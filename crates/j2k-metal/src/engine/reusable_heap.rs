// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal heaps whose memory stays mapped between the buffers allocated from them.

use std::sync::Mutex;

use j2k_core::accelerator::GpuAbi;
use j2k_metal_support::{checked_buffer_write, checked_heap_shared_buffer, checked_shared_heap};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLDevice as _, MTLHeap};

use crate::error::metal_kernel_support_error;
use crate::metal_types::{Buffer, Device};
use crate::Error;

use super::new_shared_buffer;

/// A Metal heap that hands out shared buffers and keeps their memory mapped
/// after they are released.
///
/// Writing into a new buffer faults in fresh pages on the CPU and maps them
/// for the GPU; for a 13 MB mammogram payload that cost more than the copy,
/// and for a new 29 MB output surface about 0.75 ms. Heap memory stays mapped
/// after its buffers are released, and Metal reuses a buffer's memory only
/// after every reference to it, including a command buffer's or a caller's,
/// is gone, so reuse needs no completion tracking here.
pub(in crate::engine) struct ReusableHeap {
    heap: Mutex<Option<Retained<ProtocolObject<dyn MTLHeap>>>>,
    min_bytes: usize,
    max_bytes: usize,
}

impl ReusableHeap {
    /// A heap that grows to between `min_bytes` and `max_bytes`, and never
    /// past the device or host allocation limit; requests larger than that
    /// get their own buffers.
    pub(in crate::engine) const fn new(min_bytes: usize, max_bytes: usize) -> Self {
        Self {
            heap: Mutex::new(None),
            min_bytes,
            max_bytes,
        }
    }

    /// A shared buffer of `bytes` from the heap. A request the heap is too
    /// small for replaces it with a larger one, while buffers of the old heap
    /// keep it alive; a request it cannot hold now because live buffers use
    /// it gets its own buffer, so a full heap does not churn.
    pub(in crate::engine) fn allocate(
        &self,
        device: &Device,
        bytes: usize,
    ) -> Result<Buffer, Error> {
        let bytes = bytes.max(1);
        let mut heap = self.heap.lock().map_err(|_| Error::MetalStatePoisoned {
            state: "J2K Metal reusable heap",
        })?;
        let Some(size) = self.heap_size_for(device, bytes) else {
            return new_shared_buffer(device, bytes);
        };
        if let Some(current) = heap.as_ref() {
            if let Some(buffer) = heap_buffer(current, bytes)? {
                return Ok(buffer);
            }
            if current.size() >= size {
                return new_shared_buffer(device, bytes);
            }
        }
        let grown = checked_shared_heap(device, size).map_err(|source| {
            metal_kernel_support_error("J2K Metal reusable heap allocation failed", source)
        })?;
        let buffer = heap_buffer(&grown, bytes)?;
        *heap = Some(grown);
        match buffer {
            Some(buffer) => Ok(buffer),
            None => new_shared_buffer(device, bytes),
        }
    }

    /// The heap size that serves `bytes`: room for two such buffers when that
    /// fits the limits, else for one, else `None`.
    fn heap_size_for(&self, device: &Device, bytes: usize) -> Option<usize> {
        let limit = self
            .max_bytes
            .min(device.maxBufferLength())
            .min(j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES);
        [bytes.checked_mul(2), Some(bytes)]
            .into_iter()
            .flatten()
            .filter_map(usize::checked_next_power_of_two)
            .map(|size| size.max(self.min_bytes))
            .find(|size| *size <= limit)
    }
}

fn heap_buffer(heap: &ProtocolObject<dyn MTLHeap>, bytes: usize) -> Result<Option<Buffer>, Error> {
    checked_heap_shared_buffer(heap, bytes).map_err(|source| {
        metal_kernel_support_error("J2K Metal reusable heap buffer allocation failed", source)
    })
}

/// Where a prepared HT execution allocates its immutable input buffers.
#[derive(Clone, Copy)]
pub(in crate::engine) struct HtInputAllocator<'a> {
    device: &'a Device,
    heap: Option<&'a ReusableHeap>,
}

impl<'a> HtInputAllocator<'a> {
    pub(in crate::engine) const fn new(device: &'a Device, heap: &'a ReusableHeap) -> Self {
        Self {
            device,
            heap: Some(heap),
        }
    }

    /// Allocates each input as its own buffer.
    #[cfg(test)]
    pub(in crate::engine) const fn plain(device: &'a Device) -> Self {
        Self { device, heap: None }
    }

    pub(in crate::engine) fn shared_buffer(self, bytes: usize) -> Result<Buffer, Error> {
        match self.heap {
            Some(heap) => heap.allocate(self.device, bytes),
            None => new_shared_buffer(self.device, bytes),
        }
    }

    pub(in crate::engine) fn copied_slice<T: GpuAbi>(self, values: &[T]) -> Result<Buffer, Error> {
        let buffer = self.shared_buffer(core::mem::size_of_val(values))?;
        // SAFETY: the buffer was just allocated and no command buffer has
        // referenced it yet.
        unsafe { checked_buffer_write(&buffer, 0, values) }.map_err(|source| {
            metal_kernel_support_error("J2K Metal HT input upload failed", source)
        })?;
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use objc2::rc::Retained;
    use objc2_metal::{MTLBuffer as _, MTLResource as _};

    use super::{HtInputAllocator, ReusableHeap};

    const MIN_HEAP_BYTES: usize = 4 << 20;
    const MAX_HEAP_BYTES: usize = 256 << 20;
    use crate::metal_types::Buffer;

    fn heap_of(buffer: &Buffer) -> Option<usize> {
        buffer
            .heap()
            .map(|heap| Retained::as_ptr(&heap).cast::<u8>() as usize)
    }

    #[test]
    fn inputs_share_one_heap_that_grows_and_falls_back_when_full() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let device = j2k_metal_support::system_default_device().expect("required Metal device");
        let heap = ReusableHeap::new(MIN_HEAP_BYTES, MAX_HEAP_BYTES);
        let first = heap.allocate(&device, 1 << 20).expect("first input");
        let first_heap = heap_of(&first).expect("served from the heap");
        drop(first);
        let reused = heap.allocate(&device, 1 << 20).expect("reused input");
        assert_eq!(
            heap_of(&reused),
            Some(first_heap),
            "released memory is reused"
        );

        // More than half of the current heap: replaced by a larger heap.
        let large = heap.allocate(&device, MIN_HEAP_BYTES).expect("large input");
        let grown = heap_of(&large).expect("served from a grown heap");
        assert_ne!(grown, first_heap);

        // Live inputs fill the grown heap; the next request gets its own buffer.
        let mut live = vec![large];
        let overflow = loop {
            let buffer = heap.allocate(&device, MIN_HEAP_BYTES / 2).expect("input");
            if heap_of(&buffer).is_none() {
                break buffer;
            }
            assert_eq!(heap_of(&buffer), Some(grown), "a full heap is not replaced");
            live.push(buffer);
        };
        assert!(overflow.length() >= MIN_HEAP_BYTES / 2);

        let oversized = heap
            .allocate(&device, MAX_HEAP_BYTES + 1)
            .expect("oversized input");
        assert_eq!(
            heap_of(&oversized),
            None,
            "requests over the cap never use a heap"
        );
    }

    #[test]
    fn heaps_stay_within_the_host_allocation_limit() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let device = j2k_metal_support::system_default_device().expect("required Metal device");
        let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
        let heap = ReusableHeap::new(64 << 20, usize::MAX);
        // Twice the request fits under the cap: room for two buffers.
        assert_eq!(heap.heap_size_for(&device, 100 << 20), Some(256 << 20));
        // Only one fits, as for a 16-image mammogram batch output.
        assert_eq!(heap.heap_size_for(&device, 464 << 20), Some(cap));
        // Larger requests get their own buffers.
        assert_eq!(heap.heap_size_for(&device, cap + 1), None);
        let output = heap.allocate(&device, 464 << 20).expect("large output");
        assert!(heap_of(&output).is_some());
    }

    #[test]
    fn copied_inputs_hold_their_values() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let device = j2k_metal_support::system_default_device().expect("required Metal device");
        let heap = ReusableHeap::new(MIN_HEAP_BYTES, MAX_HEAP_BYTES);
        let values: Vec<u32> = (0..1000).map(|value| value * 7 + 3).collect();
        let buffer = HtInputAllocator::new(&device, &heap)
            .copied_slice(&values)
            .expect("copied input");
        assert!(heap_of(&buffer).is_some());
        // SAFETY: no command buffer has referenced this buffer.
        let read = unsafe { j2k_metal_support::checked_buffer_read_vec::<u32>(&buffer, 0, 1000) }
            .expect("read input");
        assert_eq!(read, values);
    }
}
