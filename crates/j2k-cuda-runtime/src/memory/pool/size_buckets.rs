// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::{host_allocation_error, try_vec_with_capacity},
    memory::CudaDeviceBuffer,
    CudaError,
};

mod inventory;

#[derive(Debug)]
struct CudaBufferPoolSizeBucket {
    size: usize,
    buffers: Vec<CudaDeviceBuffer>,
}

#[derive(Debug)]
pub(crate) struct CudaBufferPoolSizeBuckets {
    buckets: Vec<CudaBufferPoolSizeBucket>,
}

impl CudaBufferPoolSizeBuckets {
    pub(crate) const fn new() -> Self {
        Self {
            buckets: Vec::new(),
        }
    }

    pub(crate) fn try_recycle(
        &mut self,
        buffer: CudaDeviceBuffer,
    ) -> Result<(), (CudaError, CudaDeviceBuffer)> {
        let size = buffer.byte_len();
        match self
            .buckets
            .binary_search_by_key(&size, |bucket| bucket.size)
        {
            Ok(index) => {
                let buffers = &mut self.buckets[index].buffers;
                if buffers.try_reserve(1).is_err() {
                    return Err((
                        host_allocation_error::<CudaDeviceBuffer>(buffers.len().saturating_add(1)),
                        buffer,
                    ));
                }
                buffers.push(buffer);
            }
            Err(index) => {
                if self.buckets.try_reserve(1).is_err() {
                    return Err((
                        host_allocation_error::<CudaBufferPoolSizeBucket>(
                            self.buckets.len().saturating_add(1),
                        ),
                        buffer,
                    ));
                }
                let mut buffers = match try_vec_with_capacity(1) {
                    Ok(buffers) => buffers,
                    Err(error) => return Err((error, buffer)),
                };
                buffers.push(buffer);
                self.buckets
                    .insert(index, CudaBufferPoolSizeBucket { size, buffers });
            }
        }
        Ok(())
    }

    pub(crate) fn evict_largest_oldest(&mut self) -> Option<CudaDeviceBuffer> {
        let index = self.buckets.len().checked_sub(1)?;
        let buffer = self.buckets[index].buffers.remove(0);
        if self.buckets[index].buffers.is_empty() {
            self.buckets.remove(index);
        }
        Some(buffer)
    }

    pub(crate) fn take(&mut self, len: usize) -> (Option<CudaDeviceBuffer>, usize) {
        let index = self.buckets.partition_point(|bucket| bucket.size < len);
        if index == self.buckets.len() {
            return (None, usize::from(!self.buckets.is_empty()));
        }
        let buffer = self.buckets[index].buffers.pop();
        if self.buckets[index].buffers.is_empty() {
            self.buckets.remove(index);
        }
        (buffer, 1)
    }
}
