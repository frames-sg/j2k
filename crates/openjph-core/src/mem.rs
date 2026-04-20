//! Memory management utilities — port of `ojph_mem.h/cpp`.
//!
//! Provides aligned allocation, line buffers, and arena-style allocators that
//! mirror the C++ OpenJPH memory model.

use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;

use crate::arch::BYTE_ALIGNMENT;
use crate::error::{OjphError, Result};

// =========================================================================
// AlignedVec<T>
// =========================================================================

/// A heap-allocated, contiguously-stored buffer with a guaranteed minimum byte
/// alignment (defaults to [`BYTE_ALIGNMENT`] = 64, suitable for AVX-512).
///
/// This is the Rust equivalent of the C++ `ojph::mem_aligned_allocator` usage
/// pattern.  It uses [`std::alloc::alloc`] / [`std::alloc::dealloc`] with a
/// custom [`Layout`] so the underlying pointer is always aligned.
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    alignment: usize,
}

// Safety: the buffer is exclusively owned.
unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

impl<T> AlignedVec<T> {
    /// Creates a new, empty `AlignedVec` with the default alignment.
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            alignment: BYTE_ALIGNMENT as usize,
        }
    }

    /// Creates a new `AlignedVec` with the specified alignment.
    pub fn with_alignment(alignment: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "alignment must be a power of two"
        );
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            alignment,
        }
    }

    /// Allocates room for exactly `count` elements, zero-initialized.
    ///
    /// Any previous contents are dropped and deallocated.
    pub fn resize(&mut self, count: usize) -> Result<()>
    where
        T: Default + Copy,
    {
        self.dealloc_inner();

        if count == 0 {
            return Ok(());
        }

        let layout = self.make_layout(count)?;
        // SAFETY: layout has non-zero size and valid alignment.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        if raw.is_null() {
            return Err(OjphError::AllocationFailed);
        }
        self.ptr = unsafe { NonNull::new_unchecked(raw.cast::<T>()) };
        self.len = count;
        self.capacity = count;
        Ok(())
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    // -- internal helpers --------------------------------------------------

    fn make_layout(&self, count: usize) -> Result<Layout> {
        let size = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(OjphError::AllocationFailed)?;
        let align = self.alignment.max(std::mem::align_of::<T>());
        Layout::from_size_align(size, align).map_err(|_| OjphError::AllocationFailed)
    }

    fn dealloc_inner(&mut self) {
        if self.capacity > 0 {
            if let Ok(layout) = self.make_layout(self.capacity) {
                // SAFETY: ptr was allocated with this layout.
                unsafe { alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
            }
            self.len = 0;
            self.capacity = 0;
            self.ptr = NonNull::dangling();
        }
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        self.dealloc_inner();
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for `len` elements.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for AlignedVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: ptr is valid and uniquely owned.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Index<usize> for AlignedVec<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: usize) -> &T {
        &(**self)[idx]
    }
}

impl<T> IndexMut<usize> for AlignedVec<T> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut (**self)[idx]
    }
}

impl<T> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// LineBuf — port of `line_buf`
// =========================================================================

/// Flag: buffer type is undefined / not yet allocated.
pub const LFT_UNDEFINED: u32 = 0x00;
/// Flag: buffer elements are 32 bits wide.
pub const LFT_32BIT: u32 = 0x04;
/// Flag: buffer elements are 64 bits wide.
pub const LFT_64BIT: u32 = 0x08;
/// Flag: buffer elements are integers (as opposed to float).
pub const LFT_INTEGER: u32 = 0x10;
/// Mask that isolates the element-size field.
pub const LFT_SIZE_MASK: u32 = 0x0F;

/// Discriminated pointer to the actual sample data in a [`LineBuf`].
#[derive(Debug, Clone, Copy)]
pub enum LineBufData {
    /// 32-bit signed integer samples.
    I32(*mut i32),
    /// 64-bit signed integer samples.
    I64(*mut i64),
    /// 32-bit floating-point samples.
    F32(*mut f32),
    /// No buffer allocated yet.
    None,
}

/// A single line (row) of sample data used throughout the wavelet and coding
/// pipeline — port of the C++ `line_buf`.
///
/// The pointer stored in [`data`](LineBuf::data) is *not* owned by this
/// struct; it borrows from an arena allocator.
#[derive(Debug)]
pub struct LineBuf {
    /// Number of samples in this line.
    pub size: usize,
    /// Extra samples prepended before the line (for filter padding).
    pub pre_size: u32,
    /// Combination of `LFT_*` flag constants describing the element type.
    pub flags: u32,
    /// Pointer into the backing buffer.
    pub data: LineBufData,
}

impl LineBuf {
    /// Creates a new, empty line buffer.
    pub fn new() -> Self {
        Self {
            size: 0,
            pre_size: 0,
            flags: LFT_UNDEFINED,
            data: LineBufData::None,
        }
    }
}

impl Default for LineBuf {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// LiftingBuf — lightweight wrapper for wavelet lifting steps
// =========================================================================

/// A line buffer reference used during wavelet lifting.
pub struct LiftingBuf {
    /// Whether this buffer slot is currently active in a lifting step.
    pub active: bool,
    /// Index into an external line-buffer array (avoids self-referential
    /// lifetime issues that a raw mutable reference would introduce).
    pub line_idx: Option<usize>,
}

impl LiftingBuf {
    /// Creates an inactive lifting buffer.
    pub fn new() -> Self {
        Self {
            active: false,
            line_idx: None,
        }
    }
}

impl Default for LiftingBuf {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// MemFixedAllocator — port of `mem_fixed_allocator`
// =========================================================================

/// A two-phase bump allocator for fixed-size, aligned sub-regions.
///
/// # Usage
///
/// 1. **Pre-allocation phase** — call [`pre_alloc_data`](Self::pre_alloc_data)
///    repeatedly to accumulate the total size needed.
/// 2. Call [`alloc_data`](Self::finalize) once to allocate the backing buffer.
/// 3. **Allocation phase** — call [`alloc_data`](Self::alloc_data) to hand out
///    sub-slices from the backing buffer.
pub struct MemFixedAllocator {
    buf: Vec<u8>,
    /// Running total of bytes needed (phase 1) / bytes dispensed (phase 2).
    offset: usize,
    alignment: usize,
}

impl MemFixedAllocator {
    /// Creates a new allocator with the default alignment.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            offset: 0,
            alignment: BYTE_ALIGNMENT as usize,
        }
    }

    /// Phase 1: registers that `size` bytes (with alignment) will be needed.
    pub fn pre_alloc_data(&mut self, size: usize, _count: usize) {
        let aligned = (size + self.alignment - 1) & !(self.alignment - 1);
        self.offset += aligned;
    }

    /// Allocates the single backing buffer based on the accumulated size.
    pub fn finalize(&mut self) -> Result<()> {
        self.buf = vec![0u8; self.offset];
        self.offset = 0;
        Ok(())
    }

    /// Phase 2: hands out the next `size`-byte slice from the backing buffer.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of this allocator.
    /// The caller must ensure `size` matches a prior `pre_alloc_data` call.
    pub fn alloc_data(&mut self, size: usize) -> Result<*mut u8> {
        let aligned = (size + self.alignment - 1) & !(self.alignment - 1);
        if self.offset + aligned > self.buf.len() {
            return Err(OjphError::AllocationFailed);
        }
        let ptr = self.buf[self.offset..].as_mut_ptr();
        self.offset += aligned;
        Ok(ptr)
    }
}

impl Default for MemFixedAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// CodedLists — linked-list node for coded data
// =========================================================================

/// A node in a singly-linked list of coded data buffers — port of
/// `coded_lists`.
pub struct CodedLists {
    /// Pointer to the next node, or `None` if this is the tail.
    pub next: Option<Box<CodedLists>>,
    /// Pointer to the start of the coded data in this node.
    pub buf: *mut u8,
    /// Number of valid coded bytes.
    pub buf_size: usize,
    /// True when the list is not ready for consumption.
    pub avail: bool,
}

impl CodedLists {
    /// Creates a new, empty coded-list node.
    pub fn new() -> Self {
        Self {
            next: None,
            buf: std::ptr::null_mut(),
            buf_size: 0,
            avail: false,
        }
    }
}

impl Default for CodedLists {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// MemElasticAllocator — port of `mem_elastic_allocator`
// =========================================================================

/// Default chunk size for elastic allocation (256 KiB).
const ELASTIC_CHUNK_SIZE: usize = 256 * 1024;

/// An arena-style allocator that grows by appending large chunks.
///
/// This is used for coded data buffers where the total size is not known
/// in advance.
pub struct MemElasticAllocator {
    chunks: Vec<Vec<u8>>,
    chunk_size: usize,
    cur_offset: usize,
}

impl MemElasticAllocator {
    /// Creates a new elastic allocator with the default chunk size.
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            chunk_size: ELASTIC_CHUNK_SIZE,
            cur_offset: 0,
        }
    }

    /// Creates a new elastic allocator with a custom chunk size.
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            chunk_size,
            cur_offset: 0,
        }
    }

    /// Allocates `size` bytes from the arena, returning a raw mutable pointer.
    ///
    /// A new chunk is appended if the current one does not have enough room.
    pub fn alloc_data(&mut self, size: usize) -> Result<*mut u8> {
        let need = size;
        // Check if the current chunk has space.
        if let Some(last) = self.chunks.last() {
            if self.cur_offset + need <= last.len() {
                let ptr = unsafe { last.as_ptr().add(self.cur_offset) as *mut u8 };
                self.cur_offset += need;
                return Ok(ptr);
            }
        }
        // Need a new chunk.
        let alloc_size = self.chunk_size.max(need);
        let chunk = vec![0u8; alloc_size];
        let ptr = chunk.as_ptr() as *mut u8;
        self.chunks.push(chunk);
        self.cur_offset = need;
        Ok(ptr)
    }

    /// Releases all memory and resets the allocator.
    pub fn reset(&mut self) {
        self.chunks.clear();
        self.cur_offset = 0;
    }
}

impl Default for MemElasticAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_vec_basic() {
        let mut v = AlignedVec::<i32>::new();
        v.resize(128).unwrap();
        assert_eq!(v.len(), 128);
        assert_eq!(v[0], 0);
        v[0] = 42;
        assert_eq!(v[0], 42);
        // Check alignment.
        assert_eq!(v.as_ptr() as usize % BYTE_ALIGNMENT as usize, 0);
    }

    #[test]
    fn fixed_allocator_round_trip() {
        let mut a = MemFixedAllocator::new();
        a.pre_alloc_data(100, 1);
        a.pre_alloc_data(200, 1);
        a.finalize().unwrap();
        let p1 = a.alloc_data(100).unwrap();
        let p2 = a.alloc_data(200).unwrap();
        assert!(!p1.is_null());
        assert!(!p2.is_null());
    }

    #[test]
    fn elastic_allocator_basic() {
        let mut a = MemElasticAllocator::new();
        let p = a.alloc_data(64).unwrap();
        assert!(!p.is_null());
        a.reset();
    }
}
