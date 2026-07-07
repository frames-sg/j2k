// SPDX-License-Identifier: MIT OR Apache-2.0

#[inline(always)]
#[allow(dead_code)]
fn simt_load<T: Copy>(ptr: *const T, index: usize) -> T {
    // SAFETY: CUDA-Oxide kernels pass validated device buffers and launch-bounded
    // indices to these helpers. Callers keep the element type aligned with the
    // buffer ABI for each kernel job.
    unsafe { *ptr.add(index) }
}

#[inline(always)]
#[allow(dead_code)]
fn simt_store<T>(ptr: *mut T, index: usize, value: T) {
    // SAFETY: CUDA-Oxide kernels pass validated device buffers and launch-bounded
    // indices to these helpers. Callers keep writes within the destination
    // buffer capacity checked by the host-side runtime.
    unsafe {
        *ptr.add(index) = value;
    }
}

#[inline(always)]
#[allow(dead_code)]
fn simt_ptr_at<T>(ptr: *const T, index: usize) -> *const T {
    // SAFETY: The returned pointer is used by callers that already validated the
    // base device buffer and index range for the active kernel job.
    unsafe { ptr.add(index) }
}

#[inline(always)]
#[allow(dead_code)]
fn simt_mut_ptr_at<T>(ptr: *mut T, index: usize) -> *mut T {
    // SAFETY: The returned pointer is used by callers that already validated the
    // base device buffer and index range for the active kernel job.
    unsafe { ptr.add(index) }
}
