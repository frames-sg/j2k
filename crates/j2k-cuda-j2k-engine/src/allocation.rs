// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use j2k_core::HostPhaseBudget;

use j2k_cuda_runtime::CudaError;

pub(crate) fn host_allocation_error<T>(element_count: usize) -> CudaError {
    CudaError::HostAllocationFailed {
        bytes: j2k_core::HostAllocationError::for_elements::<T>(element_count).requested_bytes(),
    }
}

pub(crate) fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, CudaError> {
    HostPhaseBudget::new("CUDA J2K engine allocation")
        .try_vec_with_capacity(capacity)
        .map_err(Into::into)
}

pub(crate) fn try_vec_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, CudaError> {
    HostPhaseBudget::new("CUDA J2K engine allocation")
        .try_vec_filled(len, value)
        .map_err(Into::into)
}

pub(crate) fn try_vec_from_slice<T: Copy>(source: &[T]) -> Result<Vec<T>, CudaError> {
    HostPhaseBudget::new("CUDA J2K engine allocation")
        .try_vec_from_slice(source)
        .map_err(Into::into)
}
