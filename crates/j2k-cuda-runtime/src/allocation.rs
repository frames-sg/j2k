// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;
use j2k_core::{HostAllocationError, HostAllocationLimitError};

mod phase;
pub(crate) use self::phase::HostPhaseBudget;

const CUDA_HOST_VECTOR: &str = "CUDA host vector capacity";

pub(crate) fn host_element_bytes<T>(element_count: usize) -> usize {
    element_count.saturating_mul(core::mem::size_of::<T>())
}

pub(crate) fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, CudaError> {
    HostPhaseBudget::new(CUDA_HOST_VECTOR).try_vec_with_capacity(capacity)
}

pub(crate) fn try_vec_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, CudaError> {
    HostPhaseBudget::new(CUDA_HOST_VECTOR).try_vec_filled(len, value)
}

#[cfg(any(
    feature = "cuda-oxide-jpeg-decode",
    feature = "cuda-oxide-jpeg-encode",
    test
))]
pub(crate) fn try_vec_defaulted<T: Clone + Default>(len: usize) -> Result<Vec<T>, CudaError> {
    try_vec_filled(len, T::default())
}

pub(crate) fn try_vec_from_slice<T: Copy>(source: &[T]) -> Result<Vec<T>, CudaError> {
    HostPhaseBudget::new(CUDA_HOST_VECTOR).try_vec_from_slice(source)
}

pub(crate) fn try_vec_reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), CudaError> {
    let element_count = values.len().saturating_add(additional);
    values
        .try_reserve_exact(additional)
        .map_err(|_| host_allocation_error::<T>(element_count))?;
    HostPhaseBudget::new(CUDA_HOST_VECTOR).account_vec(values)?;
    Ok(())
}

pub(crate) fn host_allocation_error<T>(element_count: usize) -> CudaError {
    cuda_allocation_error(HostAllocationError::for_elements::<T>(element_count))
}

fn cuda_allocation_error(error: HostAllocationError) -> CudaError {
    CudaError::HostAllocationFailed {
        bytes: error.requested_bytes(),
    }
}

fn capacity_error(error: HostAllocationLimitError, what: &'static str) -> CudaError {
    CudaError::HostAllocationTooLarge {
        requested: error.requested_bytes(),
        cap: error.cap_bytes(),
        what,
    }
}

#[cfg(test)]
mod tests;
