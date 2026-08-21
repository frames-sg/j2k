// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;

pub(crate) use j2k_core::HostPhaseBudget;

const CUDA_JPEG_HOST_VECTOR: &str = "CUDA JPEG host vector capacity";

pub(crate) fn host_element_bytes<T>(element_count: usize) -> usize {
    element_count.saturating_mul(core::mem::size_of::<T>())
}

pub(crate) fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, CudaError> {
    Ok(HostPhaseBudget::new(CUDA_JPEG_HOST_VECTOR).try_vec_with_capacity(capacity)?)
}

pub(crate) fn try_vec_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, CudaError> {
    Ok(HostPhaseBudget::new(CUDA_JPEG_HOST_VECTOR).try_vec_filled(len, value)?)
}

pub(crate) fn try_vec_defaulted<T: Clone + Default>(len: usize) -> Result<Vec<T>, CudaError> {
    try_vec_filled(len, T::default())
}
