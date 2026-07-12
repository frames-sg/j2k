// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    try_host_vec_filled, try_host_vec_from_slice, try_host_vec_with_capacity, HostAllocationBudget,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

use super::{capacity_error, cuda_allocation_error};
use crate::error::CudaError;

pub(crate) struct HostPhaseBudget {
    inner: HostAllocationBudget,
    what: &'static str,
}

impl HostPhaseBudget {
    pub(crate) const fn new(what: &'static str) -> Self {
        Self::with_cap(what, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    pub(crate) const fn with_cap(what: &'static str, cap: usize) -> Self {
        Self {
            inner: HostAllocationBudget::new(cap),
            what,
        }
    }

    pub(crate) const fn live_bytes(&self) -> usize {
        self.inner.live_bytes()
    }

    pub(crate) fn with_live_bytes(
        what: &'static str,
        live_bytes: usize,
    ) -> Result<Self, CudaError> {
        let mut budget = Self::new(what);
        budget.account_bytes(live_bytes)?;
        Ok(budget)
    }

    pub(crate) fn account_bytes(&mut self, bytes: usize) -> Result<(), CudaError> {
        self.inner
            .account_bytes(bytes)
            .map_err(|error| capacity_error(error, self.what))
    }

    pub(crate) fn account_vec<T>(&mut self, values: &Vec<T>) -> Result<usize, CudaError> {
        self.inner
            .account_vec(values)
            .map_err(|error| capacity_error(error, self.what))
    }

    pub(crate) fn account_capacity<T>(&mut self, capacity: usize) -> Result<usize, CudaError> {
        self.inner
            .account_capacity::<T>(capacity)
            .map_err(|error| capacity_error(error, self.what))
    }

    pub(crate) fn try_vec_with_capacity<T>(
        &mut self,
        capacity: usize,
    ) -> Result<Vec<T>, CudaError> {
        self.inner
            .check_capacity::<T>(capacity)
            .map_err(|error| capacity_error(error, self.what))?;
        let values = try_host_vec_with_capacity(capacity).map_err(cuda_allocation_error)?;
        self.account_vec(&values)?;
        Ok(values)
    }

    pub(crate) fn try_vec_filled<T: Clone>(
        &mut self,
        len: usize,
        value: T,
    ) -> Result<Vec<T>, CudaError> {
        self.inner
            .check_capacity::<T>(len)
            .map_err(|error| capacity_error(error, self.what))?;
        let values = try_host_vec_filled(len, value).map_err(cuda_allocation_error)?;
        self.account_vec(&values)?;
        Ok(values)
    }

    pub(crate) fn try_vec_from_slice<T: Copy>(
        &mut self,
        source: &[T],
    ) -> Result<Vec<T>, CudaError> {
        self.inner
            .check_capacity::<T>(source.len())
            .map_err(|error| capacity_error(error, self.what))?;
        let values = try_host_vec_from_slice(source).map_err(cuda_allocation_error)?;
        self.account_vec(&values)?;
        Ok(values)
    }
}
