// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Error;
use j2k_core::{
    try_host_vec_filled, try_host_vec_with_capacity, HostAllocationBudget, HostAllocationError,
    HostAllocationLimitError, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaError;

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

    pub(crate) fn with_live_bytes(what: &'static str, live_bytes: usize) -> Result<Self, Error> {
        let mut budget = Self::new(what);
        budget.account_bytes(live_bytes)?;
        Ok(budget)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn with_cuda_live_bytes(
        what: &'static str,
        live_bytes: usize,
    ) -> Result<Self, CudaError> {
        let mut budget = Self::new(what);
        budget
            .inner
            .account_bytes(live_bytes)
            .map_err(|error| cuda_capacity_error(error, what))?;
        Ok(budget)
    }

    pub(crate) const fn live_bytes(&self) -> usize {
        self.inner.live_bytes()
    }

    pub(crate) fn account_capacity<T>(&mut self, capacity: usize) -> Result<usize, Error> {
        self.inner
            .account_capacity::<T>(capacity)
            .map_err(|error| capacity_error(error, self.what))
    }

    pub(crate) fn account_bytes(&mut self, bytes: usize) -> Result<(), Error> {
        self.inner
            .account_bytes(bytes)
            .map_err(|error| capacity_error(error, self.what))
    }

    pub(crate) fn account_vec<T>(&mut self, values: &Vec<T>) -> Result<usize, Error> {
        self.inner
            .account_vec(values)
            .map_err(|error| capacity_error(error, self.what))
    }

    pub(crate) fn try_vec_with_capacity<T>(&mut self, capacity: usize) -> Result<Vec<T>, Error> {
        self.inner
            .check_capacity::<T>(capacity)
            .map_err(|error| capacity_error(error, self.what))?;
        let values = try_host_vec_with_capacity(capacity)
            .map_err(|error| allocation_error(error, self.what))?;
        self.account_vec(&values)?;
        Ok(values)
    }

    pub(crate) fn try_vec_filled<T: Clone>(
        &mut self,
        len: usize,
        value: T,
    ) -> Result<Vec<T>, Error> {
        self.inner
            .check_capacity::<T>(len)
            .map_err(|error| capacity_error(error, self.what))?;
        let values =
            try_host_vec_filled(len, value).map_err(|error| allocation_error(error, self.what))?;
        self.account_vec(&values)?;
        Ok(values)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn try_collect_results_exact<T, I>(&mut self, iter: I) -> Result<Vec<T>, Error>
    where
        I: ExactSizeIterator<Item = Result<T, Error>>,
    {
        let mut values = self.try_vec_with_capacity(iter.len())?;
        for value in iter {
            values.push(value?);
        }
        Ok(values)
    }

    pub(crate) fn try_vec_reserve<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), Error> {
        let required_capacity = values.len().saturating_add(additional);
        let previous_capacity = values.capacity();
        let minimum_growth = required_capacity.saturating_sub(previous_capacity);
        self.inner
            .check_capacity::<T>(minimum_growth)
            .map_err(|error| capacity_error(error, self.what))?;
        values
            .try_reserve_exact(additional)
            .map_err(|_| host_allocation_error::<T>(required_capacity, self.what))?;
        let actual_growth = values.capacity().saturating_sub(previous_capacity);
        self.inner
            .account_capacity::<T>(actual_growth)
            .map_err(|error| capacity_error(error, self.what))?;
        Ok(())
    }

    pub(crate) fn try_vec_push<T>(&mut self, values: &mut Vec<T>, value: T) -> Result<(), Error> {
        self.try_vec_reserve(values, 1)?;
        values.push(value);
        Ok(())
    }

    pub(crate) fn try_vec_extend_from_slice<T: Copy>(
        &mut self,
        values: &mut Vec<T>,
        source: &[T],
    ) -> Result<(), Error> {
        self.try_vec_reserve(values, source.len())?;
        values.extend_from_slice(source);
        Ok(())
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn try_cuda_vec_with_capacity<T>(
        &mut self,
        capacity: usize,
    ) -> Result<Vec<T>, CudaError> {
        self.inner
            .check_capacity::<T>(capacity)
            .map_err(|error| cuda_capacity_error(error, self.what))?;
        let values = try_host_vec_with_capacity(capacity).map_err(cuda_allocation_error)?;
        self.inner
            .account_vec(&values)
            .map_err(|error| cuda_capacity_error(error, self.what))?;
        Ok(values)
    }
}

pub(crate) fn try_vec_with_capacity<T>(
    capacity: usize,
    what: &'static str,
) -> Result<Vec<T>, Error> {
    HostPhaseBudget::new(what).try_vec_with_capacity(capacity)
}

pub(crate) fn try_vec_filled<T: Clone>(
    len: usize,
    value: T,
    what: &'static str,
) -> Result<Vec<T>, Error> {
    HostPhaseBudget::new(what).try_vec_filled(len, value)
}

pub(crate) fn try_collect_results_exact<T, I>(iter: I, what: &'static str) -> Result<Vec<T>, Error>
where
    I: ExactSizeIterator<Item = Result<T, Error>>,
{
    let mut values = try_vec_with_capacity(iter.len(), what)?;
    for value in iter {
        values.push(value?);
    }
    Ok(values)
}

#[cfg(test)]
pub(crate) fn try_collect_exact<T, I>(iter: I, what: &'static str) -> Result<Vec<T>, Error>
where
    I: ExactSizeIterator<Item = T>,
{
    let mut values = try_vec_with_capacity(iter.len(), what)?;
    values.extend(iter);
    Ok(values)
}

#[cfg(any(feature = "cuda-runtime", test))]
pub(crate) fn try_vec_reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    what: &'static str,
) -> Result<(), Error> {
    let element_count = values.len().saturating_add(additional);
    values
        .try_reserve_exact(additional)
        .map_err(|_| host_allocation_error::<T>(element_count, what))?;
    HostPhaseBudget::new(what).account_vec(values)?;
    Ok(())
}

#[cfg(any(feature = "cuda-runtime", test))]
pub(crate) fn try_vec_push<T>(
    values: &mut Vec<T>,
    value: T,
    what: &'static str,
) -> Result<(), Error> {
    try_vec_reserve(values, 1, what)?;
    values.push(value);
    Ok(())
}

#[cfg(test)]
pub(crate) fn try_vec_extend_from_slice<T: Copy>(
    values: &mut Vec<T>,
    source: &[T],
    what: &'static str,
) -> Result<(), Error> {
    try_vec_reserve(values, source.len(), what)?;
    values.extend_from_slice(source);
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn try_collect_cuda_results_exact<T, I>(
    budget: &mut HostPhaseBudget,
    iter: I,
) -> Result<Vec<T>, CudaError>
where
    I: ExactSizeIterator<Item = Result<T, CudaError>>,
{
    let mut values = budget.try_cuda_vec_with_capacity(iter.len())?;
    for value in iter {
        values.push(value?);
    }
    Ok(values)
}

fn allocation_error(error: HostAllocationError, what: &'static str) -> Error {
    Error::HostAllocationFailed {
        bytes: error.requested_bytes(),
        what,
    }
}

fn capacity_error(error: HostAllocationLimitError, what: &'static str) -> Error {
    Error::HostAllocationTooLarge {
        requested: error.requested_bytes(),
        cap: error.cap_bytes(),
        what,
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_allocation_error(error: HostAllocationError) -> CudaError {
    CudaError::HostAllocationFailed {
        bytes: error.requested_bytes(),
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_capacity_error(error: HostAllocationLimitError, what: &'static str) -> CudaError {
    CudaError::HostAllocationTooLarge {
        requested: error.requested_bytes(),
        cap: error.cap_bytes(),
        what,
    }
}

pub(crate) fn host_allocation_error<T>(element_count: usize, what: &'static str) -> Error {
    allocation_error(HostAllocationError::for_elements::<T>(element_count), what)
}

#[cfg(test)]
mod tests {
    use super::{
        try_collect_exact, try_vec_extend_from_slice, try_vec_push, try_vec_reserve,
        try_vec_with_capacity, HostPhaseBudget,
    };
    use crate::Error;
    #[cfg(feature = "cuda-runtime")]
    use j2k_cuda_runtime::CudaError;

    #[test]
    fn logically_oversized_capacity_is_rejected_before_allocation() {
        let error = try_vec_with_capacity::<u32>(usize::MAX, "test buffer").unwrap_err();
        assert!(matches!(
            error,
            Error::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "test buffer"
            }
        ));

        #[cfg(feature = "cuda-runtime")]
        assert!(matches!(
            HostPhaseBudget::new("CUDA adapter host vector capacity")
                .try_cuda_vec_with_capacity::<u32>(usize::MAX),
            Err(CudaError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA adapter host vector capacity",
            })
        ));
    }

    #[test]
    fn incremental_helpers_reserve_before_mutating() {
        let mut values = Vec::new();
        try_vec_push(&mut values, 1u8, "test values").unwrap();
        try_vec_extend_from_slice(&mut values, &[2, 3], "test values").unwrap();
        assert_eq!(values, [1, 2, 3]);
        assert_eq!(
            try_collect_exact([4u8, 5].into_iter(), "test collection").unwrap(),
            [4, 5]
        );

        let error = try_vec_reserve(&mut values, usize::MAX, "test values").unwrap_err();
        assert!(matches!(
            error,
            Error::HostAllocationFailed {
                bytes: usize::MAX,
                what: "test values"
            }
        ));
        assert_eq!(values, [1, 2, 3]);
    }

    #[test]
    fn actual_capacity_phase_budget_uses_allocator_reported_bytes() {
        let first = j2k_core::try_host_vec_with_capacity::<u8>(8).unwrap();
        let second = j2k_core::try_host_vec_with_capacity::<u8>(8).unwrap();
        let actual = first.capacity().saturating_add(second.capacity());
        let mut exact = HostPhaseBudget::with_cap("test phase", actual);
        exact.account_vec(&first).unwrap();
        exact.account_vec(&second).unwrap();
        assert_eq!(exact.live_bytes(), actual);

        let mut one_under = HostPhaseBudget::with_cap("test phase", actual.saturating_sub(1));
        one_under.account_vec(&first).unwrap();
        assert!(matches!(
            one_under.account_vec(&second),
            Err(Error::HostAllocationTooLarge {
                requested,
                cap,
                what: "test phase",
            }) if requested == actual && cap == actual.saturating_sub(1)
        ));
    }

    #[test]
    fn phase_budget_reconciles_existing_vector_growth() {
        let mut values = j2k_core::try_host_vec_with_capacity::<u8>(8).unwrap();
        values.extend_from_slice(&[0; 8]);
        let mut budget = HostPhaseBudget::new("growth phase");
        budget.account_vec(&values).unwrap();

        budget.try_vec_reserve(&mut values, 9).unwrap();
        assert!(values.capacity() >= 17);
        assert_eq!(budget.live_bytes(), values.capacity());
    }
}
