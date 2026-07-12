// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Error;
use j2k_core::{
    try_host_vec_filled, try_host_vec_with_capacity, HostAllocationBudget, HostAllocationError,
    HostAllocationLimitError, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
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

    #[cfg(test)]
    pub(crate) const fn live_bytes(&self) -> usize {
        self.inner.live_bytes()
    }

    #[cfg(feature = "cuda-runtime")]
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

#[cfg(feature = "cuda-runtime")]
pub(crate) fn try_collect_exact<T, I>(iter: I, what: &'static str) -> Result<Vec<T>, Error>
where
    I: ExactSizeIterator<Item = T>,
{
    let mut values = try_vec_with_capacity(iter.len(), what)?;
    values.extend(iter);
    Ok(values)
}

/// Check the adapter phase that temporarily retains both parameter ABI vectors.
#[cfg(feature = "cuda-runtime")]
pub(crate) fn checked_cuda_parameter_conversion_bytes<NeutralParams, CudaParams>(
    neutral_param_capacity: usize,
    cuda_param_capacity: usize,
) -> Result<usize, Error> {
    let mut budget = HostPhaseBudget::new("CUDA JPEG parameter conversion");
    let neutral_params = element_bytes::<NeutralParams>(neutral_param_capacity);
    let cuda_params = element_bytes::<CudaParams>(cuda_param_capacity);
    budget.account_bytes(neutral_params)?;
    budget.account_bytes(cuda_params)?;
    Ok(neutral_params.saturating_add(cuda_params))
}

#[cfg(feature = "cuda-runtime")]
fn element_bytes<T>(count: usize) -> usize {
    count.saturating_mul(core::mem::size_of::<T>())
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

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda-runtime")]
    use super::checked_cuda_parameter_conversion_bytes;
    use super::{try_vec_with_capacity, HostPhaseBudget};
    use crate::Error;
    #[cfg(feature = "cuda-runtime")]
    use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

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

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_parameter_conversion_has_an_exact_cap_boundary() {
        assert_eq!(
            checked_cuda_parameter_conversion_bytes::<u8, u8>(
                DEFAULT_MAX_HOST_ALLOCATION_BYTES - 1,
                1,
            )
            .expect("exact cap is valid"),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES
        );
        assert!(matches!(
            checked_cuda_parameter_conversion_bytes::<u8, u8>(
                DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                1,
            ),
            Err(Error::HostAllocationTooLarge {
                requested,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA JPEG parameter conversion",
            }) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
        ));
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_parameter_conversion_counts_both_allocations_and_saturates() {
        type NeutralParams = [u8; 84];
        type CudaParams = [u8; 84];

        assert!(matches!(
            checked_cuda_parameter_conversion_bytes::<NeutralParams, CudaParams>(
                usize::MAX,
                usize::MAX,
            ),
            Err(Error::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA JPEG parameter conversion",
            })
        ));
    }
}
