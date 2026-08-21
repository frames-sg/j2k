// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Error;
use j2k_core::HostAllocationError;
pub(crate) use j2k_core::HostPhaseBudget;
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaError;

pub(crate) fn try_vec_with_capacity<T>(
    capacity: usize,
    what: &'static str,
) -> Result<Vec<T>, Error> {
    Ok(HostPhaseBudget::new(what).try_vec_with_capacity(capacity)?)
}

pub(crate) fn try_vec_filled<T: Clone>(
    len: usize,
    value: T,
    what: &'static str,
) -> Result<Vec<T>, Error> {
    Ok(HostPhaseBudget::new(what).try_vec_filled(len, value)?)
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

#[cfg(feature = "cuda-runtime")]
pub(crate) fn checked_cuda_element_count(width: u32, height: u32) -> Option<usize> {
    usize::try_from(width)
        .ok()?
        .checked_mul(usize::try_from(height).ok()?)
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
    let mut values = budget
        .try_vec_with_capacity(iter.len())
        .map_err(CudaError::from)?;
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

pub(crate) fn host_allocation_error<T>(element_count: usize, what: &'static str) -> Error {
    allocation_error(HostAllocationError::for_elements::<T>(element_count), what)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda-runtime")]
    use super::checked_cuda_element_count;
    use super::{
        try_collect_exact, try_vec_extend_from_slice, try_vec_push, try_vec_reserve,
        try_vec_with_capacity, HostPhaseBudget,
    };
    use crate::Error;

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_element_count_uses_checked_target_sized_arithmetic() {
        assert_eq!(checked_cuda_element_count(17, 11), Some(187));
        assert_eq!(checked_cuda_element_count(0, u32::MAX), Some(0));
        #[cfg(target_pointer_width = "32")]
        assert_eq!(checked_cuda_element_count(u32::MAX, 2), None);
    }

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
                .try_vec_with_capacity::<u32>(usize::MAX),
            Err(j2k_core::HostPhaseError::LimitExceeded {
                requested_bytes: usize::MAX,
                cap_bytes: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
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
            Err(j2k_core::HostPhaseError::LimitExceeded {
                requested_bytes: requested,
                cap_bytes: cap,
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
