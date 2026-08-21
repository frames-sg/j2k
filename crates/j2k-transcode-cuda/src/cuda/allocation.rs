// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) use j2k_core::HostPhaseBudget;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use crate::CudaTranscodeError;

pub(super) fn checked_element_product(
    factors: &[usize],
    what: &'static str,
) -> Result<usize, CudaTranscodeError> {
    Ok(j2k_core::checked_host_phase_product(factors, what)?)
}

pub(super) fn checked_element_sum(
    values: &[usize],
    what: &'static str,
) -> Result<usize, CudaTranscodeError> {
    Ok(j2k_core::checked_host_phase_sum(values, what)?)
}

fn checked_host_element_count(
    factors: &[usize],
    element_size: usize,
    what: &'static str,
) -> Result<usize, CudaTranscodeError> {
    let element_count = checked_element_product(factors, what)?;
    let requested = element_count.saturating_mul(element_size);
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(CudaTranscodeError::HostAllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        });
    }
    Ok(element_count)
}

pub(super) fn checked_host_bytes<T>(
    element_count: usize,
    what: &'static str,
) -> Result<usize, CudaTranscodeError> {
    checked_host_element_count(&[element_count], core::mem::size_of::<T>(), what)?;
    Ok(element_count.saturating_mul(core::mem::size_of::<T>()))
}

pub(super) fn checked_host_byte_sum(
    byte_counts: &[usize],
    what: &'static str,
) -> Result<usize, CudaTranscodeError> {
    let requested = checked_element_sum(byte_counts, what)?;
    checked_host_byte_add(0, requested, what)
}

pub(super) fn checked_host_byte_add(
    current: usize,
    additional: usize,
    what: &'static str,
) -> Result<usize, CudaTranscodeError> {
    let requested = checked_element_sum(&[current, additional], what)?;
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(CudaTranscodeError::HostAllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        });
    }
    Ok(requested)
}

pub(super) fn try_transcode_vec_with_capacity<T>(
    element_count: usize,
    what: &'static str,
) -> Result<Vec<T>, CudaTranscodeError> {
    let element_count =
        checked_host_element_count(&[element_count], core::mem::size_of::<T>(), what)?;
    let mut budget = HostPhaseBudget::new(what);
    Ok(budget.try_vec_with_capacity_named(element_count, what)?)
}

pub(super) fn try_transcode_vec_for_product<T>(
    factors: &[usize],
    what: &'static str,
) -> Result<Vec<T>, CudaTranscodeError> {
    let element_count = checked_host_element_count(factors, core::mem::size_of::<T>(), what)?;
    try_transcode_vec_with_capacity(element_count, what)
}

#[cfg(test)]
mod tests {
    use super::{
        checked_host_byte_sum, checked_host_bytes, checked_host_element_count,
        try_transcode_vec_for_product, HostPhaseBudget,
    };
    use crate::CudaTranscodeError;
    use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

    #[test]
    fn host_staging_rejects_overflow_and_over_cap_before_allocation() {
        let overflow = checked_host_element_count(
            &[usize::MAX, 2],
            core::mem::size_of::<u8>(),
            "test overflow",
        )
        .unwrap_err();
        assert!(matches!(
            overflow,
            CudaTranscodeError::HostAllocationTooLarge {
                requested: usize::MAX,
                what: "test overflow",
                ..
            }
        ));

        let over_cap_elements = DEFAULT_MAX_HOST_ALLOCATION_BYTES / core::mem::size_of::<u64>() + 1;
        let over_cap =
            try_transcode_vec_for_product::<u64>(&[over_cap_elements], "test cap").unwrap_err();
        assert!(matches!(
            over_cap,
            CudaTranscodeError::HostAllocationTooLarge {
                requested,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "test cap",
            } if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES
        ));

        let half = DEFAULT_MAX_HOST_ALLOCATION_BYTES / 2 + 1;
        let bytes = checked_host_bytes::<u8>(half, "test aggregate part").unwrap();
        assert!(matches!(
            checked_host_byte_sum(&[bytes, bytes], "test aggregate"),
            Err(CudaTranscodeError::HostAllocationTooLarge {
                what: "test aggregate",
                ..
            })
        ));
    }

    #[test]
    fn allocator_reported_capacity_has_exact_and_one_under_boundaries() {
        let values = j2k_core::try_host_vec_with_capacity::<u8>(17).unwrap();
        let actual = values.capacity();
        let mut exact = HostPhaseBudget::with_cap("test actual capacity", actual);

        assert!(matches!(
            exact.account_vec(&values),
            Ok(bytes) if bytes == actual
        ));
        let mut one_under =
            HostPhaseBudget::with_cap("test actual capacity", actual.saturating_sub(1));
        assert!(matches!(
            one_under.account_vec(&values),
            Err(j2k_core::HostPhaseError::LimitExceeded {
                requested_bytes: requested,
                cap_bytes: cap,
                what: "test actual capacity",
            }) if requested == actual && cap == actual.saturating_sub(1)
        ));
    }

    #[test]
    fn phase_budget_reconciles_synthetic_allocator_capacity_and_failure() {
        let values = j2k_core::try_host_vec_with_capacity::<u8>(17).unwrap();
        let actual = values.capacity();
        let mut exact = HostPhaseBudget::with_cap("test phase", actual);
        exact
            .account_vec(&values)
            .expect("allocator-reported capacity fits exact phase cap");
        assert_eq!(exact.live_bytes(), actual);

        let oversized = j2k_core::try_host_vec_with_capacity::<u8>(17).unwrap();
        let oversized_actual = oversized.capacity();
        let mut one_under =
            HostPhaseBudget::with_cap("test phase", oversized_actual.saturating_sub(1));
        assert!(matches!(
            one_under.account_vec(&oversized),
            Err(j2k_core::HostPhaseError::LimitExceeded {
                requested_bytes: requested,
                cap_bytes: cap,
                what: "test phase",
            }) if requested == oversized_actual && cap == oversized_actual.saturating_sub(1)
        ));

        let failed = CudaTranscodeError::from(j2k_core::HostPhaseError::AllocationFailed {
            requested_bytes: 16,
            what: "test owner",
        });
        assert!(matches!(
            failed,
            CudaTranscodeError::HostAllocationFailed {
                requested: 16,
                what: "test owner",
            }
        ));
    }
}
