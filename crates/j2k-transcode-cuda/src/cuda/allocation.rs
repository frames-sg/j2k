// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    try_host_vec_with_capacity, HostAllocationBudget, HostAllocationError,
    HostAllocationLimitError, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

use crate::CudaTranscodeError;

pub(super) struct HostPhaseBudget {
    inner: HostAllocationBudget,
    what: &'static str,
}

impl HostPhaseBudget {
    pub(super) const fn new(what: &'static str) -> Self {
        Self::with_cap(what, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    pub(super) const fn with_cap(what: &'static str, cap: usize) -> Self {
        Self {
            inner: HostAllocationBudget::new(cap),
            what,
        }
    }

    pub(super) fn with_live_bytes(
        what: &'static str,
        live_bytes: usize,
    ) -> Result<Self, CudaTranscodeError> {
        let mut budget = Self::new(what);
        budget.account_bytes(live_bytes)?;
        Ok(budget)
    }

    pub(super) const fn live_bytes(&self) -> usize {
        self.inner.live_bytes()
    }

    pub(super) fn preflight_capacity<T>(
        &self,
        element_count: usize,
    ) -> Result<usize, CudaTranscodeError> {
        self.inner
            .check_capacity::<T>(element_count)
            .map_err(|error| transcode_capacity_error(error, self.what))
    }

    pub(super) fn preflight_bytes(&self, additional: usize) -> Result<(), CudaTranscodeError> {
        let mut projected = self.inner;
        projected
            .account_bytes(additional)
            .map_err(|error| transcode_capacity_error(error, self.what))
    }

    pub(super) fn account_bytes(&mut self, bytes: usize) -> Result<(), CudaTranscodeError> {
        self.inner
            .account_bytes(bytes)
            .map_err(|error| transcode_capacity_error(error, self.what))
    }

    pub(super) fn account_vec<T>(&mut self, values: &Vec<T>) -> Result<usize, CudaTranscodeError> {
        self.inner
            .account_vec(values)
            .map_err(|error| transcode_capacity_error(error, self.what))
    }

    pub(super) fn try_vec_with_capacity<T>(
        &mut self,
        element_count: usize,
        allocation_what: &'static str,
    ) -> Result<Vec<T>, CudaTranscodeError> {
        self.try_vec_with_capacity_using(element_count, allocation_what, |capacity| {
            try_host_vec_with_capacity(capacity)
        })
    }

    fn try_vec_with_capacity_using<T>(
        &mut self,
        element_count: usize,
        allocation_what: &'static str,
        allocate: impl FnOnce(usize) -> Result<Vec<T>, HostAllocationError>,
    ) -> Result<Vec<T>, CudaTranscodeError> {
        self.preflight_capacity::<T>(element_count)?;
        let values =
            allocate(element_count).map_err(|error| CudaTranscodeError::HostAllocationFailed {
                requested: error.requested_bytes(),
                what: allocation_what,
            })?;
        self.account_vec(&values)?;
        Ok(values)
    }

    pub(super) fn try_vec_for_product<T>(
        &mut self,
        factors: &[usize],
        allocation_what: &'static str,
    ) -> Result<Vec<T>, CudaTranscodeError> {
        let element_count = checked_element_product(factors, allocation_what)?;
        self.try_vec_with_capacity(element_count, allocation_what)
    }

    pub(super) fn try_vec_from_slice<T: Copy>(
        &mut self,
        source: &[T],
        allocation_what: &'static str,
    ) -> Result<Vec<T>, CudaTranscodeError> {
        let mut values = self.try_vec_with_capacity(source.len(), allocation_what)?;
        values.extend_from_slice(source);
        Ok(values)
    }

    pub(super) fn try_vec_from_array<T, const N: usize>(
        &mut self,
        source: [T; N],
        allocation_what: &'static str,
    ) -> Result<Vec<T>, CudaTranscodeError> {
        let mut values = self.try_vec_with_capacity(N, allocation_what)?;
        values.extend(source);
        Ok(values)
    }
}

pub(super) fn checked_element_product(
    factors: &[usize],
    what: &'static str,
) -> Result<usize, CudaTranscodeError> {
    let Some(element_count) = factors
        .iter()
        .try_fold(1usize, |count, factor| count.checked_mul(*factor))
    else {
        return Err(CudaTranscodeError::HostAllocationTooLarge {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        });
    };
    Ok(element_count)
}

pub(super) fn checked_element_sum(
    values: &[usize],
    what: &'static str,
) -> Result<usize, CudaTranscodeError> {
    values
        .iter()
        .try_fold(0usize, |total, value| total.checked_add(*value))
        .ok_or(CudaTranscodeError::HostAllocationTooLarge {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        })
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
    byte_counts.iter().try_fold(0usize, |total, bytes| {
        checked_host_byte_add(total, *bytes, what)
    })
}

pub(super) fn checked_host_byte_add(
    current: usize,
    additional: usize,
    what: &'static str,
) -> Result<usize, CudaTranscodeError> {
    let requested = current.saturating_add(additional);
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
    budget.try_vec_with_capacity(element_count, what)
}

fn transcode_capacity_error(
    error: HostAllocationLimitError,
    what: &'static str,
) -> CudaTranscodeError {
    CudaTranscodeError::HostAllocationTooLarge {
        requested: error.requested_bytes(),
        cap: error.cap_bytes(),
        what,
    }
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
            Err(CudaTranscodeError::HostAllocationTooLarge {
                requested,
                cap,
                what: "test actual capacity",
            }) if requested == actual && cap == actual.saturating_sub(1)
        ));
    }

    #[test]
    fn phase_budget_reconciles_synthetic_allocator_capacity_and_failure() {
        let values = j2k_core::try_host_vec_with_capacity::<u8>(17).unwrap();
        let actual = values.capacity();
        let mut exact = HostPhaseBudget::with_cap("test phase", actual);
        let accepted = exact
            .try_vec_with_capacity_using(1, "test owner", |_| Ok(values))
            .expect("allocator-reported capacity fits exact phase cap");
        assert_eq!(exact.live_bytes(), actual);
        drop(accepted);

        let oversized = j2k_core::try_host_vec_with_capacity::<u8>(17).unwrap();
        let oversized_actual = oversized.capacity();
        let mut one_under =
            HostPhaseBudget::with_cap("test phase", oversized_actual.saturating_sub(1));
        assert!(matches!(
            one_under.try_vec_with_capacity_using(1, "test owner", |_| Ok(oversized)),
            Err(CudaTranscodeError::HostAllocationTooLarge {
                requested,
                cap,
                what: "test phase",
            }) if requested == oversized_actual && cap == oversized_actual.saturating_sub(1)
        ));

        let mut failed = HostPhaseBudget::with_cap("test phase", usize::MAX);
        assert!(matches!(
            failed.try_vec_with_capacity_using::<u32>(4, "test owner", |_| {
                Err(j2k_core::HostAllocationError::for_elements::<u32>(4))
            }),
            Err(CudaTranscodeError::HostAllocationFailed {
                requested: 16,
                what: "test owner",
            })
        ));
        assert_eq!(failed.live_bytes(), 0);
    }
}
