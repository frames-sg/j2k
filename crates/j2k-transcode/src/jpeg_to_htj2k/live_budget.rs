// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actual-capacity accounting for owners retained across transcode phases.

use core::mem::size_of;

use super::{
    EncodedTranscode, J2kForwardDwt53Output, J2kForwardDwt97Output, JpegToHtj2kError,
    PrecomputedComponentBatch, PrecomputedHtj2k53Component, PrecomputedHtj2k97Component,
    TranscodeReport, TranscodeValidationMetrics,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HostLiveBudget {
    live_bytes: usize,
    cap: usize,
}

impl HostLiveBudget {
    pub(super) const fn process_cap() -> Self {
        Self::with_cap(j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    pub(super) const fn with_cap(cap: usize) -> Self {
        Self { live_bytes: 0, cap }
    }

    pub(super) const fn live_bytes(self) -> usize {
        self.live_bytes
    }

    pub(super) fn add_bytes(&mut self, additional: usize) -> Result<(), JpegToHtj2kError> {
        let requested = self
            .live_bytes
            .checked_add(additional)
            .ok_or_else(|| cap_error(usize::MAX, self.cap))?;
        if requested > self.cap {
            return Err(cap_error(requested, self.cap));
        }
        self.live_bytes = requested;
        Ok(())
    }

    pub(super) fn add_capacity<T>(
        &mut self,
        allocator_capacity: usize,
    ) -> Result<(), JpegToHtj2kError> {
        let bytes = allocator_capacity
            .checked_mul(size_of::<T>())
            .ok_or_else(|| cap_error(usize::MAX, self.cap))?;
        self.add_bytes(bytes)
    }

    pub(super) fn remaining_bytes(self) -> Result<usize, JpegToHtj2kError> {
        self.cap
            .checked_sub(self.live_bytes)
            .ok_or_else(|| cap_error(self.live_bytes, self.cap))
    }
}

pub(super) fn transcode_report_retained_bytes(
    report: &TranscodeReport,
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    budget.add_capacity::<super::TranscodeComponentReport>(report.components.capacity())?;
    for metrics in [
        report.float_reference_metrics.as_ref(),
        report.integer_reference_metrics.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        budget.add_bytes(metrics.absolute_error_histogram.retained_bytes()?)?;
    }
    Ok(budget.live_bytes())
}

pub(super) fn validation_metrics_retained_bytes(
    metrics: Option<&TranscodeValidationMetrics>,
) -> Result<usize, JpegToHtj2kError> {
    metrics.map_or(Ok(0), |metrics| {
        metrics
            .absolute_error_histogram
            .retained_bytes()
            .map_err(Into::into)
    })
}

pub(super) fn precomputed_batch_retained_bytes(
    batch: &PrecomputedComponentBatch,
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    match batch {
        PrecomputedComponentBatch::Dwt53(components) => {
            budget.add_capacity::<PrecomputedHtj2k53Component>(components.capacity())?;
            for component in components {
                budget.add_bytes(dwt53_retained_bytes(&component.dwt)?)?;
            }
        }
        PrecomputedComponentBatch::Dwt97(components) => {
            budget.add_capacity::<PrecomputedHtj2k97Component>(components.capacity())?;
            for component in components {
                budget.add_bytes(dwt97_retained_bytes(&component.dwt)?)?;
            }
        }
    }
    Ok(budget.live_bytes())
}

fn dwt53_retained_bytes(output: &J2kForwardDwt53Output) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    budget.add_capacity::<f32>(output.ll.capacity())?;
    budget.add_capacity::<super::J2kForwardDwt53Level>(output.levels.capacity())?;
    for level in &output.levels {
        for capacity in [
            level.hl.capacity(),
            level.lh.capacity(),
            level.hh.capacity(),
        ] {
            budget.add_capacity::<f32>(capacity)?;
        }
    }
    Ok(budget.live_bytes())
}

fn dwt97_retained_bytes(output: &J2kForwardDwt97Output) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    budget.add_capacity::<f32>(output.ll.capacity())?;
    budget.add_capacity::<super::J2kForwardDwt97Level>(output.levels.capacity())?;
    for level in &output.levels {
        for capacity in [
            level.hl.capacity(),
            level.lh.capacity(),
            level.hh.capacity(),
        ] {
            budget.add_capacity::<f32>(capacity)?;
        }
    }
    Ok(budget.live_bytes())
}

pub(super) fn encoded_transcode_retained_bytes(
    encoded: &EncodedTranscode,
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    budget.add_capacity::<u8>(encoded.codestream.capacity())?;
    budget.add_bytes(transcode_report_retained_bytes(&encoded.report)?)?;
    Ok(budget.live_bytes())
}

fn cap_error(requested: usize, cap: usize) -> JpegToHtj2kError {
    JpegToHtj2kError::MemoryCapExceeded { requested, cap }
}

#[cfg(test)]
mod tests {
    use super::HostLiveBudget;
    use crate::JpegToHtj2kError;

    #[test]
    fn live_budget_accepts_exact_cap_and_rejects_one_byte_over() {
        let mut exact = HostLiveBudget::with_cap(16);
        exact.add_bytes(7).expect("first retained owner fits");
        exact.add_bytes(9).expect("exact live cap fits");
        assert_eq!(exact.remaining_bytes().expect("valid budget"), 0);

        let mut one_over = HostLiveBudget::with_cap(16);
        one_over.add_bytes(7).expect("first retained owner fits");
        assert!(matches!(
            one_over.add_bytes(10),
            Err(JpegToHtj2kError::MemoryCapExceeded {
                requested: 17,
                cap: 16
            })
        ));
    }

    #[test]
    fn live_budget_uses_allocator_capacity_instead_of_logical_length() {
        let mut logical = HostLiveBudget::with_cap(16);
        logical
            .add_capacity::<u32>(4)
            .expect("logical capacity reaches exact cap");

        let mut allocator = HostLiveBudget::with_cap(16);
        assert!(matches!(
            allocator.add_capacity::<u32>(5),
            Err(JpegToHtj2kError::MemoryCapExceeded {
                requested: 20,
                cap: 16
            })
        ));
    }
}
