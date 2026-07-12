// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit work bound for identity scans during queued-owner admission.

use crate::Error;

/// At this ceiling, the identity ledger performs fewer than 16.8 million
/// comparisons in its worst full-collection scan.
pub(crate) const MAX_QUEUED_JPEG_REQUESTS: usize = 4096;

pub(super) fn preflight_request_count(retained_count: usize) -> Result<(), Error> {
    if retained_count < MAX_QUEUED_JPEG_REQUESTS {
        return Ok(());
    }
    Err(j2k_core::BatchInfrastructureError::AllocationTooLarge {
        what: "JPEG Metal queued request count",
        requested: retained_count.saturating_add(1),
        cap: MAX_QUEUED_JPEG_REQUESTS,
    }
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_scan_work_bound_accepts_exact_and_rejects_one_more() {
        preflight_request_count(MAX_QUEUED_JPEG_REQUESTS - 1)
            .expect("the final admitted request stays within the work bound");
        assert!(matches!(
            preflight_request_count(MAX_QUEUED_JPEG_REQUESTS),
            Err(Error::BatchInfrastructure(
                j2k_core::BatchInfrastructureError::AllocationTooLarge {
                    what: "JPEG Metal queued request count",
                    requested,
                    cap: MAX_QUEUED_JPEG_REQUESTS,
                }
            )) if requested == MAX_QUEUED_JPEG_REQUESTS + 1
        ));
    }
}
