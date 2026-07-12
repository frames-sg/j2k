// SPDX-License-Identifier: MIT OR Apache-2.0

//! Conservative allocation model for one classic Tier-1 worker.
//!
//! The model deliberately separates retained output from worker-local scratch.
//! A driver can therefore account every block output for the lifetime of the
//! parallel phase while charging scratch only for simultaneously active workers.

use core::mem::size_of;

use crate::{EncodeError, EncodeResult, DEFAULT_MAX_CODEC_BYTES};

use super::super::coefficient_view::validate_tier1_code_block_geometry;
use super::EncodedCodeBlockSegment;

/// Maximum number of magnitude bitplanes representable by the classic scalar
/// encoder's `u64` magnitude storage.
const MAX_CLASSIC_BITPLANES: u8 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClassicWorkerAllocation {
    /// Heap bytes retained by the encoded block after the worker returns.
    pub(crate) output_bytes: usize,
    /// Maximum additional heap bytes live while this worker is encoding.
    pub(crate) scratch_bytes: usize,
    pub(super) padded_coefficients: usize,
    pub(super) payload_bytes: usize,
    pub(super) coding_passes: usize,
}

impl ClassicWorkerAllocation {
    pub(crate) fn total_bytes(self) -> EncodeResult<usize> {
        checked_add(
            self.output_bytes,
            self.scratch_bytes,
            "classic Tier-1 worker allocation",
        )
    }
}

/// Plan a schedule-independent upper bound for one classic Tier-1 block.
///
/// Each coefficient contributes at most three MQ symbols per bitplane: one
/// significance decision, one optional sign/refinement decision, and one
/// conservative run-mode allowance. Four more symbols cover SEGSYM. An MQ
/// symbol can renormalize through at most three byte-out boundaries; four bytes
/// per symbol leaves one full boundary of margin. Four bytes per possible
/// coding-pass termination plus a fixed finalization allowance cover terminated
/// and selectively bypassed segment boundaries.
pub(crate) fn classic_worker_allocation(
    width: usize,
    height: usize,
    total_bitplanes: u8,
) -> EncodeResult<ClassicWorkerAllocation> {
    if total_bitplanes > MAX_CLASSIC_BITPLANES {
        return Err(EncodeError::InvalidInput {
            what: "classic Tier-1 bitplane count exceeds u64 magnitude precision",
        });
    }

    let coefficients = validate_tier1_code_block_geometry(width, height)?;
    let padded_width = width
        .checked_add(2)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "classic Tier-1 padded width",
        })?;
    let padded_height = height
        .checked_add(2)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "classic Tier-1 padded height",
        })?;
    let padded_coefficients = checked_mul(
        padded_width,
        padded_height,
        "classic Tier-1 padded coefficient count",
    )?;
    let bitplanes = usize::from(total_bitplanes);
    let coding_passes = if bitplanes == 0 {
        0
    } else {
        checked_add(
            1,
            checked_mul(3, bitplanes - 1, "classic Tier-1 coding-pass count")?,
            "classic Tier-1 coding-pass count",
        )?
    };

    let symbols_per_bitplane = checked_add(
        checked_mul(3, coefficients, "classic Tier-1 symbols per bitplane")?,
        4,
        "classic Tier-1 segmentation symbols",
    )?;
    let symbols = checked_mul(
        symbols_per_bitplane,
        bitplanes,
        "classic Tier-1 symbol bound",
    )?;
    let payload_bytes = checked_add(
        checked_add(
            checked_mul(4, symbols, "classic Tier-1 payload bound")?,
            checked_mul(4, coding_passes, "classic Tier-1 termination bound")?,
            "classic Tier-1 terminated payload bound",
        )?,
        16,
        "classic Tier-1 final payload bound",
    )?;

    let segment_metadata_bytes = checked_mul(
        coding_passes,
        size_of::<EncodedCodeBlockSegment>(),
        "classic Tier-1 segment metadata",
    )?;
    let output_bytes = checked_add(
        payload_bytes,
        segment_metadata_bytes,
        "classic Tier-1 retained output",
    )?;

    let padded_storage_bytes = checked_mul(
        padded_coefficients,
        size_of::<u64>() + size_of::<u8>() * 2 + size_of::<usize>(),
        "classic Tier-1 padded scratch",
    )?;
    let active_segment_bytes =
        payload_bytes
            .checked_add(1)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "classic Tier-1 active segment scratch",
            })?;
    let scratch_bytes = checked_add(
        padded_storage_bytes,
        active_segment_bytes,
        "classic Tier-1 worker scratch",
    )?;

    let allocation = ClassicWorkerAllocation {
        output_bytes,
        scratch_bytes,
        padded_coefficients,
        payload_bytes,
        coding_passes,
    };
    let requested = allocation.total_bytes()?;
    if requested > DEFAULT_MAX_CODEC_BYTES {
        return Err(EncodeError::AllocationTooLarge {
            what: "classic Tier-1 worker allocation",
            requested,
            cap: DEFAULT_MAX_CODEC_BYTES,
        });
    }
    Ok(allocation)
}

fn checked_add(left: usize, right: usize, what: &'static str) -> EncodeResult<usize> {
    left.checked_add(right)
        .ok_or(EncodeError::ArithmeticOverflow { what })
}

fn checked_mul(left: usize, right: usize, what: &'static str) -> EncodeResult<usize> {
    left.checked_mul(right)
        .ok_or(EncodeError::ArithmeticOverflow { what })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_plan_rejects_sample_count_overflow() {
        assert_eq!(
            classic_worker_allocation(usize::MAX, 2, 1),
            Err(EncodeError::ArithmeticOverflow {
                what: "Tier-1 code-block sample count",
            })
        );
    }

    #[test]
    fn worker_plan_separates_retained_output_from_scratch() {
        let plan = classic_worker_allocation(64, 64, 31).expect("valid Part 1 code-block plan");
        assert!(plan.output_bytes > plan.payload_bytes);
        assert!(plan.scratch_bytes > plan.padded_coefficients);
        assert_eq!(
            plan.total_bytes().expect("plan total fits"),
            plan.output_bytes + plan.scratch_bytes
        );
    }
}
