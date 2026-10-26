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

/// Most bits one MQ symbol can renormalize: A stays below 0x10000 and every
/// coded sub-interval is at least 1, since 1 <= Qe and A - Qe >= 0x29FF.
const MAX_SHIFTS_PER_SYMBOL: usize = 15;

/// Fewest code bits in one output byte: seven after a stuffed 0xFF.
const MIN_BITS_PER_BYTE: usize = 7;

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
/// The bound depends only on the sequence of coded symbols and output bytes,
/// so it holds for both pass implementations: the column-word passes code the
/// same symbols in the same order as the per-coefficient passes, which
/// `column_word_passes_match_neighbor_byte_passes` checks.
///
/// Symbols per bitplane, n = w * h coefficients: every coefficient is coded in
/// exactly one of the three passes, and outside run mode with at most two
/// symbols, a significance or refinement decision and a sign, a baseline of
/// 2n. Run mode applies only to a full four-row stripe column whose word is
/// zero, never to a partial bottom stripe. Such a group codes at most ten
/// symbols, the run symbol, two uniform position symbols, the first sign and
/// a decision and sign for each of the three rows below, 2 above its baseline
/// of 8; a fresh 1x4 block of ones at one bitplane codes all ten. Groups are
/// disjoint and there are at most w * floor(h / 4) of them, so run mode adds at
/// most 2 * w * floor(h / 4) <= floor(n / 2). SEGSYM adds four per cleanup
/// pass. The first bitplane has only the cleanup pass, so the per-bitplane
/// count holds for all Mb of them.
///
/// MQ bytes: between symbols 0x8000 <= A < 0x10000, and 1 <= Qe <= 0x5601, so
/// A - Qe >= 0x29FF > 0 on every branch, the conditional exchange included,
/// and the coded sub-interval is at least 1: one symbol renormalizes by at
/// most 15 shifts. BYTEOUT runs when CT reaches 0; CT starts at 12 and is reset
/// to 8, or 7 after a stuffed 0xFF, so a codeword of S shifts runs it at most
/// k <= floor(S / 7) times while coding. FLUSH calls it exactly twice. Each
/// call appends one byte, a carry only increments the byte before, and
/// finishing can only drop a terminal 0xFF, so the bound also covers the
/// bytes held before that drop. A raw bypass symbol is one bit; the raw writer
/// emits a byte per 8 bits, or 7 after 0xFF, plus at most one flush byte.
///
/// Termination: every segment is at least one coding pass, so the two flush
/// bytes of each MQ segment and the one of each raw segment fit within four
/// bytes per coding pass plus the fixed allowance. No predictable or ERTERM
/// termination path exists; its three BYTEOUT calls per T.800 D.4.2 would
/// still fit within four bytes per pass, but this proof must be revisited if
/// one is added.
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
        checked_add(
            checked_mul(2, coefficients, "classic Tier-1 symbols per bitplane")?,
            coefficients / 2,
            "classic Tier-1 run-length symbols",
        )?,
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
            checked_mul(
                MAX_SHIFTS_PER_SYMBOL,
                symbols,
                "classic Tier-1 payload bound",
            )?
            .div_ceil(MIN_BITS_PER_BYTE),
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

    // A u64 magnitude and two state bytes per padded sample. The column-word
    // path needs less: 16 bytes per column of each four-row stripe plus one
    // u32 per padded stripe column, so per padded column at most
    // 5 * height + 23 bytes against 10 * height + 20 here, for any height.
    let padded_storage_bytes = checked_mul(
        padded_coefficients,
        size_of::<u64>() + size_of::<u8>() * 2,
        "classic Tier-1 padded scratch",
    )?;
    let active_segment_bytes =
        checked_add(payload_bytes, 1, "classic Tier-1 active segment scratch")?;
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
