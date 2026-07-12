// SPDX-License-Identifier: MIT OR Apache-2.0

//! Conservative retained-output and worker-scratch model for scalar HTJ2K.

use super::super::coefficient_view::validate_tier1_code_block_geometry;
use super::writers::{MEL_SIZE, MS_SIZE, VLC_SIZE};
use crate::{EncodeError, EncodeResult, DEFAULT_MAX_CODEC_BYTES};

mod refinement;
use refinement::{ht_refinement_allocation, ht_worker_scratch};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HtWorkerAllocation {
    pub(crate) output_bytes: usize,
    pub(crate) scratch_bytes: usize,
    pub(super) cleanup_bytes: usize,
    pub(super) refinement_bytes: usize,
    pub(super) sigma_entries: usize,
    pub(super) previous_sigma_entries: usize,
    pub(super) sigprop_bytes: usize,
    pub(super) magref_bits: usize,
    pub(super) magref_bytes: usize,
}

impl HtWorkerAllocation {
    pub(crate) fn total_bytes(self) -> EncodeResult<usize> {
        checked_add(
            self.output_bytes,
            self.scratch_bytes,
            "HTJ2K Tier-1 worker allocation",
        )
    }
}

pub(crate) fn ht_worker_allocation(
    width: usize,
    height: usize,
    target_coding_passes: u8,
) -> EncodeResult<HtWorkerAllocation> {
    if !(1..=3).contains(&target_coding_passes) {
        return Err(EncodeError::InvalidInput {
            what: "HTJ2K scalar target coding passes must be 1..=3",
        });
    }
    let width_u32 = u32::try_from(width).map_err(|_| EncodeError::InvalidInput {
        what: "HTJ2K code-block width exceeds u32",
    })?;
    let height_u32 = u32::try_from(height).map_err(|_| EncodeError::InvalidInput {
        what: "HTJ2K code-block height exceeds u32",
    })?;
    let coefficients = validate_tier1_code_block_geometry(width, height)?;
    let cleanup_bytes = checked_add(
        checked_add(MEL_SIZE, VLC_SIZE, "HTJ2K cleanup reservoirs")?,
        MS_SIZE,
        "HTJ2K cleanup reservoirs",
    )?;

    let refinement =
        ht_refinement_allocation(width_u32, height_u32, coefficients, target_coding_passes)?;
    let output_bytes = checked_add(
        cleanup_bytes,
        refinement.refinement_bytes,
        "HTJ2K retained block output",
    )?;
    let scratch_bytes = ht_worker_scratch(cleanup_bytes, refinement)?;
    let allocation = HtWorkerAllocation {
        output_bytes,
        scratch_bytes,
        cleanup_bytes,
        refinement_bytes: refinement.refinement_bytes,
        sigma_entries: refinement.sigma_entries,
        previous_sigma_entries: refinement.previous_sigma_entries,
        sigprop_bytes: refinement.sigprop_bytes,
        magref_bits: refinement.magref_bits,
        magref_bytes: refinement.magref_bytes,
    };
    let requested = allocation.total_bytes()?;
    if requested > DEFAULT_MAX_CODEC_BYTES {
        return Err(EncodeError::AllocationTooLarge {
            what: "HTJ2K Tier-1 worker allocation",
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
    use super::ht_worker_allocation;
    use crate::EncodeError;

    #[test]
    fn refinement_plan_adds_only_requested_pass_workspaces() {
        let cleanup = ht_worker_allocation(64, 64, 1).expect("cleanup plan");
        let sigprop = ht_worker_allocation(64, 64, 2).expect("SigProp plan");
        let magref = ht_worker_allocation(64, 64, 3).expect("MagRef plan");
        assert_eq!(cleanup.refinement_bytes, 0);
        assert!(sigprop.refinement_bytes > 0);
        assert!(magref.refinement_bytes > sigprop.refinement_bytes);
        assert!(cleanup.scratch_bytes < sigprop.scratch_bytes);
        assert!(sigprop.scratch_bytes < magref.scratch_bytes);
    }

    #[test]
    fn invalid_pass_count_is_typed() {
        assert_eq!(
            ht_worker_allocation(1, 1, 0),
            Err(EncodeError::InvalidInput {
                what: "HTJ2K scalar target coding passes must be 1..=3",
            })
        );
    }
}
