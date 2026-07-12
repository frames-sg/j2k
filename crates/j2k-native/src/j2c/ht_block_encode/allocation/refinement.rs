// SPDX-License-Identifier: MIT OR Apache-2.0

//! Refinement-output and scratch planning for one scalar HTJ2K worker.

use core::mem::size_of;

use super::{checked_add, checked_mul};
use crate::j2c::ht_block_decode::sigma_stride;
use crate::{EncodeError, EncodeResult};

#[derive(Debug, Clone, Copy)]
pub(super) struct HtRefinementAllocation {
    pub(super) refinement_bytes: usize,
    pub(super) sigma_entries: usize,
    pub(super) previous_sigma_entries: usize,
    pub(super) sigprop_bytes: usize,
    pub(super) magref_bits: usize,
    pub(super) magref_bytes: usize,
}

pub(super) fn ht_refinement_allocation(
    width: u32,
    height: u32,
    coefficients: usize,
    target_coding_passes: u8,
) -> EncodeResult<HtRefinementAllocation> {
    let refinement_enabled = target_coding_passes > 1;
    let sigma_rows = usize::try_from(height.div_ceil(4))
        .map_err(|_| EncodeError::ArithmeticOverflow {
            what: "HTJ2K sigma rows",
        })?
        .checked_add(1)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K sigma rows",
        })?;
    let sigma_entries = if refinement_enabled {
        checked_mul(sigma_rows, sigma_stride(width), "HTJ2K sigma entries")?
    } else {
        0
    };
    let previous_sigma_entries = if refinement_enabled {
        usize::try_from(width.div_ceil(4))
            .map_err(|_| EncodeError::ArithmeticOverflow {
                what: "HTJ2K previous sigma entries",
            })?
            .checked_add(8)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "HTJ2K previous sigma entries",
            })?
    } else {
        0
    };

    let sigprop_bits = if refinement_enabled {
        checked_mul(2, coefficients, "HTJ2K SigProp bit bound")?
    } else {
        0
    };
    let sigprop_bytes = stuffed_bytes(sigprop_bits)?;
    let magref_bits = if target_coding_passes > 2 {
        coefficients
    } else {
        0
    };
    let magref_bytes = stuffed_bytes(magref_bits)?;
    let refinement_bytes =
        checked_add(sigprop_bytes, magref_bytes, "HTJ2K refinement output bound")?;
    Ok(HtRefinementAllocation {
        refinement_bytes,
        sigma_entries,
        previous_sigma_entries,
        sigprop_bytes,
        magref_bits,
        magref_bytes,
    })
}

pub(super) fn ht_worker_scratch(
    cleanup_bytes: usize,
    refinement: HtRefinementAllocation,
) -> EncodeResult<usize> {
    // Deliberately conservative: several phase-disjoint owners are summed so
    // drivers do not depend on the worker's internal sequencing.
    let sigma_bytes = checked_mul(
        refinement.sigma_entries,
        size_of::<u16>(),
        "HTJ2K sigma allocation",
    )?;
    let previous_sigma_bytes = checked_mul(
        refinement.previous_sigma_entries,
        size_of::<u16>(),
        "HTJ2K previous sigma allocation",
    )?;
    [
        cleanup_bytes,
        cleanup_bytes,
        sigma_bytes,
        previous_sigma_bytes,
        refinement.sigprop_bytes,
        refinement.magref_bits * size_of::<bool>(),
        refinement.magref_bytes,
        refinement.refinement_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_add(total, bytes, "HTJ2K worker scratch")
    })
}

fn stuffed_bytes(bits: usize) -> EncodeResult<usize> {
    if bits == 0 {
        return Ok(0);
    }
    bits.checked_add(6)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K stuffed refinement byte bound",
        })
        .map(|rounded| rounded / 7)
}
