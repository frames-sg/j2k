// SPDX-License-Identifier: MIT OR Apache-2.0

use super::invalid;
use crate::{error::CudaError, jpeg::CudaJpegEntropyCheckpoint};

pub(super) fn validate_entropy_checkpoints(
    checkpoints: &[CudaJpegEntropyCheckpoint],
    entropy_len: u32,
    total_mcus: u32,
) -> Result<(), CudaError> {
    validate_complete_mcu_partition(checkpoints, total_mcus)?;
    validate_initial_state(checkpoints)?;

    let mut previous_consumed_bits = None;
    for (index, checkpoint) in checkpoints.iter().copied().enumerate() {
        if checkpoint.reserved != 0 {
            return Err(invalid(format_args!(
                "entropy checkpoint {index} has nonzero reserved state"
            )));
        }
        if checkpoint.entropy_pos > entropy_len {
            return Err(invalid(format_args!(
                "entropy checkpoint {index} is beyond the entropy payload"
            )));
        }
        if checkpoint.bit_count > 63 {
            return Err(invalid(format_args!(
                "entropy checkpoint {index} has more than 63 buffered bits"
            )));
        }
        let loaded_bits = u64::from(checkpoint.entropy_pos) * 8;
        if u64::from(checkpoint.bit_count) > loaded_bits {
            return Err(invalid(format_args!(
                "entropy checkpoint {index} buffers bits before the entropy payload starts"
            )));
        }
        let unused_bits = 64 - checkpoint.bit_count;
        let unused_mask = if unused_bits == 64 {
            u64::MAX
        } else {
            (1u64 << unused_bits) - 1
        };
        if checkpoint.bit_acc & unused_mask != 0 {
            return Err(invalid(format_args!(
                "entropy checkpoint {index} has nonzero unused accumulator bits"
            )));
        }
        let consumed_bits = loaded_bits - u64::from(checkpoint.bit_count);
        if previous_consumed_bits.is_some_and(|previous| consumed_bits <= previous) {
            return Err(invalid(format_args!(
                "entropy checkpoint {index} does not advance through the entropy payload"
            )));
        }
        previous_consumed_bits = Some(consumed_bits);
    }
    Ok(())
}

fn validate_complete_mcu_partition(
    checkpoints: &[CudaJpegEntropyCheckpoint],
    total_mcus: u32,
) -> Result<(), CudaError> {
    let Some(first) = checkpoints.first() else {
        return Err(invalid("decode requires at least one entropy checkpoint"));
    };
    if first.mcu_index != 0 {
        return Err(invalid("first entropy checkpoint must start at MCU zero"));
    }
    let Some(last) = checkpoints.last() else {
        return Err(invalid("decode requires at least one entropy checkpoint"));
    };
    if last.mcu_index >= total_mcus {
        return Err(invalid(format_args!(
            "entropy checkpoint {} starts beyond the MCU range",
            checkpoints.len() - 1
        )));
    }

    // Every decode range is [checkpoint[i], checkpoint[i + 1]); the final
    // range ends at total_mcus. A zero first boundary and strictly increasing
    // shared boundaries therefore prove complete, non-overlapping coverage.
    for (index, pair) in checkpoints.windows(2).enumerate() {
        let start_mcu = pair[0].mcu_index;
        let end_mcu = pair[1].mcu_index;
        if start_mcu >= end_mcu {
            return Err(invalid(format_args!(
                "entropy checkpoint {} is not strictly MCU-ordered",
                index + 1
            )));
        }
        if end_mcu >= total_mcus {
            return Err(invalid(format_args!(
                "entropy checkpoint {} starts beyond the MCU range",
                index + 1
            )));
        }
    }
    Ok(())
}

fn validate_initial_state(checkpoints: &[CudaJpegEntropyCheckpoint]) -> Result<(), CudaError> {
    let Some(first) = checkpoints.first() else {
        return Err(invalid("decode requires at least one entropy checkpoint"));
    };
    if first.entropy_pos != 0
        || first.bit_acc != 0
        || first.bit_count != 0
        || first.y_prev_dc != 0
        || first.cb_prev_dc != 0
        || first.cr_prev_dc != 0
    {
        return Err(invalid(
            "first entropy checkpoint must contain the initial decoder state",
        ));
    }
    Ok(())
}
