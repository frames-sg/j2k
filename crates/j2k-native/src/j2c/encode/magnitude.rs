// SPDX-License-Identifier: MIT OR Apache-2.0

//! HT cleanup-magnitude aggregation before prepared coefficients are consumed.

use super::{
    ht_block_encode, BlockCodingMode, NativeEncodePipelineError, NativeEncodePipelineResult,
    PreparedCodeBlockCoefficients, PreparedEncodeSubband, PreparedResolutionPacket,
};
use crate::j2c::capabilities::required_magnitude_bound;

pub(super) fn decomposition_level_for_resolution(
    resolution: impl TryInto<u8>,
    num_decomposition_levels: u8,
) -> NativeEncodePipelineResult<u8> {
    let resolution = resolution.try_into().map_err(|_| {
        NativeEncodePipelineError::internal_invariant(
            "resolution index exceeds the HT magnitude-bound domain",
        )
    })?;
    if resolution == 0 {
        Ok(num_decomposition_levels)
    } else {
        num_decomposition_levels
            .checked_sub(resolution - 1)
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "resolution exceeds the decomposition count",
                )
            })
    }
}

pub(super) fn required_ht_magnitude_bound<'a>(
    packets: impl IntoIterator<Item = &'a PreparedResolutionPacket>,
    num_decomposition_levels: u8,
    reversible: bool,
) -> NativeEncodePipelineResult<Option<u8>> {
    let mut required = None::<u8>;
    for packet in packets {
        let decomposition_level =
            decomposition_level_for_resolution(packet.resolution, num_decomposition_levels)?;
        for subband in &packet.subbands {
            if subband.block_coding_mode != BlockCodingMode::HighThroughput {
                continue;
            }
            let maximum = maximum_cleanup_magnitude(subband);
            let subband_required =
                required_magnitude_bound(maximum, reversible, decomposition_level);
            required = Some(required.map_or(subband_required, |bound| bound.max(subband_required)));
        }
    }
    Ok(required)
}

pub(super) fn cleanup_magnitude_upper_bound(
    total_bitplanes: u8,
    num_zero_bitplanes: u8,
    num_coding_passes: u8,
) -> u64 {
    let cleanup_bitplanes = total_bitplanes
        .saturating_sub(num_zero_bitplanes)
        .saturating_sub(u8::from(num_coding_passes >= 2));
    if u32::from(cleanup_bitplanes) >= u64::BITS {
        u64::MAX
    } else if cleanup_bitplanes == 0 {
        0
    } else {
        (1_u64 << cleanup_bitplanes) - 1
    }
}

fn maximum_cleanup_magnitude(subband: &PreparedEncodeSubband) -> u64 {
    let refinement_shift = u8::from(
        ht_block_encode::effective_coding_passes(
            subband.total_bitplanes,
            subband.ht_target_coding_passes,
        ) >= 2,
    );
    let mut maximum = 0_u64;
    let mut exact = subband.preencoded_ht_code_blocks.is_none();
    for block in &subband.code_blocks {
        match &block.coefficients {
            PreparedCodeBlockCoefficients::I32(values) => {
                maximum = maximum.max(
                    values
                        .iter()
                        .map(|value| u64::from(value.unsigned_abs()))
                        .max()
                        .unwrap_or(0),
                );
            }
            PreparedCodeBlockCoefficients::I64(values) => {
                maximum = maximum.max(
                    values
                        .iter()
                        .map(|value| value.unsigned_abs())
                        .max()
                        .unwrap_or(0),
                );
            }
            PreparedCodeBlockCoefficients::Empty => exact = false,
        }
    }
    maximum >>= u32::from(refinement_shift);
    if let Some(exact) = subband.preencoded_ht_maximum_cleanup_magnitude {
        return maximum.max(exact);
    }
    if let Some(encoded_blocks) = &subband.preencoded_ht_code_blocks {
        let conservative = encoded_blocks
            .iter()
            .map(|block| {
                cleanup_magnitude_upper_bound(
                    subband.total_bitplanes,
                    block.num_zero_bitplanes,
                    block.num_coding_passes,
                )
            })
            .max()
            .unwrap_or(0);
        return maximum.max(conservative);
    }
    if exact {
        return maximum;
    }

    let cleanup_bitplanes = subband.total_bitplanes.saturating_sub(refinement_shift);
    let conservative = if u32::from(cleanup_bitplanes) >= u64::BITS {
        u64::MAX
    } else if cleanup_bitplanes == 0 {
        0
    } else {
        (1_u64 << cleanup_bitplanes) - 1
    };
    maximum.max(conservative)
}
