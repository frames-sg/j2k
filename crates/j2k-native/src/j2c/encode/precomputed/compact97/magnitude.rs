// SPDX-License-Identifier: MIT OR Apache-2.0

//! Magnitude-bound aggregation from validated compact code-block metadata.

use super::{NativeEncodePipelineResult, PreencodedHtj2k97CompactImage};
use crate::j2c::capabilities::required_magnitude_bound;
use crate::j2c::encode::magnitude::{
    cleanup_magnitude_upper_bound, decomposition_level_for_resolution,
};

pub(super) fn required_compact_magnitude_bound(
    image: &PreencodedHtj2k97CompactImage,
    num_decomposition_levels: u8,
) -> NativeEncodePipelineResult<Option<u8>> {
    let mut required = None::<u8>;
    for component in &image.components {
        for (resolution, packet) in component.resolutions.iter().enumerate() {
            let decomposition_level =
                decomposition_level_for_resolution(resolution, num_decomposition_levels)?;
            for subband in &packet.subbands {
                for block in &subband.code_blocks {
                    let maximum = cleanup_magnitude_upper_bound(
                        subband.total_bitplanes,
                        block.num_zero_bitplanes,
                        block.num_coding_passes,
                    );
                    let bound = required_magnitude_bound(maximum, false, decomposition_level);
                    required = Some(required.map_or(bound, |prior| prior.max(bound)));
                }
            }
        }
    }
    Ok(required)
}
