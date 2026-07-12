// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-allocating option validation shared by exact-i64 routes.

use super::super::{EncodeOptions, NativeEncodePipelineError, NativeEncodePipelineResult};

pub(super) fn validate_high_bit_options(options: &EncodeOptions) -> NativeEncodePipelineResult<()> {
    if options.num_layers == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "quality layer count must be non-zero",
        ));
    }
    if options.num_layers > 32 {
        return Err(NativeEncodePipelineError::unsupported(
            "quality layer count exceeds the encoder limit",
        ));
    }
    if options.write_ppm && options.write_ppt {
        return Err(NativeEncodePipelineError::invalid_input(
            "PPM and PPT packet header markers are mutually exclusive",
        ));
    }
    if matches!(options.tile_part_packet_limit, Some(0)) {
        return Err(NativeEncodePipelineError::invalid_input(
            "tile-part packet limit must be non-zero",
        ));
    }
    if !options.quality_layer_byte_targets.is_empty()
        && options.quality_layer_byte_targets.len() != usize::from(options.num_layers)
    {
        return Err(NativeEncodePipelineError::invalid_input(
            "quality layer byte target count must match quality layer count",
        ));
    }
    if options
        .tile_size
        .is_some_and(|(tile_width, tile_height)| tile_width == 0 || tile_height == 0)
    {
        return Err(NativeEncodePipelineError::invalid_input(
            "invalid tile dimensions",
        ));
    }
    Ok(())
}
