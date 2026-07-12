// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-allocating request checks plus fallible component-sampling materialization.

use crate::j2c::encode::{
    raw_pixel_bytes_per_sample, validate_code_block_geometry, validate_component_sample_info,
    validate_irreversible_quantization_profile, validate_reversible_i64_encode_options,
    BlockCodingMode, CodeBlockGeometry, EncodeComponentSampleInfo, EncodeOptions,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
    MAX_J2K_SPEC_COMPONENTS, MAX_PART1_SAMPLE_BIT_DEPTH, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};

use super::construction::{try_component_sampling, validate_component_sampling};
use super::{ValidatedEncodeRoute, ValidatedSingleTileInput};

#[expect(
    clippy::too_many_arguments,
    reason = "raw-pixel validation keeps the byte extent, image geometry, component metadata, and encode policy explicit"
)]
pub(in crate::j2c::encode::single_tile) fn validate_encode_request(
    pixels_len: usize,
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    component_sample_info: &[EncodeComponentSampleInfo],
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<ValidatedEncodeRoute> {
    let (num_pixels, code_block_geometry) = validate_encode_request_header(
        width,
        height,
        num_components,
        bit_depth,
        options,
        component_sample_info,
    )?;
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)
        .map_err(NativeEncodePipelineError::internal_invariant)?;
    let expected_len = num_pixels
        .checked_mul(usize::from(num_components))
        .and_then(|len| len.checked_mul(bytes_per_sample))
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("raw encode byte length"))?;
    if pixels_len < expected_len {
        return Err(NativeEncodePipelineError::invalid_input(
            "pixel data too short",
        ));
    }

    finish_validate_encode_request(
        num_pixels,
        width,
        height,
        num_components,
        bit_depth,
        options,
        block_coding_mode,
        component_sample_info,
        code_block_geometry,
        session,
    )
}

pub(in crate::j2c::encode::single_tile) struct NonPixelSingleTileRequest<'a, 'input> {
    pub(in crate::j2c::encode::single_tile) width: u32,
    pub(in crate::j2c::encode::single_tile) height: u32,
    pub(in crate::j2c::encode::single_tile) num_components: u16,
    pub(in crate::j2c::encode::single_tile) bit_depth: u8,
    pub(in crate::j2c::encode::single_tile) options: &'a EncodeOptions,
    pub(in crate::j2c::encode::single_tile) block_coding_mode: BlockCodingMode,
    pub(in crate::j2c::encode::single_tile) component_sample_info:
        &'a [EncodeComponentSampleInfo],
    pub(in crate::j2c::encode::single_tile) multi_tile_error: &'static str,
    pub(in crate::j2c::encode::single_tile) session: &'a NativeEncodeSession<'input>,
}

pub(in crate::j2c::encode::single_tile) fn validate_non_pixel_single_tile_request(
    request: &NonPixelSingleTileRequest<'_, '_>,
) -> NativeEncodePipelineResult<ValidatedSingleTileInput> {
    let (num_pixels, code_block_geometry) = validate_encode_request_header(
        request.width,
        request.height,
        request.num_components,
        request.bit_depth,
        request.options,
        request.component_sample_info,
    )?;
    match finish_validate_encode_request(
        num_pixels,
        request.width,
        request.height,
        request.num_components,
        request.bit_depth,
        request.options,
        request.block_coding_mode,
        request.component_sample_info,
        code_block_geometry,
        request.session,
    )? {
        ValidatedEncodeRoute::SingleTile(validated) => Ok(validated),
        ValidatedEncodeRoute::MultiTile { .. } => Err(NativeEncodePipelineError::unsupported(
            request.multi_tile_error,
        )),
    }
}

fn validate_encode_request_header(
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
) -> NativeEncodePipelineResult<(usize, CodeBlockGeometry)> {
    if width == 0 || height == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "invalid dimensions",
        ));
    }
    if num_components == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "component count must be non-zero",
        ));
    }
    if num_components > MAX_J2K_SPEC_COMPONENTS {
        return Err(NativeEncodePipelineError::unsupported(
            "component count exceeds the JPEG 2000 Part 1 limit",
        ));
    }
    if bit_depth == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "bit depth must be non-zero",
        ));
    }
    if bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH {
        return Err(NativeEncodePipelineError::unsupported(
            "bit depth exceeds the JPEG 2000 Part 1 limit",
        ));
    }
    let code_block_geometry =
        validate_code_block_geometry(options).map_err(NativeEncodePipelineError::invalid_input)?;
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
    if !options.reversible {
        validate_irreversible_quantization_profile(options)
            .map_err(NativeEncodePipelineError::invalid_input)?;
    }
    validate_component_sample_info(component_sample_info, usize::from(num_components))
        .map_err(NativeEncodePipelineError::invalid_input)?;

    let num_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("image pixel count"))?;
    Ok((num_pixels, code_block_geometry))
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps validated image geometry and encode policy explicit"
)]
fn finish_validate_encode_request(
    num_pixels: usize,
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    component_sample_info: &[EncodeComponentSampleInfo],
    code_block_geometry: CodeBlockGeometry,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<ValidatedEncodeRoute> {
    validate_component_sampling(options, num_components)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let component_sampling = options.component_sampling.as_deref().unwrap_or(&[]);
    let high_bit_exact = bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH;
    if high_bit_exact && options.reversible {
        validate_reversible_i64_encode_options(
            options,
            block_coding_mode,
            component_sample_info,
            component_sampling,
        )
        .map_err(NativeEncodePipelineError::unsupported)?;
    }

    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width == 0 || tile_height == 0 {
            return Err(NativeEncodePipelineError::invalid_input(
                "invalid tile dimensions",
            ));
        }
        if component_sampling
            .iter()
            .any(|sampling| *sampling != (1, 1))
        {
            return Err(NativeEncodePipelineError::unsupported(
                "multi-tile encode with component sampling is not implemented",
            ));
        }
        if tile_width < width || tile_height < height {
            return Ok(ValidatedEncodeRoute::MultiTile {
                tile_width,
                tile_height,
            });
        }
    }

    let component_sampling = try_component_sampling(options, num_components, session)?;
    Ok(ValidatedEncodeRoute::SingleTile(ValidatedSingleTileInput {
        num_pixels,
        component_sampling,
        high_bit_exact,
        code_block_geometry,
    }))
}
