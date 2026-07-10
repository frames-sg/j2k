// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    should_use_resident_htj2k_host_shape_for_auto, Buffer, EncodeBackendPreference,
    J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, MetalEncodeInputStaging,
    MetalLosslessEncodeTile, PixelFormat, ReversibleTransform,
};

#[cfg(target_os = "macos")]
const AUTO_HIGH_THROUGHPUT_RESIDENT_HOST_OUTPUT_RGB8_MIN_PIXELS: usize = 1024 * 1024;

#[cfg(target_os = "macos")]
pub(super) fn should_try_resident_lossless_host_encode(options: J2kLosslessEncodeOptions) -> bool {
    options.backend == EncodeBackendPreference::RequireDevice
}

#[cfg(target_os = "macos")]
pub(super) fn should_try_resident_lossless_host_encode_for_tiles(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: J2kLosslessEncodeOptions,
    staging: MetalEncodeInputStaging,
) -> bool {
    if should_try_resident_lossless_host_encode(options) {
        return true;
    }
    options.backend == EncodeBackendPreference::Auto
        && !tiles.is_empty()
        && tiles.iter().all(|&tile| {
            should_try_auto_resident_lossless_host_encode(tile, options, staging, tiles.len())
        })
}

#[cfg(target_os = "macos")]
pub(super) fn should_try_auto_resident_lossless_host_encode(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    staging: MetalEncodeInputStaging,
    batch_size: usize,
) -> bool {
    options.backend == EncodeBackendPreference::Auto
        && options.block_coding_mode == J2kBlockCodingMode::HighThroughput
        && matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && should_try_auto_resident_lossless_host_format(
            tile.format,
            options.reversible_transform,
            batch_size,
            tile.output_width,
            tile.output_height,
        )
}

#[cfg(target_os = "macos")]
pub(super) fn should_try_auto_resident_lossless_host_format(
    format: PixelFormat,
    reversible_transform: ReversibleTransform,
    batch_size: usize,
    output_width: u32,
    output_height: u32,
) -> bool {
    let pixels = (output_width as usize).saturating_mul(output_height as usize);
    match format {
        PixelFormat::Gray8 => {
            batch_size > 1
                && should_use_resident_htj2k_host_shape_for_auto(output_width, output_height)
        }
        PixelFormat::Rgb8 => {
            batch_size > 1
                && reversible_transform == ReversibleTransform::Rct53
                && pixels >= AUTO_HIGH_THROUGHPUT_RESIDENT_HOST_OUTPUT_RGB8_MIN_PIXELS
        }
        _ => false,
    }
}

#[cfg(target_os = "macos")]
pub(super) fn host_output_encode_options(
    mut options: J2kLosslessEncodeOptions,
) -> J2kLosslessEncodeOptions {
    options.validation = J2kEncodeValidation::External;
    options
}

#[cfg(target_os = "macos")]
pub(super) fn borrow_padded_metal_buffer_from_bytes(
    session: &crate::MetalBackendSession,
    bytes: &[u8],
) -> Result<Buffer, crate::Error> {
    if bytes.is_empty() {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal hybrid encode input is empty".to_string(),
        });
    }
    Ok(session.device().new_buffer_with_bytes_no_copy(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
        None,
    ))
}
