// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use super::{
    Buffer, EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, MetalEncodeInputStaging, MetalLosslessEncodeTile, PixelFormat,
    ReversibleTransform,
};

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
    match format {
        PixelFormat::Gray8 => {
            batch_size > 1
                && crate::generated::promotion::auto_host_output_encode_qualifies(
                    1,
                    output_width,
                    output_height,
                )
        }
        PixelFormat::Rgb8 => {
            batch_size > 1
                && reversible_transform == ReversibleTransform::Rct53
                && crate::generated::promotion::auto_host_output_encode_qualifies(
                    3,
                    output_width,
                    output_height,
                )
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
pub(super) fn copy_padded_metal_buffer_from_bytes(
    session: &crate::MetalBackendSession,
    bytes: &[u8],
) -> Result<Buffer, crate::Error> {
    if bytes.is_empty() {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal hybrid encode input is empty".to_string(),
        });
    }
    j2k_metal_support::checked_shared_buffer_with_slice(session.device(), bytes).map_err(|source| {
        crate::error::metal_kernel_support_error("J2K Metal hybrid encode input", source)
    })
}
