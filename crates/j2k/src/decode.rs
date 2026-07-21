// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::backend::Image;
use crate::J2kError;
use alloc::vec::Vec;
use core::fmt;
use j2k_core::{validate_strided_output_buffer, DecodeOutcome, PixelFormat, Rect, Unsupported};

mod component_handoff;
pub use component_handoff::{
    J2kComponentPlane, J2kDecodedColorSpace, J2kDecodedComponents, J2kDecodedNativeComponents,
    J2kNativeComponentPlane,
};
mod output;
mod settings;
use output::{
    can_decode_u8_directly, write_components_u8_output, write_u16_output, write_u8_output,
};
pub use settings::DecodeSettings;

/// Non-fatal JPEG 2000 decode warning surfaced through decode outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kDecodeWarning {
    /// Decode used lenient settings, so recoverable malformed optional metadata
    /// may be tolerated instead of rejected as it would be in strict mode.
    LenientDecodeMode,
}

// A successful batch can produce at most one warning. Keep that owner
// allocation-free until warning construction is made explicitly fallible and
// the batch metadata planner is updated for non-zero warning storage.
const _: [(); 0] = [(); core::mem::size_of::<J2kDecodeWarning>()];

impl fmt::Display for J2kDecodeWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LenientDecodeMode => f.write_str("lenient decode mode enabled"),
        }
    }
}

pub(crate) type J2kDecodeOutcome = DecodeOutcome<J2kDecodeWarning>;

pub(crate) fn decode_warnings_for_settings(settings: DecodeSettings) -> Vec<J2kDecodeWarning> {
    let mut warnings = Vec::new();
    if settings.lenient_tolerance_enabled() {
        // `J2kDecodeWarning` is statically constrained to a ZST above, so this
        // push cannot enter the allocator.
        warnings.push(J2kDecodeWarning::LenientDecodeMode);
    }
    warnings
}

pub(crate) fn decode_image_into_with_native_context<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let dims = (image.width(), image.height());
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => {
            if can_decode_u8_directly(image.color_space(), image.has_alpha(), dims, stride, fmt) {
                image
                    .decode_into(out, native_context)
                    .map_err(J2kError::from_native_decode_error)?;
                return Ok(());
            }
            let decoded = image
                .decode_with_context(native_context)
                .map_err(J2kError::from_native_decode_error)?;
            write_u8_output(
                image.color_space(),
                image.has_alpha(),
                dims,
                &decoded.data,
                out,
                stride,
                fmt,
            )
        }
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => {
            let raw = image
                .decode_native_with_context(native_context)
                .map_err(J2kError::from_native_decode_error)?;
            write_u16_output(
                image.color_space(),
                image.has_alpha(),
                &raw,
                out,
                stride,
                fmt,
            )
        }
        _ => Err(Unsupported {
            what: "pixel format is not yet supported by j2k",
        }
        .into()),
    }
}

pub(crate) fn decode_image_region_into_with_native_context<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<(), J2kError> {
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => {
            let components = image
                .decode_region_components_with_context((roi.x, roi.y, roi.w, roi.h), native_context)
                .map_err(J2kError::from_native_decode_error)?;
            write_components_u8_output(&components, out, stride, fmt)
        }
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => {
            let raw = image
                .decode_native_region_with_context((roi.x, roi.y, roi.w, roi.h), native_context)
                .map_err(J2kError::from_native_decode_error)?;
            write_u16_output(
                image.color_space(),
                image.has_alpha(),
                &raw,
                out,
                stride,
                fmt,
            )
        }
        _ => Err(Unsupported {
            what: "pixel format is not yet supported by j2k",
        }
        .into()),
    }
}

pub(crate) fn validate_buffer(
    dims: (u32, u32),
    out_len: usize,
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    validate_strided_output_buffer(dims, out_len, stride, fmt).map_err(Into::into)
}

pub(crate) fn validate_region(roi: Rect, dims: (u32, u32)) -> Result<(), J2kError> {
    if roi.is_within(dims) {
        return Ok(());
    }
    Err(J2kError::InvalidRegion {
        x: roi.x,
        y: roi.y,
        w: roi.w,
        h: roi.h,
        image_w: dims.0,
        image_h: dims.1,
    })
}

#[cfg(test)]
mod tests {
    use super::DecodeSettings;
    use super::{decode_warnings_for_settings, J2kDecodeWarning};

    #[test]
    fn decode_warnings_report_lenient_decode_mode() {
        assert_eq!(
            decode_warnings_for_settings(DecodeSettings::default()),
            vec![J2kDecodeWarning::LenientDecodeMode]
        );
        assert!(decode_warnings_for_settings(DecodeSettings::strict()).is_empty());
    }

    #[test]
    fn decode_warning_owner_is_statically_allocation_free() {
        assert_eq!(core::mem::size_of::<J2kDecodeWarning>(), 0);
        let warnings = decode_warnings_for_settings(DecodeSettings::default());
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings
                .capacity()
                .checked_mul(core::mem::size_of::<J2kDecodeWarning>()),
            Some(0)
        );
    }
}
