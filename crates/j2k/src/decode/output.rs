// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pixel-layout conversion for decoded native output.

mod native;
mod u16;
mod u8;

use crate::backend::ColorSpace;
use crate::J2kError;
use j2k_core::PixelFormat;

pub(crate) use native::{
    decode_native_components_with_native_context, native_components_route,
    native_region_components_route, NativeRoute,
};
use native::{planes_pack_floor, region_rows_preferred};

pub(super) use u16::{
    region_rows_preferred_u16, rows_match_native_u16, u16_rows_pack_floor,
    write_components_u16_output, write_plane_rows_u16, write_u16_output, write_u16_output_rows,
};
pub(super) use u8::{
    can_decode_u8_directly, expand_rgb8_to_rgba8, resolve_u8_color_space,
    write_components_u8_output, write_plane_rows_u8, write_u8_output,
};

/// Shared destination and error plumbing for the 8- and 16-bit component
/// row sinks. Its pixel format selects the matching plane writer. The 16-bit
/// path matches [`write_u16_output`] through [`rows_match_native_u16`], and
/// the first write error prevents every later write.
pub(in crate::decode) struct ComponentRowOutput<'o> {
    color_space: &'o ColorSpace,
    has_alpha: bool,
    out: &'o mut [u8],
    stride: usize,
    fmt: PixelFormat,
    error: Option<J2kError>,
}

impl<'o> ComponentRowOutput<'o> {
    /// `out` must already hold the region at `stride` in `fmt`.
    pub(in crate::decode) fn new(
        color_space: &'o ColorSpace,
        has_alpha: bool,
        out: &'o mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Self {
        Self {
            color_space: resolve_u8_color_space(color_space, fmt),
            has_alpha,
            out,
            stride,
            fmt,
            error: None,
        }
    }

    pub(in crate::decode) fn finish(self) -> Result<(), J2kError> {
        self.error.map_or(Ok(()), Err)
    }

    /// The outcome of a region decode into this sink that returned
    /// `decoded`: the sink's own error first, which also stops the decode
    /// (see `ComponentRowSink::stopped`), then the decode's.
    pub(in crate::decode) fn finish_decode(
        self,
        decoded: j2k_native::Result<()>,
    ) -> Result<(), J2kError> {
        if self.error.is_some() {
            return self.finish();
        }
        decoded.map_err(J2kError::from_native_decode_error)?;
        self.finish()
    }
}

pub(in crate::decode) type U8RowSink<'o> = ComponentRowOutput<'o>;
pub(in crate::decode) type U16RowSink<'o> = ComponentRowOutput<'o>;

impl j2k_native::ComponentRowSink for ComponentRowOutput<'_> {
    fn stopped(&self) -> bool {
        self.error.is_some()
    }

    fn write_rows(&mut self, x: u32, y: u32, planes: &[j2k_native::ComponentPlane<'_>]) {
        if self.error.is_some() {
            return;
        }
        let dims = planes
            .first()
            .map_or((0, 0), j2k_native::ComponentPlane::dimensions);
        let offset = (y as usize)
            .checked_mul(self.stride)
            .and_then(|row| row.checked_add((x as usize).checked_mul(self.fmt.bytes_per_pixel())?));
        let result = match offset.and_then(|offset| self.out.get_mut(offset..)) {
            Some(out) => match self.fmt {
                PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8 => u8::write_planes_u8(
                    self.color_space,
                    self.has_alpha,
                    planes,
                    dims,
                    out,
                    self.stride,
                    self.fmt,
                ),
                PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
                    u16::write_planes_u16(
                        self.color_space,
                        self.has_alpha,
                        planes,
                        dims,
                        out,
                        self.stride,
                        self.fmt,
                    )
                }
                _ => Err(J2kError::internal_backend(
                    "component row sink has unsupported pixel format",
                )),
            },
            None => Err(J2kError::DimensionOverflow {
                width: x,
                height: y,
            }),
        };
        self.error = result.err();
    }

    fn rgb8_region(&mut self, width: u32, height: u32) -> Option<&mut [u8]> {
        // `write_rows` packs three 8-bit planes as `write_rgb_component_rows_u8`
        // does, so a tight RGB8 buffer can take them packed by the decoder.
        let row_bytes = (width as usize).checked_mul(3)?;
        let tight_rgb8 = self.error.is_none()
            && matches!(self.color_space, ColorSpace::RGB)
            && !self.has_alpha
            && self.fmt == PixelFormat::Rgb8
            && self.stride == row_bytes;
        if !tight_rgb8 {
            return None;
        }
        self.out.get_mut(..row_bytes.checked_mul(height as usize)?)
    }
}
