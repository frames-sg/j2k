// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::backend::{ColorSpace, DecodedComponents as NativeDecodedComponents, Image, RawBitmap};
use crate::J2kError;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::convert::Infallible;
use j2k_core::{validate_strided_output_buffer, DecodeOutcome, PixelFormat, Rect, Unsupported};
pub(crate) type J2kDecodeOutcome = DecodeOutcome<Infallible>;

macro_rules! impl_component_plane_metadata_accessors {
    () => {
        /// Width and height of this decoded plane in output samples.
        #[must_use]
        pub fn dimensions(&self) -> (u32, u32) {
            self.dimensions
        }

        /// Horizontal and vertical SIZ sampling factors (`XRsiz`, `YRsiz`).
        #[must_use]
        pub fn sampling(&self) -> (u8, u8) {
            self.sampling
        }

        /// Bit depth of this component plane.
        #[must_use]
        pub fn bit_depth(&self) -> u8 {
            self.bit_depth
        }

        /// Whether this component plane stores signed sample values.
        #[must_use]
        pub fn signed(&self) -> bool {
            self.signed
        }
    };
}

macro_rules! impl_decoded_components_metadata_accessors {
    () => {
        /// Dimensions of the decoded image represented by these planes.
        #[must_use]
        pub fn dimensions(&self) -> (u32, u32) {
            self.dimensions
        }

        /// Color space after JPEG 2000 color conversion has been applied.
        #[must_use]
        pub fn color_space(&self) -> &J2kDecodedColorSpace {
            &self.color_space
        }

        /// Whether the decoded image has an alpha channel.
        #[must_use]
        pub fn has_alpha(&self) -> bool {
            self.has_alpha
        }
    };
}

/// Decoded JPEG 2000 color space metadata for component-plane outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kDecodedColorSpace {
    /// Grayscale image data.
    Gray,
    /// RGB image data.
    Rgb,
    /// CMYK image data.
    Cmyk,
    /// Unknown image data with the given number of channels.
    Unknown {
        /// Number of channels represented by the color space.
        num_channels: u16,
    },
    /// ICC-described image data.
    Icc {
        /// ICC profile bytes.
        profile: Vec<u8>,
        /// Number of channels represented by the ICC profile.
        num_channels: u16,
    },
}

impl J2kDecodedColorSpace {
    fn from_native(color_space: &ColorSpace) -> Self {
        match color_space {
            ColorSpace::Gray => Self::Gray,
            ColorSpace::RGB => Self::Rgb,
            ColorSpace::CMYK => Self::Cmyk,
            ColorSpace::Unknown { num_channels } => Self::Unknown {
                num_channels: *num_channels,
            },
            ColorSpace::Icc {
                profile,
                num_channels,
            } => Self::Icc {
                profile: profile.clone(),
                num_channels: *num_channels,
            },
        }
    }
}

/// One borrowed decoded component plane.
#[derive(Debug, Clone, Copy)]
pub struct J2kComponentPlane<'a> {
    samples: &'a [f32],
    dimensions: (u32, u32),
    bit_depth: u8,
    signed: bool,
    sampling: (u8, u8),
}

impl<'a> J2kComponentPlane<'a> {
    fn from_native(plane: &j2k_native::ComponentPlane<'a>) -> Self {
        Self {
            samples: plane.samples(),
            dimensions: plane.dimensions(),
            bit_depth: plane.bit_depth(),
            signed: plane.signed(),
            sampling: plane.sampling(),
        }
    }

    /// Component samples in row-major order.
    #[must_use]
    pub fn samples(&self) -> &'a [f32] {
        self.samples
    }

    impl_component_plane_metadata_accessors!();
}

/// Borrowed decoded component planes for an image.
#[derive(Debug, Clone)]
pub struct J2kDecodedComponents<'a> {
    dimensions: (u32, u32),
    color_space: J2kDecodedColorSpace,
    has_alpha: bool,
    planes: Vec<J2kComponentPlane<'a>>,
}

impl<'a> J2kDecodedComponents<'a> {
    pub(crate) fn from_native(decoded: &j2k_native::DecodedComponents<'a>) -> Self {
        Self {
            dimensions: decoded.dimensions(),
            color_space: J2kDecodedColorSpace::from_native(decoded.color_space()),
            has_alpha: decoded.has_alpha(),
            planes: decoded
                .planes()
                .iter()
                .map(J2kComponentPlane::from_native)
                .collect(),
        }
    }

    impl_decoded_components_metadata_accessors!();

    /// Borrowed decoded component planes in display order.
    #[must_use]
    pub fn planes(&self) -> &[J2kComponentPlane<'a>] {
        &self.planes
    }
}

/// One owned decoded component plane at native bit depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kNativeComponentPlane {
    data: Vec<u8>,
    dimensions: (u32, u32),
    bit_depth: u8,
    signed: bool,
    sampling: (u8, u8),
    bytes_per_sample: u8,
}

impl J2kNativeComponentPlane {
    fn from_native(plane: &j2k_native::NativeComponentPlane) -> Self {
        Self {
            data: plane.data().to_vec(),
            dimensions: plane.dimensions(),
            bit_depth: plane.bit_depth(),
            signed: plane.signed(),
            sampling: plane.sampling(),
            bytes_per_sample: plane.bytes_per_sample(),
        }
    }

    /// Packed little-endian sample bytes for this component in row-major order.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    impl_component_plane_metadata_accessors!();

    /// Bytes used for each packed little-endian sample in [`Self::data`].
    #[must_use]
    pub fn bytes_per_sample(&self) -> u8 {
        self.bytes_per_sample
    }
}

/// Owned decoded native-bit-depth component planes for an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kDecodedNativeComponents {
    dimensions: (u32, u32),
    color_space: J2kDecodedColorSpace,
    has_alpha: bool,
    planes: Vec<J2kNativeComponentPlane>,
}

impl J2kDecodedNativeComponents {
    pub(crate) fn from_native(decoded: &j2k_native::DecodedNativeComponents) -> Self {
        Self {
            dimensions: decoded.dimensions(),
            color_space: J2kDecodedColorSpace::from_native(decoded.color_space()),
            has_alpha: decoded.has_alpha(),
            planes: decoded
                .planes()
                .iter()
                .map(J2kNativeComponentPlane::from_native)
                .collect(),
        }
    }

    impl_decoded_components_metadata_accessors!();

    /// Decoded component planes in display order.
    #[must_use]
    pub fn planes(&self) -> &[J2kNativeComponentPlane] {
        &self.planes
    }
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
                    .map_err(|err| J2kError::Backend(err.to_string()))?;
                return Ok(());
            }
            let decoded = image
                .decode_with_context(native_context)
                .map_err(|err| J2kError::Backend(err.to_string()))?;
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
                .map_err(|err| J2kError::Backend(err.to_string()))?;
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

fn can_decode_u8_directly(
    color_space: &ColorSpace,
    has_alpha: bool,
    dims: (u32, u32),
    stride: usize,
    fmt: PixelFormat,
) -> bool {
    let width = dims.0 as usize;
    match (color_space, has_alpha, fmt) {
        (ColorSpace::RGB, false, PixelFormat::Rgb8) => stride == width * 3,
        (ColorSpace::RGB, true, PixelFormat::Rgba8) => stride == width * 4,
        (ColorSpace::Gray, false, PixelFormat::Gray8) => stride == width,
        _ => false,
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
                .map_err(|err| J2kError::Backend(err.to_string()))?;
            write_components_u8_output(&components, out, stride, fmt)
        }
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => {
            let raw = image
                .decode_native_region_with_context((roi.x, roi.y, roi.w, roi.h), native_context)
                .map_err(|err| J2kError::Backend(err.to_string()))?;
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

fn write_components_u8_output(
    components: &NativeDecodedComponents<'_>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let dims = components.dimensions();
    let expected_samples = component_sample_count(dims)?;
    let width = dims.0 as usize;
    let height = dims.1 as usize;
    let planes = components.planes();
    match (
        components.color_space(),
        components.has_alpha(),
        planes.len(),
        fmt,
    ) {
        (ColorSpace::Gray, false, 1, PixelFormat::Gray8) => {
            validate_component_planes(&planes[..1], expected_samples)?;
            write_component_rows_u8(&planes[0], out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (ColorSpace::RGB, true, 4, PixelFormat::Rgb8) => {
            validate_component_planes(&planes[..3], expected_samples)?;
            write_rgb_component_rows_u8(planes, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgba8) => {
            validate_component_planes(&planes[..3], expected_samples)?;
            write_rgba_component_rows_u8(planes, out, stride, width, height, true);
            Ok(())
        }
        (ColorSpace::RGB, true, 4, PixelFormat::Rgba8) => {
            validate_component_planes(&planes[..4], expected_samples)?;
            write_rgba_component_rows_u8(planes, out, stride, width, height, false);
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 8-bit pixel format",
        }
        .into()),
    }
}

fn component_sample_count(dims: (u32, u32)) -> Result<usize, J2kError> {
    (dims.0 as usize)
        .checked_mul(dims.1 as usize)
        .ok_or(J2kError::DimensionOverflow {
            width: dims.0,
            height: dims.1,
        })
}

fn validate_component_planes(
    planes: &[j2k_native::ComponentPlane<'_>],
    expected_samples: usize,
) -> Result<(), J2kError> {
    for (index, plane) in planes.iter().enumerate() {
        let samples = plane.samples().len();
        if samples < expected_samples {
            return Err(J2kError::Backend(format!(
                "backend component plane {index} has {samples} samples, expected at least {expected_samples}"
            )));
        }
    }
    Ok(())
}

fn write_component_rows_u8(
    plane: &j2k_native::ComponentPlane<'_>,
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let src = &plane.samples()[y * width..(y + 1) * width];
        let dst = &mut out[y * stride..y * stride + width];
        write_samples_as_u8(src, plane.bit_depth(), dst);
    }
}

fn write_rgb_component_rows_u8(
    planes: &[j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let row = y * width;
        let dst = &mut out[y * stride..y * stride + width * 3];
        for x in 0..width {
            let dst = &mut dst[x * 3..x * 3 + 3];
            for channel in 0..3 {
                dst[channel] = sample_as_u8(
                    planes[channel].samples()[row + x],
                    planes[channel].bit_depth(),
                );
            }
        }
    }
}

fn write_rgba_component_rows_u8(
    planes: &[j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    synthesize_alpha: bool,
) {
    for y in 0..height {
        let row = y * width;
        let dst = &mut out[y * stride..y * stride + width * 4];
        for x in 0..width {
            let dst = &mut dst[x * 4..x * 4 + 4];
            for channel in 0..3 {
                dst[channel] = sample_as_u8(
                    planes[channel].samples()[row + x],
                    planes[channel].bit_depth(),
                );
            }
            dst[3] = if synthesize_alpha {
                u8::MAX
            } else {
                sample_as_u8(planes[3].samples()[row + x], planes[3].bit_depth())
            };
        }
    }
}

fn write_samples_as_u8(src: &[f32], bit_depth: u8, dst: &mut [u8]) {
    for (sample, dst) in src.iter().zip(dst.iter_mut()) {
        *dst = sample_as_u8(*sample, bit_depth);
    }
}

fn sample_as_u8(sample: f32, bit_depth: u8) -> u8 {
    let rounded = sample.round();
    if bit_depth == 8 {
        return rounded.clamp(0.0, f32::from(u8::MAX)) as u8;
    }
    let max_value = if bit_depth >= 16 {
        f32::from(u16::MAX)
    } else {
        f32::from(((1_u16 << bit_depth) - 1).max(1))
    };
    ((rounded.clamp(0.0, max_value) / max_value) * f32::from(u8::MAX)).round() as u8
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

fn write_u8_output(
    color_space: &ColorSpace,
    has_alpha: bool,
    dims: (u32, u32),
    decoded: &[u8],
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let width = dims.0 as usize;
    let height = dims.1 as usize;
    match (color_space, has_alpha, fmt) {
        (ColorSpace::RGB, false, PixelFormat::Rgb8) => {
            copy_rows_exact(decoded, out, stride, width * 3, height);
            Ok(())
        }
        (ColorSpace::RGB, true, PixelFormat::Rgb8) => {
            drop_alpha_u8(decoded, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, PixelFormat::Rgba8) => {
            add_opaque_alpha_u8(decoded, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, true, PixelFormat::Rgba8) => {
            copy_rows_exact(decoded, out, stride, width * 4, height);
            Ok(())
        }
        (ColorSpace::Gray, false, PixelFormat::Gray8) => {
            copy_rows_exact(decoded, out, stride, width, height);
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 8-bit pixel format",
        }
        .into()),
    }
}

fn write_u16_output(
    color_space: &ColorSpace,
    has_alpha: bool,
    raw: &RawBitmap,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let width = raw.width as usize;
    let height = raw.height as usize;
    match (color_space, has_alpha, raw.num_components, fmt) {
        (ColorSpace::RGB, false, 3, PixelFormat::Rgb16) => {
            convert_or_copy_u16(
                &raw.data,
                raw.bytes_per_sample,
                raw.bit_depth,
                3,
                out,
                stride,
                (width, height),
            );
            Ok(())
        }
        (ColorSpace::RGB, true, 4, PixelFormat::Rgb16) => {
            write_u16_channel_rows(U16ChannelRows {
                src: &raw.data,
                bytes_per_sample: raw.bytes_per_sample,
                bit_depth: raw.bit_depth,
                source_channels: 4,
                layout: U16ChannelLayout::Drop,
                out,
                stride,
                dims: (width, height),
            });
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgba16) => {
            write_u16_channel_rows(U16ChannelRows {
                src: &raw.data,
                bytes_per_sample: raw.bytes_per_sample,
                bit_depth: raw.bit_depth,
                source_channels: 3,
                layout: U16ChannelLayout::Synthesize,
                out,
                stride,
                dims: (width, height),
            });
            Ok(())
        }
        (ColorSpace::RGB, true, 4, PixelFormat::Rgba16) => {
            write_u16_channel_rows(U16ChannelRows {
                src: &raw.data,
                bytes_per_sample: raw.bytes_per_sample,
                bit_depth: raw.bit_depth,
                source_channels: 4,
                layout: U16ChannelLayout::Preserve,
                out,
                stride,
                dims: (width, height),
            });
            Ok(())
        }
        (ColorSpace::Gray, false, 1, PixelFormat::Gray16) => {
            convert_or_copy_u16(
                &raw.data,
                raw.bytes_per_sample,
                raw.bit_depth,
                1,
                out,
                stride,
                (width, height),
            );
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 16-bit pixel format",
        }
        .into()),
    }
}

#[derive(Debug, Clone, Copy)]
enum U16ChannelLayout {
    Drop,
    Synthesize,
    Preserve,
}

struct U16ChannelRows<'src, 'out> {
    src: &'src [u8],
    bytes_per_sample: u8,
    bit_depth: u8,
    source_channels: usize,
    layout: U16ChannelLayout,
    out: &'out mut [u8],
    stride: usize,
    dims: (usize, usize),
}

fn copy_rows_exact(src: &[u8], out: &mut [u8], stride: usize, row_bytes: usize, height: usize) {
    for (src_row, dst_row) in src
        .chunks_exact(row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        dst_row[..row_bytes].copy_from_slice(src_row);
    }
}

fn add_opaque_alpha_u8(src: &[u8], out: &mut [u8], stride: usize, width: usize, height: usize) {
    let src_row_bytes = width * 3;
    let dst_row_bytes = width * 4;
    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        for (rgb, rgba) in src_row
            .chunks_exact(3)
            .zip(dst_row[..dst_row_bytes].chunks_exact_mut(4))
        {
            rgba[..3].copy_from_slice(rgb);
            rgba[3] = u8::MAX;
        }
    }
}

fn drop_alpha_u8(src: &[u8], out: &mut [u8], stride: usize, width: usize, height: usize) {
    let src_row_bytes = width * 4;
    let dst_row_bytes = width * 3;
    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        for (rgba, rgb) in src_row
            .chunks_exact(4)
            .zip(dst_row[..dst_row_bytes].chunks_exact_mut(3))
        {
            rgb.copy_from_slice(&rgba[..3]);
        }
    }
}

fn write_u16_channel_rows(job: U16ChannelRows<'_, '_>) {
    let U16ChannelRows {
        src,
        bytes_per_sample,
        bit_depth,
        source_channels,
        layout,
        out,
        stride,
        dims,
    } = job;
    let (width, height) = dims;
    let dst_channels = match layout {
        U16ChannelLayout::Drop => 3,
        U16ChannelLayout::Synthesize | U16ChannelLayout::Preserve => 4,
    };
    let bytes_per_sample = usize::from(bytes_per_sample);
    let src_row_bytes = width * source_channels * bytes_per_sample;
    let dst_row_bytes = width * dst_channels * 2;
    let alpha = opaque_alpha_u16(bytes_per_sample, bit_depth);

    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        let dst_row = &mut dst_row[..dst_row_bytes];
        for x in 0..width {
            let src_pixel = &src_row[x * source_channels * bytes_per_sample..];
            let dst_pixel = &mut dst_row[x * dst_channels * 2..(x + 1) * dst_channels * 2];
            for channel in 0..3 {
                let sample = output_u16_sample(src_pixel, channel, bytes_per_sample, bit_depth);
                dst_pixel[channel * 2..channel * 2 + 2].copy_from_slice(&sample.to_le_bytes());
            }
            match layout {
                U16ChannelLayout::Drop => {}
                U16ChannelLayout::Synthesize => {
                    dst_pixel[6..8].copy_from_slice(&alpha.to_le_bytes());
                }
                U16ChannelLayout::Preserve => {
                    let sample = output_u16_sample(src_pixel, 3, bytes_per_sample, bit_depth);
                    dst_pixel[6..8].copy_from_slice(&sample.to_le_bytes());
                }
            }
        }
    }
}

fn opaque_alpha_u16(bytes_per_sample: usize, bit_depth: u8) -> u16 {
    if bytes_per_sample == 1 {
        u16::MAX
    } else {
        ((1_u32 << bit_depth.min(16)) - 1).max(1) as u16
    }
}

fn output_u16_sample(
    src_pixel: &[u8],
    channel: usize,
    bytes_per_sample: usize,
    bit_depth: u8,
) -> u16 {
    let offset = channel * bytes_per_sample;
    if bytes_per_sample == 2 {
        return u16::from_le_bytes([src_pixel[offset], src_pixel[offset + 1]]);
    }
    widen_u8_sample_to_u16(src_pixel[offset], bit_depth)
}

fn widen_u8_sample_to_u16(sample: u8, bit_depth: u8) -> u16 {
    let max_value = ((1_u32 << bit_depth.min(16)) - 1).max(1);
    ((u32::from(sample) * u32::from(u16::MAX) + (max_value / 2)) / max_value) as u16
}

fn convert_or_copy_u16(
    src: &[u8],
    bytes_per_sample: u8,
    bit_depth: u8,
    channels: usize,
    out: &mut [u8],
    stride: usize,
    dims: (usize, usize),
) {
    let (width, height) = dims;
    let dst_row_bytes = width * channels * 2;
    let src_row_bytes = width * channels * usize::from(bytes_per_sample);
    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        let dst_row = &mut dst_row[..dst_row_bytes];
        if bytes_per_sample == 2 {
            dst_row.copy_from_slice(src_row);
            continue;
        }
        for (sample, dst_sample) in src_row.iter().zip(dst_row.chunks_exact_mut(2)) {
            let widened = widen_u8_sample_to_u16(*sample, bit_depth);
            dst_sample.copy_from_slice(&widened.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_pointer_width = "32")]
    use super::J2kError;
    use super::{can_decode_u8_directly, component_sample_count, ColorSpace, PixelFormat};

    #[test]
    fn direct_u8_decode_accepts_exact_rgb_and_gray_layouts() {
        assert!(can_decode_u8_directly(
            &ColorSpace::RGB,
            false,
            (128, 64),
            128 * 3,
            PixelFormat::Rgb8
        ));
        assert!(can_decode_u8_directly(
            &ColorSpace::Gray,
            false,
            (128, 64),
            128,
            PixelFormat::Gray8
        ));
    }

    #[test]
    fn direct_u8_decode_rejects_format_conversion_and_padded_stride() {
        assert!(!can_decode_u8_directly(
            &ColorSpace::RGB,
            false,
            (128, 64),
            128 * 4,
            PixelFormat::Rgba8
        ));
        assert!(!can_decode_u8_directly(
            &ColorSpace::RGB,
            true,
            (128, 64),
            128 * 3,
            PixelFormat::Rgb8
        ));
        assert!(!can_decode_u8_directly(
            &ColorSpace::Gray,
            false,
            (128, 64),
            160,
            PixelFormat::Gray8
        ));
    }

    #[test]
    fn component_sample_count_matches_image_dimensions() {
        assert_eq!(component_sample_count((16, 8)).expect("sample count"), 128);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn component_sample_count_reports_dimension_overflow() {
        assert!(matches!(
            component_sample_count((u32::MAX, u32::MAX)),
            Err(J2kError::DimensionOverflow {
                width: u32::MAX,
                height: u32::MAX
            })
        ));
    }
}
