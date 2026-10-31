// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sixteen-bit channel-layout conversion for native decoded samples.

use crate::backend::{ColorSpace, DecodedComponents as NativeDecodedComponents, Image, RawBitmap};
use crate::J2kError;
use j2k_core::{PixelFormat, Unsupported};

use super::u8::{component_sample_count, output_rows, validate_component_planes};

pub(in crate::decode) fn write_components_u16_output(
    components: &NativeDecodedComponents<'_>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    write_planes_u16(
        components.color_space(),
        components.has_alpha(),
        components.planes(),
        components.dimensions(),
        out,
        stride,
        fmt,
    )
}

/// Whether a region decode of `image` to `fmt` may stream its rows through
/// [`U16RowSink`]: they carry the samples `decode_native_region` packs (see
/// `Image::streamed_rows_native_precision`), every component has one
/// precision of at most 16 bits, which `U16Conversion` converts to the
/// bytes [`write_u16_output`] writes from those packed samples, and
/// `fmt` is a layout the image's color space maps to. Images that fail any
/// of these keep the native-bit-depth planes, with their output or error.
pub(in crate::decode) fn rows_match_native_u16(image: &Image<'_>, fmt: PixelFormat) -> bool {
    image
        .streamed_rows_native_precision()
        .is_some_and(|precision| precision <= 16)
        && matches!(
            (image.color_space(), image.has_alpha(), fmt),
            (ColorSpace::Gray, false, PixelFormat::Gray16)
                | (ColorSpace::RGB, _, PixelFormat::Rgb16 | PixelFormat::Rgba16)
        )
}

/// The bytes the planes route charges at least when it packs `image`'s
/// native samples for a 16-bit `fmt` (see [`super::planes_pack_floor`]),
/// when a region decode of `image` may stream its rows to `fmt` instead
/// (see [`rows_match_native_u16`]).
pub(in crate::decode) fn u16_rows_pack_floor(image: &Image<'_>, fmt: PixelFormat) -> Option<usize> {
    let precision = image.streamed_rows_native_precision()?;
    rows_match_native_u16(image, fmt)
        .then(|| super::planes_pack_floor(image, usize::from(precision).div_ceil(8)))
}

/// Write `planes`, each `dims` in size, to `out` from its first byte.
fn write_planes_u16(
    color_space: &ColorSpace,
    has_alpha: bool,
    planes: &[j2k_native::ComponentPlane<'_>],
    dims: (u32, u32),
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let expected_samples = component_sample_count(dims)?;
    let width = dims.0 as usize;
    let height = dims.1 as usize;
    match (color_space, has_alpha, planes.len(), fmt) {
        (ColorSpace::Gray, false, 1, PixelFormat::Gray16) => {
            validate_component_planes(&planes[..1], expected_samples)?;
            write_component_rows_u16(&planes[0], out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgb16)
        | (ColorSpace::RGB, true, 4, PixelFormat::Rgb16) => {
            validate_component_planes(&planes[..3], expected_samples)?;
            write_rgb_component_rows_u16(planes, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgba16) => {
            validate_component_planes(&planes[..3], expected_samples)?;
            write_rgba_component_rows_u16(planes, out, stride, width, height, true);
            Ok(())
        }
        (ColorSpace::RGB, true, 4, PixelFormat::Rgba16) => {
            validate_component_planes(&planes[..4], expected_samples)?;
            write_rgba_component_rows_u16(planes, out, stride, width, height, false);
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 16-bit pixel format",
        }
        .into()),
    }
}

/// Writes a region decode's rectangles of component rows to a 16-bit
/// output as [`write_components_u16_output`] writes whole planes, which for
/// an image that [`rows_match_native_u16`] accepts is what
/// [`write_u16_output`] writes from its native samples. The first error
/// stops every later write and is returned by [`Self::finish`].
pub(in crate::decode) struct U16RowSink<'o> {
    color_space: &'o ColorSpace,
    has_alpha: bool,
    out: &'o mut [u8],
    stride: usize,
    fmt: PixelFormat,
    error: Option<J2kError>,
}

impl<'o> U16RowSink<'o> {
    /// `out` must already hold the region at `stride` in `fmt`.
    pub(in crate::decode) fn new(
        color_space: &'o ColorSpace,
        has_alpha: bool,
        out: &'o mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Self {
        Self {
            color_space,
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
}

impl j2k_native::ComponentRowSink for U16RowSink<'_> {
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
            Some(out) => write_planes_u16(
                self.color_space,
                self.has_alpha,
                planes,
                dims,
                out,
                self.stride,
                self.fmt,
            ),
            None => Err(J2kError::DimensionOverflow {
                width: x,
                height: y,
            }),
        };
        self.error = result.err();
    }
}

fn write_component_rows_u16(
    plane: &j2k_native::ComponentPlane<'_>,
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) {
    let conversion = U16Conversion::new(plane.bit_depth(), plane.signed());
    for y in 0..height {
        conversion.visit(ChannelWriter {
            src: &plane.samples()[y * width..(y + 1) * width],
            dst: &mut out[y * stride..y * stride + width * 2],
            channels: 1,
            channel: 0,
        });
    }
}

fn write_rgb_component_rows_u16(
    planes: &[j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) {
    write_color_component_rows_u16(planes, out, stride, width, height, 3, false);
}

fn write_rgba_component_rows_u16(
    planes: &[j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    synthesize_alpha: bool,
) {
    write_color_component_rows_u16(planes, out, stride, width, height, 4, synthesize_alpha);
}

fn write_color_component_rows_u16(
    planes: &[j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    output_channels: usize,
    synthesize_alpha: bool,
) {
    let color = [0, 1, 2]
        .map(|channel| U16Conversion::new(planes[channel].bit_depth(), planes[channel].signed()));
    let fused = color[1] == color[0] && color[2] == color[0];
    let alpha = (output_channels == 4 && !synthesize_alpha)
        .then(|| U16Conversion::new(planes[3].bit_depth(), planes[3].signed()));
    let opaque = opaque_alpha_u16(
        usize::from(planes[0].bit_depth()).div_ceil(8),
        planes[0].bit_depth(),
    )
    .to_le_bytes();
    for y in 0..height {
        let row = y * width..(y + 1) * width;
        let destination = &mut out[y * stride..y * stride + width * output_channels * 2];
        if fused {
            color[0].visit(PixelWriter {
                src: [0, 1, 2].map(|channel| &planes[channel].samples()[row.clone()]),
                dst: destination,
                channels: output_channels,
            });
        } else {
            for (channel, conversion) in color.iter().enumerate() {
                conversion.visit(ChannelWriter {
                    src: &planes[channel].samples()[row.clone()],
                    dst: destination,
                    channels: output_channels,
                    channel,
                });
            }
        }
        match &alpha {
            Some(conversion) => conversion.visit(ChannelWriter {
                src: &planes[3].samples()[row],
                dst: destination,
                channels: 4,
                channel: 3,
            }),
            None if output_channels == 4 => {
                for pixel in destination.as_chunks_mut::<8>().0 {
                    pixel[6..8].copy_from_slice(&opaque);
                }
            }
            None => {}
        }
    }
}

/// How [`U16Conversion`] turns a rounded sample into its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum U16Kind {
    /// Unsigned 8 bits: the byte times 257, which is its widened value.
    Byte,
    /// Fewer unsigned bits: the byte's widened value from the table.
    ByteTable,
    /// Signed, at most 8 bits: the two's-complement byte's widened value.
    SignedByteTable,
    /// Unsigned, 9 to 16 bits: the sample itself.
    Unsigned,
    /// Signed, 9 to 16 bits: the sample's two's complement.
    Signed,
}

/// Converts the decoded samples of one component of `bit_depth` bits to
/// 16-bit output: rounded, clamped to the component's signed or unsigned
/// range (at most 16 bits) and, for at most 8 bits, clamped to a byte and
/// widened across the 16-bit range as [`write_u16_output`] widens native
/// 8-bit samples. The choices are made once per plane.
#[derive(Debug, PartialEq)]
struct U16Conversion {
    kind: U16Kind,
    /// The component's range, or the byte's for at most 8 unsigned bits.
    bounds: (f32, f32),
    /// The widened value of each byte for the table kinds.
    widened: [u16; 256],
}

impl U16Conversion {
    fn new(bit_depth: u8, signed: bool) -> Self {
        if bit_depth <= 8 {
            let (kind, bounds) = if signed {
                let magnitude = 1_i16 << bit_depth.saturating_sub(1);
                (
                    U16Kind::SignedByteTable,
                    (f32::from(-magnitude), f32::from(magnitude - 1)),
                )
            } else if bit_depth == 8 {
                (U16Kind::Byte, (0.0, f32::from(u8::MAX)))
            } else {
                (U16Kind::ByteTable, (0.0, f32::from(u8::MAX)))
            };
            return Self {
                kind,
                bounds,
                widened: core::array::from_fn(|byte| {
                    widen_u8_sample_to_u16(u8::try_from(byte).unwrap_or(u8::MAX), bit_depth)
                }),
            };
        }
        let (kind, bounds) = if signed {
            let magnitude = 1_i32 << (bit_depth.min(16) - 1);
            (
                U16Kind::Signed,
                (
                    f32::from(i16::try_from(-magnitude).unwrap_or(i16::MIN)),
                    f32::from(i16::try_from(magnitude - 1).unwrap_or(i16::MAX)),
                ),
            )
        } else {
            let max = (1_u32 << bit_depth.min(16)) - 1;
            (
                U16Kind::Unsigned,
                (0.0, f32::from(u16::try_from(max).unwrap_or(u16::MAX))),
            )
        };
        Self {
            kind,
            bounds,
            widened: [0; 256],
        }
    }

    /// Hand `writer` this conversion as a closure of its kind.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "decoded samples are rounded and clamped to the component's declared integer representation"
    )]
    #[inline(always)]
    fn visit(&self, writer: impl U16Writer) {
        let (min, max) = self.bounds;
        let widened = &self.widened;
        match self.kind {
            U16Kind::Byte => {
                writer.write(|sample: f32| u16::from(sample.round().clamp(min, max) as u8) * 257);
            }
            U16Kind::ByteTable => writer
                .write(|sample: f32| widened[usize::from(sample.round().clamp(min, max) as u8)]),
            U16Kind::SignedByteTable => writer.write(|sample: f32| {
                widened[usize::from((sample.round().clamp(min, max) as i8) as u8)]
            }),
            U16Kind::Unsigned => writer.write(|sample: f32| sample.round().clamp(min, max) as u16),
            U16Kind::Signed => {
                writer.write(|sample: f32| (sample.round().clamp(min, max) as i16) as u16);
            }
        }
    }
}

/// Writes rows of samples with a [`U16Conversion`]'s closure.
trait U16Writer {
    fn write(self, convert: impl Fn(f32) -> u16);
}

/// One component's samples into one channel of interleaved pixels.
struct ChannelWriter<'s, 'd> {
    src: &'s [f32],
    dst: &'d mut [u8],
    channels: usize,
    channel: usize,
}

impl U16Writer for ChannelWriter<'_, '_> {
    #[inline(always)]
    fn write(self, convert: impl Fn(f32) -> u16) {
        let Self {
            src,
            dst,
            channels,
            channel,
        } = self;
        for (pixel, &sample) in dst.chunks_exact_mut(channels * 2).zip(src) {
            pixel[channel * 2..channel * 2 + 2].copy_from_slice(&convert(sample).to_le_bytes());
        }
    }
}

/// Three components' samples into the first three channels of interleaved
/// pixels, one pixel at a time.
struct PixelWriter<'s, 'd> {
    src: [&'s [f32]; 3],
    dst: &'d mut [u8],
    channels: usize,
}

impl U16Writer for PixelWriter<'_, '_> {
    #[inline(always)]
    fn write(self, convert: impl Fn(f32) -> u16) {
        let Self {
            src: [red, green, blue],
            dst,
            channels,
        } = self;
        let samples = red.iter().zip(green).zip(blue);
        for (pixel, ((&red, &green), &blue)) in dst.chunks_exact_mut(channels * 2).zip(samples) {
            pixel[..2].copy_from_slice(&convert(red).to_le_bytes());
            pixel[2..4].copy_from_slice(&convert(green).to_le_bytes());
            pixel[4..6].copy_from_slice(&convert(blue).to_le_bytes());
        }
    }
}

pub(in crate::decode) fn write_u16_output(
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

    let dst_rows = output_rows(out, stride, dst_row_bytes, height);
    for (src_row, dst_row) in src.chunks_exact(src_row_bytes).zip(dst_rows) {
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
        u16::try_from(((1_u32 << bit_depth.min(16)) - 1).max(1)).unwrap_or(u16::MAX)
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
    u16::try_from((u32::from(sample) * u32::from(u16::MAX) + (max_value / 2)) / max_value)
        .unwrap_or(u16::MAX)
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
    let dst_rows = output_rows(out, stride, dst_row_bytes, height);
    for (src_row, dst_row) in src.chunks_exact(src_row_bytes).zip(dst_rows) {
        if bytes_per_sample == 2 {
            dst_row.copy_from_slice(src_row);
            continue;
        }
        for (sample, dst_sample) in src_row.iter().zip(dst_row.as_chunks_mut::<2>().0) {
            *dst_sample = widen_u8_sample_to_u16(*sample, bit_depth).to_le_bytes();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        opaque_alpha_u16, widen_u8_sample_to_u16, ChannelWriter, PixelWriter, U16Conversion,
    };

    /// The per-sample conversion the component-plane writers used before
    /// they converted a plane at a time.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "decoded samples are rounded and clamped to the component's declared integer representation"
    )]
    fn component_sample_as_u16(sample: f32, bit_depth: u8, signed: bool) -> u16 {
        let rounded = sample.round();
        if signed {
            if bit_depth <= 8 {
                let magnitude_bits = u32::from(bit_depth.saturating_sub(1));
                let min = -(1_i16 << magnitude_bits);
                let max = (1_i16 << magnitude_bits) - 1;
                return widen_u8_sample_to_u16(
                    (rounded.clamp(f32::from(min), f32::from(max)) as i8) as u8,
                    bit_depth,
                );
            }
            let magnitude_bits = u32::from(bit_depth.min(16).saturating_sub(1));
            let min = i16::try_from(-(1_i32 << magnitude_bits)).unwrap_or(i16::MIN);
            let max = i16::try_from((1_i32 << magnitude_bits) - 1).unwrap_or(i16::MAX);
            return (rounded.clamp(f32::from(min), f32::from(max)) as i16) as u16;
        }
        if bit_depth <= 8 {
            return widen_u8_sample_to_u16(rounded.clamp(0.0, f32::from(u8::MAX)) as u8, bit_depth);
        }
        let max = u16::try_from((1_u32 << u32::from(bit_depth.min(16))) - 1).unwrap_or(u16::MAX);
        rounded.clamp(0.0, f32::from(max)) as u16
    }

    #[test]
    #[expect(clippy::cast_precision_loss, reason = "test values fit f32 exactly")]
    fn plane_conversion_matches_the_sample_conversion() {
        let mut samples = vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            1.0e9,
            -1.0e9,
        ];
        for value in -70_000_i32..70_000 {
            for offset in [-0.5, 0.0, 0.25, 0.5] {
                samples.push(value as f32 + offset);
            }
        }
        for bit_depth in 1..=16 {
            for signed in [false, true] {
                let conversion = U16Conversion::new(bit_depth, signed);
                let mut out = vec![0; samples.len() * 2];
                conversion.visit(ChannelWriter {
                    src: &samples,
                    dst: &mut out,
                    channels: 1,
                    channel: 0,
                });
                let mut pixels = vec![0; samples.len() * 8];
                conversion.visit(PixelWriter {
                    src: [&samples; 3],
                    dst: &mut pixels,
                    channels: 4,
                });
                for (pixel, sample) in pixels.as_chunks::<8>().0.iter().zip(out.as_chunks::<2>().0)
                {
                    assert_eq!(
                        pixel[..6],
                        [*sample; 3].concat(),
                        "{bit_depth} bits, signed {signed}"
                    );
                }
                for (sample, bytes) in samples.iter().zip(out.as_chunks::<2>().0) {
                    assert_eq!(
                        u16::from_le_bytes(*bytes),
                        component_sample_as_u16(*sample, bit_depth, signed),
                        "{sample} at {bit_depth} bits, signed {signed}"
                    );
                }
            }
        }
    }

    #[test]
    fn eight_bit_samples_widen_across_the_complete_u16_domain() {
        assert_eq!(widen_u8_sample_to_u16(0, 8), 0);
        assert_eq!(widen_u8_sample_to_u16(u8::MAX, 8), u16::MAX);
    }

    #[test]
    fn synthesized_alpha_matches_native_sample_storage() {
        assert_eq!(opaque_alpha_u16(1, 8), u16::MAX);
        assert_eq!(opaque_alpha_u16(2, 12), 0x0fff);
        assert_eq!(opaque_alpha_u16(2, 16), u16::MAX);
    }
}
