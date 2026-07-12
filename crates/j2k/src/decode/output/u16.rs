// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sixteen-bit channel-layout conversion for native decoded samples.

use crate::backend::{ColorSpace, RawBitmap};
use crate::J2kError;
use j2k_core::{PixelFormat, Unsupported};

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
        u16::try_from(((1_u32 << bit_depth.min(16)) - 1).max(1))
            .expect("16-bit alpha mask fits u16")
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
        .expect("scaled sample is bounded to u16")
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
    use super::{opaque_alpha_u16, widen_u8_sample_to_u16};

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
