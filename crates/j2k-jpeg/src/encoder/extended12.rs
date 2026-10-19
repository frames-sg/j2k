// SPDX-License-Identifier: MIT OR Apache-2.0

//! 12-bit extended sequential (SOF1) JPEG encode, DICOM's 12-bit lossy
//! transfer syntax (JPEG Extended, process 4).
//!
//! The pipeline mirrors the 8-bit encoder at 12-bit range: RGB converts to
//! YCbCr with chroma centered on 2048, chroma subsampling takes floor
//! averages, edge blocks replicate the last row and column, and the forward
//! DCT is the 8-bit path's transform. Standard tables cannot code 12-bit
//! coefficients, so every frame carries Huffman tables built from its own
//! statistics, as libjpeg does at 12 bits.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use j2k_core::try_host_vec_with_capacity;

use super::dct_entropy::{encode_dct_scan_optimized, DctComponent, DctScan};
use super::optimized::assemble_dct_frame;
use super::transform::{cosine_table, fdct_quantize_centered};
use super::{EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSubsampling};
use crate::adapter::{
    baseline_encode_tables, checked_encode_host_live_bytes, validate_jpeg_baseline_dimensions,
    validate_jpeg_baseline_restart_interval, JpegBaselineSampling,
};
use crate::encoded_frame::FrameKind;

/// Largest 12-bit sample.
const MAX_SAMPLE: u16 = 4095;
/// Level shift and chroma center of 12-bit samples.
const CENTER: i32 = 2048;
/// Largest DC difference category of 12-bit DCT JPEG.
const MAX_DC_CATEGORY_12BIT: u8 = 15;

/// Borrowed 12-bit samples (0..=4095 in `u16`), row-major and interleaved.
#[derive(Debug, Clone, Copy)]
pub enum JpegSamples12<'a> {
    /// One sample per pixel.
    Gray {
        /// Pixel data.
        data: &'a [u16],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// Three samples (R, G, B) per pixel.
    Rgb {
        /// Pixel data.
        data: &'a [u16],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
}

impl<'a> JpegSamples12<'a> {
    const fn parts(self) -> (&'a [u16], u32, u32, usize, &'static str) {
        match self {
            Self::Gray {
                data,
                width,
                height,
            } => (data, width, height, 1, "Gray12"),
            Self::Rgb {
                data,
                width,
                height,
            } => (data, width, height, 3, "Rgb12"),
        }
    }
}

/// Encode 12-bit samples as an extended sequential (SOF1) JPEG with
/// optimized Huffman tables.
///
/// `options` select quality, subsampling (`Gray` for grayscale samples,
/// `Ybr444`/`Ybr422`/`Ybr420` for RGB), and the restart interval in MCUs,
/// as for [`super::encode_jpeg_baseline`].
///
/// # Errors
///
/// Returns an error for invalid dimensions, sample length, subsampling, or
/// restart interval; for a sample above 4095; for coefficients above the
/// host allocation cap; and for a GPU backend.
pub fn encode_jpeg_extended12(
    samples: JpegSamples12<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    if !matches!(options.backend, JpegBackend::Auto | JpegBackend::Cpu) {
        return Err(JpegEncodeError::UnsupportedBackend {
            backend: options.backend,
        });
    }
    let (data, width, height, channels, name) = samples.parts();
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;
    validate_jpeg_baseline_dimensions(width, height)?;
    let gray_subsampling = options.subsampling == JpegSubsampling::Gray;
    if gray_subsampling != (channels == 1) {
        return Err(JpegEncodeError::IncompatibleSubsampling {
            subsampling: options.subsampling,
            samples: name,
        });
    }
    let expected = width as usize * height as usize * channels;
    if data.len() != expected {
        return Err(JpegEncodeError::SampleLength {
            expected,
            actual: data.len(),
        });
    }
    if let Some((index, &value)) = data.iter().enumerate().find(|&(_, &v)| v > MAX_SAMPLE) {
        return Err(JpegEncodeError::SampleOutOfRange {
            index,
            value,
            precision: 12,
        });
    }

    let tables = baseline_encode_tables(options)?;
    let sampling = tables.sampling;
    let quant = [tables.q_luma.map(u16::from), tables.q_chroma.map(u16::from)];
    let planes = component_planes(data, (width, height), sampling)?;
    let grids: Vec<(usize, usize)> = planes
        .iter()
        .map(|plane| (plane.width.div_ceil(8), plane.height.div_ceil(8)))
        .collect();
    let block_count: usize = grids.iter().map(|(cols, rows)| cols * rows).sum();
    checked_encode_host_live_bytes([
        planes.iter().map(|plane| plane.samples.len() * 2).sum(),
        block_count.saturating_mul(core::mem::size_of::<[i16; 64]>()),
    ])?;
    let cosine = cosine_table();
    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(planes.len())
        .map_err(|_| JpegEncodeError::HostAllocationFailed {
            bytes: planes.len(),
        })?;
    for (index, plane) in planes.iter().enumerate() {
        let (cols, rows) = grids[index];
        blocks.push(plane_blocks(
            plane,
            (cols, rows),
            &quant[usize::from(index != 0)],
            cosine,
        )?);
    }
    drop(planes);
    let components: Vec<DctComponent<'_>> = blocks
        .iter()
        .enumerate()
        .map(|(index, component)| {
            let (cols, rows) = grids[index];
            let cols = u32::try_from(cols).unwrap_or(u32::MAX);
            DctComponent {
                h: sampling.h[index],
                v: sampling.v[index],
                block_cols: cols,
                real_cols: cols,
                real_rows: u32::try_from(rows).unwrap_or(u32::MAX),
                blocks: component,
                table: usize::from(index != 0),
            }
        })
        .collect();
    let entropy = encode_dct_scan_optimized(&DctScan {
        width,
        height,
        components: &components,
        restart_interval: options.restart_interval,
        max_dc_category: MAX_DC_CATEGORY_12BIT,
    })?;
    assemble_dct_frame(
        FrameKind::ExtendedSequential,
        12,
        (width, height),
        sampling,
        &quant,
        options.restart_interval,
        &entropy,
    )
}

/// One component's samples at its own resolution.
struct Plane<'a> {
    width: usize,
    height: usize,
    samples: Cow<'a, [u16]>,
}

/// Color-convert and subsample the input into component planes.
fn component_planes<'a>(
    data: &'a [u16],
    (width, height): (u32, u32),
    sampling: JpegBaselineSampling,
) -> Result<Vec<Plane<'a>>, JpegEncodeError> {
    let (width, height) = (width as usize, height as usize);
    if sampling.components == 1 {
        return Ok(vec_of([Plane {
            width,
            height,
            samples: Cow::Borrowed(data),
        }]));
    }
    let pixels = width * height;
    checked_encode_host_live_bytes([pixels.saturating_mul(6)])?;
    let mut luma = try_vec(pixels)?;
    let mut cb = try_vec(pixels)?;
    let mut cr = try_vec(pixels)?;
    for &[r, g, b] in data.as_chunks::<3>().0 {
        let (y, blue_difference, red_difference) = rgb_to_ycbcr12(r, g, b);
        luma.push(y);
        cb.push(blue_difference);
        cr.push(red_difference);
    }
    let (h_ratio, v_ratio) = (usize::from(sampling.max_h), usize::from(sampling.max_v));
    let chroma = |full: Vec<u16>| -> Result<Plane<'a>, JpegEncodeError> {
        let (chroma_width, chroma_height) = (width.div_ceil(h_ratio), height.div_ceil(v_ratio));
        if (h_ratio, v_ratio) == (1, 1) {
            return Ok(Plane {
                width,
                height,
                samples: Cow::Owned(full),
            });
        }
        let mut samples = try_vec(chroma_width * chroma_height)?;
        let covered = u32::from(sampling.max_h) * u32::from(sampling.max_v);
        for y in 0..chroma_height {
            for x in 0..chroma_width {
                // Floor average of the covered samples, replicating the last
                // row and column at odd sizes.
                let mut sum = 0_u32;
                for dy in 0..v_ratio {
                    for dx in 0..h_ratio {
                        let sx = (x * h_ratio + dx).min(width - 1);
                        let sy = (y * v_ratio + dy).min(height - 1);
                        sum += u32::from(full[sy * width + sx]);
                    }
                }
                samples.push(u16::try_from(sum / covered).unwrap_or(MAX_SAMPLE));
            }
        }
        Ok(Plane {
            width: chroma_width,
            height: chroma_height,
            samples: Cow::Owned(samples),
        })
    };
    let luma = Plane {
        width,
        height,
        samples: Cow::Owned(luma),
    };
    Ok(vec_of([luma, chroma(cb)?, chroma(cr)?]))
}

/// The 8-bit encoder's fixed-point RGB to YCbCr conversion at 12 bits.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "converted samples are clamped to the 12-bit range"
)]
fn rgb_to_ycbcr12(r: u16, g: u16, b: u16) -> (u16, u16, u16) {
    let (r, g, b) = (i32::from(r), i32::from(g), i32::from(b));
    let chroma_offset = (CENTER << 16) + 32_768;
    let y = (19_595 * r + 38_470 * g + 7_471 * b + 32_768) >> 16;
    let cb = (-11_059 * r - 21_709 * g + 32_768 * b + chroma_offset) >> 16;
    let cr = (32_768 * r - 27_439 * g - 5_329 * b + chroma_offset) >> 16;
    let clamp = |value: i32| value.clamp(0, i32::from(MAX_SAMPLE)) as u16;
    (clamp(y), clamp(cb), clamp(cr))
}

/// Transform and quantize every block covering `plane`, replicating its last
/// row and column into the blocks' samples past its edge.
fn plane_blocks(
    plane: &Plane<'_>,
    (cols, rows): (usize, usize),
    quant: &[u16; 64],
    cosine: &[[f64; 8]; 8],
) -> Result<Vec<[i16; 64]>, JpegEncodeError> {
    let mut blocks = try_vec(cols * rows)?;
    for block_y in 0..rows {
        for block_x in 0..cols {
            let centered: [i32; 64] = core::array::from_fn(|index| {
                let x = (block_x * 8 + index % 8).min(plane.width - 1);
                let y = (block_y * 8 + index / 8).min(plane.height - 1);
                i32::from(plane.samples[y * plane.width + x]) - CENTER
            });
            blocks.push(fdct_quantize_centered(&centered, quant, cosine));
        }
    }
    Ok(blocks)
}

fn try_vec<T>(capacity: usize) -> Result<Vec<T>, JpegEncodeError> {
    try_host_vec_with_capacity(capacity).map_err(|error| JpegEncodeError::HostAllocationFailed {
        bytes: error.requested_bytes(),
    })
}

fn vec_of<T, const N: usize>(items: [T; N]) -> Vec<T> {
    items.into_iter().collect()
}

#[cfg(test)]
mod tests;
