// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names
)]

use alloc::string::String;
use alloc::vec::Vec;
use core::f64::consts::PI;

use rayon::prelude::*;
use thiserror::Error;

use crate::adapter::{
    assemble_jpeg_baseline_frame, baseline_encode_tables, validate_jpeg_baseline_dimensions,
    validate_jpeg_baseline_restart_interval, JpegBaselineHuffmanTable, JpegBaselineSampling,
    JPEG_BASELINE_ZIGZAG,
};
use crate::profile::{duration_us_string, emit_jpeg_profile_row, jpeg_profile_stages_enabled};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// JPEG encoder backend selector.
pub enum JpegBackend {
    /// Choose the best available backend for the platform.
    Auto,
    /// Use the portable CPU encoder.
    Cpu,
    /// Use a Metal encoder when called through the Metal integration.
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// JPEG baseline chroma subsampling mode.
pub enum JpegSubsampling {
    /// Single-component grayscale.
    Gray,
    /// Three-component YBR/RGB 4:4:4 sampling.
    Ybr444,
    /// Three-component YBR/RGB 4:2:2 sampling.
    Ybr422,
    /// Three-component YBR/RGB 4:2:0 sampling.
    Ybr420,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Options controlling baseline JPEG encoding.
pub struct JpegEncodeOptions {
    /// JPEG quality in the conventional 1..=100 range.
    pub quality: u8,
    /// Output component sampling.
    pub subsampling: JpegSubsampling,
    /// Optional restart interval in MCUs.
    pub restart_interval: Option<u16>,
    /// Requested encoder backend.
    pub backend: JpegBackend,
}

impl Default for JpegEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Borrowed input samples for baseline JPEG encoding.
pub enum JpegSamples<'a> {
    /// Interleaved 8-bit grayscale samples.
    Gray8 {
        /// Pixel data, one byte per pixel.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// Interleaved 8-bit RGB samples.
    Rgb8 {
        /// Pixel data, three bytes per pixel.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Encoded baseline JPEG bytes and the backend that produced them.
pub struct EncodedJpeg {
    /// Complete JPEG codestream.
    pub data: Vec<u8>,
    /// Backend used to encode the codestream.
    pub backend: JpegBackend,
}

#[derive(Debug, Error)]
/// Errors produced by baseline JPEG encoding.
pub enum JpegEncodeError {
    #[error("JPEG encode requires nonzero dimensions")]
    /// Width or height was zero.
    EmptyDimensions,
    #[error("JPEG baseline dimensions must fit in u16, got {width}x{height}")]
    /// JPEG baseline SOF dimensions exceed the 16-bit marker fields.
    DimensionsTooLarge {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },
    #[error("JPEG sample buffer length mismatch: expected {expected}, got {actual}")]
    /// Input sample buffer length does not match width, height, and format.
    SampleLength {
        /// Required byte count.
        expected: usize,
        /// Supplied byte count.
        actual: usize,
    },
    #[error("JPEG subsampling {subsampling:?} is incompatible with {samples}")]
    /// Requested subsampling is incompatible with the supplied sample format.
    IncompatibleSubsampling {
        /// Requested output sampling.
        subsampling: JpegSubsampling,
        /// Human-readable sample format name.
        samples: &'static str,
    },
    #[error("JPEG restart interval must be nonzero when provided")]
    /// Restart interval was explicitly set to zero.
    InvalidRestartInterval,
    #[error("JPEG encode backend {backend:?} is unavailable in j2k-jpeg CPU crate")]
    /// Requested backend is not available in this crate.
    UnsupportedBackend {
        /// Requested backend.
        backend: JpegBackend,
    },
    #[error("JPEG encoded marker segment is too large: {name}")]
    /// A marker segment would exceed the JPEG 16-bit length field.
    SegmentTooLarge {
        /// Marker segment name.
        name: &'static str,
    },
    #[error("JPEG entropy symbol has no Huffman code: {symbol}")]
    /// Encoder attempted to emit a symbol absent from the active Huffman table.
    MissingHuffmanCode {
        /// Missing entropy symbol.
        symbol: u8,
    },
    #[error("JPEG encode failed: {0}")]
    /// Internal encoder failure with diagnostic text.
    Internal(String),
}

pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

#[derive(Default)]
struct JpegEncodeProfile {
    validation: Duration,
    setup: Duration,
    planes: Duration,
    header: Duration,
    entropy: Duration,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            used: 0,
        }
    }

    fn write_bits(&mut self, code: u16, len: u8) {
        for bit_idx in (0..len).rev() {
            let bit = ((code >> bit_idx) & 1) as u8;
            self.current = (self.current << 1) | bit;
            self.used += 1;
            if self.used == 8 {
                self.push_byte(self.current);
                self.current = 0;
                self.used = 0;
            }
        }
    }

    fn align_with_ones(&mut self) {
        if self.used == 0 {
            return;
        }
        let remaining = 8 - self.used;
        self.current <<= remaining;
        self.current |= (1u8 << remaining) - 1;
        self.push_byte(self.current);
        self.current = 0;
        self.used = 0;
    }

    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.align_with_ones();
        self.bytes
    }

    fn push_byte(&mut self, byte: u8) {
        self.bytes.push(byte);
        if byte == 0xFF {
            self.bytes.push(0x00);
        }
    }
}

/// Encode grayscale or RGB samples as a baseline JPEG codestream.
pub fn encode_jpeg_baseline(
    samples: JpegSamples<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    match options.backend {
        JpegBackend::Auto | JpegBackend::Cpu => encode_jpeg_baseline_cpu(samples, options),
        JpegBackend::Metal => Err(JpegEncodeError::UnsupportedBackend {
            backend: options.backend,
        }),
    }
}

fn encode_jpeg_baseline_cpu(
    samples: JpegSamples<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    let profile_enabled = jpeg_profile_stages_enabled();
    let total_start = profile_enabled.then(Instant::now);
    let mut profile = JpegEncodeProfile::default();

    let validation_start = profile_enabled.then(Instant::now);
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;
    let (width, height) = samples.dimensions();
    let sample_format = samples.name();
    validate_jpeg_baseline_dimensions(width, height)?;
    samples.validate(options.subsampling)?;
    if let Some(start) = validation_start {
        profile.validation = start.elapsed();
    }

    let setup_start = profile_enabled.then(Instant::now);
    let tables = baseline_encode_tables(options)?;
    let sampling = tables.sampling;
    let cosine = cosine_table();
    if let Some(start) = setup_start {
        profile.setup = start.elapsed();
    }

    let planes_start = profile_enabled.then(Instant::now);
    let planes = component_planes(samples, options.subsampling)?;
    if let Some(start) = planes_start {
        profile.planes = start.elapsed();
    }

    let entropy_start = profile_enabled.then(Instant::now);
    let entropy = encode_entropy(
        &planes,
        width,
        height,
        sampling,
        &tables.q_luma,
        &tables.q_chroma,
        [&tables.huff_dc_luma, &tables.huff_dc_chroma],
        [&tables.huff_ac_luma, &tables.huff_ac_chroma],
        &cosine,
        options.restart_interval,
    )?;
    if let Some(start) = entropy_start {
        profile.entropy = start.elapsed();
    }
    let header_start = profile_enabled.then(Instant::now);
    let encoded =
        assemble_jpeg_baseline_frame(&entropy, width, height, &tables, options, JpegBackend::Cpu)?;
    if let Some(start) = header_start {
        profile.header = start.elapsed();
    }

    if let Some(start) = total_start {
        let width_s = width.to_string();
        let height_s = height.to_string();
        let quality_s = options.quality.to_string();
        let subsampling_s = format!("{:?}", options.subsampling);
        let restart_s = options.restart_interval.unwrap_or(0).to_string();
        let components_s = sampling.components.to_string();
        let output_bytes_s = encoded.data.len().to_string();
        let threads_s = rayon::current_num_threads().to_string();
        let validation_us = duration_us_string(profile.validation);
        let setup_us = duration_us_string(profile.setup);
        let planes_us = duration_us_string(profile.planes);
        let header_us = duration_us_string(profile.header);
        let entropy_us = duration_us_string(profile.entropy);
        let total_us = duration_us_string(start.elapsed());
        emit_jpeg_profile_row(
            "encode",
            "cpu",
            &[
                ("sample", sample_format),
                ("width", width_s.as_str()),
                ("height", height_s.as_str()),
                ("components", components_s.as_str()),
                ("quality", quality_s.as_str()),
                ("subsampling", subsampling_s.as_str()),
                ("restart_interval", restart_s.as_str()),
                ("validation_us", validation_us.as_str()),
                ("setup_us", setup_us.as_str()),
                ("planes_us", planes_us.as_str()),
                ("header_us", header_us.as_str()),
                ("entropy_us", entropy_us.as_str()),
                ("total_us", total_us.as_str()),
                ("output_bytes", output_bytes_s.as_str()),
                ("rayon_threads", threads_s.as_str()),
            ],
        );
    }

    Ok(encoded)
}

impl JpegSamples<'_> {
    fn name(self) -> &'static str {
        match self {
            Self::Gray8 { .. } => "Gray8",
            Self::Rgb8 { .. } => "Rgb8",
        }
    }

    fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Gray8 { width, height, .. } | Self::Rgb8 { width, height, .. } => (width, height),
        }
    }

    fn validate(self, subsampling: JpegSubsampling) -> Result<(), JpegEncodeError> {
        let (data, width, height, components, name) = match self {
            Self::Gray8 {
                data,
                width,
                height,
            } => (data, width, height, 1usize, "Gray8"),
            Self::Rgb8 {
                data,
                width,
                height,
            } => (data, width, height, 3usize, "Rgb8"),
        };
        let expected = width as usize * height as usize * components;
        if data.len() != expected {
            return Err(JpegEncodeError::SampleLength {
                expected,
                actual: data.len(),
            });
        }
        match (name, subsampling) {
            ("Gray8", JpegSubsampling::Gray) => Ok(()),
            (
                "Rgb8",
                JpegSubsampling::Ybr444 | JpegSubsampling::Ybr422 | JpegSubsampling::Ybr420,
            ) => Ok(()),
            _ => Err(JpegEncodeError::IncompatibleSubsampling {
                subsampling,
                samples: name,
            }),
        }
    }
}

fn component_planes(
    samples: JpegSamples<'_>,
    subsampling: JpegSubsampling,
) -> Result<Vec<Vec<u8>>, JpegEncodeError> {
    match samples {
        JpegSamples::Gray8 { data, .. } => Ok(vec![data.to_vec()]),
        JpegSamples::Rgb8 {
            data,
            width,
            height,
        } => {
            if subsampling == JpegSubsampling::Gray {
                return Err(JpegEncodeError::IncompatibleSubsampling {
                    subsampling,
                    samples: "Rgb8",
                });
            }
            let pixels = width as usize * height as usize;
            let mut y_plane = Vec::with_capacity(pixels);
            let mut cb_plane = Vec::with_capacity(pixels);
            let mut cr_plane = Vec::with_capacity(pixels);
            for rgb in data.chunks_exact(3) {
                let (y, cb, cr) = rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
                y_plane.push(y);
                cb_plane.push(cb);
                cr_plane.push(cr);
            }
            Ok(vec![y_plane, cb_plane, cr_plane])
        }
    }
}

fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = i32::from(r);
    let g = i32::from(g);
    let b = i32::from(b);
    let y = (19_595 * r + 38_470 * g + 7_471 * b + 32_768) >> 16;
    let cb = (-11_059 * r - 21_709 * g + 32_768 * b + 8_421_376) >> 16;
    let cr = (32_768 * r - 27_439 * g - 5_329 * b + 8_421_376) >> 16;
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

#[allow(clippy::too_many_arguments)]
fn encode_entropy(
    planes: &[Vec<u8>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    restart_interval: Option<u16>,
) -> Result<Vec<u8>, JpegEncodeError> {
    if let Some(restart_interval) = restart_interval {
        return encode_entropy_restart_segments(
            planes,
            width,
            height,
            sampling,
            q_luma,
            q_chroma,
            dc_tables,
            ac_tables,
            cosine,
            restart_interval,
        );
    }
    encode_entropy_serial(
        planes, width, height, sampling, q_luma, q_chroma, dc_tables, ac_tables, cosine, None,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_entropy_serial(
    planes: &[Vec<u8>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    restart_interval: Option<u16>,
) -> Result<Vec<u8>, JpegEncodeError> {
    let (mcus_per_row, total_mcus) = entropy_mcu_layout(width, height, sampling)?;
    if total_mcus == 0 {
        return Ok(Vec::new());
    }
    if let Some(restart_interval) = restart_interval {
        if restart_interval == 0 {
            return Err(JpegEncodeError::InvalidRestartInterval);
        }
        let restart_interval = u32::from(restart_interval);
        let mut out = Vec::new();
        let mut rst = 0u8;
        for start_mcu in (0..total_mcus).step_by(restart_interval as usize) {
            if start_mcu > 0 {
                out.push(0xFF);
                out.push(0xD0 + rst);
                rst = (rst + 1) & 7;
            }
            let end_mcu = start_mcu.saturating_add(restart_interval).min(total_mcus);
            out.extend_from_slice(&encode_entropy_mcu_range(
                planes,
                width,
                height,
                sampling,
                q_luma,
                q_chroma,
                dc_tables,
                ac_tables,
                cosine,
                mcus_per_row,
                start_mcu,
                end_mcu,
            )?);
        }
        return Ok(out);
    }

    encode_entropy_mcu_range(
        planes,
        width,
        height,
        sampling,
        q_luma,
        q_chroma,
        dc_tables,
        ac_tables,
        cosine,
        mcus_per_row,
        0,
        total_mcus,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_entropy_restart_segments(
    planes: &[Vec<u8>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    restart_interval: u16,
) -> Result<Vec<u8>, JpegEncodeError> {
    if restart_interval == 0 {
        return Err(JpegEncodeError::InvalidRestartInterval);
    }
    let (mcus_per_row, total_mcus) = entropy_mcu_layout(width, height, sampling)?;
    if total_mcus == 0 {
        return Ok(Vec::new());
    }
    let restart_interval = u32::from(restart_interval);
    let segment_count = total_mcus.div_ceil(restart_interval);
    let segments = (0..segment_count)
        .into_par_iter()
        .map(|segment_idx| {
            let start_mcu = segment_idx * restart_interval;
            let end_mcu = (start_mcu + restart_interval).min(total_mcus);
            encode_entropy_mcu_range(
                planes,
                width,
                height,
                sampling,
                q_luma,
                q_chroma,
                dc_tables,
                ac_tables,
                cosine,
                mcus_per_row,
                start_mcu,
                end_mcu,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut out = Vec::new();
    for (idx, segment) in segments.into_iter().enumerate() {
        if idx > 0 {
            out.push(0xFF);
            out.push(0xD0 + ((idx - 1) as u8 & 0x07));
        }
        out.extend_from_slice(&segment);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn encode_entropy_mcu_range(
    planes: &[Vec<u8>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    mcus_per_row: u32,
    start_mcu: u32,
    end_mcu: u32,
) -> Result<Vec<u8>, JpegEncodeError> {
    let mut writer = BitWriter::new();
    let mut prev_dc = [0i32; 3];
    for mcu_index in start_mcu..end_mcu {
        let mcu_y = mcu_index / mcus_per_row;
        let mcu_x = mcu_index % mcus_per_row;
        for_each_mcu_block(sampling, |component, block_x, block_y| {
            let quant = if component == 0 { q_luma } else { q_chroma };
            let dc_table = if component == 0 {
                dc_tables[0]
            } else {
                dc_tables[1]
            };
            let ac_table = if component == 0 {
                ac_tables[0]
            } else {
                ac_tables[1]
            };
            let block = sample_block(
                planes, width, height, sampling, component, mcu_x, mcu_y, block_x, block_y,
            );
            let coeffs = fdct_quantize(&block, quant, cosine);
            encode_block(
                &coeffs,
                &mut prev_dc[component],
                dc_table,
                ac_table,
                &mut writer,
            )
        })?;
    }
    Ok(writer.into_bytes())
}

fn entropy_mcu_layout(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
) -> Result<(u32, u32), JpegEncodeError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = width.div_ceil(mcu_width);
    let mcu_rows = height.div_ceil(mcu_height);
    let total_mcus = mcus_per_row
        .checked_mul(mcu_rows)
        .ok_or_else(|| JpegEncodeError::Internal("JPEG MCU count overflow".into()))?;
    Ok((mcus_per_row, total_mcus))
}

fn for_each_mcu_block<F>(
    sampling: JpegBaselineSampling,
    mut visit: F,
) -> Result<(), JpegEncodeError>
where
    F: FnMut(usize, u8, u8) -> Result<(), JpegEncodeError>,
{
    for component in 0..sampling.components as usize {
        for block_y in 0..sampling.v[component] {
            for block_x in 0..sampling.h[component] {
                visit(component, block_x, block_y)?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sample_block(
    planes: &[Vec<u8>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    component: usize,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u8,
    block_y: u8,
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let comp_h = u32::from(sampling.h[component]);
    let comp_v = u32::from(sampling.v[component]);
    let x_scale = max_h / comp_h;
    let y_scale = max_v / comp_v;
    let mcu_origin_x = mcu_x * max_h * 8;
    let mcu_origin_y = mcu_y * max_v * 8;
    for y in 0..8u32 {
        for x in 0..8u32 {
            let value = if component == 0 {
                let sx = (mcu_origin_x + u32::from(block_x) * 8 + x).min(width - 1);
                let sy = (mcu_origin_y + u32::from(block_y) * 8 + y).min(height - 1);
                planes[component][(sy as usize * width as usize) + sx as usize]
            } else {
                let mut sum = 0u32;
                for dy in 0..y_scale {
                    for dx in 0..x_scale {
                        let sx = (mcu_origin_x + (u32::from(block_x) * 8 + x) * x_scale + dx)
                            .min(width - 1);
                        let sy = (mcu_origin_y + (u32::from(block_y) * 8 + y) * y_scale + dy)
                            .min(height - 1);
                        sum += u32::from(
                            planes[component][sy as usize * width as usize + sx as usize],
                        );
                    }
                }
                (sum / (x_scale * y_scale)) as u8
            };
            out[(y * 8 + x) as usize] = value;
        }
    }
    out
}

fn fdct_quantize(block: &[u8; 64], quant: &[u8; 64], cosine: &[[f64; 8]; 8]) -> [i32; 64] {
    let mut coeffs = [0i32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0;
            for y in 0..8 {
                for x in 0..8 {
                    let sample = f64::from(block[y * 8 + x]) - 128.0;
                    sum += sample * cosine[u][x] * cosine[v][y];
                }
            }
            let cu = if u == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let cv = if v == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let natural = v * 8 + u;
            let transformed = 0.25 * cu * cv * sum;
            coeffs[natural] = (transformed / f64::from(quant[natural])).round() as i32;
        }
    }
    coeffs
}

pub(crate) fn encode_block(
    coeffs: &[i32; 64],
    prev_dc: &mut i32,
    dc_table: &JpegBaselineHuffmanTable,
    ac_table: &JpegBaselineHuffmanTable,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    let diff = coeffs[0] - *prev_dc;
    *prev_dc = coeffs[0];
    let dc_size = magnitude_category(diff);
    write_huffman_symbol(dc_table, dc_size, writer)?;
    if dc_size > 0 {
        writer.write_bits(magnitude_bits(diff, dc_size), dc_size);
    }

    let mut zero_run = 0u8;
    for k in 1..64 {
        let coeff = coeffs[JPEG_BASELINE_ZIGZAG[k] as usize];
        if coeff == 0 {
            zero_run = zero_run.saturating_add(1);
            continue;
        }
        while zero_run >= 16 {
            write_huffman_symbol(ac_table, 0xF0, writer)?;
            zero_run -= 16;
        }
        let size = magnitude_category(coeff);
        let symbol = (zero_run << 4) | size;
        write_huffman_symbol(ac_table, symbol, writer)?;
        writer.write_bits(magnitude_bits(coeff, size), size);
        zero_run = 0;
    }
    if zero_run > 0 {
        write_huffman_symbol(ac_table, 0, writer)?;
    }
    Ok(())
}

fn write_huffman_symbol(
    table: &JpegBaselineHuffmanTable,
    symbol: u8,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    let len = table.lens[symbol as usize];
    if len == 0 {
        return Err(JpegEncodeError::MissingHuffmanCode { symbol });
    }
    writer.write_bits(table.codes[symbol as usize], len);
    Ok(())
}

fn magnitude_category(value: i32) -> u8 {
    if value == 0 {
        return 0;
    }
    let mut abs = value.unsigned_abs();
    let mut size = 0u8;
    while abs > 0 {
        size += 1;
        abs >>= 1;
    }
    size
}

fn magnitude_bits(value: i32, size: u8) -> u16 {
    if size == 0 {
        return 0;
    }
    if value >= 0 {
        value as u16
    } else {
        (value + ((1i32 << size) - 1)) as u16
    }
}

fn cosine_table() -> [[f64; 8]; 8] {
    let mut table = [[0.0; 8]; 8];
    for u in 0..8 {
        for x in 0..8 {
            table[u][x] = (((2 * x + 1) as f64 * u as f64 * PI) / 16.0).cos();
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned_rgb(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 3) & 0xFF) as u8);
                pixels.push(((x * 5 + y * 11 + 40) & 0xFF) as u8);
                pixels.push(((x * 13 + y * 7 + 90) & 0xFF) as u8);
            }
        }
        pixels
    }

    #[test]
    fn restart_entropy_segments_match_serial_entropy() {
        let width = 160;
        let height = 80;
        let tables = baseline_encode_tables(JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: Some(64),
            backend: JpegBackend::Cpu,
        })
        .unwrap();
        let sampling = tables.sampling;
        let cosine = cosine_table();
        let pixels = patterned_rgb(width, height);
        let planes = component_planes(
            JpegSamples::Rgb8 {
                data: &pixels,
                width,
                height,
            },
            JpegSubsampling::Ybr422,
        )
        .unwrap();

        let serial = encode_entropy_serial(
            &planes,
            width,
            height,
            sampling,
            &tables.q_luma,
            &tables.q_chroma,
            [&tables.huff_dc_luma, &tables.huff_dc_chroma],
            [&tables.huff_ac_luma, &tables.huff_ac_chroma],
            &cosine,
            Some(64),
        )
        .unwrap();
        let segmented = encode_entropy_restart_segments(
            &planes,
            width,
            height,
            sampling,
            &tables.q_luma,
            &tables.q_chroma,
            [&tables.huff_dc_luma, &tables.huff_dc_chroma],
            [&tables.huff_ac_luma, &tables.huff_ac_chroma],
            &cosine,
            64,
        )
        .unwrap();

        assert_eq!(segmented, serial);
        assert!(segmented.windows(2).any(|window| window == [0xFF, 0xD0]));
    }
}
