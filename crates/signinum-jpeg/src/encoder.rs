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

/// Backend selected for baseline JPEG encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegBackend {
    /// Let the codec choose a backend.
    Auto,
    /// Portable CPU encoder.
    Cpu,
    /// Metal encoder adapter.
    Metal,
}

/// Baseline JPEG chroma subsampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegSubsampling {
    /// Single-channel grayscale.
    Gray,
    /// Three-component 4:4:4 YBR.
    Ybr444,
    /// Three-component 4:2:2 YBR.
    Ybr422,
    /// Three-component 4:2:0 YBR.
    Ybr420,
}

/// Options for baseline JPEG encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegEncodeOptions {
    /// JPEG quality in `1..=100`; values outside this range are clamped.
    pub quality: u8,
    /// Chroma subsampling mode.
    pub subsampling: JpegSubsampling,
    /// Optional restart interval in MCUs.
    pub restart_interval: Option<u16>,
    /// Requested backend.
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

impl JpegEncodeOptions {
    /// Create baseline JPEG encode options.
    pub const fn new(
        quality: u8,
        subsampling: JpegSubsampling,
        restart_interval: Option<u16>,
        backend: JpegBackend,
    ) -> Self {
        Self {
            quality,
            subsampling,
            restart_interval,
            backend,
        }
    }
}

/// Borrowed sample data for baseline JPEG encoding.
#[derive(Debug, Clone, Copy)]
pub enum JpegSamples<'a> {
    /// 8-bit grayscale samples.
    Gray8 {
        /// Tightly packed sample bytes.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// Interleaved 8-bit RGB samples.
    Rgb8 {
        /// Tightly packed RGB bytes.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
}

/// Encoded JPEG bytes and backend metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedJpeg {
    /// Complete JPEG interchange bytes.
    pub data: Vec<u8>,
    /// Backend that produced the bytes.
    pub backend: JpegBackend,
}

/// Error returned by baseline JPEG encoding.
#[derive(Debug, Error)]
pub enum JpegEncodeError {
    /// Width or height was zero.
    #[error("JPEG encode requires nonzero dimensions")]
    EmptyDimensions,
    /// Width or height exceeds the baseline JPEG field size.
    #[error("JPEG baseline dimensions must fit in u16, got {width}x{height}")]
    DimensionsTooLarge {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },
    /// Sample buffer length does not match geometry and format.
    #[error("JPEG sample buffer length mismatch: expected {expected}, got {actual}")]
    SampleLength {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// Requested subsampling cannot be used with the sample format.
    #[error("JPEG subsampling {subsampling:?} is incompatible with {samples}")]
    IncompatibleSubsampling {
        /// Requested subsampling.
        subsampling: JpegSubsampling,
        /// Sample format name.
        samples: &'static str,
    },
    /// Restart interval was zero.
    #[error("JPEG restart interval must be nonzero when provided")]
    InvalidRestartInterval,
    /// Requested encode backend is unavailable.
    #[error("JPEG encode backend {backend:?} is unavailable in signinum-jpeg CPU crate")]
    UnsupportedBackend {
        /// Requested backend.
        backend: JpegBackend,
    },
    /// Marker segment exceeded JPEG length limits.
    #[error("JPEG encoded marker segment is too large: {name}")]
    SegmentTooLarge {
        /// Marker segment name.
        name: &'static str,
    },
    /// Entropy encoder could not find a Huffman code.
    #[error("JPEG entropy symbol has no Huffman code: {symbol}")]
    MissingHuffmanCode {
        /// Missing symbol.
        symbol: u8,
    },
    /// Internal encode failure.
    #[error("JPEG encode failed: {0}")]
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

    fn push_restart_marker(&mut self, rst: u8) {
        self.align_with_ones();
        self.bytes.push(0xFF);
        self.bytes.push(0xD0 + (rst & 0x07));
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

/// Encode borrowed samples into baseline JPEG interchange bytes.
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
        validate_sample_len(data, width, height, components)?;
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

fn validate_sample_len(
    data: &[u8],
    width: u32,
    height: u32,
    components: usize,
) -> Result<usize, JpegEncodeError> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| JpegEncodeError::Internal("JPEG pixel count overflow".into()))?;
    let expected = pixels
        .checked_mul(components)
        .ok_or_else(|| JpegEncodeError::Internal("JPEG sample byte count overflow".into()))?;
    if data.len() != expected {
        return Err(JpegEncodeError::SampleLength {
            expected,
            actual: data.len(),
        });
    }
    Ok(pixels)
}

enum ComponentPlanes<'a> {
    Gray { data: &'a [u8] },
    Ycc { data: Vec<u8>, pixels: usize },
}

impl ComponentPlanes<'_> {
    #[inline]
    fn plane(&self, component: usize) -> &[u8] {
        match self {
            Self::Gray { data } => {
                debug_assert_eq!(component, 0);
                data
            }
            Self::Ycc { data, pixels } => {
                let start = component
                    .checked_mul(*pixels)
                    .expect("JPEG component index overflow");
                let end = start + *pixels;
                &data[start..end]
            }
        }
    }

    #[cfg(test)]
    fn allocation_count(&self) -> usize {
        match self {
            Self::Gray { .. } => 0,
            Self::Ycc { .. } => 1,
        }
    }
}

fn component_planes(
    samples: JpegSamples<'_>,
    subsampling: JpegSubsampling,
) -> Result<ComponentPlanes<'_>, JpegEncodeError> {
    match samples {
        JpegSamples::Gray8 {
            data,
            width,
            height,
        } => {
            if subsampling != JpegSubsampling::Gray {
                return Err(JpegEncodeError::IncompatibleSubsampling {
                    subsampling,
                    samples: "Gray8",
                });
            }
            validate_sample_len(data, width, height, 1)?;
            Ok(ComponentPlanes::Gray { data })
        }
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
            let pixels = validate_sample_len(data, width, height, 3)?;
            let plane_bytes = pixels
                .checked_mul(3)
                .expect("validated RGB plane byte count");
            let mut planes: Vec<u8> = Vec::with_capacity(plane_bytes);
            let y_plane = planes.as_mut_ptr();
            let cb_plane = y_plane.wrapping_add(pixels);
            let cr_plane = cb_plane.wrapping_add(pixels);
            for (idx, rgb) in data.chunks_exact(3).enumerate() {
                let (y, cb, cr) = rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
                // The validated RGB buffer yields exactly `pixels` loop indexes,
                // and the reserved buffer has three `pixels`-sized regions.
                unsafe {
                    y_plane.add(idx).write(y);
                    cb_plane.add(idx).write(cb);
                    cr_plane.add(idx).write(cr);
                }
            }
            // All three component planes are fully initialized above; u8 has no drop glue
            // if a panic occurs before this point.
            unsafe {
                planes.set_len(plane_bytes);
            }
            Ok(ComponentPlanes::Ycc {
                data: planes,
                pixels,
            })
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
    planes: &ComponentPlanes<'_>,
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
    planes: &ComponentPlanes<'_>,
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
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = width.div_ceil(mcu_width);
    let mcu_rows = height.div_ceil(mcu_height);
    let mut writer = BitWriter::new();
    let mut prev_dc = [0i32; 3];
    let mut mcus_since_restart = 0u16;
    let mut rst = 0u8;

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcus_per_row {
            if let Some(interval) = restart_interval {
                if mcus_since_restart == interval {
                    writer.push_restart_marker(rst);
                    rst = (rst + 1) & 7;
                    prev_dc = [0; 3];
                    mcus_since_restart = 0;
                }
            }
            for component in 0..sampling.components as usize {
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
                for block_y in 0..sampling.v[component] {
                    for block_x in 0..sampling.h[component] {
                        let block = sample_block(
                            planes, width, height, sampling, component, mcu_x, mcu_y, block_x,
                            block_y,
                        );
                        let coeffs = fdct_quantize(&block, quant, cosine);
                        encode_block(
                            &coeffs,
                            &mut prev_dc[component],
                            dc_table,
                            ac_table,
                            &mut writer,
                        )?;
                    }
                }
            }
            mcus_since_restart = mcus_since_restart.saturating_add(1);
        }
    }

    Ok(writer.into_bytes())
}

#[allow(clippy::too_many_arguments)]
fn encode_entropy_restart_segments(
    planes: &ComponentPlanes<'_>,
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
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = width.div_ceil(mcu_width);
    let mcu_rows = height.div_ceil(mcu_height);
    let total_mcus = mcus_per_row
        .checked_mul(mcu_rows)
        .ok_or_else(|| JpegEncodeError::Internal("JPEG MCU count overflow".into()))?;
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
    planes: &ComponentPlanes<'_>,
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
        for component in 0..sampling.components as usize {
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
            for block_y in 0..sampling.v[component] {
                for block_x in 0..sampling.h[component] {
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
                    )?;
                }
            }
        }
    }
    Ok(writer.into_bytes())
}

#[allow(clippy::too_many_arguments)]
fn sample_block(
    planes: &ComponentPlanes<'_>,
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
    let plane = planes.plane(component);
    let width_usize = width as usize;
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let comp_h = u32::from(sampling.h[component]);
    let comp_v = u32::from(sampling.v[component]);
    let x_scale = max_h / comp_h;
    let y_scale = max_v / comp_v;
    let mcu_origin_x = mcu_x * max_h * 8;
    let mcu_origin_y = mcu_y * max_v * 8;
    let direct_sample = component == 0 || (x_scale == 1 && y_scale == 1);
    let block_origin_x = mcu_origin_x + u32::from(block_x) * 8;
    let block_origin_y = mcu_origin_y + u32::from(block_y) * 8;

    if direct_sample && block_origin_x + 7 < width && block_origin_y + 7 < height {
        let src_x = block_origin_x as usize;
        let src_y = block_origin_y as usize;
        for row in 0..8usize {
            let src = (src_y + row) * width_usize + src_x;
            let dst = row * 8;
            out[dst..dst + 8].copy_from_slice(&plane[src..src + 8]);
        }
        return out;
    }

    for y in 0..8u32 {
        for x in 0..8u32 {
            let value = if direct_sample {
                let sx = (block_origin_x + x).min(width - 1);
                let sy = (block_origin_y + y).min(height - 1);
                plane[(sy as usize * width_usize) + sx as usize]
            } else {
                let mut sum = 0u32;
                for dy in 0..y_scale {
                    for dx in 0..x_scale {
                        let sx = (mcu_origin_x + (u32::from(block_x) * 8 + x) * x_scale + dx)
                            .min(width - 1);
                        let sy = (mcu_origin_y + (u32::from(block_y) * 8 + y) * y_scale + dy)
                            .min(height - 1);
                        sum += u32::from(plane[sy as usize * width_usize + sx as usize]);
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
    let mut rows = [[0.0; 8]; 8];

    for y in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0;
            for x in 0..8 {
                let sample = f64::from(block[y * 8 + x]) - 128.0;
                sum += sample * cosine[u][x];
            }
            rows[y][u] = sum;
        }
    }

    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0;
            for y in 0..8 {
                sum += rows[y][u] * cosine[v][y];
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
            let normalized = transformed / f64::from(quant[natural]);
            coeffs[natural] = if is_near_rounding_boundary(normalized) {
                fdct_quantize_coefficient_reference(block, quant, cosine, u, v)
            } else {
                normalized.round() as i32
            };
        }
    }
    coeffs
}

fn is_near_rounding_boundary(value: f64) -> bool {
    let fraction = value - value.floor();
    (fraction - 0.5).abs() <= 1.0e-9
}

fn fdct_quantize_coefficient_reference(
    block: &[u8; 64],
    quant: &[u8; 64],
    cosine: &[[f64; 8]; 8],
    u: usize,
    v: usize,
) -> i32 {
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
    (transformed / f64::from(quant[natural])).round() as i32
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

    fn reference_fdct_quantize(
        block: &[u8; 64],
        quant: &[u8; 64],
        cosine: &[[f64; 8]; 8],
    ) -> [i32; 64] {
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

    #[allow(clippy::too_many_arguments)]
    fn sample_block_reference(
        planes: &ComponentPlanes<'_>,
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
        let plane = planes.plane(component);
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
                    plane[(sy as usize * width as usize) + sx as usize]
                } else {
                    let mut sum = 0u32;
                    for dy in 0..y_scale {
                        for dx in 0..x_scale {
                            let sx = (mcu_origin_x + (u32::from(block_x) * 8 + x) * x_scale + dx)
                                .min(width - 1);
                            let sy = (mcu_origin_y + (u32::from(block_y) * 8 + y) * y_scale + dy)
                                .min(height - 1);
                            sum += u32::from(plane[sy as usize * width as usize + sx as usize]);
                        }
                    }
                    (sum / (x_scale * y_scale)) as u8
                };
                out[(y * 8 + x) as usize] = value;
            }
        }
        out
    }

    #[test]
    fn separable_fdct_matches_reference_quantized_coefficients() {
        let cosine = cosine_table();
        for quality in [50, 90, 98] {
            let tables = baseline_encode_tables(JpegEncodeOptions {
                quality,
                subsampling: JpegSubsampling::Ybr422,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            })
            .unwrap();

            for quant in [&tables.q_luma, &tables.q_chroma] {
                for seed in [0u32, 1, 17, 93, 251, 997] {
                    let mut block = [0u8; 64];
                    for (idx, sample) in block.iter_mut().enumerate() {
                        *sample =
                            ((idx as u32 * 37 + seed * 19 + (idx as u32 / 8) * 11) & 0xFF) as u8;
                    }
                    assert_eq!(
                        fdct_quantize(&block, quant, &cosine),
                        reference_fdct_quantize(&block, quant, &cosine),
                        "quality {quality}, seed {seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn sample_block_fast_paths_match_clamped_reference() {
        let gray_width = 17;
        let gray_height = 19;
        let gray: Vec<_> = (0..gray_width * gray_height)
            .map(|idx| ((idx * 7 + 13) & 0xFF) as u8)
            .collect();
        let gray_planes = component_planes(
            JpegSamples::Gray8 {
                data: &gray,
                width: gray_width,
                height: gray_height,
            },
            JpegSubsampling::Gray,
        )
        .unwrap();
        let gray_sampling = baseline_encode_tables(JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Gray,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        })
        .unwrap()
        .sampling;

        for (mcu_x, mcu_y) in [(0, 0), (2, 2)] {
            assert_eq!(
                sample_block(
                    &gray_planes,
                    gray_width,
                    gray_height,
                    gray_sampling,
                    0,
                    mcu_x,
                    mcu_y,
                    0,
                    0,
                ),
                sample_block_reference(
                    &gray_planes,
                    gray_width,
                    gray_height,
                    gray_sampling,
                    0,
                    mcu_x,
                    mcu_y,
                    0,
                    0,
                ),
                "gray mcu {mcu_x},{mcu_y}"
            );
        }

        let rgb_width = 17;
        let rgb_height = 19;
        let rgb = patterned_rgb(rgb_width, rgb_height);
        let rgb_planes = component_planes(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: rgb_width,
                height: rgb_height,
            },
            JpegSubsampling::Ybr444,
        )
        .unwrap();
        let rgb_sampling = baseline_encode_tables(JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        })
        .unwrap()
        .sampling;

        for component in 0..3 {
            for (mcu_x, mcu_y) in [(0, 0), (2, 2)] {
                assert_eq!(
                    sample_block(
                        &rgb_planes,
                        rgb_width,
                        rgb_height,
                        rgb_sampling,
                        component,
                        mcu_x,
                        mcu_y,
                        0,
                        0,
                    ),
                    sample_block_reference(
                        &rgb_planes,
                        rgb_width,
                        rgb_height,
                        rgb_sampling,
                        component,
                        mcu_x,
                        mcu_y,
                        0,
                        0,
                    ),
                    "Ybr444 component {component}, mcu {mcu_x},{mcu_y}"
                );
            }
        }
    }

    #[test]
    fn component_planes_borrow_gray_and_store_rgb_contiguously() {
        let gray = [3_u8, 7, 11, 19];
        let gray_planes = component_planes(
            JpegSamples::Gray8 {
                data: &gray,
                width: 2,
                height: 2,
            },
            JpegSubsampling::Gray,
        )
        .unwrap();
        assert_eq!(gray_planes.plane(0), &gray);
        assert_eq!(gray_planes.allocation_count(), 0);

        let rgb = [10_u8, 20, 30, 40, 50, 60];
        let rgb_planes = component_planes(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: 2,
                height: 1,
            },
            JpegSubsampling::Ybr444,
        )
        .unwrap();
        let (y0, cb0, cr0) = rgb_to_ycbcr(10, 20, 30);
        let (y1, cb1, cr1) = rgb_to_ycbcr(40, 50, 60);
        assert_eq!(rgb_planes.plane(0), &[y0, y1]);
        assert_eq!(rgb_planes.plane(1), &[cb0, cb1]);
        assert_eq!(rgb_planes.plane(2), &[cr0, cr1]);
        assert_eq!(rgb_planes.plane(0).len(), 2);
        assert_eq!(rgb_planes.plane(1).len(), 2);
        assert_eq!(rgb_planes.plane(2).len(), 2);
        assert_eq!(rgb_planes.allocation_count(), 1);
    }

    #[test]
    fn component_planes_revalidates_sample_lengths_before_storage() {
        let Err(gray_err) = component_planes(
            JpegSamples::Gray8 {
                data: &[1, 2, 3],
                width: 2,
                height: 2,
            },
            JpegSubsampling::Gray,
        ) else {
            panic!("invalid gray length should be rejected");
        };
        assert!(matches!(
            gray_err,
            JpegEncodeError::SampleLength {
                expected: 4,
                actual: 3
            }
        ));

        let Err(rgb_err) = component_planes(
            JpegSamples::Rgb8 {
                data: &[],
                width: 1,
                height: 1,
            },
            JpegSubsampling::Ybr444,
        ) else {
            panic!("invalid RGB length should be rejected");
        };
        assert!(matches!(
            rgb_err,
            JpegEncodeError::SampleLength {
                expected: 3,
                actual: 0
            }
        ));
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
