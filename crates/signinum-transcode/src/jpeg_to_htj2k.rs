// SPDX-License-Identifier: Apache-2.0

//! Experimental JPEG DCT to HTJ2K codestream transcode entry point.

use core::fmt;
use std::time::Instant;

use signinum_j2k_native::{
    encode_precomputed_htj2k_53, EncodeOptions, J2kForwardDwt53Level, J2kForwardDwt53Output,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
};
use signinum_jpeg::transcode::{extract_dct_blocks, DctExtractOptions, JpegDctComponent};

use crate::dct53_2d::{dct8x8_blocks_to_dwt53_float_linear, Dct53GridError};

/// Options for the experimental JPEG-to-HTJ2K path.
#[derive(Debug, Clone)]
pub struct JpegToHtj2kOptions {
    /// Native HTJ2K encode options used after wavelet bands are produced.
    pub encode_options: EncodeOptions,
}

impl Default for JpegToHtj2kOptions {
    fn default() -> Self {
        Self {
            encode_options: EncodeOptions {
                num_decomposition_levels: 1,
                reversible: true,
                use_ht_block_coding: true,
                use_mct: false,
                validate_high_throughput_codestream: false,
                ..EncodeOptions::default()
            },
        }
    }
}

/// Encoded transcode output and validation/report metadata.
#[derive(Debug, Clone)]
pub struct EncodedTranscode {
    /// HTJ2K codestream bytes.
    pub codestream: Vec<u8>,
    /// Summary of the experimental path used.
    pub report: TranscodeReport,
}

/// Transcode summary for validation and benchmarking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeReport {
    /// Source reference-grid width.
    pub width: u32,
    /// Source reference-grid height.
    pub height: u32,
    /// Number of transformed components.
    pub component_count: usize,
    /// Native transformed component geometry and SIZ sampling.
    pub components: Vec<TranscodeComponentReport>,
    /// Name of the experimental path used.
    pub path: &'static str,
    /// Wall-clock extraction time in microseconds.
    pub extract_us: u128,
    /// Wall-clock DCT-to-wavelet time in microseconds.
    pub transform_us: u128,
    /// Wall-clock HTJ2K encode time in microseconds.
    pub encode_us: u128,
}

/// Per-component transcode geometry preserved in the generated codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeComponentReport {
    /// Component index in JPEG SOF order.
    pub component_index: usize,
    /// Native component width in samples before HTJ2K SIZ expansion.
    pub width: u32,
    /// Native component height in samples before HTJ2K SIZ expansion.
    pub height: u32,
    /// Number of DCT blocks per component row, including padded edge blocks.
    pub block_cols: u32,
    /// Number of DCT block rows, including padded edge blocks.
    pub block_rows: u32,
    /// HTJ2K SIZ horizontal sampling factor.
    pub x_rsiz: u8,
    /// HTJ2K SIZ vertical sampling factor.
    pub y_rsiz: u8,
}

/// Error returned by the experimental transcode path.
#[derive(Debug)]
pub enum JpegToHtj2kError {
    /// JPEG parse or entropy decode failed.
    Jpeg(signinum_jpeg::JpegError),
    /// Input is outside the currently implemented experimental slice.
    Unsupported(&'static str),
    /// DCT block grid metadata did not cover the component dimensions.
    Grid(Dct53GridError),
    /// Native HTJ2K encode failed.
    Encode(&'static str),
}

impl fmt::Display for JpegToHtj2kError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jpeg(err) => write!(f, "JPEG extraction failed: {err}"),
            Self::Unsupported(reason) => write!(f, "unsupported transcode input: {reason}"),
            Self::Grid(err) => write!(f, "DCT grid transform failed: {err}"),
            Self::Encode(reason) => write!(f, "HTJ2K encode failed: {reason}"),
        }
    }
}

impl std::error::Error for JpegToHtj2kError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Jpeg(err) => Some(err),
            Self::Grid(err) => Some(err),
            Self::Unsupported(_) | Self::Encode(_) => None,
        }
    }
}

impl From<signinum_jpeg::JpegError> for JpegToHtj2kError {
    fn from(value: signinum_jpeg::JpegError) -> Self {
        Self::Jpeg(value)
    }
}

impl From<Dct53GridError> for JpegToHtj2kError {
    fn from(value: Dct53GridError) -> Self {
        Self::Grid(value)
    }
}

/// Transcode a constrained baseline grayscale JPEG tile into an HTJ2K
/// codestream using direct DCT-domain 5/3 wavelet coefficients.
///
/// Current implementation scope is baseline JPEG with one or more components
/// at native JPEG component resolution, and one reversible 5/3 decomposition
/// level. Component subsampling is preserved through SIZ `XRsiz`/`YRsiz`
/// instead of chroma upsampling.
pub fn jpeg_to_htj2k(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::default())?;
    let extract_us = extract_start.elapsed().as_micros();

    if jpeg.components.is_empty() || jpeg.components.len() > 4 {
        return Err(JpegToHtj2kError::Unsupported(
            "unsupported JPEG component count for jpeg_to_htj2k",
        ));
    }
    let component_sampling =
        component_sampling_for_jpeg(&jpeg.components, jpeg.width, jpeg.height)?;
    let all_unit_sampled = component_sampling
        .iter()
        .all(|&(x_rsiz, y_rsiz)| x_rsiz == 1 && y_rsiz == 1);
    let component_reports = jpeg
        .components
        .iter()
        .zip(component_sampling.iter().copied())
        .map(|(component, (x_rsiz, y_rsiz))| TranscodeComponentReport {
            component_index: component.component_index,
            width: component.width,
            height: component.height,
            block_cols: component.block_cols,
            block_rows: component.block_rows,
            x_rsiz,
            y_rsiz,
        })
        .collect();

    let transform_start = Instant::now();
    let precomputed_components = jpeg
        .components
        .iter()
        .zip(component_sampling.iter().copied())
        .map(|(component, (x_rsiz, y_rsiz))| {
            component_to_precomputed_htj2k(component, x_rsiz, y_rsiz)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transform_us = transform_start.elapsed().as_micros();

    let precomputed = PrecomputedHtj2k53Image {
        width: jpeg.width,
        height: jpeg.height,
        bit_depth: 8,
        signed: false,
        components: precomputed_components,
    };

    let encode_start = Instant::now();
    let codestream = encode_precomputed_htj2k_53(&precomputed, &options.encode_options)
        .map_err(JpegToHtj2kError::Encode)?;
    let encode_us = encode_start.elapsed().as_micros();

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width: jpeg.width,
            height: jpeg.height,
            component_count: jpeg.components.len(),
            components: component_reports,
            path: if all_unit_sampled {
                "full_resolution_components_float_direct_53"
            } else {
                "native_component_sampling_float_direct_53"
            },
            extract_us,
            transform_us,
            encode_us,
        },
    })
}

fn component_to_precomputed_htj2k(
    component: &JpegDctComponent,
    x_rsiz: u8,
    y_rsiz: u8,
) -> Result<PrecomputedHtj2k53Component, JpegToHtj2kError> {
    let blocks = dct_blocks_to_8x8_f64(&component.dequantized_blocks);
    let bands = dct8x8_blocks_to_dwt53_float_linear(
        &blocks,
        component.block_cols as usize,
        component.block_rows as usize,
        component.width as usize,
        component.height as usize,
    )?;
    let dwt = J2kForwardDwt53Output {
        ll: bands.ll.iter().map(|&value| value as f32).collect(),
        ll_width: bands.low_width as u32,
        ll_height: bands.low_height as u32,
        levels: vec![J2kForwardDwt53Level {
            hl: bands.hl.iter().map(|&value| value as f32).collect(),
            lh: bands.lh.iter().map(|&value| value as f32).collect(),
            hh: bands.hh.iter().map(|&value| value as f32).collect(),
            width: component.width,
            height: component.height,
            low_width: bands.low_width as u32,
            low_height: bands.low_height as u32,
            high_width: bands.high_width as u32,
            high_height: bands.high_height as u32,
        }],
    };

    Ok(PrecomputedHtj2k53Component {
        x_rsiz,
        y_rsiz,
        dwt,
    })
}

fn component_sampling_for_jpeg(
    components: &[JpegDctComponent],
    reference_width: u32,
    reference_height: u32,
) -> Result<Vec<(u8, u8)>, JpegToHtj2kError> {
    let max_h = components
        .iter()
        .map(|component| component.h_samp)
        .max()
        .ok_or(JpegToHtj2kError::Unsupported("missing JPEG components"))?;
    let max_v = components
        .iter()
        .map(|component| component.v_samp)
        .max()
        .ok_or(JpegToHtj2kError::Unsupported("missing JPEG components"))?;

    components
        .iter()
        .map(|component| {
            if component.h_samp == 0 || component.v_samp == 0 {
                return Err(JpegToHtj2kError::Unsupported(
                    "JPEG component sampling factors must be non-zero",
                ));
            }
            if max_h % component.h_samp != 0 || max_v % component.v_samp != 0 {
                return Err(JpegToHtj2kError::Unsupported(
                    "fractional JPEG component sampling is not supported",
                ));
            }

            let x_rsiz = max_h / component.h_samp;
            let y_rsiz = max_v / component.v_samp;
            let expected_width = reference_width.div_ceil(u32::from(x_rsiz));
            let expected_height = reference_height.div_ceil(u32::from(y_rsiz));
            if component.width != expected_width || component.height != expected_height {
                return Err(JpegToHtj2kError::Unsupported(
                    "JPEG component dimensions do not match derived SIZ sampling",
                ));
            }

            Ok((x_rsiz, y_rsiz))
        })
        .collect()
}

fn dct_blocks_to_8x8_f64(blocks: &[[i16; 64]]) -> Vec<[[f64; 8]; 8]> {
    blocks
        .iter()
        .map(|block| {
            let mut output = [[0.0; 8]; 8];
            for (idx, &coefficient) in block.iter().enumerate() {
                output[idx / 8][idx % 8] = f64::from(coefficient);
            }
            output
        })
        .collect()
}
