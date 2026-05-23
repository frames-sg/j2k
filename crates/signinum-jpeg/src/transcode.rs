// SPDX-License-Identifier: Apache-2.0

//! Experimental JPEG coefficient extraction for transcode pipelines.

use crate::decoder::Decoder;
use crate::entropy::progressive::{decode_progressive_dct_blocks, ProgressiveDctBlocks};
use crate::entropy::sequential::{decode_scan_dct_blocks, DecodedDctBlocks};
use crate::entropy::ZIGZAG;
use crate::error::{JpegError, MarkerKind};
use crate::info::{ColorSpace, RestartIndex, SofKind};
use alloc::vec::Vec;

/// Options for experimental DCT block extraction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DctExtractOptions {}

/// JPEG image represented as entropy-decoded DCT blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegDctImage {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// JPEG color space after APP14 detection.
    pub color_space: ColorSpace,
    /// Entropy coding mode that produced the extracted DCT coefficients.
    pub coding_mode: JpegDctCodingMode,
    /// Number of SOS marker segments parsed for this image.
    pub scan_count: u16,
    /// Components in SOF declaration order, each at native resolution.
    pub components: Vec<JpegDctComponent>,
    /// Restart-marker metadata when the stream uses a non-zero DRI interval.
    pub restart_index: Option<RestartIndex>,
}

/// JPEG DCT entropy coding mode represented by [`JpegDctImage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegDctCodingMode {
    /// SOF0 baseline sequential Huffman DCT.
    BaselineSequential,
    /// SOF2 progressive Huffman DCT with accumulated scan coefficients.
    Progressive,
}

/// One JPEG component's natural-order DCT blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegDctComponent {
    /// Component index in SOF declaration order.
    pub component_index: usize,
    /// Native component width in samples.
    pub width: u32,
    /// Native component height in samples.
    pub height: u32,
    /// Horizontal JPEG sampling factor.
    pub h_samp: u8,
    /// Vertical JPEG sampling factor.
    pub v_samp: u8,
    /// Number of 8x8 blocks per component row, including padded edge blocks.
    pub block_cols: u32,
    /// Number of 8x8 block rows, including padded edge blocks.
    pub block_rows: u32,
    /// Quantization table used by this component, in JPEG zigzag table order.
    pub quant_table: [u16; 64],
    /// Quantized DCT blocks in natural row-major coefficient order.
    pub quantized_blocks: Vec<[i16; 64]>,
    /// Dequantized DCT blocks in natural row-major coefficient order.
    pub dequantized_blocks: Vec<[i16; 64]>,
}

/// Extract quantized and dequantized natural-order DCT blocks from a baseline
/// sequential JPEG.
///
/// The returned data remains in JPEG component space. No IDCT, chroma upsample,
/// RGB conversion, or color transform is performed.
pub fn extract_dct_blocks(
    bytes: &[u8],
    _options: DctExtractOptions,
) -> Result<JpegDctImage, JpegError> {
    let decoder = Decoder::new(bytes)?;
    match decoder.info().color_space {
        ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb => {}
        color_space => return Err(JpegError::UnsupportedColorSpace { color_space }),
    }

    let (coding_mode, components) = match decoder.info().sof_kind {
        SofKind::Baseline8 => {
            let scan_bytes = &decoder.bytes[decoder.plan.scan_offset..];
            let decoded_blocks = decode_scan_dct_blocks(&decoder.plan, scan_bytes)?;
            (
                JpegDctCodingMode::BaselineSequential,
                build_sequential_components(&decoder, decoded_blocks)?,
            )
        }
        SofKind::Progressive8 => {
            let progressive_plan =
                decoder
                    .progressive_plan
                    .as_ref()
                    .ok_or(JpegError::NotImplemented {
                        sof: SofKind::Progressive8,
                    })?;
            let decoded_blocks = decode_progressive_dct_blocks(progressive_plan, decoder.bytes)?;
            (
                JpegDctCodingMode::Progressive,
                build_progressive_components(&decoder, decoded_blocks)?,
            )
        }
        other => return Err(JpegError::NotImplemented { sof: other }),
    };
    let restart_index = decoder.restart_index()?;

    Ok(JpegDctImage {
        width: decoder.info().dimensions.0,
        height: decoder.info().dimensions.1,
        color_space: decoder.info().color_space,
        coding_mode,
        scan_count: decoder.info().scan_count,
        components,
        restart_index,
    })
}

/// Run the scalar ISLOW IDCT oracle on one dequantized natural-order DCT block.
///
/// The output matches signinum-jpeg's scalar decode semantics for one component
/// block, including JPEG's unsigned sample level shift and clamping.
#[must_use]
pub fn idct_islow_block(block: &[i16; 64]) -> [u8; 64] {
    let mut output = [0; 64];
    crate::idct::idct_islow(block, &mut output);
    output
}

fn build_sequential_components(
    decoder: &Decoder<'_>,
    decoded_blocks: DecodedDctBlocks,
) -> Result<Vec<JpegDctComponent>, JpegError> {
    let dimensions = decoder.info().dimensions;
    let sampling = decoder.info().sampling;
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let mcu_cols = dimensions.0.div_ceil(8 * max_h);
    let mcu_rows = dimensions.1.div_ceil(8 * max_v);

    let mut components = Vec::with_capacity(sampling.len());
    for (component_index, &(h_samp, v_samp)) in sampling.components().iter().enumerate() {
        let plan_component = decoder
            .plan
            .components
            .iter()
            .find(|component| component.output_index == component_index)
            .ok_or(JpegError::InvalidSequentialComponentSet {
                offset: decoder.plan.scan_offset,
                expected: sampling.len() as u8,
                found: decoder.plan.components.len() as u8,
            })?;
        let quantized_blocks = decoded_blocks
            .quantized
            .get(component_index)
            .cloned()
            .ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?;
        let dequantized_blocks = decoded_blocks
            .dequantized
            .get(component_index)
            .cloned()
            .ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?;

        components.push(JpegDctComponent {
            component_index,
            width: dimensions
                .0
                .saturating_mul(u32::from(h_samp))
                .div_ceil(max_h),
            height: dimensions
                .1
                .saturating_mul(u32::from(v_samp))
                .div_ceil(max_v),
            h_samp,
            v_samp,
            block_cols: mcu_cols * u32::from(h_samp),
            block_rows: mcu_rows * u32::from(v_samp),
            quant_table: *plan_component.quant.as_ref(),
            quantized_blocks,
            dequantized_blocks,
        });
    }

    Ok(components)
}

fn build_progressive_components(
    decoder: &Decoder<'_>,
    decoded_blocks: ProgressiveDctBlocks,
) -> Result<Vec<JpegDctComponent>, JpegError> {
    let plan = decoder
        .progressive_plan
        .as_ref()
        .ok_or(JpegError::NotImplemented {
            sof: SofKind::Progressive8,
        })?;
    let mut components = Vec::with_capacity(plan.components.len());
    for component in &plan.components {
        let quantized_i32 = decoded_blocks.quantized.get(component.output_index).ok_or(
            JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            },
        )?;
        let mut quantized_blocks = Vec::with_capacity(quantized_i32.len());
        let mut dequantized_blocks = Vec::with_capacity(quantized_i32.len());
        for block in quantized_i32 {
            let quantized = quantized_i16_block(block);
            let dequantized = dequantize_progressive_block(block, &component.quant);
            quantized_blocks.push(quantized);
            dequantized_blocks.push(dequantized);
        }

        components.push(JpegDctComponent {
            component_index: component.output_index,
            width: component.sample_width,
            height: component.sample_height,
            h_samp: component.h,
            v_samp: component.v,
            block_cols: component.block_cols,
            block_rows: component.block_rows,
            quant_table: *component.quant.as_ref(),
            quantized_blocks,
            dequantized_blocks,
        });
    }

    Ok(components)
}

fn quantized_i16_block(block: &[i32; 64]) -> [i16; 64] {
    let mut out = [0i16; 64];
    for (dst, &value) in out.iter_mut().zip(block.iter()) {
        *dst = value.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    out
}

fn dequantize_progressive_block(block: &[i32; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut out = [0i16; 64];
    for (zigzag_idx, &natural_idx) in ZIGZAG.iter().enumerate() {
        let value = block[usize::from(natural_idx)].wrapping_mul(i32::from(quant[zigzag_idx]));
        out[usize::from(natural_idx)] = value.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    out
}
