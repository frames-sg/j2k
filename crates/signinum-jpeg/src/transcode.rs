// SPDX-License-Identifier: Apache-2.0

//! Experimental JPEG coefficient extraction for transcode pipelines.

use crate::adapter::{
    assemble_jpeg_baseline_frame_with_quant_tables, baseline_encode_tables, JpegBaselineSampling,
};
use crate::decoder::Decoder;
use crate::encoder::{
    encode_block, BitWriter, JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSubsampling,
};
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

/// Re-emit a baseline JPEG from quantized DCT blocks.
///
/// This path writes a fresh JPEG header and entropy stream, but it does not
/// perform IDCT, chroma upsampling, color conversion, or pixel-domain JPEG
/// encoding. DC deltas are recalculated from each component's quantized DC
/// sequence as the entropy stream is written.
pub fn encode_baseline_dct_image(image: &JpegDctImage) -> Result<Vec<u8>, JpegEncodeError> {
    if image.coding_mode != JpegDctCodingMode::BaselineSequential {
        return Err(JpegEncodeError::Internal(
            "DCT JPEG re-emission supports baseline sequential input only".into(),
        ));
    }
    let component_count = image.components.len();
    if component_count != 1 && component_count != 3 {
        return Err(JpegEncodeError::Internal(format!(
            "DCT JPEG re-emission supports 1 or 3 components, got {component_count}"
        )));
    }
    let max_h = image
        .components
        .iter()
        .map(|component| component.h_samp)
        .max()
        .unwrap_or(1);
    let max_v = image
        .components
        .iter()
        .map(|component| component.v_samp)
        .max()
        .unwrap_or(1);
    if max_h == 0 || max_v == 0 {
        return Err(JpegEncodeError::Internal(
            "DCT JPEG re-emission requires nonzero sampling factors".into(),
        ));
    }
    let mut sampling = JpegBaselineSampling {
        components: component_count as u8,
        h: [0; 3],
        v: [0; 3],
        max_h,
        max_v,
    };
    for (idx, component) in image.components.iter().enumerate() {
        if component.component_index != idx {
            return Err(JpegEncodeError::Internal(
                "DCT JPEG components must be in SOF declaration order".into(),
            ));
        }
        sampling.h[idx] = component.h_samp;
        sampling.v[idx] = component.v_samp;
    }
    validate_dct_component_grids(image, sampling)?;

    let luma_quant = zigzag_quant_to_natural_u8(&image.components[0].quant_table)?;
    let chroma_quant = if component_count == 3 {
        if image.components[1].quant_table != image.components[2].quant_table {
            return Err(JpegEncodeError::Internal(
                "DCT JPEG re-emission supports one shared chroma quant table".into(),
            ));
        }
        Some(zigzag_quant_to_natural_u8(
            &image.components[1].quant_table,
        )?)
    } else {
        None
    };
    let huffman_tables = baseline_encode_tables(JpegEncodeOptions {
        quality: 90,
        subsampling: if component_count == 1 {
            JpegSubsampling::Gray
        } else {
            JpegSubsampling::Ybr420
        },
        restart_interval: None,
        backend: JpegBackend::Cpu,
    })?;
    let dc_tables = [&huffman_tables.huff_dc_luma, &huffman_tables.huff_dc_chroma];
    let ac_tables = [&huffman_tables.huff_ac_luma, &huffman_tables.huff_ac_chroma];
    let entropy = encode_dct_entropy(image, sampling, dc_tables, ac_tables)?;
    let encoded = assemble_jpeg_baseline_frame_with_quant_tables(
        &entropy,
        image.width,
        image.height,
        sampling,
        &luma_quant,
        chroma_quant.as_ref(),
        JpegBackend::Cpu,
    )?;
    Ok(encoded.data)
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

fn validate_dct_component_grids(
    image: &JpegDctImage,
    sampling: JpegBaselineSampling,
) -> Result<(), JpegEncodeError> {
    let mcu_cols = image.width.div_ceil(u32::from(sampling.max_h) * 8);
    let mcu_rows = image.height.div_ceil(u32::from(sampling.max_v) * 8);
    for (idx, component) in image.components.iter().enumerate() {
        let expected_block_cols = mcu_cols * u32::from(sampling.h[idx]);
        let expected_block_rows = mcu_rows * u32::from(sampling.v[idx]);
        let expected_blocks = expected_block_cols
            .checked_mul(expected_block_rows)
            .ok_or_else(|| JpegEncodeError::Internal("DCT block count overflow".into()))?;
        if component.block_cols != expected_block_cols
            || component.block_rows != expected_block_rows
            || component.quantized_blocks.len() != expected_blocks as usize
        {
            return Err(JpegEncodeError::Internal(format!(
                "DCT component {idx} grid is {}x{} blocks with {} blocks, expected {}x{} and {} blocks",
                component.block_cols,
                component.block_rows,
                component.quantized_blocks.len(),
                expected_block_cols,
                expected_block_rows,
                expected_blocks
            )));
        }
    }
    Ok(())
}

fn zigzag_quant_to_natural_u8(quant: &[u16; 64]) -> Result<[u8; 64], JpegEncodeError> {
    let mut natural = [0u8; 64];
    for (zigzag_idx, &natural_idx) in ZIGZAG.iter().enumerate() {
        natural[usize::from(natural_idx)] = u8::try_from(quant[zigzag_idx]).map_err(|_| {
            JpegEncodeError::Internal(
                "DCT JPEG re-emission supports 8-bit quant tables only".into(),
            )
        })?;
    }
    Ok(natural)
}

fn encode_dct_entropy(
    image: &JpegDctImage,
    sampling: JpegBaselineSampling,
    dc_tables: [&crate::adapter::JpegBaselineHuffmanTable; 2],
    ac_tables: [&crate::adapter::JpegBaselineHuffmanTable; 2],
) -> Result<Vec<u8>, JpegEncodeError> {
    let mcu_cols = image.width.div_ceil(u32::from(sampling.max_h) * 8);
    let mcu_rows = image.height.div_ceil(u32::from(sampling.max_v) * 8);
    let mut writer = BitWriter::new();
    let mut prev_dc = [0i32; 3];
    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcu_cols {
            for (component_idx, prev_dc_component) in prev_dc
                .iter_mut()
                .enumerate()
                .take(sampling.components as usize)
            {
                let component = &image.components[component_idx];
                let table_idx = usize::from(component_idx != 0);
                for block_y in 0..sampling.v[component_idx] {
                    for block_x in 0..sampling.h[component_idx] {
                        let source_block_x =
                            mcu_x * u32::from(sampling.h[component_idx]) + u32::from(block_x);
                        let source_block_y =
                            mcu_y * u32::from(sampling.v[component_idx]) + u32::from(block_y);
                        let block_idx =
                            (source_block_y * component.block_cols + source_block_x) as usize;
                        let mut coeffs = [0i32; 64];
                        for (dst, &src) in coeffs
                            .iter_mut()
                            .zip(component.quantized_blocks[block_idx].iter())
                        {
                            *dst = i32::from(src);
                        }
                        encode_block(
                            &coeffs,
                            prev_dc_component,
                            dc_tables[table_idx],
                            ac_tables[table_idx],
                            &mut writer,
                        )?;
                    }
                }
            }
        }
    }
    Ok(writer.into_bytes())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
    };

    #[test]
    fn reemits_baseline_jpeg_from_extracted_quantized_dct_blocks() {
        let width = 32;
        let height = 24;
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                rgb.push(((x * 7 + y * 3) & 0xff) as u8);
                rgb.push(((x * 5 + y * 11) & 0xff) as u8);
                rgb.push(((x * 13 + y * 2) & 0xff) as u8);
            }
        }
        let encoded = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                width: width as u32,
                height: height as u32,
                data: &rgb,
            },
            JpegEncodeOptions {
                quality: 83,
                subsampling: JpegSubsampling::Ybr420,
                restart_interval: Some(2),
                backend: JpegBackend::Cpu,
            },
        )
        .expect("encode source jpeg");
        let source = extract_dct_blocks(&encoded.data, DctExtractOptions::default())
            .expect("extract source dct");

        let reemitted = encode_baseline_dct_image(&source).expect("re-emit dct jpeg");
        let actual = extract_dct_blocks(&reemitted, DctExtractOptions::default())
            .expect("extract re-emitted dct");

        assert_eq!(actual.width, source.width);
        assert_eq!(actual.height, source.height);
        assert_eq!(actual.color_space, source.color_space);
        assert_eq!(actual.components.len(), source.components.len());
        for (actual, expected) in actual.components.iter().zip(source.components.iter()) {
            assert_eq!(actual.width, expected.width);
            assert_eq!(actual.height, expected.height);
            assert_eq!(actual.h_samp, expected.h_samp);
            assert_eq!(actual.v_samp, expected.v_samp);
            assert_eq!(actual.quant_table, expected.quant_table);
            assert_eq!(actual.quantized_blocks, expected.quantized_blocks);
        }
    }
}
