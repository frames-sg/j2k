// SPDX-License-Identifier: MIT OR Apache-2.0

//! Experimental JPEG coefficient extraction for transcode pipelines.

use crate::adapter::{
    assemble_jpeg_baseline_frame_with_quant_tables, baseline_encode_tables,
    checked_encode_host_live_bytes, jpeg_baseline_entropy_capacity_bytes, JpegBaselineSampling,
};
use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, checked_allocation_len,
    try_reserve_for_len_with_live_budget,
};
use crate::decoder::{restart_index_allocation_bytes, Decoder};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;
use crate::encoder::{
    encode_block, BitWriter, JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSubsampling,
};
use crate::entropy::progressive::{
    decode_progressive_dct_blocks, PreparedProgressiveComponentPlan, ProgressiveDctBlocks,
};
use crate::entropy::sequential::{
    decode_scan_dct_blocks, DecodedDctBlocks, SequentialDctLifecycleMetadata,
};
use crate::entropy::ZIGZAG;
use crate::error::{JpegError, MarkerKind};
use crate::info::{ColorSpace, RestartIndex, SofKind};
use alloc::vec::Vec;

mod validation;
use self::validation::validate_baseline_dct_image;
pub use self::validation::JpegDctImageError;

/// Options for experimental DCT block extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DctExtractOptions {
    /// Whether to retain quantized DCT blocks in the extracted image.
    ///
    /// This defaults to true because JPEG DCT re-emission needs quantized
    /// coefficients. Coefficient-domain transcode paths that only need
    /// dequantized coefficients can disable this to avoid extra block writes.
    pub retain_quantized_blocks: bool,
}

impl DctExtractOptions {
    /// Extract only dequantized DCT blocks.
    #[must_use]
    pub const fn dequantized_only() -> Self {
        Self {
            retain_quantized_blocks: false,
        }
    }
}

impl Default for DctExtractOptions {
    fn default() -> Self {
        Self {
            retain_quantized_blocks: true,
        }
    }
}

/// JPEG image represented as entropy-decoded DCT blocks.
///
/// Coefficient planes can approach the shared host-allocation cap. This owner
/// is intentionally move-only; pass shared images by reference or place them
/// behind `Arc` instead of duplicating all coefficient storage.
#[derive(Debug, PartialEq, Eq)]
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

impl JpegDctImage {
    /// Allocator-reported bytes retained by coefficient and restart-index
    /// backing vectors.
    ///
    /// Cross-codec pipelines use this after extraction because an allocator
    /// may return more capacity than the requested logical lengths.
    #[doc(hidden)]
    pub fn retained_bytes(&self) -> Result<usize, JpegError> {
        let component_bytes =
            dct_component_capacity_bytes(self.components.capacity(), &self.components)?;
        let restart_bytes = restart_index_capacity_bytes(self.restart_index.as_ref())?;
        checked_add_allocation_bytes(component_bytes, restart_bytes)
    }
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
///
/// The quantized and dequantized coefficient vectors are intentionally
/// move-only because their aggregate capacity is governed by the image-level
/// host-allocation budget.
#[derive(Debug, PartialEq, Eq)]
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
    options: DctExtractOptions,
) -> Result<JpegDctImage, JpegError> {
    let decoder = Decoder::new(bytes)?;
    match decoder.info().color_space {
        ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb => {}
        color_space => return Err(JpegError::UnsupportedColorSpace { color_space }),
    }

    let workspace_cap = decoder.decode_workspace_cap()?;
    let planned_restart_index_bytes =
        restart_index_allocation_bytes(decoder.info(), decoder.plan.restart_interval)?;
    let (coding_mode, components) = match decoder.info().sof_kind {
        SofKind::Baseline8 => {
            let scan_bytes = &decoder.bytes[decoder.plan.scan_offset..];
            let lifecycle = SequentialDctLifecycleMetadata::new(
                checked_allocation_bytes::<JpegDctComponent>(decoder.info().sampling.len())?,
                planned_restart_index_bytes,
                workspace_cap,
            );
            let decoded_blocks = decode_scan_dct_blocks(
                &decoder.plan,
                scan_bytes,
                options.retain_quantized_blocks,
                lifecycle,
            )?;
            (
                JpegDctCodingMode::BaselineSequential,
                build_sequential_components(&decoder, decoded_blocks, workspace_cap)?,
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
            validate_progressive_extraction_workspace(
                &progressive_plan.components,
                options.retain_quantized_blocks,
                workspace_cap,
            )?;
            let decoded_blocks = decode_progressive_dct_blocks(progressive_plan, decoder.bytes, 0)?;
            (
                JpegDctCodingMode::Progressive,
                build_progressive_components(
                    &decoder,
                    decoded_blocks,
                    options.retain_quantized_blocks,
                    workspace_cap,
                )?,
            )
        }
        other => return Err(JpegError::NotImplemented { sof: other }),
    };
    let component_live_bytes = dct_component_capacity_bytes(components.capacity(), &components)?;
    ensure_dct_output_phase(
        component_live_bytes,
        planned_restart_index_bytes,
        workspace_cap,
    )?;
    let restart_index = decoder.restart_index()?;
    ensure_dct_output_phase(
        component_live_bytes,
        restart_index_capacity_bytes(restart_index.as_ref())?,
        workspace_cap,
    )?;

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
///
/// # Errors
///
/// Returns [`JpegEncodeError::InvalidDctImage`] when caller-supplied coding,
/// dimensions, component order, sampling, block grids, or quantization tables
/// cannot form the supported canonical baseline stream. Allocation, output,
/// and entropy-coding failures retain their existing encoder error variants.
pub fn encode_baseline_dct_image(image: &JpegDctImage) -> Result<Vec<u8>, JpegEncodeError> {
    let validated = validate_baseline_dct_image(image)
        .map_err(|reason| JpegEncodeError::InvalidDctImage { reason })?;
    let sampling = validated.sampling;
    let entropy_capacity =
        jpeg_baseline_entropy_capacity_bytes(image.width, image.height, sampling, None)?;
    validate_dct_reemission_live_bytes(entropy_capacity)?;

    let huffman_tables = baseline_encode_tables(JpegEncodeOptions {
        quality: 90,
        subsampling: if sampling.components == 1 {
            JpegSubsampling::Gray
        } else {
            JpegSubsampling::Ybr420
        },
        restart_interval: None,
        backend: JpegBackend::Cpu,
    })?;
    let dc_tables = [&huffman_tables.huff_dc_luma, &huffman_tables.huff_dc_chroma];
    let ac_tables = [&huffman_tables.huff_ac_luma, &huffman_tables.huff_ac_chroma];
    let entropy = encode_dct_entropy(image, sampling, dc_tables, ac_tables, entropy_capacity)?;
    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy.len())?;
    checked_encode_host_live_bytes([entropy.capacity(), frame_capacity])?;
    let encoded = assemble_jpeg_baseline_frame_with_quant_tables(
        &entropy,
        image.width,
        image.height,
        sampling,
        &validated.luma_quant,
        validated.chroma_quant.as_ref(),
        JpegBackend::Cpu,
    )?;
    checked_encode_host_live_bytes([entropy.capacity(), encoded.data.capacity()])?;
    Ok(encoded.data)
}

fn validate_dct_reemission_live_bytes(entropy_capacity: usize) -> Result<usize, JpegEncodeError> {
    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy_capacity)?;
    checked_encode_host_live_bytes([entropy_capacity, frame_capacity])
}

/// Run the scalar ISLOW IDCT oracle on one dequantized natural-order DCT block.
///
/// The output matches j2k-jpeg's scalar decode semantics for one component
/// block, including JPEG's unsigned sample level shift and clamping.
#[must_use]
pub fn idct_islow_block(block: &[i16; 64]) -> [u8; 64] {
    let mut output = [0; 64];
    crate::idct::idct_islow(block, &mut output);
    output
}

fn encode_dct_entropy(
    image: &JpegDctImage,
    sampling: JpegBaselineSampling,
    dc_tables: [&crate::adapter::JpegBaselineHuffmanTable; 2],
    ac_tables: [&crate::adapter::JpegBaselineHuffmanTable; 2],
    entropy_capacity: usize,
) -> Result<Vec<u8>, JpegEncodeError> {
    let mcu_cols = image.width.div_ceil(u32::from(sampling.max_h) * 8);
    let mcu_rows = image.height.div_ceil(u32::from(sampling.max_v) * 8);
    let mut writer = BitWriter::try_with_max_bytes(entropy_capacity)?;
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
    writer.into_bytes()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated sequential component counts fit JPEG's component-count byte"
)]
fn build_sequential_components(
    decoder: &Decoder<'_>,
    mut decoded_blocks: DecodedDctBlocks,
    workspace_cap: usize,
) -> Result<Vec<JpegDctComponent>, JpegError> {
    let dimensions = decoder.info().dimensions;
    let sampling = decoder.info().sampling;
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let mcu_cols = dimensions.0.div_ceil(8 * max_h);
    let mcu_rows = dimensions.1.div_ceil(8 * max_v);

    let mut live_bytes = decoded_blocks.capacity_bytes()?;
    let mut components = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut components,
        sampling.len(),
        &mut live_bytes,
        workspace_cap,
    )?;
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
            .get_mut(component_index)
            .map(core::mem::take)
            .ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?;
        let dequantized_blocks = decoded_blocks
            .dequantized
            .get_mut(component_index)
            .map(core::mem::take)
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
            quant_table: plan_component.quant,
            quantized_blocks,
            dequantized_blocks,
        });
    }

    Ok(components)
}

fn dct_component_capacity_bytes(
    outer_capacity: usize,
    components: &[JpegDctComponent],
) -> Result<usize, JpegError> {
    let mut total = checked_allocation_bytes::<JpegDctComponent>(outer_capacity)?;
    for component in components {
        total = checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<[i16; 64]>(component.quantized_blocks.capacity())?,
        )?;
        total = checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<[i16; 64]>(component.dequantized_blocks.capacity())?,
        )?;
    }
    Ok(total)
}

fn restart_index_capacity_bytes(index: Option<&RestartIndex>) -> Result<usize, JpegError> {
    index.map_or(Ok(0), |index| {
        checked_allocation_bytes::<crate::info::RestartSegment>(index.segments.capacity())
    })
}

fn ensure_dct_output_phase(
    initial: usize,
    additional: usize,
    workspace_cap: usize,
) -> Result<(), JpegError> {
    let requested = initial
        .checked_add(additional)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: workspace_cap,
        })?;
    if requested > workspace_cap {
        return Err(JpegError::MemoryCapExceeded {
            requested,
            cap: workspace_cap,
        });
    }
    Ok(())
}

fn build_progressive_components(
    decoder: &Decoder<'_>,
    mut decoded_blocks: ProgressiveDctBlocks,
    retain_quantized_blocks: bool,
    workspace_cap: usize,
) -> Result<Vec<JpegDctComponent>, JpegError> {
    let plan = decoder
        .progressive_plan
        .as_ref()
        .ok_or(JpegError::NotImplemented {
            sof: SofKind::Progressive8,
        })?;
    let mut live_bytes = decoded_blocks.capacity_bytes()?;
    let mut components = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut components,
        plan.components.len(),
        &mut live_bytes,
        workspace_cap,
    )?;
    for component in &plan.components {
        let quantized_i32 = decoded_blocks
            .quantized
            .get_mut(component.output_index)
            .map(core::mem::take)
            .ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?;
        let mut quantized_blocks = Vec::new();
        if retain_quantized_blocks {
            try_reserve_for_len_with_live_budget(
                &mut quantized_blocks,
                quantized_i32.len(),
                &mut live_bytes,
                workspace_cap,
            )?;
        }
        let mut dequantized_blocks = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut dequantized_blocks,
            quantized_i32.len(),
            &mut live_bytes,
            workspace_cap,
        )?;
        for block in &quantized_i32 {
            if retain_quantized_blocks {
                quantized_blocks.push(quantized_i16_block(block));
            }
            let dequantized = dequantize_progressive_block(block, &component.quant);
            dequantized_blocks.push(dequantized);
        }
        let released_bytes = checked_allocation_bytes::<[i32; 64]>(quantized_i32.capacity())?;
        drop(quantized_i32);
        live_bytes =
            live_bytes
                .checked_sub(released_bytes)
                .ok_or(JpegError::InternalInvariant {
                    reason: "progressive DCT live-byte accounting underflow",
                })?;

        components.push(JpegDctComponent {
            component_index: component.output_index,
            width: component.sample_width,
            height: component.sample_height,
            h_samp: component.h,
            v_samp: component.v,
            block_cols: component.block_cols,
            block_rows: component.block_rows,
            quant_table: component.quant,
            quantized_blocks,
            dequantized_blocks,
        });
    }

    Ok(components)
}

fn validate_progressive_extraction_workspace(
    components: &[PreparedProgressiveComponentPlan],
    retain_quantized_blocks: bool,
    workspace_cap: usize,
) -> Result<(), JpegError> {
    let decoded_metadata = checked_allocation_bytes::<Vec<[i32; 64]>>(components.len())?;
    let output_metadata = checked_allocation_bytes::<JpegDctComponent>(components.len())?;
    let mut total = checked_add_allocation_bytes(decoded_metadata, output_metadata)?;
    let output_plane_count = usize::from(retain_quantized_blocks) + 1;
    for component in components {
        let blocks = checked_allocation_len::<[i32; 64]>(
            component.block_cols as usize,
            component.block_rows as usize,
        )?;
        total =
            checked_add_allocation_bytes(total, checked_allocation_bytes::<[i32; 64]>(blocks)?)?;
        let output_blocks = checked_allocation_len::<[i16; 64]>(blocks, output_plane_count)?;
        total = checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<[i16; 64]>(output_blocks)?,
        )?;
    }
    ensure_dct_output_phase(0, total, workspace_cap)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "quantized coefficients are explicitly clamped to i16 before storage"
)]
fn quantized_i16_block(block: &[i32; 64]) -> [i16; 64] {
    let mut out = [0i16; 64];
    for (dst, &value) in out.iter_mut().zip(block.iter()) {
        *dst = value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    out
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "dequantized coefficients are explicitly clamped to i16 before storage"
)]
fn dequantize_progressive_block(block: &[i32; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut out = [0i16; 64];
    for (zigzag_idx, &natural_idx) in ZIGZAG.iter().enumerate() {
        let value = block[usize::from(natural_idx)].wrapping_mul(i32::from(quant[zigzag_idx]));
        out[usize::from(natural_idx)] =
            value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
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
    fn progressive_extraction_rejects_aggregate_live_planes() {
        let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
        let blocks = cap / core::mem::size_of::<[i32; 64]>() * 3 / 5;
        let block_cols = u32::try_from(blocks).expect("test block count fits u32");
        let components = [PreparedProgressiveComponentPlan {
            h: 1,
            v: 1,
            output_index: 0,
            quant: [1; 64],
            block_cols,
            block_rows: 1,
            sample_width: block_cols.saturating_mul(8),
            sample_height: 8,
        }];

        assert!(validate_progressive_extraction_workspace(&components, false, cap).is_ok());
        assert!(matches!(
            validate_progressive_extraction_workspace(&components, true, cap),
            Err(JpegError::MemoryCapExceeded { requested, cap: limit })
                if requested > limit && limit == cap
        ));
    }

    #[test]
    fn retained_decoder_metadata_reduces_the_extraction_workspace() {
        let workspace_cap = 512;

        ensure_dct_output_phase(400, 112, workspace_cap).expect("exact workspace boundary");
        assert!(matches!(
            ensure_dct_output_phase(400, 113, workspace_cap),
            Err(JpegError::MemoryCapExceeded {
                requested: 513,
                cap: 512,
            })
        ));
    }

    #[test]
    fn dct_reemission_counts_entropy_and_frame_at_the_shared_cap() {
        let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
        let overhead = crate::encoded_output::JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY;
        let entropy = (cap - overhead) / 2;

        assert_eq!(
            validate_dct_reemission_live_bytes(entropy).expect("exact live boundary"),
            cap
        );
        assert!(matches!(
            validate_dct_reemission_live_bytes(entropy + 1),
            Err(JpegEncodeError::MemoryCapExceeded { requested, cap: limit })
                if requested == cap + 2 && limit == cap
        ));
    }

    #[test]
    fn reemits_baseline_jpeg_from_extracted_quantized_dct_blocks() {
        let width = 32;
        let height = 24;
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                rgb.push(u8::try_from((x * 7 + y * 3) & 0xff).expect("fixture is byte-masked"));
                rgb.push(u8::try_from((x * 5 + y * 11) & 0xff).expect("fixture is byte-masked"));
                rgb.push(u8::try_from((x * 13 + y * 2) & 0xff).expect("fixture is byte-masked"));
            }
        }
        let encoded = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                width: u32::try_from(width).expect("fixture width fits in u32"),
                height: u32::try_from(height).expect("fixture height fits in u32"),
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
