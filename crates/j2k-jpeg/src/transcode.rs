// SPDX-License-Identifier: MIT OR Apache-2.0

//! Experimental JPEG coefficient extraction for transcode pipelines.

use crate::adapter::{
    assemble_jpeg_baseline_frame_with_metadata, assemble_jpeg_baseline_frame_with_quant_tables,
    baseline_encode_tables, checked_encode_host_live_bytes, jpeg_baseline_entropy_capacity_bytes,
    JpegBaselineFrameMetadata, JpegBaselineSampling,
};
use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, checked_allocation_len,
    try_reserve_for_len_with_live_budget,
};
use crate::baseline_entropy::{encode_block, BitWriter};
use crate::decoder::{restart_index_allocation_bytes, Decoder};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;
use crate::encoder::{JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSubsampling};
use crate::entropy::progressive::{
    decode_progressive_dct_blocks, PreparedProgressiveComponentPlan, ProgressiveDctBlocks,
};
use crate::entropy::sequential::{
    decode_scan_dct_blocks, decode_scan_quantized_dct_windows_into, DecodedDctBlocks, McuWindow,
    SequentialDctLifecycleMetadata,
};
use crate::entropy::ZIGZAG;
use crate::error::{JpegError, MarkerKind};
use crate::info::{ColorSpace, RestartIndex, SofKind};
use alloc::vec::Vec;

mod validation;
use self::validation::{
    validate_baseline_dct_image, validate_baseline_dct_image_with_independent_quant_tables,
};
pub use crate::dct_contract::{JpegDctCodingMode, JpegDctImageError};

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
    /// Whether to retain dequantized DCT blocks in the extracted image.
    ///
    /// Pixel-domain transcode paths need these blocks. Coefficient-domain
    /// re-emission and lossless cropping can disable them to halve baseline
    /// extraction storage.
    pub retain_dequantized_blocks: bool,
}

impl DctExtractOptions {
    /// Extract only dequantized DCT blocks.
    #[must_use]
    pub const fn dequantized_only() -> Self {
        Self {
            retain_quantized_blocks: false,
            retain_dequantized_blocks: true,
        }
    }

    /// Extract only quantized DCT blocks for coefficient-domain re-emission.
    #[must_use]
    pub const fn quantized_only() -> Self {
        Self {
            retain_quantized_blocks: true,
            retain_dequantized_blocks: false,
        }
    }
}

impl Default for DctExtractOptions {
    fn default() -> Self {
        Self {
            retain_quantized_blocks: true,
            retain_dequantized_blocks: true,
        }
    }
}

/// MCU-aligned rectangle for coefficient extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McuRect {
    /// First MCU column.
    pub x: u32,
    /// First MCU row.
    pub y: u32,
    /// Number of MCU columns.
    pub width: u32,
    /// Number of MCU rows.
    pub height: u32,
}

/// Source signalling retained while re-emitting quantized DCT blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaselineDctEncodeOptions<'a> {
    /// Complete `APPn` marker segments written immediately after SOI.
    pub app_segments: &'a [u8],
    /// SOF and SOS component identifiers in declaration order.
    pub component_ids: &'a [u8],
}

/// Extraction working storage estimated from JPEG headers before entropy decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DctWorkingSetEstimate {
    /// Bytes retained by coefficient planes and their result metadata.
    pub retained_bytes: usize,
    /// Peak bytes for coefficients, result owners, restart data, and progressive
    /// accumulation when needed.
    pub peak_bytes: usize,
}

/// Options for coefficient-domain lossless JPEG cropping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct LosslessCropOptions {
    /// Maximum extraction working set admitted before entropy decoding.
    pub max_working_bytes: usize,
}

impl Default for LosslessCropOptions {
    fn default() -> Self {
        Self {
            max_working_bytes: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        }
    }
}

/// Result of a coefficient-domain lossless crop.
#[derive(Debug, PartialEq, Eq)]
pub struct LosslessCrop {
    /// Standalone baseline JPEG containing the crop window.
    pub data: Vec<u8>,
    /// Encoded image width.
    pub width: u32,
    /// Encoded image height.
    pub height: u32,
    /// Requested rectangle inside the encoded image.
    pub region: crate::Rect,
}

/// Errors produced by coefficient-domain lossless cropping.
#[derive(Debug, thiserror::Error)]
pub enum LosslessCropError {
    /// Source parsing or coefficient extraction failed.
    #[error(transparent)]
    Decode(#[from] JpegError),
    /// Baseline re-emission failed.
    #[error(transparent)]
    Encode(#[from] JpegEncodeError),
}

impl DctExtractOptions {
    /// Estimate whole-image extraction storage without entropy decoding.
    pub fn estimate(self, bytes: &[u8]) -> Result<DctWorkingSetEstimate, JpegError> {
        estimate_dct_windows(bytes, &[], self)
    }
}

/// Estimate extraction working storage for MCU windows without entropy decoding.
///
/// An empty `windows` slice means the whole image. Progressive and multi-scan
/// streams include their required whole-image 32-bit accumulation plane.
#[expect(
    clippy::too_many_lines,
    reason = "the estimate mirrors baseline, fallback, and progressive allocation phases explicitly"
)]
pub fn estimate_dct_windows(
    bytes: &[u8],
    windows: &[McuRect],
    options: DctExtractOptions,
) -> Result<DctWorkingSetEstimate, JpegError> {
    validate_extract_options(options)?;
    let decoder = Decoder::new(bytes)?;
    validate_extract_color_space(decoder.info().color_space)?;
    validate_extract_frame(&decoder)?;
    let (mcu_cols, mcu_rows, _, _) = decoder_mcu_grid(&decoder);
    for &window in windows {
        validate_mcu_window(window, mcu_cols, mcu_rows)?;
    }
    let output_blocks = if windows.is_empty() {
        coefficient_blocks_for_window(
            &decoder,
            McuRect {
                x: 0,
                y: 0,
                width: mcu_cols,
                height: mcu_rows,
            },
        )?
    } else {
        windows.iter().try_fold(0usize, |total, &window| {
            total
                .checked_add(coefficient_blocks_for_window(&decoder, window)?)
                .ok_or(JpegError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })
        })?
    };
    let output_planes = usize::from(options.retain_quantized_blocks)
        + usize::from(options.retain_dequantized_blocks);
    let coefficient_retained_bytes =
        checked_allocation_bytes::<[i16; 64]>(output_blocks.checked_mul(output_planes).ok_or(
            JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            },
        )?)?;
    let result_count = windows.len().max(1);
    let component_count = decoder.info().sampling.len();
    let result_metadata = checked_add_allocation_bytes(
        checked_allocation_bytes::<JpegDctImage>(result_count)?,
        checked_allocation_bytes::<JpegDctComponent>(
            result_count
                .checked_mul(component_count)
                .ok_or(JpegError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })?,
        )?,
    )?;
    let restart_bytes =
        restart_index_allocation_bytes(decoder.info(), decoder.plan.restart_interval)?;
    let retained_bytes = checked_add_allocation_bytes(
        checked_add_allocation_bytes(coefficient_retained_bytes, result_metadata)?,
        if windows.is_empty() { restart_bytes } else { 0 },
    )?;
    let progressive = decoder.progressive_plan.is_some();
    // The streamed baseline route never holds whole-image coefficients, so
    // their size must not decide its admission.
    let streamed = !windows.is_empty()
        && decoder.info().sof_kind == SofKind::Baseline8
        && !progressive
        && options == DctExtractOptions::quantized_only();
    let peak_bytes = if streamed {
        let temporary_metadata = checked_add_allocation_bytes(
            checked_allocation_bytes::<Vec<Vec<[i16; 64]>>>(windows.len())?,
            checked_add_allocation_bytes(
                checked_allocation_bytes::<Vec<[i16; 64]>>(
                    windows.len().checked_mul(component_count).ok_or(
                        JpegError::MemoryCapExceeded {
                            requested: usize::MAX,
                            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                        },
                    )?,
                )?,
                checked_allocation_bytes::<McuWindow>(windows.len())?,
            )?,
        )?;
        checked_add_allocation_bytes(retained_bytes, temporary_metadata)?
    } else {
        let full_blocks = coefficient_blocks_for_window(
            &decoder,
            McuRect {
                x: 0,
                y: 0,
                width: mcu_cols,
                height: mcu_rows,
            },
        )?;
        let full_output_bytes =
            checked_allocation_bytes::<[i16; 64]>(full_blocks.checked_mul(output_planes).ok_or(
                JpegError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                },
            )?)?;
        let full_metadata = checked_add_allocation_bytes(
            checked_allocation_bytes::<JpegDctImage>(1)?,
            checked_allocation_bytes::<JpegDctComponent>(component_count)?,
        )?;
        let full_retained = checked_add_allocation_bytes(
            checked_add_allocation_bytes(full_output_bytes, full_metadata)?,
            restart_bytes,
        )?;
        let progressive_accumulation = if progressive {
            checked_add_allocation_bytes(
                checked_allocation_bytes::<[i32; 64]>(full_blocks)?,
                checked_allocation_bytes::<Vec<[i32; 64]>>(component_count)?,
            )?
        } else {
            0
        };
        let while_decoding = checked_add_allocation_bytes(full_retained, progressive_accumulation)?;
        if windows.is_empty() {
            while_decoding
        } else {
            // Windows are copied out of the whole-image result.
            checked_add_allocation_bytes(full_retained, retained_bytes)?.max(while_decoding)
        }
    };
    Ok(DctWorkingSetEstimate {
        retained_bytes,
        peak_bytes,
    })
}

fn coefficient_blocks_for_window(
    decoder: &Decoder<'_>,
    window: McuRect,
) -> Result<usize, JpegError> {
    decoder
        .info()
        .sampling
        .components()
        .iter()
        .try_fold(0usize, |total, &(h, v)| {
            let blocks = (window.width as usize)
                .checked_mul(window.height as usize)
                .and_then(|blocks| blocks.checked_mul(usize::from(h)))
                .and_then(|blocks| blocks.checked_mul(usize::from(v)))
                .ok_or(JpegError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })?;
            total
                .checked_add(blocks)
                .ok_or(JpegError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })
        })
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
    validate_extract_options(options)?;
    let decoder = Decoder::new(bytes)?;
    match decoder.info().color_space {
        ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb => {}
        color_space => return Err(JpegError::UnsupportedColorSpace { color_space }),
    }

    let workspace_cap = decoder.decode_workspace_cap()?;
    let planned_restart_index_bytes =
        restart_index_allocation_bytes(decoder.info(), decoder.plan.restart_interval)?;
    let (coding_mode, components) = match decoder.info().sof_kind {
        SofKind::Baseline8 if decoder.progressive_plan.is_none() => {
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
                options.retain_dequantized_blocks,
                lifecycle,
            )?;
            (
                JpegDctCodingMode::BaselineSequential,
                build_sequential_components(&decoder, decoded_blocks, workspace_cap)?,
            )
        }
        // Progressive and split (multi-scan) baseline frames carry
        // whole-image coefficients.
        sof @ (SofKind::Progressive8 | SofKind::Baseline8) => {
            let progressive_plan = decoder
                .progressive_plan
                .as_ref()
                .ok_or(JpegError::NotImplemented { sof })?;
            validate_progressive_extraction_workspace(
                &progressive_plan.components,
                options.retain_quantized_blocks,
                options.retain_dequantized_blocks,
                workspace_cap,
            )?;
            let decoded_blocks = decode_progressive_dct_blocks(progressive_plan, decoder.bytes, 0)?;
            let coding_mode = if sof == SofKind::Progressive8 {
                JpegDctCodingMode::Progressive
            } else {
                JpegDctCodingMode::BaselineSequential
            };
            (
                coding_mode,
                build_progressive_components(
                    &decoder,
                    decoded_blocks,
                    options.retain_quantized_blocks,
                    options.retain_dequantized_blocks,
                    workspace_cap,
                )?,
            )
        }
        other => return Err(JpegError::NotImplemented { sof: other }),
    };
    decoder.validate_completed_warnings(&[])?;
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

/// Extract one or more MCU-aligned coefficient windows.
///
/// Baseline sequential quantized-only extraction retains only the requested
/// blocks and enters the scan at the nearest restart interval. Progressive
/// and multi-scan inputs currently accumulate their full coefficient state,
/// then copy the requested windows; this preserves refinement dependencies.
/// Results are returned in the same order as `windows`.
pub fn extract_dct_windows(
    bytes: &[u8],
    windows: &[McuRect],
    options: DctExtractOptions,
) -> Result<Vec<JpegDctImage>, JpegError> {
    validate_extract_options(options)?;
    if windows.is_empty() {
        let decoder = Decoder::new(bytes)?;
        validate_extract_color_space(decoder.info().color_space)?;
        validate_extract_frame(&decoder)?;
        return Ok(Vec::new());
    }
    let estimate = estimate_dct_windows(bytes, windows, options)?;
    let decoder = Decoder::new(bytes)?;
    validate_extract_color_space(decoder.info().color_space)?;
    validate_extract_frame(&decoder)?;
    let workspace_cap = decoder.decode_workspace_cap()?;
    if estimate.peak_bytes > workspace_cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: estimate.peak_bytes,
            cap: workspace_cap,
        });
    }
    let (mcu_cols, mcu_rows, mcu_width, mcu_height) = decoder_mcu_grid(&decoder);
    for &window in windows {
        validate_mcu_window(window, mcu_cols, mcu_rows)?;
    }
    if decoder.info().sof_kind == SofKind::Baseline8
        && decoder.progressive_plan.is_none()
        && options == DctExtractOptions::quantized_only()
    {
        let out = extract_sequential_quantized_windows(
            &decoder,
            windows,
            (mcu_width, mcu_height),
            workspace_cap,
        )?;
        decoder.validate_completed_warnings(&[])?;
        return Ok(out);
    }

    drop(decoder);
    let full = extract_dct_blocks(bytes, options)?;
    let mut live_bytes = full.retained_bytes()?;
    let mut out = Vec::new();
    try_reserve_for_len_with_live_budget(&mut out, windows.len(), &mut live_bytes, workspace_cap)?;
    for &window in windows {
        out.push(window_from_full(
            &full,
            window,
            (mcu_width, mcu_height),
            &mut live_bytes,
            workspace_cap,
        )?);
    }
    Ok(out)
}

/// Crop one standalone JPEG without changing its quantized coefficients.
///
/// The encoded crop begins at the MCU corner at or above-left of `rect` and
/// ends at the requested far edges, clamped to the source. `region` locates
/// the requested pixels in that MCU-aligned output. Baseline and progressive
/// Huffman inputs with one or three 8-bit components are emitted as baseline
/// sequential JPEGs with their JFIF/Adobe signalling, component identifiers,
/// sampling, and per-component quantization tables preserved.
pub fn lossless_crop(
    bytes: &[u8],
    rect: crate::Rect,
    options: LosslessCropOptions,
) -> Result<LosslessCrop, LosslessCropError> {
    let decoder = Decoder::new(bytes)?;
    validate_extract_color_space(decoder.info().color_space)?;
    let source = decoder.info().dimensions;
    if rect.w == 0 || rect.h == 0 || rect.x >= source.0 || rect.y >= source.1 {
        return Err(JpegError::RectOutOfBounds {
            rect,
            width: source.0,
            height: source.1,
        }
        .into());
    }
    let x_end = rect
        .x
        .checked_add(rect.w)
        .ok_or(JpegError::RectOutOfBounds {
            rect,
            width: source.0,
            height: source.1,
        })?
        .min(source.0);
    let y_end = rect
        .y
        .checked_add(rect.h)
        .ok_or(JpegError::RectOutOfBounds {
            rect,
            width: source.0,
            height: source.1,
        })?
        .min(source.1);
    let (_, _, mcu_width, mcu_height) = decoder_mcu_grid(&decoder);
    let x0 = rect.x - rect.x % mcu_width;
    let y0 = rect.y - rect.y % mcu_height;
    let window = McuRect {
        x: x0 / mcu_width,
        y: y0 / mcu_height,
        width: (x_end - x0).div_ceil(mcu_width),
        height: (y_end - y0).div_ceil(mcu_height),
    };
    drop(decoder);

    let estimate = estimate_dct_windows(bytes, &[window], DctExtractOptions::quantized_only())?;
    if estimate.peak_bytes > options.max_working_bytes {
        return Err(JpegError::MemoryCapExceeded {
            requested: estimate.peak_bytes,
            cap: options.max_working_bytes,
        }
        .into());
    }
    let mut extracted = extract_dct_windows(bytes, &[window], DctExtractOptions::quantized_only())?
        .pop()
        .ok_or(JpegError::InternalInvariant {
            reason: "lossless crop extraction returned no window",
        })?;
    // A complete progressive coefficient grid can be written as a baseline
    // sequential scan without changing any quantized coefficient.
    extracted.coding_mode = JpegDctCodingMode::BaselineSequential;
    extracted.scan_count = 1;
    extracted.width = x_end - x0;
    extracted.height = y_end - y0;
    let max_h = extracted
        .components
        .iter()
        .map(|component| component.h_samp)
        .max()
        .unwrap_or(1);
    let max_v = extracted
        .components
        .iter()
        .map(|component| component.v_samp)
        .max()
        .unwrap_or(1);
    for component in &mut extracted.components {
        component.width = extracted
            .width
            .saturating_mul(u32::from(component.h_samp))
            .div_ceil(u32::from(max_h));
        component.height = extracted
            .height
            .saturating_mul(u32::from(component.v_samp))
            .div_ceil(u32::from(max_v));
    }
    let sof = source_sof_info(bytes)?;
    let app_segments = source_color_app_segments(bytes)?;
    let data = encode_baseline_dct_image_with_options(
        &extracted,
        BaselineDctEncodeOptions {
            app_segments: &app_segments,
            component_ids: &sof.component_ids,
        },
    )?;
    Ok(LosslessCrop {
        data,
        width: extracted.width,
        height: extracted.height,
        region: crate::Rect {
            x: rect.x - x0,
            y: rect.y - y0,
            w: x_end - rect.x,
            h: y_end - rect.y,
        },
    })
}

fn source_sof_info(bytes: &[u8]) -> Result<crate::JpegSofInfo, JpegError> {
    for segment in crate::iter_segments(bytes) {
        let segment = segment?;
        if crate::is_sof_marker(segment.marker) {
            return crate::parse_sof_info(segment.marker, segment.payload);
        }
    }
    Err(JpegError::MissingMarker {
        marker: MarkerKind::Sof,
    })
}

fn source_color_app_segments(bytes: &[u8]) -> Result<Vec<u8>, JpegError> {
    let mut out = Vec::new();
    for segment in crate::iter_segments(bytes) {
        let segment = segment?;
        if segment.marker == 0xda {
            break;
        }
        let is_color = (segment.marker == 0xe0 && segment.payload.starts_with(b"JFIF\0"))
            || (segment.marker == 0xee && segment.payload.starts_with(b"Adobe"));
        if !is_color {
            continue;
        }
        let end = segment
            .payload_offset
            .checked_add(segment.payload.len())
            .ok_or(JpegError::HostAllocationFailed { bytes: usize::MAX })?;
        let marker = bytes
            .get(segment.marker_offset..end)
            .ok_or(JpegError::InternalInvariant {
                reason: "JPEG APP marker range lies outside the source",
            })?;
        let required = out
            .len()
            .checked_add(marker.len())
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
        if required > j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES {
            return Err(JpegError::MemoryCapExceeded {
                requested: required,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            });
        }
        out.try_reserve(marker.len())
            .map_err(|_| JpegError::HostAllocationFailed { bytes: required })?;
        out.extend_from_slice(marker);
    }
    Ok(out)
}

fn validate_extract_color_space(color_space: ColorSpace) -> Result<(), JpegError> {
    match color_space {
        ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb => Ok(()),
        color_space => Err(JpegError::UnsupportedColorSpace { color_space }),
    }
}

fn validate_extract_frame(decoder: &Decoder<'_>) -> Result<(), JpegError> {
    match decoder.info().sof_kind {
        SofKind::Baseline8 | SofKind::Progressive8 => Ok(()),
        sof => Err(JpegError::NotImplemented { sof }),
    }
}

fn validate_extract_options(options: DctExtractOptions) -> Result<(), JpegError> {
    if options.retain_quantized_blocks || options.retain_dequantized_blocks {
        Ok(())
    } else {
        Err(JpegError::InvalidDctExtractOptions)
    }
}

fn decoder_mcu_grid(decoder: &Decoder<'_>) -> (u32, u32, u32, u32) {
    let sampling = decoder.info().sampling;
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let (width, height) = decoder.info().dimensions;
    (
        width.div_ceil(mcu_width),
        height.div_ceil(mcu_height),
        mcu_width,
        mcu_height,
    )
}

fn validate_mcu_window(window: McuRect, cols: u32, rows: u32) -> Result<(), JpegError> {
    if window.width == 0
        || window.height == 0
        || window
            .x
            .checked_add(window.width)
            .is_none_or(|end| end > cols)
        || window
            .y
            .checked_add(window.height)
            .is_none_or(|end| end > rows)
    {
        return Err(JpegError::DctWindowOutOfBounds {
            x: window.x,
            y: window.y,
            width: window.width,
            height: window.height,
            mcu_cols: cols,
            mcu_rows: rows,
        });
    }
    Ok(())
}

fn extract_sequential_quantized_windows(
    decoder: &Decoder<'_>,
    windows: &[McuRect],
    (mcu_width, mcu_height): (u32, u32),
    workspace_cap: usize,
) -> Result<Vec<JpegDctImage>, JpegError> {
    let total_blocks = windows.iter().try_fold(0usize, |total, &window| {
        total
            .checked_add(coefficient_blocks_for_window(decoder, window)?)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: workspace_cap,
            })
    })?;
    let bytes = checked_allocation_bytes::<[i16; 64]>(total_blocks)?;
    if bytes > workspace_cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: bytes,
            cap: workspace_cap,
        });
    }
    let mut all_blocks = Vec::new();
    let mut live_bytes = 0usize;
    try_reserve_for_len_with_live_budget(
        &mut all_blocks,
        windows.len(),
        &mut live_bytes,
        workspace_cap,
    )?;
    for &window in windows {
        let mut blocks = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut blocks,
            decoder.info().sampling.len(),
            &mut live_bytes,
            workspace_cap,
        )?;
        for component_index in 0..decoder.info().sampling.len() {
            let plan_component = decoder
                .plan
                .components
                .iter()
                .find(|component| component.output_index == component_index)
                .ok_or(JpegError::InternalInvariant {
                    reason: "DCT extraction plan does not cover every component",
                })?;
            let count = (window.width as usize)
                * (window.height as usize)
                * usize::from(plan_component.h)
                * usize::from(plan_component.v);
            let mut plane = Vec::new();
            try_reserve_for_len_with_live_budget(
                &mut plane,
                count,
                &mut live_bytes,
                workspace_cap,
            )?;
            plane.resize(count, [0; 64]);
            blocks.push(plane);
        }
        all_blocks.push(blocks);
    }
    let mut mcu_windows = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut mcu_windows,
        windows.len(),
        &mut live_bytes,
        workspace_cap,
    )?;
    mcu_windows.extend(windows.iter().map(|window| McuWindow {
        col: window.x,
        cols: window.width,
        row: window.y,
        rows: window.height,
    }));
    decode_scan_quantized_dct_windows_into(
        &decoder.plan,
        &decoder.bytes[decoder.plan.scan_offset..],
        &mcu_windows,
        &mut all_blocks,
    )?;

    let mut out = Vec::new();
    try_reserve_for_len_with_live_budget(&mut out, windows.len(), &mut live_bytes, workspace_cap)?;
    for (window, blocks) in windows.iter().copied().zip(all_blocks) {
        out.push(sequential_quantized_window_image(
            decoder,
            window,
            (mcu_width, mcu_height),
            blocks,
            &mut live_bytes,
            workspace_cap,
        )?);
    }
    Ok(out)
}

fn sequential_quantized_window_image(
    decoder: &Decoder<'_>,
    window: McuRect,
    (mcu_width, mcu_height): (u32, u32),
    blocks: Vec<Vec<[i16; 64]>>,
    live_bytes: &mut usize,
    workspace_cap: usize,
) -> Result<JpegDctImage, JpegError> {
    let source_dimensions = decoder.info().dimensions;
    let width = source_dimensions
        .0
        .saturating_sub(window.x * mcu_width)
        .min(window.width * mcu_width);
    let height = source_dimensions
        .1
        .saturating_sub(window.y * mcu_height)
        .min(window.height * mcu_height);
    let max_h = u32::from(decoder.info().sampling.max_h);
    let max_v = u32::from(decoder.info().sampling.max_v);
    let mut components = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut components,
        decoder.info().sampling.len(),
        live_bytes,
        workspace_cap,
    )?;
    for (component_index, quantized_blocks) in blocks.into_iter().enumerate() {
        let plan_component = decoder
            .plan
            .components
            .iter()
            .find(|component| component.output_index == component_index)
            .ok_or(JpegError::InternalInvariant {
                reason: "DCT extraction plan does not cover every component",
            })?;
        let block_cols = window.width * u32::from(plan_component.h);
        let block_rows = window.height * u32::from(plan_component.v);
        components.push(JpegDctComponent {
            component_index,
            width: width
                .saturating_mul(u32::from(plan_component.h))
                .div_ceil(max_h),
            height: height
                .saturating_mul(u32::from(plan_component.v))
                .div_ceil(max_v),
            h_samp: plan_component.h,
            v_samp: plan_component.v,
            block_cols,
            block_rows,
            quant_table: plan_component.quant,
            quantized_blocks,
            dequantized_blocks: Vec::new(),
        });
    }
    Ok(JpegDctImage {
        width,
        height,
        color_space: decoder.info().color_space,
        coding_mode: JpegDctCodingMode::BaselineSequential,
        scan_count: 1,
        components,
        restart_index: None,
    })
}

fn window_from_full(
    full: &JpegDctImage,
    window: McuRect,
    (mcu_width, mcu_height): (u32, u32),
    live_bytes: &mut usize,
    workspace_cap: usize,
) -> Result<JpegDctImage, JpegError> {
    let width = full
        .width
        .saturating_sub(window.x * mcu_width)
        .min(window.width * mcu_width);
    let height = full
        .height
        .saturating_sub(window.y * mcu_height)
        .min(window.height * mcu_height);
    let max_h = mcu_width / 8;
    let max_v = mcu_height / 8;
    let mut components = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut components,
        full.components.len(),
        live_bytes,
        workspace_cap,
    )?;
    for component in &full.components {
        let block_x = window.x * u32::from(component.h_samp);
        let block_y = window.y * u32::from(component.v_samp);
        let block_cols = window.width * u32::from(component.h_samp);
        let block_rows = window.height * u32::from(component.v_samp);
        let mut copy_plane = |source: &[[i16; 64]]| -> Result<Vec<[i16; 64]>, JpegError> {
            if source.is_empty() {
                return Ok(Vec::new());
            }
            let mut copied = Vec::new();
            let count = block_cols as usize * block_rows as usize;
            try_reserve_for_len_with_live_budget(&mut copied, count, live_bytes, workspace_cap)?;
            for row in block_y..block_y + block_rows {
                let start = (row * component.block_cols + block_x) as usize;
                copied.extend_from_slice(&source[start..start + block_cols as usize]);
            }
            Ok(copied)
        };
        components.push(JpegDctComponent {
            component_index: component.component_index,
            width: width
                .saturating_mul(u32::from(component.h_samp))
                .div_ceil(max_h),
            height: height
                .saturating_mul(u32::from(component.v_samp))
                .div_ceil(max_v),
            h_samp: component.h_samp,
            v_samp: component.v_samp,
            block_cols,
            block_rows,
            quant_table: component.quant_table,
            quantized_blocks: copy_plane(&component.quantized_blocks)?,
            dequantized_blocks: copy_plane(&component.dequantized_blocks)?,
        });
    }
    Ok(JpegDctImage {
        width,
        height,
        color_space: full.color_space,
        coding_mode: full.coding_mode,
        scan_count: full.scan_count,
        components,
        restart_index: None,
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

/// Re-emit a baseline JPEG while preserving source component signalling.
///
/// Each component keeps its own quantization table; equal tables share one
/// DQT slot. `options.app_segments` must contain complete `APPn` segments only.
pub fn encode_baseline_dct_image_with_options(
    image: &JpegDctImage,
    options: BaselineDctEncodeOptions<'_>,
) -> Result<Vec<u8>, JpegEncodeError> {
    let validated = validate_baseline_dct_image_with_independent_quant_tables(image)
        .map_err(|reason| JpegEncodeError::InvalidDctImage { reason })?;
    let entropy_capacity =
        jpeg_baseline_entropy_capacity_bytes(image.width, image.height, validated.sampling, None)?;
    validate_dct_reemission_live_bytes(entropy_capacity)?;
    let huffman_tables = baseline_encode_tables(JpegEncodeOptions {
        quality: 90,
        subsampling: if validated.sampling.components == 1 {
            JpegSubsampling::Gray
        } else {
            JpegSubsampling::Ybr420
        },
        restart_interval: None,
        backend: JpegBackend::Cpu,
    })?;
    let entropy = encode_dct_entropy(
        image,
        validated.sampling,
        [&huffman_tables.huff_dc_luma, &huffman_tables.huff_dc_chroma],
        [&huffman_tables.huff_ac_luma, &huffman_tables.huff_ac_chroma],
        entropy_capacity,
    )?;
    let component_count = usize::from(validated.sampling.components);
    let metadata_bytes = options
        .app_segments
        .len()
        .checked_add(component_count.saturating_mul(65))
        .ok_or(JpegEncodeError::HostAllocationFailed { bytes: usize::MAX })?;
    let frame_capacity = checked_jpeg_baseline_frame_capacity(
        entropy
            .len()
            .checked_add(metadata_bytes)
            .ok_or(JpegEncodeError::HostAllocationFailed { bytes: usize::MAX })?,
    )?;
    checked_encode_host_live_bytes([entropy.capacity(), frame_capacity])?;
    let encoded = assemble_jpeg_baseline_frame_with_metadata(
        &entropy,
        image.width,
        image.height,
        JpegBaselineFrameMetadata {
            sampling: validated.sampling,
            component_ids: options.component_ids,
            component_quant: &validated.component_quant[..component_count],
            app_segments: options.app_segments,
        },
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
    retain_dequantized_blocks: bool,
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
        if retain_dequantized_blocks {
            try_reserve_for_len_with_live_budget(
                &mut dequantized_blocks,
                quantized_i32.len(),
                &mut live_bytes,
                workspace_cap,
            )?;
        }
        for block in &quantized_i32 {
            if retain_quantized_blocks {
                quantized_blocks.push(quantized_i16_block(block));
            }
            if retain_dequantized_blocks {
                dequantized_blocks.push(dequantize_progressive_block(block, &component.quant));
            }
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
    retain_dequantized_blocks: bool,
    workspace_cap: usize,
) -> Result<(), JpegError> {
    let decoded_metadata = checked_allocation_bytes::<Vec<[i32; 64]>>(components.len())?;
    let output_metadata = checked_allocation_bytes::<JpegDctComponent>(components.len())?;
    let mut total = checked_add_allocation_bytes(decoded_metadata, output_metadata)?;
    let output_plane_count =
        usize::from(retain_quantized_blocks) + usize::from(retain_dequantized_blocks);
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

        assert!(validate_progressive_extraction_workspace(&components, false, true, cap).is_ok());
        assert!(matches!(
            validate_progressive_extraction_workspace(&components, true, true, cap),
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
