// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::encoder::{
    EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSubsampling,
};
use crate::PixelFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Baseline JPEG component sampling parameters.
pub struct JpegBaselineSampling {
    /// Number of encoded components.
    pub components: u8,
    /// Horizontal sampling factor per component.
    pub h: [u8; 3],
    /// Vertical sampling factor per component.
    pub v: [u8; 3],
    /// Maximum horizontal sampling factor across components.
    pub max_h: u8,
    /// Maximum vertical sampling factor across components.
    pub max_v: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Canonical Huffman lookup table for encoding.
pub struct JpegBaselineHuffmanTable {
    /// Huffman code value by symbol.
    pub codes: [u16; 256],
    /// Huffman code length by symbol.
    pub lens: [u8; 256],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Tables needed to assemble and entropy-code a baseline JPEG frame.
pub struct JpegBaselineEncodeTables {
    /// Component sampling metadata.
    pub sampling: JpegBaselineSampling,
    /// Luma quantization table in natural order.
    pub q_luma: [u8; 64],
    /// Chroma quantization table in natural order.
    pub q_chroma: [u8; 64],
    /// Luma DC Huffman table.
    pub huff_dc_luma: JpegBaselineHuffmanTable,
    /// Luma AC Huffman table.
    pub huff_ac_luma: JpegBaselineHuffmanTable,
    /// Chroma DC Huffman table.
    pub huff_dc_chroma: JpegBaselineHuffmanTable,
    /// Chroma AC Huffman table.
    pub huff_ac_chroma: JpegBaselineHuffmanTable,
}

/// Backend-neutral metadata for a resident GPU baseline JPEG encode tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegBaselineGpuEncodeTile {
    /// Byte offset of the first source pixel in the resident buffer.
    pub byte_offset: usize,
    /// Width of the valid input region in pixels.
    pub width: u32,
    /// Height of the valid input region in pixels.
    pub height: u32,
    /// Number of bytes between consecutive input rows.
    pub pitch_bytes: usize,
    /// Encoded frame width in pixels.
    pub output_width: u32,
    /// Encoded frame height in pixels.
    pub output_height: u32,
    /// Pixel format of the source buffer.
    pub format: PixelFormat,
    /// Total resident buffer length in bytes.
    pub buffer_len: usize,
}

/// Backend-neutral baseline JPEG encode ABI parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JpegBaselineGpuEncodeParams {
    /// First input byte for this tile inside a same-buffer batch.
    pub input_offset_bytes: u32,
    /// Width of the valid input rectangle in pixels.
    pub input_width: u32,
    /// Height of the valid input rectangle in pixels.
    pub input_height: u32,
    /// Encoded frame width in pixels.
    pub output_width: u32,
    /// Encoded frame height in pixels.
    pub output_height: u32,
    /// Number of input bytes between consecutive rows.
    pub pitch_bytes: u32,
    /// Number of MCUs per encoded frame row.
    pub mcus_per_row: u32,
    /// Number of MCU rows in the encoded frame.
    pub mcu_rows: u32,
    /// Optional restart interval in MCUs, or zero when disabled.
    pub restart_interval_mcus: u32,
    /// Stable resident-encode format ABI value.
    pub format: u32,
    /// Number of encoded components.
    pub components: u32,
    /// Maximum horizontal sampling factor.
    pub max_h: u32,
    /// Maximum vertical sampling factor.
    pub max_v: u32,
    /// Component 0 horizontal sampling factor.
    pub h0: u32,
    /// Component 0 vertical sampling factor.
    pub v0: u32,
    /// Component 1 horizontal sampling factor.
    pub h1: u32,
    /// Component 1 vertical sampling factor.
    pub v1: u32,
    /// Component 2 horizontal sampling factor.
    pub h2: u32,
    /// Component 2 vertical sampling factor.
    pub v2: u32,
    /// First entropy-output byte for this tile inside a batch output allocation.
    pub entropy_offset_bytes: u32,
    /// Entropy-output capacity for this tile.
    pub entropy_capacity: u32,
}

/// Backend-neutral resident GPU baseline JPEG encode plan for one tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegBaselineGpuEncodeTilePlan {
    /// GPU ABI parameters for this tile.
    pub params: JpegBaselineGpuEncodeParams,
    /// Entropy-output capacity for this tile.
    pub entropy_capacity: usize,
}

/// Backend-neutral resident GPU baseline JPEG encode plan for one batch span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegBaselineGpuEncodeBatchPlan {
    /// GPU ABI parameters in input tile order.
    pub params: Vec<JpegBaselineGpuEncodeParams>,
    /// Combined entropy-output capacity for the batch allocation.
    pub total_entropy_capacity: usize,
}

/// Backend hooks used by the shared resident GPU baseline JPEG encode driver.
///
/// First-party CUDA and Metal adapters provide only resident-buffer identity,
/// tile metadata conversion, backend error mapping, and kernel submission. The
/// shared driver owns table construction, planning, batch span grouping, and
/// JPEG frame assembly.
pub trait JpegBaselineGpuEncodeHostAdapter<T: Copy> {
    /// Error returned by the backend adapter.
    type Error: From<JpegEncodeError>;
    /// Stable identity for a resident source allocation.
    type SourceKey: PartialEq;

    /// Backend represented by this adapter.
    fn backend(&self) -> JpegBackend;

    /// Return the resident source allocation key for grouping batch spans.
    fn source_key(&self, tile: &T) -> Self::SourceKey;

    /// Convert a backend tile into backend-neutral planning metadata.
    fn gpu_tile(&self, tile: T) -> Result<JpegBaselineGpuEncodeTile, Self::Error>;

    /// Map a backend-neutral planning error into the backend's public error.
    fn map_plan_error(&self, error: JpegBaselineGpuEncodeError) -> Self::Error;

    /// Submit one resident tile to the backend entropy encoder.
    fn encode_tile_entropy(
        &mut self,
        tile: T,
        tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeTilePlan,
    ) -> Result<Vec<u8>, Self::Error>;

    /// Submit a contiguous same-source-buffer resident tile span.
    fn encode_batch_entropy(
        &mut self,
        tiles: &[T],
        tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeBatchPlan,
    ) -> Result<Vec<Vec<u8>>, Self::Error>;
}

/// Error returned by backend-neutral resident GPU baseline JPEG encode planning.
#[derive(Debug)]
pub enum JpegBaselineGpuEncodeError {
    /// A baseline JPEG encode option was invalid.
    Encode(JpegEncodeError),
    /// The requested public backend does not match this adapter.
    UnsupportedBackend {
        /// Requested backend.
        requested: JpegBackend,
        /// Backend accepted by the caller.
        expected: JpegBackend,
    },
    /// The valid input rectangle exceeds the encoded output dimensions.
    InputExceedsOutputDimensions,
    /// The source pixel format is unsupported by resident baseline encode.
    UnsupportedPixelFormat {
        /// Source pixel format.
        format: PixelFormat,
    },
    /// The source pixel format is incompatible with the requested subsampling.
    IncompatibleSubsampling {
        /// Requested subsampling.
        subsampling: JpegSubsampling,
        /// Source sample description.
        samples: &'static str,
    },
    /// Row-byte arithmetic overflowed.
    RowByteCountOverflow,
    /// Source pitch is shorter than one row.
    PitchTooShort {
        /// Required row bytes.
        row_bytes: usize,
        /// Provided pitch bytes.
        pitch_bytes: usize,
    },
    /// Input byte-range arithmetic overflowed.
    InputRangeOverflow,
    /// Input byte range exceeds the resident buffer length.
    InputRangeExceedsBuffer {
        /// Required exclusive byte end.
        required_end: usize,
        /// Resident buffer length in bytes.
        buffer_len: usize,
    },
    /// Pitch does not fit the GPU ABI.
    PitchTooLarge,
    /// Input offset does not fit the GPU ABI.
    InputOffsetTooLarge,
    /// Entropy offset does not fit the GPU ABI.
    EntropyOffsetTooLarge,
    /// Entropy capacity does not fit the GPU ABI.
    EntropyCapacityTooLarge,
    /// Combined batch entropy capacity overflowed host arithmetic.
    BatchEntropyCapacityOverflow,
}

impl From<JpegEncodeError> for JpegBaselineGpuEncodeError {
    fn from(error: JpegEncodeError) -> Self {
        Self::Encode(error)
    }
}

/// Build quantization, sampling, and Huffman tables for baseline encoding.
pub fn baseline_encode_tables(
    options: JpegEncodeOptions,
) -> Result<JpegBaselineEncodeTables, JpegEncodeError> {
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;
    Ok(JpegBaselineEncodeTables {
        sampling: jpeg_baseline_sampling_for(options.subsampling),
        q_luma: scaled_quant_table(&STD_LUMA_Q, options.quality),
        q_chroma: scaled_quant_table(&STD_CHROMA_Q, options.quality),
        huff_dc_luma: encode_huffman_table(&STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?,
        huff_ac_luma: encode_huffman_table(&STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?,
        huff_dc_chroma: encode_huffman_table(&STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?,
        huff_ac_chroma: encode_huffman_table(&STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?,
    })
}

/// Validate that dimensions can be represented in baseline JPEG markers.
pub(crate) fn validate_jpeg_baseline_dimensions(
    width: u32,
    height: u32,
) -> Result<(), JpegEncodeError> {
    if width == 0 || height == 0 {
        return Err(JpegEncodeError::EmptyDimensions);
    }
    if width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
        return Err(JpegEncodeError::DimensionsTooLarge { width, height });
    }
    Ok(())
}

/// Validate a user-provided restart interval.
pub(crate) fn validate_jpeg_baseline_restart_interval(
    restart_interval: Option<u16>,
) -> Result<(), JpegEncodeError> {
    if restart_interval == Some(0) {
        return Err(JpegEncodeError::InvalidRestartInterval);
    }
    Ok(())
}

/// Return JPEG component sampling factors for a public subsampling mode.
fn jpeg_baseline_sampling_for(subsampling: JpegSubsampling) -> JpegBaselineSampling {
    match subsampling {
        JpegSubsampling::Gray => JpegBaselineSampling {
            components: 1,
            h: [1, 0, 0],
            v: [1, 0, 0],
            max_h: 1,
            max_v: 1,
        },
        JpegSubsampling::Ybr444 => JpegBaselineSampling {
            components: 3,
            h: [1, 1, 1],
            v: [1, 1, 1],
            max_h: 1,
            max_v: 1,
        },
        JpegSubsampling::Ybr422 => JpegBaselineSampling {
            components: 3,
            h: [2, 1, 1],
            v: [1, 1, 1],
            max_h: 2,
            max_v: 1,
        },
        JpegSubsampling::Ybr420 => JpegBaselineSampling {
            components: 3,
            h: [2, 1, 1],
            v: [2, 1, 1],
            max_h: 2,
            max_v: 2,
        },
    }
}

/// Conservative upper bound for entropy bytes produced by the CPU encoder.
fn jpeg_baseline_entropy_capacity_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
) -> Result<usize, JpegEncodeError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = u64::from(width.div_ceil(mcu_width));
    let mcu_rows = u64::from(height.div_ceil(mcu_height));
    let total_mcus = mcus_per_row
        .checked_mul(mcu_rows)
        .ok_or_else(|| JpegEncodeError::Internal("JPEG MCU count overflow".into()))?;
    let blocks_per_mcu = u64::from(
        sampling.h[0] * sampling.v[0]
            + sampling.h[1] * sampling.v[1]
            + sampling.h[2] * sampling.v[2],
    );
    let restart_markers = restart_interval.map_or(0, |interval| {
        total_mcus.saturating_sub(1) / u64::from(interval)
    });
    let capacity = total_mcus
        .checked_mul(blocks_per_mcu)
        .and_then(|blocks| blocks.checked_mul(512))
        .and_then(|bytes| bytes.checked_add(restart_markers.saturating_mul(2)))
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(|| JpegEncodeError::Internal("JPEG entropy capacity overflow".into()))?;
    usize::try_from(capacity)
        .map_err(|_| JpegEncodeError::Internal("JPEG entropy capacity exceeds usize".into()))
}

/// Validate resident GPU baseline JPEG encode tile metadata.
fn validate_jpeg_baseline_gpu_encode_tile(
    tile: JpegBaselineGpuEncodeTile,
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
) -> Result<(), JpegBaselineGpuEncodeError> {
    match options.backend {
        JpegBackend::Auto => {}
        requested if requested == expected_backend => {}
        requested => {
            return Err(JpegBaselineGpuEncodeError::UnsupportedBackend {
                requested,
                expected: expected_backend,
            });
        }
    }

    validate_jpeg_baseline_restart_interval(options.restart_interval)?;
    validate_jpeg_baseline_dimensions(tile.output_width, tile.output_height)?;
    if tile.width == 0 || tile.height == 0 {
        return Err(JpegEncodeError::EmptyDimensions.into());
    }
    if tile.width > tile.output_width || tile.height > tile.output_height {
        return Err(JpegBaselineGpuEncodeError::InputExceedsOutputDimensions);
    }

    let bytes_per_pixel = jpeg_baseline_gpu_encode_bytes_per_pixel(tile.format, options)?;
    let width = usize::try_from(tile.width)
        .map_err(|_| JpegBaselineGpuEncodeError::RowByteCountOverflow)?;
    let row_bytes = width
        .checked_mul(bytes_per_pixel)
        .ok_or(JpegBaselineGpuEncodeError::RowByteCountOverflow)?;
    if tile.pitch_bytes < row_bytes {
        return Err(JpegBaselineGpuEncodeError::PitchTooShort {
            row_bytes,
            pitch_bytes: tile.pitch_bytes,
        });
    }
    let height =
        usize::try_from(tile.height).map_err(|_| JpegBaselineGpuEncodeError::InputRangeOverflow)?;
    let last_row = height
        .checked_sub(1)
        .and_then(|row| row.checked_mul(tile.pitch_bytes))
        .ok_or(JpegBaselineGpuEncodeError::InputRangeOverflow)?;
    let required_end = tile
        .byte_offset
        .checked_add(last_row)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or(JpegBaselineGpuEncodeError::InputRangeOverflow)?;
    if required_end > tile.buffer_len {
        return Err(JpegBaselineGpuEncodeError::InputRangeExceedsBuffer {
            required_end,
            buffer_len: tile.buffer_len,
        });
    }

    Ok(())
}

/// Return a GPU ABI-safe entropy capacity for resident baseline encode.
fn jpeg_baseline_gpu_entropy_capacity_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
) -> Result<usize, JpegBaselineGpuEncodeError> {
    let capacity = jpeg_baseline_entropy_capacity_bytes(width, height, sampling, restart_interval)?;
    if capacity > u32::MAX as usize {
        return Err(JpegBaselineGpuEncodeError::EntropyCapacityTooLarge);
    }
    Ok(capacity)
}

/// Build backend-neutral GPU baseline JPEG encode parameters.
fn jpeg_baseline_gpu_encode_params(
    tile: JpegBaselineGpuEncodeTile,
    options: JpegEncodeOptions,
    sampling: JpegBaselineSampling,
    entropy_capacity: usize,
    input_offset_bytes: usize,
    entropy_offset_bytes: usize,
) -> Result<JpegBaselineGpuEncodeParams, JpegBaselineGpuEncodeError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = tile.output_width.div_ceil(mcu_width);
    let mcu_rows = tile.output_height.div_ceil(mcu_height);
    let pitch_bytes =
        u32::try_from(tile.pitch_bytes).map_err(|_| JpegBaselineGpuEncodeError::PitchTooLarge)?;
    let input_offset_bytes = u32::try_from(input_offset_bytes)
        .map_err(|_| JpegBaselineGpuEncodeError::InputOffsetTooLarge)?;
    let entropy_offset_bytes = u32::try_from(entropy_offset_bytes)
        .map_err(|_| JpegBaselineGpuEncodeError::EntropyOffsetTooLarge)?;
    let format = jpeg_baseline_gpu_encode_format_abi(tile.format)?;

    Ok(JpegBaselineGpuEncodeParams {
        input_offset_bytes,
        input_width: tile.width,
        input_height: tile.height,
        output_width: tile.output_width,
        output_height: tile.output_height,
        pitch_bytes,
        mcus_per_row,
        mcu_rows,
        restart_interval_mcus: u32::from(options.restart_interval.unwrap_or(0)),
        format,
        components: u32::from(sampling.components),
        max_h: u32::from(sampling.max_h),
        max_v: u32::from(sampling.max_v),
        h0: u32::from(sampling.h[0]),
        v0: u32::from(sampling.v[0]),
        h1: u32::from(sampling.h[1]),
        v1: u32::from(sampling.v[1]),
        h2: u32::from(sampling.h[2]),
        v2: u32::from(sampling.v[2]),
        entropy_offset_bytes,
        entropy_capacity: u32::try_from(entropy_capacity)
            .map_err(|_| JpegBaselineGpuEncodeError::EntropyCapacityTooLarge)?,
    })
}

/// Build a validated backend-neutral GPU baseline JPEG encode plan for one tile.
fn jpeg_baseline_gpu_encode_tile_plan(
    tile: JpegBaselineGpuEncodeTile,
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
    sampling: JpegBaselineSampling,
    input_offset_bytes: usize,
    entropy_offset_bytes: usize,
) -> Result<JpegBaselineGpuEncodeTilePlan, JpegBaselineGpuEncodeError> {
    validate_jpeg_baseline_gpu_encode_tile(tile, options, expected_backend)?;
    let entropy_capacity = jpeg_baseline_gpu_entropy_capacity_bytes(
        tile.output_width,
        tile.output_height,
        sampling,
        options.restart_interval,
    )?;
    let params = jpeg_baseline_gpu_encode_params(
        tile,
        options,
        sampling,
        entropy_capacity,
        input_offset_bytes,
        entropy_offset_bytes,
    )?;
    Ok(JpegBaselineGpuEncodeTilePlan {
        params,
        entropy_capacity,
    })
}

/// Build validated backend-neutral GPU baseline JPEG encode parameters for a batch span.
///
/// The caller is responsible for passing only tiles that share the same backend
/// input allocation. This helper validates each tile, computes per-tile entropy
/// offsets, and returns the combined entropy capacity for the backend batch job.
fn jpeg_baseline_gpu_encode_batch_plan(
    tiles: &[JpegBaselineGpuEncodeTile],
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
    sampling: JpegBaselineSampling,
) -> Result<JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError> {
    let mut params = Vec::with_capacity(tiles.len());
    let mut total_entropy_capacity = 0usize;
    for tile in tiles {
        let tile_plan = jpeg_baseline_gpu_encode_tile_plan(
            *tile,
            options,
            expected_backend,
            sampling,
            tile.byte_offset,
            total_entropy_capacity,
        )?;
        total_entropy_capacity = total_entropy_capacity
            .checked_add(tile_plan.entropy_capacity)
            .ok_or(JpegBaselineGpuEncodeError::BatchEntropyCapacityOverflow)?;
        params.push(tile_plan.params);
    }

    Ok(JpegBaselineGpuEncodeBatchPlan {
        params,
        total_entropy_capacity,
    })
}

/// Return the end index of a contiguous same-source-buffer batch span.
fn same_source_buffer_batch_end<T, K>(
    tiles: &[T],
    start: usize,
    mut source_key: impl FnMut(&T) -> K,
) -> usize
where
    K: PartialEq,
{
    let key = source_key(&tiles[start]);
    let mut end = start + 1;
    while end < tiles.len() && source_key(&tiles[end]) == key {
        end += 1;
    }
    end
}

/// Encode one resident GPU tile through a backend adapter.
pub fn encode_jpeg_baseline_gpu_tile<T, A>(
    tile: T,
    options: JpegEncodeOptions,
    adapter: &mut A,
) -> Result<EncodedJpeg, A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    let tables = baseline_encode_tables(options)?;
    encode_jpeg_baseline_gpu_tile_with_tables(tile, options, &tables, adapter)
}

/// Encode resident GPU tiles through a backend adapter.
///
/// The driver groups only contiguous tiles that share the same resident source
/// allocation, preserving input order in the returned frames.
pub fn encode_jpeg_baseline_gpu_batch<T, A>(
    tiles: &[T],
    options: JpegEncodeOptions,
    adapter: &mut A,
) -> Result<Vec<EncodedJpeg>, A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    if tiles.is_empty() {
        return Ok(Vec::new());
    }

    let tables = baseline_encode_tables(options)?;
    let mut encoded = Vec::with_capacity(tiles.len());
    let mut start = 0usize;
    while start < tiles.len() {
        let end = same_source_buffer_batch_end(tiles, start, |tile| adapter.source_key(tile));
        if end - start == 1 {
            encoded.push(encode_jpeg_baseline_gpu_tile_with_tables(
                tiles[start],
                options,
                &tables,
                adapter,
            )?);
            start = end;
            continue;
        }

        let gpu_tiles = tiles[start..end]
            .iter()
            .copied()
            .map(|tile| adapter.gpu_tile(tile))
            .collect::<Result<Vec<_>, _>>()?;
        let plan = jpeg_baseline_gpu_encode_batch_plan(
            &gpu_tiles,
            options,
            adapter.backend(),
            tables.sampling,
        )
        .map_err(|error| adapter.map_plan_error(error))?;
        let entropy_chunks = adapter.encode_batch_entropy(&tiles[start..end], &tables, plan)?;
        if entropy_chunks.len() != gpu_tiles.len() {
            return Err(JpegEncodeError::Internal(
                "GPU JPEG baseline batch returned the wrong number of entropy chunks".into(),
            )
            .into());
        }
        for (gpu_tile, entropy) in gpu_tiles.iter().zip(entropy_chunks.iter()) {
            encoded.push(assemble_jpeg_baseline_frame(
                entropy,
                gpu_tile.output_width,
                gpu_tile.output_height,
                &tables,
                options,
                adapter.backend(),
            )?);
        }
        start = end;
    }
    Ok(encoded)
}

fn encode_jpeg_baseline_gpu_tile_with_tables<T, A>(
    tile: T,
    options: JpegEncodeOptions,
    tables: &JpegBaselineEncodeTables,
    adapter: &mut A,
) -> Result<EncodedJpeg, A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    let gpu_tile = adapter.gpu_tile(tile)?;
    let plan = jpeg_baseline_gpu_encode_tile_plan(
        gpu_tile,
        options,
        adapter.backend(),
        tables.sampling,
        0,
        0,
    )
    .map_err(|error| adapter.map_plan_error(error))?;
    let entropy = adapter.encode_tile_entropy(tile, tables, plan)?;
    assemble_jpeg_baseline_frame(
        &entropy,
        gpu_tile.output_width,
        gpu_tile.output_height,
        tables,
        options,
        adapter.backend(),
    )
    .map_err(Into::into)
}

fn jpeg_baseline_gpu_encode_bytes_per_pixel(
    format: PixelFormat,
    options: JpegEncodeOptions,
) -> Result<usize, JpegBaselineGpuEncodeError> {
    match (format, options.subsampling) {
        (PixelFormat::Gray8, JpegSubsampling::Gray) => Ok(1),
        (
            PixelFormat::Rgb8,
            JpegSubsampling::Ybr444 | JpegSubsampling::Ybr422 | JpegSubsampling::Ybr420,
        ) => Ok(3),
        (PixelFormat::Gray8 | PixelFormat::Rgb8, _) => {
            Err(JpegBaselineGpuEncodeError::IncompatibleSubsampling {
                subsampling: options.subsampling,
                samples: if format == PixelFormat::Gray8 {
                    "Gray8"
                } else {
                    "Rgb8"
                },
            })
        }
        _ => Err(JpegBaselineGpuEncodeError::UnsupportedPixelFormat { format }),
    }
}

fn jpeg_baseline_gpu_encode_format_abi(
    format: PixelFormat,
) -> Result<u32, JpegBaselineGpuEncodeError> {
    match format {
        PixelFormat::Gray8 => Ok(0),
        PixelFormat::Rgb8 => Ok(1),
        _ => Err(JpegBaselineGpuEncodeError::UnsupportedPixelFormat { format }),
    }
}

/// Assemble a complete baseline JPEG codestream from entropy bytes and tables.
pub fn assemble_jpeg_baseline_frame(
    entropy: &[u8],
    width: u32,
    height: u32,
    tables: &JpegBaselineEncodeTables,
    options: JpegEncodeOptions,
    backend: JpegBackend,
) -> Result<EncodedJpeg, JpegEncodeError> {
    validate_jpeg_baseline_dimensions(width, height)?;
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;

    let mut out = Vec::with_capacity(768usize.saturating_add(entropy.len()));
    write_marker(&mut out, 0xD8);
    write_dqt(&mut out, 0, &tables.q_luma)?;
    if tables.sampling.components == 3 {
        write_dqt(&mut out, 1, &tables.q_chroma)?;
    }
    if let Some(restart_interval) = options.restart_interval {
        write_dri(&mut out, restart_interval)?;
    }
    write_sof0(&mut out, width, height, tables.sampling)?;
    write_dht(&mut out, 0, 0, &STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?;
    write_dht(&mut out, 1, 0, &STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?;
    if tables.sampling.components == 3 {
        write_dht(&mut out, 0, 1, &STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?;
        write_dht(&mut out, 1, 1, &STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?;
    }
    write_sos(&mut out, tables.sampling.components)?;
    out.extend_from_slice(entropy);
    write_marker(&mut out, 0xD9);

    Ok(EncodedJpeg { data: out, backend })
}

pub(crate) fn assemble_jpeg_baseline_frame_with_quant_tables(
    entropy: &[u8],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: Option<&[u8; 64]>,
    backend: JpegBackend,
) -> Result<EncodedJpeg, JpegEncodeError> {
    validate_jpeg_baseline_dimensions(width, height)?;

    let mut out = Vec::with_capacity(768usize.saturating_add(entropy.len()));
    write_marker(&mut out, 0xD8);
    write_dqt(&mut out, 0, q_luma)?;
    if sampling.components == 3 {
        let q_chroma = q_chroma.ok_or_else(|| {
            JpegEncodeError::Internal("three-component DCT JPEG requires chroma quant table".into())
        })?;
        write_dqt(&mut out, 1, q_chroma)?;
    }
    write_sof0(&mut out, width, height, sampling)?;
    write_dht(&mut out, 0, 0, &STD_LUMA_DC_BITS, &STD_LUMA_DC_VALUES)?;
    write_dht(&mut out, 1, 0, &STD_LUMA_AC_BITS, &STD_LUMA_AC_VALUES)?;
    if sampling.components == 3 {
        write_dht(&mut out, 0, 1, &STD_CHROMA_DC_BITS, &STD_CHROMA_DC_VALUES)?;
        write_dht(&mut out, 1, 1, &STD_CHROMA_AC_BITS, &STD_CHROMA_AC_VALUES)?;
    }
    write_sos(&mut out, sampling.components)?;
    out.extend_from_slice(entropy);
    write_marker(&mut out, 0xD9);

    Ok(EncodedJpeg { data: out, backend })
}

/// JPEG zigzag coefficient order used by baseline entropy coding.
pub const JPEG_BASELINE_ZIGZAG: [u8; 64] = j2k_codec_math::jpeg::ZIGZAG;

fn encode_huffman_table(
    bits: &[u8; 16],
    values: &[u8],
) -> Result<JpegBaselineHuffmanTable, JpegEncodeError> {
    let mut table = JpegBaselineHuffmanTable {
        codes: [0; 256],
        lens: [0; 256],
    };
    let mut code = 0u16;
    let mut idx = 0usize;
    for (len_minus_1, count) in bits.iter().copied().enumerate() {
        let len = u8::try_from(len_minus_1 + 1)
            .map_err(|_| JpegEncodeError::Internal("Huffman code length exceeds u8".into()))?;
        for _ in 0..count {
            let symbol = *values.get(idx).ok_or_else(|| {
                JpegEncodeError::Internal("Huffman table count exceeds values".into())
            })?;
            table.codes[symbol as usize] = code;
            table.lens[symbol as usize] = len;
            code = code
                .checked_add(1)
                .ok_or_else(|| JpegEncodeError::Internal("Huffman code overflow".into()))?;
            idx += 1;
        }
        code <<= 1;
    }
    if idx != values.len() {
        return Err(JpegEncodeError::Internal(
            "Huffman values exceed table counts".into(),
        ));
    }
    Ok(table)
}

fn scaled_quant_table(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let quality = quality.clamp(1, 100);
    let scale = if quality < 50 {
        5000 / u32::from(quality)
    } else {
        200 - u32::from(quality) * 2
    };
    let mut out = [0u8; 64];
    for (idx, value) in base.iter().copied().enumerate() {
        let scaled = (u32::from(value) * scale + 50) / 100;
        out[idx] = scaled.clamp(1, 255) as u8;
    }
    out
}

fn write_marker(out: &mut Vec<u8>, marker: u8) {
    out.push(0xFF);
    out.push(marker);
}

fn write_segment(
    out: &mut Vec<u8>,
    marker: u8,
    payload: &[u8],
    name: &'static str,
) -> Result<(), JpegEncodeError> {
    let len = payload
        .len()
        .checked_add(2)
        .and_then(|value| u16::try_from(value).ok())
        .ok_or(JpegEncodeError::SegmentTooLarge { name })?;
    write_marker(out, marker);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    Ok(())
}

fn write_dqt(out: &mut Vec<u8>, table_id: u8, quant: &[u8; 64]) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(65);
    payload.push(table_id);
    for &natural_idx in &JPEG_BASELINE_ZIGZAG {
        payload.push(quant[natural_idx as usize]);
    }
    write_segment(out, 0xDB, &payload, "DQT")
}

fn write_dri(out: &mut Vec<u8>, restart_interval: u16) -> Result<(), JpegEncodeError> {
    write_segment(out, 0xDD, &restart_interval.to_be_bytes(), "DRI")
}

fn write_sof0(
    out: &mut Vec<u8>,
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
) -> Result<(), JpegEncodeError> {
    let height =
        u16::try_from(height).map_err(|_| JpegEncodeError::DimensionsTooLarge { width, height })?;
    let width = u16::try_from(width).map_err(|_| JpegEncodeError::DimensionsTooLarge {
        width,
        height: u32::from(height),
    })?;
    let mut payload = Vec::with_capacity(6 + sampling.components as usize * 3);
    payload.push(8);
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&width.to_be_bytes());
    payload.push(sampling.components);
    for component in 0..sampling.components as usize {
        let component_id = u8::try_from(component + 1)
            .map_err(|_| JpegEncodeError::Internal("JPEG component id exceeds u8".into()))?;
        payload.push(component_id);
        payload.push((sampling.h[component] << 4) | sampling.v[component]);
        payload.push(u8::from(component != 0));
    }
    write_segment(out, 0xC0, &payload, "SOF0")
}

fn write_dht(
    out: &mut Vec<u8>,
    class: u8,
    table_id: u8,
    bits: &[u8; 16],
    values: &[u8],
) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(17 + values.len());
    payload.push((class << 4) | table_id);
    payload.extend_from_slice(bits);
    payload.extend_from_slice(values);
    write_segment(out, 0xC4, &payload, "DHT")
}

fn write_sos(out: &mut Vec<u8>, components: u8) -> Result<(), JpegEncodeError> {
    let mut payload = Vec::with_capacity(4 + components as usize * 2);
    payload.push(components);
    for component in 0..components {
        payload.push(component + 1);
        payload.push(if component == 0 { 0x00 } else { 0x11 });
    }
    payload.push(0);
    payload.push(63);
    payload.push(0);
    write_segment(out, 0xDA, &payload, "SOS")
}

const STD_LUMA_Q: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

const STD_CHROMA_Q: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

const STD_LUMA_DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const STD_LUMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const STD_CHROMA_DC_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const STD_CHROMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const STD_LUMA_AC_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
const STD_LUMA_AC_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

const STD_CHROMA_AC_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const STD_CHROMA_AC_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

#[cfg(test)]
mod gpu_encode_tests {
    use super::*;

    fn rgb_tile() -> JpegBaselineGpuEncodeTile {
        JpegBaselineGpuEncodeTile {
            byte_offset: 32,
            width: 17,
            height: 9,
            pitch_bytes: 64,
            output_width: 32,
            output_height: 16,
            format: PixelFormat::Rgb8,
            buffer_len: 32 + 8 * 64 + 17 * 3,
        }
    }

    #[test]
    fn gpu_encode_params_preserve_explicit_offsets() {
        let options = JpegEncodeOptions {
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cuda,
            ..JpegEncodeOptions::default()
        };
        let sampling = jpeg_baseline_sampling_for(options.subsampling);
        let tile = rgb_tile();

        validate_jpeg_baseline_gpu_encode_tile(tile, options, JpegBackend::Cuda)
            .expect("valid tile");
        let params =
            jpeg_baseline_gpu_encode_params(tile, options, sampling, 4096, tile.byte_offset, 128)
                .expect("gpu params");

        assert_eq!(params.input_offset_bytes, 32);
        assert_eq!(params.entropy_offset_bytes, 128);
        assert_eq!(params.entropy_capacity, 4096);
        assert_eq!(params.format, 1);
        assert_eq!(params.components, 3);
        assert_eq!(params.mcus_per_row, 2);
        assert_eq!(params.mcu_rows, 1);
        assert_eq!(params.restart_interval_mcus, 4);
    }

    #[test]
    fn gpu_encode_batch_plan_accumulates_offsets_in_tile_order() {
        let options = JpegEncodeOptions {
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cuda,
            ..JpegEncodeOptions::default()
        };
        let sampling = jpeg_baseline_sampling_for(options.subsampling);
        let first = rgb_tile();
        let mut second = rgb_tile();
        second.byte_offset = 512;
        second.buffer_len = second.byte_offset + 8 * second.pitch_bytes + 17 * 3;

        let plan = jpeg_baseline_gpu_encode_batch_plan(
            &[first, second],
            options,
            JpegBackend::Cuda,
            sampling,
        )
        .expect("valid batch plan");

        assert_eq!(plan.params.len(), 2);
        assert_eq!(plan.params[0].input_offset_bytes, first.byte_offset as u32);
        assert_eq!(plan.params[0].entropy_offset_bytes, 0);
        assert_eq!(plan.params[1].input_offset_bytes, second.byte_offset as u32);
        assert_eq!(
            plan.params[1].entropy_offset_bytes,
            plan.params[0].entropy_capacity
        );
        assert_eq!(
            plan.total_entropy_capacity,
            usize::try_from(plan.params[0].entropy_capacity).unwrap()
                + usize::try_from(plan.params[1].entropy_capacity).unwrap()
        );
    }

    #[test]
    fn gpu_encode_validation_reports_short_pitch() {
        let mut tile = rgb_tile();
        tile.pitch_bytes = 50;
        let err = validate_jpeg_baseline_gpu_encode_tile(
            tile,
            JpegEncodeOptions {
                subsampling: JpegSubsampling::Ybr444,
                backend: JpegBackend::Metal,
                ..JpegEncodeOptions::default()
            },
            JpegBackend::Metal,
        )
        .expect_err("short pitch must fail");

        match err {
            JpegBaselineGpuEncodeError::PitchTooShort {
                row_bytes,
                pitch_bytes,
            } => {
                assert_eq!(row_bytes, 51);
                assert_eq!(pitch_bytes, 50);
            }
            other => panic!("unexpected validation error: {other:?}"),
        }
    }

    #[test]
    fn gpu_encode_batch_plan_validates_every_tile() {
        let options = JpegEncodeOptions {
            subsampling: JpegSubsampling::Ybr444,
            backend: JpegBackend::Metal,
            ..JpegEncodeOptions::default()
        };
        let sampling = jpeg_baseline_sampling_for(options.subsampling);
        let mut second = rgb_tile();
        second.pitch_bytes = 50;

        let err = jpeg_baseline_gpu_encode_batch_plan(
            &[rgb_tile(), second],
            options,
            JpegBackend::Metal,
            sampling,
        )
        .expect_err("invalid second tile must fail");

        match err {
            JpegBaselineGpuEncodeError::PitchTooShort {
                row_bytes,
                pitch_bytes,
            } => {
                assert_eq!(row_bytes, 51);
                assert_eq!(pitch_bytes, 50);
            }
            other => panic!("unexpected validation error: {other:?}"),
        }
    }

    #[test]
    fn same_source_buffer_batch_end_groups_contiguous_keys() {
        let tiles = [10u64, 10, 10, 11, 10];

        assert_eq!(same_source_buffer_batch_end(&tiles, 0, |value| *value), 3);
        assert_eq!(same_source_buffer_batch_end(&tiles, 3, |value| *value), 4);
    }
}
