// SPDX-License-Identifier: Apache-2.0

//! Optional acceleration hooks for coefficient-domain transform stages.
//!
//! These hooks are intentionally narrow: accelerated backends may replace the
//! direct DCT-grid to one-level wavelet projection, while the scalar path
//! remains the default oracle and fallback.

use crate::dct53_2d::Dwt53TwoDimensional;
use crate::dct97_2d::Dwt97TwoDimensional;
use rayon::prelude::*;
pub use signinum_j2k_native::{
    IrreversibleQuantizationSubbandScales, J2kSubBandType, PreencodedHtj2k97CodeBlock,
    PreencodedHtj2k97CompactCodeBlock, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97CompactImage, PreencodedHtj2k97CompactResolution,
    PreencodedHtj2k97CompactSubband, PreencodedHtj2k97Component, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};
use signinum_jpeg::transcode::idct_islow_block;

const REVERSIBLE_DWT53_UNSUPPORTED_GRID: &str =
    "reversible DCT 5/3 job has unsupported grid geometry";

/// Direct DCT-grid to one-level reversible integer 5/3 projection job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToReversibleDwt53Job<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub dequantized_blocks: &'a [[i16; 64]],
    /// Number of DCT block columns in `dequantized_blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `dequantized_blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
}

/// One separable single-level reversible integer 5/3 transform result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReversibleDwt53FirstLevel {
    /// Low-horizontal, low-vertical band.
    pub ll: Vec<i32>,
    /// High-horizontal, low-vertical band.
    pub hl: Vec<i32>,
    /// Low-horizontal, high-vertical band.
    pub lh: Vec<i32>,
    /// High-horizontal, high-vertical band.
    pub hh: Vec<i32>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Direct DCT-grid to one-level 5/3 projection job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToDwt53Job<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub blocks: &'a [[[f64; 8]; 8]],
    /// Number of DCT block columns in `blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
}

/// Direct DCT-grid to one-level 9/7 transform job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToDwt97Job<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub blocks: &'a [[[f64; 8]; 8]],
    /// Number of DCT block columns in `blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
}

/// Direct DCT-grid to prequantized one-level 9/7 HTJ2K code-block job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToHtj2k97CodeBlockJob<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub blocks: &'a [[[f64; 8]; 8]],
    /// Number of DCT block columns in `blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
}

/// Direct dequantized i16 DCT-grid to one-level 9/7 HTJ2K code-block job.
///
/// This is for accelerators that consume the JPEG coefficient extraction
/// output directly and do not need the generic f64 block representation.
#[derive(Debug, Clone, Copy)]
pub struct DctGridI16ToHtj2k97CodeBlockJob<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub dequantized_blocks: &'a [[i16; 64]],
    /// Number of DCT block columns in `dequantized_blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `dequantized_blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
}

/// One same-geometry i16 DCT-grid HTJ2K preencode batch.
#[derive(Debug, Clone, Copy)]
pub struct DctGridI16ToHtj2k97CodeBlockBatch<'a, 'j> {
    /// Jobs in this same-geometry batch.
    pub jobs: &'j [DctGridI16ToHtj2k97CodeBlockJob<'a>],
}

/// Compact preencoded HTJ2K components backed by one payload buffer.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97CompactBatch {
    /// Contiguous encoded code-block payload bytes for every component.
    pub payload: Vec<u8>,
    /// Compact components in the same order as the submitted jobs.
    pub components: Vec<PreencodedHtj2k97CompactComponent>,
}

/// Compact preencoded HTJ2K grouped-batch output backed by one payload buffer.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97CompactBatchGroups {
    /// Contiguous encoded code-block payload bytes for every returned group.
    pub payload: Vec<u8>,
    /// Compact components grouped in the same order as submitted batches.
    pub groups: Vec<Vec<PreencodedHtj2k97CompactComponent>>,
}

/// Encode parameters needed to quantize 9/7 output directly into HTJ2K
/// code-block coefficient layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Htj2k97CodeBlockOptions {
    /// Component precision in bits.
    pub bit_depth: u8,
    /// JPEG 2000 guard bits used for QCD and code-block bitplane counts.
    pub guard_bits: u8,
    /// Code-block width exponent minus two.
    pub code_block_width_exp: u8,
    /// Code-block height exponent minus two.
    pub code_block_height_exp: u8,
    /// Multiplier applied to irreversible 9/7 scalar quantization step sizes.
    pub irreversible_quantization_scale: f32,
    /// Per-subband multipliers applied on top of
    /// [`irreversible_quantization_scale`](Self::irreversible_quantization_scale).
    pub irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
}

/// Backend-specific timing breakdown for a same-geometry 9/7 batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dwt97BatchStageTimings {
    /// Host packing, buffer allocation, and upload time in microseconds.
    pub pack_upload_us: u128,
    /// Time spent in the IDCT plus horizontal 9/7 row-lift stage.
    pub idct_row_lift_us: u128,
    /// Time spent in the vertical 9/7 column-lift stage.
    pub column_lift_us: u128,
    /// Time spent quantizing 9/7 bands into HTJ2K code-block layout.
    pub quantize_codeblock_us: u128,
    /// Time spent HT-encoding resident code-block coefficients.
    pub ht_encode_us: u128,
    /// Resident HT cleanup-pass encode kernel time in microseconds.
    pub ht_kernel_us: u128,
    /// Resident HT status-buffer device-to-host readback time in microseconds.
    pub ht_status_readback_us: u128,
    /// Resident HT encoded-byte compaction kernel time in microseconds.
    pub ht_compact_us: u128,
    /// Resident HT compacted encoded-byte device-to-host readback time in microseconds.
    pub ht_output_readback_us: u128,
    /// Number of HT code-block encode kernel dispatches in this batch.
    pub ht_codeblock_dispatches: usize,
    /// Time spent reading and unpacking Metal band buffers into host outputs.
    pub readback_us: u128,
}

/// Optional backend for SIMD, GPU, or other accelerated transform stages.
pub trait DctToWaveletStageAccelerator {
    /// Whether this accelerator wants same-geometry 9/7 batch jobs offered.
    ///
    /// The default is false so CPU-only fallback paths do not pay the memory
    /// cost of materializing batch-owned float DCT blocks before immediately
    /// falling back.
    fn supports_dwt97_batch(&self) -> bool {
        false
    }

    /// Whether this accelerator wants same-geometry 9/7 batches offered as
    /// prequantized HTJ2K code-block jobs before the float-band hook.
    fn supports_htj2k97_codeblock_batch(&self) -> bool {
        false
    }

    /// Whether this accelerator wants same-geometry 9/7 preencoded HTJ2K
    /// batches offered with dequantized i16 DCT blocks before materializing the
    /// generic f64 block representation.
    fn supports_htj2k97_i16_preencoded_batch(&self) -> bool {
        false
    }

    /// Optionally compute the direct DCT-grid to one-level reversible integer
    /// 5/3 projection.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job bit-exactly
    /// relative to signinum's scalar integer oracle. Return `Ok(None)` to use
    /// the scalar fallback.
    fn dct_grid_to_reversible_dwt53(
        &mut self,
        _job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, &'static str> {
        Ok(None)
    }

    /// Optionally compute a same-geometry batch of direct DCT-grid to
    /// one-level reversible integer 5/3 projections.
    ///
    /// Backends should return outputs in the same order as `jobs`. Return
    /// `Ok(None)` to use the scalar per-component fallback.
    fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        _jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute the direct DCT-grid to one-level 5/3 projection.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job. Return
    /// `Ok(None)` to use the scalar fallback.
    fn dct_grid_to_dwt53(
        &mut self,
        _job: DctGridToDwt53Job<'_>,
    ) -> Result<Option<Dwt53TwoDimensional<f64>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute the direct DCT-grid to one-level 9/7 transform.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job. Return
    /// `Ok(None)` to use the scalar fallback.
    fn dct_grid_to_dwt97(
        &mut self,
        _job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute a same-geometry batch of direct DCT-grid to
    /// one-level 9/7 transforms.
    ///
    /// Backends should return outputs in the same order as `jobs`. Return
    /// `Ok(None)` to use the scalar per-component fallback.
    fn dct_grid_to_dwt97_batch(
        &mut self,
        _jobs: &[DctGridToDwt97Job<'_>],
    ) -> Result<Option<Vec<Dwt97TwoDimensional<f64>>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute same-geometry DCT-grid 9/7 jobs directly into
    /// prequantized HTJ2K code-block components.
    ///
    /// Backends should return one component per input job in the same order as
    /// `jobs`. Return `Ok(None)` to use the float-band path.
    fn dct_grid_to_htj2k97_codeblock_batch(
        &mut self,
        _jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
        _options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PrequantizedHtj2k97Component>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute same-geometry DCT-grid 9/7 jobs directly into
    /// preencoded HTJ2K code-block payloads.
    ///
    /// Backends should return one component per input job in the same order as
    /// `jobs`. Return `Ok(None)` to use the prequantized or float-band path.
    fn dct_grid_to_htj2k97_preencoded_batch(
        &mut self,
        _jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
        _options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PreencodedHtj2k97Component>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute same-geometry dequantized i16 DCT-grid 9/7 jobs
    /// directly into preencoded HTJ2K code-block payloads.
    ///
    /// Backends should return one component per input job in the same order as
    /// `jobs`. Return `Ok(None)` to use the generic f64 preencoded path.
    fn dct_grid_i16_to_htj2k97_preencoded_batch(
        &mut self,
        _jobs: &[DctGridI16ToHtj2k97CodeBlockJob<'_>],
        _options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<PreencodedHtj2k97Component>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute same-geometry dequantized i16 DCT-grid 9/7 jobs into
    /// compact preencoded HTJ2K code-block payloads.
    ///
    /// Backends should return one component per input job in the same order as
    /// `jobs`, with all component ranges pointing into the returned payload.
    /// Return `Ok(None)` to use the owned preencoded path.
    fn dct_grid_i16_to_htj2k97_compact_preencoded_batch(
        &mut self,
        _jobs: &[DctGridI16ToHtj2k97CodeBlockJob<'_>],
        _options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<PreencodedHtj2k97CompactBatch>, &'static str> {
        Ok(None)
    }

    /// Optionally compute multiple same-geometry dequantized i16 DCT-grid
    /// batches directly into preencoded HTJ2K code-block payloads.
    ///
    /// Each input batch is internally same-geometry, but different batches may
    /// have different component dimensions. Backends should return one output
    /// vector per input batch, in order. Return `Ok(None)` to use the per-group
    /// fallback hooks.
    fn dct_grid_i16_to_htj2k97_preencoded_batch_groups(
        &mut self,
        _groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'_, '_>],
        _options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<Vec<Vec<PreencodedHtj2k97Component>>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute multiple same-geometry dequantized i16 DCT-grid 9/7
    /// batches into compact preencoded HTJ2K code-block payloads.
    ///
    /// Each returned item corresponds to one input batch and contains one
    /// component per job in that batch. Return `Ok(None)` to use the owned
    /// preencoded grouped hook.
    fn dct_grid_i16_to_htj2k97_compact_preencoded_batch_groups(
        &mut self,
        _groups: &[DctGridI16ToHtj2k97CodeBlockBatch<'_, '_>],
        _options: Htj2k97CodeBlockOptions,
    ) -> Result<Option<PreencodedHtj2k97CompactBatchGroups>, &'static str> {
        Ok(None)
    }

    /// Return backend stage timings for the most recent 9/7 batch dispatch.
    fn last_dwt97_batch_stage_timings(&self) -> Option<Dwt97BatchStageTimings> {
        None
    }
}

/// Accelerator that always uses the scalar CPU fallback.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuOnlyDctToWaveletStageAccelerator;

impl DctToWaveletStageAccelerator for CpuOnlyDctToWaveletStageAccelerator {}

/// CPU/Rayon accelerator for the exact reversible integer 5/3 first level.
///
/// This backend keeps signinum's scalar ISLOW IDCT semantics as the oracle:
/// each 8x8 block is decoded with `signinum-jpeg`, level-shifted to signed
/// component samples, then transformed with reversible integer 5/3 lifting.
#[derive(Debug, Default, Clone)]
pub struct RayonReversibleDwt53Accelerator {
    attempts: usize,
    dispatches: usize,
    batch_attempts: usize,
    batch_dispatches: usize,
}

impl RayonReversibleDwt53Accelerator {
    /// Number of reversible 5/3 jobs offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_attempts(&self) -> usize {
        self.attempts
    }

    /// Number of reversible 5/3 jobs handled by this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_dispatches(&self) -> usize {
        self.dispatches
    }

    /// Number of reversible 5/3 batches offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_batch_attempts(&self) -> usize {
        self.batch_attempts
    }

    /// Number of reversible 5/3 batches handled by this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_batch_dispatches(&self) -> usize {
        self.batch_dispatches
    }
}

impl DctToWaveletStageAccelerator for RayonReversibleDwt53Accelerator {
    fn dct_grid_to_reversible_dwt53(
        &mut self,
        job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, &'static str> {
        self.attempts = self.attempts.saturating_add(1);
        let output = reversible_dwt53_first_level_rayon(job)?;
        self.dispatches = self.dispatches.saturating_add(1);
        Ok(Some(output))
    }

    fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, &'static str> {
        self.batch_attempts = self.batch_attempts.saturating_add(1);
        let mut output = Vec::with_capacity(jobs.len());
        for job in jobs {
            output.push(reversible_dwt53_first_level_rayon(*job)?);
        }
        self.batch_dispatches = self.batch_dispatches.saturating_add(1);
        Ok(Some(output))
    }
}

/// Decode the job's dequantized DCT blocks into signinum's signed integer
/// component sample blocks.
///
/// This is public so hybrid GPU backends can keep JPEG parsing and exact IDCT
/// on CPU while offloading the reversible 5/3 projection.
pub fn idct_blocks_to_signed_samples_rayon(blocks: &[[i16; 64]]) -> Vec<[i32; 64]> {
    blocks
        .par_iter()
        .map(|block| {
            let decoded = idct_islow_block(block);
            decoded.map(|sample| i32::from(sample) - 128)
        })
        .collect()
}

/// Compute one exact reversible integer 5/3 level from already decoded
/// block-local signed samples.
pub fn reversible_dwt53_first_level_from_block_samples(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<ReversibleDwt53FirstLevel, &'static str> {
    validate_reversible_grid(block_samples.len(), block_cols, block_rows, width, height)?;

    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    let low_rows: Vec<(Vec<i32>, Vec<i32>)> = (0..low_height)
        .into_par_iter()
        .map(|output_y| {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                row.push(vertical_low_53_i32_at(
                    block_samples,
                    block_cols,
                    width,
                    height,
                    x,
                    output_y,
                ));
            }
            reversible_lift_53_i32(&mut row);
            (
                row.iter().step_by(2).copied().collect(),
                row.iter().skip(1).step_by(2).copied().collect(),
            )
        })
        .collect();
    let high_rows: Vec<(Vec<i32>, Vec<i32>)> = (0..high_height)
        .into_par_iter()
        .map(|output_y| {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                row.push(vertical_high_53_i32_at(
                    block_samples,
                    block_cols,
                    width,
                    height,
                    x,
                    output_y,
                ));
            }
            reversible_lift_53_i32(&mut row);
            (
                row.iter().step_by(2).copied().collect(),
                row.iter().skip(1).step_by(2).copied().collect(),
            )
        })
        .collect();

    let mut ll = Vec::with_capacity(low_width * low_height);
    let mut hl = Vec::with_capacity(high_width * low_height);
    for (low, high) in low_rows {
        ll.extend(low);
        hl.extend(high);
    }

    let mut lh = Vec::with_capacity(low_width * high_height);
    let mut hh = Vec::with_capacity(high_width * high_height);
    for (low, high) in high_rows {
        lh.extend(low);
        hh.extend(high);
    }

    Ok(ReversibleDwt53FirstLevel {
        ll,
        hl,
        lh,
        hh,
        low_width,
        low_height,
        high_width,
        high_height,
    })
}

fn reversible_dwt53_first_level_rayon(
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, &'static str> {
    validate_reversible_grid(
        job.dequantized_blocks.len(),
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
    )?;
    let block_samples = idct_blocks_to_signed_samples_rayon(job.dequantized_blocks);
    reversible_dwt53_first_level_from_block_samples(
        &block_samples,
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
    )
}

fn validate_reversible_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<(), &'static str> {
    let expected_blocks = block_cols
        .checked_mul(block_rows)
        .ok_or(REVERSIBLE_DWT53_UNSUPPORTED_GRID)?;
    let covered_width = block_cols
        .checked_mul(8)
        .ok_or(REVERSIBLE_DWT53_UNSUPPORTED_GRID)?;
    let covered_height = block_rows
        .checked_mul(8)
        .ok_or(REVERSIBLE_DWT53_UNSUPPORTED_GRID)?;

    if block_count != expected_blocks
        || width == 0
        || height == 0
        || width > covered_width
        || height > covered_height
    {
        return Err(REVERSIBLE_DWT53_UNSUPPORTED_GRID);
    }
    Ok(())
}

fn vertical_low_53_i32_at(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    width: usize,
    height: usize,
    x: usize,
    low_idx: usize,
) -> i32 {
    let even_idx = low_idx * 2;
    let current = component_sample_i32(block_samples, block_cols, width, height, x, even_idx);
    if height < 2 {
        return current;
    }

    if height.is_multiple_of(2) {
        let right = vertical_high_53_i32_at(block_samples, block_cols, width, height, x, low_idx);
        if low_idx == 0 {
            return current + floor_div_i32(right + 1, 2);
        }
        let left =
            vertical_high_53_i32_at(block_samples, block_cols, width, height, x, low_idx - 1);
        return current + floor_div_i32(left + right + 2, 4);
    }

    let high_len = height / 2;
    if high_len == 0 {
        return current;
    }
    let left = if low_idx > 0 {
        vertical_high_53_i32_at(block_samples, block_cols, width, height, x, low_idx - 1)
    } else {
        vertical_high_53_i32_at(block_samples, block_cols, width, height, x, 0)
    };
    let right = if low_idx < high_len {
        vertical_high_53_i32_at(block_samples, block_cols, width, height, x, low_idx)
    } else {
        left
    };
    current + floor_div_i32(left + right + 2, 4)
}

fn vertical_high_53_i32_at(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    width: usize,
    height: usize,
    x: usize,
    high_idx: usize,
) -> i32 {
    let odd_idx = high_idx * 2 + 1;
    let current = component_sample_i32(block_samples, block_cols, width, height, x, odd_idx);
    let left = component_sample_i32(block_samples, block_cols, width, height, x, odd_idx - 1);
    if height.is_multiple_of(2) && odd_idx + 1 == height {
        return current - left;
    }

    let right_idx = if odd_idx + 1 < height {
        odd_idx + 1
    } else {
        height - 1
    };
    let right = component_sample_i32(block_samples, block_cols, width, height, x, right_idx);
    current - floor_div_i32(left + right, 2)
}

fn component_sample_i32(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> i32 {
    debug_assert!(x < width);
    debug_assert!(y < height);
    let block_x = x / 8;
    let block_y = y / 8;
    let block_idx = block_y * block_cols + block_x;
    let local_idx = (y % 8) * 8 + (x % 8);
    block_samples[block_idx][local_idx]
}

fn reversible_lift_53_i32(values: &mut [i32]) {
    let n = values.len();
    if n < 2 {
        return;
    }

    if n.is_multiple_of(2) {
        for i in (1..n - 1).step_by(2) {
            values[i] -= floor_div_i32(values[i - 1] + values[i + 1], 2);
        }
        values[n - 1] -= values[n - 2];

        values[0] += floor_div_i32(values[1] + 1, 2);
        for i in (2..n).step_by(2) {
            values[i] += floor_div_i32(values[i - 1] + values[i + 1] + 2, 4);
        }
        return;
    }

    let last_even = n - 1;
    for i in (1..n).step_by(2) {
        let right = values.get(i + 1).copied().unwrap_or(values[last_even]);
        values[i] -= floor_div_i32(values[i - 1] + right, 2);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { values[i - 1] } else { values[1] };
        let right = values.get(i + 1).copied().unwrap_or(left);
        values[i] += floor_div_i32(left + right + 2, 4);
    }
}

fn floor_div_i32(numerator: i32, denominator: i32) -> i32 {
    numerator.div_euclid(denominator)
}

#[cfg(test)]
mod ground_truth_tests {
    //! Independent ground truth for the reversible integer 5/3.
    //!
    //! The CUDA 5/3 kernel is parity-tested against the lifting in this module,
    //! so a boundary/indexing/band-split bug here would be faithfully copied by
    //! the kernel and pass parity. Validate the lifting against the canonical
    //! JPEG2000 reversible 5/3 (ISO/IEC 15444-1 Annex F.3.8.1) evaluated per
    //! output index from a whole-sample-symmetrically extended signal — a
    //! structurally different implementation than the in-place two-pass loops.

    use super::{
        reversible_dwt53_first_level_from_block_samples, reversible_lift_53_i32,
        ReversibleDwt53FirstLevel,
    };

    fn floor2(a: i32, b: i32) -> i32 {
        a.div_euclid(b)
    }

    /// Whole-sample symmetric reflection (mirror about 0 and `n - 1`, endpoints
    /// not repeated) — the boundary extension the lifting realizes at the edges.
    fn ws_reflect(i: isize, n: usize) -> usize {
        if n == 1 {
            return 0;
        }
        let n = isize::try_from(n).unwrap();
        let period = 2 * (n - 1);
        let mut k = i.rem_euclid(period);
        if k >= n {
            k = period - k;
        }
        usize::try_from(k).unwrap()
    }

    /// Canonical forward 5/3: `(low, high)` where `low[m]` is the even/approx
    /// coefficient and `high[m]` the odd/detail coefficient. Every index is read
    /// through whole-sample symmetric extension of the original signal, so the
    /// detail-boundary behavior follows automatically (no special cases).
    fn ref_53_forward(signal: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let n = signal.len();
        if n < 2 {
            return (signal.to_vec(), Vec::new());
        }
        let sig = |i: isize| signal[ws_reflect(i, n)];
        let detail = |m: isize| {
            let c = 2 * m + 1;
            sig(c) - floor2(sig(c - 1) + sig(c + 1), 2)
        };
        let low: Vec<i32> = (0..n.div_ceil(2))
            .map(|m| {
                let mi = isize::try_from(m).unwrap();
                sig(2 * mi) + floor2(detail(mi - 1) + detail(mi) + 2, 4)
            })
            .collect();
        let high: Vec<i32> = (0..n / 2)
            .map(|m| detail(isize::try_from(m).unwrap()))
            .collect();
        (low, high)
    }

    /// Separable 2D reference matching the oracle's vertical-then-horizontal
    /// order (integer floor lifting is NOT order-independent, so order matters).
    fn ref_53_2d(plane: &[i32], width: usize, height: usize) -> ReversibleDwt53FirstLevel {
        let low_width = width.div_ceil(2);
        let high_width = width / 2;
        let low_height = height.div_ceil(2);
        let high_height = height / 2;

        let mut v_low = vec![0i32; width * low_height];
        let mut v_high = vec![0i32; width * high_height];
        for x in 0..width {
            let column: Vec<i32> = (0..height).map(|y| plane[y * width + x]).collect();
            let (lo, hi) = ref_53_forward(&column);
            for (oy, &value) in lo.iter().enumerate() {
                v_low[oy * width + x] = value;
            }
            for (oy, &value) in hi.iter().enumerate() {
                v_high[oy * width + x] = value;
            }
        }

        let horizontal = |source: &[i32], rows: usize| -> (Vec<i32>, Vec<i32>) {
            let mut low = vec![0i32; low_width * rows];
            let mut high = vec![0i32; high_width * rows];
            for oy in 0..rows {
                let (lo, hi) = ref_53_forward(&source[oy * width..oy * width + width]);
                low[oy * low_width..oy * low_width + low_width].copy_from_slice(&lo);
                high[oy * high_width..oy * high_width + high_width].copy_from_slice(&hi);
            }
            (low, high)
        };

        let (ll, hl) = horizontal(&v_low, low_height);
        let (lh, hh) = horizontal(&v_high, high_height);

        ReversibleDwt53FirstLevel {
            ll,
            hl,
            lh,
            hh,
            low_width,
            low_height,
            high_width,
            high_height,
        }
    }

    /// Pack a flat `width x height` sample plane into the block-major
    /// `[[i32; 64]]` layout `reversible_dwt53_first_level_from_block_samples`
    /// consumes (local index `(y % 8) * 8 + (x % 8)`).
    fn pack_plane(plane: &[i32], width: usize, height: usize) -> (Vec<[i32; 64]>, usize, usize) {
        let block_cols = width.div_ceil(8);
        let block_rows = height.div_ceil(8);
        let mut blocks = vec![[0i32; 64]; block_cols * block_rows];
        for y in 0..height {
            for x in 0..width {
                let block = (y / 8) * block_cols + (x / 8);
                blocks[block][(y % 8) * 8 + (x % 8)] = plane[y * width + x];
            }
        }
        (blocks, block_cols, block_rows)
    }

    fn next_sample(state: &mut u64) -> i32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 40) & 0x1ff) as i32 - 256
    }

    #[test]
    fn reversible_lift_53_matches_canonical_formula_1d() {
        let mut state = 0x0a11_ce5e_ed00_d001u64;
        for n in [2usize, 3, 4, 5, 8, 9, 12, 15, 16, 23, 32, 33, 64, 65] {
            let signal: Vec<i32> = (0..n).map(|_| next_sample(&mut state)).collect();
            let mut lifted = signal.clone();
            reversible_lift_53_i32(&mut lifted);
            let lifted_low: Vec<i32> = lifted.iter().step_by(2).copied().collect();
            let lifted_high: Vec<i32> = lifted.iter().skip(1).step_by(2).copied().collect();
            let (low, high) = ref_53_forward(&signal);
            assert_eq!(lifted_low, low, "low band mismatch for n={n}");
            assert_eq!(lifted_high, high, "high band mismatch for n={n}");
        }
    }

    #[test]
    fn reversible_dwt53_2d_matches_canonical_separable() {
        let mut state = 0xfeed_5eed_d00d_face_u64;
        for (width, height) in [
            (8usize, 8usize),
            (16, 16),
            (24, 16),
            (15, 13),
            (16, 23),
            (9, 7),
            (32, 32),
        ] {
            let plane: Vec<i32> = (0..width * height)
                .map(|_| next_sample(&mut state))
                .collect();
            let (blocks, block_cols, block_rows) = pack_plane(&plane, width, height);
            let got = reversible_dwt53_first_level_from_block_samples(
                &blocks, block_cols, block_rows, width, height,
            )
            .expect("oracle accepts the packed grid");
            let want = ref_53_2d(&plane, width, height);
            assert_eq!(
                (
                    got.low_width,
                    got.low_height,
                    got.high_width,
                    got.high_height
                ),
                (
                    want.low_width,
                    want.low_height,
                    want.high_width,
                    want.high_height
                ),
                "band dimensions for {width}x{height}"
            );
            assert_eq!(got.ll, want.ll, "LL mismatch for {width}x{height}");
            assert_eq!(got.hl, want.hl, "HL mismatch for {width}x{height}");
            assert_eq!(got.lh, want.lh, "LH mismatch for {width}x{height}");
            assert_eq!(got.hh, want.hh, "HH mismatch for {width}x{height}");
        }
    }

    #[test]
    fn reversible_lift_53_kills_dc_and_linear_detail() {
        // Constant -> low = constant, detail exactly zero.
        let mut constant = vec![7i32; 32];
        reversible_lift_53_i32(&mut constant);
        assert!(
            constant.iter().skip(1).step_by(2).all(|&v| v == 0),
            "constant produced nonzero detail"
        );
        assert!(
            constant.iter().step_by(2).all(|&v| v == 7),
            "constant low band drifted from 7"
        );

        // Linear ramp -> interior detail exactly zero (two vanishing moments).
        let ramp: Vec<i32> = (0..40_i32).map(|k| 3 * k - 5).collect();
        let mut lifted = ramp;
        reversible_lift_53_i32(&mut lifted);
        let detail: Vec<i32> = lifted.iter().skip(1).step_by(2).copied().collect();
        for &value in &detail[1..detail.len() - 1] {
            assert_eq!(value, 0, "linear ramp produced interior detail {value}");
        }
    }

    #[test]
    fn reversible_dwt53_2d_separates_horizontal_and_vertical_detail() {
        // Varies only along x -> no vertical detail (LH and HH vanish).
        let (width, height) = (16usize, 16usize);
        let varies_in_x: Vec<i32> = (0..width * height)
            .map(|i| 3 * i32::try_from(i % width).unwrap() - 7)
            .collect();
        let (blocks, bc, br) = pack_plane(&varies_in_x, width, height);
        let t = reversible_dwt53_first_level_from_block_samples(&blocks, bc, br, width, height)
            .expect("oracle accepts grid");
        assert!(
            t.lh.iter().all(|&v| v == 0),
            "x-only plane produced LH detail"
        );
        assert!(
            t.hh.iter().all(|&v| v == 0),
            "x-only plane produced HH detail"
        );

        // Varies only along y -> no horizontal detail (HL and HH vanish).
        let varies_in_y: Vec<i32> = (0..width * height)
            .map(|i| 3 * i32::try_from(i / width).unwrap() - 7)
            .collect();
        let (blocks, bc, br) = pack_plane(&varies_in_y, width, height);
        let t = reversible_dwt53_first_level_from_block_samples(&blocks, bc, br, width, height)
            .expect("oracle accepts grid");
        assert!(
            t.hl.iter().all(|&v| v == 0),
            "y-only plane produced HL detail"
        );
        assert!(
            t.hh.iter().all(|&v| v == 0),
            "y-only plane produced HH detail"
        );
    }
}
