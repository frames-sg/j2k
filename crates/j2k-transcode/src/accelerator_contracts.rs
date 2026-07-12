// SPDX-License-Identifier: MIT OR Apache-2.0
// j2k-coverage: shared-accelerator-host

// Optional acceleration hooks for coefficient-domain transform stages.
//
// These hooks are intentionally narrow: accelerated backends may replace the
// direct DCT-grid to one-level wavelet projection, while the scalar path
// remains the default oracle and fallback.

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, try_vec_filled, try_vec_with_capacity,
};
use crate::dct_grid::validate_dct_block_grid;
use crate::reversible53::{
    reversible_lift_53_high_at, reversible_lift_53_i32, reversible_lift_53_low_at,
};
use crate::{
    DctGridToReversibleDwt53Job, Dwt53TwoDimensional, Dwt97BatchStageTimings, Dwt97TwoDimensional,
    ReversibleDwt53FirstLevel, TranscodeStageError,
};
pub use j2k::{
    EncodedHtJ2kCodeBlock, IrreversibleQuantizationSubbandScales, J2kSubBandType,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Resolution, PreencodedHtj2k97Subband,
    PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component, PrequantizedHtj2k97Image,
    PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};
use j2k_jpeg::transcode::idct_islow_block;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    ParallelSliceMut,
};

const REVERSIBLE_DWT53_UNSUPPORTED_GRID: &str =
    "reversible DCT 5/3 job has unsupported grid geometry";

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
#[derive(Debug)]
pub struct PreencodedHtj2k97CompactBatch {
    /// Contiguous encoded code-block payload bytes for every component.
    pub payload: Vec<u8>,
    /// Compact components in the same order as the submitted jobs.
    pub components: Vec<PreencodedHtj2k97CompactComponent>,
}

/// Compact preencoded HTJ2K grouped-batch output backed by one payload buffer.
#[derive(Debug)]
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

/// Counter row recorded by DCT-to-wavelet stage accelerators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctToWaveletStageCounterEvent {
    /// One reversible integer 5/3 job was offered to the accelerator.
    ReversibleDwt53Attempt,
    /// One reversible integer 5/3 job was handled by the accelerator.
    ReversibleDwt53Dispatch,
    /// One reversible integer 5/3 batch was offered to the accelerator.
    ReversibleDwt53BatchAttempt,
    /// One reversible integer 5/3 batch was handled by the accelerator.
    ReversibleDwt53BatchDispatch,
    /// One 5/3 projection job was offered to the accelerator.
    Dwt53Attempt,
    /// One 5/3 projection job was handled by the accelerator.
    Dwt53Dispatch,
    /// One 9/7 transform job was offered to the accelerator.
    Dwt97Attempt,
    /// One 9/7 transform job was handled by the accelerator.
    Dwt97Dispatch,
    /// One same-geometry 9/7 transform batch was offered to the accelerator.
    Dwt97BatchAttempt,
    /// One same-geometry 9/7 transform batch was handled by the accelerator.
    Dwt97BatchDispatch,
    /// One 9/7 code-block-ready batch was offered to the accelerator.
    Htj2k97CodeblockBatchAttempt,
    /// One 9/7 code-block-ready batch was handled by the accelerator.
    Htj2k97CodeblockBatchDispatch,
}

/// Shared offered/handled counters for DCT-to-wavelet stage accelerators.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DctToWaveletStageCounters {
    reversible_dwt53_attempts: usize,
    reversible_dwt53_dispatches: usize,
    reversible_dwt53_batch_attempts: usize,
    reversible_dwt53_batch_dispatches: usize,
    dwt53_attempts: usize,
    dwt53_dispatches: usize,
    dwt97_attempts: usize,
    dwt97_dispatches: usize,
    dwt97_batch_attempts: usize,
    dwt97_batch_dispatches: usize,
    htj2k97_codeblock_batch_attempts: usize,
    htj2k97_codeblock_batch_dispatches: usize,
}

impl DctToWaveletStageCounters {
    /// Create an empty counter set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            reversible_dwt53_attempts: 0,
            reversible_dwt53_dispatches: 0,
            reversible_dwt53_batch_attempts: 0,
            reversible_dwt53_batch_dispatches: 0,
            dwt53_attempts: 0,
            dwt53_dispatches: 0,
            dwt97_attempts: 0,
            dwt97_dispatches: 0,
            dwt97_batch_attempts: 0,
            dwt97_batch_dispatches: 0,
            htj2k97_codeblock_batch_attempts: 0,
            htj2k97_codeblock_batch_dispatches: 0,
        }
    }

    /// Number of reversible integer 5/3 jobs offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_attempts(&self) -> usize {
        self.reversible_dwt53_attempts
    }

    /// Number of reversible integer 5/3 jobs handled by this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_dispatches(&self) -> usize {
        self.reversible_dwt53_dispatches
    }

    /// Number of reversible integer 5/3 batches offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_batch_attempts(&self) -> usize {
        self.reversible_dwt53_batch_attempts
    }

    /// Number of reversible integer 5/3 batches handled by this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_batch_dispatches(&self) -> usize {
        self.reversible_dwt53_batch_dispatches
    }

    /// Number of 5/3 projection jobs offered to this accelerator.
    #[must_use]
    pub const fn dwt53_attempts(&self) -> usize {
        self.dwt53_attempts
    }

    /// Number of 5/3 projection jobs handled by this accelerator.
    #[must_use]
    pub const fn dwt53_dispatches(&self) -> usize {
        self.dwt53_dispatches
    }

    /// Number of 9/7 transform jobs offered to this accelerator.
    #[must_use]
    pub const fn dwt97_attempts(&self) -> usize {
        self.dwt97_attempts
    }

    /// Number of 9/7 transform jobs handled by this accelerator.
    #[must_use]
    pub const fn dwt97_dispatches(&self) -> usize {
        self.dwt97_dispatches
    }

    /// Number of 9/7 transform batches offered to this accelerator.
    #[must_use]
    pub const fn dwt97_batch_attempts(&self) -> usize {
        self.dwt97_batch_attempts
    }

    /// Number of 9/7 transform batches handled by this accelerator.
    #[must_use]
    pub const fn dwt97_batch_dispatches(&self) -> usize {
        self.dwt97_batch_dispatches
    }

    /// Number of 9/7 code-block-ready batches offered to this accelerator.
    #[must_use]
    pub const fn htj2k97_codeblock_batch_attempts(&self) -> usize {
        self.htj2k97_codeblock_batch_attempts
    }

    /// Number of 9/7 code-block-ready batches handled by this accelerator.
    #[must_use]
    pub const fn htj2k97_codeblock_batch_dispatches(&self) -> usize {
        self.htj2k97_codeblock_batch_dispatches
    }

    /// Record one or more accelerator counter events.
    pub fn record(&mut self, event: DctToWaveletStageCounterEvent, count: usize) {
        match event {
            DctToWaveletStageCounterEvent::ReversibleDwt53Attempt => {
                self.reversible_dwt53_attempts =
                    self.reversible_dwt53_attempts.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::ReversibleDwt53Dispatch => {
                self.reversible_dwt53_dispatches =
                    self.reversible_dwt53_dispatches.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::ReversibleDwt53BatchAttempt => {
                self.reversible_dwt53_batch_attempts =
                    self.reversible_dwt53_batch_attempts.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::ReversibleDwt53BatchDispatch => {
                self.reversible_dwt53_batch_dispatches =
                    self.reversible_dwt53_batch_dispatches.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::Dwt53Attempt => {
                self.dwt53_attempts = self.dwt53_attempts.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::Dwt53Dispatch => {
                self.dwt53_dispatches = self.dwt53_dispatches.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::Dwt97Attempt => {
                self.dwt97_attempts = self.dwt97_attempts.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::Dwt97Dispatch => {
                self.dwt97_dispatches = self.dwt97_dispatches.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::Dwt97BatchAttempt => {
                self.dwt97_batch_attempts = self.dwt97_batch_attempts.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::Dwt97BatchDispatch => {
                self.dwt97_batch_dispatches = self.dwt97_batch_dispatches.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::Htj2k97CodeblockBatchAttempt => {
                self.htj2k97_codeblock_batch_attempts =
                    self.htj2k97_codeblock_batch_attempts.saturating_add(count);
            }
            DctToWaveletStageCounterEvent::Htj2k97CodeblockBatchDispatch => {
                self.htj2k97_codeblock_batch_dispatches = self
                    .htj2k97_codeblock_batch_dispatches
                    .saturating_add(count);
            }
        }
    }
}

/// Dispatch policy for optional transcode-stage accelerators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeStageDispatchMode {
    /// Treat unavailable or unsupported backend dispatch as an error.
    Explicit,
    /// Decline unavailable or unsupported backend dispatch with `Ok(None)` so
    /// callers can use the scalar fallback.
    Auto,
}

impl TranscodeStageDispatchMode {
    /// Whether this mode allows scalar fallback for recoverable backend
    /// declines.
    #[must_use]
    pub const fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Outcome for a job that the backend cannot serve because it is
    /// unavailable on the current host.
    #[doc(hidden)]
    pub const fn unavailable<T>(self) -> Result<Option<T>, TranscodeStageError> {
        match self {
            Self::Explicit => Err(TranscodeStageError::DeviceUnavailable),
            Self::Auto => Ok(None),
        }
    }

    /// Convert a backend dispatch error into the trait outcome for this mode.
    ///
    /// Auto mode recovers from backend-declared recoverable errors with
    /// `Ok(None)`; Explicit mode and hard errors propagate as
    /// [`TranscodeStageError`].
    #[doc(hidden)]
    pub fn recover<T, E>(
        self,
        error: E,
        is_recoverable: impl FnOnce(&E) -> bool,
    ) -> Result<Option<T>, TranscodeStageError>
    where
        E: Into<TranscodeStageError>,
    {
        if self.is_auto() && is_recoverable(&error) {
            Ok(None)
        } else {
            Err(error.into())
        }
    }
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

    /// Whether this accelerator wants the compact i16 preencoded HTJ2K batch
    /// hook offered before the owned preencoded hook.
    fn supports_htj2k97_compact_preencoded_batch(&self) -> bool {
        self.supports_htj2k97_i16_preencoded_batch()
    }

    /// Optionally compute the direct DCT-grid to one-level reversible integer
    /// 5/3 projection.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job bit-exactly
    /// relative to j2k's scalar integer oracle. Return `Ok(None)` to use
    /// the scalar fallback.
    fn dct_grid_to_reversible_dwt53(
        &mut self,
        _job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, TranscodeStageError> {
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
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, TranscodeStageError> {
        Ok(None)
    }

    /// Optionally compute the direct DCT-grid to one-level 5/3 projection.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job. Return
    /// `Ok(None)` to use the scalar fallback.
    fn dct_grid_to_dwt53(
        &mut self,
        _job: DctGridToDwt53Job<'_>,
    ) -> Result<Option<Dwt53TwoDimensional<f64>>, TranscodeStageError> {
        Ok(None)
    }

    /// Optionally compute the direct DCT-grid to one-level 9/7 transform.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job. Return
    /// `Ok(None)` to use the scalar fallback.
    fn dct_grid_to_dwt97(
        &mut self,
        _job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, TranscodeStageError> {
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
    ) -> Result<Option<Vec<Dwt97TwoDimensional<f64>>>, TranscodeStageError> {
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
    ) -> Result<Option<Vec<PrequantizedHtj2k97Component>>, TranscodeStageError> {
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
    ) -> Result<Option<Vec<PreencodedHtj2k97Component>>, TranscodeStageError> {
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
    ) -> Result<Option<Vec<PreencodedHtj2k97Component>>, TranscodeStageError> {
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
    ) -> Result<Option<PreencodedHtj2k97CompactBatch>, TranscodeStageError> {
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
    ) -> Result<Option<Vec<Vec<PreencodedHtj2k97Component>>>, TranscodeStageError> {
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
    ) -> Result<Option<PreencodedHtj2k97CompactBatchGroups>, TranscodeStageError> {
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

#[doc(hidden)]
impl DctToWaveletStageAccelerator for CpuOnlyDctToWaveletStageAccelerator {}

/// CPU/Rayon accelerator for the exact reversible integer 5/3 first level.
///
/// This backend keeps j2k's scalar ISLOW IDCT semantics as the oracle:
/// each 8x8 block is decoded with `j2k-jpeg`, level-shifted to signed
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

#[doc(hidden)]
impl DctToWaveletStageAccelerator for RayonReversibleDwt53Accelerator {
    fn dct_grid_to_reversible_dwt53(
        &mut self,
        job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, TranscodeStageError> {
        self.attempts = self.attempts.saturating_add(1);
        let output = reversible_dwt53_first_level_rayon(job)?;
        self.dispatches = self.dispatches.saturating_add(1);
        Ok(Some(output))
    }

    fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, TranscodeStageError> {
        self.batch_attempts = self.batch_attempts.saturating_add(1);
        validate_reversible_batch_workspace(jobs)?;
        let mut output = try_vec_with_capacity(jobs.len()).map_err(TranscodeStageError::from)?;
        for job in jobs {
            output.push(reversible_dwt53_first_level_rayon(*job)?);
        }
        self.batch_dispatches = self.batch_dispatches.saturating_add(1);
        Ok(Some(output))
    }
}

/// Decode the job's dequantized DCT blocks into j2k's signed integer
/// component sample blocks.
///
/// This is source-visible so hybrid GPU backends can keep JPEG parsing and
/// exact IDCT on CPU while offloading the reversible 5/3 projection.
#[doc(hidden)]
pub fn idct_blocks_to_signed_samples_rayon(
    blocks: &[[i16; 64]],
) -> Result<Vec<[i32; 64]>, TranscodeStageError> {
    let mut output = try_vec_filled(blocks.len(), [0i32; 64]).map_err(TranscodeStageError::from)?;
    output
        .par_iter_mut()
        .zip(blocks.par_iter())
        .for_each(|(output, block)| {
            let decoded = idct_islow_block(block);
            *output = decoded.map(|sample| i32::from(sample) - 128);
        });
    Ok(output)
}

/// Compute one exact reversible integer 5/3 level from already decoded
/// block-local signed samples.
pub(crate) fn reversible_dwt53_first_level_from_block_samples(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<ReversibleDwt53FirstLevel, TranscodeStageError> {
    validate_reversible_grid(block_samples.len(), block_cols, block_rows, width, height)?;
    validate_reversible_output_workspace(width, height)?;

    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    let low_row_count = checked_stage_product(width, low_height)?;
    let mut low_rows = try_vec_filled(low_row_count, 0i32).map_err(TranscodeStageError::from)?;
    low_rows
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(output_y, row)| {
            for (x, sample) in row.iter_mut().enumerate() {
                *sample =
                    vertical_low_53_i32_at(block_samples, block_cols, width, height, x, output_y);
            }
            reversible_lift_53_i32(row);
        });
    let high_row_count = checked_stage_product(width, high_height)?;
    let mut high_rows = try_vec_filled(high_row_count, 0i32).map_err(TranscodeStageError::from)?;
    high_rows
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(output_y, row)| {
            for (x, sample) in row.iter_mut().enumerate() {
                *sample =
                    vertical_high_53_i32_at(block_samples, block_cols, width, height, x, output_y);
            }
            reversible_lift_53_i32(row);
        });

    let mut ll = try_vec_with_capacity(checked_stage_product(low_width, low_height)?)
        .map_err(TranscodeStageError::from)?;
    let mut hl = try_vec_with_capacity(checked_stage_product(high_width, low_height)?)
        .map_err(TranscodeStageError::from)?;
    for row in low_rows.chunks_exact(width) {
        ll.extend(row.iter().step_by(2).copied());
        hl.extend(row.iter().skip(1).step_by(2).copied());
    }

    let mut lh = try_vec_with_capacity(checked_stage_product(low_width, high_height)?)
        .map_err(TranscodeStageError::from)?;
    let mut hh = try_vec_with_capacity(checked_stage_product(high_width, high_height)?)
        .map_err(TranscodeStageError::from)?;
    for row in high_rows.chunks_exact(width) {
        lh.extend(row.iter().step_by(2).copied());
        hh.extend(row.iter().skip(1).step_by(2).copied());
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
) -> Result<ReversibleDwt53FirstLevel, TranscodeStageError> {
    validate_reversible_grid(
        job.dequantized_blocks.len(),
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
    )?;
    validate_reversible_job_workspace(job)?;
    let block_samples = idct_blocks_to_signed_samples_rayon(job.dequantized_blocks)?;
    reversible_dwt53_first_level_from_block_samples(
        &block_samples,
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
    )
}

fn validate_reversible_output_workspace(
    width: usize,
    height: usize,
) -> Result<(), TranscodeStageError> {
    let sample_count = checked_stage_product(width, height)?;
    let row_bytes = checked_allocation_bytes::<i32>(sample_count)?;
    let band_bytes = checked_allocation_bytes::<i32>(sample_count)?;
    checked_add_allocation_bytes(row_bytes, band_bytes)
        .map(|_| ())
        .map_err(TranscodeStageError::from)
}

fn validate_reversible_job_workspace(
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<(), TranscodeStageError> {
    let block_bytes = checked_allocation_bytes::<[i32; 64]>(job.dequantized_blocks.len())?;
    let sample_count = checked_stage_product(job.width, job.height)?;
    let row_bytes = checked_allocation_bytes::<i32>(sample_count)?;
    let band_bytes = checked_allocation_bytes::<i32>(sample_count)?;
    let workspace = checked_add_allocation_bytes(block_bytes, row_bytes)?;
    checked_add_allocation_bytes(workspace, band_bytes)?;
    Ok(())
}

fn validate_reversible_batch_workspace(
    jobs: &[DctGridToReversibleDwt53Job<'_>],
) -> Result<(), TranscodeStageError> {
    let mut retained_output_bytes = 0usize;
    let mut max_transient_bytes = 0usize;
    for job in jobs {
        validate_reversible_grid(
            job.dequantized_blocks.len(),
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
        )?;
        let sample_count = checked_stage_product(job.width, job.height)?;
        let output_bytes = checked_allocation_bytes::<i32>(sample_count)?;
        retained_output_bytes = checked_add_allocation_bytes(retained_output_bytes, output_bytes)?;
        let block_bytes = checked_allocation_bytes::<[i32; 64]>(job.dequantized_blocks.len())?;
        let row_bytes = checked_allocation_bytes::<i32>(sample_count)?;
        max_transient_bytes =
            max_transient_bytes.max(checked_add_allocation_bytes(block_bytes, row_bytes)?);
    }
    checked_add_allocation_bytes(retained_output_bytes, max_transient_bytes)?;
    Ok(())
}

fn checked_stage_product(left: usize, right: usize) -> Result<usize, TranscodeStageError> {
    left.checked_mul(right)
        .ok_or(TranscodeStageError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })
}

fn validate_reversible_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<(), TranscodeStageError> {
    validate_dct_block_grid(block_count, block_cols, block_rows, width, height)
        .map_err(|_| TranscodeStageError::Unsupported(REVERSIBLE_DWT53_UNSUPPORTED_GRID))
}

fn vertical_low_53_i32_at(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    width: usize,
    height: usize,
    x: usize,
    low_idx: usize,
) -> i32 {
    reversible_lift_53_low_at(height, low_idx, |y| {
        component_sample_i32(block_samples, block_cols, width, height, x, y)
    })
}

fn vertical_high_53_i32_at(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    width: usize,
    height: usize,
    x: usize,
    high_idx: usize,
) -> i32 {
    reversible_lift_53_high_at(height, high_idx, |y| {
        component_sample_i32(block_samples, block_cols, width, height, x, y)
    })
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

#[cfg(test)]
mod allocation_tests {
    use super::{
        idct_blocks_to_signed_samples_rayon, validate_reversible_grid,
        validate_reversible_output_workspace, TranscodeStageError,
        REVERSIBLE_DWT53_UNSUPPORTED_GRID,
    };

    #[test]
    fn malformed_reversible_grid_is_explicitly_unsupported() {
        assert!(matches!(
            validate_reversible_grid(0, 1, 1, 8, 8),
            Err(TranscodeStageError::Unsupported(
                REVERSIBLE_DWT53_UNSUPPORTED_GRID
            ))
        ));
    }

    #[test]
    fn reversible_workspace_overflow_is_typed() {
        assert!(matches!(
            validate_reversible_output_workspace(usize::MAX, 2),
            Err(TranscodeStageError::MemoryCapExceeded {
                requested: usize::MAX,
                ..
            })
        ));
    }

    #[test]
    fn fallible_parallel_idct_preserves_signed_samples() {
        let blocks = [[0i16; 64]; 2];
        let samples = idct_blocks_to_signed_samples_rayon(&blocks)
            .expect("two block outputs fit the host cap");
        assert_eq!(samples, [[0i32; 64]; 2]);
    }
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
    fn reversible_lift_53_shared_helper_matches_canonical_formula_1d() {
        let mut state = 0x5a53_5a53_5a53_5a53u64;
        for n in [2usize, 3, 4, 5, 8, 9, 16, 17, 31, 32, 65] {
            let signal: Vec<i32> = (0..n).map(|_| next_sample(&mut state)).collect();
            let mut lifted = signal.clone();
            crate::reversible53::reversible_lift_53_i32(&mut lifted);
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
