// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    decode_ht_code_block_scalar, J2kCodeBlockSegment, J2kCodeBlockStyle, J2kSubBandType, Result,
};

define_ht_code_block_job! {
    /// Adapter HTJ2K code-block job description for backend experimentation.
    #[derive(Debug, Clone, Copy)]
    pub struct HtCodeBlockDecodeJob<'a> {
        /// Combined cleanup/refinement bytes for the code block.
        pub data: &'a [u8],
    }
}

/// Adapter HTJ2K scalar decode phase limit for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtCodeBlockDecodePhaseLimit {
    /// Stop after the cleanup pass has produced coefficient magnitudes/signs.
    Cleanup,
    /// Stop after the significance propagation refinement pass.
    SignificancePropagation,
    /// Decode through the magnitude refinement pass when present.
    MagnitudeRefinement,
}

/// Adapter HTJ2K batched code-block decode job for one sub-band.
#[derive(Debug, Clone, Copy)]
pub struct HtCodeBlockBatchJob<'a> {
    /// X offset within the target sub-band coefficient buffer.
    pub output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub output_y: u32,
    /// The actual code-block decode parameters.
    pub code_block: HtCodeBlockDecodeJob<'a>,
}

/// Adapter HTJ2K batched sub-band decode request for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct HtSubBandDecodeJob<'a> {
    /// Sub-band width in samples.
    pub width: u32,
    /// Sub-band height in samples.
    pub height: u32,
    /// Code blocks to decode into this sub-band.
    pub jobs: &'a [HtCodeBlockBatchJob<'a>],
}

/// Adapter Classic Tier-1 compact token segment for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kTier1TokenSegment {
    /// Bit offset of this segment within the compact token buffer.
    pub token_bit_offset: u32,
    /// Number of token bits in this segment.
    ///
    /// Arithmetic segments contain 6-bit MQ tokens. Raw bypass segments contain
    /// one bit per raw bypass event.
    pub token_bit_count: u32,
    /// First coding pass covered by this segment.
    pub start_coding_pass: u8,
    /// One-past-last coding pass covered by this segment.
    pub end_coding_pass: u8,
    /// Whether this segment should be packed through the MQ arithmetic path.
    pub use_arithmetic: bool,
}

/// Adapter classic J2K code-block job description for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kCodeBlockDecodeJob<'a> {
    /// Combined payload bytes for all coded segments in this code block.
    pub data: &'a [u8],
    /// Coded segments for the code block.
    pub segments: &'a [J2kCodeBlockSegment],
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Output row stride, in samples, for the target sub-band storage.
    pub output_stride: usize,
    /// Missing most-significant bit planes for this code block.
    pub missing_bit_planes: u8,
    /// Number of coding passes present for this code block.
    pub number_of_coding_passes: u8,
    /// Total coded bitplanes for the parent sub-band.
    pub total_bitplanes: u8,
    /// Region-of-interest maxshift value from RGN marker metadata.
    pub roi_shift: u8,
    /// The sub-band type containing this code block.
    pub sub_band_type: J2kSubBandType,
    /// The code-block style flags.
    pub style: J2kCodeBlockStyle,
    /// Whether strict decode validation is enabled for the parent image.
    pub strict: bool,
    /// Dequantization step to apply to decoded coefficients.
    pub dequantization_step: f32,
}

/// Adapter HTJ2K cleanup-encode shape counters for backend benchmarking.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HtCleanupEncodeDistribution {
    /// Total 2x2 cleanup quads visited.
    pub total_quads: u64,
    /// Quads encoded in the first cleanup row pair.
    pub initial_quads: u64,
    /// Quads encoded after the first cleanup row pair.
    pub non_initial_quads: u64,
    /// All-quad `rho` histogram, indexed by the low four `rho` bits.
    pub rho_counts: [u64; 16],
    /// First-row-pair `rho` histogram, indexed by the low four `rho` bits.
    pub initial_rho_counts: [u64; 16],
    /// Non-initial-row `rho` histogram, indexed by the low four `rho` bits.
    pub non_initial_rho_counts: [u64; 16],
    /// Non-initial-row `u_q` histogram.
    pub non_initial_u_q_counts: [u64; 32],
    /// Non-initial-row `e_qmax` histogram.
    pub non_initial_e_qmax_counts: [u64; 32],
    /// Non-initial-row `kappa` histogram.
    pub non_initial_kappa_counts: [u64; 32],
    /// Non-initial-row joint `rho`/`u_q` histogram.
    pub non_initial_rho_u_q_counts: [[u64; 32]; 16],
    /// Calls that emitted at least one magnitude/sign sample.
    pub mag_sign_calls: u64,
    /// Magnitude/sign call histogram, indexed by the low four `rho` bits.
    pub mag_sign_rho_counts: [u64; 16],
    /// Encoded magnitude/sign sample payload bit-count histogram.
    pub mag_sign_sample_bit_counts: [u64; 32],
    /// Number of individual magnitude/sign samples emitted.
    pub mag_sign_encoded_samples: u64,
}

/// Adapter classic J2K batched code-block decode job for one sub-band.
#[derive(Debug, Clone, Copy)]
pub struct J2kCodeBlockBatchJob<'a> {
    /// X offset within the target sub-band coefficient buffer.
    pub output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub output_y: u32,
    /// The actual code-block decode parameters.
    pub code_block: J2kCodeBlockDecodeJob<'a>,
}

/// Adapter classic J2K batched sub-band decode request for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kSubBandDecodeJob<'a> {
    /// Sub-band width in samples.
    pub width: u32,
    /// Sub-band height in samples.
    pub height: u32,
    /// Code blocks to decode into this sub-band.
    pub jobs: &'a [J2kCodeBlockBatchJob<'a>],
}

/// Adapter integer rectangle for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kRect {
    /// Inclusive minimum x coordinate.
    pub x0: u32,
    /// Inclusive minimum y coordinate.
    pub y0: u32,
    /// Exclusive maximum x coordinate.
    pub x1: u32,
    /// Exclusive maximum y coordinate.
    pub y1: u32,
}

impl J2kRect {
    /// Rectangle width in samples.
    #[must_use]
    pub fn width(self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    /// Rectangle height in samples.
    #[must_use]
    pub fn height(self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }
}

/// Adapter wavelet transform selector for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kWaveletTransform {
    /// Reversible 5/3 transform.
    Reversible53,
    /// Irreversible 9/7 transform.
    Irreversible97,
}

/// Adapter single sub-band payload for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kIdwtBand<'a> {
    /// Rect covered by this band.
    pub rect: J2kRect,
    /// Band coefficients in row-major order.
    pub coefficients: &'a [f32],
}

/// Adapter single-decomposition IDWT job for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kSingleDecompositionIdwtJob<'a> {
    /// Output rect of the decomposition level.
    pub rect: J2kRect,
    /// Transform to apply.
    pub transform: J2kWaveletTransform,
    /// LL band input.
    pub ll: J2kIdwtBand<'a>,
    /// HL band input.
    pub hl: J2kIdwtBand<'a>,
    /// LH band input.
    pub lh: J2kIdwtBand<'a>,
    /// HH band input.
    pub hh: J2kIdwtBand<'a>,
}

/// Adapter inverse MCT job for backend experimentation.
#[derive(Debug)]
pub struct J2kInverseMctJob<'a> {
    /// Transform to apply.
    pub transform: J2kWaveletTransform,
    /// First component plane, updated in place.
    pub plane0: &'a mut [f32],
    /// Second component plane, updated in place.
    pub plane1: &'a mut [f32],
    /// Third component plane, updated in place.
    pub plane2: &'a mut [f32],
    /// Constant sign-shift addend applied to the first plane after inverse MCT.
    pub addend0: f32,
    /// Constant sign-shift addend applied to the second plane after inverse MCT.
    pub addend1: f32,
    /// Constant sign-shift addend applied to the third plane after inverse MCT.
    pub addend2: f32,
}

/// Adapter component-store job for backend experimentation.
#[derive(Debug)]
pub struct J2kStoreComponentJob<'a> {
    /// Source IDWT coefficients in row-major order.
    pub input: &'a [f32],
    /// Source row width.
    pub input_width: u32,
    /// Source x offset to begin copying from.
    pub source_x: u32,
    /// Source y offset to begin copying from.
    pub source_y: u32,
    /// Number of samples to copy per row.
    pub copy_width: u32,
    /// Number of rows to copy.
    pub copy_height: u32,
    /// Destination component plane in row-major order.
    pub output: &'a mut [f32],
    /// Destination row width.
    pub output_width: u32,
    /// Destination x offset to begin writing at.
    pub output_x: u32,
    /// Destination y offset to begin writing at.
    pub output_y: u32,
    /// Constant value added to every copied sample.
    pub addend: f32,
}

/// Adapter HTJ2K code-block decode hook for backend experimentation.
pub trait HtCodeBlockDecoder {
    /// Optionally decode a full classic J2K sub-band in one batch.
    ///
    /// Implementations should return `Ok(true)` if they handled the request and
    /// wrote the decoded coefficients into `output`. Returning `Ok(false)`
    /// falls back to per-code-block decode via `decode_j2k_code_block`.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot complete the decode request.
    fn decode_j2k_sub_band(
        &mut self,
        _job: J2kSubBandDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally decode one classic J2K code block.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and wrote the decoded coefficients into `output`. Returning `Ok(false)`
    /// falls back to the scalar bitplane decoder.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot complete the decode request.
    fn decode_j2k_code_block(
        &mut self,
        _job: J2kCodeBlockDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally decode a full HTJ2K sub-band in one batch.
    ///
    /// Implementations should return `Ok(true)` if they handled the request and
    /// wrote the decoded coefficients into `output`. Returning `Ok(false)`
    /// falls back to per-code-block decode via `decode_code_block`.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot complete the decode request.
    fn decode_sub_band(
        &mut self,
        _job: HtSubBandDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally decode one single-decomposition IDWT level on a backend.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and wrote the transformed coefficients into `output`. Returning
    /// `Ok(false)` falls back to the scalar/SIMD CPU IDWT path.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot complete the transform request.
    fn decode_single_decomposition_idwt(
        &mut self,
        _job: J2kSingleDecompositionIdwtJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally apply inverse MCT on a backend.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and updated the component planes in place. Returning `Ok(false)` falls
    /// back to the scalar/SIMD CPU MCT path.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot complete the transform request.
    fn decode_inverse_mct(&mut self, _job: J2kInverseMctJob<'_>) -> Result<bool> {
        Ok(false)
    }

    /// Optionally store one component plane on a backend.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and updated the destination plane in place. Returning `Ok(false)` falls
    /// back to the CPU store path.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot complete the store request.
    fn decode_store_component(&mut self, _job: J2kStoreComponentJob<'_>) -> Result<bool> {
        Ok(false)
    }

    /// Decode one HTJ2K code block into `output`, writing `job.width` samples per row.
    ///
    /// # Errors
    ///
    /// Returns an error when the code block is malformed or decoding fails.
    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        decode_ht_code_block_scalar(job, output)
    }
}
