// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_roi_shift_to_bitplanes, checked_code_block_output_layout, j2c, CodeBlockOutputLayout,
    HtCodeBlockDecodeJob, HtCodeBlockDecodePhaseLimit, Result, Vec,
};
use crate::try_reserve_decode_elements;

/// Adapter scalar HTJ2K decoder helper for backend experimentation.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase::<{ j2c::ht_block_decode::PHASE_LIMIT_MAGREF }>(
        job, output,
    )
}

/// Adapter scalar HTJ2K decoder helper that stops after the selected phase.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar_until_phase(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    phase_limit: HtCodeBlockDecodePhaseLimit,
) -> Result<()> {
    match phase_limit {
        HtCodeBlockDecodePhaseLimit::Cleanup => decode_ht_code_block_scalar_for_phase::<
            { j2c::ht_block_decode::PHASE_LIMIT_CLEANUP },
        >(job, output),
        HtCodeBlockDecodePhaseLimit::SignificancePropagation => {
            decode_ht_code_block_scalar_for_phase::<{ j2c::ht_block_decode::PHASE_LIMIT_SIGPROP }>(
                job, output,
            )
        }
        HtCodeBlockDecodePhaseLimit::MagnitudeRefinement => {
            decode_ht_code_block_scalar_for_phase::<{ j2c::ht_block_decode::PHASE_LIMIT_MAGREF }>(
                job, output,
            )
        }
    }
}

/// Adapter reusable scalar HTJ2K decode workspace for backend experimentation.
#[derive(Debug, Default)]
#[doc(hidden)]
pub struct HtCodeBlockDecodeWorkspace {
    coefficients: Vec<u32>,
    scratch: j2c::ht_block_decode::HtBlockDecodeScratch,
}

impl HtCodeBlockDecodeWorkspace {
    pub(crate) const fn empty() -> Self {
        Self {
            coefficients: Vec::new(),
            scratch: j2c::ht_block_decode::HtBlockDecodeScratch::empty(),
        }
    }

    /// Current coefficient buffer capacity retained by this workspace.
    #[must_use]
    pub fn coefficient_capacity(&self) -> usize {
        self.coefficients.capacity()
    }

    // Keep fallible allocation and actual-capacity reconciliation on the
    // caller thread; parallel decode may initialize this owner only afterward.
    pub(crate) fn reserve(&mut self, width: u32, height: u32) -> Result<()> {
        let coefficient_count = (width as usize)
            .checked_mul(height as usize)
            .ok_or(crate::ValidationError::ImageTooLarge)?;
        try_reserve_decode_elements(&mut self.coefficients, coefficient_count)?;
        self.scratch.prepare(width, height)
    }

    pub(crate) fn initialize_reserved(&mut self, width: u32, height: u32) -> Result<()> {
        let coefficient_count = (width as usize)
            .checked_mul(height as usize)
            .ok_or(crate::ValidationError::ImageTooLarge)?;
        if self.coefficients.capacity() < coefficient_count {
            return Err(crate::DecodingError::CodeBlockDecodeFailure.into());
        }
        self.coefficients.clear();
        self.coefficients.resize(coefficient_count, 0);
        Ok(())
    }

    pub(crate) fn prepare(&mut self, width: u32, height: u32) -> Result<()> {
        self.reserve(width, height)?;
        self.initialize_reserved(width, height)
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize> {
        let coefficient_bytes = self
            .coefficients
            .capacity()
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(crate::ValidationError::ImageTooLarge)?;
        coefficient_bytes
            .checked_add(self.scratch.allocated_bytes()?)
            .ok_or(crate::ValidationError::ImageTooLarge.into())
    }
}

/// Adapter scalar HTJ2K phase timings for backend experimentation.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub struct HtCodeBlockDecodeProfile {
    /// Number of decoded HT code blocks.
    pub blocks: u128,
    /// Number of decoded HT code blocks with refinement data.
    pub refinement_blocks: u128,
    /// Total cleanup segment bytes consumed by decoded HT code blocks.
    pub cleanup_bytes: u128,
    /// Total refinement segment bytes consumed by decoded HT code blocks.
    pub refinement_bytes: u128,
    /// Cleanup phase elapsed time in microseconds.
    pub cleanup_us: u128,
    /// Magnitude/sign phase elapsed time in microseconds.
    pub mag_sgn_us: u128,
    /// Sigma build phase elapsed time in microseconds.
    pub sigma_us: u128,
    /// Significance propagation phase elapsed time in microseconds.
    pub sigprop_us: u128,
    /// Magnitude refinement phase elapsed time in microseconds.
    pub magref_us: u128,
}

/// Whole-image cleanup-only fast path. Unlike the public scalar helpers, this
/// may publish a prefix before malformed entropy is rejected; callers discard
/// their internal coefficient plane when the image decode returns an error.
#[cfg(feature = "simd")]
pub(crate) fn try_decode_ht_code_block_to_f32(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    irreversible_midpoint: bool,
) -> Result<bool> {
    if job.number_of_coding_passes != 1
        || job.roi_shift != 0
        || !j2c::ht_block_decode::supports_cleanup_to_f32(job.missing_bit_planes, job.width)
    {
        return Ok(false);
    }

    let _layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let segments = j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        job.data,
        job.cleanup_length,
        job.refinement_length,
    )?;
    if !segments.refinement.is_empty() {
        return Ok(false);
    }
    let coded_bitplanes = add_roi_shift_to_bitplanes(job.num_bitplanes, job.roi_shift, 31)?;
    let stride = u32::try_from(job.output_stride)
        .map_err(|_| crate::DecodingError::CodeBlockDecodeFailure)?;
    if irreversible_midpoint {
        j2c::ht_block_decode::decode_cleanup_validated_to_f32::<true>(
            segments.cleanup,
            job.missing_bit_planes,
            coded_bitplanes,
            job.strict,
            output,
            job.width,
            job.height,
            stride,
            job.dequantization_step,
            &mut workspace.scratch,
        )
    } else {
        j2c::ht_block_decode::decode_cleanup_validated_to_f32::<false>(
            segments.cleanup,
            job.missing_bit_planes,
            coded_bitplanes,
            job.strict,
            output,
            job.width,
            job.height,
            stride,
            job.dequantization_step,
            &mut workspace.scratch,
        )
    }
}

impl HtCodeBlockDecodeProfile {
    /// Create an empty profile accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn add_native_stats(&mut self, stats: j2c::ht_block_decode::HtBlockDecodeStats) {
        self.blocks += stats.blocks;
        self.refinement_blocks += stats.refinement_blocks;
        self.cleanup_bytes += stats.cleanup_bytes;
        self.refinement_bytes += stats.refinement_bytes;
        self.cleanup_us += stats.ht_cleanup_us;
        self.mag_sgn_us += stats.ht_mag_sgn_us;
        self.sigma_us += stats.ht_sigma_us;
        self.sigprop_us += stats.ht_sigprop_us;
        self.magref_us += stats.ht_magref_us;
    }
}

/// Adapter scalar HTJ2K decoder helper that reuses caller-owned scratch buffers.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar_with_workspace(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase_with_workspace_inner::<
        { j2c::ht_block_decode::PHASE_LIMIT_MAGREF },
    >(job, output, workspace, false)
}

/// Adapter scalar HTJ2K decoder helper using irreversible midpoint reconstruction.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar_with_workspace_midpoint(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase_with_workspace_inner::<
        { j2c::ht_block_decode::PHASE_LIMIT_MAGREF },
    >(job, output, workspace, true)
}

/// Adapter scalar HTJ2K decoder helper that reuses scratch and records phase timings.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar_with_workspace_profiled(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile: &mut HtCodeBlockDecodeProfile,
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase_with_workspace_profiled::<
        { j2c::ht_block_decode::PHASE_LIMIT_MAGREF },
    >(job, output, workspace, profile, false)
}

/// Adapter scalar HTJ2K decoder helper using irreversible midpoint
/// reconstruction while recording phase timings.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar_with_workspace_midpoint_profiled(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile: &mut HtCodeBlockDecodeProfile,
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase_with_workspace_profiled::<
        { j2c::ht_block_decode::PHASE_LIMIT_MAGREF },
    >(job, output, workspace, profile, true)
}

fn decode_ht_code_block_scalar_for_phase<const PHASE_LIMIT: u8>(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    let mut workspace = HtCodeBlockDecodeWorkspace::default();
    decode_ht_code_block_scalar_for_phase_with_workspace::<PHASE_LIMIT>(job, output, &mut workspace)
}

fn decode_ht_code_block_scalar_for_phase_with_workspace<const PHASE_LIMIT: u8>(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase_with_workspace_inner::<PHASE_LIMIT>(
        job, output, workspace, false,
    )
}

fn decode_ht_code_block_scalar_for_phase_with_workspace_inner<const PHASE_LIMIT: u8>(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    irreversible_midpoint: bool,
) -> Result<()> {
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let segments = j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        job.data,
        job.cleanup_length,
        job.refinement_length,
    )?;
    let coded_bitplanes = add_roi_shift_to_bitplanes(job.num_bitplanes, job.roi_shift, 31)?;
    workspace.prepare(job.width, job.height)?;
    j2c::ht_block_decode::decode_segments_validated_with_scratch_for_phase::<PHASE_LIMIT>(
        &segments,
        job.missing_bit_planes,
        coded_bitplanes,
        job.number_of_coding_passes,
        job.stripe_causal,
        job.strict,
        &mut workspace.coefficients,
        job.width,
        job.height,
        job.width,
        &mut workspace.scratch,
        None,
        false,
    )?;

    write_ht_code_block_output(
        &workspace.coefficients,
        job,
        layout,
        coded_bitplanes,
        output,
        irreversible_midpoint,
    );

    Ok(())
}

fn decode_ht_code_block_scalar_for_phase_with_workspace_profiled<const PHASE_LIMIT: u8>(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile: &mut HtCodeBlockDecodeProfile,
    irreversible_midpoint: bool,
) -> Result<()> {
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let segments = j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        job.data,
        job.cleanup_length,
        job.refinement_length,
    )?;
    let coded_bitplanes = add_roi_shift_to_bitplanes(job.num_bitplanes, job.roi_shift, 31)?;
    workspace.prepare(job.width, job.height)?;
    let mut stats = j2c::ht_block_decode::HtBlockDecodeStats::default();
    j2c::ht_block_decode::decode_segments_validated_with_scratch_for_phase::<PHASE_LIMIT>(
        &segments,
        job.missing_bit_planes,
        coded_bitplanes,
        job.number_of_coding_passes,
        job.stripe_causal,
        job.strict,
        &mut workspace.coefficients,
        job.width,
        job.height,
        job.width,
        &mut workspace.scratch,
        Some(&mut stats),
        true,
    )?;
    profile.add_native_stats(stats);

    write_ht_code_block_output(
        &workspace.coefficients,
        job,
        layout,
        coded_bitplanes,
        output,
        irreversible_midpoint,
    );

    Ok(())
}

fn write_ht_code_block_output(
    coefficients: &[u32],
    job: HtCodeBlockDecodeJob<'_>,
    layout: CodeBlockOutputLayout,
    coded_bitplanes: u8,
    output: &mut [f32],
    irreversible_midpoint: bool,
) {
    for (row_idx, coeff_row) in coefficients
        .chunks_exact(layout.stride)
        .enumerate()
        .take(job.height as usize)
    {
        let row_start = row_idx * job.output_stride;
        let output_row = &mut output[row_start..row_start + layout.stride];
        j2c::ht_block_decode::store_coefficient_row(
            coeff_row,
            output_row,
            coded_bitplanes,
            job.roi_shift,
            irreversible_midpoint,
            job.dequantization_step,
        );
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "simd")]
    use super::try_decode_ht_code_block_to_f32;
    use super::{
        decode_ht_code_block_scalar_with_workspace,
        decode_ht_code_block_scalar_with_workspace_midpoint, write_ht_code_block_output,
        CodeBlockOutputLayout, HtCodeBlockDecodeJob, HtCodeBlockDecodeWorkspace,
    };
    #[cfg(feature = "simd")]
    use crate::encode_ht_code_block_scalar;

    #[test]
    fn irreversible_output_retains_the_ht_midpoint_before_dequantization() {
        let job = HtCodeBlockDecodeJob {
            data: &[],
            cleanup_length: 0,
            refinement_length: 0,
            width: 1,
            height: 1,
            output_stride: 1,
            missing_bit_planes: 0,
            number_of_coding_passes: 1,
            num_bitplanes: 5,
            roi_shift: 0,
            stripe_causal: false,
            strict: true,
            dequantization_step: 2.0,
        };
        let coefficients = [3_u32 << 25];
        let layout = CodeBlockOutputLayout { stride: 1 };
        let mut reversible = [0.0];
        let mut irreversible = [0.0];

        write_ht_code_block_output(&coefficients, job, layout, 5, &mut reversible, false);
        write_ht_code_block_output(&coefficients, job, layout, 5, &mut irreversible, true);

        assert_eq!(reversible.map(f32::to_bits), [2.0_f32.to_bits()]);
        assert_eq!(irreversible.map(f32::to_bits), [3.0_f32.to_bits()]);
    }

    #[test]
    fn reserved_workspace_initialization_does_not_grow_allocations() {
        let mut workspace = HtCodeBlockDecodeWorkspace::default();
        workspace
            .reserve(64, 64)
            .expect("workspace reservation should succeed");
        assert_eq!(workspace.coefficients.len(), 0);
        let reserved_bytes = workspace
            .allocated_bytes()
            .expect("reserved workspace bytes should be measurable");

        workspace
            .initialize_reserved(64, 64)
            .expect("reserved workspace should initialize");

        assert_eq!(workspace.coefficients.len(), 64 * 64);
        assert_eq!(
            workspace
                .allocated_bytes()
                .expect("initialized workspace bytes should be measurable"),
            reserved_bytes,
            "parallel initialization must not grow beyond serially accounted capacity"
        );
    }

    #[test]
    fn unreserved_workspace_initialization_fails_without_allocating() {
        let mut workspace = HtCodeBlockDecodeWorkspace::default();

        workspace
            .initialize_reserved(64, 64)
            .expect_err("initialization must not allocate an unreserved workspace");

        assert_eq!(workspace.coefficient_capacity(), 0);
        assert_eq!(workspace.coefficients.len(), 0);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn direct_cleanup_output_matches_transactional_decode_across_precisions() {
        let width = 8u32;
        let height = 5u32;
        let stride = 11usize;
        let output_len = stride * (height as usize - 1) + width as usize + 7;

        for total_bitplanes in 1_u8..=31 {
            // The current cleanup encoder selects at most the third highest
            // cleanup bitplane. Reuse a supported 24-bit payload above that
            // point while varying the decoder's final coefficient precision.
            let encoded_bitplanes = total_bitplanes.min(24);
            let limit = 1_i64 << encoded_bitplanes;
            let mut coefficients: Vec<i32> = (0..width * height)
                .map(|index| {
                    let magnitude = (i64::from(index) * 104_729 + 17) % limit.max(1);
                    let value = if index & 1 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    };
                    i32::try_from(value).expect("test coefficient fits i32")
                })
                .collect();
            coefficients[0] = i32::try_from(limit - 1).expect("top magnitude fits i32");
            let encoded =
                encode_ht_code_block_scalar(&coefficients, width, height, encoded_bitplanes)
                    .expect("encode cleanup-only block");
            assert_eq!(encoded.num_coding_passes, 1);
            let job = HtCodeBlockDecodeJob {
                data: &encoded.data,
                cleanup_length: encoded.cleanup_length,
                refinement_length: encoded.refinement_length,
                width,
                height,
                output_stride: stride,
                missing_bit_planes: encoded.num_zero_bitplanes,
                number_of_coding_passes: encoded.num_coding_passes,
                num_bitplanes: total_bitplanes,
                roi_shift: 0,
                stripe_causal: false,
                strict: true,
                dequantization_step: 0.375,
            };

            for midpoint in [false, true] {
                let sentinel = f32::from_bits(0x7FC0_1234);
                let mut expected = vec![sentinel; output_len];
                let mut actual = expected.clone();
                let mut expected_workspace = HtCodeBlockDecodeWorkspace::default();
                let mut direct_workspace = HtCodeBlockDecodeWorkspace::default();
                if midpoint {
                    decode_ht_code_block_scalar_with_workspace_midpoint(
                        job,
                        &mut expected,
                        &mut expected_workspace,
                    )
                    .expect("transactional midpoint decode");
                } else {
                    decode_ht_code_block_scalar_with_workspace(
                        job,
                        &mut expected,
                        &mut expected_workspace,
                    )
                    .expect("transactional reversible decode");
                }
                assert!(
                    try_decode_ht_code_block_to_f32(
                        job,
                        &mut actual,
                        &mut direct_workspace,
                        midpoint,
                    )
                    .expect("direct decode"),
                    "total_bitplanes={total_bitplanes} midpoint={midpoint}"
                );
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "total_bitplanes={total_bitplanes} midpoint={midpoint}"
                );
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn direct_cleanup_rejects_fallbacks_and_preserves_output_guards_on_error() {
        let width = 8u32;
        let height = 4u32;
        let coefficients: Vec<i32> = (0..width * height)
            .map(|index| i32::try_from(index).expect("fixture index fits i32") % 31 - 15)
            .collect();
        let encoded = encode_ht_code_block_scalar(&coefficients, width, height, 6)
            .expect("encode cleanup-only block");
        let base_job = HtCodeBlockDecodeJob {
            data: &encoded.data,
            cleanup_length: encoded.cleanup_length,
            refinement_length: encoded.refinement_length,
            width,
            height,
            output_stride: width as usize,
            missing_bit_planes: encoded.num_zero_bitplanes,
            number_of_coding_passes: encoded.num_coding_passes,
            num_bitplanes: 6,
            roi_shift: 0,
            stripe_causal: false,
            strict: true,
            dequantization_step: 1.0,
        };
        let sentinel = f32::from_bits(0x7FC0_4321);

        for job in [
            HtCodeBlockDecodeJob {
                width: 7,
                ..base_job
            },
            HtCodeBlockDecodeJob {
                roi_shift: 1,
                ..base_job
            },
            HtCodeBlockDecodeJob {
                number_of_coding_passes: 2,
                ..base_job
            },
        ] {
            let mut output = vec![sentinel; width as usize * height as usize];
            assert!(!try_decode_ht_code_block_to_f32(
                job,
                &mut output,
                &mut HtCodeBlockDecodeWorkspace::default(),
                false,
            )
            .expect("unsupported direct decode falls back"));
            assert!(output
                .iter()
                .all(|value| value.to_bits() == sentinel.to_bits()));
        }

        let mut malformed = encoded.data.clone();
        let last = malformed.len() - 1;
        malformed[last] = 0xFF;
        malformed[last - 1] |= 0x0F;
        let malformed_job = HtCodeBlockDecodeJob {
            data: &malformed,
            cleanup_length: u32::try_from(malformed.len()).expect("payload length fits u32"),
            ..base_job
        };
        let guard = 5usize;
        let output_len = width as usize * height as usize;
        let mut guarded = vec![sentinel; guard + output_len + guard];
        let error = try_decode_ht_code_block_to_f32(
            malformed_job,
            &mut guarded[guard..guard + output_len],
            &mut HtCodeBlockDecodeWorkspace::default(),
            false,
        )
        .expect_err("malformed cleanup suffix must fail");
        assert_eq!(
            error,
            crate::DecodeError::Decoding(crate::DecodingError::CodeBlockDecodeFailure)
        );
        assert!(guarded[..guard]
            .iter()
            .chain(&guarded[guard + output_len..])
            .all(|value| value.to_bits() == sentinel.to_bits()));
    }
}
