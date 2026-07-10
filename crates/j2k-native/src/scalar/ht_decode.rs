// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_i32, checked_code_block_output_layout,
    j2c, CodeBlockOutputLayout, HtCodeBlockDecodeJob, HtCodeBlockDecodePhaseLimit, Result, Vec,
};

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
#[derive(Default)]
#[doc(hidden)]
pub struct HtCodeBlockDecodeWorkspace {
    coefficients: Vec<u32>,
    scratch: j2c::ht_block_decode::HtBlockDecodeScratch,
}

impl HtCodeBlockDecodeWorkspace {
    /// Current coefficient buffer capacity retained by this workspace.
    pub fn coefficient_capacity(&self) -> usize {
        self.coefficients.capacity()
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
    decode_ht_code_block_scalar_for_phase_with_workspace::<
        { j2c::ht_block_decode::PHASE_LIMIT_MAGREF },
    >(job, output, workspace)
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
    >(job, output, workspace, profile)
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
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let segments = j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        job.data,
        job.cleanup_length,
        job.refinement_length,
    )?;
    let coded_bitplanes = add_roi_shift_to_bitplanes(job.num_bitplanes, job.roi_shift, 31)?;
    workspace.coefficients.clear();
    workspace.coefficients.resize(layout.len, 0);
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
    );

    Ok(())
}

fn decode_ht_code_block_scalar_for_phase_with_workspace_profiled<const PHASE_LIMIT: u8>(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile: &mut HtCodeBlockDecodeProfile,
) -> Result<()> {
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let segments = j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        job.data,
        job.cleanup_length,
        job.refinement_length,
    )?;
    let coded_bitplanes = add_roi_shift_to_bitplanes(job.num_bitplanes, job.roi_shift, 31)?;
    workspace.coefficients.clear();
    workspace.coefficients.resize(layout.len, 0);
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
    );

    Ok(())
}

fn write_ht_code_block_output(
    coefficients: &[u32],
    job: HtCodeBlockDecodeJob<'_>,
    layout: CodeBlockOutputLayout,
    coded_bitplanes: u8,
    output: &mut [f32],
) {
    for (row_idx, coeff_row) in coefficients
        .chunks_exact(layout.stride)
        .enumerate()
        .take(job.height as usize)
    {
        let row_start = row_idx * job.output_stride;
        let output_row = &mut output[row_start..row_start + layout.stride];
        for (coefficient, sample) in coeff_row.iter().copied().zip(output_row.iter_mut()) {
            let coefficient =
                j2c::ht_block_decode::coefficient_to_i32(coefficient, coded_bitplanes);
            let coefficient = apply_roi_maxshift_inverse_i32(coefficient, job.roi_shift);
            *sample = coefficient as f32 * job.dequantization_step;
        }
    }
}
