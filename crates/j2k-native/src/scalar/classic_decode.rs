// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_i64, bail,
    internal_j2k_code_block_style, internal_j2k_sub_band_type, j2c, profile, DecodingError,
    J2kCodeBlockDecodeJob, J2kSubBandDecodeJob, Result, MAX_CLASSIC_DECODE_BITPLANES,
};

/// Adapter scalar classic J2K decoder helper for backend experimentation.
#[doc(hidden)]
pub fn decode_j2k_code_block_scalar(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    let mut workspace = J2kCodeBlockDecodeWorkspace::default();
    decode_j2k_code_block_scalar_with_workspace(job, output, &mut workspace)
}

/// Reusable scratch for scalar classic J2K code-block decoding.
#[derive(Default)]
#[doc(hidden)]
pub struct J2kCodeBlockDecodeWorkspace {
    bit_plane_decode_context: j2c::bitplane::BitPlaneDecodeContext,
}

impl J2kCodeBlockDecodeWorkspace {
    pub(crate) fn prepare(&mut self, width: u32, height: u32) -> Result<()> {
        self.bit_plane_decode_context.prepare(width, height)
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize> {
        self.bit_plane_decode_context.allocated_bytes()
    }
}

/// Adapter scalar classic J2K decoder helper that reuses caller-provided scratch.
#[doc(hidden)]
pub fn decode_j2k_code_block_scalar_with_workspace(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut J2kCodeBlockDecodeWorkspace,
) -> Result<()> {
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let style = internal_j2k_code_block_style(job.style);
    let sub_band_type = internal_j2k_sub_band_type(job.sub_band_type);
    let coded_bitplanes = add_roi_shift_to_bitplanes(
        job.total_bitplanes,
        job.roi_shift,
        MAX_CLASSIC_DECODE_BITPLANES,
    )?;

    j2c::bitplane::decode_code_block_segments_validated(
        job.data,
        job.segments,
        job.width,
        job.height,
        job.missing_bit_planes,
        job.number_of_coding_passes,
        coded_bitplanes,
        sub_band_type,
        &style,
        job.strict,
        &mut workspace.bit_plane_decode_context,
    )?;

    write_j2k_code_block_output(&workspace.bit_plane_decode_context, job, layout, output);

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CodeBlockOutputLayout {
    pub(super) stride: usize,
}

pub(super) fn checked_code_block_output_layout(
    width: u32,
    height: u32,
    output_stride: usize,
    output_len: usize,
) -> Result<CodeBlockOutputLayout> {
    let stride = usize::try_from(width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let height = height as usize;
    let required_len = if height == 0 {
        0
    } else {
        output_stride
            .checked_mul(height - 1)
            .and_then(|prefix| prefix.checked_add(stride))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
    };
    if output_len < required_len {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(CodeBlockOutputLayout { stride })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the public scalar adapter intentionally emits f32 coefficients"
)]
fn write_j2k_code_block_output(
    decode_context: &j2c::bitplane::BitPlaneDecodeContext,
    job: J2kCodeBlockDecodeJob<'_>,
    layout: CodeBlockOutputLayout,
    output: &mut [f32],
) {
    for (row_idx, coeff_row) in decode_context
        .coefficient_rows()
        .enumerate()
        .take(job.height as usize)
    {
        let row_start = row_idx * job.output_stride;
        let output_row = &mut output[row_start..row_start + layout.stride];
        for (coefficient, sample) in coeff_row.iter().zip(output_row.iter_mut()) {
            let coefficient = apply_roi_maxshift_inverse_i64(coefficient.get_i64(), job.roi_shift);
            *sample = coefficient as f32 * job.dequantization_step;
        }
    }
}

/// Adapter scalar classic J2K pass timings for backend experimentation.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub struct J2kCodeBlockDecodeProfile {
    /// Significance propagation pass elapsed time in microseconds.
    pub sigprop_us: u128,
    /// Magnitude refinement pass elapsed time in microseconds.
    pub magref_us: u128,
    /// Cleanup pass elapsed time in microseconds.
    pub cleanup_us: u128,
    /// Raw bypass pass elapsed time in microseconds.
    pub bypass_us: u128,
    /// Coefficient output conversion elapsed time in microseconds.
    pub output_convert_us: u128,
}

impl J2kCodeBlockDecodeProfile {
    /// Create an empty profile accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn add_native_stats(&mut self, stats: j2c::bitplane::J2kBlockDecodeStats) {
        self.sigprop_us += stats.sigprop_us;
        self.magref_us += stats.magref_us;
        self.cleanup_us += stats.cleanup_us;
        self.bypass_us += stats.bypass_us;
    }
}

/// Adapter scalar classic J2K decoder helper that records pass timings.
#[doc(hidden)]
pub fn decode_j2k_code_block_scalar_profiled(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    profile: &mut J2kCodeBlockDecodeProfile,
) -> Result<()> {
    let mut workspace = J2kCodeBlockDecodeWorkspace::default();
    decode_j2k_code_block_scalar_with_workspace_profiled(job, output, &mut workspace, profile)
}

/// Adapter scalar classic J2K decoder helper that records pass timings and reuses scratch.
#[doc(hidden)]
pub fn decode_j2k_code_block_scalar_with_workspace_profiled(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut J2kCodeBlockDecodeWorkspace,
    profile: &mut J2kCodeBlockDecodeProfile,
) -> Result<()> {
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let style = internal_j2k_code_block_style(job.style);
    let sub_band_type = internal_j2k_sub_band_type(job.sub_band_type);
    let coded_bitplanes = add_roi_shift_to_bitplanes(
        job.total_bitplanes,
        job.roi_shift,
        MAX_CLASSIC_DECODE_BITPLANES,
    )?;
    let mut stats = j2c::bitplane::J2kBlockDecodeStats::default();

    j2c::bitplane::decode_code_block_segments_validated_profiled(
        job.data,
        job.segments,
        job.width,
        job.height,
        job.missing_bit_planes,
        job.number_of_coding_passes,
        coded_bitplanes,
        sub_band_type,
        &style,
        job.strict,
        &mut workspace.bit_plane_decode_context,
        &mut stats,
        true,
    )?;
    profile.add_native_stats(stats);

    let output_convert_started = profile::profile_now(true);
    write_j2k_code_block_output(&workspace.bit_plane_decode_context, job, layout, output);
    profile.output_convert_us += profile::elapsed_us(output_convert_started);

    Ok(())
}

/// Adapter scalar classic J2K batched decoder helper for backend experimentation.
#[doc(hidden)]
pub fn decode_j2k_sub_band_scalar(job: J2kSubBandDecodeJob<'_>, output: &mut [f32]) -> Result<()> {
    let required_len = if job.height == 0 {
        0
    } else {
        usize::try_from(job.width)
            .ok()
            .and_then(|width| width.checked_mul(job.height as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
    };
    if output.len() < required_len {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    let sub_band_width =
        usize::try_from(job.width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;

    for batch_job in job.jobs {
        let code_block = batch_job.code_block;
        if code_block.output_stride != sub_band_width {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        if batch_job
            .output_x
            .checked_add(code_block.width)
            .is_none_or(|x1| x1 > job.width)
            || batch_job
                .output_y
                .checked_add(code_block.height)
                .is_none_or(|y1| y1 > job.height)
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        let base_idx = usize::try_from(batch_job.output_y)
            .ok()
            .and_then(|y| y.checked_mul(sub_band_width))
            .and_then(|row| row.checked_add(batch_job.output_x as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let block_output_len = if code_block.height == 0 {
            0
        } else {
            code_block
                .output_stride
                .checked_mul(code_block.height as usize - 1)
                .and_then(|prefix| prefix.checked_add(code_block.width as usize))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?
        };
        let end_idx = base_idx
            .checked_add(block_output_len)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if end_idx > output.len() {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        decode_j2k_code_block_scalar(code_block, &mut output[base_idx..end_idx])?;
    }

    Ok(())
}
