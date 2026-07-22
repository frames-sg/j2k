use super::{
    bail, decode_ht_code_block_scalar_with_workspace, decode_j2k_code_block_scalar_with_workspace,
    idwt, try_resize_decode_elements, DecodingError, DirectComponentBandScratch,
    DirectComponentPlane, DirectCpuBand, DirectWorkspaceBudget, HtCodeBlockDecodeJob,
    HtCodeBlockDecodeWorkspace, HtOwnedSubBandPlan, J2kCodeBlockDecodeJob,
    J2kCodeBlockDecodeWorkspace, J2kDirectBandId, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kDirectIdwtStep, J2kDirectStoreStep, J2kIdwtBand, J2kOwnedSubBandPlan, J2kRect,
    J2kSingleDecompositionIdwtJob, Range, Result, Vec,
};

pub(super) fn execute_component_plan(
    plan: &J2kDirectGrayscalePlan,
    bands: &mut DirectComponentBandScratch,
    output: &mut DirectComponentPlane,
    workspace_budget: DirectWorkspaceBudget,
) -> Result<()> {
    bands.reset();
    let mut output_written = false;

    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                execute_classic_sub_band(sub_band, bands, workspace_budget)?;
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                execute_ht_sub_band(sub_band, bands, workspace_budget)?;
            }
            J2kDirectGrayscaleStep::Idwt(step) => execute_idwt_step(step, bands)?,
            J2kDirectGrayscaleStep::Store(store) => {
                store_component(store, bands.active(), output, &mut output_written)?;
            }
        }
    }

    if output_written {
        Ok(())
    } else {
        Err(DecodingError::CodeBlockDecodeFailure.into())
    }
}

fn execute_classic_sub_band(
    plan: &J2kOwnedSubBandPlan,
    bands: &mut DirectComponentBandScratch,
    workspace_budget: DirectWorkspaceBudget,
) -> Result<()> {
    let (output, sub_band_width) =
        prepare_sub_band_output(bands, plan.band_id, plan.rect, plan.width, plan.height)?;
    let mut workspace = J2kCodeBlockDecodeWorkspace::default();
    if let Some((width, height)) = max_classic_job_dimensions(plan) {
        workspace.prepare(width, height)?;
        workspace_budget.validate_workspace(workspace.allocated_bytes()?)?;
    }

    for job in &plan.jobs {
        let output_range = checked_sub_band_job_output_range(&SubBandJobOutputRange {
            output_x: job.output_x,
            output_y: job.output_y,
            output_stride: job.output_stride,
            width: job.width,
            height: job.height,
            sub_band_width,
            plan_width: plan.width,
            plan_height: plan.height,
            output_len: output.len(),
        })?;

        let code_block = J2kCodeBlockDecodeJob {
            data: &job.data,
            segments: &job.segments,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            total_bitplanes: job.total_bitplanes,
            roi_shift: job.roi_shift,
            sub_band_type: job.sub_band_type,
            style: job.style,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        };
        decode_j2k_code_block_scalar_with_workspace(
            code_block,
            &mut output[output_range],
            &mut workspace,
        )?;
    }
    Ok(())
}

fn execute_ht_sub_band(
    plan: &HtOwnedSubBandPlan,
    bands: &mut DirectComponentBandScratch,
    workspace_budget: DirectWorkspaceBudget,
) -> Result<()> {
    let (output, sub_band_width) =
        prepare_sub_band_output(bands, plan.band_id, plan.rect, plan.width, plan.height)?;
    let mut workspace = HtCodeBlockDecodeWorkspace::default();
    if let Some((width, height)) = max_ht_job_dimensions(plan) {
        workspace.prepare(width, height)?;
        workspace_budget.validate_workspace(workspace.allocated_bytes()?)?;
    }

    for job in &plan.jobs {
        let output_range = checked_sub_band_job_output_range(&SubBandJobOutputRange {
            output_x: job.output_x,
            output_y: job.output_y,
            output_stride: job.output_stride,
            width: job.width,
            height: job.height,
            sub_band_width,
            plan_width: plan.width,
            plan_height: plan.height,
            output_len: output.len(),
        })?;

        let code_block = HtCodeBlockDecodeJob {
            data: &job.data,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            num_bitplanes: job.num_bitplanes,
            roi_shift: job.roi_shift,
            stripe_causal: job.stripe_causal,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        };
        decode_ht_code_block_scalar_with_workspace(
            code_block,
            &mut output[output_range],
            &mut workspace,
        )?;
    }
    Ok(())
}

fn max_classic_job_dimensions(plan: &J2kOwnedSubBandPlan) -> Option<(u32, u32)> {
    plan.jobs.iter().fold(None, |dimensions, job| {
        Some(
            dimensions.map_or((job.width, job.height), |(width, height)| {
                (width.max(job.width), height.max(job.height))
            }),
        )
    })
}

fn max_ht_job_dimensions(plan: &HtOwnedSubBandPlan) -> Option<(u32, u32)> {
    plan.jobs.iter().fold(None, |dimensions, job| {
        Some(
            dimensions.map_or((job.width, job.height), |(width, height)| {
                (width.max(job.width), height.max(job.height))
            }),
        )
    })
}

pub(super) fn prepare_sub_band_output(
    bands: &mut DirectComponentBandScratch,
    band_id: J2kDirectBandId,
    rect: J2kRect,
    width: u32,
    height: u32,
) -> Result<(&mut [f32], usize)> {
    let required_len = checked_area(width, height)?;
    let band_index = bands.prepare_band(band_id, rect, required_len)?;
    let output = bands.bands[band_index].coefficients.as_mut_slice();
    let sub_band_width =
        usize::try_from(width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    Ok((output, sub_band_width))
}

pub(super) fn execute_idwt_step(
    step: &J2kDirectIdwtStep,
    bands: &mut DirectComponentBandScratch,
) -> Result<()> {
    let output_index = bands.prepare_band(step.output_band_id, step.rect, 0)?;
    let (input_bands, output_bands) = bands.bands.split_at_mut(output_index);
    let output = &mut output_bands[0].coefficients;
    let ll = find_idwt_band(input_bands, step.ll_band_id)?;
    let hl = find_idwt_band(input_bands, step.hl_band_id)?;
    let lh = find_idwt_band(input_bands, step.lh_band_id)?;
    let hh = find_idwt_band(input_bands, step.hh_band_id)?;
    let job = J2kSingleDecompositionIdwtJob {
        rect: step.rect,
        transform: step.transform,
        ll,
        hl,
        lh,
        hh,
    };
    idwt::apply_single_decomposition_idwt_job(job, output)
}

fn find_idwt_band(bands: &[DirectCpuBand], band_id: J2kDirectBandId) -> Result<J2kIdwtBand<'_>> {
    let band = find_band(bands, band_id)?;
    Ok(J2kIdwtBand {
        rect: band.rect,
        coefficients: &band.coefficients,
    })
}

pub(super) fn store_component(
    store: &J2kDirectStoreStep,
    bands: &[DirectCpuBand],
    plane: &mut DirectComponentPlane,
    output_written: &mut bool,
) -> Result<()> {
    let input = find_band(bands, store.input_band_id)?;
    if !*output_written {
        plane.width = store.output_width;
        plane.height = store.output_height;
        let required_len = checked_area(store.output_width, store.output_height)?;
        resize_and_zero(&mut plane.samples, required_len)?;
        *output_written = true;
    }
    if plane.width != store.output_width
        || plane.height != store.output_height
        || plane.samples.len() != checked_area(store.output_width, store.output_height)?
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    validate_store_bounds(store, input, plane)?;
    let input_width = input.rect.width() as usize;
    let output_width = plane.width as usize;
    let copy_width = store.copy_width as usize;
    for row in 0..store.copy_height as usize {
        let src_start = (store.source_y as usize + row)
            .checked_mul(input_width)
            .and_then(|base| base.checked_add(store.source_x as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let dst_start = (store.output_y as usize + row)
            .checked_mul(output_width)
            .and_then(|base| base.checked_add(store.output_x as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let src = &input.coefficients[src_start..src_start + copy_width];
        let dst = &mut plane.samples[dst_start..dst_start + copy_width];
        for (src, dst) in src.iter().zip(dst.iter_mut()) {
            *dst = *src + store.addend;
        }
    }
    Ok(())
}

fn find_band(bands: &[DirectCpuBand], band_id: J2kDirectBandId) -> Result<&DirectCpuBand> {
    bands
        .iter()
        .find(|band| band.band_id == band_id)
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

fn validate_store_bounds(
    store: &J2kDirectStoreStep,
    input: &DirectCpuBand,
    output: &DirectComponentPlane,
) -> Result<()> {
    if store
        .source_x
        .checked_add(store.copy_width)
        .is_none_or(|x| x > input.rect.width())
        || store
            .source_y
            .checked_add(store.copy_height)
            .is_none_or(|y| y > input.rect.height())
        || store
            .output_x
            .checked_add(store.copy_width)
            .is_none_or(|x| x > output.width)
        || store
            .output_y
            .checked_add(store.copy_height)
            .is_none_or(|y| y > output.height)
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(())
}

pub(super) fn checked_area(width: u32, height: u32) -> Result<usize> {
    let height = usize::try_from(height).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    usize::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(height))
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

fn checked_block_base(output_x: u32, output_y: u32, stride: usize) -> Result<usize> {
    let output_x = usize::try_from(output_x).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    usize::try_from(output_y)
        .ok()
        .and_then(|y| y.checked_mul(stride))
        .and_then(|base| base.checked_add(output_x))
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

pub(super) struct SubBandJobOutputRange {
    pub(super) output_x: u32,
    pub(super) output_y: u32,
    pub(super) output_stride: usize,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) sub_band_width: usize,
    pub(super) plan_width: u32,
    pub(super) plan_height: u32,
    pub(super) output_len: usize,
}

pub(super) fn checked_sub_band_job_output_range(
    bounds: &SubBandJobOutputRange,
) -> Result<Range<usize>> {
    let base_idx = checked_block_base(bounds.output_x, bounds.output_y, bounds.sub_band_width)?;
    let block_len = checked_block_output_len(bounds.output_stride, bounds.width, bounds.height)?;
    let end_idx = base_idx
        .checked_add(block_len)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    if end_idx > bounds.output_len
        || bounds
            .output_x
            .checked_add(bounds.width)
            .is_none_or(|x| x > bounds.plan_width)
        || bounds
            .output_y
            .checked_add(bounds.height)
            .is_none_or(|y| y > bounds.plan_height)
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(base_idx..end_idx)
}

fn checked_block_output_len(stride: usize, width: u32, height: u32) -> Result<usize> {
    if height == 0 {
        return Ok(0);
    }
    let height = usize::try_from(height).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let width = usize::try_from(width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    stride
        .checked_mul(height - 1)
        .and_then(|prefix| prefix.checked_add(width))
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

pub(super) fn resize_and_zero(buffer: &mut Vec<f32>, len: usize) -> Result<()> {
    try_resize_decode_elements(buffer, len, 0.0)?;
    buffer.fill(0.0);
    Ok(())
}
