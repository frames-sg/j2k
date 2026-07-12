// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_i32, apply_roi_maxshift_inverse_i64,
    bitplane, classic_decode_job_parameters, collect_classic_code_block_data,
    decode_j2k_code_block_scalar_with_workspace, ht_block_decode,
    ht_code_block_has_decodable_passes, sub_band_decode_parameters, CodeBlock, ComponentInfo,
    CpuDecodeParallelism, DecodeAllocationBudget, DecodingError, DecompositionStorage, Header,
    HtCodeBlockBatchJob, HtCodeBlockDecodeJob, HtCodeBlockDecoder, HtSubBandDecodeJob,
    J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace, J2kSubBandDecodeJob,
    Result, SubBand, SubBandDecodeParameters, Tile, TileDecodeContext, Vec, MAX_BITPLANE_COUNT,
};

mod ht;
#[cfg(feature = "parallel")]
mod parallel;
mod pending;
use self::ht::{decode_sub_band_ht_blocks, decode_sub_band_ht_blocks_i64};
#[cfg(all(feature = "parallel", not(test)))]
use self::parallel::{copy_decoded_classic_blocks_to_sub_band, copy_decoded_ht_blocks_to_sub_band};
#[cfg(all(test, feature = "parallel"))]
pub(super) use self::parallel::{
    copy_decoded_classic_blocks_to_sub_band, copy_decoded_ht_blocks_to_sub_band,
    DecodedClassicBlock, DecodedHtBlock,
};
#[cfg(feature = "parallel")]
use self::parallel::{
    decode_classic_sub_band_blocks_parallel, decode_ht_sub_band_blocks_parallel,
    ClassicParallelParameters,
};
use self::pending::{
    collect_pending_classic_blocks, collect_pending_ht_blocks, count_classic_code_blocks,
    count_ht_code_blocks,
};

pub(crate) fn decode_component_tile_bit_planes<'a>(
    tile: &Tile<'a>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
) -> Result<()> {
    for (tile_decompositions_idx, component_info) in tile.component_infos.iter().enumerate() {
        // Only decode the resolution levels we actually care about.
        for resolution in
            0..component_info.num_resolution_levels() - header.skipped_resolution_levels
        {
            let tile_composition = &storage.tile_decompositions[tile_decompositions_idx];
            let sub_band_iter = tile_composition.sub_band_iter(resolution, &storage.decompositions);

            for sub_band_idx in sub_band_iter {
                decode_sub_band_bitplanes(
                    sub_band_idx,
                    resolution,
                    component_info,
                    tile_ctx,
                    storage,
                    header,
                    ht_decoder,
                    cpu_decode_parallelism,
                    profile_enabled,
                )?;
            }
        }
    }

    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples or metadata at this rounding boundary"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
fn decode_sub_band_bitplanes(
    sub_band_idx: usize,
    resolution: u8,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
) -> Result<()> {
    let sub_band = storage.sub_bands[sub_band_idx].clone();
    let SubBandDecodeParameters {
        dequantization_step,
        num_bitplanes,
    } = sub_band_decode_parameters(&sub_band, resolution, component_info)?;

    if component_info
        .coding_style
        .parameters
        .code_block_style
        .uses_high_throughput_block_coding()
    {
        if storage.exact_integer_decode {
            decode_sub_band_ht_blocks_i64(
                sub_band_idx,
                &sub_band,
                component_info,
                tile_ctx,
                storage,
                header,
                num_bitplanes,
                profile_enabled,
            )?;
            return Ok(());
        }
        decode_sub_band_ht_blocks(
            sub_band_idx,
            &sub_band,
            component_info,
            tile_ctx,
            storage,
            header,
            ht_decoder,
            cpu_decode_parallelism,
            num_bitplanes,
            dequantization_step,
            profile_enabled,
        )?;
        return Ok(());
    }

    let coded_bitplanes =
        add_roi_shift_to_bitplanes(num_bitplanes, component_info.roi_shift, MAX_BITPLANE_COUNT)?;

    if storage.exact_integer_decode {
        decode_sub_band_classic_blocks_i64(
            sub_band_idx,
            &sub_band,
            component_info,
            tile_ctx,
            storage,
            header,
            coded_bitplanes,
        )?;
        return Ok(());
    }

    let (classic_job_sub_band_type, classic_job_style) =
        classic_decode_job_parameters(sub_band.sub_band_type, component_info);

    if let Some(ht_decoder) = ht_decoder.as_deref_mut() {
        let mut budget = DecodeAllocationBudget::for_storage(storage)?;
        let pending_blocks = collect_pending_classic_blocks(
            sub_band_idx,
            &sub_band,
            component_info,
            storage,
            &mut budget,
        )?;

        let mut batch_jobs = Vec::new();
        budget.reserve_new(&mut batch_jobs, pending_blocks.len())?;
        for pending in &pending_blocks {
            batch_jobs.push(J2kCodeBlockBatchJob {
                output_x: pending.output_x,
                output_y: pending.output_y,
                code_block: J2kCodeBlockDecodeJob {
                    data: &pending.combined_data,
                    segments: &pending.segments,
                    width: pending.width,
                    height: pending.height,
                    output_stride: sub_band.rect.width() as usize,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    total_bitplanes: num_bitplanes,
                    roi_shift: component_info.roi_shift,
                    sub_band_type: classic_job_sub_band_type,
                    style: classic_job_style,
                    strict: header.strict,
                    dequantization_step,
                },
            });
        }

        let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
        if ht_decoder.decode_j2k_sub_band(
            J2kSubBandDecodeJob {
                width: sub_band.rect.width(),
                height: sub_band.rect.height(),
                jobs: &batch_jobs,
            },
            base_store,
        )? {
            tile_ctx.debug_counters.decoded_code_blocks += batch_jobs.len();
            return Ok(());
        }

        let (workspace_width, workspace_height) =
            batch_jobs
                .iter()
                .fold((0_u32, 0_u32), |(width, height), job| {
                    (
                        width.max(job.code_block.width),
                        height.max(job.code_block.height),
                    )
                });
        let planned_workspace =
            bitplane::classic_decode_workspace_bytes(workspace_width, workspace_height)?;
        budget.include_bytes(planned_workspace)?;
        let mut scalar_workspace = J2kCodeBlockDecodeWorkspace::default();
        scalar_workspace.prepare(workspace_width, workspace_height)?;
        let actual_workspace = scalar_workspace.allocated_bytes()?;
        if actual_workspace > planned_workspace {
            budget.include_bytes(actual_workspace - planned_workspace)?;
        }

        let output_stride = sub_band.rect.width() as usize;
        for job in batch_jobs {
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            let base_idx = (job.output_y * sub_band.rect.width()) as usize + job.output_x as usize;
            let output_len = if job.code_block.height == 0 {
                0
            } else {
                output_stride
                    .checked_mul(job.code_block.height as usize - 1)
                    .and_then(|prefix| prefix.checked_add(job.code_block.width as usize))
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?
            };
            let output_slice = &mut base_store[base_idx..base_idx + output_len];
            if ht_decoder.decode_j2k_code_block(job.code_block, output_slice)? {
                continue;
            }
            decode_j2k_code_block_scalar_with_workspace(
                job.code_block,
                output_slice,
                &mut scalar_workspace,
            )?;
        }

        return Ok(());
    }

    let code_block_count = count_classic_code_blocks(sub_band_idx, &sub_band, storage)?;
    if should_decode_classic_sub_band_in_parallel(cpu_decode_parallelism, code_block_count) {
        #[cfg(feature = "parallel")]
        {
            let mut budget = DecodeAllocationBudget::for_storage(storage)?;
            let pending_blocks = collect_pending_classic_blocks(
                sub_band_idx,
                &sub_band,
                component_info,
                storage,
                &mut budget,
            )?;
            let decoded_blocks = decode_classic_sub_band_blocks_parallel(
                &pending_blocks,
                ClassicParallelParameters {
                    sub_band_type: classic_job_sub_band_type,
                    style: classic_job_style,
                    strict: header.strict,
                    total_bitplanes: num_bitplanes,
                    roi_shift: component_info.roi_shift,
                    dequantization_step,
                },
                &mut budget,
            )?;
            tile_ctx.debug_counters.decoded_code_blocks += decoded_blocks.len();
            copy_decoded_classic_blocks_to_sub_band(&decoded_blocks, &sub_band, storage)?;
            return Ok(());
        }
    }

    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                tile_ctx.debug_counters.skipped_code_blocks += 1;
                continue;
            }
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;
            let output_stride = sub_band.rect.width() as usize;
            let base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;

            bitplane::decode(
                code_block,
                sub_band.sub_band_type,
                coded_bitplanes,
                &component_info.coding_style.parameters.code_block_style,
                tile_ctx,
                storage,
                header.strict,
            )?;

            let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
            let mut base_idx = base_idx;

            for coefficients in tile_ctx.bit_plane_decode_context.coefficient_rows() {
                let out_row = &mut base_store[base_idx..];

                for (output, coefficient) in out_row.iter_mut().zip(coefficients.iter().copied()) {
                    let coefficient = apply_roi_maxshift_inverse_i64(
                        coefficient.get_i64(),
                        component_info.roi_shift,
                    );
                    *output = coefficient as f32;
                    *output *= dequantization_step;
                }

                base_idx += output_stride;
            }
        }
    }

    Ok(())
}

fn decode_sub_band_classic_blocks_i64(
    sub_band_idx: usize,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
    coded_bitplanes: u8,
) -> Result<()> {
    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                tile_ctx.debug_counters.skipped_code_blocks += 1;
                continue;
            }
            tile_ctx.debug_counters.decoded_code_blocks += 1;
            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;
            let output_stride = sub_band.rect.width() as usize;
            let base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;

            bitplane::decode(
                code_block,
                sub_band.sub_band_type,
                coded_bitplanes,
                &component_info.coding_style.parameters.code_block_style,
                tile_ctx,
                storage,
                header.strict,
            )?;

            let base_store = &mut storage.coefficients_i64[sub_band.coefficients.clone()];
            let mut base_idx = base_idx;

            for coefficients in tile_ctx.bit_plane_decode_context.coefficient_rows() {
                let out_row = &mut base_store[base_idx..];

                for (output, coefficient) in out_row.iter_mut().zip(coefficients.iter().copied()) {
                    *output = apply_roi_maxshift_inverse_i64(
                        coefficient.get_i64(),
                        component_info.roi_shift,
                    );
                }

                base_idx += output_stride;
            }
        }
    }

    Ok(())
}

pub(super) fn code_block_required_by_index(
    storage: &DecompositionStorage<'_>,
    sub_band_idx: usize,
    code_block: &CodeBlock,
) -> bool {
    storage
        .roi_plan
        .as_ref()
        .is_none_or(|plan| plan.code_block_required(sub_band_idx, code_block.rect))
}

pub(crate) fn should_decode_classic_sub_band_in_parallel(
    parallelism: CpuDecodeParallelism,
    code_block_count: usize,
) -> bool {
    cfg!(feature = "parallel") && parallelism == CpuDecodeParallelism::Auto && code_block_count >= 4
}

pub(crate) fn should_decode_ht_sub_band_in_parallel(
    parallelism: CpuDecodeParallelism,
    code_block_count: usize,
) -> bool {
    cfg!(feature = "parallel") && parallelism == CpuDecodeParallelism::Auto && code_block_count >= 4
}
