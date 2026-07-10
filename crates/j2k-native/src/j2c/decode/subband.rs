// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_i32, apply_roi_maxshift_inverse_i64,
    bitplane, classic_decode_job_parameters, collect_classic_code_block_data,
    decode_j2k_code_block_scalar, ht_block_decode, ht_code_block_has_decodable_passes,
    sub_band_decode_parameters, CodeBlock, ComponentInfo, CpuDecodeParallelism, DecodingError,
    DecompositionStorage, Header, HtCodeBlockBatchJob, HtCodeBlockDecodeJob, HtCodeBlockDecoder,
    HtSubBandDecodeJob, J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob, J2kCodeBlockSegment,
    J2kSubBandDecodeJob, Result, SubBand, SubBandDecodeParameters, Tile, TileDecodeContext, Vec,
    MAX_BITPLANE_COUNT,
};

#[cfg(feature = "parallel")]
use super::{
    bail, decode_ht_code_block_scalar_with_workspace, vec, HtCodeBlockDecodeWorkspace,
    J2kCodeBlockStyle, J2kSubBandType,
};

mod ht;
use self::ht::{decode_sub_band_ht_blocks, decode_sub_band_ht_blocks_i64};

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
        let pending_blocks =
            collect_pending_classic_blocks(sub_band_idx, &sub_band, component_info, storage)?;

        let batch_jobs: Vec<_> = pending_blocks
            .iter()
            .map(|pending| J2kCodeBlockBatchJob {
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
            })
            .collect();

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
            decode_j2k_code_block_scalar(job.code_block, output_slice)?;
        }

        return Ok(());
    }

    let code_block_count = count_classic_code_blocks(sub_band_idx, &sub_band, storage);
    if should_decode_classic_sub_band_in_parallel(cpu_decode_parallelism, code_block_count) {
        #[cfg(feature = "parallel")]
        {
            let pending_blocks =
                collect_pending_classic_blocks(sub_band_idx, &sub_band, component_info, storage)?;
            let decoded_blocks = decode_classic_sub_band_blocks_parallel(
                &pending_blocks,
                classic_job_sub_band_type,
                classic_job_style,
                header.strict,
                num_bitplanes,
                component_info.roi_shift,
                dequantization_step,
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

struct PendingHtBlock {
    combined: ht_block_decode::CombinedCodeBlockData,
    output_x: u32,
    output_y: u32,
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
}

struct PendingClassicBlock {
    combined_data: Vec<u8>,
    segments: Vec<J2kCodeBlockSegment>,
    output_x: u32,
    output_y: u32,
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
}

#[cfg(feature = "parallel")]
pub(super) struct DecodedClassicBlock {
    pub(super) output_x: u32,
    pub(super) output_y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) coefficients: Vec<f32>,
}

#[cfg(feature = "parallel")]
pub(super) struct DecodedHtBlock {
    pub(super) output_x: u32,
    pub(super) output_y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) coefficients: Vec<f32>,
}

#[cfg(feature = "parallel")]
trait DecodedSubBandBlock {
    fn output_x(&self) -> u32;
    fn output_y(&self) -> u32;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn coefficients(&self) -> &[f32];
}

#[cfg(feature = "parallel")]
impl DecodedSubBandBlock for DecodedClassicBlock {
    fn output_x(&self) -> u32 {
        self.output_x
    }

    fn output_y(&self) -> u32 {
        self.output_y
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
}

#[cfg(feature = "parallel")]
impl DecodedSubBandBlock for DecodedHtBlock {
    fn output_x(&self) -> u32 {
        self.output_x
    }

    fn output_y(&self) -> u32 {
        self.output_y
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
}

fn count_classic_code_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
) -> usize {
    sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
        .map(|precinct| {
            precinct
                .code_blocks
                .clone()
                .filter(|idx| {
                    let code_block = &storage.code_blocks[*idx];
                    code_block_required_by_index(storage, sub_band_idx, code_block)
                })
                .count()
        })
        .sum()
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

fn collect_pending_classic_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
) -> Result<Vec<PendingClassicBlock>> {
    let mut pending_blocks =
        Vec::with_capacity(count_classic_code_blocks(sub_band_idx, sub_band, storage));
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
                continue;
            }
            let (combined_data, segments) = collect_classic_code_block_data(
                code_block,
                &component_info.coding_style.parameters.code_block_style,
                storage,
            )?;
            pending_blocks.push(PendingClassicBlock {
                combined_data,
                segments,
                output_x: code_block.rect.x0 - sub_band.rect.x0,
                output_y: code_block.rect.y0 - sub_band.rect.y0,
                width: code_block.rect.width(),
                height: code_block.rect.height(),
                missing_bit_planes: code_block.missing_bit_planes,
                number_of_coding_passes: code_block.number_of_coding_passes,
            });
        }
    }
    Ok(pending_blocks)
}

fn count_ht_code_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
) -> usize {
    sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
        .map(|precinct| {
            precinct
                .code_blocks
                .clone()
                .filter(|idx| {
                    let code_block = &storage.code_blocks[*idx];
                    code_block_required_by_index(storage, sub_band_idx, code_block)
                        && code_block.number_of_coding_passes > 0
                })
                .count()
        })
        .sum()
}

#[cfg(feature = "parallel")]
fn collect_pending_ht_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    num_bitplanes: u8,
    roi_shift: u8,
) -> Result<Vec<PendingHtBlock>> {
    let coded_bitplanes = add_roi_shift_to_bitplanes(num_bitplanes, roi_shift, 31)?;
    let mut pending_blocks =
        Vec::with_capacity(count_ht_code_blocks(sub_band_idx, sub_band, storage));
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
                continue;
            }
            if !ht_code_block_has_decodable_passes(code_block, coded_bitplanes, header.strict)? {
                continue;
            }

            pending_blocks.push(PendingHtBlock {
                combined: ht_block_decode::collect_code_block_data(code_block, storage)?,
                output_x: code_block.rect.x0 - sub_band.rect.x0,
                output_y: code_block.rect.y0 - sub_band.rect.y0,
                width: code_block.rect.width(),
                height: code_block.rect.height(),
                missing_bit_planes: code_block.missing_bit_planes,
                number_of_coding_passes: code_block.number_of_coding_passes,
            });
        }
    }
    Ok(pending_blocks)
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

#[cfg(feature = "parallel")]
fn decode_classic_sub_band_blocks_parallel(
    pending_blocks: &[PendingClassicBlock],
    sub_band_type: J2kSubBandType,
    style: J2kCodeBlockStyle,
    strict: bool,
    total_bitplanes: u8,
    roi_shift: u8,
    dequantization_step: f32,
) -> Result<Vec<DecodedClassicBlock>> {
    use rayon::prelude::*;

    pending_blocks
        .par_iter()
        .map(|pending| {
            let output_stride = pending.width as usize;
            let output_len = output_stride
                .checked_mul(pending.height as usize)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let mut coefficients = vec![0.0; output_len];
            decode_j2k_code_block_scalar(
                J2kCodeBlockDecodeJob {
                    data: &pending.combined_data,
                    segments: &pending.segments,
                    width: pending.width,
                    height: pending.height,
                    output_stride,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    total_bitplanes,
                    roi_shift,
                    sub_band_type,
                    style,
                    strict,
                    dequantization_step,
                },
                &mut coefficients,
            )?;
            Ok(DecodedClassicBlock {
                output_x: pending.output_x,
                output_y: pending.output_y,
                width: pending.width,
                height: pending.height,
                coefficients,
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

#[cfg(feature = "parallel")]
pub(super) fn copy_decoded_classic_blocks_to_sub_band(
    decoded_blocks: &[DecodedClassicBlock],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    copy_decoded_blocks_to_sub_band(decoded_blocks, sub_band, storage)
}

#[cfg(feature = "parallel")]
fn copy_decoded_blocks_to_sub_band<B: DecodedSubBandBlock>(
    decoded_blocks: &[B],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    let sub_band_width = sub_band.rect.width() as usize;
    let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
    for block in decoded_blocks {
        let output_x = block.output_x();
        let output_y = block.output_y();
        let width = block.width();
        let height = block.height();
        if output_x
            .checked_add(width)
            .is_none_or(|x1| x1 > sub_band.rect.width())
            || output_y
                .checked_add(height)
                .is_none_or(|y1| y1 > sub_band.rect.height())
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        let block_width = width as usize;
        for row in 0..height as usize {
            let dst_start = (output_y as usize + row)
                .checked_mul(sub_band_width)
                .and_then(|offset| offset.checked_add(output_x as usize))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let dst_end = dst_start
                .checked_add(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let src_start = row
                .checked_mul(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let src_end = src_start
                .checked_add(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            base_store[dst_start..dst_end]
                .copy_from_slice(&block.coefficients()[src_start..src_end]);
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn decode_ht_sub_band_blocks_parallel(
    pending_blocks: &[PendingHtBlock],
    strict: bool,
    num_bitplanes: u8,
    roi_shift: u8,
    stripe_causal: bool,
    dequantization_step: f32,
) -> Result<Vec<DecodedHtBlock>> {
    use rayon::prelude::*;

    pending_blocks
        .par_iter()
        .map(|pending| {
            let output_stride = pending.width as usize;
            let output_len = output_stride
                .checked_mul(pending.height as usize)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let mut coefficients = vec![0.0; output_len];
            let mut workspace = HtCodeBlockDecodeWorkspace::default();
            decode_ht_code_block_scalar_with_workspace(
                HtCodeBlockDecodeJob {
                    data: &pending.combined.data,
                    cleanup_length: pending.combined.cleanup_length,
                    refinement_length: pending.combined.refinement_length,
                    width: pending.width,
                    height: pending.height,
                    output_stride,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    num_bitplanes,
                    roi_shift,
                    stripe_causal,
                    strict,
                    dequantization_step,
                },
                &mut coefficients,
                &mut workspace,
            )?;
            Ok(DecodedHtBlock {
                output_x: pending.output_x,
                output_y: pending.output_y,
                width: pending.width,
                height: pending.height,
                coefficients,
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

#[cfg(feature = "parallel")]
pub(super) fn copy_decoded_ht_blocks_to_sub_band(
    decoded_blocks: &[DecodedHtBlock],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    copy_decoded_blocks_to_sub_band(decoded_blocks, sub_band, storage)
}
