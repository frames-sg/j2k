// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    add_roi_shift_to_bitplanes, bail, build, classic_decode_job_parameters,
    code_block_required_by_index, component_unsigned_level_shift, ht_block_decode,
    ht_code_block_has_decodable_passes, progression_iterator, segment, sub_band_decode_parameters,
    tile, BitReader, ColorError, ComponentInfo, ComponentTile, DecodeAllocationBudget,
    DecoderContext, DecodingError, DecompositionStorage, DirectPlanUnsupportedReason, Header,
    HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kDirectBandId, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan, J2kRect, J2kWaveletTransform, ResolutionTile,
    Result, RoiPlan, SubBand, SubBandDecodeParameters, Tile, ValidationError, Vec,
};

mod classic;
pub(super) use self::classic::collect_classic_code_block_data;

pub(crate) fn build_direct_grayscale_plan<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kDirectGrayscalePlan> {
    ctx.release_reusable_allocations();
    let result = build_direct_grayscale_plan_inner(data, header, retained_image_bytes, ctx);
    ctx.release_reusable_allocations();
    result
}

fn build_direct_grayscale_plan_inner<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kDirectGrayscalePlan> {
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header, retained_image_bytes)?;

    if tiles.len() != 1 {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::GrayscaleSingleTileCodestream
        ));
    }

    let tile = &tiles[0];
    if tile.component_infos.len() != 1 {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::GrayscaleSingleComponentCodestream
        ));
    }
    ctx.tile_decode_context.channel_data.clear();
    ctx.storage.reset_for_next_tile();

    build::build(
        tile,
        &mut ctx.storage,
        tiles.structural_workspace_bytes(),
        ctx.tile_decode_context.output_region.is_some(),
        build::BuildWorkspace::CoefficientsOnly,
    )?;
    if let Some(output_region) = ctx.tile_decode_context.output_region {
        ctx.storage.roi_plan = RoiPlan::build(tile, header, &ctx.storage, output_region)?;
        if ctx.storage.roi_plan.is_none() {
            build::release_unused_roi_workspace(&mut ctx.storage, tile.component_infos.len())?;
        }
    }

    segment::parse(tile, progression_iterator(tile)?, header, &mut ctx.storage)?;

    let component_info = &tile.component_infos[0];
    let mut budget = DecodeAllocationBudget::for_storage(&ctx.storage)?;
    build_component_plan_from_storage(
        tile,
        header,
        &ctx.storage,
        0,
        component_unsigned_level_shift(component_info),
        &mut budget,
    )
}

pub(crate) fn build_direct_color_plan<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kDirectColorPlan> {
    ctx.release_reusable_allocations();
    let result = build_direct_color_plan_inner(data, header, retained_image_bytes, ctx);
    ctx.release_reusable_allocations();
    result
}

fn build_direct_color_plan_inner<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kDirectColorPlan> {
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header, retained_image_bytes)?;

    if tiles.len() != 1 {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::ColorSingleTileCodestream
        ));
    }

    let tile = &tiles[0];
    if tile.component_infos.len() != 3 {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::ColorThreeComponentRgbCodestream
        ));
    }
    let transform = tile.component_infos[0].wavelet_transform();
    if tile.mct
        && (transform != tile.component_infos[1].wavelet_transform()
            || transform != tile.component_infos[2].wavelet_transform())
    {
        bail!(ColorError::Mct);
    }

    ctx.tile_decode_context.channel_data.clear();
    ctx.storage.reset_for_next_tile();

    build::build(
        tile,
        &mut ctx.storage,
        tiles.structural_workspace_bytes(),
        ctx.tile_decode_context.output_region.is_some(),
        build::BuildWorkspace::CoefficientsOnly,
    )?;
    if let Some(output_region) = ctx.tile_decode_context.output_region {
        ctx.storage.roi_plan = RoiPlan::build(tile, header, &ctx.storage, output_region)?;
        if ctx.storage.roi_plan.is_none() {
            build::release_unused_roi_workspace(&mut ctx.storage, tile.component_infos.len())?;
        }
    }

    segment::parse(tile, progression_iterator(tile)?, header, &mut ctx.storage)?;

    let mut bit_depths = [0_u8; 3];
    let mut budget = DecodeAllocationBudget::for_storage(&ctx.storage)?;
    let mut component_plans = Vec::new();
    budget.reserve_new(&mut component_plans, bit_depths.len())?;
    for (component_idx, bit_depth) in bit_depths.iter_mut().enumerate() {
        let component_info = &tile.component_infos[component_idx];
        *bit_depth = component_info.size_info.precision;
        let addend = if tile.mct {
            0.0
        } else {
            component_unsigned_level_shift(component_info)
        };
        component_plans.push(build_component_plan_from_storage(
            tile,
            header,
            &ctx.storage,
            component_idx,
            addend,
            &mut budget,
        )?);
    }

    Ok(J2kDirectColorPlan {
        dimensions: (
            header.size_data.image_width(),
            header.size_data.image_height(),
        ),
        bit_depths,
        mct: tile.mct,
        transform: J2kWaveletTransform::from(transform),
        component_plans,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
fn build_component_plan_from_storage(
    tile: &Tile<'_>,
    header: &Header<'_>,
    storage: &DecompositionStorage<'_>,
    component_idx: usize,
    store_addend: f32,
    budget: &mut DecodeAllocationBudget,
) -> Result<J2kDirectGrayscalePlan> {
    let component_info =
        tile.component_infos
            .get(component_idx)
            .ok_or(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ComponentIndexOutOfRange,
            ))?;
    if component_info.size_info.horizontal_resolution != 1
        || component_info.size_info.vertical_resolution != 1
    {
        bail!(DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::ComponentUnitSampled
        ));
    }

    let tile_decompositions = storage.tile_decompositions.get(component_idx).ok_or(
        DecodingError::DirectPlanUnsupported(
            DirectPlanUnsupportedReason::ComponentDecompositionIndexOutOfRange,
        ),
    )?;
    let decompositions = &storage.decompositions[tile_decompositions.decompositions.clone()];
    let active_decomposition_count = decompositions
        .len()
        .saturating_sub(header.skipped_resolution_levels as usize);
    let sub_band_step_count = (0..component_info.num_resolution_levels()
        - header.skipped_resolution_levels)
        .try_fold(0_usize, |total, resolution| {
            tile_decompositions
                .sub_band_iter(resolution, &storage.decompositions)
                .count()
                .checked_add(total)
                .ok_or(ValidationError::ImageTooLarge)
        })?;
    let step_capacity = sub_band_step_count
        .checked_add(active_decomposition_count)
        .and_then(|count| count.checked_add(1))
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut steps = Vec::new();
    budget.reserve_new(&mut steps, step_capacity)?;
    let mut next_band_id: J2kDirectBandId = 0;
    let mut sub_band_ids = Vec::new();
    budget.resize_new(&mut sub_band_ids, storage.sub_bands.len(), None)?;

    for resolution in 0..component_info.num_resolution_levels() - header.skipped_resolution_levels {
        let sub_band_iter = tile_decompositions.sub_band_iter(resolution, &storage.decompositions);
        for sub_band_idx in sub_band_iter {
            if let Some(step) = build_grayscale_sub_band_step(
                &storage.sub_bands[sub_band_idx],
                sub_band_idx,
                next_band_id,
                resolution,
                component_info,
                storage,
                header,
                budget,
            )? {
                sub_band_ids[sub_band_idx] = Some(next_band_id);
                next_band_id = next_band_id
                    .checked_add(1)
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                steps.push(step);
            }
        }
    }

    let mut current_ll_rect = storage.sub_bands[tile_decompositions.first_ll_sub_band].rect;
    let mut current_ll_band_id = sub_band_ids[tile_decompositions.first_ll_sub_band]
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let decompositions = &decompositions[..active_decomposition_count];
    for decomposition in decompositions {
        let hl = &storage.sub_bands[decomposition.sub_bands[0]];
        let lh = &storage.sub_bands[decomposition.sub_bands[1]];
        let hh = &storage.sub_bands[decomposition.sub_bands[2]];
        let output_band_id = next_band_id;
        next_band_id = next_band_id
            .checked_add(1)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        steps.push(J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
            output_band_id,
            rect: J2kRect::from(decomposition.rect),
            transform: J2kWaveletTransform::from(component_info.wavelet_transform()),
            ll_band_id: current_ll_band_id,
            ll: J2kRect::from(current_ll_rect),
            hl_band_id: sub_band_ids[decomposition.sub_bands[0]]
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            hl: J2kRect::from(hl.rect),
            lh_band_id: sub_band_ids[decomposition.sub_bands[1]]
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            lh: J2kRect::from(lh.rect),
            hh_band_id: sub_band_ids[decomposition.sub_bands[2]]
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            hh: J2kRect::from(hh.rect),
        }));
        current_ll_rect = decomposition.rect;
        current_ll_band_id = output_band_id;
    }

    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );
    let image_x_offset = header.size_data.image_area_x_offset;
    let image_y_offset = header.size_data.image_area_y_offset;
    let source_x = image_x_offset.saturating_sub(current_ll_rect.x0);
    let source_y = image_y_offset.saturating_sub(current_ll_rect.y0);
    let copy_width = resolution_tile
        .rect
        .width()
        .min(current_ll_rect.width().saturating_sub(source_x));
    let copy_height = resolution_tile
        .rect
        .height()
        .min(current_ll_rect.height().saturating_sub(source_y));
    let output_x = resolution_tile.rect.x0.saturating_sub(image_x_offset);
    let output_y = resolution_tile.rect.y0.saturating_sub(image_y_offset);
    steps.push(J2kDirectGrayscaleStep::Store(J2kDirectStoreStep {
        input_band_id: current_ll_band_id,
        input_rect: J2kRect::from(current_ll_rect),
        source_x,
        source_y,
        copy_width,
        copy_height,
        output_width: header.size_data.image_width(),
        output_height: header.size_data.image_height(),
        output_x,
        output_y,
        addend: store_addend,
    }));

    let sub_band_id_capacity = sub_band_ids.capacity();
    drop(sub_band_ids);
    budget.release_elements::<Option<J2kDirectBandId>>(sub_band_id_capacity)?;

    Ok(J2kDirectGrayscalePlan {
        dimensions: (
            header.size_data.image_width(),
            header.size_data.image_height(),
        ),
        bit_depth: component_info.size_info.precision,
        steps,
    })
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the direct-plan boundary keeps validated band identity, geometry, storage, and the shared live budget explicit"
)]
fn build_grayscale_sub_band_step(
    sub_band: &SubBand,
    sub_band_idx: usize,
    band_id: J2kDirectBandId,
    resolution: u8,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<Option<J2kDirectGrayscaleStep>> {
    let SubBandDecodeParameters {
        dequantization_step,
        num_bitplanes,
    } = sub_band_decode_parameters(sub_band, resolution, component_info)?;

    if component_info
        .coding_style
        .parameters
        .code_block_style
        .uses_high_throughput_block_coding()
    {
        let coded_bitplanes =
            add_roi_shift_to_bitplanes(num_bitplanes, component_info.roi_shift, 31)?;
        let stripe_causal = component_info
            .coding_style
            .parameters
            .code_block_style
            .vertically_causal_context;
        let job_capacity = direct_sub_band_job_capacity(sub_band, storage)?;
        let mut jobs = Vec::new();
        budget.reserve_new(&mut jobs, job_capacity)?;
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
                if !ht_code_block_has_decodable_passes(code_block, coded_bitplanes, header.strict)?
                {
                    continue;
                }

                let combined =
                    ht_block_decode::collect_code_block_data(code_block, storage, budget)?;
                jobs.push(HtOwnedCodeBlockBatchJob {
                    output_x: code_block.rect.x0 - sub_band.rect.x0,
                    output_y: code_block.rect.y0 - sub_band.rect.y0,
                    data: combined.data,
                    cleanup_length: combined.cleanup_length,
                    refinement_length: combined.refinement_length,
                    width: code_block.rect.width(),
                    height: code_block.rect.height(),
                    output_stride: sub_band.rect.width() as usize,
                    missing_bit_planes: code_block.missing_bit_planes,
                    number_of_coding_passes: code_block.number_of_coding_passes,
                    num_bitplanes,
                    roi_shift: component_info.roi_shift,
                    stripe_causal,
                    strict: header.strict,
                    dequantization_step,
                });
            }
        }

        return Ok(Some(J2kDirectGrayscaleStep::HtSubBand(
            HtOwnedSubBandPlan {
                band_id,
                rect: J2kRect::from(sub_band.rect),
                width: sub_band.rect.width(),
                height: sub_band.rect.height(),
                jobs,
            },
        )));
    }

    let (classic_job_sub_band_type, classic_job_style) =
        classic_decode_job_parameters(sub_band.sub_band_type, component_info);

    let job_capacity = direct_sub_band_job_capacity(sub_band, storage)?;
    let mut jobs = Vec::new();
    budget.reserve_new(&mut jobs, job_capacity)?;
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
                budget,
            )?;
            jobs.push(J2kOwnedCodeBlockBatchJob {
                output_x: code_block.rect.x0 - sub_band.rect.x0,
                output_y: code_block.rect.y0 - sub_band.rect.y0,
                data: combined_data,
                segments,
                width: code_block.rect.width(),
                height: code_block.rect.height(),
                output_stride: sub_band.rect.width() as usize,
                missing_bit_planes: code_block.missing_bit_planes,
                number_of_coding_passes: code_block.number_of_coding_passes,
                total_bitplanes: num_bitplanes,
                roi_shift: component_info.roi_shift,
                sub_band_type: classic_job_sub_band_type,
                style: classic_job_style,
                strict: header.strict,
                dequantization_step,
            });
        }
    }

    Ok(Some(J2kDirectGrayscaleStep::ClassicSubBand(
        J2kOwnedSubBandPlan {
            band_id,
            rect: J2kRect::from(sub_band.rect),
            width: sub_band.rect.width(),
            height: sub_band.rect.height(),
            jobs,
        },
    )))
}

fn direct_sub_band_job_capacity(
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
) -> Result<usize> {
    sub_band
        .precincts
        .clone()
        .map(|idx| storage.precincts[idx].code_blocks.len())
        .try_fold(0_usize, |total, count| {
            total
                .checked_add(count)
                .ok_or(ValidationError::ImageTooLarge.into())
        })
}
