// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{MetalRuntime, CommandBufferRef, PreparedDirectColorPlan, PixelFormat, DirectTier1Mode, DirectHybridStageTimings, Buffer, DirectStatusCheck, DirectScratchBuffer, Surface, Error, encode_prepared_direct_component_plane_in_command_buffer, DirectComponentPlaneRequest, metal_profile_stages_enabled, Instant, encode_mct_rgb8_to_surface_in_command_buffer, elapsed_us, dispatch_inverse_mct_buffers_in_command_buffer, PlaneStage, NativeColorSpace, encode_plane_stage_to_surface_in_command_buffer, J2kDirectBandId, BandRequiredRegion, size_of, DirectColorBatchCommandBuffers, Arc, repeated_shared_direct_color_plan_count, flattened_hybrid_cpu_tier1_enabled, should_flatten_hybrid_cpu_tier1_color_batch, build_flattened_cpu_tier1_cache, encode_repeated_mct_rgb8_to_surfaces_in_command_buffer, encode_batched_mct_rgb8_to_surfaces_in_command_buffer, PreparedDirectGrayscalePlan, classic_group_shapes_match, ht_group_shapes_match, PreparedDirectGrayscaleStep, classic_sub_band_shapes_match, ht_sub_band_shapes_match, idwt_shapes_match, store_shapes_match, FlattenedCpuTier1Cache, take_f32_scratch_buffer, encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer, ClassicCpuDecodeInput, CpuTier1DecodeSubstageCounters, decode_classic_inputs_on_cpu_with_plan_cache, upload_cpu_decoded_coefficients, encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer, HtCpuDecodeInput, decode_ht_inputs_on_cpu_with_plan_cache, direct_preflight_invariant, encode_distinct_classic_sub_bands_to_buffer_in_command_buffer, encode_distinct_ht_sub_bands_to_buffer_in_command_buffer, prepared_idwt_output_len, J2kWaveletTransform, repeated_idwt_params, idwt_input_windows_from_slices, PreparedIdwtInputStrides, IdwtSubBandBuffers, RepeatedIdwtDispatch, dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets, prepared_idwt_params, SingleIdwtDispatch, dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets, dispatch_store_component_repeated_in_command_buffer, J2kRepeatedStoreParams, record_hybrid_stacked_component_batch, encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer, encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer, encode_repeated_classic_sub_band_to_buffer_in_command_buffer, encode_repeated_ht_sub_band_to_buffer_in_command_buffer, DirectIdwtCommandBuffers, dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets, j2k_scalar_pack_params, encode_repeated_gray_store_to_surfaces_in_command_buffer, J2kRepeatedGrayStoreParams, encode_repeated_gray_plane_to_surfaces_in_command_buffer, J2kStoreParams, dispatch_store_component_buffer_in_command_buffer_with_offsets, encode_gray_plane_to_surface_in_command_buffer_with_offset};

#[cfg(target_os = "macos")]
pub(super) fn signed_sample_bias(bit_depth: u8) -> f32 {
    2.0_f32.powi(i32::from(bit_depth) - 1)
}

#[cfg(target_os = "macos")]
pub(super) struct DirectColorPlanRequest<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a CommandBufferRef,
    pub(super) plan: &'a PreparedDirectColorPlan,
    pub(super) fmt: PixelFormat,
    pub(super) tier1_mode: DirectTier1Mode,
    pub(super) stage_timings: &'a mut DirectHybridStageTimings,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_prepared_direct_color_plan_in_command_buffer(
    request: DirectColorPlanRequest<'_>,
) -> Result<Surface, Error> {
    let DirectColorPlanRequest {
        runtime,
        command_buffer,
        plan,
        fmt,
        tier1_mode,
        stage_timings,
        retained_buffers,
        retained_cpu_coefficients,
        status_checks,
        scratch_buffers,
    } = request;
    if plan.component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color execution expected 3 component plans, got {}",
                plan.component_plans.len()
            ),
        });
    }

    let mut planes = Vec::with_capacity(3);
    for component_plan in &plan.component_plans {
        planes.push(encode_prepared_direct_component_plane_in_command_buffer(
            DirectComponentPlaneRequest {
            runtime,
            command_buffer,
            plan: component_plan,
            tier1_mode,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
            status_checks,
            scratch_buffers,
        })?);
    }

    if plan.mct && fmt == PixelFormat::Rgb8 {
        let encode_started = metal_profile_stages_enabled().then(Instant::now);
        let surface = encode_mct_rgb8_to_surface_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            plan.dimensions,
            plan.bit_depths,
            plan.transform,
        )?;
        if let Some(started) = encode_started {
            stage_timings.metal_mct_pack_encode += elapsed_us(started);
        }
        return Ok(surface);
    }

    if plan.mct {
        let len = plan.dimensions.0 as usize * plan.dimensions.1 as usize;
        let encode_started = metal_profile_stages_enabled().then(Instant::now);
        status_checks.push(dispatch_inverse_mct_buffers_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            len,
            plan.transform,
            [
                signed_sample_bias(plan.bit_depths[0]),
                signed_sample_bias(plan.bit_depths[1]),
                signed_sample_bias(plan.bit_depths[2]),
            ],
        )?);
        if let Some(started) = encode_started {
            stage_timings.metal_mct_pack_encode += elapsed_us(started);
        }
    }

    let stage = PlaneStage {
        dims: plan.dimensions,
        plane_count: 3,
        color_space: NativeColorSpace::RGB,
        has_alpha: false,
        bit_depths: [
            u32::from(plan.bit_depths[0]),
            u32::from(plan.bit_depths[1]),
            u32::from(plan.bit_depths[2]),
            0,
        ],
        planes: [
            Some(planes[0].clone()),
            Some(planes[1].clone()),
            Some(planes[2].clone()),
            None,
        ],
    };
    let encode_started = metal_profile_stages_enabled().then(Instant::now);
    let surface =
        encode_plane_stage_to_surface_in_command_buffer(runtime, command_buffer, &stage, fmt);
    if let Some(started) = encode_started {
        stage_timings.metal_mct_pack_encode += elapsed_us(started);
    }
    surface
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(super) struct DirectBandSlice {
    pub(super) band_id: J2kDirectBandId,
    pub(super) buffer: Buffer,
    pub(super) offset_bytes: usize,
    pub(super) window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
pub(super) fn lookup_direct_band_slice_entry(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<DirectBandSlice, Error> {
    bands
        .iter()
        .find(|existing| existing.band_id == band_id)
        .cloned()
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "missing J2K MetalDirect device band {} for rect ({}, {}, {}, {})",
                band_id, rect.x0, rect.y0, rect.x1, rect.y1
            ),
        })
}

#[cfg(target_os = "macos")]
pub(super) fn lookup_direct_band_slice(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<(Buffer, usize), Error> {
    let entry = lookup_direct_band_slice_entry(bands, band_id, rect)?;
    Ok((entry.buffer, entry.offset_bytes))
}

#[cfg(target_os = "macos")]
pub(super) fn lookup_repeated_direct_band_layout_entry(
    band_sets: &[Vec<DirectBandSlice>],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<(DirectBandSlice, u32), Error> {
    let first_bands = band_sets.first().ok_or_else(|| Error::MetalKernel {
        message: "missing J2K MetalDirect repeated band set".to_string(),
    })?;
    let entry = lookup_direct_band_slice_entry(first_bands, band_id, rect)?;
    let stride_bytes = if let Some(second_bands) = band_sets.get(1) {
        let next = lookup_direct_band_slice_entry(second_bands, band_id, rect)?;
        next.offset_bytes
            .checked_sub(entry.offset_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect repeated band offsets are not monotonic".to_string(),
            })?
    } else {
        entry.window.width() as usize * entry.window.height() as usize * size_of::<f32>()
    };
    if stride_bytes % size_of::<f32>() != 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride is not f32-aligned".to_string(),
        });
    }
    let stride_elements =
        u32::try_from(stride_bytes / size_of::<f32>()).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride exceeds u32".to_string(),
        })?;
    Ok((entry, stride_elements))
}

#[cfg(target_os = "macos")]
pub(super) struct StackedDirectComponentPlane {
    pub(super) buffer: Buffer,
    pub(super) dimensions: (u32, u32),
    pub(super) count: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct StackedDirectColorBatchRequest<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffers: DirectColorBatchCommandBuffers<'a>,
    pub(super) plans: &'a [Arc<PreparedDirectColorPlan>],
    pub(super) tier1_mode: DirectTier1Mode,
    pub(super) force_flattened_cpu_tier1: bool,
    pub(super) stage_timings: &'a mut DirectHybridStageTimings,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(super) fn try_encode_stacked_mct_rgb8_direct_color_batch(
    request: StackedDirectColorBatchRequest<'_>,
) -> Result<Option<Vec<Surface>>, Error> {
    let StackedDirectColorBatchRequest {
        runtime,
        command_buffers,
        plans,
        tier1_mode,
        force_flattened_cpu_tier1,
        stage_timings,
        retained_buffers,
        retained_cpu_coefficients,
        status_checks,
        scratch_buffers,
    } = request;
    let Some(first) = plans.first() else {
        return Ok(Some(Vec::new()));
    };
    let repeated_count = repeated_shared_direct_color_plan_count(plans);
    if plans.len() <= 1
        || !first.mct
        || first.component_plans.len() != 3
        || !plans.iter().all(|plan| {
            plan.mct
                && plan.dimensions == first.dimensions
                && plan.bit_depths == first.bit_depths
                && plan.transform == first.transform
                && plan.component_plans.len() == 3
        })
    {
        return Ok(None);
    }
    let execution_plans = if repeated_count.is_some() {
        &plans[..1]
    } else {
        plans
    };

    let flattened_cpu_tier1_cache = if tier1_mode == DirectTier1Mode::CpuUpload
        && (force_flattened_cpu_tier1
            || flattened_hybrid_cpu_tier1_enabled()
            || should_flatten_hybrid_cpu_tier1_color_batch(execution_plans))
    {
        Some(build_flattened_cpu_tier1_cache(
            runtime,
            execution_plans,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
        )?)
    } else {
        None
    };

    let mut stacked_planes = Vec::with_capacity(3);
    for component_idx in 0..3 {
        let component_plan_refs = execution_plans
            .iter()
            .map(|plan| &plan.component_plans[component_idx])
            .collect::<Vec<_>>();
        if !supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            return Ok(None);
        }
        stacked_planes.push(encode_stacked_direct_component_plane_batch(
            StackedDirectComponentPlaneBatchRequest {
            runtime,
            command_buffers,
            plans: &component_plan_refs,
            component_idx,
            flattened_cpu_tier1_cache: flattened_cpu_tier1_cache.as_ref(),
            tier1_mode,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
            status_checks,
            scratch_buffers,
        })?);
    }

    if !stacked_planes
        .iter()
        .all(|plane| plane.dimensions == first.dimensions && plane.count == execution_plans.len())
    {
        return Ok(None);
    }

    let encode_started = metal_profile_stages_enabled().then(Instant::now);
    let mct_plane_buffers = [
        &stacked_planes[0].buffer,
        &stacked_planes[1].buffer,
        &stacked_planes[2].buffer,
    ];
    let surfaces = if let Some(count) = repeated_count {
        encode_repeated_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            first.dimensions,
            count,
            first.bit_depths,
            first.transform,
        )?
    } else {
        encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            first.dimensions,
            execution_plans.len(),
            first.bit_depths,
            first.transform,
        )?
    };
    if let Some(started) = encode_started {
        stage_timings.metal_mct_pack_encode += elapsed_us(started);
    }
    Ok(Some(surfaces))
}

#[cfg(target_os = "macos")]
pub(super) fn supports_stacked_direct_component_plane_batch(plans: &[&PreparedDirectGrayscalePlan]) -> bool {
    let Some(first) = plans.first() else {
        return false;
    };
    if plans.iter().any(|plan| {
        plan.dimensions != first.dimensions
            || plan.bit_depth != first.bit_depth
            || plan.steps.len() != first.steps.len()
    }) {
        return false;
    }

    let mut step_idx = 0;
    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            if group.end_step <= step_idx
                || !plans.iter().all(|plan| {
                    plan.classic_group_starting_at(step_idx)
                        .is_some_and(|other| classic_group_shapes_match(group, other))
                })
            {
                return false;
            }
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = first.ht_group_starting_at(step_idx) {
            if group.end_step <= step_idx
                || !plans.iter().all(|plan| {
                    plan.ht_group_starting_at(step_idx)
                        .is_some_and(|other| ht_group_shapes_match(group, other))
                })
            {
                return false;
            }
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::ClassicSubBand(other)
                            if classic_sub_band_shapes_match(sub_band, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::HtSubBand(other)
                            if ht_sub_band_shapes_match(sub_band, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Idwt(other)
                            if idwt_shapes_match(idwt, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Store(other)
                            if store_shapes_match(store, other)
                    )
                }) {
                    return false;
                }
            }
        }
        step_idx += 1;
    }

    true
}

#[cfg(target_os = "macos")]
pub(super) struct StackedDirectComponentPlaneBatchRequest<'a, 'p> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffers: DirectColorBatchCommandBuffers<'a>,
    pub(super) plans: &'a [&'p PreparedDirectGrayscalePlan],
    pub(super) component_idx: usize,
    pub(super) flattened_cpu_tier1_cache: Option<&'a FlattenedCpuTier1Cache>,
    pub(super) tier1_mode: DirectTier1Mode,
    pub(super) stage_timings: &'a mut DirectHybridStageTimings,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) retained_cpu_coefficients: &'a mut Vec<Vec<f32>>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_stacked_direct_component_plane_batch(
    request: StackedDirectComponentPlaneBatchRequest<'_, '_>,
) -> Result<StackedDirectComponentPlane, Error> {
    let StackedDirectComponentPlaneBatchRequest {
        runtime,
        command_buffers,
        plans,
        component_idx,
        flattened_cpu_tier1_cache,
        tier1_mode,
        stage_timings,
        retained_buffers,
        retained_cpu_coefficients,
        status_checks,
        scratch_buffers,
    } = request;
    let Some(first) = plans.first() else {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color batch has no component plans".to_string(),
        });
    };

    let count = plans.len();
    let broadcast_tier1_inputs = tier1_mode == DirectTier1Mode::CpuUpload
        && plans.iter().all(|plan| std::ptr::eq(*plan, *first));
    let mut band_sets = vec![Vec::<DirectBandSlice>::new(); count];
    let mut final_plane = None;
    let mut step_idx = 0;
    let profile_stages = tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();

    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            let groups = plans
                .iter()
                .map(|plan| {
                    plan.classic_group_starting_at(step_idx)
                        .expect("preflight validated classic group")
                })
                .collect::<Vec<_>>();
            let buffer = match tier1_mode {
                DirectTier1Mode::Metal => {
                    let output =
                        take_f32_scratch_buffer(runtime, group.total_coefficients * count)?;
                    let (buffers, status_check) =
                        encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
                            runtime,
                            command_buffers.default,
                            &groups,
                            &output.buffer,
                            scratch_buffers,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    let buffer = output.buffer.clone();
                    scratch_buffers.push(output);
                    buffer
                }
                DirectTier1Mode::CpuUpload => {
                    let input_groups = if broadcast_tier1_inputs {
                        &groups[..1]
                    } else {
                        &groups
                    };
                    if let Some(cache) = flattened_cpu_tier1_cache {
                        cache.buffer_for(
                            component_idx,
                            step_idx,
                            group.total_coefficients,
                            input_groups.len(),
                        )?
                    } else {
                        let inputs = input_groups
                            .iter()
                            .map(|group| ClassicCpuDecodeInput {
                                coded_data: &group.coded_data,
                                segments: &group.segments,
                                jobs: &group.jobs,
                                output_len: group.total_coefficients,
                            })
                            .collect::<Vec<_>>();
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_classic_inputs_on_cpu_with_plan_cache(
                            first,
                            step_idx,
                            &inputs,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                }
            };
            let stride_bytes = group.total_coefficients * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                let source_group = if broadcast_tier1_inputs {
                    groups[0]
                } else {
                    groups[instance_idx]
                };
                let instance_offset = if broadcast_tier1_inputs {
                    0
                } else {
                    instance_idx * stride_bytes
                };
                for member in &source_group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            step_idx = group.end_step;
            continue;
        }

        if let Some(group) = first.ht_group_starting_at(step_idx) {
            let groups = plans
                .iter()
                .map(|plan| {
                    plan.ht_group_starting_at(step_idx)
                        .expect("preflight validated HT group")
                })
                .collect::<Vec<_>>();
            let buffer = match tier1_mode {
                DirectTier1Mode::Metal => {
                    let output =
                        take_f32_scratch_buffer(runtime, group.total_coefficients * count)?;
                    let (buffers, status_check) =
                        encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
                            runtime,
                            command_buffers.default,
                            &groups,
                            &output.buffer,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    let buffer = output.buffer.clone();
                    scratch_buffers.push(output);
                    buffer
                }
                DirectTier1Mode::CpuUpload => {
                    let input_groups = if broadcast_tier1_inputs {
                        &groups[..1]
                    } else {
                        &groups
                    };
                    if let Some(cache) = flattened_cpu_tier1_cache {
                        cache.buffer_for(
                            component_idx,
                            step_idx,
                            group.total_coefficients,
                            input_groups.len(),
                        )?
                    } else {
                        let inputs = input_groups
                            .iter()
                            .map(|group| HtCpuDecodeInput {
                                coded_data: &group.coded_arena.data,
                                jobs: &group.jobs,
                                output_len: group.total_coefficients,
                            })
                            .collect::<Vec<_>>();
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_ht_inputs_on_cpu_with_plan_cache(
                            first,
                            step_idx,
                            &inputs,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                }
            };
            let stride_bytes = group.total_coefficients * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                let source_group = if broadcast_tier1_inputs {
                    groups[0]
                } else {
                    groups[instance_idx]
                };
                let instance_offset = if broadcast_tier1_inputs {
                    0
                } else {
                    instance_idx * stride_bytes
                };
                for member in &source_group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let sub_bands = plans
                    .iter()
                    .map(|plan| match &plan.steps[step_idx] {
                        PreparedDirectGrayscaleStep::ClassicSubBand(other) => Ok(other),
                        _ => Err(direct_preflight_invariant(
                            "classic sub-band step mismatch in stacked component batch",
                        )),
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                        let (buffers, status_check) =
                            encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
                                runtime,
                                command_buffers.default,
                                &sub_bands,
                                &output.buffer,
                                scratch_buffers,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let input_sub_bands = if broadcast_tier1_inputs {
                            &sub_bands[..1]
                        } else {
                            &sub_bands
                        };
                        if let Some(cache) = flattened_cpu_tier1_cache {
                            cache.buffer_for(
                                component_idx,
                                step_idx,
                                per_instance_len,
                                input_sub_bands.len(),
                            )?
                        } else {
                            let inputs = input_sub_bands
                                .iter()
                                .map(|sub_band| ClassicCpuDecodeInput {
                                    coded_data: &sub_band.coded_data,
                                    segments: &sub_band.segments,
                                    jobs: &sub_band.jobs,
                                    output_len: per_instance_len,
                                })
                                .collect::<Vec<_>>();
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_classic_inputs_on_cpu_with_plan_cache(
                                first,
                                step_idx,
                                &inputs,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    }
                };
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let source_sub_band = if broadcast_tier1_inputs {
                        sub_bands[0]
                    } else {
                        sub_bands[instance_idx]
                    };
                    let instance_offset = if broadcast_tier1_inputs {
                        0
                    } else {
                        instance_idx * stride_bytes
                    };
                    bands.push(DirectBandSlice {
                        band_id: source_sub_band.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset,
                        window: BandRequiredRegion::full(
                            source_sub_band.width,
                            source_sub_band.height,
                        ),
                    });
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let sub_bands = plans
                    .iter()
                    .map(|plan| match &plan.steps[step_idx] {
                        PreparedDirectGrayscaleStep::HtSubBand(other) => Ok(other),
                        _ => Err(direct_preflight_invariant(
                            "HT sub-band step mismatch in stacked component batch",
                        )),
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                        let (buffers, status_check) =
                            encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
                                runtime,
                                command_buffers.default,
                                &sub_bands,
                                &output.buffer,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let input_sub_bands = if broadcast_tier1_inputs {
                            &sub_bands[..1]
                        } else {
                            &sub_bands
                        };
                        if let Some(cache) = flattened_cpu_tier1_cache {
                            cache.buffer_for(
                                component_idx,
                                step_idx,
                                per_instance_len,
                                input_sub_bands.len(),
                            )?
                        } else {
                            let inputs = input_sub_bands
                                .iter()
                                .map(|sub_band| HtCpuDecodeInput {
                                    coded_data: &sub_band.coded_data,
                                    jobs: &sub_band.jobs,
                                    output_len: per_instance_len,
                                })
                                .collect::<Vec<_>>();
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_ht_inputs_on_cpu_with_plan_cache(
                                first,
                                step_idx,
                                &inputs,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    }
                };
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let source_sub_band = if broadcast_tier1_inputs {
                        sub_bands[0]
                    } else {
                        sub_bands[instance_idx]
                    };
                    let instance_offset = if broadcast_tier1_inputs {
                        0
                    } else {
                        instance_idx * stride_bytes
                    };
                    bands.push(DirectBandSlice {
                        band_id: source_sub_band.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset,
                        window: BandRequiredRegion::full(
                            source_sub_band.width,
                            source_sub_band.height,
                        ),
                    });
                }
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                let per_instance_len = prepared_idwt_output_len(idwt);
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let encode_started = profile_stages.then(Instant::now);
                match idwt.step.transform {
                    J2kWaveletTransform::Reversible53 => {
                        let (ll, low_low_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.ll_band_id,
                            idwt.step.ll,
                        )?;
                        let (hl, high_low_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.hl_band_id,
                            idwt.step.hl,
                        )?;
                        let (lh, low_high_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.lh_band_id,
                            idwt.step.lh,
                        )?;
                        let (hh, high_high_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.hh_band_id,
                            idwt.step.hh,
                        )?;
                        let params = repeated_idwt_params(
                            idwt,
                            idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                            PreparedIdwtInputStrides {
                                ll: low_low_stride,
                                hl: high_low_stride,
                                lh: low_high_stride,
                                hh: high_high_stride,
                            },
                            count,
                            "color",
                        )?;
                        dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                            command_buffers.idwt,
                            RepeatedIdwtDispatch {
                                runtime,
                                sub_bands: IdwtSubBandBuffers {
                                    ll: &ll.buffer,
                                    ll_offset: ll.offset_bytes,
                                    hl: &hl.buffer,
                                    hl_offset: hl.offset_bytes,
                                    lh: &lh.buffer,
                                    lh_offset: lh.offset_bytes,
                                    hh: &hh.buffer,
                                    hh_offset: hh.offset_bytes,
                                },
                                params,
                                decoded: &output.buffer,
                            },
                        );
                    }
                    J2kWaveletTransform::Irreversible97 => {
                        for (instance_idx, bands) in band_sets.iter().enumerate() {
                            let PreparedDirectGrayscaleStep::Idwt(step) =
                                &plans[instance_idx].steps[step_idx]
                            else {
                                return Err(direct_preflight_invariant(
                                    "IDWT step mismatch in stacked component batch",
                                ));
                            };
                            let ll = lookup_direct_band_slice_entry(
                                bands,
                                step.step.ll_band_id,
                                step.step.ll,
                            )?;
                            let hl = lookup_direct_band_slice_entry(
                                bands,
                                step.step.hl_band_id,
                                step.step.hl,
                            )?;
                            let lh = lookup_direct_band_slice_entry(
                                bands,
                                step.step.lh_band_id,
                                step.step.lh,
                            )?;
                            let hh = lookup_direct_band_slice_entry(
                                bands,
                                step.step.hh_band_id,
                                step.step.hh,
                            )?;
                            let params = prepared_idwt_params(
                                step,
                                idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                            );
                            status_checks.push(
                                dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                    command_buffers.idwt.interleave,
                                    SingleIdwtDispatch {
                                        runtime,
                                        sub_bands: IdwtSubBandBuffers {
                                            ll: &ll.buffer,
                                            ll_offset: ll.offset_bytes,
                                            hl: &hl.buffer,
                                            hl_offset: hl.offset_bytes,
                                            lh: &lh.buffer,
                                            lh_offset: lh.offset_bytes,
                                            hh: &hh.buffer,
                                            hh_offset: hh.offset_bytes,
                                        },
                                        params,
                                        decoded: &output.buffer,
                                        decoded_offset: instance_idx * per_instance_len * size_of::<f32>(),
                                    },
                                ),
                            );
                        }
                    }
                }
                if let Some(started) = encode_started {
                    stage_timings.metal_idwt_encode += elapsed_us(started);
                }
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let PreparedDirectGrayscaleStep::Idwt(step) =
                        &plans[instance_idx].steps[step_idx]
                    else {
                        return Err(direct_preflight_invariant(
                            "IDWT output step mismatch in stacked component batch",
                        ));
                    };
                    bands.push(DirectBandSlice {
                        band_id: step.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: step.output_window,
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                let (input, input_instance_stride) = lookup_repeated_direct_band_layout_entry(
                    &band_sets,
                    store.input_band_id,
                    store.input_rect,
                )?;
                let per_instance_len = store.output_width as usize * store.output_height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let encode_started = profile_stages.then(Instant::now);
                dispatch_store_component_repeated_in_command_buffer(
                    runtime,
                    command_buffers.store,
                    &input.buffer,
                    input.offset_bytes,
                    &output.buffer,
                    J2kRepeatedStoreParams {
                        input_width: store.input_rect.width(),
                        input_height: store.input_rect.height(),
                        input_instance_stride,
                        source_x: store.source_x,
                        source_y: store.source_y,
                        copy_width: store.copy_width,
                        copy_height: store.copy_height,
                        output_width: store.output_width,
                        output_height: store.output_height,
                        output_x: store.output_x,
                        output_y: store.output_y,
                        addend: store.addend,
                        batch_count: u32::try_from(count).map_err(|_| Error::MetalKernel {
                            message: "J2K MetalDirect color store batch count exceeds u32"
                                .to_string(),
                        })?,
                    },
                );
                if let Some(started) = encode_started {
                    stage_timings.metal_store_encode += elapsed_us(started);
                }
                final_plane = Some(output.buffer.clone());
                scratch_buffers.push(output);
            }
        }
        step_idx += 1;
    }

    let buffer = final_plane.ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color component batch did not produce a final plane".to_string(),
    })?;
    record_hybrid_stacked_component_batch(tier1_mode);
    Ok(StackedDirectComponentPlane {
        buffer,
        dimensions: first.dimensions,
        count,
    })
}

#[cfg(target_os = "macos")]
pub(super) struct RepeatedDirectGrayscalePlanRequest<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a CommandBufferRef,
    pub(super) plan: &'a PreparedDirectGrayscalePlan,
    pub(super) fmt: PixelFormat,
    pub(super) count: usize,
    pub(super) retained_buffers: &'a mut Vec<Buffer>,
    pub(super) status_checks: &'a mut Vec<DirectStatusCheck>,
    pub(super) scratch_buffers: &'a mut Vec<DirectScratchBuffer>,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_repeated_direct_grayscale_plan_in_command_buffer(
    request: RepeatedDirectGrayscalePlanRequest<'_>,
) -> Result<Vec<Surface>, Error> {
    let RepeatedDirectGrayscalePlanRequest {
        runtime,
        command_buffer,
        plan,
        fmt,
        count,
        retained_buffers,
        status_checks,
        scratch_buffers,
    } = request;
    let mut band_sets = vec![Vec::<DirectBandSlice>::new(); count];
    let mut surfaces = Vec::with_capacity(count);
    let mut stacked_outputs = true;
    let mut step_idx = 0;

    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
            let (buffers, status_check) =
                encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                    scratch_buffers,
                )?;
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
            let (buffers, status_check) =
                encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                )?;
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        let step = &plan.steps[step_idx];
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let (buffers, status_check) =
                    encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                let (buffers, status_check) =
                    encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => match idwt.step.transform {
                J2kWaveletTransform::Reversible53 if stacked_outputs => {
                    let (ll, low_low_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.ll_band_id,
                        idwt.step.ll,
                    )?;
                    let (hl, high_low_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.hl_band_id,
                        idwt.step.hl,
                    )?;
                    let (lh, low_high_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.lh_band_id,
                        idwt.step.lh,
                    )?;
                    let (hh, high_high_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.hh_band_id,
                        idwt.step.hh,
                    )?;
                    let params = repeated_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                        PreparedIdwtInputStrides {
                            ll: low_low_stride,
                            hl: high_low_stride,
                            lh: low_high_stride,
                            hh: high_high_stride,
                        },
                        count,
                        "repeated",
                    )?;
                    let per_instance_len = prepared_idwt_output_len(idwt);
                    let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                        DirectIdwtCommandBuffers::single(command_buffer),
                        RepeatedIdwtDispatch {
                            runtime,
                            sub_bands: IdwtSubBandBuffers {
                                ll: &ll.buffer,
                                ll_offset: ll.offset_bytes,
                                hl: &hl.buffer,
                                hl_offset: hl.offset_bytes,
                                lh: &lh.buffer,
                                lh_offset: lh.offset_bytes,
                                hh: &hh.buffer,
                                hh_offset: hh.offset_bytes,
                            },
                            params,
                            decoded: &output.buffer,
                        },
                    );
                    let stride_bytes = per_instance_len * size_of::<f32>();
                    for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                        bands.push(DirectBandSlice {
                            band_id: idwt.step.output_band_id,
                            buffer: output.buffer.clone(),
                            offset_bytes: instance_idx * stride_bytes,
                            window: idwt.output_window,
                        });
                    }
                    scratch_buffers.push(output);
                }
                _ => {
                    stacked_outputs = false;
                    for bands in &mut band_sets {
                        let ll = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.ll_band_id,
                            idwt.step.ll,
                        )?;
                        let hl = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.hl_band_id,
                            idwt.step.hl,
                        )?;
                        let lh = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.lh_band_id,
                            idwt.step.lh,
                        )?;
                        let hh = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.hh_band_id,
                            idwt.step.hh,
                        )?;
                        let params = prepared_idwt_params(
                            idwt,
                            idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                        );
                        let output =
                            take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt))?;
                        match idwt.step.transform {
                                J2kWaveletTransform::Reversible53 => {
                                    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        command_buffer,
                                        SingleIdwtDispatch {
                                            runtime,
                                            sub_bands: IdwtSubBandBuffers {
                                                ll: &ll.buffer,
                                                ll_offset: ll.offset_bytes,
                                                hl: &hl.buffer,
                                                hl_offset: hl.offset_bytes,
                                                lh: &lh.buffer,
                                                lh_offset: lh.offset_bytes,
                                                hh: &hh.buffer,
                                                hh_offset: hh.offset_bytes,
                                            },
                                            params,
                                            decoded: &output.buffer,
                                            decoded_offset: 0,
                                        },
                                    );
                                }
                                J2kWaveletTransform::Irreversible97 => status_checks.push(
                                    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        command_buffer,
                                        SingleIdwtDispatch {
                                            runtime,
                                            sub_bands: IdwtSubBandBuffers {
                                                ll: &ll.buffer,
                                                ll_offset: ll.offset_bytes,
                                                hl: &hl.buffer,
                                                hl_offset: hl.offset_bytes,
                                                lh: &lh.buffer,
                                                lh_offset: lh.offset_bytes,
                                                hh: &hh.buffer,
                                                hh_offset: hh.offset_bytes,
                                            },
                                            params,
                                            decoded: &output.buffer,
                                            decoded_offset: 0,
                                        },
                                    ),
                                ),
                            }
                        bands.push(DirectBandSlice {
                            band_id: idwt.step.output_band_id,
                            buffer: output.buffer.clone(),
                            offset_bytes: 0,
                            window: idwt.output_window,
                        });
                        scratch_buffers.push(output);
                    }
                }
            },
            PreparedDirectGrayscaleStep::Store(store) => {
                if stacked_outputs {
                    let (input, _) = lookup_direct_band_slice(
                        &band_sets[0],
                        store.input_band_id,
                        store.input_rect,
                    )?;
                    let batch_count = u32::try_from(count).map_err(|_| Error::MetalKernel {
                        message: "J2K MetalDirect repeated store batch count exceeds u32"
                            .to_string(),
                    })?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        surfaces.extend(encode_repeated_gray_store_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            J2kRepeatedGrayStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                            count,
                        )?);
                    } else {
                        let per_instance_len =
                            store.output_width as usize * store.output_height as usize;
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count)?;
                        dispatch_store_component_repeated_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            0,
                            &output.buffer,
                            J2kRepeatedStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                input_instance_stride: store
                                    .input_rect
                                    .width()
                                    .checked_mul(store.input_rect.height())
                                    .ok_or_else(|| Error::MetalKernel {
                                        message: "J2K MetalDirect repeated store input stride overflows u32"
                                            .to_string(),
                                    })?,
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                            },
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.extend(encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                            count,
                        )?);
                        scratch_buffers.push(output);
                    }
                } else {
                    for bands in &band_sets {
                        let (input, input_offset) =
                            lookup_direct_band_slice(bands, store.input_band_id, store.input_rect)?;
                        let output = take_f32_scratch_buffer(
                            runtime,
                            store.output_width as usize * store.output_height as usize,
                        )?;
                        let params = J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        };
                        dispatch_store_component_buffer_in_command_buffer_with_offsets(
                            runtime,
                            command_buffer,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.push(encode_gray_plane_to_surface_in_command_buffer_with_offset(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            0,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
        }
        step_idx += 1;
    }

    if surfaces.len() != count {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect repeated grayscale plan produced {} surfaces for count {}",
                surfaces.len(),
                count
            ),
        });
    }

    Ok(surfaces)
}
