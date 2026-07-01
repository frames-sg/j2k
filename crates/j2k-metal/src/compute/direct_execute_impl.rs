// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct PreparedDirectGrayscalePlan {
    dimensions: (u32, u32),
    bit_depth: u8,
    tier1_prepare_mode: DirectTier1Mode,
    steps: Vec<PreparedDirectGrayscaleStep>,
    classic_groups: Vec<PreparedClassicSubBandGroup>,
    ht_groups: Vec<PreparedHtSubBandGroup>,
    cpu_tier1_cache: Arc<CpuTier1CoefficientCache>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct PreparedDirectColorPlan {
    dimensions: (u32, u32),
    bit_depths: [u8; 3],
    mct: bool,
    transform: J2kWaveletTransform,
    component_plans: Vec<PreparedDirectGrayscalePlan>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
enum PreparedDirectGrayscaleStep {
    ClassicSubBand(PreparedClassicSubBand),
    HtSubBand(PreparedHtSubBand),
    Idwt(PreparedDirectIdwt),
    Store(J2kDirectStoreStep),
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedDirectIdwt {
    step: J2kDirectIdwtStep,
    output_window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBand {
    band_id: J2kDirectBandId,
    width: u32,
    height: u32,
    zero_fill: bool,
    coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    jobs_buffer: Buffer,
    segments: Vec<J2kClassicSegment>,
    segments_buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBandGroup {
    start_step: usize,
    end_step: usize,
    total_coefficients: usize,
    zero_fill: bool,
    coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    jobs_buffer: Buffer,
    segments: Vec<J2kClassicSegment>,
    segments_buffer: Buffer,
    members: Vec<PreparedClassicSubBandGroupMember>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBandGroupMember {
    band_id: J2kDirectBandId,
    offset_elements: usize,
    window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBand {
    band_id: J2kDirectBandId,
    width: u32,
    height: u32,
    coded_data: Vec<u8>,
    coded_buffer: Option<Buffer>,
    jobs: Vec<J2kHtCleanupBatchJob>,
    jobs_buffer: Option<Buffer>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct HtCodedArena {
    data: Vec<u8>,
    buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBandGroup {
    start_step: usize,
    end_step: usize,
    total_coefficients: usize,
    coded_arena: HtCodedArena,
    jobs: Vec<J2kHtCleanupBatchJob>,
    jobs_buffer: Buffer,
    members: Vec<PreparedHtSubBandGroupMember>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBandGroupMember {
    band_id: J2kDirectBandId,
    offset_elements: usize,
    window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
struct PlaneStage {
    dims: (u32, u32),
    plane_count: usize,
    color_space: NativeColorSpace,
    has_alpha: bool,
    bit_depths: [u32; 4],
    planes: [Option<Buffer>; 4],
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    fn from_planes(
        device: &Device,
        decoded: &NativeDecodedComponents<'_>,
        roi: Option<Rect>,
    ) -> Result<Self, Error> {
        let full_dims = decoded.dimensions();
        let roi = roi.unwrap_or(Rect {
            x: 0,
            y: 0,
            w: full_dims.0,
            h: full_dims.1,
        });
        let dims = (roi.w, roi.h);
        let plane_count = decoded.planes().len();
        if plane_count == 0 || plane_count > 4 {
            return Err(Error::MetalKernel {
                message: format!("unsupported J2K plane count {plane_count}"),
            });
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, plane) in decoded.planes().iter().enumerate() {
            bit_depths[index] = u32::from(plane.bit_depth());
            let len = dims.0 as usize * dims.1 as usize;
            let buffer = device.new_buffer(
                (len * size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            copy_plane_samples(&buffer, plane.samples(), full_dims.0 as usize, roi);
            planes[index] = Some(buffer);
        }

        Ok(Self {
            dims,
            plane_count,
            color_space: decoded.color_space().clone(),
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    fn from_captured_planes(
        decoded: &NativeDecodedComponents<'_>,
        captured_planes: Vec<Buffer>,
    ) -> Option<Self> {
        let plane_count = decoded.planes().len();
        let supported_shape = matches!(
            (decoded.color_space(), decoded.has_alpha(), plane_count),
            (NativeColorSpace::Gray, false, 1) | (NativeColorSpace::RGB, false, 3)
        );
        if !supported_shape {
            return None;
        }
        if captured_planes.len() != plane_count || plane_count == 0 || plane_count > 4 {
            return None;
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, (plane, buffer)) in decoded.planes().iter().zip(captured_planes).enumerate() {
            bit_depths[index] = u32::from(plane.bit_depth());
            planes[index] = Some(buffer);
        }

        Some(Self {
            dims: decoded.dimensions(),
            plane_count,
            color_space: decoded.color_space().clone(),
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        let command_buffer = runtime.queue.new_command_buffer();
        let surface =
            encode_plane_stage_to_surface_in_command_buffer(runtime, command_buffer, &self, fmt)?;
        commit_and_wait_metal(command_buffer)?;
        Ok(surface)
    }
}

#[cfg(target_os = "macos")]
fn encode_plane_stage_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    stage: &PlaneStage,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = stage.dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * stage.dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) = output_shape_for(
        &stage.color_space,
        stage.has_alpha,
        stage.plane_count,
        fmt,
        runtime,
    )?;
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays(stage.bit_depths);

    let params = J2kPackParams {
        width: stage.dims.0,
        height: stage.dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        output_channels,
        opaque_alpha,
        max_values,
        u8_scales,
        u16_scales,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid plane pack");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(
        0,
        stage.planes[0].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        1,
        stage.planes[1].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        2,
        stage.planes[2].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        3,
        stage.planes[3].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(4, Some(&out_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, stage.dims);
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, stage.dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_mct_rgb8_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        transform: mct_transform_code(transform),
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid MCT RGB8 pack");
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, &runtime.pack_mct_rgb8, dims);
    encoder.end_encoding();
    drop(signpost);

    Ok(Surface::from_metal_buffer(
        out_buffer,
        dims,
        PixelFormat::Rgb8,
    ))
}

#[cfg(target_os = "macos")]
fn encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    count: usize,
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Vec<Surface>, Error> {
    let count_u32 = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "J2K MetalDirect color batch count exceeds u32".to_string(),
    })?;
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect color batch output size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect color batch output size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let plane_stride = dims
        .0
        .checked_mul(dims.1)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect color batch plane stride overflow".to_string(),
        })?;
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kBatchedMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        transform: mct_transform_code(transform),
        batch_count: count_u32,
        plane_stride,
        output_stride: u32::try_from(surface_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect color batch surface stride exceeds u32".to_string(),
        })?,
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid batched MCT RGB8 pack");
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8_batched);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kBatchedMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        &runtime.pack_mct_rgb8_batched,
        (dims.0, dims.1, count_u32),
    );
    encoder.end_encoding();
    drop(signpost);

    Ok((0..count)
        .map(|index| {
            Surface::from_metal_buffer_with_offset(
                out_buffer.clone(),
                dims,
                PixelFormat::Rgb8,
                index * surface_bytes,
            )
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn encode_repeated_mct_rgb8_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    count: usize,
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Vec<Surface>, Error> {
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect repeated color batch output size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch output size overflow".to_string(),
        })?;
    let output_len = u64::try_from(total_bytes.max(1)).map_err(|_| Error::MetalKernel {
        message: "J2K MetalDirect repeated output buffer exceeds u64".to_string(),
    })?;
    let out_buffer = runtime
        .device
        .new_buffer(output_len, MTLResourceOptions::StorageModeShared);
    let plane_stride = dims
        .0
        .checked_mul(dims.1)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch plane stride overflow".to_string(),
        })?;
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kBatchedMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        transform: mct_transform_code(transform),
        batch_count: 1,
        plane_stride,
        output_stride: u32::try_from(surface_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch surface stride exceeds u32".to_string(),
        })?,
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated MCT RGB8 pack");
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8_batched);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kBatchedMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, &runtime.pack_mct_rgb8_batched, dims);
    encoder.end_encoding();
    drop(signpost);

    if surface_bytes > 0 && count > 1 {
        let blit = command_buffer.new_blit_command_encoder();
        if metal_profile_stages_enabled() {
            blit.set_label("J2K decode hybrid repeated output blit");
        }
        let mut copied = 1usize;
        while copied < count {
            let copy_count = copied.min(count - copied);
            let dst_offset =
                copied
                    .checked_mul(surface_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K MetalDirect repeated output destination offset overflow"
                            .to_string(),
                    })?;
            let copy_bytes =
                copy_count
                    .checked_mul(surface_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K MetalDirect repeated output copy size overflow".to_string(),
                    })?;
            blit.copy_from_buffer(
                &out_buffer,
                0,
                &out_buffer,
                u64::try_from(dst_offset).map_err(|_| Error::MetalKernel {
                    message: "J2K MetalDirect repeated output destination offset exceeds u64"
                        .to_string(),
                })?,
                u64::try_from(copy_bytes).map_err(|_| Error::MetalKernel {
                    message: "J2K MetalDirect repeated output copy size exceeds u64".to_string(),
                })?,
            );
            record_hybrid_repeated_output_blit();
            copied += copy_count;
        }
        blit.end_encoding();
    }

    Ok((0..count)
        .map(|index| {
            Surface::from_metal_buffer_with_offset(
                out_buffer.clone(),
                dims,
                PixelFormat::Rgb8,
                index * surface_bytes,
            )
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn repeated_shared_direct_color_plan_count(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> Option<usize> {
    let first = plans.first()?;
    (plans.len() > 1 && plans.iter().all(|plan| Arc::ptr_eq(plan, first))).then_some(plans.len())
}

#[cfg(target_os = "macos")]
fn mct_transform_code(transform: J2kWaveletTransform) -> u32 {
    match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    }
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band(
    job: &j2k_native::J2kOwnedSubBandPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBand, Error> {
    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    let mut segments = Vec::new();

    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
        })?;
        for segment in &block.segments {
            let data_offset = coded_offset
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect segment offset overflow".to_string(),
                })?;
            segments.push(J2kClassicSegment {
                data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            });
        }
        jobs.push(J2kClassicCleanupBatchJob {
            coded_offset,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            segment_offset,
            segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
            })?,
            width: block.width,
            height: block.height,
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect output offset overflow".to_string(),
                })?,
            missing_msbs: u32::from(block.missing_bit_planes),
            total_bitplanes: u32::from(block.total_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            sub_band_type: match block.sub_band_type {
                j2k_native::J2kSubBandType::LowLow => 0,
                j2k_native::J2kSubBandType::HighLow => 1,
                j2k_native::J2kSubBandType::LowHigh => 2,
                j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(block.style),
            strict: u32::from(block.strict),
            dequantization_step: block.dequantization_step,
        });
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode);
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode);
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &segments, tier1_prepare_mode);
        Ok(PreparedClassicSubBand {
            band_id: job.band_id,
            width: job.width,
            height: job.height,
            zero_fill: false,
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedClassicSubBandGroup>, Error> {
    let mut groups = Vec::new();
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        let mut sub_bands = Vec::new();
        while let Some(PreparedDirectGrayscaleStep::ClassicSubBand(sub_band)) = steps.get(step_idx)
        {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_classic_sub_band_group(
                start_step,
                step_idx,
                &sub_bands,
                tier1_prepare_mode,
            )?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedClassicSubBand],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBandGroup, Error> {
    let mut members = Vec::with_capacity(sub_bands.len());
    let mut jobs = Vec::new();
    let mut segments = Vec::new();
    let mut coded_data = Vec::new();
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedClassicSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let segment_base = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped segment table exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;

        for segment in &sub_band.segments {
            let mut grouped_segment = *segment;
            grouped_segment.data_offset =
                coded_base
                    .checked_add(segment.data_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped segment offset overflow"
                            .to_string(),
                    })?;
            segments.push(grouped_segment);
        }

        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job coded offset overflow"
                            .to_string(),
                    })?;
            grouped_job.segment_offset =
                segment_base
                    .checked_add(job.segment_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job segment offset overflow"
                            .to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped output offset overflow"
                            .to_string(),
                    })?;
            jobs.push(grouped_job);
        }

        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode);
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode);
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &segments, tier1_prepare_mode);
        Ok(PreparedClassicSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            zero_fill: sub_bands.iter().any(|sub_band| sub_band.zero_fill),
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band(
    job: &j2k_native::HtOwnedSubBandPlan,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBand, Error> {
    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    Ok(PreparedHtSubBand {
        band_id: job.band_id,
        width: job.width,
        height: job.height,
        coded_data,
        coded_buffer: None,
        jobs,
        jobs_buffer: None,
    })
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedHtSubBandGroup>, Error> {
    let mut groups = Vec::new();
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        let mut sub_bands = Vec::new();
        while let Some(PreparedDirectGrayscaleStep::HtSubBand(sub_band)) = steps.get(step_idx) {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_ht_sub_band_group(
                start_step,
                step_idx,
                &sub_bands,
                tier1_prepare_mode,
            )?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedHtSubBand],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBandGroup, Error> {
    let mut members = Vec::with_capacity(sub_bands.len());
    let mut jobs = Vec::new();
    let mut coded_data = Vec::new();
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedHtSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;
        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped coded offset overflow".to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped output offset overflow".to_string(),
                    })?;
            jobs.push(grouped_job);
        }
        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode);
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode);
        Ok(PreparedHtSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            coded_arena: HtCodedArena {
                data: coded_data,
                buffer: coded_buffer,
            },
            jobs,
            jobs_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_ungrouped_ht_sub_band_buffers(
    steps: &mut [PreparedDirectGrayscaleStep],
    groups: &[PreparedHtSubBandGroup],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<(), Error> {
    if tier1_prepare_mode != DirectTier1Mode::Metal {
        return Ok(());
    }

    for (step_idx, step) in steps.iter_mut().enumerate() {
        let PreparedDirectGrayscaleStep::HtSubBand(sub_band) = step else {
            continue;
        };
        if groups
            .iter()
            .any(|group| group.start_step <= step_idx && step_idx < group.end_step)
        {
            sub_band.coded_buffer = None;
            sub_band.jobs_buffer = None;
            continue;
        }
        with_runtime(|runtime| {
            sub_band.coded_buffer = Some(prepare_direct_tier1_input_buffer(
                runtime,
                &sub_band.coded_data,
                tier1_prepare_mode,
            ));
            sub_band.jobs_buffer = Some(prepare_direct_tier1_input_buffer(
                runtime,
                &sub_band.jobs,
                tier1_prepare_mode,
            ));
            Ok(())
        })?;
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn prepared_ht_buffer<'a>(buffer: Option<&'a Buffer>, label: &str) -> Result<&'a Buffer, Error> {
    buffer.ok_or_else(|| Error::MetalKernel {
        message: format!("HTJ2K MetalDirect ungrouped sub-band is missing prepared {label} buffer"),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_grayscale_plan(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
fn prepare_direct_grayscale_plan_for_cpu_upload(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn prepare_direct_grayscale_plan_with_tier1_mode(
    plan: &J2kDirectGrayscalePlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    let mut steps = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::ClassicSubBand(
                    prepare_classic_sub_band(sub_band, tier1_prepare_mode)?,
                ));
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::HtSubBand(prepare_ht_sub_band(
                    sub_band,
                    tier1_prepare_mode,
                )?));
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                steps.push(PreparedDirectGrayscaleStep::Idwt(PreparedDirectIdwt {
                    step: *idwt,
                    output_window: BandRequiredRegion::full(idwt.rect.width(), idwt.rect.height()),
                }));
            }
            J2kDirectGrayscaleStep::Store(store) => {
                steps.push(PreparedDirectGrayscaleStep::Store(*store));
            }
        }
    }
    let classic_groups = prepare_classic_sub_band_groups(&steps, tier1_prepare_mode)?;
    let ht_groups = prepare_ht_sub_band_groups(&steps, tier1_prepare_mode)?;
    prepare_ungrouped_ht_sub_band_buffers(&mut steps, &ht_groups, tier1_prepare_mode)?;
    Ok(PreparedDirectGrayscalePlan {
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        tier1_prepare_mode,
        steps,
        classic_groups,
        ht_groups,
        cpu_tier1_cache: Arc::new(CpuTier1CoefficientCache::default()),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn crop_prepared_direct_grayscale_plan_to_output_region(
    plan: &mut PreparedDirectGrayscalePlan,
    region: Rect,
) -> Result<(), Error> {
    if region.w == 0 || region.h == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect region-scaled grayscale plan has an empty output region"
                .to_string(),
        });
    }
    if region.x == 0
        && region.y == 0
        && region.w == plan.dimensions.0
        && region.h == plan.dimensions.1
    {
        return Ok(());
    }

    plan.clear_cpu_tier1_cache()?;
    let mut store_count = 0;
    for step in &mut plan.steps {
        if let PreparedDirectGrayscaleStep::Store(store) = step {
            crop_direct_store_step_to_output_region(store, region)?;
            store_count += 1;
        }
    }

    if store_count == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect grayscale plan has no store step to crop".to_string(),
        });
    }

    prune_prepared_direct_grayscale_plan_to_store_windows(plan)?;
    plan.dimensions = (region.w, region.h);
    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
struct BandRequiredRegion {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

#[cfg(target_os = "macos")]
impl BandRequiredRegion {
    fn full(width: u32, height: u32) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        }
    }

    fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Option<Self> {
        (x0 < x1 && y0 < y1).then_some(Self { x0, y0, x1, y1 })
    }

    fn width(self) -> u32 {
        self.x1 - self.x0
    }

    fn height(self) -> u32 {
        self.y1 - self.y0
    }

    fn expanded(self, margin: u32, width: u32, height: u32) -> Self {
        Self {
            x0: self.x0.saturating_sub(margin),
            y0: self.y0.saturating_sub(margin),
            x1: self.x1.saturating_add(margin).min(width),
            y1: self.y1.saturating_add(margin).min(height),
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    fn intersects(self, x0: u32, y0: u32, width: u32, height: u32) -> bool {
        let x1 = x0.saturating_add(width);
        let y1 = y0.saturating_add(height);
        self.x0 < x1 && x0 < self.x1 && self.y0 < y1 && y0 < self.y1
    }
}

#[cfg(target_os = "macos")]
fn prune_prepared_direct_grayscale_plan_to_store_windows(
    plan: &mut PreparedDirectGrayscalePlan,
) -> Result<(), Error> {
    let mut required = HashMap::<J2kDirectBandId, BandRequiredRegion>::new();
    for step in &plan.steps {
        if let PreparedDirectGrayscaleStep::Store(store) = step {
            let source_right = store
                .source_x
                .checked_add(store.copy_width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect ROI source width overflows u32".to_string(),
                })?;
            let source_bottom = store
                .source_y
                .checked_add(store.copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect ROI source height overflows u32".to_string(),
                })?;
            if let Some(region) =
                BandRequiredRegion::new(store.source_x, store.source_y, source_right, source_bottom)
            {
                add_required_region(&mut required, store.input_band_id, region);
            }
        }
    }

    let mut idwt_output_windows = HashMap::<J2kDirectBandId, BandRequiredRegion>::new();
    for step in plan.steps.iter().rev() {
        if let PreparedDirectGrayscaleStep::Idwt(idwt) = step {
            let Some(output_region) = required.get(&idwt.step.output_band_id).copied() else {
                continue;
            };
            let expanded = output_region.expanded(
                idwt_required_output_margin(idwt.step.transform),
                idwt.step.rect.width(),
                idwt.step.rect.height(),
            );
            idwt_output_windows.insert(idwt.step.output_band_id, expanded);
            add_idwt_input_required_regions(&mut required, &idwt.step, expanded);
        }
    }

    for step in &mut plan.steps {
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let before = sub_band.jobs.len();
                retain_classic_jobs_for_required_region(
                    &mut sub_band.jobs,
                    required.get(&sub_band.band_id).copied(),
                );
                if sub_band.jobs.len() != before {
                    sub_band.zero_fill = true;
                    if plan.tier1_prepare_mode == DirectTier1Mode::Metal {
                        with_runtime(|runtime| {
                            sub_band.jobs_buffer =
                                borrow_slice_buffer(&runtime.device, &sub_band.jobs);
                            Ok(())
                        })?;
                    }
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let before = sub_band.jobs.len();
                retain_ht_jobs_for_required_region(
                    &mut sub_band.jobs,
                    required.get(&sub_band.band_id).copied(),
                );
                if sub_band.jobs.len() != before {
                    compact_ht_sub_band_coded_data(sub_band, plan.tier1_prepare_mode)?;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => {}
        }
    }

    apply_prepared_direct_idwt_output_windows(plan, &idwt_output_windows)?;
    plan.classic_groups = prepare_classic_sub_band_groups(&plan.steps, plan.tier1_prepare_mode)?;
    plan.ht_groups = prepare_ht_sub_band_groups(&plan.steps, plan.tier1_prepare_mode)?;
    prepare_ungrouped_ht_sub_band_buffers(
        &mut plan.steps,
        &plan.ht_groups,
        plan.tier1_prepare_mode,
    )?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn apply_prepared_direct_idwt_output_windows(
    plan: &mut PreparedDirectGrayscalePlan,
    windows: &HashMap<J2kDirectBandId, BandRequiredRegion>,
) -> Result<(), Error> {
    for step in &mut plan.steps {
        if let PreparedDirectGrayscaleStep::Idwt(idwt) = step {
            idwt.output_window = windows
                .get(&idwt.step.output_band_id)
                .copied()
                .unwrap_or_else(|| {
                    BandRequiredRegion::full(idwt.step.rect.width(), idwt.step.rect.height())
                });
        }
    }

    for step in &mut plan.steps {
        let PreparedDirectGrayscaleStep::Store(store) = step else {
            continue;
        };
        let Some(window) = windows.get(&store.input_band_id).copied() else {
            continue;
        };

        store.source_x =
            store
                .source_x
                .checked_sub(window.x0)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect cropped IDWT store source x underflow".to_string(),
                })?;
        store.source_y =
            store
                .source_y
                .checked_sub(window.y0)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect cropped IDWT store source y underflow".to_string(),
                })?;
        store.input_rect = j2k_native::J2kRect {
            x0: store.input_rect.x0.saturating_add(window.x0),
            y0: store.input_rect.y0.saturating_add(window.y0),
            x1: store.input_rect.x0.saturating_add(window.x1),
            y1: store.input_rect.y0.saturating_add(window.y1),
        };
    }

    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct PreparedIdwtInputWindows {
    ll: BandRequiredRegion,
    hl: BandRequiredRegion,
    lh: BandRequiredRegion,
    hh: BandRequiredRegion,
}

fn idwt_input_windows_from_slices(
    ll: &DirectBandSlice,
    hl: &DirectBandSlice,
    lh: &DirectBandSlice,
    hh: &DirectBandSlice,
) -> PreparedIdwtInputWindows {
    PreparedIdwtInputWindows {
        ll: BandRequiredRegion::full(ll.window.width(), ll.window.height()),
        hl: BandRequiredRegion::full(hl.window.width(), hl.window.height()),
        lh: BandRequiredRegion::full(lh.window.width(), lh.window.height()),
        hh: BandRequiredRegion::full(hh.window.width(), hh.window.height()),
    }
}

#[cfg(target_os = "macos")]
fn prepared_idwt_params(
    idwt: &PreparedDirectIdwt,
    inputs: PreparedIdwtInputWindows,
) -> J2kIdwtSingleDecompositionParams {
    J2kIdwtSingleDecompositionParams {
        x0: idwt.step.rect.x0,
        y0: idwt.step.rect.y0,
        output_x: idwt.output_window.x0,
        output_y: idwt.output_window.y0,
        width: idwt.output_window.width(),
        height: idwt.output_window.height(),
        ll_x: inputs.ll.x0,
        ll_y: inputs.ll.y0,
        ll_width: inputs.ll.width(),
        ll_height: inputs.ll.height(),
        hl_x: inputs.hl.x0,
        hl_y: inputs.hl.y0,
        hl_width: inputs.hl.width(),
        hl_height: inputs.hl.height(),
        lh_x: inputs.lh.x0,
        lh_y: inputs.lh.y0,
        lh_width: inputs.lh.width(),
        lh_height: inputs.lh.height(),
        hh_x: inputs.hh.x0,
        hh_y: inputs.hh.y0,
        hh_width: inputs.hh.width(),
        hh_height: inputs.hh.height(),
    }
}

#[cfg(target_os = "macos")]
fn repeated_idwt_params(
    idwt: &PreparedDirectIdwt,
    inputs: PreparedIdwtInputWindows,
    strides: PreparedIdwtInputStrides,
    batch_count: usize,
    context: &'static str,
) -> Result<J2kRepeatedIdwtSingleDecompositionParams, Error> {
    Ok(J2kRepeatedIdwtSingleDecompositionParams {
        x0: idwt.step.rect.x0,
        y0: idwt.step.rect.y0,
        output_x: idwt.output_window.x0,
        output_y: idwt.output_window.y0,
        width: idwt.output_window.width(),
        height: idwt.output_window.height(),
        ll_x: inputs.ll.x0,
        ll_y: inputs.ll.y0,
        ll_width: inputs.ll.width(),
        ll_height: inputs.ll.height(),
        hl_x: inputs.hl.x0,
        hl_y: inputs.hl.y0,
        hl_width: inputs.hl.width(),
        hl_height: inputs.hl.height(),
        lh_x: inputs.lh.x0,
        lh_y: inputs.lh.y0,
        lh_width: inputs.lh.width(),
        lh_height: inputs.lh.height(),
        hh_x: inputs.hh.x0,
        hh_y: inputs.hh.y0,
        hh_width: inputs.hh.width(),
        hh_height: inputs.hh.height(),
        ll_instance_stride: strides.ll,
        hl_instance_stride: strides.hl,
        lh_instance_stride: strides.lh,
        hh_instance_stride: strides.hh,
        batch_count: u32::try_from(batch_count).map_err(|_| Error::MetalKernel {
            message: format!("J2K MetalDirect {context} IDWT batch count exceeds u32"),
        })?,
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct PreparedIdwtInputStrides {
    ll: u32,
    hl: u32,
    lh: u32,
    hh: u32,
}

#[cfg(target_os = "macos")]
fn prepared_idwt_output_len(idwt: &PreparedDirectIdwt) -> usize {
    idwt.output_window.width() as usize * idwt.output_window.height() as usize
}

#[cfg(target_os = "macos")]
fn add_required_region(
    required: &mut HashMap<J2kDirectBandId, BandRequiredRegion>,
    band_id: J2kDirectBandId,
    region: BandRequiredRegion,
) {
    required
        .entry(band_id)
        .and_modify(|existing| *existing = existing.union(region))
        .or_insert(region);
}

#[cfg(target_os = "macos")]
fn idwt_required_output_margin(transform: J2kWaveletTransform) -> u32 {
    match transform {
        J2kWaveletTransform::Reversible53 => 16,
        J2kWaveletTransform::Irreversible97 => 40,
    }
}

#[cfg(target_os = "macos")]
fn add_idwt_input_required_regions(
    required: &mut HashMap<J2kDirectBandId, BandRequiredRegion>,
    idwt: &J2kDirectIdwtStep,
    output_region: BandRequiredRegion,
) {
    add_required_region(
        required,
        idwt.ll_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            true,
            true,
            idwt.ll.width(),
            idwt.ll.height(),
        ),
    );
    add_required_region(
        required,
        idwt.hl_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            false,
            true,
            idwt.hl.width(),
            idwt.hl.height(),
        ),
    );
    add_required_region(
        required,
        idwt.lh_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            true,
            false,
            idwt.lh.width(),
            idwt.lh.height(),
        ),
    );
    add_required_region(
        required,
        idwt.hh_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            false,
            false,
            idwt.hh.width(),
            idwt.hh.height(),
        ),
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::fn_params_excessive_bools)]
fn idwt_input_required_region(
    output_region: BandRequiredRegion,
    output_origin_x: u32,
    output_origin_y: u32,
    low_x: bool,
    low_y: bool,
    band_width: u32,
    band_height: u32,
) -> BandRequiredRegion {
    let x0 = j2k_native::idwt_band_index(output_origin_x, output_region.x0, low_x);
    let x1 =
        j2k_native::idwt_band_index(output_origin_x, output_region.x1 - 1, low_x).saturating_add(1);
    let y0 = j2k_native::idwt_band_index(output_origin_y, output_region.y0, low_y);
    let y1 =
        j2k_native::idwt_band_index(output_origin_y, output_region.y1 - 1, low_y).saturating_add(1);
    BandRequiredRegion {
        x0: x0.min(band_width),
        y0: y0.min(band_height),
        x1: x1.min(band_width),
        y1: y1.min(band_height),
    }
}

#[cfg(target_os = "macos")]
fn retain_classic_jobs_for_required_region(
    jobs: &mut Vec<J2kClassicCleanupBatchJob>,
    required: Option<BandRequiredRegion>,
) {
    let Some(required) = required else {
        jobs.clear();
        return;
    };
    jobs.retain(|job| {
        let output_x = job.output_offset % job.output_stride;
        let output_y = job.output_offset / job.output_stride;
        required.intersects(output_x, output_y, job.width, job.height)
    });
}

#[cfg(target_os = "macos")]
fn retain_ht_jobs_for_required_region(
    jobs: &mut Vec<J2kHtCleanupBatchJob>,
    required: Option<BandRequiredRegion>,
) {
    let Some(required) = required else {
        jobs.clear();
        return;
    };
    jobs.retain(|job| {
        let output_x = job.output_offset % job.output_stride;
        let output_y = job.output_offset / job.output_stride;
        required.intersects(output_x, output_y, job.width, job.height)
    });
}

#[cfg(target_os = "macos")]
fn compact_ht_sub_band_coded_data(
    sub_band: &mut PreparedHtSubBand,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<(), Error> {
    let previous = std::mem::take(&mut sub_band.coded_data);
    let mut compacted = Vec::new();

    for job in &mut sub_band.jobs {
        let start = job.coded_offset as usize;
        let len = job.coded_len as usize;
        let end = start.checked_add(len).ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect cropped coded payload range overflow".to_string(),
        })?;
        if end > previous.len() {
            return Err(Error::MetalKernel {
                message: "HTJ2K MetalDirect cropped coded payload range out of bounds".to_string(),
            });
        }
        job.coded_offset = u32::try_from(compacted.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect cropped coded payload exceeds u32".to_string(),
        })?;
        compacted.extend_from_slice(&previous[start..end]);
    }

    sub_band.coded_data = compacted;
    sub_band.coded_buffer = None;
    sub_band.jobs_buffer = None;
    Ok(())
}

#[cfg(target_os = "macos")]
fn checked_rect_end(origin: u32, length: u32, label: &str) -> Result<u32, Error> {
    origin
        .checked_add(length)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("J2K MetalDirect region-scaled {label} overflows u32"),
        })
}

#[cfg(target_os = "macos")]
fn crop_direct_store_step_to_output_region(
    store: &mut J2kDirectStoreStep,
    region: Rect,
) -> Result<(), Error> {
    let store_bounds = (
        store.output_x,
        store.output_y,
        checked_rect_end(store.output_x, store.copy_width, "store width")?,
        checked_rect_end(store.output_y, store.copy_height, "store height")?,
    );
    let region_bounds = (
        region.x,
        region.y,
        checked_rect_end(region.x, region.w, "ROI width")?,
        checked_rect_end(region.y, region.h, "ROI height")?,
    );
    let intersection = (
        store_bounds.0.max(region_bounds.0),
        store_bounds.1.max(region_bounds.1),
        store_bounds.2.min(region_bounds.2),
        store_bounds.3.min(region_bounds.3),
    );
    if intersection.0 >= intersection.2 || intersection.1 >= intersection.3 {
        return Err(Error::MetalKernel {
            message:
                "J2K MetalDirect region-scaled ROI does not intersect the decoded store window"
                    .to_string(),
        });
    }

    let source_delta = (
        intersection.0 - store_bounds.0,
        intersection.1 - store_bounds.1,
    );
    store.source_x =
        store
            .source_x
            .checked_add(source_delta.0)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect region-scaled source x overflows u32".to_string(),
            })?;
    store.source_y =
        store
            .source_y
            .checked_add(source_delta.1)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect region-scaled source y overflows u32".to_string(),
            })?;
    store.copy_width = intersection.2 - intersection.0;
    store.copy_height = intersection.3 - intersection.1;
    store.output_width = region.w;
    store.output_height = region.h;
    store.output_x = intersection.0 - region_bounds.0;
    store.output_y = intersection.1 - region_bounds.1;
    Ok(())
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_color_plan(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_color_plan_for_cpu_upload(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn prepare_direct_color_plan_with_tier1_mode(
    plan: &J2kDirectColorPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectColorPlan, Error> {
    let component_plans = plan
        .component_plans
        .iter()
        .map(|component| match tier1_prepare_mode {
            DirectTier1Mode::Metal => prepare_direct_grayscale_plan(component),
            DirectTier1Mode::CpuUpload => prepare_direct_grayscale_plan_for_cpu_upload(component),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color plan expected 3 component plans, got {}",
                component_plans.len()
            ),
        });
    }
    Ok(PreparedDirectColorPlan {
        dimensions: plan.dimensions,
        bit_depths: plan.bit_depths,
        mct: plan.mct,
        transform: plan.transform,
        component_plans,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn crop_prepared_direct_color_plan_to_output_region(
    plan: &mut PreparedDirectColorPlan,
    region: Rect,
) -> Result<(), Error> {
    if region.w == 0 || region.h == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect region-scaled color plan has an empty output region"
                .to_string(),
        });
    }

    for component_plan in &mut plan.component_plans {
        crop_prepared_direct_grayscale_plan_to_output_region(component_plan, region)?;
        if component_plan.dimensions != (region.w, region.h) {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K MetalDirect color component crop produced {:?}, expected {:?}",
                    component_plan.dimensions,
                    (region.w, region.h)
                ),
            });
        }
    }

    plan.dimensions = (region.w, region.h);
    Ok(())
}

#[cfg(target_os = "macos")]
impl PreparedDirectGrayscalePlan {
    fn classic_group_starting_at(&self, step_idx: usize) -> Option<&PreparedClassicSubBandGroup> {
        self.classic_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }

    fn ht_group_starting_at(&self, step_idx: usize) -> Option<&PreparedHtSubBandGroup> {
        self.ht_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_direct_grayscale_plan_compute_encoder_count(
    plan: &PreparedDirectGrayscalePlan,
    _fmt: PixelFormat,
) -> usize {
    usize::from(!plan.steps.is_empty())
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_repeated_direct_ht_cleanup_dispatch_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    let mut dispatches = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            dispatches += 1;
            step_idx = group.end_step;
            continue;
        }
        if matches!(
            plan.steps[step_idx],
            PreparedDirectGrayscaleStep::HtSubBand(_)
        ) {
            dispatches += 1;
        }
        step_idx += 1;
    }
    dispatches
}

#[cfg(target_os = "macos")]
fn encode_prepared_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_surface = None;
        let mut step_idx = 0;

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                let (buffers, status_check) =
                    encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            let step = &plan.steps[step_idx];
            match step {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    )?;
                    let (buffers, status_check) =
                        encode_prepared_classic_sub_band_to_buffer_in_encoder(
                            runtime,
                            encoder,
                            sub_band,
                            &output.buffer,
                            scratch_buffers,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    )?;
                    let (buffers, status_check) = encode_prepared_ht_sub_band_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        sub_band,
                        &output.buffer,
                    )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let ll =
                        lookup_direct_band_slice_entry(&bands, idwt.step.ll_band_id, idwt.step.ll)?;
                    let hl =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hl_band_id, idwt.step.hl)?;
                    let lh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.lh_band_id, idwt.step.lh)?;
                    let hh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hh_band_id, idwt.step.hh)?;
                    let params = prepared_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    );
                    let output = take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt))?;
                    match idwt.step.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                runtime,
                                encoder,
                                &ll.buffer,
                                ll.offset_bytes,
                                &hl.buffer,
                                hl.offset_bytes,
                                &lh.buffer,
                                lh.offset_bytes,
                                &hh.buffer,
                                hh.offset_bytes,
                                params,
                                &output.buffer,
                                0,
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            let status_check =
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    runtime,
                                    encoder,
                                    &ll.buffer,
                                    ll.offset_bytes,
                                    &hl.buffer,
                                    hl.offset_bytes,
                                    &lh.buffer,
                                    lh.offset_bytes,
                                    &hh.buffer,
                                    hh.offset_bytes,
                                    params,
                                    &output.buffer,
                                    0,
                                );
                            status_checks.push(status_check);
                        }
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: idwt.output_window,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        final_surface = Some(encode_gray_store_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            J2kGrayStoreParams {
                                input_width: store.input_rect.width(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                        )?);
                    } else {
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
                        dispatch_store_component_buffer_in_encoder_with_offsets(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        final_surface = Some(encode_gray_plane_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
            step_idx += 1;
        }

        final_surface.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect prepared grayscale plan did not produce a final stored plane"
                .to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn checked_coefficient_len(width: u32, height: u32, message: &str) -> Result<usize, Error> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: message.to_string(),
        })
}

#[cfg(target_os = "macos")]
fn upload_cpu_decoded_coefficients(
    runtime: &MetalRuntime,
    mut coefficients: Vec<f32>,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
) -> Buffer {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD);
    let buffer = borrow_mut_slice_buffer(&runtime.device, &mut coefficients);
    retained_buffers.push(buffer.clone());
    retained_cpu_coefficients.push(coefficients);
    buffer
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_prepared_direct_component_plane_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Buffer, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_plane = None;
        let mut step_idx = 0;
        let profile_stages = metal_profile_stages_enabled();

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                        let (buffers, status_check) =
                            encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                                runtime,
                                encoder,
                                group,
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
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_prepared_classic_sub_band_group_on_cpu_profile(
                            group,
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
                };
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, group.total_coefficients)?;
                        let (buffers, status_check) =
                            encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                                runtime,
                                encoder,
                                group,
                                &output.buffer,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_prepared_ht_sub_band_group_on_cpu_profile(
                            group,
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
                };
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                step_idx = group.end_step;
                continue;
            }

            match &plan.steps[step_idx] {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let buffer = match tier1_mode {
                        DirectTier1Mode::Metal => {
                            let output = take_f32_scratch_buffer(
                                runtime,
                                sub_band.width as usize * sub_band.height as usize,
                            )?;
                            let (buffers, status_check) =
                                encode_prepared_classic_sub_band_to_buffer_in_encoder(
                                    runtime,
                                    encoder,
                                    sub_band,
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
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_prepared_classic_sub_band_on_cpu_profile(
                                sub_band,
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
                    };
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer,
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let buffer = match tier1_mode {
                        DirectTier1Mode::Metal => {
                            let output = take_f32_scratch_buffer(
                                runtime,
                                sub_band.width as usize * sub_band.height as usize,
                            )?;
                            let (buffers, status_check) =
                                encode_prepared_ht_sub_band_to_buffer_in_encoder(
                                    runtime,
                                    encoder,
                                    sub_band,
                                    &output.buffer,
                                )?;
                            retained_buffers.extend(buffers);
                            status_checks.push(status_check);
                            let buffer = output.buffer.clone();
                            scratch_buffers.push(output);
                            buffer
                        }
                        DirectTier1Mode::CpuUpload => {
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_prepared_ht_sub_band_on_cpu_profile(
                                sub_band,
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
                    };
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer,
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let ll =
                        lookup_direct_band_slice_entry(&bands, idwt.step.ll_band_id, idwt.step.ll)?;
                    let hl =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hl_band_id, idwt.step.hl)?;
                    let lh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.lh_band_id, idwt.step.lh)?;
                    let hh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hh_band_id, idwt.step.hh)?;
                    let params = prepared_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    );
                    let output = take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt))?;
                    let encode_started = profile_stages.then(Instant::now);
                    match idwt.step.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                runtime,
                                encoder,
                                &ll.buffer,
                                ll.offset_bytes,
                                &hl.buffer,
                                hl.offset_bytes,
                                &lh.buffer,
                                lh.offset_bytes,
                                &hh.buffer,
                                hh.offset_bytes,
                                params,
                                &output.buffer,
                                0,
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            status_checks.push(
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    runtime,
                                    encoder,
                                    &ll.buffer,
                                    ll.offset_bytes,
                                    &hl.buffer,
                                    hl.offset_bytes,
                                    &lh.buffer,
                                    lh.offset_bytes,
                                    &hh.buffer,
                                    hh.offset_bytes,
                                    params,
                                    &output.buffer,
                                    0,
                                ),
                            );
                        }
                    }
                    if let Some(started) = encode_started {
                        stage_timings.metal_idwt_encode += elapsed_us(started);
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: idwt.output_window,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    let output = take_f32_scratch_buffer(
                        runtime,
                        store.output_width as usize * store.output_height as usize,
                    )?;
                    let encode_started = profile_stages.then(Instant::now);
                    dispatch_store_component_buffer_in_encoder_with_offsets(
                        runtime,
                        encoder,
                        &input,
                        input_offset,
                        &output.buffer,
                        0,
                        J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
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

        final_plane.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect component plan did not produce a stored plane".to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_repeated_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surfaces = encode_repeated_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            count,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surface = encode_prepared_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surface)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_with_device(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_prepared_direct_grayscale_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_batch(
    plans: &[Arc<PreparedDirectGrayscalePlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }

    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let mut surfaces = Vec::with_capacity(plans.len());

        let component_plan_refs = plans.iter().map(Arc::as_ref).collect::<Vec<_>>();
        if plans.len() > 1 && supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            let stacked_plane = encode_stacked_direct_component_plane_batch(
                runtime,
                DirectColorBatchCommandBuffers::single(command_buffer),
                &component_plan_refs,
                0,
                None,
                DirectTier1Mode::Metal,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )?;
            let first = plans.first().expect("plans is not empty");
            if stacked_plane.dimensions == first.dimensions && stacked_plane.count == plans.len() {
                surfaces = encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                    runtime,
                    command_buffer,
                    &stacked_plane.buffer,
                    first.dimensions,
                    first.bit_depth,
                    fmt,
                    plans.len(),
                )?;
            }
        }

        for plan in plans {
            if !surfaces.is_empty() {
                break;
            }
            surfaces.push(encode_prepared_direct_grayscale_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                &mut retained_buffers,
                &mut status_checks,
                &mut scratch_buffers,
            )?);
        }

        commit_and_wait_metal(command_buffer)?;
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_prepared_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| execute_prepared_direct_color_plan(plan, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect hybrid color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_hybrid_cpu_tier1_direct_color_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn execute_direct_color_plan_batch_with_tier1(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, tier1_mode, false)
}

#[cfg(all(target_os = "macos", test))]
fn execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, DirectTier1Mode::CpuUpload, true)
}

#[cfg(target_os = "macos")]
fn execute_direct_color_plan_batch_with_tier1_options(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    if tier1_mode == DirectTier1Mode::Metal
        && plans
            .iter()
            .any(|plan| !prepared_direct_color_plan_supports_runtime(plan, fmt))
    {
        return Err(Error::MetalKernel {
            message: "unsupported classic kernel input in direct component plan".to_string(),
        });
    }

    with_runtime(|runtime| {
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let profile_hybrid_stages =
            tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();

        if fmt == PixelFormat::Rgb8
            && profile_hybrid_stages
            && metal_profile_decode_split_commands_enabled()
        {
            let split_command_buffers = DecodeHybridSplitCommandBuffers::new(runtime);
            if let Some(surfaces) = try_encode_stacked_mct_rgb8_direct_color_batch(
                runtime,
                split_command_buffers.refs(),
                plans,
                tier1_mode,
                force_flattened_cpu_tier1,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )? {
                split_command_buffers.commit_in_order();
                let wait_started = Instant::now();
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                wait_for_completion_metal(&split_command_buffers.mct_pack)?;
                stage_timings.command_wait += elapsed_us(wait_started);
                record_completed_decode_split_gpu_stages(
                    &mut stage_timings,
                    &split_command_buffers,
                );
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers)?;
                return Ok(surfaces);
            }

            drop(split_command_buffers);
            retained_buffers.clear();
            retained_cpu_coefficients.clear();
            status_checks.clear();
            scratch_buffers.clear();
            stage_timings = DirectHybridStageTimings::default();
        }

        let command_buffer = runtime.queue.new_command_buffer();
        if profile_hybrid_stages {
            label_command_buffer(command_buffer, "j2k decode hybrid direct color batch");
        }

        if fmt == PixelFormat::Rgb8 {
            if let Some(surfaces) = try_encode_stacked_mct_rgb8_direct_color_batch(
                runtime,
                DirectColorBatchCommandBuffers::single(command_buffer),
                plans,
                tier1_mode,
                force_flattened_cpu_tier1,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )? {
                command_buffer.commit();
                let wait_started = profile_hybrid_stages.then(Instant::now);
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                wait_for_completion_metal(command_buffer)?;
                if let Some(started) = wait_started {
                    stage_timings.command_wait += elapsed_us(started);
                }
                if profile_hybrid_stages {
                    if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                        stage_timings.gpu_command += duration.as_micros();
                    }
                }
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                if tier1_mode == DirectTier1Mode::CpuUpload {
                    emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                }
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers)?;
                return Ok(surfaces);
            }
        }

        let mut surfaces = Vec::with_capacity(plans.len());

        for plan in plans {
            let surface = encode_prepared_direct_color_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                tier1_mode,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )?;
            surfaces.push(surface);
        }

        command_buffer.commit();
        let wait_started = profile_hybrid_stages.then(Instant::now);
        let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
        wait_for_completion_metal(command_buffer)?;
        if let Some(started) = wait_started {
            stage_timings.command_wait += elapsed_us(started);
        }
        if profile_hybrid_stages {
            if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                stage_timings.gpu_command += duration.as_micros();
            }
        }
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        if tier1_mode == DirectTier1Mode::CpuUpload {
            emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers)?;
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
fn signed_sample_bias(bit_depth: u8) -> f32 {
    2.0_f32.powi(i32::from(bit_depth) - 1)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_prepared_direct_color_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
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
            runtime,
            command_buffer,
            component_plan,
            tier1_mode,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
            status_checks,
            scratch_buffers,
        )?);
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
struct DirectBandSlice {
    band_id: J2kDirectBandId,
    buffer: Buffer,
    offset_bytes: usize,
    window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
fn lookup_direct_band_slice_entry(
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
fn lookup_direct_band_slice(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: j2k_native::J2kRect,
) -> Result<(Buffer, usize), Error> {
    let entry = lookup_direct_band_slice_entry(bands, band_id, rect)?;
    Ok((entry.buffer, entry.offset_bytes))
}

#[cfg(target_os = "macos")]
fn lookup_repeated_direct_band_layout_entry(
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
struct StackedDirectComponentPlane {
    buffer: Buffer,
    dimensions: (u32, u32),
    count: usize,
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn try_encode_stacked_mct_rgb8_direct_color_batch(
    runtime: &MetalRuntime,
    command_buffers: DirectColorBatchCommandBuffers<'_>,
    plans: &[Arc<PreparedDirectColorPlan>],
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Option<Vec<Surface>>, Error> {
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
            runtime,
            command_buffers,
            &component_plan_refs,
            component_idx,
            flattened_cpu_tier1_cache.as_ref(),
            tier1_mode,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
            status_checks,
            scratch_buffers,
        )?);
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
fn supports_stacked_direct_component_plane_batch(plans: &[&PreparedDirectGrayscalePlan]) -> bool {
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
#[allow(clippy::too_many_arguments)]
fn encode_stacked_direct_component_plane_batch(
    runtime: &MetalRuntime,
    command_buffers: DirectColorBatchCommandBuffers<'_>,
    plans: &[&PreparedDirectGrayscalePlan],
    component_idx: usize,
    flattened_cpu_tier1_cache: Option<&FlattenedCpuTier1Cache>,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<StackedDirectComponentPlane, Error> {
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
                            runtime,
                            command_buffers.idwt,
                            &ll.buffer,
                            ll.offset_bytes,
                            &hl.buffer,
                            hl.offset_bytes,
                            &lh.buffer,
                            lh.offset_bytes,
                            &hh.buffer,
                            hh.offset_bytes,
                            params,
                            &output.buffer,
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
                                    runtime,
                                    command_buffers.idwt.interleave,
                                    &ll.buffer,
                                    ll.offset_bytes,
                                    &hl.buffer,
                                    hl.offset_bytes,
                                    &lh.buffer,
                                    lh.offset_bytes,
                                    &hh.buffer,
                                    hh.offset_bytes,
                                    params,
                                    &output.buffer,
                                    instance_idx * per_instance_len * size_of::<f32>(),
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
#[allow(clippy::too_many_arguments)]
fn encode_repeated_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Vec<Surface>, Error> {
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
                        runtime,
                        DirectIdwtCommandBuffers::single(command_buffer),
                        &ll.buffer,
                        ll.offset_bytes,
                        &hl.buffer,
                        hl.offset_bytes,
                        &lh.buffer,
                        lh.offset_bytes,
                        &hh.buffer,
                        hh.offset_bytes,
                        params,
                        &output.buffer,
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
                                        runtime,
                                        command_buffer,
                                        &ll.buffer,
                                        ll.offset_bytes,
                                        &hl.buffer,
                                        hl.offset_bytes,
                                        &lh.buffer,
                                        lh.offset_bytes,
                                        &hh.buffer,
                                        hh.offset_bytes,
                                        params,
                                        &output.buffer,
                                        0,
                                    );
                                }
                                J2kWaveletTransform::Irreversible97 => status_checks.push(
                                    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        runtime,
                                        command_buffer,
                                        &ll.buffer,
                                        ll.offset_bytes,
                                        &hl.buffer,
                                        hl.offset_bytes,
                                        &lh.buffer,
                                        lh.offset_bytes,
                                        &hh.buffer,
                                        hh.offset_bytes,
                                        params,
                                        &output.buffer,
                                        0,
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

#[cfg(target_os = "macos")]
fn copy_plane_samples(buffer: &Buffer, samples: &[f32], image_width: usize, roi: Rect) {
    let row_width = roi.w as usize;
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), row_width * roi.h as usize)
    };

    for row in 0..roi.h as usize {
        let src_start = (roi.y as usize + row) * image_width + roi.x as usize;
        let src_end = src_start + row_width;
        let dst_start = row * row_width;
        dst[dst_start..dst_start + row_width].copy_from_slice(&samples[src_start..src_end]);
    }
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime, encoder, plane, 0, dims, bit_depth, fmt,
    )
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_command_buffer_with_offset(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime,
        encoder,
        plane,
        plane_offset_bytes,
        dims,
        bit_depth,
        fmt,
    );
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_encoder_with_offset(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) =
        output_shape_for(&NativeColorSpace::Gray, false, 1, fmt, runtime)?;
    let mut bit_depths = [0u32; 4];
    bit_depths[0] = u32::from(bit_depth);
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays(bit_depths);
    let params = J2kPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        output_channels,
        opaque_alpha,
        max_values,
        u8_scales,
        u16_scales,
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), plane_offset_bytes as u64);
    encoder.set_buffer(1, None, 0);
    encoder.set_buffer(2, None, 0);
    encoder.set_buffer(3, None, 0);
    encoder.set_buffer(4, Some(&out_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, dims);

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_repeated_gray_plane_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    let count_u32 = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal repeated grayscale surface count exceeds u32".to_string(),
    })?;
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal repeated grayscale surface size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal repeated grayscale output size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let scale = j2k_scalar_pack_params(u32::from(bit_depth));
    let params = J2kRepeatedGrayPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: j2k_u32_param(pitch_bytes, "J2K Metal output stride exceeds u32")?,
        batch_count: count_u32,
        max_value: scale.max_value,
        u8_scale: scale.u8_scale,
        u16_scale: scale.u16_scale,
    };
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.pack_u8_repeated_gray,
        PixelFormat::Gray16 => &runtime.pack_u16_repeated_gray,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("J2K Metal repeated grayscale pack does not support {fmt:?}"),
            })
        }
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(encoder, pipeline, (dims.0, dims.1, count_u32));
    encoder.end_encoding();

    let mut surfaces = Vec::with_capacity(count);
    for instance_idx in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            fmt,
            instance_idx * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn j2k_pack_kernel_name_for(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
) -> Option<&'static str> {
    match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8) => Some("j2k_pack_gray8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => Some("j2k_pack_rgb8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => Some("j2k_pack_rgb_opaque_rgba8"),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => Some("j2k_pack_rgba8"),
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray16) => Some("j2k_pack_gray16"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16) => Some("j2k_pack_rgb16"),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn j2k_pack_pipeline_for<'a>(
    runtime: &'a MetalRuntime,
    kernel_name: &str,
) -> Result<&'a ComputePipelineState, Error> {
    let pipeline = match kernel_name {
        "j2k_pack_gray8" => &runtime.pack_gray8,
        "j2k_pack_rgb8" => &runtime.pack_rgb8,
        "j2k_pack_rgb_opaque_rgba8" => &runtime.pack_rgb_opaque_rgba8,
        "j2k_pack_rgba8" => &runtime.pack_rgba8,
        "j2k_pack_gray16" => &runtime.pack_gray16,
        "j2k_pack_rgb16" => &runtime.pack_rgb16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("unsupported validated J2K Metal pack kernel `{kernel_name}`"),
            });
        }
    };
    Ok(pipeline)
}

#[cfg(target_os = "macos")]
fn output_shape_for<'a>(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
    runtime: &'a MetalRuntime,
) -> Result<(u32, u32, &'a ComputePipelineState), Error> {
    let Some(kernel_name) = j2k_pack_kernel_name_for(color_space, has_alpha, plane_count, fmt)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "unsupported J2K Metal mapping for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
            ),
        });
    };
    let (output_channels, opaque_alpha) = match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8 | PixelFormat::Gray16) => (1, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8 | PixelFormat::Rgb16)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => (3, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => (4, 1),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => (4, 0),
        _ => {
            return Err(Error::MetalKernel {
                message: format!(
                    "unsupported validated J2K Metal pack shape for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
                ),
            });
        }
    };
    Ok((
        output_channels,
        opaque_alpha,
        j2k_pack_pipeline_for(runtime, kernel_name)?,
    ))
}

