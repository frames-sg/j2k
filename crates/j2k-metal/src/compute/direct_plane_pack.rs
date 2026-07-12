// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_metal_surface_len, commit_and_wait_metal, copy_plane_samples, dispatch_2d_pipeline,
    dispatch_3d_pipeline, hybrid_stage_signpost, j2k_pack_scale_arrays, j2k_u32_param,
    label_compute_encoder, metal_profile_stages_enabled, new_blit_command_encoder,
    new_command_buffer, new_compute_command_encoder, new_shared_buffer, output_shape_for,
    record_hybrid_repeated_output_blit, signed_sample_bias, size_of, Arc, Buffer, CommandBufferRef,
    Device, Error, J2kBatchedMctRgb8PackParams, J2kMctRgb8PackParams, J2kPackParams,
    J2kWaveletTransform, MetalRuntime, NativeColorSpace, NativeDecodedComponents, PixelFormat,
    PreparedDirectColorPlan, Rect, Surface, SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE,
};

#[cfg(target_os = "macos")]
pub(super) struct PlaneStage {
    pub(super) dims: (u32, u32),
    pub(super) plane_count: usize,
    pub(super) color_space: NativeColorSpace,
    pub(super) has_alpha: bool,
    pub(super) bit_depths: [u32; 4],
    pub(super) planes: [Option<Buffer>; 4],
}

#[cfg(target_os = "macos")]
fn allocate_direct_surface_handles(
    count: usize,
    phase: &'static str,
    what: &'static str,
) -> Result<Vec<Surface>, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(phase);
    budget.try_vec(count, what).map_err(Error::from)
}

#[cfg(target_os = "macos")]
fn supported_plane_color_space(color_space: &NativeColorSpace) -> Result<NativeColorSpace, Error> {
    match color_space {
        NativeColorSpace::Gray => Ok(NativeColorSpace::Gray),
        NativeColorSpace::RGB => Ok(NativeColorSpace::RGB),
        unsupported => Err(Error::MetalKernel {
            message: format!("unsupported J2K Metal plane mapping for {unsupported:?}"),
        }),
    }
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    pub(super) fn from_planes(
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
        let color_space = supported_plane_color_space(decoded.color_space())?;
        if plane_count == 0 || plane_count > 4 {
            return Err(Error::MetalKernel {
                message: format!("unsupported J2K plane count {plane_count}"),
            });
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, plane) in decoded.planes().iter().enumerate() {
            bit_depths[index] = u32::from(plane.bit_depth());
            let (_, len_bytes) = checked_metal_surface_len(
                dims,
                size_of::<f32>(),
                "J2K MetalDirect plane upload size overflow",
            )?;
            let mut buffer = new_shared_buffer(device, len_bytes)?;
            copy_plane_samples(&mut buffer, plane.samples(), full_dims.0 as usize, roi)?;
            planes[index] = Some(buffer);
        }

        Ok(Self {
            dims,
            plane_count,
            color_space,
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    pub(super) fn from_captured_planes(
        decoded: &NativeDecodedComponents<'_>,
        captured_planes: Vec<Buffer>,
    ) -> Option<Self> {
        let plane_count = decoded.planes().len();
        let color_space = supported_plane_color_space(decoded.color_space()).ok()?;
        let supported_shape = matches!(
            (&color_space, decoded.has_alpha(), plane_count),
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
            color_space,
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    pub(super) fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let surface =
            encode_plane_stage_to_surface_in_command_buffer(runtime, &command_buffer, &self, fmt)?;
        commit_and_wait_metal(&command_buffer)?;
        Ok(surface)
    }
}

#[cfg(target_os = "macos")]
pub(super) fn encode_plane_stage_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    stage: &PlaneStage,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let (pitch_bytes, surface_bytes) = checked_metal_surface_len(
        stage.dims,
        fmt.bytes_per_pixel(),
        "J2K MetalDirect plane pack output size overflow",
    )?;
    let out_buffer = new_shared_buffer(&runtime.device, surface_bytes)?;
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

    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode hybrid plane pack");
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
    dispatch_2d_pipeline(&encoder, pipeline, stage.dims);
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, stage.dims, fmt))
}

#[cfg(target_os = "macos")]
pub(super) fn encode_mct_rgb8_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Surface, Error> {
    let (pitch_bytes, surface_bytes) = checked_metal_surface_len(
        dims,
        PixelFormat::Rgb8.bytes_per_pixel(),
        "J2K MetalDirect MCT RGB8 output size overflow",
    )?;
    let out_buffer = new_shared_buffer(&runtime.device, surface_bytes)?;
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
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode hybrid MCT RGB8 pack");
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
    dispatch_2d_pipeline(&encoder, &runtime.pack_mct_rgb8, dims);
    encoder.end_encoding();
    drop(signpost);

    Ok(Surface::from_metal_buffer(
        out_buffer,
        dims,
        PixelFormat::Rgb8,
    ))
}

#[cfg(target_os = "macos")]
pub(super) fn encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
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
    let (pitch_bytes, surface_bytes) = checked_metal_surface_len(
        dims,
        PixelFormat::Rgb8.bytes_per_pixel(),
        "J2K MetalDirect color batch output size overflow",
    )?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect color batch output size overflow".to_string(),
        })?;
    let mut surfaces = allocate_direct_surface_handles(
        count,
        "J2K MetalDirect color batch surface metadata",
        "J2K MetalDirect color batch surface handles",
    )?;
    let out_buffer = new_shared_buffer(&runtime.device, total_bytes)?;
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
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode hybrid batched MCT RGB8 pack");
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
        &encoder,
        &runtime.pack_mct_rgb8_batched,
        (dims.0, dims.1, count_u32),
    );
    encoder.end_encoding();
    drop(signpost);

    for index in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            PixelFormat::Rgb8,
            index * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "Metal dispatch and retained-resource ordering must remain linear"
)]
pub(super) fn encode_repeated_mct_rgb8_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    count: usize,
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Vec<Surface>, Error> {
    let (pitch_bytes, surface_bytes) = checked_metal_surface_len(
        dims,
        PixelFormat::Rgb8.bytes_per_pixel(),
        "J2K MetalDirect repeated color batch output size overflow",
    )?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch output size overflow".to_string(),
        })?;
    let mut surfaces = allocate_direct_surface_handles(
        count,
        "J2K MetalDirect repeated color batch surface metadata",
        "J2K MetalDirect repeated color batch surface handles",
    )?;
    let out_buffer = new_shared_buffer(&runtime.device, total_bytes)?;
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
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K decode hybrid repeated MCT RGB8 pack");
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
    dispatch_2d_pipeline(&encoder, &runtime.pack_mct_rgb8_batched, dims);
    encoder.end_encoding();
    drop(signpost);

    if surface_bytes > 0 && count > 1 {
        let blit = new_blit_command_encoder(command_buffer)?;
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

    for index in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            PixelFormat::Rgb8,
            index * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
pub(super) fn repeated_shared_direct_color_plan_count(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> Option<usize> {
    let first = plans.first()?;
    (plans.len() > 1 && plans.iter().all(|plan| Arc::ptr_eq(plan, first))).then_some(plans.len())
}

#[cfg(target_os = "macos")]
pub(super) fn mct_transform_code(transform: J2kWaveletTransform) -> u32 {
    match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{supported_plane_color_space, Error, NativeColorSpace};

    #[test]
    fn plane_stage_color_space_ownership_accepts_only_heap_free_variants() {
        assert!(matches!(
            supported_plane_color_space(&NativeColorSpace::Gray),
            Ok(NativeColorSpace::Gray)
        ));
        assert!(matches!(
            supported_plane_color_space(&NativeColorSpace::RGB),
            Ok(NativeColorSpace::RGB)
        ));

        for unsupported in [
            NativeColorSpace::CMYK,
            NativeColorSpace::Unknown { num_channels: 5 },
        ] {
            assert!(matches!(
                supported_plane_color_space(&unsupported),
                Err(Error::MetalKernel { message })
                    if message.contains("unsupported J2K Metal plane mapping")
            ));
        }

        let icc = NativeColorSpace::Icc {
            profile: vec![1, 2, 3, 4],
            num_channels: 3,
        };
        let (profile_ptr, profile_capacity) = match &icc {
            NativeColorSpace::Icc { profile, .. } => (profile.as_ptr(), profile.capacity()),
            _ => unreachable!("constructed ICC color space"),
        };
        assert!(matches!(
            supported_plane_color_space(&icc),
            Err(Error::MetalKernel { message })
                if message.contains("unsupported J2K Metal plane mapping")
        ));
        match &icc {
            NativeColorSpace::Icc { profile, .. } => {
                assert_eq!(profile.as_ptr(), profile_ptr);
                assert_eq!(profile.capacity(), profile_capacity);
            }
            _ => unreachable!("ICC color space remains owned by the caller"),
        }
    }
}
