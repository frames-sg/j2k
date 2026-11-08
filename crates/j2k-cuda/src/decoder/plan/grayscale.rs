// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grayscale plan construction for prepared and one-shot inputs.

use super::{
    native_decode_error, parse_at_scale, parse_with_host_cap, profile, CudaHtj2kDecodePlan,
    CudaHtj2kDecodeProfileDetail, CudaHtj2kProfileReport, DecodeSettings, DeviceDecodePlan,
    Downscale, Error, HostPhaseBudget, J2kDecoder, J2kReferencedClassicPlan,
    J2kReferencedHtj2kPlan, NativeDecoderContext, NativeImage, PixelFormat, Rect,
};

pub(in crate::decoder) fn build_cuda_htj2k_grayscale_plan_from_bytes_with_profile<'a>(
    input: &'a [u8],
    fmt: PixelFormat,
    native_context: &mut NativeDecoderContext<'a>,
) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
    build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile_and_cap(
        input,
        fmt,
        None,
        DecodeSettings::default(),
        native_context,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )
}

pub(in crate::decoder) fn build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile_and_cap<
    'a,
>(
    input: &'a [u8],
    fmt: PixelFormat,
    device_plan: Option<DeviceDecodePlan>,
    settings: DecodeSettings,
    native_context: &mut NativeDecoderContext<'a>,
    host_cap: usize,
) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
    let total_start = profile::profile_now(true);
    let parse_start = profile::profile_now(true);
    let (image, mut host_budget) = parse_with_host_cap(
        input,
        &settings,
        device_plan
            .filter(|plan| plan.scale() != Downscale::None)
            .map(j2k::DeviceDecodePlan::scale),
        host_cap,
        "j2k CUDA bounded grayscale planning",
    )?;
    let parse_us = profile::elapsed_us(parse_start);

    let plan_start = profile::profile_now(true);
    let selected_rect = device_plan.and_then(|plan| {
        let full = Rect::full(plan.source_dims());
        if plan.source_rect() == full {
            None
        } else if plan.scale() == Downscale::None {
            Some(plan.source_rect())
        } else {
            Some(plan.output_rect())
        }
    });
    let native_plan = match selected_rect {
        Some(rect) => image
            .build_direct_grayscale_plan_region_with_context(
                native_context,
                (rect.x, rect.y, rect.w, rect.h),
            )
            .map_err(native_decode_error)?,
        None => image
            .build_direct_grayscale_plan_with_context(native_context)
            .map_err(native_decode_error)?,
    };
    let plan_us = profile::elapsed_us(plan_start);

    let flatten_start = profile::profile_now(true);
    host_budget.account_bytes(
        native_plan
            .retained_allocation_bytes()
            .map_err(j2k_native::DecodeError::from)
            .map_err(native_decode_error)?,
    )?;
    let cuda_plan = match selected_rect {
        Some(rect) => CudaHtj2kDecodePlan::from_grayscale_direct_plan_region_with_budget(
            &native_plan,
            fmt,
            (rect.x, rect.y),
            (rect.w, rect.h),
            &mut host_budget,
        )?,
        None => CudaHtj2kDecodePlan::from_grayscale_direct_plan_with_budget(
            &native_plan,
            fmt,
            (0, 0),
            &mut host_budget,
        )?,
    };
    let flatten_us = profile::elapsed_us(flatten_start);
    let report = CudaHtj2kProfileReport {
        parse_us,
        plan_us,
        flatten_us,
        total_us: profile::elapsed_us(total_start),
        block_count: cuda_plan.block_count(),
        classic_block_count: cuda_plan.classic_code_blocks().len(),
        ht_block_count: cuda_plan.code_blocks().len(),
        payload_bytes: cuda_plan.payload().len(),
        dispatch_count: 0,
        residency: crate::SurfaceResidency::CudaResidentDecode,
        detail: CudaHtj2kDecodeProfileDetail::default(),
        ..CudaHtj2kProfileReport::default()
    };
    report.emit("plan");
    Ok((cuda_plan, report))
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn build_cuda_htj2k_grayscale_plans_from_referenced_with_profile(
    input: &[u8],
    referenced: &J2kReferencedHtj2kPlan,
    fmt: PixelFormat,
    output_dimensions: (u32, u32),
    shared_payload: &mut Vec<u8>,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport)>, Error> {
    let total_start = profile::profile_now(true);
    let output_rect = referenced.output_rect();
    if (output_rect.width(), output_rect.height()) != output_dimensions {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(
                "prepared CUDA grayscale tile plan output geometry is inconsistent",
            ),
        ));
    }
    let mut plans = host_budget.try_vec_with_capacity(referenced.tiles().len())?;
    let mut next_payload = 0usize;
    for tile in referenced.tiles() {
        CudaHtj2kDecodePlan::validate_referenced_classic_payload_sequence(
            tile.classic_payloads(),
            tile.classic_ranges(),
        )?;
        let geometry = tile.grayscale_geometry().ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(
                "prepared CUDA grayscale batch received a color HTJ2K tile",
            ),
        ))?;
        let span = tile.payload_records();
        if span.first_record != next_payload {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(
                    "prepared CUDA grayscale tile payload spans are not contiguous",
                ),
            ));
        }
        let payload_end = span.end_record().ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::resource_limit(
                "prepared CUDA grayscale tile payload span overflows",
            ),
        ))?;
        let payloads = referenced
            .payloads()
            .get(span.first_record..payload_end)
            .ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(
                    "prepared CUDA grayscale tile payload span is out of bounds",
                ),
            ))?;
        let payload_start = shared_payload.len();
        let flatten_start = profile::profile_now(true);
        let cuda_plan = CudaHtj2kDecodePlan::from_referenced_tile_grayscale_plan_into_shared(
            geometry,
            payloads,
            tile.classic_payloads(),
            tile.classic_ranges(),
            input,
            fmt,
            (output_rect.x0, output_rect.y0),
            output_dimensions,
            shared_payload,
            host_budget,
        )?;
        let report = CudaHtj2kProfileReport {
            parse_us: 0,
            plan_us: 0,
            flatten_us: profile::elapsed_us(flatten_start),
            total_us: profile::elapsed_us(total_start),
            block_count: cuda_plan.block_count(),
            classic_block_count: cuda_plan.classic_code_blocks().len(),
            ht_block_count: cuda_plan.code_blocks().len(),
            payload_bytes: shared_payload.len().saturating_sub(payload_start),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("prepared_tile_plan");
        plans.push((cuda_plan, report));
        next_payload = payload_end;
    }
    if next_payload != referenced.payloads().len() {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(
                "prepared CUDA grayscale tile payloads contain trailing records",
            ),
        ));
    }
    Ok(plans)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn build_cuda_classic_grayscale_plans_from_referenced_with_profile(
    input: &[u8],
    referenced: &J2kReferencedClassicPlan,
    fmt: PixelFormat,
    output_dimensions: (u32, u32),
    shared_payload: &mut Vec<u8>,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport)>, Error> {
    let total_start = profile::profile_now(true);
    CudaHtj2kDecodePlan::validate_referenced_classic_payload_sequence(
        referenced.payloads(),
        referenced.ranges(),
    )?;
    let output_rect = referenced.output_rect();
    if (output_rect.width(), output_rect.height()) != output_dimensions {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(
                "prepared CUDA classic grayscale tile output geometry is inconsistent",
            ),
        ));
    }
    let mut plans = host_budget.try_vec_with_capacity(referenced.tiles().len())?;
    let mut next_payload = 0usize;
    for tile in referenced.tiles() {
        let geometry = tile.grayscale_geometry().ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(
                "prepared CUDA classic grayscale batch received a color tile",
            ),
        ))?;
        let span = tile.payload_records();
        if span.first_record != next_payload {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(
                    "prepared CUDA classic grayscale tile payload spans are not contiguous",
                ),
            ));
        }
        let payload_end = span.end_record().ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::resource_limit(
                "prepared CUDA classic grayscale tile payload span overflows",
            ),
        ))?;
        let payloads = referenced
            .payloads()
            .get(span.first_record..payload_end)
            .ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(
                    "prepared CUDA classic grayscale tile payload span is out of bounds",
                ),
            ))?;
        let payload_start = shared_payload.len();
        let flatten_start = profile::profile_now(true);
        let cuda_plan =
            CudaHtj2kDecodePlan::from_referenced_classic_tile_grayscale_plan_into_shared(
                geometry,
                payloads,
                referenced.ranges(),
                input,
                fmt,
                (output_rect.x0, output_rect.y0),
                output_dimensions,
                shared_payload,
                host_budget,
            )?;
        let report = CudaHtj2kProfileReport {
            parse_us: 0,
            plan_us: 0,
            flatten_us: profile::elapsed_us(flatten_start),
            total_us: profile::elapsed_us(total_start),
            block_count: cuda_plan.block_count(),
            classic_block_count: cuda_plan.classic_code_blocks().len(),
            ht_block_count: 0,
            payload_bytes: shared_payload.len().saturating_sub(payload_start),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("prepared_classic_tile_plan");
        plans.push((cuda_plan, report));
        next_payload = payload_end;
    }
    if next_payload != referenced.payloads().len() {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(
                "prepared CUDA classic grayscale tile payloads contain trailing records",
            ),
        ));
    }
    Ok(plans)
}

impl J2kDecoder<'_> {
    /// Build a flat CUDA HTJ2K grayscale decode plan and return stage timings.
    pub(crate) fn build_cuda_htj2k_grayscale_plan_with_profile(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
        let mut native_context = NativeDecoderContext::default();
        build_cuda_htj2k_grayscale_plan_from_bytes_with_profile(
            self.bytes,
            fmt,
            &mut native_context,
        )
    }

    /// Build a flat CUDA HTJ2K grayscale region decode plan and return stage timings.
    pub(crate) fn build_cuda_htj2k_grayscale_region_plan_with_profile(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(self.bytes, &DecodeSettings::default())
            .map_err(native_decode_error)?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_grayscale_plan_region_with_context(
                &mut native_context,
                (roi.x, roi.y, roi.w, roi.h),
            )
            .map_err(native_decode_error)?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
            &native_plan,
            fmt,
            (roi.x, roi.y),
            (roi.w, roi.h),
        )?;
        let flatten_us = profile::elapsed_us(flatten_start);

        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count: cuda_plan.block_count(),
            classic_block_count: cuda_plan.classic_code_blocks().len(),
            ht_block_count: cuda_plan.code_blocks().len(),
            payload_bytes: cuda_plan.payload().len(),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");
        Ok((cuda_plan, report))
    }

    /// Build a flat reduced-resolution CUDA HTJ2K grayscale decode plan and
    /// return stage timings.
    pub(crate) fn build_cuda_htj2k_grayscale_scaled_plan_with_profile(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = parse_at_scale(self.bytes, &DecodeSettings::default(), scale, 0)?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_grayscale_plan_with_context(&mut native_context)
            .map_err(native_decode_error)?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, fmt, (0, 0))?;
        let flatten_us = profile::elapsed_us(flatten_start);

        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count: cuda_plan.block_count(),
            classic_block_count: cuda_plan.classic_code_blocks().len(),
            ht_block_count: cuda_plan.code_blocks().len(),
            payload_bytes: cuda_plan.payload().len(),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");
        Ok((cuda_plan, report))
    }

    /// Build a flat reduced-resolution CUDA HTJ2K grayscale region decode
    /// plan and return stage timings.
    pub(crate) fn build_cuda_htj2k_grayscale_region_scaled_plan_with_profile(
        &mut self,
        fmt: PixelFormat,
        scaled_roi: Rect,
        scale: Downscale,
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = parse_at_scale(self.bytes, &DecodeSettings::default(), scale, 0)?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_grayscale_plan_region_with_context(
                &mut native_context,
                (scaled_roi.x, scaled_roi.y, scaled_roi.w, scaled_roi.h),
            )
            .map_err(native_decode_error)?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
            &native_plan,
            fmt,
            (scaled_roi.x, scaled_roi.y),
            (scaled_roi.w, scaled_roi.h),
        )?;
        let flatten_us = profile::elapsed_us(flatten_start);

        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count: cuda_plan.block_count(),
            classic_block_count: cuda_plan.classic_code_blocks().len(),
            ht_block_count: cuda_plan.code_blocks().len(),
            payload_bytes: cuda_plan.payload().len(),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");
        Ok((cuda_plan, report))
    }
}

#[cfg(test)]
mod reduction_tests {
    use super::*;
    use j2k::DeviceDecodeRequest;

    #[test]
    fn scaled_gray_plans_share_exact_reduction_and_bounded_preparation() {
        let bytes = j2k_native::encode_htj2k(
            &[42; 23 * 17],
            23,
            17,
            1,
            8,
            false,
            &j2k_native::EncodeOptions {
                num_decomposition_levels: 3,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("gray fixture");
        let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
        let scale = Downscale::Half;
        let (full, _) = decoder
            .build_cuda_htj2k_grayscale_scaled_plan_with_profile(PixelFormat::Gray8, scale)
            .expect("scaled gray plan");
        assert_eq!(full.dimensions(), (12, 9));
        let request_plan = DeviceDecodePlan::for_reference_grid(
            decoder.inner.reference_grid(),
            DeviceDecodeRequest::RegionScaled {
                roi: Rect {
                    x: 15,
                    y: 9,
                    w: 8,
                    h: 8,
                },
                scale,
            },
        )
        .expect("region plan");
        let (region, _) = decoder
            .build_cuda_htj2k_grayscale_region_scaled_plan_with_profile(
                PixelFormat::Gray8,
                request_plan.output_rect(),
                scale,
            )
            .expect("scaled gray region");
        let build = |cap| {
            build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile_and_cap(
                &bytes,
                PixelFormat::Gray8,
                Some(request_plan),
                DecodeSettings::default(),
                &mut NativeDecoderContext::default(),
                cap,
            )
        };
        let (prepared, _) =
            build(j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES).expect("prepared gray");
        assert_eq!(region.dimensions(), request_plan.output_dims());
        assert_eq!(prepared.dimensions(), region.dimensions());
        assert_eq!(prepared.output_origin(), region.output_origin());
        assert!(build(1)
            .expect_err("retained parser baseline must be enforced")
            .is_host_planning_limit());
    }
}
