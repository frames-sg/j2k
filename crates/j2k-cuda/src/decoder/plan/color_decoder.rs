// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region and reduced-resolution color planning on the decoder facade.

use super::{
    build_cuda_htj2k_color_plans_from_bytes_with_profile, flatten_cuda_color_components,
    native_decode_error, profile, rgba_bit_depths_from_rgb, CudaHtj2kColorDecodePlans,
    CudaHtj2kDecodePlan, CudaHtj2kDecodeProfileDetail, CudaHtj2kProfileReport, CudaHtj2kTransform,
    DecodeSettings, Error, J2kDecoder, NativeDecoderContext, NativeImage, PixelFormat, Rect,
};

impl J2kDecoder<'_> {
    /// Build flat CUDA HTJ2K RGB component plans and return stage timings.
    #[cfg(feature = "cuda-runtime")]
    pub(in crate::decoder) fn build_cuda_htj2k_color_plans_with_profile(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let mut native_context = NativeDecoderContext::default();
        build_cuda_htj2k_color_plans_from_bytes_with_profile(self.bytes, fmt, &mut native_context)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(in crate::decoder) fn build_cuda_htj2k_color_scaled_plans_with_profile(
        &mut self,
        fmt: PixelFormat,
        output_dimensions: (u32, u32),
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(
            self.bytes,
            &DecodeSettings {
                target_resolution: Some(output_dimensions),
                ..DecodeSettings::default()
            },
        )
        .map_err(native_decode_error)?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_color_plan_with_context(&mut native_context)
            .map_err(native_decode_error)?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let (payload, components) = flatten_cuda_color_components(
            &native_plan,
            fmt,
            None,
            "j2k CUDA scaled color decode plans",
        )?;
        let flatten_us = profile::elapsed_us(flatten_start);
        let block_count = components
            .iter()
            .map(CudaHtj2kDecodePlan::block_count)
            .sum::<usize>();
        let classic_block_count = components
            .iter()
            .map(|plan| plan.classic_code_blocks().len())
            .sum::<usize>();
        let ht_block_count = components
            .iter()
            .map(|plan| plan.code_blocks().len())
            .sum::<usize>();
        let payload_bytes = payload.len();
        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count,
            classic_block_count,
            ht_block_count,
            payload_bytes,
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");

        Ok(CudaHtj2kColorDecodePlans {
            output_index: 0,
            dimensions: native_plan.dimensions,
            mct_dimensions: native_plan.dimensions,
            bit_depths: rgba_bit_depths_from_rgb(native_plan.bit_depths),
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from_native(native_plan.transform),
            payload,
            components,
            report,
        })
    }

    #[cfg(feature = "cuda-runtime")]
    pub(in crate::decoder) fn build_cuda_htj2k_color_region_plans_with_profile(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(self.bytes, &DecodeSettings::default())
            .map_err(native_decode_error)?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_color_plan_with_context(&mut native_context)
            .map_err(native_decode_error)?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let (payload, components) = flatten_cuda_color_components(
            &native_plan,
            fmt,
            Some(((roi.x, roi.y), (roi.w, roi.h))),
            "j2k CUDA region color decode plans",
        )?;
        let flatten_us = profile::elapsed_us(flatten_start);
        let block_count = components
            .iter()
            .map(CudaHtj2kDecodePlan::block_count)
            .sum::<usize>();
        let classic_block_count = components
            .iter()
            .map(|plan| plan.classic_code_blocks().len())
            .sum::<usize>();
        let ht_block_count = components
            .iter()
            .map(|plan| plan.code_blocks().len())
            .sum::<usize>();
        let payload_bytes = payload.len();
        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count,
            classic_block_count,
            ht_block_count,
            payload_bytes,
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");

        Ok(CudaHtj2kColorDecodePlans {
            output_index: 0,
            dimensions: (roi.w, roi.h),
            mct_dimensions: native_plan.dimensions,
            bit_depths: rgba_bit_depths_from_rgb(native_plan.bit_depths),
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from_native(native_plan.transform),
            payload,
            components,
            report,
        })
    }

    #[cfg(feature = "cuda-runtime")]
    pub(in crate::decoder) fn build_cuda_htj2k_color_region_scaled_plans_with_profile(
        &mut self,
        fmt: PixelFormat,
        scaled_roi: Rect,
        output_dimensions: (u32, u32),
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(
            self.bytes,
            &DecodeSettings {
                target_resolution: Some(output_dimensions),
                ..DecodeSettings::default()
            },
        )
        .map_err(native_decode_error)?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_color_plan_with_context(&mut native_context)
            .map_err(native_decode_error)?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let (payload, components) = flatten_cuda_color_components(
            &native_plan,
            fmt,
            Some(((scaled_roi.x, scaled_roi.y), (scaled_roi.w, scaled_roi.h))),
            "j2k CUDA scaled region color decode plans",
        )?;
        let flatten_us = profile::elapsed_us(flatten_start);
        let block_count = components
            .iter()
            .map(CudaHtj2kDecodePlan::block_count)
            .sum::<usize>();
        let classic_block_count = components
            .iter()
            .map(|plan| plan.classic_code_blocks().len())
            .sum::<usize>();
        let ht_block_count = components
            .iter()
            .map(|plan| plan.code_blocks().len())
            .sum::<usize>();
        let payload_bytes = payload.len();
        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count,
            classic_block_count,
            ht_block_count,
            payload_bytes,
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");

        Ok(CudaHtj2kColorDecodePlans {
            output_index: 0,
            dimensions: (scaled_roi.w, scaled_roi.h),
            mct_dimensions: native_plan.dimensions,
            bit_depths: rgba_bit_depths_from_rgb(native_plan.bit_depths),
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from_native(native_plan.transform),
            payload,
            components,
            report,
        })
    }
}
