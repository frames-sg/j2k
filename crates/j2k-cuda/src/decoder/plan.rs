// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    native_decode_error, profile, CudaHtj2kColorDecodePlans, CudaHtj2kDecodePlan,
    CudaHtj2kDecodeProfileDetail, CudaHtj2kProfileReport, CudaHtj2kTransform, DecodeSettings,
    Error, J2kDecoder, NativeDecoderContext, NativeImage, PixelFormat, Rect,
};
#[cfg(feature = "cuda-runtime")]
mod color_owners;
#[cfg(feature = "cuda-runtime")]
use self::color_owners::flatten_cuda_color_components;

impl J2kDecoder<'_> {
    /// Build a flat CUDA HTJ2K grayscale decode plan and return stage timings.
    pub(crate) fn build_cuda_htj2k_grayscale_plan_with_profile(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(self.bytes, &DecodeSettings::default())
            .map_err(native_decode_error)?;
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
        output_dimensions: (u32, u32),
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
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
        output_dimensions: (u32, u32),
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
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

    /// Build flat CUDA HTJ2K RGB component plans and return stage timings.
    #[cfg(feature = "cuda-runtime")]
    pub(super) fn build_cuda_htj2k_color_plans_with_profile(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let mut native_context = NativeDecoderContext::default();
        build_cuda_htj2k_color_plans_from_bytes_with_profile(self.bytes, fmt, &mut native_context)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(super) fn build_cuda_htj2k_color_scaled_plans_with_profile(
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
            dimensions: native_plan.dimensions,
            mct_dimensions: native_plan.dimensions,
            bit_depths: native_plan.bit_depths,
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from_native(native_plan.transform),
            payload,
            components,
            report,
        })
    }

    #[cfg(feature = "cuda-runtime")]
    pub(super) fn build_cuda_htj2k_color_region_plans_with_profile(
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
            dimensions: (roi.w, roi.h),
            mct_dimensions: native_plan.dimensions,
            bit_depths: native_plan.bit_depths,
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from_native(native_plan.transform),
            payload,
            components,
            report,
        })
    }

    #[cfg(feature = "cuda-runtime")]
    pub(super) fn build_cuda_htj2k_color_region_scaled_plans_with_profile(
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
            dimensions: (scaled_roi.w, scaled_roi.h),
            mct_dimensions: native_plan.dimensions,
            bit_depths: native_plan.bit_depths,
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from_native(native_plan.transform),
            payload,
            components,
            report,
        })
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn build_cuda_htj2k_color_plans_from_bytes_with_profile<'a>(
    input: &'a [u8],
    fmt: PixelFormat,
    native_context: &mut NativeDecoderContext<'a>,
) -> Result<CudaHtj2kColorDecodePlans, Error> {
    let total_start = profile::profile_now(true);

    let parse_start = profile::profile_now(true);
    let image = NativeImage::new(input, &DecodeSettings::default()).map_err(native_decode_error)?;
    let parse_us = profile::elapsed_us(parse_start);

    let plan_start = profile::profile_now(true);
    let native_plan = image
        .build_direct_color_plan_with_context(native_context)
        .map_err(native_decode_error)?;
    let plan_us = profile::elapsed_us(plan_start);

    let flatten_start = profile::profile_now(true);
    let (payload, components) =
        flatten_cuda_color_components(&native_plan, fmt, None, "j2k CUDA color decode plans")?;
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
        dimensions: native_plan.dimensions,
        mct_dimensions: native_plan.dimensions,
        bit_depths: native_plan.bit_depths,
        mct: native_plan.mct,
        transform: CudaHtj2kTransform::from_native(native_plan.transform),
        payload,
        components,
        report,
    })
}
