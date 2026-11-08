// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region and reduced-resolution color planning on the decoder facade.

use super::{
    build_cuda_htj2k_color_plans_from_bytes_with_profile, flatten_cuda_color_components,
    native_decode_error, parse_at_scale, profile, rgba_bit_depths_from_rgb,
    CudaHtj2kColorDecodePlans, CudaHtj2kDecodePlan, CudaHtj2kDecodeProfileDetail,
    CudaHtj2kProfileReport, CudaHtj2kTransform, DecodeSettings, Downscale, Error, J2kDecoder,
    NativeDecoderContext, NativeImage, PixelFormat, Rect,
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
        scale: Downscale,
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = parse_at_scale(self.bytes, &DecodeSettings::default(), scale, 0)?;
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
            .build_direct_color_plan_region_with_context(
                &mut native_context,
                (roi.x, roi.y, roi.w, roi.h),
            )
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
            mct_dimensions: final_band_dimensions(&components),
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
        scale: Downscale,
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = parse_at_scale(self.bytes, &DecodeSettings::default(), scale, 0)?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_color_plan_region_with_context(
                &mut native_context,
                (scaled_roi.x, scaled_roi.y, scaled_roi.w, scaled_roi.h),
            )
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
            mct_dimensions: final_band_dimensions(&components),
            bit_depths: rgba_bit_depths_from_rgb(native_plan.bit_depths),
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from_native(native_plan.transform),
            payload,
            components,
            report,
        })
    }
}

/// Dimensions of the reconstructed component planes the colour stores read: a
/// region plan's native dimensions are the region, not the planes.
#[cfg(feature = "cuda-runtime")]
fn final_band_dimensions(components: &[CudaHtj2kDecodePlan]) -> (u32, u32) {
    components
        .first()
        .and_then(|component| component.store_steps().first())
        .map_or((0, 0), |store| {
            (
                store.input_rect.x1.saturating_sub(store.input_rect.x0),
                store.input_rect.y1.saturating_sub(store.input_rect.y0),
            )
        })
}

#[cfg(test)]
mod reduction_tests {
    use super::*;
    use crate::decoder::plan::build_cuda_color_plan_from_bytes_for_device_plan_with_profile;
    use j2k::{DeviceDecodePlan, DeviceDecodeRequest};
    use j2k_test_support::{OPENJPEG_ORIGIN_23X17_AT_3_5, OPENJPH_ORIGIN_23X17_AT_3_5};

    #[test]
    fn scaled_color_plan_preserves_reference_grid_reduction() {
        for bytes in [OPENJPEG_ORIGIN_23X17_AT_3_5, OPENJPH_ORIGIN_23X17_AT_3_5] {
            let mut decoder = J2kDecoder::new(bytes).expect("origin fixture");
            for (scale, dimensions) in [
                (Downscale::Half, (11, 8)),
                (Downscale::Quarter, (6, 4)),
                (Downscale::Eighth, (3, 2)),
            ] {
                let full = decoder
                    .build_cuda_htj2k_color_scaled_plans_with_profile(PixelFormat::Rgb8, scale)
                    .expect("scaled plan");
                assert_eq!(full.dimensions, dimensions);
                assert_eq!(full.mct_dimensions, dimensions);
                for component in &full.components {
                    assert_eq!(component.dimensions(), dimensions);
                }
                for request in [
                    DeviceDecodeRequest::Scaled { scale },
                    DeviceDecodeRequest::RegionScaled {
                        roi: Rect {
                            x: 15,
                            y: 9,
                            w: 8,
                            h: 8,
                        },
                        scale,
                    },
                ] {
                    let request_plan = DeviceDecodePlan::for_reference_grid(
                        decoder.inner.reference_grid(),
                        request,
                    )
                    .expect("geometry plan");
                    let prepared = build_cuda_color_plan_from_bytes_for_device_plan_with_profile(
                        bytes,
                        PixelFormat::Rgb8,
                        request_plan,
                        DecodeSettings::default(),
                        &mut NativeDecoderContext::default(),
                    )
                    .expect("prepared plan");
                    assert_eq!(prepared.dimensions, request_plan.output_dims());
                    assert_eq!(prepared.mct_dimensions, dimensions);
                    if let DeviceDecodeRequest::RegionScaled { .. } = request {
                        let region = decoder
                            .build_cuda_htj2k_color_region_scaled_plans_with_profile(
                                PixelFormat::Rgb8,
                                request_plan.output_rect(),
                                scale,
                            )
                            .expect("scaled edge region");
                        assert_eq!(region.dimensions, request_plan.output_dims());
                    }
                }
            }
        }
    }
}
