// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained HTJ2K and classic color-plan construction.

use super::{
    flatten_referenced_classic_cuda_color_tile_components,
    flatten_referenced_cuda_color_tile_components, profile, rgba_bit_depths_from_rgb,
    CudaHtj2kColorDecodePlans, CudaHtj2kDecodePlan, CudaHtj2kDecodeProfileDetail,
    CudaHtj2kProfileReport, CudaHtj2kTransform, DeviceDecodePlan, Error, HostPhaseBudget,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, PixelFormat,
};

#[cfg(feature = "cuda-runtime")]
#[expect(
    clippy::too_many_lines,
    reason = "tile metadata, payload spans, component owners, and exact output geometry are validated atomically"
)]
pub(in crate::decoder) fn build_cuda_htj2k_color_plans_from_referenced_with_profile(
    input: &[u8],
    referenced: &J2kReferencedHtj2kPlan,
    fmt: PixelFormat,
    device_plan: DeviceDecodePlan,
    shared_payload: &mut Vec<u8>,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<CudaHtj2kColorDecodePlans>, Error> {
    let total_start = profile::profile_now(true);
    let output_rect = referenced.output_rect();
    let output_dimensions = device_plan.output_dims();
    if (output_rect.width(), output_rect.height()) != output_dimensions {
        return Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA color tile output geometry is inconsistent",
        });
    }
    let mut colors = host_budget.try_vec_with_capacity(referenced.tiles().len())?;
    let mut next_payload = 0usize;
    for tile in referenced.tiles() {
        let (geometry_dimensions, bit_depths, mct, transform, component_plans) =
            if let Some(geometry) = tile.color_geometry() {
                (
                    geometry.dimensions,
                    rgba_bit_depths_from_rgb(geometry.bit_depths),
                    geometry.mct,
                    geometry.transform,
                    geometry.component_plans.as_slice(),
                )
            } else if let Some(geometry) = tile.rgba_geometry() {
                (
                    geometry.dimensions,
                    geometry.bit_depths,
                    geometry.mct,
                    geometry.transform,
                    geometry.component_plans.as_slice(),
                )
            } else {
                return Err(Error::UnsupportedCudaRequest {
                    reason: "prepared CUDA color batch received a grayscale HTJ2K tile",
                });
            };
        if component_plans.len() != fmt.channels() {
            return Err(Error::UnsupportedCudaRequest {
                reason: "prepared CUDA color tile component count does not match its output format",
            });
        }
        let span = tile.payload_records();
        if span.first_record != next_payload {
            return Err(Error::UnsupportedCudaRequest {
                reason: "prepared CUDA color tile payload spans are not contiguous",
            });
        }
        let payload_end = span.end_record().ok_or(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA color tile payload span overflows",
        })?;
        let payloads = referenced
            .payloads()
            .get(span.first_record..payload_end)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: "prepared CUDA color tile payload span is out of bounds",
            })?;
        let payload_start = shared_payload.len();
        let flatten_start = profile::profile_now(true);
        let components = flatten_referenced_cuda_color_tile_components(
            component_plans,
            payloads,
            input,
            fmt,
            (output_rect.x0, output_rect.y0),
            output_dimensions,
            shared_payload,
            host_budget,
        )?;
        let block_count = components
            .iter()
            .map(CudaHtj2kDecodePlan::block_count)
            .sum::<usize>();
        let report = CudaHtj2kProfileReport {
            parse_us: 0,
            plan_us: 0,
            flatten_us: profile::elapsed_us(flatten_start),
            total_us: profile::elapsed_us(total_start),
            block_count,
            classic_block_count: 0,
            ht_block_count: block_count,
            payload_bytes: shared_payload.len().saturating_sub(payload_start),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("prepared_color_tile_plan");
        colors.push(CudaHtj2kColorDecodePlans {
            output_index: 0,
            dimensions: output_dimensions,
            mct_dimensions: geometry_dimensions,
            bit_depths,
            mct,
            transform: CudaHtj2kTransform::from_native(transform),
            payload: Vec::new(),
            components,
            report,
        });
        next_payload = payload_end;
    }
    if next_payload != referenced.payloads().len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA color tile payloads contain trailing records",
        });
    }
    Ok(colors)
}

#[cfg(feature = "cuda-runtime")]
#[expect(
    clippy::too_many_lines,
    reason = "classic tile metadata, payload spans, component owners, and exact output geometry are validated atomically"
)]
pub(in crate::decoder) fn build_cuda_classic_color_plans_from_referenced_with_profile(
    input: &[u8],
    referenced: &J2kReferencedClassicPlan,
    fmt: PixelFormat,
    device_plan: DeviceDecodePlan,
    shared_payload: &mut Vec<u8>,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<CudaHtj2kColorDecodePlans>, Error> {
    let total_start = profile::profile_now(true);
    CudaHtj2kDecodePlan::validate_referenced_classic_payload_sequence(
        referenced.payloads(),
        referenced.ranges(),
    )?;
    let output_rect = referenced.output_rect();
    let output_dimensions = device_plan.output_dims();
    if (output_rect.width(), output_rect.height()) != output_dimensions {
        return Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA classic color tile output geometry is inconsistent",
        });
    }
    let mut colors = host_budget.try_vec_with_capacity(referenced.tiles().len())?;
    let mut next_payload = 0usize;
    for tile in referenced.tiles() {
        let (geometry_dimensions, bit_depths, mct, transform, component_plans) =
            if let Some(geometry) = tile.color_geometry() {
                (
                    geometry.dimensions,
                    rgba_bit_depths_from_rgb(geometry.bit_depths),
                    geometry.mct,
                    geometry.transform,
                    geometry.component_plans.as_slice(),
                )
            } else if let Some(geometry) = tile.rgba_geometry() {
                (
                    geometry.dimensions,
                    geometry.bit_depths,
                    geometry.mct,
                    geometry.transform,
                    geometry.component_plans.as_slice(),
                )
            } else {
                return Err(Error::UnsupportedCudaRequest {
                    reason: "prepared CUDA classic color batch received a grayscale tile",
                });
            };
        if component_plans.len() != fmt.channels() {
            return Err(Error::UnsupportedCudaRequest {
                reason: "prepared CUDA classic color tile component count does not match its output format",
            });
        }
        let span = tile.payload_records();
        if span.first_record != next_payload {
            return Err(Error::UnsupportedCudaRequest {
                reason: "prepared CUDA classic color tile payload spans are not contiguous",
            });
        }
        let payload_end = span.end_record().ok_or(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA classic color tile payload span overflows",
        })?;
        let payloads = referenced
            .payloads()
            .get(span.first_record..payload_end)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: "prepared CUDA classic color tile payload span is out of bounds",
            })?;
        let payload_start = shared_payload.len();
        let flatten_start = profile::profile_now(true);
        let components = flatten_referenced_classic_cuda_color_tile_components(
            component_plans,
            payloads,
            referenced.ranges(),
            input,
            fmt,
            (output_rect.x0, output_rect.y0),
            output_dimensions,
            shared_payload,
            host_budget,
        )?;
        let block_count = components
            .iter()
            .map(CudaHtj2kDecodePlan::block_count)
            .sum::<usize>();
        let classic_block_count = components
            .iter()
            .map(|plan| plan.classic_code_blocks().len())
            .sum::<usize>();
        let report = CudaHtj2kProfileReport {
            parse_us: 0,
            plan_us: 0,
            flatten_us: profile::elapsed_us(flatten_start),
            total_us: profile::elapsed_us(total_start),
            block_count,
            classic_block_count,
            ht_block_count: 0,
            payload_bytes: shared_payload.len().saturating_sub(payload_start),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("prepared_classic_color_tile_plan");
        colors.push(CudaHtj2kColorDecodePlans {
            output_index: 0,
            dimensions: output_dimensions,
            mct_dimensions: geometry_dimensions,
            bit_depths,
            mct,
            transform: CudaHtj2kTransform::from_native(transform),
            payload: Vec::new(),
            components,
            report,
        });
        next_payload = payload_end;
    }
    if next_payload != referenced.payloads().len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA classic color tile payloads contain trailing records",
        });
    }
    Ok(colors)
}
