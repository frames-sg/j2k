// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    J2kClassicCodeBlockPayload, J2kCodestreamRange, J2kRect, J2kReferencedClassicPlan,
    J2kReferencedTilePlan,
};

use super::super::{
    flatten_referenced_classic_cuda_color_tile_components, profile, rgba_bit_depths_from_rgb,
    CudaHtj2kColorDecodePlans, CudaHtj2kDecodePlan, CudaHtj2kDecodeProfileDetail,
    CudaHtj2kProfileReport, CudaHtj2kTransform, DeviceDecodePlan, Error, HostPhaseBudget,
    PixelFormat,
};
use super::ReferencedTileColorGeometry;

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
        let (payload_end, payloads) = classic_tile_payloads(referenced, tile, next_payload)?;
        let mut color = build_referenced_classic_color_tile(
            (input, referenced.ranges()),
            tile,
            payloads,
            fmt,
            output_rect,
            shared_payload,
            host_budget,
        )?;
        color.report.total_us = profile::elapsed_us(total_start);
        color.report.emit("prepared_classic_color_tile_plan");
        colors.push(color);
        next_payload = payload_end;
    }
    if next_payload != referenced.payloads().len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA classic color tile payloads contain trailing records",
        });
    }
    Ok(colors)
}

fn build_referenced_classic_color_tile(
    source: (&[u8], &[J2kCodestreamRange]),
    tile: &J2kReferencedTilePlan,
    payloads: &[J2kClassicCodeBlockPayload],
    fmt: PixelFormat,
    output_rect: J2kRect,
    shared_payload: &mut Vec<u8>,
    host_budget: &mut HostPhaseBudget,
) -> Result<CudaHtj2kColorDecodePlans, Error> {
    let (geometry_dimensions, bit_depths, mct, transform, component_plans) =
        classic_tile_color_geometry(tile)?;
    if component_plans.len() != fmt.channels() {
        return Err(Error::UnsupportedCudaRequest {
            reason:
                "prepared CUDA classic color tile component count does not match its output format",
        });
    }
    let payload_start = shared_payload.len();
    let flatten_start = profile::profile_now(true);
    let components = flatten_referenced_classic_cuda_color_tile_components(
        component_plans,
        payloads,
        source.1,
        source.0,
        fmt,
        (output_rect.x0, output_rect.y0),
        (output_rect.width(), output_rect.height()),
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
        flatten_us: profile::elapsed_us(flatten_start),
        block_count,
        classic_block_count,
        payload_bytes: shared_payload.len().saturating_sub(payload_start),
        residency: crate::SurfaceResidency::CudaResidentDecode,
        detail: CudaHtj2kDecodeProfileDetail::default(),
        ..CudaHtj2kProfileReport::default()
    };
    Ok(CudaHtj2kColorDecodePlans {
        output_index: 0,
        dimensions: (output_rect.width(), output_rect.height()),
        mct_dimensions: geometry_dimensions,
        bit_depths,
        mct,
        transform: CudaHtj2kTransform::from_native(transform),
        payload: Vec::new(),
        components,
        report,
    })
}

fn classic_tile_color_geometry(
    tile: &J2kReferencedTilePlan,
) -> Result<ReferencedTileColorGeometry<'_>, Error> {
    if let Some(geometry) = tile.color_geometry() {
        Ok((
            geometry.dimensions,
            rgba_bit_depths_from_rgb(geometry.bit_depths),
            geometry.mct,
            geometry.transform,
            &geometry.component_plans,
        ))
    } else if let Some(geometry) = tile.rgba_geometry() {
        Ok((
            geometry.dimensions,
            geometry.bit_depths,
            geometry.mct,
            geometry.transform,
            &geometry.component_plans,
        ))
    } else {
        Err(Error::UnsupportedCudaRequest {
            reason: "prepared CUDA classic color batch received a grayscale tile",
        })
    }
}

fn classic_tile_payloads<'a>(
    referenced: &'a J2kReferencedClassicPlan,
    tile: &J2kReferencedTilePlan,
    next_payload: usize,
) -> Result<(usize, &'a [J2kClassicCodeBlockPayload]), Error> {
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
    Ok((payload_end, payloads))
}
