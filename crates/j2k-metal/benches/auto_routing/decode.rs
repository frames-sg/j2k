// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use criterion::Criterion;
use j2k_core::{BackendKind, BackendRequest, DeviceSurface, Downscale, PixelFormat, Rect};
use j2k_metal::{
    J2kDecoder, MetalBackendSession, MetalDecodeRequest, MetalTileBatch, SurfaceResidency,
};
use j2k_test_support::{
    append_auto_routing_output as append_output, auto_routing_operation_label as operation_label,
    auto_routing_route_cell as route_cell, auto_routing_sha256,
    validate_auto_routing_decode_identity, AutoRoutingCell, AutoRoutingOperation,
    AutoRoutingPixelFormat, AutoRoutingWorkload,
};

use crate::assert_output_parity;

const BATCH_SIZE: usize = 16;

pub(crate) struct DecodeCase<'a> {
    id: &'a str,
    bytes: &'a [u8],
    fmt: PixelFormat,
    dimensions: (u32, u32),
    shared: Arc<[u8]>,
}

impl<'a> DecodeCase<'a> {
    pub(crate) fn new(workload: &'a AutoRoutingWorkload) -> Self {
        let fmt = pixel_format(workload.pixel_format);
        let info = j2k::J2kDecoder::inspect(&workload.bytes)
            .unwrap_or_else(|error| panic!("inspect decode workload {}: {error}", workload.id));
        let support = j2k::J2kDecoder::inspect_support(&workload.bytes)
            .unwrap_or_else(|error| panic!("inspect decode support {}: {error}", workload.id));
        validate_auto_routing_decode_identity(
            workload,
            support.transfer_syntax,
            support.payload_kind,
        )
        .unwrap_or_else(|error| panic!("validate decode workload {}: {error}", workload.id));
        Self {
            id: &workload.id,
            bytes: &workload.bytes,
            fmt,
            dimensions: info.dimensions,
            shared: Arc::from(workload.bytes.clone()),
        }
    }
}

pub(crate) fn bench_cell(
    criterion: &mut Criterion,
    case: &DecodeCase<'_>,
    operation: AutoRoutingOperation,
    session: &MetalBackendSession,
) -> AutoRoutingCell {
    let cpu = decode_once(case, operation, BackendRequest::Cpu, session)
        .unwrap_or_else(|error| panic!("CPU {} {}: {error}", case.id, operation_label(operation)));
    let hybrid =
        decode_once(case, operation, BackendRequest::Metal, session).unwrap_or_else(|error| {
            panic!(
                "hybrid Metal {} {}: {error}",
                case.id,
                operation_label(operation)
            )
        });
    assert_output_parity(case.id, operation, &cpu, &hybrid);
    let auto = decode_once(case, operation, BackendRequest::Auto, session)
        .unwrap_or_else(|error| panic!("Auto {} {}: {error}", case.id, operation_label(operation)));
    assert_output_parity(case.id, operation, &cpu, &auto);

    let group_id = format!(
        "auto-routing_{}_{id}",
        operation_label(operation),
        id = case.id
    );
    let mut group = criterion.benchmark_group(&group_id);
    group.bench_function("cpu", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(
                decode_once(case, operation, BackendRequest::Cpu, session)
                    .expect("measured CPU Metal-adapter decode"),
            )
        });
    });
    group.bench_function("hybrid", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(
                decode_once(case, operation, BackendRequest::Metal, session)
                    .expect("measured hybrid Metal decode"),
            )
        });
    });
    group.finish();
    route_cell(case.id, operation, &group_id, auto_routing_sha256(&cpu))
}

fn decode_once(
    case: &DecodeCase<'_>,
    operation: AutoRoutingOperation,
    backend: BackendRequest,
    session: &MetalBackendSession,
) -> Result<Vec<u8>, String> {
    if operation == AutoRoutingOperation::BatchDecode {
        return decode_batch_once(case, backend);
    }
    let request = decode_request(case, operation, backend)?;
    let mut decoder = J2kDecoder::new(case.bytes).map_err(|error| error.to_string())?;
    let surface = decoder
        .decode_request_to_device_with_session(request, session)
        .map_err(|error| error.to_string())?;
    assert_surface_route(&surface, backend)?;
    surface
        .as_bytes()
        .map(std::borrow::Cow::into_owned)
        .map_err(|error| error.to_string())
}

fn decode_batch_once(case: &DecodeCase<'_>, backend: BackendRequest) -> Result<Vec<u8>, String> {
    let mut batch = MetalTileBatch::with_capacity(BATCH_SIZE);
    for _ in 0..BATCH_SIZE {
        batch
            .push_shared_tile_request(
                Arc::clone(&case.shared),
                MetalDecodeRequest::full(case.fmt, backend),
            )
            .map_err(|error| error.to_string())?;
    }
    let surfaces = batch.decode_all().map_err(|error| error.to_string())?;
    if surfaces.len() != BATCH_SIZE {
        return Err(format!(
            "Metal batch returned {} surfaces for {BATCH_SIZE} inputs",
            surfaces.len()
        ));
    }
    let mut output = Vec::new();
    for surface in &surfaces {
        assert_surface_route(surface, backend)?;
        let bytes = surface.as_bytes().map_err(|error| error.to_string())?;
        append_output(&mut output, &bytes)?;
    }
    Ok(output)
}

fn decode_request(
    case: &DecodeCase<'_>,
    operation: AutoRoutingOperation,
    backend: BackendRequest,
) -> Result<MetalDecodeRequest, String> {
    match operation {
        AutoRoutingOperation::FullDecode => Ok(MetalDecodeRequest::full(case.fmt, backend)),
        AutoRoutingOperation::RoiDecode => Ok(MetalDecodeRequest::region(
            case.fmt,
            benchmark_roi(case.dimensions),
            backend,
        )),
        AutoRoutingOperation::ScaledDecode => Ok(MetalDecodeRequest::scaled(
            case.fmt,
            Downscale::Half,
            backend,
        )),
        AutoRoutingOperation::BatchDecode
        | AutoRoutingOperation::LosslessEncode
        | AutoRoutingOperation::LossyEncode => {
            Err("invalid single-image decode operation".to_string())
        }
    }
}

fn assert_surface_route(
    surface: &j2k_metal::Surface,
    backend: BackendRequest,
) -> Result<(), String> {
    match backend {
        BackendRequest::Cpu if surface.backend_kind() == BackendKind::Cpu => Ok(()),
        BackendRequest::Metal
            if surface.backend_kind() == BackendKind::Metal
                && surface.residency() == SurfaceResidency::MetalResidentDecode =>
        {
            Ok(())
        }
        BackendRequest::Auto
            if surface.backend_kind() == BackendKind::Cpu
                || (surface.backend_kind() == BackendKind::Metal
                    && surface.residency() == SurfaceResidency::MetalResidentDecode) =>
        {
            Ok(())
        }
        _ => Err(format!(
            "requested {backend:?} but received {:?}/{:?}",
            surface.backend_kind(),
            surface.residency()
        )),
    }
}

fn benchmark_roi(dimensions: (u32, u32)) -> Rect {
    let width = (dimensions.0 / 2).max(1);
    let height = (dimensions.1 / 2).max(1);
    Rect {
        x: dimensions.0.saturating_sub(width) / 2,
        y: dimensions.1.saturating_sub(height) / 2,
        w: width,
        h: height,
    }
}

const fn pixel_format(format: AutoRoutingPixelFormat) -> PixelFormat {
    match format {
        AutoRoutingPixelFormat::Gray8 => PixelFormat::Gray8,
        AutoRoutingPixelFormat::Rgb8 => PixelFormat::Rgb8,
    }
}
