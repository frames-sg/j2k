// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::Criterion;
use j2k_core::{
    BackendKind, BackendRequest, DeviceSubmission, DeviceSurface, Downscale, ImageDecodeSubmit,
    PixelFormat, Rect, TileBatchDecodeManyDevice,
};
use j2k_cuda::{Codec, CudaSession, J2kDecoder, SurfaceResidency};
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
}

impl<'a> DecodeCase<'a> {
    pub(crate) fn new(workload: &'a AutoRoutingWorkload) -> Self {
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
            fmt: pixel_format(workload.pixel_format),
            dimensions: info.dimensions,
        }
    }
}

pub(crate) fn bench_cell(
    criterion: &mut Criterion,
    case: &DecodeCase<'_>,
    operation: AutoRoutingOperation,
) -> AutoRoutingCell {
    let mut cpu_probe_session = CudaSession::default();
    let cpu = decode_once(case, operation, BackendRequest::Cpu, &mut cpu_probe_session)
        .unwrap_or_else(|error| panic!("CPU {} {}: {error}", case.id, operation_label(operation)));
    let mut hybrid_probe_session = CudaSession::default();
    let hybrid = decode_once(
        case,
        operation,
        BackendRequest::Cuda,
        &mut hybrid_probe_session,
    )
    .unwrap_or_else(|error| {
        panic!(
            "hybrid CUDA {} {}: {error}",
            case.id,
            operation_label(operation)
        )
    });
    if operation == AutoRoutingOperation::BatchDecode {
        let mut reference_session = CudaSession::default();
        let single = decode_once(
            case,
            AutoRoutingOperation::FullDecode,
            BackendRequest::Cpu,
            &mut reference_session,
        )
        .unwrap_or_else(|error| panic!("CPU {} batch reference: {error}", case.id));
        let reference = repeated_batch_output(&single)
            .unwrap_or_else(|error| panic!("CPU {} batch reference: {error}", case.id));
        assert_output_parity(
            &format!("{} CPU batch versus repeated full decode", case.id),
            operation,
            &reference,
            &cpu,
        );
        assert_output_parity(
            &format!("{} CUDA batch versus repeated CPU full decode", case.id),
            operation,
            &reference,
            &hybrid,
        );
    }
    assert_output_parity(case.id, operation, &cpu, &hybrid);

    let group_id = format!(
        "auto-routing_{}_{id}",
        operation_label(operation),
        id = case.id
    );
    let mut cpu_session = CudaSession::default();
    let mut hybrid_session = hybrid_probe_session;
    let mut group = criterion.benchmark_group(&group_id);
    group.bench_function("cpu", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(
                decode_once(case, operation, BackendRequest::Cpu, &mut cpu_session)
                    .expect("measured CPU CUDA-adapter decode"),
            )
        });
    });
    group.bench_function("hybrid", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(
                decode_once(case, operation, BackendRequest::Cuda, &mut hybrid_session)
                    .expect("measured hybrid CUDA decode"),
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
    session: &mut CudaSession,
) -> Result<Vec<u8>, String> {
    if operation == AutoRoutingOperation::BatchDecode {
        return decode_batch_once(case, backend, session);
    }
    let mut decoder = J2kDecoder::new(case.bytes).map_err(|error| error.to_string())?;
    let submission = match operation {
        AutoRoutingOperation::FullDecode => decoder
            .submit_to_device(session, case.fmt, backend)
            .map_err(|error| error.to_string())?,
        AutoRoutingOperation::RoiDecode => decoder
            .submit_region_to_device(session, case.fmt, benchmark_roi(case.dimensions), backend)
            .map_err(|error| error.to_string())?,
        AutoRoutingOperation::ScaledDecode => decoder
            .submit_scaled_to_device(session, case.fmt, Downscale::Half, backend)
            .map_err(|error| error.to_string())?,
        AutoRoutingOperation::BatchDecode
        | AutoRoutingOperation::LosslessEncode
        | AutoRoutingOperation::LossyEncode => {
            return Err("invalid single-image decode operation".to_string())
        }
    };
    let surface = submission.wait().map_err(|error| error.to_string())?;
    assert_surface_route(&surface, backend)?;
    surface_bytes(&surface)
}

fn decode_batch_once(
    case: &DecodeCase<'_>,
    backend: BackendRequest,
    session: &mut CudaSession,
) -> Result<Vec<u8>, String> {
    let mut inputs = Vec::new();
    inputs
        .try_reserve_exact(BATCH_SIZE)
        .map_err(|_| "allocate CUDA Auto-routing batch inputs".to_string())?;
    inputs.resize(BATCH_SIZE, case.bytes);
    let surfaces = if backend == BackendRequest::Cuda {
        J2kDecoder::decode_batch_to_device_with_session(&inputs, case.fmt, session)
            .map_err(|error| error.to_string())?
    } else {
        let mut context = j2k_cuda::J2kContext::default();
        let mut pool = j2k_cuda::J2kScratchPool::new();
        Codec::decode_tiles_to_device(&mut context, &mut pool, &inputs, case.fmt, backend)
            .map_err(|error| error.to_string())?
    };
    if surfaces.len() != BATCH_SIZE {
        return Err(format!(
            "CUDA batch returned {} surfaces for {BATCH_SIZE} inputs",
            surfaces.len()
        ));
    }
    let mut output = Vec::new();
    for surface in &surfaces {
        assert_surface_route(surface, backend)?;
        let bytes = surface_bytes(surface)?;
        append_output(&mut output, &bytes)?;
    }
    Ok(output)
}

fn repeated_batch_output(single: &[u8]) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    for _ in 0..BATCH_SIZE {
        append_output(&mut output, single)?;
    }
    Ok(output)
}

fn surface_bytes(surface: &j2k_cuda::Surface) -> Result<Vec<u8>, String> {
    let (width, height) = surface.dimensions();
    let stride = usize::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(surface.pixel_format().bytes_per_pixel()))
        .ok_or_else(|| "CUDA surface stride overflow".to_string())?;
    let len = stride
        .checked_mul(usize::try_from(height).map_err(|_| "CUDA surface height overflow")?)
        .ok_or_else(|| "CUDA surface byte length overflow".to_string())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| "allocate CUDA Auto-routing surface readback".to_string())?;
    bytes.resize(len, 0);
    surface
        .download_into(&mut bytes, stride)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

fn assert_surface_route(
    surface: &j2k_cuda::Surface,
    backend: BackendRequest,
) -> Result<(), String> {
    match backend {
        BackendRequest::Cpu if surface.backend_kind() == BackendKind::Cpu => Ok(()),
        BackendRequest::Cuda
            if surface.backend_kind() == BackendKind::Cuda
                && surface.residency() == SurfaceResidency::CudaResidentDecode =>
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
