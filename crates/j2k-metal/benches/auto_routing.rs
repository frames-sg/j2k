// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(not(target_os = "macos"))]
fn main() {
    assert!(
        std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_none(),
        "J2K Metal Auto-routing benchmark requires macOS"
    );
    eprintln!("J2K Metal Auto-routing benchmark skipped outside macOS");
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
mod macos {
    use std::{path::PathBuf, sync::Arc, time::Duration};

    use criterion::Criterion;
    use j2k::{
        encode_j2k_lossless, encode_j2k_lossless_with_accelerator, encode_j2k_lossy,
        encode_j2k_lossy_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
        J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kLossyEncodeOptions,
        J2kLossySamples, J2kRateTarget,
    };
    use j2k_core::{
        BackendKind, BackendRequest, CompressedTransferSyntax, DeviceSurface, Downscale,
        PixelFormat, Rect,
    };
    use j2k_metal::{
        J2kDecoder, MetalBackendSession, MetalDecodeRequest, MetalEncodeStageAccelerator,
        MetalTileBatch, SurfaceResidency,
    };
    use j2k_test_support::{
        append_auto_routing_output as append_output,
        auto_routing_operation_label as operation_label, auto_routing_route_cell as route_cell,
        auto_routing_sha256, load_auto_routing_manifest, load_auto_routing_pnm,
        write_auto_routing_evidence, AutoRoutingBackend, AutoRoutingCell, AutoRoutingEvidence,
        AutoRoutingOperation, AutoRoutingPixelFormat, AutoRoutingPlatform, AutoRoutingPnm,
        AutoRoutingWorkload, AutoRoutingWorkloadKind,
    };

    const SAMPLE_SIZE: usize = 10;
    const WARM_UP: Duration = Duration::from_secs(1);
    const MEASUREMENT: Duration = Duration::from_secs(3);
    const BATCH_SIZE: usize = 16;
    const AUTO_GRAY8_MIN_PIXELS: u64 = 2_960_793;
    const AUTO_RGB8_BATCH_MIN_PIXELS: u64 = 307_200;
    const AUTO_RGB8_LARGE_MIN_PIXELS: u64 = 5_038_848;

    pub(super) fn run() {
        let manifest_path = required_path("J2K_AUTO_ROUTING_MANIFEST");
        let corpus_root = required_path("J2K_AUTO_ROUTING_ROOT");
        let evidence_path = required_path("J2K_AUTO_ROUTING_EVIDENCE");
        let workloads = load_auto_routing_manifest(&manifest_path, &corpus_root)
            .unwrap_or_else(|error| panic!("load Metal Auto-routing workloads: {error}"));
        let session = MetalBackendSession::system_default()
            .unwrap_or_else(|error| panic!("Metal Auto-routing benchmark needs a device: {error}"));
        let mut criterion = Criterion::default()
            .sample_size(SAMPLE_SIZE)
            .warm_up_time(WARM_UP)
            .measurement_time(MEASUREMENT)
            .configure_from_args();
        let mut cells = Vec::new();

        for workload in &workloads.workloads {
            match workload.kind {
                AutoRoutingWorkloadKind::Decode => {
                    let decode = DecodeCase::new(workload);
                    for operation in [
                        AutoRoutingOperation::FullDecode,
                        AutoRoutingOperation::RoiDecode,
                        AutoRoutingOperation::ScaledDecode,
                        AutoRoutingOperation::BatchDecode,
                    ] {
                        cells.push(bench_decode_cell(
                            &mut criterion,
                            &decode,
                            operation,
                            &session,
                        ));
                    }
                }
                AutoRoutingWorkloadKind::Encode => {
                    let encode = load_auto_routing_pnm(workload).unwrap_or_else(|error| {
                        panic!("load Metal encode workload {}: {error}", workload.id)
                    });
                    for operation in [
                        AutoRoutingOperation::LosslessEncode,
                        AutoRoutingOperation::LossyEncode,
                    ] {
                        cells.push(bench_encode_cell(&mut criterion, &encode, operation));
                    }
                }
            }
        }
        criterion.final_summary();

        let evidence = AutoRoutingEvidence {
            schema_version: 1,
            candidate_sha: required_env("J2K_AUTO_ROUTING_CANDIDATE_SHA"),
            backend: AutoRoutingBackend::Metal,
            platform: AutoRoutingPlatform {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                hardware: required_env("J2K_AUTO_ROUTING_HARDWARE"),
                driver: required_env("J2K_AUTO_ROUTING_DRIVER"),
            },
            external_manifest_sha256: workloads.manifest_sha256,
            external_case_count: workloads.workloads.len(),
            cells,
        };
        write_auto_routing_evidence(&evidence_path, &evidence)
            .unwrap_or_else(|error| panic!("write Metal Auto-routing evidence: {error}"));
    }

    struct DecodeCase<'a> {
        id: &'a str,
        bytes: &'a [u8],
        fmt: PixelFormat,
        dimensions: (u32, u32),
        transfer_syntax: CompressedTransferSyntax,
        shared: Arc<[u8]>,
    }

    impl<'a> DecodeCase<'a> {
        fn new(workload: &'a AutoRoutingWorkload) -> Self {
            let fmt = pixel_format(workload.pixel_format);
            let info = j2k::J2kDecoder::inspect(&workload.bytes)
                .unwrap_or_else(|error| panic!("inspect decode workload {}: {error}", workload.id));
            let support = j2k::J2kDecoder::inspect_support(&workload.bytes)
                .unwrap_or_else(|error| panic!("inspect decode support {}: {error}", workload.id));
            Self {
                id: &workload.id,
                bytes: &workload.bytes,
                fmt,
                dimensions: info.dimensions,
                transfer_syntax: support.transfer_syntax,
                shared: Arc::from(workload.bytes.clone()),
            }
        }
    }

    type EncodeCase = AutoRoutingPnm;

    fn bench_decode_cell(
        criterion: &mut Criterion,
        case: &DecodeCase<'_>,
        operation: AutoRoutingOperation,
        session: &MetalBackendSession,
    ) -> AutoRoutingCell {
        let cpu =
            decode_once(case, operation, BackendRequest::Cpu, session).unwrap_or_else(|error| {
                panic!("CPU {} {}: {error}", case.id, operation_label(operation))
            });
        let hybrid =
            decode_once(case, operation, BackendRequest::Metal, session).unwrap_or_else(|error| {
                panic!(
                    "hybrid Metal {} {}: {error}",
                    case.id,
                    operation_label(operation)
                )
            });
        assert_output_parity(case.id, operation, &cpu, &hybrid);
        let auto =
            decode_once(case, operation, BackendRequest::Auto, session).unwrap_or_else(|error| {
                panic!("Auto {} {}: {error}", case.id, operation_label(operation))
            });
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
        assert_surface_route(&surface, backend, case, operation)?;
        surface
            .as_bytes()
            .map(std::borrow::Cow::into_owned)
            .map_err(|error| error.to_string())
    }

    fn decode_batch_once(
        case: &DecodeCase<'_>,
        backend: BackendRequest,
    ) -> Result<Vec<u8>, String> {
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
            assert_surface_route(surface, backend, case, AutoRoutingOperation::BatchDecode)?;
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
        case: &DecodeCase<'_>,
        operation: AutoRoutingOperation,
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
                if surface.backend_kind() == expected_auto_decode_backend(case, operation)
                    && (surface.backend_kind() != BackendKind::Metal
                        || surface.residency() == SurfaceResidency::MetalResidentDecode) =>
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

    fn expected_auto_decode_backend(
        case: &DecodeCase<'_>,
        operation: AutoRoutingOperation,
    ) -> BackendKind {
        if operation != AutoRoutingOperation::BatchDecode {
            return BackendKind::Cpu;
        }
        let pixels = u64::from(case.dimensions.0) * u64::from(case.dimensions.1);
        let promoted = match (case.fmt, case.transfer_syntax) {
            (PixelFormat::Gray8, CompressedTransferSyntax::Jpeg2000Lossy) => {
                pixels >= AUTO_GRAY8_MIN_PIXELS
            }
            (PixelFormat::Rgb8, CompressedTransferSyntax::Jpeg2000Lossy) => {
                pixels >= AUTO_RGB8_BATCH_MIN_PIXELS
            }
            (PixelFormat::Rgb8, CompressedTransferSyntax::Jpeg2000Lossless) => {
                pixels >= AUTO_RGB8_LARGE_MIN_PIXELS
            }
            _ => false,
        };
        if promoted {
            BackendKind::Metal
        } else {
            BackendKind::Cpu
        }
    }

    fn bench_encode_cell(
        criterion: &mut Criterion,
        case: &EncodeCase,
        operation: AutoRoutingOperation,
    ) -> AutoRoutingCell {
        let cpu = encode_cpu(case, operation).unwrap_or_else(|error| {
            panic!("CPU {} {}: {error}", case.id, operation_label(operation))
        });
        let mut probe_accelerator = MetalEncodeStageAccelerator::for_host_output_benchmark();
        let hybrid =
            encode_hybrid(case, operation, &mut probe_accelerator).unwrap_or_else(|error| {
                panic!(
                    "hybrid Metal {} {}: {error}",
                    case.id,
                    operation_label(operation)
                )
            });
        assert_output_parity(&case.id, operation, &cpu, &hybrid);
        let (auto, auto_dispatches) = encode_auto(case, operation).unwrap_or_else(|error| {
            panic!("Auto {} {}: {error}", case.id, operation_label(operation))
        });
        assert_output_parity(&case.id, operation, &cpu, &auto);
        let expected_dispatch = expected_auto_encode_dispatch(case, operation);
        assert_eq!(
            auto_dispatches > 0,
            expected_dispatch,
            "Auto dispatch decision for {} {}",
            case.id,
            operation_label(operation)
        );

        let group_id = format!(
            "auto-routing_{}_{id}",
            operation_label(operation),
            id = case.id
        );
        let mut accelerator = MetalEncodeStageAccelerator::for_host_output_benchmark();
        let mut group = criterion.benchmark_group(&group_id);
        group.bench_function("cpu", |bencher| {
            bencher.iter(|| {
                std::hint::black_box(
                    encode_cpu(case, operation).expect("measured CPU JPEG 2000 encode"),
                )
            });
        });
        group.bench_function("hybrid", |bencher| {
            bencher.iter(|| {
                std::hint::black_box(
                    encode_hybrid(case, operation, &mut accelerator)
                        .expect("measured hybrid Metal JPEG 2000 encode"),
                )
            });
        });
        group.finish();
        route_cell(&case.id, operation, &group_id, auto_routing_sha256(&cpu))
    }

    fn encode_cpu(case: &EncodeCase, operation: AutoRoutingOperation) -> Result<Vec<u8>, String> {
        match operation {
            AutoRoutingOperation::LosslessEncode => {
                let encoded = encode_j2k_lossless(
                    lossless_samples(case)?,
                    &lossless_options(EncodeBackendPreference::CpuOnly),
                )
                .map_err(|error| error.to_string())?;
                Ok(encoded.codestream)
            }
            AutoRoutingOperation::LossyEncode => {
                let encoded = encode_j2k_lossy(
                    lossy_samples(case)?,
                    &lossy_options(EncodeBackendPreference::CpuOnly),
                )
                .map_err(|error| error.to_string())?;
                Ok(encoded.codestream)
            }
            _ => Err("invalid encode operation".to_string()),
        }
    }

    fn encode_hybrid(
        case: &EncodeCase,
        operation: AutoRoutingOperation,
        accelerator: &mut MetalEncodeStageAccelerator,
    ) -> Result<Vec<u8>, String> {
        let (codestream, dispatches) = match operation {
            AutoRoutingOperation::LosslessEncode => {
                let encoded = encode_j2k_lossless_with_accelerator(
                    lossless_samples(case)?,
                    &lossless_options(EncodeBackendPreference::Auto),
                    BackendKind::Metal,
                    accelerator,
                )
                .map_err(|error| error.to_string())?;
                (encoded.codestream, encoded.dispatch_report.total())
            }
            AutoRoutingOperation::LossyEncode => {
                let encoded = encode_j2k_lossy_with_accelerator(
                    lossy_samples(case)?,
                    &lossy_options(EncodeBackendPreference::Auto),
                    BackendKind::Metal,
                    accelerator,
                )
                .map_err(|error| error.to_string())?;
                (encoded.codestream, encoded.dispatch_report.total())
            }
            _ => return Err("invalid encode operation".to_string()),
        };
        if dispatches == 0 {
            return Err("Metal hybrid encode did not dispatch any device stage".to_string());
        }
        Ok(codestream)
    }

    fn encode_auto(
        case: &EncodeCase,
        operation: AutoRoutingOperation,
    ) -> Result<(Vec<u8>, usize), String> {
        let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();
        let encoded = match operation {
            AutoRoutingOperation::LosslessEncode => encode_j2k_lossless_with_accelerator(
                lossless_samples(case)?,
                &lossless_options(EncodeBackendPreference::Auto),
                BackendKind::Metal,
                &mut accelerator,
            )
            .map(|encoded| (encoded.codestream, encoded.dispatch_report.total()))
            .map_err(|error| error.to_string()),
            AutoRoutingOperation::LossyEncode => encode_j2k_lossy_with_accelerator(
                lossy_samples(case)?,
                &lossy_options(EncodeBackendPreference::Auto),
                BackendKind::Metal,
                &mut accelerator,
            )
            .map(|encoded| (encoded.codestream, encoded.dispatch_report.total()))
            .map_err(|error| error.to_string()),
            _ => Err("invalid encode operation".to_string()),
        }?;
        Ok(encoded)
    }

    fn expected_auto_encode_dispatch(case: &EncodeCase, operation: AutoRoutingOperation) -> bool {
        if operation != AutoRoutingOperation::LossyEncode {
            return false;
        }
        let pixels = u64::from(case.width) * u64::from(case.height);
        match case.components {
            3 => pixels >= AUTO_RGB8_LARGE_MIN_PIXELS,
            _ => false,
        }
    }

    fn lossless_samples(case: &EncodeCase) -> Result<J2kLosslessSamples<'_>, String> {
        J2kLosslessSamples::new(
            &case.pixels,
            case.width,
            case.height,
            case.components,
            8,
            false,
        )
        .map_err(|error| error.to_string())
    }

    fn lossy_samples(case: &EncodeCase) -> Result<J2kLossySamples<'_>, String> {
        J2kLossySamples::new(
            &case.pixels,
            case.width,
            case.height,
            case.components,
            8,
            false,
        )
        .map_err(|error| error.to_string())
    }

    fn lossless_options(backend: EncodeBackendPreference) -> J2kLosslessEncodeOptions {
        J2kLosslessEncodeOptions::default()
            .with_backend(backend)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(3))
            .with_validation(J2kEncodeValidation::External)
    }

    fn lossy_options(backend: EncodeBackendPreference) -> J2kLossyEncodeOptions {
        let mut options = J2kLossyEncodeOptions::default()
            .with_backend(backend)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(3))
            .with_rate_target(Some(J2kRateTarget::BitsPerPixel(4.0)))
            .with_validation(J2kEncodeValidation::External);
        options.psnr_iteration_budget = 1;
        options
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

    fn assert_output_parity(
        case_id: &str,
        operation: AutoRoutingOperation,
        cpu: &[u8],
        hybrid: &[u8],
    ) {
        if cpu == hybrid {
            return;
        }
        let first_difference = cpu
            .iter()
            .zip(hybrid)
            .position(|(cpu, hybrid)| cpu != hybrid)
            .map(|index| (index, cpu[index], hybrid[index]));
        panic!(
            "Metal {} output differs for {case_id}: cpu_len={}, hybrid_len={}, first_difference={first_difference:?}",
            operation_label(operation),
            cpu.len(),
            hybrid.len(),
        );
    }

    const fn pixel_format(format: AutoRoutingPixelFormat) -> PixelFormat {
        match format {
            AutoRoutingPixelFormat::Gray8 => PixelFormat::Gray8,
            AutoRoutingPixelFormat::Rgb8 => PixelFormat::Rgb8,
        }
    }

    fn required_path(name: &str) -> PathBuf {
        PathBuf::from(required_env(name))
    }

    fn required_env(name: &str) -> String {
        std::env::var(name)
            .unwrap_or_else(|_| panic!("{name} must be set for Auto-routing benchmarks"))
    }
}
