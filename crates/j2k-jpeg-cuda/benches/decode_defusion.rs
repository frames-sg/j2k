// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-path profiler for the promoted P19 adaptive-checkpoint decode route.

use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use j2k_core::{
    BackendRequest, CodecError, DeviceSurface, Downscale, ImageDecodeDevice, PixelFormat, Rect,
};
use j2k_cuda_runtime::CudaContextDiagnostics;
use j2k_jpeg::{
    adapter::{
        build_fast420_packet, build_fast422_packet, build_fast444_packet, JpegEntropyCheckpointV1,
    },
    DecodeRequest, Decoder as CpuDecoder, JpegBackend, JpegEncodeOptions, JpegSamples,
    JpegSubsampling,
};
use j2k_jpeg_cuda::{Codec, CudaSession, Decoder as CudaDecoder};
use sha2::{Digest, Sha256};

const INPUT_HASH_DOMAIN: &[u8] = b"P19-CUDA-JPEG-DECODE-INPUTS\0";
const OUTPUT_HASH_DOMAIN: &[u8] = b"P19-CUDA-JPEG-DECODE-OUTPUTS\0";
const MAX_CPU_CHANNEL_DELTA: u8 = 2;
const PACKED_CHECKPOINT_THREADS_PER_BLOCK: u32 = 128;
const PACKED_CHECKPOINT_MIN_COUNT: u32 = 128;
const BENCHMARK_GROUP: &str = "j2k_cuda_p19_decode_adaptive_checkpoints";
const SERIAL_ROUTE_FIELD: &str = "route=serial_below_threshold";
const PACKED_ROUTE_FIELD: &str = "route=packed_checkpoints";

#[derive(Clone, Copy)]
struct CheckpointLaunch {
    route_field: &'static str,
    grid_x: u32,
    block_x: u32,
}

fn checkpoint_launch(checkpoint_count: usize) -> CheckpointLaunch {
    let checkpoint_count =
        u32::try_from(checkpoint_count).expect("P19 checkpoint count fits the CUDA ABI");
    let launch = if checkpoint_count < PACKED_CHECKPOINT_MIN_COUNT {
        CheckpointLaunch {
            route_field: SERIAL_ROUTE_FIELD,
            grid_x: checkpoint_count,
            block_x: 1,
        }
    } else {
        CheckpointLaunch {
            route_field: PACKED_ROUTE_FIELD,
            grid_x: checkpoint_count.div_ceil(PACKED_CHECKPOINT_THREADS_PER_BLOCK),
            block_x: PACKED_CHECKPOINT_THREADS_PER_BLOCK,
        }
    };
    assert_eq!(
        launch.block_x == PACKED_CHECKPOINT_THREADS_PER_BLOCK,
        checkpoint_count >= PACKED_CHECKPOINT_MIN_COUNT
    );
    assert!(launch.grid_x.saturating_mul(launch.block_x) >= checkpoint_count);
    launch
}

#[derive(Clone, Copy)]
struct BenchCase {
    id: &'static str,
    dimension: u32,
    batch_size: usize,
    sampling: JpegSubsampling,
    restart_interval: Option<u16>,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        id: "ybr420_512x512_batch16_restart_none",
        dimension: 512,
        batch_size: 16,
        sampling: JpegSubsampling::Ybr420,
        restart_interval: None,
    },
    BenchCase {
        id: "ybr420_512x512_batch1_restart_none",
        dimension: 512,
        batch_size: 1,
        sampling: JpegSubsampling::Ybr420,
        restart_interval: None,
    },
    BenchCase {
        id: "ybr420_512x512_batch16_restart16",
        dimension: 512,
        batch_size: 16,
        sampling: JpegSubsampling::Ybr420,
        restart_interval: Some(16),
    },
    BenchCase {
        id: "ybr420_512x512_batch1_restart16",
        dimension: 512,
        batch_size: 1,
        sampling: JpegSubsampling::Ybr420,
        restart_interval: Some(16),
    },
    BenchCase {
        id: "ybr422_512x512_batch16_restart_none",
        dimension: 512,
        batch_size: 16,
        sampling: JpegSubsampling::Ybr422,
        restart_interval: None,
    },
    BenchCase {
        id: "ybr422_512x512_batch1_restart_none",
        dimension: 512,
        batch_size: 1,
        sampling: JpegSubsampling::Ybr422,
        restart_interval: None,
    },
    BenchCase {
        id: "ybr444_512x512_batch16_restart_none",
        dimension: 512,
        batch_size: 16,
        sampling: JpegSubsampling::Ybr444,
        restart_interval: None,
    },
    BenchCase {
        id: "ybr444_512x512_batch1_restart_none",
        dimension: 512,
        batch_size: 1,
        sampling: JpegSubsampling::Ybr444,
        restart_interval: None,
    },
    BenchCase {
        id: "ybr420_64x64_batch1_restart_none",
        dimension: 64,
        batch_size: 1,
        sampling: JpegSubsampling::Ybr420,
        restart_interval: None,
    },
    BenchCase {
        id: "ybr420_1024x1024_batch1_restart_none",
        dimension: 1024,
        batch_size: 1,
        sampling: JpegSubsampling::Ybr420,
        restart_interval: None,
    },
];

fn bench_decode_adaptive_checkpoints(criterion: &mut Criterion) {
    let mut session = CudaSession::default();
    if let Err(error) = session.cuda_context_diagnostics() {
        assert!(
            std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_none(),
            "J2K_REQUIRE_CUDA_BENCH is set but P19 CUDA decode is unavailable: {error}"
        );
        eprintln!("skipping P19 CUDA JPEG adaptive-checkpoint profile: {error}");
        return;
    }

    run_correctness_only_seams(&mut session);
    let mut group = criterion.benchmark_group(BENCHMARK_GROUP);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for case in CASES {
        let input = generated_jpeg(
            case.dimension,
            case.dimension,
            case.sampling,
            case.restart_interval,
        );
        let packet_start = Instant::now();
        let packet = packet_summary(&input, case.sampling);
        let cold_packet_construction_us = packet_start.elapsed().as_micros();
        let inputs = bench_vec_filled(case.batch_size, input.as_slice());
        let input_sha256 = framed_sha256(INPUT_HASH_DOMAIN, inputs.iter().copied());
        let cpu = cpu_decode(&input);

        let first = profile_probe(case, &input, &cpu, &mut session);
        let repeat = profile_probe(case, &input, &cpu, &mut session);
        assert_eq!(
            first.output_sha256, repeat.output_sha256,
            "P19 production output must be deterministic"
        );
        assert_eq!(
            first.outputs, repeat.outputs,
            "P19 repeated production output bytes must be exact"
        );
        emit_probe(
            case,
            &input_sha256,
            &packet,
            cold_packet_construction_us,
            &first,
            &repeat,
        );

        group.throughput(Throughput::Elements(case.batch_size as u64));
        group.bench_function(case.id, |bencher| {
            bencher.iter(|| {
                Codec::decode_tiles_to_device_with_session(
                    std::hint::black_box(&inputs),
                    PixelFormat::Rgb8,
                    BackendRequest::Cuda,
                    &mut session,
                )
                .expect("P19 warm cached-packet CUDA decode")
            });
        });
    }
    group.finish();
}

struct PacketSummary {
    checkpoint_count: usize,
    checkpoint_mcu_range: (u32, u32),
    checkpoint_entropy_range: (u32, u32),
    total_mcus: u32,
    blocks_per_mcu: u32,
}

fn packet_summary(input: &[u8], sampling: JpegSubsampling) -> PacketSummary {
    macro_rules! summarize {
        ($packet:expr, $blocks:expr) => {{
            let packet = $packet.expect("P19 fast packet");
            summarize_packet(
                packet.mcus_per_row,
                packet.mcu_rows,
                &packet.entropy_checkpoints,
                $blocks,
            )
        }};
    }
    match sampling {
        JpegSubsampling::Ybr420 => summarize!(build_fast420_packet(input), 6),
        JpegSubsampling::Ybr422 => summarize!(build_fast422_packet(input), 4),
        JpegSubsampling::Ybr444 => summarize!(build_fast444_packet(input), 3),
        JpegSubsampling::Gray => unreachable!("P19 matrix is color-only"),
    }
}

fn summarize_packet(
    mcus_per_row: u32,
    mcu_rows: u32,
    checkpoints: &[JpegEntropyCheckpointV1],
    blocks_per_mcu: u32,
) -> PacketSummary {
    let first = checkpoints.first().expect("P19 initial checkpoint");
    let last = checkpoints.last().expect("P19 final checkpoint");
    PacketSummary {
        checkpoint_count: checkpoints.len(),
        checkpoint_mcu_range: (first.mcu_index, last.mcu_index),
        checkpoint_entropy_range: (first.entropy_pos, last.entropy_pos),
        total_mcus: mcus_per_row.checked_mul(mcu_rows).expect("P19 MCU count"),
        blocks_per_mcu,
    }
}

struct Probe {
    outputs: Vec<Vec<u8>>,
    output_sha256: String,
    resource_upload_us: u128,
    fused_decode_kernel_us: u128,
    conversion_us: u128,
    status_readback_us: u128,
    product_wall_us: u128,
    component_workspace_bytes: usize,
    kernel_dispatches: usize,
    max_cpu_channel_delta: u8,
    diagnostics: DiagnosticsDelta,
}

fn profile_probe(case: &BenchCase, input: &[u8], cpu: &[u8], session: &mut CudaSession) -> Probe {
    let before = session
        .cuda_context_diagnostics()
        .expect("P19 diagnostics before probe");
    let wall_start = Instant::now();
    let mut profiled = bench_vec_with_capacity(case.batch_size);
    for _ in 0..case.batch_size {
        profiled.push(
            Codec::profile_tile_rgb8_with_session(input, session).expect("P19 profiled decode"),
        );
    }
    let product_wall_us = wall_start.elapsed().as_micros();
    let after = session
        .cuda_context_diagnostics()
        .expect("P19 diagnostics after probe");
    let mut outputs = bench_vec_with_capacity(case.batch_size);
    let mut resource_upload_us = 0u128;
    let mut fused_decode_kernel_us = 0u128;
    let mut conversion_us = 0u128;
    let mut status_readback_us = 0u128;
    let mut component_workspace_bytes = 0usize;
    let mut kernel_dispatches = 0usize;
    let mut max_cpu_channel_delta = 0u8;
    for result in profiled {
        let (surface, timings) = result.into_parts();
        let stats = surface.cuda_surface().expect("P19 CUDA surface").stats();
        assert!(stats.used_owned_cuda_decode());
        let mut output = bench_vec_filled(surface.byte_len(), 0u8);
        surface
            .download_into(&mut output, surface.pitch_bytes())
            .expect("P19 output download");
        max_cpu_channel_delta = max_cpu_channel_delta.max(max_channel_delta(&output, cpu));
        resource_upload_us = resource_upload_us.saturating_add(timings.resource_upload_us());
        fused_decode_kernel_us =
            fused_decode_kernel_us.saturating_add(timings.fused_decode_kernel_us());
        conversion_us = conversion_us.saturating_add(timings.conversion_us());
        status_readback_us = status_readback_us.saturating_add(timings.status_readback_us());
        component_workspace_bytes =
            component_workspace_bytes.saturating_add(timings.component_workspace_bytes());
        kernel_dispatches = kernel_dispatches.saturating_add(stats.kernel_dispatches());
        outputs.push(output);
    }
    assert!(
        max_cpu_channel_delta <= MAX_CPU_CHANNEL_DELTA,
        "P19 CPU conformance delta {max_cpu_channel_delta}"
    );
    let output_sha256 = framed_sha256(OUTPUT_HASH_DOMAIN, outputs.iter().map(Vec::as_slice));
    Probe {
        outputs,
        output_sha256,
        resource_upload_us,
        fused_decode_kernel_us,
        conversion_us,
        status_readback_us,
        product_wall_us,
        component_workspace_bytes,
        kernel_dispatches,
        max_cpu_channel_delta,
        diagnostics: DiagnosticsDelta::new(before, after),
    }
}

fn emit_probe(
    case: &BenchCase,
    input_sha256: &str,
    packet: &PacketSummary,
    cold_packet_construction_us: u128,
    first: &Probe,
    repeat: &Probe,
) {
    let launch = checkpoint_launch(packet.checkpoint_count);
    eprintln!(
        "p19_cuda_jpeg_decode_probe cell={} dimensions={}x{} sampling={:?} restart_interval={} batch={} {} warm_cached_packet_product=true probe_repeat=2 cold_packet_construction_us={} input_sha256={} output_sha256={} exact_production_output=true deterministic=true cpu_conformance=true max_cpu_channel_delta={}/{} checkpoint_count={} checkpoint_mcu_range={}-{} checkpoint_entropy_range={}-{} total_mcus={} blocks_per_mcu={} decode_grid={}x1x1 decode_block={}x1x1 coefficient_scratch_bytes=0 component_workspace_bytes={}/{} resource_upload_us={}/{} fused_decode_kernel_us={}/{} conversion_us={}/{} status_readback_us={}/{} product_wall_us={}/{} kernel_dispatches={}/{} host_to_device_transfers={}/{} host_to_device_bytes={}/{} device_to_host_transfers={}/{} device_to_host_bytes={}/{} status_transfers={}/{} status_bytes={}/{} device_allocations={}/{} device_allocation_bytes={}/{} event_allocations={}/{} event_reuses={}/{} host_synchronizations={}/{}",
        case.id, case.dimension, case.dimension, case.sampling,
        case.restart_interval.map_or_else(|| "none".to_string(), |value| value.to_string()),
        case.batch_size, launch.route_field, cold_packet_construction_us, input_sha256,
        first.output_sha256,
        first.max_cpu_channel_delta, repeat.max_cpu_channel_delta,
        packet.checkpoint_count, packet.checkpoint_mcu_range.0, packet.checkpoint_mcu_range.1,
        packet.checkpoint_entropy_range.0, packet.checkpoint_entropy_range.1,
        packet.total_mcus, packet.blocks_per_mcu, launch.grid_x, launch.block_x,
        first.component_workspace_bytes, repeat.component_workspace_bytes,
        first.resource_upload_us, repeat.resource_upload_us,
        first.fused_decode_kernel_us, repeat.fused_decode_kernel_us,
        first.conversion_us, repeat.conversion_us,
        first.status_readback_us, repeat.status_readback_us,
        first.product_wall_us, repeat.product_wall_us,
        first.kernel_dispatches, repeat.kernel_dispatches,
        first.diagnostics.host_to_device_transfers, repeat.diagnostics.host_to_device_transfers,
        first.diagnostics.host_to_device_bytes, repeat.diagnostics.host_to_device_bytes,
        first.diagnostics.device_to_host_transfers, repeat.diagnostics.device_to_host_transfers,
        first.diagnostics.device_to_host_bytes, repeat.diagnostics.device_to_host_bytes,
        first.diagnostics.status_transfers, repeat.diagnostics.status_transfers,
        first.diagnostics.status_bytes, repeat.diagnostics.status_bytes,
        first.diagnostics.device_allocations, repeat.diagnostics.device_allocations,
        first.diagnostics.device_allocation_bytes, repeat.diagnostics.device_allocation_bytes,
        first.diagnostics.event_allocations, repeat.diagnostics.event_allocations,
        first.diagnostics.event_reuses, repeat.diagnostics.event_reuses,
        first.diagnostics.host_synchronizations, repeat.diagnostics.host_synchronizations,
    );
}

fn run_correctness_only_seams(session: &mut CudaSession) {
    for (label, sampling, dimensions) in [
        ("odd_420_513x517", JpegSubsampling::Ybr420, (513, 517)),
        ("odd_422_513x257", JpegSubsampling::Ybr422, (513, 257)),
    ] {
        let input = generated_jpeg(dimensions.0, dimensions.1, sampling, None);
        let cpu = cpu_decode(&input);
        let first = Codec::profile_tile_rgb8_with_session(&input, session).expect("P19 odd probe");
        let second =
            Codec::profile_tile_rgb8_with_session(&input, session).expect("P19 odd repeat");
        let actual = download_profile(&first);
        let repeated = download_profile(&second);
        assert_eq!(actual, repeated, "{label} exact repeat");
        assert!(
            max_channel_delta(&actual, &cpu) <= MAX_CPU_CHANNEL_DELTA,
            "{label} CPU conformance"
        );
        eprintln!("p19_cuda_jpeg_correctness cell={label} exact_production_output=true deterministic=true cpu_conformance=true");
    }
    restart32_seam(session);
    caller_owned_padded_output(session);
    routing_rejection_and_auto_fallback();
}

fn restart32_seam(session: &mut CudaSession) {
    let restart32 = generated_jpeg(512, 512, JpegSubsampling::Ybr420, Some(32));
    let restart32_cpu = cpu_decode(&restart32);
    let restart32_actual = Codec::profile_tile_rgb8_with_session(&restart32, session)
        .expect("P19 restart32 adaptive decode");
    let restart32_repeat = Codec::profile_tile_rgb8_with_session(&restart32, session)
        .expect("P19 restart32 adaptive repeat");
    let restart32_actual = download_profile(&restart32_actual);
    assert_eq!(restart32_actual, download_profile(&restart32_repeat));
    assert!(max_channel_delta(&restart32_actual, &restart32_cpu) <= MAX_CPU_CHANNEL_DELTA);

    eprintln!("p19_cuda_jpeg_correctness restart32_420=true exact_production_output=true deterministic=true cpu_conformance=true");
}

fn caller_owned_padded_output(session: &mut CudaSession) {
    let (width, height) = (513u32, 517u32);
    let input = generated_jpeg(width, height, JpegSubsampling::Ybr420, None);
    let cpu = cpu_decode(&input);
    let row_bytes = width as usize * 3;
    let pitch = row_bytes + 19;
    let buffer = session
        .take_owned_cuda_output_buffer(pitch * height as usize)
        .expect("P19 padded output");
    let stats =
        Codec::decode_tile_rgb8_into_cuda_buffer_with_session(&input, &buffer, pitch, session)
            .expect("P19 padded decode");
    assert!(stats.used_owned_cuda_decode());
    let mut downloaded = bench_vec_filled(buffer.byte_len(), 0u8);
    buffer
        .copy_to_host(&mut downloaded)
        .expect("P19 padded download");
    let mut tight = bench_vec_with_capacity(row_bytes * height as usize);
    for row in downloaded.chunks(pitch).take(height as usize) {
        tight.extend_from_slice(&row[..row_bytes]);
    }
    assert!(max_channel_delta(&tight, &cpu) <= MAX_CPU_CHANNEL_DELTA);
    assert!(downloaded
        .chunks(pitch)
        .take(height.saturating_sub(1) as usize)
        .all(|row| row[row_bytes..].iter().all(|byte| *byte == 0)));
    eprintln!("p19_cuda_jpeg_correctness caller_owned_padded_output=true dimensions=513x517 padding_bytes=19 cpu_conformance=true");
}

fn routing_rejection_and_auto_fallback() {
    let input = generated_jpeg(64, 64, JpegSubsampling::Ybr420, None);
    let roi = Rect {
        x: 3,
        y: 5,
        w: 41,
        h: 37,
    };
    let mut decoder = CudaDecoder::new(&input).expect("P19 routing decoder");
    assert!(decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Cuda)
        .expect_err("strict region rejection")
        .is_unsupported());
    assert!(decoder
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Cuda)
        .expect_err("strict scaled rejection")
        .is_unsupported());
    assert!(decoder
        .decode_region_scaled_to_device(
            PixelFormat::Rgb8,
            roi,
            Downscale::Half,
            BackendRequest::Cuda
        )
        .expect_err("strict region-scaled rejection")
        .is_unsupported());
    let auto = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("P19 Auto fallback");
    assert_eq!(auto.backend_kind(), j2k_core::BackendKind::Cpu);
    eprintln!("p19_cuda_jpeg_correctness strict_region_rejected=true strict_scaled_rejected=true strict_region_scaled_rejected=true auto_fallback=cpu");
}

fn download_profile(profile: &j2k_jpeg_cuda::CudaJpegDecodeProfile) -> Vec<u8> {
    let surface = profile.surface();
    let mut output = bench_vec_filled(surface.byte_len(), 0u8);
    surface
        .download_into(&mut output, surface.pitch_bytes())
        .expect("P19 profile download");
    output
}

fn bench_vec_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("allocate bounded P19 benchmark vector");
    values
}

fn bench_vec_filled<T: Clone>(len: usize, value: T) -> Vec<T> {
    let mut values = bench_vec_with_capacity(len);
    values.resize(len, value);
    values
}

fn generated_jpeg(
    width: u32,
    height: u32,
    sampling: JpegSubsampling,
    restart_interval: Option<u16>,
) -> Vec<u8> {
    let rgb = j2k_test_support::gpu_bench_rgb8(width, height);
    j2k_jpeg::encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: sampling,
            restart_interval,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("P19 generated JPEG")
    .data
}

fn cpu_decode(input: &[u8]) -> Vec<u8> {
    CpuDecoder::new(input)
        .expect("P19 CPU decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("P19 CPU decode")
        .0
}

fn max_channel_delta(actual: &[u8], expected: &[u8]) -> u8 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .max()
        .unwrap_or(0)
}

fn framed_sha256<'a>(domain: &[u8], frames: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for frame in frames {
        hasher.update((frame.len() as u64).to_le_bytes());
        hasher.update(frame);
    }
    format!("{:x}", hasher.finalize())
}

struct DiagnosticsDelta {
    host_to_device_transfers: u64,
    host_to_device_bytes: u64,
    device_to_host_transfers: u64,
    device_to_host_bytes: u64,
    status_transfers: u64,
    status_bytes: u64,
    device_allocations: u64,
    device_allocation_bytes: u64,
    event_allocations: u64,
    event_reuses: u64,
    host_synchronizations: u64,
}

impl DiagnosticsDelta {
    const fn new(before: CudaContextDiagnostics, after: CudaContextDiagnostics) -> Self {
        Self {
            host_to_device_transfers: delta(
                before.host_to_device_operations,
                after.host_to_device_operations,
            ),
            host_to_device_bytes: delta(before.host_to_device_bytes, after.host_to_device_bytes),
            device_to_host_transfers: delta(
                before.device_to_host_operations,
                after.device_to_host_operations,
            ),
            device_to_host_bytes: delta(before.device_to_host_bytes, after.device_to_host_bytes),
            status_transfers: delta(
                before.status_device_to_host_operations,
                after.status_device_to_host_operations,
            ),
            status_bytes: delta(
                before.status_device_to_host_bytes,
                after.status_device_to_host_bytes,
            ),
            device_allocations: delta(
                before.device_allocation_operations,
                after.device_allocation_operations,
            ),
            device_allocation_bytes: delta(
                before.device_allocation_bytes,
                after.device_allocation_bytes,
            ),
            event_allocations: delta(
                before.event_driver_allocations,
                after.event_driver_allocations,
            ),
            event_reuses: delta(before.event_reuses, after.event_reuses),
            host_synchronizations: delta(
                before.event_host_synchronizations,
                after.event_host_synchronizations,
            )
            .saturating_add(delta(
                before.context_host_synchronizations,
                after.context_host_synchronizations,
            )),
        }
    }
}

const fn delta(before: u64, after: u64) -> u64 {
    after.saturating_sub(before)
}

fn p19_criterion() -> Criterion {
    Criterion::default().confidence_level(0.95)
}

criterion_group! { name = benches; config = p19_criterion(); targets = bench_decode_adaptive_checkpoints }
criterion_main!(benches);
