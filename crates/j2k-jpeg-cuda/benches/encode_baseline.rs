// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decision-grade profile of the promoted staged CUDA baseline JPEG encoder.

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use j2k_core::PixelFormat;
use j2k_cuda_runtime::{CudaContext, CudaContextDiagnostics};
use j2k_jpeg::{DecodeRequest, Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
use j2k_jpeg_cuda::{
    encode_jpeg_baseline_batch_from_cuda_buffers, CudaSession, JpegBaselineCudaEncodeTile,
};
use sha2::{Digest, Sha256};

const INPUT_HASH_DOMAIN: &[u8] = b"P18-CUDA-JPEG-ENCODE-INPUTS\0";
const OUTPUT_HASH_DOMAIN: &[u8] = b"P18-CUDA-JPEG-ENCODE-OUTPUTS\0";

#[derive(Clone, Copy)]
struct BenchCase {
    id: &'static str,
    dimension: u32,
    batch_size: usize,
    restart_interval: Option<u16>,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        id: "rgb8_422_512x512_batch8_q90_restart_none",
        dimension: 512,
        batch_size: 8,
        restart_interval: None,
    },
    BenchCase {
        id: "rgb8_422_512x512_batch1_q90_restart_none",
        dimension: 512,
        batch_size: 1,
        restart_interval: None,
    },
    BenchCase {
        id: "rgb8_422_64x64_batch1_q90_restart_none",
        dimension: 64,
        batch_size: 1,
        restart_interval: None,
    },
    BenchCase {
        id: "rgb8_422_512x512_batch8_q90_restart16",
        dimension: 512,
        batch_size: 8,
        restart_interval: Some(16),
    },
    BenchCase {
        id: "rgb8_422_512x512_batch8_q90_restart32",
        dimension: 512,
        batch_size: 8,
        restart_interval: Some(32),
    },
];

fn bench_cuda_staged_encode(criterion: &mut Criterion) {
    let context = match CudaContext::system_default() {
        Ok(context) => context,
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA JPEG encode is unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA JPEG encode benchmark: {error}");
            return;
        }
    };
    let mut group = criterion.benchmark_group("j2k_cuda_p18_staged_encode");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for case in CASES {
        let tile_bytes = tile_bytes(case.dimension);
        let input = patterned_rgb8_tiles(case.dimension, case.batch_size);
        let input_frames = input.chunks_exact(tile_bytes).collect::<Vec<_>>();
        let input_sha256 = framed_sha256(INPUT_HASH_DOMAIN, input_frames.iter().copied());
        let buffer = context.upload(&input).expect("upload P18 CUDA JPEG input");
        let tiles = (0..case.batch_size)
            .map(|tile_index| JpegBaselineCudaEncodeTile {
                buffer: &buffer,
                byte_offset: tile_index * tile_bytes,
                width: case.dimension,
                height: case.dimension,
                pitch_bytes: case.dimension as usize * 3,
                output_width: case.dimension,
                output_height: case.dimension,
                format: PixelFormat::Rgb8,
            })
            .collect::<Vec<_>>();
        let options = JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: case.restart_interval,
            backend: JpegBackend::Cuda,
        };
        let mut session = CudaSession::default();
        let before = context.diagnostics().expect("P18 diagnostics before probe");
        let first = encode_jpeg_baseline_batch_from_cuda_buffers(&tiles, options, &mut session)
            .expect("first P18 CUDA JPEG staged probe");
        let after = context.diagnostics().expect("P18 diagnostics after probe");
        let repeat = encode_jpeg_baseline_batch_from_cuda_buffers(&tiles, options, &mut session)
            .expect("repeat P18 CUDA JPEG staged probe");

        assert_eq!(
            first, repeat,
            "P18 staged codestreams must be deterministic"
        );
        validate_frames(&first, case.dimension, case.batch_size);
        validate_frames(&repeat, case.dimension, case.batch_size);
        let output_sha256 = framed_sha256(
            OUTPUT_HASH_DOMAIN,
            first.iter().map(|frame| frame.data.as_slice()),
        );
        let diagnostics = DiagnosticsDelta::new(before, after);
        assert_eq!(
            diagnostics.kernel_dispatches, 2,
            "promoted P18 route must preserve its two physical dispatches"
        );
        diagnostics.emit(case, &input_sha256, &output_sha256);

        group.throughput(Throughput::Elements(case.batch_size as u64));
        group.bench_function(case.id, |bencher| {
            bencher.iter(|| {
                encode_jpeg_baseline_batch_from_cuda_buffers(
                    std::hint::black_box(&tiles),
                    options,
                    &mut session,
                )
                .expect("P18 CUDA JPEG staged encode")
            });
        });
    }
    group.finish();
}

fn validate_frames(frames: &[j2k_jpeg::EncodedJpeg], dimension: u32, batch_size: usize) {
    assert_eq!(frames.len(), batch_size);
    for frame in frames {
        assert_eq!(frame.backend, JpegBackend::Cuda);
        assert!(frame.data.starts_with(&[0xff, 0xd8]), "missing JPEG SOI");
        assert!(frame.data.ends_with(&[0xff, 0xd9]), "missing JPEG EOI");

        let decoder = Decoder::new(&frame.data).expect("repository parser accepts P18 frame");
        let (pixels, outcome) = decoder
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("repository decoder accepts P18 frame");
        assert_eq!(
            (outcome.decoded.w, outcome.decoded.h),
            (dimension, dimension)
        );
        assert_eq!(pixels.len(), tile_bytes(dimension));

        let mut independent = jpeg_decoder::Decoder::new(std::io::Cursor::new(&frame.data));
        let pixels = independent
            .decode()
            .expect("independent jpeg-decoder accepts P18 frame");
        let info = independent.info().expect("independent decoder frame info");
        assert_eq!(
            (u32::from(info.width), u32::from(info.height)),
            (dimension, dimension)
        );
        assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::RGB24);
        assert_eq!(pixels.len(), tile_bytes(dimension));
    }
}

fn patterned_rgb8_tiles(dimension: u32, batch_size: usize) -> Vec<u8> {
    let tile_bytes = tile_bytes(dimension);
    let capacity = tile_bytes
        .checked_mul(batch_size)
        .expect("P18 input byte count fits usize");
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(capacity)
        .expect("allocate P18 benchmark inputs");
    for tile in 0..batch_size {
        let tile = u32::try_from(tile).expect("P18 batch index fits u32");
        for y in 0..dimension {
            for x in 0..dimension {
                pixels.push(((x * 29 + y * 3 + tile * 31 + 11) & 0xff) as u8);
                pixels.push(((x * 7 + y * 17 + tile * 19 + 40) & 0xff) as u8);
                pixels.push(((x * 13 + y * 5 + tile * 23 + 90) & 0xff) as u8);
            }
        }
    }
    pixels
}

fn tile_bytes(dimension: u32) -> usize {
    dimension as usize * dimension as usize * 3
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
    kernel_dispatches: u64,
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
            kernel_dispatches: delta(before.kernel_launches, after.kernel_launches),
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

    fn emit(&self, case: &BenchCase, input_sha256: &str, output_sha256: &str) {
        let mcus_per_row = case.dimension.div_ceil(16) as usize;
        let mcu_rows = case.dimension.div_ceil(8) as usize;
        let coefficient_scratch_bytes =
            mcus_per_row * mcu_rows * 4 * 64 * std::mem::size_of::<i32>() * case.batch_size;
        eprintln!(
            "p18_cuda_jpeg_encode_probe cell={} dimensions={}x{} batch={} quality=90 sampling=4:2:2 restart_interval={} route=staged coefficient_scratch_bytes={} input_sha256={} output_sha256={} exact_codestreams=true deterministic=true repository_decode=true independent_decode=true kernel_dispatches={} host_to_device_transfers={} host_to_device_bytes={} device_to_host_transfers={} device_to_host_bytes={} status_transfers={} status_bytes={} device_allocations={} device_allocation_bytes={} event_allocations={} event_reuses={} host_synchronizations={}",
            case.id,
            case.dimension,
            case.dimension,
            case.batch_size,
            case.restart_interval.map_or_else(|| "none".to_string(), |value| value.to_string()),
            coefficient_scratch_bytes,
            input_sha256,
            output_sha256,
            self.kernel_dispatches,
            self.host_to_device_transfers,
            self.host_to_device_bytes,
            self.device_to_host_transfers,
            self.device_to_host_bytes,
            self.status_transfers,
            self.status_bytes,
            self.device_allocations,
            self.device_allocation_bytes,
            self.event_allocations,
            self.event_reuses,
            self.host_synchronizations,
        );
    }
}

const fn delta(before: u64, after: u64) -> u64 {
    after.saturating_sub(before)
}

fn p18_criterion() -> Criterion {
    Criterion::default().confidence_level(0.95)
}

criterion_group! {
    name = benches;
    config = p18_criterion();
    targets = bench_cuda_staged_encode
}
criterion_main!(benches);
