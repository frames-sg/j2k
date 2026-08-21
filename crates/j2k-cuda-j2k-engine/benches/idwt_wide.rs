// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use j2k_cuda_j2k_engine::{CudaJ2kIdwtJob, CudaJ2kIdwtTarget, CudaJ2kRect, J2kCudaEngine};
use j2k_cuda_runtime::{CudaContext, CudaPooledDeviceBuffer};
use sha2::{Digest, Sha256};

const BATCH_SIZES: &[usize] = &[1, 16];
const SHAPES: &[BenchShape] = &[
    BenchShape {
        label: "narrow_512x512",
        width: 512,
        height: 512,
    },
    BenchShape {
        label: "wide_2592x1944",
        width: 2_592,
        height: 1_944,
    },
];
const TRANSFORMS: &[BenchTransform] = &[
    BenchTransform {
        label: "reversible53",
        mode: 0,
    },
    BenchTransform {
        label: "irreversible97",
        mode: 1,
    },
];

#[derive(Clone, Copy)]
struct BenchShape {
    label: &'static str,
    width: usize,
    height: usize,
}

#[derive(Clone, Copy)]
struct BenchTransform {
    label: &'static str,
    mode: u32,
}

fn sample_bytes(len: usize, salt: usize) -> Vec<u8> {
    let byte_len = len
        .checked_mul(std::mem::size_of::<f32>())
        .expect("P14 input byte count");
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_len)
        .expect("P14 input host allocation");
    for index in 0..len {
        let code = i16::try_from((index * (salt * 2 + 1) + salt * 11) % 251)
            .expect("P14 sample code fits i16");
        let sample = f32::from(code - 125) * 0.125;
        bytes.extend_from_slice(&sample.to_ne_bytes());
    }
    bytes
}

fn zeroed_bytes(len: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .expect("P14 readback host allocation");
    bytes.resize(len, 0);
    bytes
}

fn sha256_framed(chunks: impl IntoIterator<Item = impl AsRef<[u8]>>) -> String {
    let mut hash = Sha256::new();
    for chunk in chunks {
        let chunk = chunk.as_ref();
        hash.update(u64::try_from(chunk.len()).unwrap_or(u64::MAX).to_le_bytes());
        hash.update(chunk);
    }
    format!("{hash:x}", hash = hash.finalize())
}

fn output_bytes(shape: BenchShape) -> usize {
    shape
        .width
        .checked_mul(shape.height)
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>()))
        .expect("P14 output byte count")
}

fn make_job(shape: BenchShape, transform: BenchTransform) -> CudaJ2kIdwtJob {
    let width = u32::try_from(shape.width).expect("P14 width fits u32");
    let height = u32::try_from(shape.height).expect("P14 height fits u32");
    let band_width = width / 2;
    let band_height = height / 2;
    let band_rect = CudaJ2kRect {
        x0: 0,
        y0: 0,
        x1: band_width,
        y1: band_height,
    };
    CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 1,
            y0: 1,
            x1: width + 1,
            y1: height + 1,
        },
        ll_rect: band_rect,
        hl_rect: band_rect,
        lh_rect: band_rect,
        hh_rect: band_rect,
        irreversible97: transform.mode,
    }
}

fn route_label(shape: BenchShape) -> &'static str {
    if shape.width <= 512 && shape.height <= 512 {
        "whole_line_cooperative"
    } else {
        "generic"
    }
}

fn verified_output_sha256(
    outputs: &[CudaPooledDeviceBuffer],
    oracle_bytes: &[u8],
    shape: BenchShape,
    transform: BenchTransform,
) -> String {
    let mut output_frames = Vec::new();
    output_frames
        .try_reserve_exact(outputs.len())
        .expect("P14 output frame owners");
    for (batch_index, output) in outputs.iter().enumerate() {
        let mut actual = zeroed_bytes(output_bytes(shape));
        output
            .copy_to_host(&mut actual)
            .expect("P14 output readback");
        assert_eq!(
            actual, oracle_bytes,
            "P14 exact parity failed for {} {} batch item {batch_index}",
            transform.label, shape.label
        );
        output_frames.push(actual);
    }
    sha256_framed(output_frames.iter())
}

fn bench_cell(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    shape: BenchShape,
    transform: BenchTransform,
    batch_size: usize,
) {
    let context = CudaContext::system_default().expect("P14 CUDA context");
    let pool = context.buffer_pool();
    let engine = J2kCudaEngine::new(&context);
    let band_len = (shape.width / 2) * (shape.height / 2);
    let host_bands = [
        sample_bytes(band_len, 1),
        sample_bytes(band_len, 3),
        sample_bytes(band_len, 5),
        sample_bytes(band_len, 7),
    ];
    let input_sha256 = sha256_framed(host_bands.iter());
    let ll = context.upload(&host_bands[0]).expect("P14 upload LL");
    let hl = context.upload(&host_bands[1]).expect("P14 upload HL");
    let lh = context.upload(&host_bands[2]).expect("P14 upload LH");
    let hh = context.upload(&host_bands[3]).expect("P14 upload HH");
    let job = make_job(shape, transform);

    let oracle = engine
        .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
        .expect("P14 generic single oracle");
    let mut oracle_bytes = zeroed_bytes(output_bytes(shape));
    oracle
        .buffer()
        .expect("P14 oracle device buffer")
        .copy_to_host(&mut oracle_bytes)
        .expect("P14 oracle readback");

    let outputs = (0..batch_size)
        .map(|_| {
            pool.take(output_bytes(shape))
                .expect("P14 output allocation")
        })
        .collect::<Vec<_>>();
    let targets = outputs
        .iter()
        .map(|output| CudaJ2kIdwtTarget {
            ll: &ll,
            hl: &hl,
            lh: &lh,
            hh: &hh,
            output: output.as_device_buffer().expect("P14 output device buffer"),
            job,
        })
        .collect::<Vec<_>>();

    let probe_start = Instant::now();
    let probe = engine
        .j2k_inverse_dwt_batch_device_with_pool(&targets, &pool)
        .expect("P14 route probe");
    let stage_wall_time_ns = probe_start.elapsed().as_nanos();
    let expected_dispatches = 2;
    assert_eq!(probe.kernel_dispatches(), expected_dispatches);

    let output_sha256 = verified_output_sha256(&outputs, &oracle_bytes, shape, transform);
    println!(
        "j2k_cuda_p14_probe transform={} shape={} batch={} odd_origin=true \
         input_sha256={} output_sha256={} exact_parity=true route={} dispatch_count={} \
         stage_wall_time_ns={} output_allocation_bytes={}",
        transform.label,
        shape.label,
        batch_size,
        input_sha256,
        output_sha256,
        route_label(shape),
        probe.kernel_dispatches(),
        stage_wall_time_ns,
        output_bytes(shape) * batch_size,
    );

    let benchmark_id = BenchmarkId::new(
        format!("{}/{}", transform.label, shape.label),
        format!("batch_{batch_size}"),
    );
    group.throughput(Throughput::Elements(
        u64::try_from(shape.width * shape.height * batch_size).expect("P14 throughput fits u64"),
    ));
    group.bench_function(benchmark_id, |b| {
        b.iter(|| {
            let execution = engine
                .j2k_inverse_dwt_batch_device_with_pool(
                    std::hint::black_box(targets.as_slice()),
                    &pool,
                )
                .expect("P14 timed IDWT stage");
            assert_eq!(execution.kernel_dispatches(), expected_dispatches);
            std::hint::black_box(execution)
        });
    });
}

fn bench_idwt_wide(c: &mut Criterion) {
    println!(
        "j2k_cuda_p14_handoff priority_end_to_end_decode=j2k_cuda_htj2k_tile_batch_decode \
         shape=narrow_512x512 batch=16"
    );
    let mut group = c.benchmark_group("j2k_cuda_idwt_stage");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));
    for &transform in TRANSFORMS {
        for &shape in SHAPES {
            for &batch_size in BATCH_SIZES {
                bench_cell(&mut group, shape, transform, batch_size);
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_idwt_wide);
criterion_main!(benches);
