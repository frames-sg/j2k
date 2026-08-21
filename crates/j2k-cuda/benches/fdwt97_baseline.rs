// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use j2k::{
    encode_j2k_lossy_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeStageAccelerator, J2kEncodeValidation, J2kForwardDwt97Job, J2kForwardDwt97Output,
    J2kLossyEncodeOptions, J2kLossySamples, J2kRateTarget,
};
use j2k_core::BackendKind;
use j2k_cuda::CudaEncodeStageAccelerator;
use j2k_native::{DecodeSettings, Image};
use sha2::{Digest, Sha256};

const BATCH_SIZES: &[usize] = &[1, 16];
const LEVELS: u8 = 3;
const SAMPLE_SIZE: usize = 10;
const WARM_UP: Duration = Duration::from_secs(1);
const MEASUREMENT: Duration = Duration::from_secs(3);
const GENERIC_LABEL: &str = "generic_baseline";
const PRODUCT_BATCH: usize = 16;
const PRODUCT_DIMENSION: u32 = 512;

#[derive(Clone, Copy)]
struct StageWorkload {
    id: &'static str,
    width: u32,
    height: u32,
}

const STAGE_WORKLOADS: &[StageWorkload] = &[
    StageWorkload {
        id: "small_512x512",
        width: 512,
        height: 512,
    },
    StageWorkload {
        id: "representative_1024x1024",
        width: 1_024,
        height: 1_024,
    },
    StageWorkload {
        id: "large_2592x1944",
        width: 2_592,
        height: 1_944,
    },
];

#[derive(Clone, Copy)]
struct StaticLoadTraffic {
    generic_loads: u128,
}

fn try_vec_with_capacity<T>(capacity: usize, label: &str) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .unwrap_or_else(|error| panic!("reserve {label}: {error}"));
    values
}

fn generate_plane(width: u32, height: u32) -> Vec<f32> {
    let mut samples = try_vec_with_capacity(width as usize * height as usize, "stage samples");
    for y in 0..height {
        for x in 0..width {
            let raw = (x * 29 + y * 43 + x.wrapping_mul(y) / 17 + (x ^ y)) & 0xff;
            samples
                .push(f32::from(u8::try_from(raw).expect("masked stage sample fits u8")) - 128.0);
        }
    }
    samples
}

fn generate_rgb8(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = try_vec_with_capacity(width as usize * height as usize * 3, "RGB pixels");
    for y in 0..height {
        for x in 0..width {
            for value in [
                x * 17 + y * 31 + x.wrapping_mul(y) / 11,
                x * 7 + y * 47 + (x ^ y),
                x * 41 + y * 5 + x.wrapping_mul(y) / 23,
            ] {
                pixels.push(u8::try_from(value & 0xff).expect("masked RGB sample fits u8"));
            }
        }
    }
    pixels
}

fn update_len(hash: &mut Sha256, len: usize) {
    hash.update(
        u64::try_from(len)
            .expect("benchmark length fits u64")
            .to_le_bytes(),
    );
}

fn input_sha256(samples: &[f32]) -> String {
    let mut hash = Sha256::new();
    update_len(&mut hash, samples.len());
    for sample in samples {
        hash.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn pixel_input_sha256(pixels: &[u8]) -> String {
    let mut hash = Sha256::new();
    update_len(&mut hash, pixels.len());
    hash.update(pixels);
    format!("{:x}", hash.finalize())
}

fn update_coefficients(hash: &mut Sha256, coefficients: &[f32]) {
    update_len(hash, coefficients.len());
    for coefficient in coefficients {
        hash.update(coefficient.to_bits().to_le_bytes());
    }
}

fn stage_output_sha256(outputs: &[J2kForwardDwt97Output]) -> String {
    let mut hash = Sha256::new();
    update_len(&mut hash, outputs.len());
    for output in outputs {
        hash.update(output.ll_width.to_le_bytes());
        hash.update(output.ll_height.to_le_bytes());
        update_coefficients(&mut hash, &output.ll);
        update_len(&mut hash, output.levels.len());
        for level in &output.levels {
            for dimension in [
                level.width,
                level.height,
                level.low_width,
                level.low_height,
                level.high_width,
                level.high_height,
            ] {
                hash.update(dimension.to_le_bytes());
            }
            update_coefficients(&mut hash, &level.hl);
            update_coefficients(&mut hash, &level.lh);
            update_coefficients(&mut hash, &level.hh);
        }
    }
    format!("{:x}", hash.finalize())
}

fn codestream_batch_sha256(codestreams: &[Vec<u8>]) -> String {
    let mut hash = Sha256::new();
    update_len(&mut hash, codestreams.len());
    for codestream in codestreams {
        update_len(&mut hash, codestream.len());
        hash.update(codestream);
    }
    format!("{:x}", hash.finalize())
}

fn high1_loads() -> u128 {
    3
}

fn low1_loads(extent: u32, index: u32) -> u128 {
    let even = index * 2;
    high1_loads() + u128::from(even + 1 < extent) * high1_loads() + 1
}

fn high2_loads(extent: u32, index: u32) -> u128 {
    let odd = index * 2 + 1;
    high1_loads()
        + low1_loads(extent, index)
        + if odd + 1 < extent {
            low1_loads(extent, index + 1)
        } else {
            let last_even = if extent.is_multiple_of(2) {
                extent - 2
            } else {
                extent - 1
            };
            low1_loads(extent, last_even / 2)
        }
}

fn low2_loads(extent: u32, index: u32) -> u128 {
    let even = index * 2;
    high2_loads(extent, index.saturating_sub(1))
        + u128::from(even + 1 < extent) * high2_loads(extent, index)
        + low1_loads(extent, index)
}

fn generic_line_loads(extent: u32) -> u128 {
    let low_extent = extent.div_ceil(2);
    let high_extent = extent / 2;
    (0..low_extent)
        .map(|index| low2_loads(extent, index))
        .sum::<u128>()
        + (0..high_extent)
            .map(|index| high2_loads(extent, index))
            .sum::<u128>()
}

fn static_load_traffic(width: u32, height: u32, levels: u8, batch: usize) -> StaticLoadTraffic {
    let mut current_width = width;
    let mut current_height = height;
    let mut generic_loads = 0_u128;
    for _ in 0..levels {
        generic_loads += generic_line_loads(current_height) * u128::from(current_width);
        generic_loads += generic_line_loads(current_width) * u128::from(current_height);
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
    }
    let batch = u128::try_from(batch).expect("batch fits u128");
    StaticLoadTraffic {
        generic_loads: generic_loads * batch,
    }
}

fn coefficient_count(output: &J2kForwardDwt97Output) -> usize {
    output.ll.len()
        + output
            .levels
            .iter()
            .map(|level| level.hl.len() + level.lh.len() + level.hh.len())
            .sum::<usize>()
}

fn stage_probe(
    samples: &[f32],
    workload: StageWorkload,
    batch: usize,
) -> (Vec<J2kForwardDwt97Output>, usize) {
    let mut accelerator = CudaEncodeStageAccelerator::default();
    let before = accelerator.dispatch_report();
    let mut outputs = try_vec_with_capacity(batch, "stage outputs");
    for _ in 0..batch {
        let output = accelerator
            .encode_forward_dwt97(J2kForwardDwt97Job {
                samples,
                width: workload.width,
                height: workload.height,
                num_levels: LEVELS,
            })
            .expect("CUDA FDWT97 stage succeeds")
            .expect("CUDA FDWT97 stage dispatches");
        outputs.push(output);
    }
    let dispatches = accelerator
        .dispatch_report()
        .saturating_delta(before)
        .forward_dwt97;
    assert_eq!(dispatches, usize::from(LEVELS) * 2 * batch);
    (outputs, dispatches)
}

fn timed_stage(
    accelerator: &mut CudaEncodeStageAccelerator,
    samples: &[f32],
    workload: StageWorkload,
    batch: usize,
) -> usize {
    let mut coefficients = 0_usize;
    for _ in 0..batch {
        let output = accelerator
            .encode_forward_dwt97(J2kForwardDwt97Job {
                samples: std::hint::black_box(samples),
                width: workload.width,
                height: workload.height,
                num_levels: LEVELS,
            })
            .expect("timed CUDA FDWT97 stage succeeds")
            .expect("timed CUDA FDWT97 stage dispatches");
        coefficients = coefficients.saturating_add(coefficient_count(&output));
        std::hint::black_box(output);
    }
    coefficients
}

fn emit_stage_probe(workload: StageWorkload, batch: usize, samples: &[f32]) {
    let (first, dispatches) = stage_probe(samples, workload, batch);
    let (second, second_dispatches) = stage_probe(samples, workload, batch);
    let output_sha256 = stage_output_sha256(&first);
    assert_eq!(output_sha256, stage_output_sha256(&second));
    assert_eq!(dispatches, second_dispatches);

    let traffic = static_load_traffic(workload.width, workload.height, LEVELS, batch);
    let generic_bytes = traffic.generic_loads * 4;
    println!(
        "j2k_cuda_fdwt97_probe workload={} route=generic_baseline width={} height={} levels={} batch={} \
         input_sha256={} output_sha256={} exact_parity=true dispatch_count={} \
         static_global_load_bytes={}",
        workload.id,
        workload.width,
        workload.height,
        LEVELS,
        batch,
        input_sha256(samples),
        output_sha256,
        dispatches,
        generic_bytes,
    );
}

fn product_options() -> J2kLossyEncodeOptions {
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(LEVELS))
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(4.0)))
        .with_validation(J2kEncodeValidation::External);
    options.psnr_iteration_budget = 1;
    options
}

fn encode_product_batch(
    accelerator: &mut CudaEncodeStageAccelerator,
    pixels: &[u8],
    options: &J2kLossyEncodeOptions,
) -> (Vec<Vec<u8>>, usize) {
    let before = accelerator.dispatch_report();
    let mut codestreams = try_vec_with_capacity(PRODUCT_BATCH, "product codestreams");
    for _ in 0..PRODUCT_BATCH {
        let samples =
            J2kLossySamples::new(pixels, PRODUCT_DIMENSION, PRODUCT_DIMENSION, 3, 8, false)
                .expect("valid P15 RGB8 product samples");
        let encoded =
            encode_j2k_lossy_with_accelerator(samples, options, BackendKind::Cuda, accelerator)
                .expect("P15 CUDA product encode");
        assert_eq!(encoded.backend, BackendKind::Cuda);
        codestreams.push(encoded.codestream);
    }
    let dispatches = accelerator
        .dispatch_report()
        .saturating_delta(before)
        .forward_dwt97;
    assert!(dispatches > 0, "P15 product must dispatch CUDA FDWT97");
    (codestreams, dispatches)
}

fn validate_product_codestreams(codestreams: &[Vec<u8>]) {
    for codestream in codestreams {
        let decoded = Image::new(codestream, &DecodeSettings::strict())
            .expect("P15 product codestream parses")
            .decode_native()
            .expect("P15 product codestream decodes");
        assert_eq!(decoded.width, PRODUCT_DIMENSION);
        assert_eq!(decoded.height, PRODUCT_DIMENSION);
        assert_eq!(decoded.num_components, 3);
    }
}

fn emit_product_probe(pixels: &[u8], options: &J2kLossyEncodeOptions) {
    let (first, dispatches) =
        encode_product_batch(&mut CudaEncodeStageAccelerator::default(), pixels, options);
    let (second, second_dispatches) =
        encode_product_batch(&mut CudaEncodeStageAccelerator::default(), pixels, options);
    validate_product_codestreams(&first);
    validate_product_codestreams(&second);
    let output_sha256 = codestream_batch_sha256(&first);
    assert_eq!(output_sha256, codestream_batch_sha256(&second));
    assert_eq!(dispatches, second_dispatches);
    println!(
        "j2k_cuda_p15_product_probe workload=product_htj2k_rgb_512x512_batch16 route=generic_baseline \
         width=512 height=512 components=3 levels={} batch=16 input_sha256={} \
         output_sha256={} exact_parity=true dispatch_count={}",
        LEVELS,
        pixel_input_sha256(pixels),
        output_sha256,
        dispatches,
    );
}

fn bench_fdwt97_stage(criterion: &mut Criterion) {
    println!(
        "j2k_cuda_fdwt97_baseline route=generic_baseline retained_after=P15_rejection label={GENERIC_LABEL}"
    );
    let mut group = criterion.benchmark_group("j2k_cuda_fdwt97_stage");
    for &workload in STAGE_WORKLOADS {
        let samples = generate_plane(workload.width, workload.height);
        for &batch in BATCH_SIZES {
            emit_stage_probe(workload, batch, &samples);
            group.throughput(Throughput::Elements(
                u64::from(workload.width) * u64::from(workload.height) * batch as u64,
            ));
            let mut accelerator = CudaEncodeStageAccelerator::default();
            group.bench_with_input(
                BenchmarkId::new(workload.id, format!("batch_{batch}")),
                &batch,
                |bencher, &batch| {
                    bencher.iter(|| {
                        std::hint::black_box(timed_stage(
                            &mut accelerator,
                            &samples,
                            workload,
                            batch,
                        ))
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_product_encode(criterion: &mut Criterion) {
    let pixels = generate_rgb8(PRODUCT_DIMENSION, PRODUCT_DIMENSION);
    let options = product_options();
    emit_product_probe(&pixels, &options);

    let mut group = criterion.benchmark_group("j2k_cuda_p15_product_encode");
    group.throughput(Throughput::Elements(
        u64::from(PRODUCT_DIMENSION) * u64::from(PRODUCT_DIMENSION) * PRODUCT_BATCH as u64,
    ));
    let mut accelerator = CudaEncodeStageAccelerator::default();
    group.bench_function("product_htj2k_rgb_512x512_batch16", |bencher| {
        bencher.iter(|| {
            let (codestreams, dispatches) =
                encode_product_batch(&mut accelerator, std::hint::black_box(&pixels), &options);
            std::hint::black_box((codestreams, dispatches))
        });
    });
    group.finish();
}

fn p15_criterion() -> Criterion {
    Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP)
        .measurement_time(MEASUREMENT)
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = p15_criterion();
    targets = bench_fdwt97_stage, bench_product_encode
}
criterion_main!(benches);
