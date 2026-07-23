// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::backend::Backend;
use burn_cuda::{Cuda, CudaDevice};
use criterion::{BenchmarkId, Criterion, Throughput};
use j2k::{BatchDecodeOptions, EncodedImage, PreparedBatch};
use j2k_cuda::{CudaBatchDecodeResult, CudaBatchDecoder};
use j2k_ml::{CpuBurnDecoder, CudaUploadBurnDecoder};

mod cuda_telemetry;
#[path = "batch_decode_cuda/profile.rs"]
mod profile;
mod support;

use support::{
    decode_case::{
        decoded_pixels_per_batch, requests, require_burn_success, require_prepared_success,
    },
    input_selection::InputMode,
    process_policy::ProcessMode,
    workload::{materialize_workload, Workload, WorkloadSpec},
    workload_catalog::{workload_specs, BATCH_SIZES},
};

fn main() {
    let input_mode = InputMode::from_env().unwrap_or_else(|error| panic!("{error}"));
    let process_mode = ProcessMode::from_env().unwrap_or_else(|error| panic!("{error}"));
    let workload_specs = workload_specs();
    match process_mode {
        ProcessMode::Criterion => {
            let mut criterion = Criterion::default().configure_from_args();
            bench_codec_resident(&mut criterion, &workload_specs, input_mode);
            bench_burn_upload(&mut criterion, &workload_specs, input_mode);
            criterion.final_summary();
        }
        ProcessMode::Profile => profile::run(&workload_specs, input_mode),
    }
}

fn bench_codec_resident(
    criterion: &mut Criterion,
    workload_specs: &[WorkloadSpec],
    input_mode: InputMode,
) {
    let options = BatchDecodeOptions::default();
    let mut group = criterion.benchmark_group(format!(
        "j2k_owned_batch_codec_resident_cuda/input_{}",
        input_mode.label()
    ));

    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        let mut one_shot = CudaBatchDecoder::with_options(options);
        let mut prepared_decoder = CudaBatchDecoder::with_options(options);
        for (request_name, request, output_pixels) in requests(workload.dimensions, true) {
            for &batch_size in BATCH_SIZES {
                let inputs = workload.inputs(request, batch_size);
                group.throughput(Throughput::Elements(
                    u64::try_from(batch_size).expect("benchmark batch size fits u64"),
                ));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}/{request_name}/prepare_images", workload.name),
                        batch_size,
                    ),
                    &inputs,
                    |bencher, inputs| {
                        bencher.iter(|| {
                            let prepared = one_shot
                                .prepare(std::hint::black_box(inputs.clone()))
                                .expect("prepare CUDA codec batch");
                            require_prepared_success(&prepared);
                            std::hint::black_box(prepared)
                        });
                    },
                );
                group.throughput(Throughput::Elements(decoded_pixels_per_batch(
                    output_pixels,
                    batch_size,
                )));

                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}/{request_name}/one_shot_pixels", workload.name),
                        batch_size,
                    ),
                    &inputs,
                    |bencher, inputs| {
                        bencher.iter(|| {
                            // `decode_batch` retires the resident CUDA work before returning.
                            let decoded = one_shot
                                .decode_batch(std::hint::black_box(inputs.clone()))
                                .expect("CUDA codec-resident batch decode");
                            std::hint::black_box(require_codec_success(decoded))
                        });
                    },
                );
                let prepared = prepared_decoder
                    .prepare(inputs.clone())
                    .expect("prepare CUDA codec benchmark batch");
                require_prepared_success(&prepared);
                bench_prepared_codec(
                    &mut group,
                    &workload,
                    request_name,
                    batch_size,
                    &prepared,
                    &mut prepared_decoder,
                );
            }
        }
    }
    group.finish();
}

fn bench_prepared_codec(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    workload: &Workload,
    request_name: &str,
    batch_size: usize,
    prepared: &PreparedBatch,
    decoder: &mut CudaBatchDecoder,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("{}/{request_name}/prepared_pixels", workload.name),
            batch_size,
        ),
        prepared,
        |bencher, prepared| {
            bencher.iter(|| {
                // `decode_prepared` retires the resident CUDA work before returning.
                let batch_result = decoder
                    .decode_prepared(std::hint::black_box(prepared))
                    .expect("prepared CUDA codec-resident batch decode");
                std::hint::black_box(require_codec_success(batch_result))
            });
        },
    );
}

fn bench_burn_upload(
    criterion: &mut Criterion,
    workload_specs: &[WorkloadSpec],
    input_mode: InputMode,
) {
    let options = BatchDecodeOptions::default();
    let device = CudaDevice::default();
    let mut group = criterion.benchmark_group(format!(
        "j2k_owned_batch_burn_staged_cuda/input_{}",
        input_mode.label()
    ));

    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        let mut one_shot = CudaUploadBurnDecoder::new(device.clone(), options);
        let mut prepared_decoder = CudaUploadBurnDecoder::new(device.clone(), options);
        let mut staged = CpuBurnDecoder::<Cuda>::new(device.clone(), options);
        for (request_name, request, output_pixels) in requests(workload.dimensions, true) {
            for &batch_size in BATCH_SIZES {
                let inputs = workload.inputs(request, batch_size);
                group.throughput(Throughput::Elements(decoded_pixels_per_batch(
                    output_pixels,
                    batch_size,
                )));
                let case = BurnBenchCase {
                    workload: &workload,
                    request_name,
                    batch_size,
                    inputs: &inputs,
                };
                bench_staged_burn(&mut group, case, &mut staged, &device);
                bench_one_shot_burn(&mut group, case, &mut one_shot, &device);
                bench_prepared_burn(&mut group, case, &mut prepared_decoder, &device);
            }
        }
    }
    group.finish();
}

#[derive(Clone, Copy)]
struct BurnBenchCase<'a> {
    workload: &'a Workload,
    request_name: &'static str,
    batch_size: usize,
    inputs: &'a [EncodedImage],
}

fn bench_staged_burn(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: BurnBenchCase<'_>,
    staged: &mut CpuBurnDecoder<Cuda>,
    device: &CudaDevice,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "{}/{}/staged_cpu_upload_pixels",
                case.workload.name, case.request_name
            ),
            case.batch_size,
        ),
        case.inputs,
        |bencher, inputs| {
            bencher.iter(|| {
                let completed_batch = staged
                    .decode(std::hint::black_box(inputs.to_vec()))
                    .expect("CPU-staged CUDA Burn batch decode");
                let completed_batch = require_burn_success(completed_batch);
                <Cuda as Backend>::sync(device)
                    .expect("synchronize staged CUDA Burn benchmark completion");
                std::hint::black_box(completed_batch)
            });
        },
    );
}

fn bench_one_shot_burn(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: BurnBenchCase<'_>,
    decoder: &mut CudaUploadBurnDecoder,
    device: &CudaDevice,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "{}/{}/one_shot_pixels",
                case.workload.name, case.request_name
            ),
            case.batch_size,
        ),
        case.inputs,
        |bencher, inputs| {
            bencher.iter(|| {
                let completed_batch = decoder
                    .decode(std::hint::black_box(inputs.to_vec()))
                    .expect("CUDA Burn-upload batch decode");
                let completed_batch = require_burn_success(completed_batch);
                <Cuda as Backend>::sync(device)
                    .expect("synchronize CUDA Burn benchmark completion");
                std::hint::black_box(completed_batch)
            });
        },
    );
}

fn bench_prepared_burn(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: BurnBenchCase<'_>,
    decoder: &mut CudaUploadBurnDecoder,
    device: &CudaDevice,
) {
    let prepared = decoder
        .prepare(case.inputs.to_vec())
        .expect("prepare CUDA Burn benchmark batch");
    require_prepared_success(&prepared);
    group.bench_with_input(
        BenchmarkId::new(
            format!(
                "{}/{}/prepared_pixels",
                case.workload.name, case.request_name
            ),
            case.batch_size,
        ),
        &prepared,
        |bencher, prepared| {
            bencher.iter(|| {
                let completed_batch = decoder
                    .decode_prepared(std::hint::black_box(prepared))
                    .expect("prepared CUDA Burn-upload batch decode");
                let completed_batch = require_burn_success(completed_batch);
                <Cuda as Backend>::sync(device)
                    .expect("synchronize prepared CUDA Burn benchmark completion");
                std::hint::black_box(completed_batch)
            });
        },
    );
}

fn require_codec_success(decoded: CudaBatchDecodeResult) -> CudaBatchDecodeResult {
    assert!(
        decoded.errors().is_empty(),
        "benchmark decode returned indexed errors: {:?}",
        decoded.errors()
    );
    assert!(
        decoded.group_errors().is_empty(),
        "benchmark decode returned group errors: {:?}",
        decoded.group_errors()
    );
    assert_eq!(
        decoded.groups().len(),
        1,
        "benchmark workload must decode to exactly one homogeneous group"
    );
    decoded
}
