// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};
use criterion::{BenchmarkId, Criterion, Throughput};
use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage, PreparedBatch};
use j2k_metal::{MetalBatchDecodeResult, MetalBatchDecoder};
use j2k_ml::{BurnBatchDecode, CpuBurnDecoder, MetalBurnDecoder};

#[path = "support/metal_process_policy.rs"]
mod metal_process_policy;
mod metal_telemetry;
#[path = "support/process_policy.rs"]
mod process_policy;
mod support;

use metal_process_policy::ensure_metal_criterion_instrumentation_disabled;
use metal_telemetry::{
    capture_burn_telemetry, capture_codec_telemetry, capture_staged_telemetry,
    print_metal_telemetry, MetalTelemetryCase, MetalTelemetryRow,
};
use process_policy::ProcessMode;
use support::{
    decode_case::{decoded_pixels_per_batch, requests, require_prepared_success},
    input_selection::InputMode,
    workload::{materialize_workload, workload_specs, Workload, WorkloadSpec},
    BATCH_SIZES, LOW_BATCH_SIZES,
};

fn main() {
    let input_mode = InputMode::from_env().unwrap_or_else(|error| panic!("{error}"));
    let process_mode = ProcessMode::from_env().unwrap_or_else(|error| panic!("{error}"));
    let workload_specs = workload_specs();
    match process_mode {
        ProcessMode::Criterion => {
            ensure_metal_criterion_instrumentation_disabled()
                .unwrap_or_else(|error| panic!("{error}"));
            let mut criterion = Criterion::default().configure_from_args();
            bench_codec_resident(&mut criterion, &workload_specs, input_mode);
            bench_burn_direct(&mut criterion, &workload_specs, input_mode);
            criterion.final_summary();
        }
        ProcessMode::Profile => {
            let mut telemetry = profile_codec_resident(&workload_specs, input_mode);
            telemetry.extend(profile_burn_direct(&workload_specs, input_mode));
            print_metal_telemetry(&telemetry);
        }
    }
}

fn bench_codec_resident(
    criterion: &mut Criterion,
    workload_specs: &[WorkloadSpec],
    input_mode: InputMode,
) {
    // Resident RGB groups expose ordinary interleaved `Surface` views, so the
    // codec-resident color matrix is explicitly NHWC. Grayscale is layout
    // equivalent and shares the same retained sessions.
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut group = criterion.benchmark_group(format!(
        "j2k_owned_batch_codec_resident_metal/input_{}",
        input_mode.label()
    ));

    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        let mut one_shot = MetalBatchDecoder::system_default_with_options(options)
            .expect("create persistent Metal codec benchmark session");
        let mut prepared_decoder = MetalBatchDecoder::system_default_with_options(options)
            .expect("create prepared Metal codec benchmark session");
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
                            std::hint::black_box(
                                one_shot
                                    .prepare(std::hint::black_box(inputs.clone()))
                                    .expect("prepare Metal codec batch"),
                            )
                        });
                    },
                );
                group.throughput(Throughput::Elements(decoded_pixels_per_batch(
                    output_pixels,
                    batch_size,
                )));

                bench_one_shot_codec(
                    &mut group,
                    &workload,
                    request_name,
                    batch_size,
                    &inputs,
                    &mut one_shot,
                );

                let prepared = prepared_decoder
                    .prepare(inputs.clone())
                    .expect("prepare Metal codec benchmark batch");
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

fn bench_one_shot_codec(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    workload: &Workload,
    request_name: &str,
    batch_size: usize,
    inputs: &[EncodedImage],
    decoder: &mut MetalBatchDecoder,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("{}/{request_name}/one_shot_pixels", workload.name),
            batch_size,
        ),
        inputs,
        |bencher, inputs| {
            bencher.iter(|| {
                // `decode_batch` waits for resident Metal output before returning.
                let batch_result = decoder
                    .decode_batch(std::hint::black_box(inputs.to_vec()))
                    .expect("Metal codec-resident batch decode");
                std::hint::black_box(require_codec_success(batch_result))
            });
        },
    );
}

fn bench_prepared_codec(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    workload: &Workload,
    request_name: &str,
    batch_size: usize,
    prepared: &PreparedBatch,
    decoder: &mut MetalBatchDecoder,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("{}/{request_name}/prepared_pixels", workload.name),
            batch_size,
        ),
        prepared,
        |bencher, prepared| {
            bencher.iter(|| {
                // `decode_prepared` waits for the resident surfaces before returning.
                let batch_result = decoder
                    .decode_prepared(std::hint::black_box(prepared))
                    .expect("prepared Metal codec-resident batch decode");
                std::hint::black_box(require_codec_success(batch_result))
            });
        },
    );
}

fn bench_burn_direct(
    criterion: &mut Criterion,
    workload_specs: &[WorkloadSpec],
    input_mode: InputMode,
) {
    let options = BatchDecodeOptions::default();
    let mut group = criterion.benchmark_group(format!(
        "j2k_owned_batch_burn_direct_metal/input_{}",
        input_mode.label()
    ));
    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        let mut one_shot = MetalBurnDecoder::system_default(options)
            .expect("create paired Metal Burn benchmark session");
        let mut prepared_decoder = MetalBurnDecoder::system_default(options)
            .expect("create prepared paired Metal Burn benchmark session");
        let one_shot_device = one_shot.device().clone();
        let mut staged = CpuBurnDecoder::<Wgpu>::new(one_shot_device.clone(), options);
        for (request_name, request, output_pixels) in requests(workload.dimensions, true) {
            for &batch_size in BATCH_SIZES {
                let inputs = workload.inputs(request, batch_size);
                group.throughput(Throughput::Elements(decoded_pixels_per_batch(
                    output_pixels,
                    batch_size,
                )));
                bench_staged_burn(
                    &mut group,
                    &workload,
                    request_name,
                    batch_size,
                    &inputs,
                    &mut staged,
                    &one_shot_device,
                );
                bench_one_shot_burn(
                    &mut group,
                    &workload,
                    request_name,
                    batch_size,
                    &inputs,
                    &mut one_shot,
                );
                let prepared = prepared_decoder
                    .prepare(inputs.clone())
                    .expect("prepare Metal Burn benchmark batch");
                require_prepared_success(&prepared);
                bench_prepared_burn(
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

fn profile_codec_resident(
    workload_specs: &[WorkloadSpec],
    input_mode: InputMode,
) -> Vec<MetalTelemetryRow> {
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut telemetry = Vec::new();

    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        let mut one_shot = MetalBatchDecoder::system_default_with_options(options)
            .expect("create Metal codec profile session");
        let mut prepared_decoder = MetalBatchDecoder::system_default_with_options(options)
            .expect("create prepared Metal codec profile session");
        for (request_name, request, output_pixels) in requests(workload.dimensions, true) {
            for &batch_size in LOW_BATCH_SIZES {
                let inputs = workload.inputs(request, batch_size);
                capture_codec_telemetry(
                    &mut telemetry,
                    MetalTelemetryCase::new(
                        "codec_resident",
                        "one_shot",
                        &workload,
                        request_name,
                        batch_size,
                        output_pixels,
                        0,
                    ),
                    &mut one_shot,
                    |decoder| {
                        decoder
                            .decode_batch(inputs.clone())
                            .map(require_codec_success)
                    },
                );
                let prepared = prepared_decoder
                    .prepare(inputs.clone())
                    .expect("prepare Metal codec profile batch");
                require_prepared_success(&prepared);
                capture_codec_telemetry(
                    &mut telemetry,
                    MetalTelemetryCase::new(
                        "codec_resident",
                        "prepared",
                        &workload,
                        request_name,
                        batch_size,
                        output_pixels,
                        0,
                    ),
                    &mut prepared_decoder,
                    |decoder| {
                        decoder
                            .decode_prepared(&prepared)
                            .map(require_codec_success)
                    },
                );
            }
        }
    }
    telemetry
}

fn profile_burn_direct(
    workload_specs: &[WorkloadSpec],
    input_mode: InputMode,
) -> Vec<MetalTelemetryRow> {
    let options = BatchDecodeOptions::default();
    let mut telemetry = Vec::new();

    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        let mut one_shot = MetalBurnDecoder::system_default(options)
            .expect("create paired Metal Burn profile session");
        let mut prepared_decoder = MetalBurnDecoder::system_default(options)
            .expect("create prepared paired Metal Burn profile session");
        let device = one_shot.device().clone();
        let mut staged = CpuBurnDecoder::<Wgpu>::new(device.clone(), options);
        for (request_name, request, output_pixels) in requests(workload.dimensions, true) {
            for &batch_size in LOW_BATCH_SIZES {
                let inputs = workload.inputs(request, batch_size);
                capture_staged_telemetry(
                    &mut telemetry,
                    MetalTelemetryCase::new(
                        "burn_staged",
                        "cpu_upload",
                        &workload,
                        request_name,
                        batch_size,
                        output_pixels,
                        1,
                    ),
                    || {
                        let decoded = staged
                            .decode(inputs.clone())
                            .expect("staged Metal telemetry decode");
                        let decoded = require_burn_success(decoded);
                        <Wgpu as Backend>::sync(&device)
                            .expect("synchronize staged Metal telemetry completion");
                        decoded
                    },
                );
                capture_burn_telemetry(
                    &mut telemetry,
                    MetalTelemetryCase::new(
                        "burn_direct",
                        "one_shot",
                        &workload,
                        request_name,
                        batch_size,
                        output_pixels,
                        0,
                    ),
                    &mut one_shot,
                    |decoder| decoder.decode(inputs.clone()).map(require_burn_success),
                );
                let prepared = prepared_decoder
                    .prepare(inputs.clone())
                    .expect("prepare Metal Burn profile batch");
                require_prepared_success(&prepared);
                capture_burn_telemetry(
                    &mut telemetry,
                    MetalTelemetryCase::new(
                        "burn_direct",
                        "prepared",
                        &workload,
                        request_name,
                        batch_size,
                        output_pixels,
                        0,
                    ),
                    &mut prepared_decoder,
                    |decoder| decoder.decode_prepared(&prepared).map(require_burn_success),
                );
            }
        }
    }
    telemetry
}

fn bench_staged_burn(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    workload: &Workload,
    request_name: &str,
    batch_size: usize,
    inputs: &[EncodedImage],
    decoder: &mut CpuBurnDecoder<Wgpu>,
    device: &WgpuDevice,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("{}/{request_name}/staged_cpu_upload_pixels", workload.name),
            batch_size,
        ),
        inputs,
        |bencher, inputs| {
            bencher.iter(|| {
                let batch_result = decoder
                    .decode(std::hint::black_box(inputs.to_vec()))
                    .expect("CPU-staged Metal Burn batch decode");
                let batch_result = require_burn_success(batch_result);
                <Wgpu as Backend>::sync(device)
                    .expect("synchronize staged Metal Burn benchmark completion");
                std::hint::black_box(batch_result)
            });
        },
    );
}

fn bench_one_shot_burn(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    workload: &Workload,
    request_name: &str,
    batch_size: usize,
    inputs: &[EncodedImage],
    decoder: &mut MetalBurnDecoder,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("{}/{request_name}/one_shot_pixels", workload.name),
            batch_size,
        ),
        inputs,
        |bencher, inputs| {
            bencher.iter(|| {
                let batch_result = decoder
                    .decode(std::hint::black_box(inputs.to_vec()))
                    .expect("Metal Burn-direct batch decode");
                let batch_result = require_burn_success(batch_result);
                std::hint::black_box(batch_result)
            });
        },
    );
}

fn bench_prepared_burn(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    workload: &Workload,
    request_name: &str,
    batch_size: usize,
    prepared: &PreparedBatch,
    decoder: &mut MetalBurnDecoder,
) {
    group.bench_with_input(
        BenchmarkId::new(
            format!("{}/{request_name}/prepared_pixels", workload.name),
            batch_size,
        ),
        prepared,
        |bencher, prepared| {
            bencher.iter(|| {
                let batch_result = decoder
                    .decode_prepared(std::hint::black_box(prepared))
                    .expect("prepared Metal Burn-direct batch decode");
                let batch_result = require_burn_success(batch_result);
                std::hint::black_box(batch_result)
            });
        },
    );
}

fn require_codec_success(decoded: MetalBatchDecodeResult) -> MetalBatchDecodeResult {
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

fn require_burn_success(decoded: BurnBatchDecode<Wgpu>) -> BurnBatchDecode<Wgpu> {
    assert!(
        decoded.errors.is_empty(),
        "benchmark adapter returned indexed errors: {:?}",
        decoded.errors
    );
    assert!(
        decoded.group_errors.is_empty(),
        "benchmark adapter returned group errors: {:?}",
        decoded.group_errors
    );
    assert_eq!(
        decoded.groups.len(),
        1,
        "benchmark workload must materialize exactly one Burn tensor group"
    );
    decoded
}
