// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::backend::Backend;
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
use burn_flex::{Flex as CpuBackend, FlexDevice as CpuDevice};
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
use burn_ndarray::NdArrayDevice::Cpu as CpuDevice;
use criterion::{BenchmarkId, Criterion, Throughput};
use j2k::{BatchDecodeOptions, CpuBatchDecodeResult, CpuBatchDecoder};
use j2k_ml::{BurnBatchDecode, CpuBurnDecoder};

mod support;

use support::{
    decode_case::{decoded_pixels_per_batch, requests, require_prepared_success},
    input_selection::InputMode,
    process_policy::ProcessMode,
    workload::{materialize_workload, WorkloadSpec},
    workload_catalog::{workload_specs, BATCH_SIZES},
};

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
type CpuBackend = burn_ndarray::NdArray<f32, i64, i8>;

fn main() {
    let input_mode = InputMode::from_env().unwrap_or_else(|error| panic!("{error}"));
    match ProcessMode::from_env().unwrap_or_else(|error| panic!("{error}")) {
        ProcessMode::Criterion => {}
        ProcessMode::Profile => panic!("the CPU benchmark does not expose profile telemetry"),
    }
    let workload_specs = workload_specs();
    let mut criterion = Criterion::default().configure_from_args();
    bench_codec(&mut criterion, &workload_specs, input_mode);
    bench_burn(&mut criterion, &workload_specs, input_mode);
    criterion.final_summary();
}

fn bench_codec(criterion: &mut Criterion, workload_specs: &[WorkloadSpec], input_mode: InputMode) {
    let mut group = criterion.benchmark_group(format!(
        "j2k_owned_batch_codec_cpu/input_{}",
        input_mode.label()
    ));
    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        for (request_name, request, output_pixels) in requests(workload.dimensions, true) {
            for &batch_size in BATCH_SIZES {
                let inputs = workload.inputs(request, batch_size);
                group.throughput(Throughput::Elements(
                    u64::try_from(batch_size).expect("benchmark batch size fits u64"),
                ));
                let preflight_session = CpuBatchDecoder::new(BatchDecodeOptions::default());
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}/{request_name}/prepare_images", workload.name),
                        batch_size,
                    ),
                    &inputs,
                    |bencher, inputs| {
                        bencher.iter(|| {
                            let prepared = preflight_session
                                .prepare(std::hint::black_box(inputs.clone()))
                                .expect("prepare CPU codec batch");
                            require_prepared_success(&prepared);
                            std::hint::black_box(prepared)
                        });
                    },
                );
                group.throughput(Throughput::Elements(decoded_pixels_per_batch(
                    output_pixels,
                    batch_size,
                )));

                let mut one_shot = CpuBatchDecoder::new(BatchDecodeOptions::default());
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}/{request_name}/end_to_end_pixels", workload.name),
                        batch_size,
                    ),
                    &inputs,
                    |bencher, inputs| {
                        bencher.iter(|| {
                            let decoded = one_shot
                                .decode(std::hint::black_box(inputs.clone()))
                                .expect("CPU batch decode");
                            std::hint::black_box(require_codec_success(decoded))
                        });
                    },
                );

                let mut prepared_decoder = CpuBatchDecoder::new(BatchDecodeOptions::default());
                let prepared = prepared_decoder
                    .prepare(inputs.clone())
                    .expect("prepare CPU batch benchmark");
                require_prepared_success(&prepared);
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}/{request_name}/prepared_pixels", workload.name),
                        batch_size,
                    ),
                    &prepared,
                    |bencher, prepared| {
                        bencher.iter(|| {
                            let decoded = prepared_decoder
                                .decode_prepared(std::hint::black_box(prepared))
                                .expect("prepared CPU batch decode");
                            std::hint::black_box(require_codec_success(decoded))
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

fn bench_burn(criterion: &mut Criterion, workload_specs: &[WorkloadSpec], input_mode: InputMode) {
    let mut group = criterion.benchmark_group(format!(
        "j2k_owned_batch_burn_cpu/input_{}",
        input_mode.label()
    ));
    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        for (request_name, request, output_pixels) in requests(workload.dimensions, true) {
            for &batch_size in BATCH_SIZES {
                let inputs = workload.inputs(request, batch_size);
                group.throughput(Throughput::Elements(decoded_pixels_per_batch(
                    output_pixels,
                    batch_size,
                )));

                let mut one_shot =
                    CpuBurnDecoder::<CpuBackend>::new(CpuDevice, BatchDecodeOptions::default());
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}/{request_name}/end_to_end_pixels", workload.name),
                        batch_size,
                    ),
                    &inputs,
                    |bencher, inputs| {
                        bencher.iter(|| {
                            let decoded = one_shot
                                .decode(std::hint::black_box(inputs.clone()))
                                .expect("CPU Burn batch decode");
                            let decoded = require_burn_success(decoded);
                            CpuBackend::sync(&CpuDevice).expect("sync CPU Burn backend");
                            std::hint::black_box(decoded)
                        });
                    },
                );

                let mut prepared_decoder =
                    CpuBurnDecoder::<CpuBackend>::new(CpuDevice, BatchDecodeOptions::default());
                let prepared = prepared_decoder
                    .prepare(inputs.clone())
                    .expect("prepare CPU Burn benchmark");
                require_prepared_success(&prepared);
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}/{request_name}/prepared_pixels", workload.name),
                        batch_size,
                    ),
                    &prepared,
                    |bencher, prepared| {
                        bencher.iter(|| {
                            let decoded = prepared_decoder
                                .decode_prepared(std::hint::black_box(prepared))
                                .expect("prepared CPU Burn batch decode");
                            let decoded = require_burn_success(decoded);
                            CpuBackend::sync(&CpuDevice).expect("sync CPU Burn backend");
                            std::hint::black_box(decoded)
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

fn require_codec_success(decoded: CpuBatchDecodeResult) -> CpuBatchDecodeResult {
    assert!(
        decoded.errors().is_empty(),
        "benchmark decode returned indexed errors: {:?}",
        decoded.errors()
    );
    assert_eq!(
        decoded.groups().len(),
        1,
        "benchmark workload must decode to exactly one homogeneous group"
    );
    decoded
}

fn require_burn_success<B: Backend>(decoded: BurnBatchDecode<B>) -> BurnBatchDecode<B> {
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
