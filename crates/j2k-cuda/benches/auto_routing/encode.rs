// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::Criterion;
use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, encode_j2k_lossy,
    encode_j2k_lossy_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kLossyEncodeOptions,
    J2kLossySamples, J2kRateTarget,
};
use j2k_core::BackendKind;
use j2k_cuda::CudaEncodeStageAccelerator;
use j2k_test_support::{
    auto_routing_operation_label as operation_label, auto_routing_route_cell as route_cell,
    auto_routing_sha256, AutoRoutingCell, AutoRoutingOperation, AutoRoutingPnm,
};

use crate::assert_output_parity;

type EncodeCase = AutoRoutingPnm;

pub(crate) fn bench_cell(
    criterion: &mut Criterion,
    case: &EncodeCase,
    operation: AutoRoutingOperation,
) -> AutoRoutingCell {
    let cpu = encode_cpu(case, operation)
        .unwrap_or_else(|error| panic!("CPU {} {}: {error}", case.id, operation_label(operation)));
    let mut probe_accelerator = CudaEncodeStageAccelerator::for_auto_host_output();
    let hybrid = encode_hybrid(case, operation, &mut probe_accelerator).unwrap_or_else(|error| {
        panic!(
            "hybrid CUDA {} {}: {error}",
            case.id,
            operation_label(operation)
        )
    });
    assert_output_parity(&case.id, operation, &cpu, &hybrid);

    let group_id = format!(
        "auto-routing_{}_{id}",
        operation_label(operation),
        id = case.id
    );
    let mut accelerator = probe_accelerator;
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
                    .expect("measured hybrid CUDA JPEG 2000 encode"),
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
                &lossless_options(case, EncodeBackendPreference::CpuOnly),
            )
            .map_err(|error| error.to_string())?;
            Ok(encoded.codestream)
        }
        AutoRoutingOperation::LossyEncode => {
            let encoded = encode_j2k_lossy(
                lossy_samples(case)?,
                &lossy_options(case, EncodeBackendPreference::CpuOnly),
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
    accelerator: &mut CudaEncodeStageAccelerator,
) -> Result<Vec<u8>, String> {
    let (codestream, dispatches) = match operation {
        AutoRoutingOperation::LosslessEncode => {
            let encoded = encode_j2k_lossless_with_accelerator(
                lossless_samples(case)?,
                &lossless_options(case, EncodeBackendPreference::Auto),
                BackendKind::Cuda,
                accelerator,
            )
            .map_err(|error| error.to_string())?;
            (encoded.codestream, encoded.dispatch_report.total())
        }
        AutoRoutingOperation::LossyEncode => {
            let encoded = encode_j2k_lossy_with_accelerator(
                lossy_samples(case)?,
                &lossy_options(case, EncodeBackendPreference::Auto),
                BackendKind::Cuda,
                accelerator,
            )
            .map_err(|error| error.to_string())?;
            (encoded.codestream, encoded.dispatch_report.total())
        }
        _ => return Err("invalid encode operation".to_string()),
    };
    if dispatches == 0 {
        return Err("CUDA hybrid encode did not dispatch any device stage".to_string());
    }
    Ok(codestream)
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

fn lossless_options(
    case: &EncodeCase,
    backend: EncodeBackendPreference,
) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(backend)
        .with_block_coding_mode(if case.codec.is_high_throughput() {
            J2kBlockCodingMode::HighThroughput
        } else {
            J2kBlockCodingMode::Classic
        })
        .with_max_decomposition_levels(Some(3))
        .with_validation(J2kEncodeValidation::External)
}

fn lossy_options(case: &EncodeCase, backend: EncodeBackendPreference) -> J2kLossyEncodeOptions {
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(backend)
        .with_block_coding_mode(if case.codec.is_high_throughput() {
            J2kBlockCodingMode::HighThroughput
        } else {
            J2kBlockCodingMode::Classic
        })
        .with_max_decomposition_levels(Some(3))
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(4.0)))
        .with_validation(J2kEncodeValidation::External);
    options.psnr_iteration_budget = 1;
    options
}
