// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::Criterion;
use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, encode_j2k_lossy,
    encode_j2k_lossy_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kLossyEncodeOptions,
    J2kLossySamples, J2kRateTarget,
};
use j2k_core::{BackendKind, PixelFormat};
use j2k_metal::MetalEncodeStageAccelerator;
use j2k_test_support::{
    append_auto_routing_output as append_output, auto_routing_operation_label as operation_label,
    auto_routing_route_cell as route_cell, auto_routing_sha256, AutoRoutingCell,
    AutoRoutingOperation, AutoRoutingPnm,
};

use crate::assert_output_parity;

type EncodeCase = AutoRoutingPnm;

const BATCH_SIZE: usize = 16;

pub(crate) fn bench_cell(
    criterion: &mut Criterion,
    case: &EncodeCase,
    operation: AutoRoutingOperation,
) -> AutoRoutingCell {
    let cpu = encode_cpu(case, operation, 1)
        .unwrap_or_else(|error| panic!("CPU {} {}: {error}", case.id, operation_label(operation)));
    let mut probe_accelerator = MetalEncodeStageAccelerator::for_host_output_benchmark();
    let hybrid =
        encode_hybrid(case, operation, 1, &mut probe_accelerator).unwrap_or_else(|error| {
            panic!(
                "hybrid Metal {} {}: {error}",
                case.id,
                operation_label(operation)
            )
        });
    assert_output_parity(&case.id, operation, &cpu, &hybrid);
    if operation == AutoRoutingOperation::LosslessEncode {
        verify_lossless_output(case, &cpu)
            .unwrap_or_else(|error| panic!("lossless output verification {}: {error}", case.id));
    }
    let auto = encode_auto(case, operation, 1)
        .unwrap_or_else(|error| panic!("Auto {} {}: {error}", case.id, operation_label(operation)));
    assert_output_parity(&case.id, operation, &cpu, &auto);

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
                encode_cpu(case, operation, 1).expect("measured CPU JPEG 2000 encode"),
            )
        });
    });
    group.bench_function("hybrid", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(
                encode_hybrid(case, operation, 1, &mut accelerator)
                    .expect("measured hybrid Metal JPEG 2000 encode"),
            )
        });
    });
    group.finish();
    route_cell(&case.id, operation, &group_id, auto_routing_sha256(&cpu))
}

pub(crate) fn bench_batch_cell(criterion: &mut Criterion, case: &EncodeCase) -> AutoRoutingCell {
    assert!(
        case.codec.is_high_throughput(),
        "resident host-output batch evidence requires HTJ2K"
    );
    let operation = AutoRoutingOperation::LosslessEncode;
    let cpu = encode_cpu(case, operation, BATCH_SIZE)
        .unwrap_or_else(|error| panic!("CPU {} batch-{BATCH_SIZE}: {error}", case.id));
    let mut probe_accelerator = MetalEncodeStageAccelerator::for_host_output_benchmark();
    let hybrid = encode_hybrid(case, operation, BATCH_SIZE, &mut probe_accelerator)
        .unwrap_or_else(|error| panic!("hybrid Metal {} batch-{BATCH_SIZE}: {error}", case.id));
    assert_output_parity(
        &format!("{} batch-{BATCH_SIZE}", case.id),
        operation,
        &cpu,
        &hybrid,
    );
    let auto = encode_auto(case, operation, BATCH_SIZE)
        .unwrap_or_else(|error| panic!("Auto {} batch-{BATCH_SIZE}: {error}", case.id));
    assert_output_parity(
        &format!("{} batch-{BATCH_SIZE}", case.id),
        operation,
        &cpu,
        &auto,
    );

    let group_id = format!("auto-routing_lossless-encode-batch{BATCH_SIZE}_{}", case.id);
    let mut accelerator = MetalEncodeStageAccelerator::for_host_output_benchmark();
    let mut group = criterion.benchmark_group(&group_id);
    group.bench_function("cpu", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(
                encode_cpu(case, operation, BATCH_SIZE).expect("measured CPU HTJ2K batch encode"),
            )
        });
    });
    group.bench_function("hybrid", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(
                encode_hybrid(case, operation, BATCH_SIZE, &mut accelerator)
                    .expect("measured resident Metal HTJ2K batch encode"),
            )
        });
    });
    group.finish();
    let mut cell = route_cell(&case.id, operation, &group_id, auto_routing_sha256(&cpu));
    cell.id = format!("lossless-encode-batch{BATCH_SIZE}-{}", case.id);
    cell
}

fn encode_cpu(
    case: &EncodeCase,
    operation: AutoRoutingOperation,
    batch_size: usize,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    for _ in 0..batch_size {
        let codestream = match operation {
            AutoRoutingOperation::LosslessEncode => encode_j2k_lossless(
                lossless_samples(case)?,
                &lossless_options(case, EncodeBackendPreference::CpuOnly),
            )
            .map(|encoded| encoded.codestream)
            .map_err(|error| error.to_string())?,
            AutoRoutingOperation::LossyEncode => encode_j2k_lossy(
                lossy_samples(case)?,
                &lossy_options(case, EncodeBackendPreference::CpuOnly),
            )
            .map(|encoded| encoded.codestream)
            .map_err(|error| error.to_string())?,
            _ => return Err("invalid encode operation".to_string()),
        };
        if batch_size == 1 {
            output = codestream;
        } else {
            append_output(&mut output, &codestream)?;
        }
    }
    Ok(output)
}

fn encode_hybrid(
    case: &EncodeCase,
    operation: AutoRoutingOperation,
    batch_size: usize,
    accelerator: &mut MetalEncodeStageAccelerator,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    for _ in 0..batch_size {
        let (codestream, dispatches) = match operation {
            AutoRoutingOperation::LosslessEncode => {
                let encoded = encode_j2k_lossless_with_accelerator(
                    lossless_samples(case)?,
                    &lossless_options(case, EncodeBackendPreference::Auto),
                    BackendKind::Metal,
                    accelerator,
                )
                .map_err(|error| error.to_string())?;
                (encoded.codestream, encoded.dispatch_report.total())
            }
            AutoRoutingOperation::LossyEncode => {
                let encoded = encode_j2k_lossy_with_accelerator(
                    lossy_samples(case)?,
                    &lossy_options(case, EncodeBackendPreference::Auto),
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
        if batch_size == 1 {
            output = codestream;
        } else {
            append_output(&mut output, &codestream)?;
        }
    }
    Ok(output)
}

fn encode_auto(
    case: &EncodeCase,
    operation: AutoRoutingOperation,
    batch_size: usize,
) -> Result<Vec<u8>, String> {
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();
    let mut output = Vec::new();
    for _ in 0..batch_size {
        let codestream = match operation {
            AutoRoutingOperation::LosslessEncode => encode_j2k_lossless_with_accelerator(
                lossless_samples(case)?,
                &lossless_options(case, EncodeBackendPreference::Auto),
                BackendKind::Metal,
                &mut accelerator,
            )
            .map(|encoded| encoded.codestream)
            .map_err(|error| error.to_string())?,
            AutoRoutingOperation::LossyEncode => encode_j2k_lossy_with_accelerator(
                lossy_samples(case)?,
                &lossy_options(case, EncodeBackendPreference::Auto),
                BackendKind::Metal,
                &mut accelerator,
            )
            .map(|encoded| encoded.codestream)
            .map_err(|error| error.to_string())?,
            _ => return Err("invalid encode operation".to_string()),
        };
        if batch_size == 1 {
            output = codestream;
        } else {
            append_output(&mut output, &codestream)?;
        }
    }
    Ok(output)
}

fn verify_lossless_output(case: &EncodeCase, codestream: &[u8]) -> Result<(), String> {
    let (format, stride) = match case.components {
        1 => (PixelFormat::Gray8, case.width as usize),
        3 => (
            PixelFormat::Rgb8,
            (case.width as usize)
                .checked_mul(3)
                .ok_or_else(|| "lossless verification stride overflow".to_string())?,
        ),
        _ => return Err("lossless verification supports Gray8 and RGB8".to_string()),
    };
    let len = stride
        .checked_mul(case.height as usize)
        .ok_or_else(|| "lossless verification output length overflow".to_string())?;
    let mut output_pixels = vec![0u8; len];
    let mut decoder = j2k::J2kDecoder::new(codestream).map_err(|error| error.to_string())?;
    decoder
        .decode_into(&mut output_pixels, stride, format)
        .map_err(|error| error.to_string())?;
    if output_pixels != case.pixels {
        return Err("decoded pixels do not match the source PNM".to_string());
    }
    Ok(())
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
