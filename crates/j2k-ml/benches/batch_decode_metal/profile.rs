// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::backend::Backend;
use burn_wgpu::Wgpu;
use j2k::{BatchDecodeOptions, BatchLayout};
use j2k_metal::MetalBatchDecoder;
use j2k_ml::{CpuBurnDecoder, MetalBurnDecoder};

use crate::{
    metal_telemetry::{
        capture_burn_telemetry, capture_codec_telemetry, capture_staged_telemetry,
        print_metal_telemetry, MetalTelemetryCase, MetalTelemetryRow,
    },
    require_codec_success,
    support::{
        decode_case::{requests, require_burn_success, require_prepared_success},
        input_selection::InputMode,
        workload::{materialize_workload, WorkloadSpec},
        workload_catalog::LOW_BATCH_SIZES,
    },
};

pub(super) fn run(workload_specs: &[WorkloadSpec], input_mode: InputMode) {
    let mut telemetry = profile_codec_resident(workload_specs, input_mode);
    telemetry.extend(profile_burn_direct(workload_specs, input_mode));
    print_metal_telemetry(&telemetry);
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
