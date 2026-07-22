// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_cuda::CudaDevice;
use j2k::BatchDecodeOptions;
use j2k_cuda::CudaBatchDecoder;
use j2k_ml::CudaBurnDecoder;

use crate::{
    cuda_telemetry::{
        capture_burn_telemetry, capture_codec_telemetry, print_cuda_telemetry, CudaTelemetryCase,
        CudaTelemetryRow,
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
    print_cuda_telemetry(&telemetry);
}

fn profile_codec_resident(
    workload_specs: &[WorkloadSpec],
    input_mode: InputMode,
) -> Vec<CudaTelemetryRow> {
    let options = BatchDecodeOptions::default();
    let mut telemetry = Vec::new();

    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        let mut one_shot = CudaBatchDecoder::with_options(options);
        let mut prepared_decoder = CudaBatchDecoder::with_options(options);
        for (request_name, request, output_pixels) in requests(workload.dimensions, true) {
            for &batch_size in LOW_BATCH_SIZES {
                let inputs = workload.inputs(request, batch_size);
                capture_codec_telemetry(
                    &mut telemetry,
                    CudaTelemetryCase::new(
                        "codec_resident",
                        "one_shot",
                        &workload,
                        request_name,
                        batch_size,
                        output_pixels,
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
                    .expect("prepare CUDA codec profile batch");
                require_prepared_success(&prepared);
                capture_codec_telemetry(
                    &mut telemetry,
                    CudaTelemetryCase::new(
                        "codec_resident",
                        "prepared",
                        &workload,
                        request_name,
                        batch_size,
                        output_pixels,
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
) -> Vec<CudaTelemetryRow> {
    let options = BatchDecodeOptions::default();
    let device = CudaDevice::default();
    let mut telemetry = Vec::new();

    for &spec in workload_specs {
        let workload = materialize_workload(spec, input_mode);
        let mut one_shot = CudaBurnDecoder::new(device.clone(), options);
        let mut prepared_decoder = CudaBurnDecoder::new(device.clone(), options);
        for (request_name, request, output_pixels) in requests(workload.dimensions, true) {
            for &batch_size in LOW_BATCH_SIZES {
                let inputs = workload.inputs(request, batch_size);
                capture_burn_telemetry(
                    &mut telemetry,
                    CudaTelemetryCase::new(
                        "burn_direct",
                        "one_shot",
                        &workload,
                        request_name,
                        batch_size,
                        output_pixels,
                    ),
                    &mut one_shot,
                    |decoder| decoder.decode(inputs.clone()).map(require_burn_success),
                );
                let prepared = prepared_decoder
                    .prepare(inputs.clone())
                    .expect("prepare CUDA Burn profile batch");
                require_prepared_success(&prepared);
                capture_burn_telemetry(
                    &mut telemetry,
                    CudaTelemetryCase::new(
                        "burn_direct",
                        "prepared",
                        &workload,
                        request_name,
                        batch_size,
                        output_pixels,
                    ),
                    &mut prepared_decoder,
                    |decoder| decoder.decode_prepared(&prepared).map(require_burn_success),
                );
            }
        }
    }
    telemetry
}
