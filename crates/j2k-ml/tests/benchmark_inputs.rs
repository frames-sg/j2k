// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../benches/support/decode_case.rs"]
mod decode_case;
#[path = "../benches/support/fixture.rs"]
mod fixture;
#[path = "../benches/support/input_selection.rs"]
mod input_selection;
#[path = "../benches/support/workload.rs"]
mod workload;

use std::{collections::HashSet, sync::Arc};

use input_selection::InputMode;
use j2k::{BatchDecodeOptions, CpuBatchDecoder, DecodeRequest, Downscale};
use workload::{materialize_workload, WorkloadSpec, GENERATED_BATCH_SIZE};

fn tiny_gray12() -> WorkloadSpec {
    WorkloadSpec::new("tiny_gray12", 16, 16, 1, 12, false)
}

#[test]
fn distinct_inputs_have_unique_owners_and_payloads() {
    let workload = materialize_workload(tiny_gray12(), InputMode::Distinct);
    assert_eq!(workload.name, "tiny_gray12");
    assert_eq!(workload.dimensions, (16, 16));
    let inputs = workload.inputs(DecodeRequest::Full, GENERATED_BATCH_SIZE);

    assert_eq!(inputs.len(), GENERATED_BATCH_SIZE);
    for (index, input) in inputs.iter().enumerate() {
        assert!(inputs[..index]
            .iter()
            .all(|previous| !Arc::ptr_eq(&previous.bytes, &input.bytes)));
    }
    let payloads = inputs
        .iter()
        .map(|input| input.bytes.as_ref())
        .collect::<HashSet<_>>();
    assert_eq!(payloads.len(), GENERATED_BATCH_SIZE);
}

#[test]
fn repeated_inputs_share_one_owner() {
    let workload = materialize_workload(tiny_gray12(), InputMode::Repeated);
    let inputs = workload.inputs(DecodeRequest::Full, GENERATED_BATCH_SIZE);

    assert!(inputs
        .iter()
        .all(|input| Arc::ptr_eq(&inputs[0].bytes, &input.bytes)));
}

#[test]
fn distinct_inputs_prepare_as_one_homogeneous_group() {
    let workload = materialize_workload(tiny_gray12(), InputMode::Distinct);
    let inputs = workload.inputs(DecodeRequest::Full, 8);
    let decoder = CpuBatchDecoder::new(BatchDecodeOptions::default());

    let prepared = decoder.prepare(inputs).expect("prepare benchmark inputs");

    decode_case::require_prepared_success(&prepared);
    assert_eq!(
        prepared.groups()[0].source_indices(),
        &[0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn benchmark_request_matrix_covers_all_owned_batch_requests() {
    let requests = decode_case::requests((16, 12), true);
    assert_eq!(requests.len(), 4);
    assert!(matches!(requests[0].1, DecodeRequest::Full));
    assert!(matches!(requests[1].1, DecodeRequest::Region { .. }));
    assert!(matches!(
        requests[2].1,
        DecodeRequest::Reduced {
            scale: Downscale::Half
        }
    ));
    assert!(matches!(
        requests[3].1,
        DecodeRequest::RegionReduced {
            scale: Downscale::Half,
            ..
        }
    ));
    assert_eq!(decode_case::decoded_pixels_per_batch(16, 8), 128);
}

#[test]
fn input_mode_defaults_to_distinct_and_rejects_unknown_values() {
    assert_eq!(InputMode::parse(None).unwrap(), InputMode::Distinct);
    assert_eq!(
        InputMode::parse(Some("distinct")).unwrap(),
        InputMode::Distinct
    );
    assert_eq!(
        InputMode::parse(Some("repeated")).unwrap(),
        InputMode::Repeated
    );

    let error = InputMode::parse(Some("broadcast")).unwrap_err();
    assert!(error.contains("J2K_ML_BATCH_INPUT_MODE"));
    assert!(error.contains("broadcast"));

    assert_eq!(InputMode::Distinct.label(), "distinct");
    assert_eq!(InputMode::Repeated.label(), "repeated");
    if let Err(error) = InputMode::from_env() {
        assert!(error.contains("J2K_ML_BATCH_INPUT_MODE"));
    }
}
