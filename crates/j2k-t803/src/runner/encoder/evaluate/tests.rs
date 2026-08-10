// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::J2kEncodeDispatchReport;

use super::route::route_evidence;
use super::validation::validate_markers;
use crate::encoder::EncoderMatrix;
use crate::{EncodeRouteStageName, ExecutionLocation};

#[test]
fn planar_input_does_not_claim_an_interleaved_colour_transform() {
    let matrix = EncoderMatrix::parse(include_str!(
        "../../../../../../corpus/j2k-conformance/encoder-matrix-v2.toml"
    ))
    .expect("valid committed matrix");
    let case = matrix
        .cases
        .iter()
        .find(|case| case.id == "planar-sampled")
        .expect("planar matrix case");

    let route = route_evidence(case, J2kEncodeDispatchReport::default(), None);
    let rct = route
        .stages
        .iter()
        .find(|stage| stage.stage == EncodeRouteStageName::ForwardRct)
        .expect("RCT disclosure");

    assert_eq!(rct.location, ExecutionLocation::NotUsed);
}

#[test]
fn coefficient_recode_does_not_claim_pixel_domain_forward_stages() {
    let matrix = EncoderMatrix::parse(include_str!(
        "../../../../../../corpus/j2k-conformance/encoder-matrix-v2.toml"
    ))
    .expect("valid committed matrix");
    let case = matrix
        .cases
        .iter()
        .find(|case| case.id == "part15-recode-classic-j2k-to-htj2k")
        .expect("coefficient recode matrix case");

    let route = route_evidence(case, J2kEncodeDispatchReport::default(), None);

    for stage_name in [
        EncodeRouteStageName::ForwardRct,
        EncodeRouteStageName::ForwardIct,
        EncodeRouteStageName::ForwardDwt53,
        EncodeRouteStageName::ForwardDwt97,
        EncodeRouteStageName::Quantization,
    ] {
        let stage = route
            .stages
            .iter()
            .find(|stage| stage.stage == stage_name)
            .expect("recode stage disclosure");
        assert_eq!(stage.location, ExecutionLocation::NotUsed, "{stage_name:?}");
    }
}

#[test]
fn ht_code_block_dispatch_is_reported_as_device_tier1_work() {
    let matrix = EncoderMatrix::parse(include_str!(
        "../../../../../../corpus/j2k-conformance/encoder-matrix-v2.toml"
    ))
    .expect("valid committed matrix");
    let case = matrix
        .cases
        .iter()
        .find(|case| case.id == "part15-boundary-gray1-singleton")
        .expect("single-component HT matrix case");
    let dispatch = J2kEncodeDispatchReport {
        ht_code_block: 1,
        ..J2kEncodeDispatchReport::default()
    };

    let route = route_evidence(case, dispatch, Some(ExecutionLocation::Cuda));
    let tier1 = route
        .stages
        .iter()
        .find(|stage| stage.stage == EncodeRouteStageName::Tier1)
        .expect("Tier-1 disclosure");

    assert_eq!(tier1.location, ExecutionLocation::Cuda);
}

#[test]
fn all_ht_encoder_output_accepts_htonly_capability_mode() {
    let matrix = EncoderMatrix::parse(include_str!(
        "../../../../../../corpus/j2k-conformance/encoder-matrix-v2.toml"
    ))
    .expect("valid committed matrix");
    let case = matrix
        .cases
        .iter()
        .find(|case| case.id == "part15-boundary-gray1-singleton")
        .expect("single-component HT matrix case");
    let input = super::super::input::generate_input(case).expect("generate HT input");
    let output = super::super::backend::encode_cpu_case(case, &input).expect("encode HT input");

    validate_markers(case, &output.codestream).expect("HTONLY CAP matches all-HT COD style");
}

#[test]
fn coefficient_derived_bmagb_may_be_below_source_precision() {
    let matrix = EncoderMatrix::parse(include_str!(
        "../../../../../../corpus/j2k-conformance/encoder-matrix-v2.toml"
    ))
    .expect("valid committed matrix");
    let case = matrix
        .cases
        .iter()
        .find(|case| case.id == "part15-pairwise-03")
        .expect("lossy HT matrix case");
    let input = super::super::input::generate_input(case).expect("generate lossy HT input");
    let output =
        super::super::backend::encode_cpu_case(case, &input).expect("encode lossy HT input");

    let capabilities = j2k_native::inspect_htj2k_capabilities(&output.codestream)
        .expect("inspect encoded capabilities")
        .expect("Part 15 capabilities");
    assert_eq!(capabilities.magnitude_bound(), 8);
    validate_markers(case, &output.codestream)
        .expect("quantized cleanup magnitudes, not source precision, determine BMAGB");
}
