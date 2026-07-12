// SPDX-License-Identifier: MIT OR Apache-2.0

use super::JpegDecodePreflightSources;

fn function_slice<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source.find(start).unwrap_or_else(|| panic!("find {start}"));
    let end = source[start..]
        .find(end)
        .map_or_else(|| panic!("find {end}"), |offset| start + offset);
    &source[start..end]
}

fn assert_validation_precedes_driver_work(body: &str, validation: &str, path: &str) {
    let validation = body
        .find(validation)
        .unwrap_or_else(|| panic!("find {path} validation"));
    let first_driver_work = [
        "self.inner.set_current()",
        "self.allocate(",
        "self.upload(",
        "self.upload_pinned(",
        "self.memset_",
        "self.decode_jpeg_rgb8_owned_validated(",
        "self.launch_jpeg_",
        ".copy_to_host(",
    ]
    .into_iter()
    .filter_map(|operation| body.find(operation))
    .min()
    .unwrap_or_else(|| panic!("find {path} first driver operation"));
    assert!(
        validation < first_driver_work,
        "{path} validation must finish before the first CUDA driver operation"
    );
}

pub(super) fn assert_preflight_ordering(sources: &JpegDecodePreflightSources) {
    let decode = format!("{}\n{}", sources.decode_api, sources.decode_launch);
    let owned = function_slice(
        &sources.decode_api,
        "pub fn decode_jpeg_rgb8_owned(",
        "pub fn decode_jpeg_rgb8_owned_into(",
    );
    let owned_into = function_slice(
        &sources.decode_api,
        "pub fn decode_jpeg_rgb8_owned_into(",
        "fn allocate_decode_statuses_with_cap(",
    );
    let validated = function_slice(
        &sources.decode_launch,
        "fn decode_jpeg_rgb8_owned_validated(",
        "fn launch_jpeg_decode_rgb8(",
    );
    let diagnostic = function_slice(
        &sources.diagnostics_execution,
        "fn diagnose_jpeg_420_entropy_self_sync_nonempty(",
        "fn launch_jpeg_entropy_sync420(",
    );
    assert_validation_precedes_driver_work(owned, "validate_jpeg_rgb8_plan(plan)?", "owned decode");
    assert_validation_precedes_driver_work(
        owned_into,
        "validate_jpeg_rgb8_plan_with_pitch(plan, pitch_bytes)?",
        "caller-owned decode",
    );
    assert_validation_precedes_driver_work(
        diagnostic,
        "validate_jpeg_entropy_chunk_plan(plan, subsequences)?",
        "entropy diagnostic",
    );
    for (path, body, allocation) in [
        ("owned decode", owned, "allocate_decode_statuses_with_cap("),
        (
            "caller-owned decode",
            owned_into,
            "allocate_decode_statuses_with_cap(",
        ),
        (
            "entropy diagnostic",
            diagnostic,
            "allocate_diagnostic_workspaces_with_cap(",
        ),
    ] {
        let last_status_allocation = body
            .rfind(allocation)
            .unwrap_or_else(|| panic!("find {path} status allocation"));
        let context_binding = body
            .find("self.inner.set_current()")
            .unwrap_or_else(|| panic!("find {path} context binding"));
        assert!(
            last_status_allocation < context_binding,
            "{path} must finish every fallible host status allocation before driver work"
        );
    }
    assert_common_output_initialization(owned, owned_into, validated, &decode);
}

fn assert_common_output_initialization(
    owned: &str,
    owned_into: &str,
    validated: &str,
    decode: &str,
) {
    let output_allocation = owned
        .find("self.allocate(validated.output_len)?")
        .expect("find owned JPEG output allocation");
    let owned_common = owned
        .find("self.decode_jpeg_rgb8_owned_validated(")
        .expect("find owned JPEG validated decode");
    assert!(output_allocation < owned_common);
    let output_size_check = owned_into
        .find("if output.byte_len() < validated.output_len")
        .expect("find caller-owned JPEG output size check");
    let caller_common = owned_into
        .find("self.decode_jpeg_rgb8_owned_validated(")
        .expect("find caller-owned JPEG validated decode");
    assert!(output_size_check < caller_common);
    let zero_fill = validated
        .find("self.memset_d8(output, 0, validated.output_len)?")
        .expect("find common JPEG output zero fill");
    let first_upload = validated
        .find("self.upload(plan.entropy_bytes)?")
        .expect("find first common JPEG upload");
    let launch = validated
        .find("self.launch_jpeg_decode_rgb8(")
        .expect("find common JPEG decode launch");
    assert!(
        zero_fill < first_upload && first_upload < launch,
        "every safe JPEG decode path must initialize its full validated extent before launch"
    );
    assert_eq!(
        decode
            .matches("self.memset_d8(output, 0, validated.output_len)?")
            .count(),
        1
    );
    assert_eq!(
        decode.matches("decode_jpeg_rgb8_owned_validated(").count(),
        3
    );
}
