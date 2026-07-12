// SPDX-License-Identifier: MIT OR Apache-2.0

//! Grouped dispatch staging and caller-live budget relationships.

use super::super::super::rust_function_policy::FunctionCalls;
use super::super::{call_arguments::FunctionCallArguments, CudaTranscodeSources};

fn calls(sources: &CudaTranscodeSources, label: &str, function: &str) -> FunctionCalls {
    FunctionCalls::parse_many(label, &sources.sources(), function)
}

fn arguments(sources: &CudaTranscodeSources, label: &str, function: &str) -> FunctionCallArguments {
    FunctionCallArguments::parse_many(label, &sources.sources(), function)
}

pub(super) fn assert_policy(sources: &CudaTranscodeSources) {
    let staging = calls(
        sources,
        "CUDA grouped live staging budget",
        "live_staging_budget",
    );
    staging.assert_ordered(
        "CUDA grouped live staging aggregate",
        &["HostPhaseBudget::with_live_bytes", "preflight_capacity"],
    );

    calls(
        sources,
        "CUDA grouped resident dispatch",
        "dispatch_with_sink",
    )
    .assert_ordered(
        "CUDA grouped resident caller-live flow",
        &[
            "HostPhaseBudget::new",
            "try_vec_with_capacity",
            "resize_with",
            "try_vec_with_capacity",
            "resize",
            "try_vec_with_capacity",
            "live_bytes",
            "stage_resident_device_groups",
            "sink",
        ],
    );
    calls(
        sources,
        "CUDA grouped resident device staging",
        "stage_resident_device_groups",
    )
    .assert_ordered(
        "CUDA grouped resident per-group staging flow",
        &[
            "checked_element_product",
            "live_staging_budget",
            "try_vec_for_product",
            "drop",
        ],
    );

    let dispatch = arguments(
        sources,
        "CUDA grouped resident dispatch",
        "dispatch_with_sink",
    );
    dispatch.assert_ident_argument(
        "CUDA grouped staging live-byte forwarding",
        "stage_resident_device_groups",
        "live_metadata_bytes",
        1,
    );
    dispatch.assert_ident_argument(
        "CUDA grouped resident sink",
        "sink",
        "live_metadata_bytes",
        1,
    );
    arguments(
        sources,
        "CUDA grouped resident device staging",
        "stage_resident_device_groups",
    )
    .assert_ident_argument(
        "CUDA grouped staging preflight",
        "live_staging_budget",
        "live_metadata_bytes",
        1,
    );

    for (owner, callee) in [
        (
            "dispatch_htj2k97_preencoded_i16_batch_groups",
            "device_band_groups_to_preencoded_components",
        ),
        (
            "dispatch_htj2k97_compact_preencoded_i16_batch_groups",
            "device_band_groups_to_compact_preencoded_components",
        ),
    ] {
        arguments(sources, "CUDA grouped sink wrapper", owner).assert_ident_argument(
            "CUDA grouped sink live-byte forwarding",
            callee,
            "live_metadata_bytes",
            1,
        );
    }
}
