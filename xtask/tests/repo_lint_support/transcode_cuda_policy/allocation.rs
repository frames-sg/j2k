// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::contains_normalized;
use super::super::rust_function_policy::FunctionCalls;
use super::CudaTranscodeSources;

mod regressions;

fn calls(sources: &CudaTranscodeSources, source_name: &str, function_name: &str) -> FunctionCalls {
    FunctionCalls::parse_many(source_name, &sources.sources(), function_name)
}

fn assert_shared_allocation_helpers(sources: &CudaTranscodeSources) {
    let checked = calls(
        sources,
        "CUDA transcode allocation helpers",
        "checked_host_element_count",
    );
    checked.assert_ordered(
        "CUDA transcode byte preflight",
        &["checked_element_product", "saturating_mul"],
    );
    checked.assert_propagated(
        "CUDA transcode byte preflight",
        &["checked_element_product"],
    );

    calls(
        sources,
        "CUDA transcode phase allocation helpers",
        "try_vec_with_capacity_using",
    )
    .assert_ordered(
        "CUDA transcode allocator-capacity reconciliation",
        &[
            "preflight_capacity",
            "allocate",
            "requested_bytes",
            "account_vec",
        ],
    );

    calls(
        sources,
        "CUDA transcode allocation helpers",
        "try_transcode_vec_with_capacity",
    )
    .assert_ordered(
        "CUDA transcode isolated vector allocation",
        &[
            "checked_host_element_count",
            "HostPhaseBudget::new",
            "try_vec_with_capacity",
        ],
    );

    calls(
        sources,
        "CUDA transcode allocation helpers",
        "try_transcode_vec_for_product",
    )
    .assert_ordered(
        "CUDA transcode product allocation",
        &[
            "checked_host_element_count",
            "try_transcode_vec_with_capacity",
        ],
    );
    calls(
        sources,
        "CUDA transcode phase allocation helpers",
        "try_vec_from_slice",
    )
    .assert_ordered(
        "CUDA transcode slice copy",
        &["try_vec_with_capacity", "extend_from_slice"],
    );
    calls(
        sources,
        "CUDA transcode phase allocation helpers",
        "try_vec_from_array",
    )
    .assert_ordered(
        "CUDA transcode fixed metadata allocation",
        &["try_vec_with_capacity", "extend"],
    );

    let combined = sources.combined();
    assert!(
        contains_normalized(
            &combined,
            "if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES",
        ) && contains_normalized(&combined, "cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES",),
        "CUDA transcode allocations must reject byte products above the shared host cap"
    );
}

fn assert_staging_preallocates_before_materialization(sources: &CudaTranscodeSources) {
    calls(
        sources,
        "CUDA transcode staging",
        "flatten_f64_blocks_to_f32",
    )
    .assert_ordered(
        "single-job CUDA DCT staging",
        &["try_vec_for_product", "append_f64_blocks_to_f32"],
    );
    calls(
        sources,
        "CUDA transcode dispatch",
        "dispatch_reversible_dwt53_batch",
    )
    .assert_ordered(
        "CUDA reversible batch outputs",
        &[
            "HostPhaseBudget::new",
            "try_vec_with_capacity",
            "run_reversible",
            "account_reversible_output",
            "push",
        ],
    );
    calls(
        sources,
        "CUDA nonuniform 9/7 fallback",
        "dispatch_dwt97_batch",
    )
    .assert_ordered(
        "CUDA nonuniform 9/7 aggregate outputs",
        &[
            "HostPhaseBudget::new",
            "try_vec_with_capacity",
            "run_dwt97",
            "account_dwt97_output",
            "push",
        ],
    );
    calls(
        sources,
        "CUDA grouped resident dispatch",
        "dispatch_with_sink",
    )
    .assert_ordered(
        "CUDA grouped resident staging",
        &[
            "HostPhaseBudget::new",
            "try_vec_with_capacity",
            "resize_with",
            "try_vec_with_capacity",
            "resize",
            "try_vec_with_capacity",
            "stage_resident_device_groups",
        ],
    );
    calls(
        sources,
        "CUDA grouped resident device staging",
        "stage_resident_device_groups",
    )
    .assert_ordered(
        "CUDA grouped resident device staging",
        &[
            "checked_element_product",
            "live_staging_budget",
            "try_vec_for_product",
            "drop",
        ],
    );
    calls(
        sources,
        "CUDA resident target planning",
        "resident_group_targets",
    )
    .assert_ordered(
        "CUDA resident target allocation",
        &["checked_element_product", "try_vec_with_capacity"],
    );
    calls(sources, "CUDA resident target planning", "resident_targets").assert_ordered(
        "CUDA fixed-subband target allocation",
        &["try_vec_with_capacity", "push"],
    );

    for source in &sources.files {
        for forbidden in ["Vec::with_capacity", ".to_vec()", "vec!["] {
            assert!(
                !source.production.contains(forbidden),
                "{} must not use infallible variable-sized staging `{forbidden}`",
                source.relative
            );
        }
    }
    super::collection_scan::assert_no_infallible_collects(sources);
}

pub(super) fn assert_policy(sources: &CudaTranscodeSources) {
    assert!(
        include_str!("allocation.rs").lines().count() < 225,
        "CUDA transcode allocation policy must remain a focused module"
    );
    assert_shared_allocation_helpers(sources);
    assert_staging_preallocates_before_materialization(sources);
    regressions::assert_policy(sources);
}
