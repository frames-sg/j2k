// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::super::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn cuda_direct_decode_plan_remains_move_only() {
    let root = repo_root();
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let direct_plan = read("crates/j2k-cuda/src/direct_plan.rs");
    let accessors = read("crates/j2k-cuda/src/direct_plan/accessors.rs");
    let required_regions = read("crates/j2k-cuda/src/direct_plan/required_regions.rs");
    let shared = read("crates/j2k-cuda/src/direct_plan/shared.rs");
    let color_owners = read("crates/j2k-cuda/src/decoder/plan/color_owners.rs");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA direct decode plan", &direct_plan)
            .required(&[
                "mod accessors;",
                "mod required_regions;",
                "mod shared;",
                "The plan is move-only because its payload",
                "#[derive(Debug)]\npub(crate) struct CudaHtj2kDecodePlan",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone)]\npub(crate) struct CudaHtj2kDecodePlan",
                "HashMap",
            ]),
        PatternCheck::new("CUDA direct-plan owner graph", &shared)
            .required(&[
                "HostPhaseBudget::new(\"CUDA direct-plan owner graph\")",
                "budget.live_bytes()",
            ])
            .forbidden(&["HashMap"]),
        PatternCheck::new("CUDA direct-plan accessors", &accessors)
            .required(&[
                "append_payload_to_shared_with_budget(",
                "account_host_owners(",
            ])
            .forbidden(&["HashMap"]),
        PatternCheck::new("CUDA direct-plan required regions", &required_regions)
            .required(&[
                "struct RequiredBandRegions",
                "try_vec_with_capacity(capacity, REQUIRED_REGIONS)?",
                "binary_search_by_key",
                "try_vec_reserve(&mut required.entries, 1, REQUIRED_REGIONS)?",
                "entries.capacity()",
                "direct_plan_actual_capacities_accept_exact_cap_and_reject_one_over",
            ])
            .forbidden(&["HashMap", ".entry("]),
        PatternCheck::new("CUDA color direct-plan owner graph", &color_owners).required(&[
            "fn flatten_cuda_color_components(",
            "fn color_owner_graph_budget(",
            "append_payload_to_shared_with_budget(",
            "component.account_host_owners(",
        ]),
    ]);
    assert!(
        required_regions.lines().count() < 250,
        "CUDA direct-plan required-region owner must stay focused"
    );
    assert!(
        include_str!("direct_plan.rs").lines().count() < 100,
        "CUDA direct-plan policy leaf must stay focused"
    );
}
