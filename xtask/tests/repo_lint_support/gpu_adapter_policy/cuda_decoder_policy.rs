// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{assert_pattern_checks, repo_root, PatternCheck};

mod architecture;
mod color_runtime;

struct CudaDecoderSources {
    direct_plan: String,
    direct_plan_required_regions: String,
    decoder: String,
    api: String,
    plan: String,
    plan_color_owners: String,
    resident: String,
    resident_cleanup_dequant: String,
    resident_component: String,
    resident_helpers: String,
    resident_idwt: String,
    resident_routing: String,
    resident_surface: String,
    color_batch: String,
    color_batch_host_owners: String,
    color_store: String,
    color_store_batch: String,
    profile: String,
}

impl CudaDecoderSources {
    fn read() -> Self {
        let root = repo_root();
        let read = |relative: &str| {
            fs::read_to_string(root.join(relative))
                .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        };
        Self {
            direct_plan: read("crates/j2k-cuda/src/direct_plan.rs"),
            direct_plan_required_regions: read(
                "crates/j2k-cuda/src/direct_plan/required_regions.rs",
            ),
            decoder: read("crates/j2k-cuda/src/decoder.rs"),
            api: read("crates/j2k-cuda/src/decoder/api.rs"),
            plan: read("crates/j2k-cuda/src/decoder/plan.rs"),
            plan_color_owners: read("crates/j2k-cuda/src/decoder/plan/color_owners.rs"),
            resident: read("crates/j2k-cuda/src/decoder/resident.rs"),
            resident_cleanup_dequant: read(
                "crates/j2k-cuda/src/decoder/resident/cleanup_dequant.rs",
            ),
            resident_component: read("crates/j2k-cuda/src/decoder/resident/component.rs"),
            resident_helpers: read("crates/j2k-cuda/src/decoder/resident/helpers.rs"),
            resident_idwt: read("crates/j2k-cuda/src/decoder/resident/idwt.rs"),
            resident_routing: read("crates/j2k-cuda/src/decoder/resident/routing.rs"),
            resident_surface: read("crates/j2k-cuda/src/decoder/resident/surface.rs"),
            color_batch: read("crates/j2k-cuda/src/decoder/color_batch.rs"),
            color_batch_host_owners: read("crates/j2k-cuda/src/decoder/color_batch/host_owners.rs"),
            color_store: read("crates/j2k-cuda/src/decoder/color_batch/store.rs"),
            color_store_batch: read("crates/j2k-cuda/src/decoder/color_batch/store/batch.rs"),
            profile: read("crates/j2k-cuda/src/decoder/profile.rs"),
        }
    }
}

#[test]
fn cuda_direct_decode_plan_remains_move_only() {
    let sources = CudaDecoderSources::read();
    assert_pattern_checks(&[
        PatternCheck::new("CUDA direct decode plan", &sources.direct_plan)
            .required(&[
                "mod required_regions;",
                "HostPhaseBudget::new(\"CUDA direct-plan owner graph\")",
                "host_budget.live_bytes()",
                "append_payload_to_shared_with_budget(",
                "account_host_owners(",
                "The plan is move-only because its payload",
                "#[derive(Debug)]\npub(crate) struct CudaHtj2kDecodePlan",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone)]\npub(crate) struct CudaHtj2kDecodePlan",
                "HashMap",
            ]),
        PatternCheck::new(
            "CUDA direct-plan required regions",
            &sources.direct_plan_required_regions,
        )
        .required(&[
            "struct RequiredBandRegions",
            "try_vec_with_capacity(capacity, REQUIRED_REGIONS)?",
            "binary_search_by_key",
            "try_vec_reserve(&mut required.entries, 1, REQUIRED_REGIONS)?",
            "entries.capacity()",
            "direct_plan_actual_capacities_accept_exact_cap_and_reject_one_over",
        ])
        .forbidden(&["HashMap", ".entry("]),
        PatternCheck::new(
            "CUDA color direct-plan owner graph",
            &sources.plan_color_owners,
        )
        .required(&[
            "fn flatten_cuda_color_components(",
            "fn color_owner_graph_budget(",
            "append_payload_to_shared_with_budget(",
            "component.account_host_owners(",
        ]),
    ]);
    assert!(
        sources.direct_plan_required_regions.lines().count() < 250,
        "CUDA direct-plan required-region owner must stay focused"
    );
}

#[test]
fn focused_modules_stay_below_line_ratchets() {
    let sources = CudaDecoderSources::read();
    assert!(
        sources.decoder.lines().count() < 1_500,
        "j2k-cuda/src/decoder.rs must stay below the post-runtime-split line-count ratchet"
    );
    for (module_name, source) in [
        ("api.rs", &sources.api),
        ("plan.rs", &sources.plan),
        ("profile.rs", &sources.profile),
    ] {
        assert!(
            source.lines().count() < 1_800,
            "j2k-cuda/src/decoder/{module_name} must stay below the focused-module line-count ratchet"
        );
    }
    assert!(
        sources.plan_color_owners.lines().count() < 100,
        "j2k-cuda/src/decoder/plan/color_owners.rs must remain a focused owner-accounting leaf"
    );
    for (module_name, source, maximum_lines) in [
        ("resident.rs", &sources.resident, 50),
        (
            "resident/cleanup_dequant.rs",
            &sources.resident_cleanup_dequant,
            325,
        ),
        ("resident/component.rs", &sources.resident_component, 225),
        ("resident/helpers.rs", &sources.resident_helpers, 200),
        ("resident/idwt.rs", &sources.resident_idwt, 350),
        ("resident/routing.rs", &sources.resident_routing, 425),
        ("resident/surface.rs", &sources.resident_surface, 175),
    ] {
        assert!(
            source.lines().count() < maximum_lines,
            "j2k-cuda/src/decoder/{module_name} must stay below its semantic-module line-count ratchet"
        );
    }
    assert!(
        sources.color_batch.lines().count() < 800,
        "j2k-cuda decoder/color_batch.rs must stay below its post-batch-store-split line-count ratchet"
    );
    assert!(
        sources.color_batch_host_owners.lines().count() < 125,
        "j2k-cuda decoder/color_batch/host_owners.rs must remain a focused owner-accounting leaf"
    );
    assert!(
        sources.color_store.lines().count() < 500,
        "j2k-cuda decoder/color_batch/store.rs must stay below its focused-module line-count ratchet"
    );
    assert!(
        sources.color_store_batch.lines().count() < 150,
        "j2k-cuda decoder/color_batch/store/batch.rs must remain a focused preparation leaf"
    );
}

#[test]
fn decoder_host_collections_remain_fallible() {
    let sources = CudaDecoderSources::read();
    let forbidden = [
        "Vec::with_capacity",
        ".collect::<Vec",
        ".collect::<Result<Vec",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("CUDA decoder plan host allocations", &sources.plan)
            .required(&["mod color_owners;"])
            .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA decoder color-plan host allocations",
            &sources.plan_color_owners,
        )
        .required(&["try_vec_with_capacity(", "color_owner_graph_budget("])
        .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA resident component host allocations",
            &sources.resident_component,
        )
        .required(&["try_vec_with_capacity(", "try_collect_results_exact("])
        .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA resident cleanup host allocations",
            &sources.resident_cleanup_dequant,
        )
        .required(&["try_vec_with_capacity("])
        .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA resident IDWT host allocations",
            &sources.resident_idwt,
        )
        .required(&[
            "try_cuda_vec_with_capacity(",
            "try_collect_cuda_results_exact(",
        ])
        .forbidden(&forbidden),
        PatternCheck::new("CUDA color batch host allocations", &sources.color_batch)
            .required(&["try_vec_with_capacity(", "try_collect_results_exact("])
            .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA color owner-graph host allocations",
            &sources.color_batch_host_owners,
        )
        .required(&[
            "HostPhaseBudget::new(",
            "host_budget.try_vec_with_capacity(",
            "host_budget.try_vec_reserve(",
        ])
        .forbidden(&forbidden),
    ]);
}
