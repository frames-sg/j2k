// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, CudaDecoderSources, PatternCheck};

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
        PatternCheck::new(
            "CUDA color batch preparation host allocations",
            &sources.color_batch_execution_preparation,
        )
        .required(&["try_vec_with_capacity("])
        .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA color batch execution host allocations",
            &sources.color_batch_execution_execution,
        )
        .required(&["try_vec_with_capacity("])
        .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA color batch store host allocations",
            &sources.color_batch_execution_completion_batch_store,
        )
        .required(&["try_vec_with_capacity(", "try_collect_results_exact("])
        .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA color batch fallback host allocations",
            &sources.color_batch_execution_completion_fallback,
        )
        .required(&["try_vec_with_capacity("])
        .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA single-color host allocations",
            &sources.color_batch_single,
        )
        .required(&["try_vec_with_capacity("])
        .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA native-color preparation host allocations",
            &sources.color_batch_native_prepare,
        )
        .required(&["try_vec_with_capacity("])
        .forbidden(&forbidden),
        PatternCheck::new(
            "CUDA native-color lifecycle host allocations",
            &sources.color_batch_native_lifecycle,
        )
        .required(&["try_vec_with_capacity("])
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
