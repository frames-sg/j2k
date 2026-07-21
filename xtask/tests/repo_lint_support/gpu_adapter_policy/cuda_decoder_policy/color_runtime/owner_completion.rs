// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

use super::CudaDecoderSources;

pub(super) fn assert_contract(sources: &CudaDecoderSources, finish: &str) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "CUDA color batch host owner graph",
            &sources.color_batch_host_owners,
        )
        .required(&[
            "fn account_colors(",
            "fn color_batch_budget(",
            "fn color_work_budget(",
            "fn account_component_work(",
            "fn append_color_payload_to_shared(",
            "host_budget.try_vec_reserve(",
        ]),
        PatternCheck::new(
            "CUDA color store aggregate host ownership",
            &sources.color_batch_execution,
        )
        .required(&[
            "j2k_store_rgb8_mct_batch_contiguous_device_with_live_host_bytes(",
            "host_budget.live_bytes()",
        ]),
    ]);
    assert_eq!(
        sources
            .color_batch_execution
            .matches("CudaQueuedIdwtBatch::resolve_optional_after_completed_work(")
            .count()
            + finish
                .matches("CudaQueuedIdwtBatch::resolve_optional_after_completed_work(")
                .count(),
        2,
        "single and batch color decode must both resolve queued IDWT ownership"
    );
}
