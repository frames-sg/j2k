// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{assert_pattern_checks, read_repo, read_runtime, PatternCheck};

mod ordering;

struct JpegDecodePreflightSources {
    allocation: String,
    core_allocation: String,
    decode_api: String,
    decode_launch: String,
    diagnostics_api: String,
    diagnostics_execution: String,
    diagnostics_allocation: String,
}

impl JpegDecodePreflightSources {
    fn read() -> Self {
        Self {
            allocation: [
                read_runtime("allocation.rs"),
                read_runtime("allocation/phase.rs"),
                read_runtime("allocation/tests.rs"),
            ]
            .join("\n"),
            core_allocation: read_repo("crates/j2k-core/src/host_allocation.rs"),
            decode_api: read_runtime("jpeg/decode.rs"),
            decode_launch: read_runtime("jpeg/decode_launch.rs"),
            diagnostics_api: read_runtime("jpeg/diagnostics.rs"),
            diagnostics_execution: read_runtime("jpeg/diagnostics_execution.rs"),
            diagnostics_allocation: read_runtime("jpeg/diagnostics_allocation.rs"),
        }
    }
}

#[test]
fn jpeg_decode_preflight_and_fallible_statuses_precede_driver_work() {
    let sources = JpegDecodePreflightSources::read();
    assert_shared_allocation_contracts(&sources);
    assert_decode_allocation_contracts(&sources);
    assert_diagnostic_allocation_contracts(&sources);
    ordering::assert_preflight_ordering(&sources);
}

fn assert_shared_allocation_contracts(sources: &JpegDecodePreflightSources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "shared fallible host vector allocation",
            &sources.core_allocation,
        )
        .required(&[
            "pub struct HostAllocationError",
            "pub fn try_host_vec_with_capacity<T>(",
            ".try_reserve_exact(capacity)",
            "pub fn try_host_vec_filled<T: Clone>(",
            "pub fn try_host_vec_from_slice<T: Copy>(",
            "impossible_capacity_reports_saturated_requested_bytes",
        ])
        .forbidden(&["Vec::with_capacity", "vec!["]),
        PatternCheck::new("CUDA allocation error-domain mapping", &sources.allocation).required(&[
            "fn try_vec_defaulted<T: Clone + Default>(",
            "try_vec_filled(len, T::default())",
            "try_host_vec_with_capacity(capacity).map_err(cuda_allocation_error)",
            "CudaError::HostAllocationFailed {",
            "logically_oversized_capacities_are_rejected_before_allocation",
        ]),
    ]);
}

fn assert_decode_allocation_contracts(sources: &JpegDecodePreflightSources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "CUDA JPEG decode fallible status preflight",
            &sources.decode_api,
        )
        .required(&[
            "fn allocate_decode_statuses_with_cap(",
            "HostPhaseBudget::with_cap(\"CUDA JPEG decode status workspace\", cap)",
            "host_budget.account_bytes(external_live_bytes)?",
            "host_budget.try_vec_filled(count, CudaJpegDecodeStatus::default())",
            "status_workspace_external_live_boundary_is_exact",
        ])
        .forbidden(&["Vec::with_capacity", "vec![CudaJpegDecodeStatus"]),
        PatternCheck::new(
            "CUDA JPEG decode common output initialization",
            &sources.decode_launch,
        )
        .required(&[
            "Both safe decode entrypoints converge here",
            "self.memset_d8(output, 0, validated.output_len)?;",
        ])
        .forbidden(&["Vec::with_capacity", "vec![CudaJpegDecodeStatus"]),
    ]);
    assert_eq!(
        sources
            .decode_api
            .matches("let statuses = allocate_decode_statuses_with_cap(")
            .count(),
        2
    );
}

fn assert_diagnostic_allocation_contracts(sources: &JpegDecodePreflightSources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "CUDA JPEG diagnostic status allocation",
            &sources.diagnostics_allocation,
        )
        .required(&[
            "fn allocate_diagnostic_workspaces_with_cap(",
            "HostPhaseBudget::with_cap(\"CUDA JPEG entropy diagnostics\", cap)",
            "host_budget.account_bytes(external_live_bytes)?",
            "host_budget.account_bytes(retained_page_locked_bytes)?",
            "host_budget.try_vec_filled(state_count, CudaJpegEntropySyncState::default())",
            "host_budget.try_vec_filled(overflow_count, CudaJpegEntropyOverflowState::default())",
            "diagnostic_state_and_overflow_external_live_boundary_is_exact",
            "diagnostic_budget_counts_new_and_reused_larger_checkout_exactly",
        ])
        .forbidden(&["Vec::with_capacity", "vec![CudaJpegEntropy"]),
        PatternCheck::new(
            "CUDA JPEG diagnostic preflight handoff",
            &sources.diagnostics_execution,
        )
        .required(&[
            "let params = validate_jpeg_entropy_chunk_plan(plan, subsequences)?;",
            "let workspaces = allocate_diagnostic_workspaces_with_cap(",
            "if let Err(error) = self.inner.set_current()",
        ]),
        PatternCheck::new(
            "CUDA JPEG diagnostic public preflight",
            &sources.diagnostics_api,
        )
        .required(&[
            "pinned_upload.ensure_for_context(self)?;",
            "plan.config.validate()?;",
            "subsequence_count_for_entropy_bytes(plan.entropy_bytes.len())?",
            "self.diagnose_jpeg_420_entropy_self_sync_nonempty(",
        ]),
    ]);
    assert_eq!(
        sources
            .diagnostics_execution
            .matches("let workspaces = allocate_diagnostic_workspaces_with_cap(")
            .count(),
        1
    );
}
