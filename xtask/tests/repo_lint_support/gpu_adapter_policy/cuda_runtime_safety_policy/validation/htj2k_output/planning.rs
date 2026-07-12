// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::read_runtime;
use super::sources::Htj2kOutputSources;

#[test]
fn htj2k_output_planning_precedes_allocation_and_initializes_gaps() {
    let decode = Htj2kOutputSources::read().decode;
    let output_regions = read_runtime("htj2k_decode/output_regions.rs");
    assert_eq!(
        output_regions
            .matches("validate_disjoint_htj2k_job_outputs_with_live_bytes(")
            .count(),
        2,
        "the shared HTJ2K output planner must own disjoint-region validation"
    );
    assert_eq!(
        decode.matches("validate_htj2k_output_layout(").count()
            + decode
                .matches("validate_htj2k_output_layout_with_live_bytes(")
                .count(),
        4,
        "allocation, single and multi-target kernel planning, and empty-output checks must share validated or live-owner-aware HTJ2K coverage planning"
    );
    let allocation = &decode[decode
        .find("pub fn allocate_htj2k_codeblock_coefficients_with_pool(")
        .expect("find pooled HTJ2K coefficient allocator")..];
    let validation = allocation
        .find("validate_htj2k_output_layout(jobs, output_words)?")
        .expect("pooled HTJ2K allocation must validate its complete output layout");
    let context_binding = allocation
        .find("self.inner.set_current()?")
        .expect("pooled HTJ2K allocation must bind its context");
    let pool_take = allocation
        .find("pool.take(output_layout.output_bytes)?")
        .expect("pooled HTJ2K allocation must use the validated byte count");
    assert!(
        validation < context_binding && context_binding < pool_take,
        "pooled HTJ2K output validation must precede context binding and allocation"
    );
    let zero_fill = allocation
        .find("self.memset_d32(coefficient_buffer, 0, output_words)?")
        .expect("pooled HTJ2K partial output must be initialized");
    let completion = allocation
        .find("self.synchronize()?")
        .expect("pooled HTJ2K zero fill must establish completion");
    assert!(
        pool_take < zero_fill && zero_fill < completion,
        "pooled HTJ2K zero fill must complete before the checkout can be returned or recycled"
    );
}
