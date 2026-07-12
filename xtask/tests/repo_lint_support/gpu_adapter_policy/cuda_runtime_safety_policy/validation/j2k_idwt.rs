// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, read_runtime, PatternCheck};

#[test]
fn cuda_j2k_idwt_jobs_require_full_validation() {
    let idwt = read_runtime("j2k_decode/idwt.rs");
    let idwt_sequence = read_runtime("j2k_decode/idwt/sequence.rs");
    let idwt_jobs = read_runtime("j2k_decode/idwt/job_validation.rs");
    let idwt_job_tests = read_runtime("j2k_decode/idwt/job_validation/tests.rs");
    let idwt_preflight = read_runtime("j2k_decode/idwt/preflight.rs");
    let decode_validation = read_runtime("j2k_decode/validation.rs");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA J2K IDWT full job validation", &idwt_jobs).required(&[
            "fn validate_idwt_job(",
            "fn validate_idwt_target(",
            "checked_rect_dimensions(\"output\"",
            "idwt_band_extent(job.rect.x0, job.rect.x1, low_x)",
            "exceeds the CUDA u32 indexing ABI",
            "exceeds the CUDA u32 iteration ABI",
            "buffer is too small: required",
        ]),
        PatternCheck::new("CUDA J2K IDWT validation integration", &idwt).required(&[
            "pub(super) mod job_validation;",
            "mod preflight;",
            "mod sequence;",
            "use job_validation::validate_idwt_job;",
            "use preflight::validate_idwt_single_request;",
            "let validated = validate_idwt_single_request(self, [ll, hl, lh, hh], job)?;",
            "let output = self.allocate(validated.output_bytes)?;",
            "let output = pool.take(validated.output_bytes)?;",
        ]),
        PatternCheck::new(
            "CUDA J2K IDWT sequence validation integration",
            &idwt_sequence,
        )
        .required(&[
            "j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes(",
            "HostPhaseBudget::with_live_bytes(",
            "host_budget.try_vec_with_capacity(total_target_count)?",
            "host_budget.try_vec_with_capacity(target_batches.len())?",
            "append_j2k_idwt_multi_kernel_jobs(targets, &mut all_jobs)?",
            "plan_idwt_batch_launch(&all_jobs[start..])?",
        ]),
        PatternCheck::new("CUDA J2K IDWT allocation-free preflight", &idwt_preflight).required(&[
            "fn validate_idwt_single_request(",
            "let validated = validate_idwt_job(bands, None, job)?;",
            "validate_idwt_single_launch(validated.width, validated.height)?;",
            "fn j2k_inverse_dwt_single_output_bytes(",
        ]),
        PatternCheck::new(
            "CUDA J2K IDWT batch validation integration",
            &decode_validation,
        )
        .required(&[
            "job_validation::validate_idwt_target(target)?",
            "try_vec_with_capacity(targets.len())?",
            "append_j2k_idwt_multi_kernel_jobs(targets, &mut kernel_jobs)?",
        ]),
        PatternCheck::new("CUDA J2K IDWT adversarial validation", &idwt_job_tests).required(&[
            "full_validator_rejects_each_undersized_input_band",
            "full_validator_accepts_valid_two_by_eight_and_eight_by_two_jobs",
            "full_validator_accepts_one_by_n_jobs_for_even_and_odd_origins",
            "full_validator_rejects_linear_index_overflow",
        ]),
    ]);

    assert_eq!(
        idwt.matches("validate_idwt_job(").count()
            + idwt_preflight.matches("validate_idwt_job(").count(),
        2,
        "single preflight and pooled-single IDWT paths must use the full job validator"
    );
    assert_eq!(
        idwt.matches("j2k_idwt_multi_kernel_jobs(").count()
            + idwt_sequence
                .matches("append_j2k_idwt_multi_kernel_jobs(")
                .count(),
        3,
        "synchronous batch, queued batch, and queued sequence IDWT paths must validate jobs"
    );
}
