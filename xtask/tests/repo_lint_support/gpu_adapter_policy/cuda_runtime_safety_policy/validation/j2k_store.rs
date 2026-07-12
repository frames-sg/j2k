// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, read_runtime, PatternCheck};

#[test]
fn cuda_j2k_store_requires_validated_initialized_destinations() {
    let store = read_runtime("j2k_decode/store.rs");
    let store_batch = read_runtime("j2k_decode/store/batch.rs");
    let destination = read_runtime("j2k_decode/store/destination.rs");
    let store_validation = read_runtime("j2k_decode/store/validation.rs");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA J2K store destination safety", &destination).required(&[
            "fn validate_store_destination(",
            "checked_add(copy_width)",
            "checked_add(copy_height)",
            "copy_width.checked_mul(copy_height)",
            "exceeds the CUDA u32 kernel ABI",
            "fn zero_unwritten_store_output(",
            "context.memset_d8(output, 0, output_bytes)?;",
        ]),
        PatternCheck::new("CUDA J2K store source ABI safety", &store_validation).required(&[
            "fn validate_store_plane_layout(",
            "last_sample > u64::from(u32::MAX)",
            "exceeds the CUDA u32 kernel ABI",
            "required_bytes > plane_bytes",
        ]),
        PatternCheck::new("CUDA J2K store destination integration", &store)
            .required(&[
                "mod batch;",
                "mod destination;",
                "zero_unwritten_store_output",
                "if zero_fill_enqueued {",
                "self.synchronize()?;",
            ])
            .forbidden(&["let dst_end ="]),
        PatternCheck::new("CUDA J2K store batch validation integration", &store_batch)
            .required(&[
                "fn validate_rgb8_mct_targets(",
                "validate_rgb8_mct_target_context(context, targets)?;",
                "validate_store_destination(",
                "try_vec_with_capacity(targets.len())?",
                "let plan = validate_rgb8_mct_targets(self, targets, 0)?;",
                "let plan = validate_rgb8_mct_targets(self, targets, live_host_bytes)?;",
                "zero_unwritten_store_output(",
            ])
            .forbidden(&[
                "Vec::with_capacity",
                ".collect::<Result<Vec<_>, CudaError>>()",
            ]),
    ]);

    assert_eq!(
        store.matches("validate_store_destination(").count(),
        5,
        "each single-output CUDA store path must validate its destination"
    );
    assert_eq!(
        store_batch.matches("validate_store_destination(").count(),
        1,
        "both batch CUDA store paths must share one destination planner"
    );
    assert_eq!(
        store_batch
            .matches("let plan = validate_rgb8_mct_targets(self, targets,")
            .count(),
        2,
        "separate-allocation and contiguous batch stores must use the shared planner"
    );
    assert_eq!(
        store.matches("zero_unwritten_store_output(").count()
            + store_batch.matches("zero_unwritten_store_output(").count(),
        12,
        "every active and zero-copy CUDA store outcome must initialize unwritten output"
    );
}
