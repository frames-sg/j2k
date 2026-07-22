// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn resident_cuda_encode_resources_are_session_owned_and_context_bound() {
    let root = repo_root();
    let runtime_context = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/context.rs"))
        .expect("read CUDA context source");
    let session = fs::read_to_string(root.join("crates/j2k-cuda/src/session.rs"))
        .expect("read CUDA session source");
    let helper = fs::read_to_string(root.join("crates/j2k-cuda/src/session/encode_resources.rs"))
        .expect("read CUDA session encode-resource helper");
    let encode = fs::read_to_string(root.join("crates/j2k-cuda/src/encode.rs"))
        .expect("read CUDA encode source");
    let encode_tests = [
        fs::read_to_string(root.join("crates/j2k-cuda/src/encode/tests/resident.rs"))
            .expect("read CUDA resident encode tests"),
        fs::read_to_string(root.join("crates/j2k-cuda/src/encode/tests/resident_session.rs"))
            .expect("read CUDA resident session tests"),
    ]
    .concat();
    let helper_production = helper
        .split("#[cfg(test)]")
        .next()
        .expect("helper production prefix");
    let encode_production = encode
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("encode production prefix");

    assert_pattern_checks(&[
        PatternCheck::new("minimum CUDA context identity API", &runtime_context).required(&[
            "pub fn is_same_context(&self, other: &Self) -> bool",
            "self.inner.context == other.inner.context",
            "#[doc(hidden)]",
        ]),
        PatternCheck::new("session-owned HTJ2K encode resources", &session).required(&[
            "#[derive(Clone, Default)]",
            "htj2k_encode_resources: Option<Arc<CudaHtj2kEncodeResources>>",
            "pub(crate) fn htj2k_encode_resources(",
            "get_or_try_init_context_bound(",
            "Error::UnsupportedCudaRequest",
            "J2K CUDA encode tile belongs to a different context than the session",
            ".upload_htj2k_encode_resources(crate::encode::cuda_htj2k_encode_tables())",
            "htj2k_encode_resources_cached",
        ]),
        PatternCheck::new("context-bound resource helper", helper_production)
            .required(&[
                "if !same_context(context, requested_context)",
                "return Err(mismatch_error());",
                "if let Some(resource) = cached_resource.as_ref()",
                "Arc::clone(resource)",
                "initialize(requested_context)?",
            ])
            .forbidden(&["expect(", "unwrap("]),
        PatternCheck::new("resident encode session cache use", encode_production)
            .required(&[
                "encode_lossless_cuda_tile_with_report(tile, *options, session)",
                "session.htj2k_encode_resources(&context)?",
                "resources: Arc<CudaHtj2kEncodeResources>",
                "CudaEncodedJ2kMetadata::from_host_encoded(&host_outcome.encoded)",
                ".upload(&host_outcome.encoded.codestream)",
                "drop(host_encoded);",
            ])
            .forbidden(&[
                ".upload_htj2k_encode_resources(",
                ".upload_pinned(&host_outcome.encoded.codestream)",
            ]),
        PatternCheck::new("resident encode resource behavior coverage", &encode_tests).required(&[
            "resident_encode_binds_external_context_and_clones_reuse_resources_when_required",
            "cuda_lossless_buffer_batch_encode_returns_resident_codestreams_in_order_when_runtime_required",
            "resident_encode_rejects_session_context_mismatch_before_resource_upload_when_required",
            "htj2k_encode_resource_uploads_for_test()",
        ]),
        PatternCheck::new("driver-independent context cache coverage", &helper).required(&[
            "compatible_context_reuses_one_shared_resource",
            "mismatched_context_is_rejected_before_resource_initialization",
            "failed_initialization_keeps_binding_and_allows_compatible_retry",
        ]),
    ]);

    assert_eq!(
        encode_production
            .matches("encode_lossless_cuda_tile_with_report(tile, *options, session)")
            .count(),
        2,
        "single and batch resident encode must both pass the owning session"
    );
    let mismatch = helper_production
        .find("if !same_context(context, requested_context)")
        .expect("mismatch guard");
    let cache = helper_production
        .find("if let Some(resource) = cached_resource.as_ref()")
        .expect("cache lookup");
    let initialize = helper_production
        .find("initialize(requested_context)?")
        .expect("resource initializer");
    assert!(
        mismatch < cache && cache < initialize,
        "context mismatch must fail before cache reuse or resource initialization"
    );
}
