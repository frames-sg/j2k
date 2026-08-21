use j2k_test_support::unwired_metal_kernels;

const SHADER_SOURCE: &str = include_str!("../src/shaders.metal");
const STAGED_ENCODE_SOURCE: &str = include_str!("../src/shaders_encode_staged.metal");
const COMPUTE_SOURCE: &str = include_str!("../src/compute/pipeline_registry.rs");

#[test]
fn metal_kernels_are_wired_to_host_pipelines() {
    let unused = unwired_metal_kernels([SHADER_SOURCE], COMPUTE_SOURCE);

    assert!(
        unused.is_empty(),
        "Metal kernels must be compiled by host pipeline setup or removed: {unused:?}"
    );
}

#[test]
fn promoted_staged_baseline_encode_kernels_are_wired() {
    for kernel in [
        "jpeg_encode_baseline_precompute_batch",
        "jpeg_encode_baseline_entropy_from_coeffs_batch",
    ] {
        assert!(
            STAGED_ENCODE_SOURCE.contains(kernel),
            "missing staged kernel {kernel}"
        );
        assert!(
            COMPUTE_SOURCE.contains(kernel),
            "unwired staged kernel {kernel}"
        );
    }
}
