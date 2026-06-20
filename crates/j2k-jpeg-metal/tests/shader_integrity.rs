use j2k_test_support::unwired_metal_kernels;

const SHADER_SOURCE: &str = include_str!("../src/shaders.metal");
const COMPUTE_SOURCE: &str = include_str!("../src/compute.rs");

#[test]
fn metal_kernels_are_wired_to_host_pipelines() {
    let unused = unwired_metal_kernels([SHADER_SOURCE], COMPUTE_SOURCE);

    assert!(
        unused.is_empty(),
        "Metal kernels must be compiled by host pipeline setup or removed: {unused:?}"
    );
}
