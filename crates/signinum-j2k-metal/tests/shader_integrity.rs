use signinum_test_support::unwired_metal_kernels;

const COMPUTE_SOURCE: &str = include_str!("../src/compute.rs");
const SHADER_SOURCES: &[&str] = &[
    COMPUTE_SOURCE,
    include_str!("../src/classic.metal"),
    include_str!("../src/encode_bitstream.metal"),
    include_str!("../src/fdwt.metal"),
    include_str!("../src/ht_cleanup.metal"),
    include_str!("../src/idwt.metal"),
    include_str!("../src/mct.metal"),
    include_str!("../src/store.metal"),
];

#[test]
fn metal_kernels_are_wired_to_host_pipelines() {
    let unused = unwired_metal_kernels(SHADER_SOURCES.iter().copied(), COMPUTE_SOURCE);

    assert!(
        unused.is_empty(),
        "Metal kernels must be compiled by host pipeline setup or removed: {unused:?}"
    );
}
