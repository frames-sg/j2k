use j2k_test_support::unwired_metal_kernels;

const HOST_SOURCE: &str = concat!(
    include_str!("../src/compute.rs"),
    "\n",
    include_str!("../src/compute/runtime.rs"),
);
const SHADER_SOURCES: &[&str] = &[
    HOST_SOURCE,
    include_str!("../src/classic.metal"),
    include_str!("../src/encode_bitstream.metal"),
    include_str!("../src/encode_bitstream_shared.metal"),
    include_str!("../src/encode_bitstream_classic_core.metal"),
    include_str!("../src/encode_bitstream_classic_tokens.metal"),
    include_str!("../src/encode_bitstream_classic_symbol_plan.metal"),
    include_str!("../src/encode_bitstream_classic_kernels.metal"),
    include_str!("../src/encode_bitstream_ht.metal"),
    include_str!("../src/encode_bitstream_packetize.metal"),
    include_str!("../src/fdwt.metal"),
    include_str!("../src/ht_cleanup.metal"),
    include_str!("../src/idwt.metal"),
    include_str!("../src/mct.metal"),
    include_str!("../src/quantize.metal"),
    include_str!("../src/store.metal"),
];

#[test]
fn metal_kernels_are_wired_to_host_pipelines() {
    let unused = unwired_metal_kernels(SHADER_SOURCES.iter().copied(), HOST_SOURCE);

    assert!(
        unused.is_empty(),
        "Metal kernels must be compiled by host pipeline setup or removed: {unused:?}"
    );
}
