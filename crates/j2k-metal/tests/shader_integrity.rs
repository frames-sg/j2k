use j2k_test_support::unwired_metal_kernels;

const HOST_SOURCE: &str = concat!(
    include_str!("../src/compute.rs"),
    "\n",
    include_str!("../src/compute/runtime.rs"),
);
const NATIVE_COLOR_BATCH_SOURCE: &str = include_str!("../src/store_native_color_batch.metal");
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
    NATIVE_COLOR_BATCH_SOURCE,
];

#[test]
fn shader_inventory_includes_batch_capable_native_color_store() {
    assert!(SHADER_SOURCES.contains(&NATIVE_COLOR_BATCH_SOURCE));
}

#[test]
fn metal_kernels_are_wired_to_host_pipelines() {
    let unused = unwired_metal_kernels(SHADER_SOURCES.iter().copied(), HOST_SOURCE);

    assert!(
        unused.is_empty(),
        "Metal kernels must be compiled by host pipeline setup or removed: {unused:?}"
    );
}

fn metal_function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing Metal function `{signature}`"));
    let open = source[start..].find('{').map_or_else(
        || panic!("Metal function `{signature}` has no body"),
        |offset| start + offset,
    );
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("Metal function `{signature}` has an unterminated body");
}

#[test]
fn cleanup_only_ht_pipelines_cannot_instantiate_refinement_state() {
    const SOURCE: &str = include_str!("../src/ht_cleanup.metal");

    let common = metal_function_body(SOURCE, "inline void decode_ht_cleanup_common(");
    let strict_pass_rejection = common
        .find("params.number_of_coding_passes > 3u")
        .expect("common HT decode must reject coding-pass counts above three");
    let empty_refinement_normalization = common
        .find("params.refinement_length == 0u")
        .expect("common HT decode must retain empty-refinement normalization");
    assert!(strict_pass_rejection < empty_refinement_normalization);
    assert!(!common.contains("J2K_HT_MAX_SIGMA"));
    assert!(!common.contains("J2K_HT_MAX_PREV_ROW_SIG"));

    let cleanup = metal_function_body(SOURCE, "inline void decode_ht_cleanup_only_impl(");
    assert!(cleanup.contains("decode_ht_cleanup_common("));
    assert!(!cleanup.contains("decode_ht_refinement_impl("));
    assert!(!cleanup.contains("sigma"));
    assert!(!cleanup.contains("prev_row_sig"));

    let refinement = metal_function_body(SOURCE, "inline bool decode_ht_refinement_impl(");
    assert!(refinement.contains("thread ushort sigma[J2K_HT_MAX_SIGMA]"));
    assert!(refinement.contains("thread ushort prev_row_sig[J2K_HT_MAX_PREV_ROW_SIG]"));

    for kernel in [
        "kernel void j2k_decode_ht_cleanup_batched_cleanup_only(",
        "kernel void j2k_decode_ht_cleanup_repeated_batched_cleanup_only(",
    ] {
        let body = metal_function_body(SOURCE, kernel);
        assert!(body.contains("decode_ht_cleanup_only_impl("));
        assert!(!body.contains("decode_ht_cleanup_impl("));
        assert!(!body.contains("decode_ht_refinement_impl("));
    }
}
