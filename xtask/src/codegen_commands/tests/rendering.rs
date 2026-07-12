// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    codec_math_codegen, compact_f32, finalize_text_snapshot,
    render_codec_math_dwt97_metal_fragment, render_codec_math_dwt97_rust_fragment, stable_api,
};

#[test]
fn text_snapshots_end_with_exactly_one_newline() {
    assert_eq!(finalize_text_snapshot("content\n\n"), "content\n");
    assert_eq!(finalize_text_snapshot("content"), "content\n");
    assert_eq!(finalize_text_snapshot(""), "");
}

#[test]
fn codec_math_fragments_render_from_one_ordered_constant_source() {
    let metal = render_codec_math_dwt97_metal_fragment();
    let rust = render_codec_math_dwt97_rust_fragment();

    assert!(metal.starts_with("// Generated from crates/j2k-codec-math/src/dwt.rs.\n"));
    assert!(rust.starts_with("// Generated from crates/j2k-codec-math/src/dwt.rs.\n"));
    for name in [
        "DWT97_ALPHA",
        "DWT97_BETA",
        "DWT97_GAMMA",
        "DWT97_DELTA",
        "DWT97_KAPPA",
        "IDWT97_NEG_ALPHA",
        "IDWT97_NEG_BETA",
        "IDWT97_NEG_GAMMA",
        "IDWT97_NEG_DELTA",
    ] {
        assert!(metal.contains(name), "Metal fragment omitted {name}");
        assert!(rust.contains(name), "Rust fragment omitted {name}");
    }
    assert!(metal.contains("1.0f / CODEC_MATH_DWT97_KAPPA"));
    assert!(rust.contains("1.0 / CODEC_MATH_DWT97_KAPPA"));
    assert!(metal.ends_with('\n'));
    assert!(rust.ends_with('\n'));
    assert_eq!(metal.lines().count(), rust.lines().count());
}

#[test]
fn compact_float_rendering_is_round_trip_stable() {
    for value in [0.0_f32, -0.0, 1.25, f32::MIN_POSITIVE, f32::MAX] {
        let rendered = compact_f32(value);
        let parsed = rendered.parse::<f32>().expect("rendered f32");
        assert_eq!(parsed.to_bits(), value.to_bits());
    }
}

#[test]
fn codegen_argument_parsers_fail_before_generation_or_writes() {
    let error = codec_math_codegen(["--unknown".to_string()].into_iter())
        .expect_err("unknown codec-math argument");
    assert!(error.contains("unknown codec-math-codegen argument"));
    let error =
        stable_api(["--unknown".to_string()].into_iter()).expect_err("unknown stable API argument");
    assert!(error.contains("unknown stable-api argument"));

    assert_eq!(
        codec_math_codegen(["--help".to_string()].into_iter()),
        Ok(())
    );
    assert_eq!(stable_api(["--help".to_string()].into_iter()), Ok(()));
}
