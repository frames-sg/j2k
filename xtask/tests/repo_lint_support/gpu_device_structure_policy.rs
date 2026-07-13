// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::*;

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the complete CUDA transcode device ABI and staging ledger is clearest in one audit"
)]
fn cuda_transcode_simt_modules_are_focused_and_staged() {
    let root = repo_root();
    let source_root = root.join("crates/j2k-cuda-runtime/src/cuda_oxide_transcode/simt/src");
    let main = fs::read_to_string(source_root.join("main.rs"))
        .expect("read CUDA Oxide transcode SIMT root");
    let build = fs::read_to_string(root.join("crates/j2k-cuda-runtime/build.rs"))
        .expect("read CUDA runtime build script");
    let modules = [
        ("abi", "abi.rs", 50),
        ("constants", "constants.rs", 75),
        ("dwt97", "dwt97.rs", 550),
        ("exports", "exports.rs", 650),
        ("helpers", "helpers.rs", 100),
        ("quantization", "quantization.rs", 100),
        ("reversible53", "reversible53.rs", 350),
    ];

    assert!(
        main.lines().count() < 40,
        "CUDA Oxide transcode SIMT main.rs must remain a focused module shell"
    );
    assert_eq!(
        main.matches("include!(\"../../../cuda_oxide_simt_prelude.rs\");")
            .count(),
        1,
        "the transcode SIMT root must include the shared prelude exactly once"
    );
    assert!(
        !main.contains("#[cuda_module]"),
        "the authoritative CUDA export surface must live in exports.rs"
    );

    let staging_start = build
        .find("const CUDA_OXIDE_TRANSCODE_EXTRA_SOURCES")
        .expect("transcode extra-source staging declaration");
    let staging_tail = &build[staging_start..];
    let staging_end = staging_tail
        .find("];")
        .expect("end of transcode extra-source staging declaration");
    let staging = &staging_tail[..staging_end];
    assert_eq!(
        staging.matches("\"simt/src/").count(),
        modules.len(),
        "the staged transcode SIMT source list must exactly match the declared modules"
    );
    assert_pattern_checks(&[
        PatternCheck::new("CUDA transcode SIMT staging", &build)
        .required(&[
            "for relative in CUDA_OXIDE_TRANSCODE_EXTRA_SOURCES",
            "for relative in cuda_oxide_extra_sources(source_dir)",
            "source_dir == Path::new(\"src/cuda_oxide_transcode\")",
        ])
        .normalized_required(&[
            "copy_cuda_oxide_file( source_dir, project_dir, Path::new(relative), codec_math_crate_path, );",
        ]),
    ]);

    let mut module_sources = String::new();
    for (module, filename, max_lines) in modules {
        assert!(
            main.contains(&format!("mod {module};")),
            "transcode SIMT root must declare module {module}"
        );
        assert!(
            staging.contains(&format!("\"simt/src/{filename}\"")),
            "transcode SIMT module {filename} must be staged for Linux/PTX builds"
        );
        let source = fs::read_to_string(source_root.join(filename))
            .unwrap_or_else(|error| panic!("read transcode SIMT module {filename}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "transcode SIMT module {filename} exceeded its focused line-count ratchet"
        );
        module_sources.push_str(&source);
    }
    assert!(
        !module_sources.contains("use super::*"),
        "transcode SIMT modules must use explicit imports"
    );

    let exports = fs::read_to_string(source_root.join("exports.rs"))
        .expect("read CUDA Oxide transcode export surface");
    assert_eq!(exports.matches("#[cuda_module]").count(), 1);
    assert_eq!(exports.matches("#[kernel]").count(), 15);
    for entrypoint in [
        "transcode_reversible53_idct",
        "transcode_reversible53_vertical_low",
        "transcode_reversible53_vertical_high",
        "transcode_reversible53_horizontal_low",
        "transcode_reversible53_horizontal_high",
        "transcode_dwt97_idct",
        "transcode_dwt97_row_lift",
        "transcode_dwt97_column_lift",
        "transcode_dwt97_idct_batch",
        "transcode_dwt97_idct_i16_batch",
        "transcode_dwt97_row_lift_batch",
        "transcode_dwt97_row_lift_batch_coop",
        "transcode_dwt97_column_lift_batch",
        "transcode_dwt97_quantize_codeblocks",
        "transcode_dwt97_column_lift_quantize_codeblocks_batch",
    ] {
        assert_eq!(
            exports
                .matches(&format!("pub unsafe fn {entrypoint}("))
                .count(),
            1,
            "CUDA transcode entrypoint {entrypoint} must have one authoritative definition"
        );
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the complete CUDA J2K encode device ABI and staging ledger is clearest in one audit"
)]
fn cuda_j2k_encode_simt_modules_are_focused_and_staged() {
    let root = repo_root();
    let source_root = root.join("crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src");
    let main = fs::read_to_string(source_root.join("main.rs"))
        .expect("read CUDA Oxide J2K encode SIMT root");
    let build = fs::read_to_string(root.join("crates/j2k-cuda-runtime/build.rs"))
        .expect("read CUDA runtime build script");
    let modules = [
        ("abi", "abi.rs", 100),
        ("constants", "constants.rs", 50),
        ("dwt53", "dwt53.rs", 75),
        ("dwt97", "dwt97.rs", 225),
        ("exports", "exports.rs", 500),
        ("helpers", "helpers.rs", 100),
        ("packet_writer", "packet_writer.rs", 200),
        ("packetization", "packetization.rs", 325),
        ("quantization", "quantization.rs", 75),
        ("tag_tree", "tag_tree.rs", 300),
    ];

    assert!(
        main.lines().count() < 40,
        "CUDA Oxide J2K encode SIMT main.rs must remain a focused module shell"
    );
    assert_eq!(
        main.matches("include!(\"../../../cuda_oxide_simt_prelude.rs\");")
            .count(),
        1,
        "the J2K encode SIMT root must include the shared prelude exactly once"
    );
    assert!(
        !main.contains("#[cuda_module]"),
        "the authoritative CUDA J2K encode export surface must live in exports.rs"
    );

    let staging_start = build
        .find("const CUDA_OXIDE_J2K_ENCODE_EXTRA_SOURCES")
        .expect("J2K encode extra-source staging declaration");
    let staging_tail = &build[staging_start..];
    let staging_end = staging_tail
        .find("];")
        .expect("end of J2K encode extra-source staging declaration");
    let staging = &staging_tail[..staging_end];
    assert_eq!(
        staging.matches("\"simt/src/").count(),
        modules.len(),
        "the staged J2K encode SIMT source list must exactly match the declared modules"
    );
    assert_pattern_checks(&[
        PatternCheck::new("CUDA J2K encode SIMT staging", &build)
        .required(&[
            "for relative in CUDA_OXIDE_J2K_ENCODE_EXTRA_SOURCES",
            "for relative in cuda_oxide_extra_sources(source_dir)",
            "source_dir == Path::new(\"src/cuda_oxide_j2k_encode\")",
        ])
        .normalized_required(&[
            "copy_cuda_oxide_file( source_dir, project_dir, Path::new(relative), codec_math_crate_path, );",
        ]),
    ]);

    let mut module_sources = String::new();
    for (module, filename, max_lines) in modules {
        assert!(
            main.contains(&format!("mod {module};")),
            "J2K encode SIMT root must declare module {module}"
        );
        assert!(
            staging.contains(&format!("\"simt/src/{filename}\"")),
            "J2K encode SIMT module {filename} must be staged for Linux/PTX builds"
        );
        let source = fs::read_to_string(source_root.join(filename))
            .unwrap_or_else(|error| panic!("read J2K encode SIMT module {filename}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "J2K encode SIMT module {filename} exceeded its focused line-count ratchet"
        );
        module_sources.push_str(&source);
    }
    assert!(
        !module_sources.contains("use super::*"),
        "J2K encode SIMT modules must use explicit imports"
    );
    assert!(
        !module_sources.contains("include!("),
        "only the J2K encode SIMT root may include the shared device prelude"
    );

    let exports = fs::read_to_string(source_root.join("exports.rs"))
        .expect("read CUDA Oxide J2K encode export surface");
    assert_eq!(exports.matches("#[cuda_module]").count(), 1);
    assert_eq!(exports.matches("#[kernel]").count(), 12);
    for entrypoint in [
        "j2k_deinterleave_to_f32",
        "j2k_deinterleave_strided_to_f32",
        "j2k_forward_rct",
        "j2k_forward_ict",
        "j2k_forward_dwt53_horizontal",
        "j2k_forward_dwt53_vertical",
        "j2k_forward_dwt97_horizontal",
        "j2k_forward_dwt97_vertical",
        "j2k_quantize_subband",
        "j2k_quantize_subband_strided",
        "j2k_htj2k_compact_codeblocks",
        "j2k_htj2k_packetize_cleanup",
    ] {
        assert_eq!(
            exports
                .matches(&format!("pub unsafe fn {entrypoint}("))
                .count(),
            1,
            "CUDA J2K encode entrypoint {entrypoint} must have one authoritative definition"
        );
    }
}

#[test]
fn metal_tier1_encode_keeps_test_support_out_of_production_source() {
    let root = repo_root();
    let source_root = root.join("crates/j2k-metal/src/compute/tier1_encode");
    let production = fs::read_to_string(root.join("crates/j2k-metal/src/compute/tier1_encode.rs"))
        .expect("read Metal Tier-1 encode production source");
    let test_support =
        fs::read_to_string(source_root.join("test_support.rs")).expect("read test-support shell");

    assert!(
        production.lines().count() < 1_300,
        "Metal Tier-1 production source exceeded its focused line-count ratchet"
    );
    assert_pattern_checks(&[
        PatternCheck::new("Metal Tier-1 production/test seam", &production)
            .required(&["#[cfg(test)]\nmod test_support;"])
            .forbidden(&[
                "fn encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test(",
                "struct ClassicTier1MsbBitWriter",
                "fn pack_classic_split_mq_raw_tokens_for_test(",
            ]),
        PatternCheck::new("Metal Tier-1 test-support shell", &test_support).required(&[
            "mod gpu_pack;",
            "mod ordered_pack;",
            "mod split_cpu_pack;",
        ]),
    ]);
    assert!(
        test_support.lines().count() < 50,
        "Metal Tier-1 test-support root must remain a focused module shell"
    );

    for (filename, max_lines, required_symbol) in [
        (
            "test_support/gpu_pack.rs",
            400,
            "fn encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test(",
        ),
        (
            "test_support/ordered_pack.rs",
            325,
            "fn encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test(",
        ),
        (
            "test_support/split_cpu_pack.rs",
            400,
            "fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test(",
        ),
    ] {
        let source = fs::read_to_string(source_root.join(filename))
            .unwrap_or_else(|error| panic!("read Metal Tier-1 {filename}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "Metal Tier-1 {filename} exceeded its focused line-count ratchet"
        );
        assert!(
            source.contains(required_symbol),
            "Metal Tier-1 {filename} must own {required_symbol}"
        );
    }
}
