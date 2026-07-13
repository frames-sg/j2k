// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

const MODULE_LIMITS: &[(&str, usize)] = &[
    ("batch.rs", 420),
    ("device_resident.rs", 280),
    ("host_fallback.rs", 190),
    ("resident_hybrid.rs", 200),
    ("resident_plan.rs", 110),
    ("resident_prepare.rs", 130),
    ("resident_submit.rs", 350),
    ("resident_validation.rs", 130),
    ("resident_wait.rs", 310),
    ("routing.rs", 130),
    ("unavailable.rs", 70),
];

const FUNCTION_OWNERS: &[(&str, &[&str])] = &[
    (
        "batch.rs",
        &[
            "encode_lossless_tiles_with_report",
            "encode_lossless_owned_tiles_with_report",
            "submit_lossless_tiles_to_metal_buffer_batch",
            "try_submit_resident_lossless_tiles_to_metal_buffer_batch",
            "submit_lossless_tiles",
        ],
    ),
    (
        "device_resident.rs",
        &[
            "try_encode_lossless_tile_device_resident_with_report",
            "try_encode_lossless_tile_device_resident_to_metal_buffer_with_report",
            "encode_lossless_tile_to_metal_buffer_with_report",
        ],
    ),
    ("host_fallback.rs", &["encode_lossless_tile_with_report"]),
    (
        "resident_hybrid.rs",
        &[
            "encode_resident_ht_tile_body_with_cpu_packetization",
            "lossless_device_coefficient_count",
            "should_try_resident_lossless_ht_cpu_packetization",
        ],
    ),
    (
        "resident_plan.rs",
        &["plan_resident_lossless_buffer_encode"],
    ),
    (
        "resident_prepare.rs",
        &["prepare_planned_resident_lossless_tiles_batch"],
    ),
    (
        "resident_submit.rs",
        &[
            "submit_planned_resident_lossless_tiles",
            "submit_planned_resident_lossless_tiles_chunked",
            "duration_share",
        ],
    ),
    (
        "resident_validation.rs",
        &[
            "validate_lossless_roundtrip_on_metal_tile_with_session",
            "validate_lossless_roundtrip_on_metal_region_with_session",
        ],
    ),
    (
        "resident_wait.rs",
        &[
            "wait_submitted_resident_lossless_buffer_encode_batch",
            "wait_submitted_resident_lossless_buffer_encode_batch_once",
            "finished_resident_lossless_buffer_encode",
            "validate_finished_resident_lossless_buffer_encode",
        ],
    ),
    (
        "routing.rs",
        &[
            "should_try_resident_lossless_host_encode",
            "should_try_resident_lossless_host_encode_for_tiles",
            "should_try_auto_resident_lossless_host_encode",
            "should_try_auto_resident_lossless_host_format",
            "host_output_encode_options",
            "copy_padded_metal_buffer_from_bytes",
        ],
    ),
];

fn source_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/encode")
}

fn read_source(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn assert_in_order(source: &str, needles: &[&str]) {
    let mut remaining = source;
    for needle in needles {
        let index = remaining
            .find(needle)
            .unwrap_or_else(|| panic!("missing ordered operation: {needle}"));
        remaining = &remaining[index + needle.len()..];
    }
}

#[test]
fn encode_root_remains_a_small_explicit_facade() {
    let root = read_source(&source_root().with_extension("rs"));
    assert!(
        root.lines().count() <= 220,
        "encode.rs facade grew too large"
    );
    assert!(
        !root.contains("pub fn "),
        "implementations belong in modules"
    );
    for (module, _) in MODULE_LIMITS {
        assert!(
            root.contains(&format!("mod {};", module.trim_end_matches(".rs"))),
            "encode.rs does not declare {module}"
        );
    }
}

#[test]
fn encode_modules_stay_focused_and_explicit() {
    let root = source_root();
    for (module, max_lines) in MODULE_LIMITS {
        let source = read_source(&root.join(module));
        assert!(
            source.lines().count() <= *max_lines,
            "{module} exceeds its {max_lines}-line structural budget"
        );
        assert!(source.starts_with("// SPDX-License-Identifier:"));
        assert!(
            !source.contains("include!("),
            "{module} hides an include split"
        );
        assert!(!source.contains("::*"), "{module} uses a wildcard import");
        assert!(!source.contains("#[allow("), "{module} suppresses a lint");
    }
}

#[test]
fn encode_responsibilities_have_single_module_owners() {
    let root = source_root();
    let all_sources = MODULE_LIMITS
        .iter()
        .map(|(module, _)| read_source(&root.join(module)))
        .collect::<Vec<_>>();
    for (owner, functions) in FUNCTION_OWNERS {
        let owner_source = read_source(&root.join(owner));
        for function in *functions {
            let signature = format!("fn {function}(");
            assert_eq!(
                owner_source.matches(&signature).count(),
                1,
                "{function} owner"
            );
            assert_eq!(
                all_sources
                    .iter()
                    .map(|source| source.matches(&signature).count())
                    .sum::<usize>(),
                1,
                "{function} must not be duplicated"
            );
        }
    }
    let routing = read_source(&root.join("routing.rs"));
    assert_eq!(
        routing
            .matches("const AUTO_HIGH_THROUGHPUT_RESIDENT_HOST_OUTPUT_RGB8_MIN_PIXELS")
            .count(),
        1
    );
    let hybrid = read_source(&root.join("resident_hybrid.rs"));
    assert_eq!(hybrid.matches("struct ResidentHybridHtTileBody").count(), 1);
    let prepare = read_source(&root.join("resident_prepare.rs"));
    assert_eq!(
        prepare
            .matches("struct PreparedResidentLosslessBatchItem")
            .count(),
        1
    );
}

#[test]
fn public_cfg_docs_and_metal_command_order_are_pinned() {
    let root = source_root();
    let batch = read_source(&root.join("batch.rs"));
    for documented_api in [
        "/// Submit a lossless tile batch that resolves to host codestream bytes.",
        "/// Submit a lossless tile batch that resolves to Metal-backed codestreams.",
        "/// Encode a lossless tile batch and return host-byte timing reports.",
    ] {
        assert!(batch.contains(documented_api));
    }
    assert_eq!(batch.matches("#[cfg(target_os = \"macos\")]").count(), 11);

    let unavailable = read_source(&root.join("unavailable.rs"));
    assert_eq!(
        unavailable
            .matches("#[cfg(not(target_os = \"macos\"))]")
            .count(),
        3
    );

    let hybrid = read_source(&root.join("resident_hybrid.rs"));
    assert_in_order(
        &hybrid,
        &[
            "compute::prepare_lossless_device_code_blocks(",
            "compute::encode_ht_prepared_device_code_blocks_resident",
            "compute::read_resident_ht_tier1_code_blocks_for_cpu_packetization",
            "j2k_native::encode_j2k_packetization_scalar",
        ],
    );

    let submit = read_source(&root.join("resident_submit.rs"));
    assert_in_order(
        &submit,
        &[
            "prepare_planned_resident_lossless_tiles_batch(chunk_planned, session)",
            "let pending = submit_chunk(session, batch_items)?;",
        ],
    );
    let wait = read_source(&root.join("resident_wait.rs"));
    assert_in_order(
        &wait,
        &[
            "compute::wait_resident_lossless_codestream_batches",
            "finished_resident_lossless_buffer_encode(",
            "validate_finished_resident_lossless_buffer_encode(",
        ],
    );
}
