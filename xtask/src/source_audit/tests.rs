// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::paths::production_path_for_test;
use super::{inventory_panic_macro_sites, mask_test_only_syntax, PanicMacroInventory};

const INLINE_TEST_A: &str = include_str!("../../tests/fixtures/clone_audit/inline_test_a.rs");
const INLINE_TEST_B: &str = include_str!("../../tests/fixtures/clone_audit/inline_test_b.rs");
const PRODUCTION_CLONE_A: &str =
    include_str!("../../tests/fixtures/clone_audit/production_clone_a.rs");
const PRODUCTION_CLONE_B: &str =
    include_str!("../../tests/fixtures/clone_audit/production_clone_b.rs");

fn repository_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest has a repository parent")
}

fn mask(path: &str, source: &str) -> super::mask::MaskedRustSource {
    mask_test_only_syntax(repository_root(), Path::new(path), source)
        .expect("fixture must have classifiable Rust syntax")
}

#[test]
fn inline_cfg_test_clones_do_not_reach_production_clone_counts() {
    assert!(longest_shared_nonblank_run(INLINE_TEST_A, INLINE_TEST_B) >= 20);

    let masked_a = mask("crates/fixture/src/inline_test_a.rs", INLINE_TEST_A);
    let masked_b = mask("crates/fixture/src/inline_test_b.rs", INLINE_TEST_B);

    assert!(masked_a.masked_nodes > 0);
    assert!(masked_b.masked_nodes > 0);
    assert!(longest_shared_nonblank_run(&masked_a.text, &masked_b.text) < 20);
    assert!(!masked_a.text.contains("repeated_test_clone"));
    assert!(!masked_b.text.contains("repeated_test_clone"));
}

#[test]
fn production_clones_remain_visible_after_source_aware_masking() {
    let masked_a = mask(
        "crates/fixture/src/production_clone_a.rs",
        PRODUCTION_CLONE_A,
    );
    let masked_b = mask(
        "crates/fixture/src/production_clone_b.rs",
        PRODUCTION_CLONE_B,
    );

    assert_eq!(masked_a.masked_nodes, 0);
    assert_eq!(masked_b.masked_nodes, 0);
    assert!(longest_shared_nonblank_run(&masked_a.text, &masked_b.text) >= 20);
    assert!(masked_a.text.contains("production_clone_fixture"));
    assert!(masked_b.text.contains("production_clone_fixture"));
}

#[test]
fn masking_preserves_source_byte_and_line_positions() {
    let masked = mask("crates/fixture/src/lib.rs", INLINE_TEST_A);
    assert_eq!(masked.text.len(), INLINE_TEST_A.len());
    assert_eq!(
        newline_positions(&masked.text),
        newline_positions(INLINE_TEST_A)
    );
    assert_eq!(
        line_number(INLINE_TEST_A, "alpha_production_value"),
        line_number(&masked.text, "alpha_production_value")
    );
}

#[test]
fn cfg_implication_is_conservative_and_mixed_lines_are_reported() {
    let source = r#"
#[cfg(any(test, feature = "shipping"))]
fn possibly_shipping() {}

#[cfg(all(test, feature = "shipping"))]
fn definitely_test_only() {}

fn mixed() {
    let values = (#[cfg(test)] panic!("test only"), 7);
}
"#;
    let masked = mask("crates/fixture/src/mixed.rs", source);

    assert!(masked.text.contains("possibly_shipping"));
    assert!(!masked.text.contains("definitely_test_only"));
    assert!(!masked.text.contains("test only"));
    assert!(masked.text.contains("7);"));
    assert!(masked.mixed_lines.contains(&9));
}

#[test]
fn panic_macro_inventory_uses_masked_production_tokens() {
    let source = r#"
pub fn production(value: u8) {
    assert!(value > 0);
    debug_assert_eq!(value, 1);
    if value == 2 {
        panic!("production");
    }
}

#[cfg(test)]
mod tests {
    fn duplicate() {
        panic!("test");
        unreachable!("test");
        assert_eq!(1, 1);
    }
}
"#;
    let masked = mask("crates/fixture/src/panic.rs", source);
    let (inventory, sites) =
        inventory_panic_macro_sites("crates/fixture/src/panic.rs", &masked.text)
            .expect("inventory");

    assert_eq!(
        inventory,
        PanicMacroInventory {
            panic: 1,
            assert: 1,
            debug_assert_eq: 1,
            ..PanicMacroInventory::default()
        }
    );
    assert_eq!(
        sites
            .iter()
            .map(|site| (site.name, site.path.as_str(), site.line, site.column))
            .collect::<Vec<_>>(),
        [
            ("assert!", "crates/fixture/src/panic.rs", 3, 5),
            ("debug_assert_eq!", "crates/fixture/src/panic.rs", 4, 5),
            ("panic!", "crates/fixture/src/panic.rs", 6, 9),
        ]
    );
}

#[test]
fn production_source_path_policy_excludes_test_and_generated_families() {
    for path in [
        "crates/codec/src/lib.rs",
        "crates/codec/src/packet.rs",
        "crates/codec/src/bin/tool.rs",
    ] {
        assert!(production_path_for_test(path), "{path}");
    }
    for path in [
        "crates/codec/build.rs",
        "crates/codec/tests/integration.rs",
        "crates/codec/src/tests.rs",
        "crates/codec/src/packet_tests.rs",
        "crates/codec/src/test_helpers.rs",
        "crates/codec/src/test_plan.rs",
        "crates/codec/benches/throughput.rs",
        "crates/codec/fuzz/fuzz_targets/decode.rs",
        "crates/j2k-test-support/src/lib.rs",
    ] {
        assert!(!production_path_for_test(path), "{path}");
    }
}

#[test]
fn source_audit_resolves_raw_identifier_modules_without_changing_paths() {
    let root = temp_dir("raw-module");
    let source_dir = root.join("crates/fixture/src");
    fs::create_dir_all(&source_dir).expect("create fixture source");
    fs::write(source_dir.join("lib.rs"), "mod r#box;\n").expect("write fixture root");
    fs::write(source_dir.join("box.rs"), "pub fn value() -> u8 { 7 }\n")
        .expect("write raw module source");

    let masked = mask_test_only_syntax(
        &root,
        Path::new("crates/fixture/src/lib.rs"),
        "mod r#box;\n",
    )
    .expect("raw identifier module must resolve");
    assert_eq!(masked.text, "mod r#box;\n");

    fs::remove_dir_all(root).expect("remove raw-module fixture");
}

fn longest_shared_nonblank_run(left: &str, right: &str) -> usize {
    let left = left.lines().collect::<Vec<_>>();
    let right = right.lines().collect::<Vec<_>>();
    let mut previous = vec![0usize; right.len() + 1];
    let mut longest = 0usize;
    for left_line in left {
        let mut current = vec![0usize; right.len() + 1];
        for (index, right_line) in right.iter().enumerate() {
            if !left_line.trim().is_empty() && left_line == *right_line {
                current[index + 1] = previous[index] + 1;
                longest = longest.max(current[index + 1]);
            }
        }
        previous = current;
    }
    longest
}

fn newline_positions(source: &str) -> Vec<usize> {
    source
        .bytes()
        .enumerate()
        .filter_map(|(index, byte)| (byte == b'\n').then_some(index))
        .collect()
}

fn line_number(source: &str, pattern: &str) -> usize {
    source
        .lines()
        .position(|line| line.contains(pattern))
        .map(|index| index + 1)
        .expect("fixture pattern")
}

fn temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "j2k-source-audit-{label}-{}-{nonce}",
        std::process::id()
    ))
}
