// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::super::enforce_panic_macro_inventory;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn temp_repository(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "j2k-panic-inventory-{label}-{}-{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(root.join("crates/codec/src/bin")).expect("create temporary source tree");
    root
}

fn metadata(root: &Path) -> String {
    serde_json::json!({
        "packages": [{
            "name": "codec",
            "targets": [
                {
                    "crate_types": ["lib"],
                    "src_path": root.join("crates/codec/src/lib.rs")
                },
                {
                    "crate_types": ["bin"],
                    "src_path": root.join("crates/codec/src/bin/tool.rs")
                }
            ]
        }]
    })
    .to_string()
}

#[test]
fn inventory_scans_production_masks_tests_and_excludes_bin_roots() {
    let root = temp_repository("pass");
    fs::write(
        root.join("crates/codec/src/lib.rs"),
        "pub fn checked(value: bool) { assert!(value); }\n#[cfg(test)] mod tests { fn helper() { panic!(\"test only\"); } }\n",
    )
    .expect("write library source");
    fs::write(
        root.join("crates/codec/src/bin/tool.rs"),
        "fn main() { panic!(\"binary only\"); }\n",
    )
    .expect("write binary source");

    let inventory = enforce_panic_macro_inventory(&metadata(&root), &["codec".to_string()], &root)
        .expect("inventory below ratchet");

    assert_eq!(inventory.assert, 1);
    assert_eq!(inventory.panic, 0);
}

#[test]
fn inventory_ratchet_reports_the_exact_production_site() {
    let root = temp_repository("failure");
    fs::write(
        root.join("crates/codec/src/lib.rs"),
        "pub fn fail() { panic!(\"production panic\"); }\n",
    )
    .expect("write library source");
    fs::write(root.join("crates/codec/src/bin/tool.rs"), "fn main() {}\n")
        .expect("write binary source");

    let error = enforce_panic_macro_inventory(&metadata(&root), &["codec".to_string()], &root)
        .expect_err("panic macro must exceed zero baseline");

    assert!(error.contains("panic! 1/0"));
    assert!(error.contains("crates/codec/src/lib.rs:1"));
    assert!(error.contains("current inventory"));
}
