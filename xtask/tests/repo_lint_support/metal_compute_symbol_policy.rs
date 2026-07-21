// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::repo_root;

#[test]
fn metal_compute_dependencies_are_imported_without_a_symbol_inventory() {
    let root = repo_root();
    let compute_path = root.join("crates/j2k-metal/src/compute.rs");
    let inventory_path = root.join("crates/j2k-metal/src/compute/symbol_inventory.rs");
    let compute = fs::read_to_string(&compute_path).expect("read Metal compute module");

    assert!(
        !inventory_path.exists(),
        "Metal compute dependencies must be imported by their actual owners, not a symbol inventory"
    );
    assert!(
        !compute.contains("symbol_inventory") && !compute.contains("wire_compute_symbols"),
        "compute.rs must not invoke an import-inventory macro"
    );
}
