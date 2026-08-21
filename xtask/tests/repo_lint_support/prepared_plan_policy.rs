// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prevent prepared-plan type erasure from returning after the typed-plan migration.

use std::{collections::BTreeSet, fs};

use super::{repo_root, rust_sources};

const TYPE_ERASURE_PATTERNS: &[&str] = &[
    "core::any::Any",
    "adapter_view",
    "downcast_ref::<j2k_native::J2kReferenced",
    "downcast_ref::<J2kReferenced",
];
const ADAPTER_SOURCE_ROOTS: &[&str] = &[
    "crates/j2k/src",
    "crates/j2k-cuda/src",
    "crates/j2k-metal/src",
];

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TypeErasureOccurrence {
    relative_path: String,
    pattern: &'static str,
    count: usize,
}

fn prepared_plan_type_erasure_inventory() -> BTreeSet<TypeErasureOccurrence> {
    let root = repo_root();
    let mut occurrences = BTreeSet::new();
    for source_root in ADAPTER_SOURCE_ROOTS {
        for path in rust_sources(&root.join(source_root)) {
            let relative_path = path
                .strip_prefix(root)
                .expect("adapter source must be inside the repository")
                .to_string_lossy();
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            occurrences.extend(type_erasure_occurrences(&relative_path, &source));
        }
    }
    occurrences
}

fn type_erasure_occurrences(relative_path: &str, source: &str) -> BTreeSet<TypeErasureOccurrence> {
    TYPE_ERASURE_PATTERNS
        .iter()
        .filter_map(|&pattern| {
            let count = source.matches(pattern).count();
            (count != 0).then(|| occurrence(relative_path, pattern, count))
        })
        .collect()
}

#[test]
fn prepared_plans_do_not_use_type_erasure() {
    let actual = prepared_plan_type_erasure_inventory();
    assert!(
        actual.is_empty(),
        "prepared-plan type erasure is forbidden: {actual:#?}"
    );
}

#[test]
fn prepared_plans_share_one_image_geometry_owner() {
    let root = repo_root();
    let neutral = fs::read_to_string(root.join("crates/j2k-types/src/decode_plan/referenced.rs"))
        .expect("read neutral referenced geometry");
    let native = fs::read_to_string(root.join("crates/j2k-native/src/direct_plan.rs"))
        .expect("read native direct-plan producer boundary");
    let facade = fs::read_to_string(root.join("crates/j2k/src/owned_batch/prepared_plan.rs"))
        .expect("read facade prepared plans");

    assert_eq!(
        neutral
            .matches("pub struct J2kReferencedImageGeometry")
            .count(),
        1,
        "prepared image geometry must have one backend-neutral owner"
    );
    for method in [
        "pub const fn is_empty",
        "pub fn is_grayscale",
        "pub fn is_color",
        "pub fn is_rgba",
        "pub fn grayscale_geometry",
        "pub fn color_geometry",
        "pub fn rgba_geometry",
        "pub fn uniform_wavelet_transform",
    ] {
        assert!(
            neutral.contains(method),
            "shared geometry method missing: {method}"
        );
    }
    assert!(
        native.contains("pub use j2k_types::{"),
        "native must produce the neutral direct-plan contract instead of owning it"
    );
    assert!(
        !native.contains("pub struct J2kDirect")
            && !native.contains("pub enum J2kReferenced")
            && !native.contains("pub struct J2kReferenced"),
        "native must not reintroduce backend-local direct-plan types"
    );
    for alias in [
        "pub type ClassicPreparedGeometry = j2k_types::J2kReferencedClassicPlan;",
        "pub type Htj2kPreparedGeometry = j2k_types::J2kReferencedHtj2kPlan;",
        "pub type PreparedImageGeometry<'a> = j2k_types::J2kReferencedImageGeometry<'a>;",
    ] {
        assert!(
            facade.contains(alias),
            "neutral facade alias missing: {alias}"
        );
    }
    assert!(
        !facade.contains("j2k_native::J2kReferenced"),
        "the public facade must not expose private native plan types"
    );
    assert!(
        !facade.contains(".all(|tile| tile.grayscale_geometry()"),
        "facade reintroduced grayscale classification logic"
    );
    assert!(
        !facade.contains("fn uniform_wavelet_transform(\n    tiles:"),
        "facade reintroduced wavelet aggregation logic"
    );
}

#[test]
fn type_erasure_counter_reports_each_pattern_independently() {
    let actual = type_erasure_occurrences(
        "fixture.rs",
        "use core::any::Any; plan.adapter_view().adapter_view();",
    );
    assert_eq!(
        actual,
        BTreeSet::from([
            occurrence("fixture.rs", "core::any::Any", 1),
            occurrence("fixture.rs", "adapter_view", 2),
        ])
    );
}

fn occurrence(relative_path: &str, pattern: &'static str, count: usize) -> TypeErasureOccurrence {
    TypeErasureOccurrence {
        relative_path: relative_path.to_owned(),
        pattern,
        count,
    }
}
