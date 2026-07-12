// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use super::support::TestRepository;
use crate::coverage::source_analysis::{SourceIndex, SourceRole};

#[test]
fn body_bearing_function_forms_have_item_and_body_spans() {
    let source = "\
/// Changed documentation.
pub async unsafe extern \"C\" fn free_function(
    value: u32,
) -> u32 {
    value
}

struct Worker;
impl Worker {
    pub(crate) fn method(&self) {}
}

trait Operation {
    fn defaulted(&self) {
        let _value = 1;
    }
    fn declaration(&self);
}

unsafe extern \"C\" {
    fn foreign_declaration();
}
";
    let index = SourceIndex::single("src/lib.rs", source).unwrap();
    let functions = &index.file("src/lib.rs").unwrap().functions;
    let names = functions
        .iter()
        .map(|function| function.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(names, ["free_function", "method", "defaulted"]);
    assert!(
        functions
            .iter()
            .all(|function| function.start <= function.body_start
                && function.body_end <= function.end)
    );
    assert!(functions[0].start < functions[0].body_start);
}

#[test]
fn nested_inline_module_uses_its_real_module_directory() {
    let repository = TestRepository::new();
    repository.write("src/lib.rs", "mod outer { mod child; }\n");
    repository.write("src/outer/child.rs", "pub fn reached() {}\n");
    let changed = BTreeMap::from([("src/outer/child.rs".to_string(), BTreeSet::from([1]))]);
    let index = SourceIndex::repository_subset(
        repository.root(),
        &changed,
        &[("src/lib.rs", SourceRole::Production)],
    )
    .unwrap();

    assert_eq!(
        index.file("src/outer/child.rs").unwrap().role,
        SourceRole::Production
    );
}

#[test]
fn module_path_cannot_escape_the_repository_root() {
    let repository = TestRepository::new();
    let outside_name = format!("j2k-coverage-outside-{}.rs", std::process::id());
    let outside = repository
        .root()
        .parent()
        .expect("temporary repository has a parent")
        .join(&outside_name);
    fs::write(&outside, "pub fn outside() {}\n")
        .unwrap_or_else(|error| panic!("write outside module {}: {error}", outside.display()));
    repository.write(
        "src/lib.rs",
        &format!("#[path = \"../../{outside_name}\"]\nmod outside;\n"),
    );
    let changed = BTreeMap::from([("src/lib.rs".to_string(), BTreeSet::from([1]))]);
    let error = SourceIndex::repository_subset(
        repository.root(),
        &changed,
        &[("src/lib.rs", SourceRole::Production)],
    )
    .unwrap_err();
    fs::remove_file(&outside)
        .unwrap_or_else(|cleanup| panic!("remove outside module {}: {cleanup}", outside.display()));

    assert!(error.contains("outside repository root"));
}

#[test]
fn unknown_custom_cfg_is_conservatively_required() {
    let index = SourceIndex::single(
        "src/lib.rs",
        "#[cfg(build_script_decides)]\npub fn conditional() {}\n",
    )
    .unwrap();
    let function = &index.file("src/lib.rs").unwrap().functions[0];

    assert!(function.required_on_host);
}

#[test]
fn unknown_cfg_in_either_polarity_is_conservatively_required() {
    let source = "\
#[cfg(coverage_unknown)]
fn positive_unknown() {}
#[cfg(not(coverage_unknown))]
fn negative_unknown() {}
#[cfg(target_feature = \"fma\")]
fn positive_target_feature() {}
#[cfg(not(target_feature = \"fma\"))]
fn negative_target_feature() {}
";
    let index = SourceIndex::single("src/lib.rs", source).unwrap();
    let functions = &index.file("src/lib.rs").unwrap().functions;

    assert_eq!(functions.len(), 4);
    assert!(functions.iter().all(|function| function.required_on_host));
}

#[test]
fn structural_cfg_attr_and_duplicate_module_paths_fail_closed() {
    let cfg_attr = SourceIndex::single(
        "src/lib.rs",
        "#[cfg_attr(test, cfg(extra))]\npub fn conditional() {}\n",
    )
    .unwrap_err();
    assert!(cfg_attr.contains("structural cfg_attr"));

    let nested_cfg_attr = SourceIndex::single(
        "src/lib.rs",
        "#[cfg_attr(coverage_unknown, cfg_attr(other, cfg(test)))]\npub fn nested() {}\n",
    )
    .unwrap_err();
    assert!(nested_cfg_attr.contains("structural cfg_attr"));

    let duplicate_path = SourceIndex::single(
        "src/lib.rs",
        "#[path = \"one.rs\"]\n#[path = \"two.rs\"]\nmod child;\n",
    )
    .unwrap_err();
    assert!(duplicate_path.contains("more than one path attribute"));
}
