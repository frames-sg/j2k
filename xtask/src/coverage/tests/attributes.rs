// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use crate::coverage::source_analysis::SourceIndex;

#[test]
fn cfg_test_module_does_not_hide_later_production_items() {
    let source = "\
fn before() {}

#[cfg(test)]
mod tests {
    #[test]
    fn only_a_test() {}
}

pub fn after() {
    let _value = 1;
}
";
    let index = SourceIndex::single("src/lib.rs", source).unwrap();
    let analysis = index.file("src/lib.rs").unwrap();

    assert_eq!(
        analysis
            .functions
            .iter()
            .map(|function| function.name.as_str())
            .collect::<Vec<_>>(),
        ["before", "after"]
    );
    assert!((3..=7).all(|line| analysis.test_only_lines.contains(&line)));
    assert!(!analysis.test_only_lines.contains(&9));
}

#[test]
fn cfg_test_attributes_on_fields_locals_arms_and_expressions_are_test_only() {
    let source = "\
struct Data {
    #[cfg(test)]
    test_field: u32,
    production_field: u32,
}
enum Mode {
    #[cfg(test)]
    TestOnly,
    Production,
}
fn inspect(input: Data) {
    #[cfg(test)]
    let test_local = 1;
    let production_local = 2;
    let Data {
        #[cfg(test)]
        test_field: _,
        production_field: _,
    } = input;
    let _value = Data {
        #[cfg(test)]
        test_field: 1,
        production_field: 2,
    };
    match production_local {
        #[cfg(test)]
        1 => {}
        _ => {}
    }
    #[cfg(test)]
    test_macro!();
    #[cfg(test)]
    test_call();
}
";
    let index = SourceIndex::single("src/lib.rs", source).unwrap();
    let analysis = index.file("src/lib.rs").unwrap();
    assert_line_dispositions(source, &analysis.test_only_lines);
}

#[test]
fn cfg_test_function_parameters_are_test_only_without_hiding_patterns() {
    let source = "\
fn parameterized(
    #[cfg(test)]
    test_parameter: u32,
    production_parameter: u32,
) {
    let (_, production_binding) = (1, 2);
    let _production = production_parameter + production_binding;
}
";
    let index = SourceIndex::single("src/lib.rs", source).unwrap();
    let analysis = index.file("src/lib.rs").unwrap();
    assert_line_dispositions(source, &analysis.test_only_lines);
}

fn assert_line_dispositions(source: &str, test_only_lines: &BTreeSet<usize>) {
    for (index, line) in source.lines().enumerate() {
        let line_number = index + 1;
        if line.trim() == "#[cfg(test)]" || line.contains("test_") || line.contains("TestOnly") {
            assert!(
                test_only_lines.contains(&line_number),
                "line {line_number} must be syntax-test-only: {line}"
            );
        }
        if line.contains("production_") || line.contains("Production") {
            assert!(
                !test_only_lines.contains(&line_number),
                "line {line_number} must remain production: {line}"
            );
        }
    }
}
