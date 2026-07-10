// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs};

use super::relative_path;
use crate::repo_lint_support::{repo_root, rust_sources};

const REVIEWED_ALLOWS: &[(&str, &str)] = &[
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        "clippy::manual_div_ceil",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        "clippy::too_many_lines",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "clippy::manual_div_ceil",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "clippy::manual_is_multiple_of",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/main.rs",
        "static_mut_refs",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_idwt/simt/src/main.rs",
        "static_mut_refs",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::cast_possible_truncation",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::cast_sign_loss",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::many_single_char_names",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_encode/simt/src/main.rs",
        "clippy::too_many_arguments",
    ),
    (
        "crates/j2k-cuda-runtime/src/cuda_oxide_simt_prelude.rs",
        "dead_code",
    ),
];

const NEVER_EXPECT_LINTS: &[&str] = &[
    "clippy::undocumented_unsafe_blocks",
    "clippy::uninit_assumed_init",
    "clippy::uninit_vec",
    "invalid_value",
    "unsafe_op_in_unsafe_fn",
];

#[test]
fn suppressions_stay_in_reviewed_device_generation_scopes() {
    let root = repo_root();
    let reviewed = REVIEWED_ALLOWS.iter().copied().collect::<BTreeSet<_>>();
    let mut sources = rust_sources(&root.join("crates"));
    sources.extend(rust_sources(&root.join("xtask")));
    sources.sort();

    let mut unreviewed = Vec::new();
    let mut file_expectations = Vec::new();
    let mut dangerous_expectations = Vec::new();
    let mut module_dead_code_expectations = Vec::new();
    let mut unexplained_expectations = Vec::new();
    for path in sources {
        let relative = relative_path(root, &path);
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let lines = source.lines().collect::<Vec<_>>();
        for (index, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            let block = suppression_attribute_block(&lines, index);
            let contains_allow = block
                .as_deref()
                .and_then(|attribute| lint_attribute_arguments(attribute, "allow"))
                .is_some();

            if contains_allow {
                let attribute = block.as_deref().unwrap_or(trimmed);
                assert!(
                    lint_attribute_has_reason(attribute, "allow"),
                    "reviewed source allowance {relative}:{} must state its device-specific reason",
                    index + 1
                );
                let attribute_lints = source_allow_lints(attribute);
                assert!(
                    !attribute_lints.is_empty(),
                    "source allowance {relative}:{} must name at least one lint",
                    index + 1
                );
                for lint in attribute_lints {
                    if !reviewed.contains(&(relative.as_str(), lint)) {
                        unreviewed.push(format!("{relative}:{} `{lint}`", index + 1));
                    }
                }
            }

            let contains_expect = block
                .as_deref()
                .and_then(|attribute| lint_attribute_arguments(attribute, "expect"))
                .is_some();
            if contains_expect {
                let attribute = block.as_deref().unwrap_or(trimmed);
                if !lint_attribute_has_reason(attribute, "expect") {
                    unexplained_expectations.push(format!("{relative}:{}", index + 1));
                }
                let expectation_lints = source_expect_lints(attribute);
                for lint in NEVER_EXPECT_LINTS {
                    if expectation_lints.contains(lint) {
                        dangerous_expectations.push(format!("{relative}:{} `{lint}`", index + 1));
                    }
                }
                if expectation_lints.contains(&"dead_code")
                    && expectation_targets_module(&lines, index)
                {
                    module_dead_code_expectations.push(format!("{relative}:{}", index + 1));
                }
            }

            if file_level_expectation(&lines, index) {
                file_expectations.push(format!("{relative}:{}", index + 1));
            }
        }
    }

    assert!(
        unreviewed.is_empty(),
        "host or unreviewed source lint allowances are forbidden: {unreviewed:?}"
    );
    assert!(
        file_expectations.is_empty(),
        "file-level lint expectations hide future findings; localize them to items: {file_expectations:?}"
    );
    assert!(
        unexplained_expectations.is_empty(),
        "lint expectations must explain the preserved contract or boundary: {unexplained_expectations:?}"
    );
    assert!(
        dangerous_expectations.is_empty(),
        "memory-safety lint expectations are forbidden; fix the unsafe boundary: {dangerous_expectations:?}"
    );
    assert!(
        module_dead_code_expectations.is_empty(),
        "module-level `dead_code` expectations hide an unbounded subtree; localize expectations to the unused items: {module_dead_code_expectations:?}"
    );
}

fn attribute_block(lines: &[&str], start: usize) -> String {
    let candidate = lines
        .iter()
        .skip(start)
        .take(32)
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    outer_attribute_end(&candidate)
        .map_or_else(|| candidate.clone(), |end| candidate[..end].to_owned())
}

fn suppression_attribute_block(lines: &[&str], start: usize) -> Option<String> {
    let trimmed = lines.get(start)?.trim_start();
    let suppression_attribute = is_direct_attribute(trimmed, "allow")
        || is_direct_attribute(trimmed, "expect")
        || is_direct_attribute(trimmed, "cfg_attr");
    suppression_attribute.then(|| attribute_block(lines, start))
}

fn source_allow_lints(attribute: &str) -> Vec<&str> {
    source_attribute_lints(attribute, "allow")
}

fn source_expect_lints(attribute: &str) -> Vec<&str> {
    source_attribute_lints(attribute, "expect")
}

fn source_attribute_lints<'a>(attribute: &'a str, action: &str) -> Vec<&'a str> {
    lint_attribute_arguments(attribute, action)
        .map(top_level_arguments)
        .unwrap_or_default()
        .into_iter()
        .filter(|argument| {
            !argument.is_empty()
                && argument
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':'))
        })
        .collect()
}

fn lint_attribute_has_reason(attribute: &str, action: &str) -> bool {
    lint_attribute_arguments(attribute, action)
        .map(top_level_arguments)
        .unwrap_or_default()
        .into_iter()
        .any(|argument| {
            argument
                .split_once('=')
                .is_some_and(|(key, _)| key.trim() == "reason")
        })
}

fn lint_attribute_arguments<'a>(attribute: &'a str, action: &str) -> Option<&'a str> {
    let bytes = attribute.as_bytes();
    let action_bytes = action.as_bytes();
    let mut cursor = 0;
    let mut quote = None;
    let mut escaped = false;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == active_quote {
                quote = None;
            }
            cursor += 1;
            continue;
        }
        if matches!(byte, b'\'' | b'"') {
            quote = Some(byte);
            cursor += 1;
            continue;
        }
        if !bytes[cursor..].starts_with(action_bytes) {
            cursor += 1;
            continue;
        }
        let start = cursor;
        let before = start
            .checked_sub(1)
            .and_then(|index| bytes.get(index))
            .copied();
        let after_name = start + action.len();
        let after = bytes.get(after_name).copied();
        if before.is_some_and(is_identifier_byte) || after.is_some_and(is_identifier_byte) {
            cursor = after_name;
            continue;
        }
        let open = after_name
            + attribute[after_name..]
                .bytes()
                .take_while(u8::is_ascii_whitespace)
                .count();
        if bytes.get(open) != Some(&b'(') {
            cursor = after_name;
            continue;
        }
        let close = matching_delimiter_end(attribute, open, b'(', b')')?;
        return Some(&attribute[open + 1..close - 1]);
    }
    None
}

fn top_level_arguments(arguments: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut delimiters = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    for (index, byte) in arguments.bytes().enumerate() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == active_quote {
                quote = None;
            }
            continue;
        }
        match byte {
            b'\'' | b'"' => quote = Some(byte),
            b'(' => delimiters.push(b')'),
            b'[' => delimiters.push(b']'),
            b'{' => delimiters.push(b'}'),
            b')' | b']' | b'}' if delimiters.last() == Some(&byte) => {
                delimiters.pop();
            }
            b',' if delimiters.is_empty() => {
                result.push(arguments[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    result.push(arguments[start..].trim());
    result
}

fn matching_delimiter_end(source: &str, open: usize, opening: u8, closing: u8) -> Option<usize> {
    let mut depth = 0_u32;
    let mut quote = None;
    let mut escaped = false;
    for (offset, byte) in source.as_bytes().iter().copied().enumerate().skip(open) {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == active_quote {
                quote = None;
            }
            continue;
        }
        if matches!(byte, b'\'' | b'"') {
            quote = Some(byte);
        } else if byte == opening {
            depth = depth.checked_add(1)?;
        } else if byte == closing {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(offset + 1);
            }
        }
    }
    None
}

fn outer_attribute_end(source: &str) -> Option<usize> {
    let open = source.find('[')?;
    matching_delimiter_end(source, open, b'[', b']')
}

fn is_direct_attribute(source: &str, name: &str) -> bool {
    let body = source
        .strip_prefix("#![")
        .or_else(|| source.strip_prefix("#["))
        .unwrap_or_default()
        .trim_start();
    let Some(after_name) = body.strip_prefix(name) else {
        return false;
    };
    !after_name
        .as_bytes()
        .first()
        .is_some_and(|byte| is_identifier_byte(*byte))
        && after_name.trim_start().starts_with('(')
}

fn is_inner_attribute(source: &str) -> bool {
    source.starts_with("#![")
}

fn file_level_expectation(lines: &[&str], start: usize) -> bool {
    let Some(trimmed) = lines.get(start).map(|line| line.trim_start()) else {
        return false;
    };
    is_inner_attribute(trimmed)
        && suppression_attribute_block(lines, start)
            .as_deref()
            .and_then(|attribute| lint_attribute_arguments(attribute, "expect"))
            .is_some()
}

const fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn expectation_targets_module(lines: &[&str], start: usize) -> bool {
    let source = lines
        .iter()
        .skip(start)
        .take(64)
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    let mut remaining = source.as_str();
    loop {
        remaining = trim_rust_trivia(remaining);
        if !remaining.starts_with("#[") {
            break;
        }
        let Some(end) = outer_attribute_end(remaining) else {
            return false;
        };
        remaining = &remaining[end..];
    }
    is_module_declaration(trim_rust_trivia(remaining))
}

fn trim_rust_trivia(mut source: &str) -> &str {
    loop {
        source = source.trim_start();
        if let Some(comment) = source.strip_prefix("//") {
            source = comment
                .find('\n')
                .map_or("", |newline| &comment[newline + 1..]);
        } else if let Some(comment) = source.strip_prefix("/*") {
            let Some(end) = comment.find("*/") else {
                return "";
            };
            source = &comment[end + 2..];
        } else {
            return source;
        }
    }
}

fn is_module_declaration(source: &str) -> bool {
    let mut item = source.trim_start();
    if let Some(after_pub) = item.strip_prefix("pub") {
        let Some(boundary) = after_pub.as_bytes().first().copied() else {
            return false;
        };
        if boundary == b'(' {
            let Some(close) = matching_delimiter_end(after_pub, 0, b'(', b')') else {
                return false;
            };
            item = &after_pub[close..];
        } else if boundary.is_ascii_whitespace() {
            item = after_pub;
        } else {
            return false;
        }
        item = item.trim_start();
    }
    item.strip_prefix("mod").is_some_and(|after_mod| {
        after_mod
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_whitespace)
    })
}

#[test]
fn attribute_block_captures_multiline_expect_reasons() {
    let lines = [
        "#[expect(",
        "    dead_code,",
        "    reason = \"shared target-specific fixture helpers\"",
        ")]",
        "mod fixture;",
    ];
    let attribute = suppression_attribute_block(&lines, 0).expect("lint attribute");
    assert!(attribute.contains("dead_code"));
    assert!(attribute.contains("reason ="));
    assert!(suppression_attribute_block(&lines, 4).is_none());
}

#[test]
fn file_expectations_cannot_use_whitespace_to_bypass_detection() {
    let direct = [
        "#![ expect (",
        "    dead_code,",
        "    reason = \"file-wide suppression\"",
        ")]",
    ];
    let conditional = [
        "#![ cfg_attr(",
        "    test,",
        "    expect (dead_code, reason = \"conditional file-wide suppression\")",
        ")]",
    ];
    let item = [
        "#[ expect (dead_code, reason = \"localized helper\")]",
        "fn helper() {}",
    ];

    assert!(file_level_expectation(&direct, 0));
    assert!(file_level_expectation(&conditional, 0));
    assert!(!file_level_expectation(&item, 0));
}

#[test]
fn module_dead_code_expectations_cannot_hide_a_subtree() {
    let direct = [
        "#[expect(",
        "    reason = \"temporary fixture module, with legacy helpers\",",
        "    dead_code,",
        ")]",
        "#[path = \"support/fixture.rs\"]",
        "pub(crate) mod fixture;",
    ];
    let conditional = [
        "#[cfg_attr(",
        "    feature = \"fixture\",",
        "    expect(dead_code, reason = \"conditional helper module\")",
        ")]",
        "mod fixture;",
    ];
    let localized = [
        "#[expect(dead_code, reason = \"single target-specific helper\")]",
        "fn fixture_helper() {}",
    ];

    assert_eq!(
        source_expect_lints(&attribute_block(&direct, 0)),
        ["dead_code"]
    );
    assert!(expectation_targets_module(&direct, 0));
    assert!(expectation_targets_module(&conditional, 0));
    assert!(!expectation_targets_module(&localized, 0));
}

#[test]
fn allow_lints_extract_the_exact_registered_ceiling() {
    let attribute = "#![allow(\n    clippy::cast_possible_truncation,\n    clippy::cast_sign_loss,\n    reason = \"bounded device ABI narrowing\"\n)]";
    assert_eq!(
        source_allow_lints(attribute),
        ["clippy::cast_possible_truncation", "clippy::cast_sign_loss"]
    );
    assert_eq!(source_allow_lints("#[allow(dead_code)]"), ["dead_code"]);
    assert!(!lint_attribute_has_reason("#[allow(dead_code)]", "allow"));
}
