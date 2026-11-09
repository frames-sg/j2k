// SPDX-License-Identifier: MIT OR Apache-2.0

//! Each canonical codec literal has exactly one owner. Every other backend
//! imports it (Rust) or generates it from the owner (Metal, through
//! `cargo xtask codec-math-codegen`), so a change to the owner reaches every
//! CPU, CUDA and Metal path at once.

use std::{collections::BTreeMap, ffi::OsStr, fs, path::Path};

use super::repo_root;

/// Canonical literals, written without sign, digit separators or type
/// suffixes, and the one path that may spell each in code.
const OWNED_LITERALS: &[(&str, &[&str])] = &[
    (
        "crates/j2k-codec-math/src/jpeg/ycbcr.rs",
        &[
            "91881", "22554", "46802", "116130", "19595", "7471", "11059", "5329",
        ],
    ),
    (
        "crates/j2k-codec-math/src/jpeg/idct.rs",
        &[
            "2446", "3196", "4433", "6270", "7373", "9633", "12299", "15137", "16069", "16819",
            "20995", "25172",
        ],
    ),
    (
        "crates/j2k-codec-math/src/mct.rs",
        &[
            "0.299", "0.587", "0.114", "0.16875", "0.33126", "0.41869", "0.08131", "1.402",
            "0.34413", "0.71414", "1.772",
        ],
    ),
    (
        "crates/j2k-codec-math/src/dwt.rs",
        &[
            "1.5861343",
            "0.052980117",
            "0.8829111",
            "0.44350687",
            "1.2301741",
            "1.6257324",
            "1.586134342059924",
            "0.052980118572961",
            "0.882911075530934",
            "0.443506852043971",
            "1.230174104914001",
        ],
    ),
    // The sYCC colour space (JP2 `colr`) uses the exact BT.601 matrix, a
    // different contract from the ICT, so it has its own owner.
    (
        "crates/j2k-native/src/color/sycc.rs",
        &["0.344136", "0.714136"],
    ),
    // Derived Q14/Q15 forms of the YCbCr matrix (see DERIVED_FORM_PATHS).
    (
        "crates/j2k-jpeg/src/backend/neon.rs",
        &["22971", "11277", "23401", "29033"],
    ),
];

/// Paths that may spell an owned literal under another contract, with the
/// literals and the reason. Review each addition; none of these is a copy.
const SHARED_DIGIT_SITES: &[(&str, &[&str], &str)] = &[(
    "crates/j2k-native/src/color/sycc.rs",
    &["1.402", "1.772"],
    "the exact BT.601 sYCC matrix shares these two digits strings with the ICT",
)];

/// SIMD converters that restructure the YCbCr arithmetic into derived
/// fixed-point forms. They stay exempt only while the exhaustive test proves
/// them equal to the scalar reference for every (Y, Cb, Cr).
const DERIVED_FORM_PATHS: &[&str] = &[
    "crates/j2k-jpeg/src/backend/neon.rs",
    "crates/j2k-jpeg/src/backend/x86.rs",
];
const DERIVED_FORM_PROOF: (&str, &str) = (
    "crates/j2k-jpeg/src/backend/tests.rs",
    "fn detected_backend_ycbcr_matches_scalar_for_every_chroma_pair()",
);

const SCANNED_EXTENSIONS: &[&str] = &["rs", "metal", "cu", "h", "c", "cc", "cpp", "hpp", "py"];

/// Generated fragments, independent test oracles, benches and the xtask that
/// renders the fragments may spell literals; product code may not.
fn exempt(relative: &str) -> bool {
    relative.starts_with("crates/j2k-codec-math/generated/")
        || relative.starts_with("crates/j2k-test-support/")
        || relative.starts_with("crates/j2k-transcode-test-support/")
        || relative.starts_with("corpus/conformance/")
        || relative.starts_with("xtask/")
        || relative.contains("/tests/")
        || relative.contains("/benches/")
        || relative.ends_with("/tests.rs")
        || relative.ends_with("_tests.rs")
}

#[test]
fn codec_literals_have_exactly_one_owner() {
    let root = repo_root();
    let owners = owner_index();
    let proof = fs::read_to_string(root.join(DERIVED_FORM_PROOF.0)).unwrap_or_default();
    let derived_forms_proven = proof.contains(DERIVED_FORM_PROOF.1);

    let mut files = Vec::new();
    collect_sources(root, root, &mut files);
    let mut copies = BTreeMap::<(String, String), Vec<usize>>::new();
    let mut owner_hits = BTreeMap::<&str, usize>::new();
    for relative in files {
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        for (line, literal) in owned_literal_sites(&relative, &source, &owners) {
            if owners.get(literal.as_str()) == Some(&relative.as_str()) {
                *owner_hits.entry(owners[literal.as_str()]).or_default() += 1;
                continue;
            }
            if exempt(&relative)
                || (derived_forms_proven && DERIVED_FORM_PATHS.contains(&relative.as_str()))
                || shares_digits(&relative, &literal)
            {
                continue;
            }
            copies
                .entry((relative.clone(), literal))
                .or_default()
                .push(line);
        }
    }

    let failures = ownership_failures(&copies, &owner_hits, derived_forms_proven);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn ownership_failures(
    copies: &BTreeMap<(String, String), Vec<usize>>,
    owner_hits: &BTreeMap<&str, usize>,
    derived_forms_proven: bool,
) -> Vec<String> {
    let owners = owner_index();
    let mut failures = Vec::new();
    if !derived_forms_proven {
        failures.push(format!(
            "{} no longer defines `{}`; the derived SIMD forms in {DERIVED_FORM_PATHS:?} are exempt only while it proves them exhaustively",
            DERIVED_FORM_PROOF.0, DERIVED_FORM_PROOF.1
        ));
    }
    for (owner, _) in OWNED_LITERALS {
        if !owner_hits.contains_key(owner) {
            failures.push(format!(
                "{owner} no longer spells its owned literals; update OWNED_LITERALS"
            ));
        }
    }
    for ((path, literal), lines) in copies {
        let owner = owners[literal.as_str()];
        for line in lines {
            failures.push(format!(
                "{path}:{line} writes `{literal}`, owned by {owner}; import it (Rust) or generate it with `cargo xtask codec-math-codegen` (Metal)"
            ));
        }
    }
    failures
}

fn owner_index() -> BTreeMap<&'static str, &'static str> {
    let mut owners = BTreeMap::new();
    for (owner, literals) in OWNED_LITERALS {
        for literal in *literals {
            let previous = owners.insert(*literal, *owner);
            assert!(previous.is_none(), "`{literal}` has two owners");
        }
    }
    owners
}

fn shares_digits(relative: &str, literal: &str) -> bool {
    SHARED_DIGIT_SITES
        .iter()
        .any(|(path, literals, _)| *path == relative && literals.contains(&literal))
}

fn collect_sources(root: &Path, directory: &Path, files: &mut Vec<String>) {
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .map(|entry| entry.expect("read directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
        if path.is_dir() {
            if !matches!(name, ".codewhale" | ".git" | ".venv" | "target") {
                collect_sources(root, &path, files);
            }
        } else if path
            .extension()
            .and_then(OsStr::to_str)
            .is_some_and(|extension| SCANNED_EXTENSIONS.contains(&extension))
        {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            files.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// Lines and normalized spellings of owned literals in `source`'s code.
fn owned_literal_sites(
    relative: &str,
    source: &str,
    owners: &BTreeMap<&str, &str>,
) -> Vec<(usize, String)> {
    // The scan admits only the exact lowercase SCANNED_EXTENSIONS.
    let extension = Path::new(relative).extension().and_then(OsStr::to_str);
    let python = extension == Some("py");
    let mut code = strip_comments(source, python);
    if extension == Some("rs") {
        code = blank_inline_test_modules(&code);
    }
    numeric_tokens(&code)
        .into_iter()
        .filter(|(_, literal)| owners.contains_key(literal.as_str()))
        .collect()
}

/// Replace comments with spaces, keeping newlines so line numbers survive.
/// String and character literals stay: a literal inside a generated string
/// is still a copy.
fn strip_comments(source: &str, python: bool) -> String {
    let bytes = source.as_bytes();
    let mut out = String::new();
    let mut index = 0;
    while index < bytes.len() {
        let rest = &source[index..];
        let comment_end = if python && rest.starts_with('#') {
            Some(rest.find('\n').map_or(source.len(), |end| index + end))
        } else if !python && rest.starts_with("//") {
            Some(rest.find('\n').map_or(source.len(), |end| index + end))
        } else if !python && rest.starts_with("/*") {
            Some(
                rest[2..]
                    .find("*/")
                    .map_or(source.len(), |end| index + 2 + end + 2),
            )
        } else {
            None
        };
        if let Some(end) = comment_end {
            out.extend(source[index..end].chars().map(
                |character| {
                    if character == '\n' {
                        '\n'
                    } else {
                        ' '
                    }
                },
            ));
            index = end;
            continue;
        }
        if bytes[index] == b'"' {
            let mut end = index + 1;
            while end < bytes.len() && bytes[end] != b'"' {
                end += if bytes[end] == b'\\' { 2 } else { 1 };
            }
            let end = (end + 1).min(bytes.len());
            out.push_str(&source[index..end]);
            index = end;
            continue;
        }
        let character = rest.chars().next().expect("non-empty remainder");
        out.push(character);
        index += character.len_utf8();
    }
    out
}

/// Blank `#[cfg(test)] mod name { ... }` bodies: inline unit tests are oracles.
fn blank_inline_test_modules(code: &str) -> String {
    let mut out = code.to_string();
    let mut search = 0;
    while let Some(found) = out[search..].find("#[cfg(test)]") {
        let start = search + found;
        let after = out[start + "#[cfg(test)]".len()..].trim_start();
        let is_module = after.starts_with("mod ") || after.starts_with("pub(crate) mod ");
        let Some(open) = out[start..].find('{').map(|open| start + open) else {
            break;
        };
        if !is_module || after.find('{') > after.find(';') {
            search = start + 1;
            continue;
        }
        let mut depth = 0_usize;
        let mut end = out.len();
        for (offset, character) in out[open..].char_indices() {
            match character {
                '{' => depth += 1,
                '}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        end = open + offset + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        // Keep byte offsets: a multi-byte character becomes as many spaces.
        let blanked = out[start..end]
            .chars()
            .map(|character| {
                if character == '\n' {
                    "\n".to_string()
                } else {
                    " ".repeat(character.len_utf8())
                }
            })
            .collect::<String>();
        out.replace_range(start..end, &blanked);
        search = end;
    }
    out
}

/// Numeric tokens with their line: digits and dots on digit/dot boundaries,
/// outside identifiers, with `_` separators and type suffixes removed. A
/// leading `-` is outside the token, so signed spellings match too.
fn numeric_tokens(code: &str) -> Vec<(usize, String)> {
    let bytes = code.as_bytes();
    let mut tokens = Vec::new();
    let mut line = 1;
    let mut index = 0;
    let identifier = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'\n' {
            line += 1;
            index += 1;
        } else if byte.is_ascii_alphabetic() || byte == b'_' {
            while index < bytes.len() && identifier(bytes[index]) {
                index += 1;
            }
        } else if byte.is_ascii_digit()
            || (byte == b'.'
                && bytes.get(index + 1).is_some_and(u8::is_ascii_digit)
                && (index == 0 || !identifier(bytes[index - 1])))
        {
            let start = index;
            while index < bytes.len()
                && (bytes[index].is_ascii_digit() || matches!(bytes[index], b'.' | b'_'))
            {
                index += 1;
            }
            let token = code[start..index].replace('_', "");
            tokens.push((line, token.trim_end_matches('.').to_string()));
            while index < bytes.len() && identifier(bytes[index]) {
                index += 1;
            }
        } else {
            index += 1;
        }
    }
    tokens
}

#[test]
fn ownership_gate_reports_a_planted_duplicate_and_ignores_comments_tests_and_neighbours() {
    let owners = owner_index();
    let planted = "// 1.402 in a comment is fine\n\
                   const K: f32 = 1.402; /* 0.299 */\n\
                   let r = fma(y2, -1.402f, y0);\n\
                   let t = 10.299 + 0.2990 + 2.0_f32;\n\
                   const FIX_1_40200: i32 = 91_881;\n\
                   const G: f32 = 0.587_f32;\n\
                   #[cfg(test)]\n\
                   mod tests {\n\
                       const ORACLE: f32 = 0.114;\n\
                   }\n";
    let sites = owned_literal_sites("crates/j2k-metal/src/planted.rs", planted, &owners);
    assert_eq!(
        sites,
        [
            (2, "1.402".to_string()),
            (3, "1.402".to_string()),
            (5, "91881".to_string()),
            (6, "0.587".to_string()),
        ]
    );

    let metal = "plane0[gid] = fma(0.114f, b, fma(0.299f, r, 0.587f * g)); // 0.33126\n";
    let mut copies = BTreeMap::<(String, String), Vec<usize>>::new();
    for (line, literal) in owned_literal_sites("crates/j2k-metal/src/planted.metal", metal, &owners)
    {
        copies
            .entry(("crates/j2k-metal/src/planted.metal".to_string(), literal))
            .or_default()
            .push(line);
    }
    let owner_hits = OWNED_LITERALS
        .iter()
        .map(|(owner, _)| (*owner, 1))
        .collect::<BTreeMap<_, _>>();
    let failures = ownership_failures(&copies, &owner_hits, true)
        .into_iter()
        .filter(|failure| failure.contains("planted"))
        .collect::<Vec<_>>();
    assert_eq!(failures.len(), 3, "{failures:?}");
    assert!(failures[0]
        .contains("planted.metal:1 writes `0.114`, owned by crates/j2k-codec-math/src/mct.rs"));

    let unproven = ownership_failures(&BTreeMap::new(), &owner_hits, false);
    assert!(unproven
        .iter()
        .any(|failure| failure.contains("derived SIMD forms")));
}
