use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::Path;

use proc_macro2::{TokenStream, TokenTree};
use syn::visit::{self, Visit};

use crate::codegen_commands::codec_math_codegen;
use crate::command_support::{
    command_output_os, run_cargo, run_cargo_with_env, run_nightly_cargo,
    run_nightly_cargo_in_dir_owned, run_program, rust_sources,
};
use crate::panic_surface::panic_surface;
use crate::release_commands::published_library_packages;

const NO_STD_TARGET: &str = "aarch64-unknown-none";
const NO_STD_CORE_PORTABLE_TARGET: &str = "wasm32-unknown-unknown";

pub(super) fn ci() -> Result<(), String> {
    fmt()?;
    codec_math_codegen(std::iter::empty::<String>())?;
    clippy()?;
    panic_surface()?;
    test()?;
    doc()?;
    verify_unsafe_audit()
}

pub(super) fn repo_lint(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    if let Some(argument) = args.next() {
        return Err(format!("unknown repo-lint argument `{argument}`"));
    }

    run_cargo(&[
        "test",
        "-p",
        "xtask",
        "--test",
        "repo_lint",
        "--",
        "--nocapture",
    ])
}

pub(super) fn fmt() -> Result<(), String> {
    run_cargo(&["fmt", "--all", "--", "--check"])
}

pub(super) fn clippy() -> Result<(), String> {
    run_cargo(&[
        "clippy",
        "--workspace",
        "--lib",
        "--all-features",
        "--",
        "-D",
        "warnings",
    ])?;
    run_cargo(&[
        "clippy",
        "--workspace",
        "--bins",
        "--examples",
        "--tests",
        "--benches",
        "--all-features",
        "--",
        "-D",
        "warnings",
        "-A",
        "clippy::disallowed_methods",
        "-A",
        "clippy::disallowed_macros",
    ])?;
    run_cargo(&[
        "clippy",
        "-p",
        "xtask",
        "--test",
        "repo_lint",
        "--all-features",
        "--",
        "-D",
        "warnings",
    ])
}

pub(super) fn clippy_strict() -> Result<(), String> {
    let mut library_args = vec![
        "clippy",
        "-p",
        "j2k-native",
        "-p",
        "j2k",
        "--lib",
        "--all-features",
        "--no-deps",
    ];
    append_strict_clippy_lints(&mut library_args);
    run_cargo(&library_args)?;

    let mut non_library_args = vec![
        "clippy",
        "-p",
        "j2k-native",
        "-p",
        "j2k",
        "--bins",
        "--examples",
        "--tests",
        "--benches",
        "--all-features",
        "--no-deps",
    ];
    append_strict_clippy_lints(&mut non_library_args);
    non_library_args.extend([
        "-A",
        "clippy::disallowed_methods",
        "-A",
        "clippy::disallowed_macros",
    ]);

    run_cargo(&non_library_args)
}

fn append_strict_clippy_lints(args: &mut Vec<&'static str>) {
    args.extend([
        "--",
        "-W",
        "clippy::pedantic",
        "-W",
        "clippy::nursery",
        "-D",
        "warnings",
    ]);

    // Keep the strict gate useful as a ratchet: enable pedantic/nursery, but
    // baseline high-noise codec-math lints so new lint classes still fail.
    for lint in STRICT_CLIPPY_BASELINE_ALLOWED_LINTS {
        args.extend(["-A", lint]);
    }
}

const STRICT_CLIPPY_BASELINE_ALLOWED_LINTS: &[&str] = &[
    "clippy::bool_to_int_with_if",
    "clippy::branches_sharing_code",
    "clippy::cast_lossless",
    "clippy::cast_possible_truncation",
    "clippy::cast_possible_wrap",
    "clippy::cast_precision_loss",
    "clippy::cast_sign_loss",
    "clippy::checked_conversions",
    "clippy::cognitive_complexity",
    "clippy::doc_markdown",
    "clippy::elidable_lifetime_names",
    "clippy::explicit_deref_methods",
    "clippy::explicit_iter_loop",
    "clippy::float_cmp",
    "clippy::if_not_else",
    "clippy::inconsistent_struct_constructor",
    "clippy::inline_always",
    "clippy::items_after_statements",
    "clippy::manual_let_else",
    "clippy::map_unwrap_or",
    "clippy::match_same_arms",
    "clippy::missing_const_for_fn",
    "clippy::missing_errors_doc",
    "clippy::must_use_candidate",
    "clippy::needless_collect",
    "clippy::needless_pass_by_ref_mut",
    "clippy::needless_pass_by_value",
    "clippy::no_effect_underscore_binding",
    "clippy::or_fun_call",
    "clippy::redundant_clone",
    "clippy::redundant_closure_for_method_calls",
    "clippy::redundant_else",
    "clippy::redundant_pub_crate",
    "clippy::similar_names",
    "clippy::struct_excessive_bools",
    "clippy::struct_field_names",
    "clippy::suboptimal_flops",
    "clippy::suspicious_operation_groupings",
    "clippy::too_many_lines",
    "clippy::trivially_copy_pass_by_ref",
    "clippy::unnecessary_wraps",
    "clippy::unreadable_literal",
    "clippy::used_underscore_binding",
    "clippy::useless_let_if_seq",
];

pub(super) fn test() -> Result<(), String> {
    if env::consts::OS != "macos" {
        test_workspace_without_benches(&[])?;
        test_alloc_probe()?;
        return test_downstream_examples();
    }

    test_workspace_without_benches(&["--exclude", "j2k-metal"])?;
    test_alloc_probe()?;
    test_j2k_metal_without_benches()?;
    test_downstream_examples()
}

fn test_workspace_without_benches(extra_args: &[&str]) -> Result<(), String> {
    let mut test_args = vec![
        "test",
        "--workspace",
        "--all-features",
        "--lib",
        "--bins",
        "--tests",
        "--exclude",
        "j2k-alloc-probe",
    ];
    test_args.extend_from_slice(extra_args);
    run_cargo(&test_args)?;
    test_facade_cuda_stub()?;

    let mut doc_args = vec!["test", "--workspace", "--all-features", "--doc"];
    doc_args.extend_from_slice(extra_args);
    run_cargo(&doc_args)
}

fn test_alloc_probe() -> Result<(), String> {
    run_cargo(&["test", "-p", "j2k-alloc-probe"])
}

fn test_facade_cuda_stub() -> Result<(), String> {
    run_cargo(&[
        "test",
        "-p",
        "j2k",
        "--test",
        "encode_lossless",
        "accelerator_facade_auto_falls_back_when_no_stage_dispatches",
    ])
}

fn test_j2k_metal_without_benches() -> Result<(), String> {
    run_cargo_with_env(
        &[
            "test",
            "-p",
            "j2k-metal",
            "--all-features",
            "--lib",
            "--bins",
            "--tests",
        ],
        &[("RUST_TEST_THREADS", "1")],
    )?;
    run_cargo(&["test", "-p", "j2k-metal", "--all-features", "--doc"])
}

pub(super) fn doc() -> Result<(), String> {
    run_cargo_with_env(
        &["doc", "--workspace", "--all-features", "--no-deps"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    )?;

    for package in published_library_packages()? {
        run_cargo_with_env(
            &["doc", "-p", package.as_str(), "--lib", "--no-deps"],
            &[("RUSTDOCFLAGS", "-D warnings -D missing_docs")],
        )?;
    }

    run_cargo_with_env(
        &["doc", "-p", "j2k-cli", "--no-deps"],
        &[("RUSTDOCFLAGS", "-D warnings -D missing_docs")],
    )
}

pub(super) fn fuzz_build() -> Result<(), String> {
    run_cargo(&["check", "--manifest-path", "crates/j2k/fuzz/Cargo.toml"])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/j2k-jpeg/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/j2k-tilecodec/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/j2k-transcode/fuzz/Cargo.toml",
    ])?;
    run_cargo(&[
        "check",
        "--manifest-path",
        "crates/j2k-t803/fuzz/Cargo.toml",
    ])
}

const FUZZ_TARGETS: &[(&str, &str)] = &[
    ("crates/j2k", "decode_fuzz"),
    ("crates/j2k", "jp2_box_fuzz"),
    ("crates/j2k", "jp2_metadata_fuzz"),
    ("crates/j2k", "srgb8_fuzz"),
    ("crates/j2k", "parse_fuzz"),
    ("crates/j2k", "region_scaled_fuzz"),
    ("crates/j2k-jpeg", "decode_fuzz"),
    ("crates/j2k-jpeg", "parse_fuzz"),
    ("crates/j2k-jpeg", "region_scaled_fuzz"),
    ("crates/j2k-jpeg", "row_stream_fuzz"),
    ("crates/j2k-tilecodec", "decompress_fuzz"),
    ("crates/j2k-transcode", "jpeg_to_htj2k_fuzz"),
    ("crates/j2k-t803", "pgx_fuzz"),
    ("crates/j2k-t803", "archive_fuzz"),
];

pub(super) fn fuzz_run() -> Result<(), String> {
    let runs = env::var("J2K_FUZZ_RUNS").unwrap_or_else(|_| "1000".to_string());
    let max_total_time = env::var("J2K_FUZZ_MAX_TOTAL_TIME_SECONDS").ok();
    let fuzz_target = fuzz_target_triple()?;

    for (crate_dir, target) in FUZZ_TARGETS {
        let mut args = vec![
            "fuzz".to_string(),
            "run".to_string(),
            "--target".to_string(),
            fuzz_target.clone(),
            (*target).to_string(),
            "--".to_string(),
            format!("-runs={runs}"),
        ];
        if let Some(seconds) = &max_total_time {
            args.push(format!("-max_total_time={seconds}"));
        }
        run_nightly_cargo_in_dir_owned(crate_dir, &args)?;
    }
    Ok(())
}

fn fuzz_target_triple() -> Result<String, String> {
    if let Ok(target) = env::var("J2K_FUZZ_TARGET") {
        if !target.trim().is_empty() {
            return Ok(target);
        }
    }

    let version = command_output_os(
        OsString::from("rustup"),
        &["run", "nightly", "rustc", "-vV"],
    )
    .map_err(|err| format!("failed to detect nightly host target for fuzz-run: {err}"))?;
    version
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_string)
        .ok_or_else(|| "failed to parse nightly host target from `rustc -vV`".to_string())
}

pub(super) fn miri() -> Result<(), String> {
    run_nightly_cargo(&["miri", "test", "-p", "j2k-core"])?;
    run_nightly_cargo(&["miri", "test", "-p", "j2k-tilecodec"])?;
    run_nightly_cargo(&[
        "miri",
        "test",
        "-p",
        "j2k-native",
        "--no-default-features",
        "inspect::",
    ])
}

pub(super) fn machete() -> Result<(), String> {
    run_program(OsString::from("cargo-machete"), &["--with-metadata"], &[])
}

pub(super) fn no_std() -> Result<(), String> {
    run_program(
        OsString::from("rustup"),
        &["target", "add", NO_STD_TARGET],
        &[],
    )?;
    run_cargo(&["check", "-p", "j2k-core", "--target", NO_STD_TARGET])?;
    run_cargo(&["check", "-p", "j2k-codec-math", "--target", NO_STD_TARGET])?;
    run_cargo(&[
        "check",
        "-p",
        "j2k-profile",
        "--no-default-features",
        "--target",
        NO_STD_TARGET,
    ])?;
    run_cargo(&[
        "check",
        "-p",
        "j2k-native",
        "--no-default-features",
        "--target",
        NO_STD_TARGET,
    ])?;
    run_program(
        OsString::from("rustup"),
        &["target", "add", NO_STD_CORE_PORTABLE_TARGET],
        &[],
    )?;
    run_cargo(&[
        "check",
        "-p",
        "j2k-core",
        "--target",
        NO_STD_CORE_PORTABLE_TARGET,
    ])?;
    run_cargo(&[
        "check",
        "-p",
        "j2k-codec-math",
        "--target",
        NO_STD_CORE_PORTABLE_TARGET,
    ])
}

pub(super) fn verify_unsafe_audit() -> Result<(), String> {
    verify_jpeg_simd_unsafe_boundary()?;

    let audit_path = Path::new("docs/unsafe-audit.md");
    let audit = fs::read_to_string(audit_path)
        .map_err(|err| format!("failed to read {}: {err}", audit_path.display()))?;
    if !audit.contains("| Path | Scope | Invariants | Regression guards |") {
        return Err(
            "docs/unsafe-audit.md must include Path/Scope/Invariants/Regression guards columns"
                .to_string(),
        );
    }
    let mut malformed_rows = Vec::new();
    let mut documented_paths = BTreeSet::new();
    for line in audit.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("| `crates/") {
            continue;
        }
        let cells = trimmed.split('|').map(str::trim).collect::<Vec<_>>();
        if let Some(path) = cells.get(1).and_then(|cell| {
            cell.strip_prefix('`')
                .and_then(|cell| cell.strip_suffix('`'))
        }) {
            documented_paths.insert(path.to_string());
        }
        if cells.len() < 6
            || cells[1].is_empty()
            || cells[2].is_empty()
            || cells[3].is_empty()
            || cells[4].is_empty()
            || cells[3].eq_ignore_ascii_case("tbd")
            || cells[4].eq_ignore_ascii_case("tbd")
        {
            malformed_rows.push(trimmed.to_string());
        }
    }
    if !malformed_rows.is_empty() {
        return Err(format!(
            "docs/unsafe-audit.md has unsafe rows missing invariants or regression guards: {malformed_rows:?}"
        ));
    }
    let mut missing = Vec::new();
    let mut current_unsafe = BTreeSet::new();
    for path in rust_sources(Path::new("crates"))? {
        let source = fs::read_to_string(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        if source_contains_unsafe_rust(&source).map_err(|err| {
            format!(
                "failed to parse {} while scanning for unsafe Rust: {err}",
                path.display()
            )
        })? {
            let relative = path.to_string_lossy().replace('\\', "/");
            current_unsafe.insert(relative);
        }
    }
    for relative in &current_unsafe {
        if !documented_paths.contains(relative) {
            missing.push(relative.clone());
        }
    }
    let stale = documented_paths
        .difference(&current_unsafe)
        .cloned()
        .collect::<Vec<_>>();
    if !stale.is_empty() {
        return Err(format!(
            "docs/unsafe-audit.md has stale unsafe source entries: {stale:?}"
        ));
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "docs/unsafe-audit.md is missing unsafe source entries: {missing:?}"
        ))
    }
}

const JPEG_SIMD_UNSAFE_BLOCK_CAP: usize = 24;
const JPEG_SIMD_UNSAFE_BOUNDARIES: &[&str] = &[
    "crates/j2k-jpeg/src/simd/neon_memory.rs",
    "crates/j2k-jpeg/src/simd/x86.rs",
    "crates/j2k-jpeg/src/simd/x86_memory.rs",
];

fn verify_jpeg_simd_unsafe_boundary() -> Result<(), String> {
    let mut paths = BTreeSet::new();
    for root in [
        "crates/j2k-jpeg/src/backend",
        "crates/j2k-jpeg/src/idct",
        "crates/j2k-jpeg/src/simd",
    ] {
        paths.extend(rust_sources(Path::new(root))?);
    }

    let mut unsafe_blocks = 0usize;
    for path in paths {
        let relative = path.to_string_lossy().replace('\\', "/");
        let source = fs::read_to_string(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let stats = audit_jpeg_simd_source(&relative, &source)?;
        unsafe_blocks += stats.unsafe_blocks;
    }

    if unsafe_blocks > JPEG_SIMD_UNSAFE_BLOCK_CAP {
        return Err(format!(
            "j2k-jpeg SIMD contains {unsafe_blocks} unsafe blocks; cap is {JPEG_SIMD_UNSAFE_BLOCK_CAP}"
        ));
    }

    Ok(())
}

#[derive(Debug, Default, PartialEq, Eq)]
struct JpegSimdUnsafeStats {
    unsafe_blocks: usize,
}

fn audit_jpeg_simd_source(relative: &str, source: &str) -> Result<JpegSimdUnsafeStats, String> {
    let file = syn::parse_file(source)
        .map_err(|err| format!("failed to parse {relative} for the SIMD unsafe ratchet: {err}"))?;
    let mut visitor = JpegSimdUnsafeVisitor::default();
    visitor.visit_file(&file);

    if !visitor.unsafe_functions.is_empty() {
        return Err(format!(
            "{relative} contains SIMD unsafe fn declarations at lines {:?}; use safe token-backed entry points",
            visitor.unsafe_functions
        ));
    }

    if !visitor.unsafe_blocks.is_empty() && !JPEG_SIMD_UNSAFE_BOUNDARIES.contains(&relative) {
        return Err(format!(
            "{relative} contains SIMD unsafe at lines {:?} outside private boundary modules",
            visitor.unsafe_blocks
        ));
    }

    for line in &visitor.unsafe_blocks {
        verify_simd_safety_proof(relative, source, *line)?;
    }

    Ok(JpegSimdUnsafeStats {
        unsafe_blocks: visitor.unsafe_blocks.len(),
    })
}

fn verify_simd_safety_proof(
    relative: &str,
    source: &str,
    unsafe_line: usize,
) -> Result<(), String> {
    let lines = source.lines().collect::<Vec<_>>();
    let start = unsafe_line.saturating_sub(20);
    let end = unsafe_line.min(lines.len());
    let proof = lines[start..end].join("\n").to_ascii_lowercase();
    let required = [
        "safety:",
        "feature availability",
        "bounds",
        "alignment",
        "aliasing",
        "initialization",
    ];
    let missing = required
        .into_iter()
        .filter(|term| !proof.contains(term))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{relative}:{unsafe_line} SIMD unsafe block is missing proof terms: {missing:?}"
        ))
    }
}

#[derive(Default)]
struct JpegSimdUnsafeVisitor {
    unsafe_blocks: Vec<usize>,
    unsafe_functions: Vec<usize>,
}

impl<'ast> Visit<'ast> for JpegSimdUnsafeVisitor {
    fn visit_expr_unsafe(&mut self, expression: &'ast syn::ExprUnsafe) {
        self.unsafe_blocks
            .push(expression.unsafe_token.span.start().line);
        visit::visit_expr_unsafe(self, expression);
    }

    fn visit_item_macro(&mut self, item_macro: &'ast syn::ItemMacro) {
        scan_simd_macro_tokens(
            &item_macro.mac.tokens,
            &mut self.unsafe_blocks,
            &mut self.unsafe_functions,
        );
    }

    fn visit_signature(&mut self, signature: &'ast syn::Signature) {
        if let Some(unsafety) = signature.unsafety {
            self.unsafe_functions.push(unsafety.span.start().line);
        }
        visit::visit_signature(self, signature);
    }
}

fn scan_simd_macro_tokens(
    tokens: &TokenStream,
    unsafe_blocks: &mut Vec<usize>,
    unsafe_functions: &mut Vec<usize>,
) {
    let trees = tokens.clone().into_iter().collect::<Vec<_>>();
    for (index, tree) in trees.iter().enumerate() {
        match tree {
            TokenTree::Group(group) => {
                scan_simd_macro_tokens(&group.stream(), unsafe_blocks, unsafe_functions);
            }
            TokenTree::Ident(ident) if ident == "unsafe" => {
                let is_function = trees
                    .get(index + 1)
                    .is_some_and(|next| matches!(next, TokenTree::Ident(next) if next == "fn"));
                if is_function {
                    unsafe_functions.push(ident.span().start().line);
                } else {
                    unsafe_blocks.push(ident.span().start().line);
                }
            }
            TokenTree::Ident(_) | TokenTree::Literal(_) | TokenTree::Punct(_) => {}
        }
    }
}

fn source_contains_unsafe_rust(source: &str) -> syn::Result<bool> {
    let file = syn::parse_file(source)?;
    let mut detector = UnsafeRustDetector::default();
    detector.visit_file(&file);
    Ok(detector.found)
}

#[derive(Default)]
struct UnsafeRustDetector {
    found: bool,
}

impl<'ast> Visit<'ast> for UnsafeRustDetector {
    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        let contains_unsafe = attribute.path().is_ident("unsafe")
            || match &attribute.meta {
                syn::Meta::List(list) => tokens_contain_unsafe(&list.tokens),
                syn::Meta::Path(_) | syn::Meta::NameValue(_) => false,
            };
        if contains_unsafe {
            self.found = true;
        } else {
            visit::visit_attribute(self, attribute);
        }
    }

    fn visit_expr_unsafe(&mut self, _expression: &'ast syn::ExprUnsafe) {
        self.found = true;
    }

    fn visit_item_foreign_mod(&mut self, _foreign_mod: &'ast syn::ItemForeignMod) {
        // Foreign declarations are an unsafe boundary even in editions where
        // the `unsafe extern` spelling is not mandatory.
        self.found = true;
    }

    fn visit_item_impl(&mut self, item_impl: &'ast syn::ItemImpl) {
        if item_impl.unsafety.is_some() {
            self.found = true;
        } else {
            visit::visit_item_impl(self, item_impl);
        }
    }

    fn visit_item_mod(&mut self, item_mod: &'ast syn::ItemMod) {
        if item_mod.unsafety.is_some() {
            self.found = true;
        } else {
            visit::visit_item_mod(self, item_mod);
        }
    }

    fn visit_item_trait(&mut self, item_trait: &'ast syn::ItemTrait) {
        if item_trait.unsafety.is_some() {
            self.found = true;
        } else {
            visit::visit_item_trait(self, item_trait);
        }
    }

    fn visit_signature(&mut self, signature: &'ast syn::Signature) {
        if signature.unsafety.is_some() {
            self.found = true;
        } else {
            visit::visit_signature(self, signature);
        }
    }

    fn visit_token_stream(&mut self, tokens: &'ast TokenStream) {
        // Syn preserves macro bodies, attribute arguments, and syntax it
        // represents verbatim as token streams. Scan those token trees so an
        // unsafe boundary cannot hide behind a macro or newer Rust syntax.
        if tokens_contain_unsafe(tokens) {
            self.found = true;
        }
    }

    fn visit_type_bare_fn(&mut self, bare_fn: &'ast syn::TypeBareFn) {
        if bare_fn.unsafety.is_some() {
            self.found = true;
        } else {
            visit::visit_type_bare_fn(self, bare_fn);
        }
    }
}

fn tokens_contain_unsafe(tokens: &TokenStream) -> bool {
    tokens.clone().into_iter().any(|token| match token {
        TokenTree::Group(group) => tokens_contain_unsafe(&group.stream()),
        TokenTree::Ident(ident) => ident == "unsafe",
        TokenTree::Literal(_) | TokenTree::Punct(_) => false,
    })
}

fn test_downstream_examples() -> Result<(), String> {
    run_cargo(&["test", "-p", "j2k", "--examples"])?;
    run_cargo(&["test", "-p", "j2k-transcode", "--examples"])
}

#[cfg(test)]
mod unsafe_audit_tests {
    use super::{audit_jpeg_simd_source, source_contains_unsafe_rust};

    #[test]
    fn detects_unsafe_rust_across_supported_syntax() {
        let cases = [
            ("block", "fn boundary(pointer: *const u8) { unsafe\n{ let _ = *pointer; } }"),
            ("free function", "unsafe\nfn boundary() {}"),
            ("unsafe trait", "unsafe trait Boundary {}"),
            (
                "unsafe implementation",
                "trait Boundary {} struct Value; unsafe impl Boundary for Value {}",
            ),
            ("foreign block", "extern \"C\" { fn boundary(); }"),
            (
                "unsafe associated function",
                "trait Boundary { unsafe fn call(); }",
            ),
            (
                "unsafe bare function type",
                "type Boundary = unsafe extern \"C\" fn();",
            ),
            (
                "unsafe attribute",
                "#[unsafe(no_mangle)] pub extern \"C\" fn boundary() {}",
            ),
            (
                "macro-contained unsafe block",
                "macro_rules! boundary { () => { unsafe\n{ core::hint::unreachable_unchecked() } } }",
            ),
        ];

        for (label, source) in cases {
            assert!(
                source_contains_unsafe_rust(source).expect("valid Rust syntax"),
                "failed to detect {label}"
            );
        }
    }

    #[test]
    fn ignores_unsafe_text_in_comments_and_literals() {
        let source = r##"
            // unsafe { comment_only(); }
            /* pub unsafe fn also_comment_only() {} */
            const MESSAGE: &str = "pub unsafe fn string_only()";
            const RAW: &str = r#"unsafe { raw_string_only(); }"#;
            fn unsafe_count() -> usize { 0 }
            pub extern "C" fn safe_callback() {}
            macro_rules! literal_only { () => { "unsafe { macro_literal(); }" } }
        "##;

        assert!(!source_contains_unsafe_rust(source).expect("valid Rust syntax"));
    }

    #[test]
    fn rejects_unparsable_rust_instead_of_skipping_it() {
        let error = source_contains_unsafe_rust("fn incomplete( {")
            .expect_err("invalid Rust must fail the audit scan");
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn jpeg_simd_boundary_rejects_unsafe_functions() {
        let error = audit_jpeg_simd_source(
            "crates/j2k-jpeg/src/simd/x86_memory.rs",
            "pub(crate) unsafe fn load() {}",
        )
        .expect_err("SIMD unsafe functions must be rejected");
        assert!(error.contains("unsafe fn"));
    }

    #[test]
    fn jpeg_simd_boundary_rejects_unsafe_outside_boundary_modules() {
        let error = audit_jpeg_simd_source(
            "crates/j2k-jpeg/src/backend/x86.rs",
            "fn kernel() { unsafe { core::hint::unreachable_unchecked() } }",
        )
        .expect_err("ordinary SIMD modules must not contain unsafe blocks");
        assert!(error.contains("outside private boundary modules"));
    }

    #[test]
    fn jpeg_simd_boundary_requires_the_complete_safety_proof() {
        let error = audit_jpeg_simd_source(
            "crates/j2k-jpeg/src/simd/x86_memory.rs",
            "fn load() { /* SAFETY: bounds only */ unsafe { core::hint::unreachable_unchecked() } }",
        )
        .expect_err("incomplete SIMD safety proofs must be rejected");
        assert!(error.contains("feature availability"));
        assert!(error.contains("alignment"));
        assert!(error.contains("aliasing"));
        assert!(error.contains("initialization"));
    }
}

#[cfg(all(test, unix))]
mod command_tests;
