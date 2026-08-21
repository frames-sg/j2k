// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural review triggers for production source size and root-module logic.

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};

use syn::spanned::Spanned;

use super::{repo_root, rust_sources};

const ROOT_MODULE_SOFT_LIMIT: usize = 400;
const RUST_MODULE_HARD_LIMIT: usize = 1_200;
const SHADER_MODULE_HARD_LIMIT: usize = 1_500;
const ROOT_FUNCTION_SOFT_LIMIT: usize = 75;
const TOO_MANY_LINES_EXPECTATION_CEILING: usize = 98;

const REVIEWED_LARGE_ROOT_MODULES: &[(&str, usize, &str)] = &[
    (
        "crates/j2k-native/src/lib.rs",
        432,
        "legacy native root retained while public module boundaries stabilize",
    ),
    (
        "crates/j2k-profile/src/lib.rs",
        899,
        "single-purpose profiling contracts and report types",
    ),
    (
        "crates/j2k-transcode-cuda/src/lib.rs",
        921,
        "legacy operational crate root targeted by C1",
    ),
];

const REVIEWED_LARGE_RUST_MODULES: &[(&str, usize, &str)] = &[
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_htj2k_decode/simt/src/main.rs",
        1_397,
        "cohesive no-std CUDA-Oxide HTJ2K decode kernel moved intact from the runtime",
    ),
    (
        "crates/j2k-cuda-j2k-engine/src/cuda_oxide_htj2k_encode/simt/src/main.rs",
        1_968,
        "cohesive no-std CUDA-Oxide HTJ2K encode kernel moved intact from the runtime",
    ),
    (
        "crates/j2k-cuda-jpeg-engine/src/cuda_oxide_jpeg_decode/simt/src/main.rs",
        1_770,
        "cohesive no-std CUDA-Oxide JPEG decode kernel moved intact from the runtime",
    ),
    (
        "crates/j2k-jpeg/src/backend/neon.rs",
        1_645,
        "cohesive architecture-specific JPEG backend",
    ),
    (
        "crates/j2k-metal/src/engine/tier1_encode.rs",
        1_249,
        "review trigger for the active Metal Tier-1 engine",
    ),
];

const REVIEWED_LARGE_SHADER_MODULES: &[(&str, usize, &str)] = &[
    (
        "crates/j2k-jpeg-metal/src/shaders_encode.metal",
        1_810,
        "legacy JPEG encode shader targeted by P18",
    ),
    (
        "crates/j2k-metal/src/encode_bitstream_classic_core.metal",
        1_635,
        "shared classic encode shader core",
    ),
    (
        "crates/j2k-metal/src/encode_bitstream_classic_symbol_plan.metal",
        2_022,
        "classic symbol-plan shader with reviewed cohesive ownership",
    ),
    (
        "crates/j2k-metal/src/encode_bitstream_packetize.metal",
        1_913,
        "packetization shader targeted by P11",
    ),
];

const REVIEWED_LONG_ROOT_FUNCTIONS: &[(&str, &str, usize, &str)] = &[];

fn production_source_size_violations() -> Vec<String> {
    let root = repo_root();
    let crate_roots = publishable_crate_roots(root);
    let rust = production_rust_sources(&crate_roots);
    let shaders = metal_sources(&crate_roots);
    let mut violations = Vec::new();

    check_source_sizes(
        root,
        rust.iter().filter(|path| path.ends_with("src/lib.rs")),
        ROOT_MODULE_SOFT_LIMIT,
        REVIEWED_LARGE_ROOT_MODULES,
        "crate root",
        &mut violations,
    );
    check_source_sizes(
        root,
        rust.iter().filter(|path| !path.ends_with("src/lib.rs")),
        RUST_MODULE_HARD_LIMIT,
        REVIEWED_LARGE_RUST_MODULES,
        "Rust module",
        &mut violations,
    );
    check_source_sizes(
        root,
        shaders.iter(),
        SHADER_MODULE_HARD_LIMIT,
        REVIEWED_LARGE_SHADER_MODULES,
        "Metal shader",
        &mut violations,
    );
    check_root_functions(root, &rust, &mut violations);

    let expectation_count = rust
        .iter()
        .map(|path| {
            fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
                .matches("clippy::too_many_lines")
                .count()
        })
        .sum::<usize>();
    if expectation_count > TOO_MANY_LINES_EXPECTATION_CEILING {
        violations.push(format!(
            "production clippy::too_many_lines expectations increased from ceiling \
             {TOO_MANY_LINES_EXPECTATION_CEILING} to {expectation_count}"
        ));
    }

    violations
}

fn publishable_crate_roots(root: &Path) -> Vec<PathBuf> {
    let mut crates = fs::read_dir(root.join("crates"))
        .expect("read crates directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.join("Cargo.toml").is_file())
        .filter(|path| {
            let manifest = fs::read_to_string(path.join("Cargo.toml"))
                .unwrap_or_else(|error| panic!("read {}/Cargo.toml: {error}", path.display()));
            let manifest = toml::from_str::<toml::Value>(&manifest)
                .unwrap_or_else(|error| panic!("parse {}/Cargo.toml: {error}", path.display()));
            is_publishable_manifest(&manifest)
        })
        .collect::<Vec<_>>();
    crates.sort();
    crates
}

fn is_publishable_manifest(manifest: &toml::Value) -> bool {
    match manifest
        .get("package")
        .and_then(|package| package.get("publish"))
    {
        Some(toml::Value::Boolean(false)) => false,
        Some(toml::Value::Array(registries)) if registries.is_empty() => false,
        None | Some(_) => true,
    }
}

fn production_rust_sources(crate_roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut sources = crate_roots
        .iter()
        .flat_map(|crate_root| rust_sources(&crate_root.join("src")))
        .filter(|path| is_production_rust_source(path))
        .collect::<Vec<_>>();
    sources.sort();
    sources
}

fn is_production_rust_source(path: &Path) -> bool {
    !path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some("tests" | "test_support")
        )
    }) && !path
        .file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| name == "tests.rs" || name.ends_with("_tests.rs"))
}

fn metal_sources(crate_roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    for crate_root in crate_roots {
        collect_metal_sources(crate_root, &mut sources);
    }
    sources.sort();
    sources
}

fn collect_metal_sources(directory: &Path, sources: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
    {
        let path = entry.expect("read directory entry").path();
        if path.is_dir() {
            collect_metal_sources(&path, sources);
        } else if path.extension().and_then(OsStr::to_str) == Some("metal") {
            sources.push(path);
        }
    }
}

fn check_source_sizes<'a>(
    root: &Path,
    paths: impl Iterator<Item = &'a PathBuf>,
    limit: usize,
    reviewed: &[(&str, usize, &str)],
    kind: &str,
    violations: &mut Vec<String>,
) {
    let reviewed = reviewed
        .iter()
        .map(|&(path, ceiling, reason)| (path, (ceiling, reason)))
        .collect::<BTreeMap<_, _>>();
    let mut observed = BTreeSet::new();
    for path in paths {
        let relative = relative_path(root, path);
        let source =
            fs::read_to_string(path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let lines = physical_line_count(&source);
        if lines <= limit {
            continue;
        }
        match reviewed.get(relative.as_str()) {
            Some(&(ceiling, reason)) if lines <= ceiling && !reason.is_empty() => {
                observed.insert(relative);
            }
            Some(&(ceiling, _)) => violations.push(format!(
                "{kind} {relative} has {lines} lines, above reviewed ceiling {ceiling}"
            )),
            None => violations.push(format!(
                "unreviewed {kind} {relative} has {lines} lines, above limit {limit}"
            )),
        }
    }
    for path in reviewed.keys() {
        if !observed.contains(*path) {
            violations.push(format!(
                "stale {kind} size allowance for {path}; remove or lower the reviewed inventory"
            ));
        }
    }
}

fn check_root_functions(root: &Path, rust: &[PathBuf], violations: &mut Vec<String>) {
    let reviewed = REVIEWED_LONG_ROOT_FUNCTIONS
        .iter()
        .map(|&(path, function, ceiling, reason)| ((path, function), (ceiling, reason)))
        .collect::<BTreeMap<_, _>>();
    let mut observed = BTreeSet::new();
    for path in rust.iter().filter(|path| path.ends_with("src/lib.rs")) {
        let relative = relative_path(root, path);
        let source =
            fs::read_to_string(path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let syntax = syn::parse_file(&source)
            .unwrap_or_else(|error| panic!("parse {relative} as Rust: {error}"));
        for function in syntax.items.iter().filter_map(|item| match item {
            syn::Item::Fn(function) => Some(function),
            _ => None,
        }) {
            let span = function.block.span();
            let lines = span
                .end()
                .line
                .saturating_sub(span.start().line)
                .saturating_add(1);
            if lines <= ROOT_FUNCTION_SOFT_LIMIT {
                continue;
            }
            let function_name = function.sig.ident.to_string();
            match reviewed.get(&(relative.as_str(), function_name.as_str())) {
                Some(&(ceiling, reason)) if lines <= ceiling && !reason.is_empty() => {
                    observed.insert((relative.clone(), function_name));
                }
                Some(&(ceiling, _)) => violations.push(format!(
                    "root function {relative}::{function_name} has {lines} lines, above reviewed ceiling {ceiling}"
                )),
                None => violations.push(format!(
                    "unreviewed root function {relative}::{function_name} has {lines} lines, above soft limit {ROOT_FUNCTION_SOFT_LIMIT}"
                )),
            }
        }
    }
    for &(path, function, _, _) in REVIEWED_LONG_ROOT_FUNCTIONS {
        if !observed.contains(&(path.to_owned(), function.to_owned())) {
            violations.push(format!(
                "stale root-function allowance for {path}::{function}; remove or lower it"
            ));
        }
    }
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn physical_line_count(source: &str) -> usize {
    source.lines().count()
}

#[test]
fn production_source_sizes_stay_within_reviewed_limits() {
    let violations = production_source_size_violations();
    assert!(
        violations.is_empty(),
        "production source-size policy violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn j2k_types_root_is_reexport_oriented() {
    let root = repo_root();
    let crate_source = root.join("crates/j2k-types/src");
    let required_modules = [
        "transform/mod.rs",
        "transform/mct.rs",
        "transform/dwt53.rs",
        "transform/dwt97.rs",
        "transform/quantization.rs",
        "tier1/mod.rs",
        "tier1/classic.rs",
        "tier1/htj2k.rs",
        "packetization/mod.rs",
        "packetization/jobs.rs",
        "packetization/progression.rs",
        "dispatch/mod.rs",
        "dispatch/accelerator.rs",
        "dispatch/report.rs",
        "prepared_plan/mod.rs",
    ];
    let missing = required_modules
        .iter()
        .filter(|relative| !crate_source.join(relative).is_file())
        .copied()
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "j2k-types M1 ownership modules are missing: {}",
        missing.join(", ")
    );

    let lib = fs::read_to_string(crate_source.join("lib.rs")).expect("read j2k-types crate root");
    assert!(
        physical_line_count(&lib) <= 160,
        "j2k-types crate root must contain declarations and re-exports, not {} lines of types and logic",
        physical_line_count(&lib)
    );
    assert!(
        !lib.contains("pub struct J2k")
            && !lib.contains("pub enum J2k")
            && !lib.contains("pub trait J2k"),
        "j2k-types public definitions belong in ownership modules"
    );
}

#[test]
fn j2k_native_color_is_split_by_ownership() {
    let root = repo_root();
    let color_source = root.join("crates/j2k-native/src/color");
    let required_modules = [
        "mod.rs",
        "types.rs",
        "metadata.rs",
        "output_planes.rs",
        "allocation.rs",
        "packing.rs",
        "palette.rs",
        "icc.rs",
        "sycc.rs",
        "cielab.rs",
    ];
    let missing = required_modules
        .iter()
        .filter(|relative| !color_source.join(relative).is_file())
        .copied()
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "j2k-native M2 color ownership modules are missing: {}",
        missing.join(", ")
    );
    assert!(
        !root.join("crates/j2k-native/src/color.rs").exists(),
        "the color module root belongs at color/mod.rs"
    );

    let module = fs::read_to_string(color_source.join("mod.rs")).expect("read native color root");
    assert!(
        physical_line_count(&module) <= 160,
        "native color module root must contain declarations and re-exports, not {} lines",
        physical_line_count(&module)
    );
    assert!(
        !module.contains("pub type NativeComponentPlaneParts = (")
            && !module.contains("pub type ComponentPlaneParts<'a> = (")
            && !module.contains("pub struct ColorSpace")
            && !module.contains("pub enum ColorSpace"),
        "native color values and facade handoff parts belong in named ownership modules"
    );
}

#[test]
fn j2k_jpeg_metal_compute_root_delegates_runtime_ownership() {
    let root = repo_root();
    let compute_source = root.join("crates/j2k-jpeg-metal/src/compute");
    let required_modules = [
        "mod.rs",
        "runtime.rs",
        "pipeline_registry.rs",
        "command.rs",
        "status.rs",
        "encode.rs",
        "fast_packets.rs",
        "pack_dispatch.rs",
        "batch_entry.rs",
        "batch_full.rs",
        "batch_region.rs",
        "single_decode.rs",
    ];
    let missing = required_modules
        .iter()
        .filter(|relative| !compute_source.join(relative).is_file())
        .copied()
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "JPEG Metal M3 compute ownership modules are missing: {}",
        missing.join(", ")
    );
    assert!(
        !root.join("crates/j2k-jpeg-metal/src/compute.rs").exists(),
        "the JPEG Metal compute root belongs at compute/mod.rs"
    );

    let module =
        fs::read_to_string(compute_source.join("mod.rs")).expect("read JPEG Metal compute root");
    assert!(
        physical_line_count(&module) <= 220,
        "JPEG Metal compute root must delegate domain ownership, not contain {} lines",
        physical_line_count(&module)
    );
    for forbidden in [
        "struct MetalRuntime",
        "const SHADER_SOURCE",
        "fn new_command_buffer",
        "fn new_compute_command_encoder",
        "fn new_blit_command_encoder",
    ] {
        assert!(
            !module.contains(forbidden),
            "JPEG Metal compute root still owns `{forbidden}`"
        );
    }
}

#[test]
fn j2k_jpeg_metal_codec_batch_separates_semantics_from_execution() {
    let root = repo_root();
    let batch_source = root.join("crates/j2k-jpeg-metal/src/codec_batch");
    let required_modules = [
        "mod.rs",
        "request.rs",
        "source.rs",
        "inspect.rs",
        "plan.rs",
        "owner_accounting.rs",
        "buffer_target.rs",
        "texture_target.rs",
        "submit.rs",
    ];
    let missing = required_modules
        .iter()
        .filter(|relative| !batch_source.join(relative).is_file())
        .copied()
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "JPEG Metal M4 codec-batch ownership modules are missing: {}",
        missing.join(", ")
    );
    assert!(
        !root
            .join("crates/j2k-jpeg-metal/src/codec_batch.rs")
            .exists(),
        "codec-batch semantics belong under codec_batch/mod.rs"
    );
    let module =
        fs::read_to_string(batch_source.join("mod.rs")).expect("read JPEG Metal codec-batch root");
    assert!(
        physical_line_count(&module) <= 100,
        "JPEG Metal codec-batch root must declare focused owners, not contain {} lines",
        physical_line_count(&module)
    );
    assert!(
        !module.contains("struct Rgb8BatchBuildContext")
            && !module.contains("fn build_rgb8_batch_plan")
            && !module.contains("impl Codec"),
        "codec-batch behavior belongs in focused ownership modules"
    );

    let execution = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/batch.rs"))
        .expect("read JPEG Metal execution batch module");
    assert!(
        execution.contains("struct QueuedRequest") && execution.contains("struct MetalSubmission"),
        "batch.rs must remain the queue/execution owner"
    );
}

#[test]
fn accelerator_crate_roots_are_declaration_oriented() {
    let root = repo_root();
    let expectations = [
        (
            "crates/j2k-metal/src/lib.rs",
            130_usize,
            [
                "fn benchmark_region_scaled_direct_plan_prepare",
                "fn benchmark_private_buffer_with_bytes",
            ],
        ),
        (
            "crates/j2k-jpeg-metal/src/lib.rs",
            160_usize,
            [
                "fn decode_surface_from_decoder",
                "fn decode_cpu_request_upload",
            ],
        ),
        (
            "crates/j2k-cuda-runtime/src/lib.rs",
            150_usize,
            [
                "macro_rules! cuda_kernel_params",
                "macro_rules! impl_cuda_htj2k_encoded_status_accessors",
            ],
        ),
    ];
    for (relative, ceiling, forbidden) in expectations {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            physical_line_count(&source) <= ceiling,
            "{relative} must be declaration-oriented, not {} lines",
            physical_line_count(&source)
        );
        for symbol in forbidden {
            assert!(
                !source.contains(symbol),
                "{relative} still owns operational `{symbol}`"
            );
        }
    }
    for required in [
        "crates/j2k-metal/src/bench_support.rs",
        "crates/j2k-jpeg-metal/src/codec.rs",
        "crates/j2k-jpeg-metal/src/decode_surface.rs",
        "crates/j2k-cuda-runtime/src/macros.rs",
    ] {
        assert!(
            root.join(required).is_file(),
            "M5 owner missing: {required}"
        );
    }
}

#[test]
fn jpeg_capabilities_are_split_by_decision_responsibility() {
    let root = repo_root();
    let capabilities = root.join("crates/j2k-jpeg/src/capabilities");
    for owner in [
        "mod.rs",
        "request.rs",
        "output_geometry.rs",
        "cpu.rs",
        "cuda.rs",
        "metal.rs",
        "rejection.rs",
        "resolve.rs",
    ] {
        assert!(
            capabilities.join(owner).is_file(),
            "M6 capability owner missing: {owner}"
        );
    }
    assert!(
        !root.join("crates/j2k-jpeg/src/capabilities.rs").exists(),
        "the capability monolith must be replaced by a module directory"
    );

    let module = fs::read_to_string(capabilities.join("mod.rs"))
        .expect("read JPEG capabilities module root");
    assert!(
        physical_line_count(&module) <= 220,
        "capabilities/mod.rs must coordinate reports rather than own backend rules"
    );
    for (owner, symbol) in [
        ("cpu.rs", "fn cpu_eligibility"),
        ("cuda.rs", "fn owned_cuda_eligibility"),
        ("metal.rs", "fn metal_fast_eligibility"),
        ("output_geometry.rs", "fn output_rect_for_request"),
        ("resolve.rs", "impl JpegResolvedDecode"),
    ] {
        let source = fs::read_to_string(capabilities.join(owner))
            .unwrap_or_else(|error| panic!("read capability owner {owner}: {error}"));
        assert!(source.contains(symbol), "{owner} must own `{symbol}`");
    }
}

#[test]
fn facade_encode_entrypoints_delegate_to_focused_owners() {
    let root = repo_root();
    let encode = root.join("crates/j2k/src/encode");
    for owner in [
        "mod.rs",
        "api.rs",
        "geometry.rs",
        "cpu.rs",
        "accelerator.rs",
        "high_bit.rs",
        "lossless.rs",
        "lossy.rs",
        "roi.rs",
        "validation.rs",
        "tests/mod.rs",
    ] {
        assert!(
            encode.join(owner).is_file(),
            "M7 encode owner missing: {owner}"
        );
    }
    assert!(
        !root.join("crates/j2k/src/encode.rs").exists(),
        "the facade encode monolith must be replaced by a module directory"
    );

    let module = fs::read_to_string(encode.join("mod.rs")).expect("read facade encode root");
    assert!(
        physical_line_count(&module) <= 160,
        "encode/mod.rs must declare and re-export rather than execute"
    );
    let api = fs::read_to_string(encode.join("api.rs")).expect("read facade encode API owner");
    for forbidden in [
        "validate_lossless_roundtrip",
        "validate_lossy_options",
        "encode_cpu_lossy",
        "resolve_accelerated_encode_backend",
        "EncodedJ2k {",
    ] {
        assert!(
            !api.contains(forbidden),
            "encode/api.rs must delegate instead of owning `{forbidden}`"
        );
    }
}

#[test]
fn classic_metal_shader_is_split_by_generated_and_control_flow_ownership() {
    let root = repo_root();
    let classic = root.join("crates/j2k-metal/src/classic");
    for unit in [
        "abi.metal",
        "constants.metal",
        "qe_table.metal",
        "context_tables.metal",
        "support.metal",
        "mq_decoder.metal",
        "bypass_decoder.metal",
        "pass_logic.metal",
        "decode_kernels.metal",
    ] {
        assert!(
            classic.join(unit).is_file(),
            "M8 classic unit missing: {unit}"
        );
    }
    assert!(
        !root.join("crates/j2k-metal/src/classic.metal").exists(),
        "the classic Metal monolith must be removed"
    );

    let qe = fs::read_to_string(classic.join("qe_table.metal")).expect("read classic QE table");
    let contexts = fs::read_to_string(classic.join("context_tables.metal"))
        .expect("read classic context tables");
    assert!(qe.contains("J2K_QE_TABLE[47]"));
    assert!(!qe.contains("kernel void"));
    assert!(contexts.contains("SIGN_CONTEXT_LOOKUP[256]"));
    assert!(contexts.contains("ZERO_CTX_HH_LOOKUP[256]"));
    assert!(!contexts.contains("kernel void"));

    let kernels = fs::read_to_string(classic.join("decode_kernels.metal"))
        .expect("read classic decode kernels");
    assert!(kernels.contains("kernel void j2k_decode_classic_cleanup_batched"));
    let composer = fs::read_to_string(root.join("crates/j2k-metal/src/engine/shader_source.rs"))
        .expect("read Metal shader composer");
    assert!(composer.contains("../classic/abi.metal"));
    assert!(composer.contains("../classic/decode_kernels.metal"));
}

#[test]
fn physical_line_counter_handles_empty_and_unterminated_sources() {
    assert_eq!(physical_line_count(""), 0);
    assert_eq!(physical_line_count("one"), 1);
    assert_eq!(physical_line_count("one\ntwo\n"), 2);
}

#[test]
fn publishability_parser_handles_cargo_private_forms() {
    let default =
        toml::from_str::<toml::Value>("[package]\nname = 'public'").expect("parse public fixture");
    let boolean = toml::from_str::<toml::Value>("[package]\nname = 'private'\npublish = false")
        .expect("parse boolean-private fixture");
    let registries = toml::from_str::<toml::Value>("[package]\nname = 'private'\npublish = []")
        .expect("parse registry-private fixture");

    assert!(is_publishable_manifest(&default));
    assert!(!is_publishable_manifest(&boolean));
    assert!(!is_publishable_manifest(&registries));
}
