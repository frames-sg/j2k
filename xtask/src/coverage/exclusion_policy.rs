// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use syn::{Attribute, Item};

const CUDA_SIMT_EVIDENCE: &[EvidenceTest] = &[
    supplemental_evidence(
        "crates/j2k-cuda-runtime/src/tests.rs",
        "cuda_oxide_copy_u8_matches_builtin_copy_and_cpu_when_required",
    ),
    supplemental_evidence(
        "crates/j2k-cuda/tests/htj2k_encode_parity.rs",
        "cuda_facade_byte_matches_native_across_matrix_when_required",
    ),
    supplemental_evidence(
        "crates/j2k-transcode-cuda/tests/jpeg_to_htj2k.rs",
        "ycbcr_420_batch_transcodes_to_htj2k_with_explicit_cuda_97_codeblock_path",
    ),
    primary_evidence(
        "crates/j2k-cuda-runtime/src/tests.rs",
        "kernel_module_names_cover_htj2k_decode_and_encode_stages",
    ),
];

const CUDA_SCAFFOLD_EVIDENCE: &[EvidenceTest] = &[primary_evidence(
    "crates/j2k-cuda-runtime/src/tests.rs",
    "kernel_module_names_cover_htj2k_decode_and_encode_stages",
)];

const CUDA_FFI_EVIDENCE: &[EvidenceTest] = &[primary_evidence(
    "crates/j2k-cuda-runtime/src/tests.rs",
    "runtime_raii_primitives_smoke_when_required",
)];

const METAL_SHADER_EVIDENCE: &[EvidenceTest] = &[
    primary_evidence(
        "crates/j2k-metal/tests/shader_integrity.rs",
        "metal_kernels_are_wired_to_host_pipelines",
    ),
    supplemental_evidence(
        "crates/j2k-metal/tests/device.rs",
        "full_classic_grayscale_decode_to_metal_matches_host_decode",
    ),
];

const GENERATED_DWT_EVIDENCE: &[EvidenceTest] = &[
    primary_evidence(
        "crates/j2k-codec-math/tests/generated_freshness.rs",
        "metal_dwt97_fragment_matches_rust_constants",
    ),
    primary_evidence(
        "crates/j2k-codec-math/tests/generated_freshness.rs",
        "rust_dwt97_fragment_matches_rust_constants",
    ),
];

const VENDORED_BLOCK_EVIDENCE: &[EvidenceTest] = &[
    primary_evidence(
        "xtask/tests/repo_lint_support/dependency_policy.rs",
        "patched_block_dependency_has_pinned_provenance_and_documented_abi_delta",
    ),
    supplemental_evidence(
        "crates/j2k-metal-support/src/tests.rs",
        "commit_and_wait_accepts_unlabeled_command_buffer",
    ),
    supplemental_evidence(
        "crates/j2k-metal-support/src/tests.rs",
        "buffer_readback_copies_typed_shared_buffer_values",
    ),
];

const fn primary_evidence(path: &'static str, name: &'static str) -> EvidenceTest {
    EvidenceTest {
        path,
        name,
        class: EvidenceClass::Primary,
    }
}

const fn supplemental_evidence(path: &'static str, name: &'static str) -> EvidenceTest {
    EvidenceTest {
        path,
        name,
        class: EvidenceClass::Supplemental,
    }
}

pub(super) const COVERAGE_EXCLUSIONS: &[CoverageExclusion] = &[
    CoverageExclusion {
        id: "cuda-simt-device-rust",
        reason: "CUDA SIMT device Rust is cross-compiled to PTX and cannot be instrumented by host LLVM coverage",
        matcher: ExclusionMatcher::PathPattern {
            prefix: "crates/j2k-cuda-runtime/src/cuda_oxide_",
            contains: Some("/simt/src/"),
            excludes: None,
            suffix: ".rs",
        },
        evidence: CUDA_SIMT_EVIDENCE,
    },
    CoverageExclusion {
        id: "cuda-generated-host-scaffold",
        reason: "generated cuda-oxide host project scaffolds contain only the build entry point",
        matcher: ExclusionMatcher::PathPattern {
            prefix: "crates/j2k-cuda-runtime/src/cuda_oxide_",
            contains: Some("/src/"),
            excludes: Some("/simt/"),
            suffix: "main.rs",
        },
        evidence: CUDA_SCAFFOLD_EVIDENCE,
    },
    CoverageExclusion {
        id: "cuda-shared-simt-prelude",
        reason: "shared CUDA SIMT device helpers are included into PTX crates and are not host-instrumentable",
        matcher: ExclusionMatcher::WholeFile {
            path: "crates/j2k-cuda-runtime/src/cuda_oxide_simt_prelude.rs",
        },
        evidence: CUDA_SIMT_EVIDENCE,
    },
    CoverageExclusion {
        id: "cuda-driver-ffi-declarations",
        reason: "CUDA Driver and NVTX FFI type declarations have no executable host coverage region",
        matcher: ExclusionMatcher::MarkerSpan {
            path: "crates/j2k-cuda-runtime/src/driver.rs",
            start: "pub(crate) type CuResult = c_int;",
            end: "pub(crate) type NvtxRangePop = unsafe extern \"C\" fn() -> c_int;",
        },
        evidence: CUDA_FFI_EVIDENCE,
    },
    CoverageExclusion {
        id: "metal-embedded-shader-body",
        reason: "the embedded Metal shader body is MSL text, not executable host Rust",
        matcher: ExclusionMatcher::MarkerSpan {
            path: "crates/j2k-metal/src/compute/shader_source.rs",
            start: "        r\"",
            end: "\",",
        },
        evidence: METAL_SHADER_EVIDENCE,
    },
    CoverageExclusion {
        id: "generated-codec-math-fragment",
        reason: "generated DWT constants have no executable coverage region and are freshness-checked against their canonical Rust source",
        matcher: ExclusionMatcher::WholeFile {
            path: "crates/j2k-codec-math/generated/dwt97_constants.rs",
        },
        evidence: GENERATED_DWT_EVIDENCE,
    },
    CoverageExclusion {
        id: "vendored-block-ffi-binding",
        reason: "the reviewed patched block dependency is outside workspace instrumentation; real Metal tests exercise its callback and lifecycle boundary",
        matcher: ExclusionMatcher::WholeFile {
            path: "third_party/block-0.1.6-patched/src/lib.rs",
        },
        evidence: VENDORED_BLOCK_EVIDENCE,
    },
];

#[derive(Clone, Copy, Debug)]
pub(super) struct EvidenceTest {
    pub(super) path: &'static str,
    pub(super) name: &'static str,
    class: EvidenceClass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvidenceClass {
    Primary,
    Supplemental,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CoverageExclusion {
    pub(super) id: &'static str,
    pub(super) reason: &'static str,
    pub(super) matcher: ExclusionMatcher,
    pub(super) evidence: &'static [EvidenceTest],
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ExclusionMatcher {
    WholeFile {
        path: &'static str,
    },
    PathPattern {
        prefix: &'static str,
        contains: Option<&'static str>,
        excludes: Option<&'static str>,
        suffix: &'static str,
    },
    MarkerSpan {
        path: &'static str,
        start: &'static str,
        end: &'static str,
    },
}

pub(super) fn matching_exclusion(
    path: &str,
    line_number: usize,
    source: &[&str],
) -> Result<Option<&'static CoverageExclusion>, String> {
    for exclusion in COVERAGE_EXCLUSIONS {
        if exclusion_matches(exclusion, path, line_number, source)? {
            return Ok(Some(exclusion));
        }
    }
    Ok(None)
}

fn exclusion_matches(
    exclusion: &CoverageExclusion,
    path: &str,
    line_number: usize,
    source: &[&str],
) -> Result<bool, String> {
    match exclusion.matcher {
        ExclusionMatcher::WholeFile { path: exact } => Ok(path == exact),
        ExclusionMatcher::PathPattern {
            prefix,
            contains,
            excludes,
            suffix,
        } => Ok(path.starts_with(prefix)
            && contains.is_none_or(|needle| path.contains(needle))
            && excludes.is_none_or(|needle| !path.contains(needle))
            && path.ends_with(suffix)),
        ExclusionMatcher::MarkerSpan {
            path: exact,
            start,
            end,
        } => {
            if path != exact {
                return Ok(false);
            }
            let start_line = unique_marker_line(source, start, exclusion.id)?;
            let end_line = unique_marker_line(source, end, exclusion.id)?;
            if start_line > end_line {
                return Err(format!(
                    "coverage exclusion `{}` marker order is invalid",
                    exclusion.id
                ));
            }
            Ok((start_line..=end_line).contains(&line_number))
        }
    }
}

fn unique_marker_line(source: &[&str], marker: &str, id: &str) -> Result<usize, String> {
    let matches = source
        .iter()
        .enumerate()
        .filter_map(|(index, line)| (*line == marker).then_some(index + 1))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [line] => Ok(*line),
        [] => Err(format!(
            "coverage exclusion `{id}` marker `{marker}` is missing"
        )),
        _ => Err(format!(
            "coverage exclusion `{id}` marker `{marker}` is ambiguous"
        )),
    }
}

pub(super) fn validate_exclusion_policy(root: &Path) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for exclusion in COVERAGE_EXCLUSIONS {
        if !ids.insert(exclusion.id) {
            return Err(format!(
                "duplicate coverage exclusion id `{}`",
                exclusion.id
            ));
        }
        if exclusion.reason.trim().is_empty() || exclusion.evidence.is_empty() {
            return Err(format!(
                "coverage exclusion `{}` needs a reason and evidence tests",
                exclusion.id
            ));
        }
        require_primary_evidence(exclusion)?;
        for evidence in exclusion.evidence {
            let source = fs::read_to_string(root.join(evidence.path)).map_err(|err| {
                format!(
                    "coverage exclusion `{}` evidence {} is unavailable: {err}",
                    exclusion.id, evidence.path
                )
            })?;
            validate_evidence_test_source(evidence.path, evidence.name, evidence.class, &source)
                .map_err(|error| {
                    format!(
                        "coverage exclusion `{}` has invalid evidence test `{}::{}`: {error}",
                        exclusion.id, evidence.path, evidence.name
                    )
                })?;
        }
        validate_exclusion_matcher(root, exclusion)?;
    }
    Ok(())
}

fn require_primary_evidence(exclusion: &CoverageExclusion) -> Result<(), String> {
    if exclusion
        .evidence
        .iter()
        .any(|evidence| evidence.class == EvidenceClass::Primary)
    {
        return Ok(());
    }
    Err(format!(
        "coverage exclusion `{}` needs at least one unconditional primary evidence test",
        exclusion.id
    ))
}

#[derive(Default)]
struct EvidenceSymbolMatches {
    count: usize,
    direct_test: bool,
    ignored_or_should_panic: bool,
    conditional: bool,
}

fn validate_evidence_test_source(
    path: &str,
    name: &str,
    expected_class: EvidenceClass,
    source: &str,
) -> Result<(), String> {
    let file = syn::parse_file(source)
        .map_err(|error| format!("failed to parse evidence source `{path}`: {error}"))?;
    let mut matches = EvidenceSymbolMatches::default();
    collect_evidence_symbols(
        &file.items,
        name,
        enclosing_cfg_is_conditional(&file.attrs),
        &mut matches,
    );
    match matches.count {
        0 => return Err("no matching Rust function symbol exists".to_string()),
        1 => {}
        count => {
            return Err(format!(
                "{count} matching Rust function symbols are ambiguous"
            ))
        }
    }
    if !matches.direct_test {
        return Err("matching function is not directly annotated with #[test]".to_string());
    }
    if matches.ignored_or_should_panic {
        return Err(
            "coverage evidence tests must not be ignored or use #[should_panic]".to_string(),
        );
    }
    let observed_class = if matches.conditional {
        EvidenceClass::Supplemental
    } else {
        EvidenceClass::Primary
    };
    if observed_class != expected_class {
        return Err(format!(
            "matching function is {} but registered as {} evidence",
            evidence_class_description(observed_class),
            evidence_class_description(expected_class)
        ));
    }
    Ok(())
}

fn collect_evidence_symbols(
    items: &[Item],
    expected_name: &str,
    inherited_conditional: bool,
    matches: &mut EvidenceSymbolMatches,
) {
    for item in items {
        match item {
            Item::Fn(function) if function.sig.ident == expected_name => {
                matches.count += 1;
                matches.direct_test |= function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("test"));
                matches.ignored_or_should_panic |= function.attrs.iter().any(|attribute| {
                    attribute.path().is_ident("ignore") || attribute.path().is_ident("should_panic")
                });
                matches.conditional |= inherited_conditional
                    || function
                        .attrs
                        .iter()
                        .any(attribute_is_conditional_compilation);
            }
            Item::Mod(module) => {
                if let Some((_, nested)) = &module.content {
                    collect_evidence_symbols(
                        nested,
                        expected_name,
                        inherited_conditional || enclosing_cfg_is_conditional(&module.attrs),
                        matches,
                    );
                }
            }
            _ => {}
        }
    }
}

fn enclosing_cfg_is_conditional(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("cfg_attr")
            || (attribute.path().is_ident("cfg") && !is_exact_cfg_test(attribute))
    })
}

fn attribute_is_conditional_compilation(attribute: &Attribute) -> bool {
    attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr")
}

fn is_exact_cfg_test(attribute: &Attribute) -> bool {
    attribute.path().is_ident("cfg")
        && attribute
            .parse_args::<syn::Path>()
            .is_ok_and(|path| path.is_ident("test"))
}

const fn evidence_class_description(class: EvidenceClass) -> &'static str {
    match class {
        EvidenceClass::Primary => "unconditional primary",
        EvidenceClass::Supplemental => "conditionally compiled supplemental",
    }
}

fn validate_exclusion_matcher(root: &Path, exclusion: &CoverageExclusion) -> Result<(), String> {
    match exclusion.matcher {
        ExclusionMatcher::WholeFile { path } => {
            if !root.join(path).is_file() {
                return Err(format!(
                    "coverage exclusion `{}` file `{path}` is missing",
                    exclusion.id
                ));
            }
        }
        ExclusionMatcher::PathPattern { .. } => {
            let cuda_runtime_src = root.join("crates/j2k-cuda-runtime/src");
            let matched = collect_rust_files(&cuda_runtime_src, root)?
                .into_iter()
                .any(|path| exclusion_matches(exclusion, &path, 1, &[]).unwrap_or(false));
            if !matched {
                return Err(format!(
                    "coverage exclusion `{}` path pattern matches no Rust file",
                    exclusion.id
                ));
            }
        }
        ExclusionMatcher::MarkerSpan { path, .. } => {
            let source_path = root.join(path);
            let source = fs::read_to_string(&source_path).map_err(|err| {
                format!(
                    "coverage exclusion `{}` cannot read {}: {err}",
                    exclusion.id,
                    source_path.display()
                )
            })?;
            let lines = source.lines().collect::<Vec<_>>();
            let matched = (1..=lines.len())
                .any(|line| exclusion_matches(exclusion, path, line, &lines).unwrap_or(false));
            if !matched {
                return Err(format!(
                    "coverage exclusion `{}` marker span matches no line",
                    exclusion.id
                ));
            }
        }
    }
    Ok(())
}

fn collect_rust_files(directory: &Path, root: &Path) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(current) = pending.pop() {
        for entry in fs::read_dir(&current)
            .map_err(|err| format!("failed to inspect {}: {err}", current.display()))?
        {
            let entry = entry.map_err(|err| format!("failed to inspect directory entry: {err}"))?;
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(
                    path.strip_prefix(root)
                        .map_err(|err| format!("failed to normalize {}: {err}", path.display()))?
                        .components()
                        .map(|part| part.as_os_str().to_string_lossy())
                        .collect::<Vec<_>>()
                        .join("/"),
                );
            }
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests;
