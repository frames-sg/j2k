// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use syn::{Attribute, Item};

use self::evidence_modules::collect_evidence_symbols_from_file;
use super::cuda_oxide_paths::{
    is_cuda_oxide_device_rust, is_cuda_oxide_host_scaffold, CUDA_OXIDE_PACKAGE_ROOTS,
};

mod evidence_modules;

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
        "crates/j2k-cuda-j2k-engine/src/kernels.rs",
        "htj2k_decode_and_dequantize_entrypoints_are_stable",
    ),
    primary_evidence(
        "crates/j2k-cuda-j2k-engine/src/kernels.rs",
        "j2k_encode_and_forward_transform_entrypoints_are_stable",
    ),
];

const CUDA_SCAFFOLD_EVIDENCE: &[EvidenceTest] = &[
    primary_evidence(
        "crates/j2k-cuda-j2k-engine/src/kernels.rs",
        "htj2k_decode_and_dequantize_entrypoints_are_stable",
    ),
    primary_evidence(
        "crates/j2k-cuda-j2k-engine/src/kernels.rs",
        "j2k_encode_and_forward_transform_entrypoints_are_stable",
    ),
];

const CUDA_FFI_EVIDENCE: &[EvidenceTest] = &[primary_evidence(
    "crates/j2k-cuda-runtime/src/tests.rs",
    "runtime_raii_primitives_smoke_when_required",
)];

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

const PYTHON_EXTENSION_EVIDENCE: &[EvidenceTest] = &[
    python_primary_evidence(
        "crates/j2k-python/tests/test_adapter.py",
        "test_jpeg_and_jpeg2000_share_a_batch_with_original_indices",
    ),
    python_primary_evidence(
        "crates/j2k-python/tests/test_adapter.py",
        "test_prepared_mixed_batches_preserve_indices_and_errors",
    ),
    python_primary_evidence(
        "crates/j2k-python/tests/test_adapter.py",
        "test_dlpack_transfers_ownership_once_and_handles_unconsumed_capsule",
    ),
    python_primary_evidence(
        "crates/j2k-python/tests/test_adapter.py",
        "test_jpeg_integrity_and_recovery_metadata",
    ),
    python_supplemental_evidence(
        "crates/j2k-python/tests/test_cuda.py",
        "test_mixed_inputs_preserve_indices_and_gpu_export_after_numpy_rejection",
    ),
    python_supplemental_evidence(
        "crates/j2k-python/tests/test_metal.py",
        "test_native_types_layouts_and_requests_match_cpu",
    ),
];

const fn primary_evidence(path: &'static str, name: &'static str) -> EvidenceTest {
    EvidenceTest {
        path,
        name,
        class: EvidenceClass::Primary,
        language: EvidenceLanguage::Rust,
    }
}

const fn supplemental_evidence(path: &'static str, name: &'static str) -> EvidenceTest {
    EvidenceTest {
        path,
        name,
        class: EvidenceClass::Supplemental,
        language: EvidenceLanguage::Rust,
    }
}

const fn python_primary_evidence(path: &'static str, name: &'static str) -> EvidenceTest {
    EvidenceTest {
        path,
        name,
        class: EvidenceClass::Primary,
        language: EvidenceLanguage::Python,
    }
}

const fn python_supplemental_evidence(path: &'static str, name: &'static str) -> EvidenceTest {
    EvidenceTest {
        path,
        name,
        class: EvidenceClass::Supplemental,
        language: EvidenceLanguage::Python,
    }
}

pub(super) const COVERAGE_EXCLUSIONS: &[CoverageExclusion] = &[
    CoverageExclusion {
        id: "cuda-simt-device-rust",
        reason: "CUDA SIMT device Rust is cross-compiled to PTX and cannot be instrumented by host LLVM coverage",
        matcher: ExclusionMatcher::CudaOxideDeviceRust,
        evidence: CUDA_SIMT_EVIDENCE,
    },
    CoverageExclusion {
        id: "cuda-generated-host-scaffold",
        reason: "generated cuda-oxide host project scaffolds contain only the build entry point",
        matcher: ExclusionMatcher::CudaOxideHostScaffold,
        evidence: CUDA_SCAFFOLD_EVIDENCE,
    },
    CoverageExclusion {
        id: "cuda-shared-simt-prelude",
        reason: "shared CUDA SIMT device helpers are included into PTX crates and are not host-instrumentable",
        matcher: ExclusionMatcher::WholeFile {
            path: "crates/j2k-cuda-build-support/src/cuda_oxide_simt_prelude.rs",
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
        id: "generated-codec-math-fragment",
        reason: "generated DWT constants have no executable coverage region and are freshness-checked against their canonical Rust source",
        matcher: ExclusionMatcher::WholeFile {
            path: "crates/j2k-codec-math/generated/dwt97_constants.rs",
        },
        evidence: GENERATED_DWT_EVIDENCE,
    },
    CoverageExclusion {
        id: "python-extension-bindings",
        reason: "PyO3 extension code runs only inside a Python interpreter, which the Rust host coverage lane does not load; the Python adapter suites exercise it",
        matcher: ExclusionMatcher::PathPrefix {
            prefix: "crates/j2k-python/src/",
        },
        evidence: PYTHON_EXTENSION_EVIDENCE,
    },
];

#[derive(Clone, Copy, Debug)]
pub(super) struct EvidenceTest {
    pub(super) path: &'static str,
    pub(super) name: &'static str,
    class: EvidenceClass,
    language: EvidenceLanguage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvidenceClass {
    Primary,
    Supplemental,
}

/// Rust evidence is a `#[test]` function; Python evidence is a `unittest.TestCase` method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvidenceLanguage {
    Rust,
    Python,
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
    CudaOxideDeviceRust,
    CudaOxideHostScaffold,
    WholeFile {
        path: &'static str,
    },
    /// Every file below one crate source directory; `prefix` ends with `/src/`.
    PathPrefix {
        prefix: &'static str,
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
        ExclusionMatcher::CudaOxideDeviceRust => Ok(is_cuda_oxide_device_rust(path)),
        ExclusionMatcher::CudaOxideHostScaffold => Ok(is_cuda_oxide_host_scaffold(path)),
        ExclusionMatcher::WholeFile { path: exact } => Ok(path == exact),
        ExclusionMatcher::PathPrefix { prefix } => Ok(path.starts_with(prefix)),
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
            let validation = match evidence.language {
                EvidenceLanguage::Rust => {
                    validate_evidence_test_path(root, evidence.path, evidence.name, evidence.class)
                }
                EvidenceLanguage::Python => validate_python_evidence_test_path(
                    root,
                    evidence.path,
                    evidence.name,
                    evidence.class,
                ),
            };
            validation.map_err(|error| {
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

fn validate_evidence_test_path(
    root: &Path,
    path: &str,
    name: &str,
    expected_class: EvidenceClass,
) -> Result<(), String> {
    let mut matches = EvidenceSymbolMatches::default();
    collect_evidence_symbols_from_file(root, Path::new(path), name, false, &mut matches)?;
    validate_evidence_symbol_matches(&matches, expected_class)
}

fn validate_python_evidence_test_path(
    root: &Path,
    path: &str,
    name: &str,
    expected_class: EvidenceClass,
) -> Result<(), String> {
    let source = fs::read_to_string(root.join(path))
        .map_err(|err| format!("failed to read Python evidence source `{path}`: {err}"))?;
    validate_python_evidence_test_source(name, expected_class, &source)
}

/// Resolves one `unittest.TestCase` method. A `skipIf`/`skipUnless` decorator on the method or
/// its class makes the test conditional, so it can only be supplemental evidence.
fn validate_python_evidence_test_source(
    name: &str,
    expected_class: EvidenceClass,
    source: &str,
) -> Result<(), String> {
    if !name.starts_with("test_") {
        return Err("Python evidence must be a unittest `test_` method".to_string());
    }
    let lines = source.lines().collect::<Vec<_>>();
    let definition = format!("    def {name}(self");
    let definitions = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| line.starts_with(&definition).then_some(index))
        .collect::<Vec<_>>();
    let method = match definitions.as_slice() {
        [method] => *method,
        [] => return Err("no matching Python unittest method exists".to_string()),
        many => {
            return Err(format!(
                "{} matching Python unittest methods are ambiguous",
                many.len()
            ))
        }
    };
    let class = lines[..method]
        .iter()
        .rposition(|line| line.starts_with("class "))
        .ok_or_else(|| "matching Python method is not inside a test class".to_string())?;
    if !lines[class].contains("(unittest.TestCase)") {
        return Err("matching Python method is not in a unittest.TestCase class".to_string());
    }
    let mut conditional = false;
    for decorator in decorators_of(&lines, class, "").chain(decorators_of(&lines, method, "    ")) {
        if decorator.contains("unittest.skip(") || decorator.contains("expectedFailure") {
            return Err(
                "coverage evidence tests must not be skipped or expected to fail".to_string(),
            );
        }
        conditional |= decorator.contains("skipIf") || decorator.contains("skipUnless");
    }
    let observed_class = if conditional {
        EvidenceClass::Supplemental
    } else {
        EvidenceClass::Primary
    };
    if observed_class != expected_class {
        return Err(format!(
            "matching Python method is {} but registered as {} evidence",
            evidence_class_description(observed_class),
            evidence_class_description(expected_class)
        ));
    }
    Ok(())
}

/// Decorator lines at `indent` between the previous definition at that indent and `index`. The
/// first line of a multi-line decorator is included, so a wrapped skip still counts.
fn decorators_of<'a>(
    lines: &'a [&'a str],
    index: usize,
    indent: &'static str,
) -> impl Iterator<Item = &'a str> + 'a {
    let block_start = lines[..index]
        .iter()
        .rposition(|line| {
            line.strip_prefix(indent)
                .is_some_and(|rest| rest.starts_with("def ") || rest.starts_with("class "))
        })
        .map_or(0, |previous| previous + 1);
    lines[block_start..index]
        .iter()
        .filter(move |line| {
            line.strip_prefix(indent)
                .is_some_and(|rest| rest.starts_with('@'))
        })
        .copied()
}

#[cfg(test)]
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
    validate_evidence_symbol_matches(&matches, expected_class)
}

fn validate_evidence_symbol_matches(
    matches: &EvidenceSymbolMatches,
    expected_class: EvidenceClass,
) -> Result<(), String> {
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
        ExclusionMatcher::CudaOxideDeviceRust | ExclusionMatcher::CudaOxideHostScaffold => {
            let mut matched = false;
            for package_root in CUDA_OXIDE_PACKAGE_ROOTS {
                if collect_rust_files(&root.join(package_root).join("src"), root)?
                    .into_iter()
                    .any(|path| exclusion_matches(exclusion, &path, 1, &[]).unwrap_or(false))
                {
                    matched = true;
                    break;
                }
            }
            if !matched {
                return Err(format!(
                    "coverage exclusion `{}` CUDA-Oxide matcher matches no Rust file",
                    exclusion.id
                ));
            }
        }
        ExclusionMatcher::PathPrefix { prefix } => {
            let directory = prefix.strip_suffix('/').unwrap_or(prefix);
            if !prefix.ends_with("/src/")
                || collect_rust_files(&root.join(directory), root)?.is_empty()
            {
                return Err(format!(
                    "coverage exclusion `{}` prefix `{prefix}` must name a crate source directory with Rust files",
                    exclusion.id
                ));
            }
        }
        ExclusionMatcher::WholeFile { path } => {
            if !root.join(path).is_file() {
                return Err(format!(
                    "coverage exclusion `{}` file `{path}` is missing",
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
