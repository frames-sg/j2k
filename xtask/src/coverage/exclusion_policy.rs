// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const CUDA_SIMT_EVIDENCE: &[EvidenceTest] = &[
    EvidenceTest {
        path: "crates/j2k-cuda-runtime/src/tests.rs",
        name: "cuda_oxide_copy_u8_matches_builtin_copy_and_cpu_when_required",
    },
    EvidenceTest {
        path: "crates/j2k-cuda/tests/htj2k_encode_parity.rs",
        name: "cuda_facade_byte_matches_native_across_matrix_when_required",
    },
    EvidenceTest {
        path: "crates/j2k-transcode-cuda/tests/jpeg_to_htj2k.rs",
        name: "ycbcr_420_batch_transcodes_to_htj2k_with_explicit_cuda_97_codeblock_path",
    },
];

const CUDA_SCAFFOLD_EVIDENCE: &[EvidenceTest] = &[EvidenceTest {
    path: "crates/j2k-cuda-runtime/src/tests.rs",
    name: "kernel_module_names_cover_htj2k_decode_and_encode_stages",
}];

const CUDA_FFI_EVIDENCE: &[EvidenceTest] = &[EvidenceTest {
    path: "crates/j2k-cuda-runtime/src/tests.rs",
    name: "runtime_raii_primitives_smoke_when_required",
}];

const METAL_SHADER_EVIDENCE: &[EvidenceTest] = &[
    EvidenceTest {
        path: "crates/j2k-metal/tests/shader_integrity.rs",
        name: "metal_kernels_are_wired_to_host_pipelines",
    },
    EvidenceTest {
        path: "crates/j2k-metal/tests/device.rs",
        name: "full_classic_grayscale_decode_to_metal_matches_host_decode",
    },
];

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
];

#[derive(Clone, Copy, Debug)]
pub(super) struct EvidenceTest {
    pub(super) path: &'static str,
    pub(super) name: &'static str,
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
        for evidence in exclusion.evidence {
            let source = fs::read_to_string(root.join(evidence.path)).map_err(|err| {
                format!(
                    "coverage exclusion `{}` evidence {} is unavailable: {err}",
                    exclusion.id, evidence.path
                )
            })?;
            if !source.contains(&format!("fn {}(", evidence.name)) {
                return Err(format!(
                    "coverage exclusion `{}` evidence test `{}::{}` is missing",
                    exclusion.id, evidence.path, evidence.name
                ));
            }
        }
        validate_exclusion_matcher(root, exclusion)?;
    }
    Ok(())
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
