// SPDX-License-Identifier: MIT OR Apache-2.0

//! Annex G JP2/JPH reader comparisons through the production decoder APIs.

use std::{fs, path::Path};

use j2k::{J2kDecoder, J2kSrgb8Layout};

use crate::{compare_peak_samples, CaseReport, CaseStatus, Jp2Case, JphBset, T803Manifest};

use super::{codestream_requirements, cpu_route, unpack_native_plane, RouteEvidence};

type FileFormatFailure = (String, Box<RouteEvidence>);

pub(in crate::runner) fn run_jp2_cases(manifest: &T803Manifest, corpus: &Path) -> Vec<CaseReport> {
    manifest
        .jp2_cases
        .iter()
        .map(|case| {
            compare_color_case(case, "G.1", corpus)
                .unwrap_or_else(|(error, route)| error_report(case, "G.1", error, *route))
        })
        .collect()
}

pub(in crate::runner) fn run_jph_cases(manifest: &T803Manifest, corpus: &Path) -> Vec<CaseReport> {
    manifest
        .jph_bsets
        .iter()
        .map(|bset| compare_jph_case(bset, manifest, corpus))
        .collect()
}

fn compare_jph_case(bset: &JphBset, manifest: &T803Manifest, corpus: &Path) -> CaseReport {
    let Some(candidate) = bset.selected_candidate() else {
        return error_report_fields(
            &bset.id,
            "G.5",
            bset.peak,
            format!("no JPH candidate is at or below MMAGB {}", bset.mmagb),
            cpu_route(false),
        );
    };
    if let Some(base_id) = &bset.base_jp2_case {
        let Some(base) = manifest.jp2_cases.iter().find(|case| case.id == *base_id) else {
            return error_report_fields(
                &bset.id,
                "G.5",
                bset.peak,
                format!("unknown JP2 comparison {base_id}"),
                cpu_route(false),
            );
        };
        let mut case = base.clone();
        case.id.clone_from(&bset.id);
        case.input.clone_from(&candidate.path);
        return compare_color_case(&case, "G.5", corpus)
            .unwrap_or_else(|(error, route)| error_report(&case, "G.5", error, *route));
    }
    compare_native_jph(bset, &candidate.path, corpus).unwrap_or_else(|(error, route)| {
        error_report_fields(&bset.id, "G.5", bset.peak, error, *route)
    })
}

fn compare_color_case(
    case: &Jp2Case,
    table: &str,
    corpus: &Path,
) -> Result<CaseReport, FileFormatFailure> {
    let route = cpu_route(false);
    let compare = || -> Result<CaseReport, String> {
        let input_path = corpus.join(&case.input);
        let input = fs::read(&input_path)
            .map_err(|error| format!("read {}: {error}", input_path.display()))?;
        let component_transform = codestream_requirements(&input)?.component_transform;
        let support = J2kDecoder::inspect_support(&input).map_err(|error| error.to_string())?;
        if support.component_count() != u16::from(case.components) {
            return Err(format!(
                "codestream has {} components, expected {}",
                support.component_count(),
                case.components
            ));
        }
        let mut iut = J2kDecoder::new(&input).map_err(|error| error.to_string())?;
        let case_route = cpu_route(component_transform.is_some());
        let normalized = iut.decode_srgb8().map_err(|error| error.to_string())?;
        if normalized.dimensions() != (case.width, case.height) {
            return Err(format!(
                "decoded dimensions are {:?}, expected {}x{}",
                normalized.dimensions(),
                case.width,
                case.height
            ));
        }
        let reference_path = corpus.join(&case.reference);
        let reference = image::open(&reference_path)
            .map_err(|error| format!("read {}: {error}", reference_path.display()))?
            .into_rgb8();
        if reference.dimensions() != (case.width, case.height) {
            return Err("TIFF dimensions do not match the pinned case".to_string());
        }
        let reference = reference.into_raw();
        let (expected_samples, actual_samples) = match normalized.layout() {
            J2kSrgb8Layout::Gray => {
                let mut gray = Vec::new();
                gray.try_reserve_exact(reference.len() / 3)
                    .map_err(|_| "cannot allocate Annex G grayscale reference".to_string())?;
                for pixel in reference.chunks_exact(3) {
                    if pixel[0] != pixel[1] || pixel[0] != pixel[2] {
                        return Err(
                            "grayscale TIFF reference contains non-neutral pixels".to_string()
                        );
                    }
                    gray.push(i64::from(pixel[0]));
                }
                let actual_samples = normalized
                    .data()
                    .iter()
                    .map(|&sample| i64::from(sample))
                    .collect();
                (gray, actual_samples)
            }
            J2kSrgb8Layout::Rgb => (
                reference
                    .iter()
                    .map(|&sample| i64::from(sample))
                    .collect::<Vec<_>>(),
                normalized
                    .data()
                    .iter()
                    .map(|&sample| i64::from(sample))
                    .collect::<Vec<_>>(),
            ),
            J2kSrgb8Layout::Rgba => {
                return Err("Annex G case unexpectedly produced alpha".to_string());
            }
            _ => return Err("Annex G case produced an unknown sRGB8 layout".to_string()),
        };
        let comparison = compare_peak_samples(&expected_samples, &actual_samples, case.peak)
            .map_err(|error| error.to_string())?;
        Ok(CaseReport {
            id: case.id.clone(),
            table: table.to_string(),
            status: if comparison.passed {
                CaseStatus::Pass
            } else {
                CaseStatus::Fail
            },
            route: case_route.kind,
            peak: Some(comparison.peak),
            mse: None,
            allowed_peak: case.peak,
            allowed_mse: None,
            error: None,
            stages: case_route.stages,
            accelerator_execution: case_route.accelerator_execution,
            part15: None,
        })
    };
    compare().map_err(|error| (error, Box::new(route)))
}

fn compare_native_jph(
    bset: &JphBset,
    input: &str,
    corpus: &Path,
) -> Result<CaseReport, FileFormatFailure> {
    let route = cpu_route(false);
    let compare = || -> Result<CaseReport, String> {
        let input_path = corpus.join(input);
        let input = fs::read(&input_path)
            .map_err(|error| format!("read {}: {error}", input_path.display()))?;
        let component_transform = codestream_requirements(&input)?.component_transform;
        let mut iut = J2kDecoder::new(&input).map_err(|error| error.to_string())?;
        let decoded = iut
            .decode_native_components_at_reduction(0)
            .map_err(|error| error.to_string())?;
        if decoded.planes().len() != usize::from(bset.components) {
            return Err(format!(
                "decoded output has {} components, expected {}",
                decoded.planes().len(),
                bset.components
            ));
        }
        let mut peak = 0_u64;
        let mut passed = true;
        for (plane, reference) in decoded.planes().iter().zip(&bset.native_references) {
            if plane.dimensions() != (bset.width, bset.height)
                || plane.bit_depth() != bset.bit_depth
                || plane.signed()
            {
                return Err(format!(
                    "native component metadata differs from {}x{} unsigned {}-bit",
                    bset.width, bset.height, bset.bit_depth
                ));
            }
            let reference_path = corpus.join(reference);
            let reference = image::open(&reference_path)
                .map_err(|error| format!("read {}: {error}", reference_path.display()))?
                .into_luma8();
            if reference.dimensions() != (bset.width, bset.height) {
                return Err(format!(
                    "PGM dimensions differ from {}x{}",
                    bset.width, bset.height
                ));
            }
            let expected = reference
                .into_raw()
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>();
            let actual = unpack_native_plane(plane)?;
            let comparison = compare_peak_samples(&expected, &actual, bset.peak)
                .map_err(|error| error.to_string())?;
            peak = peak.max(comparison.peak);
            passed &= comparison.passed;
        }
        let case_route = cpu_route(component_transform.is_some());
        Ok(CaseReport {
            id: bset.id.clone(),
            table: "G.5".to_string(),
            status: if passed {
                CaseStatus::Pass
            } else {
                CaseStatus::Fail
            },
            route: case_route.kind,
            peak: Some(peak),
            mse: None,
            allowed_peak: bset.peak,
            allowed_mse: None,
            error: None,
            stages: case_route.stages,
            accelerator_execution: case_route.accelerator_execution,
            part15: None,
        })
    };
    compare().map_err(|error| (error, Box::new(route)))
}

fn error_report(case: &Jp2Case, table: &str, error: String, route: RouteEvidence) -> CaseReport {
    error_report_fields(&case.id, table, case.peak, error, route)
}

fn error_report_fields(
    id: &str,
    table: &str,
    allowed_peak: u64,
    error: String,
    route: RouteEvidence,
) -> CaseReport {
    CaseReport {
        id: id.to_string(),
        table: table.to_string(),
        status: CaseStatus::Error,
        route: route.kind,
        peak: None,
        mse: None,
        allowed_peak,
        allowed_mse: None,
        error: Some(error),
        stages: route.stages,
        accelerator_execution: route.accelerator_execution,
        part15: None,
    }
}
