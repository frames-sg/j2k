// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CaseReport, T803Manifest, T803Suite};

pub(super) fn verify_decoder_evidence(
    iut_name: &str,
    suite: T803Suite,
    claim: &str,
    cases: &[CaseReport],
    manifest: &T803Manifest,
) -> Result<(), String> {
    if suite != T803Suite::All {
        return Err(format!(
            "{iut_name} release report must use T.803 suite all"
        ));
    }
    let required_claim_points = [
        "Profile-1 Cclass-1;",
        "Profile-1 Cclass-1HF",
        "Annex G JP2 reader",
        "HTJ2K DS1-HM Cclass-1h, MMAGB 15",
        "including DS1-HT, DS0-HM, and DS0-HT subset evidence",
        "HTJ2K Cclass-1HFh, MMAGB 20",
        "Annex G JPH reader",
    ];
    if required_claim_points
        .iter()
        .any(|required| !claim.contains(required))
        || ["full Part 1", "full Part 15"]
            .iter()
            .any(|forbidden| claim.contains(forbidden))
    {
        return Err(format!("{iut_name} report uses an invalid claim label"));
    }

    let decoder_cases = manifest
        .decoder_cases_for_suite(T803Suite::All)
        .map_err(|error| error.to_string())?;
    let expected_count = decoder_cases.len() + manifest.jp2_cases.len() + manifest.jph_bsets.len();
    if cases.len() != expected_count {
        return Err(format!(
            "{iut_name} report contains {} cases, expected {expected_count}",
            cases.len()
        ));
    }
    let (observed_decoder, observed_file_formats) = cases.split_at(decoder_cases.len());
    for (observed, expected) in observed_decoder.iter().zip(&decoder_cases) {
        if observed.id != expected.id
            || observed.table != expected.table
            || observed.allowed_peak != expected.peak
            || observed.allowed_mse != Some(expected.mse)
        {
            return Err(format!(
                "{iut_name} report case {} differs from the pinned decoder matrix",
                observed.id
            ));
        }
        if observed.part15.as_ref().map(|part15| &part15.selection) != expected.part15.as_ref() {
            return Err(format!(
                "{iut_name} report case {} differs from the pinned Part 15 selection",
                observed.id
            ));
        }
    }
    let (classic_readers, ht_readers) = observed_file_formats.split_at(manifest.jp2_cases.len());
    for (observed, expected) in classic_readers.iter().zip(&manifest.jp2_cases) {
        if observed.id != expected.id
            || observed.table != "G.1"
            || observed.allowed_peak != expected.peak
            || observed.allowed_mse.is_some()
        {
            return Err(format!(
                "{iut_name} report case {} differs from the pinned Annex G JP2 matrix",
                observed.id
            ));
        }
    }
    for (observed, expected) in ht_readers.iter().zip(&manifest.jph_bsets) {
        if observed.id != expected.id
            || observed.table != "G.5"
            || observed.allowed_peak != expected.peak
            || observed.allowed_mse.is_some()
        {
            return Err(format!(
                "{iut_name} report case {} differs from the pinned Annex G JPH matrix",
                observed.id
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CaseStatus, HtCodeBlockSetMode, Part15CaseEvidence, Part15CodestreamEvidence,
        Part15EvidenceClassification, RouteKind,
    };

    const PART1_CLAIM: &str =
        "Profile-1 Cclass-1; Profile-1 Cclass-1HF; Annex G JP2 reader (candidate evidence)";
    const ALL_CLAIM: &str = "Profile-1 Cclass-1; Profile-1 Cclass-1HF; Annex G JP2 reader; HTJ2K DS1-HM Cclass-1h, MMAGB 15, including DS1-HT, DS0-HM, and DS0-HT subset evidence; HTJ2K Cclass-1HFh, MMAGB 20; Annex G JPH reader at MMAGB 15 (candidate evidence)";

    #[test]
    fn all_suite_requires_the_part15_decoder_and_jph_inventory() {
        let manifest = committed_manifest();
        let cases = all_cases(&manifest);

        verify_decoder_evidence("j2k", T803Suite::All, ALL_CLAIM, &cases, &manifest)
            .expect("the release verifier must accept the complete all-suite inventory");
    }

    #[test]
    fn all_suite_requires_the_exact_part15_claim_points() {
        let manifest = committed_manifest();
        let cases = all_cases(&manifest);

        let error = verify_decoder_evidence("j2k", T803Suite::All, PART1_CLAIM, &cases, &manifest)
            .expect_err("all-suite evidence must identify its formal Part 15 claim points");
        assert!(error.contains("claim label"), "{error}");
    }

    #[test]
    fn all_suite_requires_the_standalone_cclass1_claim() {
        let manifest = committed_manifest();
        let cases = all_cases(&manifest);
        let claim = ALL_CLAIM.replacen("Profile-1 Cclass-1; ", "", 1);

        let error = verify_decoder_evidence("j2k", T803Suite::All, &claim, &cases, &manifest)
            .expect_err("Cclass-1HF must not stand in for the distinct Cclass-1 claim");
        assert!(error.contains("claim label"), "{error}");
    }

    #[test]
    fn all_suite_rejects_part15_selection_tampering() {
        let manifest = committed_manifest();
        let mut cases = all_cases(&manifest);
        let altered = cases
            .iter_mut()
            .find_map(|case| case.part15.as_mut())
            .expect("Part 15 case evidence");
        altered.selection.bmagb += 1;

        let error = verify_decoder_evidence("j2k", T803Suite::All, ALL_CLAIM, &cases, &manifest)
            .expect_err("release evidence must retain the selected BSET metadata");
        assert!(error.contains("Part 15 selection"), "{error}");
    }

    #[test]
    fn release_verification_rejects_a_part1_only_report() {
        let manifest = committed_manifest();
        let mut cases = manifest
            .decoder_cases_for_suite(T803Suite::Part1)
            .expect("resolve Part 1 decoder cases")
            .into_iter()
            .map(|case| case_report(&case.id, &case.table, case.peak, Some(case.mse), None))
            .collect::<Vec<_>>();
        cases.extend(
            manifest
                .jp2_cases
                .iter()
                .map(|case| case_report(&case.id, "G.1", case.peak, None, None)),
        );

        let error =
            verify_decoder_evidence("j2k", T803Suite::Part1, PART1_CLAIM, &cases, &manifest)
                .expect_err("release evidence must cover Part 1 and Part 15 together");
        assert!(error.contains("suite all"), "{error}");
    }

    fn committed_manifest() -> T803Manifest {
        T803Manifest::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../corpus/j2k-conformance/t803-v3.toml"
        )))
        .expect("valid committed manifest")
    }

    fn all_cases(manifest: &T803Manifest) -> Vec<CaseReport> {
        let mut cases = manifest
            .decoder_cases_for_suite(T803Suite::All)
            .expect("resolve all decoder cases")
            .into_iter()
            .map(|case| {
                let part15 = case.part15.map(|selection| {
                    let bmagb = selection.bmagb;
                    Part15CaseEvidence {
                        classification: Part15EvidenceClassification::Formal,
                        selection,
                        codestream: Part15CodestreamEvidence {
                            pcap: 1,
                            ccap15: 1,
                            mode: HtCodeBlockSetMode::HtDeclared,
                            multiple_ht_sets: false,
                            roi: false,
                            heterogeneous: false,
                            ht_irreversible: false,
                            bmagb,
                            quality_layers: 1,
                            default_ht_block_coding: true,
                            default_mixed_block_coding: false,
                            corresponding_profile_words: Vec::new(),
                            reversible: true,
                            component_count: 1,
                        },
                    }
                });
                case_report(&case.id, &case.table, case.peak, Some(case.mse), part15)
            })
            .collect::<Vec<_>>();
        cases.extend(
            manifest
                .jp2_cases
                .iter()
                .map(|case| case_report(&case.id, "G.1", case.peak, None, None)),
        );
        cases.extend(
            manifest
                .jph_bsets
                .iter()
                .map(|bset| case_report(&bset.id, "G.5", bset.peak, None, None)),
        );
        cases
    }

    fn case_report(
        id: &str,
        table: &str,
        allowed_peak: u64,
        allowed_mse: Option<f64>,
        part15: Option<Part15CaseEvidence>,
    ) -> CaseReport {
        CaseReport {
            id: id.to_string(),
            table: table.to_string(),
            status: CaseStatus::Pass,
            route: RouteKind::Cpu,
            peak: Some(0),
            mse: allowed_mse.map(|_| 0.0),
            allowed_peak,
            allowed_mse,
            error: None,
            stages: Vec::new(),
            accelerator_execution: None,
            part15,
        }
    }
}
