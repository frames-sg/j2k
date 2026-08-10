use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{
    has_exact_extension, validate_case_id, validate_inventory_reference, validation, DecoderCase,
    ManifestError, T803Manifest, T803Suite,
};

const REQUIRED_HT_BSETS: usize = 26;
const REQUIRED_HT_CANDIDATES: usize = 41;
const REQUIRED_JPH_BSETS: usize = 10;
const REQUIRED_JPH_CANDIDATES: usize = 20;

/// HTJ2K derived set or high-fidelity ETS represented by one BSET.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum HtClaimSet {
    /// Profile-0-derived HTONLY codestreams.
    Ds0Ht,
    /// Profile-0-derived codestreams that are not HTONLY.
    Ds0Hm,
    /// Profile-1-derived HTONLY codestreams.
    Ds1Ht,
    /// Profile-1-derived codestreams that are not HTONLY.
    Ds1Hm,
    /// The dedicated HTJ2K high-fidelity ETS.
    HighFidelity,
}

/// Formal HTJ2K compliance class represented by one selected case.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum HtComplianceClass {
    /// Cclass-1h.
    Cclass1h,
    /// Cclass-1HFh.
    Cclass1hF,
}

/// One non-zero additional allowance from an HTJ2K `bis` table.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HtAdditionalError {
    /// Zero-based decoded component index.
    pub component: usize,
    /// Additional peak-error allowance at the candidate reference depth.
    pub peak: u64,
    /// Additional MSE allowance at the candidate reference depth.
    pub mse: f64,
}

/// One codestream in an official HTJ2K BSET.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HtCodestream {
    /// Codestream path in the external corpus.
    pub path: String,
    /// Magnitude bound advertised by the codestream.
    pub bmagb: u8,
    /// Reference depth used by the applicable `bis` table.
    pub reference_depth: u8,
    /// Non-zero per-component additional allowances.
    #[serde(default)]
    pub additional_errors: Vec<HtAdditionalError>,
}

/// One official HTJ2K BSET linked to the corresponding Part 1 comparison rows.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HtBset {
    /// Stable BSET identifier.
    pub id: String,
    /// Corresponding J2K codestream used to find the shared reference rows.
    pub base_codestream: String,
    /// Derived set or high-fidelity ETS.
    pub claim_set: HtClaimSet,
    /// Formal compliance class.
    pub cclass: HtComplianceClass,
    /// Decoder magnitude-bound guarantee used for selection.
    pub mmagb: u8,
    /// Official candidate codestreams in this BSET.
    pub candidates: Vec<HtCodestream>,
}

/// One codestream in an official JPH BSET.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JphCodestream {
    /// JPH path in the external corpus.
    pub path: String,
    /// Magnitude bound advertised by its embedded codestream.
    pub bmagb: u8,
}

/// One official Annex G JPH BSET.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JphBset {
    /// Stable BSET identifier.
    pub id: String,
    /// Matching JP2 case identifier for sRGB output; absent for native-component cases.
    pub base_jp2_case: Option<String>,
    /// Decoder magnitude-bound guarantee used for selection.
    pub mmagb: u8,
    /// Official candidate JPH files.
    pub candidates: Vec<JphCodestream>,
    /// Native component references, used by JPH file 10.
    #[serde(default)]
    pub native_references: Vec<String>,
    /// Expected native component count.
    pub components: u8,
    /// Reference precision.
    pub bit_depth: u8,
    /// Reference width.
    pub width: u32,
    /// Reference height.
    pub height: u32,
    /// Inclusive peak-error bound.
    pub peak: u64,
}

impl JphBset {
    /// Select the largest official encoding within this reader's magnitude guarantee.
    pub fn selected_candidate(&self) -> Option<&JphCodestream> {
        self.candidates
            .iter()
            .filter(|candidate| candidate.bmagb <= self.mmagb)
            .max_by_key(|candidate| candidate.bmagb)
    }
}

/// Part 15 selection and tolerance evidence attached to a resolved comparison.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Part15CaseMetadata {
    /// Official BSET identifier.
    pub bset: String,
    /// Derived set or high-fidelity ETS.
    pub claim_set: HtClaimSet,
    /// Formal compliance class.
    pub cclass: HtComplianceClass,
    /// IUT magnitude-bound guarantee.
    pub mmagb: u8,
    /// Selected codestream magnitude bound.
    pub bmagb: u8,
    /// Base peak-error allowance before the `bis` addition.
    pub base_peak: u64,
    /// Base MSE allowance before the `bis` addition.
    pub base_mse: f64,
    /// Scaled additional peak-error allowance.
    pub additional_peak: u64,
    /// Scaled additional MSE allowance.
    pub additional_mse: f64,
}

impl T803Manifest {
    /// Resolve the decoder comparisons selected for one formal suite.
    pub fn decoder_cases_for_suite(
        &self,
        suite: T803Suite,
    ) -> Result<Vec<DecoderCase>, ManifestError> {
        let mut selected = Vec::new();
        if matches!(suite, T803Suite::Part1 | T803Suite::All) {
            selected.extend(self.decoder_cases.iter().cloned());
        }
        if !matches!(suite, T803Suite::Part15 | T803Suite::All) {
            return Ok(selected);
        }

        for bset in &self.ht_bsets {
            let base_table = match bset.claim_set {
                HtClaimSet::Ds0Ht | HtClaimSet::Ds0Hm => "C.6",
                HtClaimSet::Ds1Ht | HtClaimSet::Ds1Hm => "C.7",
                HtClaimSet::HighFidelity => "C.8",
            };
            let candidate = bset
                .candidates
                .iter()
                .filter(|candidate| candidate.bmagb <= bset.mmagb)
                .max_by_key(|candidate| candidate.bmagb)
                .ok_or_else(|| {
                    ManifestError::Validation(format!(
                        "{} has no candidate at or below MMAGB {}",
                        bset.id, bset.mmagb
                    ))
                })?;
            let mut matched = false;
            for base in self
                .decoder_cases
                .iter()
                .filter(|case| case.table == base_table && case.codestream == bset.base_codestream)
            {
                matched = true;
                let allowance = candidate
                    .additional_errors
                    .iter()
                    .find(|allowance| allowance.component == base.component);
                let depth_delta = base
                    .bit_depth
                    .checked_sub(candidate.reference_depth)
                    .ok_or_else(|| {
                        ManifestError::Validation(format!(
                            "{} reference depth exceeds comparison depth",
                            candidate.path
                        ))
                    })?;
                let peak_scale = 1_u64.checked_shl(u32::from(depth_delta)).ok_or_else(|| {
                    ManifestError::Validation(format!(
                        "{} peak-error scaling overflows",
                        candidate.path
                    ))
                })?;
                let additional_peak = allowance
                    .map_or(0, |allowance| allowance.peak)
                    .checked_mul(peak_scale)
                    .ok_or_else(|| {
                        ManifestError::Validation(format!(
                            "{} peak-error allowance overflows",
                            candidate.path
                        ))
                    })?;
                let additional_mse = allowance.map_or(0.0, |allowance| allowance.mse)
                    * 4_f64.powi(i32::from(depth_delta));
                let peak = base.peak.checked_add(additional_peak).ok_or_else(|| {
                    ManifestError::Validation(format!(
                        "{} effective peak-error allowance overflows",
                        candidate.path
                    ))
                })?;
                let mse = base.mse + additional_mse;
                if !mse.is_finite() {
                    return validation(format!(
                        "{} effective MSE allowance is not finite",
                        candidate.path
                    ));
                }
                let mut case = base.clone();
                case.id = format!("{}-{}", bset.id, base.id);
                case.codestream.clone_from(&candidate.path);
                case.peak = peak;
                case.mse = mse;
                case.part15 = Some(Part15CaseMetadata {
                    bset: bset.id.clone(),
                    claim_set: bset.claim_set,
                    cclass: bset.cclass,
                    mmagb: bset.mmagb,
                    bmagb: candidate.bmagb,
                    base_peak: base.peak,
                    base_mse: base.mse,
                    additional_peak,
                    additional_mse,
                });
                selected.push(case);
            }
            if !matched {
                return validation(format!(
                    "{} does not reference a selected Part 1 codestream",
                    bset.id
                ));
            }
        }
        Ok(selected)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the ordered Part 15 manifest inventory validation keeps cross-table uniqueness and exact counts in one transaction"
)]
pub(super) fn validate<'a>(
    manifest: &'a T803Manifest,
    inventory: &BTreeSet<&'a str>,
    used_files: &mut BTreeSet<&'a str>,
) -> Result<(), ManifestError> {
    let mut bset_ids = BTreeSet::new();
    let mut ht_candidate_paths = BTreeSet::new();
    let mut ht_candidate_count = 0_usize;
    for bset in &manifest.ht_bsets {
        validate_case_id(&mut bset_ids, &bset.id)?;
        validate_inventory_reference(inventory, &bset.base_codestream)?;
        if !has_exact_extension(&bset.base_codestream, "j2k") {
            return validation(format!("{} has a misnamed base codestream", bset.id));
        }
        match (bset.claim_set, bset.cclass, bset.mmagb) {
            (HtClaimSet::HighFidelity, HtComplianceClass::Cclass1hF, 20)
            | (
                HtClaimSet::Ds0Ht | HtClaimSet::Ds0Hm | HtClaimSet::Ds1Ht | HtClaimSet::Ds1Hm,
                HtComplianceClass::Cclass1h,
                15,
            ) => {}
            _ => {
                return validation(format!(
                    "{} has an invalid formal claim-set, Cclass, or MMAGB combination",
                    bset.id
                ));
            }
        }
        if bset.candidates.is_empty() {
            return validation(format!("{} has no HTJ2K candidates", bset.id));
        }
        let mut previous_bmagb = None;
        for candidate in &bset.candidates {
            ht_candidate_count += 1;
            validate_inventory_reference(inventory, &candidate.path)?;
            if !has_exact_extension(&candidate.path, "j2k") {
                return validation(format!("{} has a misnamed HTJ2K candidate", bset.id));
            }
            if !ht_candidate_paths.insert(candidate.path.as_str()) {
                return validation(format!(
                    "HTJ2K candidate {} belongs to more than one BSET",
                    candidate.path
                ));
            }
            if !(8..=74).contains(&candidate.bmagb)
                || !(1..=32).contains(&candidate.reference_depth)
                || previous_bmagb.is_some_and(|previous| previous >= candidate.bmagb)
            {
                return validation(format!(
                    "{} candidates must have increasing legal BMAGB values and reference depths",
                    bset.id
                ));
            }
            previous_bmagb = Some(candidate.bmagb);
            let mut components = BTreeSet::new();
            for allowance in &candidate.additional_errors {
                if !components.insert(allowance.component)
                    || !allowance.mse.is_finite()
                    || allowance.mse < 0.0
                    || (allowance.peak == 0 && allowance.mse == 0.0)
                {
                    return validation(format!(
                        "{} has an invalid or duplicate additional-error allowance",
                        candidate.path
                    ));
                }
            }
            used_files.insert(candidate.path.as_str());
        }
        if bset
            .candidates
            .iter()
            .all(|candidate| candidate.bmagb > bset.mmagb)
        {
            return validation(format!(
                "{} has no candidate at or below MMAGB {}",
                bset.id, bset.mmagb
            ));
        }
    }

    let mut jph_candidate_paths = BTreeSet::new();
    let mut jph_candidate_count = 0_usize;
    for bset in &manifest.jph_bsets {
        validate_case_id(&mut bset_ids, &bset.id)?;
        if bset.mmagb != 15
            || bset.candidates.is_empty()
            || bset.components == 0
            || bset.width == 0
            || bset.height == 0
            || !(1..=32).contains(&bset.bit_depth)
        {
            return validation(format!("{} has invalid JPH comparison metadata", bset.id));
        }
        let base_case = bset
            .base_jp2_case
            .as_ref()
            .and_then(|id| manifest.jp2_cases.iter().find(|case| case.id == *id));
        match (bset.base_jp2_case.as_ref(), base_case) {
            (Some(_), Some(case)) => {
                if !bset.native_references.is_empty()
                    || (
                        case.components,
                        case.bit_depth,
                        case.width,
                        case.height,
                        case.peak,
                    ) != (
                        bset.components,
                        bset.bit_depth,
                        bset.width,
                        bset.height,
                        bset.peak,
                    )
                {
                    return validation(format!("{} does not match its JP2 comparison", bset.id));
                }
            }
            (Some(id), None) => {
                return validation(format!("{} references unknown JP2 case {id}", bset.id));
            }
            (None, None) => {
                if bset.native_references.len() != usize::from(bset.components) {
                    return validation(format!(
                        "{} must provide one native reference per component",
                        bset.id
                    ));
                }
            }
            (None, Some(_)) => unreachable!("a JP2 case is only looked up for Some(id)"),
        }
        let mut previous_bmagb = None;
        for candidate in &bset.candidates {
            jph_candidate_count += 1;
            validate_inventory_reference(inventory, &candidate.path)?;
            if !has_exact_extension(&candidate.path, "jph")
                || !(8..=74).contains(&candidate.bmagb)
                || previous_bmagb.is_some_and(|previous| previous >= candidate.bmagb)
                || !jph_candidate_paths.insert(candidate.path.as_str())
            {
                return validation(format!("{} has invalid JPH candidates", bset.id));
            }
            previous_bmagb = Some(candidate.bmagb);
            used_files.insert(candidate.path.as_str());
        }
        if bset.selected_candidate().is_none() {
            return validation(format!(
                "{} has no JPH candidate at or below MMAGB {}",
                bset.id, bset.mmagb
            ));
        }
        for reference in &bset.native_references {
            validate_inventory_reference(inventory, reference)?;
            if !has_exact_extension(reference, "pgm") {
                return validation(format!("{} has a misnamed native reference", bset.id));
            }
            used_files.insert(reference.as_str());
        }
    }

    if manifest.ht_bsets.len() != REQUIRED_HT_BSETS || ht_candidate_count != REQUIRED_HT_CANDIDATES
    {
        return validation(format!(
            "Part 15 must contain {REQUIRED_HT_BSETS} HT BSETs and {REQUIRED_HT_CANDIDATES} candidates"
        ));
    }
    if manifest.jph_bsets.len() != REQUIRED_JPH_BSETS
        || jph_candidate_count != REQUIRED_JPH_CANDIDATES
    {
        return validation(format!(
            "Annex G must contain {REQUIRED_JPH_BSETS} JPH BSETs and {REQUIRED_JPH_CANDIDATES} candidates"
        ));
    }
    if manifest.decoder_cases_for_suite(T803Suite::Part15)?.len() != 60 {
        return validation("formal Part 15 decoder selection must contain 60 comparisons");
    }
    Ok(())
}
