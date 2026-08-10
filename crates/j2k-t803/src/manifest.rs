use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    path::{Component, Path},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

mod part15;

pub use part15::{
    HtAdditionalError, HtBset, HtClaimSet, HtCodestream, HtComplianceClass, JphBset, JphCodestream,
    Part15CaseMetadata,
};

pub(crate) const STANDARD: &str = "ISO/IEC 15444-4:2024 / ITU-T T.803 v3";
pub(crate) const SOURCE_URL: &str = "https://www.itu.int/wftp3/public/t/testsignal/SpeImage/T803/v2024_02/T.803v3_15444-4ed4-ElecAtt-codestreams.zip";
const TABLE_COUNTS: [(&str, usize); 5] = [
    ("C.1", 18),
    ("C.4", 8),
    ("C.6", 35),
    ("C.7", 17),
    ("C.8", 3),
];
const REQUIRED_CODESTREAMS: [&str; 24] = [
    "files/codestreams_profile0/p0_01.j2k",
    "files/codestreams_profile0/p0_02.j2k",
    "files/codestreams_profile0/p0_03.j2k",
    "files/codestreams_profile0/p0_04.j2k",
    "files/codestreams_profile0/p0_05.j2k",
    "files/codestreams_profile0/p0_06.j2k",
    "files/codestreams_profile0/p0_07.j2k",
    "files/codestreams_profile0/p0_08.j2k",
    "files/codestreams_profile0/p0_09.j2k",
    "files/codestreams_profile0/p0_10.j2k",
    "files/codestreams_profile0/p0_11.j2k",
    "files/codestreams_profile0/p0_12.j2k",
    "files/codestreams_profile0/p0_13.j2k",
    "files/codestreams_profile0/p0_14.j2k",
    "files/codestreams_profile0/p0_15.j2k",
    "files/codestreams_profile0/p0_16.j2k",
    "files/codestreams_profile1/p1_01.j2k",
    "files/codestreams_profile1/p1_02.j2k",
    "files/codestreams_profile1/p1_03.j2k",
    "files/codestreams_profile1/p1_04.j2k",
    "files/codestreams_profile1/p1_05.j2k",
    "files/codestreams_profile1/p1_06.j2k",
    "files/codestreams_profile1/p1_07.j2k",
    "files/codestreams_hifi/hifi_p1_02.j2k",
];
/// T.803 suite selected for one run.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum T803Suite {
    /// Part 1 J2K decoder and JP2 reader cases.
    Part1,
    /// Part 15 HTJ2K decoder and JPH reader cases.
    Part15,
    /// Both formal suites.
    All,
}

/// Pinned source metadata for the external electronic attachment.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct T803Source {
    /// Official ITU attachment handle.
    pub url: String,
    /// Expected SHA-256 of the attachment archive.
    pub archive_sha256: String,
    /// Expected archive size in bytes.
    pub archive_bytes: u64,
}

/// One externally stored file used by the selected test cases.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusFile {
    /// Normalized path below the extracted corpus root.
    pub path: String,
    /// Expected file SHA-256.
    pub sha256: String,
}

/// One reference-component comparison from an Annex C table.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DecoderCase {
    /// Stable case identifier.
    pub id: String,
    /// T.803 table containing the case.
    pub table: String,
    /// Codestream path in the external corpus.
    pub codestream: String,
    /// PGX reference path in the external corpus.
    pub reference: String,
    /// Zero-based component to compare.
    pub component: usize,
    /// Exact decoder resolution reduction level.
    pub reduction_levels: u8,
    /// Whether the reference component is signed.
    pub signed: bool,
    /// Reference precision.
    pub bit_depth: u8,
    /// Reference width.
    pub width: u32,
    /// Reference height.
    pub height: u32,
    /// Inclusive peak-error bound.
    pub peak: u64,
    /// Inclusive MSE bound.
    pub mse: f64,
    /// Resolved Part 15 selection metadata; populated by [`T803Manifest::decoder_cases_for_suite`].
    #[serde(skip)]
    pub part15: Option<Part15CaseMetadata>,
}

/// One Annex G JP2 reader comparison.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Jp2Case {
    /// Stable case identifier.
    pub id: String,
    /// JP2 input path in the external corpus.
    pub input: String,
    /// TIFF reference path in the external corpus.
    pub reference: String,
    /// Expected component count.
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

/// Complete, pinned T.803 v3 Part 1 and Annex G selection.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct T803Manifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Standard edition represented by the case data.
    pub standard: String,
    /// External attachment provenance.
    pub source: T803Source,
    /// Hash inventory for every selected external file.
    pub files: Vec<CorpusFile>,
    /// Annex C J2K decoder comparisons.
    pub decoder_cases: Vec<DecoderCase>,
    /// Annex G JP2 reader comparisons.
    pub jp2_cases: Vec<Jp2Case>,
    /// Official HTJ2K BSET inventory linked to the shared comparison rows.
    #[serde(default)]
    pub ht_bsets: Vec<HtBset>,
    /// Official Annex G JPH BSET inventory.
    #[serde(default)]
    pub jph_bsets: Vec<JphBset>,
}

/// Error returned when the pinned manifest is malformed or incomplete.
#[derive(Debug, Error)]
pub enum ManifestError {
    /// TOML syntax or schema error.
    #[error("invalid T.803 manifest TOML: {0}")]
    Toml(#[from] toml::de::Error),
    /// Semantic or inventory error.
    #[error("invalid T.803 manifest: {0}")]
    Validation(String),
}

impl T803Manifest {
    /// Parse and validate a complete T.803 v3 manifest.
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        let mut manifest = toml::from_str::<Self>(text)?;
        manifest
            .files
            .sort_unstable_by(|left, right| left.path.cmp(&right.path));
        manifest.validate()?;
        Ok(manifest)
    }

    /// Return the number of selected comparisons from one Annex C table.
    pub fn table_case_count(&self, table: &str) -> usize {
        self.decoder_cases
            .iter()
            .filter(|case| case.table == table)
            .count()
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the manifest root validates schema identity, shared inventory ownership, and exact suite counts together"
    )]
    fn validate(&self) -> Result<(), ManifestError> {
        if self.schema_version != 2 {
            return validation("schema_version must be 2");
        }
        if self.standard != STANDARD {
            return validation(format!("standard must be {STANDARD:?}"));
        }
        if self.source.url != SOURCE_URL {
            return validation(format!("source URL must be {SOURCE_URL}"));
        }
        validate_sha256(&self.source.archive_sha256, "archive")?;
        if self.source.archive_bytes == 0 {
            return validation("archive_bytes must be non-zero");
        }

        let mut inventory = BTreeSet::new();
        for file in &self.files {
            validate_path(&file.path)?;
            validate_sha256(&file.sha256, &file.path)?;
            if !inventory.insert(file.path.as_str()) {
                return validation(format!("duplicate file inventory path {}", file.path));
            }
        }

        let mut case_ids = BTreeSet::new();
        let mut used_files = BTreeSet::new();
        let mut table_counts = BTreeMap::new();
        let mut codestreams = BTreeSet::new();
        for case in &self.decoder_cases {
            validate_case_id(&mut case_ids, &case.id)?;
            if !TABLE_COUNTS.iter().any(|(table, _)| *table == case.table) {
                return validation(format!("{} has unknown Annex C table", case.id));
            }
            validate_inventory_reference(&inventory, &case.codestream)?;
            validate_inventory_reference(&inventory, &case.reference)?;
            if !has_exact_extension(&case.codestream, "j2k")
                || !has_exact_extension(&case.reference, "pgx")
            {
                return validation(format!("{} has misnamed input or reference", case.id));
            }
            if case.width == 0
                || case.height == 0
                || !(1..=32).contains(&case.bit_depth)
                || !case.mse.is_finite()
                || case.mse < 0.0
            {
                return validation(format!("{} has invalid comparison bounds", case.id));
            }
            *table_counts.entry(case.table.as_str()).or_insert(0_usize) += 1;
            codestreams.insert(case.codestream.as_str());
            used_files.insert(case.codestream.as_str());
            used_files.insert(case.reference.as_str());
        }

        for case in &self.jp2_cases {
            validate_case_id(&mut case_ids, &case.id)?;
            validate_inventory_reference(&inventory, &case.input)?;
            validate_inventory_reference(&inventory, &case.reference)?;
            if !has_exact_extension(&case.input, "jp2")
                || !has_exact_extension(&case.reference, "tif")
            {
                return validation(format!("{} has misnamed input or reference", case.id));
            }
            if case.components == 0
                || case.width == 0
                || case.height == 0
                || !(1..=32).contains(&case.bit_depth)
            {
                return validation(format!("{} has invalid comparison shape", case.id));
            }
            used_files.insert(case.input.as_str());
            used_files.insert(case.reference.as_str());
        }

        part15::validate(self, &inventory, &mut used_files)?;

        if inventory != used_files {
            let unused = inventory
                .difference(&used_files)
                .copied()
                .collect::<Vec<_>>();
            return validation(format!(
                "file inventory contains unused entries: {unused:?}"
            ));
        }
        for (table, expected) in TABLE_COUNTS {
            let actual = table_counts.get(table).copied().unwrap_or_default();
            if actual != expected {
                return validation(format!(
                    "table {table} must contain {expected} cases, found {actual}"
                ));
            }
        }
        if self.jp2_cases.len() != 9 {
            return validation(format!(
                "Annex G must contain 9 cases, found {}",
                self.jp2_cases.len()
            ));
        }
        if codestreams.len() != REQUIRED_CODESTREAMS.len()
            || REQUIRED_CODESTREAMS
                .iter()
                .any(|path| !codestreams.contains(path))
        {
            return validation("Annex C codestream set is incomplete or contains extra entries");
        }
        Ok(())
    }
}

fn validate_case_id<'a>(ids: &mut BTreeSet<&'a str>, id: &'a str) -> Result<(), ManifestError> {
    if id.is_empty() {
        return validation("case id must not be empty");
    }
    if !ids.insert(id) {
        return validation(format!("duplicate case id {id}"));
    }
    Ok(())
}

fn validate_inventory_reference(
    inventory: &BTreeSet<&str>,
    path: &str,
) -> Result<(), ManifestError> {
    validate_path(path)?;
    if !inventory.contains(path) {
        return validation(format!("{path} is not present in the file inventory"));
    }
    Ok(())
}

pub(crate) fn validate_path(path: &str) -> Result<(), ManifestError> {
    let normalized = !path.is_empty()
        && !path.contains('\\')
        && !path.contains("//")
        && !path.split('/').any(|segment| matches!(segment, "." | ".."))
        && !Path::new(path).is_absolute()
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
    if !normalized {
        return validation(format!("{path:?} must be a relative normalized path"));
    }
    Ok(())
}

pub(crate) fn validate_sha256(value: &str, subject: &str) -> Result<(), ManifestError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return validation(format!("{subject} must have a lowercase SHA-256"));
    }
    Ok(())
}

fn has_exact_extension(path: &str, extension: &str) -> bool {
    Path::new(path).extension() == Some(OsStr::new(extension))
}

fn validation<T>(message: impl Into<String>) -> Result<T, ManifestError> {
    Err(ManifestError::Validation(message.into()))
}
