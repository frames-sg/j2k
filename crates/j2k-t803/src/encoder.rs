// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declarative Annex D/F encoder test scope and inventory validation.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::manifest::{validate_sha256, STANDARD};

const MATRIX_PATH: &str = "corpus/j2k-conformance/encoder-matrix-v1.toml";
const REFERENCE_STANDARD: &str = "ISO/IEC 15444-5 / ITU-T T.804";
const REFERENCE_IMPLEMENTATION: &str = "OpenJPEG";
const REFERENCE_VERSION: &str = "2.5.3";

/// Encoder implementation under test.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderIut {
    /// Portable `j2k` CPU encoder surfaces.
    Cpu,
    /// `j2k-cuda` adapter encoder surfaces.
    Cuda,
    /// `j2k-metal` adapter encoder surfaces.
    Metal,
}

/// Compression mode exercised by one encoder case.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EncoderMode {
    /// Reversible Part 1 encode.
    Lossless,
    /// Irreversible Part 1 encode.
    Lossy,
}

/// Part 1 packet progression selected in COD.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EncoderProgression {
    /// Layer-resolution-component-position.
    Lrcp,
    /// Resolution-layer-component-position.
    Rlcp,
    /// Resolution-position-component-layer.
    Rpcl,
    /// Position-component-resolution-layer.
    Pcrl,
    /// Component-position-resolution-layer.
    Cprl,
}

/// Marker listed in T.803 Table F.1.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum EncoderMarker {
    Soc,
    Cap,
    Prf,
    Cpf,
    Sot,
    Sod,
    Eoc,
    Siz,
    Cod,
    Coc,
    Rgn,
    Qcd,
    Qcc,
    Poc,
    Tlm,
    Plm,
    Plt,
    Ppm,
    Ppt,
    Sop,
    Eph,
    Crg,
    Com,
}

const TABLE_F1_MARKERS: [EncoderMarker; 23] = [
    EncoderMarker::Soc,
    EncoderMarker::Cap,
    EncoderMarker::Prf,
    EncoderMarker::Cpf,
    EncoderMarker::Sot,
    EncoderMarker::Sod,
    EncoderMarker::Eoc,
    EncoderMarker::Siz,
    EncoderMarker::Cod,
    EncoderMarker::Coc,
    EncoderMarker::Rgn,
    EncoderMarker::Qcd,
    EncoderMarker::Qcc,
    EncoderMarker::Poc,
    EncoderMarker::Tlm,
    EncoderMarker::Plm,
    EncoderMarker::Plt,
    EncoderMarker::Ppm,
    EncoderMarker::Ppt,
    EncoderMarker::Sop,
    EncoderMarker::Eph,
    EncoderMarker::Crg,
    EncoderMarker::Com,
];

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum EncoderInputKind {
    #[default]
    Interleaved,
    ComponentPlanes,
    TypedComponentPlanes,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum EncoderPattern {
    #[default]
    Gradient,
    Checkerboard,
    DeterministicNoise,
    Impulse,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub(crate) enum EncoderRateTarget {
    BitsPerPixel(f64),
    Bytes(u64),
    PsnrDb(f64),
}

/// One rectangular maxshift request.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EncoderRoi {
    pub(crate) component: u16,
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) shift: u8,
}

/// One stable Annex D encoder test case.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderCase {
    /// Stable case identifier.
    pub id: String,
    /// Adapter IUTs to which this case applies.
    pub iuts: Vec<EncoderIut>,
    /// Reversible or irreversible coding.
    pub mode: EncoderMode,
    #[serde(default)]
    pub(crate) input: EncoderInputKind,
    /// Reference-grid width.
    pub width: u32,
    /// Reference-grid height.
    pub height: u32,
    /// Component count.
    pub components: u16,
    /// Common sample precision for interleaved and homogeneous planar inputs.
    pub bit_depth: u8,
    /// Common signedness for interleaved and homogeneous planar inputs.
    pub signed: bool,
    #[serde(default)]
    pub(crate) pattern: EncoderPattern,
    #[serde(default)]
    pub(crate) sampling: Vec<[u8; 2]>,
    #[serde(default)]
    pub(crate) component_bit_depths: Vec<u8>,
    #[serde(default)]
    pub(crate) component_signedness: Vec<bool>,
    /// COD packet progression.
    pub progression: EncoderProgression,
    /// Requested wavelet decomposition levels.
    pub decomposition_levels: u8,
    #[serde(default = "one_quality_layer")]
    pub(crate) lossless_quality_layers: u8,
    #[serde(default)]
    pub(crate) lossy_rate_target: Option<EncoderRateTarget>,
    #[serde(default)]
    pub(crate) lossy_quality_layers: Vec<EncoderRateTarget>,
    #[serde(default)]
    pub(crate) minimum_psnr_db: Option<f64>,
    #[serde(default)]
    pub(crate) maximum_rate_overshoot_percent: Option<f64>,
    #[serde(default)]
    pub(crate) tile_size: Option<[u32; 2]>,
    #[serde(default)]
    pub(crate) tile_part_packet_limit: Option<u16>,
    #[serde(default)]
    pub(crate) precinct_exponents: Vec<[u8; 2]>,
    #[serde(default)]
    pub(crate) roi: Option<EncoderRoi>,
    /// Optional markers this case explicitly requests.
    #[serde(default)]
    pub markers: Vec<EncoderMarker>,
    /// Whether this row participates in the declared pairwise covering array.
    #[serde(default)]
    pub pairwise: bool,
}

const fn one_quality_layer() -> u8 {
    1
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PairwiseScope {
    modes: Vec<EncoderMode>,
    dimensions: Vec<[u32; 2]>,
    signedness: Vec<bool>,
    bit_depths: Vec<u8>,
    component_counts: Vec<u16>,
    progressions: Vec<EncoderProgression>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct MatrixInventory {
    iut: EncoderIut,
    case_count: usize,
    case_sha256: String,
}

/// Versioned, tamper-evident Annex D encoder case matrix.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderMatrix {
    /// Matrix schema version.
    pub schema_version: u32,
    /// T.803 edition used to define the procedure.
    pub standard: String,
    pairwise: PairwiseScope,
    inventories: Vec<MatrixInventory>,
    /// Cases in stable execution order.
    pub cases: Vec<EncoderCase>,
}

/// Error returned for malformed encoder matrices or ICS files.
#[derive(Debug, Error)]
pub enum EncoderMatrixError {
    /// TOML syntax or schema error.
    #[error("invalid encoder evidence TOML: {0}")]
    Toml(#[from] toml::de::Error),
    /// Semantic or inventory error.
    #[error("invalid encoder evidence: {0}")]
    Validation(String),
    /// Canonical case serialization failed.
    #[error("serialize encoder case inventory: {0}")]
    Json(#[from] serde_json::Error),
}

impl EncoderMatrix {
    /// Parse and validate a complete encoder matrix.
    pub fn parse(text: &str) -> Result<Self, EncoderMatrixError> {
        let matrix = toml::from_str::<Self>(text)?;
        matrix.validate()?;
        Ok(matrix)
    }

    fn validate(&self) -> Result<(), EncoderMatrixError> {
        if self.schema_version != 1 || self.standard != STANDARD {
            return validation("schema or standard does not match T.803 v3");
        }
        validate_pairwise_scope(&self.pairwise)?;

        let mut previous_id = None;
        for case in &self.cases {
            if case.id.is_empty()
                || previous_id.is_some_and(|previous| previous >= case.id.as_str())
            {
                return validation("case ids must be non-empty, sorted, and unique");
            }
            previous_id = Some(case.id.as_str());
            validate_case(case)?;
        }
        validate_pairwise_coverage(&self.pairwise, &self.cases)?;
        validate_boundaries(&self.cases)?;
        self.validate_inventories()
    }

    fn validate_inventories(&self) -> Result<(), EncoderMatrixError> {
        let expected_iuts = [EncoderIut::Cpu, EncoderIut::Cuda, EncoderIut::Metal];
        if self.inventories.len() != expected_iuts.len()
            || self
                .inventories
                .iter()
                .map(|inventory| inventory.iut)
                .ne(expected_iuts)
        {
            return validation("matrix inventories must contain CPU, CUDA, and Metal in order");
        }
        for inventory in &self.inventories {
            validate_sha256(&inventory.case_sha256, "encoder case inventory")
                .map_err(|error| EncoderMatrixError::Validation(error.to_string()))?;
            let cases = self
                .cases
                .iter()
                .filter(|case| case.iuts.contains(&inventory.iut))
                .collect::<Vec<_>>();
            if cases.len() != inventory.case_count {
                return validation(format!(
                    "{:?} case count is {}, expected {}",
                    inventory.iut,
                    cases.len(),
                    inventory.case_count
                ));
            }
            let actual = canonical_case_sha256(&cases)?;
            if actual != inventory.case_sha256 {
                return validation(format!(
                    "{:?} case inventory SHA-256 is {actual}, expected {}",
                    inventory.iut, inventory.case_sha256
                ));
            }
        }
        Ok(())
    }

    fn inventory(&self, iut: EncoderIut) -> Option<&MatrixInventory> {
        self.inventories.iter().find(|entry| entry.iut == iut)
    }

    #[cfg(feature = "runner")]
    pub(crate) fn selected_cases(&self, iut: EncoderIut) -> impl Iterator<Item = &EncoderCase> {
        self.cases
            .iter()
            .filter(move |case| case.iuts.contains(&iut))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum MarkerUse {
    Always,
    Conditional,
    CallerControlled,
    NotProduced,
    OutsidePart1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct IcsMarker {
    marker: EncoderMarker,
    usage: MarkerUse,
}

/// Published T.803 Annex F implementation compliance statement.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderIcs {
    /// ICS schema version.
    pub schema_version: u32,
    /// T.803 edition used by the statement.
    pub standard: String,
    /// CPU, CUDA adapter, or Metal adapter IUT.
    pub iut: EncoderIut,
    /// Precise informative-encoder scope statement.
    pub scope: String,
    /// Public encoder entry points covered by the statement.
    pub surfaces: Vec<String>,
    matrix_path: String,
    matrix_case_count: usize,
    matrix_case_sha256: String,
    reference_decoder_standard: String,
    reference_decoder_implementation: String,
    reference_decoder_version: String,
    /// Public maximum sample precision; the reference decoder may cover less.
    pub public_max_bit_depth: u8,
    /// Highest precision included in the T.804 `OpenJPEG` matrix.
    pub reference_validated_max_bit_depth: u8,
    /// Public maximum component count for the listed surfaces.
    pub public_max_components: u16,
    /// Whether the listed surface accepts component sampling.
    pub component_sampling: bool,
    /// Known API ranges not validated by this reference implementation.
    pub reference_limitations: Vec<String>,
    markers: Vec<IcsMarker>,
}

impl EncoderIcs {
    /// Parse and validate one Annex F ICS.
    pub fn parse(text: &str) -> Result<Self, EncoderMatrixError> {
        let ics = toml::from_str::<Self>(text)?;
        ics.validate()?;
        Ok(ics)
    }

    /// Verify that this ICS pins the selected IUT's exact matrix inventory.
    pub fn validate_against(&self, matrix: &EncoderMatrix) -> Result<(), EncoderMatrixError> {
        self.validate()?;
        let inventory = matrix.inventory(self.iut).ok_or_else(|| {
            EncoderMatrixError::Validation("IUT inventory is missing".to_string())
        })?;
        if self.matrix_case_count != inventory.case_count
            || self.matrix_case_sha256 != inventory.case_sha256
        {
            return validation("ICS matrix count or SHA-256 does not match the case inventory");
        }
        for entry in self
            .markers
            .iter()
            .filter(|entry| entry.usage == MarkerUse::CallerControlled)
        {
            if !matrix
                .cases
                .iter()
                .any(|case| case.iuts.contains(&self.iut) && case.markers.contains(&entry.marker))
            {
                return validation(format!(
                    "{:?} does not exercise caller-controlled {:?}",
                    self.iut, entry.marker
                ));
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), EncoderMatrixError> {
        if self.schema_version != 1 || self.standard != STANDARD {
            return validation("ICS schema or standard does not match T.803 v3");
        }
        if self.scope.is_empty()
            || self.surfaces.is_empty()
            || self.surfaces.iter().any(String::is_empty)
        {
            return validation("ICS scope and surfaces must not be empty");
        }
        if self.matrix_path != MATRIX_PATH {
            return validation(format!("ICS matrix path must be {MATRIX_PATH}"));
        }
        validate_sha256(&self.matrix_case_sha256, "ICS matrix inventory")
            .map_err(|error| EncoderMatrixError::Validation(error.to_string()))?;
        if self.matrix_case_count == 0
            || self.reference_decoder_standard != REFERENCE_STANDARD
            || self.reference_decoder_implementation != REFERENCE_IMPLEMENTATION
            || self.reference_decoder_version != REFERENCE_VERSION
        {
            return validation("ICS reference decoder or matrix metadata is invalid");
        }
        if self.public_max_bit_depth != 38
            || self.reference_validated_max_bit_depth != 31
            || self.public_max_components != 16_384
            || self.reference_limitations.is_empty()
        {
            return validation("ICS public and reference-decoder limits are incomplete");
        }
        if self.iut != EncoderIut::Cpu && self.component_sampling {
            return validation("adapter ICS must not claim a sampled-component surface");
        }
        let markers = self
            .markers
            .iter()
            .map(|entry| entry.marker)
            .collect::<Vec<_>>();
        if markers != TABLE_F1_MARKERS {
            return validation("ICS must list every Table F.1 marker in table order");
        }
        validate_marker_usage(self.iut, &self.markers)
    }

    #[cfg(feature = "runner")]
    pub(crate) fn matrix_case_count(&self) -> usize {
        self.matrix_case_count
    }

    #[cfg(feature = "runner")]
    pub(crate) fn matrix_case_sha256(&self) -> &str {
        &self.matrix_case_sha256
    }
}

#[cfg(feature = "runner")]
pub(crate) fn reference_decoder_identity() -> (&'static str, &'static str, &'static str) {
    (
        REFERENCE_STANDARD,
        REFERENCE_IMPLEMENTATION,
        REFERENCE_VERSION,
    )
}

#[cfg(feature = "runner")]
pub(crate) fn matrix_path() -> &'static str {
    MATRIX_PATH
}

#[cfg(feature = "runner")]
pub(crate) const fn ics_path(iut: EncoderIut) -> &'static str {
    match iut {
        EncoderIut::Cpu => "corpus/j2k-conformance/encoder-ics-cpu.toml",
        EncoderIut::Cuda => "corpus/j2k-conformance/encoder-ics-cuda.toml",
        EncoderIut::Metal => "corpus/j2k-conformance/encoder-ics-metal.toml",
    }
}

fn validate_marker_usage(iut: EncoderIut, markers: &[IcsMarker]) -> Result<(), EncoderMatrixError> {
    for entry in markers {
        let expected = match entry.marker {
            EncoderMarker::Soc
            | EncoderMarker::Sot
            | EncoderMarker::Sod
            | EncoderMarker::Eoc
            | EncoderMarker::Siz
            | EncoderMarker::Cod
            | EncoderMarker::Qcd => MarkerUse::Always,
            EncoderMarker::Cap | EncoderMarker::Cpf => MarkerUse::OutsidePart1,
            EncoderMarker::Coc | EncoderMarker::Qcc => MarkerUse::Conditional,
            EncoderMarker::Rgn if iut == EncoderIut::Cpu => MarkerUse::CallerControlled,
            EncoderMarker::Tlm
            | EncoderMarker::Plm
            | EncoderMarker::Plt
            | EncoderMarker::Ppm
            | EncoderMarker::Ppt
            | EncoderMarker::Sop
            | EncoderMarker::Eph => MarkerUse::CallerControlled,
            EncoderMarker::Prf
            | EncoderMarker::Rgn
            | EncoderMarker::Poc
            | EncoderMarker::Crg
            | EncoderMarker::Com => MarkerUse::NotProduced,
        };
        if entry.usage != expected {
            return validation(format!(
                "{:?} has incorrect Table F.1 usage for {:?}",
                iut, entry.marker
            ));
        }
    }
    Ok(())
}

fn validate_case(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    if case.iuts.is_empty()
        || !case.iuts.windows(2).all(|pair| pair[0] < pair[1])
        || case.width == 0
        || case.height == 0
        || !(1..=16_384).contains(&case.components)
        || !(1..=38).contains(&case.bit_depth)
        || case.decomposition_levels > 32
        || case.lossless_quality_layers == 0
        || case.markers.iter().collect::<BTreeSet<_>>().len() != case.markers.len()
    {
        return validation(format!("{} has invalid basic parameters", case.id));
    }
    if case
        .tile_size
        .is_some_and(|[width, height]| width == 0 || height == 0)
        || case.tile_part_packet_limit == Some(0)
        || case
            .precinct_exponents
            .iter()
            .any(|[width, height]| *width > 15 || *height > 15)
    {
        return validation(format!("{} has invalid tiling or precinct data", case.id));
    }
    if let Some(roi) = case.roi {
        let fits = roi.component < case.components
            && roi.width > 0
            && roi.height > 0
            && roi.shift > 0
            && roi
                .x
                .checked_add(roi.width)
                .is_some_and(|x1| x1 <= case.width)
            && roi
                .y
                .checked_add(roi.height)
                .is_some_and(|y1| y1 <= case.height);
        if !fits || !case.markers.contains(&EncoderMarker::Rgn) {
            return validation(format!("{} has invalid ROI data", case.id));
        }
    } else if case.markers.contains(&EncoderMarker::Rgn) {
        return validation(format!("{} requests RGN without an ROI", case.id));
    }
    validate_input_surface(case)?;
    validate_mode(case)
}

fn validate_input_surface(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    match case.input {
        EncoderInputKind::Interleaved => {
            if !case.sampling.is_empty()
                || !case.component_bit_depths.is_empty()
                || !case.component_signedness.is_empty()
            {
                return validation(format!("{} mixes interleaved and planar metadata", case.id));
            }
        }
        EncoderInputKind::ComponentPlanes => {
            validate_cpu_planar_case(case)?;
            if !case.component_bit_depths.is_empty() || !case.component_signedness.is_empty() {
                return validation(format!(
                    "{} has typed metadata on homogeneous planes",
                    case.id
                ));
            }
        }
        EncoderInputKind::TypedComponentPlanes => {
            validate_cpu_planar_case(case)?;
            if case.component_bit_depths.len() != usize::from(case.components)
                || case.component_signedness.len() != usize::from(case.components)
                || case
                    .component_bit_depths
                    .iter()
                    .any(|depth| !(1..=38).contains(depth))
            {
                return validation(format!("{} has invalid typed component metadata", case.id));
            }
        }
    }
    Ok(())
}

fn validate_mode(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    match case.mode {
        EncoderMode::Lossless => {
            if case.lossy_rate_target.is_some()
                || !case.lossy_quality_layers.is_empty()
                || case.minimum_psnr_db.is_some()
                || case.maximum_rate_overshoot_percent.is_some()
            {
                return validation(format!("{} has lossy targets on a lossless case", case.id));
            }
        }
        EncoderMode::Lossy => {
            if case.input != EncoderInputKind::Interleaved || case.roi.is_some() {
                return validation(format!(
                    "{} uses an unsupported lossy matrix surface",
                    case.id
                ));
            }
            for target in case
                .lossy_rate_target
                .iter()
                .chain(case.lossy_quality_layers.iter())
            {
                validate_rate_target(*target, &case.id)?;
            }
            if case
                .minimum_psnr_db
                .is_none_or(|value| !value.is_finite() || value <= 0.0)
            {
                return validation(format!("{} has no finite minimum PSNR gate", case.id));
            }
            let has_rate_gate = case
                .lossy_rate_target
                .iter()
                .chain(case.lossy_quality_layers.last())
                .any(|target| {
                    matches!(
                        target,
                        EncoderRateTarget::BitsPerPixel(_) | EncoderRateTarget::Bytes(_)
                    )
                });
            if has_rate_gate
                != case
                    .maximum_rate_overshoot_percent
                    .is_some_and(|value| value.is_finite() && (0.0..=100.0).contains(&value))
            {
                return validation(format!("{} has an invalid rate overshoot gate", case.id));
            }
        }
    }
    Ok(())
}

fn validate_cpu_planar_case(case: &EncoderCase) -> Result<(), EncoderMatrixError> {
    if case.mode != EncoderMode::Lossless
        || case.iuts != [EncoderIut::Cpu]
        || case.sampling.len() != usize::from(case.components)
        || case
            .sampling
            .iter()
            .any(|[x_rsiz, y_rsiz]| *x_rsiz == 0 || *y_rsiz == 0)
    {
        return validation(format!("{} has invalid planar surface metadata", case.id));
    }
    Ok(())
}

fn validate_rate_target(target: EncoderRateTarget, id: &str) -> Result<(), EncoderMatrixError> {
    let valid = match target {
        EncoderRateTarget::BitsPerPixel(value) | EncoderRateTarget::PsnrDb(value) => {
            value.is_finite() && value > 0.0
        }
        EncoderRateTarget::Bytes(value) => value > 0,
    };
    if valid {
        Ok(())
    } else {
        validation(format!("{id} has an invalid lossy rate target"))
    }
}

fn validate_pairwise_scope(scope: &PairwiseScope) -> Result<(), EncoderMatrixError> {
    let exact = scope.modes == [EncoderMode::Lossless, EncoderMode::Lossy]
        && scope.dimensions == [[32, 32], [63, 47]]
        && scope.signedness == [false, true]
        && scope.bit_depths == [8, 12]
        && scope.component_counts == [1, 3]
        && scope.progressions
            == [
                EncoderProgression::Lrcp,
                EncoderProgression::Rlcp,
                EncoderProgression::Rpcl,
                EncoderProgression::Pcrl,
                EncoderProgression::Cprl,
            ];
    if exact {
        Ok(())
    } else {
        validation("pairwise scope does not match the committed Part 1 coverage contract")
    }
}

fn validate_pairwise_coverage(
    scope: &PairwiseScope,
    cases: &[EncoderCase],
) -> Result<(), EncoderMatrixError> {
    let axes = [
        scope.modes.iter().map(debug_value).collect::<Vec<_>>(),
        scope
            .dimensions
            .iter()
            .map(|[width, height]| format!("{width}x{height}"))
            .collect(),
        scope.signedness.iter().map(debug_value).collect(),
        scope.bit_depths.iter().map(debug_value).collect(),
        scope.component_counts.iter().map(debug_value).collect(),
        scope.progressions.iter().map(debug_value).collect(),
    ];
    let rows = cases
        .iter()
        .filter(|case| case.pairwise)
        .map(|case| {
            [
                debug_value(&case.mode),
                format!("{}x{}", case.width, case.height),
                debug_value(&case.signed),
                debug_value(&case.bit_depth),
                debug_value(&case.components),
                debug_value(&case.progression),
            ]
        })
        .collect::<Vec<_>>();
    for left in 0..axes.len() {
        for right in (left + 1)..axes.len() {
            for left_value in &axes[left] {
                for right_value in &axes[right] {
                    if !rows
                        .iter()
                        .any(|row| row[left] == *left_value && row[right] == *right_value)
                    {
                        return validation(format!(
                            "pairwise rows do not cover axes {left}/{right} values {left_value}/{right_value}"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_boundaries(cases: &[EncoderCase]) -> Result<(), EncoderMatrixError> {
    for bit_depth in [1, 8, 12, 16, 31] {
        require(
            cases.iter().any(|case| case.bit_depth == bit_depth),
            "bit-depth boundary",
        )?;
    }
    for components in [1, 2, 3, 4, 5] {
        require(
            cases.iter().any(|case| case.components == components),
            "component-count boundary",
        )?;
    }
    for level in [0, 1, 2, 3, 5] {
        require(
            cases.iter().any(|case| case.decomposition_levels == level),
            "decomposition-level boundary",
        )?;
    }
    require(
        cases.iter().any(|case| (case.width, case.height) == (1, 1)),
        "singleton geometry",
    )?;
    require(cases.iter().any(|case| case.tile_size.is_some()), "tiling")?;
    require(
        cases
            .iter()
            .any(|case| case.tile_part_packet_limit.is_some()),
        "tile parts",
    )?;
    require(
        cases.iter().any(|case| !case.precinct_exponents.is_empty()),
        "precincts",
    )?;
    require(cases.iter().any(|case| case.roi.is_some()), "ROI maxshift")?;
    require(
        cases
            .iter()
            .any(|case| case.input == EncoderInputKind::ComponentPlanes),
        "component sampling",
    )?;
    require(
        cases
            .iter()
            .any(|case| case.input == EncoderInputKind::TypedComponentPlanes),
        "mixed typed components",
    )?;
    for target_kind in ["bits-per-pixel", "bytes", "psnr-db"] {
        require(
            cases.iter().any(|case| {
                case.lossy_rate_target
                    .iter()
                    .chain(case.lossy_quality_layers.iter())
                    .any(|target| rate_kind(*target) == target_kind)
            }),
            "lossy rate-target variant",
        )?;
    }
    require(
        cases.iter().any(|case| case.lossless_quality_layers > 1)
            && cases.iter().any(|case| case.lossy_quality_layers.len() > 1),
        "lossless and lossy quality layers",
    )?;
    Ok(())
}

fn require(present: bool, what: &str) -> Result<(), EncoderMatrixError> {
    if present {
        Ok(())
    } else {
        validation(format!("matrix does not cover {what}"))
    }
}

fn rate_kind(target: EncoderRateTarget) -> &'static str {
    match target {
        EncoderRateTarget::BitsPerPixel(_) => "bits-per-pixel",
        EncoderRateTarget::Bytes(_) => "bytes",
        EncoderRateTarget::PsnrDb(_) => "psnr-db",
    }
}

fn debug_value(value: &impl std::fmt::Debug) -> String {
    format!("{value:?}")
}

fn canonical_case_sha256(cases: &[&EncoderCase]) -> Result<String, EncoderMatrixError> {
    let bytes = serde_json::to_vec(cases)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validation<T>(message: impl Into<String>) -> Result<T, EncoderMatrixError> {
    Err(EncoderMatrixError::Validation(message.into()))
}
