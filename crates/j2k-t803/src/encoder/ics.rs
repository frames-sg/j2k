// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::manifest::{validate_sha256, STANDARD};

use super::{
    matrix::validation, EncoderIut, EncoderMarker, EncoderMatrix, EncoderMatrixError,
    TABLE_F1_MARKERS,
};

const MATRIX_PATH: &str = "corpus/j2k-conformance/encoder-matrix-v2.toml";
const REFERENCE_STANDARD: &str = "ISO/IEC 15444-5 / ITU-T T.804";
const REFERENCE_IMPLEMENTATION: &str = "OpenJPEG";
const REFERENCE_VERSION: &str = "2.5.3";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum MarkerUse {
    Always,
    Conditional,
    CallerControlled,
    NotProduced,
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
        if self.schema_version != 2 || self.standard != STANDARD {
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
            EncoderMarker::Cap | EncoderMarker::Coc | EncoderMarker::Qcc => MarkerUse::Conditional,
            EncoderMarker::Rgn if iut == EncoderIut::Cpu => MarkerUse::CallerControlled,
            EncoderMarker::Tlm
            | EncoderMarker::Plm
            | EncoderMarker::Plt
            | EncoderMarker::Ppm
            | EncoderMarker::Ppt
            | EncoderMarker::Sop
            | EncoderMarker::Eph => MarkerUse::CallerControlled,
            EncoderMarker::Prf
            | EncoderMarker::Cpf
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
