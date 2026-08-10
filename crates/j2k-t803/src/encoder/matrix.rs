// SPDX-License-Identifier: MIT OR Apache-2.0

mod case;
mod pairwise;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::manifest::{validate_sha256, STANDARD};

use self::case::{validate_boundaries, validate_case};
use self::pairwise::validate_pairwise_coverage;
use super::{EncoderCase, EncoderIut};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MatrixInventory {
    pub(super) iut: EncoderIut,
    pub(super) case_count: usize,
    pub(super) case_sha256: String,
}

/// Versioned, tamper-evident Annex D encoder case matrix.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderMatrix {
    /// Matrix schema version.
    pub schema_version: u32,
    /// T.803 edition used to define the procedure.
    pub standard: String,
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
        if self.schema_version != 2 || self.standard != STANDARD {
            return validation("schema or standard does not match T.803 v3");
        }
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
        validate_pairwise_coverage(&self.cases)?;
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

    pub(super) fn inventory(&self, iut: EncoderIut) -> Option<&MatrixInventory> {
        self.inventories.iter().find(|entry| entry.iut == iut)
    }

    #[cfg(feature = "runner")]
    pub(crate) fn selected_cases(&self, iut: EncoderIut) -> impl Iterator<Item = &EncoderCase> {
        self.cases
            .iter()
            .filter(move |case| case.iuts.contains(&iut))
    }
}

fn canonical_case_sha256(cases: &[&EncoderCase]) -> Result<String, EncoderMatrixError> {
    let bytes = serde_json::to_vec(cases)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(super) fn validation<T>(message: impl Into<String>) -> Result<T, EncoderMatrixError> {
    Err(EncoderMatrixError::Validation(message.into()))
}
