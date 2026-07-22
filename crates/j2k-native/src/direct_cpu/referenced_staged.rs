// SPDX-License-Identifier: MIT OR Apache-2.0

//! Staged parse-free CPU execution for cross-image code-block scheduling.

mod entropy;
mod finish;
mod plan_access;
mod prepare;
mod state;

pub use entropy::{execute_referenced_classic_entropy_job, execute_referenced_htj2k_entropy_job};
pub use finish::{
    finish_referenced_classic_staged, finish_referenced_classic_tile_staged,
    finish_referenced_htj2k_staged, finish_referenced_htj2k_tile_staged,
};
pub use prepare::{
    prepare_referenced_classic_entropy_workspace, prepare_referenced_classic_staged,
    prepare_referenced_classic_tile_staged, prepare_referenced_htj2k_entropy_workspace,
    prepare_referenced_htj2k_staged, prepare_referenced_htj2k_tile_staged,
};

use crate::error::{DecodingError, Result};
use crate::{HtCodeBlockDecodeWorkspace, J2kCodeBlockDecodeWorkspace};

use super::allocation::DirectWorkspaceBudget;

/// Stable location of one code block inside retained component geometry.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kDirectCodeBlockIndex {
    /// Raster-ordered referenced tile-plan index.
    pub tile: usize,
    /// Component-plan index.
    pub component: usize,
    /// Step index inside the component plan.
    pub step: usize,
    /// Code-block index inside the sub-band step.
    pub code_block: usize,
}

/// Per-worker scalar entropy workspace shared by staged images.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct J2kDirectCpuEntropyWorkspace {
    classic: J2kCodeBlockDecodeWorkspace,
    ht: HtCodeBlockDecodeWorkspace,
}

impl J2kDirectCpuEntropyWorkspace {
    /// Retained HT scalar workspace bytes.
    #[must_use]
    pub fn retained_ht_bytes(&self) -> usize {
        self.ht.allocated_bytes().unwrap_or(usize::MAX)
    }

    /// Retained classic scalar workspace bytes.
    #[must_use]
    pub fn retained_classic_bytes(&self) -> usize {
        self.classic.allocated_bytes().unwrap_or(usize::MAX)
    }

    fn prepare_ht(
        &mut self,
        dimensions: Option<(u32, u32)>,
        budget: DirectWorkspaceBudget,
    ) -> Result<()> {
        let (width, height) = self.prepare_ht_dimensions(dimensions)?;
        if budget
            .validate_workspace(self.ht.allocated_bytes()?)
            .is_err()
        {
            self.ht = HtCodeBlockDecodeWorkspace::default();
            self.ht.prepare(width, height)?;
            budget.validate_workspace(self.ht.allocated_bytes()?)?;
        }
        Ok(())
    }

    fn prepare_classic(
        &mut self,
        dimensions: Option<(u32, u32)>,
        budget: DirectWorkspaceBudget,
    ) -> Result<()> {
        let (width, height) = self.prepare_classic_dimensions(dimensions)?;
        if budget
            .validate_workspace(self.classic.allocated_bytes()?)
            .is_err()
        {
            self.classic = J2kCodeBlockDecodeWorkspace::default();
            self.classic.prepare(width, height)?;
            budget.validate_workspace(self.classic.allocated_bytes()?)?;
        }
        Ok(())
    }

    fn prepare_ht_dimensions(&mut self, dimensions: Option<(u32, u32)>) -> Result<(u32, u32)> {
        self.classic = J2kCodeBlockDecodeWorkspace::default();
        let dimensions = dimensions.ok_or(DecodingError::CodeBlockDecodeFailure)?;
        self.ht.prepare(dimensions.0, dimensions.1)?;
        Ok(dimensions)
    }

    fn prepare_classic_dimensions(&mut self, dimensions: Option<(u32, u32)>) -> Result<(u32, u32)> {
        self.ht = HtCodeBlockDecodeWorkspace::default();
        let dimensions = dimensions.ok_or(DecodingError::CodeBlockDecodeFailure)?;
        self.classic.prepare(dimensions.0, dimensions.1)?;
        Ok(dimensions)
    }
}
