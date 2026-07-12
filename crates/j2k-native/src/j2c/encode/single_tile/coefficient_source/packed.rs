// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned decomposed-or-packed coefficient sources.

use alloc::vec::Vec;

use super::{DwtBandView, DwtComponentSource, DwtLevelView};
use crate::j2c::encode::{DwtDecomposition, NativeEncodePipelineResult};
use crate::j2c::fdwt::{PackedDwtGeometry, PackedSubbandView};

pub(in crate::j2c::encode) enum OwnedDwtComponent {
    Decomposed(DwtDecomposition),
    Packed(PackedF32DwtComponent),
}

pub(in crate::j2c::encode) struct PackedF32DwtComponent {
    pub(in crate::j2c::encode) coefficients: Vec<f32>,
    pub(in crate::j2c::encode) geometry: PackedDwtGeometry,
}

impl DwtComponentSource for OwnedDwtComponent {
    fn ll(&self) -> NativeEncodePipelineResult<DwtBandView<'_>> {
        match self {
            Self::Decomposed(decomposition) => decomposition.ll(),
            Self::Packed(packed) => packed.ll(),
        }
    }

    fn level_count(&self) -> usize {
        match self {
            Self::Decomposed(decomposition) => decomposition.level_count(),
            Self::Packed(packed) => packed.level_count(),
        }
    }

    fn level(&self, index: usize) -> NativeEncodePipelineResult<Option<DwtLevelView<'_>>> {
        match self {
            Self::Decomposed(decomposition) => decomposition.level(index),
            Self::Packed(packed) => packed.level(index),
        }
    }
}

impl DwtComponentSource for PackedF32DwtComponent {
    fn ll(&self) -> NativeEncodePipelineResult<DwtBandView<'_>> {
        let rect = self.geometry.ll()?;
        Ok(DwtBandView::Packed(PackedSubbandView::try_new(
            &self.coefficients,
            rect,
        )?))
    }

    fn level_count(&self) -> usize {
        usize::from(self.geometry.num_levels())
    }

    fn level(&self, index: usize) -> NativeEncodePipelineResult<Option<DwtLevelView<'_>>> {
        let Ok(index) = u8::try_from(index) else {
            return Ok(None);
        };
        if index >= self.geometry.num_levels() {
            return Ok(None);
        }
        let level = self.geometry.level(index)?;
        Ok(Some(DwtLevelView {
            hl: DwtBandView::Packed(PackedSubbandView::try_new(&self.coefficients, level.hl)?),
            lh: DwtBandView::Packed(PackedSubbandView::try_new(&self.coefficients, level.lh)?),
            hh: DwtBandView::Packed(PackedSubbandView::try_new(&self.coefficients, level.hh)?),
        }))
    }
}
