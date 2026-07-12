// SPDX-License-Identifier: MIT OR Apache-2.0

//! Borrowed views over contiguous native and precomputed DWT owners.

use super::{DwtBandView, DwtComponentSource, DwtLevelView};
use crate::j2c::encode::{
    DwtDecomposition, NativeEncodePipelineResult, PrecomputedHtj2k53Component,
    PrecomputedHtj2k97Component,
};

fn band(coefficients: &[f32], width: u32, height: u32) -> DwtBandView<'_> {
    DwtBandView::Contiguous {
        coefficients,
        width,
        height,
    }
}

fn level_view<'a>(
    hl: &'a [f32],
    lh: &'a [f32],
    hh: &'a [f32],
    low_shape: (u32, u32),
    high_shape: (u32, u32),
) -> DwtLevelView<'a> {
    DwtLevelView {
        hl: band(hl, high_shape.0, low_shape.1),
        lh: band(lh, low_shape.0, high_shape.1),
        hh: band(hh, high_shape.0, high_shape.1),
    }
}

impl DwtComponentSource for DwtDecomposition {
    fn ll(&self) -> NativeEncodePipelineResult<DwtBandView<'_>> {
        Ok(band(&self.ll, self.ll_width, self.ll_height))
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }

    fn level(&self, index: usize) -> NativeEncodePipelineResult<Option<DwtLevelView<'_>>> {
        Ok(self.levels.get(index).map(|level| {
            level_view(
                &level.hl,
                &level.lh,
                &level.hh,
                (level.low_width, level.low_height),
                (level.high_width, level.high_height),
            )
        }))
    }
}

impl DwtComponentSource for PrecomputedHtj2k53Component {
    fn ll(&self) -> NativeEncodePipelineResult<DwtBandView<'_>> {
        Ok(band(&self.dwt.ll, self.dwt.ll_width, self.dwt.ll_height))
    }

    fn level_count(&self) -> usize {
        self.dwt.levels.len()
    }

    fn level(&self, index: usize) -> NativeEncodePipelineResult<Option<DwtLevelView<'_>>> {
        Ok(self.dwt.levels.get(index).map(|level| {
            level_view(
                &level.hl,
                &level.lh,
                &level.hh,
                (level.low_width, level.low_height),
                (level.high_width, level.high_height),
            )
        }))
    }
}

impl DwtComponentSource for PrecomputedHtj2k97Component {
    fn ll(&self) -> NativeEncodePipelineResult<DwtBandView<'_>> {
        Ok(band(&self.dwt.ll, self.dwt.ll_width, self.dwt.ll_height))
    }

    fn level_count(&self) -> usize {
        self.dwt.levels.len()
    }

    fn level(&self, index: usize) -> NativeEncodePipelineResult<Option<DwtLevelView<'_>>> {
        Ok(self.dwt.levels.get(index).map(|level| {
            level_view(
                &level.hl,
                &level.lh,
                &level.hh,
                (level.low_width, level.low_height),
                (level.high_width, level.high_height),
            )
        }))
    }
}
