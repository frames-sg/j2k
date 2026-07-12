// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free coefficient views shared by transformed encode sources.

use super::super::{NativeEncodePipelineError, NativeEncodePipelineResult};
use crate::j2c::fdwt::PackedSubbandView;

mod contiguous;
mod packed;

pub(in crate::j2c::encode) use packed::{OwnedDwtComponent, PackedF32DwtComponent};

pub(super) enum DwtBandView<'a> {
    Contiguous {
        coefficients: &'a [f32],
        width: u32,
        height: u32,
    },
    Packed(PackedSubbandView<'a, f32>),
}

impl DwtBandView<'_> {
    pub(super) fn width(&self) -> u32 {
        match self {
            Self::Contiguous { width, .. } => *width,
            Self::Packed(view) => view.width(),
        }
    }

    pub(super) fn height(&self) -> u32 {
        match self {
            Self::Contiguous { height, .. } => *height,
            Self::Packed(view) => view.height(),
        }
    }
}

pub(super) struct DwtLevelView<'a> {
    pub(super) hl: DwtBandView<'a>,
    pub(super) lh: DwtBandView<'a>,
    pub(super) hh: DwtBandView<'a>,
}

pub(super) trait DwtComponentSource {
    fn ll(&self) -> NativeEncodePipelineResult<DwtBandView<'_>>;
    fn level_count(&self) -> usize;
    fn level(&self, index: usize) -> NativeEncodePipelineResult<Option<DwtLevelView<'_>>>;

    fn dimensions(&self) -> NativeEncodePipelineResult<(u32, u32)> {
        let Some(level_index) = self.level_count().checked_sub(1) else {
            let ll = self.ll()?;
            return Ok((ll.width(), ll.height()));
        };
        let level = self
            .level(level_index)?
            .ok_or(crate::EncodeError::InternalInvariant {
                what: "DWT source highest-resolution level is missing",
            })?;
        let width = level.lh.width().checked_add(level.hl.width()).ok_or(
            crate::EncodeError::ArithmeticOverflow {
                what: "DWT source width",
            },
        )?;
        let height = level.hl.height().checked_add(level.lh.height()).ok_or(
            crate::EncodeError::ArithmeticOverflow {
                what: "DWT source height",
            },
        )?;
        Ok((width, height))
    }
}

pub(super) fn validate_component_sampling_dwt_geometry<S: DwtComponentSource>(
    components: &[S],
    reference_width: u32,
    reference_height: u32,
    component_sampling: &[(u8, u8)],
) -> NativeEncodePipelineResult<()> {
    if components.len() != component_sampling.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "component sampling count does not match component count",
        ));
    }
    for (component, &(x_rsiz, y_rsiz)) in components.iter().zip(component_sampling) {
        let expected_width = reference_width.div_ceil(u32::from(x_rsiz.max(1)));
        let expected_height = reference_height.div_ceil(u32::from(y_rsiz.max(1)));
        if component.dimensions()? != (expected_width, expected_height) {
            return Err(NativeEncodePipelineError::internal_invariant(
                "component sampling requires component-sized DWT geometry",
            ));
        }
    }
    Ok(())
}
