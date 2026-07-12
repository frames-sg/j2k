// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated and phase-accounted standard single-tile encode plans.

use crate::j2c::encode::{
    CodeBlockGeometry, ComponentRoiEncodePlan, EncodeParams, QuantStepSize, Vec,
};

mod build;
mod construction;
#[cfg(test)]
mod tests;
mod validation;

pub(super) use build::build_single_tile_plan;
pub(super) use validation::{
    validate_encode_request, validate_non_pixel_single_tile_request, NonPixelSingleTileRequest,
};

pub(super) enum ValidatedEncodeRoute {
    MultiTile { tile_width: u32, tile_height: u32 },
    SingleTile(ValidatedSingleTileInput),
}

pub(super) struct ValidatedSingleTileInput {
    pub(super) num_pixels: usize,
    pub(super) component_sampling: Vec<(u8, u8)>,
    pub(super) high_bit_exact: bool,
    pub(super) code_block_geometry: CodeBlockGeometry,
}

pub(super) struct SingleTilePlan {
    pub(super) num_pixels: usize,
    pub(super) high_bit_exact: bool,
    pub(super) use_mct: bool,
    pub(super) num_levels: u8,
    pub(super) guard_bits: u8,
    pub(super) step_sizes: Vec<QuantStepSize>,
    pub(super) quant_params: Vec<(u16, u16)>,
    pub(super) component_step_sizes: Vec<Vec<QuantStepSize>>,
    pub(super) roi_plans: Vec<ComponentRoiEncodePlan>,
    pub(super) roi_component_shifts: Vec<u8>,
    pub(super) cb_width: u32,
    pub(super) cb_height: u32,
    pub(super) ht_target_coding_passes: u8,
    pub(super) params: EncodeParams,
}

/// Marker-only state retained after transform, Tier-1, and packetization.
pub(super) struct CodestreamFinalPlan {
    pub(super) params: EncodeParams,
    pub(super) quant_params: Vec<(u16, u16)>,
}

impl SingleTilePlan {
    pub(super) fn into_codestream_final_plan(self) -> CodestreamFinalPlan {
        let Self {
            params,
            quant_params,
            ..
        } = self;
        CodestreamFinalPlan {
            params,
            quant_params,
        }
    }
}
