// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked scratch and launch planning for staged CUDA baseline JPEG encode.

use super::CudaJpegBaselineEncodeParams;
use crate::{error::CudaError, kernels::CudaLaunchGeometry};

const PRECOMPUTE_THREADS: u32 = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CudaJpegStagedEncodePlan {
    pub(super) coefficient_bytes: usize,
    pub(super) total_mcus: u32,
    pub(super) precompute_geometry: CudaLaunchGeometry,
}

pub(super) fn checked_staged_encode_plan(
    params: &[CudaJpegBaselineEncodeParams],
) -> Result<CudaJpegStagedEncodePlan, CudaError> {
    let mut coefficient_count = 0usize;
    let mut total_mcus = 0usize;
    for (tile_index, params) in params.iter().copied().enumerate() {
        let components = usize::try_from(params.components)
            .map_err(|_| invalid_staged_plan(tile_index, "component count exceeds usize"))?;
        if !(1..=3).contains(&components) {
            return Err(invalid_staged_plan(
                tile_index,
                "staged encode requires one to three components",
            ));
        }
        let component_blocks = [
            params.h0.checked_mul(params.v0),
            params.h1.checked_mul(params.v1),
            params.h2.checked_mul(params.v2),
        ];
        let mut blocks_per_mcu = 0usize;
        for blocks in component_blocks.into_iter().take(components) {
            let blocks = blocks
                .ok_or_else(|| invalid_staged_plan(tile_index, "block geometry overflowed"))?;
            if blocks == 0 {
                return Err(invalid_staged_plan(
                    tile_index,
                    "component block geometry must be nonzero",
                ));
            }
            blocks_per_mcu = blocks_per_mcu.checked_add(blocks as usize).ok_or_else(|| {
                invalid_staged_plan(tile_index, "blocks-per-MCU count overflowed")
            })?;
        }
        let tile_mcus = (params.mcus_per_row as usize)
            .checked_mul(params.mcu_rows as usize)
            .ok_or_else(|| invalid_staged_plan(tile_index, "MCU count overflowed"))?;
        if tile_mcus == 0 {
            return Err(invalid_staged_plan(
                tile_index,
                "MCU geometry must be nonzero",
            ));
        }
        total_mcus = total_mcus
            .checked_add(tile_mcus)
            .ok_or_else(|| invalid_staged_plan(tile_index, "batch MCU count overflowed"))?;
        let tile_coefficients = tile_mcus
            .checked_mul(blocks_per_mcu)
            .and_then(|blocks| blocks.checked_mul(64))
            .ok_or_else(|| invalid_staged_plan(tile_index, "coefficient count overflowed"))?;
        coefficient_count = coefficient_count
            .checked_add(tile_coefficients)
            .ok_or_else(|| invalid_staged_plan(tile_index, "batch coefficient count overflowed"))?;
    }

    let total_mcus =
        u32::try_from(total_mcus).map_err(|_| CudaError::LengthTooLarge { len: total_mcus })?;
    u32::try_from(coefficient_count).map_err(|_| CudaError::LengthTooLarge {
        len: coefficient_count,
    })?;
    let coefficient_bytes = coefficient_count
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(CudaError::LengthTooLarge {
            len: coefficient_count,
        })?;
    let precompute_geometry = CudaLaunchGeometry::new(
        (total_mcus.div_ceil(PRECOMPUTE_THREADS), 1, 1),
        (PRECOMPUTE_THREADS, 1, 1),
    )
    .ok_or_else(|| CudaError::InvalidArgument {
        message: "JPEG CUDA staged coefficient launch exceeds static CUDA limits".to_string(),
    })?;
    Ok(CudaJpegStagedEncodePlan {
        coefficient_bytes,
        total_mcus,
        precompute_geometry,
    })
}

fn invalid_staged_plan(tile_index: usize, message: &str) -> CudaError {
    CudaError::InvalidArgument {
        message: format!("JPEG CUDA staged encode tile {tile_index}: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{checked_staged_encode_plan, CudaJpegBaselineEncodeParams};

    fn params(components: u32, mcus_per_row: u32, mcu_rows: u32) -> CudaJpegBaselineEncodeParams {
        CudaJpegBaselineEncodeParams {
            components,
            mcus_per_row,
            mcu_rows,
            h0: if components == 1 { 1 } else { 2 },
            v0: 1,
            h1: u32::from(components > 1),
            v1: u32::from(components > 1),
            h2: u32::from(components > 2),
            v2: u32::from(components > 2),
            ..CudaJpegBaselineEncodeParams::default()
        }
    }

    #[test]
    fn staged_plan_counts_one_precompute_work_item_per_mcu_and_exact_scratch() {
        let plan = checked_staged_encode_plan(&[params(3, 2, 3), params(1, 1, 2)])
            .expect("checked staged encode plan");

        // First tile: 6 MCUs * 4 blocks. Second: 2 MCUs * 1 block.
        assert_eq!(plan.total_mcus, 8);
        assert_eq!(plan.coefficient_bytes, (6 * 4 + 2) * 64 * 4);
        assert_eq!(plan.precompute_geometry.grid(), (1, 1, 1));
        assert_eq!(plan.precompute_geometry.block(), (128, 1, 1));
    }

    #[test]
    fn staged_plan_rejects_component_and_coefficient_index_overflow() {
        let invalid = params(0, 1, 1);
        assert!(checked_staged_encode_plan(&[invalid]).is_err());

        let overflow = params(3, u32::MAX, u32::MAX);
        assert!(checked_staged_encode_plan(&[overflow]).is_err());
    }
}
