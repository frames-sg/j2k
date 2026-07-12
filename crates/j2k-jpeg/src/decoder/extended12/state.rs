// SPDX-License-Identifier: MIT OR Apache-2.0

//! Extended-precision entropy block and restart state.

pub(super) use super::super::lossless_helpers::Extended12RestartTracker;
use super::super::{
    decode_block_with_activity, BitReader, BlockActivity, CoefficientBlock, JpegError,
    ResolvedPreparedComponentPlan,
};

pub(super) fn decode_extended12_block_pixels(
    br: &mut BitReader<'_>,
    component: ResolvedPreparedComponentPlan<'_>,
    prev_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    pixels: &mut [u16; 64],
) -> Result<(), JpegError> {
    let activity = decode_block_with_activity(
        br,
        component.dc_table,
        component.ac_table,
        prev_dc,
        component.quant,
        coeff,
    )?;
    match activity {
        BlockActivity::DcOnly => {
            pixels.fill(crate::idct::idct_islow_12bit_dc_only_sample(
                coeff.dc_coeff(),
            ));
        }
        BlockActivity::BottomHalfZero | BlockActivity::General => {
            crate::idct::idct_islow_12bit(coeff.coefficients(), pixels);
        }
    }
    Ok(())
}
