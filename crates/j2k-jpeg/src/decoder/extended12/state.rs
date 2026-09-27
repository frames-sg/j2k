// SPDX-License-Identifier: MIT OR Apache-2.0

//! Extended-precision entropy block and restart state.

pub(super) use super::super::lossless_helpers::Extended12RestartTracker;
use super::super::{
    decode_block_with_activity, Backend, BitReader, BlockActivity, CoefficientBlock, JpegError,
    ResolvedPreparedComponentPlan,
};

/// Decode one block and inverse-transform it to 12-bit samples.
pub(super) fn decode_extended12_block_pixels(
    backend: Backend,
    br: &mut BitReader<'_>,
    component: ResolvedPreparedComponentPlan<'_>,
    prev_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    pixels: &mut [u16; 64],
) -> Result<(), JpegError> {
    // Decode through a local copy so the accumulator stays in registers for
    // the whole block instead of being stored back after every symbol; it is
    // written back even on error.
    let mut local = br.clone();
    let decoded = decode_block_with_activity(
        &mut local,
        component.dc_table,
        component.ac_table,
        prev_dc,
        component.quant,
        coeff,
    );
    *br = local;
    let activity = decoded?;
    match activity {
        BlockActivity::DcOnly => {
            pixels.fill(crate::idct::idct_islow_12bit_dc_only_sample(
                coeff.dc_coeff(),
            ));
        }
        BlockActivity::BottomHalfZero | BlockActivity::General => {
            backend.idct_12bit(coeff.coefficients(), pixels);
        }
    }
    Ok(())
}
