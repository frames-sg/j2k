// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only conversion from the internal 5/3 decomposition to the typed API output.

use super::super::single_tile::ownership::dwt_decompositions_retained_bytes;
#[cfg(test)]
use super::super::NativeEncodeRetainedInput;
use super::super::{
    allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed},
    DwtDecomposition, J2kForwardDwt53Level, J2kForwardDwt53Output, NativeEncodePipelineResult,
    NativeEncodeSession, Vec,
};

pub(in crate::j2c::encode) fn try_forward_dwt53_output_from_decomposition(
    decomposition: DwtDecomposition,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
) -> NativeEncodePipelineResult<J2kForwardDwt53Output> {
    let source_bytes = dwt_decompositions_retained_bytes(core::slice::from_ref(&decomposition), 0)?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            source_bytes,
            "typed component DWT source",
        )?,
        "typed component DWT source",
    )?;
    let requested_owner_bytes = checked_element_bytes::<J2kForwardDwt53Level>(
        decomposition.levels.len(),
        "typed component DWT level owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(
                source_bytes,
                requested_owner_bytes,
                "typed component DWT conversion overlap",
            )?,
            "typed component DWT conversion overlap",
        )?,
        "typed component DWT conversion overlap",
    )?;
    let mut levels = Vec::new();
    levels
        .try_reserve_exact(decomposition.levels.len())
        .map_err(|_| {
            host_allocation_failed("typed component DWT level owners", requested_owner_bytes)
        })?;
    let actual_owner_bytes = checked_element_bytes::<J2kForwardDwt53Level>(
        levels.capacity(),
        "typed component DWT level owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(
                source_bytes,
                actual_owner_bytes,
                "typed component DWT conversion overlap",
            )?,
            "typed component DWT conversion overlap",
        )?,
        "typed component DWT conversion overlap",
    )?;
    levels.extend(decomposition.levels.into_iter().map(|level| {
        let width = level.low_width + level.high_width;
        let height = level.low_height + level.high_height;
        J2kForwardDwt53Level {
            hl: level.hl,
            lh: level.lh,
            hh: level.hh,
            width,
            height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
        }
    }));
    Ok(J2kForwardDwt53Output {
        ll: decomposition.ll,
        ll_width: decomposition.ll_width,
        ll_height: decomposition.ll_height,
        levels,
    })
}

#[cfg(test)]
pub(in crate::j2c::encode) fn forward_dwt53_output_from_decomposition(
    decomposition: DwtDecomposition,
) -> J2kForwardDwt53Output {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("test DWT conversion session");
    try_forward_dwt53_output_from_decomposition(decomposition, &session, 0)
        .expect("test DWT conversion")
}
