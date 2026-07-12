// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validation and fallible materialization of component sampling metadata.

use alloc::vec::Vec;

use crate::j2c::encode::{
    EncodeOptions, NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
};

use super::PlanConstruction;

pub(in crate::j2c::encode::single_tile::plan) fn validate_component_sampling(
    options: &EncodeOptions,
    num_components: u16,
) -> Result<(), &'static str> {
    let Some(component_sampling) = &options.component_sampling else {
        return Ok(());
    };
    if component_sampling.len() != usize::from(num_components) {
        return Err("component sampling count does not match component count");
    }
    if component_sampling
        .iter()
        .any(|&(x_rsiz, y_rsiz)| x_rsiz == 0 || y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    Ok(())
}

pub(in crate::j2c::encode::single_tile::plan) fn try_component_sampling(
    options: &EncodeOptions,
    num_components: u16,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<(u8, u8)>> {
    validate_component_sampling(options, num_components)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let mut construction = PlanConstruction::new(session, 0);
    let count = usize::from(num_components);
    let mut sampling = construction.try_vec(count, "single-tile component sampling")?;
    match &options.component_sampling {
        Some(source) => sampling.extend_from_slice(source),
        None => sampling.resize(count, (1, 1)),
    }
    Ok(sampling)
}
