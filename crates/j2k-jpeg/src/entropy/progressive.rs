// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive JPEG decode facade over focused metadata, allocation, scan,
//! and rendering owners.

use alloc::vec::Vec;

use crate::backend::Backend;
use crate::error::{JpegError, Warning};
use crate::output::OutputWriter;

mod allocation;
mod model;
mod render;
mod scan;
mod terminal;

pub(crate) use self::allocation::COMPONENT_IMAGE_METADATA_BYTES;
pub(crate) use self::model::{
    PreparedProgressiveComponentPlan, PreparedProgressivePlan, PreparedProgressiveScan,
    PreparedProgressiveScanComponent, ProgressiveDctBlocks,
};
pub(crate) use self::scan::decode_progressive_dct_blocks;

use self::allocation::{checked_phase_capacity, component_image_capacity_bytes};
use self::render::{emit_component_images, render_component_images};

pub(crate) fn decode_progressive<W: OutputWriter>(
    plan: &PreparedProgressivePlan,
    backend: Backend,
    bytes: &[u8],
    writer: &mut W,
    external_live_bytes: usize,
) -> Result<Vec<Warning>, JpegError> {
    let dct_blocks = decode_progressive_dct_blocks(plan, bytes, external_live_bytes)?;
    let coefficient_live_bytes = checked_phase_capacity(
        external_live_bytes,
        dct_blocks.capacity_bytes()?,
        plan.scratch_bytes,
    )?;
    let ProgressiveDctBlocks { quantized: coeffs } = dct_blocks;
    let images = render_component_images(plan, backend, &coeffs, coefficient_live_bytes)?;
    drop(coeffs);
    let image_live_bytes = checked_phase_capacity(
        external_live_bytes,
        component_image_capacity_bytes(images.capacity(), &images)?,
        plan.scratch_bytes,
    )?;
    emit_component_images(plan, &images, image_live_bytes, writer)?;
    Ok(Vec::new())
}

#[cfg(test)]
mod tests;
