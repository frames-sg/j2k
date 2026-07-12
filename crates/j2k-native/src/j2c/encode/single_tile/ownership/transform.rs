// SPDX-License-Identifier: MIT OR Apache-2.0

//! Nested sample-plane and DWT owner accounting.

use alloc::vec::Vec;

use super::super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::super::DwtDecomposition;
use super::super::accelerator::PreparedComponentTransforms;
use super::super::coefficient_source::OwnedDwtComponent;
use crate::j2c::fdwt::DwtLevel;
use crate::EncodeResult;

pub(in crate::j2c::encode::single_tile) fn component_planes_retained_bytes(
    components: &[Vec<f32>],
    outer_capacity: usize,
) -> EncodeResult<usize> {
    let mut bytes =
        add_capacity::<Vec<f32>>(0, outer_capacity, "transform component plane owners")?;
    for component in components {
        bytes = add_capacity::<f32>(
            bytes,
            component.capacity(),
            "transform component plane samples",
        )?;
    }
    Ok(bytes)
}

pub(in crate::j2c::encode) fn dwt_decompositions_retained_bytes(
    decompositions: &[DwtDecomposition],
    outer_capacity: usize,
) -> EncodeResult<usize> {
    let mut bytes =
        add_capacity::<DwtDecomposition>(0, outer_capacity, "transform DWT component owners")?;
    for decomposition in decompositions {
        bytes = add_capacity::<f32>(bytes, decomposition.ll.capacity(), "transform LL samples")?;
        bytes = add_capacity::<DwtLevel>(
            bytes,
            decomposition.levels.capacity(),
            "transform DWT level owners",
        )?;
        for level in &decomposition.levels {
            bytes = add_capacity::<f32>(bytes, level.hl.capacity(), "transform HL samples")?;
            bytes = add_capacity::<f32>(bytes, level.lh.capacity(), "transform LH samples")?;
            bytes = add_capacity::<f32>(bytes, level.hh.capacity(), "transform HH samples")?;
        }
    }
    Ok(bytes)
}

pub(in crate::j2c::encode::single_tile) fn dwt_component_sources_retained_bytes(
    components: &[OwnedDwtComponent],
    outer_capacity: usize,
) -> EncodeResult<usize> {
    let mut bytes =
        add_capacity::<OwnedDwtComponent>(0, outer_capacity, "transform DWT component owners")?;
    for component in components {
        match component {
            OwnedDwtComponent::Decomposed(decomposition) => {
                bytes = checked_add_bytes(
                    bytes,
                    dwt_decompositions_retained_bytes(core::slice::from_ref(decomposition), 0)?,
                    "transform decomposed DWT component",
                )?;
            }
            OwnedDwtComponent::Packed(packed) => {
                bytes = add_capacity::<f32>(
                    bytes,
                    packed.coefficients.capacity(),
                    "transform packed DWT samples",
                )?;
            }
        }
    }
    Ok(bytes)
}

/// Peak owner bytes created by the CPU DWT for one component while its input
/// plane remains owned by the caller. Extracted bands partition the source
/// grid, so the larger overlap is two full planes plus level metadata.
pub(in crate::j2c::encode) fn cpu_dwt_transient_bytes(
    sample_count: usize,
    num_levels: u8,
) -> EncodeResult<usize> {
    let coefficient_bytes =
        checked_element_bytes::<f32>(sample_count, "CPU DWT packed coefficient samples")?;
    let extracted_bytes =
        checked_element_bytes::<f32>(sample_count, "CPU DWT extracted coefficient samples")?;
    let bytes = checked_add_bytes(
        coefficient_bytes,
        extracted_bytes,
        "CPU DWT transient samples",
    )?;
    checked_add_bytes(
        bytes,
        checked_element_bytes::<DwtLevel>(usize::from(num_levels), "CPU DWT level metadata")?,
        "CPU DWT transient owners",
    )
}

pub(in crate::j2c::encode::single_tile) fn prepared_transforms_retained_bytes(
    prepared: &PreparedComponentTransforms,
) -> EncodeResult<usize> {
    dwt_component_sources_retained_bytes(
        &prepared.decompositions,
        prepared.decompositions.capacity(),
    )
}

fn add_capacity<T>(bytes: usize, capacity: usize, what: &'static str) -> EncodeResult<usize> {
    checked_add_bytes(bytes, checked_element_bytes::<T>(capacity, what)?, what)
}

#[cfg(test)]
#[path = "transform/tests.rs"]
mod tests;
