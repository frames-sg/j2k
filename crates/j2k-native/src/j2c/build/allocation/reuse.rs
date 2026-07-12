// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reuse policy and transient-safe reservation for empty tile workspaces.

use super::DecompositionAllocationPlan;
use crate::error::{DecodingError, Result, ValidationError};
use crate::j2c::decode::DecompositionStorage;
use crate::{try_reserve_decode_elements, DEFAULT_MAX_DECODE_BYTES};
use alloc::vec::Vec;
use core::mem::size_of;

pub(super) fn discard_stale_capacity(
    storage: &mut DecompositionStorage<'_>,
    plan: &DecompositionAllocationPlan,
) {
    // Segment demand cannot be known until packet headers are parsed. Starting
    // empty prevents a prior segment-heavy tile from reducing the set of
    // otherwise valid current tiles.
    storage.segments = Vec::new();
    release_capacity_above(&mut storage.tile_decompositions, plan.tile_decompositions);
    release_capacity_above(&mut storage.decompositions, plan.decompositions);
    release_capacity_above(&mut storage.sub_bands, plan.sub_bands);
    release_capacity_above(&mut storage.precincts, plan.precincts);
    release_capacity_above(&mut storage.code_blocks, plan.code_blocks);
    release_capacity_above(&mut storage.layers, plan.layers);
    release_capacity_above(&mut storage.tag_tree_nodes, plan.tag_tree_nodes);
    release_capacity_above(&mut storage.coefficients, plan.coefficients);
    let integer_target = usize::from(storage.exact_integer_decode) * plan.coefficients;
    release_capacity_above(&mut storage.coefficients_i64, integer_target);
}

fn release_capacity_above<T>(values: &mut Vec<T>, target_capacity: usize) {
    values.clear();
    if values.capacity() > target_capacity {
        *values = Vec::new();
    }
}

pub(super) struct ReallocationBudget {
    live_bytes: usize,
}

impl ReallocationBudget {
    pub(super) fn for_storage(
        storage: &DecompositionStorage<'_>,
        retained_baseline_bytes: usize,
    ) -> Result<Self> {
        let live_bytes = retained_baseline_bytes
            .checked_add(storage.retained_capacity_bytes()?)
            .ok_or(ValidationError::ImageTooLarge)?;
        if live_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(Self { live_bytes })
    }

    pub(super) fn reserve<T>(&mut self, values: &mut Vec<T>, target_len: usize) -> Result<()> {
        values.clear();
        if target_len <= values.capacity() {
            return Ok(());
        }

        // A growing Vec may hold old and new buffers concurrently. If that
        // peak would cross the contract, release the empty old owner first.
        let requested_bytes = target_len
            .checked_mul(size_of::<T>())
            .ok_or(ValidationError::ImageTooLarge)?;
        let mut old_bytes = values
            .capacity()
            .checked_mul(size_of::<T>())
            .ok_or(ValidationError::ImageTooLarge)?;
        let mut transient_bytes = self
            .live_bytes
            .checked_add(requested_bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        if transient_bytes > DEFAULT_MAX_DECODE_BYTES && old_bytes != 0 {
            *values = Vec::new();
            self.live_bytes = self
                .live_bytes
                .checked_sub(old_bytes)
                .ok_or(ValidationError::ImageTooLarge)?;
            old_bytes = 0;
            transient_bytes = self
                .live_bytes
                .checked_add(requested_bytes)
                .ok_or(ValidationError::ImageTooLarge)?;
        }
        if transient_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }

        try_reserve_decode_elements(values, target_len)?;
        let new_bytes = values
            .capacity()
            .checked_mul(size_of::<T>())
            .ok_or(ValidationError::ImageTooLarge)?;
        let new_live_bytes = self
            .live_bytes
            .checked_sub(old_bytes)
            .and_then(|bytes| bytes.checked_add(new_bytes))
            .ok_or(ValidationError::ImageTooLarge)?;
        if new_live_bytes > DEFAULT_MAX_DECODE_BYTES {
            // Allocators may return a larger capacity than requested. Do not
            // retain that surprise allocation after reporting the cap breach.
            *values = Vec::new();
            return Err(ValidationError::ImageTooLarge.into());
        }
        self.live_bytes = new_live_bytes;
        Ok(())
    }
}

pub(super) fn reserve_decomposition_storage(
    plan: &DecompositionAllocationPlan,
    storage: &mut DecompositionStorage<'_>,
    retained_baseline_bytes: usize,
) -> Result<()> {
    let mut budget = ReallocationBudget::for_storage(storage, retained_baseline_bytes)?;
    budget.reserve(&mut storage.tile_decompositions, plan.tile_decompositions)?;
    budget.reserve(&mut storage.decompositions, plan.decompositions)?;
    budget.reserve(&mut storage.sub_bands, plan.sub_bands)?;
    budget.reserve(&mut storage.precincts, plan.precincts)?;
    budget.reserve(&mut storage.code_blocks, plan.code_blocks)?;
    budget.reserve(&mut storage.layers, plan.layers)?;
    budget.reserve(&mut storage.tag_tree_nodes, plan.tag_tree_nodes)?;
    budget.reserve(&mut storage.coefficients, plan.coefficients)?;
    if storage.exact_integer_decode {
        budget.reserve(&mut storage.coefficients_i64, plan.coefficients)?;
    }
    Ok(())
}

pub(in crate::j2c::build) fn push_preallocated<T>(values: &mut Vec<T>, value: T) -> Result<()> {
    if values.len() == values.capacity() {
        return Err(DecodingError::HostAllocationFailed.into());
    }
    values.push(value);
    Ok(())
}
