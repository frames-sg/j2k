// SPDX-License-Identifier: MIT OR Apache-2.0

//! Derived offsets for repeated single-tile-part RPCL region decoding.
//! Offsets never impersonate signalled PLT metadata.

use alloc::vec::Vec;
use core::mem::size_of;

use super::{parse_with_walk, PacketWalk};
use crate::error::{Result, ValidationError};
use crate::j2c::build::{self, PrecinctSelection};
use crate::j2c::codestream::{Header, ProgressionOrder};
use crate::j2c::decode::{DecodeAllocationBudget, DecompositionStorage};
use crate::j2c::tile::Tile;

#[derive(Debug)]
pub(crate) struct PacketDirectory {
    boundaries: Vec<u64>,
}

impl PacketDirectory {
    pub(crate) fn retained_bytes(&self) -> usize {
        self.boundaries.capacity() * size_of::<u64>()
    }

    pub(super) fn range(&self, ordinal: usize) -> Option<(usize, usize)> {
        Some((
            usize::try_from(*self.boundaries.get(ordinal)?).ok()?,
            usize::try_from(*self.boundaries.get(ordinal.checked_add(1)?)?).ok()?,
        ))
    }

    /// Any error here disables the optional index, after ordinary tile
    /// validation has already succeeded. In particular, a bad trailing packet
    /// must not make an earlier, otherwise valid ROI fail eagerly.
    pub(crate) fn prepare(
        tile: &Tile<'_>,
        header: &Header<'_>,
        live_bytes: usize,
        max_directory_bytes: usize,
    ) -> Result<Option<Self>> {
        if !matches!(
            tile.progression_order,
            ProgressionOrder::ResolutionPositionComponentLayer
        ) || !tile.progression_changes.is_empty()
            || tile.tile_parts.len() != 1
        {
            return Ok(None);
        }
        let Some((start, end)) = tile.tile_parts[0].unindexed_inline_bounds() else {
            return Ok(None);
        };
        // Reject oversized directories before allocating a selection map.
        let precincts = tile
            .component_tiles()
            .try_fold(0_usize, |count, component| {
                component
                    .resolution_tiles()
                    .try_fold(count, |count, resolution| {
                        count.checked_add(usize::try_from(resolution.num_precincts()).ok()?)
                    })
            })
            .ok_or(ValidationError::ImageTooLarge)?;
        let count = precincts
            .checked_mul(usize::from(tile.num_layers))
            .ok_or(ValidationError::ImageTooLarge)?
            .checked_add(1)
            .ok_or(ValidationError::ImageTooLarge)?;
        if count
            .checked_mul(size_of::<u64>())
            .ok_or(ValidationError::ImageTooLarge)?
            > max_directory_bytes
        {
            return Ok(None);
        }
        let selection = PrecinctSelection::for_packet_scan(tile, live_bytes)?;
        let mut budget = DecodeAllocationBudget::from_live_bytes(live_bytes)?;
        budget.include_bytes(selection.owned_bytes())?;
        let mut boundaries = Vec::new();
        budget.reserve_new(&mut boundaries, count)?;
        if boundaries.capacity() * size_of::<u64>() > max_directory_bytes {
            return Ok(None);
        }
        boundaries.push(u64::try_from(start).map_err(|_| ValidationError::ImageTooLarge)?);
        let mut storage = DecompositionStorage::default();
        build::build(
            tile,
            &mut storage,
            budget.live_bytes(),
            false,
            build::BuildWorkspace::MetadataOnly,
            Some(&selection),
        )?;
        let mut walk = PacketWalk {
            recording: Some(&mut boundaries),
            ..PacketWalk::default()
        };
        parse_with_walk(tile, header, &mut storage, Some(&selection), &mut walk)?;
        // Exact expected coverage, including all layers, and the entire
        // tile-part. No partial directory is ever published.
        if boundaries.len() != count || boundaries.last().copied() != u64::try_from(end).ok() {
            return Ok(None);
        }
        Ok(Some(Self { boundaries }))
    }
}
