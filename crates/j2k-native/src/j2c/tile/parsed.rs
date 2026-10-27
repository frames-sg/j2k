// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parsed tile ownership and its carried live-allocation baseline.

use alloc::vec::Vec;
use core::ops::Deref;

use super::Tile;

pub(crate) struct ParsedTiles<'a> {
    tiles: Vec<Tile<'a>>,
    structural_workspace_bytes: usize,
    metadata_owner_bytes: usize,
}

impl<'a> ParsedTiles<'a> {
    #[cfg(feature = "std")]
    pub(crate) fn build_packet_directory(
        &self,
        header: &super::Header<'_>,
        retained_live_bytes: usize,
        max_directory_bytes: usize,
    ) -> Option<super::super::segment::directory::PacketDirectory> {
        if self.tiles.len() != 1 || max_directory_bytes == 0 {
            return None;
        }
        super::super::segment::directory::PacketDirectory::prepare(
            &self.tiles[0],
            header,
            retained_live_bytes,
            max_directory_bytes,
        )
        .ok()
        .flatten()
    }

    #[cfg(feature = "std")]
    pub(crate) fn set_packet_directory(
        &mut self,
        directory: Option<alloc::sync::Arc<super::super::segment::directory::PacketDirectory>>,
    ) {
        if self.tiles.len() == 1 {
            self.tiles[0].packet_directory = directory;
        }
    }

    pub(super) fn new(
        tiles: Vec<Tile<'a>>,
        structural_workspace_bytes: usize,
        metadata_owner_bytes: usize,
    ) -> Self {
        Self {
            tiles,
            structural_workspace_bytes,
            metadata_owner_bytes,
        }
    }

    pub(crate) fn structural_workspace_bytes(&self) -> usize {
        self.structural_workspace_bytes
    }

    pub(crate) fn metadata_owner_bytes(&self) -> usize {
        self.metadata_owner_bytes
    }
}

impl<'a> Deref for ParsedTiles<'a> {
    type Target = [Tile<'a>];

    fn deref(&self) -> &Self::Target {
        &self.tiles
    }
}
