// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parsed tile ownership and its carried live-allocation baseline.

use alloc::vec::Vec;
use core::ops::Deref;

use super::Tile;

pub(crate) struct ParsedTiles<'a> {
    tiles: Vec<Tile<'a>>,
    structural_workspace_bytes: usize,
}

impl<'a> ParsedTiles<'a> {
    pub(super) fn new(tiles: Vec<Tile<'a>>, structural_workspace_bytes: usize) -> Self {
        Self {
            tiles,
            structural_workspace_bytes,
        }
    }

    pub(crate) fn structural_workspace_bytes(&self) -> usize {
        self.structural_workspace_bytes
    }
}

impl<'a> Deref for ParsedTiles<'a> {
    type Target = [Tile<'a>];

    fn deref(&self) -> &Self::Target {
        &self.tiles
    }
}
