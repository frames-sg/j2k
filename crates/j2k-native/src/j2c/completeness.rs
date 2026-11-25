// SPDX-License-Identifier: MIT OR Apache-2.0

//! Which resolutions of a partial codestream can be decoded exactly.

use alloc::vec::Vec;

use super::codestream::Header;
use super::tile;
use crate::error::{Result, ValidationError};
use crate::reader::BitReader;

/// How much of a partial codestream is present, by resolution level.
///
/// Built by [`crate::Image::codestream_completeness`] for an image parsed
/// with [`crate::DecodeSettings::partial_codestream`]. Decoding at a
/// reduction of [`Self::min_exact_reduction`] levels or more reads only
/// packets that are present, so its samples equal those of the complete
/// codestream decoded at the same reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodestreamCompleteness {
    complete: bool,
    min_exact_reduction: Option<u8>,
    tile_min_exact_reduction: Vec<Option<u8>>,
}

impl CodestreamCompleteness {
    /// Whether every packet of every tile is present. A codestream can be
    /// complete without its EOC marker.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// The fewest resolution levels to drop for a decode of the whole image
    /// that needs only packets that are present: `Some(0)` when every packet
    /// is present, `None` while some tile still lacks part of its lowest
    /// resolution or needs a reduction deeper than every tile supports.
    #[must_use]
    pub fn min_exact_reduction(&self) -> Option<u8> {
        self.min_exact_reduction
    }

    /// [`Self::min_exact_reduction`] for each tile, in raster order. A region
    /// that only covers tiles with a value of `Some(r)` or less decodes
    /// exactly at reduction `r`.
    #[must_use]
    pub fn tile_min_exact_reduction(&self) -> &[Option<u8>] {
        &self.tile_min_exact_reduction
    }

    /// [`Self::tile_min_exact_reduction`], taken without copying.
    #[must_use]
    pub fn into_tile_min_exact_reduction(self) -> Vec<Option<u8>> {
        self.tile_min_exact_reduction
    }
}

/// Walk the packets of every tile of `data` and report which reductions read
/// only packets that are present.
pub(crate) fn codestream_completeness(
    data: &[u8],
    header: &Header<'_>,
    retained_baseline_bytes: usize,
) -> Result<CodestreamCompleteness> {
    let tiles = tile::parse(&mut BitReader::new(data), header, retained_baseline_bytes)?;
    let live_bytes = retained_baseline_bytes
        .checked_add(tiles.structural_workspace_bytes())
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut tile_min_exact_reduction = Vec::new();
    crate::try_reserve_decode_elements(&mut tile_min_exact_reduction, tiles.len())?;
    let mut complete = true;
    // The shallowest reduction every component of every tile can take.
    let mut deepest_supported = u8::MAX;
    for tile in tiles.iter() {
        let levels = super::segment::complete_resolution_levels(tile, header, live_bytes)?;
        let mut tile_reduction = Some(0_u8);
        for (component, &complete_levels) in tile.component_infos.iter().zip(&levels) {
            let total = component.num_resolution_levels();
            deepest_supported = deepest_supported.min(component.num_decomposition_levels());
            complete &= complete_levels >= total;
            // Reduction r keeps resolutions 0..total - r, which must all be
            // complete: r >= total - complete_levels, with at least the
            // lowest resolution present.
            tile_reduction = match (tile_reduction, complete_levels) {
                (_, 0) | (None, _) => None,
                (Some(reduction), levels) => Some(reduction.max(total.saturating_sub(levels))),
            };
        }
        tile_min_exact_reduction.push(tile_reduction);
    }
    let min_exact_reduction = tile_min_exact_reduction
        .iter()
        .try_fold(0_u8, |deepest, tile| {
            tile.map(|reduction| deepest.max(reduction))
        })
        .filter(|&reduction| reduction <= deepest_supported);
    Ok(CodestreamCompleteness {
        complete,
        min_exact_reduction,
        tile_min_exact_reduction,
    })
}
