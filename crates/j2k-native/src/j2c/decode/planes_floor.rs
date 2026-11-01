// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a full-image decode into component planes charges the decode budget
//! at least, planned from the codestream before anything is decoded.
//!
//! The planes decode charges, in order: the parsed tiles' structural bytes
//! and every component's image-sized plane (`TileDecodeContext::reset`), then
//! for each tile its code-block records, the coefficients of its largest
//! component and that component's IDWT workspace (`build`'s allocation plan,
//! with the component coefficient layout). Each charge here is the same or
//! smaller, so a decode whose floor passes the budget is refused on the
//! planes, serially or on tile workers, and a decode the planes can finish
//! never has a floor past the budget. The floor leaves out what `build`
//! counts by walking precincts (most tag-tree nodes) and every later stage.

use core::mem::size_of;

use crate::checked_decode_sample_count;
use crate::error::{Result, ValidationError};
use crate::j2c::build::{global_code_block_count, IdwtWorkspaceTracker, RecordCounts, SubBandType};
use crate::j2c::codestream::Header;
use crate::j2c::rect::IntRect;
use crate::j2c::tile::{self, ResolutionTile, Tile};
use crate::j2c::ComponentData;
use crate::math::{SimdBuffer, SIMD_WIDTH};
use crate::reader::BitReader;

/// The bytes a full-image decode of `data` into component planes charges at
/// least, when its full-image region decode could stream its rows instead:
/// the image streams as far as its main header says (the caller checks
/// `Image::region_decode_streams_rows`), and every tile has its components
/// at full sampling and an inverse MCT only over three or more components
/// whose first three share a wavelet (as `streams_rows` requires). `None`
/// when it could not, or when the header alone puts the planes, one tile's
/// coefficients and its IDWT buffers within half of `budget`, which leaves
/// the planes decode far from it unless its metadata is unusually large.
///
/// `retained_image_bytes` is what the image holds while it decodes, the
/// tile parse's baseline. Errors are the tile parse's, which the planes
/// decode reports too.
pub(crate) fn full_decode_planes_floor(
    data: &[u8],
    header: &Header<'_>,
    retained_image_bytes: usize,
    budget: usize,
) -> Result<Option<usize>> {
    let planes = planes_bytes(header)?;
    if header_estimate(header, planes).saturating_mul(2) <= budget {
        return Ok(None);
    }
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header, retained_image_bytes)?;
    if tiles.is_empty() {
        return Ok(None);
    }
    let skipped = header.skipped_resolution_levels;
    let mut tile_floor = 0_usize;
    for tile in tiles.iter() {
        if !tile_streams_rows(tile, header.component_infos.len()) {
            return Ok(None);
        }
        tile_floor = tile_floor.max(tile_workspace_floor(tile, skipped));
    }
    Ok(Some(
        tiles
            .structural_workspace_bytes()
            .saturating_add(planes)
            .saturating_add(tile_floor),
    ))
}

/// What `reset` charges for every component's plane of the output image.
fn planes_bytes(header: &Header<'_>) -> Result<usize> {
    region_planes_bytes(
        header,
        header.size_data.image_width(),
        header.size_data.image_height(),
    )
}

/// What `reset` charges for every component's plane of a `width` x
/// `height` output.
pub(super) fn region_planes_bytes(header: &Header<'_>, width: u32, height: u32) -> Result<usize> {
    let samples = checked_decode_sample_count(width, height)?;
    let padded =
        SimdBuffer::<SIMD_WIDTH>::padded_len(samples).ok_or(ValidationError::ImageTooLarge)?;
    let components = header.component_infos.len();
    Ok(padded
        .saturating_mul(size_of::<f32>())
        .saturating_mul(components)
        .saturating_add(components.saturating_mul(size_of::<ComponentData>())))
}

/// The planes, plus the coefficients and twice the output-resolution
/// samples of one SIZ tile in `f32`: about what the planes decode charges
/// for its largest tile, the IDWT's output, next-level scratch and band
/// scratch included, without records or metadata.
fn header_estimate(header: &Header<'_>, planes: usize) -> usize {
    let size = &header.size_data;
    let full = |tile: u32, extent: u32, offset: u32| tile.min(extent.saturating_sub(offset));
    let width = full(
        size.tile_width,
        size.reference_grid_width,
        size.tile_x_offset,
    );
    let height = full(
        size.tile_height,
        size.reference_grid_height,
        size.tile_y_offset,
    );
    let reduced = |length: u32| {
        u32::try_from(u64::from(length) >> header.skipped_resolution_levels)
            .unwrap_or(u32::MAX)
            .saturating_add(2)
    };
    let full_samples = (width as usize).saturating_mul(height as usize);
    let output_samples = (reduced(width) as usize).saturating_mul(reduced(height) as usize);
    planes.saturating_add(
        full_samples
            .saturating_add(output_samples.saturating_mul(2))
            .saturating_mul(size_of::<f32>()),
    )
}

/// The tile-level conditions of `streams_rows` that `tile`'s header
/// settles.
fn tile_streams_rows(tile: &Tile<'_>, components: usize) -> bool {
    let infos = &tile.component_infos;
    infos.len() == components
        && infos.iter().all(|component| {
            component.size_info.horizontal_resolution == 1
                && component.size_info.vertical_resolution == 1
                && !component.requires_exact_integer_decode()
        })
        && (!tile.mct
            || (components >= 3
                && infos[1..3].iter().all(|component| {
                    component.wavelet_transform() == infos[0].wavelet_transform()
                })))
}

/// What `build` charges for `tile` at least beside the planes: its records,
/// its largest component's `f32` coefficients at full resolution, and the
/// largest component IDWT workspace at the output resolution.
fn tile_workspace_floor(tile: &Tile<'_>, skipped: u8) -> usize {
    let mut records = RecordCounts::default();
    let mut coefficients = 0_usize;
    let mut idwt = 0_usize;
    for component_tile in tile.component_tiles() {
        coefficients = coefficients.max(rect_samples(component_tile.rect));
        records.tile_decompositions = records.tile_decompositions.saturating_add(1);
        let active = usize::from(
            component_tile
                .component_info
                .num_decomposition_levels()
                .saturating_sub(skipped),
        );
        let mut idwt_tracker = IdwtWorkspaceTracker::new(
            Some(active),
            component_tile.component_info.wavelet_transform(),
        );
        let mut ll_rect = IntRect::from_xywh(0, 0, 0, 0);
        for resolution_tile in component_tile.resolution_tiles() {
            if resolution_tile.resolution == 0 {
                ll_rect = resolution_tile.sub_band_rect(SubBandType::LowLow);
                add_sub_band_records(
                    &mut records,
                    &resolution_tile,
                    SubBandType::LowLow,
                    tile.num_layers,
                );
                continue;
            }
            records.decompositions = records.decompositions.saturating_add(1);
            idwt_tracker.observe(resolution_tile.rect);
            for sub_band_type in [
                SubBandType::HighLow,
                SubBandType::LowHigh,
                SubBandType::HighHigh,
            ] {
                add_sub_band_records(
                    &mut records,
                    &resolution_tile,
                    sub_band_type,
                    tile.num_layers,
                );
            }
        }
        idwt = idwt.max(
            idwt_tracker
                .finish(ll_rect)
                .ok()
                .flatten()
                .unwrap_or(usize::MAX),
        );
    }
    let records_bytes = records.checked_bytes().unwrap_or(usize::MAX);
    records_bytes.saturating_add(
        coefficients
            .saturating_add(idwt)
            .saturating_mul(size_of::<f32>()),
    )
}

fn rect_samples(rect: IntRect) -> usize {
    (rect.width() as usize).saturating_mul(rect.height() as usize)
}

/// Add one sub-band's records to the floor, counting tag-tree nodes only as
/// the two trees of each precinct need at least: a root and one leaf per
/// code-block.
fn add_sub_band_records(
    records: &mut RecordCounts,
    resolution_tile: &ResolutionTile<'_>,
    sub_band_type: SubBandType,
    layers: u8,
) {
    let precincts = usize::try_from(resolution_tile.num_precincts()).unwrap_or(usize::MAX);
    let code_blocks = global_code_block_count(
        resolution_tile,
        resolution_tile.sub_band_rect(sub_band_type),
    )
    .unwrap_or(usize::MAX);
    records.sub_bands = records.sub_bands.saturating_add(1);
    records.precincts = records.precincts.saturating_add(precincts);
    records.code_blocks = records.code_blocks.saturating_add(code_blocks);
    records.layers = records
        .layers
        .saturating_add(code_blocks.saturating_mul(usize::from(layers)));
    records.tag_tree_nodes = records
        .tag_tree_nodes
        .saturating_add(precincts.max(code_blocks).saturating_mul(2));
}
