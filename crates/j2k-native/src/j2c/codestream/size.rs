// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::bitplane::BITPLANE_BIT_SIZE;
use super::{ComponentSizeInfo, SizeData};
use crate::error::{bail, MarkerError, Result, ValidationError};
use crate::reader::BitReader;
use crate::{try_reserve_decode_elements, MAX_J2K_SPEC_COMPONENTS};

const MAX_PART1_COMPONENT_PRECISION: u8 = 38;

/// SIZ marker (A.5.1).
pub(super) fn size_marker(
    reader: &mut BitReader<'_>,
    max_component_bytes: usize,
) -> Result<SizeData> {
    let size_data = size_marker_inner(reader, max_component_bytes)?;

    if size_data.tile_width == 0
        || size_data.tile_height == 0
        || size_data.reference_grid_width == 0
        || size_data.reference_grid_height == 0
    {
        bail!(ValidationError::InvalidDimensions);
    }

    if size_data.tile_x_offset >= size_data.reference_grid_width
        || size_data.tile_y_offset >= size_data.reference_grid_height
    {
        bail!(ValidationError::InvalidDimensions);
    }

    // The tile grid offsets (XTOsiz, YTOsiz) are constrained to be no greater than the
    // image area offsets (B-3).
    if size_data.tile_x_offset > size_data.image_area_x_offset
        || size_data.tile_y_offset > size_data.image_area_y_offset
    {
        bail!(crate::TileError::InvalidOffsets);
    }

    // Also, the tile size plus the tile offset shall be greater than the image area offset.
    // This ensures that the first tile (tile 0) will contain at least one reference grid point
    // from the image area (B-4).
    if size_data
        .tile_x_offset
        .checked_add(size_data.tile_width)
        .ok_or(crate::TileError::InvalidOffsets)?
        <= size_data.image_area_x_offset
        || size_data
            .tile_y_offset
            .checked_add(size_data.tile_height)
            .ok_or(crate::TileError::InvalidOffsets)?
            <= size_data.image_area_y_offset
    {
        bail!(crate::TileError::InvalidOffsets);
    }

    for comp in &size_data.component_sizes {
        if comp.precision == 0 || comp.vertical_resolution == 0 || comp.horizontal_resolution == 0 {
            bail!(ValidationError::InvalidComponentMetadata);
        }
    }

    if size_data.image_width() > crate::MAX_J2K_IMAGE_DIMENSION
        || size_data.image_height() > crate::MAX_J2K_IMAGE_DIMENSION
    {
        bail!(ValidationError::ImageTooLarge);
    }

    // Isot is a u16, so no conforming codestream addresses more than 65,536
    // tiles (the encoder enforces the same ceiling). Rejecting here also stops
    // crafted SIZ values from overflowing num_tiles() or driving the eager
    // per-tile allocation in tile parsing.
    let num_tiles = u64::from(size_data.num_x_tiles()) * u64::from(size_data.num_y_tiles());
    if num_tiles > crate::MAX_J2K_TILE_COUNT {
        bail!(ValidationError::TooManyTiles);
    }

    Ok(size_data)
}

fn read_siz_byte(reader: &mut BitReader<'_>) -> Result<u8> {
    reader
        .read_byte()
        .ok_or(MarkerError::ParseFailure("SIZ").into())
}

fn read_siz_u16(reader: &mut BitReader<'_>) -> Result<u16> {
    reader
        .read_u16()
        .ok_or(MarkerError::ParseFailure("SIZ").into())
}

fn read_siz_u32(reader: &mut BitReader<'_>) -> Result<u32> {
    reader
        .read_u32()
        .ok_or(MarkerError::ParseFailure("SIZ").into())
}

#[expect(
    clippy::similar_names,
    reason = "paired axis, subband, and marker names follow JPEG 2000 specification notation"
)]
fn size_marker_inner(reader: &mut BitReader<'_>, max_component_bytes: usize) -> Result<SizeData> {
    // Length.
    let _ = read_siz_u16(reader)?;
    // Decoder capabilities.
    let _ = read_siz_u16(reader)?;

    let xsiz = read_siz_u32(reader)?;
    let ysiz = read_siz_u32(reader)?;
    let x_osiz = read_siz_u32(reader)?;
    let y_osiz = read_siz_u32(reader)?;
    let xt_siz = read_siz_u32(reader)?;
    let yt_siz = read_siz_u32(reader)?;
    let xto_siz = read_siz_u32(reader)?;
    let yto_siz = read_siz_u32(reader)?;
    let csiz = read_siz_u16(reader)?;

    if x_osiz >= xsiz || y_osiz >= ysiz {
        bail!(ValidationError::InvalidDimensions);
    }

    if csiz == 0 {
        bail!(ValidationError::InvalidComponentMetadata);
    }

    if csiz > MAX_J2K_SPEC_COMPONENTS {
        bail!(ValidationError::TooManyChannels);
    }

    let component_bytes = usize::from(csiz)
        .checked_mul(core::mem::size_of::<ComponentSizeInfo>())
        .ok_or(ValidationError::ImageTooLarge)?;
    if component_bytes > max_component_bytes {
        bail!(ValidationError::ImageTooLarge);
    }
    let mut components = Vec::new();
    try_reserve_decode_elements(&mut components, usize::from(csiz))?;
    let actual_component_bytes = components
        .capacity()
        .checked_mul(core::mem::size_of::<ComponentSizeInfo>())
        .ok_or(ValidationError::ImageTooLarge)?;
    if actual_component_bytes > max_component_bytes {
        bail!(ValidationError::ImageTooLarge);
    }
    for _ in 0..csiz {
        let ssiz = read_siz_byte(reader)?;
        let x_rsiz = read_siz_byte(reader)?;
        let y_rsiz = read_siz_byte(reader)?;

        let precision = (ssiz & 0x7F) + 1;
        let signed = (ssiz & 0x80) != 0;

        if precision > MAX_PART1_COMPONENT_PRECISION || u32::from(precision) > BITPLANE_BIT_SIZE {
            bail!(ValidationError::InvalidComponentMetadata);
        }

        components.push(ComponentSizeInfo {
            precision,
            signed,
            horizontal_resolution: x_rsiz,
            vertical_resolution: y_rsiz,
        });
    }

    // In case all components are sub-sampled at the same level, we
    // don't want to render them at the original resolution but instead
    // reduce their dimension so that we can assume a resolution of 1 for
    // all components. This makes the images much smaller.

    let mut x_shrink_factor = 1;
    let mut y_shrink_factor = 1;

    let hr = components[0].horizontal_resolution;
    let vr = components[0].vertical_resolution;
    let mut same_resolution = true;

    for component in &components[1..] {
        same_resolution &= component.horizontal_resolution == hr;
        same_resolution &= component.vertical_resolution == vr;
    }

    if same_resolution {
        x_shrink_factor = u32::from(hr);
        y_shrink_factor = u32::from(vr);
    }

    Ok(SizeData {
        reference_grid_width: xsiz,
        reference_grid_height: ysiz,
        image_area_x_offset: x_osiz,
        image_area_y_offset: y_osiz,
        tile_width: xt_siz,
        tile_height: yt_siz,
        tile_x_offset: xto_siz,
        tile_y_offset: yto_siz,
        component_sizes: components,
        x_shrink_factor,
        y_shrink_factor,
        x_resolution_shrink_factor: 1,
        y_resolution_shrink_factor: 1,
    })
}
