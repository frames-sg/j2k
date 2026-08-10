// SPDX-License-Identifier: MIT OR Apache-2.0

//! SIZ marker image, tile-grid, and component geometry.

use alloc::vec::Vec;

use crate::error::{Result, ValidationError};

#[derive(Debug)]
pub(crate) struct SizeData {
    /// Decoder capabilities/profile word (`Rsiz`).
    pub(crate) decoder_capabilities: u16,
    /// Width of the reference grid (Xsiz).
    pub(crate) reference_grid_width: u32,
    /// Height of the reference grid (Ysiz).
    pub(crate) reference_grid_height: u32,
    /// Horizontal offset from the origin of the reference grid to the
    /// left side of the image area (`XOsiz`).
    pub(crate) image_area_x_offset: u32,
    /// Vertical offset from the origin of the reference grid to the top side of the image area (`YOsiz`).
    pub(crate) image_area_y_offset: u32,
    /// Width of one reference tile with respect to the reference grid (`XTSiz`).
    pub(crate) tile_width: u32,
    /// Height of one reference tile with respect to the reference grid (`YTSiz`).
    pub(crate) tile_height: u32,
    /// Horizontal offset from the origin of the reference grid to the left side of the first tile (`XTOSiz`).
    pub(crate) tile_x_offset: u32,
    /// Vertical offset from the origin of the reference grid to the top side of the first tile (`YTOSiz`).
    pub(crate) tile_y_offset: u32,
    /// Component information (SSiz/XRSiz/YRSiz).
    pub(crate) component_sizes: Vec<ComponentSizeInfo>,
    /// Shrink factor in the x direction. See the comment in the parsing method.
    pub(crate) x_shrink_factor: u32,
    /// Shrink factor in the y direction. See the comment in the parsing method.
    pub(crate) y_shrink_factor: u32,
    /// Shrink factor in the x direction due to requesting a lower resolution level.
    pub(crate) x_resolution_shrink_factor: u32,
    /// Shrink factor in the y direction due to requesting a lower resolution level.
    pub(crate) y_resolution_shrink_factor: u32,
}

/// Component information (A.5.1 and Table A.11).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ComponentSizeInfo {
    pub(crate) precision: u8,
    pub(crate) signed: bool,
    pub(crate) horizontal_resolution: u8,
    pub(crate) vertical_resolution: u8,
}

impl SizeData {
    pub(crate) fn tile_x_coord(&self, idx: u32) -> u32 {
        // See B-6.
        idx % self.num_x_tiles()
    }

    pub(crate) fn tile_y_coord(&self, idx: u32) -> u32 {
        // See B-6.
        idx / self.num_x_tiles()
    }

    /// The number of tiles in the x direction.
    pub(crate) fn num_x_tiles(&self) -> u32 {
        // See formula B-5.
        (self.reference_grid_width - self.tile_x_offset).div_ceil(self.tile_width)
    }

    /// The number of tiles in the y direction.
    pub(crate) fn num_y_tiles(&self) -> u32 {
        // See formula B-5.
        (self.reference_grid_height - self.tile_y_offset).div_ceil(self.tile_height)
    }

    /// The total number of tiles.
    ///
    /// Saturating: `size_marker` rejects grids beyond `MAX_TILES`, so any
    /// validated header stays far below the saturation point; saturation only
    /// keeps unvalidated values panic-free.
    pub(crate) fn num_tiles(&self) -> u32 {
        self.num_x_tiles().saturating_mul(self.num_y_tiles())
    }

    /// Return the overall width of the image.
    pub(crate) fn image_width(&self) -> u32 {
        self.checked_image_width()
            .expect("validated JPEG 2000 horizontal shrink factors")
    }

    /// Return the overall height of the image.
    pub(crate) fn image_height(&self) -> u32 {
        self.checked_image_height()
            .expect("validated JPEG 2000 vertical shrink factors")
    }

    pub(crate) fn checked_image_width(&self) -> Result<u32> {
        let shrink_factor = self.checked_x_shrink_factor()?;
        Ok((self.reference_grid_width - self.image_area_x_offset).div_ceil(shrink_factor))
    }

    pub(crate) fn checked_image_height(&self) -> Result<u32> {
        let shrink_factor = self.checked_y_shrink_factor()?;
        Ok((self.reference_grid_height - self.image_area_y_offset).div_ceil(shrink_factor))
    }

    pub(crate) fn checked_x_shrink_factor(&self) -> Result<u32> {
        self.x_shrink_factor
            .checked_mul(self.x_resolution_shrink_factor)
            .filter(|factor| *factor != 0)
            .ok_or(ValidationError::InvalidDimensions.into())
    }

    pub(crate) fn checked_y_shrink_factor(&self) -> Result<u32> {
        self.y_shrink_factor
            .checked_mul(self.y_resolution_shrink_factor)
            .filter(|factor| *factor != 0)
            .ok_or(ValidationError::InvalidDimensions.into())
    }

    /// Return the reference-grid image width before component or resolution
    /// downscaling is applied.
    pub(crate) fn reference_image_width(&self) -> u32 {
        self.reference_grid_width - self.image_area_x_offset
    }

    /// Return the reference-grid image height before component or resolution
    /// downscaling is applied.
    pub(crate) fn reference_image_height(&self) -> u32 {
        self.reference_grid_height - self.image_area_y_offset
    }
}
