// SPDX-License-Identifier: MIT OR Apache-2.0

//! Kernel parameters for one hybrid DCT decode: which coefficient blocks each
//! component needs for the output rectangle, and where its decoded window sits.

use j2k_core::{PixelFormat, Rect};
use j2k_jpeg::{
    adapter::{
        JpegDeviceComponentGeometry, JpegDeviceDctLayout, JpegDeviceUpsample,
        JPEG_DEVICE_MAX_COMPONENTS,
    },
    ColorSpace as JpegColorSpace,
};

use crate::abi::{
    JpegHybridComponentParams, JpegHybridIdctParams, JpegHybridPackParams, HYBRID_COLOR_CMYK,
    HYBRID_COLOR_GRAY, HYBRID_COLOR_RGB, HYBRID_COLOR_YCBCR, HYBRID_COLOR_YCCK,
    HYBRID_UPSAMPLE_H1V2, HYBRID_UPSAMPLE_H2V1, HYBRID_UPSAMPLE_H2V2, HYBRID_UPSAMPLE_NONE,
    HYBRID_UPSAMPLE_REPLICATE, OUT_GRAY, OUT_RGB, OUT_RGBA,
};
use crate::Error;

use super::super::checked_u32;

/// Everything the IDCT and pack passes need for one image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HybridDctPlan {
    pub(super) idct: [JpegHybridIdctParams; JPEG_DEVICE_MAX_COMPONENTS],
    pub(super) component_count: usize,
    /// Samples held by each component's decoded window.
    pub(super) plane_samples: [usize; JPEG_DEVICE_MAX_COMPONENTS],
    /// Pack parameters shared by every output rectangle.
    pack_template: JpegHybridPackParams,
    window: Rect,
}

impl HybridDctPlan {
    /// Plan IDCT windows covering `window`, the union of every output
    /// rectangle (in scaled image coordinates) the pack passes will read.
    pub(super) fn new(
        color_space: JpegColorSpace,
        layout: &JpegDeviceDctLayout,
        geometry: &[JpegDeviceComponentGeometry],
        window: Rect,
        fmt: PixelFormat,
    ) -> Result<Self, Error> {
        let grids = layout.components();
        if grids.len() != geometry.len() || grids.is_empty() {
            return Err(invariant(
                "component geometry disagrees with the block layout",
            ));
        }
        if window.w == 0 || window.h == 0 {
            return Err(invariant("output rectangle is empty"));
        }
        let bases = layout.block_bases()?;
        let total_blocks = layout.total_blocks()?;
        // The IDCT kernel addresses coefficients as `block * 64` in u32.
        checked_u32(
            total_blocks
                .checked_mul(64)
                .ok_or_else(|| invariant("coefficient count overflowed"))?,
            "hybrid coefficient count",
        )?;

        let mut idct = [JpegHybridIdctParams::default(); JPEG_DEVICE_MAX_COMPONENTS];
        let mut components = [JpegHybridComponentParams::default(); JPEG_DEVICE_MAX_COMPONENTS];
        let mut plane_samples = [0; JPEG_DEVICE_MAX_COMPONENTS];
        let blocks = component_block_windows(geometry, window)?;
        for (index, (grid, geometry)) in grids.iter().zip(geometry).enumerate() {
            let size = geometry.idct_size;
            let [first_col, first_row, end_col, end_row] = blocks[index];
            let (cols, rows) = (end_col - first_col, end_row - first_row);
            // Coefficients are stored for the layout's window of the image's
            // block grid; the IDCT addresses them relative to its origin.
            let (origin_col, origin_row) = layout.origin(index);
            if first_col < origin_col
                || first_row < origin_row
                || end_col - origin_col > grid.block_cols
                || end_row - origin_row > grid.block_rows
            {
                return Err(invariant("component window exceeds its block grid"));
            }
            let plane_stride = cols * size;
            let plane_len = (plane_stride as usize)
                .checked_mul((rows * size) as usize)
                .ok_or_else(|| invariant("component plane size overflowed"))?;
            checked_u32(plane_len, "hybrid component plane")?;
            plane_samples[index] = plane_len;
            idct[index] = JpegHybridIdctParams {
                block_base: checked_u32(bases[index], "hybrid block base")?,
                block_cols: grid.block_cols,
                block_x0: first_col - origin_col,
                block_y0: first_row - origin_row,
                block_x_count: cols,
                block_y_count: rows,
                idct_size: size,
                plane_stride,
            };
            components[index] = JpegHybridComponentParams {
                plane_stride,
                origin_x: first_col * size,
                origin_y: first_row * size,
                width: geometry.width,
                height: geometry.height,
                h_ratio: geometry.h_ratio,
                v_ratio: geometry.v_ratio,
                upsample: upsample_code(geometry.upsample),
            };
        }

        Ok(Self {
            idct,
            component_count: grids.len(),
            plane_samples,
            pack_template: JpegHybridPackParams {
                components,
                component_count: checked_u32(grids.len(), "hybrid component count")?,
                color_space: color_code(color_space),
                out_format: out_format_code(fmt)?,
                ..JpegHybridPackParams::default()
            },
            window,
        })
    }

    /// Pack parameters writing `source` (scaled image coordinates, inside
    /// the planned window) to `dest` in an output `dest_dims` pixels large.
    pub(super) fn pack_params(
        &self,
        source: Rect,
        dest: (u32, u32),
        dest_dims: (u32, u32),
        fmt: PixelFormat,
    ) -> Result<JpegHybridPackParams, Error> {
        let inside_window = source.x >= self.window.x
            && source.y >= self.window.y
            && source.x + source.w <= self.window.x + self.window.w
            && source.y + source.h <= self.window.y + self.window.h;
        let inside_dest = dest
            .0
            .checked_add(source.w)
            .is_some_and(|end| end <= dest_dims.0)
            && dest
                .1
                .checked_add(source.h)
                .is_some_and(|end| end <= dest_dims.1);
        if !inside_window || !inside_dest {
            return Err(invariant(
                "pack rectangle lies outside its window or destination",
            ));
        }
        // The pack kernel addresses output samples in u32.
        let output_samples = (dest_dims.0 as usize)
            .checked_mul(dest_dims.1 as usize)
            .and_then(|pixels| pixels.checked_mul(fmt.channels()))
            .ok_or_else(|| invariant("output size overflowed"))?;
        checked_u32(output_samples, "hybrid output size")?;
        Ok(JpegHybridPackParams {
            src_x: source.x,
            src_y: source.y,
            width: source.w,
            height: source.h,
            dest_x: dest.0,
            dest_y: dest.1,
            dest_stride: dest_dims.0,
            ..self.pack_template
        })
    }
}

/// Each component's needed blocks `[first_column, first_row, end_column,
/// end_row]` (ends exclusive) for output `window`.
pub(super) fn component_block_windows(
    geometry: &[JpegDeviceComponentGeometry],
    window: Rect,
) -> Result<[[u32; 4]; JPEG_DEVICE_MAX_COMPONENTS], Error> {
    let mut blocks = [[0; 4]; JPEG_DEVICE_MAX_COMPONENTS];
    for (slot, geometry) in blocks.iter_mut().zip(geometry) {
        let samples = ComponentWindow::new(*geometry, window)?;
        let size = geometry.idct_size;
        *slot = [
            samples.x0 / size,
            samples.y0 / size,
            samples.x1 / size + 1,
            samples.y1 / size + 1,
        ];
    }
    Ok(blocks)
}

/// Inclusive range of component samples the output rectangle reads, including
/// the neighbours the triangle filters blend in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ComponentWindow {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

impl ComponentWindow {
    fn new(geometry: JpegDeviceComponentGeometry, rect: Rect) -> Result<Self, Error> {
        if geometry.width == 0
            || geometry.height == 0
            || geometry.h_ratio == 0
            || geometry.v_ratio == 0
            || !matches!(geometry.idct_size, 1 | 2 | 4 | 8)
        {
            return Err(invariant("component geometry is degenerate"));
        }
        let (margin_x, margin_y) = match geometry.upsample {
            JpegDeviceUpsample::FancyH2V1 => (1, 0),
            JpegDeviceUpsample::FancyH1V2 => (0, 1),
            JpegDeviceUpsample::FancyH2V2 => (1, 1),
            JpegDeviceUpsample::None | JpegDeviceUpsample::Replicate => (0, 0),
        };
        let span = |start: u32, len: u32, ratio: u32, margin: u32, extent: u32| {
            let last = extent - 1;
            let first = (start / ratio).saturating_sub(margin).min(last);
            let end = ((start + len - 1) / ratio + margin).min(last);
            (first, end)
        };
        let (x0, x1) = span(rect.x, rect.w, geometry.h_ratio, margin_x, geometry.width);
        let (y0, y1) = span(rect.y, rect.h, geometry.v_ratio, margin_y, geometry.height);
        Ok(Self { x0, x1, y0, y1 })
    }
}

fn upsample_code(upsample: JpegDeviceUpsample) -> u32 {
    match upsample {
        JpegDeviceUpsample::None => HYBRID_UPSAMPLE_NONE,
        JpegDeviceUpsample::FancyH2V1 => HYBRID_UPSAMPLE_H2V1,
        JpegDeviceUpsample::FancyH1V2 => HYBRID_UPSAMPLE_H1V2,
        JpegDeviceUpsample::FancyH2V2 => HYBRID_UPSAMPLE_H2V2,
        JpegDeviceUpsample::Replicate => HYBRID_UPSAMPLE_REPLICATE,
    }
}

fn color_code(color_space: JpegColorSpace) -> u32 {
    match color_space {
        JpegColorSpace::Grayscale => HYBRID_COLOR_GRAY,
        JpegColorSpace::YCbCr => HYBRID_COLOR_YCBCR,
        JpegColorSpace::Rgb => HYBRID_COLOR_RGB,
        JpegColorSpace::Cmyk => HYBRID_COLOR_CMYK,
        JpegColorSpace::Ycck => HYBRID_COLOR_YCCK,
    }
}

pub(super) fn out_format_code(fmt: PixelFormat) -> Result<u32, Error> {
    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => Ok(OUT_GRAY),
        PixelFormat::Rgb8 | PixelFormat::Rgb16 => Ok(OUT_RGB),
        PixelFormat::Rgba8 | PixelFormat::Rgba16 => Ok(OUT_RGBA),
        _ => Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_format(
                "JPEG Metal hybrid decode supports Gray, RGB, and RGBA 8- or 16-bit output",
            ),
        )),
    }
}

pub(super) fn invariant(reason: &str) -> Error {
    Error::MetalKernel {
        message: format!("JPEG Metal hybrid decode plan invariant failed: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(
        idct_size: u32,
        ratios: (u32, u32),
        extent: (u32, u32),
        upsample: JpegDeviceUpsample,
    ) -> JpegDeviceComponentGeometry {
        JpegDeviceComponentGeometry {
            idct_size,
            h_ratio: ratios.0,
            v_ratio: ratios.1,
            width: extent.0,
            height: extent.1,
            upsample,
        }
    }

    #[test]
    fn fancy_windows_keep_one_neighbour_sample_and_clamp_at_edges() {
        let chroma = geometry(8, (2, 2), (34, 23), JpegDeviceUpsample::FancyH2V2);
        let window = ComponentWindow::new(
            chroma,
            Rect {
                x: 10,
                y: 0,
                w: 20,
                h: 45,
            },
        )
        .unwrap();
        assert_eq!(
            window,
            ComponentWindow {
                x0: 4,
                x1: 15,
                y0: 0,
                y1: 22,
            }
        );
    }

    #[test]
    fn replicated_windows_cover_only_sampled_positions() {
        let chroma = geometry(8, (4, 1), (17, 45), JpegDeviceUpsample::Replicate);
        let window = ComponentWindow::new(
            chroma,
            Rect {
                x: 5,
                y: 3,
                w: 7,
                h: 2,
            },
        )
        .unwrap();
        assert_eq!(
            window,
            ComponentWindow {
                x0: 1,
                x1: 2,
                y0: 3,
                y1: 4,
            }
        );
    }

    #[test]
    fn unsupported_formats_are_capability_rejections() {
        assert!(matches!(
            out_format_code(PixelFormat::RgbI16),
            Err(Error::UnsupportedMetalRequest { .. })
        ));
    }
}
