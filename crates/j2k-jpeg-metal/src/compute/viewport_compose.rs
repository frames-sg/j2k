// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;
use j2k_jpeg::Decoder as CpuDecoder;

use crate::viewport::ViewportTile;
use crate::{Error, Surface};

use super::viewport_cache::{cached_plane_stage, ViewportPlaneWriter};
use super::with_runtime;
#[cfg(test)]
use super::with_runtime_for_session;

pub(crate) fn compose_rgb_viewport_from_regions(
    decoder: &CpuDecoder<'_>,
    pool: &mut j2k_jpeg::ScratchPool,
    scale: j2k_core::Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut stage = cached_plane_stage(runtime, decoder.info().color_space, viewport_dims)?;
        for tile in tiles {
            let dims = tile.source_roi.scaled_covering(scale);
            if (dims.w, dims.h) != (tile.dest.w, tile.dest.h) {
                return Err(Error::MetalKernel {
                    message: format!(
                        "viewport tile dims {:?} do not match destination rect {:?}",
                        (dims.w, dims.h),
                        tile.dest
                    ),
                });
            }
            let mut writer = ViewportPlaneWriter {
                stage: &mut stage,
                dest: tile.dest,
            };
            decoder.decode_region_component_rows_with_scratch(
                pool,
                &mut writer,
                j2k_jpeg::Rect {
                    x: tile.source_roi.x,
                    y: tile.source_roi.y,
                    w: tile.source_roi.w,
                    h: tile.source_roi.h,
                },
                scale,
            )?;
        }
        stage.finish_with_runtime(runtime, PixelFormat::Rgb8)
    })
}

#[cfg(test)]
pub(crate) fn compose_rgb_viewport_from_regions_into_output_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut j2k_jpeg::ScratchPool,
    scale: j2k_core::Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
    output: &crate::MetalBatchOutputBuffer,
    session: &crate::MetalBackendSession,
) -> Result<Surface, Error> {
    with_runtime_for_session(session, |runtime| {
        let mut stage = cached_plane_stage(runtime, decoder.info().color_space, viewport_dims)?;
        for tile in tiles {
            let dims = tile.source_roi.scaled_covering(scale);
            if (dims.w, dims.h) != (tile.dest.w, tile.dest.h) {
                return Err(Error::MetalKernel {
                    message: format!(
                        "viewport tile dims {:?} do not match destination rect {:?}",
                        (dims.w, dims.h),
                        tile.dest
                    ),
                });
            }
            let mut writer = ViewportPlaneWriter {
                stage: &mut stage,
                dest: tile.dest,
            };
            decoder.decode_region_component_rows_with_scratch(
                pool,
                &mut writer,
                j2k_jpeg::Rect {
                    x: tile.source_roi.x,
                    y: tile.source_roi.y,
                    w: tile.source_roi.w,
                    h: tile.source_roi.h,
                },
                scale,
            )?;
        }
        stage.finish_rgb8_into_output_with_runtime(runtime, output)
    })
}

#[cfg(test)]
pub(crate) fn compose_rgb_viewport_from_regions_into_textures_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut j2k_jpeg::ScratchPool,
    scale: j2k_core::Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
    output: &crate::MetalBatchTextureOutput,
    session: &crate::MetalBackendSession,
) -> Result<crate::MetalTextureTile, Error> {
    with_runtime_for_session(session, |runtime| {
        let mut stage = cached_plane_stage(runtime, decoder.info().color_space, viewport_dims)?;
        for tile in tiles {
            let dims = tile.source_roi.scaled_covering(scale);
            if (dims.w, dims.h) != (tile.dest.w, tile.dest.h) {
                return Err(Error::MetalKernel {
                    message: format!(
                        "viewport tile dims {:?} do not match destination rect {:?}",
                        (dims.w, dims.h),
                        tile.dest
                    ),
                });
            }
            let mut writer = ViewportPlaneWriter {
                stage: &mut stage,
                dest: tile.dest,
            };
            decoder.decode_region_component_rows_with_scratch(
                pool,
                &mut writer,
                j2k_jpeg::Rect {
                    x: tile.source_roi.x,
                    y: tile.source_roi.y,
                    w: tile.source_roi.w,
                    h: tile.source_roi.h,
                },
                scale,
            )?;
        }
        stage.finish_rgba8_into_texture_output_with_runtime(runtime, output)
    })
}
