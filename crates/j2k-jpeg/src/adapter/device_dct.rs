// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hybrid device decode planning for DCT-coded JPEG.
//!
//! Device adapters that run IDCT, upsampling, color conversion, and output
//! packing on a GPU use this module for the serial CPU stage: entropy decode
//! into caller-owned dequantized coefficient storage. The geometry helpers
//! expose libjpeg-turbo's per-component reduced-IDCT and upsampler choices so
//! device kernels reproduce the CPU decoder bit for bit.

use crate::color::scaled_sampling::{ScaledSampling, Upsample};
use crate::decoder::Decoder;
use crate::entropy::block::clamp_i16;
use crate::entropy::progressive::decode_progressive_dct_blocks;
use crate::entropy::sequential::{
    decode_scan_dct_blocks_into, decode_scan_dct_window_into, sequential_mcu_grid, McuWindow,
};
use crate::entropy::ZIGZAG;
use crate::error::JpegError;
use crate::info::{DownscaleFactor, Info, SofKind};
use crate::Warning;
use alloc::vec::Vec;
use j2k_core::Downscale;

/// Maximum number of JPEG frame components a device DCT plan describes.
pub const JPEG_DEVICE_MAX_COMPONENTS: usize = 4;

/// One component's MCU-padded grid of 8x8 coefficient blocks.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JpegDeviceDctComponentLayout {
    /// Blocks per row, including MCU padding.
    pub block_cols: u32,
    /// Block rows, including MCU padding.
    pub block_rows: u32,
}

impl JpegDeviceDctComponentLayout {
    /// Number of blocks in the grid.
    ///
    /// # Errors
    /// Returns [`JpegError::InternalInvariant`] when the count overflows.
    pub fn block_count(self) -> Result<usize, JpegError> {
        (self.block_cols as usize)
            .checked_mul(self.block_rows as usize)
            .ok_or(JpegError::InternalInvariant {
                reason: "device DCT block count overflowed",
            })
    }
}

/// Coefficient storage layout for one DCT-coded JPEG.
///
/// Blocks of every component are stored back to back in frame (SOF
/// declaration) order. Each block holds 64 dequantized coefficients in natural
/// row-major order: exactly the input of the CPU decoder's IDCT.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegDeviceDctLayout {
    /// Sample precision in bits: 8 or 12.
    pub precision: u8,
    components: [JpegDeviceDctComponentLayout; JPEG_DEVICE_MAX_COMPONENTS],
    origins: [(u32, u32); JPEG_DEVICE_MAX_COMPONENTS],
    component_count: usize,
}

impl JpegDeviceDctLayout {
    /// Per-component block grids in frame order.
    #[must_use]
    pub fn components(&self) -> &[JpegDeviceDctComponentLayout] {
        &self.components[..self.component_count]
    }

    /// Block `(column, row)` of the image at which `component`'s stored grid
    /// starts: zero for a whole-image layout, the window origin for
    /// [`device_dct_window_layout`].
    #[must_use]
    pub fn origin(&self, component: usize) -> (u32, u32) {
        self.origins[component]
    }

    /// Index of each component's first block in the shared block storage.
    ///
    /// # Errors
    /// Returns [`JpegError::InternalInvariant`] when the counts overflow.
    pub fn block_bases(&self) -> Result<[usize; JPEG_DEVICE_MAX_COMPONENTS], JpegError> {
        let mut bases = [0; JPEG_DEVICE_MAX_COMPONENTS];
        let mut total = 0_usize;
        for (base, component) in bases.iter_mut().zip(self.components()) {
            *base = total;
            total = total.checked_add(component.block_count()?).ok_or(
                JpegError::InternalInvariant {
                    reason: "device DCT block count overflowed",
                },
            )?;
        }
        Ok(bases)
    }

    /// Total number of blocks across every component.
    ///
    /// # Errors
    /// Returns [`JpegError::InternalInvariant`] when the counts overflow.
    pub fn total_blocks(&self) -> Result<usize, JpegError> {
        self.components()
            .iter()
            .try_fold(0_usize, |total, component| {
                total
                    .checked_add(component.block_count()?)
                    .ok_or(JpegError::InternalInvariant {
                        reason: "device DCT block count overflowed",
                    })
            })
    }
}

/// Return the coefficient storage layout for a DCT-coded decoder.
///
/// # Errors
/// Returns [`JpegError::NotImplemented`] for lossless JPEG and decoders without
/// a prepared DCT plan.
pub fn device_dct_layout(decoder: &Decoder<'_>) -> Result<JpegDeviceDctLayout, JpegError> {
    let info = decoder.info();
    let mut components = [JpegDeviceDctComponentLayout::default(); JPEG_DEVICE_MAX_COMPONENTS];
    let component_count = match info.sof_kind {
        SofKind::Baseline8 | SofKind::Extended8 | SofKind::Extended12
            if decoder.progressive_plan.is_none() =>
        {
            let plan = &decoder.plan;
            let (mcus_per_row, mcu_rows) = sequential_mcu_grid(plan);
            for component in &plan.components {
                let slot = components.get_mut(component.output_index).ok_or(
                    JpegError::InternalInvariant {
                        reason: "device DCT component index exceeds the frame limit",
                    },
                )?;
                *slot = JpegDeviceDctComponentLayout {
                    block_cols: mcus_per_row * u32::from(component.h),
                    block_rows: mcu_rows * u32::from(component.v),
                };
            }
            plan.components.len()
        }
        // Progressive and split (multi-scan) sequential frames.
        SofKind::Baseline8
        | SofKind::Extended8
        | SofKind::Extended12
        | SofKind::Progressive8
        | SofKind::Progressive12 => {
            let plan = decoder
                .progressive_plan
                .as_ref()
                .ok_or(JpegError::NotImplemented { sof: info.sof_kind })?;
            for component in &plan.components {
                let slot = components.get_mut(component.output_index).ok_or(
                    JpegError::InternalInvariant {
                        reason: "device DCT component index exceeds the frame limit",
                    },
                )?;
                *slot = JpegDeviceDctComponentLayout {
                    block_cols: component.block_cols,
                    block_rows: component.block_rows,
                };
            }
            plan.components.len()
        }
        SofKind::Lossless => return Err(JpegError::NotImplemented { sof: info.sof_kind }),
    };
    if component_count != info.sampling.len() || component_count > JPEG_DEVICE_MAX_COMPONENTS {
        return Err(JpegError::InternalInvariant {
            reason: "device DCT plan must cover every frame component",
        });
    }
    Ok(JpegDeviceDctLayout {
        precision: info.bit_depth,
        components,
        origins: [(0, 0); JPEG_DEVICE_MAX_COMPONENTS],
        component_count,
    })
}

/// Rectangle of MCUs a windowed sequential device decode entropy-decodes.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegDeviceMcuWindow {
    /// First MCU column.
    pub col: u32,
    /// MCU columns.
    pub cols: u32,
    /// First MCU row.
    pub row: u32,
    /// MCU rows.
    pub rows: u32,
}

impl From<JpegDeviceMcuWindow> for McuWindow {
    fn from(window: JpegDeviceMcuWindow) -> Self {
        Self {
            col: window.col,
            cols: window.cols,
            row: window.row,
            rows: window.rows,
        }
    }
}

/// MCU window covering every component's needed blocks for an 8-bit
/// sequential decoder, or `None` when the whole scan must be decoded.
///
/// `component_blocks[c]` is `[first_column, first_row, end_column, end_row]`
/// in component `c`'s block units (ends exclusive). Like CPU region decode,
/// the window keeps one MCU row above and below the needed rows. 12-bit,
/// progressive, and split (multi-scan) sequential scans always decode in
/// full, as CPU decode does.
#[must_use]
pub fn device_dct_mcu_window(
    decoder: &Decoder<'_>,
    component_blocks: &[[u32; 4]],
) -> Option<JpegDeviceMcuWindow> {
    if !matches!(
        decoder.info().sof_kind,
        SofKind::Baseline8 | SofKind::Extended8
    ) || decoder.progressive_plan.is_some()
    {
        return None;
    }
    let plan = &decoder.plan;
    let (mcus_per_row, mcu_rows) = sequential_mcu_grid(plan);
    let (mut col0, mut row0, mut col1, mut row1) = (u32::MAX, u32::MAX, 0, 0);
    for component in &plan.components {
        let [x0, y0, x1, y1] = *component_blocks.get(component.output_index)?;
        let (h, v) = (u32::from(component.h), u32::from(component.v));
        col0 = col0.min(x0 / h);
        row0 = row0.min(y0 / v);
        col1 = col1.max(x1.div_ceil(h));
        row1 = row1.max(y1.div_ceil(v));
    }
    let row0 = row0.saturating_sub(1);
    let row1 = (row1 + 1).min(mcu_rows);
    let col1 = col1.min(mcus_per_row);
    if col0 >= col1 || row0 >= row1 || (col0, row0, col1, row1) == (0, 0, mcus_per_row, mcu_rows) {
        return None;
    }
    Some(JpegDeviceMcuWindow {
        col: col0,
        cols: col1 - col0,
        row: row0,
        rows: row1 - row0,
    })
}

/// Coefficient storage layout for the MCUs of `window`.
///
/// # Errors
/// Returns [`JpegError::NotImplemented`] for scans windowed decode does not
/// cover, or [`JpegError::InternalInvariant`] for a window outside the scan.
pub fn device_dct_window_layout(
    decoder: &Decoder<'_>,
    window: JpegDeviceMcuWindow,
) -> Result<JpegDeviceDctLayout, JpegError> {
    let info = decoder.info();
    if !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8)
        || decoder.progressive_plan.is_some()
    {
        return Err(JpegError::NotImplemented { sof: info.sof_kind });
    }
    let plan = &decoder.plan;
    let (mcus_per_row, mcu_rows) = sequential_mcu_grid(plan);
    if window.cols == 0
        || window.rows == 0
        || window.col + window.cols > mcus_per_row
        || window.row + window.rows > mcu_rows
    {
        return Err(JpegError::InternalInvariant {
            reason: "device DCT window lies outside the scan",
        });
    }
    let mut components = [JpegDeviceDctComponentLayout::default(); JPEG_DEVICE_MAX_COMPONENTS];
    let mut origins = [(0, 0); JPEG_DEVICE_MAX_COMPONENTS];
    for component in &plan.components {
        let index = component.output_index;
        if index >= JPEG_DEVICE_MAX_COMPONENTS {
            return Err(JpegError::InternalInvariant {
                reason: "device DCT component index exceeds the frame limit",
            });
        }
        let (h, v) = (u32::from(component.h), u32::from(component.v));
        components[index] = JpegDeviceDctComponentLayout {
            block_cols: window.cols * h,
            block_rows: window.rows * v,
        };
        origins[index] = (window.col * h, window.row * v);
    }
    Ok(JpegDeviceDctLayout {
        precision: info.bit_depth,
        components,
        origins,
        component_count: plan.components.len(),
    })
}

/// Entropy-decode the MCUs of `window` into `blocks`, laid out per
/// [`device_dct_window_layout`]. As CPU region decode does, the scan is
/// entered at the nearest preceding restart marker when there is one and is
/// not validated past the window.
///
/// # Errors
/// Returns the same entropy and restart errors as CPU decode, or
/// [`JpegError::InternalInvariant`] when `blocks` does not match the layout.
pub fn decode_device_dct_window_into(
    decoder: &Decoder<'_>,
    window: JpegDeviceMcuWindow,
    blocks: &mut [[i16; 64]],
) -> Result<(), JpegError> {
    let info = decoder.info();
    if !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8)
        || decoder.progressive_plan.is_some()
    {
        return Err(JpegError::NotImplemented { sof: info.sof_kind });
    }
    let plan = &decoder.plan;
    decode_scan_dct_window_into(
        plan,
        &decoder.bytes[plan.scan_offset..],
        window.into(),
        blocks,
    )
}

/// Entropy-decode every DCT block into `blocks`, laid out per
/// [`device_dct_layout`].
///
/// Sequential scans decode straight into the storage; progressive scans
/// accumulate their quantized coefficients on the host first and then write
/// dequantized blocks. Every block is overwritten.
///
/// # Errors
/// Returns the same entropy, restart, and marker errors as CPU decode, or
/// [`JpegError::InternalInvariant`] when `blocks` does not match the layout.
pub fn decode_device_dct_blocks_into(
    decoder: &Decoder<'_>,
    blocks: &mut [[i16; 64]],
) -> Result<Vec<Warning>, JpegError> {
    let info = decoder.info();
    match info.sof_kind {
        SofKind::Baseline8 | SofKind::Extended8 | SofKind::Extended12
            if decoder.progressive_plan.is_none() =>
        {
            let plan = &decoder.plan;
            decode_scan_dct_blocks_into(plan, &decoder.bytes[plan.scan_offset..], blocks)
        }
        // Progressive and split (multi-scan) sequential frames.
        SofKind::Baseline8
        | SofKind::Extended8
        | SofKind::Extended12
        | SofKind::Progressive8
        | SofKind::Progressive12 => {
            let layout = device_dct_layout(decoder)?;
            if blocks.len() != layout.total_blocks()? {
                return Err(JpegError::InternalInvariant {
                    reason: "progressive device DCT storage disagrees with the block grid",
                });
            }
            let plan = decoder
                .progressive_plan
                .as_ref()
                .ok_or(JpegError::NotImplemented { sof: info.sof_kind })?;
            let decoded = decode_progressive_dct_blocks(plan, decoder.bytes, 0)?;
            let bases = layout.block_bases()?;
            for component in &plan.components {
                let index = component.output_index;
                let quantized =
                    decoded
                        .quantized
                        .get(index)
                        .ok_or(JpegError::InternalInvariant {
                            reason: "progressive coefficient/component count mismatch",
                        })?;
                let count = layout.components()[index].block_count()?;
                if quantized.len() != count {
                    return Err(JpegError::InternalInvariant {
                        reason: "progressive coefficient grid disagrees with the device layout",
                    });
                }
                let natural_quant = natural_order_quant(&component.quant);
                let target = &mut blocks[bases[index]..bases[index] + count];
                for (out, coeffs) in target.iter_mut().zip(quantized) {
                    for ((slot, &coeff), &q) in out.iter_mut().zip(coeffs).zip(&natural_quant) {
                        *slot = clamp_i16(coeff.wrapping_mul(i32::from(q)));
                    }
                }
            }
            Ok(Vec::new())
        }
        SofKind::Lossless => Err(JpegError::NotImplemented { sof: info.sof_kind }),
    }
}

/// Quantization table permuted from zigzag to natural coefficient order.
fn natural_order_quant(quant: &[u16; 64]) -> [u16; 64] {
    let mut natural = [0u16; 64];
    for (&q, &natural_idx) in quant.iter().zip(&ZIGZAG) {
        natural[usize::from(natural_idx)] = q;
    }
    natural
}

/// How one component's decoded samples reach the output grid.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegDeviceUpsample {
    /// The component is already at output resolution.
    None,
    /// libjpeg-turbo `h2v1_fancy_upsample`.
    FancyH2V1,
    /// libjpeg-turbo `h1v2_fancy_upsample`.
    FancyH1V2,
    /// libjpeg-turbo `h2v2_fancy_upsample`.
    FancyH2V2,
    /// Sample replication by `h_ratio` x `v_ratio`.
    Replicate,
}

/// Decode geometry of one component at one output scale.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegDeviceComponentGeometry {
    /// Reduced IDCT output size per block side: 8, 4, 2, or 1.
    pub idct_size: u32,
    /// Output pixels per decoded sample, horizontally.
    pub h_ratio: u32,
    /// Output pixels per decoded sample, vertically.
    pub v_ratio: u32,
    /// Decoded samples across the whole image.
    pub width: u32,
    /// Decoded sample rows in the whole image.
    pub height: u32,
    /// Upsampler the CPU decoder applies to this component.
    pub upsample: JpegDeviceUpsample,
}

/// Return every component's decode geometry at `scale`, in frame order.
///
/// # Errors
/// Returns [`JpegError::InternalInvariant`] for more than four components.
pub fn device_component_geometry(
    info: &Info,
    scale: Downscale,
) -> Result<Vec<JpegDeviceComponentGeometry>, JpegError> {
    let count = info.sampling.len();
    if count > JPEG_DEVICE_MAX_COMPONENTS {
        return Err(JpegError::InternalInvariant {
            reason: "device geometry supports at most four components",
        });
    }
    let downscale = match scale {
        Downscale::None => DownscaleFactor::Full,
        Downscale::Half => DownscaleFactor::Half,
        Downscale::Quarter => DownscaleFactor::Quarter,
        Downscale::Eighth => DownscaleFactor::Eighth,
        _ => {
            return Err(JpegError::InternalInvariant {
                reason: "device geometry supports full, half, quarter, or eighth scale",
            })
        }
    };
    let scaled = ScaledSampling::new(info.sampling, downscale, info.dimensions);
    let mut geometry = Vec::new();
    geometry
        .try_reserve_exact(count)
        .map_err(|_| JpegError::MemoryCapExceeded {
            requested: count,
            cap: JPEG_DEVICE_MAX_COMPONENTS,
        })?;
    for index in 0..count {
        let component = scaled.component(index);
        geometry.push(JpegDeviceComponentGeometry {
            idct_size: component.idct_size,
            h_ratio: component.h_ratio,
            v_ratio: component.v_ratio,
            width: component.width,
            height: component.height,
            upsample: match component.upsample {
                Upsample::None => JpegDeviceUpsample::None,
                Upsample::FancyH2V1 => JpegDeviceUpsample::FancyH2V1,
                Upsample::FancyH1V2 => JpegDeviceUpsample::FancyH1V2,
                Upsample::FancyH2V2 => JpegDeviceUpsample::FancyH2V2,
                Upsample::Replicate => JpegDeviceUpsample::Replicate,
            },
        });
    }
    Ok(geometry)
}

#[cfg(test)]
mod tests;
