// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hybrid device decode planning for lossless (SOF3) JPEG.
//!
//! The CPU runs the serial Huffman stage and stores every visible
//! difference; device kernels then reconstruct samples with each scan's
//! predictor, apply the point transform, replicate subsampled components,
//! convert color, and pack the output exactly as CPU decode does.

use crate::decoder::{Decoder, DeviceLosslessScan};
use crate::error::JpegError;
use crate::info::ColorSpace;
use crate::Warning;
use alloc::vec::Vec;
use j2k_core::PixelFormat;

/// Largest component count of a lossless JPEG CPU decode accepts.
pub const JPEG_DEVICE_LOSSLESS_MAX_COMPONENTS: usize = 3;

/// One component's reconstruction grid and parameters, in frame (SOF) order.
/// A split (multi-scan) frame codes its components in different scans, so
/// predictor, point transform, and limits are per component.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JpegDeviceLosslessComponent {
    /// Visible samples per row.
    pub width: u32,
    /// Visible sample rows.
    pub height: u32,
    /// Component rows per restart interval; zero without restart markers.
    /// The first row of the scan and of every restart interval uses the 1-D
    /// predictor seeded with [`Self::initial_predictor`].
    pub restart_rows: u32,
    /// Predictor selection value of the component's scan, 1..=7.
    pub predictor: u8,
    /// Point transform `Pt` of the component's scan; output samples are
    /// shifted left by it.
    pub point_transform: u8,
    /// `2^(P - Pt - 1)`, the 1-D predictor seed.
    pub initial_predictor: u16,
    /// Largest valid reconstructed sample; larger values are a decode error.
    pub sample_limit: u16,
    /// Output pixels each sample covers, `Hmax / h` by `Vmax / v`: output
    /// replicates subsampled components, as libjpeg-turbo's lossless
    /// upsampler does.
    pub h_expand: u8,
    pub v_expand: u8,
}

/// Validated description of one lossless device decode.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegDeviceLosslessLayout {
    /// Sample precision `P`, 2..=16. Output is 8-bit for `P <= 8` and
    /// 16-bit otherwise, holding native-range samples.
    pub precision: u8,
    /// Header-derived color space: grayscale, RGB, or YCbCr.
    pub color_space: ColorSpace,
    components: [JpegDeviceLosslessComponent; JPEG_DEVICE_LOSSLESS_MAX_COMPONENTS],
    component_count: usize,
}

impl JpegDeviceLosslessLayout {
    /// Component grids in frame order.
    #[must_use]
    pub fn components(&self) -> &[JpegDeviceLosslessComponent] {
        &self.components[..self.component_count]
    }

    /// Whether every component shares one predictor, as in single-scan
    /// frames and split frames whose scans agree.
    #[must_use]
    pub fn uniform_predictor(&self) -> Option<u8> {
        let first = self.components().first()?.predictor;
        self.components()
            .iter()
            .all(|component| component.predictor == first)
            .then_some(first)
    }

    /// Index of each component's first difference in the shared storage.
    #[must_use]
    pub fn sample_bases(&self) -> [usize; JPEG_DEVICE_LOSSLESS_MAX_COMPONENTS] {
        let mut bases = [0; JPEG_DEVICE_LOSSLESS_MAX_COMPONENTS];
        let mut total = 0;
        for (base, component) in bases.iter_mut().zip(self.components()) {
            *base = total;
            total += component.width as usize * component.height as usize;
        }
        bases
    }

    /// Total visible samples across every component.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.components()
            .iter()
            .map(|component| component.width as usize * component.height as usize)
            .sum()
    }
}

/// Validate a lossless device decode to `fmt` exactly as CPU decode does and
/// describe it.
///
/// # Errors
/// Returns the error CPU decode returns for an unsupported precision,
/// predictor, color space, sampling, or output format.
pub fn device_lossless_layout(
    decoder: &Decoder<'_>,
    fmt: PixelFormat,
) -> Result<JpegDeviceLosslessLayout, JpegError> {
    let info = decoder.info();
    let component_count = info.sampling.len();
    if component_count > JPEG_DEVICE_LOSSLESS_MAX_COMPONENTS {
        return Err(JpegError::InternalInvariant {
            reason: "device lossless decode supports at most three components",
        });
    }
    let narrow = |value: usize| {
        u32::try_from(value).map_err(|_| JpegError::InternalInvariant {
            reason: "lossless component geometry exceeds u32",
        })
    };
    let expand = |max: u8, factor: u8| {
        max.checked_div(factor).ok_or(JpegError::InternalInvariant {
            reason: "lossless sampling factor is zero",
        })
    };
    let mut components =
        [JpegDeviceLosslessComponent::default(); JPEG_DEVICE_LOSSLESS_MAX_COMPONENTS];
    let mut seen = [false; JPEG_DEVICE_LOSSLESS_MAX_COMPONENTS];
    for index in 0..decoder.device_lossless_scan_count(fmt)? {
        let DeviceLosslessScan {
            scan,
            frame_components,
            component_count: scan_components,
            ..
        } = decoder.device_lossless_scan(index)?;
        let (initial_predictor, sample_limit) = scan.device_reconstruction_limits();
        for (slot, &frame) in frame_components[..scan_components].iter().enumerate() {
            let (width, height, restart_rows) = scan.device_component_geometry(slot);
            let (h, v) = info
                .sampling
                .component(frame)
                .ok_or(JpegError::InternalInvariant {
                    reason: "lossless scan component exceeds the frame",
                })?;
            let entry = seen.get_mut(frame).ok_or(JpegError::InternalInvariant {
                reason: "lossless scan component exceeds the device limit",
            })?;
            if core::mem::replace(entry, true) {
                return Err(JpegError::InternalInvariant {
                    reason: "lossless scans code a component twice",
                });
            }
            components[frame] = JpegDeviceLosslessComponent {
                width: narrow(width)?,
                height: narrow(height)?,
                restart_rows: narrow(restart_rows)?,
                predictor: scan.predictor(),
                point_transform: scan.point_transform(),
                initial_predictor,
                sample_limit,
                h_expand: expand(info.sampling.max_h, h)?,
                v_expand: expand(info.sampling.max_v, v)?,
            };
        }
    }
    if seen[..component_count].contains(&false) {
        return Err(JpegError::InternalInvariant {
            reason: "lossless scans do not cover every frame component",
        });
    }
    Ok(JpegDeviceLosslessLayout {
        precision: info.bit_depth,
        color_space: info.color_space,
        components,
        component_count,
    })
}

/// Entropy-decode every lossless difference into `diffs`, laid out per
/// [`device_lossless_layout`]: each component's visible `width * height`
/// differences row-major, back to back in frame order.
///
/// # Errors
/// Returns the same entropy, restart, and marker errors as CPU decode. CPU
/// decode reconstructs as it goes, so on an entropy error it may instead
/// report an out-of-range sample decoded earlier; callers that need the CPU
/// error for a failed decode must take it from CPU decode.
pub fn decode_device_lossless_differences_into(
    decoder: &Decoder<'_>,
    fmt: PixelFormat,
    diffs: &mut [u16],
) -> Result<Vec<Warning>, JpegError> {
    let layout = device_lossless_layout(decoder, fmt)?;
    if diffs.len() != layout.total_samples() {
        return Err(JpegError::InternalInvariant {
            reason: "device lossless difference storage disagrees with the layout",
        });
    }
    // Split the storage into frame-order planes, then hand each scan its
    // components' planes in scan order.
    let mut frame_planes: [Option<&mut [u16]>; JPEG_DEVICE_LOSSLESS_MAX_COMPONENTS] =
        [None, None, None];
    let mut rest = diffs;
    for (frame, component) in layout.components().iter().enumerate() {
        let len = component.width as usize * component.height as usize;
        let (plane, tail) = rest.split_at_mut(len);
        frame_planes[frame] = Some(plane);
        rest = tail;
    }
    let mut warnings = Vec::new();
    for index in 0..decoder.device_lossless_scan_count(fmt)? {
        let DeviceLosslessScan {
            scan,
            frame_components,
            component_count,
            validate_eoi,
        } = decoder.device_lossless_scan(index)?;
        let mut planes: [&mut [u16]; JPEG_DEVICE_LOSSLESS_MAX_COMPONENTS] =
            [&mut [], &mut [], &mut []];
        for (slot, &frame) in planes.iter_mut().zip(&frame_components[..component_count]) {
            *slot = frame_planes.get_mut(frame).and_then(Option::take).ok_or(
                JpegError::InternalInvariant {
                    reason: "lossless scan components do not map one-to-one onto the frame",
                },
            )?;
        }
        warnings = scan.decode_differences_into(&mut planes[..component_count], validate_eoi)?;
    }
    Ok(warnings)
}

#[cfg(test)]
mod tests;
