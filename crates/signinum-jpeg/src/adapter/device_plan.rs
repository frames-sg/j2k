// SPDX-License-Identifier: Apache-2.0

use crate::decoder::Decoder;
use crate::error::{JpegError, MarkerKind};
use crate::info::{ColorSpace, SofKind};
use crate::internal::checkpoint::{build_checkpoint_plan, DeviceCheckpoint};
use crate::Warning;
use alloc::borrow::Cow;
use alloc::vec::Vec;

/// One component entry in a device decode plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceComponentPlan {
    /// Horizontal sampling factor.
    pub h: u8,
    /// Vertical sampling factor.
    pub v: u8,
    /// Output component index.
    pub output_index: usize,
}

/// Baseline JPEG device decode plan shared with adapter crates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDecodePlan<'a> {
    /// Image dimensions in pixels.
    pub dimensions: (u32, u32),
    /// JPEG color space.
    pub color_space: ColorSpace,
    /// Optional restart interval in MCUs.
    pub restart_interval: Option<u16>,
    /// Warnings carried from CPU header parsing.
    pub warnings: Vec<Warning>,
    /// Entropy-coded scan bytes.
    pub scan_bytes: Cow<'a, [u8]>,
    /// Per-component sampling/output metadata.
    pub components: Vec<DeviceComponentPlan>,
    /// Entropy checkpoints for restart or synthetic cadence boundaries.
    pub checkpoints: Vec<DeviceCheckpoint>,
    /// True when the plan matches the Metal 4:2:0 fast path.
    pub matches_fast_420: bool,
    /// True when the plan matches the Metal 4:2:2 fast path.
    pub matches_fast_422: bool,
    /// True when the plan matches the Metal 4:4:4 fast path.
    pub matches_fast_444: bool,
}

/// Cheap summary of a device decode plan for batch planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBatchSummary {
    /// Optional restart interval in MCUs.
    pub restart_interval: Option<u16>,
    /// Number of checkpoints that would be produced.
    pub checkpoint_count: usize,
    /// True when the decoder matches the Metal 4:2:0 fast path.
    pub matches_fast_420: bool,
    /// True when the decoder matches the Metal 4:2:2 fast path.
    pub matches_fast_422: bool,
    /// True when the decoder matches the Metal 4:4:4 fast path.
    pub matches_fast_444: bool,
}

/// Build a full adapter decode plan for a decoder.
pub fn build_device_plan<'a>(
    decoder: &'a Decoder<'a>,
    cadence_mcus: u32,
) -> Result<DeviceDecodePlan<'a>, JpegError> {
    if !matches!(
        decoder.info().sof_kind,
        SofKind::Baseline8 | SofKind::Extended8
    ) {
        return Err(JpegError::NotImplemented {
            sof: decoder.info().sof_kind,
        });
    }
    let plan = &decoder.plan;
    let restart_interval = plan.restart_interval.filter(|&interval| interval > 0);
    let (scan_bytes, missing_eoi) =
        scan_payload_bytes(decoder.bytes, plan.scan_offset, restart_interval.is_some())?;
    let checkpoints = build_checkpoint_plan(plan, scan_bytes.as_ref(), cadence_mcus)?;
    let mut warnings = decoder.warnings.to_vec();
    if missing_eoi {
        warnings.push(Warning::MissingEoi);
    }

    Ok(DeviceDecodePlan {
        dimensions: plan.dimensions,
        color_space: plan.color_space,
        restart_interval,
        warnings,
        scan_bytes,
        components: plan
            .components
            .iter()
            .map(|component| DeviceComponentPlan {
                h: component.h,
                v: component.v,
                output_index: component.output_index,
            })
            .collect(),
        checkpoints,
        matches_fast_420: plan.matches_fast_tile_shape(),
        matches_fast_422: plan.matches_fast_rgb422_shape(),
        matches_fast_444: plan.matches_fast_rgb444_shape(),
    })
}

/// Summarize device-batch properties without materializing scan bytes.
pub fn summarize_device_batch(decoder: &Decoder<'_>, cadence_mcus: u32) -> DeviceBatchSummary {
    if !matches!(
        decoder.info().sof_kind,
        SofKind::Baseline8 | SofKind::Extended8
    ) {
        return DeviceBatchSummary {
            restart_interval: None,
            checkpoint_count: 0,
            matches_fast_420: false,
            matches_fast_422: false,
            matches_fast_444: false,
        };
    }
    let plan = &decoder.plan;
    let restart_interval = plan.restart_interval.filter(|&interval| interval > 0);
    let total_mcus = total_mcus(plan);
    let cadence_mcus = cadence_mcus.max(1);
    let checkpoint_count = match restart_interval {
        Some(restart) => 1usize.saturating_add(
            total_mcus
                .saturating_sub(1)
                .checked_div(u32::from(restart))
                .unwrap_or(0) as usize,
        ),
        None => 1usize.saturating_add(
            total_mcus
                .saturating_sub(1)
                .checked_div(cadence_mcus)
                .unwrap_or(0) as usize,
        ),
    };

    DeviceBatchSummary {
        restart_interval,
        checkpoint_count,
        matches_fast_420: plan.matches_fast_tile_shape(),
        matches_fast_422: plan.matches_fast_rgb422_shape(),
        matches_fast_444: plan.matches_fast_rgb444_shape(),
    }
}

fn scan_payload_bytes(
    bytes: &[u8],
    scan_offset: usize,
    allow_restart_markers: bool,
) -> Result<(Cow<'_, [u8]>, bool), JpegError> {
    let scan = &bytes[scan_offset..];
    let mut index = 0usize;
    while index < scan.len() {
        if scan[index] != 0xff {
            index += 1;
            continue;
        }

        let marker_start = index;
        let next = index + 1;
        if next >= scan.len() {
            return Ok((Cow::Borrowed(scan), true));
        }

        match scan[next] {
            0x00 => {
                index = next + 1;
            }
            0xd0..=0xd7 if allow_restart_markers => {
                index = next + 1;
            }
            0xd9 => return Ok((Cow::Borrowed(&scan[..=next]), false)),
            found => {
                return Err(JpegError::UnexpectedMarker {
                    offset: scan_offset + marker_start,
                    expected: MarkerKind::Eoi,
                    found,
                })
            }
        }
    }

    Ok((Cow::Borrowed(scan), true))
}

fn total_mcus(plan: &crate::entropy::sequential::PreparedDecodePlan) -> u32 {
    let mcu_width = u32::from(plan.sampling.max_h) * 8;
    let mcu_height = u32::from(plan.sampling.max_v) * 8;
    let mcus_per_row = plan.dimensions.0.div_ceil(mcu_width);
    let mcu_rows = plan.dimensions.1.div_ceil(mcu_height);
    mcus_per_row.saturating_mul(mcu_rows)
}
