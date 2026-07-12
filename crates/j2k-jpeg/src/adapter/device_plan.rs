// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, ensure_allocation_bytes,
    try_reserve_for_len_with_live_budget,
};
use crate::decoder::Decoder;
use crate::error::JpegError;
use crate::info::{ColorSpace, SofKind};
use crate::internal::checkpoint::{
    build_checkpoint_plan_from_validated_with_live_budget, checkpoint_count_summary, total_mcus,
    validate_scan_bytes, DeviceCheckpoint,
};
use crate::Warning;
use alloc::borrow::Cow;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
/// Component metadata needed by device-side JPEG decoders.
pub struct DeviceComponentPlan {
    /// Horizontal sampling factor.
    pub h: u8,
    /// Vertical sampling factor.
    pub v: u8,
    /// Output component index in decoded pixel/component order.
    pub output_index: usize,
}

#[derive(Debug, PartialEq, Eq)]
#[doc(hidden)]
/// Validated decode plan and entropy payload for device-side decoders.
///
/// The plan is move-only because an owned scan payload, warning list, and
/// checkpoint graph can together approach the shared host-allocation cap.
/// Borrow it or use `Arc` when explicit shared ownership is required.
pub struct DeviceDecodePlan<'a> {
    /// Image dimensions as `(width, height)` in pixels.
    pub dimensions: (u32, u32),
    /// Header-derived JPEG color space.
    pub color_space: ColorSpace,
    /// Restart interval in MCUs, if restart markers are present.
    pub restart_interval: Option<u16>,
    /// Non-fatal warnings collected while building the plan.
    pub warnings: Vec<Warning>,
    /// Entropy scan bytes with device-compatible marker handling.
    pub scan_bytes: Cow<'a, [u8]>,
    /// Component sampling and output placement.
    pub components: Vec<DeviceComponentPlan>,
    /// Entropy checkpoints for restart or synthetic resume points.
    pub checkpoints: Vec<DeviceCheckpoint>,
    /// True when the tile matches the Metal/CUDA fast 4:2:0 path.
    pub matches_fast_420: bool,
    /// True when the tile matches the Metal/CUDA fast 4:2:2 path.
    pub matches_fast_422: bool,
    /// True when the tile matches the Metal/CUDA fast 4:4:4 path.
    pub matches_fast_444: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Lightweight summary used for batching without materializing scan bytes.
pub struct DeviceBatchSummary {
    /// Restart interval in MCUs, if restart markers are present.
    pub restart_interval: Option<u16>,
    /// Number of checkpoints the full device plan would contain.
    pub checkpoint_count: usize,
    /// True when the tile matches the fast 4:2:0 device path.
    pub matches_fast_420: bool,
    /// True when the tile matches the fast 4:2:2 device path.
    pub matches_fast_422: bool,
    /// True when the tile matches the fast 4:4:4 device path.
    pub matches_fast_444: bool,
}

/// Build a full device decode plan for an inspected decoder.
#[doc(hidden)]
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
    let validated_scan = validate_scan_bytes(
        &decoder.bytes[plan.scan_offset..],
        restart_interval.is_some(),
        plan.scan_offset,
    )?;
    let scan_bytes = Cow::Borrowed(validated_scan.payload());
    let missing_eoi = validated_scan.is_missing_eoi();
    let warning_count = decoder
        .warnings
        .len()
        .checked_add(usize::from(missing_eoi))
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    let expected_checkpoint_count = checkpoint_count_summary(
        total_mcus(plan),
        cadence_mcus.max(1),
        restart_interval.map(u32::from),
    );
    let retained_decoder_bytes = retained_decoder_allocation_bytes(decoder)?;
    let output_bytes = device_plan_output_allocation_bytes(
        expected_checkpoint_count,
        warning_count,
        plan.components.len(),
    )?;
    let terminated_copy_bytes =
        validated_scan.terminated_copy_len(j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
    ensure_allocation_bytes(checked_add_allocation_bytes(
        checked_add_allocation_bytes(retained_decoder_bytes, output_bytes)?,
        terminated_copy_bytes,
    )?)?;

    let mut live_bytes = retained_decoder_bytes;
    let mut warnings = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut warnings,
        warning_count,
        &mut live_bytes,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    warnings.extend_from_slice(&decoder.warnings);
    if missing_eoi {
        warnings.push(Warning::MissingEoi);
    }

    let mut components = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut components,
        plan.components.len(),
        &mut live_bytes,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    components.extend(plan.components.iter().map(|component| DeviceComponentPlan {
        h: component.h,
        v: component.v,
        output_index: component.output_index,
    }));
    let checkpoints = build_checkpoint_plan_from_validated_with_live_budget(
        plan,
        validated_scan,
        cadence_mcus,
        &mut live_bytes,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;
    if checkpoints.len() != expected_checkpoint_count {
        return Err(JpegError::InternalInvariant {
            reason: "device checkpoint count disagrees with its allocation plan",
        });
    }

    Ok(DeviceDecodePlan {
        dimensions: plan.dimensions,
        color_space: plan.color_space,
        restart_interval,
        warnings,
        scan_bytes,
        components,
        checkpoints,
        matches_fast_420: plan.matches_fast_tile_shape(),
        matches_fast_422: plan.matches_fast_rgb422_shape(),
        matches_fast_444: plan.matches_fast_rgb444_shape(),
    })
}

pub(super) fn retained_decoder_allocation_bytes(decoder: &Decoder<'_>) -> Result<usize, JpegError> {
    let baseline = decoder.retained_allocation_bytes_excluding_cpu_checkpoint_cache()?;
    let checkpoint_cache_bytes = decoder
        .cpu_entropy_checkpoints
        .lock()
        .map_err(|_| JpegError::InternalInvariant {
            reason: "CPU entropy checkpoint cache mutex poisoned",
        })?
        .retained_allocation_bytes()?;
    checked_add_allocation_bytes(baseline, checkpoint_cache_bytes)
}

fn device_plan_output_allocation_bytes(
    checkpoint_count: usize,
    warning_count: usize,
    component_count: usize,
) -> Result<usize, JpegError> {
    let mut total = checked_allocation_bytes::<DeviceCheckpoint>(checkpoint_count)?;
    total =
        checked_add_allocation_bytes(total, checked_allocation_bytes::<Warning>(warning_count)?)?;
    checked_add_allocation_bytes(
        total,
        checked_allocation_bytes::<DeviceComponentPlan>(component_count)?,
    )
}

/// Summarize device planning properties without copying entropy data.
#[doc(hidden)]
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
    let checkpoint_count = checkpoint_count_summary(
        total_mcus(plan),
        cadence_mcus,
        restart_interval.map(u32::from),
    );

    DeviceBatchSummary {
        restart_interval,
        checkpoint_count,
        matches_fast_420: plan.matches_fast_tile_shape(),
        matches_fast_422: plan.matches_fast_rgb422_shape(),
        matches_fast_444: plan.matches_fast_rgb444_shape(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_plan_output_rejects_aggregate_retained_vectors() {
        let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
        let checkpoint_count = (cap * 3 / 5) / core::mem::size_of::<DeviceCheckpoint>();
        let warning_count = (cap * 3 / 5) / core::mem::size_of::<Warning>();
        assert!(checked_allocation_bytes::<DeviceCheckpoint>(checkpoint_count).is_ok());
        assert!(checked_allocation_bytes::<Warning>(warning_count).is_ok());
        assert!(matches!(
            device_plan_output_allocation_bytes(checkpoint_count, warning_count, 0),
            Err(JpegError::MemoryCapExceeded { requested, cap: limit })
                if requested > limit && limit == cap
        ));
    }

    #[test]
    fn device_plan_output_boundary_counts_all_three_payloads() {
        let checkpoints = 3usize;
        let warnings = 2usize;
        let components = 4usize;
        let expected = checkpoints * core::mem::size_of::<DeviceCheckpoint>()
            + warnings * core::mem::size_of::<Warning>()
            + components * core::mem::size_of::<DeviceComponentPlan>();
        assert_eq!(
            device_plan_output_allocation_bytes(checkpoints, warnings, components).unwrap(),
            expected
        );
    }

    #[test]
    fn retained_decoder_bytes_include_populated_cpu_checkpoint_cache() {
        let bytes = j2k_test_support::minimal_baseline_jpeg();
        let decoder = Decoder::new(&bytes).unwrap();
        let before = retained_decoder_allocation_bytes(&decoder).unwrap();
        let retained_checkpoint_bytes = {
            let mut cache = decoder.cpu_entropy_checkpoints.lock().unwrap();
            cache.checkpoints.try_reserve_exact(4).unwrap();
            cache.retained_allocation_bytes().unwrap()
        };
        let after = retained_decoder_allocation_bytes(&decoder).unwrap();

        assert!(retained_checkpoint_bytes >= 4 * core::mem::size_of::<DeviceCheckpoint>());
        assert_eq!(after, before + retained_checkpoint_bytes);
    }
}
