// SPDX-License-Identifier: MIT OR Apache-2.0

use super::pipeline::decode_impl;
#[cfg(test)]
use super::pipeline::PHASE_LIMIT_MAGREF;
#[cfg(test)]
use super::segments::CombinedCodeBlockData;
use super::segments::HtCodeBlockSegments;
use super::state::{
    HtBlockDecodeScratch, HtBlockDecodeStats, NoHtDecodeStats, RecordingHtDecodeStats,
};
use crate::error::{bail, DecodingError, Result};

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "this monomorphized adapter preserves the validated decode phase inputs and observer choice"
)]
#[inline(always)]
fn decode_segments_with_scratch_for_phase<const PHASE_LIMIT: u8>(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    width: u32,
    height: u32,
    stride: u32,
    stripe_causal: bool,
    decoded_data: &mut [u32],
    scratch: &mut HtBlockDecodeScratch,
    stats: Option<&mut HtBlockDecodeStats>,
    profile_enabled: bool,
) -> Result<()> {
    let decoded = if let Some(stats) = stats {
        let mut observer = RecordingHtDecodeStats {
            stats,
            profile_enabled,
        };
        decode_impl::<PHASE_LIMIT, _>(
            segments.cleanup,
            segments.refinement,
            decoded_data,
            u32::from(missing_bit_planes),
            u32::from(number_of_coding_passes),
            width,
            height,
            stride,
            stripe_causal,
            scratch,
            &mut observer,
        )
    } else {
        let mut observer = NoHtDecodeStats;
        decode_impl::<PHASE_LIMIT, _>(
            segments.cleanup,
            segments.refinement,
            decoded_data,
            u32::from(missing_bit_planes),
            u32::from(number_of_coding_passes),
            width,
            height,
            stride,
            stripe_causal,
            scratch,
            &mut observer,
        )
    };

    decoded.ok_or(DecodingError::CodeBlockDecodeFailure.into())
}

#[cfg_attr(test, expect(clippy::too_many_arguments, reason = "HT test helper"))]
#[cfg(test)]
pub(crate) fn decode_segments_validated(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
) -> Result<()> {
    decode_segments_validated_for_phase::<PHASE_LIMIT_MAGREF>(
        segments,
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        stripe_causal,
        strict,
        decoded_data,
        width,
        height,
        stride,
    )
}

#[cfg_attr(test, expect(clippy::too_many_arguments, reason = "HT test helper"))]
#[cfg_attr(
    test,
    expect(clippy::inline_always, reason = "const phase specialization")
)]
#[inline(always)]
#[cfg(test)]
pub(crate) fn decode_segments_validated_for_phase<const PHASE_LIMIT: u8>(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
) -> Result<()> {
    if !validate_combined_decode(
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        strict,
    )? {
        return Ok(());
    }

    let mut scratch = HtBlockDecodeScratch::default();
    decode_segments_with_scratch_for_phase::<PHASE_LIMIT>(
        segments,
        missing_bit_planes,
        number_of_coding_passes,
        width,
        height,
        stride,
        stripe_causal,
        decoded_data,
        &mut scratch,
        None,
        false,
    )
}

#[cfg_attr(test, expect(clippy::too_many_arguments, reason = "HT test helper"))]
#[cfg(test)]
pub(super) fn decode_segments_validated_with_scratch(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    scratch: &mut HtBlockDecodeScratch,
) -> Result<()> {
    decode_segments_validated_with_scratch_for_phase::<PHASE_LIMIT_MAGREF>(
        segments,
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        stripe_causal,
        strict,
        decoded_data,
        width,
        height,
        stride,
        scratch,
        None,
        false,
    )
}

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "this stable validated facade keeps decode geometry, scratch, statistics, and profiling explicit"
)]
#[inline(always)]
pub(crate) fn decode_segments_validated_with_scratch_for_phase<const PHASE_LIMIT: u8>(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    scratch: &mut HtBlockDecodeScratch,
    stats: Option<&mut HtBlockDecodeStats>,
    profile_enabled: bool,
) -> Result<()> {
    if !validate_combined_decode(
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        strict,
    )? {
        return Ok(());
    }

    decode_segments_with_scratch_for_phase::<PHASE_LIMIT>(
        segments,
        missing_bit_planes,
        number_of_coding_passes,
        width,
        height,
        stride,
        stripe_causal,
        decoded_data,
        scratch,
        stats,
        profile_enabled,
    )
}

pub(super) fn validate_combined_decode(
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    strict: bool,
) -> Result<bool> {
    if total_bitplanes == 0 {
        return Ok(false);
    }

    if total_bitplanes > 31 {
        bail!(DecodingError::TooManyBitplanes);
    }

    let actual_bitplanes = if strict {
        total_bitplanes
            .checked_sub(missing_bit_planes)
            .ok_or(DecodingError::InvalidBitplaneCount)?
    } else {
        total_bitplanes.saturating_sub(missing_bit_planes)
    };

    let max_coding_passes = if actual_bitplanes == 0 {
        0
    } else {
        1 + 3 * (actual_bitplanes - 1)
    };

    if number_of_coding_passes > max_coding_passes && strict {
        bail!(DecodingError::TooManyCodingPasses);
    }

    Ok(number_of_coding_passes != 0 && actual_bitplanes != 0)
}

#[cfg_attr(test, expect(clippy::too_many_arguments, reason = "HT test helper"))]
#[cfg(test)]
pub(crate) fn decode_combined_validated(
    combined: &CombinedCodeBlockData,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
) -> Result<()> {
    let segments = combined.segments()?;
    decode_segments_validated(
        &segments,
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        stripe_causal,
        strict,
        decoded_data,
        width,
        height,
        stride,
    )
}

#[cfg_attr(test, expect(clippy::too_many_arguments, reason = "HT test helper"))]
#[cfg(test)]
pub(super) fn decode_combined_validated_with_scratch(
    combined: &CombinedCodeBlockData,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    decoded_data: &mut [u32],
    width: u32,
    height: u32,
    stride: u32,
    scratch: &mut HtBlockDecodeScratch,
) -> Result<()> {
    let segments = combined.segments()?;
    decode_segments_validated_with_scratch(
        &segments,
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        stripe_causal,
        strict,
        decoded_data,
        width,
        height,
        stride,
        scratch,
    )
}
