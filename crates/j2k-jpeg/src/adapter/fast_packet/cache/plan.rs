// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{JpegPlanCacheError, SharedJpegFastPacket, SharedJpegInput};
use crate::adapter::device_plan::DeviceBatchSummary;
use crate::adapter::fast_packet::JpegFastPacketFamily;
use crate::ColorSpace;

#[derive(Clone, Debug)]
#[doc(hidden)]
/// Result of the inspect-once fast-packet capability decision.
pub enum JpegFastPacketState {
    /// The input was inspected successfully but has no supported packet family.
    Unsupported,
    /// Exactly one validated fast-packet family is ready for an adapter.
    Ready(SharedJpegFastPacket),
}

#[derive(Clone, Debug)]
#[doc(hidden)]
/// One inspect-once accelerator plan cached for one complete JPEG input.
pub struct JpegCachedPlan {
    input: SharedJpegInput,
    summary: DeviceBatchSummary,
    color_space: ColorSpace,
    packet_state: JpegFastPacketState,
}

impl JpegCachedPlan {
    /// Construct one inspected plan, validating any ready packet family.
    ///
    /// # Errors
    ///
    /// Returns an invariant error if a ready packet contradicts the device
    /// batch summary. Parse, build, and allocation errors must be returned by
    /// callers before constructing a cached plan; they are not cached as
    /// [`JpegFastPacketState::Unsupported`].
    pub fn try_new(
        input: SharedJpegInput,
        summary: DeviceBatchSummary,
        color_space: ColorSpace,
        packet_state: JpegFastPacketState,
    ) -> Result<Self, JpegPlanCacheError> {
        let family_count = usize::from(summary.matches_fast_420)
            + usize::from(summary.matches_fast_422)
            + usize::from(summary.matches_fast_444);
        match &packet_state {
            JpegFastPacketState::Unsupported if family_count != 0 => {
                return Err(JpegPlanCacheError::Invariant(
                    "unsupported JPEG packet has a fast-family device batch summary",
                ));
            }
            JpegFastPacketState::Ready(packet) => {
                let (matches, color_matches) = match packet.as_packet().family() {
                    JpegFastPacketFamily::Fast420 => {
                        (summary.matches_fast_420, color_space == ColorSpace::YCbCr)
                    }
                    JpegFastPacketFamily::Fast422 => {
                        (summary.matches_fast_422, color_space == ColorSpace::YCbCr)
                    }
                    JpegFastPacketFamily::Fast444 => (
                        summary.matches_fast_444,
                        matches!(color_space, ColorSpace::YCbCr | ColorSpace::Rgb),
                    ),
                };
                if family_count != 1 || !matches || !color_matches {
                    return Err(JpegPlanCacheError::Invariant(
                        "ready JPEG packet must match exactly one device family and color mode",
                    ));
                }
            }
            JpegFastPacketState::Unsupported => {}
        }
        Ok(Self {
            input,
            summary,
            color_space,
            packet_state,
        })
    }

    /// Shared complete input bytes.
    #[must_use]
    pub const fn input(&self) -> &SharedJpegInput {
        &self.input
    }

    /// Inspect-once device batch summary.
    #[must_use]
    pub const fn batch_summary(&self) -> DeviceBatchSummary {
        self.summary
    }

    /// Inspect-once JPEG color interpretation used by adapter plane routing.
    #[must_use]
    pub const fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    /// Explicit supported/unsupported packet state.
    #[must_use]
    pub const fn packet_state(&self) -> &JpegFastPacketState {
        &self.packet_state
    }

    /// Borrow the ready shared packet, if this is a supported family.
    #[must_use]
    pub const fn fast_packet(&self) -> Option<&SharedJpegFastPacket> {
        match &self.packet_state {
            JpegFastPacketState::Unsupported => None,
            JpegFastPacketState::Ready(packet) => Some(packet),
        }
    }

    pub(super) fn retained_cache_bytes(&self) -> Result<usize, JpegPlanCacheError> {
        let input_bytes = self.input.retained_cache_bytes()?;
        let packet_bytes = match &self.packet_state {
            JpegFastPacketState::Unsupported => 0,
            JpegFastPacketState::Ready(packet) => packet.retained_cache_bytes()?,
        };
        input_bytes
            .checked_add(packet_bytes)
            .ok_or(JpegPlanCacheError::Invariant(
                "cached JPEG plan retained-byte count overflow",
            ))
    }
}
