// SPDX-License-Identifier: MIT OR Apache-2.0

mod accounting;

use alloc::sync::Arc;

use super::shared_allocation::{checked_live_bytes, shared_owner_bytes};
use super::JpegPlanCacheError;
use crate::adapter::fast_packet::{
    JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1, JpegFastPacketFamily,
};
use accounting::color_packet_capacity_bytes;

#[derive(Debug)]
#[doc(hidden)]
/// Exactly one validated color fast-packet family for a JPEG input.
pub enum JpegFastPacket {
    /// 8-bit 4:2:0 packet.
    Fast420(JpegFast420PacketV1),
    /// 8-bit 4:2:2 packet.
    Fast422(JpegFast422PacketV1),
    /// 8-bit 4:4:4 packet.
    Fast444(JpegFast444PacketV1),
}

impl JpegFastPacket {
    /// Packet sampling family.
    #[must_use]
    pub const fn family(&self) -> JpegFastPacketFamily {
        match self {
            Self::Fast420(_) => JpegFastPacketFamily::Fast420,
            Self::Fast422(_) => JpegFastPacketFamily::Fast422,
            Self::Fast444(_) => JpegFastPacketFamily::Fast444,
        }
    }

    /// Borrow a 4:2:0 packet when this is the matching family.
    #[must_use]
    pub const fn fast420(&self) -> Option<&JpegFast420PacketV1> {
        match self {
            Self::Fast420(packet) => Some(packet),
            Self::Fast422(_) | Self::Fast444(_) => None,
        }
    }

    /// Borrow a 4:2:2 packet when this is the matching family.
    #[must_use]
    pub const fn fast422(&self) -> Option<&JpegFast422PacketV1> {
        match self {
            Self::Fast422(packet) => Some(packet),
            Self::Fast420(_) | Self::Fast444(_) => None,
        }
    }

    /// Borrow a 4:4:4 packet when this is the matching family.
    #[must_use]
    pub const fn fast444(&self) -> Option<&JpegFast444PacketV1> {
        match self {
            Self::Fast444(packet) => Some(packet),
            Self::Fast420(_) | Self::Fast422(_) => None,
        }
    }

    fn nested_capacity_bytes(&self) -> Result<usize, JpegPlanCacheError> {
        match self {
            Self::Fast420(packet) => color_packet_capacity_bytes(
                &packet.restart_offsets,
                &packet.entropy_checkpoints,
                &packet.entropy_bytes,
            ),
            Self::Fast422(packet) => color_packet_capacity_bytes(
                &packet.restart_offsets,
                &packet.entropy_checkpoints,
                &packet.entropy_bytes,
            ),
            Self::Fast444(packet) => color_packet_capacity_bytes(
                &packet.restart_offsets,
                &packet.entropy_checkpoints,
                &packet.entropy_bytes,
            ),
        }
    }
}

impl From<JpegFast420PacketV1> for JpegFastPacket {
    fn from(packet: JpegFast420PacketV1) -> Self {
        Self::Fast420(packet)
    }
}

impl From<JpegFast422PacketV1> for JpegFastPacket {
    fn from(packet: JpegFast422PacketV1) -> Self {
        Self::Fast422(packet)
    }
}

impl From<JpegFast444PacketV1> for JpegFastPacket {
    fn from(packet: JpegFast444PacketV1) -> Self {
        Self::Fast444(packet)
    }
}

#[derive(Clone)]
#[doc(hidden)]
/// Cheap shared ownership for exactly one color fast-packet family.
///
/// Packet vectors are move-only and charged by actual capacity. The single
/// `Arc` allocation is intentionally small relative to the packet graph; see
/// [`super::SharedJpegInput`] for the stable-Rust control-block accounting limitation.
pub struct SharedJpegFastPacket(Arc<JpegFastPacket>);

impl SharedJpegFastPacket {
    /// Move a validated packet behind one preflighted shared owner.
    ///
    /// # Errors
    ///
    /// Returns a typed limit or invariant error before allocating the Arc.
    pub fn try_new(packet: JpegFastPacket) -> Result<Self, JpegPlanCacheError> {
        Self::try_new_with_external_live(packet, 0)
    }

    /// Share a packet while charging owners already live in the operation.
    ///
    /// # Errors
    ///
    /// Returns a typed limit or invariant error before allocating the Arc.
    #[doc(hidden)]
    pub fn try_new_with_external_live(
        packet: JpegFastPacket,
        external_live_bytes: usize,
    ) -> Result<Self, JpegPlanCacheError> {
        Self::try_new_with_external_live_and_cap(
            packet,
            external_live_bytes,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    fn try_new_with_external_live_and_cap(
        packet: JpegFastPacket,
        external_live_bytes: usize,
        cap: usize,
    ) -> Result<Self, JpegPlanCacheError> {
        checked_live_bytes(
            "shared JPEG fast-packet owner graph",
            external_live_bytes,
            shared_owner_bytes::<JpegFastPacket>(packet.nested_capacity_bytes()?)?,
            cap,
        )?;
        Ok(Self(Arc::new(packet)))
    }

    #[cfg(test)]
    pub(in crate::adapter::fast_packet::cache) fn try_new_with_cap_for_test(
        packet: JpegFastPacket,
        external_live_bytes: usize,
        cap: usize,
    ) -> Result<Self, JpegPlanCacheError> {
        Self::try_new_with_external_live_and_cap(packet, external_live_bytes, cap)
    }

    /// Borrow the underlying one-family packet.
    #[must_use]
    pub fn as_packet(&self) -> &JpegFastPacket {
        self.0.as_ref()
    }

    /// Borrow a 4:2:0 packet when this is the matching family.
    #[must_use]
    pub fn fast420(&self) -> Option<&JpegFast420PacketV1> {
        self.as_packet().fast420()
    }

    /// Borrow a 4:2:2 packet when this is the matching family.
    #[must_use]
    pub fn fast422(&self) -> Option<&JpegFast422PacketV1> {
        self.as_packet().fast422()
    }

    /// Borrow a 4:4:4 packet when this is the matching family.
    #[must_use]
    pub fn fast444(&self) -> Option<&JpegFast444PacketV1> {
        self.as_packet().fast444()
    }

    /// Whether two handles share the same packet allocation.
    #[must_use]
    pub fn ptr_eq(left: &Self, right: &Self) -> bool {
        Arc::ptr_eq(&left.0, &right.0)
    }

    /// Retained bytes charged to a cache entry for this packet graph.
    ///
    /// Nested vector capacities are exact; the fixed shared allocation has the
    /// stable-Rust limitation documented on [`super::SharedJpegInput`].
    ///
    /// # Errors
    ///
    /// Returns an invariant error if retained-byte arithmetic overflows.
    pub fn retained_cache_bytes(&self) -> Result<usize, JpegPlanCacheError> {
        shared_owner_bytes::<JpegFastPacket>(self.0.nested_capacity_bytes()?)
    }
}

impl core::fmt::Debug for SharedJpegFastPacket {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SharedJpegFastPacket")
            .field("family", &self.as_packet().family())
            .finish_non_exhaustive()
    }
}
