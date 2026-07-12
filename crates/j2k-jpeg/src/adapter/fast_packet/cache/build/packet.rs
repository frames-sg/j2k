// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast-family inspection and packet materialization for cached plans.

use crate::adapter::device_plan::{retained_decoder_allocation_bytes, DeviceBatchSummary};
use crate::adapter::fast_packet::build::{
    build_fast420_packet_from_decoder, build_fast422_packet_from_decoder,
    build_fast444_packet_from_decoder,
};
use crate::adapter::fast_packet::cache::shared_allocation::checked_live_bytes;
use crate::adapter::fast_packet::cache::{
    JpegCachedPlanBuildError, JpegFastPacketState, SharedJpegFastPacket,
};
use crate::adapter::fast_packet::family::{classify_color_fast_packet_info, JpegFastPacketFamily};
use crate::adapter::fast_packet::header::{
    ColorFastHeader, FastLayout, FAST420_LAYOUT, FAST422_LAYOUT, FAST444_LAYOUT,
};
use crate::adapter::fast_packet::{FastPacketError, JpegFastPacket};
use crate::decoder::{Decoder, JpegView};

pub(super) fn inspect_fast_header(
    view: &JpegView<'_>,
) -> Result<Option<(JpegFastPacketFamily, ColorFastHeader)>, FastPacketError> {
    let selected = classify_color_fast_packet_info(view.info())
        .map(|family| (family, layout_for_family(family)));
    match selected {
        Some((family, layout)) => match ColorFastHeader::inspect(view.parsed_header(), layout) {
            Ok(header) => Ok(Some((family, header))),
            Err(error) if error.is_capability_mismatch() => Ok(None),
            Err(error) => Err(error),
        },
        None => Ok(None),
    }
}

pub(super) fn materialize_packet_state(
    bytes: &[u8],
    decoder: &Decoder<'_>,
    family: JpegFastPacketFamily,
    header: ColorFastHeader,
    owner_live_bytes: usize,
    summary: &mut DeviceBatchSummary,
) -> Result<JpegFastPacketState, JpegCachedPlanBuildError> {
    match build_selected_packet(bytes, decoder, family, header, owner_live_bytes) {
        Ok(packet) => {
            set_fast_family(summary, family);
            let packet_external_live = checked_live_bytes(
                "cached JPEG packet external and decoder owners",
                owner_live_bytes,
                retained_decoder_allocation_bytes(decoder)?,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )?;
            Ok(JpegFastPacketState::Ready(
                SharedJpegFastPacket::try_new_with_external_live(packet, packet_external_live)?,
            ))
        }
        Err(error) if error.is_capability_mismatch() => {
            clear_fast_families(summary);
            Ok(JpegFastPacketState::Unsupported)
        }
        Err(error) => Err(error.into()),
    }
}

const fn layout_for_family(family: JpegFastPacketFamily) -> FastLayout {
    match family {
        JpegFastPacketFamily::Fast420 => FAST420_LAYOUT,
        JpegFastPacketFamily::Fast422 => FAST422_LAYOUT,
        JpegFastPacketFamily::Fast444 => FAST444_LAYOUT,
    }
}

fn build_selected_packet(
    bytes: &[u8],
    decoder: &Decoder<'_>,
    family: JpegFastPacketFamily,
    header: ColorFastHeader,
    external_live_bytes: usize,
) -> Result<JpegFastPacket, FastPacketError> {
    match family {
        JpegFastPacketFamily::Fast420 => {
            build_fast420_packet_from_decoder(bytes, decoder, header, external_live_bytes)
                .map(Into::into)
        }
        JpegFastPacketFamily::Fast422 => {
            build_fast422_packet_from_decoder(bytes, decoder, header, external_live_bytes)
                .map(Into::into)
        }
        JpegFastPacketFamily::Fast444 => {
            build_fast444_packet_from_decoder(bytes, decoder, header, external_live_bytes)
                .map(Into::into)
        }
    }
}

pub(super) const fn clear_fast_families(summary: &mut DeviceBatchSummary) {
    summary.matches_fast_420 = false;
    summary.matches_fast_422 = false;
    summary.matches_fast_444 = false;
}

const fn set_fast_family(summary: &mut DeviceBatchSummary, family: JpegFastPacketFamily) {
    clear_fast_families(summary);
    match family {
        JpegFastPacketFamily::Fast420 => summary.matches_fast_420 = true,
        JpegFastPacketFamily::Fast422 => summary.matches_fast_422 = true,
        JpegFastPacketFamily::Fast444 => summary.matches_fast_444 = true,
    }
}
