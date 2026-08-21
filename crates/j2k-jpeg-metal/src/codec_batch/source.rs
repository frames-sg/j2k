// SPDX-License-Identifier: MIT OR Apache-2.0

//! Normalized raw/prepared RGB8 batch source values.

use crate::{batch, Decoder};

pub(super) struct ResolvedRgb8BatchSource {
    pub(super) request: batch::QueuedRequest,
    pub(super) output_dimensions: (u32, u32),
    pub(super) sampling_family: batch::SamplingFamily,
    pub(super) restart_coded: bool,
    pub(super) cache_retained_bytes: usize,
}

pub(super) fn decoder_resident_sampling_family(decoder: &Decoder<'_>) -> batch::SamplingFamily {
    if decoder.fast420_packet().is_some() {
        batch::SamplingFamily::Fast420
    } else if decoder.fast422_packet().is_some() {
        batch::SamplingFamily::Fast422
    } else if decoder.fast444_packet().is_some() {
        batch::SamplingFamily::Fast444
    } else {
        batch::SamplingFamily::Other
    }
}

pub(super) fn decoder_resident_restart_interval_mcus(decoder: &Decoder<'_>) -> u32 {
    if let Some(packet) = decoder.fast420_packet() {
        packet.restart_interval_mcus
    } else if let Some(packet) = decoder.fast422_packet() {
        packet.restart_interval_mcus
    } else if let Some(packet) = decoder.fast444_packet() {
        packet.restart_interval_mcus
    } else {
        0
    }
}
