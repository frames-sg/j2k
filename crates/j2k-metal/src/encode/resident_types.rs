// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use j2k::J2kLosslessEncodeOptions;
use j2k_native::J2kPacketizationPacketDescriptor;

use crate::compute;

use super::{
    submitted::OwnedMetalLosslessEncodeTile, LosslessDeviceEncodePlan, MetalEncodeInputStaging,
    MetalEncodedJ2k, MetalLosslessEncodeBatchStats,
};

pub(super) struct ResidentLosslessBufferEncodeMetadata {
    pub(super) tile: OwnedMetalLosslessEncodeTile,
    pub(super) components: u8,
    pub(super) bit_depth: u8,
    pub(super) bytes_per_pixel: usize,
    pub(super) plan: LosslessDeviceEncodePlan,
    pub(super) packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    pub(super) packetization_resolutions: Vec<compute::J2kResidentPacketizationResolution>,
}

pub(super) struct PreparedResidentLosslessBufferEncode {
    pub(super) metadata: ResidentLosslessBufferEncodeMetadata,
    pub(super) prepared: compute::J2kPreparedLosslessDeviceCodeBlocks,
}

pub(super) struct PlannedResidentLosslessBufferEncode {
    pub(super) index: usize,
    pub(super) metadata: ResidentLosslessBufferEncodeMetadata,
    pub(super) coefficient_count: usize,
    pub(super) bytes_per_sample: u8,
    pub(super) estimated_peak_bytes: usize,
    #[cfg(test)]
    pub(super) failure_injection_index: Option<usize>,
}

impl PlannedResidentLosslessBufferEncode {
    pub(super) fn estimated_peak_bytes(&self) -> usize {
        self.estimated_peak_bytes
    }
}

pub(super) struct SubmittedResidentLosslessMetalBufferEncodeBatch {
    pub(super) options: J2kLosslessEncodeOptions,
    pub(super) session: crate::MetalBackendSession,
    pub(super) stats: MetalLosslessEncodeBatchStats,
    pub(super) encode_started: Instant,
    pub(super) tiles: Vec<OwnedMetalLosslessEncodeTile>,
    pub(super) staging: MetalEncodeInputStaging,
    pub(super) kind: SubmittedResidentLosslessMetalBufferEncodeBatchKind,
}

pub(super) enum SubmittedResidentLosslessMetalBufferEncodeBatchKind {
    Empty,
    Chunks(Vec<SubmittedResidentLosslessMetalBufferEncodeChunk>),
}

pub(super) struct SubmittedResidentLosslessMetalBufferEncodeChunk {
    pub(super) metadatas: Vec<ResidentLosslessBufferEncodeMetadata>,
    pub(super) prepare_durations: Vec<Duration>,
    pub(super) pending: compute::J2kPendingResidentLosslessCodestreamBatch,
    pub(super) batch_started: Instant,
}

pub(super) struct FinishedResidentLosslessBufferEncode {
    pub(super) metadata: ResidentLosslessBufferEncodeMetadata,
    pub(super) encoded: MetalEncodedJ2k,
    pub(super) encode_duration: Duration,
    pub(super) gpu_duration: Option<Duration>,
}
