// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_native::{
    J2kDirectBandId, J2kDirectIdwtStep, J2kDirectStoreStep, J2kRequiredBandRegion,
    J2kWaveletTransform,
};
use metal::Buffer;

use super::{
    CpuTier1CoefficientCache, DirectTier1Mode, J2kClassicCleanupBatchJob, J2kClassicSegment,
    J2kHtCleanupBatchJob,
};

#[derive(Clone)]
pub(crate) struct PreparedDirectGrayscalePlan {
    pub(super) dimensions: (u32, u32),
    pub(super) bit_depth: u8,
    pub(super) tier1_prepare_mode: DirectTier1Mode,
    pub(super) steps: Vec<PreparedDirectGrayscaleStep>,
    pub(super) classic_groups: Vec<PreparedClassicSubBandGroup>,
    pub(super) ht_groups: Vec<PreparedHtSubBandGroup>,
    pub(super) cpu_tier1_cache: Arc<CpuTier1CoefficientCache>,
}

#[derive(Clone)]
pub(crate) struct PreparedDirectColorPlan {
    pub(super) dimensions: (u32, u32),
    pub(super) bit_depths: [u8; 3],
    pub(super) mct: bool,
    pub(super) transform: J2kWaveletTransform,
    pub(super) component_plans: Vec<PreparedDirectGrayscalePlan>,
}

#[derive(Clone)]
pub(super) enum PreparedDirectGrayscaleStep {
    ClassicSubBand(PreparedClassicSubBand),
    HtSubBand(PreparedHtSubBand),
    Idwt(PreparedDirectIdwt),
    Store(J2kDirectStoreStep),
}

#[derive(Clone)]
pub(super) struct PreparedDirectIdwt {
    pub(super) step: J2kDirectIdwtStep,
    pub(super) output_window: J2kRequiredBandRegion,
}

#[derive(Clone)]
pub(super) struct PreparedClassicSubBand {
    pub(super) band_id: J2kDirectBandId,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) zero_fill: bool,
    pub(super) coded_data: Vec<u8>,
    pub(super) coded_buffer: Buffer,
    pub(super) jobs: Vec<J2kClassicCleanupBatchJob>,
    pub(super) jobs_buffer: Buffer,
    pub(super) segments: Vec<J2kClassicSegment>,
    pub(super) segments_buffer: Buffer,
}

#[derive(Clone)]
pub(super) struct PreparedClassicSubBandGroup {
    pub(super) start_step: usize,
    pub(super) end_step: usize,
    pub(super) total_coefficients: usize,
    pub(super) zero_fill: bool,
    pub(super) coded_data: Vec<u8>,
    pub(super) coded_buffer: Buffer,
    pub(super) jobs: Vec<J2kClassicCleanupBatchJob>,
    pub(super) jobs_buffer: Buffer,
    pub(super) segments: Vec<J2kClassicSegment>,
    pub(super) segments_buffer: Buffer,
    pub(super) members: Vec<PreparedClassicSubBandGroupMember>,
}

#[derive(Clone)]
pub(super) struct PreparedClassicSubBandGroupMember {
    pub(super) band_id: J2kDirectBandId,
    pub(super) offset_elements: usize,
    pub(super) window: J2kRequiredBandRegion,
}

#[derive(Clone)]
pub(super) struct PreparedHtSubBand {
    pub(super) band_id: J2kDirectBandId,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) coded_data: Vec<u8>,
    pub(super) coded_buffer: Option<Buffer>,
    pub(super) jobs: Vec<J2kHtCleanupBatchJob>,
    pub(super) jobs_buffer: Option<Buffer>,
}

#[derive(Clone)]
pub(super) struct HtCodedArena {
    pub(super) data: Vec<u8>,
    pub(super) buffer: Buffer,
}

#[derive(Clone)]
pub(super) struct PreparedHtSubBandGroup {
    pub(super) start_step: usize,
    pub(super) end_step: usize,
    pub(super) total_coefficients: usize,
    pub(super) coded_arena: HtCodedArena,
    pub(super) jobs: Vec<J2kHtCleanupBatchJob>,
    pub(super) jobs_buffer: Buffer,
    pub(super) members: Vec<PreparedHtSubBandGroupMember>,
}

#[derive(Clone)]
pub(super) struct PreparedHtSubBandGroupMember {
    pub(super) band_id: J2kDirectBandId,
    pub(super) offset_elements: usize,
    pub(super) window: J2kRequiredBandRegion,
}
