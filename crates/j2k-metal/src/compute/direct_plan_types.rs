// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{borrow::Cow, sync::Arc};

use j2k_native::{
    HtCodeBlockPayloadRanges, J2kDirectBandId, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kRequiredBandRegion, J2kWaveletTransform,
};
use metal::Buffer;

use super::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob};
use super::{CpuTier1CoefficientCache, DirectTier1Mode};

mod allocation;

pub(crate) struct PreparedDirectGrayscalePlan {
    pub(super) dimensions: (u32, u32),
    pub(super) bit_depth: u8,
    pub(super) tier1_prepare_mode: DirectTier1Mode,
    pub(super) steps: Vec<PreparedDirectGrayscaleStep>,
    pub(super) classic_groups: Vec<PreparedClassicSubBandGroup>,
    pub(super) ht_groups: Vec<PreparedHtSubBandGroup>,
    pub(super) cpu_tier1_cache: Arc<CpuTier1CoefficientCache>,
}

pub(crate) struct PreparedDirectColorPlan {
    pub(super) dimensions: (u32, u32),
    pub(super) bit_depths: [u8; 3],
    pub(super) alpha_bit_depth: Option<u8>,
    pub(super) signed: bool,
    pub(super) mct: bool,
    pub(super) transform: J2kWaveletTransform,
    pub(super) component_plans: Vec<PreparedDirectGrayscalePlan>,
}

pub(super) enum PreparedDirectGrayscaleStep {
    ClassicSubBand(PreparedClassicSubBand),
    HtSubBand(PreparedHtSubBand),
    Idwt(PreparedDirectIdwt),
    Store(J2kDirectStoreStep),
}

pub(super) struct PreparedDirectIdwt {
    pub(super) step: J2kDirectIdwtStep,
    pub(super) output_window: J2kRequiredBandRegion,
}

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

pub(super) struct PreparedClassicSubBandGroupMember {
    pub(super) band_id: J2kDirectBandId,
    pub(super) offset_elements: usize,
    pub(super) window: J2kRequiredBandRegion,
}

pub(super) struct PreparedHtSubBand {
    pub(super) band_id: J2kDirectBandId,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) payload_source: PreparedHtPayloadSource,
    pub(super) jobs: Vec<J2kHtCleanupBatchJob>,
    pub(super) execution_owner: Arc<PreparedHtExecutionOwner>,
}

pub(super) struct PreparedHtExecutionOwner;

pub(super) enum PreparedHtPayloadSource {
    Contiguous(Vec<u8>),
    Referenced {
        input: Arc<[u8]>,
        ranges: Vec<HtCodeBlockPayloadRanges>,
    },
}

impl PreparedHtPayloadSource {
    #[cfg(test)]
    pub(super) fn owned_payload_capacity(&self) -> usize {
        match self {
            Self::Contiguous(data) => data.capacity(),
            Self::Referenced { .. } => 0,
        }
    }

    pub(super) fn contiguous(&self) -> Option<&[u8]> {
        match self {
            Self::Contiguous(data) => Some(data),
            Self::Referenced { .. } => None,
        }
    }

    pub(super) fn contiguous_mut(&mut self) -> Option<&mut Vec<u8>> {
        match self {
            Self::Contiguous(data) => Some(data),
            Self::Referenced { .. } => None,
        }
    }

    pub(super) fn materialize_for_cpu(&self) -> Result<Cow<'_, [u8]>, crate::Error> {
        let Self::Referenced { input, ranges } = self else {
            return Ok(Cow::Borrowed(self.contiguous().ok_or(
                crate::Error::MetalStateInvariant {
                    state: "HTJ2K Metal prepared payload source",
                    reason: "contiguous payload source could not expose its bytes",
                },
            )?));
        };
        let payload_len = crate::batch_allocation::checked_count_sum(
            ranges.iter().flat_map(|payload| {
                core::iter::once(payload.cleanup.length)
                    .chain(payload.refinement.map(|range| range.length))
            }),
            "HTJ2K Metal CPU fallback referenced payload",
        )?;
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "HTJ2K Metal CPU fallback referenced payload",
        );
        let mut materialized =
            budget.try_vec(payload_len, "HTJ2K Metal CPU fallback referenced payload")?;
        for payload in ranges {
            append_referenced_payload_range(&mut materialized, input, payload.cleanup)?;
            if let Some(refinement) = payload.refinement {
                append_referenced_payload_range(&mut materialized, input, refinement)?;
            }
        }
        Ok(Cow::Owned(materialized))
    }
}

fn append_referenced_payload_range(
    destination: &mut Vec<u8>,
    input: &[u8],
    range: j2k_native::J2kCodestreamRange,
) -> Result<(), crate::Error> {
    let end = range.end().ok_or_else(|| crate::Error::MetalKernel {
        message: "HTJ2K Metal CPU fallback referenced payload range overflow".to_string(),
    })?;
    let bytes = input
        .get(range.offset..end)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "HTJ2K Metal CPU fallback referenced payload range exceeds retained input"
                .to_string(),
        })?;
    destination.extend_from_slice(bytes);
    Ok(())
}

pub(super) struct PreparedHtSubBandGroup {
    pub(super) start_step: usize,
    pub(super) end_step: usize,
    pub(super) total_coefficients: usize,
    pub(super) payload_source: PreparedHtPayloadSource,
    pub(super) jobs: Vec<J2kHtCleanupBatchJob>,
    pub(super) members: Vec<PreparedHtSubBandGroupMember>,
    pub(super) execution_owner: Arc<PreparedHtExecutionOwner>,
}

pub(super) struct PreparedHtSubBandGroupMember {
    pub(super) band_id: J2kDirectBandId,
    pub(super) offset_elements: usize,
    pub(super) window: J2kRequiredBandRegion,
}
