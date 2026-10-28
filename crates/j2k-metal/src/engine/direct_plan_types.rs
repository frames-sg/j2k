// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::{
    borrow::Cow,
    sync::{Arc, OnceLock},
};

use crate::metal_types::Buffer;
use j2k_native::{
    HtCodeBlockPayloadRanges, J2kDirectBandId, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kRequiredBandRegion, J2kWaveletTransform,
};

use super::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob, J2kZeroRect};
use super::{CpuTier1CoefficientCache, DirectTier1Mode};

mod allocation;

pub(crate) struct PreparedDirectGrayscalePlan {
    pub(super) dimensions: (u32, u32),
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
    /// Whether the final store rounds centered samples ties-to-even before
    /// the level shift, as the CPU does for integer output. False only for
    /// the first three components of an MCT color plan, which the inverse
    /// transform rounds instead.
    pub(super) round_centered_store: bool,
    pub(super) tier1_prepare_mode: DirectTier1Mode,
    pub(super) steps: Vec<PreparedDirectGrayscaleStep>,
    pub(super) classic_groups: Vec<PreparedClassicSubBandGroup>,
    pub(super) ht_groups: Vec<PreparedHtSubBandGroup>,
    pub(super) cpu_tier1_cache: Arc<CpuTier1CoefficientCache>,
}

// SAFETY: A prepared plan is immutable after construction, except that each
// classic sub-band and group fills its Tier-1 input buffers on first use. The
// `OnceLock` holding them synchronizes that first fill, and the buffers are
// immutable afterwards. Retained Metal buffers are cross-thread resources and
// are bound only for read access; all host-owned vectors and the coefficient
// cache are immutable through this API.
unsafe impl Send for PreparedDirectGrayscalePlan {}
// SAFETY: The same contract permits concurrent readers: they observe either an
// empty `OnceLock` or fully built buffers, and GPU writes target per-submission
// scratch rather than these cached buffers.
unsafe impl Sync for PreparedDirectGrayscalePlan {}

pub(crate) struct PreparedDirectColorPlan {
    pub(super) dimensions: (u32, u32),
    pub(super) bit_depths: [u8; 3],
    pub(super) alpha_bit_depth: Option<u8>,
    pub(super) signed: bool,
    pub(super) mct: bool,
    pub(super) transform: J2kWaveletTransform,
    /// SIZ sampling factors of the three color components. A component plan
    /// with factors other than `(1, 1)` decodes on its own grid and is
    /// replicated onto `dimensions` before the final store. Alpha is always
    /// unit-sampled.
    pub(super) color_sampling: [(u8, u8); 3],
    pub(super) component_plans: Vec<PreparedDirectGrayscalePlan>,
}

pub(super) const UNIT_COLOR_SAMPLING: [(u8, u8); 3] = [(1, 1); 3];

impl PreparedDirectColorPlan {
    /// Sampling factors of component `index`.
    pub(super) fn component_sampling(&self, index: usize) -> (u8, u8) {
        self.color_sampling.get(index).copied().unwrap_or((1, 1))
    }

    /// Dimensions component `index` decodes at before any replication.
    pub(super) fn component_dimensions(&self, index: usize) -> (u32, u32) {
        sampled_dimensions(self.dimensions, self.component_sampling(index))
    }

    /// Whether any component decodes on a smaller grid than the image.
    pub(super) fn has_sampled_components(&self) -> bool {
        (0..self.component_plans.len())
            .any(|index| self.component_dimensions(index) != self.dimensions)
    }
}

#[cfg(test)]
impl PreparedDirectColorPlan {
    /// Output dimensions and the coefficients each component's Tier-1 groups
    /// store.
    pub(crate) fn geometry_for_test(&self) -> ((u32, u32), Vec<usize>) {
        let coefficients = self
            .component_plans
            .iter()
            .map(|component| {
                component
                    .classic_groups
                    .iter()
                    .map(|group| group.total_coefficients)
                    .chain(
                        component
                            .ht_groups
                            .iter()
                            .map(|group| group.total_coefficients),
                    )
                    .sum()
            })
            .collect();
        (self.dimensions, coefficients)
    }

    /// Tier-1 jobs and owned coded bytes across all components.
    pub(crate) fn jobs_for_test(&self) -> (usize, usize) {
        let mut jobs = 0;
        let mut coded = 0;
        for component in &self.component_plans {
            for step in &component.steps {
                match step {
                    PreparedDirectGrayscaleStep::ClassicSubBand(band) => {
                        jobs += band.jobs.len();
                        coded += band.coded_data.len();
                    }
                    PreparedDirectGrayscaleStep::HtSubBand(band) => {
                        jobs += band.jobs.len();
                        coded += band.payload_source.owned_payload_capacity();
                    }
                    _ => {}
                }
            }
        }
        (jobs, coded)
    }
}

/// Component-grid dimensions of an origin-zero image. Zero factors are
/// rejected by codestream parsing; they map to the image grid here so the
/// caller's dimension check fails instead of dividing by zero.
pub(super) fn sampled_dimensions(dimensions: (u32, u32), sampling: (u8, u8)) -> (u32, u32) {
    let (sx, sy) = (u32::from(sampling.0.max(1)), u32::from(sampling.1.max(1)));
    (dimensions.0.div_ceil(sx), dimensions.1.div_ceil(sy))
}

// SAFETY: The color plan contains only immutable metadata and prepared
// grayscale plans satisfying the cross-thread contract above.
unsafe impl Send for PreparedDirectColorPlan {}
// SAFETY: No mutable state is exposed after construction, so concurrent cache
// readers cannot race host or GPU writes through this owner.
unsafe impl Sync for PreparedDirectColorPlan {}

/// All components of a direct color store must share source signedness.
pub(super) fn color_signedness(
    plans: &[j2k_native::J2kDirectGrayscalePlan],
) -> Result<bool, crate::Error> {
    let Some(first) = plans.first() else {
        return Err(crate::Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch("Metal color plan has no components"),
        ));
    };
    if plans.iter().any(|plan| plan.signed != first.signed) {
        return Err(crate::Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_operation(
                "Metal decode needs every component to share one signedness",
            ),
        ));
    }
    Ok(first.signed)
}

/// Applies the CPU's color rounding split to freshly prepared component plans:
/// with MCT, components 0-2 are rounded after the inverse transform and any
/// alpha component at its own store.
pub(super) fn with_color_store_rounding(
    mut component_plans: Vec<PreparedDirectGrayscalePlan>,
    mct: bool,
) -> Vec<PreparedDirectGrayscalePlan> {
    for (index, plan) in component_plans.iter_mut().enumerate() {
        plan.round_centered_store = !(mct && index < 3);
    }
    component_plans
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
    /// The stored coefficients: `width` x `height` from `origin` in sub-band
    /// coordinates, the whole sub-band unless a region crop narrowed it.
    pub(super) origin: (u32, u32),
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) zero_fill: bool,
    pub(super) coded_data: Vec<u8>,
    pub(super) jobs: Vec<J2kClassicCleanupBatchJob>,
    pub(super) segments: Vec<J2kClassicSegment>,
    pub(super) tier1_inputs: ClassicTier1Inputs,
}

pub(super) struct PreparedClassicSubBandGroup {
    pub(super) start_step: usize,
    pub(super) end_step: usize,
    pub(super) total_coefficients: usize,
    pub(super) zero_fill: bool,
    pub(super) coded_data: Vec<u8>,
    pub(super) jobs: Vec<J2kClassicCleanupBatchJob>,
    pub(super) segments: Vec<J2kClassicSegment>,
    pub(super) tier1_inputs: ClassicTier1Inputs,
    pub(super) members: Vec<PreparedClassicSubBandGroupMember>,
}

/// Device copies of a classic sub-band's or group's `coded_data`, `jobs` and
/// `segments`.
pub(super) struct ClassicTier1Buffers {
    pub(super) coded: Buffer,
    pub(super) jobs: Buffer,
    pub(super) segments: Buffer,
}

/// The Tier-1 input buffers of a classic sub-band or group, built from its
/// host vectors the first time a dispatch binds them. The batch RGB routes
/// upload their own concatenated copies, so plans decoded only through them
/// never build these.
pub(super) struct ClassicTier1Inputs {
    pub(super) mode: DirectTier1Mode,
    pub(super) buffers: OnceLock<ClassicTier1Buffers>,
}

pub(super) struct PreparedClassicSubBandGroupMember {
    pub(super) band_id: J2kDirectBandId,
    pub(super) offset_elements: usize,
    pub(super) window: J2kRequiredBandRegion,
}

pub(super) struct PreparedHtSubBand {
    pub(super) band_id: J2kDirectBandId,
    /// As [`PreparedClassicSubBand::origin`].
    pub(super) origin: (u32, u32),
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) payload_source: PreparedHtPayloadSource,
    pub(super) jobs: Vec<J2kHtCleanupBatchJob>,
    pub(super) execution_owner: Arc<PreparedHtExecutionOwner>,
}

pub(super) struct PreparedHtExecutionOwner;

impl PreparedClassicSubBand {
    /// The stored coefficients' window in sub-band coordinates.
    pub(super) fn window(&self) -> J2kRequiredBandRegion {
        stored_window(self.origin, self.width, self.height)
    }
}

impl PreparedHtSubBand {
    /// The stored coefficients' window in sub-band coordinates.
    pub(super) fn window(&self) -> J2kRequiredBandRegion {
        stored_window(self.origin, self.width, self.height)
    }
}

fn stored_window((x0, y0): (u32, u32), width: u32, height: u32) -> J2kRequiredBandRegion {
    J2kRequiredBandRegion {
        x0,
        y0,
        x1: x0 + width,
        y1: y0 + height,
    }
}

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
    /// Arena rectangles no job writes, which decoding zeroes first.
    pub(super) zero_rects: Vec<J2kZeroRect>,
    pub(super) execution_owner: Arc<PreparedHtExecutionOwner>,
}

impl PreparedHtSubBandGroup {
    pub(super) fn zero_layout(&self) -> super::decode_dispatch::HtZeroLayout<'_> {
        super::decode_dispatch::HtZeroLayout {
            extent: self.total_coefficients,
            uncovered: &self.zero_rects,
        }
    }
}

pub(super) struct PreparedHtSubBandGroupMember {
    pub(super) band_id: J2kDirectBandId,
    pub(super) offset_elements: usize,
    pub(super) window: J2kRequiredBandRegion,
}
