use alloc::vec::Vec;
use core::ops::Range;

use crate::error::{bail, DecodingError, Result};
use crate::j2c::idwt;
use crate::math::{floor_f32, round_f32};
use crate::{
    decode_ht_code_block_scalar_with_workspace, decode_j2k_code_block_scalar_with_workspace,
    try_resize_decode_elements, HtCodeBlockDecodeJob, HtCodeBlockDecodeWorkspace,
    HtOwnedSubBandPlan, J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace, J2kDirectBandId,
    J2kDirectColorPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep,
    J2kDirectStoreStep, J2kIdwtBand, J2kOwnedSubBandPlan, J2kRect, J2kSingleDecompositionIdwtJob,
    J2kWaveletTransform,
};

mod allocation;
use allocation::{prepare_direct_scratch, DirectWorkspaceBudget};
mod color;
use color::apply_inverse_mct_region;
pub use color::{execute_direct_color_plan_rgb8_into, execute_direct_color_plan_rgba8_into};
mod component;
use component::{
    checked_area, checked_sub_band_job_output_range, execute_component_plan, execute_idwt_step,
    prepare_sub_band_output, resize_and_zero, store_component, SubBandJobOutputRange,
};
mod referenced;
pub use referenced::{
    execute_referenced_htj2k_plan, execute_referenced_htj2k_plan_from_payloads,
    J2kDirectDecodedComponents, J2kDirectDecodedPlane,
};
mod referenced_staged;
pub use referenced_staged::{
    execute_referenced_classic_entropy_job, execute_referenced_htj2k_entropy_job,
    finish_referenced_classic_staged, finish_referenced_classic_tile_staged,
    finish_referenced_htj2k_staged, finish_referenced_htj2k_tile_staged,
    prepare_referenced_classic_entropy_workspace, prepare_referenced_classic_staged,
    prepare_referenced_classic_tile_staged, prepare_referenced_htj2k_entropy_workspace,
    prepare_referenced_htj2k_staged, prepare_referenced_htj2k_tile_staged, J2kDirectCodeBlockIndex,
    J2kDirectCpuEntropyWorkspace,
};
mod referenced_classic;
pub use referenced_classic::{
    execute_referenced_classic_plan, execute_referenced_classic_plan_from_payloads,
};

/// Adapter reusable scratch for executing direct J2K RGB plans on the CPU.
#[derive(Debug, Default)]
pub struct J2kDirectCpuScratch {
    component_band_sets: Vec<DirectComponentBandScratch>,
    component_planes: Vec<DirectComponentPlane>,
    compressed_payload: Vec<u8>,
    classic_workspace: J2kCodeBlockDecodeWorkspace,
    ht_workspace: HtCodeBlockDecodeWorkspace,
    staged_state: Option<StagedDirectState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StagedDirectRoute {
    Classic,
    Htj2k,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StagedDirectState {
    route: StagedDirectRoute,
    next_tile: usize,
    active_tile: Option<usize>,
    tile_count: usize,
}

impl J2kDirectCpuScratch {
    /// Create empty direct-plan CPU scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            component_band_sets: Vec::new(),
            component_planes: Vec::new(),
            compressed_payload: Vec::new(),
            classic_workspace: J2kCodeBlockDecodeWorkspace::empty(),
            ht_workspace: HtCodeBlockDecodeWorkspace::empty(),
            staged_state: None,
        }
    }

    /// Release retained scratch allocations.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Bytes retained by the parse-free HT code-block workspace.
    #[doc(hidden)]
    #[must_use]
    pub fn retained_ht_workspace_bytes(&self) -> usize {
        self.ht_workspace.allocated_bytes().unwrap_or(usize::MAX)
    }

    /// Bytes retained by the parse-free classic code-block workspace.
    #[doc(hidden)]
    #[must_use]
    pub fn retained_classic_workspace_bytes(&self) -> usize {
        self.classic_workspace
            .allocated_bytes()
            .unwrap_or(usize::MAX)
    }

    /// Number of coefficient/IDWT buffer owners retained by this scratch.
    ///
    /// Referenced multi-tile execution retains the maximum live tile shape,
    /// not the sum of every tile's bands.
    #[doc(hidden)]
    #[must_use]
    pub fn retained_band_owner_count(&self) -> usize {
        self.component_band_sets
            .iter()
            .map(|component| component.bands.len())
            .sum()
    }

    /// Prepare retained scratch for `plan` and report the actual-capacity peak
    /// allocation for one execution, including the retained plan and temporary
    /// scalar code-block workspace.
    ///
    /// This is an adapter accounting hook used to admit concurrent direct CPU
    /// executions without treating every small ROI as a worst-case full-frame
    /// native decode.
    #[doc(hidden)]
    pub fn prepare_execution_allocation_bytes(
        &mut self,
        plan: &J2kDirectColorPlan,
    ) -> Result<usize> {
        prepare_direct_scratch(plan, self).map(DirectWorkspaceBudget::peak_bytes)
    }

    #[cfg(test)]
    fn allocation_profile_for_tests(&self) -> DirectScratchAllocationProfile {
        let band_buffers = self
            .component_band_sets
            .iter()
            .map(|component| component.bands.len())
            .sum();
        let band_sample_len = self
            .component_band_sets
            .iter()
            .flat_map(|component| component.bands.iter())
            .map(|band| band.coefficients.len())
            .sum();
        let band_sample_capacity = self
            .component_band_sets
            .iter()
            .flat_map(|component| component.bands.iter())
            .map(|band| band.coefficients.capacity())
            .sum();
        let component_sample_len = self
            .component_planes
            .iter()
            .map(|plane| plane.samples.len())
            .sum();
        let component_sample_capacity = self
            .component_planes
            .iter()
            .map(|plane| plane.samples.capacity())
            .sum();
        DirectScratchAllocationProfile {
            component_band_sets: self.component_band_sets.len(),
            component_planes: self.component_planes.len(),
            band_buffers,
            band_sample_len,
            band_sample_capacity,
            component_sample_len,
            component_sample_capacity,
            compressed_payload_capacity: self.compressed_payload.capacity(),
            classic_workspace_bytes: self.retained_classic_workspace_bytes(),
            ht_workspace_bytes: self.retained_ht_workspace_bytes(),
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectScratchAllocationProfile {
    component_band_sets: usize,
    component_planes: usize,
    band_buffers: usize,
    band_sample_len: usize,
    band_sample_capacity: usize,
    component_sample_len: usize,
    component_sample_capacity: usize,
    compressed_payload_capacity: usize,
    classic_workspace_bytes: usize,
    ht_workspace_bytes: usize,
}

#[derive(Debug, Default)]
struct DirectComponentBandScratch {
    bands: Vec<DirectCpuBand>,
    active_len: usize,
}

impl DirectComponentBandScratch {
    fn reset(&mut self) {
        self.active_len = 0;
    }

    fn active(&self) -> &[DirectCpuBand] {
        &self.bands[..self.active_len]
    }

    fn prepare_band(
        &mut self,
        band_id: J2kDirectBandId,
        rect: J2kRect,
        len: usize,
    ) -> Result<usize> {
        let index = self.active_len;
        if index == self.bands.len() {
            return Err(DecodingError::HostAllocationFailed.into());
        }
        let band = &mut self.bands[index];
        band.band_id = band_id;
        band.rect = rect;
        resize_and_zero(&mut band.coefficients, len)?;
        self.active_len += 1;
        Ok(index)
    }
}

#[derive(Debug)]
struct DirectCpuBand {
    band_id: J2kDirectBandId,
    rect: J2kRect,
    coefficients: Vec<f32>,
}

impl DirectCpuBand {
    const fn empty() -> Self {
        Self {
            band_id: 0,
            rect: J2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            coefficients: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
struct DirectComponentPlane {
    width: u32,
    height: u32,
    samples: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image};
    use alloc::vec;

    const EMPTY_DIRECT_CPU_SCRATCH: J2kDirectCpuScratch = J2kDirectCpuScratch::new();

    #[test]
    fn direct_cpu_scratch_constructor_remains_const() {
        let scratch = EMPTY_DIRECT_CPU_SCRATCH;
        assert_eq!(
            scratch.allocation_profile_for_tests(),
            DirectScratchAllocationProfile {
                component_band_sets: 0,
                component_planes: 0,
                band_buffers: 0,
                band_sample_len: 0,
                band_sample_capacity: 0,
                component_sample_len: 0,
                component_sample_capacity: 0,
                compressed_payload_capacity: 0,
                classic_workspace_bytes: 0,
                ht_workspace_bytes: 0,
            }
        );
    }

    fn direct_htj2k_rgb_plan() -> (J2kDirectColorPlan, J2kRect) {
        let pixels = (0..16 * 16 * 3)
            .map(|idx| {
                u8::try_from((idx * 13 + idx / 3) & 0xff)
                    .expect("test pattern is masked to one byte")
            })
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 16, 16, 3, 8, false, &options).expect("encode HTJ2K RGB");
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((4, 4)),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled image");
        let output_region = J2kRect {
            x0: 1,
            y0: 1,
            x1: 3,
            y1: 3,
        };
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_color_plan_region_with_context(&mut context, (1, 1, 2, 2))
            .expect("direct color plan");
        (plan, output_region)
    }

    #[test]
    fn direct_cpu_scratch_retains_component_buffers_between_executions() {
        let (plan, output_region) = direct_htj2k_rgb_plan();
        let stride = output_region.width() as usize * 3;
        let mut out = vec![0_u8; stride * output_region.height() as usize];
        let mut scratch = J2kDirectCpuScratch::new();

        execute_direct_color_plan_rgb8_into(&plan, output_region, &mut scratch, &mut out, stride)
            .expect("first direct execute");
        let first = scratch.allocation_profile_for_tests();

        execute_direct_color_plan_rgb8_into(&plan, output_region, &mut scratch, &mut out, stride)
            .expect("second direct execute");
        let second = scratch.allocation_profile_for_tests();

        assert_eq!(first.component_band_sets, 3);
        assert_eq!(first.component_planes, 3);
        assert_eq!(second.component_band_sets, first.component_band_sets);
        assert_eq!(second.component_planes, first.component_planes);
        assert_eq!(second.band_buffers, first.band_buffers);
        assert_eq!(
            second.component_sample_capacity,
            first.component_sample_capacity
        );
        assert_eq!(second.band_sample_capacity, first.band_sample_capacity);
        assert!(second.band_sample_capacity >= second.band_sample_len);
        assert!(second.component_sample_capacity >= second.component_sample_len);
    }

    #[test]
    fn direct_cpu_scratch_clear_releases_every_retained_owner() {
        let (plan, output_region) = direct_htj2k_rgb_plan();
        let stride = output_region.width() as usize * 3;
        let mut out = vec![0_u8; stride * output_region.height() as usize];
        let mut scratch = J2kDirectCpuScratch::new();
        execute_direct_color_plan_rgb8_into(&plan, output_region, &mut scratch, &mut out, stride)
            .expect("populate direct scratch");

        scratch.clear();

        assert_eq!(
            scratch.allocation_profile_for_tests(),
            DirectScratchAllocationProfile {
                component_band_sets: 0,
                component_planes: 0,
                band_buffers: 0,
                band_sample_len: 0,
                band_sample_capacity: 0,
                component_sample_len: 0,
                component_sample_capacity: 0,
                compressed_payload_capacity: 0,
                classic_workspace_bytes: 0,
                ht_workspace_bytes: 0,
            }
        );
        assert_eq!(scratch.component_band_sets.capacity(), 0);
        assert_eq!(scratch.component_planes.capacity(), 0);
    }

    #[test]
    fn prepared_direct_execution_reports_its_actual_peak_allocation() {
        let (plan, _) = direct_htj2k_rgb_plan();
        let plan_bytes = plan.retained_allocation_bytes().expect("plan bytes");
        let mut scratch = J2kDirectCpuScratch::new();

        let peak = scratch
            .prepare_execution_allocation_bytes(&plan)
            .expect("prepare direct execution");

        assert!(peak >= plan_bytes);
        assert!(peak <= crate::DEFAULT_MAX_DECODE_BYTES);
        assert_eq!(
            scratch
                .prepare_execution_allocation_bytes(&plan)
                .expect("re-prepare retained direct execution"),
            peak
        );
    }

    #[test]
    fn direct_cpu_rejects_aggregate_planes_before_allocating_scratch() {
        let (mut plan, _) = direct_htj2k_rgb_plan();
        for component in &mut plan.component_plans {
            for step in &mut component.steps {
                if let J2kDirectGrayscaleStep::Store(store) = step {
                    store.output_width = 60_000;
                    store.output_height = 60_000;
                }
            }
        }
        let mut scratch = J2kDirectCpuScratch::new();

        assert_eq!(
            prepare_direct_scratch(&plan, &mut scratch),
            Err(crate::DecodeError::Validation(
                crate::ValidationError::ImageTooLarge
            ))
        );
        assert_eq!(scratch.component_band_sets.capacity(), 0);
        assert_eq!(scratch.component_planes.capacity(), 0);
    }
}
