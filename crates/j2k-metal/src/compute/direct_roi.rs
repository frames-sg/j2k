// SPDX-License-Identifier: MIT OR Apache-2.0

use super::abi::{
    J2kClassicCleanupBatchJob, J2kHtCleanupBatchJob, J2kIdwtSingleDecompositionParams,
    J2kRepeatedIdwtSingleDecompositionParams,
};
use super::direct_prepare::{prepare_classic_sub_band_groups, prepare_ht_sub_band_groups};
use super::{
    copied_slice_buffer, idwt_required_input_windows, idwt_required_output_margin, with_runtime,
    DirectBandSlice, DirectTier1Mode, Error, J2kDirectBandId, J2kDirectIdwtStep,
    J2kDirectStoreStep, J2kRequiredBandRegion, PreparedDirectGrayscalePlan,
    PreparedDirectGrayscaleStep, PreparedDirectIdwt, PreparedHtSubBand, Rect,
};

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CheckedF32Span {
    pub(super) elements: usize,
    pub(super) bytes: usize,
}

#[cfg(target_os = "macos")]
pub(super) fn checked_f32_span(
    width: usize,
    height: usize,
    context: &str,
) -> Result<CheckedF32Span, Error> {
    let elements = width
        .checked_mul(height)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("{context} element count overflow"),
        })?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::MetalKernel {
            message: format!("{context} byte count overflow"),
        })?;
    Ok(CheckedF32Span { elements, bytes })
}

#[cfg(target_os = "macos")]
pub(crate) fn crop_prepared_direct_grayscale_plan_to_output_region(
    plan: &mut PreparedDirectGrayscalePlan,
    region: Rect,
) -> Result<(), Error> {
    if region.w == 0 || region.h == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect region-scaled grayscale plan has an empty output region"
                .to_string(),
        });
    }
    if region.x == 0
        && region.y == 0
        && region.w == plan.dimensions.0
        && region.h == plan.dimensions.1
    {
        return Ok(());
    }

    plan.clear_cpu_tier1_cache()?;
    let mut store_count = 0;
    for step in &mut plan.steps {
        if let PreparedDirectGrayscaleStep::Store(store) = step {
            crop_direct_store_step_to_output_region(store, region)?;
            store_count += 1;
        }
    }

    if store_count == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect grayscale plan has no store step to crop".to_string(),
        });
    }

    prune_prepared_direct_grayscale_plan_to_store_windows(plan)?;
    plan.dimensions = (region.w, region.h);
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) type BandRequiredRegion = J2kRequiredBandRegion;

#[cfg(target_os = "macos")]
pub(super) type BandRequiredRegions = Vec<(J2kDirectBandId, BandRequiredRegion)>;

#[cfg(target_os = "macos")]
struct RoiBandMaps {
    required: BandRequiredRegions,
    idwt_outputs: BandRequiredRegions,
}

#[cfg(target_os = "macos")]
fn required_region(
    regions: &BandRequiredRegions,
    band_id: J2kDirectBandId,
) -> Option<BandRequiredRegion> {
    regions
        .iter()
        .find_map(|(candidate, region)| (*candidate == band_id).then_some(*region))
}

#[cfg(target_os = "macos")]
fn allocate_roi_band_maps(steps: &[PreparedDirectGrayscaleStep]) -> Result<RoiBandMaps, Error> {
    let required_capacity = steps.iter().try_fold(0usize, |total, step| {
        let added = match step {
            PreparedDirectGrayscaleStep::Idwt(_) => 4,
            PreparedDirectGrayscaleStep::Store(_) => 1,
            PreparedDirectGrayscaleStep::ClassicSubBand(_)
            | PreparedDirectGrayscaleStep::HtSubBand(_) => 0,
        };
        crate::batch_allocation::checked_count_sum(
            [total, added],
            "J2K MetalDirect ROI required bands",
        )
    })?;
    let idwt_capacity = steps
        .iter()
        .filter(|step| matches!(step, PreparedDirectGrayscaleStep::Idwt(_)))
        .count();
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K MetalDirect ROI band maps");
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<(J2kDirectBandId, BandRequiredRegion)>(
            required_capacity,
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<(J2kDirectBandId, BandRequiredRegion)>(
            idwt_capacity,
        ),
    ])?;
    Ok(RoiBandMaps {
        required: budget.try_vec(required_capacity, "J2K MetalDirect ROI required bands")?,
        idwt_outputs: budget.try_vec(idwt_capacity, "J2K MetalDirect ROI IDWT output windows")?,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn prune_prepared_direct_grayscale_plan_to_store_windows(
    plan: &mut PreparedDirectGrayscalePlan,
) -> Result<(), Error> {
    let RoiBandMaps {
        mut required,
        mut idwt_outputs,
    } = allocate_roi_band_maps(&plan.steps)?;
    for step in &plan.steps {
        if let PreparedDirectGrayscaleStep::Store(store) = step {
            let source_right = store
                .source_x
                .checked_add(store.copy_width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect ROI source width overflows u32".to_string(),
                })?;
            let source_bottom = store
                .source_y
                .checked_add(store.copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect ROI source height overflows u32".to_string(),
                })?;
            if let Some(region) =
                BandRequiredRegion::new(store.source_x, store.source_y, source_right, source_bottom)
            {
                add_required_region(&mut required, store.input_band_id, region)?;
            }
        }
    }

    for step in plan.steps.iter().rev() {
        if let PreparedDirectGrayscaleStep::Idwt(idwt) = step {
            let Some(output_region) = required_region(&required, idwt.step.output_band_id) else {
                continue;
            };
            // Native ROI uses conservative synthesis-support margins: 16 samples
            // for reversible 5/3 and 40 for irreversible 9/7. Expanding before
            // back-propagation keeps parity/filter support available when a
            // later store crops the final output region.
            let expanded = output_region.expanded_within_band(
                idwt_required_output_margin(idwt.step.transform),
                idwt.step.rect.width(),
                idwt.step.rect.height(),
            );
            set_required_region(&mut idwt_outputs, idwt.step.output_band_id, expanded)?;
            add_idwt_input_required_regions(&mut required, &idwt.step, expanded)?;
        }
    }

    for step in &mut plan.steps {
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let before = sub_band.jobs.len();
                retain_classic_jobs_for_required_region(
                    &mut sub_band.jobs,
                    required_region(&required, sub_band.band_id),
                );
                if sub_band.jobs.len() != before {
                    sub_band.zero_fill = true;
                    if plan.tier1_prepare_mode == DirectTier1Mode::Metal {
                        with_runtime(|runtime| {
                            sub_band.jobs_buffer =
                                copied_slice_buffer(&runtime.device, &sub_band.jobs)?;
                            Ok(())
                        })?;
                    }
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let before = sub_band.jobs.len();
                retain_ht_jobs_for_required_region(
                    &mut sub_band.jobs,
                    required_region(&required, sub_band.band_id),
                );
                if sub_band.jobs.len() != before {
                    compact_ht_sub_band_coded_data(sub_band, plan.tier1_prepare_mode)?;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => {}
        }
    }

    apply_prepared_direct_idwt_output_windows(plan, &idwt_outputs)?;
    plan.classic_groups = prepare_classic_sub_band_groups(&plan.steps, plan.tier1_prepare_mode)?;
    plan.ht_groups = prepare_ht_sub_band_groups(&plan.steps, plan.tier1_prepare_mode)?;
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn apply_prepared_direct_idwt_output_windows(
    plan: &mut PreparedDirectGrayscalePlan,
    windows: &BandRequiredRegions,
) -> Result<(), Error> {
    for step in &mut plan.steps {
        if let PreparedDirectGrayscaleStep::Idwt(idwt) = step {
            idwt.output_window =
                required_region(windows, idwt.step.output_band_id).unwrap_or_else(|| {
                    BandRequiredRegion::full(idwt.step.rect.width(), idwt.step.rect.height())
                });
        }
    }

    for step in &mut plan.steps {
        let PreparedDirectGrayscaleStep::Store(store) = step else {
            continue;
        };
        let Some(window) = required_region(windows, store.input_band_id) else {
            continue;
        };

        store.source_x =
            store
                .source_x
                .checked_sub(window.x0)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect cropped IDWT store source x underflow".to_string(),
                })?;
        store.source_y =
            store
                .source_y
                .checked_sub(window.y0)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect cropped IDWT store source y underflow".to_string(),
                })?;
        store.input_rect = j2k_native::J2kRect {
            x0: store.input_rect.x0.saturating_add(window.x0),
            y0: store.input_rect.y0.saturating_add(window.y0),
            x1: store.input_rect.x0.saturating_add(window.x1),
            y1: store.input_rect.y0.saturating_add(window.y1),
        };
    }

    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct PreparedIdwtInputWindows {
    pub(super) ll: BandRequiredRegion,
    pub(super) hl: BandRequiredRegion,
    pub(super) lh: BandRequiredRegion,
    pub(super) hh: BandRequiredRegion,
}

pub(super) fn idwt_input_windows_from_slices(
    ll: &DirectBandSlice,
    hl: &DirectBandSlice,
    lh: &DirectBandSlice,
    hh: &DirectBandSlice,
) -> PreparedIdwtInputWindows {
    PreparedIdwtInputWindows {
        ll: BandRequiredRegion::full(ll.window.width(), ll.window.height()),
        hl: BandRequiredRegion::full(hl.window.width(), hl.window.height()),
        lh: BandRequiredRegion::full(lh.window.width(), lh.window.height()),
        hh: BandRequiredRegion::full(hh.window.width(), hh.window.height()),
    }
}

#[cfg(target_os = "macos")]
pub(super) fn prepared_idwt_params(
    idwt: &PreparedDirectIdwt,
    inputs: PreparedIdwtInputWindows,
) -> J2kIdwtSingleDecompositionParams {
    J2kIdwtSingleDecompositionParams {
        x0: idwt.step.rect.x0,
        y0: idwt.step.rect.y0,
        output_x: idwt.output_window.x0,
        output_y: idwt.output_window.y0,
        width: idwt.output_window.width(),
        height: idwt.output_window.height(),
        ll_x: inputs.ll.x0,
        ll_y: inputs.ll.y0,
        ll_width: inputs.ll.width(),
        ll_height: inputs.ll.height(),
        hl_x: inputs.hl.x0,
        hl_y: inputs.hl.y0,
        hl_width: inputs.hl.width(),
        hl_height: inputs.hl.height(),
        lh_x: inputs.lh.x0,
        lh_y: inputs.lh.y0,
        lh_width: inputs.lh.width(),
        lh_height: inputs.lh.height(),
        hh_x: inputs.hh.x0,
        hh_y: inputs.hh.y0,
        hh_width: inputs.hh.width(),
        hh_height: inputs.hh.height(),
    }
}

#[cfg(target_os = "macos")]
pub(super) fn repeated_idwt_params(
    idwt: &PreparedDirectIdwt,
    inputs: PreparedIdwtInputWindows,
    strides: PreparedIdwtInputStrides,
    batch_count: usize,
    context: &'static str,
) -> Result<J2kRepeatedIdwtSingleDecompositionParams, Error> {
    Ok(J2kRepeatedIdwtSingleDecompositionParams {
        x0: idwt.step.rect.x0,
        y0: idwt.step.rect.y0,
        output_x: idwt.output_window.x0,
        output_y: idwt.output_window.y0,
        width: idwt.output_window.width(),
        height: idwt.output_window.height(),
        ll_x: inputs.ll.x0,
        ll_y: inputs.ll.y0,
        ll_width: inputs.ll.width(),
        ll_height: inputs.ll.height(),
        hl_x: inputs.hl.x0,
        hl_y: inputs.hl.y0,
        hl_width: inputs.hl.width(),
        hl_height: inputs.hl.height(),
        lh_x: inputs.lh.x0,
        lh_y: inputs.lh.y0,
        lh_width: inputs.lh.width(),
        lh_height: inputs.lh.height(),
        hh_x: inputs.hh.x0,
        hh_y: inputs.hh.y0,
        hh_width: inputs.hh.width(),
        hh_height: inputs.hh.height(),
        ll_instance_stride: strides.ll,
        hl_instance_stride: strides.hl,
        lh_instance_stride: strides.lh,
        hh_instance_stride: strides.hh,
        batch_count: u32::try_from(batch_count).map_err(|_| Error::MetalKernel {
            message: format!("J2K MetalDirect {context} IDWT batch count exceeds u32"),
        })?,
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct PreparedIdwtInputStrides {
    pub(super) ll: u32,
    pub(super) hl: u32,
    pub(super) lh: u32,
    pub(super) hh: u32,
}

#[cfg(target_os = "macos")]
pub(super) fn prepared_idwt_output_len(idwt: &PreparedDirectIdwt) -> Result<usize, Error> {
    checked_f32_span(
        idwt.output_window.width() as usize,
        idwt.output_window.height() as usize,
        "J2K MetalDirect IDWT output",
    )
    .map(|span| span.elements)
}

#[cfg(target_os = "macos")]
pub(super) fn add_required_region(
    required: &mut BandRequiredRegions,
    band_id: J2kDirectBandId,
    region: BandRequiredRegion,
) -> Result<(), Error> {
    if let Some((_, existing)) = required
        .iter_mut()
        .find(|(candidate, _)| *candidate == band_id)
    {
        *existing = existing.union(region);
        return Ok(());
    }
    crate::batch_allocation::try_reserve_for_push(required, "J2K MetalDirect ROI required bands")?;
    required.push((band_id, region));
    Ok(())
}

#[cfg(target_os = "macos")]
fn set_required_region(
    regions: &mut BandRequiredRegions,
    band_id: J2kDirectBandId,
    region: BandRequiredRegion,
) -> Result<(), Error> {
    if let Some((_, existing)) = regions
        .iter_mut()
        .find(|(candidate, _)| *candidate == band_id)
    {
        *existing = region;
        return Ok(());
    }
    crate::batch_allocation::try_reserve_for_push(
        regions,
        "J2K MetalDirect ROI IDWT output windows",
    )?;
    regions.push((band_id, region));
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn add_idwt_input_required_regions(
    required: &mut BandRequiredRegions,
    idwt: &J2kDirectIdwtStep,
    output_region: BandRequiredRegion,
) -> Result<(), Error> {
    let windows = idwt_required_input_windows(idwt, output_region);
    add_required_region(required, idwt.ll_band_id, windows.ll)?;
    add_required_region(required, idwt.hl_band_id, windows.hl)?;
    add_required_region(required, idwt.lh_band_id, windows.lh)?;
    add_required_region(required, idwt.hh_band_id, windows.hh)
}

#[cfg(target_os = "macos")]
pub(super) trait RequiredRegionJob {
    fn output_offset(&self) -> u32;
    fn output_stride(&self) -> u32;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}

#[cfg(target_os = "macos")]
impl RequiredRegionJob for J2kClassicCleanupBatchJob {
    fn output_offset(&self) -> u32 {
        self.output_offset
    }

    fn output_stride(&self) -> u32 {
        self.output_stride
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}

#[cfg(target_os = "macos")]
impl RequiredRegionJob for J2kHtCleanupBatchJob {
    fn output_offset(&self) -> u32 {
        self.output_offset
    }

    fn output_stride(&self) -> u32 {
        self.output_stride
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}

#[cfg(target_os = "macos")]
pub(super) fn retain_jobs_for_required_region<J: RequiredRegionJob>(
    jobs: &mut Vec<J>,
    required: Option<BandRequiredRegion>,
) {
    let Some(required) = required else {
        jobs.clear();
        return;
    };
    jobs.retain(|job| {
        let output_x = job.output_offset() % job.output_stride();
        let output_y = job.output_offset() / job.output_stride();
        required.intersects(output_x, output_y, job.width(), job.height())
    });
}

#[cfg(target_os = "macos")]
pub(super) fn retain_classic_jobs_for_required_region(
    jobs: &mut Vec<J2kClassicCleanupBatchJob>,
    required: Option<BandRequiredRegion>,
) {
    retain_jobs_for_required_region(jobs, required);
}

#[cfg(target_os = "macos")]
pub(super) fn retain_ht_jobs_for_required_region(
    jobs: &mut Vec<J2kHtCleanupBatchJob>,
    required: Option<BandRequiredRegion>,
) {
    retain_jobs_for_required_region(jobs, required);
}

#[cfg(target_os = "macos")]
pub(super) fn compact_ht_sub_band_coded_data(
    sub_band: &mut PreparedHtSubBand,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<(), Error> {
    let coded_data =
        sub_band
            .payload_source
            .contiguous_mut()
            .ok_or(Error::MetalStateInvariant {
                state: "HTJ2K MetalDirect cropped plan",
                reason: "post-prepare ROI compaction requires a legacy contiguous payload plan",
            })?;
    let compacted_len = crate::batch_allocation::checked_count_sum(
        sub_band.jobs.iter().map(|job| job.coded_len as usize),
        "HTJ2K MetalDirect cropped coded payload",
    )?;
    u32::try_from(compacted_len).map_err(|_| Error::MetalKernel {
        message: "HTJ2K MetalDirect cropped coded payload exceeds u32".to_string(),
    })?;
    for job in &sub_band.jobs {
        let start = job.coded_offset as usize;
        let end = start
            .checked_add(job.coded_len as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect cropped coded payload range overflow".to_string(),
            })?;
        if end > coded_data.len() {
            return Err(Error::MetalKernel {
                message: "HTJ2K MetalDirect cropped coded payload range out of bounds".to_string(),
            });
        }
    }
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K MetalDirect cropped coded payload",
    );
    let mut compacted = budget.try_vec(compacted_len, "HTJ2K MetalDirect cropped coded payload")?;
    let previous = std::mem::take(coded_data);

    for job in &mut sub_band.jobs {
        let start = job.coded_offset as usize;
        let len = job.coded_len as usize;
        let end = start + len;
        job.coded_offset = u32::try_from(compacted.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect cropped coded payload exceeds u32".to_string(),
        })?;
        compacted.extend_from_slice(&previous[start..end]);
    }

    *coded_data = compacted;
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn checked_rect_end(origin: u32, length: u32, label: &str) -> Result<u32, Error> {
    origin
        .checked_add(length)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("J2K MetalDirect region-scaled {label} overflows u32"),
        })
}

#[cfg(target_os = "macos")]
pub(super) fn crop_direct_store_step_to_output_region(
    store: &mut J2kDirectStoreStep,
    region: Rect,
) -> Result<(), Error> {
    let store_bounds = (
        store.output_x,
        store.output_y,
        checked_rect_end(store.output_x, store.copy_width, "store width")?,
        checked_rect_end(store.output_y, store.copy_height, "store height")?,
    );
    let region_bounds = (
        region.x,
        region.y,
        checked_rect_end(region.x, region.w, "ROI width")?,
        checked_rect_end(region.y, region.h, "ROI height")?,
    );
    let intersection = (
        store_bounds.0.max(region_bounds.0),
        store_bounds.1.max(region_bounds.1),
        store_bounds.2.min(region_bounds.2),
        store_bounds.3.min(region_bounds.3),
    );
    if intersection.0 >= intersection.2 || intersection.1 >= intersection.3 {
        return Err(Error::MetalKernel {
            message:
                "J2K MetalDirect region-scaled ROI does not intersect the decoded store window"
                    .to_string(),
        });
    }

    let source_delta = (
        intersection.0 - store_bounds.0,
        intersection.1 - store_bounds.1,
    );
    store.source_x =
        store
            .source_x
            .checked_add(source_delta.0)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect region-scaled source x overflows u32".to_string(),
            })?;
    store.source_y =
        store
            .source_y
            .checked_add(source_delta.1)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect region-scaled source y overflows u32".to_string(),
            })?;
    store.copy_width = intersection.2 - intersection.0;
    store.copy_height = intersection.3 - intersection.1;
    store.output_width = region.w;
    store.output_height = region.h;
    store.output_x = intersection.0 - region_bounds.0;
    store.output_y = intersection.1 - region_bounds.1;
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod span_tests {
    use std::mem::size_of;

    use super::*;

    #[test]
    fn non_stacked_f32_span_accepts_exact_byte_boundary_and_rejects_one_over() {
        let exact_elements = usize::MAX / size_of::<f32>();
        let exact = checked_f32_span(exact_elements, 1, "J2K MetalDirect non-stacked test span")
            .expect("largest exactly representable f32 span");
        assert_eq!(exact.elements, exact_elements);
        assert_eq!(exact.bytes, exact_elements * size_of::<f32>());

        assert!(matches!(
            checked_f32_span(
                exact_elements + 1,
                1,
                "J2K MetalDirect non-stacked test span",
            ),
            Err(Error::MetalKernel { message }) if message.contains("byte count overflow")
        ));
        assert!(matches!(
            checked_f32_span(
                usize::MAX,
                2,
                "J2K MetalDirect non-stacked test span",
            ),
            Err(Error::MetalKernel { message }) if message.contains("element count overflow")
        ));
    }
}
