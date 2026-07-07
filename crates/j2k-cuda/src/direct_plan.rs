use std::collections::HashMap;

use j2k_core::PixelFormat;
use j2k_native::{
    idwt_required_input_windows, idwt_required_output_margin, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep, J2kRect,
    J2kRequiredBandRegion as RequiredBandRegion, J2kWaveletTransform,
};

use crate::Error;

const CLASSIC_J2K_NOT_CUDA_HTJ2K: &str =
    "strict CUDA codestream decode only accepts HTJ2K direct-plan subbands";
const EMPTY_HTJ2K_PLAN: &str = "strict CUDA HTJ2K plan contains no HT code blocks";
const MIXED_TRANSFORMS_UNSUPPORTED: &str = "strict CUDA HTJ2K plan contains mixed DWT transforms";
const PLAN_PAYLOAD_TOO_LARGE: &str = "strict CUDA HTJ2K plan payload is too large";
const PLAN_BLOCK_LENGTH_MISMATCH: &str =
    "strict CUDA HTJ2K plan block lengths do not match payload bytes";
const PLAN_OUTPUT_RECT_MISMATCH: &str =
    "strict CUDA HTJ2K plan store does not fit the requested output rectangle";
const ROI_MAXSHIFT_UNSUPPORTED: &str =
    "strict CUDA HTJ2K plan does not support ROI maxshift decode";

/// CUDA-side DWT transform selector for a flat HTJ2K plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum CudaHtj2kTransform {
    /// Reversible 5/3 transform.
    Reversible53,
    /// Irreversible 9/7 transform.
    Irreversible97,
}

/// Stable CUDA-side identifier for a direct-plan coefficient band.
pub(crate) type CudaHtj2kBandId = u32;

impl CudaHtj2kTransform {
    pub(crate) fn from_native(value: J2kWaveletTransform) -> Self {
        match value {
            J2kWaveletTransform::Reversible53 => Self::Reversible53,
            J2kWaveletTransform::Irreversible97 => Self::Irreversible97,
        }
    }
}

/// Flat POD HTJ2K code-block metadata consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct CudaHtj2kCodeBlock {
    /// Index of the parent sub-band in [`CudaHtj2kDecodePlan::subbands`].
    pub(crate) subband_index: u32,
    /// Byte offset into [`CudaHtj2kDecodePlan::payload`].
    pub(crate) payload_offset: u64,
    /// Total payload byte length for this code block.
    pub(crate) payload_len: u32,
    /// Cleanup segment length in bytes.
    pub(crate) cleanup_length: u32,
    /// Refinement segment length in bytes.
    pub(crate) refinement_length: u32,
    /// X offset within the target sub-band coefficient buffer.
    pub(crate) output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub(crate) output_y: u32,
    /// Code-block width in samples.
    pub(crate) width: u32,
    /// Code-block height in samples.
    pub(crate) height: u32,
    /// Output row stride, in samples.
    pub(crate) output_stride: u32,
    /// Missing most-significant bit planes.
    pub(crate) missing_bit_planes: u8,
    /// Number of coding passes present.
    pub(crate) number_of_coding_passes: u8,
    /// Total coded bitplanes for the parent sub-band.
    pub(crate) num_bitplanes: u8,
    /// Nonzero when vertically causal context was enabled.
    pub(crate) stripe_causal: u8,
    /// Dequantization step to apply to decoded coefficients.
    pub(crate) dequantization_step: f32,
}

/// Flat POD sub-band geometry consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct CudaHtj2kSubband {
    /// Stable CUDA direct-plan band id.
    pub(crate) band_id: CudaHtj2kBandId,
    /// Absolute x0 coordinate in component space.
    pub(crate) x0: u32,
    /// Absolute y0 coordinate in component space.
    pub(crate) y0: u32,
    /// Absolute x1 coordinate in component space.
    pub(crate) x1: u32,
    /// Absolute y1 coordinate in component space.
    pub(crate) y1: u32,
    /// Sub-band width in samples.
    pub(crate) width: u32,
    /// Sub-band height in samples.
    pub(crate) height: u32,
    /// First code-block index for this sub-band.
    pub(crate) code_block_start: u32,
    /// Number of code blocks for this sub-band.
    pub(crate) code_block_count: u32,
}

/// Flat POD IDWT step consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct CudaHtj2kIdwtStep {
    /// Stable identifier of the output coefficient band produced by this step.
    pub(crate) output_band_id: CudaHtj2kBandId,
    /// DWT transform to apply.
    pub(crate) transform: CudaHtj2kTransform,
    /// Output rectangle.
    pub(crate) rect: CudaHtj2kRect,
    /// LL input band id.
    pub(crate) ll_band_id: CudaHtj2kBandId,
    /// LL input rectangle.
    pub(crate) ll_rect: CudaHtj2kRect,
    /// HL input band id.
    pub(crate) hl_band_id: CudaHtj2kBandId,
    /// HL input rectangle.
    pub(crate) hl_rect: CudaHtj2kRect,
    /// LH input band id.
    pub(crate) lh_band_id: CudaHtj2kBandId,
    /// LH input rectangle.
    pub(crate) lh_rect: CudaHtj2kRect,
    /// HH input band id.
    pub(crate) hh_band_id: CudaHtj2kBandId,
    /// HH input rectangle.
    pub(crate) hh_rect: CudaHtj2kRect,
}

/// Flat POD store step consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct CudaHtj2kStoreStep {
    /// Stable identifier of the input coefficient band.
    pub(crate) input_band_id: CudaHtj2kBandId,
    /// Source rectangle.
    pub(crate) input_rect: CudaHtj2kRect,
    /// Source x offset.
    pub(crate) source_x: u32,
    /// Source y offset.
    pub(crate) source_y: u32,
    /// Number of samples copied per row.
    pub(crate) copy_width: u32,
    /// Number of rows copied.
    pub(crate) copy_height: u32,
    /// Destination row width.
    pub(crate) output_width: u32,
    /// Destination height.
    pub(crate) output_height: u32,
    /// Destination x offset.
    pub(crate) output_x: u32,
    /// Destination y offset.
    pub(crate) output_y: u32,
    /// Constant level-shift addend.
    pub(crate) addend: f32,
}

/// Flat POD rectangle used inside CUDA HTJ2K plan metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct CudaHtj2kRect {
    /// Inclusive left coordinate.
    pub(crate) x0: u32,
    /// Inclusive top coordinate.
    pub(crate) y0: u32,
    /// Exclusive right coordinate.
    pub(crate) x1: u32,
    /// Exclusive bottom coordinate.
    pub(crate) y1: u32,
}

/// Flat CUDA HTJ2K decode plan.
#[derive(Debug, Clone)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct CudaHtj2kDecodePlan {
    dimensions: (u32, u32),
    bit_depth: u8,
    output_format: PixelFormat,
    output_origin: (u32, u32),
    transform: CudaHtj2kTransform,
    payload: Vec<u8>,
    code_blocks: Vec<CudaHtj2kCodeBlock>,
    subbands: Vec<CudaHtj2kSubband>,
    idwt_steps: Vec<CudaHtj2kIdwtStep>,
    store_steps: Vec<CudaHtj2kStoreStep>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl CudaHtj2kDecodePlan {
    pub(crate) fn from_grayscale_direct_plan(
        plan: &J2kDirectGrayscalePlan,
        output_format: PixelFormat,
        output_origin: (u32, u32),
    ) -> Result<Self, Error> {
        Self::from_grayscale_direct_plan_region(plan, output_format, output_origin, plan.dimensions)
    }

    pub(crate) fn from_grayscale_direct_plan_region(
        plan: &J2kDirectGrayscalePlan,
        output_format: PixelFormat,
        output_origin: (u32, u32),
        output_dimensions: (u32, u32),
    ) -> Result<Self, Error> {
        let capacity_hint = cuda_plan_capacity_hint(plan)?;
        let mut payload = Vec::with_capacity(capacity_hint.payload_bytes);
        let mut code_blocks = Vec::with_capacity(capacity_hint.code_blocks);
        let mut subbands = Vec::with_capacity(capacity_hint.subbands);
        let mut idwt_steps = Vec::with_capacity(capacity_hint.idwt_steps);
        let mut store_steps = Vec::with_capacity(capacity_hint.store_steps);
        let mut transform = None;
        let mut saw_classic = false;
        let required_regions = if output_origin == (0, 0) && output_dimensions == plan.dimensions {
            None
        } else {
            Some(required_regions_for_direct_plan(plan)?)
        };

        for step in &plan.steps {
            match step {
                J2kDirectGrayscaleStep::HtSubBand(subband) => {
                    let subband_index = u32::try_from(subbands.len()).map_err(|_| {
                        Error::UnsupportedCudaRequest {
                            reason: PLAN_PAYLOAD_TOO_LARGE,
                        }
                    })?;
                    let code_block_start = u32::try_from(code_blocks.len()).map_err(|_| {
                        Error::UnsupportedCudaRequest {
                            reason: PLAN_PAYLOAD_TOO_LARGE,
                        }
                    })?;
                    for job in &subband.jobs {
                        let payload_offset = u64::try_from(payload.len()).map_err(|_| {
                            Error::UnsupportedCudaRequest {
                                reason: PLAN_PAYLOAD_TOO_LARGE,
                            }
                        })?;
                        let payload_len = u32::try_from(job.data.len()).map_err(|_| {
                            Error::UnsupportedCudaRequest {
                                reason: PLAN_PAYLOAD_TOO_LARGE,
                            }
                        })?;
                        let expected_len = job
                            .cleanup_length
                            .checked_add(job.refinement_length)
                            .ok_or(Error::UnsupportedCudaRequest {
                                reason: PLAN_BLOCK_LENGTH_MISMATCH,
                            })?;
                        if expected_len != payload_len {
                            return Err(Error::UnsupportedCudaRequest {
                                reason: PLAN_BLOCK_LENGTH_MISMATCH,
                            });
                        }
                        let output_stride = u32::try_from(job.output_stride).map_err(|_| {
                            Error::UnsupportedCudaRequest {
                                reason: PLAN_PAYLOAD_TOO_LARGE,
                            }
                        })?;
                        if let Some(required_regions) = &required_regions {
                            if !required_regions
                                .get(&subband.band_id)
                                .is_some_and(|required| {
                                    required.intersects(
                                        job.output_x,
                                        job.output_y,
                                        job.width,
                                        job.height,
                                    )
                                })
                            {
                                continue;
                            }
                        }
                        if job.roi_shift != 0 {
                            return Err(Error::UnsupportedCudaRequest {
                                reason: ROI_MAXSHIFT_UNSUPPORTED,
                            });
                        }
                        payload.extend_from_slice(&job.data);
                        code_blocks.push(CudaHtj2kCodeBlock {
                            subband_index,
                            payload_offset,
                            payload_len,
                            cleanup_length: job.cleanup_length,
                            refinement_length: job.refinement_length,
                            output_x: job.output_x,
                            output_y: job.output_y,
                            width: job.width,
                            height: job.height,
                            output_stride,
                            missing_bit_planes: job.missing_bit_planes,
                            number_of_coding_passes: job.number_of_coding_passes,
                            num_bitplanes: job.num_bitplanes,
                            stripe_causal: u8::from(job.stripe_causal),
                            dequantization_step: job.dequantization_step,
                        });
                    }
                    let code_block_count = u32::try_from(
                        code_blocks.len() - code_block_start as usize,
                    )
                    .map_err(|_| Error::UnsupportedCudaRequest {
                        reason: PLAN_PAYLOAD_TOO_LARGE,
                    })?;
                    subbands.push(CudaHtj2kSubband {
                        band_id: subband.band_id,
                        x0: subband.rect.x0,
                        y0: subband.rect.y0,
                        x1: subband.rect.x1,
                        y1: subband.rect.y1,
                        width: subband.width,
                        height: subband.height,
                        code_block_start,
                        code_block_count,
                    });
                }
                J2kDirectGrayscaleStep::ClassicSubBand(_) => saw_classic = true,
                J2kDirectGrayscaleStep::Idwt(step) => {
                    let step_transform = CudaHtj2kTransform::from_native(step.transform);
                    match transform {
                        Some(existing) if existing != step_transform => {
                            return Err(Error::UnsupportedCudaRequest {
                                reason: MIXED_TRANSFORMS_UNSUPPORTED,
                            });
                        }
                        Some(_) => {}
                        None => transform = Some(step_transform),
                    }
                    idwt_steps.push(convert_idwt_step(*step));
                }
                J2kDirectGrayscaleStep::Store(step) => {
                    store_steps.push(convert_store_step(*step, output_origin, output_dimensions)?);
                }
            }
        }

        if saw_classic {
            return Err(Error::UnsupportedCudaRequest {
                reason: CLASSIC_J2K_NOT_CUDA_HTJ2K,
            });
        }
        if code_blocks.is_empty() {
            return Err(Error::UnsupportedCudaRequest {
                reason: EMPTY_HTJ2K_PLAN,
            });
        }

        Ok(Self {
            dimensions: output_dimensions,
            bit_depth: plan.bit_depth,
            output_format,
            output_origin,
            transform: transform.unwrap_or(CudaHtj2kTransform::Reversible53),
            payload,
            code_blocks,
            subbands,
            idwt_steps,
            store_steps,
        })
    }

    /// Output dimensions of the decoded surface.
    pub(crate) fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Source component bit depth.
    pub(crate) fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// Output pixel format requested by the caller.
    pub(crate) fn output_format(&self) -> PixelFormat {
        self.output_format
    }

    /// Destination origin in the caller-visible output surface.
    pub(crate) fn output_origin(&self) -> (u32, u32) {
        self.output_origin
    }

    /// DWT transform used by IDWT kernels.
    pub(crate) fn transform(&self) -> CudaHtj2kTransform {
        self.transform
    }

    /// Contiguous cleanup/refinement payload bytes.
    pub(crate) fn payload(&self) -> &[u8] {
        &self.payload
    }

    #[cfg_attr(not(feature = "cuda-runtime"), allow(dead_code))]
    pub(crate) fn append_payload_to_shared(
        &mut self,
        shared_payload: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let base =
            u64::try_from(shared_payload.len()).map_err(|_| Error::UnsupportedCudaRequest {
                reason: PLAN_PAYLOAD_TOO_LARGE,
            })?;
        shared_payload
            .try_reserve(self.payload.len())
            .map_err(|_| Error::UnsupportedCudaRequest {
                reason: PLAN_PAYLOAD_TOO_LARGE,
            })?;
        for block in &mut self.code_blocks {
            block.payload_offset =
                block
                    .payload_offset
                    .checked_add(base)
                    .ok_or(Error::UnsupportedCudaRequest {
                        reason: PLAN_PAYLOAD_TOO_LARGE,
                    })?;
        }
        shared_payload.append(&mut self.payload);
        Ok(())
    }

    #[cfg_attr(not(feature = "cuda-runtime"), allow(dead_code))]
    pub(crate) fn rebase_payload_offsets(&mut self, base: u64) -> Result<(), Error> {
        for block in &mut self.code_blocks {
            block.payload_offset =
                block
                    .payload_offset
                    .checked_add(base)
                    .ok_or(Error::UnsupportedCudaRequest {
                        reason: PLAN_PAYLOAD_TOO_LARGE,
                    })?;
        }
        Ok(())
    }

    /// Flat code-block metadata.
    pub(crate) fn code_blocks(&self) -> &[CudaHtj2kCodeBlock] {
        &self.code_blocks
    }

    /// Flat sub-band metadata.
    pub(crate) fn subbands(&self) -> &[CudaHtj2kSubband] {
        &self.subbands
    }

    /// Flat IDWT step metadata.
    pub(crate) fn idwt_steps(&self) -> &[CudaHtj2kIdwtStep] {
        &self.idwt_steps
    }

    /// Flat store step metadata.
    pub(crate) fn store_steps(&self) -> &[CudaHtj2kStoreStep] {
        &self.store_steps
    }

    /// Number of per-code-block decode dispatches implied by the plan.
    pub(crate) fn dispatch_count_hint(&self) -> usize {
        self.code_blocks.len()
    }
}

#[derive(Debug, Default)]
struct CudaPlanCapacityHint {
    payload_bytes: usize,
    code_blocks: usize,
    subbands: usize,
    idwt_steps: usize,
    store_steps: usize,
}

fn cuda_plan_capacity_hint(plan: &J2kDirectGrayscalePlan) -> Result<CudaPlanCapacityHint, Error> {
    let mut hint = CudaPlanCapacityHint::default();
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::HtSubBand(subband) => {
                hint.subbands = hint.subbands.saturating_add(1);
                hint.code_blocks = hint.code_blocks.checked_add(subband.jobs.len()).ok_or(
                    Error::UnsupportedCudaRequest {
                        reason: PLAN_PAYLOAD_TOO_LARGE,
                    },
                )?;
                for job in &subband.jobs {
                    hint.payload_bytes = hint.payload_bytes.checked_add(job.data.len()).ok_or(
                        Error::UnsupportedCudaRequest {
                            reason: PLAN_PAYLOAD_TOO_LARGE,
                        },
                    )?;
                }
            }
            J2kDirectGrayscaleStep::ClassicSubBand(_) => {}
            J2kDirectGrayscaleStep::Idwt(_) => {
                hint.idwt_steps = hint.idwt_steps.saturating_add(1);
            }
            J2kDirectGrayscaleStep::Store(_) => {
                hint.store_steps = hint.store_steps.saturating_add(1);
            }
        }
    }
    Ok(hint)
}

fn convert_idwt_step(step: J2kDirectIdwtStep) -> CudaHtj2kIdwtStep {
    CudaHtj2kIdwtStep {
        output_band_id: step.output_band_id,
        transform: CudaHtj2kTransform::from_native(step.transform),
        rect: convert_rect(step.rect),
        ll_band_id: step.ll_band_id,
        ll_rect: convert_rect(step.ll),
        hl_band_id: step.hl_band_id,
        hl_rect: convert_rect(step.hl),
        lh_band_id: step.lh_band_id,
        lh_rect: convert_rect(step.lh),
        hh_band_id: step.hh_band_id,
        hh_rect: convert_rect(step.hh),
    }
}

fn required_regions_for_direct_plan(
    plan: &J2kDirectGrayscalePlan,
) -> Result<HashMap<CudaHtj2kBandId, RequiredBandRegion>, Error> {
    let mut required = HashMap::<CudaHtj2kBandId, RequiredBandRegion>::new();
    for step in &plan.steps {
        let J2kDirectGrayscaleStep::Store(store) = step else {
            continue;
        };
        let source_right =
            store
                .source_x
                .checked_add(store.copy_width)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: PLAN_OUTPUT_RECT_MISMATCH,
                })?;
        let source_bottom =
            store
                .source_y
                .checked_add(store.copy_height)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: PLAN_OUTPUT_RECT_MISMATCH,
                })?;
        if let Some(region) =
            RequiredBandRegion::new(store.source_x, store.source_y, source_right, source_bottom)
        {
            add_required_region(&mut required, store.input_band_id, region);
        }
    }

    for step in plan.steps.iter().rev() {
        let J2kDirectGrayscaleStep::Idwt(idwt) = step else {
            continue;
        };
        let Some(output_region) = required.get(&idwt.output_band_id).copied() else {
            continue;
        };
        let expanded = output_region.expanded_within_band(
            idwt_required_output_margin(idwt.transform),
            idwt.rect.width(),
            idwt.rect.height(),
        );
        add_idwt_input_required_regions(&mut required, idwt, expanded);
    }
    Ok(required)
}

fn add_required_region(
    required: &mut HashMap<CudaHtj2kBandId, RequiredBandRegion>,
    band_id: CudaHtj2kBandId,
    region: RequiredBandRegion,
) {
    required
        .entry(band_id)
        .and_modify(|existing| *existing = existing.union(region))
        .or_insert(region);
}

fn add_idwt_input_required_regions(
    required: &mut HashMap<CudaHtj2kBandId, RequiredBandRegion>,
    idwt: &J2kDirectIdwtStep,
    output_region: RequiredBandRegion,
) {
    let windows = idwt_required_input_windows(idwt, output_region);
    add_required_region(required, idwt.ll_band_id, windows.ll);
    add_required_region(required, idwt.hl_band_id, windows.hl);
    add_required_region(required, idwt.lh_band_id, windows.lh);
    add_required_region(required, idwt.hh_band_id, windows.hh);
}

fn convert_store_step(
    step: J2kDirectStoreStep,
    output_origin: (u32, u32),
    output_dimensions: (u32, u32),
) -> Result<CudaHtj2kStoreStep, Error> {
    if output_dimensions.0 == 0 || output_dimensions.1 == 0 {
        return Err(Error::UnsupportedCudaRequest {
            reason: PLAN_OUTPUT_RECT_MISMATCH,
        });
    }
    let region_end_x =
        output_origin
            .0
            .checked_add(output_dimensions.0)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: PLAN_OUTPUT_RECT_MISMATCH,
            })?;
    let region_end_y =
        output_origin
            .1
            .checked_add(output_dimensions.1)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: PLAN_OUTPUT_RECT_MISMATCH,
            })?;
    let store_end_x =
        step.output_x
            .checked_add(step.copy_width)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: PLAN_OUTPUT_RECT_MISMATCH,
            })?;
    let store_end_y =
        step.output_y
            .checked_add(step.copy_height)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: PLAN_OUTPUT_RECT_MISMATCH,
            })?;
    if output_origin.0 < step.output_x
        || output_origin.1 < step.output_y
        || region_end_x > store_end_x
        || region_end_y > store_end_y
    {
        return Err(Error::UnsupportedCudaRequest {
            reason: PLAN_OUTPUT_RECT_MISMATCH,
        });
    }
    let source_x = step
        .source_x
        .checked_add(output_origin.0 - step.output_x)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: PLAN_OUTPUT_RECT_MISMATCH,
        })?;
    let source_y = step
        .source_y
        .checked_add(output_origin.1 - step.output_y)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: PLAN_OUTPUT_RECT_MISMATCH,
        })?;
    Ok(CudaHtj2kStoreStep {
        input_band_id: step.input_band_id,
        input_rect: convert_rect(step.input_rect),
        source_x,
        source_y,
        copy_width: output_dimensions.0,
        copy_height: output_dimensions.1,
        output_width: output_dimensions.0,
        output_height: output_dimensions.1,
        output_x: 0,
        output_y: 0,
        addend: step.addend,
    })
}

fn convert_rect(rect: J2kRect) -> CudaHtj2kRect {
    CudaHtj2kRect {
        x0: rect.x0,
        y0: rect.y0,
        x1: rect.x1,
        y1: rect.y1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_core::CodecError;
    use j2k_native::{HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan};

    fn one_block_direct_plan(
        cleanup_length: u32,
        refinement_length: u32,
        data: Vec<u8>,
        output_stride: usize,
    ) -> J2kDirectGrayscalePlan {
        J2kDirectGrayscalePlan {
            dimensions: (1, 1),
            bit_depth: 8,
            steps: vec![
                J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                    band_id: 0,
                    rect: J2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    width: 1,
                    height: 1,
                    jobs: vec![HtOwnedCodeBlockBatchJob {
                        output_x: 0,
                        output_y: 0,
                        data,
                        cleanup_length,
                        refinement_length,
                        width: 1,
                        height: 1,
                        output_stride,
                        missing_bit_planes: 0,
                        number_of_coding_passes: 1,
                        num_bitplanes: 8,
                        roi_shift: 0,
                        stripe_causal: false,
                        strict: true,
                        dequantization_step: 1.0,
                    }],
                }),
                J2kDirectGrayscaleStep::Store(J2kDirectStoreStep {
                    input_band_id: 0,
                    input_rect: J2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 1,
                        y1: 1,
                    },
                    source_x: 0,
                    source_y: 0,
                    copy_width: 1,
                    copy_height: 1,
                    output_width: 1,
                    output_height: 1,
                    output_x: 0,
                    output_y: 0,
                    addend: 128.0,
                }),
            ],
        }
    }

    fn one_block_plan(data: Vec<u8>) -> CudaHtj2kDecodePlan {
        let payload_len = u32::try_from(data.len()).expect("fixture payload length");
        let direct = one_block_direct_plan(payload_len, 0, data, 1);
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
            .expect("CUDA plan")
    }

    fn two_block_direct_plan() -> J2kDirectGrayscalePlan {
        J2kDirectGrayscalePlan {
            dimensions: (2, 1),
            bit_depth: 8,
            steps: vec![
                J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                    band_id: 0,
                    rect: J2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 2,
                        y1: 1,
                    },
                    width: 2,
                    height: 1,
                    jobs: vec![
                        HtOwnedCodeBlockBatchJob {
                            output_x: 0,
                            output_y: 0,
                            data: vec![1],
                            cleanup_length: 1,
                            refinement_length: 0,
                            width: 1,
                            height: 1,
                            output_stride: 2,
                            missing_bit_planes: 0,
                            number_of_coding_passes: 1,
                            num_bitplanes: 8,
                            roi_shift: 0,
                            stripe_causal: false,
                            strict: true,
                            dequantization_step: 1.0,
                        },
                        HtOwnedCodeBlockBatchJob {
                            output_x: 1,
                            output_y: 0,
                            data: vec![2],
                            cleanup_length: 1,
                            refinement_length: 0,
                            width: 1,
                            height: 1,
                            output_stride: 2,
                            missing_bit_planes: 0,
                            number_of_coding_passes: 1,
                            num_bitplanes: 8,
                            roi_shift: 0,
                            stripe_causal: false,
                            strict: true,
                            dequantization_step: 1.0,
                        },
                    ],
                }),
                J2kDirectGrayscaleStep::Store(J2kDirectStoreStep {
                    input_band_id: 0,
                    input_rect: J2kRect {
                        x0: 0,
                        y0: 0,
                        x1: 2,
                        y1: 1,
                    },
                    source_x: 0,
                    source_y: 0,
                    copy_width: 2,
                    copy_height: 1,
                    output_width: 2,
                    output_height: 1,
                    output_x: 0,
                    output_y: 0,
                    addend: 128.0,
                }),
            ],
        }
    }

    #[test]
    fn append_payload_to_shared_offsets_blocks_and_drains_local_payload() {
        let mut first = one_block_plan(vec![1, 2]);
        let mut second = one_block_plan(vec![3, 4, 5]);
        let mut shared = Vec::new();

        first
            .append_payload_to_shared(&mut shared)
            .expect("append first payload");
        second
            .append_payload_to_shared(&mut shared)
            .expect("append second payload");

        assert_eq!(shared, vec![1, 2, 3, 4, 5]);
        assert!(first.payload().is_empty());
        assert!(second.payload().is_empty());
        assert_eq!(first.code_blocks()[0].payload_offset, 0);
        assert_eq!(second.code_blocks()[0].payload_offset, 2);
    }

    #[test]
    fn rebase_payload_offsets_preserves_shared_payload_for_larger_batch() {
        let mut plan = one_block_plan(vec![7, 8]);
        let mut shared = Vec::new();
        plan.append_payload_to_shared(&mut shared)
            .expect("append local payload");

        plan.rebase_payload_offsets(4096).expect("rebase payload");

        assert_eq!(shared, vec![7, 8]);
        assert_eq!(plan.code_blocks()[0].payload_offset, 4096);
    }

    #[test]
    fn full_frame_plan_keeps_all_blocks_while_region_plan_prunes() {
        let direct = two_block_direct_plan();
        let full =
            CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
                .expect("full CUDA plan");
        let mut region_direct = two_block_direct_plan();
        let J2kDirectGrayscaleStep::Store(store) = &mut region_direct.steps[1] else {
            panic!("expected store fixture");
        };
        store.source_x = 1;
        store.copy_width = 1;
        store.output_x = 1;
        let region = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
            &region_direct,
            PixelFormat::Gray8,
            (1, 0),
            (1, 1),
        )
        .expect("region CUDA plan");

        assert_eq!(full.code_blocks().len(), 2);
        assert_eq!(region.code_blocks().len(), 1);
        assert_eq!(region.code_blocks()[0].output_x, 1);
    }

    #[test]
    fn rejects_block_length_mismatch() {
        let direct = one_block_direct_plan(1, 2, vec![0xAA, 0xBB], 1);

        let error =
            CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
                .expect_err("mismatched cleanup/refinement lengths must be rejected");

        assert!(error.is_unsupported());
        assert!(
            error
                .to_string()
                .contains("block lengths do not match payload bytes"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_roi_maxshift_jobs() {
        let mut direct = one_block_direct_plan(1, 0, vec![0xAA], 1);
        let J2kDirectGrayscaleStep::HtSubBand(subband) = &mut direct.steps[0] else {
            panic!("fixture starts with one HT sub-band");
        };
        subband.jobs[0].roi_shift = 7;

        let error =
            CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
                .expect_err("ROI maxshift jobs must be rejected");

        assert!(error.is_unsupported());
        assert!(
            error.to_string().contains("ROI maxshift decode"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_output_stride_overflow() {
        let direct = one_block_direct_plan(1, 0, vec![0xAA], usize::MAX);

        let error =
            CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
                .expect_err("unrepresentable output stride must be rejected");

        assert!(error.is_unsupported());
    }

    #[test]
    fn rejects_mixed_idwt_transforms() {
        let mut direct = one_block_direct_plan(1, 0, vec![0xAA], 1);
        let rect = J2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        };
        direct.steps.insert(
            1,
            J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
                output_band_id: 4,
                rect,
                transform: J2kWaveletTransform::Reversible53,
                ll_band_id: 0,
                ll: rect,
                hl_band_id: 1,
                hl: rect,
                lh_band_id: 2,
                lh: rect,
                hh_band_id: 3,
                hh: rect,
            }),
        );
        direct.steps.insert(
            2,
            J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
                output_band_id: 8,
                rect,
                transform: J2kWaveletTransform::Irreversible97,
                ll_band_id: 4,
                ll: rect,
                hl_band_id: 5,
                hl: rect,
                lh_band_id: 6,
                lh: rect,
                hh_band_id: 7,
                hh: rect,
            }),
        );

        let error =
            CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
                .expect_err("mixed transforms must be rejected");

        assert!(error.is_unsupported());
        assert!(
            error.to_string().contains("mixed DWT transforms"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn region_plan_rejects_store_outside_output_rect() {
        let direct = one_block_direct_plan(1, 0, vec![0xAA], 1);

        let error = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
            &direct,
            PixelFormat::Gray8,
            (1, 1),
            (0, 0),
        )
        .expect_err("store outside compact output rectangle must be rejected");

        assert!(error.is_unsupported());
        assert!(
            error
                .to_string()
                .contains("store does not fit the requested output rectangle"),
            "unexpected error: {error}"
        );
    }
}
