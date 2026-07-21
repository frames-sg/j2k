// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep, J2kRect,
};

use super::{
    CudaClassicCodeBlock, CudaClassicSegment, CudaClassicSubband, CudaHtj2kCodeBlock,
    CudaHtj2kDecodePlan, CudaHtj2kIdwtStep, CudaHtj2kRect, CudaHtj2kStoreStep, CudaHtj2kSubband,
    CudaHtj2kTransform, Error, EMPTY_CUDA_PLAN, MIXED_TRANSFORMS_UNSUPPORTED,
    PLAN_OUTPUT_RECT_MISMATCH, PLAN_PAYLOAD_TOO_LARGE,
};
use crate::allocation::HostPhaseBudget;

#[derive(Debug, Default)]
pub(super) struct CudaPlanCapacityHint {
    payload_bytes: usize,
    code_blocks: usize,
    subbands: usize,
    classic_code_blocks: usize,
    classic_segments: usize,
    classic_subbands: usize,
    idwt_steps: usize,
    store_steps: usize,
}

pub(super) struct CudaPlanOwners {
    pub(super) payload: Vec<u8>,
    pub(super) code_blocks: Vec<CudaHtj2kCodeBlock>,
    pub(super) classic_code_blocks: Vec<CudaClassicCodeBlock>,
    pub(super) classic_segments: Vec<CudaClassicSegment>,
    pub(super) classic_subbands: Vec<CudaClassicSubband>,
    pub(super) subbands: Vec<CudaHtj2kSubband>,
    pub(super) idwt_steps: Vec<CudaHtj2kIdwtStep>,
    pub(super) store_steps: Vec<CudaHtj2kStoreStep>,
    transform: Option<CudaHtj2kTransform>,
}

impl CudaPlanOwners {
    pub(super) fn from_plan(plan: &J2kDirectGrayscalePlan) -> Result<(Self, usize), Error> {
        let hint = cuda_plan_capacity_hint(plan)?;
        let mut budget = HostPhaseBudget::new("CUDA direct-plan owner graph");
        let owners = Self {
            payload: budget.try_vec_with_capacity(hint.payload_bytes)?,
            code_blocks: budget.try_vec_with_capacity(hint.code_blocks)?,
            classic_code_blocks: budget.try_vec_with_capacity(hint.classic_code_blocks)?,
            classic_segments: budget.try_vec_with_capacity(hint.classic_segments)?,
            classic_subbands: budget.try_vec_with_capacity(hint.classic_subbands)?,
            subbands: budget.try_vec_with_capacity(hint.subbands)?,
            idwt_steps: budget.try_vec_with_capacity(hint.idwt_steps)?,
            store_steps: budget.try_vec_with_capacity(hint.store_steps)?,
            transform: None,
        };
        Ok((owners, budget.live_bytes()))
    }

    pub(super) fn append_idwt(&mut self, step: J2kDirectIdwtStep) -> Result<(), Error> {
        let transform = CudaHtj2kTransform::from_native(step.transform);
        match self.transform {
            Some(existing) if existing != transform => {
                return Err(Error::UnsupportedCudaRequest {
                    reason: MIXED_TRANSFORMS_UNSUPPORTED,
                });
            }
            Some(_) => {}
            None => self.transform = Some(transform),
        }
        self.idwt_steps.push(convert_idwt_step(step));
        Ok(())
    }

    pub(super) fn finish(
        self,
        plan: &J2kDirectGrayscalePlan,
        output_format: j2k_core::PixelFormat,
        output_origin: (u32, u32),
        dimensions: (u32, u32),
    ) -> Result<CudaHtj2kDecodePlan, Error> {
        if self.code_blocks.is_empty() && self.classic_code_blocks.is_empty() {
            return Err(Error::UnsupportedCudaRequest {
                reason: EMPTY_CUDA_PLAN,
            });
        }
        Ok(CudaHtj2kDecodePlan {
            dimensions,
            bit_depth: plan.bit_depth,
            output_format,
            output_origin,
            transform: self.transform.unwrap_or(CudaHtj2kTransform::Reversible53),
            payload: self.payload,
            code_blocks: self.code_blocks,
            classic_code_blocks: self.classic_code_blocks,
            classic_segments: self.classic_segments,
            classic_subbands: self.classic_subbands,
            subbands: self.subbands,
            idwt_steps: self.idwt_steps,
            store_steps: self.store_steps,
        })
    }
}

fn cuda_plan_capacity_hint(plan: &J2kDirectGrayscalePlan) -> Result<CudaPlanCapacityHint, Error> {
    let mut hint = CudaPlanCapacityHint::default();
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::HtSubBand(subband) => {
                hint.subbands = hint.subbands.saturating_add(1);
                hint.code_blocks = checked_add(hint.code_blocks, subband.jobs.len())?;
                for job in &subband.jobs {
                    hint.payload_bytes = checked_add(hint.payload_bytes, job.data.len())?;
                }
            }
            J2kDirectGrayscaleStep::ClassicSubBand(subband) => {
                hint.classic_subbands = hint.classic_subbands.saturating_add(1);
                hint.classic_code_blocks =
                    checked_add(hint.classic_code_blocks, subband.jobs.len())?;
                for job in &subband.jobs {
                    hint.payload_bytes = checked_add(hint.payload_bytes, job.data.len())?;
                    hint.classic_segments = checked_add(hint.classic_segments, job.segments.len())?;
                }
            }
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

fn checked_add(current: usize, additional: usize) -> Result<usize, Error> {
    current
        .checked_add(additional)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: PLAN_PAYLOAD_TOO_LARGE,
        })
}

pub(super) fn convert_idwt_step(step: J2kDirectIdwtStep) -> CudaHtj2kIdwtStep {
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

pub(super) fn convert_store_step(
    step: J2kDirectStoreStep,
    output_origin: (u32, u32),
    output_dimensions: (u32, u32),
) -> Result<CudaHtj2kStoreStep, Error> {
    if output_dimensions.0 == 0 || output_dimensions.1 == 0 {
        return output_rect_error();
    }
    let region_end_x = checked_end(output_origin.0, output_dimensions.0)?;
    let region_end_y = checked_end(output_origin.1, output_dimensions.1)?;
    let store_end_x = checked_end(step.output_x, step.copy_width)?;
    let store_end_y = checked_end(step.output_y, step.copy_height)?;
    if output_origin.0 < step.output_x
        || output_origin.1 < step.output_y
        || region_end_x > store_end_x
        || region_end_y > store_end_y
    {
        return output_rect_error();
    }
    let source_x = checked_end(step.source_x, output_origin.0 - step.output_x)?;
    let source_y = checked_end(step.source_y, output_origin.1 - step.output_y)?;
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

fn checked_end(start: u32, len: u32) -> Result<u32, Error> {
    start.checked_add(len).ok_or(Error::UnsupportedCudaRequest {
        reason: PLAN_OUTPUT_RECT_MISMATCH,
    })
}

fn output_rect_error<T>() -> Result<T, Error> {
    Err(Error::UnsupportedCudaRequest {
        reason: PLAN_OUTPUT_RECT_MISMATCH,
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
