// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parse-free prepared-plan execution and dense CPU final store.

use j2k_core::Rect;

use super::cpu_fast::{CpuFlattenedPayloadJob, CpuGroupFastWorkspace};
use super::{BatchCodecRoute, BatchGroupInfo, BatchLayout, PreparedCodecPlan, PreparedImage};
use crate::{batch::worker::BatchWorker, J2kError};

pub(super) fn decode_prepared_plan_samples<T: Copy>(
    image: &PreparedImage,
    info: &BatchGroupInfo,
    worker: &mut BatchWorker,
    out: &mut [T],
    convert: impl Fn(f32, u8) -> T,
) -> Result<bool, J2kError> {
    let components = match image.codec_plan() {
        PreparedCodecPlan::Htj2k(prepared_plan) => worker
            .execute_prepared_htj2k_plan(prepared_plan.native_plan(), image.bytes(), info.signed)
            .map_err(J2kError::from_native_decode_error)?,
        PreparedCodecPlan::Classic(prepared_plan) => worker
            .execute_prepared_classic_plan(prepared_plan.native_plan(), image.bytes(), info.signed)
            .map_err(J2kError::from_native_decode_error)?,
        PreparedCodecPlan::MetadataOnly => return Ok(false),
    };
    let output_rect = image.plan().output_rect();
    validate_components(&components, output_rect, info, out.len())?;
    write_samples(&components, output_rect, info, out, convert)?;
    Ok(true)
}

pub(super) fn prepare_staged_image(
    image: &PreparedImage,
    route: BatchCodecRoute,
    worker: &mut BatchWorker,
    scratch: &mut j2k_native::J2kDirectCpuScratch,
) -> Result<(), J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => worker
            .prepare_staged_htj2k_plan(plan.native_plan(), scratch)
            .map_err(J2kError::from_native_decode_error),
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => worker
            .prepare_staged_classic_plan(plan.native_plan(), scratch)
            .map_err(J2kError::from_native_decode_error),
        _ => Err(J2kError::internal_backend(
            "staged CPU route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn execute_staged_entropy_job(
    image: &PreparedImage,
    job: CpuFlattenedPayloadJob,
    flattened: &CpuGroupFastWorkspace,
    worker: &mut BatchWorker,
    scratch: &mut j2k_native::J2kDirectCpuScratch,
) -> Result<(), J2kError> {
    match (flattened.route(), image.codec_plan()) {
        (Some(BatchCodecRoute::Htj2k), PreparedCodecPlan::Htj2k(plan)) => {
            let payload = flattened.ht_payload(job).ok_or_else(|| {
                J2kError::internal_backend("staged HT payload descriptor is missing")
            })?;
            worker
                .execute_staged_htj2k_job(
                    plan.native_plan(),
                    job.block_index,
                    flattened.arena(),
                    payload,
                    scratch,
                )
                .map_err(J2kError::from_native_decode_error)
        }
        (Some(BatchCodecRoute::Classic), PreparedCodecPlan::Classic(plan)) => {
            let payload = flattened.classic_payload_range(job).ok_or_else(|| {
                J2kError::internal_backend("staged classic payload descriptor is missing")
            })?;
            worker
                .execute_staged_classic_job(
                    plan.native_plan(),
                    job.block_index,
                    flattened.arena(),
                    payload,
                    scratch,
                )
                .map_err(J2kError::from_native_decode_error)
        }
        _ => Err(J2kError::internal_backend(
            "staged CPU route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn prepare_staged_entropy_worker(
    image: &PreparedImage,
    route: BatchCodecRoute,
    worker: &mut BatchWorker,
) -> Result<(), J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => worker
            .prepare_staged_htj2k_entropy_workspace(plan.native_plan())
            .map_err(J2kError::from_native_decode_error),
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => worker
            .prepare_staged_classic_entropy_workspace(plan.native_plan())
            .map_err(J2kError::from_native_decode_error),
        _ => Err(J2kError::internal_backend(
            "staged CPU worker route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn staged_tile_count(
    image: &PreparedImage,
    route: BatchCodecRoute,
) -> Result<usize, J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => {
            Ok(plan.native_plan().tiles().len())
        }
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => {
            Ok(plan.native_plan().tiles().len())
        }
        _ => Err(J2kError::internal_backend(
            "staged CPU route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn prepare_staged_tile(
    image: &PreparedImage,
    route: BatchCodecRoute,
    tile_index: usize,
    scratch: &mut j2k_native::J2kDirectCpuScratch,
) -> Result<(), J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => {
            j2k_native::prepare_referenced_htj2k_tile_staged(
                plan.native_plan(),
                tile_index,
                scratch,
            )
            .map_err(J2kError::from_native_decode_error)
        }
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => {
            j2k_native::prepare_referenced_classic_tile_staged(
                plan.native_plan(),
                tile_index,
                scratch,
            )
            .map_err(J2kError::from_native_decode_error)
        }
        _ => Err(J2kError::internal_backend(
            "staged CPU route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn finish_staged_tile(
    image: &PreparedImage,
    route: BatchCodecRoute,
    tile_index: usize,
    signed: bool,
    scratch: &mut j2k_native::J2kDirectCpuScratch,
) -> Result<(), J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => {
            j2k_native::finish_referenced_htj2k_tile_staged(
                plan.native_plan(),
                tile_index,
                signed,
                scratch,
            )
            .map_err(J2kError::from_native_decode_error)
        }
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => {
            j2k_native::finish_referenced_classic_tile_staged(
                plan.native_plan(),
                tile_index,
                signed,
                scratch,
            )
            .map_err(J2kError::from_native_decode_error)
        }
        _ => Err(J2kError::internal_backend(
            "staged CPU route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn finish_staged_plan_samples<T: Copy>(
    image: &PreparedImage,
    info: &BatchGroupInfo,
    scratch: &mut j2k_native::J2kDirectCpuScratch,
    out: &mut [T],
    convert: impl Fn(f32, u8) -> T,
) -> Result<(), J2kError> {
    let components = match image.codec_plan() {
        PreparedCodecPlan::Htj2k(plan) => {
            j2k_native::finish_referenced_htj2k_staged(plan.native_plan(), info.signed, scratch)
                .map_err(J2kError::from_native_decode_error)?
        }
        PreparedCodecPlan::Classic(plan) => {
            j2k_native::finish_referenced_classic_staged(plan.native_plan(), info.signed, scratch)
                .map_err(J2kError::from_native_decode_error)?
        }
        PreparedCodecPlan::MetadataOnly => {
            return Err(J2kError::internal_backend(
                "staged CPU finish has no prepared codec plan",
            ));
        }
    };
    let output_rect = image.plan().output_rect();
    validate_components(&components, output_rect, info, out.len())?;
    write_samples(&components, output_rect, info, out, convert)
}

fn validate_components(
    components: &j2k_native::J2kDirectDecodedComponents<'_>,
    output_rect: Rect,
    info: &BatchGroupInfo,
    output_len: usize,
) -> Result<(), J2kError> {
    if components.component_count() != info.color.channels()
        || (output_rect.w, output_rect.h) != info.dimensions
        || components.dimensions() != info.dimensions
        || info.samples_per_image() != Some(output_len)
    {
        return Err(J2kError::internal_backend(
            "prepared codec plan output metadata changed during decode",
        ));
    }
    let plane_len = (components.dimensions().0 as usize)
        .checked_mul(components.dimensions().1 as usize)
        .ok_or_else(|| J2kError::internal_backend("prepared HTJ2K plane size overflow"))?;
    for component in 0..components.component_count() {
        let plane = components.plane(component).ok_or_else(|| {
            J2kError::internal_backend("prepared codec component plane is missing")
        })?;
        if plane.dimensions() != components.dimensions()
            || plane.bit_depth() != info.precision
            || plane.samples().len() < plane_len
        {
            return Err(J2kError::BackendComponentPlaneTooShort {
                component,
                samples: plane.samples().len(),
                expected: plane_len,
            });
        }
    }
    Ok(())
}

fn write_samples<T: Copy>(
    components: &j2k_native::J2kDirectDecodedComponents<'_>,
    output_rect: Rect,
    info: &BatchGroupInfo,
    out: &mut [T],
    convert: impl Fn(f32, u8) -> T,
) -> Result<(), J2kError> {
    let output_width = output_rect.w as usize;
    let output_height = output_rect.h as usize;
    let source_width = components.dimensions().0 as usize;
    let channels = components.component_count();
    let output_pixels = output_width * output_height;
    for y in 0..output_height {
        for x in 0..output_width {
            let output_pixel = y * output_width + x;
            let source_pixel = y * source_width + x;
            for channel in 0..channels {
                let plane = components.plane(channel).ok_or_else(|| {
                    J2kError::internal_backend("prepared codec component plane is missing")
                })?;
                let destination = match info.layout {
                    BatchLayout::Nchw => channel * output_pixels + output_pixel,
                    BatchLayout::Nhwc => output_pixel * channels + channel,
                };
                out[destination] = convert(plane.samples()[source_pixel], plane.bit_depth());
            }
        }
    }
    Ok(())
}
