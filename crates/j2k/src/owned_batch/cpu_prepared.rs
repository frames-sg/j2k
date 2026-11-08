// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parse-free prepared-plan execution and dense CPU final store.

use super::cpu_fast::{CpuFlattenedPayloadJob, CpuGroupFastWorkspace};
use super::cpu_materialize::write_batch_samples;
use super::{BatchCodecRoute, BatchGroupInfo, CpuPreparedImage, PreparedCodecPlan};
use crate::{batch::worker::BatchWorker, J2kError};

pub(super) fn decode_prepared_plan_samples<T: Copy>(
    image: &CpuPreparedImage,
    info: &BatchGroupInfo,
    worker: &mut BatchWorker,
    out: &mut [T],
    convert: impl Fn(f32, u8) -> T,
) -> Result<bool, J2kError> {
    // General native reconstruction carries operation policy through bounded
    // loops. Direct offset execution retains separate, uncontrolled scratch.
    if image.control().is_some() {
        return Ok(false);
    }
    let components = match image.codec_plan() {
        PreparedCodecPlan::Htj2k(prepared_plan) => worker
            .execute_prepared_htj2k_plan(prepared_plan.cpu_plan(), image.bytes(), info.signed)
            .map_err(J2kError::from_native_decode_error)?,
        PreparedCodecPlan::Classic(prepared_plan) => worker
            .execute_prepared_classic_plan(prepared_plan.cpu_plan(), image.bytes(), info.signed)
            .map_err(J2kError::from_native_decode_error)?,
        PreparedCodecPlan::MetadataOnly => return Ok(false),
    };
    let output_rect = image.output_rect();
    if (output_rect.w, output_rect.h) != info.dimensions {
        return Err(J2kError::internal_backend(
            "prepared codec plan output metadata changed during decode",
        ));
    }
    write_direct_samples(&components, info, out, convert)?;
    Ok(true)
}

pub(super) fn prepare_staged_image(
    image: &CpuPreparedImage,
    route: BatchCodecRoute,
    worker: &mut BatchWorker,
    scratch: &mut j2k_native::J2kDirectCpuScratch,
) -> Result<(), J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => worker
            .prepare_staged_htj2k_plan(plan.cpu_plan(), scratch)
            .map_err(J2kError::from_native_decode_error),
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => worker
            .prepare_staged_classic_plan(plan.cpu_plan(), scratch)
            .map_err(J2kError::from_native_decode_error),
        _ => Err(J2kError::internal_backend(
            "staged CPU route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn execute_staged_entropy_job(
    image: &CpuPreparedImage,
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
                    plan.cpu_plan(),
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
                    plan.cpu_plan(),
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
    image: &CpuPreparedImage,
    route: BatchCodecRoute,
    worker: &mut BatchWorker,
) -> Result<(), J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => worker
            .prepare_staged_htj2k_entropy_workspace(plan.cpu_plan())
            .map_err(J2kError::from_native_decode_error),
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => worker
            .prepare_staged_classic_entropy_workspace(plan.cpu_plan())
            .map_err(J2kError::from_native_decode_error),
        _ => Err(J2kError::internal_backend(
            "staged CPU worker route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn staged_tile_count(
    image: &CpuPreparedImage,
    route: BatchCodecRoute,
) -> Result<usize, J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => {
            Ok(plan.cpu_plan().tiles().len())
        }
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => {
            Ok(plan.cpu_plan().tiles().len())
        }
        _ => Err(J2kError::internal_backend(
            "staged CPU route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn prepare_staged_tile(
    image: &CpuPreparedImage,
    route: BatchCodecRoute,
    tile_index: usize,
    scratch: &mut j2k_native::J2kDirectCpuScratch,
) -> Result<(), J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => {
            j2k_native::prepare_referenced_htj2k_tile_staged(plan.cpu_plan(), tile_index, scratch)
                .map_err(J2kError::from_native_decode_error)
        }
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => {
            j2k_native::prepare_referenced_classic_tile_staged(plan.cpu_plan(), tile_index, scratch)
                .map_err(J2kError::from_native_decode_error)
        }
        _ => Err(J2kError::internal_backend(
            "staged CPU route does not match its prepared codec plan",
        )),
    }
}

pub(super) fn finish_staged_tile(
    image: &CpuPreparedImage,
    route: BatchCodecRoute,
    tile_index: usize,
    signed: bool,
    scratch: &mut j2k_native::J2kDirectCpuScratch,
) -> Result<(), J2kError> {
    match (route, image.codec_plan()) {
        (BatchCodecRoute::Htj2k, PreparedCodecPlan::Htj2k(plan)) => {
            j2k_native::finish_referenced_htj2k_tile_staged(
                plan.cpu_plan(),
                tile_index,
                signed,
                scratch,
            )
            .map_err(J2kError::from_native_decode_error)
        }
        (BatchCodecRoute::Classic, PreparedCodecPlan::Classic(plan)) => {
            j2k_native::finish_referenced_classic_tile_staged(
                plan.cpu_plan(),
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
    image: &CpuPreparedImage,
    info: &BatchGroupInfo,
    scratch: &mut j2k_native::J2kDirectCpuScratch,
    out: &mut [T],
    convert: impl Fn(f32, u8) -> T,
) -> Result<(), J2kError> {
    let components = match image.codec_plan() {
        PreparedCodecPlan::Htj2k(plan) => {
            j2k_native::finish_referenced_htj2k_staged(plan.cpu_plan(), info.signed, scratch)
                .map_err(J2kError::from_native_decode_error)?
        }
        PreparedCodecPlan::Classic(plan) => {
            j2k_native::finish_referenced_classic_staged(plan.cpu_plan(), info.signed, scratch)
                .map_err(J2kError::from_native_decode_error)?
        }
        PreparedCodecPlan::MetadataOnly => {
            return Err(J2kError::internal_backend(
                "staged CPU finish has no prepared codec plan",
            ));
        }
    };
    let output_rect = image.output_rect();
    if (output_rect.w, output_rect.h) != info.dimensions {
        return Err(J2kError::internal_backend(
            "prepared codec plan output metadata changed during decode",
        ));
    }
    write_direct_samples(&components, info, out, convert)
}

fn write_direct_samples<T: Copy>(
    components: &j2k_native::J2kDirectDecodedComponents<'_>,
    info: &BatchGroupInfo,
    out: &mut [T],
    convert: impl Fn(f32, u8) -> T,
) -> Result<(), J2kError> {
    write_batch_samples(
        components.dimensions(),
        components.component_count(),
        |component| {
            components
                .plane(component)
                .map(|plane| (plane.dimensions(), plane.bit_depth(), plane.samples()))
        },
        info,
        out,
        convert,
    )
}
