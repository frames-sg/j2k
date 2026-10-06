// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    native_decode_error, profile, CudaHtj2kColorDecodePlans, CudaHtj2kDecodePlan,
    CudaHtj2kDecodeProfileDetail, CudaHtj2kProfileReport, CudaHtj2kTransform, DecodeSettings,
    DeviceDecodePlan, DeviceDecodeRequest, Downscale, Error, J2kDecoder, NativeDecoderContext,
    NativeImage, PixelFormat, Rect,
};
#[cfg(feature = "cuda-runtime")]
use crate::allocation::HostPhaseBudget;
#[cfg(feature = "cuda-runtime")]
use j2k_native::{J2kReferencedClassicPlan, J2kReferencedHtj2kPlan};
#[cfg(feature = "cuda-runtime")]
mod color_owners;
#[cfg(feature = "cuda-runtime")]
use self::color_owners::{
    flatten_cuda_color_components, flatten_cuda_color_components_with_budget,
    flatten_direct_source_cuda_color_tile_components,
    flatten_referenced_classic_cuda_color_tile_components,
    flatten_referenced_cuda_color_tile_components,
};

#[cfg(feature = "cuda-runtime")]
mod color;
#[cfg(feature = "cuda-runtime")]
mod color_decoder;
#[cfg(feature = "cuda-runtime")]
mod color_referenced;
mod grayscale;

#[cfg(feature = "cuda-runtime")]
pub(super) use self::color::{
    build_cuda_color_plan_from_bytes_for_device_plan_with_profile,
    build_cuda_htj2k_color_plans_from_bytes_with_profile,
    build_cuda_htj2k_color_plans_from_bytes_with_profile_and_cap,
};
#[cfg(feature = "cuda-runtime")]
pub(super) use self::color_referenced::{
    build_cuda_classic_color_plans_from_referenced_with_profile,
    build_cuda_htj2k_color_plan_from_referenced_direct_source,
    build_cuda_htj2k_color_plans_from_referenced_with_profile,
};
pub(super) use self::grayscale::{
    build_cuda_classic_grayscale_plans_from_referenced_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile_and_cap,
    build_cuda_htj2k_grayscale_plans_from_referenced_with_profile,
};

#[cfg(feature = "cuda-runtime")]
const fn rgba_bit_depths_from_rgb(bit_depths: [u8; 3]) -> [u8; 4] {
    [bit_depths[0], bit_depths[1], bit_depths[2], 0]
}

/// Plans `inputs` in parallel waves of up to one input per Rayon thread,
/// splitting the remaining `host_cap` evenly across a wave's workers. `accept`
/// receives results in input order and accounts what it retains in
/// `retained`. When a worker hits the host planning limit, the results before
/// it are kept and planning resumes from that input with half as many
/// workers.
#[cfg(feature = "cuda-runtime")]
pub(super) fn plan_in_waves<I, C, T>(
    inputs: &[I],
    host_cap: usize,
    retained: &mut HostPhaseBudget,
    build: impl Fn(&I, &mut C, usize) -> Result<T, Error> + Sync + Send,
    mut accept: impl FnMut(&mut HostPhaseBudget, usize, T) -> Result<(), Error>,
) -> Result<(), Error>
where
    I: Sync,
    C: Default,
    T: Send,
{
    use rayon::prelude::*;

    let mut worker_limit = rayon::current_num_threads().max(1);
    let mut next_input = 0usize;
    while next_input < inputs.len() {
        let wave_inputs = &inputs[next_input..inputs.len().min(next_input + worker_limit)];
        let mut wave_budget = HostPhaseBudget::with_cap("j2k CUDA batch planning wave", host_cap);
        wave_budget.account_bytes(retained.live_bytes())?;
        let mut wave = wave_budget.try_vec_with_capacity(wave_inputs.len())?;
        let worker_cap = host_cap.saturating_sub(wave_budget.live_bytes()) / wave_inputs.len();
        if let [input] = wave_inputs {
            wave.push(build(input, &mut C::default(), worker_cap));
        } else {
            wave_inputs
                .par_iter()
                .map_init(C::default, |context, input| {
                    build(input, context, worker_cap)
                })
                .collect_into_vec(&mut wave);
        }

        let wave_len = wave.len();
        let mut kept = 0usize;
        for result in wave {
            match result {
                Ok(planned) => accept(retained, next_input + kept, planned)?,
                Err(error) if error.is_host_planning_limit() && wave_len > 1 => {
                    worker_limit = wave_len.div_ceil(2);
                    break;
                }
                Err(error) => return Err(error),
            }
            kept += 1;
        }
        next_input += kept;
    }
    Ok(())
}

/// Bytes needed to copy every referenced HT cleanup and refinement range.
#[cfg(feature = "cuda-runtime")]
pub(super) fn referenced_ht_payload_bytes(
    records: &[j2k_native::HtCodeBlockPayloadRanges],
) -> Result<usize, Error> {
    records
        .iter()
        .try_fold(0usize, |total, record| {
            total
                .checked_add(record.cleanup.length)?
                .checked_add(record.refinement.map_or(0, |range| range.length))
        })
        .ok_or_else(referenced_payload_overflow)
}

/// Bytes needed to copy every referenced classic code-block payload.
#[cfg(feature = "cuda-runtime")]
pub(super) fn referenced_classic_payload_bytes(
    tiles: &[j2k_native::J2kReferencedTilePlan],
) -> Result<usize, Error> {
    tiles
        .iter()
        .flat_map(j2k_native::J2kReferencedTilePlan::classic_payloads)
        .try_fold(0usize, |total, payload| {
            total.checked_add(payload.combined_length)
        })
        .ok_or_else(referenced_payload_overflow)
}

#[cfg(feature = "cuda-runtime")]
fn referenced_payload_overflow() -> Error {
    Error::capability_rejected(j2k_core::CapabilityRejection::resource_limit(
        "prepared CUDA referenced payload size overflows",
    ))
}

/// Offset of the next input's payload in a back-to-back batch upload.
#[cfg(feature = "cuda-runtime")]
pub(super) fn next_payload_base(payload_base: u64, payload_len: usize) -> Result<u64, Error> {
    u64::try_from(payload_len)
        .ok()
        .and_then(|len| payload_base.checked_add(len))
        .ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::resource_limit(
                super::CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
            ),
        ))
}

/// Parses `input` with the host budget outside `host_cap` treated as already
/// live, and starts a `host_cap` planning budget charged with the parsed image.
#[cfg(feature = "cuda-runtime")]
pub(super) fn parse_with_host_cap<'a>(
    input: &'a [u8],
    settings: &DecodeSettings,
    host_cap: usize,
    what: &'static str,
) -> Result<(NativeImage<'a>, HostPhaseBudget), Error> {
    let retained_baseline = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
        .checked_sub(host_cap)
        .ok_or(Error::HostAllocationTooLarge {
            requested: host_cap,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        })?;
    let image = NativeImage::new_with_retained_baseline(input, settings, retained_baseline)
        .map_err(native_decode_error)?;
    let mut budget = HostPhaseBudget::with_cap(what, host_cap);
    budget.account_bytes(
        image
            .retained_allocation_bytes()
            .map_err(native_decode_error)?,
    )?;
    Ok((image, budget))
}

/// Smallest host cap at which `plan` succeeds, found by bisection. Any error
/// other than the host planning limit fails the calling test.
#[cfg(test)]
pub(super) fn minimum_planning_cap<T>(mut plan: impl FnMut(usize) -> Result<T, Error>) -> usize {
    let mut rejected = 0usize;
    let mut accepted = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
    while accepted - rejected > 1 {
        let candidate = usize::midpoint(rejected, accepted);
        match plan(candidate) {
            Ok(_) => accepted = candidate,
            Err(error) if error.is_host_planning_limit() => rejected = candidate,
            Err(error) => panic!("valid bounded planning fixture failed: {error:?}"),
        }
    }
    accepted
}
