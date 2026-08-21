// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    HtCodeBlockPayloadRanges, HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kCodestreamRange,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
};

use super::{
    required_regions::RequiredBandRegions, shared::CudaPlanOwners, CudaHtj2kCodeBlock,
    CudaHtj2kSubband, Error, PLAN_PAYLOAD_TOO_LARGE,
};

const PLAN_BLOCK_LENGTH_MISMATCH: &str =
    "strict CUDA HTJ2K plan block lengths do not match payload bytes";
const PLAN_BITPLANES_UNSUPPORTED: &str =
    "strict CUDA HTJ2K plan has invalid coded bitplane or ROI maxshift metadata";

#[cfg(test)]
mod tests;

pub(super) fn append_ht_subband(
    owners: &mut CudaPlanOwners,
    subband: &HtOwnedSubBandPlan,
    required_regions: Option<&RequiredBandRegions>,
) -> Result<(), Error> {
    let subband_index = checked_u32(owners.subbands.len())?;
    let code_block_start = checked_u32(owners.code_blocks.len())?;
    for job in &subband.jobs {
        if required_regions.is_some_and(|regions| {
            !regions.get(subband.band_id).is_some_and(|required| {
                required.intersects(job.output_x, job.output_y, job.width, job.height)
            })
        }) {
            continue;
        }
        let payload_offset = checked_u64(owners.payload.len())?;
        let payload_len = checked_u32(job.data.len())?;
        let expected_len = job
            .cleanup_length
            .checked_add(job.refinement_length)
            .ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
            ))?;
        if expected_len != payload_len {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
            ));
        }
        if job
            .num_bitplanes
            .checked_add(job.roi_shift)
            .is_none_or(|coded_bitplanes| coded_bitplanes > 31)
        {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::unsupported_bit_depth(PLAN_BITPLANES_UNSUPPORTED),
            ));
        }
        let output_stride = checked_u32(job.output_stride)?;
        owners.payload.extend_from_slice(&job.data);
        owners.code_blocks.push(CudaHtj2kCodeBlock {
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
            roi_shift: job.roi_shift,
            stripe_causal: u8::from(job.stripe_causal),
            irreversible_midpoint: subband.irreversible_midpoint,
            dequantization_step: job.dequantization_step,
        });
    }
    owners.subbands.push(CudaHtj2kSubband {
        band_id: subband.band_id,
        x0: subband.rect.x0,
        y0: subband.rect.y0,
        x1: subband.rect.x1,
        y1: subband.rect.y1,
        width: subband.width,
        height: subband.height,
        code_block_start,
        code_block_count: checked_u32(owners.code_blocks.len() - code_block_start as usize)?,
    });
    Ok(())
}

pub(super) fn append_referenced_ht_subband<'a>(
    owners: &mut CudaPlanOwners,
    subband: &HtOwnedSubBandPlan,
    required_regions: Option<&RequiredBandRegions>,
    payloads: &mut impl Iterator<Item = &'a HtCodeBlockPayloadRanges>,
    encoded: &[u8],
    shared_payload: &mut Vec<u8>,
) -> Result<(), Error> {
    let subband_index = checked_u32(owners.subbands.len())?;
    let code_block_start = checked_u32(owners.code_blocks.len())?;
    for job in &subband.jobs {
        let required = !required_regions.is_some_and(|regions| {
            !regions.get(subband.band_id).is_some_and(|required| {
                required.intersects(job.output_x, job.output_y, job.width, job.height)
            })
        });
        if !job.data.is_empty() {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
            ));
        }
        if job
            .num_bitplanes
            .checked_add(job.roi_shift)
            .is_none_or(|coded_bitplanes| coded_bitplanes > 31)
        {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::unsupported_bit_depth(PLAN_BITPLANES_UNSUPPORTED),
            ));
        }
        let payload_offset = checked_u64(shared_payload.len())?;
        let (payload_len, _) = consume_referenced_ht_payload(
            payloads,
            job,
            encoded,
            required.then_some(shared_payload),
        )?;
        if !required {
            continue;
        }
        let payload_len = checked_u32(payload_len)?;
        owners.code_blocks.push(CudaHtj2kCodeBlock {
            subband_index,
            payload_offset,
            payload_len,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            output_x: job.output_x,
            output_y: job.output_y,
            width: job.width,
            height: job.height,
            output_stride: checked_u32(job.output_stride)?,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            num_bitplanes: job.num_bitplanes,
            roi_shift: job.roi_shift,
            stripe_causal: u8::from(job.stripe_causal),
            irreversible_midpoint: subband.irreversible_midpoint,
            dequantization_step: job.dequantization_step,
        });
    }
    owners.subbands.push(CudaHtj2kSubband {
        band_id: subband.band_id,
        x0: subband.rect.x0,
        y0: subband.rect.y0,
        x1: subband.rect.x1,
        y1: subband.rect.y1,
        width: subband.width,
        height: subband.height,
        code_block_start,
        code_block_count: checked_u32(owners.code_blocks.len() - code_block_start as usize)?,
    });
    Ok(())
}

fn consume_referenced_ht_payload<'a>(
    payloads: &mut impl Iterator<Item = &'a HtCodeBlockPayloadRanges>,
    job: &HtOwnedCodeBlockBatchJob,
    encoded: &[u8],
    mut output: Option<&mut Vec<u8>>,
) -> Result<(usize, usize), Error> {
    let output_start = output.as_ref().map_or(0, |bytes| bytes.len());
    let result = (|| {
        let first = payloads.next().ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
        ))?;
        let mut record_count = 1usize;
        let cleanup = referenced_slice(encoded, first.cleanup)?;
        if cleanup.len() != job.cleanup_length as usize {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
            ));
        }
        if let Some(bytes) = output.as_deref_mut() {
            bytes.extend_from_slice(cleanup);
        }

        let expected_refinement = job.refinement_length as usize;
        let mut refinement_len = 0usize;
        if let Some(range) = first.refinement {
            let refinement = referenced_slice(encoded, range)?;
            refinement_len = refinement.len();
            if let Some(bytes) = output.as_deref_mut() {
                bytes.extend_from_slice(refinement);
            }
        }
        while refinement_len < expected_refinement {
            let continuation = payloads.next().ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
            ))?;
            record_count = record_count
                .checked_add(1)
                .ok_or(Error::capability_rejected(
                    j2k_core::CapabilityRejection::resource_limit(PLAN_PAYLOAD_TOO_LARGE),
                ))?;
            if continuation.cleanup.length != 0 {
                return Err(Error::capability_rejected(
                    j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
                ));
            }
            referenced_slice(encoded, continuation.cleanup)?;
            let range = continuation.refinement.ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
            ))?;
            let refinement = referenced_slice(encoded, range)?;
            if refinement.is_empty() {
                return Err(Error::capability_rejected(
                    j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
                ));
            }
            refinement_len =
                refinement_len
                    .checked_add(refinement.len())
                    .ok_or(Error::capability_rejected(
                        j2k_core::CapabilityRejection::resource_limit(PLAN_PAYLOAD_TOO_LARGE),
                    ))?;
            if refinement_len > expected_refinement {
                return Err(Error::capability_rejected(
                    j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
                ));
            }
            if let Some(bytes) = output.as_deref_mut() {
                bytes.extend_from_slice(refinement);
            }
        }
        if refinement_len != expected_refinement {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
            ));
        }
        let payload_len =
            cleanup
                .len()
                .checked_add(refinement_len)
                .ok_or(Error::capability_rejected(
                    j2k_core::CapabilityRejection::resource_limit(PLAN_PAYLOAD_TOO_LARGE),
                ))?;
        Ok((payload_len, record_count))
    })();
    if result.is_err() {
        if let Some(bytes) = output {
            bytes.truncate(output_start);
        }
    }
    result
}

pub(crate) fn referenced_ht_payload_record_count(
    plan: &J2kDirectGrayscalePlan,
    payloads: &[HtCodeBlockPayloadRanges],
    encoded: &[u8],
) -> Result<usize, Error> {
    let mut records = payloads.iter();
    let mut record_count = 0usize;
    for step in &plan.steps {
        if let J2kDirectGrayscaleStep::HtSubBand(subband) = step {
            for job in &subband.jobs {
                let (_, consumed) =
                    consume_referenced_ht_payload(&mut records, job, encoded, None)?;
                record_count =
                    record_count
                        .checked_add(consumed)
                        .ok_or(Error::capability_rejected(
                            j2k_core::CapabilityRejection::resource_limit(PLAN_PAYLOAD_TOO_LARGE),
                        ))?;
            }
        }
    }
    Ok(record_count)
}

pub(super) fn referenced_payload_bytes(
    encoded: &[u8],
    payloads: &[HtCodeBlockPayloadRanges],
) -> Result<usize, Error> {
    payloads.iter().try_fold(0usize, |total, payload| {
        let cleanup = referenced_slice(encoded, payload.cleanup)?.len();
        let refinement = payload.refinement.map_or(Ok(0), |range| {
            referenced_slice(encoded, range).map(<[u8]>::len)
        })?;
        total
            .checked_add(cleanup)
            .and_then(|value| value.checked_add(refinement))
            .ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::resource_limit(PLAN_PAYLOAD_TOO_LARGE),
            ))
    })
}

fn referenced_slice(encoded: &[u8], range: J2kCodestreamRange) -> Result<&[u8], Error> {
    let end = range.end().ok_or(Error::capability_rejected(
        j2k_core::CapabilityRejection::resource_limit(PLAN_PAYLOAD_TOO_LARGE),
    ))?;
    encoded
        .get(range.offset..end)
        .ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::geometry_mismatch(PLAN_BLOCK_LENGTH_MISMATCH),
        ))
}

fn checked_u32(value: usize) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| {
        Error::capability_rejected(j2k_core::CapabilityRejection::resource_limit(
            PLAN_PAYLOAD_TOO_LARGE,
        ))
    })
}

fn checked_u64(value: usize) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| {
        Error::capability_rejected(j2k_core::CapabilityRejection::resource_limit(
            PLAN_PAYLOAD_TOO_LARGE,
        ))
    })
}
