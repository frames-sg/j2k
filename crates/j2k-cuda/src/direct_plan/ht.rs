// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{HtCodeBlockPayloadRanges, HtOwnedSubBandPlan, J2kCodestreamRange};

use super::{
    required_regions::RequiredBandRegions, shared::CudaPlanOwners, CudaHtj2kCodeBlock,
    CudaHtj2kSubband, Error, PLAN_PAYLOAD_TOO_LARGE,
};

const PLAN_BLOCK_LENGTH_MISMATCH: &str =
    "strict CUDA HTJ2K plan block lengths do not match payload bytes";
const ROI_MAXSHIFT_UNSUPPORTED: &str =
    "strict CUDA HTJ2K plan does not support ROI maxshift decode";

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
            .ok_or(Error::UnsupportedCudaRequest {
                reason: PLAN_BLOCK_LENGTH_MISMATCH,
            })?;
        if expected_len != payload_len {
            return Err(Error::UnsupportedCudaRequest {
                reason: PLAN_BLOCK_LENGTH_MISMATCH,
            });
        }
        if job.roi_shift != 0 {
            return Err(Error::UnsupportedCudaRequest {
                reason: ROI_MAXSHIFT_UNSUPPORTED,
            });
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
            stripe_causal: u8::from(job.stripe_causal),
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
        let ranges = payloads.next().ok_or(Error::UnsupportedCudaRequest {
            reason: PLAN_BLOCK_LENGTH_MISMATCH,
        })?;
        if required_regions.is_some_and(|regions| {
            !regions.get(subband.band_id).is_some_and(|required| {
                required.intersects(job.output_x, job.output_y, job.width, job.height)
            })
        }) {
            continue;
        }
        if !job.data.is_empty() || job.roi_shift != 0 {
            return Err(Error::UnsupportedCudaRequest {
                reason: if job.roi_shift != 0 {
                    ROI_MAXSHIFT_UNSUPPORTED
                } else {
                    PLAN_BLOCK_LENGTH_MISMATCH
                },
            });
        }
        let cleanup = referenced_slice(encoded, ranges.cleanup)?;
        let refinement = match (job.refinement_length, ranges.refinement) {
            (0, None) => &[][..],
            (0, Some(range)) if range.length == 0 => &[][..],
            (_, Some(range)) => referenced_slice(encoded, range)?,
            (_, None) => {
                return Err(Error::UnsupportedCudaRequest {
                    reason: PLAN_BLOCK_LENGTH_MISMATCH,
                });
            }
        };
        if cleanup.len() != job.cleanup_length as usize
            || refinement.len() != job.refinement_length as usize
        {
            return Err(Error::UnsupportedCudaRequest {
                reason: PLAN_BLOCK_LENGTH_MISMATCH,
            });
        }
        let payload_offset = checked_u64(shared_payload.len())?;
        let payload_len =
            cleanup
                .len()
                .checked_add(refinement.len())
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: PLAN_PAYLOAD_TOO_LARGE,
                })?;
        let payload_len = checked_u32(payload_len)?;
        shared_payload.extend_from_slice(cleanup);
        shared_payload.extend_from_slice(refinement);
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
            stripe_causal: u8::from(job.stripe_causal),
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
            .ok_or(Error::UnsupportedCudaRequest {
                reason: PLAN_PAYLOAD_TOO_LARGE,
            })
    })
}

fn referenced_slice(encoded: &[u8], range: J2kCodestreamRange) -> Result<&[u8], Error> {
    let end = range.end().ok_or(Error::UnsupportedCudaRequest {
        reason: PLAN_PAYLOAD_TOO_LARGE,
    })?;
    encoded
        .get(range.offset..end)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: PLAN_BLOCK_LENGTH_MISMATCH,
        })
}

fn checked_u32(value: usize) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::UnsupportedCudaRequest {
        reason: PLAN_PAYLOAD_TOO_LARGE,
    })
}

fn checked_u64(value: usize) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| Error::UnsupportedCudaRequest {
        reason: PLAN_PAYLOAD_TOO_LARGE,
    })
}
