// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bitplane_encode, coefficients_fit_i32, default_public_code_block_style,
    downcast_i64_coefficients_to_i32, ht_block_encode, internal_sub_band_type,
    public_sub_band_type, BlockCodingMode, CodeBlockPacketData, EncodedJ2kCodeBlock,
    J2kEncodeStageAccelerator, J2kTier1CodeBlockEncodeJob, PreparedEncodeSubband, SubBandType,
    SubbandPrecinct, Vec, HT_CPU_PARALLEL_FALLBACK_MIN_JOBS,
};

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

pub(super) fn encode_prepared_subbands(
    prepared_subbands: Vec<PreparedEncodeSubband>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<SubbandPrecinct>, &'static str> {
    let block_coding_mode = prepared_subbands
        .iter()
        .find(|subband| !subband.code_blocks.is_empty())
        .map(|subband| subband.block_coding_mode);
    let encoded_blocks = match block_coding_mode {
        Some(BlockCodingMode::HighThroughput) => {
            encode_all_ht_code_blocks(&prepared_subbands, accelerator)?
        }
        Some(BlockCodingMode::Classic) => {
            encode_all_tier1_code_blocks(&prepared_subbands, accelerator)?
        }
        None => Vec::new(),
    };

    let mut encoded_iter = encoded_blocks.into_iter();
    let mut precincts = Vec::with_capacity(prepared_subbands.len());
    for subband in prepared_subbands {
        let mut code_blocks = Vec::with_capacity(subband.code_blocks.len());
        for _ in 0..subband.code_blocks.len() {
            let encoded = encoded_iter
                .next()
                .ok_or("encoded code-block count mismatch")?;
            code_blocks.push(CodeBlockPacketData {
                data: encoded.data,
                ht_cleanup_length: if subband.block_coding_mode == BlockCodingMode::HighThroughput {
                    encoded.ht_cleanup_length
                } else {
                    0
                },
                ht_refinement_length: if subband.block_coding_mode
                    == BlockCodingMode::HighThroughput
                {
                    encoded.ht_refinement_length
                } else {
                    0
                },
                num_coding_passes: encoded.num_coding_passes,
                classic_segment_lengths: Vec::new(),
                num_zero_bitplanes: encoded.num_zero_bitplanes,
                previously_included: false,
                l_block: 3,
                block_coding_mode: subband.block_coding_mode,
            });
        }
        precincts.push(SubbandPrecinct {
            code_blocks,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
        });
    }
    if encoded_iter.next().is_some() {
        return Err("encoded code-block count mismatch");
    }

    Ok(precincts)
}

pub(super) fn encode_all_ht_code_blocks(
    prepared_subbands: &[PreparedEncodeSubband],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if prepared_subbands.iter().all(|subband| {
        subband.code_blocks.is_empty() || subband.preencoded_ht_code_blocks.is_some()
    }) {
        let total_blocks = prepared_subbands
            .iter()
            .map(|subband| subband.code_blocks.len())
            .sum();
        let mut encoded = Vec::with_capacity(total_blocks);
        for subband in prepared_subbands {
            if let Some(blocks) = &subband.preencoded_ht_code_blocks {
                if blocks.len() != subband.code_blocks.len() {
                    return Err("preencoded HT subband code-block count mismatch");
                }
                encoded.extend(
                    blocks
                        .iter()
                        .cloned()
                        .map(ht_encoded_code_block_from_accelerator),
                );
            }
        }
        return Ok(encoded);
    }
    if prepared_subbands
        .iter()
        .any(|subband| subband.preencoded_ht_code_blocks.is_some())
    {
        return Err("mixed preencoded and quantized HT subbands are unsupported");
    }

    let job_coefficients = prepared_subbands
        .iter()
        .flat_map(|subband| subband.code_blocks.iter())
        .map(|block| downcast_i64_coefficients_to_i32(&block.coefficients))
        .collect::<Result<Vec<_>, _>>()?;
    let mut jobs = Vec::with_capacity(job_coefficients.len());
    let mut coefficient_idx = 0usize;
    for subband in prepared_subbands {
        for block in &subband.code_blocks {
            let coefficients = job_coefficients
                .get(coefficient_idx)
                .ok_or("HT coefficient storage count mismatch")?;
            jobs.push(crate::J2kHtCodeBlockEncodeJob {
                coefficients,
                width: block.width,
                height: block.height,
                total_bitplanes: subband.total_bitplanes,
                target_coding_passes: subband.ht_target_coding_passes,
            });
            coefficient_idx = coefficient_idx
                .checked_add(1)
                .ok_or("HT coefficient storage count overflow")?;
        }
    }

    if let Some(encoded) = accelerator.encode_ht_code_blocks(&jobs)? {
        if encoded.len() != jobs.len() {
            return Err("accelerated HT code-block batch length mismatch");
        }
        return Ok(encoded
            .into_iter()
            .map(ht_encoded_code_block_from_accelerator)
            .collect());
    }

    if accelerator.prefer_parallel_cpu_code_block_fallback() {
        if jobs.len() < HT_CPU_PARALLEL_FALLBACK_MIN_JOBS {
            return encode_all_ht_code_blocks_serial_cpu(&jobs);
        }
        return encode_all_ht_code_blocks_parallel(&jobs);
    }

    jobs.iter()
        .map(|job| {
            encode_ht_code_block(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
                accelerator,
            )
        })
        .collect()
}

pub(super) fn encode_all_tier1_code_blocks(
    prepared_subbands: &[PreparedEncodeSubband],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    let style = default_public_code_block_style();
    let can_use_i32_jobs = prepared_subbands
        .iter()
        .flat_map(|subband| &subband.code_blocks)
        .all(|block| coefficients_fit_i32(&block.coefficients));
    if !can_use_i32_jobs {
        let mut encoded = Vec::new();
        for subband in prepared_subbands {
            encoded.reserve(subband.code_blocks.len());
            for block in &subband.code_blocks {
                encoded.push(bitplane_encode::encode_code_block_i64(
                    &block.coefficients,
                    block.width,
                    block.height,
                    subband.sub_band_type,
                    subband.total_bitplanes,
                ));
            }
        }
        return Ok(encoded);
    }

    let job_coefficients = prepared_subbands
        .iter()
        .flat_map(|subband| subband.code_blocks.iter())
        .map(|block| downcast_i64_coefficients_to_i32(&block.coefficients))
        .collect::<Result<Vec<_>, _>>()?;
    let mut jobs = Vec::with_capacity(job_coefficients.len());
    let mut coefficient_idx = 0usize;
    for subband in prepared_subbands {
        let public_sub_band_type = public_sub_band_type(subband.sub_band_type);
        for block in &subband.code_blocks {
            let coefficients = job_coefficients
                .get(coefficient_idx)
                .ok_or("classic coefficient storage count mismatch")?;
            jobs.push(J2kTier1CodeBlockEncodeJob {
                coefficients,
                width: block.width,
                height: block.height,
                sub_band_type: public_sub_band_type,
                total_bitplanes: subband.total_bitplanes,
                style,
            });
            coefficient_idx = coefficient_idx
                .checked_add(1)
                .ok_or("classic coefficient storage count overflow")?;
        }
    }

    if let Some(encoded) = accelerator.encode_tier1_code_blocks(&jobs)? {
        if encoded.len() != jobs.len() {
            return Err("accelerated classic code-block batch length mismatch");
        }
        return Ok(encoded
            .into_iter()
            .map(encoded_code_block_from_accelerator)
            .collect());
    }

    if accelerator.prefer_parallel_cpu_code_block_fallback() {
        return encode_all_tier1_code_blocks_parallel(&jobs);
    }

    let mut encoded = Vec::with_capacity(jobs.len());
    for job in &jobs {
        encoded.push(encode_tier1_code_block(
            job.coefficients,
            job.width,
            job.height,
            internal_sub_band_type(job.sub_band_type),
            job.total_bitplanes,
            accelerator,
        )?);
    }
    Ok(encoded)
}

pub(super) fn encode_all_ht_code_blocks_serial_cpu(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if jobs
        .iter()
        .any(|job| !(1..=3).contains(&job.target_coding_passes))
    {
        return Err("CPU HTJ2K code-block fallback supports at most three HT coding passes");
    }
    jobs.iter()
        .map(|job| {
            ht_block_encode::encode_code_block_with_passes(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
            )
        })
        .collect()
}

#[cfg(feature = "parallel")]
pub(super) fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if jobs
        .iter()
        .any(|job| !(1..=3).contains(&job.target_coding_passes))
    {
        return Err("CPU HTJ2K code-block fallback supports at most three HT coding passes");
    }
    jobs.par_iter()
        .map(|job| {
            ht_block_encode::encode_code_block_with_passes(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
            )
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
pub(super) fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if jobs
        .iter()
        .any(|job| !(1..=3).contains(&job.target_coding_passes))
    {
        return Err("CPU HTJ2K code-block fallback supports at most three HT coding passes");
    }
    jobs.iter()
        .map(|job| {
            ht_block_encode::encode_code_block_with_passes(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
            )
        })
        .collect()
}

#[cfg(feature = "parallel")]
pub(super) fn encode_all_tier1_code_blocks_parallel(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    jobs.par_iter()
        .map(|job| {
            Ok(bitplane_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                internal_sub_band_type(job.sub_band_type),
                job.total_bitplanes,
            ))
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
pub(super) fn encode_all_tier1_code_blocks_parallel(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    jobs.iter()
        .map(|job| {
            Ok(bitplane_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                internal_sub_band_type(job.sub_band_type),
                job.total_bitplanes,
            ))
        })
        .collect()
}

pub(super) fn encode_ht_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    target_coding_passes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bitplane_encode::EncodedCodeBlock, &'static str> {
    if let Some(encoded) = accelerator.encode_ht_code_block(crate::J2kHtCodeBlockEncodeJob {
        coefficients,
        width,
        height,
        total_bitplanes,
        target_coding_passes,
    })? {
        return Ok(ht_encoded_code_block_from_accelerator(encoded));
    }

    ht_block_encode::encode_code_block_with_passes(
        coefficients,
        width,
        height,
        total_bitplanes,
        target_coding_passes,
    )
}

pub(super) fn ht_encoded_code_block_from_accelerator(
    encoded: crate::EncodedHtJ2kCodeBlock,
) -> bitplane_encode::EncodedCodeBlock {
    bitplane_encode::EncodedCodeBlock {
        data: encoded.data,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
        ht_cleanup_length: encoded.cleanup_length,
        ht_refinement_length: encoded.refinement_length,
    }
}

pub(super) fn encode_tier1_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bitplane_encode::EncodedCodeBlock, &'static str> {
    if let Some(encoded) = accelerator.encode_tier1_code_block(J2kTier1CodeBlockEncodeJob {
        coefficients,
        width,
        height,
        sub_band_type: public_sub_band_type(sub_band_type),
        total_bitplanes,
        style: default_public_code_block_style(),
    })? {
        return Ok(encoded_code_block_from_accelerator(encoded));
    }

    Ok(bitplane_encode::encode_code_block(
        coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
    ))
}

pub(super) fn encoded_code_block_from_accelerator(
    encoded: EncodedJ2kCodeBlock,
) -> bitplane_encode::EncodedCodeBlock {
    bitplane_encode::EncodedCodeBlock {
        data: encoded.data,
        num_coding_passes: encoded.number_of_coding_passes,
        num_zero_bitplanes: encoded.missing_bit_planes,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
    }
}
