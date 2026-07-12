// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only Tier-1 orchestration under one retained allocation phase.

use super::allocation::checked_add_bytes;
use super::tier1_allocation::{
    prepared_subbands_ownership, public_classic_blocks_ownership, public_ht_blocks_ownership,
    subband_precincts_ownership, Tier1PhaseTracker,
};
#[cfg(test)]
use super::NativeEncodeRetainedInput;
use super::{
    bitplane_encode, default_public_code_block_style, ht_block_encode, internal_sub_band_type,
    public_sub_band_type, BlockCodingMode, J2kEncodeStageAccelerator, J2kTier1CodeBlockEncodeJob,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
    PreparedCodeBlockCoefficients, PreparedEncodeSubband, SubbandPrecinct, Vec,
};

mod cpu;
use cpu::{
    coefficients_fit_i32, encode_classic_cpu_results_accounted, encode_classic_i64_direct,
    encode_ht_cpu_results_accounted,
};
#[cfg(test)]
pub(super) use cpu::{encode_all_ht_code_blocks_parallel, encode_all_ht_code_blocks_serial_cpu};
mod output;
use output::{
    ht_encoded_code_block_from_accelerator, move_native_result_iter, move_public_classic_outputs,
    move_public_ht_outputs, push_packet_block, validate_classic_batch_outputs,
    validate_ht_batch_outputs, validated_classic_output, validated_ht_output,
};
mod layout;
use layout::{consistent_block_coding_mode, try_packet_shells};
mod scratch;
use scratch::{check_classic_wave, check_ht_wave};

#[cfg(test)]
pub(super) fn encode_prepared_subbands(
    prepared_subbands: Vec<PreparedEncodeSubband>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<SubbandPrecinct>> {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())?;
    encode_prepared_subbands_for_session(prepared_subbands, &session, 0, accelerator)
        .map_err(NativeEncodePipelineError::into_encode_error)
}

pub(super) fn encode_prepared_subbands_for_session(
    prepared_subbands: Vec<PreparedEncodeSubband>,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<SubbandPrecinct>> {
    encode_prepared_subbands_accounted(prepared_subbands, session, retained_base_bytes, accelerator)
        .map(|outcome| outcome.precincts)
}

struct Tier1EncodeOutcome {
    precincts: Vec<SubbandPrecinct>,
    #[cfg(test)]
    peak_phase_bytes: usize,
}

fn encode_prepared_subbands_accounted(
    mut prepared_subbands: Vec<PreparedEncodeSubband>,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Tier1EncodeOutcome> {
    let prepared = prepared_subbands_ownership(&prepared_subbands, prepared_subbands.capacity())?;
    let prepared_bytes = prepared.total()?;
    let mut tracker = Tier1PhaseTracker::new(session, retained_base_bytes);
    tracker.check([prepared_bytes], "prepared Tier-1 owners")?;

    let (mut precincts, packet_structural_bytes) =
        try_packet_shells(&prepared_subbands, prepared_bytes, &mut tracker)?;
    let block_coding_mode = consistent_block_coding_mode(&prepared_subbands)
        .map_err(NativeEncodePipelineError::unsupported)?;
    let all_preencoded = prepared_subbands.iter().all(|subband| {
        subband.code_blocks.is_empty() || subband.preencoded_ht_code_blocks.is_some()
    });
    let any_preencoded = prepared_subbands
        .iter()
        .any(|subband| subband.preencoded_ht_code_blocks.is_some());

    if all_preencoded && any_preencoded {
        move_preencoded_ht_blocks(
            &mut prepared_subbands,
            &mut precincts,
            prepared_bytes,
            packet_structural_bytes,
            &mut tracker,
        )?;
    } else {
        if any_preencoded {
            return Err(NativeEncodePipelineError::unsupported(
                "mixed preencoded and quantized HT subbands are unsupported",
            ));
        }
        match block_coding_mode {
            Some(BlockCodingMode::HighThroughput) => encode_ht_subbands(
                &prepared_subbands,
                &mut precincts,
                prepared_bytes,
                packet_structural_bytes,
                &mut tracker,
                accelerator,
            )?,
            Some(BlockCodingMode::Classic) => encode_classic_subbands(
                &prepared_subbands,
                &mut precincts,
                prepared_bytes,
                packet_structural_bytes,
                &mut tracker,
                accelerator,
            )?,
            None => {}
        }
    }

    drop(prepared_subbands);
    let packet_bytes = subband_precincts_ownership(&precincts, precincts.capacity())?;
    tracker.check([packet_bytes], "completed Tier-1 packet owners")?;
    Ok(Tier1EncodeOutcome {
        precincts,
        #[cfg(test)]
        peak_phase_bytes: tracker.peak_phase_bytes(),
    })
}

fn move_preencoded_ht_blocks(
    prepared_subbands: &mut [PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
    prepared_bytes: usize,
    packet_structural_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    tracker.check(
        [prepared_bytes, packet_structural_bytes],
        "preencoded HT Tier-1 handoff",
    )?;
    for (subband, precinct) in prepared_subbands.iter_mut().zip(precincts) {
        let Some(encoded_blocks) = subband.preencoded_ht_code_blocks.take() else {
            if subband.code_blocks.is_empty() {
                continue;
            }
            return Err(NativeEncodePipelineError::internal_invariant(
                "preencoded HT subband payload is missing",
            ));
        };
        if encoded_blocks.len() != subband.code_blocks.len() {
            return Err(NativeEncodePipelineError::internal_invariant(
                "preencoded HT subband code-block count mismatch",
            ));
        }
        for encoded in encoded_blocks {
            push_packet_block(
                precinct,
                ht_encoded_code_block_from_accelerator(encoded),
                BlockCodingMode::HighThroughput,
            )?;
        }
    }
    Ok(())
}

fn encode_ht_subbands(
    prepared_subbands: &[PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
    prepared_bytes: usize,
    packet_structural_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    let (downcast, downcast_bytes) = try_downcast_i64_coefficients(
        prepared_subbands,
        true,
        prepared_bytes,
        packet_structural_bytes,
        tracker,
    )?;
    let (jobs, job_bytes) = try_ht_jobs(
        prepared_subbands,
        &downcast,
        prepared_bytes,
        packet_structural_bytes,
        downcast_bytes,
        tracker,
    )?;
    let fixed = [
        prepared_bytes,
        packet_structural_bytes,
        downcast_bytes,
        job_bytes,
    ];

    if let Some(encoded) = accelerator.encode_ht_code_blocks(&jobs).map_err(|source| {
        crate::EncodeError::Accelerator {
            operation: "HT Tier-1 code-block batch encode",
            source,
        }
    })? {
        validate_ht_batch_outputs(&encoded, &jobs)?;
        let encoded_bytes = public_ht_blocks_ownership(&encoded, encoded.capacity())?;
        tracker.check(
            fixed.into_iter().chain([encoded_bytes]),
            "accelerated HT Tier-1 output",
        )?;
        move_public_ht_outputs(encoded, prepared_subbands, precincts)?;
        return Ok(());
    }

    if accelerator.prefer_parallel_cpu_code_block_fallback() {
        let encoded = encode_ht_cpu_results_accounted(&jobs, tracker, fixed)?;
        move_native_result_iter(
            encoded.into_iter().map(|slot| {
                slot.unwrap_or(Err(crate::EncodeError::InternalInvariant {
                    what: "HT Tier-1 worker result is missing",
                }))
            }),
            prepared_subbands,
            precincts,
            BlockCodingMode::HighThroughput,
        )?;
        return Ok(());
    }

    encode_ht_serial(
        &jobs,
        prepared_subbands,
        precincts,
        fixed,
        tracker,
        accelerator,
    )
}

fn encode_ht_serial(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    prepared_subbands: &[PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
    fixed: [usize; 4],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    let mut packet_payload_bytes = 0usize;
    let mut job_index = 0usize;
    for (subband, precinct) in prepared_subbands.iter().zip(precincts) {
        for _block in &subband.code_blocks {
            let job = jobs.get(job_index).ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant("HT Tier-1 job count mismatch")
            })?;
            let wave_fixed = [fixed[0], fixed[1], fixed[2], fixed[3], packet_payload_bytes];
            check_ht_wave(core::slice::from_ref(job), tracker, &wave_fixed, 1)?;
            let encoded = encode_ht_code_block_typed(job, accelerator)?;
            packet_payload_bytes = checked_add_bytes(
                packet_payload_bytes,
                encoded.data.capacity(),
                "HT Tier-1 packet payload",
            )?;
            tracker.check(
                fixed.into_iter().chain([packet_payload_bytes]),
                "serial HT Tier-1 packet output",
            )?;
            push_packet_block(precinct, encoded, subband.block_coding_mode)?;
            job_index = job_index
                .checked_add(1)
                .ok_or(crate::EncodeError::ArithmeticOverflow {
                    what: "HT Tier-1 job index",
                })?;
        }
    }
    if job_index != jobs.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HT Tier-1 job count mismatch",
        ));
    }
    Ok(())
}

fn encode_classic_subbands(
    prepared_subbands: &[PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
    prepared_bytes: usize,
    packet_structural_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    if classic_requires_direct_i64(prepared_subbands) {
        return encode_classic_i64_direct(
            prepared_subbands,
            precincts,
            prepared_bytes,
            packet_structural_bytes,
            tracker,
        );
    }

    let (downcast, downcast_bytes) = try_downcast_i64_coefficients(
        prepared_subbands,
        false,
        prepared_bytes,
        packet_structural_bytes,
        tracker,
    )?;
    let (jobs, job_bytes) = try_classic_jobs(
        prepared_subbands,
        &downcast,
        prepared_bytes,
        packet_structural_bytes,
        downcast_bytes,
        tracker,
    )?;
    let fixed = [
        prepared_bytes,
        packet_structural_bytes,
        downcast_bytes,
        job_bytes,
    ];

    if let Some(encoded) = accelerator
        .encode_tier1_code_blocks(&jobs)
        .map_err(|source| crate::EncodeError::Accelerator {
            operation: "classic Tier-1 code-block batch encode",
            source,
        })?
    {
        validate_classic_batch_outputs(&encoded, &jobs)?;
        let encoded_bytes = public_classic_blocks_ownership(&encoded, encoded.capacity())?;
        tracker.check(
            fixed.into_iter().chain([encoded_bytes]),
            "accelerated classic Tier-1 output",
        )?;
        move_public_classic_outputs(encoded, prepared_subbands, precincts)?;
        return Ok(());
    }

    if accelerator.prefer_parallel_cpu_code_block_fallback() {
        let encoded = encode_classic_cpu_results_accounted(&jobs, tracker, fixed)?;
        move_native_result_iter(
            encoded.into_iter().map(|slot| {
                slot.unwrap_or(Err(crate::EncodeError::InternalInvariant {
                    what: "classic Tier-1 worker result is missing",
                }))
            }),
            prepared_subbands,
            precincts,
            BlockCodingMode::Classic,
        )?;
        return Ok(());
    }

    encode_classic_serial(
        &jobs,
        prepared_subbands,
        precincts,
        fixed,
        tracker,
        accelerator,
    )
}

fn encode_classic_serial(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    prepared_subbands: &[PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
    fixed: [usize; 4],
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    let mut packet_payload_bytes = 0usize;
    let mut job_index = 0usize;
    for (subband, precinct) in prepared_subbands.iter().zip(precincts) {
        for _block in &subband.code_blocks {
            let job = jobs.get(job_index).ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant("classic Tier-1 job count mismatch")
            })?;
            let wave_fixed = [fixed[0], fixed[1], fixed[2], fixed[3], packet_payload_bytes];
            check_classic_wave(core::slice::from_ref(job), tracker, &wave_fixed, 1)?;
            let encoded = encode_tier1_code_block_accounted(
                job,
                accelerator,
                tracker,
                fixed,
                packet_payload_bytes,
            )?;
            packet_payload_bytes = checked_add_bytes(
                packet_payload_bytes,
                encoded.data.capacity(),
                "classic Tier-1 packet payload",
            )?;
            tracker.check(
                fixed.into_iter().chain([packet_payload_bytes]),
                "serial classic Tier-1 packet output",
            )?;
            push_packet_block(precinct, encoded, subband.block_coding_mode)?;
            job_index = job_index
                .checked_add(1)
                .ok_or(crate::EncodeError::ArithmeticOverflow {
                    what: "classic Tier-1 job index",
                })?;
        }
    }
    if job_index != jobs.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "classic Tier-1 job count mismatch",
        ));
    }
    Ok(())
}

fn classic_requires_direct_i64(prepared_subbands: &[PreparedEncodeSubband]) -> bool {
    prepared_subbands.iter().any(|subband| {
        subband.code_blocks.iter().any(|block| {
            matches!(
                &block.coefficients,
                PreparedCodeBlockCoefficients::I64(values) if !coefficients_fit_i32(values)
            )
        })
    })
}

fn try_downcast_i64_coefficients(
    prepared_subbands: &[PreparedEncodeSubband],
    ht_mode: bool,
    prepared_bytes: usize,
    packet_structural_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<(Vec<Vec<i32>>, usize)> {
    let i64_count = prepared_subbands
        .iter()
        .flat_map(|subband| &subband.code_blocks)
        .filter(|block| matches!(&block.coefficients, PreparedCodeBlockCoefficients::I64(_)))
        .count();
    let (mut downcast, outer_bytes) = tracker.try_vec::<Vec<i32>>(
        i64_count,
        [prepared_bytes, packet_structural_bytes],
        "Tier-1 downcast coefficient owners",
    )?;
    let mut downcast_bytes = outer_bytes;
    for block in prepared_subbands
        .iter()
        .flat_map(|subband| &subband.code_blocks)
    {
        let PreparedCodeBlockCoefficients::I64(values) = &block.coefficients else {
            continue;
        };
        if ht_mode && !coefficients_fit_i32(values) {
            return Err(NativeEncodePipelineError::unsupported(
                "HTJ2K/accelerated code-block encode does not support i64 coefficients",
            ));
        }
        let (mut converted, converted_bytes) = tracker.try_vec::<i32>(
            values.len(),
            [prepared_bytes, packet_structural_bytes, downcast_bytes],
            "Tier-1 downcast coefficients",
        )?;
        for &value in values {
            converted.push(i32::try_from(value).map_err(|_| {
                NativeEncodePipelineError::unsupported(
                    "HTJ2K/accelerated code-block encode does not support i64 coefficients",
                )
            })?);
        }
        downcast_bytes = checked_add_bytes(
            downcast_bytes,
            converted_bytes,
            "Tier-1 downcast coefficient graph",
        )?;
        downcast.push(converted);
    }
    Ok((downcast, downcast_bytes))
}

fn try_ht_jobs<'a>(
    prepared_subbands: &'a [PreparedEncodeSubband],
    downcast: &'a [Vec<i32>],
    prepared_bytes: usize,
    packet_structural_bytes: usize,
    downcast_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<(Vec<crate::J2kHtCodeBlockEncodeJob<'a>>, usize)> {
    let block_count = total_block_count(prepared_subbands)?;
    let (mut jobs, job_bytes) = tracker.try_vec::<crate::J2kHtCodeBlockEncodeJob<'_>>(
        block_count,
        [prepared_bytes, packet_structural_bytes, downcast_bytes],
        "HT Tier-1 job descriptors",
    )?;
    let mut downcast_iter = downcast.iter();
    for subband in prepared_subbands {
        for block in &subband.code_blocks {
            let coefficients = job_coefficients(block, &mut downcast_iter)
                .map_err(NativeEncodePipelineError::internal_invariant)?;
            jobs.push(crate::J2kHtCodeBlockEncodeJob {
                coefficients,
                width: block.width,
                height: block.height,
                total_bitplanes: subband.total_bitplanes,
                target_coding_passes: subband.ht_target_coding_passes,
            });
        }
    }
    if downcast_iter.next().is_some() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HT coefficient storage count mismatch",
        ));
    }
    Ok((jobs, job_bytes))
}

fn try_classic_jobs<'a>(
    prepared_subbands: &'a [PreparedEncodeSubband],
    downcast: &'a [Vec<i32>],
    prepared_bytes: usize,
    packet_structural_bytes: usize,
    downcast_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<(Vec<J2kTier1CodeBlockEncodeJob<'a>>, usize)> {
    let block_count = total_block_count(prepared_subbands)?;
    let (mut jobs, job_bytes) = tracker.try_vec::<J2kTier1CodeBlockEncodeJob<'_>>(
        block_count,
        [prepared_bytes, packet_structural_bytes, downcast_bytes],
        "classic Tier-1 job descriptors",
    )?;
    let style = default_public_code_block_style();
    let mut downcast_iter = downcast.iter();
    for subband in prepared_subbands {
        let sub_band_type = public_sub_band_type(subband.sub_band_type);
        for block in &subband.code_blocks {
            let coefficients = job_coefficients(block, &mut downcast_iter)
                .map_err(NativeEncodePipelineError::internal_invariant)?;
            jobs.push(J2kTier1CodeBlockEncodeJob {
                coefficients,
                width: block.width,
                height: block.height,
                sub_band_type,
                total_bitplanes: subband.total_bitplanes,
                style,
            });
        }
    }
    if downcast_iter.next().is_some() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "classic coefficient storage count mismatch",
        ));
    }
    Ok((jobs, job_bytes))
}

fn job_coefficients<'a>(
    block: &'a super::PreparedEncodeCodeBlock,
    downcast: &mut impl Iterator<Item = &'a Vec<i32>>,
) -> Result<&'a [i32], &'static str> {
    match &block.coefficients {
        PreparedCodeBlockCoefficients::I32(values) => Ok(values),
        PreparedCodeBlockCoefficients::I64(_) => downcast
            .next()
            .map(Vec::as_slice)
            .ok_or("Tier-1 downcast coefficient storage count mismatch"),
        PreparedCodeBlockCoefficients::Empty => Err("Tier-1 coefficient storage is missing"),
    }
}

fn total_block_count(
    prepared_subbands: &[PreparedEncodeSubband],
) -> Result<usize, crate::EncodeError> {
    prepared_subbands.iter().try_fold(0usize, |count, subband| {
        count
            .checked_add(subband.code_blocks.len())
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "Tier-1 code-block count",
            })
    })
}

fn encode_ht_code_block_typed(
    job: &crate::J2kHtCodeBlockEncodeJob<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<bitplane_encode::EncodedCodeBlock> {
    if let Some(encoded) = accelerator.encode_ht_code_block(*job).map_err(|source| {
        crate::EncodeError::Accelerator {
            operation: "HT Tier-1 code-block encode",
            source,
        }
    })? {
        return validated_ht_output(encoded, job);
    }
    Ok(ht_block_encode::try_encode_code_block_with_passes(
        job.coefficients,
        job.width,
        job.height,
        job.total_bitplanes,
        job.target_coding_passes,
    )?)
}

fn encode_tier1_code_block_accounted(
    job: &J2kTier1CodeBlockEncodeJob<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    fixed: [usize; 4],
    retained_packet_payload_bytes: usize,
) -> NativeEncodePipelineResult<bitplane_encode::EncodedCodeBlock> {
    if let Some(encoded) = accelerator
        .encode_tier1_code_block(*job)
        .map_err(|source| crate::EncodeError::Accelerator {
            operation: "classic Tier-1 code-block encode",
            source,
        })?
    {
        let public_output_bytes =
            public_classic_blocks_ownership(core::slice::from_ref(&encoded), 0)?;
        tracker.check(
            fixed
                .into_iter()
                .chain([retained_packet_payload_bytes, public_output_bytes]),
            "serial accelerated classic Tier-1 output",
        )?;
        return validated_classic_output(encoded, job);
    }
    Ok(bitplane_encode::try_encode_code_block(
        job.coefficients,
        job.width,
        job.height,
        internal_sub_band_type(job.sub_band_type),
        job.total_bitplanes,
    )?)
}

#[cfg(test)]
mod tests;
