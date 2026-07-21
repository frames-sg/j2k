// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use j2k_metal_support::{dispatch_1d_pipeline, dispatch_single_thread};

use crate::profile_env::{label_command_buffer, label_compute_encoder};

use super::abi::{
    J2kClassicEncodeBatchJob, J2kClassicEncodeParams, J2kClassicEncodeStatus, J2kClassicSegment,
    J2kHtEncodeBatchJob, J2kHtEncodeParams, J2kHtEncodeStatus, J2kPacketEncodeStatus,
    J2K_ENCODE_STATUS_FAIL, J2K_ENCODE_STATUS_OK, J2K_ENCODE_STATUS_UNSUPPORTED,
};
use super::resident_tier1::{J2kResidentLosslessHtCodeBlocks, J2kResidentLosslessTier1CodeBlocks};
use super::{
    checked_buffer_read, checked_buffer_slice, checked_buffer_slice_at,
    classic_encode_code_blocks_pipeline, classic_encode_output_capacity,
    classic_encode_segment_capacity, classic_style_flags, commit_and_wait_metal,
    copied_slice_buffer, ht_encode_output_capacity, new_blit_command_encoder, new_command_buffer,
    new_compute_command_encoder, new_private_buffer, new_shared_buffer, with_runtime,
    with_runtime_for_session, zeroed_shared_buffer, Buffer, EncodedHtJ2kCodeBlock,
    EncodedJ2kCodeBlock, Error, J2kCodeBlockSegment, J2kHtCodeBlockEncodeJob,
    J2kLosslessDeviceCodeBlock, J2kPreparedLosslessDeviceCodeBlocks, J2kTier1CodeBlockEncodeJob,
    MetalRuntime,
};

fn checked_type_buffer_bytes<T>(count: usize, context: &'static str) -> Result<usize, Error> {
    count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::MetalKernel {
            message: format!("{context} byte size overflow"),
        })
}
#[cfg(test)]
mod test_support;
#[cfg(test)]
pub(crate) use self::test_support::{
    encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test,
    encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test,
    encode_classic_tier1_code_blocks_via_split_mq_byte_raw_tokens_gpu_pack_for_test,
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test,
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test,
};

#[cfg(target_os = "macos")]
pub(super) fn encode_status_error(stage: &str, code: u32, detail: u32) -> Error {
    let kind = match code {
        J2K_ENCODE_STATUS_FAIL => "failure",
        J2K_ENCODE_STATUS_UNSUPPORTED => "unsupported input",
        _ => "unexpected status",
    };
    let message = format!("{stage} Metal encode kernel {kind} (detail={detail})");
    if let Some(retry_class) = encode_status_retry_class(stage, code, detail) {
        Error::MetalKernelRetryable {
            message,
            retry_class,
        }
    } else {
        Error::MetalKernel { message }
    }
}

#[cfg(target_os = "macos")]
pub(super) fn encode_status_retry_class(
    stage: &str,
    code: u32,
    detail: u32,
) -> Option<crate::MetalKernelRetryClass> {
    use crate::MetalKernelRetryClass::{
        ResidentClassicBatch, ResidentClassicOrHtBatch, ResidentHtBatch,
    };

    if code != J2K_ENCODE_STATUS_FAIL {
        return None;
    }
    match (stage, detail) {
        ("classic Tier-1", 4 | 5) | ("J2K batched codestream assembly", 2 | 3) => {
            Some(ResidentClassicBatch)
        }
        ("packetization", 3..=5) => Some(ResidentClassicOrHtBatch),
        ("HTJ2K batched codestream assembly", 2 | 3) => Some(ResidentHtBatch),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
pub(super) fn packet_encode_status_error(status: J2kPacketEncodeStatus) -> Error {
    if status.code == J2K_ENCODE_STATUS_FAIL && status.detail == 7 {
        let message = format!(
            "packetization Metal encode kernel failure (detail=7, tier1_detail={})",
            status.data_len
        );
        return if matches!(status.data_len, 4 | 5) {
            Error::MetalKernelRetryable {
                message,
                retry_class: crate::MetalKernelRetryClass::ResidentClassicBatch,
            }
        } else {
            Error::MetalKernel { message }
        };
    }
    encode_status_error("packetization", status.code, status.detail)
}

pub(super) fn classic_encode_sub_band_code(sub_band_type: j2k_native::J2kSubBandType) -> u32 {
    match sub_band_type {
        j2k_native::J2kSubBandType::LowLow => 0,
        j2k_native::J2kSubBandType::HighLow => 1,
        j2k_native::J2kSubBandType::LowHigh => 2,
        j2k_native::J2kSubBandType::HighHigh => 3,
    }
}

#[cfg(target_os = "macos")]
fn project_classic_encode_segments(
    raw_segments: &[J2kClassicSegment],
) -> Result<Vec<J2kCodeBlockSegment>, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "classic J2K Metal encoded segment metadata",
    );
    let mut segments = budget.try_vec(
        raw_segments.len(),
        "classic J2K Metal encoded segment metadata",
    )?;
    for segment in raw_segments {
        segments.push(J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: u8::try_from(segment.start_coding_pass).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal encode segment start pass exceeds u8".to_string(),
                }
            })?,
            end_coding_pass: u8::try_from(segment.end_coding_pass).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal encode segment end pass exceeds u8".to_string(),
                }
            })?,
            use_arithmetic: segment.use_arithmetic != 0,
        });
    }
    Ok(segments)
}

#[cfg(target_os = "macos")]
pub(super) fn read_classic_encoded_code_block(
    status: J2kClassicEncodeStatus,
    output: &Buffer,
    output_offset: usize,
    output_capacity: usize,
    segment_buffer: &Buffer,
    segment_offset: usize,
    segment_capacity: usize,
) -> Result<EncodedJ2kCodeBlock, Error> {
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "classic Tier-1",
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode length exceeds usize".to_string(),
    })?;
    let payload_skip = usize::try_from(status.reserved0).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode payload skip exceeds usize".to_string(),
    })?;
    let number_of_coding_passes =
        u8::try_from(status.number_of_coding_passes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode pass count exceeds u8".to_string(),
        })?;
    let missing_bit_planes =
        u8::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode missing bitplanes exceeds u8".to_string(),
        })?;
    let segment_count = usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode segment count exceeds usize".to_string(),
    })?;
    if segment_count > segment_capacity {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal encode segment count exceeds buffer".to_string(),
        });
    }
    let raw_segments = if segment_count == 0 {
        Vec::new()
    } else {
        checked_buffer_slice_at::<J2kClassicSegment>(
            segment_buffer,
            segment_offset
                .checked_mul(size_of::<J2kClassicSegment>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode segment offset overflow".to_string(),
                })?,
            segment_count,
            "classic encode segments",
        )?
    };
    let data = if data_len == 0 {
        Vec::new()
    } else {
        let payload_span =
            data_len
                .checked_add(payload_skip)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode payload span overflow".to_string(),
                })?;
        if payload_span > output_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let payload_offset =
            output_offset
                .checked_add(payload_skip)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode payload offset overflow".to_string(),
                })?;
        checked_buffer_slice_at::<u8>(output, payload_offset, data_len, "classic encode payload")?
    };
    let segments = project_classic_encode_segments(&raw_segments)?;

    Ok(EncodedJ2kCodeBlock {
        data,
        segments,
        number_of_coding_passes,
        missing_bit_planes,
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "batched Tier-1 submission keeps buffer slices and status ordering aligned"
)]
pub(crate) fn encode_classic_tier1_code_blocks(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let coefficient_count = jobs.iter().try_fold(0usize, |total, job| {
            let width = usize::try_from(job.width).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode width exceeds usize".to_string(),
            })?;
            let height = usize::try_from(job.height).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode height exceeds usize".to_string(),
            })?;
            let count = width
                .checked_mul(height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient count overflow".to_string(),
                })?;
            total.checked_add(count).ok_or_else(|| {
                Error::from(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                    what: "classic J2K Metal encode coefficients",
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })
            })
        })?;
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "classic J2K Metal Tier-1 encode batch",
        );
        let mut coefficients =
            budget.try_vec(coefficient_count, "classic J2K Metal encode coefficients")?;
        let mut batch_jobs = budget.try_vec(jobs.len(), "classic J2K Metal encode batch jobs")?;
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient slice is too small".to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient table exceeds u32".to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment table exceeds u32".to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode output capacity exceeds u32".to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode segment buffer overflow".to_string(),
                })?;
        }

        let coefficient_buffer = copied_slice_buffer(&runtime.device, &coefficients)?;
        let job_buffer = copied_slice_buffer(&runtime.device, &batch_jobs)?;
        let output = new_shared_buffer(&runtime.device, output_capacity_total.max(1))?;
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kClassicEncodeStatus>(
                jobs.len(),
                "classic J2K Metal encode status buffer",
            )?,
        )?;
        let segment_buffer = new_shared_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kClassicSegment>(
                segment_capacity_total,
                "classic J2K Metal encode segment buffer",
            )?,
        )?;
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        let classic_encode_pipeline = classic_encode_code_blocks_pipeline(runtime, &batch_jobs);
        encoder.set_compute_pipeline_state(classic_encode_pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(&encoder, classic_encode_pipeline, u64::from(job_count));
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;

        let statuses = checked_buffer_slice::<J2kClassicEncodeStatus>(
            &status_buffer,
            jobs.len(),
            "classic encode statuses",
        )?;
        let mut results = budget.try_vec(jobs.len(), "J2K Metal classic Tier-1 encoded blocks")?;
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output capacity exceeds usize".to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "resident Tier-1 submission keeps command bindings and retained buffers ordered"
)]
pub(crate) fn encode_classic_tier1_prepared_device_code_blocks_resident(
    session: &crate::MetalBackendSession,
    prepared: J2kPreparedLosslessDeviceCodeBlocks,
) -> Result<J2kResidentLosslessTier1CodeBlocks, Error> {
    let J2kPreparedLosslessDeviceCodeBlocks {
        coefficient_buffer,
        coefficient_byte_offset: _,
        coefficient_byte_len: _,
        coefficient_buffer_is_batch_shared: _,
        code_blocks,
        recyclable_private_buffers: _,
        _prepare_command_buffer: prepare_command_buffer,
        _prepare_deinterleave_rct_command_buffer: _,
        _prepare_dwt53_command_buffer: _,
        _prepare_dwt53_vertical_command_buffers: _,
        _prepare_dwt53_horizontal_command_buffers: _,
        _prepare_coefficient_extract_command_buffer: _,
        _deinterleave_status_buffer: deinterleave_status_buffer,
        _plane_buffers: plane_buffers,
        _scratch_buffers: scratch_buffers,
        _coefficient_job_buffer: coefficient_job_buffer,
    } = prepared;
    with_runtime_for_session(session, |runtime| {
        if code_blocks.is_empty() {
            let output = new_private_buffer(&runtime.device, 1)?;
            let status_buffer = zeroed_shared_buffer(&runtime.device, 1)?;
            let segment_buffer = new_private_buffer(&runtime.device, 1)?;
            let job_buffer = new_shared_buffer(&runtime.device, 1)?;
            let command_buffer = new_command_buffer(&runtime.queue)?;
            command_buffer.commit();
            return Ok(J2kResidentLosslessTier1CodeBlocks {
                output_buffer: output,
                status_buffer,
                job_buffer,
                batch_jobs: Vec::new(),
                code_blocks,
                output_capacity_total: 0,
                _segment_buffer: segment_buffer,
                tier1_command_buffer: command_buffer,
                _coefficient_buffer: coefficient_buffer,
                prepare_command_buffer,
                _deinterleave_status_buffer: deinterleave_status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal resident classic Tier-1 encode jobs",
        );
        let mut batch_jobs = budget.try_vec(
            code_blocks.len(),
            "J2K Metal resident classic Tier-1 encode jobs",
        )?;
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for block in &code_blocks {
            let output_capacity =
                classic_encode_output_capacity(block.width, block.height, block.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal resident encode output table exceeds u32"
                        .to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal resident encode segment table exceeds u32"
                        .to_string(),
                })?;
            let style_flags = 0;
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, block.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset: block.coefficient_offset,
                output_offset,
                segment_offset,
                width: block.width,
                height: block.height,
                sub_band_type: classic_encode_sub_band_code(block.sub_band_type),
                total_bitplanes: u32::from(block.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal resident encode output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal resident encode segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal resident encode output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal resident encode segment buffer overflow"
                        .to_string(),
                })?;
        }

        let job_buffer = copied_slice_buffer(&runtime.device, &batch_jobs)?;
        let output = new_private_buffer(&runtime.device, output_capacity_total.max(1))?;
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kClassicEncodeStatus>(
                batch_jobs.len(),
                "classic J2K Metal resident status buffer",
            )?,
        )?;
        let segment_buffer = new_private_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kClassicSegment>(
                segment_capacity_total,
                "classic J2K Metal resident segment buffer",
            )?,
        )?;
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal resident encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        let classic_encode_pipeline = classic_encode_code_blocks_pipeline(runtime, &batch_jobs);
        encoder.set_compute_pipeline_state(classic_encode_pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(&encoder, classic_encode_pipeline, u64::from(job_count));
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kResidentLosslessTier1CodeBlocks {
            output_buffer: output,
            status_buffer,
            job_buffer,
            batch_jobs,
            code_blocks,
            output_capacity_total,
            _segment_buffer: segment_buffer,
            tier1_command_buffer: command_buffer,
            _coefficient_buffer: coefficient_buffer,
            prepare_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "resident HT submission keeps command bindings and retained buffers ordered"
)]
pub(crate) fn encode_ht_prepared_device_code_blocks_resident(
    session: &crate::MetalBackendSession,
    prepared: J2kPreparedLosslessDeviceCodeBlocks,
) -> Result<J2kResidentLosslessHtCodeBlocks, Error> {
    let J2kPreparedLosslessDeviceCodeBlocks {
        coefficient_buffer,
        coefficient_byte_offset: _,
        coefficient_byte_len: _,
        coefficient_buffer_is_batch_shared: _,
        code_blocks,
        recyclable_private_buffers: _,
        _prepare_command_buffer: prepare_command_buffer,
        _prepare_deinterleave_rct_command_buffer: _,
        _prepare_dwt53_command_buffer: _,
        _prepare_dwt53_vertical_command_buffers: _,
        _prepare_dwt53_horizontal_command_buffers: _,
        _prepare_coefficient_extract_command_buffer: _,
        _deinterleave_status_buffer: deinterleave_status_buffer,
        _plane_buffers: plane_buffers,
        _scratch_buffers: scratch_buffers,
        _coefficient_job_buffer: coefficient_job_buffer,
    } = prepared;
    with_runtime_for_session(session, |runtime| {
        if code_blocks.is_empty() {
            let output = new_private_buffer(&runtime.device, 1)?;
            let status_buffer = zeroed_shared_buffer(&runtime.device, 1)?;
            let job_buffer = new_shared_buffer(&runtime.device, 1)?;
            let command_buffer = new_command_buffer(&runtime.queue)?;
            command_buffer.commit();
            return Ok(J2kResidentLosslessHtCodeBlocks {
                output_buffer: output,
                status_buffer,
                job_buffer,
                batch_jobs: Vec::new(),
                code_blocks,
                output_capacity_total: 0,
                tier1_command_buffer: command_buffer,
                _coefficient_buffer: coefficient_buffer,
                prepare_command_buffer,
                _deinterleave_status_buffer: deinterleave_status_buffer,
                plane_buffers,
                scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }

        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal resident HT Tier-1 encode jobs",
        );
        let mut batch_jobs = budget.try_vec(
            code_blocks.len(),
            "J2K Metal resident HT Tier-1 encode jobs",
        )?;
        let mut output_capacity_total = 0usize;

        for block in &code_blocks {
            let output_capacity = ht_encode_output_capacity(block.width, block.height)?;
            let output_capacity_u32 =
                u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal resident encode output capacity exceeds u32".to_string(),
                })?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal resident encode output table exceeds u32".to_string(),
                })?;
            batch_jobs.push(J2kHtEncodeBatchJob {
                coefficient_offset: block.coefficient_offset,
                output_offset,
                width: block.width,
                height: block.height,
                total_bitplanes: u32::from(block.total_bitplanes),
                output_capacity: output_capacity_u32,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal resident encode output buffer overflow".to_string(),
                })?;
        }

        let job_buffer = copied_slice_buffer(&runtime.device, &batch_jobs)?;
        let output = new_private_buffer(&runtime.device, output_capacity_total.max(1))?;
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kHtEncodeStatus>(
                batch_jobs.len(),
                "HTJ2K Metal resident status buffer",
            )?,
        )?;
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal resident encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&command_buffer, "j2k htj2k resident tier1");
        let encoder = new_compute_command_encoder(&command_buffer)?;
        label_compute_encoder(&encoder, "HTJ2K Tier-1 encode");
        let pipeline = &runtime.ht_encode_code_blocks;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
        encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
        encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
        encoder.set_buffer(6, Some(&status_buffer), 0);
        encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(&encoder, pipeline, u64::from(job_count));
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kResidentLosslessHtCodeBlocks {
            output_buffer: output,
            status_buffer,
            job_buffer,
            batch_jobs,
            code_blocks,
            output_capacity_total,
            tier1_command_buffer: command_buffer,
            _coefficient_buffer: coefficient_buffer,
            prepare_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            plane_buffers,
            scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "single-block dispatch and validated readback form one operation"
)]
pub(crate) fn encode_classic_tier1_code_block(
    job: J2kTier1CodeBlockEncodeJob<'_>,
) -> Result<EncodedJ2kCodeBlock, Error> {
    with_runtime(|runtime| {
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode coefficient slice is too small".to_string(),
            });
        }

        let output_capacity =
            classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let style_flags = classic_style_flags(job.style);
        let segment_capacity = classic_encode_segment_capacity(style_flags, job.total_bitplanes);
        let params = J2kClassicEncodeParams {
            width: job.width,
            height: job.height,
            sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
            total_bitplanes: u32::from(job.total_bitplanes),
            style_flags,
            output_capacity: output_capacity_u32,
            segment_capacity: u32::try_from(segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds u32".to_string(),
            })?,
        };
        let coefficients =
            copied_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients])?;
        let output = new_shared_buffer(&runtime.device, output_capacity)?;
        let status_buffer =
            zeroed_shared_buffer(&runtime.device, size_of::<J2kClassicEncodeStatus>())?;
        let segment_capacity =
            usize::try_from(params.segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
            })?;
        let segment_buffer = new_shared_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kClassicSegment>(
                segment_capacity,
                "classic J2K Metal single segment buffer",
            )?,
        )?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.classic_encode_code_block);
        encoder.set_buffer(0, Some(&coefficients), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kClassicEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        dispatch_single_thread(&encoder);
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;

        let status =
            checked_buffer_read::<J2kClassicEncodeStatus>(&status_buffer, "classic Tier-1 status")?;
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode length exceeds usize".to_string(),
        })?;
        let payload_skip = usize::try_from(status.reserved0).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode payload skip exceeds usize".to_string(),
        })?;
        let payload_span =
            data_len
                .checked_add(payload_skip)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode payload span overflow".to_string(),
                })?;
        if payload_span > output_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let payload_offset = payload_skip;
        let data = if data_len == 0 {
            Vec::new()
        } else {
            checked_buffer_slice_at::<u8>(
                &output,
                payload_offset,
                data_len,
                "classic Tier-1 payload",
            )?
        };
        let number_of_coding_passes =
            u8::try_from(status.number_of_coding_passes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode pass count exceeds u8".to_string(),
            })?;
        let missing_bit_planes =
            u8::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode missing bitplanes exceeds u8".to_string(),
            })?;
        let segment_count =
            usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment count exceeds usize".to_string(),
            })?;
        let segment_capacity =
            usize::try_from(params.segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
            })?;
        if segment_count > segment_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode segment count exceeds buffer".to_string(),
            });
        }
        let raw_segments = if segment_count == 0 {
            Vec::new()
        } else {
            checked_buffer_slice::<J2kClassicSegment>(
                &segment_buffer,
                segment_count,
                "classic Tier-1 segments",
            )?
        };
        let segments = project_classic_encode_segments(&raw_segments)?;

        Ok(EncodedJ2kCodeBlock {
            data,
            segments,
            number_of_coding_passes,
            missing_bit_planes,
        })
    })
}

#[cfg(target_os = "macos")]
pub(super) fn read_ht_encoded_code_block(
    status: J2kHtEncodeStatus,
    output: &Buffer,
    output_offset: usize,
    output_capacity: usize,
) -> Result<EncodedHtJ2kCodeBlock, Error> {
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "HTJ2K cleanup",
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal encode length exceeds usize".to_string(),
    })?;
    if data_len > output_capacity {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal encode length exceeds output buffer".to_string(),
        });
    }
    let data = if data_len == 0 {
        Vec::new()
    } else {
        checked_buffer_slice_at::<u8>(output, output_offset, data_len, "HTJ2K encode payload")?
    };
    Ok(EncodedHtJ2kCodeBlock {
        data,
        cleanup_length: status.data_len,
        refinement_length: 0,
        num_coding_passes: u8::try_from(status.num_coding_passes).map_err(|_| {
            Error::MetalKernel {
                message: "HTJ2K Metal encode pass count exceeds u8".to_string(),
            }
        })?,
        num_zero_bitplanes: u8::try_from(status.num_zero_bitplanes).map_err(|_| {
            Error::MetalKernel {
                message: "HTJ2K Metal encode zero bitplanes exceeds u8".to_string(),
            }
        })?,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn read_resident_ht_tier1_code_blocks_for_cpu_packetization(
    session: &crate::MetalBackendSession,
    tier1: &J2kResidentLosslessHtCodeBlocks,
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    with_runtime_for_session(session, |runtime| {
        if tier1.batch_jobs.is_empty() {
            return Ok(Vec::new());
        }
        let output_bytes = tier1.output_capacity_total.max(1);
        let status_bytes = tier1
            .batch_jobs
            .len()
            .checked_mul(size_of::<J2kHtEncodeStatus>())
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal resident status readback size overflow".to_string(),
            })?;
        let mut result_budget = crate::batch_allocation::BatchMetadataBudget::new(
            "HTJ2K Metal resident Tier-1 readback metadata",
        );
        result_budget.account_capacity::<J2kHtEncodeBatchJob>(tier1.batch_jobs.capacity())?;
        result_budget
            .account_capacity::<J2kLosslessDeviceCodeBlock>(tier1.code_blocks.capacity())?;
        result_budget.account_capacity::<Buffer>(tier1.plane_buffers.capacity())?;
        result_budget.account_capacity::<Buffer>(tier1.scratch_buffers.capacity())?;
        result_budget.preflight(&[crate::batch_allocation::BatchMetadataRequest::of::<
            EncodedHtJ2kCodeBlock,
        >(tier1.batch_jobs.len())])?;
        let mut encoded_blocks = result_budget.try_vec(
            tier1.batch_jobs.len(),
            "HTJ2K Metal resident Tier-1 encoded block results",
        )?;
        let output = new_shared_buffer(&runtime.device, output_bytes)?;
        let status_buffer = zeroed_shared_buffer(&runtime.device, status_bytes)?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&command_buffer, "j2k htj2k resident tier1 cpu readback");
        let blit = new_blit_command_encoder(&command_buffer)?;
        blit.copy_from_buffer(
            &tier1.output_buffer,
            0,
            &output,
            0,
            tier1.output_capacity_total as u64,
        );
        blit.copy_from_buffer(
            &tier1.status_buffer,
            0,
            &status_buffer,
            0,
            status_bytes as u64,
        );
        blit.end_encoding();
        commit_and_wait_metal(&command_buffer)?;

        let statuses = checked_buffer_slice::<J2kHtEncodeStatus>(
            &status_buffer,
            tier1.batch_jobs.len(),
            "resident HT encode statuses",
        )?;
        for (batch_job, status) in tier1.batch_jobs.iter().zip(statuses.iter().copied()) {
            encoded_blocks.push(read_ht_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal resident output offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal resident output capacity exceeds usize".to_string(),
                })?,
            )?);
        }
        Ok(encoded_blocks)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_cleanup_code_blocks(
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    with_runtime(|runtime| encode_ht_cleanup_code_blocks_with_runtime(runtime, jobs))
}

#[cfg(target_os = "macos")]
pub(super) fn encode_ht_cleanup_code_blocks_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    let blocks = encode_ht_cleanup_code_blocks_with_runtime_and_statuses(runtime, jobs)?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K Metal encoded block projection metadata",
    );
    budget.account_capacity::<(EncodedHtJ2kCodeBlock, J2kHtEncodeStatus)>(blocks.capacity())?;
    budget.preflight(&[crate::batch_allocation::BatchMetadataRequest::of::<
        EncodedHtJ2kCodeBlock,
    >(blocks.len())])?;
    let mut encoded = budget.try_vec(blocks.len(), "HTJ2K Metal encoded block results")?;
    for (block, _status) in blocks {
        encoded.push(block);
    }
    Ok(encoded)
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "HT batch dispatch keeps output/status slice ownership aligned"
)]
pub(super) fn encode_ht_cleanup_code_blocks_with_runtime_and_statuses(
    runtime: &MetalRuntime,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<(EncodedHtJ2kCodeBlock, J2kHtEncodeStatus)>, Error> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    if jobs.iter().any(|job| job.target_coding_passes != 1) {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal cleanup encode supports one coding pass".to_string(),
        });
    }

    let coefficient_count = jobs.iter().try_fold(0usize, |total, job| {
        let width = usize::try_from(job.width).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal encode width exceeds usize".to_string(),
        })?;
        let height = usize::try_from(job.height).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal encode height exceeds usize".to_string(),
        })?;
        let count = width
            .checked_mul(height)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient count overflow".to_string(),
            })?;
        total.checked_add(count).ok_or_else(|| {
            Error::from(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "HTJ2K Metal encode coefficients",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
        })
    })?;
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K Metal Tier-1 encode batch");
    let mut coefficients = budget.try_vec(coefficient_count, "HTJ2K Metal encode coefficients")?;
    let mut batch_jobs = budget.try_vec(jobs.len(), "HTJ2K Metal encode batch jobs")?;
    let mut output_capacity_total = 0usize;

    for job in jobs {
        let output_capacity = ht_encode_output_capacity(job.width, job.height)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient slice is too small".to_string(),
            });
        }
        let coefficient_offset =
            u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient table exceeds u32".to_string(),
            })?;
        coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
        let output_offset =
            u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output table exceeds u32".to_string(),
            })?;
        batch_jobs.push(J2kHtEncodeBatchJob {
            coefficient_offset,
            output_offset,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_capacity: output_capacity_u32,
        });
        output_capacity_total = output_capacity_total
            .checked_add(output_capacity)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode output buffer overflow".to_string(),
            })?;
    }

    let coefficient_buffer = copied_slice_buffer(&runtime.device, &coefficients)?;
    let job_buffer = copied_slice_buffer(&runtime.device, &batch_jobs)?;
    let output = new_shared_buffer(&runtime.device, output_capacity_total.max(1))?;
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        checked_type_buffer_bytes::<J2kHtEncodeStatus>(
            jobs.len(),
            "HTJ2K Metal encode status buffer",
        )?,
    )?;
    let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal encode job count exceeds u32".to_string(),
    })?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    label_command_buffer(&command_buffer, "j2k htj2k tier1 batch");
    let encoder = new_compute_command_encoder(&command_buffer)?;
    label_compute_encoder(&encoder, "HTJ2K Tier-1 encode");
    let pipeline = &runtime.ht_encode_code_blocks;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&coefficient_buffer), 0);
    encoder.set_buffer(1, Some(&output), 0);
    encoder.set_buffer(2, Some(&job_buffer), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
    encoder.set_buffer(6, Some(&status_buffer), 0);
    encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
    dispatch_1d_pipeline(&encoder, pipeline, u64::from(job_count));
    encoder.end_encoding();
    commit_and_wait_metal(&command_buffer)?;

    let statuses = checked_buffer_slice::<J2kHtEncodeStatus>(
        &status_buffer,
        jobs.len(),
        "HT encode statuses",
    )?;
    let mut results = budget.try_vec(jobs.len(), "J2K Metal HT Tier-1 encoded blocks")?;
    for (idx, status) in statuses.iter().copied().enumerate() {
        let batch_job = batch_jobs[idx];
        let encoded_block = read_ht_encoded_code_block(
            status,
            &output,
            usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output offset exceeds usize".to_string(),
            })?,
            usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds usize".to_string(),
            })?,
        )?;
        results.push((encoded_block, status));
    }

    Ok(results)
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_cleanup_code_block(
    job: J2kHtCodeBlockEncodeJob<'_>,
) -> Result<EncodedHtJ2kCodeBlock, Error> {
    with_runtime(|runtime| {
        if job.target_coding_passes != 1 {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal cleanup encode supports one coding pass".to_string(),
            });
        }
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient slice is too small".to_string(),
            });
        }
        let output_capacity = ht_encode_output_capacity(job.width, job.height)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let params = J2kHtEncodeParams {
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_capacity: output_capacity_u32,
        };
        let coefficients =
            copied_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients])?;
        let output = new_shared_buffer(&runtime.device, output_capacity)?;
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kHtEncodeStatus>())?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.ht_encode_code_block);
        encoder.set_buffer(0, Some(&coefficients), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kHtEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
        encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
        encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
        encoder.set_buffer(6, Some(&status_buffer), 0);
        dispatch_single_thread(&encoder);
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;

        let status = checked_buffer_read::<J2kHtEncodeStatus>(&status_buffer, "HT encode status")?;
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "HTJ2K cleanup",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal encode length exceeds usize".to_string(),
        })?;
        if data_len > output_capacity {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let data = if data_len == 0 {
            Vec::new()
        } else {
            checked_buffer_slice::<u8>(&output, data_len, "HT encode payload")?
        };
        Ok(EncodedHtJ2kCodeBlock {
            data,
            cleanup_length: status.data_len,
            refinement_length: 0,
            num_coding_passes: u8::try_from(status.num_coding_passes).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Metal encode pass count exceeds u8".to_string(),
                }
            })?,
            num_zero_bitplanes: u8::try_from(status.num_zero_bitplanes).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Metal encode zero bitplanes exceeds u8".to_string(),
                }
            })?,
        })
    })
}
