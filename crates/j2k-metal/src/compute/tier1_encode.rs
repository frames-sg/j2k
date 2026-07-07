// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    borrow_slice_buffer, checked_buffer_read, checked_buffer_slice, checked_buffer_slice_at,
    classic_encode_code_blocks_pipeline, classic_encode_output_capacity,
    classic_encode_segment_capacity, classic_style_flags, commit_and_wait_metal,
    dispatch_1d_pipeline, dispatch_single_thread, ht_encode_output_capacity, label_command_buffer,
    label_compute_encoder, owned_slice_buffer, size_of, with_runtime, with_runtime_for_session,
    zeroed_shared_buffer, Buffer, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, Error,
    J2kClassicEncodeBatchJob, J2kClassicEncodeParams, J2kClassicEncodeStatus, J2kClassicSegment,
    J2kCodeBlockSegment, J2kHtCodeBlockEncodeJob, J2kHtEncodeBatchJob, J2kHtEncodeParams,
    J2kHtEncodeStatus, J2kPacketEncodeStatus, J2kPreparedLosslessDeviceCodeBlocks,
    J2kResidentLosslessHtCodeBlocks, J2kResidentLosslessTier1CodeBlocks,
    J2kTier1CodeBlockEncodeJob, MTLResourceOptions, MetalRuntime, J2K_ENCODE_STATUS_FAIL,
    J2K_ENCODE_STATUS_OK, J2K_ENCODE_STATUS_UNSUPPORTED,
};
#[cfg(test)]
use super::{
    classic_tier1_gpu_token_pack_supported, dispatch_classic_tier1_split_token_emit_for_cpu_pack,
    dispatch_classic_tier1_split_token_emit_for_gpu_pack,
    dispatch_classic_tier1_split_token_pack_from_gpu_tokens,
    dispatch_classic_tier1_token_emit_for_gpu_pack,
    dispatch_classic_tier1_token_pack_from_gpu_tokens,
    pack_j2k_code_block_scalar_from_tier1_tokens, J2kClassicTier1SymbolPlanCounters,
    J2kClassicTier1TokenSegment, J2kTier1TokenSegment,
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
        &[][..]
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
            .to_vec()
    };
    let segments = raw_segments
        .iter()
        .map(|segment| {
            Ok(J2kCodeBlockSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: u8::try_from(segment.start_coding_pass).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment start pass exceeds u8"
                            .to_string(),
                    }
                })?,
                end_coding_pass: u8::try_from(segment.end_coding_pass).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment end pass exceeds u8".to_string(),
                    }
                })?,
                use_arithmetic: segment.use_arithmetic != 0,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(EncodedJ2kCodeBlock {
        data,
        segments,
        number_of_coding_passes,
        missing_bit_planes,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_classic_tier1_code_blocks(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
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

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            jobs.len() * size_of::<J2kClassicEncodeStatus>(),
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let classic_encode_pipeline = classic_encode_code_blocks_pipeline(runtime, &batch_jobs);
        encoder.set_compute_pipeline_state(classic_encode_pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(encoder, classic_encode_pipeline, u64::from(job_count));
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        let statuses = checked_buffer_slice::<J2kClassicEncodeStatus>(
            &status_buffer,
            jobs.len(),
            "classic encode statuses",
        )?;
        let mut results = Vec::with_capacity(jobs.len());
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

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
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
                    message: "classic J2K Metal token-pack coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal token-pack coefficient slice is too small"
                        .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment table exceeds u32".to_string(),
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
                        message: "classic J2K Metal token-pack output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal token-pack segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment buffer overflow".to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message:
                    "classic J2K Metal token-pack parity helper supports only bypass_u16_32 jobs"
                        .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            jobs.len() * size_of::<J2kClassicEncodeStatus>(),
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal token-pack job count exceeds u32".to_string(),
        })?;
        let command_buffer = runtime.queue.new_command_buffer();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
        )?;
        debug_assert_eq!(token_buffers.job_count, job_count);
        dispatch_classic_tier1_token_pack_from_gpu_tokens(
            runtime,
            command_buffer,
            &job_buffer,
            &token_buffers,
            &output,
            &status_buffer,
            &segment_buffer,
        );
        commit_and_wait_metal(command_buffer)?;

        let statuses = checked_buffer_slice::<J2kClassicEncodeStatus>(
            &status_buffer,
            jobs.len(),
            "classic GPU-pack encode statuses",
        )?;
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output capacity exceeds usize"
                        .to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment offset exceeds usize"
                        .to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment capacity exceeds usize"
                        .to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test_with_emit_route(
        jobs, false,
    )
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_split_mq_byte_raw_tokens_gpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test_with_emit_route(
        jobs, true,
    )
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test_with_emit_route(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    use_mq_byte_emit: bool,
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
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
                    message: "classic J2K Metal split GPU token-pack coefficient count overflow"
                        .to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message:
                        "classic J2K Metal split GPU token-pack coefficient slice is too small"
                            .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output table exceeds u32"
                        .to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack segment table exceeds u32"
                        .to_string(),
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
                        message:
                            "classic J2K Metal split GPU token-pack output capacity exceeds u32"
                                .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "classic J2K Metal split GPU token-pack segment capacity exceeds u32"
                                .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output buffer overflow"
                        .to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack segment buffer overflow"
                        .to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message:
                    "classic J2K Metal split GPU token-pack helper supports only bypass_u16_32 jobs"
                        .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            jobs.len() * size_of::<J2kClassicEncodeStatus>(),
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = runtime.queue.new_command_buffer();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let split_buffers = dispatch_classic_tier1_split_token_emit_for_gpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
            use_mq_byte_emit,
        )?;
        dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
            runtime,
            command_buffer,
            &job_buffer,
            &split_buffers,
            &output,
            &status_buffer,
            &segment_buffer,
        );
        commit_and_wait_metal(command_buffer)?;

        let statuses = checked_buffer_slice::<J2kClassicEncodeStatus>(
            &status_buffer,
            jobs.len(),
            "classic split-token encode statuses",
        )?;
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output offset exceeds usize"
                        .to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output capacity exceeds usize"
                        .to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack segment offset exceeds usize"
                        .to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message:
                        "classic J2K Metal split GPU token-pack segment capacity exceeds usize"
                            .to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
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
                    message: "classic J2K Metal ordered-token coefficient count overflow"
                        .to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coefficient slice is too small"
                        .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment table exceeds u32"
                        .to_string(),
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
                        message: "classic J2K Metal ordered-token output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal ordered-token segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment buffer overflow".to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal ordered-token helper supports only bypass_u16_32 jobs"
                    .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let command_buffer = runtime.queue.new_command_buffer();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
        )?;
        let job_count =
            usize::try_from(token_buffers.job_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal ordered-token job count exceeds usize".to_string(),
            })?;
        let token_stride_bytes =
            usize::try_from(token_buffers.token_stride_bytes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal ordered-token byte stride exceeds usize".to_string(),
            })?;
        let token_segment_stride =
            usize::try_from(token_buffers.token_segment_stride).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment stride exceeds usize"
                        .to_string(),
                }
            })?;
        let counter_byte_len = job_count
            .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal ordered-token counter readback overflow".to_string(),
            })?;
        let token_byte_len =
            job_count
                .checked_mul(token_stride_bytes)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token byte readback overflow".to_string(),
                })?;
        let token_segment_byte_len = job_count
            .checked_mul(token_segment_stride)
            .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal ordered-token segment readback overflow".to_string(),
            })?;
        let counter_readback = runtime.device.new_buffer(
            counter_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let token_readback = runtime.device.new_buffer(
            token_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let token_segment_readback = runtime.device.new_buffer(
            token_segment_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &token_buffers.counter_buffer,
            0,
            &counter_readback,
            0,
            counter_byte_len as u64,
        );
        blit.copy_from_buffer(
            &token_buffers.token_buffer,
            0,
            &token_readback,
            0,
            token_byte_len as u64,
        );
        blit.copy_from_buffer(
            &token_buffers.segment_buffer,
            0,
            &token_segment_readback,
            0,
            token_segment_byte_len as u64,
        );
        blit.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        let counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
            &counter_readback,
            job_count,
            "classic token-pack counters",
        )?;
        let token_bytes =
            checked_buffer_slice::<u8>(&token_readback, token_byte_len, "classic token bytes")?;
        let token_segments = checked_buffer_slice::<J2kClassicTier1TokenSegment>(
            &token_segment_readback,
            job_count.saturating_mul(token_segment_stride),
            "classic token segments",
        )?;

        let mut results = Vec::with_capacity(job_count);
        for (block_idx, counter) in counters.iter().enumerate() {
            if counter.code != J2K_ENCODE_STATUS_OK {
                return Err(encode_status_error(
                    "classic Tier-1 ordered-token emit",
                    counter.code,
                    counter.detail,
                ));
            }
            let segment_count =
                usize::try_from(counter.segment_count).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment count exceeds usize"
                        .to_string(),
                })?;
            if segment_count > token_segment_stride {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment count exceeds capacity"
                        .to_string(),
                });
            }
            let token_start =
                block_idx
                    .checked_mul(token_stride_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token byte offset overflow".to_string(),
                    })?;
            let segment_start =
                block_idx
                    .checked_mul(token_segment_stride)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token segment offset overflow"
                            .to_string(),
                    })?;
            let mut native_segments = Vec::with_capacity(segment_count);
            for segment in &token_segments[segment_start..segment_start + segment_count] {
                let start_coding_pass =
                    u8::try_from(segment.pass_range & 0xFFFF).map_err(|_| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token start pass exceeds u8"
                            .to_string(),
                    })?;
                let end_coding_pass =
                    u8::try_from(segment.pass_range >> 16).map_err(|_| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token end pass exceeds u8".to_string(),
                    })?;
                native_segments.push(J2kTier1TokenSegment {
                    token_bit_offset: segment.token_bit_offset,
                    token_bit_count: segment.token_bit_count,
                    start_coding_pass,
                    end_coding_pass,
                    use_arithmetic: (segment.flags & 1) != 0,
                });
            }
            let packed = pack_j2k_code_block_scalar_from_tier1_tokens(
                &token_bytes[token_start..token_start + token_stride_bytes],
                &native_segments,
                u8::try_from(counter.coding_passes).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coding-pass count exceeds u8"
                        .to_string(),
                })?,
                u8::try_from(counter.missing_bit_planes).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token missing bitplanes exceed u8"
                        .to_string(),
                })?,
            )
            .map_err(|message| Error::MetalKernel {
                message: format!("classic J2K Metal ordered-token CPU pack failed: {message}"),
            })?;
            results.push(packed);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
#[derive(Default)]
pub(super) struct ClassicTier1MsbBitWriter {
    pub(super) bytes: Vec<u8>,
    pub(super) current_byte: u8,
    pub(super) bits_in_current: u8,
    pub(super) bit_count: usize,
}

#[cfg(all(test, target_os = "macos"))]
impl ClassicTier1MsbBitWriter {
    fn write_bit(&mut self, bit: u8) {
        self.current_byte = (self.current_byte << 1) | (bit & 1);
        self.bits_in_current += 1;
        self.bit_count += 1;
        if self.bits_in_current == 8 {
            self.bytes.push(self.current_byte);
            self.current_byte = 0;
            self.bits_in_current = 0;
        }
    }

    fn bit_count_u32(&self) -> Result<u32, Error> {
        u32::try_from(self.bit_count).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal split-token combined bit offset exceeds u32".to_string(),
        })
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_in_current != 0 {
            self.bytes
                .push(self.current_byte << (8 - self.bits_in_current));
        }
        self.bytes
    }
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn classic_tier1_split_token_bit(source: &[u8], bit_offset: usize) -> Result<u8, Error> {
    if bit_offset >= source.len().saturating_mul(8) {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal split-token bit offset exceeds stream".to_string(),
        });
    }
    let byte = source[bit_offset / 8];
    let shift = 7 - (bit_offset % 8);
    Ok((byte >> shift) & 1)
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn classic_tier1_append_split_token_bits(
    writer: &mut ClassicTier1MsbBitWriter,
    source: &[u8],
    bit_offset: usize,
    bit_count: usize,
) -> Result<(), Error> {
    let end = bit_offset
        .checked_add(bit_count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal split-token bit range overflow".to_string(),
        })?;
    if end > source.len().saturating_mul(8) {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal split-token bit range exceeds stream".to_string(),
        });
    }
    for bit_idx in 0..bit_count {
        writer.write_bit(classic_tier1_split_token_bit(source, bit_offset + bit_idx)?);
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
pub(super) fn pack_classic_split_mq_raw_tokens_for_test(
    mq_token_bytes: &[u8],
    raw_token_bytes: &[u8],
    split_segments: &[J2kClassicTier1TokenSegment],
    counter: J2kClassicTier1SymbolPlanCounters,
) -> Result<EncodedJ2kCodeBlock, Error> {
    if counter.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "classic Tier-1 split-token emit",
            counter.code,
            counter.detail,
        ));
    }

    let mut combined = ClassicTier1MsbBitWriter::default();
    let mut native_segments = Vec::with_capacity(split_segments.len());
    for segment in split_segments {
        if (segment.flags & !1) != 0 {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal split-token segment has unsupported flags".to_string(),
            });
        }
        let use_arithmetic = (segment.flags & 1) != 0;
        let source = if use_arithmetic {
            mq_token_bytes
        } else {
            raw_token_bytes
        };
        let source_bit_offset =
            usize::try_from(segment.token_bit_offset).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token bit offset exceeds usize".to_string(),
            })?;
        let source_bit_count =
            usize::try_from(segment.token_bit_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token bit count exceeds usize".to_string(),
            })?;
        let combined_bit_offset = combined.bit_count_u32()?;
        classic_tier1_append_split_token_bits(
            &mut combined,
            source,
            source_bit_offset,
            source_bit_count,
        )?;
        let start_coding_pass =
            u8::try_from(segment.pass_range & 0xFFFF).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token start pass exceeds u8".to_string(),
            })?;
        let end_coding_pass =
            u8::try_from(segment.pass_range >> 16).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token end pass exceeds u8".to_string(),
            })?;
        native_segments.push(J2kTier1TokenSegment {
            token_bit_offset: combined_bit_offset,
            token_bit_count: segment.token_bit_count,
            start_coding_pass,
            end_coding_pass,
            use_arithmetic,
        });
    }

    pack_j2k_code_block_scalar_from_tier1_tokens(
        &combined.finish(),
        &native_segments,
        u8::try_from(counter.coding_passes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal split-token coding-pass count exceeds u8".to_string(),
        })?,
        u8::try_from(counter.missing_bit_planes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal split-token missing bitplanes exceed u8".to_string(),
        })?,
    )
    .map_err(|message| Error::MetalKernel {
        message: format!("classic J2K Metal split-token CPU pack failed: {message}"),
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
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
                    message: "classic J2K Metal split-token coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal split-token coefficient slice is too small"
                        .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token segment table exceeds u32".to_string(),
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
                        message: "classic J2K Metal split-token output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal split-token segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token segment buffer overflow".to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal split-token helper supports only bypass_u16_32 jobs"
                    .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let command_buffer = runtime.queue.new_command_buffer();
        let split_buffers = dispatch_classic_tier1_split_token_emit_for_cpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
        )?;
        commit_and_wait_metal(command_buffer)?;

        let job_count =
            usize::try_from(split_buffers.job_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token job count exceeds usize".to_string(),
            })?;
        let mq_token_stride_bytes =
            usize::try_from(split_buffers.mq_token_stride_bytes).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal split-token MQ byte stride exceeds usize"
                        .to_string(),
                }
            })?;
        let raw_token_stride_bytes = usize::try_from(split_buffers.raw_token_stride_bytes)
            .map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token raw byte stride exceeds usize".to_string(),
            })?;
        let token_segment_stride =
            usize::try_from(split_buffers.token_segment_stride).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal split-token segment stride exceeds usize"
                        .to_string(),
                }
            })?;
        let counters = checked_buffer_slice::<J2kClassicTier1SymbolPlanCounters>(
            &split_buffers.counter_buffer,
            job_count,
            "classic split-token counters",
        )?;
        let mq_token_bytes = checked_buffer_slice::<u8>(
            &split_buffers.mq_token_buffer,
            job_count.saturating_mul(mq_token_stride_bytes),
            "classic split MQ token bytes",
        )?;
        let raw_token_bytes = checked_buffer_slice::<u8>(
            &split_buffers.raw_token_buffer,
            job_count.saturating_mul(raw_token_stride_bytes),
            "classic split raw token bytes",
        )?;
        let token_segments = checked_buffer_slice::<J2kClassicTier1TokenSegment>(
            &split_buffers.segment_buffer,
            job_count.saturating_mul(token_segment_stride),
            "classic split token segments",
        )?;

        let mut results = Vec::with_capacity(job_count);
        for (block_idx, counter) in counters.iter().copied().enumerate() {
            let segment_count =
                usize::try_from(counter.segment_count).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token segment count exceeds usize"
                        .to_string(),
                })?;
            if segment_count > token_segment_stride {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal split-token segment count exceeds capacity"
                        .to_string(),
                });
            }
            let mq_token_start = block_idx
                .checked_mul(mq_token_stride_bytes)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token MQ byte offset overflow".to_string(),
                })?;
            let raw_token_start =
                block_idx
                    .checked_mul(raw_token_stride_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal split-token raw byte offset overflow"
                            .to_string(),
                    })?;
            let segment_start =
                block_idx
                    .checked_mul(token_segment_stride)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal split-token segment offset overflow"
                            .to_string(),
                    })?;
            results.push(pack_classic_split_mq_raw_tokens_for_test(
                &mq_token_bytes[mq_token_start..mq_token_start + mq_token_stride_bytes],
                &raw_token_bytes[raw_token_start..raw_token_start + raw_token_stride_bytes],
                &token_segments[segment_start..segment_start + segment_count],
                counter,
            )?);
        }

        Ok(results)
    })
}

#[cfg(target_os = "macos")]
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
            let output = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let status_buffer = zeroed_shared_buffer(&runtime.device, 1);
            let segment_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let job_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModeShared);
            let command_buffer = runtime.queue.new_command_buffer();
            command_buffer.commit();
            return Ok(J2kResidentLosslessTier1CodeBlocks {
                output_buffer: output,
                status_buffer,
                job_buffer,
                batch_jobs: Vec::new(),
                code_blocks,
                output_capacity_total: 0,
                _segment_buffer: segment_buffer,
                tier1_command_buffer: command_buffer.to_owned(),
                _coefficient_buffer: coefficient_buffer,
                prepare_command_buffer,
                _deinterleave_status_buffer: deinterleave_status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(code_blocks.len());
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

        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            batch_jobs.len() * size_of::<J2kClassicEncodeStatus>(),
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal resident encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let classic_encode_pipeline = classic_encode_code_blocks_pipeline(runtime, &batch_jobs);
        encoder.set_compute_pipeline_state(classic_encode_pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(encoder, classic_encode_pipeline, u64::from(job_count));
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
            tier1_command_buffer: command_buffer.to_owned(),
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
            let output = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let status_buffer = zeroed_shared_buffer(&runtime.device, 1);
            let job_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModeShared);
            let command_buffer = runtime.queue.new_command_buffer();
            command_buffer.commit();
            return Ok(J2kResidentLosslessHtCodeBlocks {
                output_buffer: output,
                status_buffer,
                job_buffer,
                batch_jobs: Vec::new(),
                code_blocks,
                output_capacity_total: 0,
                tier1_command_buffer: command_buffer.to_owned(),
                _coefficient_buffer: coefficient_buffer,
                prepare_command_buffer,
                _deinterleave_status_buffer: deinterleave_status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }

        let mut batch_jobs = Vec::<J2kHtEncodeBatchJob>::with_capacity(code_blocks.len());
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

        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            batch_jobs.len() * size_of::<J2kHtEncodeStatus>(),
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal resident encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k htj2k resident tier1");
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "HTJ2K Tier-1 encode");
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
        dispatch_1d_pipeline(encoder, pipeline, u64::from(job_count));
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kResidentLosslessHtCodeBlocks {
            output_buffer: output,
            status_buffer,
            job_buffer,
            batch_jobs,
            code_blocks,
            output_capacity_total,
            tier1_command_buffer: command_buffer.to_owned(),
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
            borrow_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients]);
        let output = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer =
            zeroed_shared_buffer(&runtime.device, size_of::<J2kClassicEncodeStatus>());
        let segment_buffer = runtime.device.new_buffer(
            (usize::try_from(params.segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
            })? * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
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
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

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
            .to_vec()
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
            &[][..]
        } else {
            checked_buffer_slice::<J2kClassicSegment>(
                &segment_buffer,
                segment_count,
                "classic Tier-1 segments",
            )?
        };
        let segments = raw_segments
            .iter()
            .map(|segment| {
                Ok(J2kCodeBlockSegment {
                    data_offset: segment.data_offset,
                    data_length: segment.data_length,
                    start_coding_pass: u8::try_from(segment.start_coding_pass).map_err(|_| {
                        Error::MetalKernel {
                            message: "classic J2K Metal encode segment start pass exceeds u8"
                                .to_string(),
                        }
                    })?,
                    end_coding_pass: u8::try_from(segment.end_coding_pass).map_err(|_| {
                        Error::MetalKernel {
                            message: "classic J2K Metal encode segment end pass exceeds u8"
                                .to_string(),
                        }
                    })?,
                    use_arithmetic: segment.use_arithmetic != 0,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

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
            .to_vec()
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
        let output = runtime
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let status_buffer = zeroed_shared_buffer(&runtime.device, status_bytes);

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k htj2k resident tier1 cpu readback");
        let blit = command_buffer.new_blit_command_encoder();
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
        commit_and_wait_metal(command_buffer)?;

        let statuses = checked_buffer_slice::<J2kHtEncodeStatus>(
            &status_buffer,
            tier1.batch_jobs.len(),
            "resident HT encode statuses",
        )?;
        tier1
            .batch_jobs
            .iter()
            .zip(statuses.iter().copied())
            .map(|(batch_job, status)| {
                read_ht_encoded_code_block(
                    status,
                    &output,
                    usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal resident output offset exceeds usize".to_string(),
                    })?,
                    usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal resident output capacity exceeds usize".to_string(),
                    })?,
                )
            })
            .collect()
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
    encode_ht_cleanup_code_blocks_with_runtime_and_statuses(runtime, jobs).map(|blocks| {
        blocks
            .into_iter()
            .map(|(encoded, _status)| encoded)
            .collect()
    })
}

#[cfg(target_os = "macos")]
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

    let mut coefficients = Vec::<i32>::new();
    let mut batch_jobs = Vec::<J2kHtEncodeBatchJob>::with_capacity(jobs.len());
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

    let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
    let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
    let output = runtime.device.new_buffer(
        output_capacity_total.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let status_buffer =
        zeroed_shared_buffer(&runtime.device, jobs.len() * size_of::<J2kHtEncodeStatus>());
    let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal encode job count exceeds u32".to_string(),
    })?;

    let command_buffer = runtime.queue.new_command_buffer();
    label_command_buffer(command_buffer, "j2k htj2k tier1 batch");
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "HTJ2K Tier-1 encode");
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
    dispatch_1d_pipeline(encoder, pipeline, u64::from(job_count));
    encoder.end_encoding();
    commit_and_wait_metal(command_buffer)?;

    let statuses = checked_buffer_slice::<J2kHtEncodeStatus>(
        &status_buffer,
        jobs.len(),
        "HT encode statuses",
    )?;
    let mut results = Vec::with_capacity(jobs.len());
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
            borrow_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients]);
        let output = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kHtEncodeStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
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
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

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
            checked_buffer_slice::<u8>(&output, data_len, "HT encode payload")?.to_vec()
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
