// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(all(test, target_os = "macos"))]
#[expect(
    clippy::too_many_lines,
    reason = "test-only route reproduces the complete GPU token-pack pipeline"
)]
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

        let coefficient_buffer = copied_slice_buffer(&runtime.device, &coefficients)?;
        let job_buffer = copied_slice_buffer(&runtime.device, &batch_jobs)?;
        let output = new_shared_buffer(&runtime.device, output_capacity_total.max(1))?;
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kClassicEncodeStatus>(
                jobs.len(),
                "classic J2K Metal token-pack status buffer",
            )?,
        )?;
        let segment_buffer = new_shared_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kClassicSegment>(
                segment_capacity_total,
                "classic J2K Metal token-pack segment buffer",
            )?,
        )?;
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal token-pack job count exceeds u32".to_string(),
        })?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let mut recyclable_private_buffers = Vec::new();
        let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
            runtime,
            &command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
        )?;
        debug_assert_eq!(token_buffers.job_count, job_count);
        dispatch_classic_tier1_token_pack_from_gpu_tokens(
            runtime,
            &command_buffer,
            &job_buffer,
            &token_buffers,
            &output,
            &status_buffer,
            &segment_buffer,
        )?;
        commit_and_wait_metal(&command_buffer)?;

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
#[expect(
    clippy::too_many_lines,
    reason = "test-only route reproduces the complete split-token GPU pipeline"
)]
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

        let coefficient_buffer = copied_slice_buffer(&runtime.device, &coefficients)?;
        let job_buffer = copied_slice_buffer(&runtime.device, &batch_jobs)?;
        let output = new_shared_buffer(&runtime.device, output_capacity_total.max(1))?;
        let status_buffer = zeroed_shared_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kClassicEncodeStatus>(
                jobs.len(),
                "classic J2K Metal split token-pack status buffer",
            )?,
        )?;
        let segment_buffer = new_shared_buffer(
            &runtime.device,
            checked_type_buffer_bytes::<J2kClassicSegment>(
                segment_capacity_total,
                "classic J2K Metal split token-pack segment buffer",
            )?,
        )?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let mut recyclable_private_buffers = Vec::new();
        let split_buffers = dispatch_classic_tier1_split_token_emit_for_gpu_pack(
            runtime,
            &command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
            use_mq_byte_emit,
        )?;
        dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
            runtime,
            &command_buffer,
            &job_buffer,
            &split_buffers,
            &output,
            &status_buffer,
            &segment_buffer,
        )?;
        commit_and_wait_metal(&command_buffer)?;

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
