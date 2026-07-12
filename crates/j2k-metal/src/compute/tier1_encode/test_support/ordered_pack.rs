// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(all(test, target_os = "macos"))]
#[expect(
    clippy::too_many_lines,
    reason = "test-only route reproduces the complete ordered-token pipeline"
)]
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

        let coefficient_buffer = copied_slice_buffer(&runtime.device, &coefficients)?;
        let job_buffer = copied_slice_buffer(&runtime.device, &batch_jobs)?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
            runtime,
            &command_buffer,
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
        let counter_readback = new_shared_buffer(&runtime.device, counter_byte_len.max(1))?;
        let token_readback = new_shared_buffer(&runtime.device, token_byte_len.max(1))?;
        let token_segment_readback =
            new_shared_buffer(&runtime.device, token_segment_byte_len.max(1))?;

        let blit = new_blit_command_encoder(&command_buffer)?;
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
        commit_and_wait_metal(&command_buffer)?;

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
            .map_err(|source| {
                crate::error::native_encode_error("classic Tier-1 ordered-token CPU pack", source)
            })?;
            results.push(packed);
        }

        Ok(results)
    })
}
