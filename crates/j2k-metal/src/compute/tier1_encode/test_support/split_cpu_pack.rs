// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

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
