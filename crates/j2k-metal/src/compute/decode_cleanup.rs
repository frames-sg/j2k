// SPDX-License-Identifier: MIT OR Apache-2.0

use super::abi::{
    J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob, J2kHtCleanupParams,
};
use super::decode_dispatch::{dispatch_classic_cleanup_batched, required_ht_output_len};
use super::resident_codestream::{dispatch_ht_cleanup, dispatch_ht_cleanup_batched};
use super::{
    checked_buffer_slice, classic_style_flags, copied_slice_buffer, required_classic_output_len,
    with_runtime, Error, HtCodeBlockDecodeJob, HtSubBandDecodeJob, J2kCodeBlockDecodeJob,
    J2kSubBandDecodeJob,
};

#[cfg(target_os = "macos")]
struct ClassicCleanupOwners {
    jobs: Vec<J2kClassicCleanupBatchJob>,
    coded_data: Vec<u8>,
    segments: Vec<J2kClassicSegment>,
}

#[cfg(target_os = "macos")]
fn allocate_classic_cleanup_owners(
    job: &J2kSubBandDecodeJob<'_>,
) -> Result<ClassicCleanupOwners, Error> {
    let coded_len = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.code_block.data.len()),
        "classic J2K Metal batched coded payload",
    )?;
    let segment_count = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.code_block.segments.len()),
        "classic J2K Metal batched segment table",
    )?;
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("classic J2K Metal cleanup batch");
    Ok(ClassicCleanupOwners {
        jobs: budget.try_vec(job.jobs.len(), "classic J2K Metal cleanup jobs")?,
        coded_data: budget.try_vec(coded_len, "classic J2K Metal batched coded payload")?,
        segments: budget.try_vec(segment_count, "classic J2K Metal batched segment table")?,
    })
}

#[cfg(target_os = "macos")]
fn validate_classic_sub_band_output(
    job: &J2kSubBandDecodeJob<'_>,
    output_len: usize,
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal sub-band size overflow".to_string(),
        })?;
    if output_len < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal sub-band output slice is too small".to_string(),
        });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_code_block(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_classic_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    let mut segment_budget = crate::batch_allocation::BatchMetadataBudget::new(
        "classic J2K Metal cleanup segment metadata",
    );
    let mut segments =
        segment_budget.try_vec(job.segments.len(), "classic J2K Metal cleanup segments")?;
    for segment in job.segments {
        segments.push(J2kClassicSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: u32::from(segment.start_coding_pass),
            end_coding_pass: u32::from(segment.end_coding_pass),
            use_arithmetic: u32::from(segment.use_arithmetic),
        });
    }

    with_runtime(|runtime| {
        let decoded = copied_slice_buffer(&runtime.device, output)?;
        let batch_job = J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal coded payload exceeds u32".to_string(),
            })?,
            segment_offset: 0,
            segment_count: u32::try_from(job.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment count exceeds u32".to_string(),
            })?,
            width: job.width,
            height: job.height,
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            missing_msbs: u32::from(job.missing_bit_planes),
            total_bitplanes: u32::from(job.total_bitplanes),
            roi_shift: u32::from(job.roi_shift),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            sub_band_type: match job.sub_band_type {
                j2k_native::J2kSubBandType::LowLow => 0,
                j2k_native::J2kSubBandType::HighLow => 1,
                j2k_native::J2kSubBandType::LowHigh => 2,
                j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(job.style),
            strict: u32::from(job.strict),
            dequantization_step: job.dequantization_step,
        };
        dispatch_classic_cleanup_batched(runtime, job.data, &[batch_job], &segments, &decoded)?;
        let decoded_host =
            checked_buffer_slice::<f32>(&decoded, output.len(), "classic cleanup output")?;
        output.copy_from_slice(&decoded_host);
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_sub_band(
    job: J2kSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    validate_classic_sub_band_output(&job, output.len())?;
    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = copied_slice_buffer(&runtime.device, output)?;
        let ClassicCleanupOwners {
            mut jobs,
            mut coded_data,
            mut segments,
        } = allocate_classic_cleanup_owners(&job)?;

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);
            let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment table exceeds u32".to_string(),
            })?;
            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal batched block lies outside sub-band bounds"
                        .to_string(),
                });
            }
            for segment in block.code_block.segments {
                let data_offset =
                    coded_offset
                        .checked_add(segment.data_offset)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "classic J2K Metal segment offset overflow".to_string(),
                        })?;
                segments.push(J2kClassicSegment {
                    data_offset,
                    data_length: segment.data_length,
                    start_coding_pass: u32::from(segment.start_coding_pass),
                    end_coding_pass: u32::from(segment.end_coding_pass),
                    use_arithmetic: u32::from(segment.use_arithmetic),
                });
            }
            jobs.push(J2kClassicCleanupBatchJob {
                coded_offset,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                segment_offset,
                segment_count: u32::try_from(block.code_block.segments.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal segment count exceeds u32".to_string(),
                    }
                })?,
                width: block.code_block.width,
                height: block.code_block.height,
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal output offset overflow".to_string(),
                    })?,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                total_bitplanes: u32::from(block.code_block.total_bitplanes),
                roi_shift: u32::from(block.code_block.roi_shift),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                sub_band_type: match block.code_block.sub_band_type {
                    j2k_native::J2kSubBandType::LowLow => 0,
                    j2k_native::J2kSubBandType::HighLow => 1,
                    j2k_native::J2kSubBandType::LowHigh => 2,
                    j2k_native::J2kSubBandType::HighHigh => 3,
                },
                style_flags: classic_style_flags(block.code_block.style),
                strict: u32::from(block.code_block.strict),
                dequantization_step: block.code_block.dequantization_step,
            });
        }

        dispatch_classic_cleanup_batched(runtime, &coded_data, &jobs, &segments, &decoded)?;
        let decoded_host =
            checked_buffer_slice::<f32>(&decoded, output.len(), "classic sub-band output")?;
        output.copy_from_slice(&decoded_host);
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_code_block(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_ht_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    with_runtime(|runtime| {
        let params = J2kHtCleanupParams {
            width: job.width,
            height: job.height,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            missing_msbs: u32::from(job.missing_bit_planes),
            num_bitplanes: u32::from(job.num_bitplanes),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            dequantization_step: job.dequantization_step,
            stripe_causal: u32::from(job.stripe_causal),
        };
        let decoded = copied_slice_buffer(&runtime.device, output)?;
        dispatch_ht_cleanup(runtime, job.data, params, &decoded)?;
        let decoded_host =
            checked_buffer_slice::<f32>(&decoded, output.len(), "HT cleanup output")?;
        output.copy_from_slice(&decoded_host);

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_sub_band(
    job: HtSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal sub-band size overflow".to_string(),
        })?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal sub-band output slice is too small".to_string(),
        });
    }

    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = copied_slice_buffer(&runtime.device, output)?;

        let coded_len = crate::batch_allocation::checked_count_sum(
            job.jobs.iter().map(|block| block.code_block.data.len()),
            "HTJ2K Metal batched coded payload",
        )?;
        let mut budget =
            crate::batch_allocation::BatchMetadataBudget::new("HTJ2K Metal cleanup batch");
        let mut jobs = budget.try_vec(job.jobs.len(), "HTJ2K Metal cleanup jobs")?;
        let mut coded_data = budget.try_vec(coded_len, "HTJ2K Metal batched coded payload")?;

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);

            jobs.push(J2kHtCleanupBatchJob {
                coded_offset,
                width: block.code_block.width,
                height: block.code_block.height,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                cleanup_length: block.code_block.cleanup_length,
                refinement_length: block.code_block.refinement_length,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                num_bitplanes: u32::from(block.code_block.num_bitplanes),
                roi_shift: u32::from(block.code_block.roi_shift),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal output offset overflow".to_string(),
                    })?,
                dequantization_step: block.code_block.dequantization_step,
                stripe_causal: u32::from(block.code_block.stripe_causal),
            });

            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "HTJ2K Metal batched block lies outside sub-band bounds".to_string(),
                });
            }
        }

        dispatch_ht_cleanup_batched(runtime, &coded_data, &jobs, &decoded)?;
        let decoded_host =
            checked_buffer_slice::<f32>(&decoded, output.len(), "HT sub-band output")?;
        output.copy_from_slice(&decoded_host);
        Ok(())
    })
}
