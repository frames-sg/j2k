// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    classic_style_flags, prepare_direct_tier1_input_buffer, with_runtime, Arc, BandRequiredRegion,
    Buffer, CpuTier1CoefficientCache, DirectTier1Mode, Error, HtCodedArena,
    J2kClassicCleanupBatchJob, J2kClassicSegment, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kHtCleanupBatchJob, PreparedClassicSubBand, PreparedClassicSubBandGroup,
    PreparedClassicSubBandGroupMember, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
    PreparedDirectIdwt, PreparedHtSubBand, PreparedHtSubBandGroup, PreparedHtSubBandGroupMember,
};

#[cfg(target_os = "macos")]
struct ClassicGroupOwners {
    members: Vec<PreparedClassicSubBandGroupMember>,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    segments: Vec<J2kClassicSegment>,
    coded_data: Vec<u8>,
}

#[cfg(target_os = "macos")]
fn allocate_classic_group_owners(
    sub_bands: &[&PreparedClassicSubBand],
) -> Result<ClassicGroupOwners, Error> {
    let job_count = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.jobs.len()),
        "classic J2K MetalDirect grouped jobs",
    )?;
    let segment_count = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.segments.len()),
        "classic J2K MetalDirect grouped segment table",
    )?;
    let coded_len = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.coded_data.len()),
        "classic J2K MetalDirect grouped coded payload",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "classic J2K MetalDirect prepared sub-band group",
    );
    Ok(ClassicGroupOwners {
        members: budget.try_vec(sub_bands.len(), "classic J2K MetalDirect grouped members")?,
        jobs: budget.try_vec(job_count, "classic J2K MetalDirect grouped jobs")?,
        segments: budget.try_vec(segment_count, "classic J2K MetalDirect grouped segments")?,
        coded_data: budget.try_vec(coded_len, "classic J2K MetalDirect grouped coded payload")?,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_classic_sub_band(
    job: &j2k_native::J2kOwnedSubBandPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBand, Error> {
    let coded_len = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.data.len()),
        "classic J2K MetalDirect coded payload",
    )?;
    let segment_count = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.segments.len()),
        "classic J2K MetalDirect segment table",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "classic J2K MetalDirect prepared sub-band",
    );
    let mut jobs = budget.try_vec(job.jobs.len(), "classic J2K MetalDirect jobs")?;
    let mut coded_data = budget.try_vec(coded_len, "classic J2K MetalDirect coded payload")?;
    let mut segments = budget.try_vec(segment_count, "classic J2K MetalDirect segment table")?;

    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
        })?;
        for segment in &block.segments {
            let data_offset = coded_offset
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect segment offset overflow".to_string(),
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
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            segment_offset,
            segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
            })?,
            width: block.width,
            height: block.height,
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect output offset overflow".to_string(),
                })?,
            missing_msbs: u32::from(block.missing_bit_planes),
            total_bitplanes: u32::from(block.total_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            sub_band_type: match block.sub_band_type {
                j2k_native::J2kSubBandType::LowLow => 0,
                j2k_native::J2kSubBandType::HighLow => 1,
                j2k_native::J2kSubBandType::LowHigh => 2,
                j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(block.style),
            strict: u32::from(block.strict),
            dequantization_step: block.dequantization_step,
        });
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode)?;
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode)?;
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &segments, tier1_prepare_mode)?;
        Ok(PreparedClassicSubBand {
            band_id: job.band_id,
            width: job.width,
            height: job.height,
            zero_fill: false,
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_sub_band_groups<'a, SubBand: 'a, Group>(
    steps: &'a [PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
    mut sub_band_for_step: impl FnMut(&'a PreparedDirectGrayscaleStep) -> Option<&'a SubBand>,
    mut prepare_group: impl FnMut(usize, usize, &[&'a SubBand], DirectTier1Mode) -> Result<Group, Error>,
) -> Result<Vec<Group>, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect prepared sub-band groups",
    );
    let mut groups = budget.try_vec(
        steps.len(),
        "J2K MetalDirect prepared sub-band group results",
    )?;
    let mut sub_bands =
        budget.try_vec(steps.len(), "J2K MetalDirect grouped sub-band references")?;
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        sub_bands.clear();
        while let Some(sub_band) = steps.get(step_idx).and_then(&mut sub_band_for_step) {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_group(
                start_step,
                step_idx,
                &sub_bands,
                tier1_prepare_mode,
            )?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_classic_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedClassicSubBandGroup>, Error> {
    prepare_sub_band_groups(
        steps,
        tier1_prepare_mode,
        |step| match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => Some(sub_band),
            _ => None,
        },
        prepare_classic_sub_band_group,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_classic_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedClassicSubBand],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBandGroup, Error> {
    let ClassicGroupOwners {
        mut members,
        mut jobs,
        mut segments,
        mut coded_data,
    } = allocate_classic_group_owners(sub_bands)?;
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedClassicSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let segment_base = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped segment table exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;

        for segment in &sub_band.segments {
            let mut grouped_segment = *segment;
            grouped_segment.data_offset =
                coded_base
                    .checked_add(segment.data_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped segment offset overflow"
                            .to_string(),
                    })?;
            segments.push(grouped_segment);
        }

        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job coded offset overflow"
                            .to_string(),
                    })?;
            grouped_job.segment_offset =
                segment_base
                    .checked_add(job.segment_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job segment offset overflow"
                            .to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped output offset overflow"
                            .to_string(),
                    })?;
            jobs.push(grouped_job);
        }

        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode)?;
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode)?;
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &segments, tier1_prepare_mode)?;
        Ok(PreparedClassicSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            zero_fill: sub_bands.iter().any(|sub_band| sub_band.zero_fill),
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_ht_sub_band(
    job: &j2k_native::HtOwnedSubBandPlan,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBand, Error> {
    let coded_len = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.data.len()),
        "HTJ2K MetalDirect coded payload",
    )?;
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K MetalDirect prepared sub-band");
    let mut jobs = budget.try_vec(job.jobs.len(), "HTJ2K MetalDirect jobs")?;
    let mut coded_data = budget.try_vec(coded_len, "HTJ2K MetalDirect coded payload")?;
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    Ok(PreparedHtSubBand {
        band_id: job.band_id,
        width: job.width,
        height: job.height,
        coded_data,
        coded_buffer: None,
        jobs,
        jobs_buffer: None,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_ht_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedHtSubBandGroup>, Error> {
    prepare_sub_band_groups(
        steps,
        tier1_prepare_mode,
        |step| match step {
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => Some(sub_band),
            _ => None,
        },
        prepare_ht_sub_band_group,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_ht_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedHtSubBand],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBandGroup, Error> {
    let job_count = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.jobs.len()),
        "HTJ2K MetalDirect grouped jobs",
    )?;
    let coded_len = crate::batch_allocation::checked_count_sum(
        sub_bands.iter().map(|sub_band| sub_band.coded_data.len()),
        "HTJ2K MetalDirect grouped coded payload",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K MetalDirect prepared sub-band group",
    );
    let mut members = budget.try_vec(sub_bands.len(), "HTJ2K MetalDirect grouped members")?;
    let mut jobs = budget.try_vec(job_count, "HTJ2K MetalDirect grouped jobs")?;
    let mut coded_data = budget.try_vec(coded_len, "HTJ2K MetalDirect grouped coded payload")?;
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedHtSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;
        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped coded offset overflow".to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped output offset overflow".to_string(),
                    })?;
            jobs.push(grouped_job);
        }
        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode)?;
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode)?;
        Ok(PreparedHtSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            coded_arena: HtCodedArena {
                data: coded_data,
                buffer: coded_buffer,
            },
            jobs,
            jobs_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_ungrouped_ht_sub_band_buffers(
    steps: &mut [PreparedDirectGrayscaleStep],
    groups: &[PreparedHtSubBandGroup],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<(), Error> {
    if tier1_prepare_mode != DirectTier1Mode::Metal {
        return Ok(());
    }

    for (step_idx, step) in steps.iter_mut().enumerate() {
        let PreparedDirectGrayscaleStep::HtSubBand(sub_band) = step else {
            continue;
        };
        if groups
            .iter()
            .any(|group| group.start_step <= step_idx && step_idx < group.end_step)
        {
            sub_band.coded_buffer = None;
            sub_band.jobs_buffer = None;
            continue;
        }
        with_runtime(|runtime| {
            sub_band.coded_buffer = Some(prepare_direct_tier1_input_buffer(
                runtime,
                &sub_band.coded_data,
                tier1_prepare_mode,
            )?);
            sub_band.jobs_buffer = Some(prepare_direct_tier1_input_buffer(
                runtime,
                &sub_band.jobs,
                tier1_prepare_mode,
            )?);
            Ok(())
        })?;
    }

    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn prepared_ht_buffer<'a>(
    buffer: Option<&'a Buffer>,
    label: &str,
) -> Result<&'a Buffer, Error> {
    buffer.ok_or_else(|| Error::MetalKernel {
        message: format!("HTJ2K MetalDirect ungrouped sub-band is missing prepared {label} buffer"),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_grayscale_plan(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_direct_grayscale_plan_for_cpu_upload(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_direct_grayscale_plan_with_tier1_mode(
    plan: &J2kDirectGrayscalePlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect prepared grayscale plan",
    );
    let mut steps = budget.try_vec(plan.steps.len(), "J2K MetalDirect prepared grayscale steps")?;
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::ClassicSubBand(
                    prepare_classic_sub_band(sub_band, tier1_prepare_mode)?,
                ));
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::HtSubBand(prepare_ht_sub_band(
                    sub_band,
                    tier1_prepare_mode,
                )?));
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                steps.push(PreparedDirectGrayscaleStep::Idwt(PreparedDirectIdwt {
                    step: *idwt,
                    output_window: BandRequiredRegion::full(idwt.rect.width(), idwt.rect.height()),
                }));
            }
            J2kDirectGrayscaleStep::Store(store) => {
                steps.push(PreparedDirectGrayscaleStep::Store(*store));
            }
        }
    }
    let classic_groups = prepare_classic_sub_band_groups(&steps, tier1_prepare_mode)?;
    let ht_groups = prepare_ht_sub_band_groups(&steps, tier1_prepare_mode)?;
    prepare_ungrouped_ht_sub_band_buffers(&mut steps, &ht_groups, tier1_prepare_mode)?;
    Ok(PreparedDirectGrayscalePlan {
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        tier1_prepare_mode,
        steps,
        classic_groups,
        ht_groups,
        cpu_tier1_cache: Arc::new(CpuTier1CoefficientCache::default()),
    })
}
