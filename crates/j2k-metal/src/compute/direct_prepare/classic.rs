// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic JPEG 2000 sub-band and grouped payload preparation.

use super::{
    classic_style_flags, prepare_direct_tier1_input_buffer, with_runtime, BandRequiredRegion,
    DirectTier1Mode, Error, J2kClassicCleanupBatchJob, J2kClassicCodeBlockPayload,
    J2kClassicSegment, J2kCodestreamRange, J2kReferencedClassicPlan, PreparedClassicSubBand,
    PreparedClassicSubBandGroup, PreparedClassicSubBandGroupMember, PreparedDirectGrayscaleStep,
};

#[cfg(target_os = "macos")]
struct ClassicGroupOwners {
    members: Vec<PreparedClassicSubBandGroupMember>,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    segments: Vec<J2kClassicSegment>,
    coded_data: Vec<u8>,
}

#[cfg(target_os = "macos")]
pub(super) struct ReferencedClassicPayloadCursor<'a> {
    input: &'a [u8],
    payloads: &'a [J2kClassicCodeBlockPayload],
    ranges: &'a [J2kCodestreamRange],
    pub(super) next_payload: usize,
    next_range: usize,
}

#[cfg(target_os = "macos")]
impl<'a> ReferencedClassicPayloadCursor<'a> {
    pub(super) fn new(input: &'a [u8], plan: &'a J2kReferencedClassicPlan) -> Self {
        Self {
            input,
            payloads: plan.payloads(),
            ranges: plan.ranges(),
            next_payload: 0,
            next_range: 0,
        }
    }

    fn expected_payload_bytes(&self, count: usize) -> Result<usize, Error> {
        let end = self
            .next_payload
            .checked_add(count)
            .ok_or(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "payload traversal count overflowed",
            })?;
        let payloads =
            self.payloads
                .get(self.next_payload..end)
                .ok_or(Error::MetalStateInvariant {
                    state: "classic J2K referenced payload cursor",
                    reason: "geometry contains more jobs than retained payload descriptors",
                })?;
        Ok(crate::batch_allocation::checked_count_sum(
            payloads.iter().map(|payload| payload.combined_length),
            "classic J2K referenced Metal coded payload",
        )?)
    }

    fn append_next(&mut self, coded_data: &mut Vec<u8>) -> Result<usize, Error> {
        let payload =
            self.payloads
                .get(self.next_payload)
                .copied()
                .ok_or(Error::MetalStateInvariant {
                    state: "classic J2K referenced payload cursor",
                    reason: "geometry contains more jobs than retained payload descriptors",
                })?;
        if payload.first_range != self.next_range {
            return Err(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "payload fragment ranges are not contiguous in traversal order",
            });
        }
        let end_range = payload.end_range().ok_or(Error::MetalStateInvariant {
            state: "classic J2K referenced payload cursor",
            reason: "payload fragment range overflowed",
        })?;
        let fragments =
            self.ranges
                .get(payload.first_range..end_range)
                .ok_or(Error::MetalStateInvariant {
                    state: "classic J2K referenced payload cursor",
                    reason: "payload fragment range exceeds the retained range table",
                })?;
        let before = coded_data.len();
        for range in fragments {
            let end = range.end().ok_or(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "encoded payload byte range overflowed",
            })?;
            let fragment = self
                .input
                .get(range.offset..end)
                .ok_or(Error::MetalStateInvariant {
                    state: "classic J2K referenced payload cursor",
                    reason: "encoded payload byte range exceeds the retained input",
                })?;
            coded_data.extend_from_slice(fragment);
        }
        let appended = coded_data
            .len()
            .checked_sub(before)
            .ok_or(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "coded payload length moved backwards",
            })?;
        if appended != payload.combined_length {
            return Err(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "concatenated fragments do not match their retained payload length",
            });
        }
        self.next_payload = self
            .next_payload
            .checked_add(1)
            .ok_or(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "payload cursor overflowed",
            })?;
        self.next_range = end_range;
        Ok(appended)
    }

    pub(super) fn ensure_exhausted(&self) -> Result<(), Error> {
        if self.next_payload == self.payloads.len() && self.next_range == self.ranges.len() {
            Ok(())
        } else {
            Err(Error::MetalStateInvariant {
                state: "classic J2K referenced payload cursor",
                reason: "retained payload descriptors or ranges were left unused",
            })
        }
    }
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
pub(in crate::compute) fn prepare_classic_sub_band(
    job: &j2k_native::J2kOwnedSubBandPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBand, Error> {
    let coded_len = crate::batch_allocation::checked_count_sum(
        job.jobs.iter().map(|block| block.data.len()),
        "classic J2K MetalDirect coded payload",
    )?;
    prepare_classic_sub_band_with_payloads(
        job,
        tier1_prepare_mode,
        coded_len,
        |block_index, coded_data| {
            let before = coded_data.len();
            coded_data.extend_from_slice(&job.jobs[block_index].data);
            Ok(coded_data.len() - before)
        },
    )
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_with_payloads(
    job: &j2k_native::J2kOwnedSubBandPlan,
    tier1_prepare_mode: DirectTier1Mode,
    coded_len: usize,
    mut append_payload: impl FnMut(usize, &mut Vec<u8>) -> Result<usize, Error>,
) -> Result<PreparedClassicSubBand, Error> {
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

    for (block_index, block) in job.jobs.iter().enumerate() {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        let block_coded_len = append_payload(block_index, &mut coded_data)?;
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
            coded_len: u32::try_from(block_coded_len).map_err(|_| Error::MetalKernel {
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
    if coded_data.len() != coded_len {
        return Err(Error::MetalStateInvariant {
            state: "classic J2K MetalDirect prepared sub-band",
            reason: "appended payload bytes do not match the preflight allocation",
        });
    }
    let zero_fill = jobs
        .iter()
        .any(|job| job.coded_len == 0 || job.number_of_coding_passes == 0);

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
            zero_fill,
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
pub(super) fn prepare_referenced_classic_sub_band(
    job: &j2k_native::J2kOwnedSubBandPlan,
    payloads: &mut ReferencedClassicPayloadCursor<'_>,
) -> Result<PreparedClassicSubBand, Error> {
    if job.jobs.iter().any(|block| !block.data.is_empty()) {
        return Err(Error::MetalStateInvariant {
            state: "classic J2K referenced Metal sub-band",
            reason: "referenced geometry unexpectedly owns compressed payload bytes",
        });
    }
    let coded_len = payloads.expected_payload_bytes(job.jobs.len())?;
    prepare_classic_sub_band_with_payloads(
        job,
        DirectTier1Mode::Metal,
        coded_len,
        |_, coded_data| payloads.append_next(coded_data),
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn prepare_sub_band_groups<'a, SubBand: 'a, Group>(
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
pub(in crate::compute) fn prepare_classic_sub_band_groups(
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
