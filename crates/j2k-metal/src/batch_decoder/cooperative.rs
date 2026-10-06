// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit CPU/Metal overlap using the existing prepared plans and completion guards.

use std::{num::NonZeroUsize, sync::Arc};

use super::{
    submission::{allocate_codec_owned_destination, CodecOwnedMetalGroupDestination},
    BatchColor, Error, MetalBatchDecodeResult, MetalBatchDecoder, MetalBatchGroup,
    MetalBatchGroupError, MetalResidentGroupMetadata, PixelFormat, PreparedBatch,
    PreparedBatchGroup, SubmittedMetalResidentGroup,
};
use crate::{engine, metal_types::prelude::*};

struct Split {
    group_index: usize,
    plans: Vec<Arc<engine::PreparedDirectColorPlan>>,
    cpu: Vec<bool>,
}

struct Cohort {
    plans: Vec<Arc<engine::PreparedDirectColorPlan>>,
    positions: Vec<usize>,
    sources: Vec<usize>,
    allocation: CodecOwnedMetalGroupDestination,
}

struct CooperativeSubmission {
    pending: Vec<(usize, SubmittedMetalResidentGroup)>,
    cooperative: Option<CooperativeGroup>,
    group_errors: Vec<MetalBatchGroupError>,
}

struct CooperativeGroup {
    // Rust drops fields in declaration order: retire the producer before
    // releasing either cohort's exclusive output owner on an early return.
    gpu_submission: engine::SubmittedDirectDestination,
    cpu: Cohort,
    gpu: Cohort,
    output: CodecOwnedMetalGroupDestination,
    metadata: MetalResidentGroupMetadata,
}

impl MetalBatchDecoder {
    /// Decode a prepared batch with bounded CPU entropy work overlapping Metal.
    ///
    /// All GPU Tier-1 work is submitted before CPU workers start. At most one
    /// homogeneous group is split, using code-block coefficient/pass counts and
    /// payload sizes to estimate work. `max_cpu_workers` limits the scoped CPU
    /// workers and is capped by the host's available parallelism. Lower budgets
    /// assign less work to the CPU.
    ///
    /// The initial policy admits distinct, full-resolution, reversible classic
    /// RGB8 images with matching plans: at least eight images of at least 16,384
    /// pixels in a group. Other groups, repeated plans, and budgets below two
    /// workers use the existing GPU route. This is an opt-in scheduling policy,
    /// not a guarantee of lower latency on every device or input.
    ///
    /// Output layout, source order, preparation warnings, and group-error handling
    /// match [`Self::decode_prepared`]. Results remain densely packed on Metal;
    /// CPU coefficients are uploaded and both cohorts are gathered on the GPU.
    /// [`crate::MetalBatchGroup::dispatch_report`] records CPU Tier-1 image counts.
    pub fn decode_prepared_cooperative(
        &mut self,
        prepared: &PreparedBatch,
        max_cpu_workers: NonZeroUsize,
    ) -> Result<MetalBatchDecodeResult, Error> {
        let available = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        let workers = max_cpu_workers.get().min(available);
        if workers < 2 {
            return self.decode_prepared(prepared);
        }
        let Some(split) = self.cooperative_split(prepared, workers, available)? else {
            return self.decode_prepared(prepared);
        };
        let CooperativeSubmission {
            pending,
            cooperative,
            mut group_errors,
        } = self.submit_cooperative(prepared, &split)?;
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new("J2K cooperative batch");
        let mut completed =
            budget.try_vec(prepared.groups().len(), "J2K cooperative completed groups")?;
        let mut errors =
            budget.try_vec(prepared.errors().len(), "J2K cooperative indexed errors")?;
        errors.extend_from_slice(prepared.errors());
        let mut fatal = None;
        let mut collect = |index: usize, result: Result<MetalBatchGroup, Error>| match result {
            Ok(group) => completed.push((index, group)),
            Err(source) if source.session_is_unusable() => {
                if fatal.is_none() {
                    fatal = Some(source);
                }
            }
            Err(source) => {
                group_errors.push(MetalBatchGroupError::new(&prepared.groups()[index], source));
            }
        };
        // Every GPU group is now committed. CPU workers cannot delay GPU enqueue.
        if let Some(work) = cooperative {
            collect(
                split.group_index,
                work.complete(self, &prepared.groups()[split.group_index], max_cpu_workers),
            );
        }
        for (index, work) in pending {
            collect(index, work.wait().map_err(|(_, source)| *source));
        }
        if let Some(source) = fatal {
            return Err(source);
        }
        completed.sort_unstable_by_key(|(index, _)| *index);
        let mut groups = budget.try_vec(completed.len(), "J2K cooperative output groups")?;
        groups.extend(completed.into_iter().map(|(_, group)| group));
        Ok(MetalBatchDecodeResult {
            groups,
            errors,
            group_errors,
        })
    }

    /// Commits the GPU cohort and every other group, sharing one classic
    /// Tier-1 dispatch. Submission errors are returned in group order.
    fn submit_cooperative(
        &mut self,
        prepared: &PreparedBatch,
        split: &Split,
    ) -> Result<CooperativeSubmission, Error> {
        let group_count = prepared.groups().len();
        let mut budget =
            crate::batch_allocation::BatchMetadataBudget::new("J2K cooperative submission");
        let mut pending = budget.try_vec(group_count, "J2K cooperative pending groups")?;
        let mut indexed_errors = budget.try_vec(group_count, "J2K cooperative group errors")?;
        let mut residents = budget.try_vec(group_count, "J2K cooperative resident groups")?;
        let mut resident_indices =
            budget.try_vec(group_count, "J2K cooperative resident indices")?;
        let mut cooperative_parts = None;
        for (index, group) in prepared.groups().iter().enumerate() {
            let result = if index == split.group_index {
                CooperativeParts::new(self, group, split)
                    .map(|parts| cooperative_parts = Some(parts))
            } else {
                self.prepare_resident_group(group, prepared.options())
                    .map(|resident| {
                        residents.push(resident);
                        resident_indices.push(index);
                    })
            };
            if let Err(source) = result {
                if source.session_is_unusable() {
                    return Err(source);
                }
                indexed_errors.push((index, MetalBatchGroupError::new(group, source)));
            }
        }
        let split_group = &prepared.groups()[split.group_index];
        let gpu_cohort = cooperative_parts
            .as_ref()
            .map(|parts| engine::ColorGroupSubmission {
                plans: &parts.gpu.plans,
                fmt: PixelFormat::Rgb8,
                layout: split_group.info().layout,
                destination: &parts.gpu.allocation.destination,
                source_indices: Some(&parts.gpu.sources),
                consumer_ordering: engine::DirectDestinationConsumerOrdering::HostCompletionOnly,
            });
        // Pending guards retire all committed work on a fatal error.
        let (submitted, cohort_results) =
            self.submit_resident_groups(residents, gpu_cohort.into_iter().collect())?;
        for (index, result) in resident_indices.into_iter().zip(submitted) {
            match result {
                Ok(work) => pending.push((index, work)),
                Err(error) => indexed_errors.push((index, error)),
            }
        }
        let mut cooperative = None;
        if let (Some(parts), Some(result)) = (cooperative_parts, cohort_results.into_iter().next())
        {
            match result {
                Ok(gpu_submission) => {
                    self.record_submission();
                    cooperative = Some(parts.into_group(gpu_submission));
                }
                Err(source) if source.session_is_unusable() => return Err(source),
                Err(source) => indexed_errors.push((
                    split.group_index,
                    MetalBatchGroupError::new(split_group, source),
                )),
            }
        }
        indexed_errors.sort_by_key(|(index, _)| *index);
        let mut group_errors = budget.try_vec(group_count, "J2K cooperative group errors")?;
        group_errors.extend(indexed_errors.into_iter().map(|(_, error)| error));
        Ok(CooperativeSubmission {
            pending,
            cooperative,
            group_errors,
        })
    }

    fn cooperative_split(
        &mut self,
        prepared: &PreparedBatch,
        workers: usize,
        available: usize,
    ) -> Result<Option<Split>, Error> {
        let eligible_pixels = |group: &PreparedBatchGroup| -> u64 {
            let info = group.info();
            let pixels = u64::from(info.dimensions.0) * u64::from(info.dimensions.1);
            if info.color != BatchColor::Rgb
                || info.precision != 8
                || info.signed
                || info.route != j2k::BatchCodecRoute::Classic
                || info.transform != j2k::BatchWaveletTransform::Reversible53
                || pixels < 128 * 128
                || group
                    .images()
                    .iter()
                    .any(|image| image.request() != j2k::DecodeRequest::Full)
            {
                return 0;
            }
            pixels.saturating_mul(group.images().len() as u64)
        };
        let Some((group_index, group)) = prepared
            .groups()
            .iter()
            .enumerate()
            .filter(|(_, group)| group.images().len() >= 8 && eligible_pixels(group) > 0)
            .max_by_key(|(_, group)| eligible_pixels(group))
        else {
            return Ok(None);
        };
        let plans = match self.prepared_color_group_plans(group, PixelFormat::Rgb8) {
            Ok(plans) => plans,
            Err(source) if source.session_is_unusable() => return Err(source),
            // Ordinary group validation will report this through the existing GPU path.
            Err(_) => return Ok(None),
        };
        let Some(costs) = engine::cooperative_color_costs(&plans)? else {
            return Ok(None);
        };
        let total = costs
            .iter()
            .fold(0u64, |sum, cost| sum.saturating_add(*cost));
        if total == 0 {
            return Ok(None);
        }
        let all_pixels = prepared.groups().iter().fold(0u64, |sum, group| {
            sum.saturating_add(eligible_pixels(group))
        });
        // Calibrated on an M4 Pro (12 CPU cores, 16-core GPU) with all GPU
        // groups sharing one Tier-1 dispatch: at the full worker budget, 3/8 of
        // the batch on the CPU was fastest (8-9 of 24 Kodak images); 10 or more
        // made the CPU the critical path. Partial budgets keep the 1/2 target,
        // which scales to 8 images at 8 workers and 4 at 4, matching the best
        // measured splits.
        let (cpu_share, group_cap) = if workers == available {
            ((3_u128, 8_u128), (4_usize, 5_usize))
        } else {
            ((1, 2), (2, 3))
        };
        let target = (u128::from(total)
            .saturating_mul(u128::from(all_pixels))
            .saturating_mul(workers as u128)
            .saturating_mul(cpu_share.0)
            / u128::from(eligible_pixels(group))
            / available as u128
            / cpu_share.1)
            .min(u128::from(total) * group_cap.0 as u128 / group_cap.1 as u128);
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new("J2K cooperative split");
        let mut ranked = budget.try_vec(costs.len(), "J2K cooperative ranked image work")?;
        ranked.extend(costs.iter().copied().enumerate());
        ranked.sort_unstable_by_key(|&(index, cost)| (std::cmp::Reverse(cost), index));
        let mut cpu = budget.try_filled(plans.len(), false, "J2K cooperative CPU image mask")?;
        let (mut cpu_cost, mut gpu_cost, mut count) = (0u64, 0u64, 0usize);
        for (index, cost) in ranked {
            if count < (plans.len() * group_cap.0 / group_cap.1).min(plans.len() - 2)
                && u128::from(cpu_cost) * (u128::from(total) - target)
                    <= u128::from(gpu_cost) * target
            {
                cpu[index] = true;
                cpu_cost = cpu_cost.saturating_add(cost);
                count += 1;
            } else {
                gpu_cost = gpu_cost.saturating_add(cost);
            }
        }
        if count < 2 {
            return Ok(None);
        }
        Ok(Some(Split {
            group_index,
            plans,
            cpu,
        }))
    }
}

impl Cohort {
    fn new(
        decoder: &MetalBatchDecoder,
        group: &PreparedBatchGroup,
        split: &Split,
        cpu: bool,
    ) -> Result<Self, Error> {
        let count = split.cpu.iter().filter(|&&value| value == cpu).count();
        let mut budget =
            crate::batch_allocation::BatchMetadataBudget::new("J2K cooperative cohort");
        let mut plans = budget.try_vec(count, "J2K cooperative cohort plans")?;
        let mut positions = budget.try_vec(count, "J2K cooperative output positions")?;
        let mut sources = budget.try_vec(count, "J2K cooperative source indices")?;
        for (index, plan) in split.plans.iter().enumerate() {
            if split.cpu[index] == cpu {
                plans.push(plan.clone());
                positions.push(index);
                sources.push(group.source_indices()[index]);
            }
        }
        let allocation = allocate_codec_owned_destination(
            decoder.backend_session().device(),
            group.info().dimensions,
            count,
            PixelFormat::Rgb8,
        )?;
        Ok(Self {
            plans,
            positions,
            sources,
            allocation,
        })
    }
}

/// The split group's cohorts and output before the GPU cohort is submitted.
struct CooperativeParts {
    cpu: Cohort,
    gpu: Cohort,
    output: CodecOwnedMetalGroupDestination,
    metadata: MetalResidentGroupMetadata,
}

impl CooperativeParts {
    fn new(
        decoder: &MetalBatchDecoder,
        group: &PreparedBatchGroup,
        split: &Split,
    ) -> Result<Self, Error> {
        let metadata = MetalResidentGroupMetadata::from_prepared(group, group.options())?;
        let output = super::allocate_codec_owned_group_destination(
            decoder.backend_session().device(),
            group,
            PixelFormat::Rgb8,
        )?;
        let cpu = Cohort::new(decoder, group, split, true)?;
        let gpu = Cohort::new(decoder, group, split, false)?;
        Ok(Self {
            cpu,
            gpu,
            output,
            metadata,
        })
    }

    fn into_group(self, gpu_submission: engine::SubmittedDirectDestination) -> CooperativeGroup {
        CooperativeGroup {
            gpu_submission,
            cpu: self.cpu,
            gpu: self.gpu,
            output: self.output,
            metadata: self.metadata,
        }
    }
}

impl CooperativeGroup {
    fn complete(
        self,
        decoder: &MetalBatchDecoder,
        group: &PreparedBatchGroup,
        workers: NonZeroUsize,
    ) -> Result<MetalBatchGroup, Error> {
        let Self {
            cpu,
            gpu,
            gpu_submission,
            output,
            metadata,
        } = self;
        let runtime = decoder.backend_session().runtime()?;
        let cpu_submission = engine::submit_cooperative_cpu_color_group(
            runtime.clone(),
            &cpu.plans,
            group.info().layout,
            &cpu.allocation.destination,
            &cpu.sources,
            workers,
        );
        // Retire both branches even when either CPU decoding or a GPU command fails.
        let gpu_result = gpu_submission.wait();
        let cpu_result = cpu_submission.and_then(engine::SubmittedDirectDestination::wait);
        let (mut report, cpu_report) = match (gpu_result, cpu_result) {
            (Err(gpu), Err(cpu)) => {
                return Err(if cpu.session_is_unusable() { cpu } else { gpu });
            }
            (Ok(gpu), Ok(cpu)) => (gpu, cpu),
            (Err(source), _) | (_, Err(source)) => return Err(source),
        };
        report.cpu_tier1_images = cpu_report.cpu_tier1_images;
        report.idwt = report.idwt.saturating_add(cpu_report.idwt);
        report.mct = report.mct.saturating_add(cpu_report.mct);
        report.color_output = report.color_output.saturating_add(cpu_report.color_output);
        gather(&runtime, [&gpu, &cpu], &output)?;
        drop(cpu);
        drop(gpu);
        drop(output.destination);
        metadata
            .complete(output.output, output.layout, report)
            .map_err(|(_, source)| *source)
    }
}

fn gather(
    runtime: &engine::MetalRuntime,
    cohorts: [&Cohort; 2],
    output: &CodecOwnedMetalGroupDestination,
) -> Result<(), Error> {
    let support =
        |source| crate::error::metal_kernel_support_error("J2K cooperative output gather", source);
    let command = j2k_metal_support::checked_command_buffer(&runtime.queue).map_err(support)?;
    let blit = j2k_metal_support::checked_blit_command_encoder(&command).map_err(support)?;
    let image_bytes = output.layout.image_stride_bytes();
    let result: Result<(), Error> = (|| {
        for cohort in cohorts {
            for (index, &position) in cohort.positions.iter().enumerate() {
                blit.copy_from_buffer(
                    &cohort.allocation.output,
                    (index * image_bytes) as u64,
                    &output.output,
                    (position * image_bytes) as u64,
                    image_bytes as u64,
                )?;
            }
        }
        Ok(())
    })();
    blit.endEncoding();
    result?;
    j2k_metal_support::commit_and_wait(&command).map_err(support)
}
