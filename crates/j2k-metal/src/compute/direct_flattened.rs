// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{sync::Arc, time::Instant};

use j2k_native::HtCodeBlockDecodeWorkspace;
use metal::Buffer;

use crate::profile_env::{hybrid_stage_signpost, SIGNPOST_DECODE_HYBRID_CPU_TIER1};
use crate::Error;

use super::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob};
use super::direct_grayscale_execute::{checked_coefficient_len, upload_cpu_decoded_coefficients};
use super::{
    decode_prepared_classic_jobs_on_cpu_with_scratch,
    decode_prepared_classic_jobs_on_cpu_with_scratch_profiled,
    decode_prepared_ht_jobs_on_cpu_with_workspace,
    decode_prepared_ht_jobs_on_cpu_with_workspace_profiled, elapsed_us,
    metal_profile_stages_enabled, record_flattened_hybrid_cpu_decode_batch,
    record_hybrid_cpu_decode_inputs, record_hybrid_cpu_decode_worker_init, ClassicCpuDecodeScratch,
    CpuTier1DecodeSubstageCounters, DirectHybridStageTimings, MetalRuntime,
    PreparedDirectColorPlan, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
    PreparedHtPayloadSource, HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK,
};

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FlattenedCpuTier1Key {
    component_idx: usize,
    step_idx: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
enum FlattenedCpuTier1Source<'a> {
    Classic {
        coded_data: &'a [u8],
        segments: &'a [J2kClassicSegment],
        jobs: &'a [J2kClassicCleanupBatchJob],
    },
    Ht {
        payload_source: &'a PreparedHtPayloadSource,
        jobs: &'a [J2kHtCleanupBatchJob],
    },
}

#[cfg(target_os = "macos")]
struct FlattenedCpuTier1BucketSpec<'a> {
    key: FlattenedCpuTier1Key,
    output_len: usize,
    cache_plan: Option<&'a PreparedDirectGrayscalePlan>,
    inputs: Vec<FlattenedCpuTier1Source<'a>>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct FlattenedCpuTier1Bucket {
    buffer: Buffer,
    output_len: usize,
    input_count: usize,
}

#[cfg(target_os = "macos")]
pub(super) struct FlattenedCpuTier1Cache {
    buckets: Vec<(FlattenedCpuTier1Key, FlattenedCpuTier1Bucket)>,
}

#[cfg(target_os = "macos")]
impl FlattenedCpuTier1Cache {
    pub(super) fn buffer_for(
        &self,
        component_idx: usize,
        step_idx: usize,
        output_len: usize,
        input_count: usize,
    ) -> Result<Buffer, Error> {
        let key = FlattenedCpuTier1Key {
            component_idx,
            step_idx,
        };
        let bucket = self
            .buckets
            .iter()
            .find_map(|(candidate, bucket)| (*candidate == key).then_some(bucket))
            .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "J2K MetalDirect flattened hybrid cache is missing component {component_idx} step {step_idx}"
            ),
        })?;
        if bucket.output_len != output_len || bucket.input_count != input_count {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K MetalDirect flattened hybrid cache shape mismatch at component {component_idx} step {step_idx}"
                ),
            });
        }
        Ok(bucket.buffer.clone())
    }
}

#[cfg(target_os = "macos")]
struct FlattenedCpuTier1WorkItem<'a> {
    output_len: usize,
    output: FlattenedCpuTier1Output,
    source: FlattenedCpuTier1Source<'a>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct FlattenedCpuTier1Output(*mut f32);

// SAFETY: Work items are constructed from non-overlapping ranges in preallocated
// packed coefficient slabs. Each pointer is written exactly once before the
// owning Vec is moved or exposed again.
#[cfg(target_os = "macos")]
// SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
unsafe impl Send for FlattenedCpuTier1Output {}

#[cfg(target_os = "macos")]
// SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
unsafe impl Sync for FlattenedCpuTier1Output {}

#[cfg(target_os = "macos")]
impl FlattenedCpuTier1Output {
    unsafe fn as_slice_mut<'a>(self, len: usize) -> &'a mut [f32] {
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        unsafe { std::slice::from_raw_parts_mut(self.0, len) }
    }
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct FlattenedCpuTier1DecodeScratch {
    classic: ClassicCpuDecodeScratch,
    ht: HtCodeBlockDecodeWorkspace,
}

#[cfg(target_os = "macos")]
pub(super) fn build_flattened_cpu_tier1_cache(
    runtime: &MetalRuntime,
    plans: &[Arc<PreparedDirectColorPlan>],
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
) -> Result<FlattenedCpuTier1Cache, Error> {
    let specs = collect_flattened_cpu_tier1_bucket_specs(plans)?;
    stage_timings.cpu_tier1_flattened_batches =
        stage_timings.cpu_tier1_flattened_batches.saturating_add(1);
    let decode_started = metal_profile_stages_enabled().then(Instant::now);
    let cpu_tier1_counters =
        metal_profile_stages_enabled().then(CpuTier1DecodeSubstageCounters::default);
    let decoded_buckets = decode_flattened_cpu_tier1_buckets(&specs, cpu_tier1_counters.as_ref())?;
    if let Some(started) = decode_started {
        stage_timings.cpu_tier1 += elapsed_us(started);
    }
    if let Some(counters) = &cpu_tier1_counters {
        counters.add_to_stage_timings(stage_timings);
    }

    let upload_started = metal_profile_stages_enabled().then(Instant::now);
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K MetalDirect flattened Tier-1 cache");
    let mut buckets = budget.try_vec(
        specs.len(),
        "J2K MetalDirect flattened Tier-1 cache buckets",
    )?;
    for (spec, coefficients) in specs.iter().zip(decoded_buckets) {
        let input_count = spec.inputs.len();
        let buffer = upload_cpu_decoded_coefficients(runtime, &coefficients, retained_buffers)?;
        buckets.push((
            spec.key,
            FlattenedCpuTier1Bucket {
                buffer,
                output_len: spec.output_len,
                input_count,
            },
        ));
    }
    if let Some(started) = upload_started {
        stage_timings.coefficient_upload += elapsed_us(started);
    }

    Ok(FlattenedCpuTier1Cache { buckets })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "single-pass bucket planning keeps cross-plan offsets and validation co-located"
)]
fn collect_flattened_cpu_tier1_bucket_specs(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> Result<Vec<FlattenedCpuTier1BucketSpec<'_>>, Error> {
    let Some(first) = plans.first() else {
        return Ok(Vec::new());
    };
    let spec_capacity = crate::batch_allocation::checked_count_sum(
        first
            .component_plans
            .iter()
            .map(|component| component.steps.len()),
        "J2K MetalDirect flattened bucket specifications",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect flattened bucket planning",
    );
    let mut specs = budget.try_vec(
        spec_capacity,
        "J2K MetalDirect flattened bucket specifications",
    )?;

    for component_idx in 0..3 {
        let mut component_plans = budget.try_vec(
            plans.len(),
            "J2K MetalDirect flattened component plan references",
        )?;
        component_plans.extend(
            plans
                .iter()
                .map(|plan| &plan.component_plans[component_idx]),
        );
        let Some(first_component) = component_plans.first().copied() else {
            continue;
        };
        let broadcast_tier1_inputs = component_plans
            .iter()
            .all(|plan| std::ptr::eq(*plan, first_component));
        let mut step_idx = 0;
        while step_idx < first.component_plans[component_idx].steps.len() {
            if let Some(group) = first_component.classic_group_starting_at(step_idx) {
                let input_count = if broadcast_tier1_inputs {
                    1
                } else {
                    component_plans.len()
                };
                let mut inputs = budget.try_vec(
                    input_count,
                    "J2K MetalDirect flattened classic group inputs",
                )?;
                for plan in component_plans.iter().take(input_count) {
                    let group = plan.classic_group_starting_at(step_idx).ok_or_else(|| {
                        Error::MetalKernel {
                            message: "J2K MetalDirect flattened hybrid missing classic group"
                                .to_string(),
                        }
                    })?;
                    inputs.push(FlattenedCpuTier1Source::Classic {
                        coded_data: &group.coded_data,
                        segments: &group.segments,
                        jobs: &group.jobs,
                    });
                }
                specs.push(FlattenedCpuTier1BucketSpec {
                    key: FlattenedCpuTier1Key {
                        component_idx,
                        step_idx,
                    },
                    output_len: group.total_coefficients,
                    cache_plan: (inputs.len() == 1).then_some(first_component),
                    inputs,
                });
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = first_component.ht_group_starting_at(step_idx) {
                let input_count = if broadcast_tier1_inputs {
                    1
                } else {
                    component_plans.len()
                };
                let mut inputs =
                    budget.try_vec(input_count, "J2K MetalDirect flattened HT group inputs")?;
                for plan in component_plans.iter().take(input_count) {
                    let group =
                        plan.ht_group_starting_at(step_idx)
                            .ok_or_else(|| Error::MetalKernel {
                                message: "J2K MetalDirect flattened hybrid missing HT group"
                                    .to_string(),
                            })?;
                    inputs.push(FlattenedCpuTier1Source::Ht {
                        payload_source: &group.payload_source,
                        jobs: &group.jobs,
                    });
                }
                specs.push(FlattenedCpuTier1BucketSpec {
                    key: FlattenedCpuTier1Key {
                        component_idx,
                        step_idx,
                    },
                    output_len: group.total_coefficients,
                    cache_plan: (inputs.len() == 1).then_some(first_component),
                    inputs,
                });
                step_idx = group.end_step;
                continue;
            }

            match &first_component.steps[step_idx] {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let output_len = checked_coefficient_len(
                        sub_band.width,
                        sub_band.height,
                        "classic J2K MetalDirect flattened hybrid sub-band size overflow",
                    )?;
                    let input_count = if broadcast_tier1_inputs {
                        1
                    } else {
                        component_plans.len()
                    };
                    let mut inputs = budget.try_vec(
                        input_count,
                        "J2K MetalDirect flattened classic sub-band inputs",
                    )?;
                    for plan in component_plans.iter().take(input_count) {
                        match &plan.steps[step_idx] {
                            PreparedDirectGrayscaleStep::ClassicSubBand(other) => {
                                inputs.push(FlattenedCpuTier1Source::Classic {
                                    coded_data: &other.coded_data,
                                    segments: &other.segments,
                                    jobs: &other.jobs,
                                });
                            }
                            _ => {
                                return Err(Error::MetalKernel {
                                    message:
                                        "J2K MetalDirect flattened hybrid missing classic sub-band"
                                            .to_string(),
                                });
                            }
                        }
                    }
                    specs.push(FlattenedCpuTier1BucketSpec {
                        key: FlattenedCpuTier1Key {
                            component_idx,
                            step_idx,
                        },
                        output_len,
                        cache_plan: (inputs.len() == 1).then_some(first_component),
                        inputs,
                    });
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let output_len = checked_coefficient_len(
                        sub_band.width,
                        sub_band.height,
                        "HTJ2K MetalDirect flattened hybrid sub-band size overflow",
                    )?;
                    let input_count = if broadcast_tier1_inputs {
                        1
                    } else {
                        component_plans.len()
                    };
                    let mut inputs = budget
                        .try_vec(input_count, "J2K MetalDirect flattened HT sub-band inputs")?;
                    for plan in component_plans.iter().take(input_count) {
                        match &plan.steps[step_idx] {
                            PreparedDirectGrayscaleStep::HtSubBand(other) => {
                                inputs.push(FlattenedCpuTier1Source::Ht {
                                    payload_source: &other.payload_source,
                                    jobs: &other.jobs,
                                });
                            }
                            _ => {
                                return Err(Error::MetalKernel {
                                    message: "J2K MetalDirect flattened hybrid missing HT sub-band"
                                        .to_string(),
                                });
                            }
                        }
                    }
                    specs.push(FlattenedCpuTier1BucketSpec {
                        key: FlattenedCpuTier1Key {
                            component_idx,
                            step_idx,
                        },
                        output_len,
                        cache_plan: (inputs.len() == 1).then_some(first_component),
                        inputs,
                    });
                }
                PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => {}
            }
            step_idx += 1;
        }
    }

    Ok(specs)
}

#[cfg(target_os = "macos")]
fn decode_flattened_cpu_tier1_buckets(
    specs: &[FlattenedCpuTier1BucketSpec<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<Vec<f32>>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal flattened CPU Tier-1 batch");
    let mut buckets = budget.try_vec(specs.len(), "J2K Metal flattened Tier-1 buckets")?;
    let mut cache_targets =
        budget.try_vec(specs.len(), "J2K Metal flattened Tier-1 cache targets")?;
    for spec in specs {
        if let Some(cache_plan) = spec.cache_plan {
            if let Some(coefficients) = cache_plan.cached_cpu_tier1_coefficients(
                &mut budget,
                spec.key.step_idx,
                spec.output_len,
            )? {
                buckets.push(coefficients);
                cache_targets.push(None);
                continue;
            }
            cache_targets.push(Some((cache_plan, spec.key.step_idx, spec.output_len)));
        } else {
            cache_targets.push(None);
        }
        buckets.push(packed_cpu_decode_coefficients_in(
            &mut budget,
            spec.inputs.len(),
            spec.output_len,
        )?);
    }
    let work_item_count = specs.iter().try_fold(0usize, |total, spec| {
        total.checked_add(spec.inputs.len()).ok_or(
            j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "J2K Metal flattened Tier-1 work items",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            },
        )
    })?;
    let mut work_items =
        budget.try_vec(work_item_count, "J2K Metal flattened Tier-1 work items")?;
    for (bucket_idx, spec) in specs.iter().enumerate() {
        if cache_targets[bucket_idx].is_none() && spec.cache_plan.is_some() {
            continue;
        }
        for (input_idx, source) in spec.inputs.iter().copied().enumerate() {
            let start =
                input_idx
                    .checked_mul(spec.output_len)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K MetalDirect flattened hybrid bucket offset overflow"
                            .to_string(),
                    })?;
            let end = start
                .checked_add(spec.output_len)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect flattened hybrid bucket end overflow".to_string(),
                })?;
            if end > buckets[bucket_idx].len() {
                return Err(Error::MetalKernel {
                    message: "J2K MetalDirect flattened hybrid bucket slice is too small"
                        .to_string(),
                });
            }
            let output =
                // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
                FlattenedCpuTier1Output(unsafe { buckets[bucket_idx].as_mut_ptr().add(start) });
            work_items.push(FlattenedCpuTier1WorkItem {
                output_len: spec.output_len,
                output,
                source,
            });
        }
    }

    if !work_items.is_empty() {
        record_flattened_hybrid_cpu_decode_batch();
        record_hybrid_cpu_decode_inputs(work_items.len());

        decode_flattened_cpu_tier1_work_items_chunked(&work_items, profile_counters)?;

        for (bucket, cache_target) in buckets.iter_mut().zip(cache_targets) {
            if let Some((cache_plan, step_idx, output_len)) = cache_target {
                let coefficients = std::mem::take(bucket);
                *bucket =
                    cache_plan.store_cpu_tier1_coefficients(step_idx, output_len, coefficients)?;
            }
        }
    }

    Ok(buckets)
}

#[cfg(target_os = "macos")]
fn decode_flattened_cpu_tier1_work_items_chunked(
    work_items: &[FlattenedCpuTier1WorkItem<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<(), Error> {
    if work_items.is_empty() {
        return Ok(());
    }

    let worker_count = hybrid_cpu_decode_worker_count(work_items.len());
    let chunk_size = work_items.len().div_ceil(worker_count);
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal flattened Tier-1 worker handles",
    );
    std::thread::scope(|scope| -> Result<(), Error> {
        let mut handles =
            budget.try_vec(worker_count, "J2K Metal flattened Tier-1 worker handles")?;
        for chunk in work_items.chunks(chunk_size) {
            handles.push(scope.spawn(move || {
                record_hybrid_cpu_decode_worker_init();
                let mut scratch = FlattenedCpuTier1DecodeScratch::default();
                for item in chunk {
                    decode_flattened_cpu_tier1_work_item(item, &mut scratch, profile_counters)?;
                }
                Ok(())
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(error),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(super) fn hybrid_cpu_decode_worker_count(item_count: usize) -> usize {
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let useful = item_count
        .div_ceil(HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK.max(1))
        .max(1);
    available.min(useful).max(1)
}

#[cfg(target_os = "macos")]
fn decode_flattened_cpu_tier1_work_item(
    item: &FlattenedCpuTier1WorkItem<'_>,
    scratch: &mut FlattenedCpuTier1DecodeScratch,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<(), Error> {
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let output = unsafe { item.output.as_slice_mut(item.output_len) };
    match item.source {
        FlattenedCpuTier1Source::Classic {
            coded_data,
            segments,
            jobs,
        } => {
            if let Some(counters) = profile_counters {
                decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
                    coded_data,
                    segments,
                    jobs,
                    output,
                    &mut scratch.classic,
                    counters,
                )
            } else {
                decode_prepared_classic_jobs_on_cpu_with_scratch(
                    coded_data,
                    segments,
                    jobs,
                    output,
                    &mut scratch.classic,
                )
            }
        }
        FlattenedCpuTier1Source::Ht {
            payload_source,
            jobs,
        } => {
            let coded_data = payload_source.materialize_for_cpu()?;
            if let Some(counters) = profile_counters {
                decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
                    coded_data.as_ref(),
                    jobs,
                    output,
                    &mut scratch.ht,
                    counters,
                )
            } else {
                decode_prepared_ht_jobs_on_cpu_with_workspace(
                    coded_data.as_ref(),
                    jobs,
                    output,
                    &mut scratch.ht,
                )
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) fn packed_cpu_decode_output_len(
    output_lens: impl IntoIterator<Item = usize>,
    label: &str,
) -> Result<Option<usize>, Error> {
    let mut output_lens = output_lens.into_iter();
    let Some(output_len) = output_lens.next() else {
        return Ok(None);
    };
    if output_len == 0 {
        return Ok(None);
    }
    if output_lens.any(|other| other != output_len) {
        return Err(Error::MetalKernel {
            message: format!("{label} has mixed coefficient lengths"),
        });
    }
    Ok(Some(output_len))
}

#[cfg(target_os = "macos")]
pub(super) fn packed_cpu_decode_coefficients(
    count: usize,
    output_len: usize,
) -> Result<Vec<f32>, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K MetalDirect hybrid packed coefficients",
    );
    packed_cpu_decode_coefficients_in(&mut budget, count, output_len)
}

#[cfg(target_os = "macos")]
pub(super) fn packed_cpu_decode_coefficients_in(
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
    count: usize,
    output_len: usize,
) -> Result<Vec<f32>, Error> {
    let total_len = crate::batch_allocation::checked_count_product(
        count,
        output_len,
        "J2K MetalDirect hybrid packed coefficients",
    )?;
    Ok(budget.try_filled(
        total_len,
        0.0_f32,
        "J2K MetalDirect hybrid packed coefficient values",
    )?)
}

#[cfg(test)]
mod tests;
