// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::HashMap, sync::Arc, time::Instant};

use j2k_native::HtCodeBlockDecodeWorkspace;
use metal::Buffer;

use crate::Error;

use super::{
    checked_coefficient_len, decode_prepared_classic_jobs_on_cpu_with_scratch,
    decode_prepared_classic_jobs_on_cpu_with_scratch_profiled,
    decode_prepared_ht_jobs_on_cpu_with_workspace,
    decode_prepared_ht_jobs_on_cpu_with_workspace_profiled, elapsed_us, hybrid_stage_signpost,
    metal_profile_stages_enabled, record_flattened_hybrid_cpu_decode_batch,
    record_hybrid_cpu_decode_inputs, record_hybrid_cpu_decode_worker_init,
    upload_cpu_decoded_coefficients, ClassicCpuDecodeScratch, CpuTier1DecodeSubstageCounters,
    DirectHybridStageTimings, J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob,
    MetalRuntime, PreparedDirectColorPlan, PreparedDirectGrayscalePlan,
    PreparedDirectGrayscaleStep, HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK,
    SIGNPOST_DECODE_HYBRID_CPU_TIER1,
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
        coded_data: &'a [u8],
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
    buckets: HashMap<FlattenedCpuTier1Key, FlattenedCpuTier1Bucket>,
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
        let bucket = self.buckets.get(&key).ok_or_else(|| Error::MetalKernel {
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
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
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
    let mut buckets = HashMap::with_capacity(specs.len());
    for (spec, coefficients) in specs.iter().zip(decoded_buckets) {
        let input_count = spec.inputs.len();
        let buffer = upload_cpu_decoded_coefficients(
            runtime,
            coefficients,
            retained_buffers,
            retained_cpu_coefficients,
        );
        buckets.insert(
            spec.key,
            FlattenedCpuTier1Bucket {
                buffer,
                output_len: spec.output_len,
                input_count,
            },
        );
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
    let mut specs = Vec::new();

    for component_idx in 0..3 {
        let component_plans = plans
            .iter()
            .map(|plan| &plan.component_plans[component_idx])
            .collect::<Vec<_>>();
        let Some(first_component) = component_plans.first().copied() else {
            continue;
        };
        let broadcast_tier1_inputs = component_plans
            .iter()
            .all(|plan| std::ptr::eq(*plan, first_component));
        let mut step_idx = 0;
        while step_idx < first.component_plans[component_idx].steps.len() {
            if let Some(group) = first_component.classic_group_starting_at(step_idx) {
                let inputs = component_plans
                    .iter()
                    .take(if broadcast_tier1_inputs {
                        1
                    } else {
                        component_plans.len()
                    })
                    .map(|plan| {
                        let group = plan.classic_group_starting_at(step_idx).ok_or_else(|| {
                            Error::MetalKernel {
                                message: "J2K MetalDirect flattened hybrid missing classic group"
                                    .to_string(),
                            }
                        })?;
                        Ok(FlattenedCpuTier1Source::Classic {
                            coded_data: &group.coded_data,
                            segments: &group.segments,
                            jobs: &group.jobs,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
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
                let inputs = component_plans
                    .iter()
                    .take(if broadcast_tier1_inputs {
                        1
                    } else {
                        component_plans.len()
                    })
                    .map(|plan| {
                        let group = plan.ht_group_starting_at(step_idx).ok_or_else(|| {
                            Error::MetalKernel {
                                message: "J2K MetalDirect flattened hybrid missing HT group"
                                    .to_string(),
                            }
                        })?;
                        Ok(FlattenedCpuTier1Source::Ht {
                            coded_data: &group.coded_arena.data,
                            jobs: &group.jobs,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
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
                    let inputs = component_plans
                        .iter()
                        .take(if broadcast_tier1_inputs {
                            1
                        } else {
                            component_plans.len()
                        })
                        .map(|plan| match &plan.steps[step_idx] {
                            PreparedDirectGrayscaleStep::ClassicSubBand(other) => {
                                Ok(FlattenedCpuTier1Source::Classic {
                                    coded_data: &other.coded_data,
                                    segments: &other.segments,
                                    jobs: &other.jobs,
                                })
                            }
                            _ => Err(Error::MetalKernel {
                                message:
                                    "J2K MetalDirect flattened hybrid missing classic sub-band"
                                        .to_string(),
                            }),
                        })
                        .collect::<Result<Vec<_>, Error>>()?;
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
                    let inputs = component_plans
                        .iter()
                        .take(if broadcast_tier1_inputs {
                            1
                        } else {
                            component_plans.len()
                        })
                        .map(|plan| match &plan.steps[step_idx] {
                            PreparedDirectGrayscaleStep::HtSubBand(other) => {
                                Ok(FlattenedCpuTier1Source::Ht {
                                    coded_data: &other.coded_data,
                                    jobs: &other.jobs,
                                })
                            }
                            _ => Err(Error::MetalKernel {
                                message: "J2K MetalDirect flattened hybrid missing HT sub-band"
                                    .to_string(),
                            }),
                        })
                        .collect::<Result<Vec<_>, Error>>()?;
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
    let mut buckets = Vec::with_capacity(specs.len());
    let mut cache_targets = Vec::with_capacity(specs.len());
    for spec in specs {
        if let Some(cache_plan) = spec.cache_plan {
            if let Some(coefficients) =
                cache_plan.cached_cpu_tier1_coefficients(spec.key.step_idx, spec.output_len)?
            {
                buckets.push(coefficients);
                cache_targets.push(None);
                continue;
            }
            cache_targets.push(Some((cache_plan, spec.key.step_idx, spec.output_len)));
        } else {
            cache_targets.push(None);
        }
        buckets.push(packed_cpu_decode_coefficients(
            spec.inputs.len(),
            spec.output_len,
        )?);
    }
    let mut work_items = Vec::new();
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
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
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
        FlattenedCpuTier1Source::Ht { coded_data, jobs } => {
            if let Some(counters) = profile_counters {
                decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
                    coded_data,
                    jobs,
                    output,
                    &mut scratch.ht,
                    counters,
                )
            } else {
                decode_prepared_ht_jobs_on_cpu_with_workspace(
                    coded_data,
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
    let total_len = count
        .checked_mul(output_len)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect hybrid packed coefficient length overflows usize".to_string(),
        })?;
    Ok(vec![0.0_f32; total_len])
}
