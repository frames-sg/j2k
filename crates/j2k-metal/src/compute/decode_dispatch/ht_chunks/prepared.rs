// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device-resident immutable arenas for an exact ordered prepared HT group.

use core::mem::size_of;
use std::sync::Arc;

use j2k_core::HtGpuJobChunkLimits;
use metal::{Buffer, Device};

use super::{
    execution::validate_pass_homogeneous_chunk, HtBatchInput, HtPayloadSource,
    J2kHtCleanupBatchJob, PackedMetalHtChunk,
};
use crate::compute::{copied_slice_buffer, Error, MetalRuntime, PreparedHtExecutionOwner};
use crate::session::{PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES, PREPARED_PLAN_CACHE_MAX_HOST_BYTES};

const PREPARED_METAL_HT_EXECUTION_CACHE_CAP: usize = 128;

struct PreparedMetalHtInputKey {
    owner: Arc<PreparedHtExecutionOwner>,
    payload: PreparedMetalHtPayloadKey,
    jobs_ptr: usize,
    jobs_len: usize,
    source_index: usize,
    output_base: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum PreparedMetalHtPayloadKey {
    Contiguous {
        data_ptr: usize,
        data_len: usize,
    },
    Referenced {
        input_ptr: usize,
        input_len: usize,
        ranges_ptr: usize,
        ranges_len: usize,
    },
}

impl PreparedMetalHtPayloadKey {
    fn from_source(source: HtPayloadSource<'_>) -> Self {
        match source {
            HtPayloadSource::Contiguous(data) => Self::Contiguous {
                data_ptr: data.as_ptr() as usize,
                data_len: data.len(),
            },
            HtPayloadSource::Referenced { input, ranges } => Self::Referenced {
                input_ptr: input.as_ptr() as usize,
                input_len: input.len(),
                ranges_ptr: ranges.as_ptr() as usize,
                ranges_len: ranges.len(),
            },
        }
    }
}

struct PreparedMetalHtExecutionCacheEntry {
    inputs: Vec<PreparedMetalHtInputKey>,
    limits: HtGpuJobChunkLimits,
    execution: Arc<PreparedMetalHtExecution>,
    host_bytes: usize,
    device_bytes: usize,
    last_used: u64,
}

pub(in crate::compute) struct PreparedMetalHtExecutionCache {
    entries: Vec<PreparedMetalHtExecutionCacheEntry>,
    retained_host_bytes: usize,
    retained_device_bytes: usize,
    access_clock: u64,
}

pub(in crate::compute) struct PreparedMetalHtExecution {
    chunks: Vec<PreparedMetalHtChunk>,
    job_count: usize,
}

pub(in crate::compute) struct PreparedMetalHtChunk {
    pub(in crate::compute) bucket: j2k_core::HtGpuJobPassBucket,
    coded_data: Vec<u8>,
    jobs: Vec<J2kHtCleanupBatchJob>,
    pub(in crate::compute) source_indices: Vec<usize>,
    pub(in crate::compute) coded_buffer: Buffer,
    pub(in crate::compute) jobs_buffer: Buffer,
}

impl PreparedMetalHtExecutionCache {
    pub(in crate::compute) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            retained_host_bytes: 0,
            retained_device_bytes: 0,
            access_clock: 0,
        }
    }

    fn get_or_prepare(
        &mut self,
        device: &Device,
        batches: &[HtBatchInput<'_>],
        limits: HtGpuJobChunkLimits,
    ) -> Result<Arc<PreparedMetalHtExecution>, Error> {
        if let Some(index) = self.find(batches, limits) {
            self.access_clock = self.access_clock.saturating_add(1);
            self.entries[index].last_used = self.access_clock;
            return Ok(self.entries[index].execution.clone());
        }

        let execution = Arc::new(PreparedMetalHtExecution::prepare(device, batches, limits)?);
        let (execution_host_bytes, device_bytes) = execution.retained_bytes()?;
        let mut inputs = Vec::new();
        inputs.try_reserve_exact(batches.len()).map_err(|source| {
            Error::PreparedPlanCacheAllocation {
                context: "J2K Metal prepared HT execution cache key",
                source,
            }
        })?;
        inputs.extend(batches.iter().map(|batch| PreparedMetalHtInputKey {
            owner: batch.execution_owner.clone(),
            payload: PreparedMetalHtPayloadKey::from_source(batch.payload),
            jobs_ptr: batch.jobs.as_ptr() as usize,
            jobs_len: batch.jobs.len(),
            source_index: batch.source_index,
            output_base: batch.output_base,
        }));
        let key_host_bytes =
            prepared_metal_ht_input_key_host_bytes(inputs.capacity(), inputs.len())?;
        let host_bytes = execution_host_bytes
            .checked_add(key_host_bytes)
            .and_then(|bytes| {
                bytes
                    .checked_add(size_of::<PreparedMetalHtExecution>())
                    .and_then(|bytes| bytes.checked_add(2 * size_of::<usize>()))
            })
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "prepared execution host byte count overflow",
            })?;
        self.ensure_metadata_capacity()?;
        let metadata_host_bytes = self
            .entries
            .capacity()
            .checked_mul(size_of::<PreparedMetalHtExecutionCacheEntry>())
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "entry metadata byte count overflow",
            })?;
        if !prepared_metal_ht_execution_fits_empty_cache(
            metadata_host_bytes,
            host_bytes,
            device_bytes,
        ) {
            return Ok(execution);
        }

        self.evict_until_fits(host_bytes, device_bytes)?;
        self.access_clock = self.access_clock.saturating_add(1);
        self.entries
            .try_reserve(1)
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K Metal prepared HT execution cache entry",
                source,
            })?;
        self.entries.push(PreparedMetalHtExecutionCacheEntry {
            inputs,
            limits,
            execution: execution.clone(),
            host_bytes,
            device_bytes,
            last_used: self.access_clock,
        });
        self.retained_host_bytes =
            self.retained_host_bytes
                .checked_add(host_bytes)
                .ok_or(Error::MetalStateInvariant {
                    state: "J2K Metal prepared HT execution cache",
                    reason: "retained host byte count overflow",
                })?;
        self.retained_device_bytes = self.retained_device_bytes.checked_add(device_bytes).ok_or(
            Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "retained device byte count overflow",
            },
        )?;
        Ok(execution)
    }

    fn find(&self, batches: &[HtBatchInput<'_>], limits: HtGpuJobChunkLimits) -> Option<usize> {
        self.entries.iter().position(|entry| {
            same_limits(entry.limits, limits)
                && entry.inputs.len() == batches.len()
                && entry.inputs.iter().zip(batches).all(|(key, batch)| {
                    Arc::ptr_eq(&key.owner, batch.execution_owner)
                        && key.payload == PreparedMetalHtPayloadKey::from_source(batch.payload)
                        && key.jobs_ptr == batch.jobs.as_ptr() as usize
                        && key.jobs_len == batch.jobs.len()
                        && key.source_index == batch.source_index
                        && key.output_base == batch.output_base
                })
        })
    }

    fn ensure_metadata_capacity(&mut self) -> Result<(), Error> {
        if self.entries.capacity() >= PREPARED_METAL_HT_EXECUTION_CACHE_CAP {
            return Ok(());
        }
        if !self.entries.is_empty() {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "entry metadata capacity changed after cache insertion",
            });
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(PREPARED_METAL_HT_EXECUTION_CACHE_CAP)
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K Metal prepared HT execution cache metadata",
                source,
            })?;
        let metadata_bytes = entries
            .capacity()
            .checked_mul(size_of::<PreparedMetalHtExecutionCacheEntry>())
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "entry metadata byte count overflow",
            })?;
        if metadata_bytes > PREPARED_PLAN_CACHE_MAX_HOST_BYTES {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "entry metadata exceeds the shared host cache limit",
            });
        }
        self.entries = entries;
        self.retained_host_bytes = metadata_bytes;
        Ok(())
    }

    fn evict_until_fits(
        &mut self,
        incoming_host_bytes: usize,
        incoming_device_bytes: usize,
    ) -> Result<(), Error> {
        while self.entries.len() >= PREPARED_METAL_HT_EXECUTION_CACHE_CAP
            || self
                .retained_host_bytes
                .checked_add(incoming_host_bytes)
                .is_none_or(|bytes| bytes > PREPARED_PLAN_CACHE_MAX_HOST_BYTES)
            || self
                .retained_device_bytes
                .checked_add(incoming_device_bytes)
                .is_none_or(|bytes| bytes > PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES)
        {
            let index = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| (entry.last_used, *index))
                .map(|(index, _)| index)
                .ok_or(Error::MetalStateInvariant {
                    state: "J2K Metal prepared HT execution cache",
                    reason: "cache limits require eviction but no entry is retained",
                })?;
            let entry = self.entries.remove(index);
            self.retained_host_bytes = self
                .retained_host_bytes
                .checked_sub(entry.host_bytes)
                .ok_or(Error::MetalStateInvariant {
                    state: "J2K Metal prepared HT execution cache",
                    reason: "retained host byte count underflow",
                })?;
            self.retained_device_bytes = self
                .retained_device_bytes
                .checked_sub(entry.device_bytes)
                .ok_or(Error::MetalStateInvariant {
                    state: "J2K Metal prepared HT execution cache",
                    reason: "retained device byte count underflow",
                })?;
        }
        Ok(())
    }
}

fn prepared_metal_ht_input_key_host_bytes(capacity: usize, len: usize) -> Result<usize, Error> {
    capacity
        .checked_mul(size_of::<PreparedMetalHtInputKey>())
        .and_then(|bytes| {
            len.checked_mul(2 * size_of::<usize>())
                .and_then(|owners| bytes.checked_add(owners))
        })
        .ok_or(Error::MetalStateInvariant {
            state: "J2K Metal prepared HT execution cache",
            reason: "prepared execution key byte count overflow",
        })
}

fn prepared_metal_ht_execution_fits_empty_cache(
    metadata_host_bytes: usize,
    entry_host_bytes: usize,
    entry_device_bytes: usize,
) -> bool {
    metadata_host_bytes
        .checked_add(entry_host_bytes)
        .is_some_and(|bytes| bytes <= PREPARED_PLAN_CACHE_MAX_HOST_BYTES)
        && entry_device_bytes <= PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES
}

impl PreparedMetalHtExecution {
    fn prepare(
        device: &Device,
        batches: &[HtBatchInput<'_>],
        limits: HtGpuJobChunkLimits,
    ) -> Result<Self, Error> {
        let plan = super::plan_metal_ht_chunks(batches, limits)?;
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(plan.chunk_count())
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K Metal prepared HT execution chunks",
                source,
            })?;
        for chunk_index in 0..plan.chunk_count() {
            let packed = plan.pack_chunk(chunk_index)?;
            validate_pass_homogeneous_chunk(&packed)?;
            #[cfg(test)]
            crate::compute::test_counters::record_ht_immutable_payload_upload();
            let coded_buffer = copied_slice_buffer(device, &packed.coded_data)?;
            #[cfg(test)]
            crate::compute::test_counters::record_ht_immutable_job_upload();
            let jobs_buffer = copied_slice_buffer(device, &packed.jobs)?;
            chunks.push(PreparedMetalHtChunk::new(packed, coded_buffer, jobs_buffer));
        }
        Ok(Self {
            chunks,
            job_count: plan.job_count(),
        })
    }

    pub(in crate::compute) fn chunks(&self) -> &[PreparedMetalHtChunk] {
        &self.chunks
    }

    pub(in crate::compute) const fn job_count(&self) -> usize {
        self.job_count
    }

    fn retained_bytes(&self) -> Result<(usize, usize), Error> {
        let mut host_bytes = self
            .chunks
            .capacity()
            .checked_mul(size_of::<PreparedMetalHtChunk>())
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "chunk owner host byte count overflow",
            })?;
        let mut device_bytes = 0usize;
        for chunk in &self.chunks {
            host_bytes =
                host_bytes
                    .checked_add(chunk.host_bytes()?)
                    .ok_or(Error::MetalStateInvariant {
                        state: "J2K Metal prepared HT execution cache",
                        reason: "aggregate chunk host byte count overflow",
                    })?;
            device_bytes = device_bytes.checked_add(chunk.device_bytes()?).ok_or(
                Error::MetalStateInvariant {
                    state: "J2K Metal prepared HT execution cache",
                    reason: "aggregate chunk device byte count overflow",
                },
            )?;
        }
        Ok((host_bytes, device_bytes))
    }
}

impl PreparedMetalHtChunk {
    fn new(packed: PackedMetalHtChunk, coded_buffer: Buffer, jobs_buffer: Buffer) -> Self {
        Self {
            bucket: packed.bucket,
            coded_data: packed.coded_data,
            jobs: packed.jobs,
            source_indices: packed.source_indices,
            coded_buffer,
            jobs_buffer,
        }
    }

    pub(in crate::compute) fn job_count(&self) -> usize {
        self.jobs.len()
    }

    fn host_bytes(&self) -> Result<usize, Error> {
        self.coded_data
            .capacity()
            .checked_add(
                self.jobs
                    .capacity()
                    .checked_mul(size_of::<J2kHtCleanupBatchJob>())
                    .ok_or(Error::MetalStateInvariant {
                        state: "J2K Metal prepared HT execution cache",
                        reason: "job arena host byte count overflow",
                    })?,
            )
            .and_then(|bytes| {
                self.source_indices
                    .capacity()
                    .checked_mul(size_of::<usize>())
                    .and_then(|indices| bytes.checked_add(indices))
            })
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "chunk host byte count overflow",
            })
    }

    fn device_bytes(&self) -> Result<usize, Error> {
        usize::try_from(self.coded_buffer.length())
            .ok()
            .and_then(|coded| {
                usize::try_from(self.jobs_buffer.length())
                    .ok()
                    .and_then(|jobs| coded.checked_add(jobs))
            })
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal prepared HT execution cache",
                reason: "chunk device byte count overflow",
            })
    }
}

pub(in crate::compute) fn prepared_metal_ht_execution(
    runtime: &MetalRuntime,
    batches: &[HtBatchInput<'_>],
    limits: HtGpuJobChunkLimits,
) -> Result<Arc<PreparedMetalHtExecution>, Error> {
    runtime
        .prepared_ht_execution_cache
        .lock()
        .map_err(|_| Error::MetalStatePoisoned {
            state: "J2K Metal prepared HT execution cache",
        })?
        .get_or_prepare(&runtime.device, batches, limits)
}

fn same_limits(left: HtGpuJobChunkLimits, right: HtGpuJobChunkLimits) -> bool {
    left.max_jobs() == right.max_jobs()
        && left.max_payload_bytes() == right.max_payload_bytes()
        && left.max_descriptor_bytes() == right.max_descriptor_bytes()
}

#[cfg(test)]
mod tests;
