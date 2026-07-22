// SPDX-License-Identifier: MIT OR Apache-2.0

//! Identity and retained-byte accounting for one prepared HT cache entry.

use core::mem::size_of;
use std::sync::Arc;

use j2k_core::HtGpuJobChunkLimits;
use metal::Device;

use super::super::super::{HtBatchInput, HtPayloadSource, J2kHtCleanupBatchJob};
use super::super::{PreparedMetalHtChunk, PreparedMetalHtExecution};
use crate::compute::{Error, PreparedHtExecutionOwner};
use crate::session::{PREPARED_PLAN_CACHE_MAX_DEVICE_BYTES, PREPARED_PLAN_CACHE_MAX_HOST_BYTES};

pub(super) struct PreparedMetalHtInputKey {
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

pub(super) struct PreparedMetalHtExecutionCacheEntry {
    inputs: Vec<PreparedMetalHtInputKey>,
    limits: HtGpuJobChunkLimits,
    execution: Arc<PreparedMetalHtExecution>,
    host_bytes: usize,
    device_bytes: usize,
    last_used: u64,
}

impl PreparedMetalHtExecutionCacheEntry {
    pub(super) fn prepare(
        device: &Device,
        batches: &[HtBatchInput<'_>],
        limits: HtGpuJobChunkLimits,
        metadata_host_bytes: usize,
    ) -> Result<(Arc<PreparedMetalHtExecution>, Option<Self>), Error> {
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
        let key_host_bytes = input_key_host_bytes(inputs.capacity(), inputs.len())?;
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
        if !fits_empty_cache(metadata_host_bytes, host_bytes, device_bytes) {
            return Ok((execution, None));
        }

        let entry = Self {
            inputs,
            limits,
            execution: execution.clone(),
            host_bytes,
            device_bytes,
            last_used: 0,
        };
        Ok((execution, Some(entry)))
    }

    pub(super) fn matches(
        &self,
        batches: &[HtBatchInput<'_>],
        limits: HtGpuJobChunkLimits,
    ) -> bool {
        same_limits(self.limits, limits)
            && self.inputs.len() == batches.len()
            && self.inputs.iter().zip(batches).all(|(key, batch)| {
                Arc::ptr_eq(&key.owner, batch.execution_owner)
                    && key.payload == PreparedMetalHtPayloadKey::from_source(batch.payload)
                    && key.jobs_ptr == batch.jobs.as_ptr() as usize
                    && key.jobs_len == batch.jobs.len()
                    && key.source_index == batch.source_index
                    && key.output_base == batch.output_base
            })
    }

    pub(super) fn execution(&self) -> Arc<PreparedMetalHtExecution> {
        self.execution.clone()
    }

    pub(super) const fn host_bytes(&self) -> usize {
        self.host_bytes
    }

    pub(super) const fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    pub(super) const fn last_used(&self) -> u64 {
        self.last_used
    }

    pub(super) fn mark_used(&mut self, access_clock: u64) {
        self.last_used = access_clock;
    }

    #[cfg(test)]
    pub(super) fn set_host_bytes_for_test(&mut self, host_bytes: usize) {
        self.host_bytes = host_bytes;
    }
}

pub(super) fn input_key_host_bytes(capacity: usize, len: usize) -> Result<usize, Error> {
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

pub(super) fn fits_empty_cache(
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

fn same_limits(left: HtGpuJobChunkLimits, right: HtGpuJobChunkLimits) -> bool {
    left.max_jobs() == right.max_jobs()
        && left.max_payload_bytes() == right.max_payload_bytes()
        && left.max_descriptor_bytes() == right.max_descriptor_bytes()
}
