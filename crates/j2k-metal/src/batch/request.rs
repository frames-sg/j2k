// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    cell::OnceCell,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
};

use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};

use crate::{Error, J2kDecoder};

use super::heuristics::group_metal_requests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchOp {
    Full,
    Region(Rect),
    Scaled(Downscale),
    RegionScaled { roi: Rect, scale: Downscale },
}

#[derive(Clone)]
pub(super) struct QueuedRequest {
    pub(super) input: Arc<[u8]>,
    pub(super) fmt: PixelFormat,
    pub(super) backend: BackendRequest,
    pub(super) op: BatchOp,
    pub(super) output_slot: usize,
    pub(super) max_image_dim: OnceCell<Option<u32>>,
    pub(super) input_fingerprint: OnceCell<u64>,
}

impl QueuedRequest {
    pub(super) fn new(
        input: Arc<[u8]>,
        fmt: PixelFormat,
        backend: BackendRequest,
        op: BatchOp,
        output_slot: usize,
    ) -> Self {
        Self {
            input,
            fmt,
            backend,
            op,
            output_slot,
            max_image_dim: OnceCell::new(),
            input_fingerprint: OnceCell::new(),
        }
    }

    pub(super) fn max_image_dim(&self) -> Option<u32> {
        *self.max_image_dim.get_or_init(|| {
            let decoder = J2kDecoder::new(self.input.as_ref()).ok()?;
            let dims = decoder.inner.info().dimensions;
            Some(dims.0.max(dims.1))
        })
    }

    pub(super) fn input_fingerprint(&self) -> u64 {
        *self.input_fingerprint.get_or_init(|| {
            let mut hasher = DefaultHasher::new();
            self.input.len().hash(&mut hasher);
            if !self.input.is_empty() {
                let len = self.input.len();
                for offset in [0, len / 4, len / 2, len.saturating_sub(8)] {
                    let end = offset.saturating_add(8).min(len);
                    self.input[offset..end].hash(&mut hasher);
                }
            }
            hasher.finish()
        })
    }

    #[cfg(test)]
    pub(super) fn max_image_dim_cache_filled_for_test(&self) -> bool {
        self.max_image_dim.get().is_some()
    }

    #[cfg(test)]
    pub(super) fn input_fingerprint_cache_filled_for_test(&self) -> bool {
        self.input_fingerprint.get().is_some()
    }
}

pub(super) fn batch_scheduler_invariant(message: &'static str) -> Error {
    Error::MetalKernel {
        message: format!("internal J2K Metal batch scheduler error: {message}"),
    }
}

#[doc(hidden)]
pub struct BenchmarkGroupedRequests {
    pub batch_count: usize,
    pub max_batch_len: usize,
}

#[doc(hidden)]
pub fn benchmark_group_region_scaled_requests(
    inputs: &[Arc<[u8]>],
    fmt: PixelFormat,
    backend: BackendRequest,
    roi: Rect,
    scale: Downscale,
) -> Result<BenchmarkGroupedRequests, Error> {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal benchmark grouping requests");
    let mut queued = budget.try_vec(inputs.len(), "J2K Metal benchmark queued requests")?;
    queued.extend(inputs.iter().enumerate().map(|(output_slot, input)| {
        QueuedRequest::new(
            input.clone(),
            fmt,
            backend,
            BatchOp::RegionScaled { roi, scale },
            output_slot,
        )
    }));
    let batches = group_metal_requests(queued)?;
    Ok(BenchmarkGroupedRequests {
        batch_count: batches.len(),
        max_batch_len: batches
            .iter()
            .map(|batch| batch.requests.len())
            .max()
            .unwrap_or(0),
    })
}
