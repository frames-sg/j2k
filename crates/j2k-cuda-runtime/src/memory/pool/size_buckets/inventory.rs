// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaBufferPoolSizeBuckets;

impl CudaBufferPoolSizeBuckets {
    pub(crate) fn cached_count(&self) -> usize {
        self.buckets.iter().map(|bucket| bucket.buffers.len()).sum()
    }

    pub(crate) fn cached_bytes(&self) -> usize {
        self.buckets.iter().fold(0usize, |total, bucket| {
            total.saturating_add(bucket.size.saturating_mul(bucket.buffers.len()))
        })
    }

    pub(crate) fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    pub(crate) fn contains_size(&self, size: usize) -> bool {
        self.buckets
            .binary_search_by_key(&size, |bucket| bucket.size)
            .is_ok()
    }
}
