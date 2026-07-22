// SPDX-License-Identifier: MIT OR Apache-2.0

//! Caller-owned CUDA destination submission and completion.

use super::{BatchGroupInfo, CudaBatchError, Error, J2kDecodeWarning, Rect};

/// Metadata for a homogeneous group decoded directly into caller-owned CUDA
/// storage.
#[cfg(feature = "cuda-runtime")]
#[derive(Debug)]
pub struct CudaExternalBatchGroup {
    pub(super) info: BatchGroupInfo,
    pub(super) source_indices: Vec<usize>,
    pub(super) decoded_rects: Vec<Rect>,
    pub(super) warnings: Vec<Vec<J2kDecodeWarning>>,
    pub(super) ranges: Vec<j2k_cuda_runtime::CudaDeviceBufferRange>,
}

/// Asynchronously submitted CUDA external-destination batch.
///
/// The destination ranges are metadata-only until the caller stream has been
/// ordered after codec completion or [`Self::wait`] succeeds. Dropping this
/// value waits before releasing codec-internal resources.
#[cfg(feature = "cuda-runtime")]
#[must_use = "submitted CUDA decode must be retained or waited"]
pub struct SubmittedCudaExternalBatch {
    pub(super) group: CudaExternalBatchGroup,
    pub(super) pending: SubmittedCudaCodecBatch,
}

#[cfg(feature = "cuda-runtime")]
pub(super) enum SubmittedCudaCodecBatch {
    Grayscale(crate::decoder::grayscale_batch::SubmittedGrayscaleExternalBatch),
    Color(crate::decoder::SubmittedNativeColorExternalBatch),
}

#[cfg(feature = "cuda-runtime")]
impl SubmittedCudaCodecBatch {
    pub(super) fn ranges(&self) -> &[j2k_cuda_runtime::CudaDeviceBufferRange] {
        match self {
            Self::Grayscale(pending) => pending.ranges(),
            Self::Color(pending) => pending.ranges(),
        }
    }

    fn is_complete(&self) -> Result<bool, Error> {
        match self {
            Self::Grayscale(pending) => pending.is_complete(),
            Self::Color(pending) => pending.is_complete(),
        }
    }

    fn finish(
        self,
    ) -> Result<
        (
            Vec<j2k_cuda_runtime::CudaDeviceBufferRange>,
            crate::CudaHtj2kProfileReport,
        ),
        Error,
    > {
        match self {
            Self::Grayscale(pending) => pending.finish(),
            Self::Color(pending) => pending.finish(),
        }
    }
}

/// Result of a nonblocking attempt to retire an external CUDA batch.
#[cfg(feature = "cuda-runtime")]
#[derive(Debug)]
#[must_use = "pending CUDA work must remain retained until it completes"]
#[expect(
    clippy::large_enum_variant,
    reason = "boxing would allocate on every incomplete retirement poll in the throughput path"
)]
pub enum CudaExternalBatchTryFinish {
    /// GPU work is still in flight; retain this completion owner.
    Pending(SubmittedCudaExternalBatch),
    /// GPU work completed and codec status validation succeeded.
    Complete(CudaExternalBatchGroup),
}

#[cfg(feature = "cuda-runtime")]
impl core::fmt::Debug for SubmittedCudaExternalBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubmittedCudaExternalBatch")
            .field("group", &self.group)
            .field("pending", &true)
            .finish()
    }
}

#[cfg(feature = "cuda-runtime")]
impl SubmittedCudaExternalBatch {
    /// Metadata and destination ranges for the submitted group.
    #[must_use]
    pub fn group(&self) -> &CudaExternalBatchGroup {
        &self.group
    }

    /// Query final-store completion without waiting on the host.
    pub fn is_complete(&self) -> Result<bool, CudaBatchError> {
        let source_indices = self.group().source_indices.clone();
        self.pending
            .is_complete()
            .map_err(|source| CudaBatchError::GroupExecution {
                source_indices,
                source: Box::new(source),
            })
    }

    /// Retire completed work without waiting, or return the still-pending
    /// completion owner unchanged.
    pub fn try_finish(self) -> Result<CudaExternalBatchTryFinish, CudaBatchError> {
        if self.is_complete()? {
            self.wait().map(CudaExternalBatchTryFinish::Complete)
        } else {
            Ok(CudaExternalBatchTryFinish::Pending(self))
        }
    }

    /// Wait for final-store completion and validate entropy-kernel status.
    pub fn wait(self) -> Result<CudaExternalBatchGroup, CudaBatchError> {
        let Self { mut group, pending } = self;
        let source_indices = group.source_indices.clone();
        let (ranges, _report) =
            pending
                .finish()
                .map_err(|source| CudaBatchError::GroupExecution {
                    source_indices,
                    source: Box::new(source),
                })?;
        group.ranges = ranges;
        Ok(group)
    }
}

#[cfg(feature = "cuda-runtime")]
impl CudaExternalBatchGroup {
    /// Shared decoded dimensions, type, color, transform, route, and layout.
    #[must_use]
    pub const fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Original input indices in destination batch order.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Actual decoded rectangle for each image.
    #[must_use]
    pub fn decoded_rects(&self) -> &[Rect] {
        &self.decoded_rects
    }

    /// Non-fatal codec warnings for each image.
    #[must_use]
    pub fn warnings(&self) -> &[Vec<J2kDecodeWarning>] {
        &self.warnings
    }

    /// Validated byte ranges written inside the caller allocation.
    #[must_use]
    pub fn ranges(&self) -> &[j2k_cuda_runtime::CudaDeviceBufferRange] {
        &self.ranges
    }
}
