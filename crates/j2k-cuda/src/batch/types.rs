// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA batch result and resident-output contracts.

#[cfg(feature = "cuda-runtime")]
use super::Arc;
use super::{
    BatchGroupInfo, BatchInfrastructureError, Error, IndexedBatchError, J2kDecodeWarning,
    PreparedBatchGroup, Rect, Surface,
};

/// Failure while preparing or executing an owned CUDA batch.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CudaBatchError {
    /// Shared codec preparation or host-side batch infrastructure failed.
    #[error(transparent)]
    Infrastructure(#[from] BatchInfrastructureError),
    /// CUDA execution failed for one homogeneous group.
    ///
    /// No output surfaces from the affected group are exposed.
    #[error("CUDA batch group containing source indices {source_indices:?} failed: {source}")]
    GroupExecution {
        /// Every original input index whose dense group output was discarded.
        source_indices: Vec<usize>,
        /// Strict CUDA adapter or runtime failure.
        #[source]
        source: Box<Error>,
    },
}

impl CudaBatchError {
    pub(super) fn group(group: &PreparedBatchGroup, source: Error) -> Self {
        Self::GroupExecution {
            source_indices: group.source_indices().to_vec(),
            source: Box::new(source),
        }
    }

    /// Whether submitted CUDA work may still reference an external
    /// destination because completion could not be established.
    #[cfg(feature = "cuda-runtime")]
    #[doc(hidden)]
    pub fn completion_is_uncertain(&self) -> bool {
        match self {
            Self::GroupExecution { source, .. } => source.completion_is_uncertain(),
            Self::Infrastructure(_) => false,
        }
    }

    /// Whether this failure prevents the current persistent batch operation
    /// from safely continuing with later groups.
    #[doc(hidden)]
    #[must_use]
    pub fn session_is_unusable(&self) -> bool {
        match self {
            Self::Infrastructure(_) => true,
            Self::GroupExecution { source, .. } => source.session_is_unusable(),
        }
    }
}

#[cfg(test)]
mod classification_tests {
    use j2k_core::BatchInfrastructureError;

    use super::CudaBatchError;
    use crate::Error;

    fn group_error(source: Error) -> CudaBatchError {
        CudaBatchError::GroupExecution {
            source_indices: vec![3],
            source: Box::new(source),
        }
    }

    #[test]
    fn cuda_batch_session_classification_is_owned_by_codec_errors() {
        assert!(
            CudaBatchError::Infrastructure(BatchInfrastructureError::EmptyBatchPlan)
                .session_is_unusable()
        );
        assert!(group_error(Error::CudaUnavailable).session_is_unusable());
        assert!(!group_error(Error::UnsupportedCudaRequest {
            reason: "test contract rejection",
        })
        .session_is_unusable());
    }
}

/// Failure while executing one homogeneous CUDA group.
///
/// No partially written dense output from the affected group is exposed.
/// Other prepared groups may still succeed when the retained CUDA session
/// remains usable.
#[derive(Debug, thiserror::Error)]
#[error("CUDA batch group containing source indices {source_indices:?} failed: {source}")]
pub struct CudaBatchGroupError {
    source_indices: Vec<usize>,
    #[source]
    source: Box<Error>,
}

impl CudaBatchGroupError {
    #[cfg(feature = "cuda-runtime")]
    pub(super) fn new(group: &PreparedBatchGroup, source: Error) -> Self {
        Self {
            source_indices: group.source_indices().to_vec(),
            source: Box::new(source),
        }
    }

    #[cfg(feature = "cuda-runtime")]
    pub(super) fn from_parts(source_indices: Vec<usize>, source: Error) -> Self {
        Self {
            source_indices,
            source: Box::new(source),
        }
    }

    /// Original input indices whose dense group output was discarded.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Strict CUDA adapter or runtime failure for this group.
    #[must_use]
    pub fn source(&self) -> &Error {
        &self.source
    }

    /// Consume the group failure into affected indices and its source.
    #[must_use]
    pub fn into_parts(self) -> (Vec<usize>, Error) {
        (self.source_indices, *self.source)
    }
}

/// One successful homogeneous CUDA-resident output group.
#[derive(Debug)]
pub struct CudaBatchGroup {
    pub(super) info: BatchGroupInfo,
    pub(super) source_indices: Vec<usize>,
    pub(super) decoded_rects: Vec<Rect>,
    pub(super) warnings: Vec<Vec<J2kDecodeWarning>>,
    pub(super) surfaces: Vec<Surface>,
    #[cfg(feature = "cuda-runtime")]
    pub(super) dense_output: CudaResidentBatchBuffer,
}

/// One codec-owned dense CUDA allocation containing a homogeneous batch.
///
/// This owner is the canonical resident representation for every exact-native
/// grayscale or color group. Grayscale and NHWC RGB/RGBA groups also expose
/// ordinary [`Surface`] views over the same allocation for compatibility.
#[cfg(feature = "cuda-runtime")]
#[derive(Debug)]
pub struct CudaResidentBatchBuffer {
    pub(super) buffer: Arc<j2k_cuda_runtime::CudaDeviceBuffer>,
    pub(super) ranges: Vec<j2k_cuda_runtime::CudaDeviceBufferRange>,
}

#[cfg(feature = "cuda-runtime")]
impl CudaResidentBatchBuffer {
    /// Codec-owned CUDA allocation containing every image range.
    #[must_use]
    pub fn buffer(&self) -> &j2k_cuda_runtime::CudaDeviceBuffer {
        &self.buffer
    }

    /// Tightly concatenated per-image byte ranges in dense batch order.
    #[must_use]
    pub fn ranges(&self) -> &[j2k_cuda_runtime::CudaDeviceBufferRange] {
        &self.ranges
    }
}

impl CudaBatchGroup {
    /// Shared decoded dimensions, type, color, transform, route, and layout.
    #[must_use]
    pub const fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Original input indices in dense batch order.
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

    /// CUDA-resident image views in dense batch order.
    ///
    /// Grayscale groups expose one view per image. NHWC RGB/RGBA groups also
    /// expose compatible interleaved views over their dense group allocation.
    /// No decoded host staging is used.
    #[must_use]
    pub fn surfaces(&self) -> &[Surface] {
        &self.surfaces
    }

    /// Dense codec-owned allocation for exact-native grayscale or color output.
    ///
    /// NCHW color groups must be consumed through this owner because a
    /// [`Surface`] describes interleaved pixels. Grayscale and NHWC RGB/RGBA
    /// groups return both this owner and compatible surface views.
    #[cfg(feature = "cuda-runtime")]
    #[must_use]
    pub const fn dense_output(&self) -> &CudaResidentBatchBuffer {
        &self.dense_output
    }

    /// Consume the group into metadata and CUDA-resident views.
    #[must_use]
    #[expect(
        clippy::type_complexity,
        reason = "the tuple mirrors the group's five explicitly documented owners"
    )]
    #[cfg(not(feature = "cuda-runtime"))]
    pub fn into_parts(
        self,
    ) -> (
        BatchGroupInfo,
        Vec<usize>,
        Vec<Rect>,
        Vec<Vec<J2kDecodeWarning>>,
        Vec<Surface>,
    ) {
        (
            self.info,
            self.source_indices,
            self.decoded_rects,
            self.warnings,
            self.surfaces,
        )
    }

    /// Consume the group into metadata, compatible surfaces, and its required
    /// dense exact-native allocation.
    #[cfg(feature = "cuda-runtime")]
    #[must_use]
    #[expect(
        clippy::type_complexity,
        reason = "the tuple mirrors the group's explicitly documented owners"
    )]
    pub fn into_parts(
        self,
    ) -> (
        BatchGroupInfo,
        Vec<usize>,
        Vec<Rect>,
        Vec<Vec<J2kDecodeWarning>>,
        Vec<Surface>,
        CudaResidentBatchBuffer,
    ) {
        (
            self.info,
            self.source_indices,
            self.decoded_rects,
            self.warnings,
            self.surfaces,
            self.dense_output,
        )
    }
}

/// CUDA batch successes plus indexed codec preflight failures.
#[derive(Debug)]
pub struct CudaBatchDecodeResult {
    pub(super) groups: Vec<CudaBatchGroup>,
    pub(super) errors: Vec<IndexedBatchError>,
    pub(super) group_errors: Vec<CudaBatchGroupError>,
}

impl CudaBatchDecodeResult {
    /// Successfully decoded homogeneous device groups.
    #[must_use]
    pub fn groups(&self) -> &[CudaBatchGroup] {
        &self.groups
    }

    /// Per-input parsing and representability failures from shared preflight.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }

    /// Homogeneous groups that failed during recoverable CUDA execution.
    #[must_use]
    pub fn group_errors(&self) -> &[CudaBatchGroupError] {
        &self.group_errors
    }

    /// Consume this result into successful groups, indexed preflight errors,
    /// and homogeneous execution failures.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<CudaBatchGroup>,
        Vec<IndexedBatchError>,
        Vec<CudaBatchGroupError>,
    ) {
        (self.groups, self.errors, self.group_errors)
    }
}
