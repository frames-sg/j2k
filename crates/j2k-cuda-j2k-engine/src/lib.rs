// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA JPEG 2000 and HTJ2K codec-engine boundary.
//!
//! The borrowed engine preserves the stable low-level CUDA context identity
//! while C1 moves J2K, HTJ2K, and ML domain ownership out of the runtime.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(unreachable_pub)]

#[macro_use]
mod macros;
mod allocation;
mod build_flags;
mod bytes;
mod classic_decode;
mod context;
mod driver;
mod error;
mod execution;
mod htj2k_decode;
mod htj2k_encode;
mod htj2k_packetize;
mod j2k_decode;
mod j2k_encode;
mod kernels;
mod memory;
mod ml;
#[cfg(test)]
mod tests;

pub use classic_decode::{
    CudaClassicCodeBlockJob, CudaClassicDecodeStageTimings, CudaClassicDecodeTableResources,
    CudaClassicDecodeTarget, CudaClassicSegment, CudaClassicStatus, CudaQueuedClassicDecode,
};
pub use htj2k_decode::{
    htj2k_cleanup_multi_descriptor_bytes, CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob,
    CudaHtj2kDecodeOutput, CudaHtj2kDecodeResources, CudaHtj2kDecodeStageTimings,
    CudaHtj2kDecodeTableResources, CudaHtj2kDecodeTables, CudaHtj2kDequantizeTarget,
    CudaHtj2kStatus, CudaPooledHtj2kDecodeOutput, CudaQueuedHtj2kCleanup,
    CudaQueuedHtj2kCleanupGroup,
};
pub use htj2k_encode::{
    CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kCompactEncodedCodeBlocks,
    CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob, CudaHtj2kEncodeResidentTarget,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeStatus,
    CudaHtj2kEncodeTables, CudaHtj2kEncodedCodeBlock, CudaHtj2kEncodedCodeBlocks,
};
pub use htj2k_packetize::{
    CudaHtj2kPacketizationBlock, CudaHtj2kPacketizationPacket, CudaHtj2kPacketizationStageTimings,
    CudaHtj2kPacketizationStatus, CudaHtj2kPacketizationSubband,
    CudaHtj2kPacketizationSubbandTagState, CudaHtj2kPacketizationTagNodeState,
    CudaHtj2kPacketizedTile,
};
pub use j2k_decode::{
    CudaJ2kIdwtBatchStageProfile, CudaJ2kIdwtJob, CudaJ2kIdwtNormalization, CudaJ2kIdwtTarget,
    CudaJ2kInverseMctJob, CudaJ2kRect, CudaJ2kStoreGray16Job, CudaJ2kStoreGray16Target,
    CudaJ2kStoreGray8Job, CudaJ2kStoreGray8Target, CudaJ2kStoreGrayI16Target, CudaJ2kStoreRgb16Job,
    CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctJob,
    CudaJ2kStoreRgb8MctTarget, CudaJ2kStoreRgbNativeJob, CudaJ2kStoreRgbNativeTarget,
    CudaJ2kStoreRgbaNativeJob, CudaJ2kStoreRgbaNativeTarget, CudaJ2kStridedInterleavedPixels,
    CudaQueuedJ2kStoreBatch,
};
pub use j2k_encode::{
    CudaDwt53LevelShape, CudaDwt53Output, CudaDwt97Output, CudaJ2kDeinterleavedComponents,
    CudaJ2kQuantizeJob, CudaJ2kQuantizeSubbandRegionJob, CudaJ2kQuantizedSubband,
    CudaJ2kResidentComponents, CudaJ2kResidentQuantizedSubband, CudaResidentDwt53Output,
    CudaResidentDwt97Output,
};
pub use ml::{CudaJ2kMlKernelConfig, CudaJ2kMlLayout, CudaJ2kMlNormalization, CudaJ2kMlSample};

pub(crate) use j2k_cuda_runtime::CudaContext;

/// Borrowed J2K/HTJ2K codec operations over one low-level CUDA context.
#[derive(Clone)]
pub struct J2kCudaEngine<'context> {
    pub(crate) context: &'context CudaContext,
}

impl<'context> J2kCudaEngine<'context> {
    /// Bind codec operations to `context` without changing its ownership.
    #[must_use]
    pub const fn new(context: &'context CudaContext) -> Self {
        Self { context }
    }

    /// Return the borrowed low-level context.
    #[must_use]
    pub const fn context(&self) -> &'context CudaContext {
        self.context
    }
}

#[cfg(test)]
pub(crate) use bytes::{f32_slice_as_bytes, f32_slice_as_bytes_mut};
#[cfg(test)]
pub(crate) use j2k_cuda_runtime::{
    CudaDeviceBuffer, CudaDeviceBufferRange, CudaError, CudaExternalDeviceBufferViewMut,
};

#[cfg(test)]
mod engine_tests {
    use super::J2kCudaEngine;

    #[test]
    fn engine_constructor_preserves_the_low_level_context_type() {
        assert_eq!(
            std::mem::size_of::<J2kCudaEngine<'static>>(),
            std::mem::size_of::<&j2k_cuda_runtime::CudaContext>()
        );
    }
}
