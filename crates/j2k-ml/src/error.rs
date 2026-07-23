// SPDX-License-Identifier: MIT OR Apache-2.0

/// Failure while decoding or staging a codec batch into Burn.
#[derive(Debug, thiserror::Error)]
pub enum BurnDecodeError {
    /// The codec could not allocate or schedule the requested batch.
    #[error("JPEG 2000 batch infrastructure failed: {0}")]
    Infrastructure(#[from] j2k::BatchInfrastructureError),
    /// The selected Burn backend cannot represent the codec's exact integer type.
    #[error("Burn backend does not support exact codec dtype {dtype:?}")]
    UnsupportedDType {
        /// Required Burn storage dtype.
        dtype: burn_core::tensor::DType,
    },
    /// Codec group metadata and the returned native sample owner disagreed.
    #[error("codec batch sample owner did not match its declared sample type")]
    SampleTypeMismatch,
    /// Tensor shape arithmetic overflowed the host index type.
    #[error("Burn tensor shape overflow")]
    SizeOverflow,
    /// A newer codec contract cannot be represented by this adapter version.
    #[error("unsupported codec batch layout or sample type")]
    UnsupportedCodecContract,
    /// Completed codec output did not contain the exact dense byte count.
    #[error("staged codec output has {actual} bytes; expected {expected}")]
    StagingSizeMismatch {
        /// Exact byte count implied by the group metadata.
        expected: usize,
        /// Byte count returned by the accelerator codec.
        actual: usize,
    },
    /// CUDA rejected or could not complete one homogeneous codec group.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] j2k_cuda::CudaBatchError),
    /// A completed CUDA allocation could not be copied to host staging.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaTransfer(#[from] j2k_cuda_runtime::CudaError),
    /// One CUDA group failed after other groups remained usable.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    CudaCodec(#[from] j2k_cuda::Error),
    /// Metal rejected or could not complete one homogeneous codec group.
    #[cfg(feature = "metal")]
    #[error(transparent)]
    Metal(#[from] j2k_metal::Error),
    /// A framework allocation, readback, or upload boundary failed.
    #[error("{backend} tensor transfer failed: {message}")]
    AcceleratorInterop {
        /// Accelerator runtime at the failing boundary.
        backend: &'static str,
        /// Actionable transfer, bounds, or platform detail.
        message: String,
    },
}
