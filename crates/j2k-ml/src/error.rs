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
    /// CUDA rejected or could not complete one homogeneous codec group.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] j2k_cuda::CudaBatchError),
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
