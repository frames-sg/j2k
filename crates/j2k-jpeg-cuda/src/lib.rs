// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA-facing device-output adapter for `j2k-jpeg`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter. CPU requests return host-backed surfaces. Scalar auto
//! requests stay on CPU. Explicit CUDA requests use J2K-owned CUDA JPEG
//! decode kernels when the runtime can handle the image, and otherwise return
//! a clear unsupported or unavailable error.

#![warn(unreachable_pub)]

mod allocation;
mod batch;
mod codec;
mod decoder;
mod encode;
mod error;
mod owned_decode;
mod profile;
mod runtime;
mod session;
mod surface;

#[doc(hidden)]
pub use batch::{CudaJpegBatch, CudaJpegBatchIntoIter};
pub use codec::Codec;
#[cfg(feature = "cuda-runtime")]
pub use codec::CudaJpegDecodeOutputTile;
pub use decoder::Decoder;
pub use encode::{
    encode_jpeg_baseline_batch_from_cuda_buffers, encode_jpeg_baseline_from_cuda_buffer,
    JpegBaselineCudaEncodeTile,
};
pub use error::Error;
pub use j2k_jpeg::{DecoderContext, ScratchPool};
#[cfg(feature = "cuda-runtime")]
pub use owned_decode::CudaJpegChunkedEntropyReport;
#[cfg(feature = "cuda-runtime")]
pub use session::CudaJpegHostMemoryDiagnostics;
pub use session::CudaSession;
pub use surface::{CudaJpegDecodePath, CudaSurface, CudaSurfaceStats, Surface};
