// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA-facing device-output adapter for `j2k`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter. CPU requests return host-backed surfaces; auto requests can
//! return CUDA-resident surfaces for promoted workloads. Explicit CUDA decode
//! supports classic JPEG 2000 and HTJ2K within the adapter's image and code-block
//! limits, with reversible 5/3 or irreversible 9/7 transforms. Encode from CUDA
//! buffers requires HTJ2K block coding and external validation. Unsupported
//! explicit CUDA requests return an error. CPU-decode-then-upload is exposed
//! through explicit CPU-staged APIs.

#![deny(missing_docs)]
#![warn(unreachable_pub)]
mod allocation;
mod batch;
mod codec;
mod decoder;
#[cfg(any(feature = "cuda-runtime", test))]
mod direct_plan;
mod encode;
mod error;
mod generated;
mod profile;
mod routing;
mod runtime;
mod session;
mod surface;

pub use batch::{
    CudaBatchDecodeResult, CudaBatchDecoder, CudaBatchError, CudaBatchGroup, CudaBatchGroupError,
};
#[cfg(feature = "cuda-runtime")]
pub use batch::{
    CudaExternalBatchGroup, CudaExternalBatchTryFinish, CudaResidentBatchBuffer,
    SubmittedCudaExternalBatch, SubmittedCudaResidentBatch,
};
pub use codec::Codec;
pub use decoder::J2kDecoder;
#[cfg(feature = "cuda-runtime")]
pub(crate) use direct_plan::{
    CudaHtj2kBandId, CudaHtj2kCodeBlock, CudaHtj2kDecodePlan, CudaHtj2kIdwtStep, CudaHtj2kRect,
    CudaHtj2kStoreStep, CudaHtj2kTransform,
};
pub use encode::{
    encode_j2k_lossless_with_cuda, encode_j2k_lossless_with_cuda_and_profile,
    CudaEncodeFallbackReason, CudaEncodeStageAccelerator, CudaEncodeStageTimings,
    CudaLosslessEncodeResult, CudaLosslessEncoder,
};
#[cfg(feature = "cuda-runtime")]
pub use encode::{
    encode_lossless_from_cuda_buffer, encode_lossless_from_cuda_buffer_to_cuda_buffer,
    encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report,
    encode_lossless_from_cuda_buffer_with_report, encode_lossless_from_cuda_buffers,
    encode_lossless_from_cuda_buffers_to_cuda_buffers,
    encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report,
    encode_lossless_from_cuda_buffers_with_report, submit_lossless_from_cuda_buffer,
    submit_lossless_from_cuda_buffers, CudaEncodedJ2k, CudaEncodedJ2kMetadata,
    CudaLosslessBufferEncodeOutcome, CudaLosslessEncodeOutcome, CudaLosslessEncodeResidency,
    CudaLosslessEncodeTile, CudaResidentCodestreamBuffer, SubmittedJ2kLosslessCudaEncode,
    SubmittedJ2kLosslessCudaEncodeBatch,
};
pub use error::{Error, NativeBackendError};
pub use j2k::{J2kContext, J2kScratchPool};
pub use profile::{
    CudaHtj2kDecodeProfileDetail, CudaHtj2kEncodeProfileReport, CudaHtj2kProfileReport,
};
pub use session::CudaSession;
#[cfg(feature = "cuda-runtime")]
pub use session::{CudaDecodePoolDiagnostics, CudaDecodePoolSnapshot, CudaSessionDiagnostics};
pub use surface::{CudaSurface, CudaSurfaceStats, Surface, SurfaceResidency};
