// SPDX-License-Identifier: Apache-2.0

//! CUDA-facing device-output adapter for `signinum-j2k`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter. CPU and auto requests return host-backed surfaces. Strict
//! CUDA requests are reserved for CUDA-resident HTJ2K codestream decode; the
//! CPU-decode-then-upload path is exposed through explicit CPU-staged APIs.

#![deny(missing_docs)]
#![warn(unreachable_pub)]

mod codec;
mod decoder;
mod direct_plan;
mod encode;
mod error;
mod profile;
mod runtime;
mod session;
mod surface;

pub use codec::Codec;
pub use decoder::J2kDecoder;
pub use direct_plan::{
    CudaHtj2kCodeBlock, CudaHtj2kDecodePlan, CudaHtj2kIdwtStep, CudaHtj2kRect, CudaHtj2kStoreStep,
    CudaHtj2kSubband, CudaHtj2kTransform,
};
#[cfg(feature = "cuda-runtime")]
#[doc(hidden)]
pub use encode::cuda_dwt53_output_to_j2k_for_test;
pub use encode::{
    encode_j2k_lossless_with_cuda, encode_j2k_lossless_with_cuda_and_profile,
    CudaEncodeStageAccelerator, CudaEncodeStageTimings,
};
pub use error::Error;
pub use profile::{
    CudaHtj2kDecodeProfileDetail, CudaHtj2kEncodeProfileReport, CudaHtj2kProfileReport,
};
pub use session::CudaSession;
pub use signinum_j2k::{J2kContext, J2kScratchPool};
pub use surface::{CudaSurface, CudaSurfaceStats, Surface, SurfaceResidency};
