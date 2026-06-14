// SPDX-License-Identifier: Apache-2.0

//! CUDA-facing device-output adapter for `signinum-jpeg`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter. CPU requests return host-backed surfaces. Scalar auto
//! requests stay on CPU. Explicit CUDA requests use Signinum-owned CUDA JPEG
//! decode kernels when the runtime can handle the image, and otherwise return
//! a clear unsupported or unavailable error.

#![warn(unreachable_pub)]

mod codec;
mod decoder;
mod error;
mod owned_decode;
mod runtime;
mod session;
mod surface;

pub use codec::Codec;
pub use decoder::Decoder;
pub use error::Error;
pub use session::CudaSession;
pub use signinum_jpeg::{DecoderContext, ScratchPool};
pub use surface::{CudaJpegDecodePath, CudaSurface, CudaSurfaceStats, Surface};
