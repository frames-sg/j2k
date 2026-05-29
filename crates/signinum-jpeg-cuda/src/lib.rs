// SPDX-License-Identifier: Apache-2.0

//! CUDA-facing device-output adapter for `signinum-jpeg`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter. CPU requests return host-backed surfaces. Scalar auto
//! requests stay on CPU, while full-tile batch auto requests may use nvJPEG
//! when the CUDA runtime and library are available. Explicit CUDA requests
//! return CUDA-backed surfaces or a clear unavailable error.

#![deny(missing_docs)]
#![warn(unreachable_pub)]

mod codec;
mod decoder;
mod error;
mod profile;
mod runtime;
mod session;
mod surface;

pub use codec::Codec;
pub use decoder::Decoder;
pub use error::Error;
pub use session::CudaSession;
pub use signinum_jpeg::{DecoderContext, ScratchPool};
pub use surface::{CudaSurface, CudaSurfaceStats, Surface};
