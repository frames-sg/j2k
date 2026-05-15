// SPDX-License-Identifier: Apache-2.0

//! CUDA-facing device-output adapter for `signinum-j2k`.
//!
//! This crate intentionally exposes the same backend-selection surface as the
//! Metal adapter. CPU and auto requests return host-backed surfaces, while
//! explicit CUDA requests upload decoded output into CUDA device memory when
//! the `cuda-runtime` feature and a CUDA driver are available.

#![warn(unreachable_pub)]

mod codec;
mod decoder;
mod encode;
mod error;
mod profile;
mod runtime;
mod session;
mod surface;

pub use codec::Codec;
pub use decoder::J2kDecoder;
pub use encode::CudaEncodeStageAccelerator;
pub use error::Error;
pub use session::CudaSession;
pub use signinum_j2k::{J2kContext, J2kScratchPool};
pub use surface::{CudaSurface, CudaSurfaceStats, Surface};
