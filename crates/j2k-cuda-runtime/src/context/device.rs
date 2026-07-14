// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CudaError;

use super::{creation::create_context, CudaContext};

impl CudaContext {
    /// Create a context for the system default CUDA device.
    pub fn system_default() -> Result<Self, CudaError> {
        Self::create_owned(0)
    }

    /// Retain the CUDA primary context for a selected device ordinal.
    ///
    /// This is the interoperability constructor for runtimes such as `CubeCL`
    /// that also use CUDA primary contexts. Dropping the final clone releases
    /// the retain; it never destroys the primary context directly.
    pub fn retain_primary(device_ordinal: usize) -> Result<Self, CudaError> {
        create_context(device_ordinal, true)
    }

    fn create_owned(device_ordinal: usize) -> Result<Self, CudaError> {
        create_context(device_ordinal, false)
    }
}
