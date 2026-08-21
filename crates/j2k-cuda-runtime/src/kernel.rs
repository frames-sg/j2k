// SPDX-License-Identifier: MIT OR Apache-2.0

//! Low-level compiled-kernel contracts shared by codec engines.

use std::hash::{Hash, Hasher};

use crate::CudaError;

/// Static PTX image and entry point requested by a codec engine.
///
/// The module identifier is diagnostic only; cache identity also includes the
/// static PTX allocation and entry-point bytes so unrelated engines cannot
/// collide by choosing the same label.
#[derive(Clone, Copy, Debug)]
pub struct CudaKernelSpec {
    module_id: &'static str,
    ptx: &'static [u8],
    entrypoint: &'static [u8],
}

impl CudaKernelSpec {
    /// Validate a static PTX image and NUL-terminated entry point.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidArgument`] for an empty module identifier,
    /// a PTX image without a trailing NUL byte, or an empty, unterminated, or
    /// internally NUL-containing entry point.
    pub fn new(
        module_id: &'static str,
        ptx: &'static [u8],
        entrypoint: &'static [u8],
    ) -> Result<Self, CudaError> {
        if module_id.is_empty() {
            return Err(CudaError::InvalidArgument {
                message: "CUDA kernel module identifier must not be empty".to_string(),
            });
        }
        if ptx.last().copied() != Some(0) {
            return Err(CudaError::InvalidArgument {
                message: format!("CUDA PTX image for {module_id} must be NUL-terminated"),
            });
        }
        if entrypoint.len() < 2
            || entrypoint.last().copied() != Some(0)
            || entrypoint[..entrypoint.len() - 1].contains(&0)
        {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "CUDA kernel entry point for {module_id} must be non-empty and NUL-terminated"
                ),
            });
        }
        Ok(Self {
            module_id,
            ptx,
            entrypoint,
        })
    }

    /// Diagnostic module identifier supplied by the engine.
    #[must_use]
    pub const fn module_id(self) -> &'static str {
        self.module_id
    }

    /// NUL-terminated static PTX image.
    #[must_use]
    pub const fn ptx(self) -> &'static [u8] {
        self.ptx
    }

    /// Non-empty NUL-terminated kernel entry point.
    #[must_use]
    pub const fn entrypoint(self) -> &'static [u8] {
        self.entrypoint
    }
}

impl PartialEq for CudaKernelSpec {
    fn eq(&self, other: &Self) -> bool {
        self.module_id == other.module_id
            && std::ptr::eq(self.ptx, other.ptx)
            && self.entrypoint == other.entrypoint
    }
}

impl Eq for CudaKernelSpec {}

impl Hash for CudaKernelSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.module_id.hash(state);
        self.ptx.as_ptr().hash(state);
        self.ptx.len().hash(state);
        self.entrypoint.hash(state);
    }
}
