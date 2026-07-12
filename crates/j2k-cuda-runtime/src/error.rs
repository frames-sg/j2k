use crate::driver::CuResult;

/// Error returned by CUDA driver and J2K CUDA kernel helpers.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CudaError {
    /// CUDA driver library or device is unavailable.
    #[error("CUDA driver is unavailable: {message}")]
    Unavailable {
        /// Human-readable availability failure.
        message: String,
    },
    /// CUDA Driver API call failed.
    #[error("CUDA driver call {operation} failed with CUresult {code}{name}")]
    Driver {
        /// Driver operation name.
        operation: &'static str,
        /// Raw CUDA result code.
        code: CuResult,
        /// CUDA error name, when available.
        name: String,
    },
    /// Host output buffer is too small for a device download.
    #[error("CUDA copy output buffer too small: required {required}, have {have}")]
    OutputTooSmall {
        /// Required byte count.
        required: usize,
        /// Provided byte count.
        have: usize,
    },
    /// Byte length cannot be represented by the kernel ABI.
    #[error("CUDA byte length is too large for kernel launch: {len}")]
    LengthTooLarge {
        /// Byte length.
        len: usize,
    },
    /// A host-side allocation needed by a CUDA operation could not be reserved.
    #[error("CUDA host allocation failed for {bytes} bytes")]
    HostAllocationFailed {
        /// Requested host allocation size in bytes.
        bytes: usize,
    },
    /// Simultaneously live host allocations would exceed the codec policy.
    #[error(
        "CUDA host allocation for {what} is too large: requested {requested} bytes, cap {cap} bytes"
    )]
    HostAllocationTooLarge {
        /// Aggregate requested host byte count, saturated on overflow.
        requested: usize,
        /// Maximum permitted simultaneously live host bytes.
        cap: usize,
        /// Logical operation requiring the allocation.
        what: &'static str,
    },
    /// Device byte length is not aligned to the requested typed view element.
    #[error("CUDA buffer length {bytes} is not a multiple of typed element size {element_size}")]
    LengthNotElementAligned {
        /// Byte length.
        bytes: usize,
        /// Requested element size.
        element_size: usize,
    },
    /// Image dimensions overflowed allocation or launch geometry.
    #[error("CUDA image allocation size overflow for {width}x{height}x{channels}")]
    ImageTooLarge {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
        /// Channel count.
        channels: usize,
    },
    /// Internal runtime state lock was poisoned.
    #[error("CUDA runtime state lock is poisoned: {message}")]
    StatePoisoned {
        /// Poison error message.
        message: String,
    },
    /// A CUDA operation failed and the follow-up completion check also failed.
    #[error(
        "CUDA operation failed ({primary}); context completion could not be established ({completion})"
    )]
    CompletionFailed {
        /// Error returned by the original CUDA operation.
        primary: Box<CudaError>,
        /// Error returned while establishing context-wide completion.
        completion: Box<CudaError>,
    },
    /// A CUDA operation failed and releasing its retained resources also failed.
    #[error(
        "CUDA operation failed ({primary}); retained resource release also failed ({release})"
    )]
    ResourceReleaseFailed {
        /// Error returned by the original CUDA operation.
        primary: Box<CudaError>,
        /// Error returned while releasing retained resources.
        release: Box<CudaError>,
    },
    /// A J2K CUDA kernel reported a validated runtime failure.
    #[error("CUDA kernel {kernel} reported status {code} detail {detail}")]
    KernelStatus {
        /// Kernel entry point or logical stage name.
        kernel: &'static str,
        /// Kernel-defined status code.
        code: u32,
        /// Kernel-defined detail code.
        detail: u32,
    },
    /// Caller supplied arguments that cannot be represented by this runtime API.
    #[error("CUDA invalid argument: {message}")]
    InvalidArgument {
        /// Human-readable validation failure.
        message: String,
    },
    /// Validated host-side planning state contradicted materialized launch data.
    #[error("CUDA internal invariant failed: {what}")]
    InternalInvariant {
        /// Stable description of the failed invariant.
        what: &'static str,
    },
}

impl CudaError {
    /// True when the error means the CUDA driver or device is unavailable.
    pub fn is_unavailable(&self) -> bool {
        match self {
            Self::Unavailable { .. } => true,
            Self::CompletionFailed {
                primary,
                completion,
            } => primary.is_unavailable() && completion.is_unavailable(),
            _ => false,
        }
    }
}

pub(crate) fn select_uncertain_completion_error(
    primary_error: CudaError,
    completion_error: Option<CudaError>,
) -> CudaError {
    if let Some(completion_error) = completion_error {
        return CudaError::CompletionFailed {
            primary: Box::new(primary_error),
            completion: Box::new(completion_error),
        };
    }
    if matches!(
        &primary_error,
        CudaError::Driver { .. }
            | CudaError::StatePoisoned { .. }
            | CudaError::CompletionFailed { .. }
            | CudaError::ResourceReleaseFailed { .. }
    ) {
        return primary_error;
    }
    CudaError::StatePoisoned {
        message: format!(
            "CUDA context became poisoned while the preceding operation failed: {primary_error}"
        ),
    }
}

pub(crate) fn select_resource_release_error(
    primary_error: CudaError,
    release_error: CudaError,
) -> CudaError {
    CudaError::ResourceReleaseFailed {
        primary: Box::new(primary_error),
        release: Box::new(release_error),
    }
}

#[cfg(test)]
mod tests {
    use super::{select_resource_release_error, CudaError};

    fn unavailable(message: &str) -> CudaError {
        CudaError::Unavailable {
            message: message.to_string(),
        }
    }

    fn driver(operation: &'static str) -> CudaError {
        CudaError::Driver {
            operation,
            code: 1,
            name: "CUDA_ERROR_TEST".to_string(),
        }
    }

    #[test]
    fn mixed_completion_failure_is_not_fallback_eligible_unavailability() {
        let mixed = CudaError::CompletionFailed {
            primary: Box::new(unavailable("driver missing")),
            completion: Box::new(driver("cuCtxSynchronize")),
        };
        assert!(!mixed.is_unavailable());

        let unavailable_only = CudaError::CompletionFailed {
            primary: Box::new(unavailable("driver missing")),
            completion: Box::new(unavailable("completion unavailable")),
        };
        assert!(unavailable_only.is_unavailable());
    }

    #[test]
    fn resource_release_failure_preserves_both_diagnostics_and_blocks_fallback() {
        let error = select_resource_release_error(
            unavailable("primary unavailable"),
            driver("release pool hold"),
        );
        assert!(!error.is_unavailable());
        assert!(matches!(error, CudaError::ResourceReleaseFailed { .. }));
        let rendered = error.to_string();
        assert!(rendered.contains("primary unavailable"));
        assert!(rendered.contains("release pool hold"));
    }
}
