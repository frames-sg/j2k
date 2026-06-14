use crate::driver::CuResult;

/// Error returned by CUDA driver and Signinum CUDA kernel helpers.
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
    /// A Signinum CUDA kernel reported a validated runtime failure.
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
}

impl CudaError {
    /// True when the error means the CUDA driver or device is unavailable.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable { .. })
    }
}
