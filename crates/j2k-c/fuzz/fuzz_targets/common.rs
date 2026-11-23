// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared harness helpers. Keeping the handle guard and diagnostic checks in
//! one place makes both targets exercise the same C ABI contract.

use std::ffi::CStr;
use std::ptr;

use j2k_c::{
    j2k_decoder_close, j2k_decoder_open, J2kCDecoder, J2kCError, J2kCOpenOptions,
    J2K_INTERNAL_ERROR, J2K_OK, J2K_PANIC,
};

/// Keep fuzz allocations small enough that a malformed header cannot turn a
/// run into an OOM test. Valid fixtures are far below this bound.
pub const MAX_OUTPUT_BYTES: u64 = 4 * 1024 * 1024;

/// Owns a decoder handle and closes it exactly once.
pub struct Decoder {
    handle: *mut J2kCDecoder,
}

impl Decoder {
    /// Open `input` with explicit limits. Returns `None` for any rejected
    /// input; all rejection statuses are checked for ABI consistency first.
    pub fn open(
        input: &[u8],
        workers: u32,
        max_input_bytes: u64,
        max_output_bytes: u64,
    ) -> Option<Self> {
        let mut handle = ptr::null_mut();
        let mut error = J2kCError::default();
        let options = J2kCOpenOptions {
            max_input_bytes,
            max_output_bytes,
            workers,
            ..J2kCOpenOptions::default()
        };
        // SAFETY: `input` is readable for `input.len()` bytes; `options`, `handle`,
        // and `error` are valid disjoint locals.
        let status = unsafe {
            j2k_decoder_open(
                input.as_ptr(),
                input.len(),
                &options,
                &mut handle,
                &mut error,
            )
        };
        check_error(status, &error);
        if status == J2K_OK && !handle.is_null() {
            Some(Self { handle })
        } else {
            None
        }
    }

    /// Open with the codec's default input limit.
    pub fn open_default(input: &[u8], workers: u32, max_output_bytes: u64) -> Option<Self> {
        Self::open(input, workers, 0, max_output_bytes)
    }

    pub fn as_ptr(&self) -> *mut J2kCDecoder {
        self.handle
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: `handle` is either null or a live handle owned by this guard.
        unsafe { j2k_decoder_close(&mut self.handle, ptr::null_mut()) };
    }
}

/// The ABI promises a NUL-terminated UTF-8 message.
pub fn message(error: &J2kCError) -> &CStr {
    // SAFETY: every diagnostic written by the library is NUL-terminated.
    unsafe { CStr::from_ptr(error.message.as_ptr()) }
}

/// Reject statuses that indicate a panic or an internal invariant failure.
/// Ordinary invalid-input statuses remain expected fuzz outcomes.
pub fn check_error(status: u32, error: &J2kCError) {
    let text = message(error).to_string_lossy();
    assert_ne!(
        status, J2K_PANIC,
        "panic crossed the C ABI boundary: {text}"
    );
    assert_ne!(
        status, J2K_INTERNAL_ERROR,
        "internal error for untrusted input: {text}"
    );
    assert_eq!(
        error.code, status,
        "diagnostic code disagrees with return status: {text}"
    );
    assert!(
        message(error).to_str().is_ok(),
        "diagnostic message is not UTF-8"
    );
}
