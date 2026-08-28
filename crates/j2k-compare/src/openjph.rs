// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(have_openjph)]
use std::{ffi::c_void, ptr};

/// Whether the optional pinned `OpenJPH` library was linked into this build.
#[must_use]
pub const fn is_available() -> bool {
    cfg!(have_openjph)
}

/// Version of the linked `OpenJPH` library.
#[must_use]
pub fn version() -> &'static str {
    option_env!("J2K_OPENJPH_VERSION").unwrap_or("unavailable")
}

/// Directory containing the linked `OpenJPH` library.
#[must_use]
pub fn library_path() -> &'static str {
    option_env!("J2K_OPENJPH_LIB_DIR").unwrap_or("unavailable")
}

/// Decode an HTJ2K codestream to packed 8-bit grayscale in-process.
pub fn decode_gray(bytes: &[u8], reduce: u8) -> Result<Vec<u8>, String> {
    decode(bytes, reduce, 1)
}

/// Decode an HTJ2K codestream to packed interleaved RGB8 in-process.
pub fn decode_rgb(bytes: &[u8], reduce: u8) -> Result<Vec<u8>, String> {
    decode(bytes, reduce, 3)
}

#[cfg_attr(
    have_openjph,
    expect(
        unsafe_code,
        reason = "OpenJPH decode uses the optional checked C++ shim and frees its output exactly once"
    )
)]
fn decode(bytes: &[u8], reduce: u8, channels: u32) -> Result<Vec<u8>, String> {
    #[cfg(have_openjph)]
    {
        let mut out = ptr::null_mut();
        let mut out_len = 0_usize;
        let mut out_width = 0_u32;
        let mut out_height = 0_u32;
        // SAFETY: the input slice remains live for the call and the output
        // pointers refer to initialized writable locals governed by the shim ABI.
        let ok = unsafe {
            j2k_openjph_decode_u8(
                bytes.as_ptr(),
                bytes.len(),
                reduce,
                channels,
                &raw mut out,
                &raw mut out_len,
                &raw mut out_width,
                &raw mut out_height,
            )
        };
        let output = OpenJphOutput(out);
        if ok == 0 || output.0.is_null() {
            return Err("openjph: decode failed".to_string());
        }
        let expected =
            crate::checked_external_output_len("openjph", out_width, out_height, channels)?;
        if out_len != expected {
            return Err(format!(
                "openjph: unexpected output length {out_len} != {expected}"
            ));
        }
        // SAFETY: the non-null shim allocation has been independently bounded
        // and its length matches the checked dimensions and channel count.
        Ok(unsafe { std::slice::from_raw_parts(output.0, expected) }.to_vec())
    }

    #[cfg(not(have_openjph))]
    {
        let _ = (bytes, reduce, channels);
        Err("openjph: local library not available".to_string())
    }
}

#[cfg(have_openjph)]
struct OpenJphOutput(*mut u8);

#[cfg(have_openjph)]
impl Drop for OpenJphOutput {
    #[expect(
        unsafe_code,
        reason = "the guard frees the OpenJPH shim allocation exactly once"
    )]
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the pointer is the allocation returned by the paired shim.
            unsafe { j2k_openjph_free(self.0.cast()) };
        }
    }
}

#[cfg(have_openjph)]
#[expect(
    unsafe_code,
    reason = "these declarations are the optional OpenJPH C++ shim's complete ABI"
)]
unsafe extern "C" {
    fn j2k_openjph_decode_u8(
        bytes: *const u8,
        len: usize,
        reduce: u8,
        channels: u32,
        out_data: *mut *mut u8,
        out_len: *mut usize,
        out_width: *mut u32,
        out_height: *mut u32,
    ) -> i32;
    fn j2k_openjph_free(ptr: *mut c_void);
}

#[cfg(test)]
mod tests {
    #[test]
    fn decodes_pinned_gray8_fixture_in_process() {
        let fixture = j2k_test_support::openjph_batch_fixtures()
            .iter()
            .find(|fixture| fixture.name == "openjph-gray-u8-53-raw")
            .expect("pinned gray8 OpenJPH fixture");

        if !super::is_available() {
            return;
        }
        let decoded =
            super::decode_gray(fixture.encoded, 0).expect("decode through pinned OpenJPH library");
        assert_eq!(decoded, fixture.oracle);
        assert_eq!(super::version(), "0.31.0");
    }
}
