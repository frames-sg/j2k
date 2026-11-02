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

/// Decode to packed 8-bit grayscale directly into `out`, whose length must
/// equal the decoded image's.
pub fn decode_gray_into(bytes: &[u8], reduce: u8, out: &mut [u8]) -> Result<(), String> {
    decode_into(bytes, reduce, 1, out)
}

/// Decode to packed interleaved RGB8 directly into `out`, whose length must
/// equal the decoded image's.
pub fn decode_rgb_into(bytes: &[u8], reduce: u8, out: &mut [u8]) -> Result<(), String> {
    decode_into(bytes, reduce, 3, out)
}

/// Decode component 0 to native-endian `u16` samples (unscaled values,
/// clamped to `0..=65535`) directly into `out`.
pub fn decode_gray16_into(bytes: &[u8], reduce: u8, out: &mut [u8]) -> Result<(), String> {
    call_shim(bytes, reduce, 1, 2, Some(out)).map(|_| ())
}

#[cfg_attr(
    have_openjph,
    expect(
        unsafe_code,
        reason = "views the shim allocation that the guard frees exactly once"
    )
)]
fn decode(bytes: &[u8], reduce: u8, channels: u32) -> Result<Vec<u8>, String> {
    let (ptr, len) = call_shim(bytes, reduce, channels, 1, None)?;
    #[cfg(have_openjph)]
    {
        let output = OpenJphOutput(ptr);
        // SAFETY: the non-null shim allocation has been independently bounded
        // and its length matches the checked dimensions and channel count.
        Ok(unsafe { std::slice::from_raw_parts(output.0, len) }.to_vec())
    }
    #[cfg(not(have_openjph))]
    {
        let _ = (ptr, len);
        Err("openjph: local library not available".to_string())
    }
}

fn decode_into(bytes: &[u8], reduce: u8, channels: u32, out: &mut [u8]) -> Result<(), String> {
    call_shim(bytes, reduce, channels, 1, Some(out)).map(|_| ())
}

/// Runs the shim. With `dest`, the pixels land in `dest` and the returned
/// pointer is `dest`'s; without it, the returned pointer is a shim allocation
/// the caller must free.
#[cfg_attr(
    have_openjph,
    expect(
        unsafe_code,
        reason = "OpenJPH decode uses the optional checked C++ shim"
    )
)]
fn call_shim(
    bytes: &[u8],
    reduce: u8,
    channels: u32,
    bytes_per_sample: u32,
    dest: Option<&mut [u8]>,
) -> Result<(*mut u8, usize), String> {
    #[cfg(have_openjph)]
    {
        let (dest_ptr, dest_len) = dest.map_or((ptr::null_mut(), 0), |d| (d.as_mut_ptr(), d.len()));
        let mut out = ptr::null_mut();
        let mut out_len = 0_usize;
        let mut out_width = 0_u32;
        let mut out_height = 0_u32;
        // SAFETY: the input slice and optional destination stay live for the
        // call; the shim writes at most `dest_len` bytes into the destination
        // and the output pointers refer to initialized writable locals.
        let ok = unsafe {
            j2k_openjph_decode_u8(
                bytes.as_ptr(),
                bytes.len(),
                reduce,
                channels,
                bytes_per_sample,
                dest_ptr,
                dest_len,
                &raw mut out,
                &raw mut out_len,
                &raw mut out_width,
                &raw mut out_height,
            )
        };
        if ok == 0 || out.is_null() {
            if dest_ptr.is_null() && !out.is_null() {
                drop(OpenJphOutput(out));
            }
            return Err("openjph: decode failed".to_string());
        }
        let expected =
            crate::checked_external_output_len("openjph", out_width, out_height, channels)?
                * usize::try_from(bytes_per_sample).unwrap_or(1);
        if out_len != expected {
            if dest_ptr.is_null() {
                drop(OpenJphOutput(out));
            }
            return Err(format!(
                "openjph: unexpected output length {out_len} != {expected}"
            ));
        }
        Ok((out, out_len))
    }

    #[cfg(not(have_openjph))]
    {
        let _ = (bytes, reduce, channels, bytes_per_sample, dest);
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
        bytes_per_sample: u32,
        dest: *mut u8,
        dest_len: usize,
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
    fn decodes_pinned_gray8_and_rgb8_fixtures_in_process() {
        if !super::is_available() {
            return;
        }
        for name in ["openjph-gray-u8-53-raw", "openjph-rgb-u8-53-raw"] {
            let fixture = j2k_test_support::openjph_batch_fixtures()
                .iter()
                .find(|fixture| fixture.name == name)
                .expect("pinned OpenJPH fixture");
            let decoded = if fixture.components == 1 {
                super::decode_gray(fixture.encoded, 0)
            } else {
                super::decode_rgb(fixture.encoded, 0)
            }
            .expect("decode through pinned OpenJPH library");
            assert_eq!(decoded, fixture.oracle, "{name}");
        }
        assert_eq!(super::version(), "0.33.0");
    }
}
