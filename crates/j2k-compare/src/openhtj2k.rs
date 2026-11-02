// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(have_openhtj2k)]
use std::{ffi::c_void, ptr};

/// Whether the optional pinned `OpenHTJ2K` library was linked into this build.
#[must_use]
pub const fn is_available() -> bool {
    cfg!(have_openhtj2k)
}

/// Version of the linked `OpenHTJ2K` library.
#[must_use]
pub fn version() -> &'static str {
    option_env!("J2K_OPENHTJ2K_VERSION").unwrap_or("unavailable")
}

/// Directory containing the linked `OpenHTJ2K` library.
#[must_use]
pub fn library_path() -> &'static str {
    option_env!("J2K_OPENHTJ2K_LIB_DIR").unwrap_or("unavailable")
}

/// Decode a raw HTJ2K/JPH input to packed 8-bit grayscale in-process.
pub fn decode_gray(bytes: &[u8], reduce: u8, threads: u32) -> Result<Vec<u8>, String> {
    decode(bytes, reduce, threads, 1)
}

/// Decode a raw HTJ2K/JPH input to packed interleaved RGB8 in-process.
pub fn decode_rgb(bytes: &[u8], reduce: u8, threads: u32) -> Result<Vec<u8>, String> {
    decode(bytes, reduce, threads, 3)
}

/// Decode `request` directly into `out`, whose length must equal the
/// output's. A region is in full-resolution coordinates and is decoded with
/// `OpenHTJ2K`'s own viewport API, then scaled by `2^reduce`.
pub fn decode_into(
    bytes: &[u8],
    request: crate::ExternalDecodeRequest,
    out: &mut [u8],
) -> Result<(), String> {
    let reduce = u8::try_from(request.reduce.unwrap_or(0))
        .map_err(|_| "openhtj2k: reduction factor exceeds u8".to_string())?;
    call_shim(
        bytes,
        reduce,
        request.threads,
        (request.color.channels(), request.color.bytes_per_sample()),
        request.region,
        Some(out),
    )
    .map(|_| ())
}

#[cfg_attr(
    have_openhtj2k,
    expect(
        unsafe_code,
        reason = "views the shim allocation that the guard frees exactly once"
    )
)]
fn decode(bytes: &[u8], reduce: u8, threads: u32, channels: u32) -> Result<Vec<u8>, String> {
    let (ptr, len) = call_shim(bytes, reduce, threads, (channels, 1), None, None)?;
    #[cfg(have_openhtj2k)]
    {
        let output = OpenHtj2kOutput(ptr);
        // SAFETY: the non-null shim allocation has been independently bounded
        // and its length matches the checked dimensions and channel count.
        Ok(unsafe { std::slice::from_raw_parts(output.0, len) }.to_vec())
    }
    #[cfg(not(have_openhtj2k))]
    {
        let _ = (ptr, len);
        Err("openhtj2k: local library not available".to_string())
    }
}

/// Runs the shim. With `dest`, the pixels land in `dest` and the returned
/// pointer is `dest`'s; without it, the returned pointer is a shim allocation
/// the caller must free.
#[cfg_attr(
    have_openhtj2k,
    expect(
        unsafe_code,
        reason = "OpenHTJ2K decode uses the optional checked C++ shim"
    )
)]
fn call_shim(
    bytes: &[u8],
    reduce: u8,
    threads: u32,
    (channels, bytes_per_sample): (u32, u32),
    region: Option<j2k_core::Rect>,
    dest: Option<&mut [u8]>,
) -> Result<(*mut u8, usize), String> {
    #[cfg(have_openhtj2k)]
    {
        let (has_region, x0, y0, x1, y1) = match region {
            Some(roi) => {
                let [x0, y0, x1, y1] = crate::checked_region_bounds("openhtj2k", roi)?;
                (1, x0, y0, x1, y1)
            }
            None => (0, 0, 0, 0, 0),
        };
        let (dest_ptr, dest_len) = dest.map_or((ptr::null_mut(), 0), |d| (d.as_mut_ptr(), d.len()));
        let mut out = ptr::null_mut();
        let mut out_len = 0_usize;
        let mut out_width = 0_u32;
        let mut out_height = 0_u32;
        // SAFETY: the input slice and optional destination stay live for the
        // call; the shim writes at most `dest_len` bytes into the destination
        // and the output pointers refer to initialized writable locals.
        let ok = unsafe {
            j2k_openhtj2k_decode_u8(
                bytes.as_ptr(),
                bytes.len(),
                reduce,
                threads.max(1),
                channels,
                bytes_per_sample,
                has_region,
                x0,
                y0,
                x1,
                y1,
                dest_ptr,
                dest_len,
                &raw mut out,
                &raw mut out_len,
                &raw mut out_width,
                &raw mut out_height,
            )
        };
        let owned = dest_ptr.is_null() && !out.is_null();
        crate::checked_shim_output(
            "openhtj2k",
            ok != 0 && !out.is_null(),
            (out_len, out_width, out_height),
            (channels, bytes_per_sample),
            || {
                if owned {
                    drop(OpenHtj2kOutput(out));
                }
            },
        )?;
        Ok((out, out_len))
    }

    #[cfg(not(have_openhtj2k))]
    {
        let _ = (
            bytes,
            reduce,
            threads,
            channels,
            bytes_per_sample,
            region,
            dest,
        );
        Err("openhtj2k: local library not available".to_string())
    }
}

#[cfg(have_openhtj2k)]
struct OpenHtj2kOutput(*mut u8);

#[cfg(have_openhtj2k)]
impl Drop for OpenHtj2kOutput {
    #[expect(
        unsafe_code,
        reason = "the guard frees the OpenHTJ2K shim allocation exactly once"
    )]
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the pointer is the allocation returned by the paired shim.
            unsafe { j2k_openhtj2k_free(self.0.cast()) };
        }
    }
}

#[cfg(have_openhtj2k)]
#[expect(
    unsafe_code,
    reason = "these declarations are the optional OpenHTJ2K C++ shim's complete ABI"
)]
unsafe extern "C" {
    fn j2k_openhtj2k_decode_u8(
        bytes: *const u8,
        len: usize,
        reduce: u8,
        threads: u32,
        channels: u32,
        bytes_per_sample: u32,
        has_region: i32,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        dest: *mut u8,
        dest_len: usize,
        out_data: *mut *mut u8,
        out_len: *mut usize,
        out_width: *mut u32,
        out_height: *mut u32,
    ) -> i32;
    fn j2k_openhtj2k_free(ptr: *mut c_void);
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
        let decoded = super::decode_gray(fixture.encoded, 0, 1)
            .expect("decode through pinned OpenHTJ2K library");
        assert_eq!(decoded, fixture.oracle);
        assert_eq!(super::version(), "0.23.4");
    }

    #[test]
    fn decodes_native_qfactor_rgb_within_one_lsb() {
        if !super::is_available() {
            return;
        }
        let pixels = (0_usize..64 * 64 * 3)
            .map(|index| u8::try_from(index.wrapping_mul(37) & 0xff).expect("masked byte"))
            .collect::<Vec<_>>();
        let options = j2k_native::EncodeOptions {
            reversible: false,
            use_ht_block_coding: true,
            num_decomposition_levels: 5,
            validate_high_throughput_codestream: false,
            ..j2k_native::EncodeOptions::default()
        };
        let codestream =
            j2k_native::encode_htj2k_with_qfactor(&pixels, 64, 64, 3, 8, false, 90, &options)
                .expect("native Qfactor encode");
        let native = j2k_native::Image::new(&codestream, &j2k_native::DecodeSettings::default())
            .expect("parse native Qfactor codestream")
            .decode_native()
            .expect("decode native Qfactor codestream");
        let reference =
            super::decode_rgb(&codestream, 0, 1).expect("OpenHTJ2K decode of native codestream");

        assert_eq!(native.data.len(), reference.len());
        let max_delta = native
            .data
            .iter()
            .zip(reference)
            .map(|(&left, right)| left.abs_diff(right))
            .max()
            .unwrap_or_default();
        assert!(
            max_delta <= 1,
            "native/OpenHTJ2K max byte delta {max_delta}"
        );
    }
}
