// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Decode `request` directly into `out`, whose length must equal the
/// output's. `OpenJPH` has no region decode and decodes on the calling
/// thread, so a region request fails and the thread count is ignored.
#[cfg_attr(
    have_openjph,
    expect(
        unsafe_code,
        reason = "OpenJPH decode uses the optional checked C++ shim"
    )
)]
pub fn decode_into(
    bytes: &[u8],
    request: crate::ExternalDecodeRequest,
    out: &mut [u8],
) -> Result<(), String> {
    if request.region.is_some() {
        return Err("openjph: no region decode".to_string());
    }
    let reduce = u8::try_from(request.reduce.unwrap_or(0))
        .map_err(|_| "openjph: reduction factor exceeds u8".to_string())?;
    let (channels, bytes_per_sample) = (request.color.channels(), request.color.bytes_per_sample());
    #[cfg(have_openjph)]
    {
        let mut out_len = 0_usize;
        let mut out_width = 0_u32;
        let mut out_height = 0_u32;
        // SAFETY: the input slice and destination stay live for the call; the
        // shim writes at most `out.len()` bytes into the destination and the
        // output pointers refer to initialized writable locals.
        let ok = unsafe {
            j2k_openjph_decode_u8(
                bytes.as_ptr(),
                bytes.len(),
                reduce,
                channels,
                bytes_per_sample,
                out.as_mut_ptr(),
                out.len(),
                &raw mut out_len,
                &raw mut out_width,
                &raw mut out_height,
            )
        };
        crate::checked_shim_output(
            "openjph",
            ok != 0,
            (out_len, out_width, out_height),
            (channels, bytes_per_sample),
            || {},
        )
    }

    #[cfg(not(have_openjph))]
    {
        let _ = (bytes, reduce, channels, bytes_per_sample, out);
        Err("openjph: local library not available".to_string())
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
        out_len: *mut usize,
        out_width: *mut u32,
        out_height: *mut u32,
    ) -> i32;
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
            let request = if fixture.components == 1 {
                crate::ExternalDecodeRequest::gray()
            } else {
                crate::ExternalDecodeRequest::rgb()
            };
            let mut decoded = vec![0_u8; fixture.oracle.len()];
            super::decode_into(fixture.encoded, request, &mut decoded)
                .expect("decode through pinned OpenJPH library");
            assert_eq!(decoded, fixture.oracle, "{name}");
        }
        assert_eq!(super::version(), "0.33.0");
    }
}
