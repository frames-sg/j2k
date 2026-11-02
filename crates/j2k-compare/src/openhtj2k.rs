// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Decode `request` directly into `out`, whose length must equal the
/// output's. A region is in full-resolution coordinates and is decoded with
/// `OpenHTJ2K`'s own viewport API, then scaled by `2^reduce`.
#[cfg_attr(
    have_openhtj2k,
    expect(
        unsafe_code,
        reason = "OpenHTJ2K decode uses the optional checked C++ shim"
    )
)]
pub fn decode_into(
    bytes: &[u8],
    request: crate::ExternalDecodeRequest,
    out: &mut [u8],
) -> Result<(), String> {
    let reduce = u8::try_from(request.reduce.unwrap_or(0))
        .map_err(|_| "openhtj2k: reduction factor exceeds u8".to_string())?;
    let (channels, bytes_per_sample) = (request.color.channels(), request.color.bytes_per_sample());
    #[cfg(have_openhtj2k)]
    {
        let (has_region, x0, y0, x1, y1) = match request.region {
            Some(roi) => {
                let [x0, y0, x1, y1] = crate::checked_region_bounds("openhtj2k", roi)?;
                (1, x0, y0, x1, y1)
            }
            None => (0, 0, 0, 0, 0),
        };
        let mut out_len = 0_usize;
        let mut out_width = 0_u32;
        let mut out_height = 0_u32;
        // SAFETY: the input slice and destination stay live for the call; the
        // shim writes at most `out.len()` bytes into the destination and the
        // output pointers refer to initialized writable locals.
        let ok = unsafe {
            j2k_openhtj2k_decode_u8(
                bytes.as_ptr(),
                bytes.len(),
                reduce,
                request.threads.max(1),
                channels,
                bytes_per_sample,
                has_region,
                x0,
                y0,
                x1,
                y1,
                out.as_mut_ptr(),
                out.len(),
                &raw mut out_len,
                &raw mut out_width,
                &raw mut out_height,
            )
        };
        crate::checked_shim_output(
            "openhtj2k",
            ok != 0,
            (out_len, out_width, out_height),
            (channels, bytes_per_sample),
            || {},
        )
    }

    #[cfg(not(have_openhtj2k))]
    {
        let _ = (bytes, reduce, channels, bytes_per_sample, out);
        Err("openhtj2k: local library not available".to_string())
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
        out_len: *mut usize,
        out_width: *mut u32,
        out_height: *mut u32,
    ) -> i32;
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
        let mut decoded = vec![0_u8; fixture.oracle.len()];
        super::decode_into(
            fixture.encoded,
            crate::ExternalDecodeRequest::gray(),
            &mut decoded,
        )
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
        let mut reference = vec![0_u8; native.data.len()];
        super::decode_into(
            &codestream,
            crate::ExternalDecodeRequest::rgb(),
            &mut reference,
        )
        .expect("OpenHTJ2K decode of native codestream");

        assert_eq!(native.data.len(), reference.len());
        let max_delta = native
            .data
            .iter()
            .zip(&reference)
            .map(|(&left, &right)| left.abs_diff(right))
            .max()
            .unwrap_or_default();
        assert!(
            max_delta <= 1,
            "native/OpenHTJ2K max byte delta {max_delta}"
        );
    }
}
