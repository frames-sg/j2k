#![no_main]

//! State-machine fuzzing for the persistent C decoder.
//!
//! Every input is fed to a decoder that already holds a valid codestream. A
//! rejected feed must leave that codestream inspectable and decodable, and a
//! successful feed must leave a consistent transfer syntax and header. The
//! target also probes the input limit captured at open and null-handle close.

mod common;

use common::{check_error, Decoder, MAX_OUTPUT_BYTES};
use j2k_c::{
    j2k_decoder_close, j2k_decoder_decode_full_into, j2k_decoder_feed, j2k_decoder_info,
    j2k_decoder_output_info_native, j2k_decoder_transfer_syntax, J2kCDecodeOptions, J2kCDecoder,
    J2kCError, J2kCImageInfo, J2kCOutputInfo, J2K_DECODE_FAILED, J2K_INVALID_INPUT, J2K_OK,
    J2K_RESOURCE_LIMIT, J2K_SYNTAX_JPEG2000_LOSSLESS, J2K_SYNTAX_OTHER, J2K_UNSUPPORTED,
};
use libfuzzer_sys::fuzz_target;
use std::ptr;

/// A valid 16x16 HTJ2K RGB8 codestream from the native decoder corpus.
const BASELINE: &[u8] = include_bytes!(
    "../../../j2k-native/fuzz/corpus/native_decode_fuzz/htj2k_rgb8_rpcl_sop_eph.j2c"
);
const MAX_PIXELS: u64 = 1 << 18;

fuzz_target!(|data: &[u8]| {
    let Some(baseline) = Decoder::open_default(BASELINE, 1, MAX_OUTPUT_BYTES) else {
        panic!("the embedded baseline seed must open");
    };
    let mut error = J2kCError::default();
    let mut baseline_info = J2kCImageInfo::default();
    // SAFETY: `baseline` owns a live handle and this target uses it serially;
    // the output slot is a valid local.
    let status = unsafe { j2k_decoder_info(baseline.as_ptr(), &mut baseline_info, &mut error) };
    assert_eq!(status, J2K_OK, "baseline info must be readable");
    check_error(status, &error);
    let mut baseline_syntax = 0_u32;
    // SAFETY: `baseline` owns a live handle and this target uses it serially;
    // the output slot is a valid local.
    let status =
        unsafe { j2k_decoder_transfer_syntax(baseline.as_ptr(), &mut baseline_syntax, &mut error) };
    assert_eq!(status, J2K_OK, "baseline syntax must be readable");
    check_error(status, &error);
    assert!((J2K_SYNTAX_JPEG2000_LOSSLESS..=J2K_SYNTAX_OTHER).contains(&baseline_syntax));
    let baseline_pixels = decode_full(&baseline, &mut error).expect("baseline must decode");

    // SAFETY: `baseline` owns a live handle and this target uses it serially;
    // `data` is a readable slice for its full length.
    let status =
        unsafe { j2k_decoder_feed(baseline.as_ptr(), data.as_ptr(), data.len(), &mut error) };
    if status == J2K_OK {
        check_error(status, &error);
        let mut syntax = 0_u32;
        // SAFETY: `baseline` owns a live handle and this target uses it serially;
        // the output slot is a valid local.
        let status =
            unsafe { j2k_decoder_transfer_syntax(baseline.as_ptr(), &mut syntax, &mut error) };
        assert_eq!(status, J2K_OK, "a held codestream must report its syntax");
        check_error(status, &error);
        assert!((J2K_SYNTAX_JPEG2000_LOSSLESS..=J2K_SYNTAX_OTHER).contains(&syntax));

        let mut info = J2kCImageInfo::default();
        // SAFETY: `baseline` owns a live handle and this target uses it serially;
        // the output slot is a valid local.
        let status = unsafe { j2k_decoder_info(baseline.as_ptr(), &mut info, &mut error) };
        if status == J2K_OK {
            check_error(status, &error);
            if info.width > 0
                && info.height > 0
                && u64::from(info.width) * u64::from(info.height) <= MAX_PIXELS
            {
                if let Some(first_pixels) = decode_full(&baseline, &mut error) {
                    let second_pixels = decode_full(&baseline, &mut error)
                        .expect("second decode of held codestream must succeed");
                    assert_eq!(
                        first_pixels, second_pixels,
                        "second decode of held codestream did not match first decode"
                    );
                }
            }
        } else {
            check_error(status, &error);
        }
    } else {
        check_error(status, &error);
        assert!(
            matches!(
                status,
                J2K_INVALID_INPUT | J2K_UNSUPPORTED | J2K_RESOURCE_LIMIT | J2K_DECODE_FAILED
            ),
            "unexpected feed rejection: {status}"
        );

        let mut syntax_after_fail = 0_u32;
        // SAFETY: `baseline` owns a live handle and this target uses it serially;
        // the output slot is a valid local.
        let status = unsafe {
            j2k_decoder_transfer_syntax(baseline.as_ptr(), &mut syntax_after_fail, &mut error)
        };
        assert_eq!(
            status, J2K_OK,
            "failed feed must not prevent reading syntax"
        );
        check_error(status, &error);
        assert_eq!(
            syntax_after_fail, baseline_syntax,
            "failed feed altered transfer syntax"
        );

        let mut info = J2kCImageInfo::default();
        // SAFETY: `baseline` owns a live handle and this target uses it serially;
        // the output slot is a valid local.
        let status = unsafe { j2k_decoder_info(baseline.as_ptr(), &mut info, &mut error) };
        assert_eq!(status, J2K_OK, "failed feed must not poison the handle");
        check_error(status, &error);
        assert_eq!(info.width, baseline_info.width);
        assert_eq!(info.height, baseline_info.height);
        assert_eq!(info.components, baseline_info.components);
        let pixels =
            decode_full(&baseline, &mut error).expect("failed feed must preserve the codestream");
        assert_eq!(
            pixels, baseline_pixels,
            "failed feed changed decoded output"
        );

        // SAFETY: `baseline` owns a live handle and this target uses it serially;
        // `BASELINE` is a readable slice for its full length.
        let status = unsafe {
            j2k_decoder_feed(
                baseline.as_ptr(),
                BASELINE.as_ptr(),
                BASELINE.len(),
                &mut error,
            )
        };
        assert_eq!(status, J2K_OK, "feeding baseline again must succeed");
        check_error(status, &error);
        let mut syntax_restored = 0_u32;
        // SAFETY: `baseline` owns a live handle and this target uses it serially;
        // the output slot is a valid local.
        let status = unsafe {
            j2k_decoder_transfer_syntax(baseline.as_ptr(), &mut syntax_restored, &mut error)
        };
        assert_eq!(status, J2K_OK);
        check_error(status, &error);
        assert_eq!(
            syntax_restored, baseline_syntax,
            "transfer syntax must return to baseline"
        );
    }

    // The same bytes may be a valid codestream for a fresh handle.
    if let Some(raw) = Decoder::open_default(data, 1, MAX_OUTPUT_BYTES) {
        let mut syntax = 0_u32;
        // SAFETY: `raw` owns a live handle and this target uses it serially;
        // the output slot is a valid local.
        let status = unsafe { j2k_decoder_transfer_syntax(raw.as_ptr(), &mut syntax, &mut error) };
        assert_eq!(
            status, J2K_OK,
            "a freshly opened codestream must report its syntax"
        );
        check_error(status, &error);
        assert!((J2K_SYNTAX_JPEG2000_LOSSLESS..=J2K_SYNTAX_OTHER).contains(&syntax));
    }

    // The input limit is captured at open and must also protect feed.
    if data.len() > BASELINE.len() {
        let Some(limited) = Decoder::open(BASELINE, 1, BASELINE.len() as u64, MAX_OUTPUT_BYTES)
        else {
            panic!("the baseline seed must fit its own limit");
        };
        // SAFETY: `limited` owns a live handle and this target uses it serially;
        // `data` is a readable slice for its full length.
        let status =
            unsafe { j2k_decoder_feed(limited.as_ptr(), data.as_ptr(), data.len(), &mut error) };
        assert_eq!(status, J2K_RESOURCE_LIMIT);
        check_error(status, &error);
        let _ = decode_full(&limited, &mut error)
            .expect("an oversized feed must not poison the previous codestream");
    }

    let mut null: *mut J2kCDecoder = ptr::null_mut();
    // SAFETY: A null handle is a documented no-op, and the slot is valid for
    // writing the resulting null pointer.
    let status = unsafe { j2k_decoder_close(&mut null, &mut error) };
    assert_eq!(status, J2K_OK);
    check_error(status, &error);
    assert!(null.is_null());
});

/// Decode the held codestream through the header-fact core route. Returns
/// `None` for layouts the route does not represent or expected decode errors.
fn decode_full(decoder: &Decoder, error: &mut J2kCError) -> Option<Vec<u8>> {
    let options = J2kCDecodeOptions::default();
    let mut info = J2kCOutputInfo::default();
    // SAFETY: `decoder` owns a live handle and this target uses it serially;
    // `options` and the output slot are valid locals.
    let status =
        unsafe { j2k_decoder_output_info_native(decoder.as_ptr(), &options, &mut info, error) };
    check_error(status, error);
    if status != J2K_OK {
        return None;
    }
    let len = usize::try_from(info.buffer_bytes).ok()?;
    let row_bytes = usize::try_from(info.row_bytes).ok()?;
    if len == 0 || len > MAX_OUTPUT_BYTES as usize || row_bytes == 0 {
        return None;
    }
    let mut output = vec![0_u8; len];
    // SAFETY: `decoder` owns a live handle and this target uses it serially;
    // `output` covers `len` writable bytes and `row_bytes` was checked above.
    let status = unsafe {
        j2k_decoder_decode_full_into(
            decoder.as_ptr(),
            &options,
            output.as_mut_ptr(),
            len,
            row_bytes,
            error,
        )
    };
    check_error(status, error);
    if status != J2K_OK {
        return None;
    }
    Some(output)
}
