#![no_main]

//! Differential fuzzing for the full-image C ABI routes.
//!
//! The input is a complete raw codestream, so seed files stay valid from byte
//! zero. The input length selects inter-row padding only. For representable
//! layouts the target compares the header-fact core route with the batch
//! route, checks row padding, and probes the documented short-buffer and
//! invalid-stride errors. The native route rejects nonzero image origins, so this target only compares
//! decoded pixels for zero-origin images; unrepresented layouts (including nonzero
//! origins) exercise the matching rejection path.

mod common;

use common::{check_error, Decoder, MAX_OUTPUT_BYTES};
use j2k_c::{
    j2k_decoder_decode_full_into, j2k_decoder_decode_into, j2k_decoder_info,
    j2k_decoder_output_info, j2k_decoder_output_info_native, J2kCDecodeOptions, J2kCError,
    J2kCImageInfo, J2kCOutputInfo, J2K_BUFFER_TOO_SMALL, J2K_INVALID_ARGUMENT, J2K_OK,
    J2K_RESOURCE_LIMIT, J2K_UNSUPPORTED,
};
use libfuzzer_sys::fuzz_target;

const MAX_PIXELS: u64 = 1 << 18;

fuzz_target!(|data: &[u8]| {
    let Some(decoder) = Decoder::open_default(data, 1, MAX_OUTPUT_BYTES) else {
        return;
    };
    let mut error = J2kCError::default();
    let mut image = J2kCImageInfo::default();
    // SAFETY: `decoder` owns a live handle and this target uses it serially;
    // the output slot is a valid local.
    let status = unsafe { j2k_decoder_info(decoder.as_ptr(), &mut image, &mut error) };
    if status != J2K_OK {
        check_error(status, &error);
        return;
    }
    if image.width == 0 || image.height == 0 {
        return;
    }
    if u64::from(image.width) * u64::from(image.height) > MAX_PIXELS {
        return;
    }

    let full = J2kCDecodeOptions::default();
    let mut native = J2kCOutputInfo::default();
    // SAFETY: `decoder` owns a live handle and this target uses it serially;
    // `full` and the output slot are valid locals.
    let status =
        unsafe { j2k_decoder_output_info_native(decoder.as_ptr(), &full, &mut native, &mut error) };
    if status != J2K_OK {
        check_error(status, &error);
        assert!(
            matches!(
                status,
                J2K_UNSUPPORTED | J2K_INVALID_ARGUMENT | J2K_RESOURCE_LIMIT
            ),
            "unexpected native output info rejection: {status}"
        );
        let mut canary = [0x5A_u8; 128];
        let mut decode_error = J2kCError::default();
        // SAFETY: `decoder` owns a live handle and this target uses it serially;
        // `canary` covers the declared length and `full` is a valid local.
        let decode_status = unsafe {
            j2k_decoder_decode_full_into(
                decoder.as_ptr(),
                &full,
                canary.as_mut_ptr(),
                canary.len(),
                canary.len(),
                &mut decode_error,
            )
        };
        assert_eq!(
            decode_status, status,
            "decode_full_into status does not match output_info_native"
        );
        check_error(decode_status, &decode_error);
        assert!(
            canary.iter().all(|&b| b == 0x5A),
            "canary buffer modified on rejection"
        );
        return;
    }
    check_error(status, &error);

    let mut general = J2kCOutputInfo::default();
    // SAFETY: `decoder` owns a live handle and this target uses it serially;
    // `full` and the output slot are valid locals.
    let status =
        unsafe { j2k_decoder_output_info(decoder.as_ptr(), &full, &mut general, &mut error) };
    check_error(status, &error);
    if status != J2K_OK {
        return;
    }
    assert_eq!(
        layout_key(&native),
        layout_key(&general),
        "native and general layouts disagree"
    );

    let row_bytes = usize::try_from(native.row_bytes).expect("row_bytes fits usize");
    let height = usize::try_from(native.height).expect("height fits usize");
    assert!(row_bytes > 0 && height > 0);
    let extra_stride = data.len() % 8;
    let stride = row_bytes
        .checked_add(extra_stride)
        .expect("stride overflow");
    let span = stride
        .checked_mul(height - 1)
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .expect("output span overflow");
    if span > MAX_OUTPUT_BYTES as usize {
        return;
    }

    let mut core = vec![0xA5_u8; span];
    let mut batch = vec![0x5A_u8; span];

    // The documented boundary errors are checked before pixel decoding, so
    // they must hold even for codestreams whose tile data is corrupt.
    if span > 1 {
        let mut short = vec![0xCC_u8; span];
        // SAFETY: `decoder` owns a live handle and this target uses it serially;
        // `short` covers `span - 1` writable bytes and `full` is a valid local.
        let status = unsafe {
            j2k_decoder_decode_full_into(
                decoder.as_ptr(),
                &full,
                short.as_mut_ptr(),
                span - 1,
                stride,
                &mut error,
            )
        };
        assert_eq!(status, J2K_BUFFER_TOO_SMALL);
        check_error(status, &error);
        assert!(short.iter().all(|&byte| byte == 0xCC));
    }

    if row_bytes > 1 {
        // SAFETY: `decoder` owns a live handle and this target uses it serially;
        // `core` covers `span` writable bytes and `full` is a valid local.
        let status = unsafe {
            j2k_decoder_decode_full_into(
                decoder.as_ptr(),
                &full,
                core.as_mut_ptr(),
                span,
                row_bytes - 1,
                &mut error,
            )
        };
        assert_eq!(status, J2K_INVALID_ARGUMENT);
        check_error(status, &error);
    }

    // SAFETY: `decoder` owns a live handle and this target uses it serially;
    // `core` covers `span` writable bytes and `full` is a valid local.
    let status_core = unsafe {
        j2k_decoder_decode_full_into(
            decoder.as_ptr(),
            &full,
            core.as_mut_ptr(),
            span,
            stride,
            &mut error,
        )
    };
    // Header facts can accept a codestream whose tile data is corrupt; a
    // decode rejection is a normal fuzz outcome. Only a panic or an internal
    // invariant failure is a bug.
    check_error(status_core, &error);

    // SAFETY: `decoder` owns a live handle and this target uses it serially;
    // `batch` covers `span` writable bytes and `full` is a valid local.
    let status_full = unsafe {
        j2k_decoder_decode_into(
            decoder.as_ptr(),
            &full,
            batch.as_mut_ptr(),
            span,
            stride,
            &mut error,
        )
    };
    check_error(status_full, &error);
    // Both routes accepted the layout via output_info_native and output_info, so
    // they must agree on success or failure, and on failure map errors through
    // the same Error::codec taxonomy while leaving the batch buffer untouched.
    assert_eq!(
        status_full == J2K_OK,
        status_core == J2K_OK,
        "core and batch full-image routes disagreed on success: core={status_core}, batch={status_full}"
    );
    if status_core != J2K_OK {
        assert_eq!(
            status_full, status_core,
            "core and batch full-image routes disagreed on failure status"
        );
        assert!(
            batch.iter().all(|&byte| byte == 0x5A),
            "batch decode_into wrote to the output buffer on failure"
        );
        return;
    }
    assert_rows_equal(&core, &batch, stride, row_bytes, height);
    assert_padding(&core, stride, row_bytes, height, 0xA5);
    assert_padding(&batch, stride, row_bytes, height, 0x5A);

    // A full-image region request stays on the batch engine. Compare it with
    // the header-fact core route; both are zero-origin here because the native
    // route rejects other origins.
    let region = J2kCDecodeOptions {
        x: 0,
        y: 0,
        width: image.width,
        height: image.height,
        ..J2kCDecodeOptions::default()
    };
    let mut forced_batch = vec![0x5A_u8; span];
    // SAFETY: `decoder` owns a live handle and this target uses it serially;
    // `forced_batch` covers `span` writable bytes and `region` is a valid local.
    let status_region = unsafe {
        j2k_decoder_decode_into(
            decoder.as_ptr(),
            &region,
            forced_batch.as_mut_ptr(),
            span,
            stride,
            &mut error,
        )
    };
    assert_eq!(status_region, J2K_OK, "forced-batch region decode failed");
    check_error(status_region, &error);
    assert_rows_equal(&core, &forced_batch, stride, row_bytes, height);
    assert_padding(&forced_batch, stride, row_bytes, height, 0x5A);
});

fn layout_key(info: &J2kCOutputInfo) -> (u32, u32, u32, u32, u32, u32, u32, u32, u64, u64) {
    (
        info.x,
        info.y,
        info.width,
        info.height,
        info.channels,
        info.sample_type,
        info.precision,
        info.alpha,
        info.row_bytes,
        info.buffer_bytes,
    )
}

fn assert_rows_equal(
    expected: &[u8],
    actual: &[u8],
    stride: usize,
    row_bytes: usize,
    height: usize,
) {
    for y in 0..height {
        let start = y * stride;
        let end = start + row_bytes;
        assert_eq!(&expected[start..end], &actual[start..end], "row {y}");
    }
}

fn assert_padding(buffer: &[u8], stride: usize, row_bytes: usize, height: usize, canary: u8) {
    for y in 0..height.saturating_sub(1) {
        let start = y * stride + row_bytes;
        let end = (y + 1) * stride;
        assert!(
            buffer[start..end].iter().all(|&byte| byte == canary),
            "row {y} padding was written"
        );
    }
}
