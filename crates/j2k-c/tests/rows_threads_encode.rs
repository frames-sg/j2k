// SPDX-License-Identifier: MIT OR Apache-2.0

//! Row streaming, decoder threads, lenient decoding, and lossless encoding
//! through the exported C ABI.

#[path = "support/encode.rs"]
mod support;

use j2k_c::*;
use std::{ffi::c_void, ptr};

struct Handle(*mut J2kCDecoder);

fn message(error: &J2kCError) -> String {
    // SAFETY: The library NUL-terminates every diagnostic it writes.
    unsafe { std::ffi::CStr::from_ptr(error.message.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl Handle {
    fn try_open(bytes: &[u8], options: &J2kCOpenOptions) -> Result<Self, (u32, String)> {
        let mut handle = ptr::null_mut();
        let mut error = J2kCError::default();
        // SAFETY: All buffers and output slots are valid and disjoint for this call.
        let status = unsafe {
            j2k_decoder_open(
                bytes.as_ptr(),
                bytes.len(),
                options,
                &raw mut handle,
                &raw mut error,
            )
        };
        if status == J2K_OK {
            Ok(Self(handle))
        } else {
            Err((status, message(&error)))
        }
    }

    fn open(bytes: &[u8], options: &J2kCOpenOptions) -> Self {
        Self::try_open(bytes, options).unwrap_or_else(|(status, text)| panic!("{status}: {text}"))
    }

    fn output_info(&self, options: &J2kCDecodeOptions) -> Result<J2kCOutputInfo, u32> {
        let mut out = J2kCOutputInfo::default();
        // SAFETY: Handle is live; the call is exclusive; arguments are disjoint.
        let status =
            unsafe { j2k_decoder_output_info(self.0, options, &raw mut out, ptr::null_mut()) };
        if status == J2K_OK {
            Ok(out)
        } else {
            Err(status)
        }
    }

    fn row_info(&self) -> Result<J2kCOutputInfo, u32> {
        let mut out = J2kCOutputInfo::default();
        // SAFETY: Handle is live; the call is exclusive; arguments are disjoint.
        let status =
            unsafe { j2k_decoder_row_info(self.0, ptr::null(), &raw mut out, ptr::null_mut()) };
        if status == J2K_OK {
            Ok(out)
        } else {
            Err(status)
        }
    }

    /// The full image, tightly packed.
    fn decode_full(&self) -> Vec<u8> {
        let options = J2kCDecodeOptions::default();
        let info = self.output_info(&options).unwrap();
        let mut output = vec![0; usize::try_from(info.buffer_bytes).unwrap()];
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; the buffer covers the reported layout.
        let status = unsafe {
            j2k_decoder_decode_into(
                self.0,
                &raw const options,
                output.as_mut_ptr(),
                output.len(),
                usize::try_from(info.row_bytes).unwrap(),
                &raw mut error,
            )
        };
        assert_eq!(status, J2K_OK, "{}", message(&error));
        output
    }

    fn decode_rows(&self, stripe_rows: u32, stop_at: Option<u32>) -> RowRun {
        let mut state = RowState {
            stop_at,
            ..RowState::default()
        };
        let mut out = J2kCOutputInfo::default();
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; `state` outlives the call and the callback
        // only touches it through `user_data`.
        let status = unsafe {
            j2k_decoder_decode_rows(
                self.0,
                ptr::null(),
                stripe_rows,
                Some(collect_row),
                ptr::from_mut(&mut state).cast(),
                &raw mut out,
                &raw mut error,
            )
        };
        RowRun {
            status,
            message: message(&error),
            out,
            state,
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: This is the only owner and its slot is valid for writing.
        unsafe { j2k_decoder_close(&raw mut self.0, ptr::null_mut()) };
    }
}

#[derive(Default)]
struct RowState {
    stop_at: Option<u32>,
    ys: Vec<u32>,
    bytes: Vec<u8>,
    threads: Vec<Option<String>>,
}

struct RowRun {
    status: u32,
    message: String,
    out: J2kCOutputInfo,
    state: RowState,
}

unsafe extern "C" fn collect_row(
    user_data: *mut c_void,
    y: u32,
    row: *const u8,
    row_bytes: usize,
) -> i32 {
    // SAFETY: `decode_rows` passes its `RowState`, and the library a row of
    // `row_bytes` readable bytes valid for this call.
    let (state, row) = unsafe {
        (
            &mut *user_data.cast::<RowState>(),
            std::slice::from_raw_parts(row, row_bytes),
        )
    };
    state.ys.push(y);
    state.bytes.extend_from_slice(row);
    state
        .threads
        .push(std::thread::current().name().map(str::to_owned));
    i32::from(state.stop_at == Some(y))
}

fn options(workers: u32, flags: u32) -> J2kCOpenOptions {
    J2kCOpenOptions {
        workers,
        flags,
        ..J2kCOpenOptions::default()
    }
}

/// Interleaved test samples of `bits` and their encoded bytes.
fn image(width: u32, height: u32, channels: u16, bits: u8, signed: bool, ht: bool) -> Vec<u8> {
    let count = width as usize * height as usize * usize::from(channels);
    let mut input = Vec::new();
    for i in 0..count {
        let i = i32::try_from(i).unwrap();
        let value: i32 = ((i * 37 + 11) % (1 << bits)) - if signed { 1 << (bits - 1) } else { 0 };
        if bits <= 8 {
            input.push(value.to_le_bytes()[0]);
        } else {
            input.extend_from_slice(&value.to_le_bytes()[..2]);
        }
    }
    support::encode(&input, width, height, channels, bits, signed, ht)
}

#[test]
fn rows_equal_decode_into_for_every_native_row_layout_and_thread_count() {
    for (channels, bits, signed, ht) in [
        (1, 8, false, false),
        (3, 8, false, true),
        (1, 12, false, true),
        (3, 16, false, false),
        (1, 12, true, false),
        (3, 16, true, true),
    ] {
        let encoded = image(37, 29, channels, bits, signed, ht);
        let expected = Handle::open(&encoded, &options(0, 0)).decode_full();
        for (workers, stripe_rows) in [(0, 0), (1, 1), (3, 5)] {
            let handle = Handle::open(&encoded, &options(workers, 0));
            let full = handle.output_info(&J2kCDecodeOptions::default()).unwrap();
            let rows = handle.row_info().unwrap();
            assert_eq!(rows, full, "{channels}x{bits} signed={signed}");
            let run = handle.decode_rows(stripe_rows, None);
            let case = format!("{channels}x{bits} signed={signed} workers={workers}");
            assert_eq!(run.status, J2K_OK, "{case}: {}", run.message);
            assert_eq!(run.out, rows, "{case}");
            assert_eq!(run.state.ys, (0..29).collect::<Vec<_>>(), "{case}");
            assert_eq!(run.state.bytes, expected, "{case}");
            assert_eq!(handle.decode_full(), expected, "{case}");
        }
    }
}

#[test]
fn rows_carry_rgba_samples_and_alpha_of_a_jp2_file() {
    use j2k::{
        J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kFileBoxMetadata,
        J2kFileWrapOptions,
    };
    let encoded = image(16, 9, 4, 8, false, true);
    let mut definitions: Vec<_> = (0..3)
        .map(|index| J2kChannelDefinition {
            channel_index: index,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: index + 1 },
        })
        .collect();
    definitions.push(J2kChannelDefinition {
        channel_index: 3,
        channel_type: J2kChannelType::Opacity,
        association: J2kChannelAssociation::WholeImage,
    });
    let file = j2k::wrap_j2k_codestream(
        &encoded,
        J2kFileWrapOptions::jph().with_metadata(J2kFileBoxMetadata {
            channel_definitions: &definitions,
            ..J2kFileBoxMetadata::empty()
        }),
    )
    .unwrap();
    let handle = Handle::open(&file, &options(0, 0));
    let rows = handle.row_info().unwrap();
    assert_eq!((rows.channels, rows.alpha), (4, J2K_ALPHA_STRAIGHT));
    let run = handle.decode_rows(4, None);
    assert_eq!(run.status, J2K_OK, "{}", run.message);
    assert_eq!(run.state.bytes, handle.decode_full());
}

#[test]
fn rows_reject_layouts_they_cannot_keep_native_and_partial_requests() {
    for (bits, signed) in [(5, false), (8, true)] {
        let handle = Handle::open(&image(8, 8, 1, bits, signed, false), &options(0, 0));
        assert_eq!(
            handle.row_info(),
            Err(J2K_UNSUPPORTED),
            "{bits} signed={signed}"
        );
        assert_eq!(handle.decode_rows(0, None).status, J2K_UNSUPPORTED);
        assert_eq!(handle.decode_full().len(), 64 * if signed { 2 } else { 1 });
    }
    let handle = Handle::open(&image(8, 8, 1, 8, false, false), &options(0, 0));
    for partial in [
        J2kCDecodeOptions {
            width: 4,
            height: 4,
            ..J2kCDecodeOptions::default()
        },
        J2kCDecodeOptions {
            reduction: 1,
            ..J2kCDecodeOptions::default()
        },
    ] {
        let mut out = J2kCOutputInfo::default();
        // SAFETY: Handle is live; arguments are disjoint and initialized.
        let status = unsafe {
            j2k_decoder_row_info(handle.0, &raw const partial, &raw mut out, ptr::null_mut())
        };
        assert_eq!(status, J2K_UNSUPPORTED);
    }
}

#[test]
fn a_callback_stops_the_decode_and_the_decoder_stays_usable() {
    let encoded = image(16, 40, 3, 8, false, true);
    let handle = Handle::open(&encoded, &options(2, 0));
    let run = handle.decode_rows(8, Some(3));
    assert_eq!(run.status, J2K_STOPPED, "{}", run.message);
    assert_eq!(run.state.ys, [0, 1, 2, 3]);
    let complete = handle.decode_rows(8, None);
    assert_eq!(complete.status, J2K_OK, "{}", complete.message);
    assert_eq!(complete.state.bytes, handle.decode_full());

    let mut error = J2kCError::default();
    // SAFETY: Handle is live; a null callback is rejected before any decode.
    let status = unsafe {
        j2k_decoder_decode_rows(
            handle.0,
            ptr::null(),
            0,
            None,
            ptr::null_mut(),
            ptr::null_mut(),
            &raw mut error,
        )
    };
    assert_eq!(status, J2K_INVALID_ARGUMENT);
}

#[test]
fn rows_stream_an_image_whose_whole_output_passes_the_output_limit() {
    let encoded = image(32, 24, 3, 16, false, true);
    let limited = J2kCOpenOptions {
        max_output_bytes: 1024,
        ..J2kCOpenOptions::default()
    };
    let handle = Handle::open(&encoded, &limited);
    assert_eq!(
        handle.output_info(&J2kCDecodeOptions::default()),
        Err(J2K_RESOURCE_LIMIT)
    );
    let run = handle.decode_rows(2, None);
    assert_eq!(run.status, J2K_OK, "{}", run.message);
    assert_eq!(run.out.buffer_bytes, 32 * 24 * 3 * 2);
    assert_eq!(
        run.state.bytes,
        Handle::open(&encoded, &options(0, 0)).decode_full()
    );
}

#[test]
fn a_decoder_with_workers_runs_its_rows_on_its_own_threads() {
    let encoded = image(64, 64, 1, 8, false, true);
    let own = Handle::open(&encoded, &options(2, 0)).decode_rows(16, None);
    assert_eq!(own.status, J2K_OK, "{}", own.message);
    assert!(own.state.threads.iter().all(|name| name
        .as_deref()
        .is_some_and(|name| name.starts_with("j2k-c-"))));
    let shared = Handle::open(&encoded, &options(0, 0)).decode_rows(16, None);
    assert_eq!(shared.status, J2K_OK, "{}", shared.message);
    assert!(shared.state.threads.iter().all(|name| !name
        .as_deref()
        .is_some_and(|name| name.starts_with("j2k-c-"))));
    assert_eq!(own.state.bytes, shared.state.bytes);
    assert_eq!(
        Handle::try_open(&encoded, &options(65, 0))
            .err()
            .map(|e| e.0),
        Some(J2K_INVALID_ARGUMENT)
    );
}

#[test]
fn lenient_decoders_recover_a_malformed_trailing_box_and_report_it() {
    let pixels: Vec<u8> = (0..16 * 8)
        .map(|i| u8::try_from(i * 2 % 256).unwrap())
        .collect();
    let codestream = support::encode(&pixels, 16, 8, 1, 8, false, false);
    let mut file = j2k::wrap_j2k_codestream(&codestream, j2k::J2kFileWrapOptions::jp2()).unwrap();
    // A trailing box that claims more bytes than the file holds.
    file.extend_from_slice(&[0, 0, 0, 64, b'x', b'm', b'l', b' ', 1, 2, 3]);

    let strict = Handle::try_open(&file, &options(0, 0));
    assert!(strict.is_err(), "strict decoders reject the truncated box");
    let lenient = Handle::open(&file, &options(0, J2K_OPEN_LENIENT));
    let info = lenient.output_info(&J2kCDecodeOptions::default()).unwrap();
    assert_eq!(info.warnings, J2K_WARNING_LENIENT_RECOVERY);
    assert_eq!(lenient.decode_full(), pixels);
    let run = lenient.decode_rows(0, None);
    assert_eq!(run.status, J2K_OK, "{}", run.message);
    assert_eq!(run.out.warnings, J2K_WARNING_LENIENT_RECOVERY);
    assert_eq!(run.state.bytes, pixels);
}

#[test]
fn image_info_reports_the_reference_grid() {
    let handle = Handle::open(
        j2k_test_support::OPENJPEG_ORIGIN_23X17_AT_3_5,
        &options(0, 0),
    );
    let mut info = J2kCImageInfo::default();
    // SAFETY: Handle is live; arguments are disjoint and initialized.
    let status = unsafe { j2k_decoder_info(handle.0, &raw mut info, ptr::null_mut()) };
    assert_eq!(status, J2K_OK);
    assert_eq!((info.width, info.height), (23, 17));
    assert_eq!((info.image_x0, info.image_y0), (3, 5));
    assert_eq!((info.tile_width, info.tile_height), (26, 22));
}

/// Encode through the C ABI: the status, and the bytes on success.
fn encode(samples: &[u8], options: &J2kCEncodeOptions) -> (u32, Vec<u8>, String) {
    let mut data = ptr::null_mut();
    let mut length = 0;
    let mut error = J2kCError::default();
    // SAFETY: All buffers and output slots are valid and disjoint for this call.
    let status = unsafe {
        j2k_encode_lossless(
            samples.as_ptr(),
            samples.len(),
            options,
            &raw mut data,
            &raw mut length,
            &raw mut error,
        )
    };
    if status != J2K_OK {
        assert!(
            data.is_null() && length == 0,
            "failure leaves the slots empty"
        );
        return (status, Vec::new(), message(&error));
    }
    // SAFETY: A successful encode returns `length` readable bytes at `data`,
    // released here exactly once.
    let bytes = unsafe { std::slice::from_raw_parts(data, length) }.to_vec();
    // SAFETY: `data` and `length` came from `j2k_encode_lossless`.
    unsafe { j2k_buffer_free(data, length) };
    (status, bytes, String::new())
}

#[test]
fn lossless_encode_round_trips_through_the_decoder() {
    let gray: Vec<u8> = (0..24 * 16)
        .map(|i| u8::try_from(i * 7 % 256).unwrap())
        .collect();
    let signed: Vec<u8> = (0..12 * 8 * 3)
        .flat_map(|i: i32| (((i * 91) % 4096) - 2048).to_le_bytes()[..2].to_vec())
        .collect();
    for (samples, width, height, components, bit_depth, flags) in [
        (&gray, 24, 16, 1, 8, 0),
        (&gray, 24, 16, 1, 8, J2K_ENCODE_HTJ2K),
        (&gray, 24, 16, 1, 8, J2K_ENCODE_FILE),
        (&gray, 24, 16, 1, 8, J2K_ENCODE_HTJ2K | J2K_ENCODE_FILE),
        (&signed, 12, 8, 3, 12, J2K_ENCODE_SIGNED | J2K_ENCODE_HTJ2K),
    ] {
        let options = J2kCEncodeOptions {
            width,
            height,
            components,
            bit_depth,
            flags,
            ..J2kCEncodeOptions::default()
        };
        let (status, encoded, text) = encode(samples, &options);
        assert_eq!(status, J2K_OK, "flags {flags}: {text}");
        let file = encoded.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']);
        assert_eq!(file, flags & J2K_ENCODE_FILE != 0, "flags {flags}");
        let handle = Handle::open(&encoded, &self::options(0, 0));
        assert_eq!(handle.decode_full(), *samples, "flags {flags}");
    }
}

#[test]
fn lossless_encode_rejects_bad_arguments_and_leaves_the_result_empty() {
    let samples = [0_u8; 64];
    let valid = J2kCEncodeOptions {
        width: 8,
        height: 8,
        components: 1,
        bit_depth: 8,
        ..J2kCEncodeOptions::default()
    };
    for (case, options) in [
        (
            "bit depth 0",
            J2kCEncodeOptions {
                bit_depth: 0,
                ..valid
            },
        ),
        (
            "bit depth 17",
            J2kCEncodeOptions {
                bit_depth: 17,
                ..valid
            },
        ),
        ("unknown flag", J2kCEncodeOptions { flags: 8, ..valid }),
        (
            "reserved",
            J2kCEncodeOptions {
                reserved: 1,
                ..valid
            },
        ),
        ("too few samples", J2kCEncodeOptions { height: 9, ..valid }),
    ] {
        assert_eq!(encode(&samples, &options).0, J2K_INVALID_ARGUMENT, "{case}");
    }
    let mut data = ptr::without_provenance_mut(1);
    let mut length = 7;
    // SAFETY: Null options are rejected; the result slots are valid.
    let status = unsafe {
        j2k_encode_lossless(
            samples.as_ptr(),
            samples.len(),
            ptr::null(),
            &raw mut data,
            &raw mut length,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, J2K_INVALID_ARGUMENT);
    assert!(data.is_null() && length == 0);
    // SAFETY: Releasing null is a documented no-op.
    unsafe { j2k_buffer_free(ptr::null_mut(), 0) };
}
