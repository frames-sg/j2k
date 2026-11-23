// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calls the exported C ABI with real codec inputs and caller-owned storage.

mod support;

use j2k_c::*;
use std::ptr;

struct Handle(*mut J2kCDecoder);

impl Handle {
    fn open(bytes: &[u8], options: Option<&J2kCOpenOptions>) -> Self {
        let mut handle = ptr::null_mut();
        let mut error = J2kCError::default();
        // SAFETY: All buffers and output slots are valid and disjoint for this call.
        let status = unsafe {
            j2k_decoder_open(
                bytes.as_ptr(),
                bytes.len(),
                options.map_or(ptr::null(), ptr::from_ref),
                &raw mut handle,
                &raw mut error,
            )
        };
        assert_eq!(status, J2K_OK, "{}", message(&error));
        assert!(!handle.is_null());
        Self(handle)
    }

    fn layout(&self, options: &J2kCDecodeOptions) -> J2kCOutputInfo {
        let mut output = J2kCOutputInfo::default();
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; call is exclusive; arguments are disjoint and initialized.
        let status =
            unsafe { j2k_decoder_output_info(self.0, options, &raw mut output, &raw mut error) };
        assert_eq!(status, J2K_OK, "{}", message(&error));
        output
    }

    fn try_layout_native(
        &self,
        options: &J2kCDecodeOptions,
    ) -> Result<J2kCOutputInfo, (u32, String)> {
        let mut output = J2kCOutputInfo::default();
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; call is exclusive; arguments are disjoint and initialized.
        let status = unsafe {
            j2k_decoder_output_info_native(self.0, options, &raw mut output, &raw mut error)
        };
        if status == J2K_OK {
            Ok(output)
        } else {
            Err((status, message(&error)))
        }
    }

    fn image_info(&self) -> J2kCImageInfo {
        let mut info = J2kCImageInfo::default();
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; call is exclusive; the output slot is valid.
        let status = unsafe { j2k_decoder_info(self.0, &raw mut info, &raw mut error) };
        assert_eq!(status, J2K_OK, "{}", message(&error));
        info
    }

    fn try_feed(&self, input: &[u8]) -> Result<(), (u32, String)> {
        let mut error = J2kCError::default();
        // SAFETY: The input slice is readable for its length; handle use is serial.
        let status =
            unsafe { j2k_decoder_feed(self.0, input.as_ptr(), input.len(), &raw mut error) };
        if status == J2K_OK {
            Ok(())
        } else {
            Err((status, message(&error)))
        }
    }

    fn feed(&self, input: &[u8]) {
        self.try_feed(input)
            .unwrap_or_else(|(status, text)| panic!("{status}: {text}"));
    }

    fn transfer_syntax(&self) -> u32 {
        let mut syntax = 0;
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; the output slot is a valid u32.
        let status =
            unsafe { j2k_decoder_transfer_syntax(self.0, &raw mut syntax, &raw mut error) };
        assert_eq!(status, J2K_OK, "{}", message(&error));
        syntax
    }

    fn decode(
        &self,
        full: bool,
        options: &J2kCDecodeOptions,
        output: &mut [u8],
        stride: usize,
    ) -> (u32, String) {
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; the buffer covers the declared span; the call is exclusive.
        let status = unsafe {
            if full {
                j2k_decoder_decode_full_into(
                    self.0,
                    options,
                    output.as_mut_ptr(),
                    output.len(),
                    stride,
                    &raw mut error,
                )
            } else {
                j2k_decoder_decode_into(
                    self.0,
                    options,
                    output.as_mut_ptr(),
                    output.len(),
                    stride,
                    &raw mut error,
                )
            }
        };
        (status, message(&error))
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        assert_eq!(
            // SAFETY: This is the only owner and its slot is valid for writing.
            unsafe { j2k_decoder_close(&raw mut self.0, ptr::null_mut()) },
            J2K_OK
        );
        assert!(self.0.is_null());
    }
}

fn message(error: &J2kCError) -> String {
    let bytes: Vec<_> = error
        .message
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| u8::from_ne_bytes(c.to_ne_bytes()))
        .collect();
    String::from_utf8(bytes).unwrap()
}

#[test]
fn concurrent_classic_and_htj2k_handles_own_input_and_preserve_pixels_and_padding() {
    assert_eq!(j2k_abi_version(), J2K_ABI_VERSION);
    // SAFETY: The library returns a static NUL-terminated string.
    let version = unsafe { std::ffi::CStr::from_ptr(j2k_version()) };
    assert_eq!(version.to_str(), Ok(env!("CARGO_PKG_VERSION")));
    // Separate handles may be used concurrently; each thread owns one.
    std::thread::scope(|scope| {
        for ht in [false, true] {
            scope.spawn(move || decode_full_and_region(ht));
        }
    });
}

fn decode_full_and_region(ht: bool) {
    let (mut encoded, pixels) = support::gray_fixture(ht);
    let handle = Handle::open(&encoded, None);
    encoded.fill(0); // Successful open must no longer borrow the input.
    let options = J2kCDecodeOptions::default();
    let layout = handle.layout(&options);
    assert_eq!((layout.width, layout.height, layout.channels), (32, 24, 1));
    assert_eq!((layout.sample_type, layout.precision), (J2K_U8, 8));
    let mut output = vec![0; usize::try_from(layout.buffer_bytes).unwrap()];
    let mut error = J2kCError::default();
    // SAFETY: Live exclusive handle and a correctly sized output allocation.
    let status = unsafe {
        j2k_decoder_decode_into(
            handle.0,
            &raw const options,
            output.as_mut_ptr(),
            output.len(),
            usize::try_from(layout.row_bytes).unwrap(),
            &raw mut error,
        )
    };
    assert_eq!(status, J2K_OK, "{}", message(&error));
    assert_eq!(output, pixels, "ht={ht}");

    let options = J2kCDecodeOptions {
        x: 3,
        y: 5,
        width: 7,
        height: 6,
        ..Default::default()
    };
    let layout = handle.layout(&options);
    assert_eq!(
        (layout.x, layout.y, layout.width, layout.height),
        (3, 5, 7, 6)
    );
    let stride = 11;
    let needed = stride * 5 + 7;
    let mut output = vec![0xA5; needed + 4];
    // SAFETY: Buffer capacity is valid; the intentionally short declared span must be rejected.
    let status = unsafe {
        j2k_decoder_decode_into(
            handle.0,
            &raw const options,
            output.as_mut_ptr(),
            needed - 1,
            stride,
            &raw mut error,
        )
    };
    assert_eq!(status, J2K_BUFFER_TOO_SMALL);
    assert!(output.iter().all(|&v| v == 0xA5));
    // SAFETY: Same handle and valid destination now cover the whole requested image.
    let status = unsafe {
        j2k_decoder_decode_into(
            handle.0,
            &raw const options,
            output.as_mut_ptr(),
            needed,
            stride,
            &raw mut error,
        )
    };
    assert_eq!(status, J2K_OK, "{}", message(&error));
    assert_eq!(error.code, J2K_OK);
    assert_eq!(error.message[0], 0);
    for row in 0..6 {
        assert_eq!(
            &output[row * stride..row * stride + 7],
            &pixels[(row + 5) * 32 + 3..(row + 5) * 32 + 10]
        );
        let padding_end = ((row + 1) * stride).min(output.len());
        assert!(output[row * stride + 7..padding_end]
            .iter()
            .all(|&v| v == 0xA5));
    }
}

fn samples(width: u32, height: u32, channels: u32, bits: u8) -> Vec<u8> {
    let mut samples = Vec::new();
    for index in 0..width * height * channels {
        let value = (index * 37 + 11) % (1_u32 << bits);
        if bits <= 8 {
            samples.push(u8::try_from(value).unwrap());
        } else {
            samples.extend_from_slice(&u16::try_from(value).unwrap().to_le_bytes());
        }
    }
    samples
}

fn tight_rows(output: &[u8], stride: usize, row_bytes: usize, height: usize) -> Vec<u8> {
    output
        .chunks(stride)
        .take(height)
        .flat_map(|line| line[..row_bytes].iter().copied())
        .collect()
}

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

#[test]
fn feed_replaces_the_codestream_and_failed_feeds_keep_the_previous_one() {
    let (first, first_pixels) = support::gray_fixture(false);
    let second_pixels = samples(40, 20, 3, 8);
    let second = support::encode(&second_pixels, 40, 20, 3, 8, false, true);
    let handle = Handle::open(&first, None);
    let info = handle.image_info();
    assert_eq!((info.width, info.height, info.components), (32, 24, 1));
    assert_eq!(first_pixels.len(), 32 * 24);

    handle.feed(&second);
    let info = handle.image_info();
    assert_eq!((info.width, info.height, info.components), (40, 20, 3));
    let options = J2kCDecodeOptions::default();
    let layout = handle.layout(&options);
    let stride = usize::try_from(layout.row_bytes).unwrap();
    let mut output = vec![0; usize::try_from(layout.buffer_bytes).unwrap()];
    let (status, text) = handle.decode(false, &options, &mut output, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_eq!(output, second_pixels);

    let syntax = handle.transfer_syntax();
    for (bad, expected_status) in [
        (&b""[..], J2K_INVALID_INPUT),
        (&b"not a JPEG 2000 codestream"[..], J2K_INVALID_INPUT),
        (
            &b"\x00\x00\x00\x0cjP  \r\n\x87\n\x00"[..],
            J2K_INVALID_INPUT,
        ),
    ] {
        let (status, text) = handle.try_feed(bad).unwrap_err();
        assert_eq!(status, expected_status, "{text}");
        assert_ne!(text, "", "a rejected feed should carry a diagnostic");
        let info = handle.image_info();
        assert_eq!(
            (info.width, info.height, info.components),
            (40, 20, 3),
            "a failed feed replaced metadata"
        );
        assert_eq!(
            handle.transfer_syntax(),
            syntax,
            "a failed feed changed transfer syntax"
        );
        let mut output = vec![0; usize::try_from(layout.buffer_bytes).unwrap()];
        let (status, text) = handle.decode(false, &options, &mut output, stride);
        assert_eq!(status, J2K_OK, "{text}");
        assert_eq!(
            output, second_pixels,
            "a failed feed replaced the codestream"
        );
    }
}

#[test]
fn feed_enforces_the_input_limit_captured_at_open() {
    let (small, small_pixels) = support::gray_fixture(false);
    let big_pixels = samples(128, 128, 3, 8);
    let big = support::encode(&big_pixels, 128, 128, 3, 8, false, false);
    assert!(big.len() > small.len());
    let options = J2kCOpenOptions {
        max_input_bytes: u64::try_from(small.len()).unwrap(),
        ..Default::default()
    };
    let handle = Handle::open(&small, Some(&options));
    let (status, text) = handle.try_feed(&big).unwrap_err();
    assert_eq!(status, J2K_RESOURCE_LIMIT);
    assert!(text.contains("max_input_bytes"));
    let options = J2kCDecodeOptions::default();
    let layout = handle.layout(&options);
    let mut output = vec![0; usize::try_from(layout.buffer_bytes).unwrap()];
    let (status, text) = handle.decode(
        false,
        &options,
        &mut output,
        usize::try_from(layout.row_bytes).unwrap(),
    );
    assert_eq!(status, J2K_OK, "{text}");
    assert_eq!(output, small_pixels);
}

#[test]
fn transfer_syntax_reports_the_held_codestream_class() {
    let (classic, _) = support::gray_fixture(false);
    let (ht, _) = support::gray_fixture(true);
    let handle = Handle::open(&classic, None);
    assert_eq!(handle.transfer_syntax(), J2K_SYNTAX_JPEG2000_LOSSLESS);
    handle.feed(&ht);
    assert_eq!(handle.transfer_syntax(), J2K_SYNTAX_HT_JPEG2000_LOSSLESS);
    handle.feed(&classic);
    assert_eq!(handle.transfer_syntax(), J2K_SYNTAX_JPEG2000_LOSSLESS);
}

#[test]
fn native_layout_matches_the_general_query_and_full_decode_matches_decode_into() {
    for ht in [false, true] {
        for (channels, bits) in [(1_u16, 8_u8), (3, 8), (1, 12), (3, 16)] {
            let pixels = samples(32, 24, u32::from(channels), bits);
            let encoded = support::encode(&pixels, 32, 24, channels, bits, false, ht);
            let handle = Handle::open(&encoded, None);
            let options = J2kCDecodeOptions::default();
            let general = handle.layout(&options);
            let native = handle
                .try_layout_native(&options)
                .unwrap_or_else(|(status, text)| {
                    panic!("channels={channels} bits={bits} ht={ht}: {status}: {text}")
                });
            assert_eq!(layout_key(&native), layout_key(&general));
            let row_bytes = usize::try_from(native.row_bytes).unwrap();
            let height = usize::try_from(native.height).unwrap();
            let stride = row_bytes + 3;
            let length = stride * (height - 1) + row_bytes;

            let mut expected = vec![0xA5; length];
            let (status, text) = handle.decode(false, &options, &mut expected, stride);
            assert_eq!(status, J2K_OK, "{text}");
            let mut full = vec![0xA5; length];
            let (status, text) = handle.decode(true, &options, &mut full, stride);
            assert_eq!(status, J2K_OK, "{text}");
            assert_eq!(full, expected);
            assert_eq!(tight_rows(&full, stride, row_bytes, height), pixels);
            for line in full.chunks(stride) {
                assert!(line[row_bytes..].iter().all(|&value| value == 0xA5));
            }

            let mut short = vec![0xA5; length - 1];
            let (status, text) = handle.decode(true, &options, &mut short, stride);
            assert_eq!(status, J2K_BUFFER_TOO_SMALL, "{text}");
            assert!(short.iter().all(|&value| value == 0xA5));

            let mut full = vec![0; length];
            let (status, text) = handle.decode(true, &options, &mut full, row_bytes - 1);
            assert_eq!(status, J2K_INVALID_ARGUMENT, "{text}");
        }
    }
}

#[test]
fn native_route_rejects_precisions_the_core_decoder_does_not_preserve() {
    for bits in [1_u8, 7] {
        let pixels = samples(32, 24, 1, bits);
        let encoded = support::encode(&pixels, 32, 24, 1, bits, false, false);
        let handle = Handle::open(&encoded, None);
        let options = J2kCDecodeOptions::default();
        let general = handle.layout(&options);
        assert_eq!(
            (general.sample_type, general.precision),
            (J2K_U8, u32::from(bits))
        );
        let (status, text) = handle.try_layout_native(&options).unwrap_err();
        assert_eq!(status, J2K_UNSUPPORTED, "{text}");

        let row_bytes = usize::try_from(general.row_bytes).unwrap();
        let height = usize::try_from(general.height).unwrap();
        let stride = row_bytes + 2;
        let length = stride * (height - 1) + row_bytes;
        let mut output = vec![0xA5; length];
        let (status, text) = handle.decode(false, &options, &mut output, stride);
        assert_eq!(status, J2K_OK, "{text}");
        assert_eq!(tight_rows(&output, stride, row_bytes, height), pixels);
        for line in output.chunks(stride) {
            assert!(line[row_bytes..].iter().all(|&value| value == 0xA5));
        }

        let mut full = vec![0xA5; length];
        let (status, text) = handle.decode(true, &options, &mut full, stride);
        assert_eq!(status, J2K_UNSUPPORTED, "{text}");
        assert!(full.iter().all(|&value| value == 0xA5));
    }

    let mut pixels = Vec::new();
    for index in 0..32 * 24 {
        let value = (index * 37 + 11) % (1_u32 << 17);
        pixels.extend_from_slice(&value.to_le_bytes()[..3]);
    }
    let encoded = support::encode(&pixels, 32, 24, 1, 17, false, false);
    let handle = Handle::open(&encoded, None);
    let options = J2kCDecodeOptions::default();
    let (status, text) = handle.try_layout_native(&options).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
}

#[test]
fn native_route_rejects_images_with_a_nonzero_origin() {
    use j2k_test_support::{OPENJPEG_ORIGIN_23X17_AT_3_5, ORIGIN_SOURCE_23X17_RGB};

    let handle = Handle::open(OPENJPEG_ORIGIN_23X17_AT_3_5, None);
    let options = J2kCDecodeOptions::default();
    let general = handle.layout(&options);
    assert_eq!((general.width, general.height), (23, 17));
    let row_bytes = usize::try_from(general.row_bytes).unwrap();
    let height = usize::try_from(general.height).unwrap();
    let stride = row_bytes + 2;
    let length = stride * (height - 1) + row_bytes;
    let mut output = vec![0xA5; length];
    let (status, text) = handle.decode(false, &options, &mut output, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_eq!(
        tight_rows(&output, stride, row_bytes, height),
        ORIGIN_SOURCE_23X17_RGB
    );

    let (status, text) = handle.try_layout_native(&options).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    let mut full = vec![0xA5; length];
    let (status, text) = handle.decode(true, &options, &mut full, stride);
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    assert!(full.iter().all(|&value| value == 0xA5));
}

#[test]
fn native_route_rejects_requests_the_core_decoder_does_not_represent() {
    let (encoded, _) = support::gray_fixture(false);
    let handle = Handle::open(&encoded, None);
    for options in [
        J2kCDecodeOptions {
            x: 1,
            y: 0,
            width: 8,
            height: 8,
            ..Default::default()
        },
        J2kCDecodeOptions {
            reduction: 1,
            ..Default::default()
        },
    ] {
        let (status, text) = handle.try_layout_native(&options).unwrap_err();
        assert_eq!(status, J2K_UNSUPPORTED, "{text}");
        let mut output = vec![0; 1024];
        let (status, text) = handle.decode(true, &options, &mut output, 32);
        assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    }

    let signed_pixels: Vec<u8> = (0..32 * 24)
        .map(|index| {
            i8::try_from((index * 37 + 11) % 256 - 128)
                .unwrap()
                .cast_unsigned()
        })
        .collect();
    let signed = support::encode(&signed_pixels, 32, 24, 1, 8, true, false);
    let signed_handle = Handle::open(&signed, None);
    let options = J2kCDecodeOptions::default();
    let general = signed_handle.layout(&options);
    assert_eq!(general.sample_type, J2K_I16_LE);
    let mut output = vec![0; usize::try_from(general.buffer_bytes).unwrap()];
    let (status, text) = signed_handle.decode(
        false,
        &options,
        &mut output,
        usize::try_from(general.row_bytes).unwrap(),
    );
    assert_eq!(status, J2K_OK, "{text}");
    let (status, text) = signed_handle.try_layout_native(&options).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");

    let wrapped = j2k::wrap_j2k_codestream(&encoded, j2k::J2kFileWrapOptions::jp2()).unwrap();
    let wrapped_handle = Handle::open(&wrapped, None);
    assert_eq!(wrapped_handle.layout(&options).width, 32);
    let (status, text) = wrapped_handle.try_layout_native(&options).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
}

fn assert_padding_intact(
    buffer: &[u8],
    stride: usize,
    row_bytes: usize,
    height: usize,
    canary: u8,
) {
    for row in 0..height.saturating_sub(1) {
        let padding_start = row * stride + row_bytes;
        let padding_end = (row + 1) * stride;
        assert!(
            buffer[padding_start..padding_end]
                .iter()
                .all(|&v| v == canary),
            "padding in row {row} was modified"
        );
    }
}

fn assert_rows_match(a: &[u8], b: &[u8], stride: usize, row_bytes: usize, height: usize) {
    for row in 0..height {
        let start = row * stride;
        let end = start + row_bytes;
        assert_eq!(&a[start..end], &b[start..end], "mismatch at row {row}");
    }
}

#[test]
fn batch_baseline_parity_for_supported_fixtures() {
    for ht in [false, true] {
        for channels in [1_u16, 3] {
            for bits in [8_u8, 12, 16] {
                let pixels = samples(32, 24, u32::from(channels), bits);
                let encoded = support::encode(&pixels, 32, 24, channels, bits, false, ht);
                let handle = Handle::open(&encoded, None);
                let options_full = J2kCDecodeOptions::default();
                let native_layout = handle.try_layout_native(&options_full).unwrap();
                let row_bytes = usize::try_from(native_layout.row_bytes).unwrap();
                let height = usize::try_from(native_layout.height).unwrap();
                let stride = row_bytes + 7;
                let length = stride * (height - 1) + row_bytes;

                // (a) native j2k_decoder_decode_full_into
                let mut native_out = vec![0xAA; length];
                let (status, text) = handle.decode(true, &options_full, &mut native_out, stride);
                assert_eq!(status, J2K_OK, "{text}");
                assert_padding_intact(&native_out, stride, row_bytes, height, 0xAA);

                // (b) batch decode_into with default full-image options
                let mut full_out = vec![0xBB; length];
                let (status, text) = handle.decode(false, &options_full, &mut full_out, stride);
                assert_eq!(status, J2K_OK, "{text}");
                assert_padding_intact(&full_out, stride, row_bytes, height, 0xBB);

                // (c) forced batch region options {x:0, y:0, width:W, height:H}
                let options_region = J2kCDecodeOptions {
                    x: 0,
                    y: 0,
                    width: 32,
                    height: 24,
                    ..Default::default()
                };
                let mut region_out = vec![0xCC; length];
                let (status, text) = handle.decode(false, &options_region, &mut region_out, stride);
                assert_eq!(status, J2K_OK, "{text}");
                assert_padding_intact(&region_out, stride, row_bytes, height, 0xCC);

                assert_rows_match(&native_out, &full_out, stride, row_bytes, height);
                assert_rows_match(&full_out, &region_out, stride, row_bytes, height);
                assert_eq!(tight_rows(&native_out, stride, row_bytes, height), pixels);
            }
        }
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "sweeps subsampled and mixed-sampling fixtures"
)]
fn uniform_and_mixed_component_subsampling_fixtures() {
    use j2k_test_support::j2k_native::{
        encode_component_planes_53, EncodeComponentPlane, EncodeOptions,
    };

    let width = 17_u32;
    let height = 13_u32;
    let subsampled_samples = usize::try_from(width.div_ceil(2) * height.div_ceil(2)).unwrap(); // 9 * 7 = 63
    let full_samples = usize::try_from(width * height).unwrap(); // 17 * 13 = 221

    let d_sub1: Vec<u8> = (0..subsampled_samples)
        .map(|i| u8::try_from((i * 37 + 11) % 256).unwrap())
        .collect();
    let d_sub2: Vec<u8> = (0..subsampled_samples)
        .map(|i| u8::try_from((i * 41 + 17) % 256).unwrap())
        .collect();
    let d_sub3: Vec<u8> = (0..subsampled_samples)
        .map(|i| u8::try_from((i * 43 + 23) % 256).unwrap())
        .collect();
    let d_full: Vec<u8> = (0..full_samples)
        .map(|i| u8::try_from((i * 19 + 7) % 256).unwrap())
        .collect();

    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct: false,
        ..EncodeOptions::default()
    };
    let decode_options = J2kCDecodeOptions::default();

    // (a) One component at 2x2
    let planes_1c = [EncodeComponentPlane {
        data: &d_sub1,
        x_rsiz: 2,
        y_rsiz: 2,
    }];
    let codestream_1c = encode_component_planes_53(&planes_1c, width, height, 8, false, &options)
        .expect("encode 1-component 2x2 subsampled fixture");
    let handle_1c = Handle::open(&codestream_1c, None);
    let (status, _) = handle_1c.try_layout_native(&decode_options).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED);
    let mut canary_1c = vec![0xA5_u8; 512];
    let (status, _) = handle_1c.decode(true, &decode_options, &mut canary_1c, 32);
    assert_eq!(status, J2K_UNSUPPORTED);
    assert!(canary_1c.iter().all(|&b| b == 0xA5));

    // (b) Three components all at 2x2
    let planes_3c = [
        EncodeComponentPlane {
            data: &d_sub1,
            x_rsiz: 2,
            y_rsiz: 2,
        },
        EncodeComponentPlane {
            data: &d_sub2,
            x_rsiz: 2,
            y_rsiz: 2,
        },
        EncodeComponentPlane {
            data: &d_sub3,
            x_rsiz: 2,
            y_rsiz: 2,
        },
    ];
    let codestream_3c = encode_component_planes_53(&planes_3c, width, height, 8, false, &options)
        .expect("encode 3-component 2x2 subsampled fixture");
    let handle_3c = Handle::open(&codestream_3c, None);
    let (status, _) = handle_3c.try_layout_native(&decode_options).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED);
    let mut canary_3c = vec![0xA5_u8; 1024];
    let (status, _) = handle_3c.decode(true, &decode_options, &mut canary_3c, 64);
    assert_eq!(status, J2K_UNSUPPORTED);
    assert!(canary_3c.iter().all(|&b| b == 0xA5));

    // (c) Mixed-sampling parity case (comp 0 at 1x1, others at 2x2)
    let planes_mixed = [
        EncodeComponentPlane {
            data: &d_full,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        EncodeComponentPlane {
            data: &d_sub2,
            x_rsiz: 2,
            y_rsiz: 2,
        },
        EncodeComponentPlane {
            data: &d_sub3,
            x_rsiz: 2,
            y_rsiz: 2,
        },
    ];
    let codestream_mixed =
        encode_component_planes_53(&planes_mixed, width, height, 8, false, &options)
            .expect("encode mixed-sampling fixture");
    let handle_mixed = Handle::open(&codestream_mixed, None);
    let native_layout = handle_mixed
        .try_layout_native(&decode_options)
        .expect("native info must succeed for mixed component sampling");
    assert_eq!((native_layout.width, native_layout.height), (17, 13));
    let row_bytes = usize::try_from(native_layout.row_bytes).unwrap();
    let height_mixed = usize::try_from(native_layout.height).unwrap();
    let stride = row_bytes + 5;
    let length = stride * (height_mixed - 1) + row_bytes;

    let mut native_out = vec![0xA5_u8; length];
    let (status, text) = handle_mixed.decode(true, &decode_options, &mut native_out, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_padding_intact(&native_out, stride, row_bytes, height_mixed, 0xA5);

    let region_options = J2kCDecodeOptions {
        x: 0,
        y: 0,
        width,
        height,
        ..Default::default()
    };
    let mut batch_out = vec![0x5A_u8; length];
    let (status, text) = handle_mixed.decode(false, &region_options, &mut batch_out, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_padding_intact(&batch_out, stride, row_bytes, height_mixed, 0x5A);

    assert_rows_match(&native_out, &batch_out, stride, row_bytes, height_mixed);
}

#[test]
#[expect(
    clippy::too_many_lines,
    clippy::similar_names,
    reason = "verifies native route rejections and batch fallback parity for wrapped and 4c layouts"
)]
fn wrapped_and_four_component_codestreams_reject_native_route_and_preserve_batch_parity() {
    use j2k_test_support::j2k_native::{
        encode_component_planes_53, EncodeComponentPlane, EncodeOptions,
    };

    let rgb_pixels = samples(32, 24, 3, 8);
    let raw_rgb = support::encode(&rgb_pixels, 32, 24, 3, 8, false, false);
    let options_full = J2kCDecodeOptions::default();
    let options_region = J2kCDecodeOptions {
        x: 0,
        y: 0,
        width: 32,
        height: 24,
        ..Default::default()
    };

    // 1. JP2-wrapped codestream
    let jp2_bytes = j2k_test_support::wrap_codestream_jp2(&raw_rgb, 32, 24, 3, 8, 16);
    let handle_jp2 = Handle::open(&jp2_bytes, None);
    let (status, text) = handle_jp2.try_layout_native(&options_full).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");

    let general_layout = handle_jp2.layout(&options_full);
    let row_bytes = usize::try_from(general_layout.row_bytes).unwrap();
    let height = usize::try_from(general_layout.height).unwrap();
    let stride = row_bytes + 6;
    let length = stride * (height - 1) + row_bytes;

    let mut canary = vec![0xA5_u8; length];
    let (status, text) = handle_jp2.decode(true, &options_full, &mut canary, stride);
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    assert!(canary.iter().all(|&b| b == 0xA5));

    let mut full_out = vec![0xBB_u8; length];
    let (status, text) = handle_jp2.decode(false, &options_full, &mut full_out, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_padding_intact(&full_out, stride, row_bytes, height, 0xBB);

    let mut region_out = vec![0xCC_u8; length];
    let (status, text) = handle_jp2.decode(false, &options_region, &mut region_out, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_padding_intact(&region_out, stride, row_bytes, height, 0xCC);

    assert_rows_match(&full_out, &region_out, stride, row_bytes, height);

    // 2. JPH-wrapped codestream
    let raw_ht = support::encode(&rgb_pixels, 32, 24, 3, 8, false, true);
    let jph_bytes = j2k::wrap_j2k_codestream(&raw_ht, j2k::J2kFileWrapOptions::jph()).unwrap();
    let handle_jph = Handle::open(&jph_bytes, None);
    let (status, text) = handle_jph.try_layout_native(&options_full).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");

    let mut canary = vec![0xA5_u8; length];
    let (status, text) = handle_jph.decode(true, &options_full, &mut canary, stride);
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    assert!(canary.iter().all(|&b| b == 0xA5));

    let mut full_out = vec![0xBB_u8; length];
    let (status, text) = handle_jph.decode(false, &options_full, &mut full_out, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_padding_intact(&full_out, stride, row_bytes, height, 0xBB);

    let mut region_out = vec![0xCC_u8; length];
    let (status, text) = handle_jph.decode(false, &options_region, &mut region_out, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_padding_intact(&region_out, stride, row_bytes, height, 0xCC);

    assert_rows_match(&full_out, &region_out, stride, row_bytes, height);

    // 3. 4-component raw codestream
    let p0: Vec<u8> = (0..32 * 24)
        .map(|i| u8::try_from((i * 17) % 256).unwrap())
        .collect();
    let p1: Vec<u8> = (0..32 * 24)
        .map(|i| u8::try_from((i * 19 + 3) % 256).unwrap())
        .collect();
    let p2: Vec<u8> = (0..32 * 24)
        .map(|i| u8::try_from((i * 23 + 7) % 256).unwrap())
        .collect();
    let p3: Vec<u8> = (0..32 * 24)
        .map(|i| u8::try_from((i * 29 + 11) % 256).unwrap())
        .collect();
    let planes = [
        EncodeComponentPlane {
            data: &p0,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        EncodeComponentPlane {
            data: &p1,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        EncodeComponentPlane {
            data: &p2,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        EncodeComponentPlane {
            data: &p3,
            x_rsiz: 1,
            y_rsiz: 1,
        },
    ];
    let raw_4c = encode_component_planes_53(
        &planes,
        32,
        24,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode 4-component codestream");
    let handle_4c = Handle::open(&raw_4c, None);
    let (status, text) = handle_4c.try_layout_native(&options_full).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");

    let mut canary = vec![0xA5_u8; 1024];
    let (status, text) = handle_4c.decode(true, &options_full, &mut canary, 128);
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    assert!(canary.iter().all(|&b| b == 0xA5));

    // Raw 4-component input lacks channel definitions, so decode_into rejects it
    // as J2K_UNSUPPORTED on both full and region options without touching the buffer.
    let mut full_4c = vec![0xBB_u8; 1024];
    let (status_full, _) = handle_4c.decode(false, &options_full, &mut full_4c, 128);
    let mut region_4c = vec![0xCC_u8; 1024];
    let options_region_4c = J2kCDecodeOptions {
        x: 0,
        y: 0,
        width: 32,
        height: 24,
        ..Default::default()
    };
    let (status_region, _) = handle_4c.decode(false, &options_region_4c, &mut region_4c, 128);
    assert_eq!(status_full, J2K_UNSUPPORTED);
    assert_eq!(status_region, J2K_UNSUPPORTED);
    assert!(full_4c.iter().all(|&b| b == 0xBB));
    assert!(region_4c.iter().all(|&b| b == 0xCC));

    // 4. 4-channel RGBA wrapped codestream with channel definitions (successful batch decode)
    let rgba_pixels: Vec<u8> = (0..32 * 24)
        .flat_map(|i| {
            [
                u8::try_from(i % 128).unwrap(),
                u8::try_from((i * 2) % 128).unwrap(),
                u8::try_from((i * 3) % 128).unwrap(),
                255_u8,
            ]
        })
        .collect();
    let raw_rgba = support::encode(&rgba_pixels, 32, 24, 4, 8, false, false);
    let channel_defs = [
        j2k::J2kChannelDefinition {
            channel_index: 0,
            channel_type: j2k::J2kChannelType::Color,
            association: j2k::J2kChannelAssociation::Color { index: 1 },
        },
        j2k::J2kChannelDefinition {
            channel_index: 1,
            channel_type: j2k::J2kChannelType::Color,
            association: j2k::J2kChannelAssociation::Color { index: 2 },
        },
        j2k::J2kChannelDefinition {
            channel_index: 2,
            channel_type: j2k::J2kChannelType::Color,
            association: j2k::J2kChannelAssociation::Color { index: 3 },
        },
        j2k::J2kChannelDefinition {
            channel_index: 3,
            channel_type: j2k::J2kChannelType::Opacity,
            association: j2k::J2kChannelAssociation::WholeImage,
        },
    ];
    let wrapped_rgba = j2k::wrap_j2k_codestream(
        &raw_rgba,
        j2k::J2kFileWrapOptions::jp2().with_metadata(j2k::J2kFileBoxMetadata {
            channel_definitions: &channel_defs,
            ..j2k::J2kFileBoxMetadata::empty()
        }),
    )
    .unwrap();
    let handle_rgba = Handle::open(&wrapped_rgba, None);
    let (status, text) = handle_rgba.try_layout_native(&options_full).unwrap_err();
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");

    let general_layout_rgba = handle_rgba.layout(&options_full);
    let row_bytes_rgba = usize::try_from(general_layout_rgba.row_bytes).unwrap();
    let height_rgba = usize::try_from(general_layout_rgba.height).unwrap();
    let stride_rgba = row_bytes_rgba + 8;
    let length_rgba = stride_rgba * (height_rgba - 1) + row_bytes_rgba;

    let mut canary_rgba = vec![0xA5_u8; length_rgba];
    let (status, text) = handle_rgba.decode(true, &options_full, &mut canary_rgba, stride_rgba);
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    assert!(canary_rgba.iter().all(|&b| b == 0xA5));

    let mut full_rgba = vec![0xBB_u8; length_rgba];
    let (status, text) = handle_rgba.decode(false, &options_full, &mut full_rgba, stride_rgba);
    assert_eq!(status, J2K_OK, "{text}");
    assert_padding_intact(&full_rgba, stride_rgba, row_bytes_rgba, height_rgba, 0xBB);

    let mut region_rgba = vec![0xCC_u8; length_rgba];
    let (status, text) =
        handle_rgba.decode(false, &options_region_4c, &mut region_rgba, stride_rgba);
    assert_eq!(status, J2K_OK, "{text}");
    assert_padding_intact(&region_rgba, stride_rgba, row_bytes_rgba, height_rgba, 0xCC);

    assert_rows_match(
        &full_rgba,
        &region_rgba,
        stride_rgba,
        row_bytes_rgba,
        height_rgba,
    );
}

/// The batch route keeps `j2k_decoder_decode_into`'s guarantee that a failed
/// decode leaves caller memory untouched, even when the header is valid and the
/// failure happens inside the tile data. In contrast,
/// `j2k_decoder_decode_full_into` streams directly into the caller's buffer on
/// eligible inputs and writes rows before hitting the corrupt tail.
#[test]
fn decode_into_leaves_the_output_untouched_when_the_payload_is_corrupt() {
    let (encoded, _) = support::gray_fixture(true);
    let sod = encoded
        .windows(2)
        .position(|marker| marker == [0xFF, 0x93])
        .expect("fixture has an SOD marker");
    let mut corrupt = encoded;
    let start = corrupt.len() - 18;
    assert!(start > sod + 2, "corruption must land in the tile data");
    corrupt[start..start + 16].fill(0xFF);

    let handle = Handle::open(&corrupt, None);
    let options = J2kCDecodeOptions::default();
    let layout = handle.layout(&options);
    let stride = usize::try_from(layout.row_bytes).unwrap();
    let mut output = vec![0xA5_u8; usize::try_from(layout.buffer_bytes).unwrap()];
    let (status, text) = handle.decode(false, &options, &mut output, stride);
    assert_eq!(status, J2K_DECODE_FAILED, "{text}");
    assert!(
        output.iter().all(|&byte| byte == 0xA5),
        "decode_into wrote to the output buffer before reporting the corrupt payload"
    );

    // Discriminating case: on a single-worker 512x512 16-bit HTJ2K gray
    // codestream, the core whole-image route streams rows directly into the
    // caller's buffer and writes partial output before failing on the corrupt
    // tail, whereas decode_into keeps the batch contract and leaves it untouched.
    let large_pixels = samples(512, 512, 1, 16);
    let mut large_corrupt = support::encode(&large_pixels, 512, 512, 1, 16, false, true);
    let large_sod = large_corrupt
        .windows(2)
        .position(|marker| marker == [0xFF, 0x93])
        .expect("512x512 fixture has an SOD marker");
    let large_start = large_corrupt.len() - 18;
    assert!(
        large_start > large_sod + 2,
        "corruption must land in the tile data"
    );
    large_corrupt[large_start..large_start + 16].fill(0xFF);

    let single_worker = J2kCOpenOptions {
        workers: 1,
        ..Default::default()
    };
    let large_handle = Handle::open(&large_corrupt, Some(&single_worker));
    let native_layout = large_handle.try_layout_native(&options).unwrap();
    let batch_layout = large_handle.layout(&options);
    assert_eq!(layout_key(&native_layout), layout_key(&batch_layout));
    let large_stride = usize::try_from(batch_layout.row_bytes).unwrap();
    let large_len = usize::try_from(batch_layout.buffer_bytes).unwrap();

    let mut core_output = vec![0xA5_u8; large_len];
    let (core_status, core_text) =
        large_handle.decode(true, &options, &mut core_output, large_stride);
    assert_eq!(core_status, J2K_DECODE_FAILED, "{core_text}");
    assert!(
        core_output.iter().any(|&byte| byte != 0xA5),
        "expected decode_full_into to stream partial rows before failing"
    );

    let mut batch_output = vec![0xA5_u8; large_len];
    let (batch_status, batch_text) =
        large_handle.decode(false, &options, &mut batch_output, large_stride);
    assert_eq!(batch_status, core_status, "{batch_text}");
    assert!(
        batch_output.iter().all(|&byte| byte == 0xA5),
        "decode_into wrote to the output buffer before reporting the corrupt payload"
    );
}
