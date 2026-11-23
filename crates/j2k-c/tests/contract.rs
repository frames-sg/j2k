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
        options: &J2kCDecodeOptions,
        output: &mut [u8],
        stride: usize,
    ) -> (u32, String) {
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; the buffer covers the declared span; the call is exclusive.
        let status = unsafe {
            j2k_decoder_decode_into(
                self.0,
                options,
                output.as_mut_ptr(),
                output.len(),
                stride,
                &raw mut error,
            )
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
    let (status, text) = handle.decode(&options, &mut output, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_eq!(output, second_pixels);

    let syntax = handle.transfer_syntax();
    for bad in [
        &b""[..],
        &b"not a JPEG 2000 codestream"[..],
        &b"\x00\x00\x00\x0cjP  \r\n\x87\n\x00"[..],
    ] {
        let (status, text) = handle.try_feed(bad).unwrap_err();
        assert_ne!(status, J2K_OK);
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
        let (status, text) = handle.decode(&options, &mut output, stride);
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
