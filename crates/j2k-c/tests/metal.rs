// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal decode through the exported C ABI.

#![cfg(all(feature = "metal", target_os = "macos"))]

mod support;

use j2k_c::*;
use std::ptr;

struct Handle(*mut J2kCDecoder);

impl Handle {
    fn open(bytes: &[u8]) -> Self {
        let mut handle = ptr::null_mut();
        let mut error = J2kCError::default();
        // SAFETY: The input is readable for its length and the output slot is valid.
        let status = unsafe {
            j2k_decoder_open(
                bytes.as_ptr(),
                bytes.len(),
                ptr::null(),
                &raw mut handle,
                &raw mut error,
            )
        };
        assert_eq!(status, J2K_OK, "{}", message(&error));
        Self(handle)
    }

    fn feed(&self, bytes: &[u8]) {
        let mut error = J2kCError::default();
        // SAFETY: The input is readable for its length; handle use is serial.
        let status =
            unsafe { j2k_decoder_feed(self.0, bytes.as_ptr(), bytes.len(), &raw mut error) };
        assert_eq!(status, J2K_OK, "{}", message(&error));
    }

    fn layout(&self, options: &J2kCDecodeOptions) -> J2kCOutputInfo {
        let mut layout = J2kCOutputInfo::default();
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; the output slot is valid and disjoint.
        let status = unsafe {
            j2k_decoder_output_info_native(self.0, options, &raw mut layout, &raw mut error)
        };
        assert_eq!(status, J2K_OK, "{}", message(&error));
        layout
    }

    fn metal_init(&self) -> u32 {
        let mut error = J2kCError::default();
        // SAFETY: Handle is live and exclusively used.
        unsafe { j2k_decoder_metal_init(self.0, &raw mut error) }
    }

    fn metal_decode(
        &self,
        options: &J2kCDecodeOptions,
        output: &mut [u8],
        stride: usize,
    ) -> (u32, String) {
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; the buffer covers the declared span.
        let status = unsafe {
            j2k_decoder_metal_decode_into(
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
    }
}

fn message(error: &J2kCError) -> String {
    // SAFETY: The library NUL-terminates every diagnostic it writes.
    unsafe { std::ffi::CStr::from_ptr(error.message.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn tight_rows(output: &[u8], stride: usize, row_bytes: usize, height: usize) -> Vec<u8> {
    output
        .chunks(stride)
        .take(height)
        .flat_map(|line| line[..row_bytes].iter().copied())
        .collect()
}

#[test]
fn metal_decode_matches_the_cpu_samples_and_never_falls_back() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let (encoded, pixels) = support::gray_fixture(true);
    let handle = Handle::open(&encoded);
    let options = J2kCDecodeOptions::default();
    let layout = handle.layout(&options);
    let row_bytes = usize::try_from(layout.row_bytes).unwrap();
    let height = usize::try_from(layout.height).unwrap();
    let stride = row_bytes + 5;
    let length = stride * (height - 1) + row_bytes;

    let mut before_init = vec![0xA5; length];
    let (status, text) = handle.metal_decode(&options, &mut before_init, stride);
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    assert!(before_init.iter().all(|&value| value == 0xA5));

    match handle.metal_init() {
        J2K_OK => {}
        J2K_UNSUPPORTED => {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        }
        status => panic!("j2k_decoder_metal_init returned {status}"),
    }
    assert_eq!(handle.metal_init(), J2K_OK, "metal_init is not idempotent");

    // Reuse the session for a second decode before replacing the codestream.
    for _ in 0..2 {
        let mut output = vec![0xA5; length];
        let (status, text) = handle.metal_decode(&options, &mut output, stride);
        assert_eq!(status, J2K_OK, "{text}");
        assert_eq!(tight_rows(&output, stride, row_bytes, height), pixels);
        for line in output.chunks(stride) {
            assert!(line[row_bytes..].iter().all(|&value| value == 0xA5));
        }
    }

    let mut short = vec![0xA5; length - 1];
    let (status, text) = handle.metal_decode(&options, &mut short, stride);
    assert_eq!(status, J2K_BUFFER_TOO_SMALL, "{text}");
    assert!(short.iter().all(|&value| value == 0xA5));

    let replacement_pixels: Vec<u8> = (0..24 * 16 * 3)
        .map(|index| u8::try_from((index * 53 + 17) % 256).unwrap())
        .collect();
    let replacement = support::encode(&replacement_pixels, 24, 16, 3, 8, false, false);
    handle.feed(&replacement);
    let layout = handle.layout(&options);
    let row_bytes = usize::try_from(layout.row_bytes).unwrap();
    let height = usize::try_from(layout.height).unwrap();
    let stride = row_bytes + 7;
    let mut output = vec![0xA5; stride * (height - 1) + row_bytes];
    let (status, text) = handle.metal_decode(&options, &mut output, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_eq!(
        tight_rows(&output, stride, row_bytes, height),
        replacement_pixels
    );
}
