// SPDX-License-Identifier: MIT OR Apache-2.0

//! The always-exported Metal entry points in builds without the `metal` feature.

#![cfg(not(feature = "metal"))]

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

    fn layout(&self, options: &J2kCDecodeOptions) -> J2kCOutputInfo {
        let mut layout = J2kCOutputInfo::default();
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; the output slot is valid and disjoint.
        let status =
            unsafe { j2k_decoder_output_info(self.0, options, &raw mut layout, &raw mut error) };
        assert_eq!(status, J2K_OK, "{}", message(&error));
        layout
    }

    fn metal_init(&self) -> (u32, String) {
        let mut error = J2kCError::default();
        // SAFETY: Handle is live and exclusively used.
        let status = unsafe { j2k_decoder_metal_init(self.0, &raw mut error) };
        (status, message(&error))
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

    fn decode(
        &self,
        options: &J2kCDecodeOptions,
        output: &mut [u8],
        stride: usize,
    ) -> (u32, String) {
        let mut error = J2kCError::default();
        // SAFETY: Handle is live; the buffer covers the declared span.
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
fn metal_entry_points_report_unsupported_without_the_feature() {
    let (encoded, pixels) = support::gray_fixture(false);
    let handle = Handle::open(&encoded);
    let options = J2kCDecodeOptions::default();

    let (status, text) = handle.metal_init();
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    assert!(text.contains("without the Metal feature"), "{text}");

    let layout = handle.layout(&options);
    let row_bytes = usize::try_from(layout.row_bytes).unwrap();
    let height = usize::try_from(layout.height).unwrap();
    let stride = row_bytes + 3;
    let length = stride * (height - 1) + row_bytes;

    let mut output = vec![0xA5; length];
    let (status, text) = handle.metal_decode(&options, &mut output, stride);
    assert_eq!(status, J2K_UNSUPPORTED, "{text}");
    assert!(text.contains("without the Metal feature"), "{text}");
    assert!(output.iter().all(|&value| value == 0xA5));

    // A failed Metal initialization must leave the decoder usable on the CPU.
    let mut cpu = vec![0xA5; length];
    let (status, text) = handle.decode(&options, &mut cpu, stride);
    assert_eq!(status, J2K_OK, "{text}");
    assert_eq!(tight_rows(&cpu, stride, row_bytes, height), pixels);
    for line in cpu.chunks(stride) {
        assert!(line[row_bytes..].iter().all(|&value| value == 0xA5));
    }
}
