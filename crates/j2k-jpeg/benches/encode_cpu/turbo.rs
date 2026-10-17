// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned-output `TurboJPEG` comparison using its stable version-2 API.

use std::ffi::{c_int, c_ulong, c_void, CStr};
use std::ptr::NonNull;

use j2k_jpeg::JpegSubsampling;

unsafe extern "C" {
    fn tjInitCompress() -> *mut c_void;
    fn tjDestroy(handle: *mut c_void) -> c_int;
    fn tjGetErrorStr2(handle: *mut c_void) -> *const std::ffi::c_char;
    fn tjFree(buffer: *mut u8);
    fn tjCompress2(
        handle: *mut c_void,
        input: *const u8,
        width: c_int,
        pitch: c_int,
        height: c_int,
        pixel_format: c_int,
        output: *mut *mut u8,
        output_size: *mut c_ulong,
        sampling: c_int,
        quality: c_int,
        flags: c_int,
    ) -> c_int;
}

pub(super) struct TurboEncoder {
    handle: NonNull<c_void>,
    width: c_int,
    height: c_int,
    pitch: c_int,
    input_len: usize,
    pixel_format: c_int,
    sampling: c_int,
    quality: c_int,
}

impl TurboEncoder {
    pub(super) fn new(
        width: u32,
        height: u32,
        gray: bool,
        sampling: JpegSubsampling,
        quality: u8,
    ) -> Self {
        // SAFETY: creates a compressor owned by this value and freed in Drop.
        let handle = NonNull::new(unsafe { tjInitCompress() }).expect("TurboJPEG compressor");
        let channels = if gray { 1 } else { 3 };
        let pitch = width.checked_mul(channels).expect("benchmark row bytes");
        Self {
            handle,
            width: width.try_into().expect("benchmark width fits c_int"),
            height: height.try_into().expect("benchmark height fits c_int"),
            pitch: pitch.try_into().expect("benchmark pitch fits c_int"),
            input_len: (pitch as usize)
                .checked_mul(height as usize)
                .expect("benchmark input bytes"),
            pixel_format: if gray { 6 } else { 0 },
            sampling: match sampling {
                JpegSubsampling::Ybr444 => 0,
                JpegSubsampling::Ybr422 => 1,
                JpegSubsampling::Ybr420 => 2,
                JpegSubsampling::Gray => 3,
            },
            quality: c_int::from(quality),
        }
    }

    pub(super) fn encode(&mut self, input: &[u8]) -> usize {
        assert_eq!(input.len(), self.input_len);
        let mut output = std::ptr::null_mut();
        let mut output_size = 0;
        // SAFETY: input contains the validated full image. The live compressor
        // allocates an output owned by OutputBuffer and freed before returning.
        // TJFLAG_ACCURATEDCT (4096) explicitly selects the accurate integer DCT.
        let status = unsafe {
            tjCompress2(
                self.handle.as_ptr(),
                input.as_ptr(),
                self.width,
                self.pitch,
                self.height,
                self.pixel_format,
                &raw mut output,
                &raw mut output_size,
                self.sampling,
                self.quality,
                4096,
            )
        };
        let output = OutputBuffer(output);
        if status != 0 {
            // SAFETY: TurboJPEG returns an error string for the live handle.
            let message = unsafe { tjGetErrorStr2(self.handle.as_ptr()) };
            let message = if message.is_null() {
                "unknown TurboJPEG error".into()
            } else {
                // SAFETY: TurboJPEG's error string is NUL-terminated.
                unsafe { CStr::from_ptr(message) }.to_string_lossy()
            };
            panic!("TurboJPEG encode: {message}");
        }
        let length = usize::try_from(output_size).expect("JPEG output length fits usize");
        assert!(!output.0.is_null() && length != 0);
        // SAFETY: successful tjCompress2 returned this allocated JPEG span.
        std::hint::black_box(unsafe { std::slice::from_raw_parts(output.0, length) });
        length
    }
}

impl Drop for TurboEncoder {
    fn drop(&mut self) {
        // SAFETY: this value exclusively owns the live compressor handle.
        unsafe { tjDestroy(self.handle.as_ptr()) };
    }
}

struct OutputBuffer(*mut u8);

impl Drop for OutputBuffer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the buffer was allocated by tjCompress2 and is freed once.
            unsafe { tjFree(self.0) };
        }
    }
}
