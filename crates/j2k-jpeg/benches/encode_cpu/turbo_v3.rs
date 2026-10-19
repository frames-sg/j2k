// SPDX-License-Identifier: MIT OR Apache-2.0

//! `TurboJPEG` version-3 compressor for the lossless and optimized-Huffman
//! comparisons, which the version-2 API cannot express.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr::NonNull;

use j2k_jpeg::JpegSubsampling;

const TJINIT_COMPRESS: c_int = 0;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPARAM_PRECISION: c_int = 7;
const TJPARAM_OPTIMIZE: c_int = 11;
const TJPARAM_LOSSLESS: c_int = 15;
const TJPARAM_LOSSLESSPSV: c_int = 16;
const TJPARAM_LOSSLESSPT: c_int = 17;
const TJPF_RGB: c_int = 0;
const TJPF_GRAY: c_int = 6;

unsafe extern "C" {
    fn tj3Init(init_type: c_int) -> *mut c_void;
    fn tj3Destroy(handle: *mut c_void);
    fn tj3Set(handle: *mut c_void, param: c_int, value: c_int) -> c_int;
    fn tj3GetErrorStr(handle: *mut c_void) -> *const c_char;
    fn tj3Free(buffer: *mut c_void);
    fn tj3Compress8(
        handle: *mut c_void,
        input: *const u8,
        width: c_int,
        pitch: c_int,
        height: c_int,
        pixel_format: c_int,
        output: *mut *mut u8,
        output_size: *mut usize,
    ) -> c_int;
    fn tj3Compress12(
        handle: *mut c_void,
        input: *const i16,
        width: c_int,
        pitch: c_int,
        height: c_int,
        pixel_format: c_int,
        output: *mut *mut u8,
        output_size: *mut usize,
    ) -> c_int;
    fn tj3Compress16(
        handle: *mut c_void,
        input: *const u16,
        width: c_int,
        pitch: c_int,
        height: c_int,
        pixel_format: c_int,
        output: *mut *mut u8,
        output_size: *mut usize,
    ) -> c_int;
}

/// What the compressor writes.
#[derive(Debug, Clone, Copy)]
pub(super) enum TurboV3Mode {
    /// Baseline with libjpeg's two-pass optimal Huffman tables.
    Optimized {
        quality: u8,
        subsampling: JpegSubsampling,
    },
    /// Lossless SOF3 with predictor 1 and no point transform.
    LosslessSv1 { precision: u8 },
}

pub(super) struct TurboV3Encoder {
    handle: NonNull<c_void>,
    width: c_int,
    height: c_int,
    pixel_format: c_int,
}

impl TurboV3Encoder {
    pub(super) fn new(width: u32, height: u32, gray: bool, mode: TurboV3Mode) -> Self {
        // SAFETY: creates a compressor owned by this value and freed in Drop.
        let handle =
            NonNull::new(unsafe { tj3Init(TJINIT_COMPRESS) }).expect("TurboJPEG 3 compressor");
        let encoder = Self {
            handle,
            width: width.try_into().expect("benchmark width fits c_int"),
            height: height.try_into().expect("benchmark height fits c_int"),
            pixel_format: if gray { TJPF_GRAY } else { TJPF_RGB },
        };
        match mode {
            TurboV3Mode::Optimized {
                quality,
                subsampling,
            } => {
                encoder.set(TJPARAM_QUALITY, c_int::from(quality));
                encoder.set(
                    TJPARAM_SUBSAMP,
                    match subsampling {
                        JpegSubsampling::Ybr444 => 0,
                        JpegSubsampling::Ybr422 => 1,
                        JpegSubsampling::Ybr420 => 2,
                        JpegSubsampling::Gray => 3,
                    },
                );
                encoder.set(TJPARAM_OPTIMIZE, 1);
            }
            TurboV3Mode::LosslessSv1 { precision } => {
                encoder.set(TJPARAM_LOSSLESS, 1);
                encoder.set(TJPARAM_LOSSLESSPSV, 1);
                encoder.set(TJPARAM_LOSSLESSPT, 0);
                encoder.set(TJPARAM_PRECISION, c_int::from(precision));
            }
        }
        encoder
    }

    fn set(&self, param: c_int, value: c_int) {
        // SAFETY: the live compressor accepts any parameter id; failures are reported.
        let status = unsafe { tj3Set(self.handle.as_ptr(), param, value) };
        assert_eq!(
            status,
            0,
            "TurboJPEG 3 tj3Set({param}, {value}): {}",
            self.error()
        );
    }

    fn error(&self) -> String {
        // SAFETY: TurboJPEG returns a NUL-terminated error string for the live handle.
        let message = unsafe { tj3GetErrorStr(self.handle.as_ptr()) };
        if message.is_null() {
            "unknown TurboJPEG error".into()
        } else {
            // SAFETY: checked non-null above; the string outlives this call.
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        }
    }

    fn samples(&self) -> usize {
        let channels = if self.pixel_format == TJPF_GRAY { 1 } else { 3 };
        usize::try_from(self.width).expect("benchmark width is nonnegative")
            * usize::try_from(self.height).expect("benchmark height is nonnegative")
            * channels
    }

    /// Compress 8-bit samples and return a copy of the JPEG bytes.
    pub(super) fn encode8(&mut self, input: &[u8]) -> Vec<u8> {
        assert_eq!(input.len(), self.samples());
        self.run(|handle, output, size, width, height, format| {
            // SAFETY: input holds width * height * channels samples; pitch 0
            // means contiguous rows.
            unsafe {
                tj3Compress8(
                    handle,
                    input.as_ptr(),
                    width,
                    0,
                    height,
                    format,
                    output,
                    size,
                )
            }
        })
    }

    /// Compress 9- to 12-bit samples and return a copy of the JPEG bytes.
    pub(super) fn encode12(&mut self, input: &[i16]) -> Vec<u8> {
        assert_eq!(input.len(), self.samples());
        self.run(|handle, output, size, width, height, format| {
            // SAFETY: as encode8, with 16-bit storage per sample.
            unsafe {
                tj3Compress12(
                    handle,
                    input.as_ptr(),
                    width,
                    0,
                    height,
                    format,
                    output,
                    size,
                )
            }
        })
    }

    /// Compress 13- to 16-bit samples and return a copy of the JPEG bytes.
    pub(super) fn encode16(&mut self, input: &[u16]) -> Vec<u8> {
        assert_eq!(input.len(), self.samples());
        self.run(|handle, output, size, width, height, format| {
            // SAFETY: as encode8, with 16-bit storage per sample.
            unsafe {
                tj3Compress16(
                    handle,
                    input.as_ptr(),
                    width,
                    0,
                    height,
                    format,
                    output,
                    size,
                )
            }
        })
    }

    fn run(
        &mut self,
        compress: impl FnOnce(*mut c_void, *mut *mut u8, *mut usize, c_int, c_int, c_int) -> c_int,
    ) -> Vec<u8> {
        let mut output = std::ptr::null_mut();
        let mut output_size = 0usize;
        let status = compress(
            self.handle.as_ptr(),
            &raw mut output,
            &raw mut output_size,
            self.width,
            self.height,
            self.pixel_format,
        );
        let output = OutputBuffer(output);
        assert_eq!(status, 0, "TurboJPEG 3 compress: {}", self.error());
        assert!(!output.0.is_null() && output_size != 0);
        // SAFETY: a successful compress returned this allocated JPEG span.
        // The copy mirrors the owned Vec the j2k encoder returns.
        unsafe { std::slice::from_raw_parts(output.0, output_size) }.to_vec()
    }
}

impl Drop for TurboV3Encoder {
    fn drop(&mut self) {
        // SAFETY: this value exclusively owns the live compressor handle.
        unsafe { tj3Destroy(self.handle.as_ptr()) };
    }
}

struct OutputBuffer(*mut u8);

impl Drop for OutputBuffer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the buffer was allocated by tj3Compress* and is freed once.
            unsafe { tj3Free(self.0.cast()) };
        }
    }
}
