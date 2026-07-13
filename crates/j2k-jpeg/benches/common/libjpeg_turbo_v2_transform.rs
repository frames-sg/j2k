// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ffi::{c_char, c_int, c_ulong, c_void, CStr};
use std::ptr::NonNull;

const TJXOP_NONE: c_int = 0;
const TJXOPT_CROP: c_int = 4;
const TJFLAGS_DEFAULT: c_int = 0;

type TjHandle = *mut c_void;
type TjCustomFilter = unsafe extern "C" fn(
    coeffs: *mut i16,
    array_region: TjRegion,
    plane_region: TjRegion,
    component_index: c_int,
    transform_index: c_int,
    transform: *mut TjTransform,
) -> c_int;

#[derive(Clone, Copy)]
#[repr(C)]
struct TjRegion {
    x: c_int,
    y: c_int,
    w: c_int,
    h: c_int,
}

#[repr(C)]
struct TjTransform {
    region: TjRegion,
    operation: c_int,
    options: c_int,
    data: *mut c_void,
    custom_filter: Option<TjCustomFilter>,
}

unsafe extern "C" {
    fn tjInitTransform() -> TjHandle;
    fn tjTransform(
        handle: TjHandle,
        jpeg_buf: *const u8,
        jpeg_size: c_ulong,
        count: c_int,
        destination_buffers: *mut *mut u8,
        destination_sizes: *mut c_ulong,
        transforms: *mut TjTransform,
        flags: c_int,
    ) -> c_int;
    fn tjDestroy(handle: TjHandle) -> c_int;
    fn tjFree(buffer: *mut u8);
    fn tjGetErrorStr2(handle: TjHandle) -> *mut c_char;
}

pub(super) struct Transformer {
    handle: NonNull<c_void>,
}

impl Transformer {
    pub(super) fn new() -> Result<Self, String> {
        // SAFETY: `tjInitTransform` takes no arguments and returns an owned handle.
        let handle = unsafe { tjInitTransform() };
        let Some(handle) = NonNull::new(handle) else {
            return Err("tjInitTransform returned null".to_string());
        };
        Ok(Self { handle })
    }

    pub(super) fn crop(&mut self, bytes: &[u8], region: CropRegion) -> Result<OwnedJpeg, String> {
        let jpeg_size = c_ulong::try_from(bytes.len()).map_err(|_| {
            format!(
                "JPEG transform input length {} does not fit into c_ulong",
                bytes.len()
            )
        })?;
        let mut destination = std::ptr::null_mut();
        let mut destination_size = 0;
        let mut transform = TjTransform {
            region: TjRegion {
                x: to_c_int(region.x, "crop x")?,
                y: to_c_int(region.y, "crop y")?,
                w: to_c_int(region.width, "crop width")?,
                h: to_c_int(region.height, "crop height")?,
            },
            operation: TJXOP_NONE,
            options: TJXOPT_CROP,
            data: std::ptr::null_mut(),
            custom_filter: None,
        };

        // SAFETY: The transformer handle is live; the source slice remains valid; output pointers
        // refer to writable storage; and the C-compatible transform structure contains no callback.
        let rc = unsafe {
            tjTransform(
                self.handle.as_ptr(),
                bytes.as_ptr(),
                jpeg_size,
                1,
                &raw mut destination,
                &raw mut destination_size,
                &raw mut transform,
                TJFLAGS_DEFAULT,
            )
        };
        if rc != 0 {
            if !destination.is_null() {
                // SAFETY: TurboJPEG allocated this destination during the failed transform.
                unsafe { tjFree(destination) };
            }
            return Err(error_string(self.handle));
        }
        let Some(destination) = NonNull::new(destination) else {
            return Err("tjTransform succeeded with a null destination".to_string());
        };
        let len = usize::try_from(destination_size).map_err(|_| {
            // SAFETY: TurboJPEG allocated this destination and ownership has not escaped.
            unsafe { tjFree(destination.as_ptr()) };
            format!("transformed JPEG size {destination_size} exceeds usize")
        })?;
        if len == 0 {
            // SAFETY: TurboJPEG allocated this destination and ownership has not escaped.
            unsafe { tjFree(destination.as_ptr()) };
            return Err("tjTransform succeeded with an empty destination".to_string());
        }
        Ok(OwnedJpeg {
            data: destination,
            len,
        })
    }
}

impl Drop for Transformer {
    fn drop(&mut self) {
        // SAFETY: This object owns the live transformer handle and destroys it exactly once.
        let _ = unsafe { tjDestroy(self.handle.as_ptr()) };
    }
}

pub(super) struct OwnedJpeg {
    data: NonNull<u8>,
    len: usize,
}

impl OwnedJpeg {
    pub(super) fn as_bytes(&self) -> &[u8] {
        // SAFETY: `data` owns a TurboJPEG allocation of exactly `len` bytes until `Drop`.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

impl Drop for OwnedJpeg {
    fn drop(&mut self) {
        // SAFETY: This object owns the TurboJPEG allocation and frees it exactly once.
        unsafe { tjFree(self.data.as_ptr()) };
    }
}

#[derive(Clone, Copy)]
pub(super) struct CropRegion {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

fn error_string(handle: NonNull<c_void>) -> String {
    // SAFETY: The handle remains live for the duration of the call.
    let ptr = unsafe { tjGetErrorStr2(handle.as_ptr()) };
    if ptr.is_null() {
        return "libjpeg-turbo transform error".to_string();
    }
    // SAFETY: TurboJPEG returns a NUL-terminated error string owned by the live handle.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn to_c_int(value: u32, label: &str) -> Result<c_int, String> {
    c_int::try_from(value).map_err(|_| format!("{label} {value} does not fit into c_int"))
}
