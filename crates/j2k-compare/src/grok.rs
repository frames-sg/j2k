// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(have_grok)]
use std::{ffi::c_void, ptr, sync::Once};

use crate::ExternalDecodeRequest;
#[cfg(any(have_grok, test))]
use crate::MAX_EXTERNAL_OUTPUT_BYTES;

pub fn is_available() -> bool {
    cfg!(have_grok)
}

pub fn version() -> &'static str {
    option_env!("J2K_GROK_VERSION").unwrap_or("unavailable")
}

pub fn library_path() -> &'static str {
    option_env!("J2K_GROK_LIB_DIR").unwrap_or("unavailable")
}

crate::external_decode_wrappers!(decode);

#[cfg(have_grok)]
static GROK_INIT: Once = Once::new();

#[cfg(have_grok)]
struct GrokOutput(*mut u8);

#[cfg(have_grok)]
impl Drop for GrokOutput {
    #[expect(
        unsafe_code,
        reason = "the guard frees the Grok shim's optional output allocation exactly once"
    )]
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the pointer is either null or the allocation returned by
            // `j2k_grok_decode_u8`, and this guard is its single owner.
            unsafe { j2k_grok_free(self.0.cast()) };
        }
    }
}

#[cfg(have_grok)]
#[expect(
    unsafe_code,
    reason = "Rust Once serializes the Grok process-global runtime initializer"
)]
fn initialize() {
    GROK_INIT.call_once(|| {
        // SAFETY: `Once` guarantees the process-global Grok initializer runs once,
        // before any shim decode is allowed to proceed.
        unsafe { j2k_grok_initialize() };
    });
}

#[cfg_attr(
    have_grok,
    expect(
        unsafe_code,
        reason = "Grok decode uses the optional checked C shim and frees its output exactly once"
    )
)]
fn decode(bytes: &[u8], request: ExternalDecodeRequest) -> Result<Vec<u8>, String> {
    #[cfg(have_grok)]
    {
        initialize();
        let mut out = ptr::null_mut();
        let mut out_len = 0_usize;
        let mut out_width = 0_u32;
        let mut out_height = 0_u32;
        let channels = request.color.channels();
        let reduce = checked_reduce(request.reduce)?;
        let (has_region, x0, y0, x1, y1) = match request.region {
            Some(roi) => {
                let [x0, y0, x1, y1] = checked_region_bounds(roi)?;
                (1, x0, y0, x1, y1)
            }
            None => (0, 0, 0, 0, 0),
        };
        // SAFETY: the source slice is live for the call and all output pointers
        // refer to initialized writable locals governed by the C shim's ABI.
        let ok = unsafe {
            j2k_grok_decode_u8(
                bytes.as_ptr(),
                bytes.len(),
                reduce,
                has_region,
                x0,
                y0,
                x1,
                y1,
                channels,
                &raw mut out,
                &raw mut out_len,
                &raw mut out_width,
                &raw mut out_height,
            )
        };
        let output = GrokOutput(out);
        if ok == 0 || output.0.is_null() {
            return Err("grok: decode failed".to_string());
        }
        let expected = checked_output_len(out_width, out_height, channels)?;
        if out_len != expected {
            return Err(format!(
                "grok: unexpected output length {out_len} != {expected}"
            ));
        }
        // SAFETY: the non-null shim allocation has been independently bounded and
        // its reported length exactly matches the checked dimensions and channels.
        let packed = unsafe { std::slice::from_raw_parts(output.0, expected) }.to_vec();
        Ok(packed)
    }

    #[cfg(not(have_grok))]
    {
        let _ = (bytes, request);
        Err("grok: local library not available".to_string())
    }
}

#[cfg(any(have_grok, test))]
fn checked_reduce(reduce: Option<u32>) -> Result<u8, String> {
    u8::try_from(reduce.unwrap_or(0))
        .map_err(|_| "grok: reduction factor exceeds Grok u8 range".to_string())
}

#[cfg(any(have_grok, test))]
fn checked_region_bounds(roi: j2k_core::Rect) -> Result<[u32; 4], String> {
    let x1 = roi
        .x
        .checked_add(roi.w)
        .ok_or_else(|| "grok: decode area x coordinate overflow".to_string())?;
    let y1 = roi
        .y
        .checked_add(roi.h)
        .ok_or_else(|| "grok: decode area y coordinate overflow".to_string())?;
    Ok([roi.x, roi.y, x1, y1])
}

#[cfg(any(have_grok, test))]
fn checked_output_len(width: u32, height: u32, channels: u32) -> Result<usize, String> {
    if !matches!(channels, 1 | 3) {
        return Err(format!(
            "grok: unsupported channel count {channels}, expected 1 or 3"
        ));
    }
    if width == 0 || height == 0 {
        return Err("grok: image has zero-sized output".to_string());
    }
    let width =
        usize::try_from(width).map_err(|_| "grok: width exceeds platform usize".to_string())?;
    let height =
        usize::try_from(height).map_err(|_| "grok: height exceeds platform usize".to_string())?;
    let channels = usize::try_from(channels)
        .map_err(|_| "grok: channel count exceeds platform usize".to_string())?;
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| "grok: output pixel count overflow".to_string())?;
    let len = pixels
        .checked_mul(channels)
        .ok_or_else(|| "grok: output byte count overflow".to_string())?;
    if len > MAX_EXTERNAL_OUTPUT_BYTES {
        return Err(format!(
            "grok: output exceeds {MAX_EXTERNAL_OUTPUT_BYTES} byte cap"
        ));
    }
    Ok(len)
}

#[cfg(have_grok)]
#[expect(
    unsafe_code,
    reason = "these declarations are the optional Grok C shim's complete ABI"
)]
unsafe extern "C" {
    fn j2k_grok_initialize();
    fn j2k_grok_decode_u8(
        bytes: *const u8,
        len: usize,
        reduce: u8,
        has_region: i32,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        channels: u32,
        out_data: *mut *mut u8,
        out_len: *mut usize,
        out_width: *mut u32,
        out_height: *mut u32,
    ) -> i32;
    fn j2k_grok_free(ptr: *mut c_void);
}

#[cfg(test)]
mod tests {
    use j2k_core::Rect;

    use super::{checked_output_len, checked_reduce, checked_region_bounds};
    use crate::MAX_EXTERNAL_OUTPUT_BYTES;

    #[test]
    fn region_bounds_reject_coordinate_overflow() {
        assert!(checked_region_bounds(Rect {
            x: u32::MAX,
            y: 0,
            w: 1,
            h: 1,
        })
        .is_err());
        assert_eq!(
            checked_region_bounds(Rect {
                x: 1,
                y: 2,
                w: 3,
                h: 4,
            })
            .expect("bounded Grok decode area"),
            [1, 2, 4, 6]
        );
    }

    #[test]
    fn output_len_is_bounded_before_slice_construction() {
        assert!(checked_output_len(0, 1, 1).is_err());
        assert!(checked_output_len(1, 1, 2).is_err());
        assert!(checked_output_len(u32::MAX, u32::MAX, 3).is_err());
        let over_cap =
            u32::try_from(MAX_EXTERNAL_OUTPUT_BYTES + 1).expect("the shared output cap fits u32");
        assert!(checked_output_len(over_cap, 1, 1).is_err());
        assert_eq!(checked_output_len(2, 3, 3), Ok(18));
    }

    #[test]
    fn reduce_rejects_values_that_the_c_abi_cannot_represent() {
        assert_eq!(checked_reduce(None), Ok(0));
        assert_eq!(checked_reduce(Some(u32::from(u8::MAX))), Ok(u8::MAX));
        assert!(checked_reduce(Some(u32::from(u8::MAX) + 1)).is_err());
    }

    #[cfg(have_grok)]
    #[test]
    fn initialization_is_serialized_across_threads() {
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(super::initialize);
            }
        });
    }
}
