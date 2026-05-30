// SPDX-License-Identifier: Apache-2.0

//! NVIDIA GPU codec baseline (nvJPEG + nvJPEG2000) for benchmarking signinum's
//! coefficient-domain JPEG → HTJ2K transcode against the conventional
//! decode-to-pixels-then-encode pipeline.
//!
//! The intricate codec orchestration lives in `cuda/nv_baseline.cu`; this module
//! is a thin FFI plus safe wrappers. The FFI is only present when `build.rs`
//! compiled and linked the C++ helper (cfg `nvbaseline_built`) — i.e. on a host
//! with `nvcc`, `libnvjpeg`, and a separately-installed `libnvjpeg2k`. Elsewhere
//! the wrappers report "unavailable" so the workspace still builds.

#[cfg(nvbaseline_built)]
mod ffi {
    use std::os::raw::{c_int, c_uchar};

    extern "C" {
        pub fn nvb_available() -> c_int;

        pub fn nvb_decode_jpeg_rgb(
            jpeg: *const c_uchar,
            jpeg_len: usize,
            out_rgb: *mut c_uchar,
            out_cap: usize,
            width: *mut c_int,
            height: *mut c_int,
        ) -> c_int;

        pub fn nvb_transcode_jpeg_to_htj2k(
            jpeg: *const c_uchar,
            jpeg_len: usize,
            out: *mut c_uchar,
            out_cap: usize,
            out_len: *mut usize,
            decode_ms: *mut f64,
            encode_ms: *mut f64,
            width: *mut c_int,
            height: *mut c_int,
            num_components: *mut c_int,
        ) -> c_int;
    }
}

/// One NVIDIA transcode result: the HTJ2K codestream plus GPU stage timings.
#[derive(Debug, Clone)]
pub struct NvTranscodeResult {
    /// The produced HTJ2K codestream.
    pub codestream: Vec<u8>,
    /// nvJPEG decode (JPEG → planar RGB) GPU time, milliseconds.
    pub decode_ms: f64,
    /// nvJPEG2000 HT encode (RGB → HTJ2K) GPU time, milliseconds.
    pub encode_ms: f64,
    /// Decoded image width.
    pub width: u32,
    /// Decoded image height.
    pub height: u32,
    /// Component count (3 for the RGB pipeline).
    pub num_components: u32,
}

/// Whether the NVIDIA baseline is compiled in and the codec handles initialize.
#[must_use]
pub fn nvidia_baseline_available() -> bool {
    #[cfg(nvbaseline_built)]
    {
        // SAFETY: `nvb_available` takes no arguments and only creates/destroys
        // codec handles to confirm the libraries load.
        unsafe { ffi::nvb_available() == 1 }
    }
    #[cfg(not(nvbaseline_built))]
    {
        false
    }
}

/// Why the NVIDIA baseline could not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvBaselineError {
    /// The C++ baseline was not compiled into this build (no nvcc / libraries).
    NotBuilt,
    /// The C++ pipeline returned a non-zero stage code.
    Stage(i32),
}

/// Reference-decode a JPEG to interleaved RGB (untimed), for PSNR comparison.
pub fn nvidia_decode_jpeg_rgb(jpeg: &[u8]) -> Result<(Vec<u8>, u32, u32), NvBaselineError> {
    #[cfg(not(nvbaseline_built))]
    {
        let _ = jpeg;
        Err(NvBaselineError::NotBuilt)
    }
    #[cfg(nvbaseline_built)]
    {
        let mut width = 0i32;
        let mut height = 0i32;
        // Two-pass: read dimensions cheaply from the header first.
        let info = nvjpeg_image_dimensions(jpeg)?;
        let mut out = vec![0u8; (info.0 as usize) * (info.1 as usize) * 3];
        // SAFETY: `out` is sized to width*height*3 and `jpeg` is a valid slice.
        let rc = unsafe {
            ffi::nvb_decode_jpeg_rgb(
                jpeg.as_ptr(),
                jpeg.len(),
                out.as_mut_ptr(),
                out.len(),
                &mut width,
                &mut height,
            )
        };
        if rc != 0 {
            return Err(NvBaselineError::Stage(rc));
        }
        Ok((out, width as u32, height as u32))
    }
}

/// Transcode a JPEG to HTJ2K via nvJPEG decode + nvJPEG2000 HT encode.
pub fn nvidia_transcode_jpeg_to_htj2k(jpeg: &[u8]) -> Result<NvTranscodeResult, NvBaselineError> {
    #[cfg(not(nvbaseline_built))]
    {
        let _ = jpeg;
        Err(NvBaselineError::NotBuilt)
    }
    #[cfg(nvbaseline_built)]
    {
        // HTJ2K of an 8-bit RGB image is well under the raw pixel size; allocate
        // raw RGB size plus header slack and grow once if the codec disagrees.
        let dims = nvjpeg_image_dimensions(jpeg)?;
        let mut capacity = (dims.0 as usize) * (dims.1 as usize) * 3 + (1 << 16);
        loop {
            let mut out = vec![0u8; capacity];
            let mut out_len = 0usize;
            let mut decode_ms = 0f64;
            let mut encode_ms = 0f64;
            let mut width = 0i32;
            let mut height = 0i32;
            let mut num_components = 0i32;
            // SAFETY: all pointers reference live, correctly-sized allocations.
            let rc = unsafe {
                ffi::nvb_transcode_jpeg_to_htj2k(
                    jpeg.as_ptr(),
                    jpeg.len(),
                    out.as_mut_ptr(),
                    out.len(),
                    &mut out_len,
                    &mut decode_ms,
                    &mut encode_ms,
                    &mut width,
                    &mut height,
                    &mut num_components,
                )
            };
            // rc 212 == output buffer too small; double and retry once more.
            if rc == 212 && capacity < (1 << 30) {
                capacity *= 2;
                continue;
            }
            if rc != 0 {
                return Err(NvBaselineError::Stage(rc));
            }
            out.truncate(out_len);
            return Ok(NvTranscodeResult {
                codestream: out,
                decode_ms,
                encode_ms,
                width: width as u32,
                height: height as u32,
                num_components: num_components as u32,
            });
        }
    }
}

/// Read (width, height) from a JPEG header via a cheap reference decode probe.
#[cfg(nvbaseline_built)]
fn nvjpeg_image_dimensions(jpeg: &[u8]) -> Result<(u32, u32), NvBaselineError> {
    let mut width = 0i32;
    let mut height = 0i32;
    // A zero-capacity decode returns the dimensions via the out params and a
    // benign "too small" code (120); we only consume the dimensions here.
    // SAFETY: out_rgb is null with zero capacity; the helper fills width/height
    // from the header before checking capacity.
    let rc = unsafe {
        ffi::nvb_decode_jpeg_rgb(
            jpeg.as_ptr(),
            jpeg.len(),
            std::ptr::null_mut(),
            0,
            &mut width,
            &mut height,
        )
    };
    // 0 (unexpected for zero cap) or 120 (capacity too small) both yield dims.
    if (rc == 0 || rc == 120) && width > 0 && height > 0 {
        Ok((width as u32, height as u32))
    } else {
        Err(NvBaselineError::Stage(rc))
    }
}

/// Peak-signal-to-noise ratio (dB) between two equal-length 8-bit buffers.
///
/// Returns `f64::INFINITY` for identical inputs and `None` on a length mismatch.
#[must_use]
pub fn psnr_u8(a: &[u8], b: &[u8]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut sum_sq = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = f64::from(x) - f64::from(y);
        sum_sq += diff * diff;
    }
    let mse = sum_sq / a.len() as f64;
    if mse == 0.0 {
        return Some(f64::INFINITY);
    }
    Some(10.0 * (255.0f64 * 255.0 / mse).log10())
}
