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

    #[repr(C)]
    pub struct NvbSession {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn nvb_available() -> c_int;

        pub fn nvb_session_create(out: *mut *mut NvbSession) -> c_int;

        pub fn nvb_session_destroy(session: *mut NvbSession);

        pub fn nvb_session_transcode_jpeg_to_htj2k(
            session: *mut NvbSession,
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

        pub fn nvb_session_decode_j2k_interleaved(
            session: *mut NvbSession,
            j2k: *const c_uchar,
            j2k_len: usize,
            requested_format: c_int,
            out: *mut c_uchar,
            out_cap: usize,
            out_len: *mut usize,
            decode_ms: *mut f64,
            width: *mut c_int,
            height: *mut c_int,
            num_components: *mut c_int,
            bit_depth: *mut c_int,
            bytes_per_sample: *mut c_int,
        ) -> c_int;

        pub fn nvb_decode_jpeg_rgb(
            jpeg: *const c_uchar,
            jpeg_len: usize,
            out_rgb: *mut c_uchar,
            out_cap: usize,
            width: *mut c_int,
            height: *mut c_int,
        ) -> c_int;

        pub fn nvb_session_decode_jpeg_rgb_interleaved_timed(
            session: *mut NvbSession,
            jpeg: *const c_uchar,
            jpeg_len: usize,
            decode_ms: *mut f64,
            width: *mut c_int,
            height: *mut c_int,
        ) -> c_int;
    }
}

/// Output pixel layout requested from direct nvJPEG2000 decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvJ2kDecodeFormat {
    /// Interleaved 8-bit RGB output.
    Rgb8,
    /// 8-bit grayscale output.
    Gray8,
}

impl NvJ2kDecodeFormat {
    #[cfg(nvbaseline_built)]
    const fn ffi_code(self) -> i32 {
        match self {
            Self::Rgb8 => 1,
            Self::Gray8 => 2,
        }
    }

    #[cfg(nvbaseline_built)]
    const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 => 3,
            Self::Gray8 => 1,
        }
    }
}

/// Direct nvJPEG2000 decode result.
#[derive(Debug, Clone)]
pub struct NvJ2kDecodeResult {
    /// Interleaved decoded host pixels in the requested format.
    pub pixels: Vec<u8>,
    /// nvJPEG2000 GPU decode time, milliseconds.
    pub decode_ms: f64,
    /// Decoded image width.
    pub width: u32,
    /// Decoded image height.
    pub height: u32,
    /// Component count in `pixels`.
    pub num_components: u32,
    /// Source codestream component bit depth.
    pub bit_depth: u32,
    /// Output bytes per sample.
    pub bytes_per_sample: u32,
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

/// One reused-session NVIDIA JPEG decode timing.
#[derive(Debug, Clone, Copy)]
pub struct NvJpegDecodeTiming {
    /// nvJPEG device-resident RGBI decode time, milliseconds.
    pub decode_ms: f64,
    /// Decoded image width.
    pub width: u32,
    /// Decoded image height.
    pub height: u32,
}

/// Reusable NVIDIA baseline session.
///
/// This keeps nvJPEG/nvJPEG2000 handles, CUDA stream/events, encode state, and
/// reusable device RGB planes alive across tile transcodes.
pub struct NvBaselineSession {
    #[cfg(nvbaseline_built)]
    raw: std::ptr::NonNull<ffi::NvbSession>,
}

impl NvBaselineSession {
    /// Create a reusable nvJPEG + nvJPEG2000 transcode session.
    pub fn new() -> Result<Self, NvBaselineError> {
        #[cfg(not(nvbaseline_built))]
        {
            Err(NvBaselineError::NotBuilt)
        }
        #[cfg(nvbaseline_built)]
        {
            let mut raw = std::ptr::null_mut();
            // SAFETY: `raw` is a valid out pointer. On success the C side
            // returns an owned session pointer that `Drop` releases.
            let rc = unsafe { ffi::nvb_session_create(std::ptr::addr_of_mut!(raw)) };
            if rc != 0 {
                return Err(NvBaselineError::Stage(rc));
            }
            let raw = std::ptr::NonNull::new(raw).ok_or(NvBaselineError::Stage(900))?;
            Ok(Self { raw })
        }
    }

    /// Transcode one JPEG to HTJ2K using reused session resources.
    pub fn transcode_jpeg_to_htj2k(
        &mut self,
        jpeg: &[u8],
    ) -> Result<NvTranscodeResult, NvBaselineError> {
        #[cfg(not(nvbaseline_built))]
        {
            let _ = jpeg;
            Err(NvBaselineError::NotBuilt)
        }
        #[cfg(nvbaseline_built)]
        {
            nvidia_transcode_with_session(self.raw.as_ptr(), jpeg)
        }
    }

    /// Decode one JPEG to device-resident interleaved RGB and return the nvJPEG event time.
    pub fn decode_jpeg_rgb_interleaved_timed(
        &mut self,
        jpeg: &[u8],
    ) -> Result<NvJpegDecodeTiming, NvBaselineError> {
        #[cfg(not(nvbaseline_built))]
        {
            let _ = jpeg;
            Err(NvBaselineError::NotBuilt)
        }
        #[cfg(nvbaseline_built)]
        {
            let mut decode_ms = 0f64;
            let mut width = 0i32;
            let mut height = 0i32;
            // SAFETY: `self.raw` owns a live NVIDIA baseline session and
            // output pointers refer to initialized stack locals.
            let rc = unsafe {
                ffi::nvb_session_decode_jpeg_rgb_interleaved_timed(
                    self.raw.as_ptr(),
                    jpeg.as_ptr(),
                    jpeg.len(),
                    std::ptr::addr_of_mut!(decode_ms),
                    std::ptr::addr_of_mut!(width),
                    std::ptr::addr_of_mut!(height),
                )
            };
            if rc != 0 {
                return Err(NvBaselineError::Stage(rc));
            }
            Ok(NvJpegDecodeTiming {
                decode_ms,
                width: width as u32,
                height: height as u32,
            })
        }
    }

    /// Decode one JPEG 2000 / HTJ2K codestream to interleaved host pixels.
    pub fn decode_j2k_interleaved(
        &mut self,
        codestream: &[u8],
        format: NvJ2kDecodeFormat,
    ) -> Result<NvJ2kDecodeResult, NvBaselineError> {
        #[cfg(not(nvbaseline_built))]
        {
            let _ = (codestream, format);
            Err(NvBaselineError::NotBuilt)
        }
        #[cfg(nvbaseline_built)]
        {
            nvidia_decode_j2k_with_session(self.raw.as_ptr(), codestream, format)
        }
    }
}

#[cfg(nvbaseline_built)]
impl Drop for NvBaselineSession {
    fn drop(&mut self) {
        // SAFETY: `raw` is owned by this wrapper and is destroyed exactly once.
        unsafe { ffi::nvb_session_destroy(self.raw.as_ptr()) };
    }
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

/// Whether direct nvJPEG2000 decode is compiled in and initializes.
#[must_use]
pub fn nvidia_j2k_decode_available() -> bool {
    nvidia_baseline_available()
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
        // Size the host RGB buffer from the JPEG SOF header without creating
        // extra codec handles.
        let info = jpeg_dimensions_from_header(jpeg)?;
        let mut out = vec![0u8; (info.0 as usize) * (info.1 as usize) * 3];
        // SAFETY: `out` is sized to width*height*3 and `jpeg` is a valid slice.
        let rc = unsafe {
            ffi::nvb_decode_jpeg_rgb(
                jpeg.as_ptr(),
                jpeg.len(),
                out.as_mut_ptr(),
                out.len(),
                std::ptr::addr_of_mut!(width),
                std::ptr::addr_of_mut!(height),
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
        let mut session = NvBaselineSession::new()?;
        session.transcode_jpeg_to_htj2k(jpeg)
    }
}

/// Decode a JPEG 2000 / HTJ2K codestream via nvJPEG2000.
pub fn nvidia_decode_j2k_interleaved(
    codestream: &[u8],
    format: NvJ2kDecodeFormat,
) -> Result<NvJ2kDecodeResult, NvBaselineError> {
    #[cfg(not(nvbaseline_built))]
    {
        let _ = (codestream, format);
        Err(NvBaselineError::NotBuilt)
    }
    #[cfg(nvbaseline_built)]
    {
        let mut session = NvBaselineSession::new()?;
        session.decode_j2k_interleaved(codestream, format)
    }
}

#[cfg(nvbaseline_built)]
fn nvidia_transcode_with_session(
    session: *mut ffi::NvbSession,
    jpeg: &[u8],
) -> Result<NvTranscodeResult, NvBaselineError> {
    // HTJ2K of an 8-bit RGB image is well under the raw pixel size; allocate
    // raw RGB size plus header slack and grow once if the codec disagrees.
    let dims = jpeg_dimensions_from_header(jpeg)?;
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
            ffi::nvb_session_transcode_jpeg_to_htj2k(
                session,
                jpeg.as_ptr(),
                jpeg.len(),
                out.as_mut_ptr(),
                out.len(),
                std::ptr::addr_of_mut!(out_len),
                std::ptr::addr_of_mut!(decode_ms),
                std::ptr::addr_of_mut!(encode_ms),
                std::ptr::addr_of_mut!(width),
                std::ptr::addr_of_mut!(height),
                std::ptr::addr_of_mut!(num_components),
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

#[cfg(nvbaseline_built)]
fn nvidia_decode_j2k_with_session(
    session: *mut ffi::NvbSession,
    codestream: &[u8],
    format: NvJ2kDecodeFormat,
) -> Result<NvJ2kDecodeResult, NvBaselineError> {
    let image = signinum_j2k_native::Image::new(
        codestream,
        &signinum_j2k_native::DecodeSettings::default(),
    )
    .map_err(|_| NvBaselineError::Stage(241))?;
    let dims = (image.width(), image.height());
    let mut capacity = (dims.0 as usize)
        .saturating_mul(dims.1 as usize)
        .saturating_mul(format.bytes_per_pixel());
    if capacity == 0 {
        return Err(NvBaselineError::Stage(240));
    }
    loop {
        let mut out = vec![0u8; capacity];
        let mut out_len = 0usize;
        let mut decode_ms = 0f64;
        let mut width = 0i32;
        let mut height = 0i32;
        let mut num_components = 0i32;
        let mut bit_depth = 0i32;
        let mut bytes_per_sample = 0i32;
        // SAFETY: all pointers reference live, correctly-sized Rust slices or
        // out parameters. The C++ side writes at most `out.len()` bytes.
        let rc = unsafe {
            ffi::nvb_session_decode_j2k_interleaved(
                session,
                codestream.as_ptr(),
                codestream.len(),
                format.ffi_code(),
                out.as_mut_ptr(),
                out.len(),
                std::ptr::addr_of_mut!(out_len),
                std::ptr::addr_of_mut!(decode_ms),
                std::ptr::addr_of_mut!(width),
                std::ptr::addr_of_mut!(height),
                std::ptr::addr_of_mut!(num_components),
                std::ptr::addr_of_mut!(bit_depth),
                std::ptr::addr_of_mut!(bytes_per_sample),
            )
        };
        if rc == 234 && capacity < (1 << 30) {
            capacity = capacity.saturating_mul(2);
            continue;
        }
        if rc != 0 {
            return Err(NvBaselineError::Stage(rc));
        }
        out.truncate(out_len);
        return Ok(NvJ2kDecodeResult {
            pixels: out,
            decode_ms,
            width: width as u32,
            height: height as u32,
            num_components: num_components as u32,
            bit_depth: bit_depth as u32,
            bytes_per_sample: bytes_per_sample as u32,
        });
    }
}

/// Read `(width, height)` from a JPEG SOF marker without initializing nvJPEG.
#[cfg(nvbaseline_built)]
fn jpeg_dimensions_from_header(jpeg: &[u8]) -> Result<(u32, u32), NvBaselineError> {
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return Err(NvBaselineError::Stage(103));
    }

    let mut offset = 2usize;
    while offset + 3 < jpeg.len() {
        if jpeg[offset] != 0xFF {
            offset = offset.saturating_add(1);
            continue;
        }
        while offset + 1 < jpeg.len() && jpeg[offset + 1] == 0xFF {
            offset = offset.saturating_add(1);
        }
        if offset + 1 >= jpeg.len() {
            break;
        }

        let marker = jpeg[offset + 1];
        if marker == 0xD9 || marker == 0xDA {
            break;
        }
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) {
            offset = offset.saturating_add(2);
            continue;
        }
        if offset + 3 >= jpeg.len() {
            break;
        }

        let segment_len = (usize::from(jpeg[offset + 2]) << 8) | usize::from(jpeg[offset + 3]);
        if segment_len < 2 || offset + 2 + segment_len > jpeg.len() {
            break;
        }
        let is_sof = matches!(
            marker,
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF
        );
        if is_sof {
            if segment_len < 7 {
                break;
            }
            let height = (u32::from(jpeg[offset + 5]) << 8) | u32::from(jpeg[offset + 6]);
            let width = (u32::from(jpeg[offset + 7]) << 8) | u32::from(jpeg[offset + 8]);
            if width != 0 && height != 0 {
                return Ok((width, height));
            }
            break;
        }
        offset = offset.saturating_add(2 + segment_len);
    }

    Err(NvBaselineError::Stage(103))
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
