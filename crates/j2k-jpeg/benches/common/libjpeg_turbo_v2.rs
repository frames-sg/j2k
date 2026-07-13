// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Downscale, Rect};
use std::ffi::{c_char, c_int, c_ulong, c_void, CStr};
use std::ptr::NonNull;

#[path = "libjpeg_turbo_v2_transform.rs"]
mod transform;

use self::transform::{CropRegion, Transformer};

const TJPF_RGB: c_int = 0;
const TJPF_GRAY: c_int = 6;
const TJFLAGS_DEFAULT: c_int = 0;
const TJMCU_WIDTH: [u32; 7] = [8, 16, 16, 8, 8, 32, 8];
const TJMCU_HEIGHT: [u32; 7] = [8, 8, 16, 8, 16, 8, 32];

type TjHandle = *mut c_void;

unsafe extern "C" {
    fn tjInitDecompress() -> TjHandle;
    fn tjDestroy(handle: TjHandle) -> c_int;
    fn tjGetErrorStr2(handle: TjHandle) -> *mut c_char;
    fn tjDecompressHeader3(
        handle: TjHandle,
        jpeg_buf: *const u8,
        jpeg_size: c_ulong,
        width: *mut c_int,
        height: *mut c_int,
        jpeg_subsamp: *mut c_int,
        jpeg_colorspace: *mut c_int,
    ) -> c_int;
    fn tjDecompress2(
        handle: TjHandle,
        jpeg_buf: *const u8,
        jpeg_size: c_ulong,
        dst_buf: *mut u8,
        width: c_int,
        pitch: c_int,
        height: c_int,
        pixel_format: c_int,
        flags: c_int,
    ) -> c_int;
}

pub(crate) struct TurboJpegDecoder {
    handle: NonNull<c_void>,
    prepared_dimensions: Option<(u32, u32)>,
    transformer: Transformer,
}

impl TurboJpegDecoder {
    pub(crate) fn new() -> Result<Self, String> {
        let transformer = Transformer::new()?;
        // SAFETY: `tjInitDecompress` takes no arguments and returns an owned handle.
        let handle = unsafe { tjInitDecompress() };
        let Some(handle) = NonNull::new(handle) else {
            return Err("tjInitDecompress returned null".to_string());
        };
        Ok(Self {
            handle,
            prepared_dimensions: None,
            transformer,
        })
    }

    pub(crate) fn decode_rgb(&mut self, bytes: &[u8]) -> Result<Vec<u8>, String> {
        self.decode(bytes, TJPF_RGB, None, Downscale::None)
    }

    pub(crate) fn prepare_full_frame(&mut self, bytes: &[u8]) -> Result<(u32, u32, i32), String> {
        let header = self.read_header(bytes)?;
        self.prepared_dimensions = Some((header.0, header.1));
        Ok(header)
    }

    pub(crate) fn decode(
        &mut self,
        bytes: &[u8],
        pixel_format: c_int,
        roi: Option<Rect>,
        factor: Downscale,
    ) -> Result<Vec<u8>, String> {
        let unscaled_full_frame = roi.is_none() && factor == Downscale::None;
        let (width, height, subsamp) = if unscaled_full_frame {
            self.prepare_full_frame(bytes)?
        } else {
            self.read_header(bytes)?
        };
        if let Some(roi) = roi {
            return self.decode_region(bytes, (width, height, subsamp), roi, factor, pixel_format);
        }
        self.decode_full(bytes, width, height, factor, pixel_format)
    }

    fn decode_full(
        &mut self,
        bytes: &[u8],
        width: u32,
        height: u32,
        factor: Downscale,
        pixel_format: c_int,
    ) -> Result<Vec<u8>, String> {
        let output_width = scaled_dimension(width, factor);
        let output_height = scaled_dimension(height, factor);
        let bytes_per_pixel = bytes_per_pixel(pixel_format)?;
        let (pitch, output_len) = packed_layout(output_width, output_height, bytes_per_pixel)?;
        let mut output = vec![0_u8; output_len];
        if factor == Downscale::None {
            self.prepared_dimensions = Some((width, height));
            self.decompress(bytes, &mut output, pitch, pixel_format)?;
        } else {
            self.decompress_dimensions(
                bytes,
                &mut output,
                pitch,
                output_width,
                output_height,
                pixel_format,
            )?;
        }
        Ok(output)
    }

    fn decode_region(
        &mut self,
        bytes: &[u8],
        header: (u32, u32, i32),
        roi: Rect,
        factor: Downscale,
        pixel_format: c_int,
    ) -> Result<Vec<u8>, String> {
        let plan = region_decode_plan(header, roi, factor)?;
        let transformed = self.transformer.crop(bytes, plan.transform)?;
        let (width, height, _) = self.read_header(transformed.as_bytes())?;
        let output =
            self.decode_full(transformed.as_bytes(), width, height, factor, pixel_format)?;
        crop_packed_rows(
            &output,
            scaled_dimension(width, factor),
            scaled_dimension(height, factor),
            plan.output_crop,
            bytes_per_pixel(pixel_format)?,
        )
    }

    pub(crate) fn read_header(&mut self, bytes: &[u8]) -> Result<(u32, u32, i32), String> {
        let jpeg_size = to_c_ulong(bytes.len(), "JPEG input length")?;
        let mut width = 0;
        let mut height = 0;
        let mut subsamp = 0;
        let mut colorspace = 0;
        // SAFETY: The handle is live, the input slice remains valid for the call, and each output
        // pointer refers to initialized, writable storage of the type required by TurboJPEG.
        let rc = unsafe {
            tjDecompressHeader3(
                self.handle.as_ptr(),
                bytes.as_ptr(),
                jpeg_size,
                &raw mut width,
                &raw mut height,
                &raw mut subsamp,
                &raw mut colorspace,
            )
        };
        if rc != 0 {
            return Err(self.error_string());
        }
        if width < 0 || height < 0 || subsamp < 0 {
            return Err("tjDecompressHeader3 returned incomplete header parameters".to_string());
        }
        Ok((
            u32::try_from(width).map_err(|_| format!("negative libjpeg-turbo width {width}"))?,
            u32::try_from(height).map_err(|_| format!("negative libjpeg-turbo height {height}"))?,
            subsamp,
        ))
    }

    pub(crate) fn decompress(
        &mut self,
        bytes: &[u8],
        out: &mut [u8],
        pitch: usize,
        pixel_format: c_int,
    ) -> Result<(), String> {
        let Some((width, height)) = self.prepared_dimensions else {
            return Err("libjpeg-turbo decoder was not prepared".to_string());
        };
        self.decompress_dimensions(bytes, out, pitch, width, height, pixel_format)
    }

    fn decompress_dimensions(
        &mut self,
        bytes: &[u8],
        out: &mut [u8],
        pitch: usize,
        width: u32,
        height: u32,
        pixel_format: c_int,
    ) -> Result<(), String> {
        let jpeg_size = to_c_ulong(bytes.len(), "JPEG input length")?;
        let width = to_c_int(width, "output width")?;
        let height = to_c_int(height, "output height")?;
        let pitch = c_int::try_from(pitch)
            .map_err(|_| format!("output pitch {pitch} does not fit into c_int"))?;
        // SAFETY: The handle is live, both slices remain valid for the call, and the output layout
        // was checked before allocation or by the prepared-buffer validator.
        let rc = unsafe {
            tjDecompress2(
                self.handle.as_ptr(),
                bytes.as_ptr(),
                jpeg_size,
                out.as_mut_ptr(),
                width,
                pitch,
                height,
                pixel_format,
                TJFLAGS_DEFAULT,
            )
        };
        if rc != 0 {
            return Err(self.error_string());
        }
        Ok(())
    }

    fn error_string(&self) -> String {
        // SAFETY: The handle remains live for the duration of the call.
        let ptr = unsafe { tjGetErrorStr2(self.handle.as_ptr()) };
        if ptr.is_null() {
            return "libjpeg-turbo error".to_string();
        }
        // SAFETY: TurboJPEG returns a NUL-terminated error string owned by the live handle.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

impl Drop for TurboJpegDecoder {
    fn drop(&mut self) {
        // SAFETY: This object owns the live handle and destroys it exactly once.
        let _ = unsafe { tjDestroy(self.handle.as_ptr()) };
    }
}

#[cfg(not(has_libjpeg_turbo_v3))]
pub(crate) fn is_available() -> bool {
    true
}

fn bytes_per_pixel(pixel_format: c_int) -> Result<usize, String> {
    match pixel_format {
        TJPF_RGB => Ok(3),
        TJPF_GRAY => Ok(1),
        _ => Err(format!("unsupported TurboJPEG pixel format {pixel_format}")),
    }
}

fn scaled_dimension(dimension: u32, factor: Downscale) -> u32 {
    dimension.div_ceil(factor.denominator())
}

fn scaled_rect(rect: Rect, factor: Downscale) -> Result<Rect, String> {
    let denominator = factor.denominator();
    let x_end = rect
        .x
        .checked_add(rect.w)
        .ok_or_else(|| "TurboJPEG region x extent overflow".to_string())?;
    let y_end = rect
        .y
        .checked_add(rect.h)
        .ok_or_else(|| "TurboJPEG region y extent overflow".to_string())?;
    Ok(Rect {
        x: rect.x / denominator,
        y: rect.y / denominator,
        w: x_end.div_ceil(denominator) - rect.x / denominator,
        h: y_end.div_ceil(denominator) - rect.y / denominator,
    })
}

struct RegionDecodePlan {
    transform: CropRegion,
    output_crop: Rect,
}

fn region_decode_plan(
    (source_width, source_height, subsamp): (u32, u32, i32),
    roi: Rect,
    factor: Downscale,
) -> Result<RegionDecodePlan, String> {
    if roi.w == 0 || roi.h == 0 {
        return Err("TurboJPEG region dimensions must be non-zero".to_string());
    }
    let roi_x_end = roi
        .x
        .checked_add(roi.w)
        .ok_or_else(|| "TurboJPEG region x extent overflow".to_string())?;
    let roi_y_end = roi
        .y
        .checked_add(roi.h)
        .ok_or_else(|| "TurboJPEG region y extent overflow".to_string())?;
    if roi_x_end > source_width || roi_y_end > source_height {
        return Err(format!(
            "TurboJPEG region ({}, {}, {}, {}) exceeds source {}x{}",
            roi.x, roi.y, roi.w, roi.h, source_width, source_height
        ));
    }

    let subsamp = usize::try_from(subsamp)
        .ok()
        .filter(|index| *index < TJMCU_WIDTH.len())
        .ok_or_else(|| format!("unsupported TurboJPEG subsampling {subsamp}"))?;
    let denominator = factor.denominator();
    let scaled_roi = scaled_rect(roi, factor)?;
    let scaled_mcu_width = TJMCU_WIDTH[subsamp].div_ceil(denominator).max(1);
    let scaled_mcu_height = TJMCU_HEIGHT[subsamp].div_ceil(denominator).max(1);
    let aligned_x = scaled_roi.x - scaled_roi.x % scaled_mcu_width;
    let aligned_y = scaled_roi.y - scaled_roi.y % scaled_mcu_height;
    let trim_left = scaled_roi.x - aligned_x;
    let trim_top = scaled_roi.y - aligned_y;
    let scaled_width = trim_left
        .checked_add(scaled_roi.w)
        .ok_or_else(|| "TurboJPEG scaled region width overflow".to_string())?;
    let scaled_height = trim_top
        .checked_add(scaled_roi.h)
        .ok_or_else(|| "TurboJPEG scaled region height overflow".to_string())?;

    let transform_x = aligned_x
        .checked_mul(denominator)
        .ok_or_else(|| "TurboJPEG transform x overflow".to_string())?;
    let transform_y = aligned_y
        .checked_mul(denominator)
        .ok_or_else(|| "TurboJPEG transform y overflow".to_string())?;
    let requested_width = scaled_width
        .checked_mul(denominator)
        .ok_or_else(|| "TurboJPEG transform width overflow".to_string())?;
    let requested_height = scaled_height
        .checked_mul(denominator)
        .ok_or_else(|| "TurboJPEG transform height overflow".to_string())?;
    let transform_height = requested_height
        .div_ceil(TJMCU_HEIGHT[subsamp])
        .checked_mul(TJMCU_HEIGHT[subsamp])
        .ok_or_else(|| "TurboJPEG transform height alignment overflow".to_string())?;
    let width = requested_width.min(source_width - transform_x);
    let height = transform_height.min(source_height - transform_y);
    if width == 0 || height == 0 {
        return Err("TurboJPEG transform region is empty".to_string());
    }

    Ok(RegionDecodePlan {
        transform: CropRegion {
            x: transform_x,
            y: transform_y,
            width,
            height,
        },
        output_crop: Rect {
            x: trim_left,
            y: trim_top,
            w: scaled_roi.w,
            h: scaled_roi.h,
        },
    })
}

fn packed_layout(
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
) -> Result<(usize, usize), String> {
    let width = usize::try_from(width).map_err(|_| "output width exceeds usize".to_string())?;
    let height = usize::try_from(height).map_err(|_| "output height exceeds usize".to_string())?;
    let pitch = width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| "TurboJPEG output row size overflow".to_string())?;
    let output_len = pitch
        .checked_mul(height)
        .ok_or_else(|| "TurboJPEG output buffer size overflow".to_string())?;
    Ok((pitch, output_len))
}

fn crop_packed_rows(
    full: &[u8],
    source_width: u32,
    source_height: u32,
    roi: Rect,
    bytes_per_pixel: usize,
) -> Result<Vec<u8>, String> {
    let x_end = roi
        .x
        .checked_add(roi.w)
        .ok_or_else(|| "TurboJPEG crop x extent overflow".to_string())?;
    let y_end = roi
        .y
        .checked_add(roi.h)
        .ok_or_else(|| "TurboJPEG crop y extent overflow".to_string())?;
    if x_end > source_width || y_end > source_height {
        return Err(format!(
            "TurboJPEG crop ({}, {}, {}, {}) exceeds output {}x{}",
            roi.x, roi.y, roi.w, roi.h, source_width, source_height
        ));
    }

    let (source_pitch, expected_len) = packed_layout(source_width, source_height, bytes_per_pixel)?;
    if full.len() < expected_len {
        return Err(format!(
            "TurboJPEG source buffer too small: need {expected_len}, got {}",
            full.len()
        ));
    }
    let (target_pitch, target_len) = packed_layout(roi.w, roi.h, bytes_per_pixel)?;
    let x = usize::try_from(roi.x).map_err(|_| "crop x exceeds usize".to_string())?;
    let y = usize::try_from(roi.y).map_err(|_| "crop y exceeds usize".to_string())?;
    let height = usize::try_from(roi.h).map_err(|_| "crop height exceeds usize".to_string())?;
    let x_offset = x
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| "TurboJPEG crop byte offset overflow".to_string())?;
    let mut output = vec![0_u8; target_len];
    for row in 0..height {
        let source_start = (y + row) * source_pitch + x_offset;
        let target_start = row * target_pitch;
        output[target_start..target_start + target_pitch]
            .copy_from_slice(&full[source_start..source_start + target_pitch]);
    }
    Ok(output)
}

fn to_c_int(value: u32, label: &str) -> Result<c_int, String> {
    c_int::try_from(value).map_err(|_| format!("{label} {value} does not fit into c_int"))
}

fn to_c_ulong(value: usize, label: &str) -> Result<c_ulong, String> {
    c_ulong::try_from(value).map_err(|_| format!("{label} {value} does not fit into c_ulong"))
}
