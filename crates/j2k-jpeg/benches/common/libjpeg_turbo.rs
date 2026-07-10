// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(has_libjpeg_turbo)]
use j2k_jpeg::{Downscale, Rect};

#[cfg(has_libjpeg_turbo)]
mod imp {
    use super::{Downscale, Rect};
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const TJINIT_DECOMPRESS: c_int = 1;
    const TJPF_RGB: c_int = 0;
    const TJPF_GRAY: c_int = 6;
    const TJPARAM_SUBSAMP: c_int = 4;
    const TJPARAM_JPEGWIDTH: c_int = 5;
    const TJPARAM_JPEGHEIGHT: c_int = 6;
    const TJUNSCALED: TjScalingFactor = TjScalingFactor { num: 1, denom: 1 };
    const TJMCU_WIDTH: [u32; 7] = [8, 16, 16, 8, 8, 32, 8];
    const TJUNCROPPED: TjRegion = TjRegion {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TjScalingFactor {
        num: c_int,
        denom: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TjRegion {
        x: c_int,
        y: c_int,
        w: c_int,
        h: c_int,
    }

    type TjHandle = *mut c_void;

    unsafe extern "C" {
        fn tj3Init(init_type: c_int) -> TjHandle;
        fn tj3Destroy(handle: TjHandle);
        fn tj3GetErrorStr(handle: TjHandle) -> *mut c_char;
        fn tj3Get(handle: TjHandle, param: c_int) -> c_int;
        fn tj3DecompressHeader(handle: TjHandle, jpeg_buf: *const u8, jpeg_size: usize) -> c_int;
        fn tj3SetScalingFactor(handle: TjHandle, scaling_factor: TjScalingFactor) -> c_int;
        fn tj3SetCroppingRegion(handle: TjHandle, cropping_region: TjRegion) -> c_int;
        fn tj3Decompress8(
            handle: TjHandle,
            jpeg_buf: *const u8,
            jpeg_size: usize,
            dst_buf: *mut u8,
            pitch: c_int,
            pixel_format: c_int,
        ) -> c_int;
    }

    pub(crate) struct TurboJpegDecoder {
        handle: NonNull<c_void>,
    }

    impl TurboJpegDecoder {
        pub(crate) fn new() -> Result<Self, String> {
            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            let handle = unsafe { tj3Init(TJINIT_DECOMPRESS) };
            let Some(handle) = NonNull::new(handle) else {
                return Err("tj3Init returned null".to_string());
            };
            Ok(Self { handle })
        }

        pub(crate) fn decode_rgb(&mut self, bytes: &[u8]) -> Result<Vec<u8>, String> {
            self.decode(bytes, TJPF_RGB, None, Downscale::None)
        }

        pub(crate) fn prepare_full_frame(
            &mut self,
            bytes: &[u8],
        ) -> Result<(u32, u32, i32), String> {
            let header = self.read_header(bytes)?;
            self.set_scaling(TJUNSCALED)?;
            self.set_crop(TJUNCROPPED)?;
            Ok(header)
        }

        pub(crate) fn decode(
            &mut self,
            bytes: &[u8],
            pixel_format: c_int,
            roi: Option<Rect>,
            factor: Downscale,
        ) -> Result<Vec<u8>, String> {
            let is_unscaled_full_frame = roi.is_none() && factor == Downscale::None;
            let header = if is_unscaled_full_frame {
                self.prepare_full_frame(bytes)?
            } else {
                let header = self.read_header(bytes)?;
                self.set_scaling(scaling_factor(factor))?;
                header
            };
            let scale = scaling_factor(factor);
            let bytes_per_pixel = bytes_per_pixel(pixel_format);

            if let Some(roi) = roi {
                let scaled_roi = scaled_rect(roi, factor);
                let scaled_mcu = scaled_mcu_width(header.2, scale);
                let aligned_x = scaled_roi.x - scaled_roi.x % scaled_mcu;
                let trim_left = scaled_roi.x - aligned_x;
                let crop_width = trim_left + scaled_roi.w;
                self.set_crop(TjRegion {
                    x: to_c_int(aligned_x)?,
                    y: to_c_int(scaled_roi.y)?,
                    w: to_c_int(crop_width)?,
                    h: to_c_int(scaled_roi.h)?,
                })?;

                let pitch = crop_width as usize * bytes_per_pixel;
                let mut out = vec![0u8; pitch * scaled_roi.h as usize];
                self.decompress(bytes, &mut out, pitch, pixel_format)?;
                if trim_left == 0 {
                    return Ok(out);
                }
                return Ok(trim_packed_rows(
                    &out,
                    crop_width as usize,
                    scaled_roi.w as usize,
                    scaled_roi.h as usize,
                    trim_left as usize,
                    bytes_per_pixel,
                ));
            }

            if !is_unscaled_full_frame {
                self.set_crop(TJUNCROPPED)?;
            }
            let out_width = scaled_dimension(header.0, scale);
            let out_height = scaled_dimension(header.1, scale);
            let pitch = out_width as usize * bytes_per_pixel;
            let mut out = vec![0u8; pitch * out_height as usize];
            self.decompress(bytes, &mut out, pitch, pixel_format)?;
            Ok(out)
        }

        pub(crate) fn read_header(&mut self, bytes: &[u8]) -> Result<(u32, u32, i32), String> {
            let rc =
                // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
                unsafe { tj3DecompressHeader(self.handle.as_ptr(), bytes.as_ptr(), bytes.len()) };
            if rc != 0 {
                return Err(self.error_string());
            }

            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            let width = unsafe { tj3Get(self.handle.as_ptr(), TJPARAM_JPEGWIDTH) };
            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            let height = unsafe { tj3Get(self.handle.as_ptr(), TJPARAM_JPEGHEIGHT) };
            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            let subsamp = unsafe { tj3Get(self.handle.as_ptr(), TJPARAM_SUBSAMP) };
            if width < 0 || height < 0 || subsamp < 0 {
                return Err("tj3Get returned incomplete header parameters".to_string());
            }

            Ok((
                u32::try_from(width)
                    .map_err(|_| format!("negative libjpeg-turbo width {width}"))?,
                u32::try_from(height)
                    .map_err(|_| format!("negative libjpeg-turbo height {height}"))?,
                subsamp,
            ))
        }

        fn set_scaling(&mut self, scale: TjScalingFactor) -> Result<(), String> {
            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            let rc = unsafe { tj3SetScalingFactor(self.handle.as_ptr(), scale) };
            if rc != 0 {
                return Err(self.error_string());
            }
            Ok(())
        }

        fn set_crop(&mut self, region: TjRegion) -> Result<(), String> {
            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            let rc = unsafe { tj3SetCroppingRegion(self.handle.as_ptr(), region) };
            if rc != 0 {
                return Err(self.error_string());
            }
            Ok(())
        }

        pub(crate) fn decompress(
            &mut self,
            bytes: &[u8],
            out: &mut [u8],
            pitch: usize,
            pixel_format: c_int,
        ) -> Result<(), String> {
            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            let rc = unsafe {
                tj3Decompress8(
                    self.handle.as_ptr(),
                    bytes.as_ptr(),
                    bytes.len(),
                    out.as_mut_ptr(),
                    to_c_int(
                        u32::try_from(pitch)
                            .map_err(|_| format!("output pitch {pitch} exceeds u32"))?,
                    )?,
                    pixel_format,
                )
            };
            if rc != 0 {
                return Err(self.error_string());
            }
            Ok(())
        }

        fn error_string(&self) -> String {
            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            let ptr = unsafe { tj3GetErrorStr(self.handle.as_ptr()) };
            if ptr.is_null() {
                return "libjpeg-turbo error".to_string();
            }
            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }

    impl Drop for TurboJpegDecoder {
        fn drop(&mut self) {
            // SAFETY: Benchmark FFI calls use a live libjpeg-turbo handle and sized outputs.
            unsafe { tj3Destroy(self.handle.as_ptr()) };
        }
    }

    pub(crate) fn is_available() -> bool {
        true
    }

    fn scaling_factor(factor: Downscale) -> TjScalingFactor {
        match factor {
            Downscale::None => TJUNSCALED,
            Downscale::Half => TjScalingFactor { num: 1, denom: 2 },
            Downscale::Quarter => TjScalingFactor { num: 1, denom: 4 },
            Downscale::Eighth => TjScalingFactor { num: 1, denom: 8 },
            _ => unreachable!("unsupported Downscale variant"),
        }
    }

    fn bytes_per_pixel(pixel_format: c_int) -> usize {
        match pixel_format {
            TJPF_RGB => 3,
            TJPF_GRAY => 1,
            _ => unreachable!("unsupported pixel format"),
        }
    }

    fn scaled_dimension(dimension: u32, scale: TjScalingFactor) -> u32 {
        let numerator = u32::try_from(scale.num).expect("scaling numerator is non-negative");
        let denominator = u32::try_from(scale.denom).expect("scaling denominator is positive");
        (dimension * numerator).div_ceil(denominator)
    }

    fn scaled_rect(rect: Rect, factor: Downscale) -> Rect {
        let denom = factor.denominator();
        let x_end = rect.x + rect.w;
        let y_end = rect.y + rect.h;
        Rect {
            x: rect.x / denom,
            y: rect.y / denom,
            w: x_end.div_ceil(denom) - rect.x / denom,
            h: y_end.div_ceil(denom) - rect.y / denom,
        }
    }

    fn scaled_mcu_width(subsamp: i32, scale: TjScalingFactor) -> u32 {
        let mcu = usize::try_from(subsamp)
            .ok()
            .and_then(|index| TJMCU_WIDTH.get(index))
            .copied()
            .unwrap_or(8);
        scaled_dimension(mcu, scale).max(1)
    }

    fn trim_packed_rows(
        full: &[u8],
        source_width: usize,
        target_width: usize,
        height: usize,
        trim_left: usize,
        bytes_per_pixel: usize,
    ) -> Vec<u8> {
        let source_stride = source_width * bytes_per_pixel;
        let target_stride = target_width * bytes_per_pixel;
        let mut out = vec![0u8; target_stride * height];
        for row in 0..height {
            let src_start = row * source_stride + trim_left * bytes_per_pixel;
            let src_end = src_start + target_stride;
            let dst_start = row * target_stride;
            out[dst_start..dst_start + target_stride].copy_from_slice(&full[src_start..src_end]);
        }
        out
    }

    fn to_c_int(value: u32) -> Result<c_int, String> {
        c_int::try_from(value).map_err(|_| format!("value {value} does not fit into c_int"))
    }
}

#[cfg(not(has_libjpeg_turbo))]
mod imp {
    pub(crate) struct TurboJpegDecoder;

    impl TurboJpegDecoder {
        pub(crate) fn new() -> Result<Self, String> {
            super::unavailable()
        }

        pub(crate) fn decode_rgb(&mut self, _bytes: &[u8]) -> Result<Vec<u8>, String> {
            let _ = self;
            super::unavailable()
        }
    }

    pub(crate) fn is_available() -> bool {
        false
    }
}

pub(crate) use imp::{is_available, TurboJpegDecoder};

#[cfg(not(has_libjpeg_turbo))]
pub(crate) fn unavailable<T>() -> Result<T, String> {
    Err("libjpeg-turbo not available".to_string())
}
