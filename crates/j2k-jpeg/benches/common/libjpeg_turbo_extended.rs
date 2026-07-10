// SPDX-License-Identifier: MIT OR Apache-2.0

use super::libjpeg_turbo::TurboJpegDecoder;
use j2k_jpeg::{Downscale, Rect};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InspectInfo {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) subsamp: i32,
}

#[cfg(has_libjpeg_turbo)]
impl TurboJpegDecoder {
    pub(crate) fn inspect(&mut self, bytes: &[u8]) -> Result<InspectInfo, String> {
        let (width, height, subsamp) = self.read_header(bytes)?;
        Ok(InspectInfo {
            width,
            height,
            subsamp,
        })
    }

    pub(crate) fn decode_gray(&mut self, bytes: &[u8]) -> Result<Vec<u8>, String> {
        self.decode(bytes, 6, None, Downscale::None)
    }

    pub(crate) fn prepare_rgb(&mut self, bytes: &[u8]) -> Result<InspectInfo, String> {
        let (width, height, subsamp) = self.prepare_full_frame(bytes)?;
        Ok(InspectInfo {
            width,
            height,
            subsamp,
        })
    }

    pub(crate) fn decode_prepared_rgb_into(
        &mut self,
        bytes: &[u8],
        out: &mut [u8],
        pitch: usize,
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        validate_output_buffer(width, height, out, pitch)?;
        self.decompress(bytes, out, pitch, 0)
    }

    pub(crate) fn decode_scaled_rgb(
        &mut self,
        bytes: &[u8],
        factor: Downscale,
    ) -> Result<Vec<u8>, String> {
        self.decode(bytes, 0, None, factor)
    }

    pub(crate) fn decode_region_rgb(&mut self, bytes: &[u8], roi: Rect) -> Result<Vec<u8>, String> {
        self.decode(bytes, 0, Some(roi), Downscale::None)
    }

    pub(crate) fn decode_region_scaled_rgb(
        &mut self,
        bytes: &[u8],
        roi: Rect,
        factor: Downscale,
    ) -> Result<Vec<u8>, String> {
        self.decode(bytes, 0, Some(roi), factor)
    }
}

#[cfg(not(has_libjpeg_turbo))]
impl TurboJpegDecoder {
    pub(crate) fn inspect(&mut self, _bytes: &[u8]) -> Result<InspectInfo, String> {
        let _ = self;
        super::libjpeg_turbo::unavailable()
    }

    pub(crate) fn decode_gray(&mut self, _bytes: &[u8]) -> Result<Vec<u8>, String> {
        let _ = self;
        super::libjpeg_turbo::unavailable()
    }

    pub(crate) fn prepare_rgb(&mut self, _bytes: &[u8]) -> Result<InspectInfo, String> {
        let _ = self;
        super::libjpeg_turbo::unavailable()
    }

    pub(crate) fn decode_prepared_rgb_into(
        &mut self,
        _bytes: &[u8],
        _out: &mut [u8],
        _pitch: usize,
        _width: usize,
        _height: usize,
    ) -> Result<(), String> {
        let _ = self;
        super::libjpeg_turbo::unavailable()
    }

    pub(crate) fn decode_scaled_rgb(
        &mut self,
        _bytes: &[u8],
        _factor: Downscale,
    ) -> Result<Vec<u8>, String> {
        let _ = self;
        super::libjpeg_turbo::unavailable()
    }

    pub(crate) fn decode_region_rgb(
        &mut self,
        _bytes: &[u8],
        _roi: Rect,
    ) -> Result<Vec<u8>, String> {
        let _ = self;
        super::libjpeg_turbo::unavailable()
    }

    pub(crate) fn decode_region_scaled_rgb(
        &mut self,
        _bytes: &[u8],
        _roi: Rect,
        _factor: Downscale,
    ) -> Result<Vec<u8>, String> {
        let _ = self;
        super::libjpeg_turbo::unavailable()
    }
}

#[cfg(has_libjpeg_turbo)]
fn validate_output_buffer(
    width: usize,
    height: usize,
    out: &[u8],
    pitch: usize,
) -> Result<(), String> {
    let row_bytes = width
        .checked_mul(3)
        .ok_or("libjpeg-turbo output row size overflow")?;
    if pitch < row_bytes {
        return Err(format!(
            "libjpeg-turbo output pitch {pitch} is smaller than row bytes {row_bytes}"
        ));
    }
    let required_len = required_strided_len(height, pitch, row_bytes)?;
    if out.len() < required_len {
        return Err(format!(
            "libjpeg-turbo output buffer too small: need {required_len}, got {}",
            out.len()
        ));
    }
    Ok(())
}

#[cfg(has_libjpeg_turbo)]
fn required_strided_len(height: usize, pitch: usize, row_bytes: usize) -> Result<usize, String> {
    if height == 0 {
        return Ok(0);
    }
    pitch
        .checked_mul(height - 1)
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or_else(|| "libjpeg-turbo output buffer size overflow".to_string())
}
