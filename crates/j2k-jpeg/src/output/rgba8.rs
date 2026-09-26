// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Rgba8Writer` — 4-byte-per-pixel RGBA output. Alpha is a constant captured
//! at construction time and written verbatim into every pixel's A channel.

use crate::backend::Backend;
use crate::error::JpegError;
use crate::output::{InterleavedRgbWriter, OutputWriter};

pub(crate) struct Rgba8Writer<'o> {
    out: &'o mut [u8],
    stride: usize,
    width: u32,
    alpha: u8,
    backend: Backend,
}

impl InterleavedRgbWriter for Rgba8Writer<'_> {
    fn with_rgb_rows<R, F>(&mut self, y: u32, row_count: usize, fill: F) -> Result<R, JpegError>
    where
        F: FnOnce(&mut [u8], Option<&mut [u8]>) -> Result<R, JpegError>,
    {
        let width = self.width as usize;
        let start = y as usize * self.stride;
        let (top, bottom) = match row_count {
            1 => (&mut self.out[start..start + width * 4], None),
            2 => {
                let (head, tail) = self.out.split_at_mut(start + self.stride);
                (
                    &mut head[start..start + width * 4],
                    Some(&mut tail[..width * 4]),
                )
            }
            _ => unreachable!("Rgba8Writer only supports one or two rows"),
        };
        let mut bottom = bottom;
        let result = fill(
            &mut top[..width * 3],
            bottom.as_deref_mut().map(|row| &mut row[..width * 3]),
        )?;
        // The fused RGB kernels write into the caller's RGBA storage. Expanding
        // backwards preserves unread RGB pixels and needs no intermediate rows.
        for row in core::iter::once(top).chain(bottom) {
            for x in (0..width).rev() {
                let rgb = [row[x * 3], row[x * 3 + 1], row[x * 3 + 2]];
                row[x * 4..x * 4 + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], self.alpha]);
            }
        }
        Ok(result)
    }
}

impl<'o> Rgba8Writer<'o> {
    #[cfg(test)]
    pub(crate) fn new(out: &'o mut [u8], stride: usize, width: u32, alpha: u8) -> Self {
        Self::new_with_backend(out, stride, width, alpha, Backend::detect())
    }

    pub(crate) fn new_with_backend(
        out: &'o mut [u8],
        stride: usize,
        width: u32,
        alpha: u8,
        backend: Backend,
    ) -> Self {
        Self {
            out,
            stride,
            width,
            alpha,
            backend,
        }
    }
}

impl OutputWriter for Rgba8Writer<'_> {
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 4];
        self.backend
            .fill_rgba_row_from_rgb(r_row, g_row, b_row, dst, self.alpha);
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError> {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 4];
        self.backend
            .fill_rgba_row_from_ycbcr(y_row, cb_row, cr_row, dst, self.alpha);
        Ok(())
    }

    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        let dst_start = (y as usize) * self.stride;
        let width = self.width as usize;
        let dst = &mut self.out[dst_start..dst_start + width * 4];
        self.backend
            .fill_rgba_row_from_gray(gray_row, dst, self.alpha);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn writes_alpha_byte_per_pixel() {
        let mut buf = vec![0u8; 2 * 4];
        let mut w = Rgba8Writer::new(&mut buf, 8, 2, 200);
        w.write_ycbcr_row(0, &[128, 128], &[128, 128], &[128, 128])
            .unwrap();
        assert_eq!(buf[3], 200);
        assert_eq!(buf[7], 200);
    }

    #[test]
    fn grayscale_expands_with_alpha() {
        let mut buf = vec![0u8; 3 * 4];
        let mut w = Rgba8Writer::new(&mut buf, 12, 3, 255);
        w.write_gray_row(0, &[10, 20, 30]).unwrap();
        assert_eq!(buf, vec![10, 10, 10, 255, 20, 20, 20, 255, 30, 30, 30, 255]);
    }

    #[test]
    fn ycbcr_color_conversion_honors_alpha() {
        let mut buf = vec![0u8; 4];
        let mut w = Rgba8Writer::new(&mut buf, 4, 1, 99);
        w.write_ycbcr_row(0, &[128], &[128], &[128]).unwrap();
        assert_eq!(buf, vec![128, 128, 128, 99]);
    }
}
