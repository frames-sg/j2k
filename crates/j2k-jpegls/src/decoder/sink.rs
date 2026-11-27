// SPDX-License-Identifier: MIT OR Apache-2.0

//! Destinations for decoded lines: interleaved samples in a byte buffer (16-bit
//! samples little-endian) or in a `u16` buffer.

use crate::error::JpegLsError;

/// Stores one decoded line of one component into interleaved output.
pub(crate) trait LineSink {
    fn store(&mut self, line: u32, component: usize, samples: &[i32]) -> Result<(), JpegLsError>;
}

/// Interleaved samples in bytes: one per sample, or two little-endian.
pub(crate) struct ByteSink<'o> {
    pub(crate) out: &'o mut [u8],
    pub(crate) stride: usize,
    pub(crate) components: usize,
    pub(crate) wide: bool,
}

impl LineSink for ByteSink<'_> {
    fn store(&mut self, line: u32, component: usize, samples: &[i32]) -> Result<(), JpegLsError> {
        let sample_bytes = if self.wide { 2 } else { 1 };
        let pixel_bytes = self.components * sample_bytes;
        let start = (line as usize) * self.stride;
        let end = start + samples.len() * pixel_bytes;
        let actual = self.out.len();
        let row = self
            .out
            .get_mut(start..end)
            .ok_or(JpegLsError::OutputBufferTooSmall {
                required: end,
                actual,
            })?;
        let offset = component * sample_bytes;
        for (pixel, &sample) in row.chunks_exact_mut(pixel_bytes).zip(samples) {
            // Samples are clamped to MAXVAL, which fits the output width.
            let value = u16::try_from(sample).unwrap_or(u16::MAX);
            if self.wide {
                pixel[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            } else {
                pixel[offset] = u8::try_from(value).unwrap_or(u8::MAX);
            }
        }
        Ok(())
    }
}

/// Interleaved 16-bit samples, tightly packed.
pub(crate) struct WordSink<'o> {
    pub(crate) out: &'o mut [u16],
    pub(crate) components: usize,
}

impl LineSink for WordSink<'_> {
    fn store(&mut self, line: u32, component: usize, samples: &[i32]) -> Result<(), JpegLsError> {
        let row_len = samples.len() * self.components;
        let start = (line as usize) * row_len;
        let actual = self.out.len() * 2;
        let row =
            self.out
                .get_mut(start..start + row_len)
                .ok_or(JpegLsError::OutputBufferTooSmall {
                    required: (start + row_len) * 2,
                    actual,
                })?;
        for (pixel, &sample) in row.chunks_exact_mut(self.components).zip(samples) {
            pixel[component] = u16::try_from(sample).unwrap_or(u16::MAX);
        }
        Ok(())
    }
}
