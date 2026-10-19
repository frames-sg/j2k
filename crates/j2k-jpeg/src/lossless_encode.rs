// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless (SOF3) JPEG encode, as DICOM's lossless transfer syntaxes carry
//! it: any precision from 2 to 16 bits, predictors 1-7 (1 is DICOM's SV1),
//! point transforms, and restart intervals, with a Huffman table built from
//! the image's own statistics.
//!
//! Output matches libjpeg-turbo's `cjpeg -lossless ... -optimize` byte for
//! byte: grayscale frames carry a JFIF marker; color frames are coded as
//! RGB, which lossless JPEG never color-converts, with an Adobe marker and
//! component IDs `R`, `G`, `B`, sharing one Huffman table.

use alloc::vec::Vec;

use j2k_codec_math::jpeg::lossless::{
    additional_bits, difference_category, predict_at, DIFFERENCE_SYMBOLS,
};
use j2k_core::PixelFormat;

use crate::adapter::JpegBaselineHuffmanTable;
use crate::baseline_entropy::BitWriter;
use crate::encoded_frame::{
    ColorMarker, FrameComponent, FrameKind, FrameWriter, ScanComponent, ScanParameters,
};
use crate::encoded_output::JPEG_BASELINE_MAX_ENTROPY_BYTES;
use crate::encoder::{EncodedJpeg, JpegBackend, JpegEncodeError};
use crate::optimal_huffman::{optimal_huffman_spec, HuffmanFrequencies, HuffmanSpec};

mod gpu;
mod plan;

pub use self::gpu::{
    finish_jpeg_lossless_gpu_entropy, plan_jpeg_lossless_gpu_encode, JpegLosslessGpuEncodeError,
    JpegLosslessGpuEncodeParams, JpegLosslessGpuEncodePlan, JpegLosslessGpuEncodeTile,
    JPEG_LOSSLESS_GPU_CHUNK_PIXELS,
};
pub use self::plan::{plan_jpeg_lossless_encode, JpegLosslessEncodePlan};

/// Borrowed samples for lossless JPEG encode, row-major and interleaved.
#[derive(Debug, Clone, Copy)]
pub enum JpegLosslessSamples<'a> {
    /// 8-bit grayscale, precision up to 8.
    Gray8 {
        /// One sample per pixel.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// 16-bit grayscale, precision up to 16, as DICOM stores 10- to 16-bit
    /// monochrome images.
    Gray16 {
        /// One sample per pixel.
        data: &'a [u16],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// 8-bit RGB, precision up to 8.
    Rgb8 {
        /// Three samples per pixel.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// 16-bit RGB, precision up to 16.
    Rgb16 {
        /// Three samples per pixel.
        data: &'a [u16],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
}

impl JpegLosslessSamples<'_> {
    /// The pixel format of the samples.
    #[must_use]
    pub const fn pixel_format(&self) -> PixelFormat {
        match self {
            Self::Gray8 { .. } => PixelFormat::Gray8,
            Self::Gray16 { .. } => PixelFormat::Gray16,
            Self::Rgb8 { .. } => PixelFormat::Rgb8,
            Self::Rgb16 { .. } => PixelFormat::Rgb16,
        }
    }

    /// `(width, height)` in pixels.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        match *self {
            Self::Gray8 { width, height, .. }
            | Self::Gray16 { width, height, .. }
            | Self::Rgb8 { width, height, .. }
            | Self::Rgb16 { width, height, .. } => (width, height),
        }
    }

    const fn len(&self) -> usize {
        match self {
            Self::Gray8 { data, .. } | Self::Rgb8 { data, .. } => data.len(),
            Self::Gray16 { data, .. } | Self::Rgb16 { data, .. } => data.len(),
        }
    }
}

/// Options for lossless JPEG encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct JpegLosslessEncodeOptions {
    /// Sample precision `P` in bits: 2..=8 for 8-bit samples, 2..=16 for
    /// 16-bit samples. Every sample must be below `2^P`.
    pub precision: u8,
    /// Predictor selection value 1..=7. DICOM's "first-order prediction"
    /// transfer syntax (SV1) uses 1.
    pub predictor: u8,
    /// Point transform `Pt`, below `precision`. Each sample is shifted right
    /// by it before coding, so its low `Pt` bits are not preserved.
    pub point_transform: u8,
    /// Rows per restart interval, or `None` for no restart markers.
    pub restart_rows: Option<u16>,
    /// Requested encoder backend.
    pub backend: JpegBackend,
}

impl JpegLosslessEncodeOptions {
    /// SV1 (predictor 1) encode at `precision` bits with no point transform
    /// or restart markers.
    #[must_use]
    pub const fn new(precision: u8) -> Self {
        Self {
            precision,
            predictor: 1,
            point_transform: 0,
            restart_rows: None,
            backend: JpegBackend::Auto,
        }
    }
}

/// Encode samples as a lossless (SOF3) JPEG codestream.
///
/// # Errors
///
/// Returns an error for invalid dimensions, sample length, precision,
/// predictor, point transform, or restart interval; for a sample at or
/// above `2^precision`; for output above the host allocation cap; and for a
/// GPU backend, which this crate cannot run.
pub fn encode_jpeg_lossless(
    samples: JpegLosslessSamples<'_>,
    options: JpegLosslessEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    match options.backend {
        JpegBackend::Auto | JpegBackend::Cpu => encode_jpeg_lossless_cpu(samples, options),
        JpegBackend::Metal | JpegBackend::Cuda => Err(JpegEncodeError::UnsupportedBackend {
            backend: options.backend,
        }),
    }
}

fn encode_jpeg_lossless_cpu(
    samples: JpegLosslessSamples<'_>,
    options: JpegLosslessEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    let plan = plan_jpeg_lossless_encode(samples.pixel_format(), samples.dimensions(), options)?;
    if samples.len() != plan.samples() {
        return Err(JpegEncodeError::SampleLength {
            expected: plan.samples(),
            actual: samples.len(),
        });
    }
    let entropy = match samples {
        JpegLosslessSamples::Gray8 { data, .. } | JpegLosslessSamples::Rgb8 { data, .. } => {
            encode_entropy(data, &plan)?
        }
        JpegLosslessSamples::Gray16 { data, .. } | JpegLosslessSamples::Rgb16 { data, .. } => {
            encode_entropy(data, &plan)?
        }
    };
    assemble_jpeg_lossless_frame(&plan, &entropy.table, &entropy.bytes, JpegBackend::Cpu)
}

struct LosslessEntropy {
    table: JpegLosslessHuffmanTable,
    bytes: Vec<u8>,
}

/// Check the range of every sample, gather statistics, build the table, and
/// code the scan.
fn encode_entropy<T: Copy + Into<u16>>(
    data: &[T],
    plan: &JpegLosslessEncodePlan,
) -> Result<LosslessEntropy, JpegEncodeError> {
    if let Some((index, value)) = data
        .iter()
        .map(|&sample| sample.into())
        .enumerate()
        .find(|&(_, value)| u32::from(value) > plan.max_sample())
    {
        return Err(JpegEncodeError::SampleOutOfRange {
            index,
            value,
            precision: plan.precision,
        });
    }
    let mut histogram = Histogram([0; DIFFERENCE_SYMBOLS]);
    for_each_difference(data, plan, &mut histogram)?;
    let table = jpeg_lossless_huffman_table(&histogram.0)?;
    let codes = table.codes();
    let writer = BitWriter::try_with_capacity(
        entropy_size_estimate(plan, &histogram.0, &codes),
        JPEG_BASELINE_MAX_ENTROPY_BYTES,
    )?;
    let mut sink = EntropySink {
        writer,
        codes: &codes,
        restarts: 0,
    };
    for_each_difference(data, plan, &mut sink)?;
    Ok(LosslessEntropy {
        bytes: sink.writer.into_bytes()?,
        table,
    })
}

/// Receives each sample's difference category, in scan order.
trait DifferenceSink {
    /// A restart interval ends before the next difference.
    fn restart(&mut self) -> Result<(), JpegEncodeError>;
    fn difference(&mut self, category: u8, bits: u16) -> Result<(), JpegEncodeError>;
}

struct Histogram([u64; DIFFERENCE_SYMBOLS]);

impl DifferenceSink for Histogram {
    fn restart(&mut self) -> Result<(), JpegEncodeError> {
        Ok(())
    }

    fn difference(&mut self, category: u8, _bits: u16) -> Result<(), JpegEncodeError> {
        let count = &mut self.0[usize::from(category)];
        *count = count.saturating_add(1);
        Ok(())
    }
}

struct EntropySink<'t> {
    writer: BitWriter,
    codes: &'t JpegBaselineHuffmanTable,
    restarts: u8,
}

impl DifferenceSink for EntropySink<'_> {
    fn restart(&mut self) -> Result<(), JpegEncodeError> {
        self.writer.write_restart_marker(0xD0 + self.restarts)?;
        self.restarts = (self.restarts + 1) % 8;
        Ok(())
    }

    fn difference(&mut self, category: u8, bits: u16) -> Result<(), JpegEncodeError> {
        self.writer.write_huffman_value(
            self.codes,
            category,
            u32::from(bits),
            additional_bits(category),
        )
    }
}

/// Predict and difference every sample in scan order: rows top to bottom,
/// pixels left to right, components interleaved. The first row of the image
/// and of each restart interval uses the 1-D predictor.
fn for_each_difference<T: Copy + Into<u16>, S: DifferenceSink>(
    data: &[T],
    plan: &JpegLosslessEncodePlan,
    sink: &mut S,
) -> Result<(), JpegEncodeError> {
    let components = usize::from(plan.components);
    let stride = plan.width as usize * components;
    let shift = plan.point_transform;
    let value = |sample: T| sample.into() >> shift;
    let mut above: Option<&[T]> = None;
    for (y, row) in data.chunks_exact(stride).enumerate() {
        let row_in_interval = match plan.restart_rows {
            0 => y,
            rows => y % rows as usize,
        };
        if y > 0 && row_in_interval == 0 {
            sink.restart()?;
        }
        for (index, &sample) in row.iter().enumerate() {
            let x = index / components;
            let left = index.checked_sub(components);
            let ra = left.map_or(0, |left| value(row[left]));
            let (rb, rc) = above.map_or((0, 0), |above| {
                (
                    value(above[index]),
                    left.map_or(0, |left| value(above[left])),
                )
            });
            let prediction = predict_at(
                plan.predictor,
                plan.initial_predictor,
                x,
                row_in_interval,
                [ra, rb, rc],
            );
            let (category, bits) = difference_category(value(sample).wrapping_sub(prediction));
            sink.difference(category, bits)?;
        }
        above = Some(row);
    }
    Ok(())
}

/// Entropy bytes the scan will take, with a margin for byte stuffing; the
/// writer grows past it if a scan needs more.
fn entropy_size_estimate(
    plan: &JpegLosslessEncodePlan,
    histogram: &[u64; DIFFERENCE_SYMBOLS],
    codes: &JpegBaselineHuffmanTable,
) -> usize {
    let bits: u64 = histogram
        .iter()
        .zip(0_u8..)
        .map(|(&count, category)| {
            count.saturating_mul(u64::from(
                codes.lens[usize::from(category)] + additional_bits(category),
            ))
        })
        .fold(0, u64::saturating_add);
    let bytes = usize::try_from(bits.div_ceil(8)).unwrap_or(usize::MAX);
    // Restart markers, padding, and about one stuffed byte per 256.
    let overhead = plan.restart_intervals().saturating_mul(3) + bytes / 256 + 16;
    bytes
        .saturating_add(overhead)
        .min(JPEG_BASELINE_MAX_ENTROPY_BYTES)
}

/// The Huffman table of one lossless scan, built from its difference
/// categories.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegLosslessHuffmanTable {
    spec: HuffmanSpec,
}

impl JpegLosslessHuffmanTable {
    /// Code and length of every category, for device entropy coding.
    #[must_use]
    pub fn codes(&self) -> JpegBaselineHuffmanTable {
        self.spec.table()
    }
}

/// Build the optimal table for a histogram of difference categories 0..=16,
/// as `cjpeg -optimize` does.
///
/// # Errors
/// Returns an invariant error when the histogram is empty.
#[doc(hidden)]
pub fn jpeg_lossless_huffman_table(
    histogram: &[u64; DIFFERENCE_SYMBOLS],
) -> Result<JpegLosslessHuffmanTable, JpegEncodeError> {
    let mut frequencies = HuffmanFrequencies::new();
    for (&count, category) in histogram.iter().zip(0_u8..) {
        frequencies.add_count(category, count);
    }
    Ok(JpegLosslessHuffmanTable {
        spec: optimal_huffman_spec(&frequencies)?,
    })
}

/// Assemble a complete lossless codestream around entropy-coded data.
///
/// # Errors
/// Returns an error when a segment does not fit its length field or the
/// frame exceeds the host allocation cap.
#[doc(hidden)]
pub fn assemble_jpeg_lossless_frame(
    plan: &JpegLosslessEncodePlan,
    table: &JpegLosslessHuffmanTable,
    entropy: &[u8],
    backend: JpegBackend,
) -> Result<EncodedJpeg, JpegEncodeError> {
    let components = usize::from(plan.components);
    let rgb = components == 3;
    let ids: [u8; 3] = if rgb { *b"RGB" } else { [1, 0, 0] };
    let mut frame_components = [FrameComponent {
        id: 0,
        h: 1,
        v: 1,
        quant_table: 0,
    }; 3];
    let mut scan_components = [ScanComponent {
        id: 0,
        dc_table: 0,
        ac_table: 0,
    }; 3];
    for ((frame_component, scan_component), id) in frame_components
        .iter_mut()
        .zip(&mut scan_components)
        .zip(ids)
    {
        frame_component.id = id;
        scan_component.id = id;
    }
    let mut frame = FrameWriter::new(entropy.len())?;
    frame.color_marker(if rgb {
        ColorMarker::AdobeUntransformed
    } else {
        ColorMarker::Jfif
    })?;
    frame.sof(
        FrameKind::Lossless,
        plan.precision,
        (plan.width, plan.height),
        &frame_components[..components],
    )?;
    frame.dht(0, 0, &table.spec)?;
    if let Some(interval) = plan.restart_interval_mcus() {
        frame.dri(interval)?;
    }
    frame.sos(
        &scan_components[..components],
        ScanParameters {
            ss: plan.predictor,
            se: 0,
            ah: 0,
            al: plan.point_transform,
        },
    )?;
    frame.finish(entropy, backend)
}

#[cfg(test)]
mod tests;
