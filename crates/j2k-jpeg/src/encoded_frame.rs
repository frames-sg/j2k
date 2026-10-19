// SPDX-License-Identifier: MIT OR Apache-2.0

//! Marker-segment assembly for frames whose tables are chosen per image:
//! lossless, 12-bit extended, and Huffman-optimized baseline JPEG.
//!
//! Segments are written in libjpeg's order (SOI, `APPn`, DQT, SOF, DHT, DRI,
//! SOS) so that output can be compared byte for byte with libjpeg-turbo.

use crate::encoded_output::CappedBytes;
use crate::encoder::{EncodedJpeg, JpegBackend, JpegEncodeError};
use crate::optimal_huffman::HuffmanSpec;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

/// Bytes every marker segment other than the entropy-coded data can need:
/// four 16-bit DQT tables, four full DHT tables, and the remaining headers.
const HEADER_CAPACITY: usize = 4 * 133 + 4 * 277 + 256;

/// SOF marker codes this module writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameKind {
    /// SOF0, baseline sequential DCT.
    Baseline,
    /// SOF1, extended sequential DCT, Huffman coded.
    ExtendedSequential,
    /// SOF3, lossless sequential, Huffman coded.
    Lossless,
}

impl FrameKind {
    const fn marker(self) -> u8 {
        match self {
            Self::Baseline => 0xC0,
            Self::ExtendedSequential => 0xC1,
            Self::Lossless => 0xC3,
        }
    }
}

/// One component's SOF entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameComponent {
    pub(crate) id: u8,
    pub(crate) h: u8,
    pub(crate) v: u8,
    pub(crate) quant_table: u8,
}

/// One component's SOS entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScanComponent {
    pub(crate) id: u8,
    pub(crate) dc_table: u8,
    pub(crate) ac_table: u8,
}

/// The SOS spectral and approximation parameters: for lossless frames, `ss`
/// is the predictor and `al` the point transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScanParameters {
    pub(crate) ss: u8,
    pub(crate) se: u8,
    pub(crate) ah: u8,
    pub(crate) al: u8,
}

/// Markers that tell decoders how to read the color components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorMarker {
    /// JFIF APP0, as libjpeg writes for grayscale and YCbCr.
    Jfif,
    /// Adobe APP14 with transform 0: components are stored untransformed.
    AdobeUntransformed,
}

/// Builder for one complete JPEG codestream.
pub(crate) struct FrameWriter {
    out: CappedBytes,
}

impl FrameWriter {
    /// Start a frame that will carry `entropy_len` entropy-coded bytes.
    pub(crate) fn new(entropy_len: usize) -> Result<Self, JpegEncodeError> {
        let capacity =
            entropy_len
                .checked_add(HEADER_CAPACITY)
                .ok_or(JpegEncodeError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })?;
        let mut out = CappedBytes::try_with_capacity(capacity, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
        write_marker(&mut out, 0xD8)?;
        Ok(Self { out })
    }

    pub(crate) fn color_marker(&mut self, marker: ColorMarker) -> Result<(), JpegEncodeError> {
        match marker {
            ColorMarker::Jfif => write_segment(
                &mut self.out,
                0xE0,
                // Version 1.01, no units, 1:1 density, no thumbnail.
                b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00",
                "APP0",
            ),
            ColorMarker::AdobeUntransformed => write_segment(
                &mut self.out,
                0xEE,
                // Version 100, no flags, transform 0.
                b"Adobe\0\x64\x00\x00\x00\x00\x00",
                "APP14",
            ),
        }
    }

    /// Write a quantization table given in natural order; 16-bit precision
    /// (`Pq = 1`) is used only when an entry exceeds 255.
    pub(crate) fn dqt(&mut self, table_id: u8, natural: &[u16; 64]) -> Result<(), JpegEncodeError> {
        let wide = natural.iter().any(|&value| value > 255);
        write_segment_header(&mut self.out, 0xDB, if wide { 129 } else { 65 }, "DQT")?;
        self.out.push((u8::from(wide) << 4) | table_id)?;
        for &natural_index in &j2k_codec_math::jpeg::ZIGZAG {
            let value = natural[usize::from(natural_index)];
            if wide {
                self.out.extend_from_slice(&value.to_be_bytes())?;
            } else {
                self.out.push(u8::try_from(value).unwrap_or(u8::MAX))?;
            }
        }
        Ok(())
    }

    pub(crate) fn sof(
        &mut self,
        kind: FrameKind,
        precision: u8,
        (width, height): (u32, u32),
        components: &[FrameComponent],
    ) -> Result<(), JpegEncodeError> {
        let too_large = JpegEncodeError::DimensionsTooLarge { width, height };
        let (Ok(width16), Ok(height16)) = (u16::try_from(width), u16::try_from(height)) else {
            return Err(too_large);
        };
        if width16 == 0 || height16 == 0 {
            return Err(JpegEncodeError::EmptyDimensions);
        }
        let count =
            u8::try_from(components.len()).map_err(|_| JpegEncodeError::InternalInvariant {
                reason: "JPEG frame component count exceeds u8",
            })?;
        write_segment_header(
            &mut self.out,
            kind.marker(),
            6 + components.len() * 3,
            "SOF",
        )?;
        self.out.push(precision)?;
        self.out.extend_from_slice(&height16.to_be_bytes())?;
        self.out.extend_from_slice(&width16.to_be_bytes())?;
        self.out.push(count)?;
        for component in components {
            self.out.push(component.id)?;
            self.out.push((component.h << 4) | component.v)?;
            self.out.push(component.quant_table)?;
        }
        Ok(())
    }

    /// Write one Huffman table; `class` is 0 for DC and lossless tables, 1
    /// for AC tables.
    pub(crate) fn dht(
        &mut self,
        class: u8,
        table_id: u8,
        spec: &HuffmanSpec,
    ) -> Result<(), JpegEncodeError> {
        write_segment_header(&mut self.out, 0xC4, 17 + spec.values().len(), "DHT")?;
        self.out.push((class << 4) | table_id)?;
        self.out.extend_from_slice(&spec.bits)?;
        self.out.extend_from_slice(spec.values())
    }

    pub(crate) fn dri(&mut self, restart_interval: u16) -> Result<(), JpegEncodeError> {
        write_segment(&mut self.out, 0xDD, &restart_interval.to_be_bytes(), "DRI")
    }

    pub(crate) fn sos(
        &mut self,
        components: &[ScanComponent],
        parameters: ScanParameters,
    ) -> Result<(), JpegEncodeError> {
        let count =
            u8::try_from(components.len()).map_err(|_| JpegEncodeError::InternalInvariant {
                reason: "JPEG scan component count exceeds u8",
            })?;
        write_segment_header(&mut self.out, 0xDA, 4 + components.len() * 2, "SOS")?;
        self.out.push(count)?;
        for component in components {
            self.out.push(component.id)?;
            self.out
                .push((component.dc_table << 4) | component.ac_table)?;
        }
        self.out.push(parameters.ss)?;
        self.out.push(parameters.se)?;
        self.out.push((parameters.ah << 4) | parameters.al)
    }

    /// Append the entropy-coded data, end the image, and return it.
    pub(crate) fn finish(
        mut self,
        entropy: &[u8],
        backend: JpegBackend,
    ) -> Result<EncodedJpeg, JpegEncodeError> {
        self.out.extend_from_slice(entropy)?;
        write_marker(&mut self.out, 0xD9)?;
        Ok(EncodedJpeg {
            data: self.out.into_vec(),
            backend,
        })
    }
}

pub(crate) fn write_marker(out: &mut CappedBytes, marker: u8) -> Result<(), JpegEncodeError> {
    out.push(0xFF)?;
    out.push(marker)
}

pub(crate) fn write_segment(
    out: &mut CappedBytes,
    marker: u8,
    payload: &[u8],
    name: &'static str,
) -> Result<(), JpegEncodeError> {
    write_segment_header(out, marker, payload.len(), name)?;
    out.extend_from_slice(payload)
}

pub(crate) fn write_segment_header(
    out: &mut CappedBytes,
    marker: u8,
    payload_len: usize,
    name: &'static str,
) -> Result<(), JpegEncodeError> {
    let len = payload_len
        .checked_add(2)
        .and_then(|value| u16::try_from(value).ok())
        .ok_or(JpegEncodeError::SegmentTooLarge { name })?;
    write_marker(out, marker)?;
    out.extend_from_slice(&len.to_be_bytes())?;
    Ok(())
}
