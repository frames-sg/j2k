// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared marker-segment serialization and sequential DCT frame assembly.
//!
//! DCT frames retain the baseline encoder's DQT, DRI, SOF, DHT, SOS order.
//! Lossless callers choose their APP metadata and place DRI after DHT.

use crate::adapter::JpegBaselineSampling;
use crate::encoded_output::{checked_jpeg_baseline_frame_capacity, CappedBytes};
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

/// Borrowed code counts and symbols, in JPEG DHT order.
#[derive(Clone, Copy)]
pub(crate) struct HuffmanTable<'a> {
    pub(crate) bits: &'a [u8; 16],
    pub(crate) values: &'a [u8],
}

impl<'a> From<&'a HuffmanSpec> for HuffmanTable<'a> {
    fn from(spec: &'a HuffmanSpec) -> Self {
        Self {
            bits: &spec.bits,
            values: spec.values(),
        }
    }
}

/// Explicit tables and format for one sequential DCT frame.
#[derive(Clone, Copy)]
pub(crate) struct DctFrame<'a> {
    pub(crate) kind: FrameKind,
    pub(crate) precision: u8,
    pub(crate) dimensions: (u32, u32),
    pub(crate) sampling: JpegBaselineSampling,
    pub(crate) quant: &'a [[u16; 64]; 2],
    /// Each used table set contains its DC and AC tables, in that order.
    pub(crate) huffman: [Option<[HuffmanTable<'a>; 2]>; 2],
    pub(crate) restart_interval: Option<u16>,
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
        Self::with_capacity(capacity, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    /// Preserve the smaller capacity charged by baseline CPU/GPU admission.
    pub(crate) fn baseline(entropy_len: usize) -> Result<Self, JpegEncodeError> {
        let capacity = checked_jpeg_baseline_frame_capacity(entropy_len)?;
        Self::with_capacity(capacity, capacity)
    }

    fn with_capacity(capacity: usize, max_len: usize) -> Result<Self, JpegEncodeError> {
        let mut out = CappedBytes::try_with_capacity(capacity, max_len)?;
        write_marker(&mut out, 0xD8)?;
        Ok(Self { out })
    }

    /// Write a complete sequential DCT frame using the supplied tables.
    pub(crate) fn assemble_dct(
        mut self,
        frame: DctFrame<'_>,
        entropy: &[u8],
        backend: JpegBackend,
    ) -> Result<EncodedJpeg, JpegEncodeError> {
        let count = usize::from(frame.sampling.components);
        self.dqt(0, &frame.quant[0])?;
        if count == 3 {
            self.dqt(1, &frame.quant[1])?;
        }
        if let Some(interval) = frame.restart_interval {
            self.dri(interval)?;
        }
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
        for (index, (component, scan)) in frame_components
            .iter_mut()
            .zip(&mut scan_components)
            .enumerate()
            .take(count)
        {
            let id = u8::try_from(index + 1).unwrap_or(u8::MAX);
            let table = u8::from(index != 0);
            *component = FrameComponent {
                id,
                h: frame.sampling.h[index],
                v: frame.sampling.v[index],
                quant_table: table,
            };
            *scan = ScanComponent {
                id,
                dc_table: table,
                ac_table: table,
            };
        }
        self.sof(
            frame.kind,
            frame.precision,
            frame.dimensions,
            &frame_components[..count],
        )?;
        for (table_id, tables) in (0_u8..).zip(frame.huffman) {
            if let Some([dc, ac]) = tables {
                self.dht(0, table_id, dc)?;
                self.dht(1, table_id, ac)?;
            }
        }
        self.sos(
            &scan_components[..count],
            ScanParameters {
                ss: 0,
                se: 63,
                ah: 0,
                al: 0,
            },
        )?;
        self.finish(entropy, backend)
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
        table: HuffmanTable<'_>,
    ) -> Result<(), JpegEncodeError> {
        write_segment_header(&mut self.out, 0xC4, 17 + table.values.len(), "DHT")?;
        self.out.push((class << 4) | table_id)?;
        self.out.extend_from_slice(table.bits)?;
        self.out.extend_from_slice(table.values)
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

fn write_marker(out: &mut CappedBytes, marker: u8) -> Result<(), JpegEncodeError> {
    out.push(0xFF)?;
    out.push(marker)
}

fn write_segment(
    out: &mut CappedBytes,
    marker: u8,
    payload: &[u8],
    name: &'static str,
) -> Result<(), JpegEncodeError> {
    write_segment_header(out, marker, payload.len(), name)?;
    out.extend_from_slice(payload)
}

fn write_segment_header(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::baseline_encode_tables;
    use crate::encoder::{JpegEncodeOptions, JpegSubsampling};
    use crate::info::SofKind;
    use crate::parse::{header::parse_header, markers::MarkerWalker};
    use crate::standard_huffman::{
        STD_LUMA_AC_BITS, STD_LUMA_AC_VALUES, STD_LUMA_DC_BITS, STD_LUMA_DC_VALUES,
    };

    #[test]
    fn dct_frames_preserve_table_widths_precision_restart_and_marker_order() {
        let sampling = baseline_encode_tables(JpegEncodeOptions {
            subsampling: JpegSubsampling::Gray,
            ..JpegEncodeOptions::default()
        })
        .expect("gray sampling")
        .sampling;
        let entropy = [0x12, 0xff, 0x00];
        for (kind, precision, largest_quant, sof) in [
            (FrameKind::Baseline, 8, 255, SofKind::Baseline8),
            (FrameKind::ExtendedSequential, 12, 256, SofKind::Extended12),
        ] {
            let mut quant = [
                core::array::from_fn(|index| u16::try_from(index + 1).unwrap()),
                [0; 64],
            ];
            quant[0][17] = largest_quant;
            let writer = if kind == FrameKind::Baseline {
                FrameWriter::baseline(entropy.len())
            } else {
                FrameWriter::new(entropy.len())
            }
            .expect("frame allocation");
            let encoded = writer
                .assemble_dct(
                    DctFrame {
                        kind,
                        precision,
                        dimensions: (17, 9),
                        sampling,
                        quant: &quant,
                        huffman: [
                            Some([
                                HuffmanTable {
                                    bits: &STD_LUMA_DC_BITS,
                                    values: &STD_LUMA_DC_VALUES,
                                },
                                HuffmanTable {
                                    bits: &STD_LUMA_AC_BITS,
                                    values: &STD_LUMA_AC_VALUES,
                                },
                            ]),
                            None,
                        ],
                        restart_interval: Some(7),
                    },
                    &entropy,
                    JpegBackend::Cuda,
                )
                .expect("assemble DCT frame");
            assert_eq!(encoded.backend, JpegBackend::Cuda);
            assert_eq!(encoded.data[6], if largest_quant > 255 { 0x10 } else { 0 });
            assert!(encoded.data.ends_with(&[0x12, 0xff, 0x00, 0xff, 0xd9]));
            let header = parse_header(&encoded.data).expect("parse assembled frame");
            assert_eq!(header.sof_kind, sof);
            assert_eq!(header.bit_depth, precision);
            assert_eq!(header.dimensions, (17, 9));
            assert_eq!(header.restart_interval, Some(7));
            assert_eq!(
                header.quant_tables.active(0),
                Some(&j2k_codec_math::jpeg::ZIGZAG.map(|index| quant[0][usize::from(index)]))
            );
            let mut walker = MarkerWalker::new(&encoded.data);
            walker.read_soi().expect("SOI");
            let mut markers = alloc::vec::Vec::new();
            while let Some(marker) = walker.next_marker().expect("marker") {
                markers.push(marker.code);
                if marker.code == 0xda {
                    break;
                }
            }
            assert_eq!(markers, [0xdb, 0xdd, kind.marker(), 0xc4, 0xc4, 0xda]);
            if kind == FrameKind::Baseline {
                assert!(
                    encoded.data.capacity()
                        <= checked_jpeg_baseline_frame_capacity(entropy.len()).unwrap()
                );
            }
        }
    }
}
