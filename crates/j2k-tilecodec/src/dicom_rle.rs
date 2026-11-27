// SPDX-License-Identifier: MIT OR Apache-2.0

//! DICOM RLE Lossless frames (transfer syntax 1.2.840.10008.1.2.5, DICOM
//! PS3.5 Annex G).

mod decode;
mod encode;

use crate::{error::unsupported, TileCodecError};
use j2k_core::BufferError;

/// Length of the RLE header that starts every frame.
const HEADER_LEN: usize = 64;
/// Most segments an RLE header can describe.
const MAX_SEGMENTS: usize = 15;

/// Arrangement of the samples of multi-sample pixels in a native
/// (uncompressed) frame buffer, as in Planar Configuration (0028,0006).
///
/// The RLE stream itself always stores each sample in its own segments,
/// color-by-plane, whatever the data set's Planar Configuration says
/// (PS3.5 Section 8.2.2 and Annex G.2). In an RLE data set the attribute
/// describes the native layout of the pixel data once it is decompressed;
/// Annex G.2 recommends 0 unless the IOD requires otherwise. Choose
/// [`Interleaved`](Self::Interleaved) for a pixel array, which is also what
/// pydicom returns for RLE frames whatever the attribute says, or the
/// variant matching the data set's Planar Configuration when the decoded
/// frame replaces encapsulated Pixel Data that keeps the attribute. With
/// one sample per pixel both variants give the same bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DicomRlePlanarConfiguration {
    /// Planar Configuration 0, color-by-pixel: the samples of each pixel are
    /// adjacent (R1 G1 B1 R2 G2 B2 …).
    Interleaved,
    /// Planar Configuration 1, color-by-plane: each sample forms a
    /// contiguous plane (R1 R2 … G1 G2 … B1 B2 …).
    Planar,
}

/// Geometry and sample format of one DICOM RLE frame, and the layout of its
/// native samples.
///
/// Native samples are `bits_allocated / 8` bytes each, little-endian, and
/// rows run top to bottom. Build one with [`DicomRleLayout::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DicomRleLayout {
    rows: u16,
    columns: u16,
    samples_per_pixel: u16,
    bytes_per_sample: u8,
    segment_count: u8,
    planar_configuration: DicomRlePlanarConfiguration,
    frame_len: usize,
}

impl DicomRleLayout {
    /// Describe a frame from the data set's Rows (0028,0010), Columns
    /// (0028,0011), Samples per Pixel (0028,0002) and Bits Allocated
    /// (0028,0100), and the native sample layout the caller wants.
    ///
    /// # Errors
    ///
    /// Returns [`TileCodecError::Unsupported`] when rows, columns or samples
    /// per pixel is zero, when bits allocated is not 8, 16 or 32 (1-bit
    /// frames and other widths are not supported), or when the frame would
    /// need more than the 15 segments an RLE header can describe (samples
    /// per pixel × bytes per sample). Returns
    /// [`BufferError::SizeOverflow`] when the frame length does not fit in
    /// `usize`.
    pub fn new(
        rows: u16,
        columns: u16,
        samples_per_pixel: u16,
        bits_allocated: u16,
        planar_configuration: DicomRlePlanarConfiguration,
    ) -> Result<Self, TileCodecError> {
        if rows == 0 || columns == 0 {
            return Err(unsupported("DICOM RLE frame with zero rows or columns"));
        }
        if samples_per_pixel == 0 {
            return Err(unsupported("DICOM RLE frame with zero samples per pixel"));
        }
        let bytes_per_sample = match bits_allocated {
            8 => 1_u8,
            16 => 2,
            32 => 4,
            _ => {
                return Err(unsupported(
                    "DICOM RLE bits allocated other than 8, 16 or 32",
                ))
            }
        };
        let segment_count = u32::from(samples_per_pixel) * u32::from(bytes_per_sample);
        let segment_count = u8::try_from(segment_count)
            .ok()
            .filter(|&count| usize::from(count) <= MAX_SEGMENTS)
            .ok_or_else(|| unsupported("DICOM RLE frame needing more than 15 segments"))?;
        let frame_len = usize::from(rows)
            .checked_mul(usize::from(columns))
            .and_then(|pixels| pixels.checked_mul(usize::from(segment_count)))
            .ok_or(TileCodecError::Buffer(BufferError::SizeOverflow {
                what: "DICOM RLE frame length",
            }))?;
        Ok(Self {
            rows,
            columns,
            samples_per_pixel,
            bytes_per_sample,
            segment_count,
            planar_configuration,
            frame_len,
        })
    }

    /// Number of rows.
    pub const fn rows(&self) -> u16 {
        self.rows
    }

    /// Number of columns.
    pub const fn columns(&self) -> u16 {
        self.columns
    }

    /// Samples per pixel.
    pub const fn samples_per_pixel(&self) -> u16 {
        self.samples_per_pixel
    }

    /// Bits allocated per sample: 8, 16 or 32.
    pub fn bits_allocated(&self) -> u16 {
        u16::from(self.bytes_per_sample) * 8
    }

    /// Layout of the native samples.
    pub const fn planar_configuration(&self) -> DicomRlePlanarConfiguration {
        self.planar_configuration
    }

    /// Number of RLE segments in the frame: samples per pixel × bytes per
    /// sample.
    pub fn segment_count(&self) -> usize {
        usize::from(self.segment_count)
    }

    /// Length in bytes of the native frame: rows × columns × samples per
    /// pixel × bytes per sample.
    pub const fn frame_len(&self) -> usize {
        self.frame_len
    }

    fn pixels(&self) -> usize {
        usize::from(self.rows) * usize::from(self.columns)
    }

    /// Where segment `segment` lives in the native frame: the index of its
    /// first byte and the distance between its consecutive bytes.
    ///
    /// Segments go by sample, then by byte significance, most significant
    /// first (PS3.5 Annex G.2). Native samples are little-endian, so the most
    /// significant byte is the last byte of each sample.
    fn segment_lane(&self, segment: usize) -> (usize, usize) {
        let bytes = usize::from(self.bytes_per_sample);
        let sample = segment / bytes;
        let byte_in_sample = bytes - 1 - segment % bytes;
        match self.planar_configuration {
            DicomRlePlanarConfiguration::Interleaved => (
                sample * bytes + byte_in_sample,
                usize::from(self.samples_per_pixel) * bytes,
            ),
            DicomRlePlanarConfiguration::Planar => {
                (sample * self.pixels() * bytes + byte_in_sample, bytes)
            }
        }
    }
}

/// What [`DicomRleCodec::decode_frame_lenient`] had to accept to decode a
/// frame.
///
/// A report from a conformant frame is [`is_conformant`](Self::is_conformant).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[must_use = "a lenient decode can accept non-conformant data; check the report"]
pub struct DicomRleDecodeReport {
    overlong_segments: u16,
    nonzero_unused_offsets: bool,
}

impl DicomRleDecodeReport {
    /// True when the frame decoded without any of the leniency below, so
    /// [`DicomRleCodec::decode_frame`] would have accepted it too.
    pub const fn is_conformant(&self) -> bool {
        self.overlong_segments == 0 && !self.nonzero_unused_offsets
    }

    /// Segments whose data continued after their rows were complete, as a
    /// bit mask: bit `i` is set for segment `i`, counting from zero in header
    /// order. Such a segment decoded to more than rows × columns bytes, or
    /// ended with bytes that are not padding; the extra data was discarded.
    pub const fn overlong_segments(&self) -> u16 {
        self.overlong_segments
    }

    /// True when header offsets after the last segment were not zero.
    pub const fn has_nonzero_unused_offsets(&self) -> bool {
        self.nonzero_unused_offsets
    }
}

/// DICOM RLE Lossless frame decoder and encoder (DICOM PS3.5 Annex G).
///
/// A frame is a 64-byte header (the segment count and up to 15 segment
/// offsets, little-endian `u32`) followed by one PackBits-coded segment per
/// byte of each sample. The caller supplies the frame layout from the data
/// set and buffers of exactly [`DicomRleLayout::frame_len`] bytes; nothing
/// is allocated from counts read from the frame.
///
/// ```
/// use j2k_tilecodec::{DicomRleCodec, DicomRleLayout, DicomRlePlanarConfiguration};
///
/// let layout = DicomRleLayout::new(2, 3, 1, 16, DicomRlePlanarConfiguration::Interleaved)?;
/// let samples: Vec<u8> = [100_u16, 100, 100, 7, 8, 9]
///     .iter()
///     .flat_map(|sample| sample.to_le_bytes())
///     .collect();
///
/// let mut frame = Vec::new();
/// DicomRleCodec::encode_frame(&samples, layout, &mut frame)?;
///
/// let mut decoded = vec![0_u8; layout.frame_len()];
/// DicomRleCodec::decode_frame(&frame, layout, &mut decoded)?;
/// assert_eq!(decoded, samples);
/// # Ok::<(), j2k_tilecodec::TileCodecError>(())
/// ```
pub struct DicomRleCodec;

impl DicomRleCodec {
    /// Decode one RLE frame into `out`, which must be exactly
    /// [`DicomRleLayout::frame_len`] bytes, as native little-endian samples
    /// in the layout's planar configuration.
    ///
    /// Decoding is strict. The header must hold 1 to 15 segments, exactly as
    /// many as the layout needs; its first offset must be 64, the offsets
    /// must increase and stay inside `input`, and unused offsets must be
    /// zero. Each segment must decode to exactly rows × columns bytes. After
    /// that it may hold -128 no-op codes and then one zero padding byte, and
    /// nothing else. Segment lengths need not be even. Use
    /// [`decode_frame_lenient`](Self::decode_frame_lenient) to read frames
    /// that break the segment rules the way some encoders do.
    ///
    /// # Errors
    ///
    /// - [`TileCodecError::LengthMismatch`] when `out` is not
    ///   [`DicomRleLayout::frame_len`] bytes.
    /// - [`TileCodecError::Input`] when the frame is truncated: shorter than
    ///   its header, an offset past its end, or a last segment that ends
    ///   before its rows are complete.
    /// - [`TileCodecError::Malformed`] for any other header or segment
    ///   violation, including an earlier segment that ends early and a run
    ///   that continues past the end of its segment's rows.
    ///
    /// On error the contents of `out` are unspecified.
    pub fn decode_frame(
        input: &[u8],
        layout: DicomRleLayout,
        out: &mut [u8],
    ) -> Result<(), TileCodecError> {
        decode::decode_frame(input, layout, out, decode::Conformance::Strict).map(|_| ())
    }

    /// Decode one RLE frame like [`decode_frame`](Self::decode_frame), but
    /// accept the non-conformant data that pydicom's RLE handler accepts,
    /// and report it.
    ///
    /// A segment whose data continues after its rows are complete is
    /// accepted: a run that decodes past the end of the rows, further runs,
    /// trailing bytes that are not padding, or a literal run cut off by the
    /// end of the segment after it has filled the rows. The extra data is
    /// discarded, and the segment is listed in
    /// [`DicomRleDecodeReport::overlong_segments`]. Nonzero unused header
    /// offsets are ignored and reported. Everything else, including a
    /// segment that decodes short, is still an error.
    ///
    /// # Errors
    ///
    /// The same as [`decode_frame`](Self::decode_frame), less the cases
    /// above.
    pub fn decode_frame_lenient(
        input: &[u8],
        layout: DicomRleLayout,
        out: &mut [u8],
    ) -> Result<DicomRleDecodeReport, TileCodecError> {
        decode::decode_frame(input, layout, out, decode::Conformance::Lenient)
    }

    /// Encode one frame of native little-endian samples, in the layout's
    /// planar configuration, as a DICOM RLE frame in `out`.
    ///
    /// `out` is cleared first. The encoder follows PS3.5 Annex G.3: each row
    /// is encoded on its own, so no run crosses a row boundary; every run of
    /// two or more equal bytes becomes a replicate run, and other bytes form
    /// literal runs of up to 128 bytes; each segment is padded to an even
    /// length with a zero byte; the header lists the segment offsets and
    /// leaves the unused ones zero. The output is the same, byte for byte,
    /// as pydicom 2.4's RLE encoder.
    ///
    /// # Errors
    ///
    /// - [`TileCodecError::LengthMismatch`] when `samples` is not
    ///   [`DicomRleLayout::frame_len`] bytes.
    /// - [`BufferError::HostAllocationFailed`] when `out` cannot grow.
    /// - [`BufferError::SizeOverflow`] when a segment offset does not fit in
    ///   the header's 32-bit fields.
    ///
    /// On error the contents of `out` are unspecified.
    pub fn encode_frame(
        samples: &[u8],
        layout: DicomRleLayout,
        out: &mut Vec<u8>,
    ) -> Result<(), TileCodecError> {
        encode::encode_frame(samples, layout, out)
    }
}
