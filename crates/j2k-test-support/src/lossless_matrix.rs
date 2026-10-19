// SPDX-License-Identifier: MIT OR Apache-2.0

//! libjpeg-turbo lossless (SOF3) JPEG matrix.
//!
//! `corpus/conformance/lossless_matrix.py` encodes each case with
//! libjpeg-turbo 3.1.4.1 `cjpeg -lossless` (or its reference encoder for
//! YCbCr and subsampled color): every predictor at every precision from 2 to
//! 16 bits, point transforms, restart intervals, RGB and YCbCr color at
//! 4:4:4, 4:2:2, 4:2:0, 4:4:0, and 4:1:1 including odd sizes, and split
//! frames coding their components in several scans. Every case libjpeg-turbo
//! decodes carries its `djpeg` output as the reference.

const CASES_TSV: &str = include_str!("../fixtures/lossless_matrix/cases.tsv");
const DATA: &[u8] = include_bytes!("../fixtures/lossless_matrix/data.bin");

/// One lossless JPEG in the matrix.
#[derive(Clone, Copy, Debug)]
pub struct LosslessMatrixCase {
    /// Unique case name.
    pub name: &'static str,
    /// `gray`, or `rgb`/`ycbcr` followed by `444`, `422`, `420`, `440`, or
    /// `411`.
    pub layout: &'static str,
    /// Sample precision `P`.
    pub precision: u8,
    /// Predictor selection value, 1..=7 (of the first scan of a split frame).
    pub predictor: u8,
    /// Point transform `Pt` (of the first scan of a split frame).
    pub point_transform: u8,
    /// Restart interval in MCU rows; zero without restart markers.
    pub restart_rows: u32,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// The JPEG codestream.
    pub jpeg: &'static [u8],
    /// Frame component groups of a split frame's scans in coding order, as
    /// `0/12`; `None` for one interleaved scan.
    pub scans: Option<&'static str>,
    /// `djpeg -pnm` output without its header: 8-bit samples, or big-endian
    /// 16-bit samples above 8 bits. `None` for YCbCr color, which
    /// libjpeg-turbo does not convert in lossless mode.
    pub reference: Option<&'static [u8]>,
}

impl LosslessMatrixCase {
    /// Whether the case is single-component grayscale.
    pub fn is_gray(&self) -> bool {
        self.layout == "gray"
    }

    /// The reference as sample values.
    pub fn reference_samples(&self) -> Option<Vec<u16>> {
        let reference = self.reference?;
        Some(if self.precision > 8 {
            reference
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_be_bytes(*pair))
                .collect()
        } else {
            reference.iter().map(|&sample| u16::from(sample)).collect()
        })
    }
}

fn slice(field: &str) -> Option<&'static [u8]> {
    if field == "-" {
        return None;
    }
    let (offset, len) = field
        .split_once(':')
        .expect("lossless matrix slices are offset:length");
    let offset: usize = offset.parse().expect("lossless matrix offset");
    let len: usize = len.parse().expect("lossless matrix length");
    Some(&DATA[offset..offset + len])
}

/// Every case in the committed matrix, in generation order.
///
/// # Panics
///
/// Panics if the committed manifest is malformed.
pub fn lossless_matrix_cases() -> Vec<LosslessMatrixCase> {
    CASES_TSV
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&'static str> = line.split('\t').collect();
            assert_eq!(fields.len(), 11, "lossless matrix row: {line}");
            LosslessMatrixCase {
                name: fields[0],
                layout: fields[1],
                precision: fields[2].parse().expect("precision"),
                predictor: fields[3].parse().expect("predictor"),
                point_transform: fields[4].parse().expect("point transform"),
                restart_rows: fields[5].parse().expect("restart rows"),
                width: fields[6].parse().expect("width"),
                height: fields[7].parse().expect("height"),
                jpeg: slice(fields[8]).expect("lossless matrix JPEG"),
                scans: (fields[9] != "-").then_some(fields[9]),
                reference: slice(fields[10]),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::lossless_matrix_cases;

    #[test]
    fn every_case_is_a_lossless_jpeg_with_its_declared_parameters() {
        let cases = lossless_matrix_cases();
        assert_eq!(cases.len(), 81);
        for case in cases {
            // SOI, then a SOF3 frame header somewhere before the scan.
            assert_eq!(&case.jpeg[..2], &[0xFF, 0xD8], "{}", case.name);
            let sof3 = case
                .jpeg
                .windows(2)
                .position(|marker| marker == [0xFF, 0xC3])
                .unwrap_or_else(|| panic!("{} has no SOF3 marker", case.name));
            let header = &case.jpeg[sof3 + 4..];
            assert_eq!(header[0], case.precision, "{}", case.name);
            let height = u32::from(u16::from_be_bytes([header[1], header[2]]));
            let width = u32::from(u16::from_be_bytes([header[3], header[4]]));
            assert_eq!((width, height), (case.width, case.height), "{}", case.name);
            assert!((1..=7).contains(&case.predictor), "{}", case.name);
            if let Some(reference) = case.reference_samples() {
                let channels = if case.is_gray() { 1 } else { 3 };
                let pixels = case.width as usize * case.height as usize;
                assert_eq!(reference.len(), pixels * channels, "{}", case.name);
            }
            assert_eq!(
                case.reference.is_some(),
                !case.layout.starts_with("ycbcr"),
                "{}",
                case.name
            );
        }
    }
}
