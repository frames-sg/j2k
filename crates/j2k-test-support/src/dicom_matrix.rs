// SPDX-License-Identifier: MIT OR Apache-2.0

//! DICOM interoperability matrix.
//!
//! `corpus/conformance/dicom_matrix.py` writes synthetic phantoms as DICOM
//! images, compresses them with DCMTK 3.7 `dcmcjpeg` (lossless SV1 and other
//! predictors at 8 to 16 bits stored, 12-bit extended and progressive DCT,
//! and RGB), and records DCMTK's own `dcmdjpeg` decode as the reference.

const CASES_TSV: &str = include_str!("../fixtures/dicom_matrix/cases.tsv");
const DATA: &[u8] = include_bytes!("../fixtures/dicom_matrix/data.bin");

/// One DCMTK-written JPEG codestream and DCMTK's decode of it.
#[derive(Clone, Copy, Debug)]
pub struct DicomMatrixCase {
    /// Unique case name.
    pub name: &'static str,
    /// The `dcmcjpeg` options that wrote it.
    pub options: &'static str,
    /// JPEG sample precision `P` (DCMTK codes 10- to 16-bit stored lossless
    /// data at 16 bits).
    pub precision: u8,
    /// Samples per pixel: 1 for monochrome, 3 for RGB.
    pub channels: usize,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// The JPEG codestream from the encapsulated pixel data.
    pub jpeg: &'static [u8],
    reference: &'static [u8],
}

impl DicomMatrixCase {
    /// DCMTK's decoded samples (interleaved for RGB).
    pub fn reference_samples(&self) -> Vec<u16> {
        if self.precision > 8 {
            self.reference
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_be_bytes(*pair))
                .collect()
        } else {
            self.reference
                .iter()
                .map(|&sample| u16::from(sample))
                .collect()
        }
    }
}

fn slice(field: &str) -> &'static [u8] {
    let (offset, len) = field
        .split_once(':')
        .expect("DICOM matrix slices are offset:length");
    let offset: usize = offset.parse().expect("DICOM matrix offset");
    let len: usize = len.parse().expect("DICOM matrix length");
    &DATA[offset..offset + len]
}

/// Every case in the committed matrix, in generation order.
///
/// # Panics
///
/// Panics if the committed manifest is malformed.
pub fn dicom_matrix_cases() -> Vec<DicomMatrixCase> {
    CASES_TSV
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&'static str> = line.split('\t').collect();
            assert_eq!(fields.len(), 8, "DICOM matrix row: {line}");
            DicomMatrixCase {
                name: fields[0],
                options: fields[1],
                precision: fields[2].parse().expect("precision"),
                channels: fields[3].parse().expect("channels"),
                width: fields[4].parse().expect("width"),
                height: fields[5].parse().expect("height"),
                jpeg: slice(fields[6]),
                reference: slice(fields[7]),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::dicom_matrix_cases;

    #[test]
    fn every_case_is_a_jpeg_with_a_full_reference() {
        let cases = dicom_matrix_cases();
        assert_eq!(cases.len(), 11);
        for case in cases {
            assert_eq!(&case.jpeg[..2], &[0xFF, 0xD8], "{}", case.name);
            let pixels = case.width as usize * case.height as usize;
            assert_eq!(
                case.reference_samples().len(),
                pixels * case.channels,
                "{}",
                case.name
            );
        }
    }
}
