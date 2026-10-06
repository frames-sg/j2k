// SPDX-License-Identifier: MIT OR Apache-2.0

//! libjpeg-turbo DCT-scaling reference matrix.
//!
//! `corpus/conformance/scaled_matrix.py` encodes each case with libjpeg-turbo
//! 3.1.4.1 `cjpeg` and records `djpeg -dct int -scale 1/N` output for N = 1,
//! 2, 4, 8 (default fancy upsampling, as `OpenSlide` uses it). The cases cover
//! every chroma layout libjpeg-turbo decodes differently at reduced scale,
//! chroma at most two samples wide, restart intervals, progressive scans and
//! 12-bit precision.

const CASES_TSV: &str = include_str!("../fixtures/scaled_matrix/cases.tsv");
const DATA: &[u8] = include_bytes!("../fixtures/scaled_matrix/data.bin");

/// Scale denominators in reference order.
pub const SCALED_MATRIX_DENOMINATORS: [u32; 4] = [1, 2, 4, 8];

/// One encoded image and its libjpeg-turbo references at every scale.
#[derive(Clone, Copy, Debug)]
pub struct ScaledMatrixCase {
    /// Unique case name, `{layout}_{w}x{h}_{coding}_{precision}bit`.
    pub name: &'static str,
    /// `gray`, `444`, `422`, `440`, `420`, `411`, `410` or `1x4` (luma
    /// sampling; chroma is 1x1).
    pub layout: &'static str,
    /// `baseline`, `restart` (one-MCU interval), `progressive` or `extended`
    /// (12-bit sequential).
    pub coding: &'static str,
    /// Sample precision, 8 or 12.
    pub precision: u8,
    /// Full-size width.
    pub width: u32,
    /// Full-size height.
    pub height: u32,
    /// The JPEG codestream.
    pub jpeg: &'static [u8],
    references: [&'static [u8]; 4],
}

impl ScaledMatrixCase {
    /// Whether the case is single-component grayscale.
    pub fn is_gray(&self) -> bool {
        self.layout == "gray"
    }

    /// Interleaved samples per pixel in the references.
    pub fn channels(&self) -> usize {
        if self.is_gray() {
            1
        } else {
            3
        }
    }

    /// Output dimensions at `1/denominator`, rounded up as libjpeg does.
    pub fn scaled_dimensions(&self, denominator: u32) -> (u32, u32) {
        (
            self.width.div_ceil(denominator),
            self.height.div_ceil(denominator),
        )
    }

    /// `djpeg -scale 1/denominator` output without its PNM header: 8-bit
    /// samples, or big-endian 16-bit samples for 12-bit cases.
    ///
    /// # Panics
    ///
    /// Panics if `denominator` is not 1, 2, 4 or 8.
    pub fn reference(&self, denominator: u32) -> &'static [u8] {
        let index = SCALED_MATRIX_DENOMINATORS
            .iter()
            .position(|&d| d == denominator)
            .expect("scaled matrix references cover 1/1, 1/2, 1/4 and 1/8");
        self.references[index]
    }

    /// 12-bit references as sample values.
    ///
    /// # Panics
    ///
    /// Panics if `denominator` is unsupported or the case is 8-bit.
    pub fn reference_u16(&self, denominator: u32) -> Vec<u16> {
        assert!(self.precision > 8, "{} is an 8-bit case", self.name);
        self.reference(denominator)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect()
    }
}

fn slice(field: &str) -> &'static [u8] {
    let (offset, len) = field
        .split_once(':')
        .expect("scaled matrix slices are offset:length");
    let offset: usize = offset.parse().expect("scaled matrix offset");
    let len: usize = len.parse().expect("scaled matrix length");
    &DATA[offset..offset + len]
}

/// Every case in the committed matrix, in generation order.
///
/// # Panics
///
/// Panics if the committed manifest is malformed.
pub fn scaled_matrix_cases() -> Vec<ScaledMatrixCase> {
    CASES_TSV
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&'static str> = line.split('\t').collect();
            assert_eq!(fields.len(), 11, "scaled matrix row: {line}");
            ScaledMatrixCase {
                name: fields[0],
                layout: fields[1],
                coding: fields[2],
                precision: fields[3].parse().expect("precision"),
                width: fields[4].parse().expect("width"),
                height: fields[5].parse().expect("height"),
                jpeg: slice(fields[6]),
                references: [
                    slice(fields[7]),
                    slice(fields[8]),
                    slice(fields[9]),
                    slice(fields[10]),
                ],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{scaled_matrix_cases, SCALED_MATRIX_DENOMINATORS};

    #[test]
    fn every_reference_has_the_scaled_geometry() {
        let cases = scaled_matrix_cases();
        assert_eq!(cases.len(), 208);
        for case in cases {
            assert_eq!(&case.jpeg[..2], &[0xFF, 0xD8], "{}", case.name);
            let bytes_per_sample = if case.precision > 8 { 2 } else { 1 };
            for denominator in SCALED_MATRIX_DENOMINATORS {
                let (w, h) = case.scaled_dimensions(denominator);
                assert_eq!(
                    case.reference(denominator).len(),
                    w as usize * h as usize * case.channels() * bytes_per_sample,
                    "{} 1/{denominator}",
                    case.name
                );
            }
        }
    }
}
