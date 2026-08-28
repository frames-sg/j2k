// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SampleFormat {
    Gray8,
    Rgb8,
    Gray16,
}

impl SampleFormat {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Gray8 => "gray8",
            Self::Rgb8 => "rgb8",
            Self::Gray16 => "gray16",
        }
    }

    pub(super) const fn components(self) -> u16 {
        if matches!(self, Self::Rgb8) {
            3
        } else {
            1
        }
    }

    pub(super) const fn bit_depth(self) -> u8 {
        if matches!(self, Self::Gray16) {
            16
        } else {
            8
        }
    }

    pub(super) const fn pnm_extension(self) -> &'static str {
        if matches!(self, Self::Rgb8) {
            "ppm"
        } else {
            "pgm"
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Profile {
    Lossless,
    Qfactor90,
}

impl Profile {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Lossless => "reversible-53",
            Self::Qfactor90 => "irreversible-qfactor90",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MatrixCell {
    pub(super) format: SampleFormat,
    pub(super) profile: Profile,
}

pub(super) const fn matrix_cells() -> [MatrixCell; 6] {
    [
        MatrixCell {
            format: SampleFormat::Gray8,
            profile: Profile::Lossless,
        },
        MatrixCell {
            format: SampleFormat::Rgb8,
            profile: Profile::Lossless,
        },
        MatrixCell {
            format: SampleFormat::Gray16,
            profile: Profile::Lossless,
        },
        MatrixCell {
            format: SampleFormat::Gray8,
            profile: Profile::Qfactor90,
        },
        MatrixCell {
            format: SampleFormat::Rgb8,
            profile: Profile::Qfactor90,
        },
        MatrixCell {
            format: SampleFormat::Gray16,
            profile: Profile::Qfactor90,
        },
    ]
}

#[derive(Clone, Copy)]
pub(super) enum Producer {
    J2k,
    OpenJph,
}

impl Producer {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::J2k => "j2k",
            Self::OpenJph => "openjph",
        }
    }
}

pub(super) struct SourceImage {
    pub(super) format: SampleFormat,
    pub(super) pixels_le: Vec<u8>,
    pub(super) pnm_bytes: Vec<u8>,
}

pub(super) struct EncodedSample {
    pub(super) codestream: Vec<u8>,
    pub(super) median_us: f64,
}

pub(super) struct MatrixContext {
    pub(super) compress: PathBuf,
    pub(super) expand: PathBuf,
    pub(super) work_dir: PathBuf,
    pub(super) repeats: usize,
}

#[cfg(test)]
mod tests {
    use super::{matrix_cells, Profile, SampleFormat};

    #[test]
    fn balanced_matrix_covers_requested_formats_and_profiles() {
        let cells = matrix_cells();
        assert_eq!(cells.len(), 6);
        for format in [
            SampleFormat::Gray8,
            SampleFormat::Rgb8,
            SampleFormat::Gray16,
        ] {
            assert!(cells
                .iter()
                .any(|cell| cell.format == format && cell.profile == Profile::Lossless));
            assert!(cells
                .iter()
                .any(|cell| cell.format == format && cell.profile == Profile::Qfactor90));
        }
    }
}
