// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `CharLS` fixture matrix written by `corpus/conformance/jpegls_matrix.py`.

use j2k_jpegls::{
    DecodedSamples, InterleaveMode, JpegLsEncodeOptions, JpegLsSamples, PresetCodingParameters,
};

const CASES_TSV: &str = include_str!("../../fixtures/jpegls_matrix/cases.tsv");
const DATA: &[u8] = include_bytes!("../../fixtures/jpegls_matrix/data.bin");

/// Who wrote a fixture's codestream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Source {
    /// `CharLS`; the reference is `CharLS`'s decode.
    Charls,
    /// j2k-jpegls; for restart cases the reference is `CharLS`'s decode, for
    /// two-component interleaved cases it is the input.
    J2kJpegls,
    /// A DICOM file's pixel data; the reference is `CharLS`'s decode.
    Dicom,
}

/// One fixture case.
#[derive(Debug, Clone)]
pub(crate) struct MatrixCase {
    pub(crate) name: &'static str,
    pub(crate) source: Source,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) components: u8,
    pub(crate) bits: u8,
    pub(crate) interleave: InterleaveMode,
    pub(crate) near: u8,
    pub(crate) restart: u16,
    pub(crate) preset: Option<PresetCodingParameters>,
    pub(crate) input: &'static [u8],
    pub(crate) codestream: &'static [u8],
    pub(crate) reference: &'static [u8],
}

impl MatrixCase {
    pub(crate) fn input_samples(&self) -> Vec<u16> {
        samples(self.input, self.bits)
    }

    pub(crate) fn reference_samples(&self) -> Vec<u16> {
        samples(self.reference, self.bits)
    }

    /// Encode options matching the case's parameters.
    pub(crate) fn options(&self) -> JpegLsEncodeOptions {
        let mut options = JpegLsEncodeOptions::new(self.bits)
            .with_near(self.near)
            .with_interleave(self.interleave);
        if self.restart != 0 {
            options = options.with_restart_interval(self.restart);
        }
        if let Some(preset) = self.preset {
            options = options.with_preset(preset);
        }
        options
    }

    /// Runs `f` with the case's input as encoder samples.
    pub(crate) fn with_samples<R>(&self, f: impl FnOnce(JpegLsSamples<'_>) -> R) -> R {
        let (width, height, components) = (self.width, self.height, self.components);
        if self.bits > 8 {
            let data = self.input_samples();
            f(JpegLsSamples::U16 {
                data: &data,
                width,
                height,
                components,
            })
        } else {
            f(JpegLsSamples::U8 {
                data: self.input,
                width,
                height,
                components,
            })
        }
    }
}

fn samples(bytes: &[u8], bits: u8) -> Vec<u16> {
    if bits > 8 {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect()
    } else {
        bytes.iter().map(|&sample| u16::from(sample)).collect()
    }
}

fn slice(field: &str) -> &'static [u8] {
    let (offset, length) = field.split_once(':').expect("offset:length");
    let offset: usize = offset.parse().expect("offset");
    let length: usize = length.parse().expect("length");
    &DATA[offset..offset + length]
}

fn number<T: core::str::FromStr>(field: &str) -> T {
    field
        .parse()
        .unwrap_or_else(|_| panic!("numeric fixture field, got {field}"))
}

/// Every case in the matrix, in file order.
pub(crate) fn matrix_cases() -> Vec<MatrixCase> {
    CASES_TSV
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| {
            let fields: Vec<&'static str> = line.split('\t').collect();
            assert_eq!(fields.len(), 14, "malformed fixture line: {line}");
            let source = match fields[1] {
                "charls" => Source::Charls,
                "j2k-jpegls" => Source::J2kJpegls,
                "dicom" => Source::Dicom,
                other => panic!("unknown fixture source {other}"),
            };
            let interleave = match fields[7] {
                "0" => InterleaveMode::None,
                "1" => InterleaveMode::Line,
                "2" => InterleaveMode::Sample,
                other => panic!("unknown interleave mode {other}"),
            };
            let preset = (fields[10] != "-").then(|| {
                let values: Vec<u16> = fields[10].split(',').map(number).collect();
                PresetCodingParameters {
                    max_value: values[0],
                    t1: values[1],
                    t2: values[2],
                    t3: values[3],
                    reset: values[4],
                }
            });
            MatrixCase {
                name: fields[0],
                source,
                width: number(fields[3]),
                height: number(fields[4]),
                components: number(fields[5]),
                bits: number(fields[6]),
                interleave,
                near: number(fields[8]),
                restart: number(fields[9]),
                preset,
                input: slice(fields[11]),
                codestream: slice(fields[12]),
                reference: slice(fields[13]),
            }
        })
        .collect()
}

/// A case by name.
pub(crate) fn matrix_case(name: &str) -> MatrixCase {
    matrix_cases()
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no fixture case {name}"))
}

/// Decoded samples widened to `u16`.
pub(crate) fn decoded_samples(samples: &DecodedSamples) -> Vec<u16> {
    match samples {
        DecodedSamples::U8(values) => values.iter().map(|&value| u16::from(value)).collect(),
        DecodedSamples::U16(values) => values.clone(),
        other => panic!("unexpected sample type {other:?}"),
    }
}
