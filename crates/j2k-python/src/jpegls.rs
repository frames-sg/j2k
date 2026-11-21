// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG-LS (ISO/IEC 14495-1) codestreams, as DICOM stores them under the
//! JPEG-LS lossless and near-lossless transfer syntaxes. Samples cross the
//! boundary as bytes: one per sample up to 8 bits, otherwise two,
//! little-endian, interleaved by pixel.

use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};

use crate::{input::EncodedInput, owned_capacity, value_error};
use j2k_jpegls::{
    encode_jpegls as encode_codestream, Decoder, InterleaveMode, JpegLsEncodeOptions, JpegLsSamples,
};

fn interleave_name(mode: InterleaveMode) -> &'static str {
    match mode {
        InterleaveMode::None => "none",
        InterleaveMode::Line => "line",
        InterleaveMode::Sample => "sample",
    }
}

/// Read a JPEG-LS codestream's headers without decoding samples.
///
/// Returns `width`, `height`, `components`, `bit_depth`, `near` (0 for
/// lossless), `interleave` (`"none"`, `"line"` or `"sample"`) and
/// `restart_interval` (lines, or `None`).
#[pyfunction]
pub(crate) fn inspect_jpegls<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let input = EncodedInput::extract(data)?;
    let bytes = input.as_bytes();
    let info = py.detach(|| Decoder::inspect(bytes)).map_err(value_error)?;
    let out = PyDict::new(py);
    out.set_item("width", info.dimensions.0)?;
    out.set_item("height", info.dimensions.1)?;
    out.set_item("components", info.components)?;
    out.set_item("bit_depth", info.bit_depth)?;
    out.set_item("near", info.near)?;
    out.set_item("interleave", interleave_name(info.interleave))?;
    out.set_item("restart_interval", info.restart_interval)?;
    Ok(out)
}

/// Decode a JPEG-LS codestream to samples interleaved by pixel: one byte per
/// sample up to 8 bits, otherwise two, little-endian.
#[pyfunction]
pub(crate) fn decode_jpegls<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let input = EncodedInput::extract(data)?;
    let bytes = input.as_bytes();
    let samples = py.detach(|| {
        let decoder = Decoder::new(bytes).map_err(value_error)?;
        let info = decoder.info();
        let (length, row_bytes) = info
            .output_len()
            .zip(info.row_bytes())
            .ok_or_else(|| value_error("JPEG-LS image size overflows"))?;
        let mut samples = owned_capacity(length)?;
        samples.resize(length, 0_u8);
        decoder
            .decode_into(&mut samples, row_bytes)
            .map_err(value_error)?;
        Ok::<_, PyErr>(samples)
    })?;
    Ok(PyBytes::new(py, &samples))
}

/// Encode samples interleaved by pixel as a JPEG-LS codestream: lossless
/// unless `near` allows each sample to differ by up to that much. The
/// encoder decodes its output and compares it with the samples before
/// returning it.
#[pyfunction]
#[pyo3(signature = (
    data, width, height, *, components=1, bit_depth=8, near=0, interleave="none",
    restart_interval=None,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "Python exposes these controls as keyword arguments"
)]
pub(crate) fn encode_jpegls<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyAny>,
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
    near: u8,
    interleave: &str,
    restart_interval: Option<u16>,
) -> PyResult<Bound<'py, PyBytes>> {
    let interleave = match interleave {
        "none" => InterleaveMode::None,
        "line" => InterleaveMode::Line,
        "sample" => InterleaveMode::Sample,
        _ => return Err(value_error("interleave must be none, line, or sample")),
    };
    let input = EncodedInput::extract(data)?;
    let bytes = input.as_bytes();
    let encoded = py.detach(|| {
        let mut options = JpegLsEncodeOptions::new(bit_depth)
            .with_near(near)
            .with_interleave(interleave);
        if let Some(lines) = restart_interval {
            options = options.with_restart_interval(lines);
        }
        let encoded = if bit_depth > 8 {
            let (pairs, rest) = bytes.as_chunks::<2>();
            if !rest.is_empty() {
                return Err(value_error(
                    "samples deeper than 8 bits take two bytes each",
                ));
            }
            let mut words = owned_capacity(pairs.len())?;
            words.extend(pairs.iter().map(|&pair| u16::from_le_bytes(pair)));
            let samples = JpegLsSamples::U16 {
                data: &words,
                width,
                height,
                components,
            };
            encode_codestream(samples, &options)
        } else {
            let samples = JpegLsSamples::U8 {
                data: bytes,
                width,
                height,
                components,
            };
            encode_codestream(samples, &options)
        };
        encoded.map(|encoded| encoded.data).map_err(value_error)
    })?;
    Ok(PyBytes::new(py, &encoded))
}
