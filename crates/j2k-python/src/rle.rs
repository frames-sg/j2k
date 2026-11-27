// SPDX-License-Identifier: MIT OR Apache-2.0

//! DICOM RLE Lossless frames (transfer syntax 1.2.840.10008.1.2.5). Samples
//! cross the boundary as the native frame: `bits_allocated / 8` bytes per
//! sample, little-endian, in the requested planar configuration.

use pyo3::{prelude::*, types::PyBytes};

use crate::{input::EncodedInput, owned_capacity, value_error};
use j2k_tilecodec::{DicomRleCodec, DicomRleLayout, DicomRlePlanarConfiguration};

fn layout(
    rows: u16,
    columns: u16,
    samples_per_pixel: u16,
    bits_allocated: u16,
    planar_configuration: u8,
) -> PyResult<DicomRleLayout> {
    let planar_configuration = match planar_configuration {
        0 => DicomRlePlanarConfiguration::Interleaved,
        1 => DicomRlePlanarConfiguration::Planar,
        _ => return Err(value_error("planar_configuration must be 0 or 1")),
    };
    DicomRleLayout::new(
        rows,
        columns,
        samples_per_pixel,
        bits_allocated,
        planar_configuration,
    )
    .map_err(value_error)
}

/// Describe what a lenient decode accepted, or `None` for a conformant frame.
fn leniency(report: j2k_tilecodec::DicomRleDecodeReport) -> Option<String> {
    if report.is_conformant() {
        return None;
    }
    let mut accepted = Vec::new();
    let overlong = report.overlong_segments();
    if overlong != 0 {
        let segments = (0..16)
            .filter(|segment| overlong & (1 << segment) != 0)
            .map(|segment| segment.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        accepted.push(format!(
            "data after the rows of segment(s) {segments} was discarded"
        ));
    }
    if report.has_nonzero_unused_offsets() {
        accepted.push(String::from("nonzero unused header offsets were ignored"));
    }
    Some(format!(
        "non-conformant DICOM RLE frame decoded leniently: {}",
        accepted.join("; ")
    ))
}

/// Decode one DICOM RLE frame to native samples: `bits_allocated / 8`
/// bytes per sample, little-endian, color-by-pixel for
/// `planar_configuration=0` or color-by-plane for 1.
///
/// Decoding is strict. `lenient=True` also accepts segments with data after
/// their rows and nonzero unused header offsets, as some encoders write
/// them, and emits a `RecoveredImageWarning` saying what it accepted.
#[pyfunction]
#[pyo3(signature = (
    data, rows, columns, *, samples_per_pixel=1, bits_allocated=8, planar_configuration=0,
    lenient=false,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "Python exposes the data set attributes as keyword arguments"
)]
pub(crate) fn decode_rle<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyAny>,
    rows: u16,
    columns: u16,
    samples_per_pixel: u16,
    bits_allocated: u16,
    planar_configuration: u8,
    lenient: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let layout = layout(
        rows,
        columns,
        samples_per_pixel,
        bits_allocated,
        planar_configuration,
    )?;
    if layout.frame_len() > j2k_native::DEFAULT_MAX_CODEC_BYTES {
        return Err(value_error("DICOM RLE frame exceeds the codec byte limit"));
    }
    let input = EncodedInput::extract(data)?;
    let bytes = input.as_bytes();
    let (frame, accepted) = py.detach(|| {
        let mut frame = owned_capacity(layout.frame_len())?;
        frame.resize(layout.frame_len(), 0_u8);
        let accepted = if lenient {
            DicomRleCodec::decode_frame_lenient(bytes, layout, &mut frame)
                .map(leniency)
                .map_err(value_error)?
        } else {
            DicomRleCodec::decode_frame(bytes, layout, &mut frame).map_err(value_error)?;
            None
        };
        Ok::<_, PyErr>((frame, accepted))
    })?;
    if let Some(message) = accepted {
        crate::decode::warn_metadata_recoveries(py, &[message])?;
    }
    Ok(PyBytes::new(py, &frame))
}

/// Encode one frame of native samples (`bits_allocated / 8` bytes per
/// sample, little-endian, in `planar_configuration`) as a DICOM RLE frame.
#[pyfunction]
#[pyo3(signature = (
    data, rows, columns, *, samples_per_pixel=1, bits_allocated=8, planar_configuration=0,
))]
pub(crate) fn encode_rle<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyAny>,
    rows: u16,
    columns: u16,
    samples_per_pixel: u16,
    bits_allocated: u16,
    planar_configuration: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    let layout = layout(
        rows,
        columns,
        samples_per_pixel,
        bits_allocated,
        planar_configuration,
    )?;
    let input = EncodedInput::extract(data)?;
    let bytes = input.as_bytes();
    let frame = py.detach(|| {
        let mut frame = Vec::new();
        DicomRleCodec::encode_frame(bytes, layout, &mut frame).map_err(value_error)?;
        Ok::<_, PyErr>(frame)
    })?;
    Ok(PyBytes::new(py, &frame))
}
