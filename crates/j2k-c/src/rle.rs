// SPDX-License-Identifier: MIT OR Apache-2.0

//! DICOM RLE Lossless frames. This module only translates ABI fields into
//! `j2k-tilecodec` calls; the codec validates every frame.

#![forbid(unsafe_code)]

use crate::{
    abi::{
        J2kCRleLayout, J2kCRleReport, J2K_BUFFER_TOO_SMALL, J2K_ENCODE_FAILED,
        J2K_INVALID_ARGUMENT, J2K_INVALID_INPUT, J2K_RLE_FLAGS, J2K_RLE_LENIENT,
    },
    error::{Error, Result},
};
use j2k_tilecodec::{
    DicomRleCodec, DicomRleDecodeReport, DicomRleLayout, DicomRlePlanarConfiguration,
};

/// The codec layout `layout` describes, and whether it asks for lenient
/// decoding.
fn codec_layout(layout: &J2kCRleLayout) -> Result<(DicomRleLayout, bool)> {
    if layout.flags & !J2K_RLE_FLAGS != 0 {
        return Err(Error::argument("unknown DICOM RLE layout flags"));
    }
    let count = |value: u32, message| {
        u16::try_from(value)
            .ok()
            .filter(|&value| value != 0)
            .ok_or_else(|| Error::argument(message))
    };
    let rows = count(layout.rows, "DICOM RLE rows must be between 1 and 65535")?;
    let columns = count(
        layout.columns,
        "DICOM RLE columns must be between 1 and 65535",
    )?;
    let samples = count(
        layout.samples_per_pixel,
        "DICOM RLE samples per pixel must be between 1 and 65535",
    )?;
    let bits_allocated = u16::try_from(layout.bits_allocated)
        .map_err(|_| Error::argument("DICOM RLE bits allocated exceeds 65535"))?;
    let planar = match layout.planar_configuration {
        0 => DicomRlePlanarConfiguration::Interleaved,
        1 => DicomRlePlanarConfiguration::Planar,
        _ => {
            return Err(Error::argument(
                "DICOM RLE planar configuration must be 0 or 1",
            ))
        }
    };
    let codec = DicomRleLayout::new(rows, columns, samples, bits_allocated, planar)
        .map_err(|error| Error::tile_codec(&error, J2K_INVALID_ARGUMENT))?;
    Ok((codec, layout.flags & J2K_RLE_LENIENT != 0))
}

/// Decode one frame into the first `frame_len` bytes of `output`.
pub(crate) fn decode(
    input: &[u8],
    layout: &J2kCRleLayout,
    output: &mut [u8],
) -> Result<J2kCRleReport> {
    let (codec, lenient) = codec_layout(layout)?;
    let (frame_len, have) = (codec.frame_len(), output.len());
    let output = output.get_mut(..frame_len).ok_or_else(|| {
        Error::new(
            J2K_BUFFER_TOO_SMALL,
            format!("output buffer holds {have} bytes, the frame needs {frame_len}"),
        )
    })?;
    let report = if lenient {
        DicomRleCodec::decode_frame_lenient(input, codec, output)
    } else {
        DicomRleCodec::decode_frame(input, codec, output).map(|()| DicomRleDecodeReport::default())
    }
    .map_err(|error| Error::tile_codec(&error, J2K_INVALID_INPUT))?;
    Ok(J2kCRleReport {
        overlong_segments: u32::from(report.overlong_segments()),
        nonzero_unused_offsets: u32::from(report.has_nonzero_unused_offsets()),
        ..J2kCRleReport::default()
    })
}

/// Encode one frame of native samples.
pub(crate) fn encode(samples: &[u8], layout: &J2kCRleLayout) -> Result<Vec<u8>> {
    let (codec, lenient) = codec_layout(layout)?;
    if lenient {
        return Err(Error::argument("J2K_RLE_LENIENT applies to decoding only"));
    }
    let mut frame = Vec::new();
    DicomRleCodec::encode_frame(samples, codec, &mut frame)
        .map_err(|error| Error::tile_codec(&error, J2K_ENCODE_FAILED))?;
    Ok(frame)
}
