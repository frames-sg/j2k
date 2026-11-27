// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG-LS inspection, decoding and encoding. This module only translates ABI
//! fields into `j2k-jpegls` calls; the codec validates every coding rule.

#![forbid(unsafe_code)]

use crate::{
    abi::{
        J2kCJpegLsEncodeOptions, J2kCJpegLsInfo, J2K_ALLOCATION_FAILED, J2K_JPEGLS_INTERLEAVE_LINE,
        J2K_JPEGLS_INTERLEAVE_NONE, J2K_JPEGLS_INTERLEAVE_SAMPLE,
    },
    error::{Error, Result},
};
use j2k_jpegls::{
    encode_jpegls, DecodeOptions, Decoder, InterleaveMode, JpegLsEncodeOptions, JpegLsSamples,
};

/// The header metadata of `input`, and the layout a decode writes.
pub(crate) fn inspect(input: &[u8]) -> Result<J2kCJpegLsInfo> {
    let info = Decoder::inspect(input).map_err(|error| Error::jpegls(&error))?;
    let (row_bytes, buffer_bytes) = info
        .row_bytes()
        .zip(info.output_len())
        .and_then(|(row, total)| Some((u64::try_from(row).ok()?, u64::try_from(total).ok()?)))
        .ok_or_else(|| Error::limit("JPEG-LS image size exceeds the address space"))?;
    Ok(J2kCJpegLsInfo {
        width: info.dimensions.0,
        height: info.dimensions.1,
        components: u32::from(info.components),
        bit_depth: u32::from(info.bit_depth),
        near: u32::from(info.near),
        interleave: u32::from(info.interleave.code()),
        restart_interval: info.restart_interval.unwrap_or(0),
        row_bytes,
        buffer_bytes,
        ..J2kCJpegLsInfo::default()
    })
}

/// Decode `input` into `output` with rows `stride` bytes apart.
pub(crate) fn decode(input: &[u8], output: &mut [u8], stride: usize) -> Result<()> {
    // Decoding writes only into the caller's buffer, which bounds the image;
    // the codec's default output limit guards allocations this call never makes.
    let options = DecodeOptions::default().with_max_output_bytes(usize::MAX);
    Decoder::with_options(input, options)
        .and_then(|decoder| decoder.decode_into(output, stride))
        .map_err(|error| Error::jpegls(&error))
}

/// Encode interleaved samples as a JPEG-LS codestream.
pub(crate) fn encode(samples: &[u8], options: &J2kCJpegLsEncodeOptions) -> Result<Vec<u8>> {
    if options.reserved != 0 {
        return Err(Error::argument("nonzero reserved field"));
    }
    let components = u8::try_from(options.components)
        .map_err(|_| Error::argument("JPEG-LS component count exceeds 255"))?;
    let bit_depth = u8::try_from(options.bit_depth)
        .map_err(|_| Error::argument("JPEG-LS bit depth must be between 2 and 16"))?;
    let near =
        u8::try_from(options.near).map_err(|_| Error::argument("JPEG-LS NEAR exceeds 255"))?;
    let interleave = match options.interleave {
        J2K_JPEGLS_INTERLEAVE_NONE => InterleaveMode::None,
        J2K_JPEGLS_INTERLEAVE_LINE => InterleaveMode::Line,
        J2K_JPEGLS_INTERLEAVE_SAMPLE => InterleaveMode::Sample,
        _ => return Err(Error::argument("unknown JPEG-LS interleave mode")),
    };
    let mut coding = JpegLsEncodeOptions::new(bit_depth)
        .with_near(near)
        .with_interleave(interleave);
    if options.restart_interval != 0 {
        let lines = u16::try_from(options.restart_interval)
            .map_err(|_| Error::argument("JPEG-LS restart interval exceeds 65535 lines"))?;
        coding = coding.with_restart_interval(lines);
    }
    let (width, height) = (options.width, options.height);
    let encoded = if bit_depth <= 8 {
        let samples = JpegLsSamples::U8 {
            data: samples,
            width,
            height,
            components,
        };
        encode_jpegls(samples, &coding)
    } else {
        let words = little_endian_words(samples)?;
        let samples = JpegLsSamples::U16 {
            data: &words,
            width,
            height,
            components,
        };
        encode_jpegls(samples, &coding)
    };
    encoded
        .map(|encoded| encoded.data)
        .map_err(|error| Error::jpegls_encode(&error))
}

/// Two-byte little-endian samples as `u16` values.
fn little_endian_words(bytes: &[u8]) -> Result<Vec<u16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(Error::argument(
            "samples above 8 bits take two bytes each, but the length is odd",
        ));
    }
    let mut words = Vec::new();
    words.try_reserve_exact(bytes.len() / 2).map_err(|_| {
        Error::new(
            J2K_ALLOCATION_FAILED,
            "could not allocate the 16-bit samples",
        )
    })?;
    words.extend(
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| u16::from_le_bytes(pair)),
    );
    Ok(words)
}
