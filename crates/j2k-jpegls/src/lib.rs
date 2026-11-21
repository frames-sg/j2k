// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG-LS inspect, decode, and encode (ISO/IEC 14495-1, ITU-T T.87).
//!
//! This covers the codestreams DICOM stores under the JPEG-LS lossless
//! (1.2.840.10008.1.2.4.80) and near-lossless (1.2.840.10008.1.2.4.81)
//! transfer syntaxes: 2 to 16 bits per sample, any number of components, all
//! three interleave modes, restart intervals, and preset coding parameters.
//!
//! [`Decoder::inspect`] reads the headers without decoding. [`Decoder`]
//! decodes to interleaved samples whatever the file's interleave mode:
//! one byte per sample up to 8 bits, and two bytes (little-endian in a byte
//! buffer, or `u16` from [`Decoder::decode`]) above 8 bits.
//! [`encode_jpegls`] writes a raw JPEG-LS codestream and, by default, checks
//! it by decoding it again.
//!
//! ```
//! use j2k_jpegls::{encode_jpegls, Decoder, DecodedSamples, JpegLsEncodeOptions, JpegLsSamples};
//!
//! let pixels: Vec<u16> = (0..64 * 48).map(|i| (i * 37 % 4096) as u16).collect();
//! let samples = JpegLsSamples::U16 { data: &pixels, width: 64, height: 48, components: 1 };
//! let encoded = encode_jpegls(samples, &JpegLsEncodeOptions::new(12))?;
//!
//! let decoder = Decoder::new(&encoded.data)?;
//! assert_eq!(decoder.info().dimensions, (64, 48));
//! let image = decoder.decode()?;
//! assert_eq!(image.samples, DecodedSamples::U16(pixels));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Decoding is strict: truncated or malformed data, and features this crate
//! does not implement (mapping tables, subsampled components, point
//! transforms, HP color transforms), return an error rather than a partial or
//! approximate image.

mod allocation;
mod bitio;
mod context;
mod decoder;
mod encoder;
mod error;
mod golomb;
mod info;
mod markers;
mod params;

pub use decoder::{DecodedImage, DecodedSamples, Decoder};
pub use encoder::{
    encode_jpegls, EncodedJpegLs, JpegLsEncodeOptions, JpegLsEncodeValidation, JpegLsSamples,
};
pub use error::{JpegLsEncodeError, JpegLsError, UnsupportedFeature};
pub use info::{DecodeOptions, Info, InterleaveMode, ScanInfo};
pub use params::PresetCodingParameters;
