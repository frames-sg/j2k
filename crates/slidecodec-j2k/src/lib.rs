// SPDX-License-Identifier: Apache-2.0

//! JPEG 2000 inspect support for slidecodec.

extern crate alloc;

mod decode;

pub mod error;
pub use error::J2kError;

pub mod scratch;
pub use scratch::J2kScratchPool;

pub mod view;
pub use view::{J2kDecoder, J2kView};

pub use slidecodec_core::{
    BufferError, CodecError, DecodeOutcome, Downscale, ImageCodec, ImageDecode, PixelFormat, Rect,
};

pub(crate) mod parse;
