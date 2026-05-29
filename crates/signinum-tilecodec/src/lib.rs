// SPDX-License-Identifier: Apache-2.0

//! Tile payload decompression codecs used by Signinum container integrations.

#![deny(missing_docs)]

mod bounded;
mod deflate;
mod error;
mod lzw;
mod pool;
mod uncompressed;
mod zstd_codec;

pub use deflate::DeflateCodec;
pub use error::TileCodecError;
pub use lzw::LzwCodec;
pub use pool::{DeflatePool, LzwPool, NoPool, ZstdPool};
pub use signinum_core::TileDecompress;
pub use uncompressed::UncompressedCodec;
pub use zstd_codec::ZstdCodec;
