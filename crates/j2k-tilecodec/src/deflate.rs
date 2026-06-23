// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bounded::{read_to_output_bounded, BoundedReadError},
    pool::DeflatePool,
    TileCodecError,
};
use flate2::read::{DeflateDecoder, ZlibDecoder};
use j2k_core::TileDecompress;

/// Decoder for zlib-wrapped or raw DEFLATE tile payloads.
pub struct DeflateCodec;

impl TileDecompress for DeflateCodec {
    type Error = TileCodecError;
    type Pool = DeflatePool;

    fn expected_size(_input: &[u8]) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    fn decompress_into(
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error> {
        match read_to_output_bounded(ZlibDecoder::new(input), &mut pool.scratch, out) {
            Ok(written) => Ok(written),
            Err(BoundedReadError::OutputTooSmall(error)) => Err(error.into()),
            Err(BoundedReadError::Io(_zlib_error)) => {
                pool.scratch.clear();
                match read_to_output_bounded(DeflateDecoder::new(input), &mut pool.scratch, out) {
                    Ok(written) => Ok(written),
                    Err(BoundedReadError::OutputTooSmall(error)) => Err(error.into()),
                    Err(BoundedReadError::Io(raw_error)) => {
                        Err(crate::error::input_or_backend_io_error(
                            &raw_error,
                            "deflate decode failed",
                        ))
                    }
                }
            }
        }
    }
}
