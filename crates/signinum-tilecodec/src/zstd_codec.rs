// SPDX-License-Identifier: Apache-2.0

use crate::{
    bounded::{read_to_output_bounded, BoundedReadError},
    pool::ZstdPool,
    TileCodecError,
};
use signinum_core::TileDecompress;

/// Decoder for Zstandard-compressed tile payloads.
pub struct ZstdCodec;

impl TileDecompress for ZstdCodec {
    type Error = TileCodecError;
    type Pool = ZstdPool;

    fn expected_size(_input: &[u8]) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    fn decompress_into(
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error> {
        pool.scratch.clear();
        let mut decoder = zstd::stream::read::Decoder::new(input)
            .map_err(|error| crate::error::malformed_io_error(&error, "zstd decoder init"))?;
        read_to_output_bounded(&mut decoder, &mut pool.scratch, out).map_err(|error| match error {
            BoundedReadError::OutputTooSmall(error) => TileCodecError::Buffer(error),
            BoundedReadError::Io(error) => {
                crate::error::malformed_io_error(&error, "zstd decode failed")
            }
        })
    }
}
