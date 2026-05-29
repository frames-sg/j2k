// SPDX-License-Identifier: Apache-2.0

use crate::{
    bounded::{copy_scratch_to_output, read_to_scratch_bounded, BoundedReadError},
    pool::ZstdPool,
    TileCodecError,
};
use signinum_core::{InputError, TileDecompress};

/// Zstandard tile decompressor.
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
        let mut decoder = zstd::stream::read::Decoder::new(input).map_err(|_| {
            TileCodecError::Input(InputError::TruncatedAt {
                offset: input.len(),
                segment: "zstd header",
            })
        })?;
        let written = match read_to_scratch_bounded(&mut decoder, &mut pool.scratch, out.len()) {
            Ok(written) => written,
            Err(BoundedReadError::OutputTooSmall(error)) => return Err(error.into()),
            Err(BoundedReadError::Io(error)) => {
                let _ = error;
                return Err(TileCodecError::Input(InputError::TruncatedAt {
                    offset: input.len(),
                    segment: "zstd payload",
                }));
            }
        };

        copy_scratch_to_output(&pool.scratch, out);
        Ok(written)
    }
}
