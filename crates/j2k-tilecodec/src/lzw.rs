// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bounded::{copy_scratch_to_output, observed_too_small},
    pool::LzwPool,
    TileCodecError,
};
use j2k_core::TileDecompress;
use weezl::{decode::Decoder, BitOrder};

/// Decoder for TIFF-style LZW tile payloads.
pub struct LzwCodec;

#[doc(hidden)]
impl TileDecompress for LzwCodec {
    type Error = TileCodecError;
    type Pool = LzwPool;

    fn expected_size(_input: &[u8]) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }

    fn decompress_into(
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error> {
        // TIFF LZW widens codes one dictionary entry before standard LZW does.
        let mut decoder = Decoder::with_tiff_size_switch(BitOrder::Msb, 8);
        let limit = out.len().saturating_add(1);
        if pool.scratch.len() != limit {
            pool.scratch.resize(limit, 0);
        }

        let mut input_offset = 0usize;
        let mut output_offset = 0usize;
        loop {
            if output_offset == pool.scratch.len() {
                return Err(TileCodecError::Buffer(observed_too_small(
                    output_offset,
                    out.len(),
                )));
            }

            let result =
                decoder.decode_bytes(&input[input_offset..], &mut pool.scratch[output_offset..]);
            input_offset += result.consumed_in;
            output_offset += result.consumed_out;

            if output_offset > out.len() {
                return Err(TileCodecError::Buffer(observed_too_small(
                    output_offset,
                    out.len(),
                )));
            }

            // weezl reads input into its bit buffer ahead of decoding, so a
            // call can consume the last byte while codes up to and including
            // the end code are still buffered. Keep calling with no input
            // until it finishes; only a call that then makes no progress
            // means the stream ended.
            let exhausted = input_offset == input.len();
            let progressed = result.consumed_in != 0 || result.consumed_out != 0;
            match result.status {
                Ok(weezl::LzwStatus::Done) => {
                    pool.scratch.truncate(output_offset);
                    let written = copy_scratch_to_output(&pool.scratch, out);
                    return Ok(written);
                }
                Ok(weezl::LzwStatus::Ok | weezl::LzwStatus::NoProgress)
                    if exhausted && !progressed =>
                {
                    // Like libtiff, accept a stream without an end code once
                    // it has filled the output; short of that it was cut off.
                    if output_offset != out.len() {
                        return Err(crate::error::truncated_input("lzw decode"));
                    }
                    pool.scratch.truncate(output_offset);
                    return Ok(copy_scratch_to_output(&pool.scratch, out));
                }
                Ok(weezl::LzwStatus::Ok) => {}
                Ok(weezl::LzwStatus::NoProgress) => {
                    return Err(crate::error::malformed_input("lzw decode", "no progress"));
                }
                Err(_error) => {
                    return Err(crate::error::malformed_input("lzw decode", "decoder error"));
                }
            }
        }
    }
}
