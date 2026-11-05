// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless encoding of interleaved samples.

#![forbid(unsafe_code)]

use crate::{
    abi::{
        J2kCEncodeOptions, J2K_ENCODE_FILE, J2K_ENCODE_FLAGS, J2K_ENCODE_HTJ2K, J2K_ENCODE_SIGNED,
    },
    error::{Error, Result},
};
use j2k::{
    encode_j2k_lossless, wrap_j2k_codestream, J2kBlockCodingMode, J2kFileWrapOptions,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};

/// Encode `samples` losslessly as `options` describe: a raw codestream, or a
/// JP2/JPH file with `J2K_ENCODE_FILE`. The CPU encoder decodes its own output
/// and compares it with `samples` before returning it.
pub(crate) fn encode_lossless(samples: &[u8], options: J2kCEncodeOptions) -> Result<Vec<u8>> {
    if options.flags & !J2K_ENCODE_FLAGS != 0 || options.reserved != 0 {
        return Err(Error::argument(
            "unknown encode flags, or nonzero reserved field",
        ));
    }
    let components = u16::try_from(options.components)
        .map_err(|_| Error::argument("component count exceeds 65535"))?;
    let bit_depth = u8::try_from(options.bit_depth)
        .ok()
        .filter(|bits| (1..=16).contains(bits))
        .ok_or_else(|| Error::argument("bit depth must be between 1 and 16"))?;
    let ht = options.flags & J2K_ENCODE_HTJ2K != 0;
    let input = J2kLosslessSamples::new(
        samples,
        options.width,
        options.height,
        components,
        bit_depth,
        options.flags & J2K_ENCODE_SIGNED != 0,
    )?;
    let settings = J2kLosslessEncodeOptions::default()
        .with_cpu_only_backend()
        .with_block_coding_mode(if ht {
            J2kBlockCodingMode::HighThroughput
        } else {
            J2kBlockCodingMode::Classic
        });
    let codestream = encode_j2k_lossless(input, &settings)?.codestream;
    if options.flags & J2K_ENCODE_FILE == 0 {
        return Ok(codestream);
    }
    let wrap = if ht {
        J2kFileWrapOptions::jph()
    } else {
        J2kFileWrapOptions::jp2()
    };
    Ok(wrap_j2k_codestream(&codestream, wrap)?)
}
