// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use j2k_jpegls::{DecodeOptions, Decoder};
use libfuzzer_sys::fuzz_target;

/// The invariant: on any input, `Decoder::with_options` + `decode` either
/// succeeds or returns a typed `JpegLsError`, and never panics. libfuzzer
/// aborts on panic; that is the detection mechanism.
///
/// The output limit stops a fuzzer-crafted SOF55 with enormous dimensions
/// from exhausting memory; the decoder rejects it before allocating.
const MAX_OUTPUT_BYTES: usize = 1 << 20;

fuzz_target!(|data: &[u8]| {
    let options = DecodeOptions::default().with_max_output_bytes(MAX_OUTPUT_BYTES);
    let Ok(decoder) = Decoder::with_options(data, options) else {
        return;
    };
    let _ = decoder.decode();
});
