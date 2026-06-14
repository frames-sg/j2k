// SPDX-License-Identifier: Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use signinum_transcode::{jpeg_to_htj2k, JpegToHtj2kOptions};

const MAX_INPUT_BYTES: usize = 256 * 1024;
const MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let Ok(encoded) = jpeg_to_htj2k(data, &JpegToHtj2kOptions::lossless_53()) else {
        return;
    };

    if encoded.codestream.len() > MAX_OUTPUT_BYTES {
        return;
    }
});
