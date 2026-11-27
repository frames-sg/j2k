// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    // Invariant: never panic. Every input must produce either `Ok(Info)` or
    // `Err(JpegLsError)`. libfuzzer aborts on panic, which is the detection
    // mechanism.
    let _ = j2k_jpegls::Decoder::inspect(data);
});
