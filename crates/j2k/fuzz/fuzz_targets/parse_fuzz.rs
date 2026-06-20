#![no_main]

use j2k::J2kDecoder;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = J2kDecoder::inspect(data);
});
