#![no_main]

//! Decodes a codestream at native bit depth, and depending on the input
//! length also a small region and the reversible 5/3 coefficients. Taking the
//! options from the length rather than from a leading byte keeps seed
//! codestreams valid from their first byte.

use j2k_native::{DecodeSettings, Image};
use libfuzzer_sys::fuzz_target;

const MAX_PIXELS: u64 = 1 << 18;

fuzz_target!(|data: &[u8]| {
    let mode = data.len().to_le_bytes()[0];
    let settings = if mode & 1 == 0 {
        DecodeSettings::strict()
    } else {
        DecodeSettings::lenient()
    };
    let Ok(image) = Image::new(data, &settings) else {
        return;
    };
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return;
    }
    let _ = image.decode_native();
    if mode & 2 != 0 {
        let x = u32::from(mode >> 4) % width;
        let y = u32::from(mode >> 2 & 3) % height;
        let _ = image.decode_native_region((x, y, (width - x).min(17), (height - y).min(13)));
    }
    if mode & 4 != 0 {
        let _ = image.decode_reversible_53_coefficients();
    }
});
