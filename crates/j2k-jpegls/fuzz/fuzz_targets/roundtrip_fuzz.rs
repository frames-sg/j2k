// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use j2k_jpegls::{
    encode_jpegls, DecodedSamples, Decoder, InterleaveMode, JpegLsEncodeOptions, JpegLsSamples,
};
use libfuzzer_sys::fuzz_target;

// The invariant: any samples the encoder accepts decode back exactly when
// lossless and within NEAR otherwise. The first bytes choose the
// parameters; the rest are samples.
fuzz_target!(|data: &[u8]| {
    let Some((&[bits, near, shape, interleave, restart], samples)) = data.split_first_chunk::<5>()
    else {
        return;
    };
    let bits = 2 + bits % 15;
    let components = 1 + shape % 4;
    let width = 1 + u32::from(shape >> 2) % 32;
    let max_value = (1_u32 << bits) - 1;
    let near = u8::try_from(u32::from(near) % (max_value / 2 + 1)).unwrap_or(0);
    let mut options = JpegLsEncodeOptions::new(bits).with_near(near);
    options.interleave = match interleave % 3 {
        0 => InterleaveMode::None,
        1 => InterleaveMode::Line,
        _ => InterleaveMode::Sample,
    };
    if restart % 4 != 0 {
        options = options.with_restart_interval(u16::from(restart % 4));
    }
    let mask = u16::try_from(max_value).unwrap_or(u16::MAX);
    let values: Vec<u16> = samples
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]) & mask)
        .collect();
    let per_row = width as usize * usize::from(components);
    let height = values.len() / per_row;
    if height == 0 {
        return;
    }
    let values = &values[..height * per_row];
    let input = JpegLsSamples::U16 {
        data: values,
        width,
        height: u32::try_from(height).unwrap_or(u32::MAX),
        components,
    };
    let encoded = encode_jpegls(input, &options).expect("encoder accepts in-range samples");
    let decoded = Decoder::new(&encoded.data)
        .and_then(|decoder| decoder.decode())
        .expect("encoder output decodes");
    let decoded: Vec<u16> = match decoded.samples {
        DecodedSamples::U8(samples) => samples.into_iter().map(u16::from).collect(),
        DecodedSamples::U16(samples) => samples,
        _ => unreachable!("only 8- and 16-bit samples exist"),
    };
    for (&expected, &actual) in values.iter().zip(&decoded) {
        assert!(expected.abs_diff(actual) <= u16::from(near));
    }
});
