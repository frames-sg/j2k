// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn fallible_rgb_deinterleave_preserves_fast_path_samples() {
    // The larger image fills its planes on the Rayon pool when allowed.
    for (num_pixels, parallel) in [(32, false), (32, true), (70_000, false), (70_000, true)] {
        let pixels = (0..3 * num_pixels)
            .map(|value| u8::try_from((value * 19 + 7) & 0xff).expect("sample fits u8"))
            .collect::<Vec<_>>();
        assert_eq!(
            try_deinterleave_to_f32(&pixels, num_pixels, 3, 8, false, parallel)
                .expect("fallible deinterleave"),
            deinterleave_to_f32(&pixels, num_pixels, 3, 8, false),
            "{num_pixels} pixels, parallel {parallel}"
        );
    }
}

#[test]
fn fallible_signed_multibyte_deinterleave_preserves_samples() {
    let mut pixels = Vec::new();
    for values in [[-2048_i16, 17], [1023, -9], [0, 2047]] {
        for value in values {
            pixels.extend_from_slice(&(value.cast_unsigned() & 0x0fff).to_le_bytes());
        }
    }
    assert_eq!(
        try_deinterleave_to_f32(&pixels, 3, 2, 12, true, false).expect("fallible deinterleave"),
        deinterleave_to_f32(&pixels, 3, 2, 12, true)
    );
}

#[test]
fn fallible_deinterleave_matches_the_reference_on_and_off_the_pool() {
    // Images of at least 3 x 65536 samples fill their planes on the Rayon pool
    // when allowed; a short buffer leaves the rest of each plane zero.
    for (num_components, bit_depth, signed) in [
        (1_u16, 8_u8, false),
        (1, 16, false),
        (2, 12, true),
        (3, 16, false),
        (4, 8, false),
        (1, 24, true),
        (1, 32, false),
        (2, 38, false),
    ] {
        let bytes_per_sample = usize::from(bit_depth).div_ceil(8);
        for num_pixels in [32_usize, 200_000] {
            let pixels = (0..num_pixels * usize::from(num_components) * bytes_per_sample)
                .map(|index| u8::try_from((index * 37 + 11) & 0xff).expect("byte fits u8"))
                .collect::<Vec<_>>();
            for (pixels, parallel) in [
                (&pixels[..], false),
                (&pixels[..], true),
                (&pixels[..=pixels.len() / 2], true),
            ] {
                assert_eq!(
                    try_deinterleave_to_f32(
                        pixels,
                        num_pixels,
                        num_components,
                        bit_depth,
                        signed,
                        parallel
                    )
                    .expect("fallible deinterleave"),
                    deinterleave_to_f32(pixels, num_pixels, num_components, bit_depth, signed),
                    "{num_components}x{bit_depth} bits, {num_pixels} pixels, {} bytes, parallel {parallel}",
                    pixels.len()
                );
            }
        }
    }
}
