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
