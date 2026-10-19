// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    target_bytes_for_bpp, J2kError, J2kLossyEncodeOptions, J2kLossySamples, J2kRateTarget,
    Unsupported,
};
use crate::encode_j2k_lossy;

mod rate_validation;
mod target_contracts;

const TWO_TO_THE_64: f64 = 18_446_744_073_709_551_616.0;

fn single_pixel_samples() -> J2kLossySamples<'static> {
    J2kLossySamples::new(&[0], 1, 1, 1, 8, false).expect("valid single-pixel samples")
}

#[test]
fn bits_per_pixel_target_rejects_exactly_two_to_the_64_bytes() {
    let bits_per_pixel = TWO_TO_THE_64 * 8.0;
    let options = J2kLossyEncodeOptions::default()
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(bits_per_pixel)));

    let result = encode_j2k_lossy(single_pixel_samples(), &options);

    assert!(matches!(
        result,
        Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy bits-per-pixel target overflows byte target"
        }))
    ));
}

#[test]
fn bits_per_pixel_target_accepts_largest_f64_below_two_to_the_64_bytes() {
    let largest_representable_byte_target = TWO_TO_THE_64.next_down();
    let bits_per_pixel = largest_representable_byte_target * 8.0;

    let target = target_bytes_for_bpp(single_pixel_samples(), bits_per_pixel)
        .expect("largest representable f64 byte target below 2^64 should fit");

    assert_eq!(target, u64::MAX - 2_047);
}

#[test]
fn explicit_lossy_depth_and_component_profile_survive_rate_search() {
    let pixels = (0..64 * 64 * 3)
        .map(|i| u8::try_from(i % 251).unwrap())
        .collect::<Vec<_>>();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 3, 8, false).unwrap();
    let options = J2kLossyEncodeOptions::default()
        .with_cpu_only_backend()
        .with_max_decomposition_levels(Some(2))
        .with_quantization_profile(Some(vec![vec![1.0; 7], vec![2.0; 7], vec![3.0; 7]]))
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(8.0)));
    let encoded = encode_j2k_lossy(samples, &options).unwrap();
    assert!(encoded.report.actual_bytes <= 4096 + super::byte_target_tolerance(4096));
    let header = j2k_native::inspect_j2k_codestream_header(&encoded.codestream).unwrap();
    assert_eq!(header.resolution_levels, 3);
}
