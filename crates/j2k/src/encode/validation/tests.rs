// SPDX-License-Identifier: MIT OR Apache-2.0

use super::first_native_sample_mismatch;

mod errors;

#[test]
fn lossless_comparison_masks_unused_unsigned_bits() {
    assert_eq!(
        first_native_sample_mismatch(&[0xff, 0xff], &[0xff, 0x0f], 12, false),
        None
    );
}

#[test]
fn lossless_comparison_sign_extends_non_byte_aligned_values() {
    assert_eq!(
        first_native_sample_mismatch(&[0x00, 0x08], &[0x00, 0xf8], 12, true),
        None
    );
}

#[test]
fn lossless_comparison_reports_sample_not_byte_index() {
    assert_eq!(
        first_native_sample_mismatch(
            &[0x01, 0x00, 0x02, 0x00],
            &[0x01, 0x00, 0x03, 0x00],
            12,
            false,
        ),
        Some(1)
    );
}
