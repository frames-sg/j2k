// SPDX-License-Identifier: MIT OR Apache-2.0

use super::collect_encode_distribution;
use alloc::vec;

#[test]
fn distribution_rejects_mismatched_and_oversized_geometry_before_walking() {
    assert_eq!(
        collect_encode_distribution(&[0], 2, 2, 8)
            .expect_err("mismatched coefficient storage is invalid"),
        crate::EncodeError::InvalidInput {
            what: "contiguous coefficient block length mismatch",
        }
    );
    for (width, height) in [(1025_u32, 1_u32), (1024, 5)] {
        let coefficients = vec![0; width as usize * height as usize];
        assert_eq!(
            collect_encode_distribution(&coefficients, width, height, 8)
                .expect_err("oversized geometry is invalid"),
            crate::EncodeError::InvalidInput {
                what: "Tier-1 code-block geometry exceeds JPEG 2000 limits",
            }
        );
    }
}

#[test]
fn distribution_rejects_bitplane_range_and_magnitude_with_typed_input_errors() {
    assert_eq!(
        collect_encode_distribution(&[0], 1, 1, 0).expect_err("zero bitplanes are invalid"),
        crate::EncodeError::InvalidInput {
            what: "HTJ2K scalar encoder currently supports 1..=31 bitplanes",
        }
    );
    assert_eq!(
        collect_encode_distribution(&[2], 1, 1, 1).expect_err("magnitude needs two bitplanes"),
        crate::EncodeError::InvalidInput {
            what: "HTJ2K block magnitude exceeds configured bitplane count",
        }
    );
}
