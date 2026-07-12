// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::string::ToString;

use super::{J2kResidentEncodeInput, J2kResidentEncodeInputError, J2kResidentHtj2kTileEncodeJob};
use crate::{
    J2kPacketizationProgressionOrder, MAX_JPEG2000_PART1_COMPONENTS,
    MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH,
};

#[test]
fn resident_input_accepts_part1_component_and_precision_boundaries() {
    let input = J2kResidentEncodeInput::new(
        1,
        1,
        MAX_JPEG2000_PART1_COMPONENTS,
        MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH,
        true,
    )
    .expect("Part 1 boundaries are valid");
    assert_eq!(input.num_components(), MAX_JPEG2000_PART1_COMPONENTS);
    assert_eq!(input.bit_depth(), MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH);
    assert!(input.signed());
}

#[test]
fn resident_tile_job_accessors_delegate_to_the_validated_input() {
    let input = J2kResidentEncodeInput::new(17, 9, 3, 12, true).expect("valid resident input");
    let job = J2kResidentHtj2kTileEncodeJob {
        input,
        num_decomposition_levels: 1,
        reversible: true,
        use_mct: true,
        guard_bits: 1,
        code_block_width: 64,
        code_block_height: 64,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        component_sampling: &[(1, 1); 3],
        quantization_steps: &[(0, 0)],
    };

    assert_eq!(job.width(), 17);
    assert_eq!(job.height(), 9);
    assert_eq!(job.num_components(), 3);
    assert_eq!(job.bit_depth(), 12);
    assert!(job.signed());
}

#[test]
fn resident_input_rejects_values_above_part1_boundaries() {
    assert_eq!(
        J2kResidentEncodeInput::new(1, 1, MAX_JPEG2000_PART1_COMPONENTS + 1, 8, false,),
        Err(J2kResidentEncodeInputError::ComponentCountOutOfRange {
            num_components: MAX_JPEG2000_PART1_COMPONENTS + 1,
        })
    );
    assert_eq!(
        J2kResidentEncodeInput::new(1, 1, 1, MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH + 1, false,),
        Err(J2kResidentEncodeInputError::PrecisionOutOfRange {
            bit_depth: MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH + 1,
        })
    );
}

#[test]
fn resident_input_errors_keep_stable_reasons_and_typed_context() {
    let empty = J2kResidentEncodeInput::new(0, 7, 1, 8, false).expect_err("zero width must fail");
    assert_eq!(
        empty,
        J2kResidentEncodeInputError::EmptyGeometry {
            width: 0,
            height: 7,
        }
    );
    assert_eq!(
        empty.reason(),
        "resident encode input dimensions must be non-zero"
    );
    assert_eq!(empty.to_string(), empty.reason());

    let overflow = J2kResidentEncodeInput::new(
        u32::MAX,
        u32::MAX,
        MAX_JPEG2000_PART1_COMPONENTS,
        MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH,
        false,
    )
    .expect_err("unaddressable logical storage must fail");
    assert_eq!(overflow, J2kResidentEncodeInputError::AddressSpaceOverflow);
    assert_eq!(
        overflow.reason(),
        "resident encode input dimensions overflow address space"
    );
}
