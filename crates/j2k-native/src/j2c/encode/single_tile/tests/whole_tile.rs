// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{
    BlockCodingMode, EncodeOptions, J2kResidentEncodeInput, NativeEncodeRetainedInput,
    NativeEncodeSession, ResidentHtj2kEncodeError, Vec,
};
use super::super::accelerator::{encode_complete_resident_ht_tile, try_encode_complete_ht_tile};
use super::super::ownership::single_tile_plan_retained_bytes;
use super::super::plan::{
    build_single_tile_plan, validate_encode_request, SingleTilePlan, ValidatedEncodeRoute,
};

struct FakeWholeTileAccelerator {
    output: Option<Vec<u8>>,
    error: Option<&'static str>,
    calls: usize,
}

impl FakeWholeTileAccelerator {
    fn dispatch(&mut self) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
        self.calls += 1;
        if let Some(error) = self.error {
            return Err(crate::J2kEncodeStageError::internal_invariant(error));
        }
        Ok(self.output.take())
    }
}

impl crate::J2kEncodeStageAccelerator for FakeWholeTileAccelerator {
    fn encode_htj2k_tile(
        &mut self,
        _job: crate::J2kHtj2kTileEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
        self.dispatch()
    }

    fn encode_resident_htj2k_tile(
        &mut self,
        _job: crate::J2kResidentHtj2kTileEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
        self.dispatch()
    }
}

fn vector_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small whole-tile output test allocation");
    values
}

fn whole_tile_fixture() -> (Vec<u8>, EncodeOptions, SingleTilePlan) {
    let pixels = (0..8 * 8)
        .map(|index| u8::try_from((index * 23 + 5) & 0xff).expect("sample fits u8"))
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        use_ht_block_coding: true,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    };
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("whole-tile plan session");
    let validated = validate_encode_request(
        pixels.len(),
        8,
        8,
        1,
        8,
        &options,
        BlockCodingMode::HighThroughput,
        &[],
        &session,
    )
    .expect("whole-tile request");
    let ValidatedEncodeRoute::SingleTile(validated) = validated else {
        panic!("fixture must be single tile");
    };
    let plan = build_single_tile_plan(
        validated,
        8,
        8,
        1,
        8,
        false,
        &options,
        BlockCodingMode::HighThroughput,
        &[],
        &[],
        &session,
    )
    .expect("whole-tile plan");
    (pixels, options, plan)
}

#[test]
fn whole_tile_accelerator_output_accepts_exact_cap_without_copying() {
    let (pixels, options, plan) = whole_tile_fixture();
    let mut output = vector_with_capacity::<u8>(17);
    output.extend_from_slice(&[1, 4, 9]);
    let output_capacity = output.capacity();
    let output_ptr = output.as_ptr();
    let plan_bytes = single_tile_plan_retained_bytes(&plan).expect("plan bytes");
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        plan_bytes + output_capacity,
    )
    .expect("exact whole-tile output session");
    let mut accelerator = FakeWholeTileAccelerator {
        output: Some(output),
        error: None,
        calls: 0,
    };

    let (tile_data, _) = try_encode_complete_ht_tile(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &options,
        &[],
        &[],
        &plan,
        false,
        &session,
        &mut accelerator,
    )
    .expect("exact whole-tile accelerator output")
    .expect("whole-tile hook accepted");

    assert_eq!(accelerator.calls, 1);
    assert_eq!(tile_data.as_ptr(), output_ptr);
    assert_eq!(tile_data.capacity(), output_capacity);
    assert_eq!(tile_data, [1, 4, 9]);
}

#[test]
fn whole_tile_accelerator_output_rejects_one_byte_over_without_fallback() {
    let (pixels, options, plan) = whole_tile_fixture();
    let output = vector_with_capacity::<u8>(19);
    let output_capacity = output.capacity();
    let plan_bytes = single_tile_plan_retained_bytes(&plan).expect("plan bytes");
    let cap = plan_bytes + output_capacity - 1;
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("whole-tile plan remains below cap");
    let mut accelerator = FakeWholeTileAccelerator {
        output: Some(output),
        error: None,
        calls: 0,
    };

    let error = try_encode_complete_ht_tile(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &options,
        &[],
        &[],
        &plan,
        false,
        &session,
        &mut accelerator,
    )
    .expect_err("one-byte-over whole-tile output");

    assert_eq!(accelerator.calls, 1);
    assert_eq!(
        error.into_encode_error(),
        crate::EncodeError::AllocationTooLarge {
            what: "accelerator whole-tile HTJ2K output",
            requested: plan_bytes + output_capacity,
            cap,
        }
    );
}

#[test]
fn whole_tile_accelerator_decline_and_failure_keep_distinct_categories() {
    let (pixels, options, plan) = whole_tile_fixture();
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("whole-tile session");
    let mut decline = FakeWholeTileAccelerator {
        output: None,
        error: None,
        calls: 0,
    };
    assert!(try_encode_complete_ht_tile(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &options,
        &[],
        &[],
        &plan,
        false,
        &session,
        &mut decline,
    )
    .expect("decline is not an error")
    .is_none());

    let mut failure = FakeWholeTileAccelerator {
        output: None,
        error: Some("synthetic whole-tile backend failure"),
        calls: 0,
    };
    let error = try_encode_complete_ht_tile(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &options,
        &[],
        &[],
        &plan,
        false,
        &session,
        &mut failure,
    )
    .expect_err("whole-tile backend failure");
    assert_eq!(decline.calls, 1);
    assert_eq!(failure.calls, 1);
    assert_eq!(
        error.into_encode_error(),
        crate::EncodeError::Accelerator {
            operation: "whole-tile HTJ2K encode",
            source: crate::J2kEncodeStageError::internal_invariant(
                "synthetic whole-tile backend failure",
            ),
        }
    );
}

#[test]
fn resident_whole_tile_over_cap_keeps_resource_category() {
    let (_, options, plan) = whole_tile_fixture();
    let input = J2kResidentEncodeInput::new(8, 8, 1, 8, false).expect("resident input");
    let output = vector_with_capacity::<u8>(23);
    let output_capacity = output.capacity();
    let plan_bytes = single_tile_plan_retained_bytes(&plan).expect("plan bytes");
    let cap = plan_bytes + output_capacity - 1;
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("resident plan remains below cap");
    let mut accelerator = FakeWholeTileAccelerator {
        output: Some(output),
        error: None,
        calls: 0,
    };

    let error =
        encode_complete_resident_ht_tile(input, &options, &plan, false, &session, &mut accelerator)
            .expect_err("resident output exceeds cap");

    assert_eq!(accelerator.calls, 1);
    assert_eq!(
        error,
        ResidentHtj2kEncodeError::Resource(crate::EncodeError::AllocationTooLarge {
            what: "resident accelerator whole-tile HTJ2K output",
            requested: plan_bytes + output_capacity,
            cap,
        })
    );
}
