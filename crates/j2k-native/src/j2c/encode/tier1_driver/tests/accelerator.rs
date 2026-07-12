// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::build::SubBandType;
use crate::{EncodedJ2kCodeBlock, J2kCodeBlockSegment};

#[derive(Default)]
struct BorrowRecordingAccelerator {
    expected_ptr: usize,
    borrowed_directly: bool,
    batch_calls: usize,
}

impl J2kEncodeStageAccelerator for BorrowRecordingAccelerator {
    fn encode_tier1_code_blocks(
        &mut self,
        jobs: &[crate::J2kTier1CodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedJ2kCodeBlock>>> {
        self.batch_calls += 1;
        self.borrowed_directly = jobs
            .first()
            .is_some_and(|job| job.coefficients.as_ptr() as usize == self.expected_ptr);
        Ok(None)
    }
}

#[test]
fn ordinary_i32_coefficients_are_borrowed_without_a_downcast_graph() {
    let fixture = classic_fixture();
    let expected_ptr = match &fixture[0].code_blocks[0].coefficients {
        PreparedCodeBlockCoefficients::I32(values) => values.as_ptr() as usize,
        _ => panic!("classic fixture must use i32 coefficients"),
    };
    let mut accelerator = BorrowRecordingAccelerator {
        expected_ptr,
        ..BorrowRecordingAccelerator::default()
    };
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let encoded = encode_prepared_subbands_for_session(fixture, &session, 0, &mut accelerator)
        .expect("classic Tier-1 encode");

    assert_eq!(accelerator.batch_calls, 1);
    assert!(accelerator.borrowed_directly);
    let reference = bitplane_encode::encode_code_block(
        &[4, -3, 2, 0, -1, 5, 0, 2, 1, -2, 3, 0, -4, 1, 2, -1],
        4,
        4,
        SubBandType::LowLow,
        5,
    );
    assert_eq!(encoded[0].code_blocks[0].data, reference.data);
}

#[derive(Default)]
struct LargeBatchAccelerator {
    batch_calls: usize,
    single_calls: usize,
}

impl J2kEncodeStageAccelerator for LargeBatchAccelerator {
    fn encode_tier1_code_blocks(
        &mut self,
        jobs: &[crate::J2kTier1CodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedJ2kCodeBlock>>> {
        self.batch_calls += 1;
        let mut outputs = exact_vec(jobs.len());
        for _ in jobs {
            let mut data = exact_vec(4_096);
            data.push(0x5a);
            let mut segments = exact_vec(1);
            segments.push(J2kCodeBlockSegment {
                data_offset: 0,
                data_length: 1,
                start_coding_pass: 0,
                end_coding_pass: 7,
                use_arithmetic: true,
            });
            outputs.push(EncodedJ2kCodeBlock {
                data,
                segments,
                number_of_coding_passes: 7,
                missing_bit_planes: 2,
            });
        }
        Ok(Some(outputs))
    }

    fn encode_tier1_code_block(
        &mut self,
        _job: crate::J2kTier1CodeBlockEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<EncodedJ2kCodeBlock>> {
        self.single_calls += 1;
        Ok(None)
    }
}

#[test]
fn accepted_tier1_batch_over_cap_does_not_fall_back() {
    let measurement_session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("measurement session");
    let measured = encode_prepared_subbands_accounted(
        classic_fixture(),
        &measurement_session,
        0,
        &mut LargeBatchAccelerator::default(),
    )
    .expect("measure accelerated Tier-1 peak");
    let cap = measured.peak_phase_bytes - 1;
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("cap-minus-one session");
    let mut accelerator = LargeBatchAccelerator::default();
    let error =
        encode_prepared_subbands_for_session(classic_fixture(), &session, 0, &mut accelerator)
            .expect_err("accepted over-cap accelerator output must fail")
            .into_encode_error();

    assert!(matches!(error, EncodeError::AllocationTooLarge { .. }));
    assert_eq!(accelerator.batch_calls, 1);
    assert_eq!(accelerator.single_calls, 0);
}

const SINGLE_SEGMENT_CAPACITY: usize = 16_384;

#[derive(Default)]
struct SegmentedSingleAccelerator {
    single_calls: usize,
}

impl J2kEncodeStageAccelerator for SegmentedSingleAccelerator {
    fn encode_tier1_code_block(
        &mut self,
        _job: crate::J2kTier1CodeBlockEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<EncodedJ2kCodeBlock>> {
        self.single_calls += 1;
        let mut data = exact_vec(1);
        data.push(0x5a);
        let mut segments = exact_vec(SINGLE_SEGMENT_CAPACITY);
        segments.push(J2kCodeBlockSegment {
            data_offset: 0,
            data_length: 1,
            start_coding_pass: 0,
            end_coding_pass: 7,
            use_arithmetic: true,
        });
        Ok(Some(EncodedJ2kCodeBlock {
            data,
            segments,
            number_of_coding_passes: 7,
            missing_bit_planes: 2,
        }))
    }
}

#[test]
fn serial_accelerator_segment_metadata_is_checked_before_conversion() {
    let metadata_bytes = SINGLE_SEGMENT_CAPACITY * core::mem::size_of::<J2kCodeBlockSegment>();
    let cap = metadata_bytes - 1;
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("segment-accounting session");
    let mut accelerator = SegmentedSingleAccelerator::default();
    let error =
        encode_prepared_subbands_for_session(classic_fixture(), &session, 0, &mut accelerator)
            .expect_err("transient public segment metadata must count against the phase cap")
            .into_encode_error();

    assert!(matches!(
        error,
        EncodeError::AllocationTooLarge {
            what: "serial accelerated classic Tier-1 output",
            requested,
            cap: observed,
        } if requested > metadata_bytes && observed == cap
    ));
    assert_eq!(accelerator.single_calls, 1);
}
