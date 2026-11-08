// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kCodeBlockSegment};

fn assert_accelerator_error(
    error: NativeEncodePipelineError,
    operation: &'static str,
    detail: &'static str,
) {
    assert_eq!(
        error.into_encode_error(),
        EncodeError::Accelerator {
            operation,
            source: crate::J2kEncodeStageError::internal_invariant(detail),
        }
    );
}

fn encoded_ht_block(
    data_len: usize,
    cleanup_length: u32,
    refinement_length: u32,
    num_coding_passes: u8,
) -> EncodedHtJ2kCodeBlock {
    let mut data = exact_vec(data_len);
    data.resize(data_len, 0x5a);
    EncodedHtJ2kCodeBlock {
        data,
        cleanup_length,
        refinement_length,
        num_coding_passes,
        num_zero_bitplanes: 0,
    }
}

struct MalformedHtBatchAccelerator;

impl J2kEncodeStageAccelerator for MalformedHtBatchAccelerator {
    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedHtJ2kCodeBlock>>> {
        let mut outputs = exact_vec(jobs.len());
        for _ in jobs {
            outputs.push(encoded_ht_block(3, 2, 2, 3));
        }
        Ok(Some(outputs))
    }
}

#[test]
fn malformed_ht_batch_metadata_keeps_the_accelerator_operation_category() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        ht_refinement_fixture(),
        &session,
        0,
        &mut MalformedHtBatchAccelerator,
    )
    .expect_err("malformed HT metadata must surface at the batch boundary");

    assert_accelerator_error(
        error,
        "HT Tier-1 code-block batch encode",
        "HTJ2K payload segment length mismatch",
    );
}

struct DivergentHtSingleAccelerator;

impl J2kEncodeStageAccelerator for DivergentHtSingleAccelerator {
    fn encode_ht_code_block(
        &mut self,
        _job: crate::J2kHtCodeBlockEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<EncodedHtJ2kCodeBlock>> {
        Ok(Some(encoded_ht_block(3, 2, 1, 2)))
    }
}

struct OmittedHtBatchAccelerator;

impl J2kEncodeStageAccelerator for OmittedHtBatchAccelerator {
    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedHtJ2kCodeBlock>>> {
        let mut outputs = exact_vec(jobs.len());
        for job in jobs {
            outputs.push(EncodedHtJ2kCodeBlock {
                data: exact_vec(0),
                cleanup_length: 0,
                refinement_length: 0,
                num_coding_passes: 0,
                num_zero_bitplanes: job.total_bitplanes,
            });
        }
        Ok(Some(outputs))
    }
}

#[test]
fn divergent_ht_single_pass_count_keeps_the_accelerator_operation_category() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        ht_refinement_fixture(),
        &session,
        0,
        &mut DivergentHtSingleAccelerator,
    )
    .expect_err("an accelerator must honor the requested HT pass count");

    assert_accelerator_error(
        error,
        "HT Tier-1 code-block encode",
        "accelerated HT code-block coding pass count differs from request",
    );
}

#[test]
fn omitted_nonzero_ht_block_keeps_the_accelerator_operation_category() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        ht_refinement_fixture(),
        &session,
        0,
        &mut OmittedHtBatchAccelerator,
    )
    .expect_err("an accelerator must not silently omit nonzero HT coefficients");

    assert_accelerator_error(
        error,
        "HT Tier-1 code-block batch encode",
        "accelerated HT code-block omitted nonzero coefficients",
    );
}

#[derive(Default)]
struct NonemptyHtBatchAccelerator {
    jobs_seen: usize,
}

impl J2kEncodeStageAccelerator for NonemptyHtBatchAccelerator {
    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedHtJ2kCodeBlock>>> {
        assert!(jobs.iter().all(|job| job.total_bitplanes != 0));
        self.jobs_seen += jobs.len();
        jobs.iter()
            .map(|job| {
                crate::encode_ht_code_block_scalar_with_passes(
                    job.coefficients,
                    job.width,
                    job.height,
                    job.total_bitplanes,
                    job.target_coding_passes,
                )
                .map_err(|error| {
                    crate::J2kEncodeStageError::backend("test", "HT batch encode", error)
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
}

#[test]
fn zero_bitplane_blocks_bypass_batch_accelerators_without_hiding_other_work() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let mut accelerator = NonemptyHtBatchAccelerator::default();
    let encoded = encode_prepared_subbands_for_session(
        ht_fixture_with_zero_bitplane_subband(),
        &session,
        0,
        &mut accelerator,
    )
    .expect("mixed empty and coded HT subbands");

    assert_eq!(accelerator.jobs_seen, 1);
    assert_eq!(encoded[0].code_blocks[0].num_coding_passes, 0);
    assert_eq!(encoded[0].code_blocks[0].data, [] as [u8; 0]);
    assert_eq!(encoded[1].code_blocks[0].num_coding_passes, 3);
    assert_ne!(encoded[1].code_blocks[0].data, [] as [u8; 0]);
}

#[test]
fn zero_bitplane_nonzero_coefficients_are_rejected_before_batch_accelerators() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let mut subbands = ht_zero_bitplane_fixture();
    let PreparedCodeBlockCoefficients::I32(coefficients) =
        &mut subbands[0].code_blocks[0].coefficients
    else {
        panic!("zero-bitplane fixture must use i32 coefficients");
    };
    coefficients[0] = 1;
    let mut accelerator = NonemptyHtBatchAccelerator::default();

    let error = encode_prepared_subbands_for_session(subbands, &session, 0, &mut accelerator)
        .expect_err("nonzero coefficients must not be accepted with zero bitplanes");

    assert_eq!(accelerator.jobs_seen, 0);
    assert_eq!(
        error.into_encode_error(),
        EncodeError::InvalidInput {
            what: "HTJ2K block magnitude exceeds configured bitplane count",
        }
    );
}

#[derive(Default)]
struct NonemptyHtSingleAccelerator {
    batch_calls: usize,
    single_calls: usize,
}

impl J2kEncodeStageAccelerator for NonemptyHtSingleAccelerator {
    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedHtJ2kCodeBlock>>> {
        assert!(jobs.iter().all(|job| job.total_bitplanes != 0));
        self.batch_calls += 1;
        Ok(None)
    }

    fn encode_ht_code_block(
        &mut self,
        job: crate::J2kHtCodeBlockEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<EncodedHtJ2kCodeBlock>> {
        assert_ne!(job.total_bitplanes, 0);
        self.single_calls += 1;
        crate::encode_ht_code_block_scalar_with_passes(
            job.coefficients,
            job.width,
            job.height,
            job.total_bitplanes,
            job.target_coding_passes,
        )
        .map(Some)
        .map_err(|error| crate::J2kEncodeStageError::backend("test", "HT encode", error))
    }
}

#[test]
fn zero_bitplane_blocks_bypass_single_accelerators_after_batch_declines() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let mut accelerator = NonemptyHtSingleAccelerator::default();
    let encoded = encode_prepared_subbands_for_session(
        ht_fixture_with_zero_bitplane_subband(),
        &session,
        0,
        &mut accelerator,
    )
    .expect("serial mixed empty and coded HT subbands");

    assert_eq!(accelerator.batch_calls, 1);
    assert_eq!(accelerator.single_calls, 1);
    assert_eq!(encoded[0].code_blocks[0].num_coding_passes, 0);
    assert_eq!(encoded[1].code_blocks[0].num_coding_passes, 3);
}

#[derive(Clone, Copy)]
enum ClassicMetadataFault {
    TooManyPasses,
    PayloadGap,
    WrongZeroBitplanes,
}

struct MalformedClassicBatchAccelerator(ClassicMetadataFault);

impl MalformedClassicBatchAccelerator {
    fn output(&self) -> EncodedJ2kCodeBlock {
        let (number_of_coding_passes, missing_bit_planes, data_offset, data_length) = match self.0 {
            ClassicMetadataFault::TooManyPasses => (165, 0, 0, 1),
            ClassicMetadataFault::PayloadGap => (7, 2, 1, 0),
            ClassicMetadataFault::WrongZeroBitplanes => (1, 4, 0, 1),
        };
        let mut data = exact_vec(1);
        data.push(0x5a);
        let mut segments = exact_vec(1);
        segments.push(J2kCodeBlockSegment {
            data_offset,
            data_length,
            start_coding_pass: 0,
            end_coding_pass: number_of_coding_passes,
            use_arithmetic: true,
        });
        EncodedJ2kCodeBlock {
            data,
            segments,
            number_of_coding_passes,
            missing_bit_planes,
        }
    }
}

impl J2kEncodeStageAccelerator for MalformedClassicBatchAccelerator {
    fn encode_tier1_code_blocks(
        &mut self,
        jobs: &[crate::J2kTier1CodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedJ2kCodeBlock>>> {
        let mut outputs = exact_vec(jobs.len());
        for _ in jobs {
            outputs.push(self.output());
        }
        Ok(Some(outputs))
    }
}

#[test]
fn excessive_classic_pass_count_keeps_the_accelerator_operation_category() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        classic_fixture(),
        &session,
        0,
        &mut MalformedClassicBatchAccelerator(ClassicMetadataFault::TooManyPasses),
    )
    .expect_err("excessive classic coding passes must be rejected at the boundary");

    assert_accelerator_error(
        error,
        "classic Tier-1 code-block batch encode",
        "accelerated classic code-block coding pass count out of range",
    );
}

#[test]
fn malformed_classic_segments_keep_the_accelerator_operation_category() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        classic_fixture(),
        &session,
        0,
        &mut MalformedClassicBatchAccelerator(ClassicMetadataFault::PayloadGap),
    )
    .expect_err("classic segment gaps must be rejected at the boundary");

    assert_accelerator_error(
        error,
        "classic Tier-1 code-block batch encode",
        "accelerated classic code-block segments do not cover payload contiguously",
    );
}

#[test]
fn wrong_classic_zero_bitplanes_keep_the_accelerator_operation_category() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        classic_fixture(),
        &session,
        0,
        &mut MalformedClassicBatchAccelerator(ClassicMetadataFault::WrongZeroBitplanes),
    )
    .expect_err("classic zero-bitplane metadata must match the input coefficients");

    assert_accelerator_error(
        error,
        "classic Tier-1 code-block batch encode",
        "accelerated classic code-block zero-bitplane metadata mismatch",
    );
}
