// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::EncodedJ2kCodeBlock;

struct FailingBatchAccelerator;

impl J2kEncodeStageAccelerator for FailingBatchAccelerator {
    fn encode_tier1_code_blocks(
        &mut self,
        _jobs: &[crate::J2kTier1CodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedJ2kCodeBlock>>> {
        Err(crate::J2kEncodeStageError::internal_invariant(
            "injected Tier-1 batch failure",
        ))
    }
}

struct MalformedBatchAccelerator;

impl J2kEncodeStageAccelerator for MalformedBatchAccelerator {
    fn encode_tier1_code_blocks(
        &mut self,
        _jobs: &[crate::J2kTier1CodeBlockEncodeJob<'_>],
    ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedJ2kCodeBlock>>> {
        Ok(Some(Vec::new()))
    }
}

#[test]
fn tier1_batch_failure_keeps_the_accelerator_operation_category() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        classic_fixture(),
        &session,
        0,
        &mut FailingBatchAccelerator,
    )
    .expect_err("injected accelerator failure must surface")
    .into_encode_error();
    assert_eq!(
        error,
        EncodeError::Accelerator {
            operation: "classic Tier-1 code-block batch encode",
            source: crate::J2kEncodeStageError::internal_invariant("injected Tier-1 batch failure",),
        }
    );
}

#[test]
fn malformed_tier1_batch_output_keeps_the_accelerator_operation_category() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        classic_fixture(),
        &session,
        0,
        &mut MalformedBatchAccelerator,
    )
    .expect_err("malformed accelerator output must surface")
    .into_encode_error();
    assert_eq!(
        error,
        EncodeError::Accelerator {
            operation: "classic Tier-1 code-block batch encode",
            source: crate::J2kEncodeStageError::internal_invariant(
                "accelerated classic code-block batch length mismatch",
            ),
        }
    );
}
