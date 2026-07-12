// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compact packetization accelerator acceptance, failure, and fallback behavior.

use super::*;
use crate::J2kPacketizationProgressionOrder;

struct FakePacketizationAccelerator {
    output: Option<Vec<u8>>,
    error: Option<&'static str>,
    calls: usize,
}

fn reserved_vector<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    assert!(values.try_reserve_exact(capacity).is_ok());
    values
}

impl J2kEncodeStageAccelerator for FakePacketizationAccelerator {
    fn encode_packetization(
        &mut self,
        _job: J2kPacketizationEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
        self.calls += 1;
        if let Some(error) = self.error {
            return Err(crate::J2kEncodeStageError::internal_invariant(error));
        }
        Ok(self.output.take())
    }
}

fn empty_packetization_job() -> J2kPacketizationEncodeJob<'static> {
    J2kPacketizationEncodeJob {
        resolution_count: 0,
        num_layers: 1,
        num_components: 1,
        code_block_count: 0,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[],
        resolutions: &[],
    }
}

fn packetize_with_fake(
    session: &NativeEncodeSession<'_>,
    phase: &NativeEncodePhase<'_, '_>,
    phase_bytes: usize,
    accelerator: &mut FakePacketizationAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    packetize_compact_job(
        &empty_packetization_job(),
        phase,
        phase_bytes,
        0,
        session,
        accelerator,
    )
}

#[test]
fn compact_accelerator_packet_output_accepts_exact_cap_without_copying() {
    let mut output = reserved_vector::<u8>(17);
    output.extend_from_slice(&[2, 3, 5]);
    let output_capacity = output.capacity();
    let output_ptr = output.as_ptr();
    let phase_bytes = 11;
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        phase_bytes + output_capacity,
    )
    .expect("exact compact packet session");
    let phase = session
        .checked_phase(phase_bytes, PACKET_OWNERS)
        .expect("compact packet phase");
    let mut accelerator = FakePacketizationAccelerator {
        output: Some(output),
        error: None,
        calls: 0,
    };

    let accepted = packetize_with_fake(&session, &phase, phase_bytes, &mut accelerator)
        .expect("exact compact accelerator output");

    assert_eq!(accelerator.calls, 1);
    assert_eq!(accepted.as_ptr(), output_ptr);
    assert_eq!(accepted.capacity(), output_capacity);
    assert_eq!(accepted, [2, 3, 5]);
}

#[test]
fn compact_accelerator_packet_output_rejects_cap_minus_one_without_scalar_fallback() {
    let output = reserved_vector::<u8>(19);
    let output_capacity = output.capacity();
    let phase_bytes = 7;
    let cap = phase_bytes + output_capacity - 1;
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("compact packet owners remain below cap");
    let phase = session
        .checked_phase(phase_bytes, PACKET_OWNERS)
        .expect("compact packet phase");
    let mut accelerator = FakePacketizationAccelerator {
        output: Some(output),
        error: None,
        calls: 0,
    };

    let error = packetize_with_fake(&session, &phase, phase_bytes, &mut accelerator)
        .expect_err("cap-minus-one accelerator output must not enter scalar fallback");

    assert_eq!(accelerator.calls, 1);
    assert_eq!(
        error.into_encode_error(),
        EncodeError::AllocationTooLarge {
            what: ACCELERATOR_OUTPUT,
            requested: phase_bytes + output_capacity,
            cap,
        }
    );
}

#[test]
fn compact_accelerator_packet_failure_keeps_accelerator_category() {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("compact packet session");
    let phase = session
        .checked_phase(0, PACKET_OWNERS)
        .expect("compact packet phase");
    let mut accelerator = FakePacketizationAccelerator {
        output: None,
        error: Some("synthetic compact packet failure"),
        calls: 0,
    };

    let error = packetize_with_fake(&session, &phase, 0, &mut accelerator)
        .expect_err("compact packet backend failure must not enter scalar fallback");

    assert_eq!(accelerator.calls, 1);
    assert_eq!(
        error.into_encode_error(),
        EncodeError::Accelerator {
            operation: "compact preencoded 9/7 packetization",
            source: crate::J2kEncodeStageError::internal_invariant(
                "synthetic compact packet failure",
            ),
        }
    );
}

#[test]
fn compact_accelerator_decline_runs_the_checked_scalar_fallback() {
    let job = empty_packetization_job();
    let expected = packet_encode::form_borrowed_packetization_scalar(job, 0)
        .expect("reference scalar fallback");
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("compact scalar session");
    let phase = session
        .checked_phase(0, PACKET_OWNERS)
        .expect("compact scalar phase");
    let mut accelerator = FakePacketizationAccelerator {
        output: None,
        error: None,
        calls: 0,
    };

    let actual = packetize_compact_job(&job, &phase, 0, 0, &session, &mut accelerator)
        .expect("declined compact packetization falls back");

    assert_eq!(accelerator.calls, 1);
    assert_eq!(actual, expected);
}
