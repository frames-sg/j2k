// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packetization accelerator ownership and retained-capacity regressions.

use super::*;
use crate::{J2kPacketizationCodeBlock, J2kPacketizationResolution, J2kPacketizationSubband};

struct FakePacketizationAccelerator {
    output: Option<Vec<u8>>,
    error: Option<&'static str>,
    calls: usize,
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

fn vector_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small accelerator-output test allocation");
    values
}

fn empty_packetization_job() -> J2kPacketizationEncodeJob<'static> {
    J2kPacketizationEncodeJob {
        resolution_count: 0,
        num_layers: 1,
        num_components: 1,
        code_block_count: 0,
        progression_order: crate::J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[],
        resolutions: &[],
    }
}

#[test]
fn packet_accelerator_output_accepts_exact_cap_without_copying() {
    let mut output = vector_with_capacity::<u8>(11);
    output.extend_from_slice(&[3, 5, 8]);
    let output_capacity = output.capacity();
    let output_ptr = output.as_ptr();
    let phase_bytes = 7;
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        phase_bytes + output_capacity,
    )
    .expect("exact packet output session");
    let phase = session
        .checked_phase(phase_bytes, "test packet owners")
        .expect("packet phase");
    let mut accelerator = FakePacketizationAccelerator {
        output: Some(output),
        error: None,
        calls: 0,
    };

    let packetized =
        try_packetization_accelerator(empty_packetization_job(), &phase, &mut accelerator)
            .expect("exact accelerator output")
            .expect("accelerator accepted packetization");

    assert_eq!(accelerator.calls, 1);
    assert_eq!(packetized.data.as_ptr(), output_ptr);
    assert_eq!(packetized.data.capacity(), output_capacity);
    assert_eq!(packetized.data, [3, 5, 8]);
    assert!(packetized.packet_lengths.is_empty());
    assert!(packetized.packet_headers.is_empty());
}

#[test]
fn packet_accelerator_output_rejects_one_byte_over_without_fallback() {
    let output = vector_with_capacity::<u8>(13);
    let output_capacity = output.capacity();
    let phase_bytes = 5;
    let cap = phase_bytes + output_capacity - 1;
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("packet owner phase remains below cap");
    let phase = session
        .checked_phase(phase_bytes, "test packet owners")
        .expect("packet phase");
    let mut accelerator = FakePacketizationAccelerator {
        output: Some(output),
        error: None,
        calls: 0,
    };

    let error = try_packetization_accelerator(empty_packetization_job(), &phase, &mut accelerator)
        .err()
        .expect("one-byte-over packet output");

    assert_eq!(accelerator.calls, 1);
    assert_eq!(
        error.into_encode_error(),
        crate::EncodeError::AllocationTooLarge {
            what: "accelerator packetization output",
            requested: phase_bytes + output_capacity,
            cap,
        }
    );
}

#[test]
fn packet_accelerator_decline_and_failure_keep_distinct_categories() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("packet session");
    let phase = session
        .checked_phase(0, "empty packet phase")
        .expect("phase");
    let mut decline = FakePacketizationAccelerator {
        output: None,
        error: None,
        calls: 0,
    };
    assert!(
        try_packetization_accelerator(empty_packetization_job(), &phase, &mut decline)
            .expect("decline is not an error")
            .is_none()
    );

    let mut failure = FakePacketizationAccelerator {
        output: None,
        error: Some("synthetic packet backend failure"),
        calls: 0,
    };
    let error = try_packetization_accelerator(empty_packetization_job(), &phase, &mut failure)
        .err()
        .expect("backend failure");
    assert_eq!(decline.calls, 1);
    assert_eq!(failure.calls, 1);
    assert_eq!(
        error.into_encode_error(),
        crate::EncodeError::Accelerator {
            operation: "packetization",
            source: crate::J2kEncodeStageError::internal_invariant(
                "synthetic packet backend failure",
            ),
        }
    );
}

#[test]
fn packetized_accelerator_output_counts_nested_metadata_capacities() {
    let data = vector_with_capacity::<u8>(5);
    let packet_lengths = vector_with_capacity::<u32>(3);
    let header = vector_with_capacity::<u8>(7);
    let header_capacity = header.capacity();
    let mut packet_headers = vector_with_capacity::<Vec<u8>>(2);
    packet_headers.push(header);
    let expected = data.capacity()
        + packet_lengths.capacity() * core::mem::size_of::<u32>()
        + packet_headers.capacity() * core::mem::size_of::<Vec<u8>>()
        + header_capacity;
    let packetized = packet_encode::PacketizedTileData {
        data,
        packet_lengths,
        packet_headers,
    };

    assert_eq!(
        packet_encode::packetized_tile_retained_bytes(&packetized)
            .expect("nested packet output bytes"),
        expected
    );
}

#[test]
fn packet_accelerator_phase_counts_nested_public_metadata_capacities() {
    let payload = [2_u8, 7];
    let mut code_blocks = vector_with_capacity::<J2kPacketizationCodeBlock<'_>>(4);
    code_blocks.push(J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: 2,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 0,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    });
    let mut subbands = vector_with_capacity::<J2kPacketizationSubband<'_>>(3);
    subbands.push(J2kPacketizationSubband {
        code_blocks,
        num_cbs_x: 1,
        num_cbs_y: 1,
    });
    let mut resolutions = vector_with_capacity::<J2kPacketizationResolution<'_>>(2);
    resolutions.push(J2kPacketizationResolution { subbands });
    let additional = 11;
    let expected = additional
        + resolutions.capacity() * core::mem::size_of::<J2kPacketizationResolution<'_>>()
        + resolutions[0].subbands.capacity() * core::mem::size_of::<J2kPacketizationSubband<'_>>()
        + resolutions[0].subbands[0].code_blocks.capacity()
            * core::mem::size_of::<J2kPacketizationCodeBlock<'_>>();

    assert_eq!(
        packet_encode::packet_metadata_retained_bytes(
            &resolutions,
            resolutions.capacity(),
            additional,
        )
        .expect("nested public packet metadata bytes"),
        expected
    );
}
