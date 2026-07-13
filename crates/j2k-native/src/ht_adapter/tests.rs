// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::{
    decode_ht_sigprop_benchmark_state, prepare_ht_sigprop_benchmark_state, HtSigPropBenchmarkState,
};
use crate::{DecodeError, DecodeErrorClass, DecodingError, HtCodeBlockDecodeJob};

const SIGPROP_BLOCK: [u8; 12] = [
    0x0E, 0xB2, 0x3E, 0x30, 0xFD, 0x6B, 0x5C, 0x7A, 0xF7, 0x56, 0x00, 0x02,
];

fn sigprop_job(data: &[u8]) -> HtCodeBlockDecodeJob<'_> {
    HtCodeBlockDecodeJob {
        data,
        cleanup_length: 11,
        refinement_length: 1,
        width: 4,
        height: 4,
        output_stride: 4,
        missing_bit_planes: 4,
        number_of_coding_passes: 2,
        num_bitplanes: 6,
        roi_shift: 0,
        stripe_causal: false,
        strict: true,
        dequantization_step: 1.0,
    }
}

fn owned_state() -> HtSigPropBenchmarkState {
    let payload = Vec::from(SIGPROP_BLOCK);
    prepare_ht_sigprop_benchmark_state(sigprop_job(&payload))
        .expect("valid two-pass block prepares benchmark state")
}

#[test]
fn prepared_state_owns_inputs_and_decodes_exact_sigprop_output() {
    let mut state = owned_state();
    assert_eq!(state.output_len(), 16);

    let mut output = vec![0_u32; state.output_len()];
    decode_ht_sigprop_benchmark_state(&mut state, &mut output)
        .expect("owned benchmark state decodes after payload drop");

    assert_eq!(
        output,
        vec![0, 0x0300_0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
    );
}

#[test]
fn short_output_fails_transactionally_and_state_remains_usable() {
    let mut state = owned_state();
    let mut short = vec![0xDEAD_BEEF; state.output_len() - 1];

    let error = decode_ht_sigprop_benchmark_state(&mut state, &mut short)
        .expect_err("short SigProp output must fail");
    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
    );
    assert_eq!(error.classify(), DecodeErrorClass::Backend);
    assert!(short.iter().all(|&word| word == 0xDEAD_BEEF));

    let mut exact = vec![0_u32; state.output_len()];
    decode_ht_sigprop_benchmark_state(&mut state, &mut exact)
        .expect("failed preflight leaves state reusable");
}

#[test]
fn overflowing_segment_metadata_is_rejected_before_state_construction() {
    let job = HtCodeBlockDecodeJob {
        cleanup_length: u32::MAX,
        refinement_length: 1,
        ..sigprop_job(&SIGPROP_BLOCK)
    };

    let Err(error) = prepare_ht_sigprop_benchmark_state(job) else {
        panic!("oversized segment metadata must not construct state");
    };
    assert_eq!(
        error,
        DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
    );
    assert_eq!(error.classify(), DecodeErrorClass::Backend);
}
