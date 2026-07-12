// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::codestream_write::BlockCodingMode;
use crate::j2c::encode::tier1_allocation::prepared_packets_ownership;
use crate::j2c::encode::{
    NativeEncodePipelineError, NativeEncodeRetainedInput, NativeEncodeSession,
    PreparedCodeBlockCoefficients,
};
use crate::EncodeError;

const RETAINED_BASE: usize = 11;

fn samples() -> Vec<i64> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(16)
        .expect("small exact i64 preparation fixture");
    values.extend((0..16).map(|value| i64::from(value) - 7));
    values
}

fn prepare_with_steps_and_cap(
    step_sizes: &[QuantStepSize],
    cap: usize,
) -> NativeEncodePipelineResult<Vec<PreparedResolutionPacket>> {
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("valid exact i64 preparation cap");
    prepare_i64_component_packets(
        samples(),
        I64ComponentPrepareRequest {
            component: 0,
            width: 4,
            height: 4,
            num_levels: 1,
            step_sizes,
            subband_settings: I64SubbandEncodeSettings {
                guard_bits: 1,
                cb_width: 2,
                cb_height: 2,
                roi_shift: 0,
                roi_regions: &[],
                roi_scale: 1,
                block_coding_mode: BlockCodingMode::Classic,
                ht_target_coding_passes: 1,
            },
            retained_base_bytes: RETAINED_BASE,
            session: &session,
        },
    )
}

fn prepare_with_cap(cap: usize) -> NativeEncodePipelineResult<Vec<PreparedResolutionPacket>> {
    let steps = [QuantStepSize {
        exponent: 25,
        mantissa: 0,
    }; 4];
    prepare_with_steps_and_cap(&steps, cap)
}

#[test]
fn packed_component_preparation_is_i64_exact_and_enforces_peak() {
    let prepared = prepare_with_cap(crate::DEFAULT_MAX_CODEC_BYTES)
        .expect("discover packed i64 preparation peak");
    let coefficient_count = prepared
        .iter()
        .flat_map(|packet| &packet.subbands)
        .flat_map(|subband| &subband.code_blocks)
        .map(|block| match &block.coefficients {
            PreparedCodeBlockCoefficients::I64(values) => values.len(),
            PreparedCodeBlockCoefficients::I32(_) | PreparedCodeBlockCoefficients::Empty => {
                panic!("exact i64 preparation changed coefficient representation")
            }
        })
        .sum::<usize>();
    assert_eq!(coefficient_count, 16);

    let source_bytes = samples().capacity() * core::mem::size_of::<i64>();
    let mut scratch = Vec::<i64>::new();
    scratch
        .try_reserve_exact(4)
        .expect("small exact i64 scratch fixture");
    let scratch_bytes = scratch.capacity() * core::mem::size_of::<i64>();
    let prepared_bytes = prepared_packets_ownership(&prepared, prepared.capacity())
        .expect("prepared ownership")
        .total()
        .expect("prepared byte total");
    let exact_cap = RETAINED_BASE + source_bytes + scratch_bytes + prepared_bytes;
    prepare_with_cap(exact_cap).expect("exact packed i64 preparation peak");
    let error = prepare_with_cap(exact_cap - 1)
        .err()
        .expect("one byte below packed i64 preparation peak must fail");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge {
            requested,
            cap,
            ..
        }) if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn missing_planned_quantization_step_is_an_internal_invariant() {
    let error = prepare_with_steps_and_cap(&[], crate::DEFAULT_MAX_CODEC_BYTES)
        .err()
        .expect("the validated plan must contain its LL quantization step");

    assert!(matches!(
        error,
        NativeEncodePipelineError::InternalInvariant("reversible quantization step missing")
    ));
}

#[test]
fn excessive_roi_bitplanes_remain_an_unsupported_capability() {
    let coefficients = [1_i64];
    let view = PackedSubbandView::try_new(
        &coefficients,
        PackedSubbandRect {
            offset: 0,
            width: 1,
            height: 1,
            row_stride: 1,
        },
    )
    .expect("one-sample packed subband");
    let step_size = QuantStepSize {
        exponent: 1,
        mantissa: 0,
    };
    let session = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
    .expect("valid exact i64 preparation cap");
    let error = prepare_packed_subband_i64(&PackedSubbandRequest {
        view,
        step_size: &step_size,
        sub_band_type: SubBandType::LowLow,
        settings: I64SubbandEncodeSettings {
            guard_bits: u8::MAX,
            cb_width: 1,
            cb_height: 1,
            roi_shift: 2,
            roi_regions: &[],
            roi_scale: 1,
            block_coding_mode: BlockCodingMode::Classic,
            ht_target_coding_passes: 1,
        },
        retained_base_bytes: 0,
        session: &session,
    })
    .err()
    .expect("coded bitplane overflow must be rejected");

    assert!(matches!(
        error,
        NativeEncodePipelineError::Unsupported(
            "ROI maxshift exceeds supported coded bitplane count"
        )
    ));
}
