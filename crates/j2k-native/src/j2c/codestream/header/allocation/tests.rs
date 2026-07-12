// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use core::mem::size_of;

use super::*;
use crate::error::DecodeError;
use crate::j2c::codestream::{
    CodeBlockStyle, CodingStyleFlags, CodingStyleParameters, ProgressionOrder, QuantizationStyle,
    WaveletTransform,
};

fn none_vec<T>(len: usize) -> Vec<Option<T>> {
    let mut values = Vec::with_capacity(len);
    values.resize_with(len, || None);
    values
}

#[test]
fn component_preflight_rejects_quantization_clone_amplification() {
    let component_count = 4_097;
    let component_size = ComponentSizeInfo {
        precision: 8,
        signed: false,
        horizontal_resolution: 1,
        vertical_resolution: 1,
    };
    let coding_default = CodingStyleDefault {
        progression_order: ProgressionOrder::LayerResolutionComponentPosition,
        num_layers: 1,
        mct: false,
        component_parameters: CodingStyleComponent {
            flags: CodingStyleFlags::default(),
            parameters: CodingStyleParameters {
                num_decomposition_levels: 0,
                num_resolution_levels: 1,
                code_block_width: 6,
                code_block_height: 6,
                code_block_style: CodeBlockStyle::default(),
                transformation: WaveletTransform::Reversible53,
                precinct_exponents: vec![(15, 15)],
            },
        },
    };
    let quantization_default = QuantizationInfo {
        quantization_style: QuantizationStyle::ScalarExpounded,
        guard_bits: 2,
        step_sizes: vec![
            StepSize {
                mantissa: 0,
                exponent: 8,
            };
            32_766
        ],
    };
    let component_sizes = vec![component_size; component_count];
    let coding_overrides = none_vec(component_count);
    let quantization_overrides = none_vec(component_count);
    let mut budget = HeaderMarkerBudget::default();

    assert!(matches!(
        account_component_metadata_peak(
            &component_sizes,
            &coding_overrides,
            &quantization_overrides,
            &coding_default,
            &quantization_default,
            &mut budget,
        ),
        Err(DecodeError::AllocationTooLarge { requested, cap, .. })
            if requested > cap && cap == DEFAULT_MAX_DECODE_BYTES
    ));
}

#[test]
fn retained_container_baseline_constrains_header_before_marker_growth() {
    let fixed_header_bytes = size_of::<super::super::super::Header<'static>>();
    let exact_baseline = DEFAULT_MAX_DECODE_BYTES - fixed_header_bytes;
    let exact = HeaderMarkerBudget::with_retained_baseline(exact_baseline)
        .expect("container plus fixed header fits exactly");
    assert_eq!(exact.remaining_bytes(), 0);

    assert!(matches!(
        HeaderMarkerBudget::with_retained_baseline(exact_baseline + 1),
        Err(DecodeError::AllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_DECODE_BYTES,
            ..
        }) if requested == DEFAULT_MAX_DECODE_BYTES + 1
    ));
}
