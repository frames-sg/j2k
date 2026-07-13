// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    try_clone_coding_parameters, try_clone_quantization_info, CodingStyleParameters,
    QuantizationInfo, StepSize, TileMetadataBudget,
};
use crate::error::{DecodeError, ValidationError};
use crate::j2c::codestream::{CodeBlockStyle, QuantizationStyle, WaveletTransform};
use alloc::vec::Vec;
use core::mem::size_of;

fn coding_parameters(capacity: usize, precincts: &[(u8, u8)]) -> CodingStyleParameters {
    let mut precinct_exponents = Vec::with_capacity(capacity);
    precinct_exponents.extend_from_slice(precincts);
    CodingStyleParameters {
        num_decomposition_levels: 2,
        num_resolution_levels: 3,
        code_block_width: 5,
        code_block_height: 6,
        code_block_style: CodeBlockStyle {
            selective_arithmetic_coding_bypass: true,
            reset_context_probabilities: false,
            termination_on_each_pass: true,
            vertically_causal_context: false,
            segmentation_symbols: true,
            high_throughput_block_coding: false,
        },
        transformation: WaveletTransform::Irreversible97,
        precinct_exponents,
    }
}

fn quantization(capacity: usize, steps: &[(u16, u16)]) -> QuantizationInfo {
    let mut step_sizes = Vec::with_capacity(capacity);
    step_sizes.extend(
        steps
            .iter()
            .map(|&(mantissa, exponent)| StepSize { mantissa, exponent }),
    );
    QuantizationInfo {
        quantization_style: QuantizationStyle::ScalarExpounded,
        guard_bits: 3,
        step_sizes,
    }
}

fn assert_coding_semantics(actual: &CodingStyleParameters, expected: &CodingStyleParameters) {
    assert_eq!(
        (
            actual.num_decomposition_levels,
            actual.num_resolution_levels,
            actual.code_block_width,
            actual.code_block_height,
            actual.transformation,
        ),
        (
            expected.num_decomposition_levels,
            expected.num_resolution_levels,
            expected.code_block_width,
            expected.code_block_height,
            expected.transformation,
        )
    );
    let actual_style = &actual.code_block_style;
    let expected_style = &expected.code_block_style;
    assert_eq!(
        [
            actual_style.selective_arithmetic_coding_bypass,
            actual_style.reset_context_probabilities,
            actual_style.termination_on_each_pass,
            actual_style.vertically_causal_context,
            actual_style.segmentation_symbols,
            actual_style.high_throughput_block_coding,
        ],
        [
            expected_style.selective_arithmetic_coding_bypass,
            expected_style.reset_context_probabilities,
            expected_style.termination_on_each_pass,
            expected_style.vertically_causal_context,
            expected_style.segmentation_symbols,
            expected_style.high_throughput_block_coding,
        ]
    );
    assert_eq!(actual.precinct_exponents, expected.precinct_exponents);
}

fn assert_quantization_semantics(actual: &QuantizationInfo, expected: &QuantizationInfo) {
    assert_eq!(actual.quantization_style, expected.quantization_style);
    assert_eq!(actual.guard_bits, expected.guard_bits);
    assert_eq!(actual.step_sizes.len(), expected.step_sizes.len());
    for (actual, expected) in actual.step_sizes.iter().zip(&expected.step_sizes) {
        assert_eq!(
            (actual.mantissa, actual.exponent),
            (expected.mantissa, expected.exponent)
        );
    }
}

#[test]
fn cloned_metadata_replaces_old_owners_with_actual_capacity_accounting() {
    let mut destination_coding = coding_parameters(7, &[(1, 2)]);
    let mut destination_quantization = quantization(6, &[(3, 4)]);
    let source_coding = coding_parameters(9, &[(5, 6), (7, 8), (9, 10)]);
    let source_quantization = quantization(8, &[(11, 12), (13, 14)]);
    let source_coding_ptr = source_coding.precinct_exponents.as_ptr();
    let source_quantization_ptr = source_quantization.step_sizes.as_ptr();
    let initial_bytes = destination_coding.precinct_exponents.capacity() * size_of::<(u8, u8)>()
        + destination_quantization.step_sizes.capacity() * size_of::<StepSize>();
    let mut budget = TileMetadataBudget::with_cap(initial_bytes, initial_bytes + 4_096)
        .expect("metadata budget");
    let (replacement_coding_bytes, replacement_quantization_bytes);

    {
        let mut transaction = budget.transaction();
        let replacement_coding = try_clone_coding_parameters(&source_coding, &mut transaction)
            .expect("fallible coding clone");
        let replacement_quantization =
            try_clone_quantization_info(&source_quantization, &mut transaction)
                .expect("fallible quantization clone");

        assert_coding_semantics(&replacement_coding, &source_coding);
        assert_quantization_semantics(&replacement_quantization, &source_quantization);
        assert_ne!(
            replacement_coding.precinct_exponents.as_ptr(),
            source_coding_ptr
        );
        assert_ne!(
            replacement_quantization.step_sizes.as_ptr(),
            source_quantization_ptr
        );
        replacement_coding_bytes =
            replacement_coding.precinct_exponents.capacity() * size_of::<(u8, u8)>();
        replacement_quantization_bytes =
            replacement_quantization.step_sizes.capacity() * size_of::<StepSize>();

        transaction
            .replace_coding_parameters(&mut destination_coding, replacement_coding)
            .expect("transfer coding owner");
        transaction
            .replace_quantization(&mut destination_quantization, replacement_quantization)
            .expect("transfer quantization owner");
    }

    assert_coding_semantics(&destination_coding, &source_coding);
    assert_quantization_semantics(&destination_quantization, &source_quantization);
    assert_eq!(
        budget.retained_bytes(),
        replacement_coding_bytes + replacement_quantization_bytes
    );
}

#[test]
fn untracked_replacements_fail_without_mutating_existing_owners_or_ledger() {
    let mut destination_coding = coding_parameters(7, &[(1, 2), (3, 4)]);
    let coding_ptr = destination_coding.precinct_exponents.as_ptr();
    let coding_capacity = destination_coding.precinct_exponents.capacity();
    let coding_bytes = coding_capacity * size_of::<(u8, u8)>();
    let mut coding_budget =
        TileMetadataBudget::with_cap(coding_bytes, coding_bytes + 1_024).expect("coding budget");
    {
        let mut transaction = coding_budget.transaction();
        assert_eq!(
            transaction.replace_coding_parameters(
                &mut destination_coding,
                coding_parameters(5, &[(9, 10)])
            ),
            Err(DecodeError::Validation(ValidationError::ImageTooLarge))
        );
    }
    assert_eq!(destination_coding.precinct_exponents.as_ptr(), coding_ptr);
    assert_eq!(
        destination_coding.precinct_exponents.capacity(),
        coding_capacity
    );
    assert_eq!(destination_coding.precinct_exponents, [(1, 2), (3, 4)]);
    assert_eq!(coding_budget.retained_bytes(), coding_bytes);

    let mut destination_quantization = quantization(6, &[(5, 6), (7, 8)]);
    let quantization_ptr = destination_quantization.step_sizes.as_ptr();
    let quantization_capacity = destination_quantization.step_sizes.capacity();
    let quantization_bytes = quantization_capacity * size_of::<StepSize>();
    let mut quantization_budget =
        TileMetadataBudget::with_cap(quantization_bytes, quantization_bytes + 1_024)
            .expect("quantization budget");
    {
        let mut transaction = quantization_budget.transaction();
        assert_eq!(
            transaction
                .replace_quantization(&mut destination_quantization, quantization(4, &[(11, 12)])),
            Err(DecodeError::Validation(ValidationError::ImageTooLarge))
        );
    }
    assert_eq!(
        destination_quantization.step_sizes.as_ptr(),
        quantization_ptr
    );
    assert_eq!(
        destination_quantization.step_sizes.capacity(),
        quantization_capacity
    );
    assert_quantization_semantics(
        &destination_quantization,
        &quantization(2, &[(5, 6), (7, 8)]),
    );
    assert_eq!(quantization_budget.retained_bytes(), quantization_bytes);
}

#[test]
fn clone_preflight_failure_preserves_sources_and_rolls_back_accounting() {
    let source_coding = coding_parameters(7, &[(1, 2), (3, 4)]);
    let source_quantization = quantization(6, &[(5, 6), (7, 8)]);
    let coding_ptr = source_coding.precinct_exponents.as_ptr();
    let quantization_ptr = source_quantization.step_sizes.as_ptr();
    let mut budget = TileMetadataBudget::with_cap(0, 0).expect("zero-cap budget");

    {
        let mut transaction = budget.transaction();
        assert!(matches!(
            try_clone_coding_parameters(&source_coding, &mut transaction),
            Err(DecodeError::Validation(ValidationError::ImageTooLarge))
        ));
    }
    assert_eq!(budget.retained_bytes(), 0);
    {
        let mut transaction = budget.transaction();
        assert!(matches!(
            try_clone_quantization_info(&source_quantization, &mut transaction),
            Err(DecodeError::Validation(ValidationError::ImageTooLarge))
        ));
    }

    assert_eq!(source_coding.precinct_exponents.as_ptr(), coding_ptr);
    assert_eq!(source_quantization.step_sizes.as_ptr(), quantization_ptr);
    assert_eq!(source_coding.precinct_exponents, [(1, 2), (3, 4)]);
    assert_quantization_semantics(&source_quantization, &quantization(2, &[(5, 6), (7, 8)]));
    assert_eq!(budget.retained_bytes(), 0);
}
