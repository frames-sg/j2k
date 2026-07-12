// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dwt97_block_value_count, projection_dispatch_sizes, validate_codeblock_projection_allocations,
    validate_float_projection_allocations, validate_htj2k97_codeblock_options, Dwt97TwoDimensional,
    Htj2k97CodeBlockOptions, MetalTranscodeError, METAL_DCT97_UNSUPPORTED_GRID,
    METAL_READBACK_CHUNK_BYTES,
};
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
use j2k_transcode::IrreversibleQuantizationSubbandScales;
use std::mem::size_of;

#[test]
fn projection_dispatch_sizes_use_16_by_8_threadgroups() {
    let (threads, threadgroup) = projection_dispatch_sizes(5, 6, 7);

    assert_eq!((threads.width, threads.height, threads.depth), (5, 6, 7));
    assert_eq!(
        (threadgroup.width, threadgroup.height, threadgroup.depth),
        (16, 8, 1)
    );
}

#[test]
fn dwt97_block_value_count_rejects_overflow() {
    assert_eq!(
        dwt97_block_value_count(2).expect("bounded block count"),
        128
    );
    assert!(matches!(
        dwt97_block_value_count(usize::MAX),
        Err(MetalTranscodeError::UnsupportedJob(reason))
            if reason == METAL_DCT97_UNSUPPORTED_GRID
    ));
}

#[test]
fn huge_projection_grid_and_batch_reject_before_allocation() {
    let over_cap_samples = DEFAULT_MAX_HOST_ALLOCATION_BYTES / size_of::<f64>() + 1;
    assert!(matches!(
        validate_float_projection_allocations(1, over_cap_samples, 1, 1, 0, 0, 1),
        Err(MetalTranscodeError::HostAllocationTooLarge {
            what: "projected wavelet output bands",
            ..
        })
    ));

    let batch_count = DEFAULT_MAX_HOST_ALLOCATION_BYTES / (224 * 224 * size_of::<f64>()) + 1;
    assert!(matches!(
        validate_float_projection_allocations(1, 224, 224, batch_count, 0, 0, 1),
        Err(MetalTranscodeError::HostAllocationTooLarge {
            what: "projected wavelet output bands",
            ..
        })
    ));
}

#[test]
fn projection_conversion_counts_both_live_outer_metadata_arrays() {
    let batch_count = METAL_READBACK_CHUNK_BYTES / size_of::<Dwt97TwoDimensional<f64>>() + 1;
    let output_bytes = batch_count * size_of::<f64>();
    let source_metadata_bytes = batch_count * size_of::<super::ProjectedBands>();
    let destination_metadata_bytes = batch_count * size_of::<Dwt97TwoDimensional<f64>>();
    let readback_peak = output_bytes + source_metadata_bytes + METAL_READBACK_CHUNK_BYTES;
    let conversion_peak = output_bytes + source_metadata_bytes + destination_metadata_bytes;
    assert!(conversion_peak > readback_peak);

    let weight_host_bytes = DEFAULT_MAX_HOST_ALLOCATION_BYTES - conversion_peak + 1;
    assert!(matches!(
        validate_float_projection_allocations(
            batch_count,
            1,
            1,
            batch_count,
            weight_host_bytes,
            0,
            1,
        ),
        Err(MetalTranscodeError::HostAllocationTooLarge {
            what: "projected wavelet host workspace",
            ..
        })
    ));
}

#[test]
fn huge_codeblock_batch_rejects_before_device_work() {
    let options = Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: 1,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        irreversible_quantization_scale: 1.0,
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales::default(),
    };
    let batch_count = DEFAULT_MAX_HOST_ALLOCATION_BYTES / (256 * 256 * size_of::<i32>()) + 1;
    assert!(matches!(
        validate_codeblock_projection_allocations(1, 256, 256, batch_count, options),
        Err(MetalTranscodeError::HostAllocationTooLarge {
            what: "prequantized HTJ2K coefficients",
            ..
        })
    ));
}

#[test]
fn htj2k97_option_declines_preserve_each_typed_category() {
    let valid = Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: 2,
        code_block_width_exp: 4,
        code_block_height_exp: 4,
        irreversible_quantization_scale: 1.0,
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales::default(),
    };
    let mut quantization = valid;
    quantization
        .irreversible_quantization_subband_scales
        .low_low = 0.0;
    let cases = [
        (
            Htj2k97CodeBlockOptions {
                bit_depth: 31,
                ..valid
            },
            "9/7 code-block options are outside supported numeric range",
        ),
        (
            quantization,
            "9/7 code-block quantization options are outside supported range",
        ),
        (
            Htj2k97CodeBlockOptions {
                code_block_width_exp: u8::MAX,
                ..valid
            },
            "9/7 code-block dimension exponent is unsupported",
        ),
        (
            Htj2k97CodeBlockOptions {
                code_block_width_exp: 8,
                code_block_height_exp: 8,
                ..valid
            },
            "9/7 code-block dimensions exceed HTJ2K limits",
        ),
    ];

    for (options, reason) in cases {
        assert!(matches!(
            validate_htj2k97_codeblock_options(options),
            Err(MetalTranscodeError::UnsupportedJob(actual)) if actual == reason
        ));
    }
}
