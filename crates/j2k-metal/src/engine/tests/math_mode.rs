// SPDX-License-Identifier: MIT OR Apache-2.0

//! The J2K Metal libraries compile under safe math without implicit FMA
//! contraction (see `PRELUDE`). A probe kernel appended to each library
//! source fails here if any shader file stops that policy from reaching the
//! end of its library.

use crate::metal_types::prelude::*;
use j2k_metal_support::{dispatch_single_thread, MetalPipelineLoader};

use super::super::{
    checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer, new_command_buffer,
    new_compute_command_encoder, shader_source, with_runtime,
};
use super::runtime::should_run_metal_runtime;

const PROBE: &str = r"
kernel void j2k_audit_float_math_mode(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]]
) {
    output[0] = input[0] * input[1] + input[2];
    output[1] = (input[3] + input[4]) - input[4];
    output[2] = isnan(input[5]) ? 1.0f : 0.0f;
    output[3] = input[6] * 0.0f;
}
";

/// Inputs on which contraction, reassociation and finite-math assumptions
/// each change the result.
fn probe_inputs() -> [f32; 7] {
    let one_plus_ulp_half = f32::from_bits(0x3F80_0800); // 1 + 2^-12
    let one_plus_ulp = f32::from_bits(0x3F80_1000); // 1 + 2^-11
    [
        one_plus_ulp_half,
        one_plus_ulp_half,
        -one_plus_ulp,
        1.0,
        16_777_216.0, // 2^24
        f32::NAN,
        f32::INFINITY,
    ]
}

fn run_probe(library: &str, source: &str) -> [f32; 4] {
    let source = format!("{source}\n{PROBE}");
    with_runtime(|runtime| {
        let loader = MetalPipelineLoader::new(&runtime.device, &source)
            .unwrap_or_else(|error| panic!("{library} library with probe: {error}"));
        let pipeline = loader
            .pipeline("j2k_audit_float_math_mode")
            .unwrap_or_else(|error| panic!("{library} probe pipeline: {error}"));
        let input = copied_slice_buffer(&runtime.device, &probe_inputs())?;
        let output = copied_slice_buffer(&runtime.device, &[0.0_f32; 4])?;
        let command = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command)?;
        encoder.setComputePipelineState(&pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&output), 0);
        dispatch_single_thread(&encoder);
        encoder.endEncoding();
        commit_and_wait_metal(&command)?;
        let values = checked_buffer_slice::<f32>(&output, 4, "math-mode probe")?;
        Ok([values[0], values[1], values[2], values[3]])
    })
    .unwrap_or_else(|error| panic!("{library} probe dispatch: {error}"))
}

#[test]
fn every_library_rounds_each_float_operation_like_the_cpu() {
    if !should_run_metal_runtime() {
        return;
    }
    let [multiplicand, multiplier, addend, small, large, _, _] = probe_inputs();
    // The CPU rounds the product before the add, and the fused result differs.
    let unfused = multiplicand * multiplier + addend;
    assert_eq!(unfused.to_bits(), 0.0_f32.to_bits());
    assert_ne!(
        multiplicand.mul_add(multiplier, addend).to_bits(),
        unfused.to_bits()
    );
    assert_eq!(((small + large) - large).to_bits(), 0.0_f32.to_bits());

    for (library, source) in [
        ("decode", shader_source::decode_shader_source()),
        ("encode", shader_source::encode_shader_source()),
        ("profile", shader_source::profile_shader_source()),
        ("buffers", shader_source::buffers_shader_source()),
    ] {
        assert!(
            source.starts_with(shader_source::PRELUDE),
            "{library} library must open with the math-mode prelude"
        );
        let [product_sum, reassociated, nan_seen, infinity_times_zero] =
            run_probe(library, &source);
        assert_eq!(
            product_sum.to_bits(),
            0.0_f32.to_bits(),
            "{library}: a * b + c must not contract into an fma"
        );
        assert_eq!(
            reassociated.to_bits(),
            0.0_f32.to_bits(),
            "{library}: (x + y) - y must not be reassociated"
        );
        assert_eq!(
            nan_seen.to_bits(),
            1.0_f32.to_bits(),
            "{library}: isnan must see a NaN input"
        );
        assert!(
            infinity_times_zero.is_nan(),
            "{library}: infinity times zero must stay NaN, got {infinity_times_zero}"
        );
    }
}
