// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::engine::abi::J2kIdwt97StepParams;
use crate::engine::runtime::MetalRuntime;
use crate::metal_types::ComputePipelineState;
use j2k_metal_support::MetalPipelineLoader;

// Original full-grid lifting kernels, kept only as an independent test oracle.
const REFERENCE_STEPS: &str = r"struct J2kIdwt97ReferenceStep {
    float coefficient;
    uint parity;
    uint reserved0;
    uint reserved1;
};

kernel void audit_idwt97_horizontal_reference(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant J2kIdwt97ReferenceStep &step [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || params.width <= 1u
        || (gid.x & 1u) != step.parity) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const uint left = periodic_symmetric_extension_left_u32(gid.x, 1u);
    const uint right = periodic_symmetric_extension_right_u32(gid.x, 1u, params.width);
    const uint idx = gid.y * params.width + gid.x;
    out[idx] = fma(out[gid.y * params.width + left] + out[gid.y * params.width + right],
                   step.coefficient,
                   out[idx]);
}

kernel void audit_idwt97_vertical_scale_reference(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant float &high_pass [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const float KAPPA = CODEC_MATH_DWT97_KAPPA;
    float sample = out[gid.y * params.width + gid.x];

    if (params.height == 1u) {
        if (((params.y0 + params.output_y) & 1u) != 0u) {
            sample *= 0.5f;
        }
    } else {
        const uint first_even_y = (params.y0 + params.output_y) & 1u;
        sample *= (gid.y & 1u) == first_even_y ? KAPPA : high_pass;
    }

    out[gid.y * params.width + gid.x] = sample;
}

kernel void audit_idwt97_vertical_reference(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant J2kIdwt97ReferenceStep &step [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || params.height <= 1u
        || (gid.y & 1u) != step.parity) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const uint above = periodic_symmetric_extension_left_u32(gid.y, 1u);
    const uint below = periodic_symmetric_extension_right_u32(gid.y, 1u, params.height);
    const uint idx = gid.y * params.width + gid.x;
    out[idx] = fma(out[above * params.width + gid.x] + out[below * params.width + gid.x],
                   step.coefficient,
                   out[idx]);
}
";

struct References {
    horizontal: ComputePipelineState,
    vertical_scale: ComputePipelineState,
    vertical: ComputePipelineState,
}

fn geometry_params(geometry: (u32, u32, u32, u32, u32, u32)) -> J2kIdwtSingleDecompositionParams {
    let (width, height, x0, y0, output_x, output_y) = geometry;
    J2kIdwtSingleDecompositionParams {
        x0,
        y0,
        output_x,
        output_y,
        width,
        height,
        ll_x: 0,
        ll_y: 0,
        ll_width: 0,
        ll_height: 0,
        hl_x: 0,
        hl_y: 0,
        hl_width: 0,
        hl_height: 0,
        lh_x: 0,
        lh_y: 0,
        lh_width: 0,
        lh_height: 0,
        hh_x: 0,
        hh_y: 0,
        hh_width: 0,
        hh_height: 0,
    }
}

/// Runs the original one-pass-per-step lifting sequence on `buffer`.
fn encode_reference_stages(
    encoder: &ComputeCommandEncoderRef,
    references: &References,
    buffer: &Buffer,
    offset: u64,
    params: &J2kIdwtSingleDecompositionParams,
    batch: u32,
    high_pass: f32,
) {
    let grid = (params.width, params.height, batch);
    let lifts = |origin: u32, pipeline: &ComputePipelineState| {
        let even = origin & 1;
        for (coefficient, parity) in [
            (dwt::IDWT97_NEG_DELTA_F32, even),
            (dwt::IDWT97_NEG_GAMMA_F32, 1 - even),
            (dwt::IDWT97_NEG_BETA_F32, even),
            (dwt::IDWT97_NEG_ALPHA_F32, 1 - even),
        ] {
            let step = J2kIdwt97StepParams {
                coefficient,
                parity,
                _reserved0: 0,
                _reserved1: 0,
            };
            encoder.setComputePipelineState(pipeline);
            encoder.set_buffer(0, Some(buffer), offset);
            encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, params);
            encoder.set_bytes::<J2kIdwt97StepParams>(2, &step);
            dispatch_3d_pipeline(encoder, pipeline, grid);
            encoder.memory_barrier_with_resources(&[buffer]);
        }
    };
    lifts(params.x0 + params.output_x, &references.horizontal);
    encoder.setComputePipelineState(&references.vertical_scale);
    encoder.set_buffer(0, Some(buffer), offset);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, params);
    encoder.set_bytes::<f32>(2, &high_pass);
    dispatch_3d_pipeline(encoder, &references.vertical_scale, grid);
    encoder.memory_barrier_with_resources(&[buffer]);
    lifts(params.y0 + params.output_y, &references.vertical);
}

fn compare_geometry(
    runtime: &MetalRuntime,
    references: &References,
    geometry: (u32, u32, u32, u32, u32, u32),
    batch: u32,
    high_pass: f32,
) {
    const PREFIX: usize = 4;
    let (width, height, ..) = geometry;
    let params = geometry_params(geometry);
    let count = width as usize * height as usize * batch as usize;
    let mut seed = vec![7.0; count + 2 * PREFIX];
    for (index, value) in seed[PREFIX..count + PREFIX].iter_mut().enumerate() {
        *value = f32::from(i16::try_from(index % 257).unwrap() - 128) * 0.03125;
    }
    let buffers = [
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
    ];
    let offset = (PREFIX * size_of::<f32>()) as u64;
    let command = new_command_buffer(&runtime.queue).expect("comparison command");
    let encoder = new_compute_command_encoder(&command).expect("comparison encoder");
    encode_reference_stages(
        &encoder,
        references,
        &buffers[0],
        offset,
        &params,
        batch,
        high_pass,
    );
    dispatch_irreversible97_stages_after_horizontal_scale(
        &encoder,
        runtime.decode().expect("production kernels"),
        &buffers[1],
        PREFIX * size_of::<f32>(),
        params,
        high_pass,
        batch,
    );
    encoder.endEncoding();
    commit_and_wait_metal(&command).expect("completed comparison");
    let len = count + 2 * PREFIX;
    let expected = checked_buffer_slice::<f32>(&buffers[0], len, "reference stages").unwrap();
    let actual = checked_buffer_slice::<f32>(&buffers[1], len, "fused stages").unwrap();
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(actual.is_finite());
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "geometry {geometry:?} batch {batch}: coefficient {index}"
        );
    }
    for value in actual[..PREFIX].iter().chain(&actual[len - PREFIX..]) {
        assert_eq!(value.to_bits(), 7.0_f32.to_bits(), "offset guard changed");
    }
}

#[test]
fn irreversible97_fused_lifting_matches_full_grid_reference_bits() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        let source = format!(
            "{}\n{REFERENCE_STEPS}",
            crate::engine::shader_source::decode_shader_source()
        );
        let loader = MetalPipelineLoader::new(&runtime.device, &source).expect("reference library");
        let references = References {
            horizontal: loader
                .pipeline("audit_idwt97_horizontal_reference")
                .unwrap(),
            vertical_scale: loader
                .pipeline("audit_idwt97_vertical_scale_reference")
                .unwrap(),
            vertical: loader.pipeline("audit_idwt97_vertical_reference").unwrap(),
        };
        // Small edge cases, then shapes spanning several row tiles (128) and
        // column tiles (64), with partial final tiles and odd origins.
        for geometry in [
            (1, 1, 0, 0, 0, 0),
            (1, 1, 1, 1, 0, 0),
            (1, 7, 1, 0, 2, 3),
            (9, 1, 0, 1, 3, 2),
            (2, 3, 0, 1, 0, 0),
            (3, 2, 1, 0, 0, 0),
            (5, 7, 0, 0, 1, 1),
            (18, 31, 1, 1, 2, 3),
            (129, 7, 0, 1, 3, 2),
            (320, 240, 0, 0, 0, 0),
            (257, 131, 1, 0, 0, 1),
            (131, 257, 0, 1, 1, 0),
            (640, 65, 1, 1, 0, 0),
            (2, 200, 0, 0, 0, 1),
            (200, 2, 1, 0, 0, 0),
        ] {
            for batch in [1, 3, 16] {
                for high_pass in [
                    dwt::DWT97_INV_KAPPA_F32,
                    dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5,
                ] {
                    compare_geometry(runtime, &references, geometry, batch, high_pass);
                }
            }
        }
        Ok(())
    })
    .expect("fused lifting bitwise comparison");
}

#[test]
fn irreversible97_fused_stages_encode_one_pass_per_axis() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        for (width, height, batch) in [(5, 7, 3), (1, 1, 1), (1, 7, 3), (9, 1, 16), (128, 64, 16)] {
            let params = geometry_params((width, height, 1, 0, 2, 3));
            let samples = vec![0.25_f32; width as usize * height as usize * batch as usize];
            let buffer = copied_slice_buffer(&runtime.device, &samples).unwrap();
            let command = new_command_buffer(&runtime.queue).unwrap();
            let encoder = new_compute_command_encoder(&command).unwrap();
            crate::engine::test_counters::reset_idwt97_logical_dispatches_for_test();
            dispatch_irreversible97_stages(
                &encoder,
                runtime.decode().unwrap(),
                &buffer,
                0,
                params,
                dwt::DWT97_INV_KAPPA_F32,
                batch,
            );
            encoder.endEncoding();
            commit_and_wait_metal(&command)?;
            let (positions, dispatches) =
                crate::engine::test_counters::idwt97_logical_dispatches_for_test();
            let horizontal_lift = usize::from(width > 1);
            assert_eq!(
                positions,
                (2 + horizontal_lift) * samples.len(),
                "horizontal scale, horizontal lifts, and vertical scale plus lifts each cover every sample once"
            );
            assert_eq!(
                dispatches,
                2 + horizontal_lift,
                "single-column planes skip the horizontal lifting pass"
            );
        }
        Ok(())
    })
    .expect("production fused grids");
}

#[path = "interleave_tests.rs"]
mod interleave_tests;
