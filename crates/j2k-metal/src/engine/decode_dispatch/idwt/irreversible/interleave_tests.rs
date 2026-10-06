// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::engine::abi::J2kRepeatedIdwtSingleDecompositionParams;
use crate::engine::decode_dispatch::idwt::batched_irreversible::{
    dispatch_irreversible97_repeated_interleave_horizontal_fused,
    dispatch_irreversible97_repeated_interleave_horizontal_scale,
};
use crate::engine::decode_dispatch::idwt::{
    dispatch_reversible53_horizontal_repeated, dispatch_reversible53_horizontal_single,
    dispatch_reversible53_interleave_horizontal_fused_repeated,
    dispatch_reversible53_interleave_horizontal_fused_single,
    dispatch_reversible53_vertical_repeated, dispatch_reversible53_vertical_single,
    RepeatedIdwtDispatch,
};

fn repeated_params(
    p: J2kIdwtSingleDecompositionParams,
    batch: u32,
) -> J2kRepeatedIdwtSingleDecompositionParams {
    J2kRepeatedIdwtSingleDecompositionParams {
        x0: p.x0,
        y0: p.y0,
        output_x: p.output_x,
        output_y: p.output_y,
        width: p.width,
        height: p.height,
        ll_x: p.ll_x,
        ll_y: p.ll_y,
        ll_width: p.ll_width,
        ll_height: p.ll_height,
        hl_x: p.hl_x,
        hl_y: p.hl_y,
        hl_width: p.hl_width,
        hl_height: p.hl_height,
        lh_x: p.lh_x,
        lh_y: p.lh_y,
        lh_width: p.lh_width,
        lh_height: p.lh_height,
        hh_x: p.hh_x,
        hh_y: p.hh_y,
        hh_width: p.hh_width,
        hh_height: p.hh_height,
        ll_instance_stride: p.ll_width * p.ll_height + 3,
        hl_instance_stride: p.hl_width * p.hl_height + 5,
        lh_instance_stride: p.lh_width * p.lh_height + 7,
        hh_instance_stride: p.hh_width * p.hh_height + 9,
        batch_count: batch,
    }
}

fn reference_interleave_scale(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
    repeated: Option<J2kRepeatedIdwtSingleDecompositionParams>,
    high_pass: f32,
) {
    let SingleIdwtDispatch {
        kernels,
        sub_bands: b,
        params,
        decoded,
        decoded_offset,
    } = dispatch;
    let pipeline = if repeated.is_some() {
        &kernels.idwt_interleave_batched
    } else {
        &kernels.idwt_interleave
    };
    encoder.setComputePipelineState(pipeline);
    for (index, buffer, offset) in [
        (0, b.ll, b.ll_offset),
        (1, b.hl, b.hl_offset),
        (2, b.lh, b.lh_offset),
        (3, b.hh, b.hh_offset),
    ] {
        encoder.set_buffer(index, Some(buffer), offset as u64);
    }
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    if let Some(p) = repeated {
        encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &p);
        dispatch_3d_pipeline(encoder, pipeline, (p.width, p.height, p.batch_count));
    } else {
        encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &params);
        dispatch_2d_pipeline(encoder, pipeline, (params.width, params.height));
    }
    encoder.memory_barrier_with_resources(&[decoded]);
    encoder.setComputePipelineState(&kernels.idwt_irreversible97_horizontal_scale);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &params);
    encoder.set_bytes::<f32>(2, &high_pass);
    dispatch_3d_pipeline(
        encoder,
        &kernels.idwt_irreversible97_horizontal_scale,
        (
            params.width,
            params.height,
            repeated.map_or(1, |p| p.batch_count),
        ),
    );
}

fn reference_reversible53_vertical(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
    repeated: Option<J2kRepeatedIdwtSingleDecompositionParams>,
) {
    let pipeline = if repeated.is_some() {
        &dispatch.kernels.idwt_reversible53_vertical_batched
    } else {
        &dispatch.kernels.idwt_reversible53_vertical
    };
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(dispatch.decoded), dispatch.decoded_offset as u64);
    if let Some(params) = repeated {
        encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(1, &params);
        let threads = pipeline.threadExecutionWidth().max(1);
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(u64::from(params.width), u64::from(params.batch_count), 1),
            j2k_metal_support::mtl_size(threads as u64, 1, 1),
        );
    } else {
        encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(1, &dispatch.params);
        let threads = pipeline.threadExecutionWidth().max(1);
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(u64::from(dispatch.params.width), 1, 1),
            j2k_metal_support::mtl_size(threads as u64, 1, 1),
        );
    }
}

fn compare_interleave(
    runtime: &MetalRuntime,
    params: J2kIdwtSingleDecompositionParams,
    batch: u32,
    high_pass: f32,
) {
    let repeated = repeated_params(params, batch.max(1));
    let offsets = [4, 8, 12, 16];
    let strides = [
        repeated.ll_instance_stride,
        repeated.hl_instance_stride,
        repeated.lh_instance_stride,
        repeated.hh_instance_stride,
    ];
    let bands: Vec<Buffer> = strides
        .iter()
        .zip(offsets)
        .enumerate()
        .map(|(band, (stride, offset))| {
            let len = offset + *stride as usize * batch.max(1) as usize;
            let seed: Vec<f32> = (0..len)
                .map(|i| {
                    let value = i16::try_from((i + band * 71) % 257).unwrap() - 128;
                    f32::from(value) * 0.03125
                })
                .collect();
            copied_slice_buffer(&runtime.device, &seed).unwrap()
        })
        .collect();
    let sub_bands = IdwtSubBandBuffers {
        ll: &bands[0],
        ll_offset: offsets[0] * size_of::<f32>(),
        hl: &bands[1],
        hl_offset: offsets[1] * size_of::<f32>(),
        lh: &bands[2],
        lh_offset: offsets[2] * size_of::<f32>(),
        hh: &bands[3],
        hh_offset: offsets[3] * size_of::<f32>(),
    };
    // The production repeated API starts output at zero. Single dispatches also
    // exercise a nonzero byte offset, and both lanes retain a guarded tail.
    let prefix = if batch == 0 { 4 } else { 0 };
    let len = prefix + params.width as usize * params.height as usize * batch.max(1) as usize + 4;
    let seed = vec![7.0_f32; len];
    let outputs = [
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
        copied_slice_buffer(&runtime.device, &seed).unwrap(),
    ];
    let command = new_command_buffer(&runtime.queue).unwrap();
    let encoder = new_compute_command_encoder(&command).unwrap();
    for (index, decoded) in outputs.iter().enumerate() {
        let dispatch = SingleIdwtDispatch {
            kernels: runtime.decode().unwrap(),
            sub_bands,
            params,
            decoded,
            decoded_offset: prefix * size_of::<f32>(),
        };
        if index == 0 {
            reference_interleave_scale(
                &encoder,
                dispatch,
                (batch != 0).then_some(repeated),
                high_pass,
            );
        } else if index == 1 && batch == 0 {
            dispatch_irreversible97_interleave_horizontal_scale(&encoder, dispatch, high_pass);
        } else if index == 1 {
            dispatch_irreversible97_repeated_interleave_horizontal_scale(
                &encoder,
                RepeatedIdwtDispatch {
                    kernels: dispatch.kernels,
                    sub_bands,
                    params: repeated,
                    decoded,
                },
                high_pass,
            );
        } else if index == 2 && batch == 0 {
            dispatch_irreversible97_interleave_horizontal_scale(&encoder, dispatch, high_pass);
            dispatch_irreversible97_stages_after_horizontal_scale(
                &encoder,
                dispatch.kernels,
                decoded,
                dispatch.decoded_offset,
                params,
                high_pass,
                1,
            );
        } else if index == 2 {
            dispatch_irreversible97_repeated_interleave_horizontal_scale(
                &encoder,
                RepeatedIdwtDispatch {
                    kernels: dispatch.kernels,
                    sub_bands,
                    params: repeated,
                    decoded,
                },
                high_pass,
            );
            dispatch_irreversible97_stages_after_horizontal_scale(
                &encoder,
                dispatch.kernels,
                decoded,
                0,
                params,
                high_pass,
                repeated.batch_count,
            );
        } else if index == 3 && batch == 0 {
            dispatch_irreversible97_interleave_horizontal_fused(&encoder, dispatch, high_pass);
            dispatch_irreversible97_vertical_after_fused_horizontal(
                &encoder,
                dispatch.kernels,
                decoded,
                dispatch.decoded_offset,
                params,
                high_pass,
                1,
            );
        } else if index == 3 {
            dispatch_irreversible97_repeated_interleave_horizontal_fused(
                &encoder,
                RepeatedIdwtDispatch {
                    kernels: dispatch.kernels,
                    sub_bands,
                    params: repeated,
                    decoded,
                },
                high_pass,
            );
            dispatch_irreversible97_vertical_after_fused_horizontal(
                &encoder,
                dispatch.kernels,
                decoded,
                0,
                params,
                high_pass,
                repeated.batch_count,
            );
        } else if index == 4 || index == 6 {
            let pipeline = if batch == 0 {
                &dispatch.kernels.idwt_interleave
            } else {
                &dispatch.kernels.idwt_interleave_batched
            };
            encoder.setComputePipelineState(pipeline);
            for (slot, buffer, offset) in [
                (0, sub_bands.ll, sub_bands.ll_offset),
                (1, sub_bands.hl, sub_bands.hl_offset),
                (2, sub_bands.lh, sub_bands.lh_offset),
                (3, sub_bands.hh, sub_bands.hh_offset),
            ] {
                encoder.set_buffer(slot, Some(buffer), offset as u64);
            }
            encoder.set_buffer(4, Some(decoded), dispatch.decoded_offset as u64);
            if batch == 0 {
                encoder.set_bytes::<J2kIdwtSingleDecompositionParams>(5, &params);
                dispatch_2d_pipeline(&encoder, pipeline, (params.width, params.height));
            } else {
                encoder.set_bytes::<J2kRepeatedIdwtSingleDecompositionParams>(5, &repeated);
                dispatch_3d_pipeline(
                    &encoder,
                    pipeline,
                    (repeated.width, repeated.height, repeated.batch_count),
                );
            }
            encoder.memory_barrier_with_resources(&[decoded]);
            if batch == 0 {
                dispatch_reversible53_horizontal_single(
                    &encoder,
                    dispatch.kernels,
                    decoded,
                    dispatch.decoded_offset,
                    params,
                );
            } else {
                dispatch_reversible53_horizontal_repeated(
                    &encoder,
                    dispatch.kernels,
                    decoded,
                    repeated,
                );
            }
            if index == 6 {
                encoder.memory_barrier_with_resources(&[decoded]);
                reference_reversible53_vertical(
                    &encoder,
                    dispatch,
                    (batch != 0).then_some(repeated),
                );
            }
        } else {
            if batch == 0 {
                dispatch_reversible53_interleave_horizontal_fused_single(&encoder, dispatch);
            } else {
                dispatch_reversible53_interleave_horizontal_fused_repeated(
                    &encoder,
                    RepeatedIdwtDispatch {
                        kernels: dispatch.kernels,
                        sub_bands,
                        params: repeated,
                        decoded,
                    },
                );
            }
            if index == 7 {
                encoder.memory_barrier_with_resources(&[decoded]);
                if batch == 0 {
                    dispatch_reversible53_vertical_single(
                        &encoder,
                        dispatch.kernels,
                        decoded,
                        dispatch.decoded_offset,
                        params,
                    );
                } else {
                    dispatch_reversible53_vertical_repeated(
                        &encoder,
                        dispatch.kernels,
                        decoded,
                        repeated,
                    );
                }
            }
        }
    }
    encoder.endEncoding();
    commit_and_wait_metal(&command).unwrap();
    let expected =
        checked_buffer_slice::<f32>(&outputs[0], len, "reference interleave/scale").unwrap();
    let actual =
        checked_buffer_slice::<f32>(&outputs[1], len, "production interleave/scale").unwrap();
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "interleave coefficient {index}, batch {batch}"
        );
    }
    for value in actual[..prefix].iter().chain(&actual[len - 4..]) {
        assert_eq!(value.to_bits(), 7.0_f32.to_bits(), "output guard changed");
    }
    let expected = checked_buffer_slice::<f32>(&outputs[2], len, "existing full IDWT").unwrap();
    let actual = checked_buffer_slice::<f32>(&outputs[3], len, "fused interleave IDWT").unwrap();
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "fused interleave coefficient {index}, batch {batch}"
        );
    }
    for value in actual[..prefix].iter().chain(&actual[len - 4..]) {
        assert_eq!(value.to_bits(), 7.0_f32.to_bits(), "output guard changed");
    }
    let expected =
        checked_buffer_slice::<f32>(&outputs[4], len, "existing 5/3 horizontal").unwrap();
    let actual = checked_buffer_slice::<f32>(&outputs[5], len, "fused 5/3 horizontal").unwrap();
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "fused 5/3 interleave coefficient {index}, batch {batch}"
        );
    }
    for value in actual[..prefix].iter().chain(&actual[len - 4..]) {
        assert_eq!(value.to_bits(), 7.0_f32.to_bits(), "output guard changed");
    }
    let expected = checked_buffer_slice::<f32>(&outputs[6], len, "existing full 5/3 IDWT").unwrap();
    let actual = checked_buffer_slice::<f32>(&outputs[7], len, "fused full 5/3 IDWT").unwrap();
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "fused 5/3 vertical coefficient {index}, batch {batch}"
        );
    }
    for value in actual[..prefix].iter().chain(&actual[len - 4..]) {
        assert_eq!(value.to_bits(), 7.0_f32.to_bits(), "output guard changed");
    }
}

#[test]
fn irreversible97_interleave_scale_matches_separate_dispatch_bits() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        for width in [1, 2, 3, 9, 129, 257] {
            for origin in [0, 1] {
                for offset in [0, 1, 2, 3] {
                    let mut params =
                        geometry_params((width, 7, origin, 1 - origin, offset, 3 - offset));
                    let broad_band_width = width.div_ceil(2) + 4;
                    // Broad windows, then cropped/missing bands, exercise each
                    // selection branch and unsigned below-window subtraction.
                    for cropped in [false, true] {
                        params.ll_width = broad_band_width;
                        params.ll_height = 6;
                        params.hl_width = if cropped { 0 } else { broad_band_width };
                        params.hl_height = 6;
                        params.lh_width = broad_band_width;
                        params.lh_height = if cropped { 0 } else { 5 };
                        params.hh_width = if cropped { 2 } else { broad_band_width };
                        params.hh_height = 5;
                        params.ll_x = u32::from(cropped);
                        params.ll_y = u32::from(cropped);
                        params.hh_x = if cropped { 2 } else { 0 };
                        for batch in [0, 1, 3, 16] {
                            for high_pass in [
                                dwt::DWT97_INV_KAPPA_F32,
                                dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5,
                            ] {
                                compare_interleave(runtime, params, batch, high_pass);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    })
    .expect("interleave/scale bitwise comparison");
}

#[test]
fn reversible53_tiled_vertical_matches_serial_across_chunk_boundaries() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        for (width, height) in [(17, 129), (65, 257), (17, 513)] {
            for (origin_y, output_y, cropped) in [(0, 0, false), (1, 3, true)] {
                let mut params = geometry_params((width, height, 1, origin_y, 2, output_y));
                let broad_band_width = width.div_ceil(2) + 4;
                let broad_band_height = height.div_ceil(2) + 4;
                params.ll_width = broad_band_width;
                params.ll_height = broad_band_height;
                params.hl_width = if cropped { 0 } else { broad_band_width };
                params.hl_height = broad_band_height;
                params.lh_width = broad_band_width;
                params.lh_height = if cropped { 0 } else { broad_band_height };
                params.hh_width = if cropped { 2 } else { broad_band_width };
                params.hh_height = broad_band_height;
                params.ll_x = u32::from(cropped);
                params.ll_y = u32::from(cropped);
                params.hh_x = if cropped { 2 } else { 0 };
                for batch in [0, 1, 3, 16] {
                    compare_interleave(runtime, params, batch, dwt::DWT97_INV_KAPPA_F32);
                }
            }
        }
        Ok(())
    })
    .expect("tiled reversible 5/3 vertical bitwise comparison");
}

#[test]
fn irreversible97_production_decomposition_dispatch_count() {
    use crate::engine::decode_dispatch::idwt::dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets;
    use crate::engine::test_counters::{
        idwt97_logical_dispatches_for_test, reset_idwt97_logical_dispatches_for_test,
    };
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    with_runtime(|runtime| {
        let band = copied_slice_buffer(&runtime.device, &[0.25_f32; 256]).unwrap();
        let sub_bands = IdwtSubBandBuffers {
            ll: &band,
            ll_offset: 0,
            hl: &band,
            hl_offset: 0,
            lh: &band,
            lh_offset: 0,
            hh: &band,
            hh_offset: 0,
        };
        for width in [9, 129] {
            let params = geometry_params((width, 7, 1, 0, 2, 3));
            for batch in [0, 1, 3, 16] {
                let count = params.width as usize * params.height as usize * batch.max(1) as usize;
                let decoded = copied_slice_buffer(&runtime.device, &vec![7.0_f32; count]).unwrap();
                let command = new_command_buffer(&runtime.queue).unwrap();
                let encoder = new_compute_command_encoder(&command).unwrap();
                reset_idwt97_logical_dispatches_for_test();
                if batch == 0 {
                    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_high_pass(
                        &encoder,
                        SingleIdwtDispatch {
                            kernels: runtime.decode().unwrap(),
                            sub_bands,
                            params,
                            decoded: &decoded,
                            decoded_offset: 0,
                        },
                        dwt::DWT97_INV_KAPPA_F32,
                    );
                } else {
                    dispatch_irreversible97_repeated_buffers_in_encoder_with_offsets(
                        &encoder,
                        RepeatedIdwtDispatch {
                            kernels: runtime.decode().unwrap(),
                            sub_bands,
                            params: repeated_params(params, batch),
                            decoded: &decoded,
                        },
                    );
                }
                encoder.endEncoding();
                commit_and_wait_metal(&command).unwrap();
                let (positions, dispatches) = idwt97_logical_dispatches_for_test();
                let expected_dispatches = if width > IDWT97_ROW_TILE { 2 } else { 3 };
                assert_eq!(
                    dispatches, expected_dispatches,
                    "actual production launches, width {width}, batch {batch}"
                );
                assert_eq!(
                    positions,
                    expected_dispatches * count,
                    "each production pass covers every sample once"
                );
            }
        }
        Ok(())
    })
    .expect("production dispatch counts");
}
