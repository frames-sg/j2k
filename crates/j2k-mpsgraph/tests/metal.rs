// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use core::{ffi::c_void, ptr::NonNull};
use std::sync::Arc;

use j2k::{
    wrap_j2k_codestream, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, EncodedImage, J2kChannelAssociation, J2kChannelDefinition, J2kChannelType,
    J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions,
};
use j2k_core::{Colorspace, Downscale, PixelLayout, Rect, SampleType};
use j2k_mpsgraph::{
    Error, MpsGraphBatchDecoder, MpsGraphElementType, MpsGraphProgram, MpsGraphTensorSpec,
};
use j2k_test_support::{htj2k_rgb8_97_fixture, htj2k_rgb8_fixture, metal_runtime_gate};
use objc2_foundation::{NSArray, NSNumber};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::MPSGraph;

fn read_u8_result(output: &j2k_mpsgraph::MpsGraphRunOutput, len: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; len];
    // SAFETY: the graph run has completed and `bytes` exactly covers the
    // identity graph's validated U8 output shape.
    unsafe {
        output.results()[0].mpsndarray().readBytes_strideBytes(
            NonNull::new(bytes.as_mut_ptr().cast::<c_void>()).expect("nonempty output"),
            core::ptr::null_mut(),
        );
    }
    bytes
}

fn read_result_bytes(output: &j2k_mpsgraph::MpsGraphRunOutput, len: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; len];
    // SAFETY: the graph run has completed and `bytes` exactly covers the
    // identity graph's validated native integer output shape.
    unsafe {
        output.results()[0].mpsndarray().readBytes_strideBytes(
            NonNull::new(bytes.as_mut_ptr().cast::<c_void>()).expect("nonempty output"),
            core::ptr::null_mut(),
        );
    }
    bytes
}

fn native_fixture(color: PixelLayout, sample_type: SampleType) -> Arc<[u8]> {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;
    let channels = color.channels();
    let sample_count = WIDTH as usize * HEIGHT as usize * channels;
    let (pixels, precision, signed): (Vec<u8>, u8, bool) = match sample_type {
        SampleType::U8 => (
            (0..sample_count)
                .map(|index| u8::try_from((index * 37 + 11) & 0xff).expect("masked fixture sample"))
                .collect(),
            8,
            false,
        ),
        SampleType::U16 => (
            (0..sample_count)
                .flat_map(|index| {
                    u16::try_from((index * 977 + 31) & 0x0fff)
                        .expect("masked fixture sample")
                        .to_le_bytes()
                })
                .collect(),
            12,
            false,
        ),
        SampleType::I16 => (
            (0..sample_count)
                .flat_map(|index| {
                    let index = i32::try_from(index).expect("fixture index fits i32");
                    i16::try_from((index * 113 + 19) % 20_001 - 10_000)
                        .expect("bounded fixture sample")
                        .to_le_bytes()
                })
                .collect(),
            16,
            true,
        ),
        _ => unreachable!("native matrix only covers integer batch types"),
    };
    let codestream = j2k_native::encode_htj2k(
        &pixels,
        WIDTH,
        HEIGHT,
        u16::try_from(channels).expect("native color channel count fits u16"),
        precision,
        signed,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            use_mct: false,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode native matrix fixture");
    if color != PixelLayout::Rgba {
        return Arc::from(codestream);
    }
    let channel_definitions = [
        J2kChannelDefinition {
            channel_index: 0,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 1 },
        },
        J2kChannelDefinition {
            channel_index: 1,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 2 },
        },
        J2kChannelDefinition {
            channel_index: 2,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 3 },
        },
        J2kChannelDefinition {
            channel_index: 3,
            channel_type: J2kChannelType::Opacity,
            association: J2kChannelAssociation::WholeImage,
        },
    ];
    Arc::from(
        wrap_j2k_codestream(
            &codestream,
            J2kFileWrapOptions::jph()
                .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
                .with_metadata(J2kFileBoxMetadata {
                    palette: None,
                    component_mappings: &[],
                    channel_definitions: &channel_definitions,
                }),
        )
        .expect("wrap RGBA native matrix fixture"),
    )
}

fn cpu_sample_bytes(samples: &CpuBatchSamples) -> Vec<u8> {
    match samples {
        CpuBatchSamples::U8(samples) => samples.clone(),
        CpuBatchSamples::U16(samples) => samples
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect(),
        CpuBatchSamples::I16(samples) => samples
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect(),
        _ => unreachable!("native matrix only covers integer batch types"),
    }
}

#[test]
fn completed_rgb8_batch_becomes_mpsgraph_tensor_without_host_pixels() {
    if !metal_runtime_gate("j2k-mpsgraph completed resident batch") {
        return;
    }
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MpsGraphBatchDecoder::system_default(options).expect("Apple Silicon MPSGraph decoder");
    let encoded = Arc::<[u8]>::from(htj2k_rgb8_fixture(8, 8));
    let output = decoder
        .decode(vec![
            EncodedImage::full(encoded.clone()),
            EncodedImage::full(encoded),
        ])
        .expect("direct MPSGraph decode");

    assert!(output.errors().is_empty());
    assert!(output.group_errors().is_empty());
    assert_eq!(output.groups().len(), 1);
    let group = &output.groups()[0];
    assert_eq!(group.spec().shape(), [2, 8, 8, 3]);
    assert_eq!(group.spec().element_type(), MpsGraphElementType::U8);
    assert_eq!(group.source_indices(), [0, 1]);
    assert_eq!(group.resident_batch().byte_offset(), 0);
    assert_eq!(group.resident_batch().byte_len(), 2 * 8 * 8 * 3);
}

#[test]
fn identity_program_rejects_a_different_static_input_contract() {
    let expected = MpsGraphTensorSpec::new([2, 8, 8, 3], MpsGraphElementType::U8)
        .expect("valid expected spec");
    let actual =
        MpsGraphTensorSpec::new([1, 8, 8, 3], MpsGraphElementType::U8).expect("valid actual spec");
    let program = MpsGraphProgram::identity(expected).expect("identity graph");

    assert!(matches!(
        program.validate_input_spec(actual),
        Err(Error::InvalidTensorContract { .. })
    ));
}

#[test]
fn program_rejects_wrong_placeholder_rank_and_dtype() {
    let expected = MpsGraphTensorSpec::new([1, 8, 8, 3], MpsGraphElementType::U8)
        .expect("valid expected spec");
    for (shape, dtype) in [
        (vec![1, 8, 8], MPSDataType::UInt8),
        (vec![1, 8, 8, 3], MPSDataType::UInt16),
    ] {
        // SAFETY: standard owning Objective-C graph construction.
        let graph = unsafe { MPSGraph::new() };
        let dimensions = shape
            .into_iter()
            .map(NSNumber::new_usize)
            .collect::<Vec<_>>();
        let shape = NSArray::from_retained_slice(&dimensions);
        // SAFETY: static positive dimensions and supported integer dtype.
        let placeholder =
            unsafe { graph.placeholderWithShape_dataType_name(Some(&shape), dtype, None) };
        assert!(matches!(
            MpsGraphProgram::new(graph, placeholder.clone(), vec![placeholder], expected,),
            Err(Error::InvalidTensorContract { .. })
        ));
    }
}

#[test]
fn prepared_group_submits_decode_and_identity_graph_without_a_cpu_wait() {
    if !metal_runtime_gate("j2k-mpsgraph pipelined identity graph") {
        return;
    }
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MpsGraphBatchDecoder::system_default(options).expect("Apple Silicon MPSGraph decoder");
    let encoded = Arc::<[u8]>::from(htj2k_rgb8_fixture(8, 8));
    let prepared = decoder
        .prepare(vec![EncodedImage::full(encoded)])
        .expect("prepared image");
    let group = &prepared.groups()[0];
    let spec = MpsGraphTensorSpec::from_group_info(group.info(), group.images().len())
        .expect("group tensor spec");
    let program = MpsGraphProgram::identity(spec).expect("identity graph");

    let submitted = decoder
        .submit_prepared_group(&program, group)
        .expect("nonblocking direct submission");
    let output = submitted.wait().expect("decode and graph completion");

    assert_eq!(output.source_indices(), [0]);
    assert_eq!(output.results().len(), 1);
}

#[test]
fn dropping_an_inflight_run_waits_before_releasing_its_input() {
    if !metal_runtime_gate("j2k-mpsgraph immediate-drop safety") {
        return;
    }
    let mut decoder = MpsGraphBatchDecoder::system_default(BatchDecodeOptions::default())
        .expect("Apple Silicon MPSGraph decoder");
    let encoded = Arc::<[u8]>::from(htj2k_rgb8_fixture(8, 8));
    let prepared = decoder
        .prepare(vec![EncodedImage::full(encoded)])
        .expect("prepared image");
    let group = &prepared.groups()[0];
    let spec = MpsGraphTensorSpec::from_group_info(group.info(), group.images().len())
        .expect("group tensor spec");
    let program = MpsGraphProgram::identity(spec).expect("identity graph");

    drop(
        decoder
            .submit_prepared_group(&program, group)
            .expect("nonblocking direct submission"),
    );

    assert_eq!(decoder.submissions().expect("submission count"), 1);
}

#[test]
fn identity_graph_matches_cpu_for_layouts_and_all_request_shapes() {
    if !metal_runtime_gate("j2k-mpsgraph request/layout identity parity") {
        return;
    }
    let encoded = Arc::<[u8]>::from(htj2k_rgb8_fixture(16, 16));
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region {
            roi: Rect {
                x: 2,
                y: 3,
                w: 9,
                h: 8,
            },
        },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi: Rect {
                x: 2,
                y: 3,
                w: 9,
                h: 8,
            },
            scale: Downscale::Half,
        },
    ];
    for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        for request in requests {
            let inputs = vec![EncodedImage::new(encoded.clone(), request)];
            let mut cpu = CpuBatchDecoder::new(options);
            let expected = cpu.decode(inputs.clone()).expect("CPU oracle");
            let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
                panic!("RGB8 oracle must use U8 storage")
            };
            let mut decoder =
                MpsGraphBatchDecoder::system_default(options).expect("MPSGraph decoder");
            let prepared = decoder.prepare(inputs).expect("prepared request");
            let group = &prepared.groups()[0];
            let spec = MpsGraphTensorSpec::from_group_info(group.info(), group.images().len())
                .expect("request tensor spec");
            let program = MpsGraphProgram::identity(spec).expect("identity graph");
            let actual = decoder
                .run_prepared_group(&program, group)
                .expect("direct identity run");
            assert_eq!(read_u8_result(&actual, expected.len()), *expected);
        }
    }
}

#[test]
fn irreversible_identity_graph_stays_within_one_integer_lsb() {
    if !metal_runtime_gate("j2k-mpsgraph irreversible identity parity") {
        return;
    }
    let encoded = Arc::<[u8]>::from(htj2k_rgb8_97_fixture(16, 16));
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let inputs = vec![EncodedImage::full(encoded)];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU oracle");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("RGB8 oracle must use U8 storage")
    };
    let mut decoder = MpsGraphBatchDecoder::system_default(options).expect("MPSGraph decoder");
    let prepared = decoder
        .prepare(inputs)
        .expect("prepared irreversible input");
    let group = &prepared.groups()[0];
    let spec = MpsGraphTensorSpec::from_group_info(group.info(), group.images().len())
        .expect("irreversible tensor spec");
    let output = decoder
        .run_prepared_group(
            &MpsGraphProgram::identity(spec).expect("identity graph"),
            group,
        )
        .expect("irreversible direct identity run");
    let actual = read_u8_result(&output, expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| actual.abs_diff(*expected) <= 1));
}

#[test]
fn identity_graph_is_exact_for_every_native_color_dtype_and_layout() {
    if !metal_runtime_gate("j2k-mpsgraph complete native identity matrix") {
        return;
    }
    for color in [PixelLayout::Gray, PixelLayout::Rgb, PixelLayout::Rgba] {
        for sample_type in [SampleType::U8, SampleType::U16, SampleType::I16] {
            let encoded = native_fixture(color, sample_type);
            for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
                let options = BatchDecodeOptions {
                    layout,
                    ..BatchDecodeOptions::default()
                };
                let inputs = vec![EncodedImage::full(encoded.clone())];
                let mut cpu = CpuBatchDecoder::new(options);
                let expected = cpu.decode(inputs.clone()).expect("CPU native oracle");
                assert!(expected.errors().is_empty());
                let expected = cpu_sample_bytes(expected.groups()[0].samples());
                let mut decoder = MpsGraphBatchDecoder::system_default(options)
                    .expect("MPSGraph native matrix decoder");
                let prepared = decoder.prepare(inputs).expect("prepare native matrix cell");
                let group = &prepared.groups()[0];
                let spec = MpsGraphTensorSpec::from_group_info(group.info(), group.images().len())
                    .expect("native matrix tensor spec");
                let actual = decoder
                    .run_prepared_group(
                        &MpsGraphProgram::identity(spec).expect("identity graph"),
                        group,
                    )
                    .expect("native matrix identity run");
                assert_eq!(
                    read_result_bytes(&actual, expected.len()),
                    expected,
                    "identity mismatch for {color:?}/{sample_type:?}/{layout:?}",
                );
            }
        }
    }
}

#[test]
fn completed_pipelined_and_nonblocking_paths_are_equivalent() {
    if !metal_runtime_gate("j2k-mpsgraph execution path equivalence") {
        return;
    }
    let encoded = Arc::<[u8]>::from(htj2k_rgb8_fixture(16, 16));
    let inputs = vec![EncodedImage::full(encoded)];
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = MpsGraphBatchDecoder::system_default(options).expect("MPSGraph decoder");
    let prepared = decoder.prepare(inputs.clone()).expect("prepared input");
    let group = &prepared.groups()[0];
    let spec = MpsGraphTensorSpec::from_group_info(group.info(), group.images().len())
        .expect("tensor spec");
    let program = MpsGraphProgram::identity(spec).expect("identity graph");
    let byte_len = spec.shape().into_iter().product::<usize>();

    let completed = decoder.decode(inputs).expect("completed decode");
    let (mut completed_groups, errors, group_errors) = completed.into_parts();
    assert!(errors.is_empty());
    assert!(group_errors.is_empty());
    assert_eq!(completed_groups.len(), 1);
    let completed = program
        .submit_completed(decoder.command_queue(), completed_groups.remove(0))
        .expect("completed-buffer submission")
        .wait()
        .expect("completed-buffer graph output");
    let completed = read_u8_result(&completed, byte_len);

    let pipelined = decoder
        .run_prepared_group(&program, group)
        .expect("pipelined blocking output");
    let pipelined = read_u8_result(&pipelined, byte_len);

    let submitted = decoder
        .submit_prepared_group(&program, group)
        .expect("nonblocking submission");
    let nonblocking = submitted.wait().expect("nonblocking output");
    let nonblocking = read_u8_result(&nonblocking, byte_len);

    assert_eq!(completed, pipelined);
    assert_eq!(pipelined, nonblocking);
}

#[test]
#[ignore = "1,000-submission release soak"]
fn one_thousand_direct_submissions_reuse_one_session() {
    if !metal_runtime_gate("j2k-mpsgraph 1,000-submission soak") {
        return;
    }
    let encoded = Arc::<[u8]>::from(htj2k_rgb8_fixture(8, 8));
    let mut decoder = MpsGraphBatchDecoder::system_default(BatchDecodeOptions::default())
        .expect("MPSGraph decoder");
    let prepared = decoder
        .prepare(vec![EncodedImage::full(encoded)])
        .expect("prepared soak input");
    let group = &prepared.groups()[0];
    let spec = MpsGraphTensorSpec::from_group_info(group.info(), group.images().len())
        .expect("soak tensor spec");
    let program = MpsGraphProgram::identity(spec).expect("identity graph");
    for _ in 0..1_000 {
        decoder
            .run_prepared_group(&program, group)
            .expect("reusable direct submission");
    }
    assert_eq!(decoder.submissions().expect("submission count"), 1_000);
}
