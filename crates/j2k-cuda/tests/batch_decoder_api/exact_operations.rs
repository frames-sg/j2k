// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact CUDA operations against the CPU operation oracle.

use std::sync::Arc;

use j2k::{
    wrap_j2k_codestream, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    EncodedImage, J2kCancellationToken, J2kChannelAssociation, J2kChannelDefinition,
    J2kChannelType, J2kDecodeControl, J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions,
    PreparedBatch, PreparedImageOperation, Rect,
};
use j2k_core::Colorspace;
use j2k_cuda::{CudaBatchDecoder, CudaBatchError, CudaSession};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};
use j2k_native::{encode, encode_htj2k, EncodeOptions};

/// Decoded rectangle and native sample bytes of one operation.
type Decoded = (Rect, Vec<u8>);

fn options(layout: BatchLayout) -> BatchDecodeOptions {
    let mut options = BatchDecodeOptions::default();
    options.layout = layout;
    options
}

fn decoder(context: &CudaContext, layout: BatchLayout) -> CudaBatchDecoder {
    CudaBatchDecoder::with_session_and_options(
        CudaSession::with_context(context.clone()),
        options(layout),
    )
}

/// Odd-sized textured image with a five-level ladder and 16 x 16 code-blocks.
/// Four components are wrapped as sRGB with straight alpha.
fn deep_textured(components: u16, ht: bool, reversible: bool) -> Arc<[u8]> {
    let (width, height) = (301_u32, 259_u32);
    let pixels: Vec<u8> = (0..width * height * u32::from(components))
        .map(|index| {
            let pixel = index / u32::from(components);
            let (x, y) = (pixel % width, pixel / width);
            u8::try_from((x * 7 + y * 13 + (x * y) / 5 + index % 3 * 41) & 0xff).expect("masked")
        })
        .collect();
    let options = EncodeOptions {
        reversible,
        use_mct: components == 3,
        num_decomposition_levels: 5,
        code_block_width_exp: 4,
        code_block_height_exp: 4,
        ..EncodeOptions::default()
    };
    let encoder = if ht { encode_htj2k } else { encode };
    let codestream = encoder(&pixels, width, height, components, 8, false, &options)
        .expect("encode deep textured fixture");
    if components != 4 {
        return codestream.into();
    }
    let definitions: Vec<_> = (0..4_u16)
        .map(|channel| J2kChannelDefinition {
            channel_index: channel,
            channel_type: if channel == 3 {
                J2kChannelType::Opacity
            } else {
                J2kChannelType::Color
            },
            association: if channel == 3 {
                J2kChannelAssociation::WholeImage
            } else {
                J2kChannelAssociation::Color { index: channel + 1 }
            },
        })
        .collect();
    wrap_j2k_codestream(
        &codestream,
        if ht {
            J2kFileWrapOptions::jph()
        } else {
            J2kFileWrapOptions::jp2()
        }
        .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
        .with_metadata(J2kFileBoxMetadata {
            channel_definitions: &definitions,
            ..J2kFileBoxMetadata::empty()
        }),
    )
    .expect("wrap RGBA fixture")
    .into()
}

fn prepare(bytes: Arc<[u8]>, layout: BatchLayout) -> PreparedBatch {
    let prepared = j2k::prepare_batch(vec![EncodedImage::full(bytes)], options(layout))
        .expect("prepare source");
    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    prepared
}

/// Per-operation samples and decoded rectangles from the CPU operation path.
fn cpu_operations(operations: &[PreparedImageOperation<'_>], layout: BatchLayout) -> Vec<Decoded> {
    let result = CpuBatchDecoder::new(options(layout))
        .decode_operations(operations)
        .expect("CPU operation oracle");
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    let mut images = vec![(Rect::full((0, 0)), Vec::new()); operations.len()];
    for group in result.groups() {
        let CpuBatchSamples::U8(samples) = group.samples() else {
            panic!("8-bit fixtures decode to U8");
        };
        let per_image = samples.len() / group.source_indices().len();
        for (position, &index) in group.source_indices().iter().enumerate() {
            images[index] = (
                group.decoded_rects()[position],
                samples[position * per_image..][..per_image].to_vec(),
            );
        }
    }
    images
}

/// Per-operation samples read back from each resident group's dense output.
fn cuda_operations(
    decoder: &mut CudaBatchDecoder,
    operations: &[PreparedImageOperation<'_>],
) -> Vec<Decoded> {
    let result = decoder
        .decode_operations(operations)
        .expect("CUDA operations");
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    assert!(
        result.group_errors().is_empty(),
        "{:?}",
        result.group_errors()
    );
    let mut images = vec![(Rect::full((0, 0)), Vec::new()); operations.len()];
    for group in result.groups() {
        let dense = group.dense_output();
        for ((&index, &rect), range) in group
            .source_indices()
            .iter()
            .zip(group.decoded_rects())
            .zip(dense.ranges())
        {
            let mut samples = vec![0_u8; range.len];
            dense
                .buffer()
                .copy_range_to_host(range.offset, &mut samples)
                .expect("download operation output");
            images[index] = (rect, samples);
        }
    }
    images
}

/// Reversible and irreversible 9/7 paths both match the CPU exactly.
fn assert_operations_match(actual: &[Decoded], expected: &[Decoded], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, ((rect, samples), (expected_rect, expected_samples))) in
        actual.iter().zip(expected).enumerate()
    {
        assert_eq!(rect, expected_rect, "{label} op={index}");
        assert_eq!(samples.len(), expected_samples.len(), "{label} op={index}");
        let worst = samples
            .iter()
            .zip(expected_samples)
            .map(|(&sample, &expected)| sample.abs_diff(expected))
            .max()
            .unwrap_or(0);
        assert_eq!(worst, 0, "{label} op={index}: samples differ by {worst}");
    }
}

fn exact_operations(prepared: &PreparedBatch) -> Vec<PreparedImageOperation<'_>> {
    let image = &prepared.groups()[0].images()[0];
    let full = Rect::full((301, 259));
    let edge = Rect {
        x: 260,
        y: 200,
        w: 41,
        h: 59,
    };
    let interior = Rect {
        x: 37,
        y: 21,
        w: 129,
        h: 97,
    };
    [
        (full, 0),
        (full, 4),
        (full, 5),
        (edge, 0),
        (edge, 4),
        (interior, 2),
        (interior, 5),
    ]
    .into_iter()
    .map(|(roi, levels)| PreparedImageOperation::new(image, roi, levels))
    .collect()
}

#[test]
fn exact_operations_match_the_cpu_at_deep_reductions_and_edge_regions_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    for (components, ht, reversible, layout) in [
        (1, true, true, BatchLayout::Nhwc),
        (1, false, true, BatchLayout::Nhwc),
        (1, true, false, BatchLayout::Nhwc),
        (3, true, true, BatchLayout::Nhwc),
        (3, true, true, BatchLayout::Nchw),
        (3, false, false, BatchLayout::Nhwc),
        (4, true, true, BatchLayout::Nhwc),
    ] {
        let prepared = prepare(deep_textured(components, ht, reversible), layout);
        let operations = exact_operations(&prepared);
        let mut decoder = decoder(&context, layout);
        assert_operations_match(
            &cuda_operations(&mut decoder, &operations),
            &cpu_operations(&operations, layout),
            &format!("components={components} ht={ht} reversible={reversible} {layout:?}"),
        );
    }
}

#[test]
fn exact_operations_of_an_odd_origin_image_match_the_cpu_at_every_level_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let mut decoder = decoder(&context, BatchLayout::Nhwc);
    for bytes in [
        j2k_test_support::OPENJPEG_ORIGIN_23X17_AT_3_5,
        j2k_test_support::OPENJPH_ORIGIN_23X17_AT_3_5,
    ] {
        let max_levels = j2k::J2kDecoder::new(bytes)
            .expect("origin decoder")
            .max_reduction_levels()
            .expect("origin ladder");
        let prepared = prepare(bytes.into(), BatchLayout::Nhwc);
        let image = &prepared.groups()[0].images()[0];
        let edge = Rect {
            x: 15,
            y: 9,
            w: 8,
            h: 8,
        };
        let operations: Vec<_> = (0..=max_levels)
            .flat_map(|levels| {
                [Rect::full((23, 17)), edge]
                    .map(|roi| PreparedImageOperation::new(image, roi, levels))
            })
            .collect();
        assert_operations_match(
            &cuda_operations(&mut decoder, &operations),
            &cpu_operations(&operations, BatchLayout::Nhwc),
            "odd origin",
        );
    }
}

/// Allocate and wrap a caller-owned destination for `len` bytes of 8-bit
/// samples, decode `group` into it, and read it back.
fn decode_group_externally(
    context: &CudaContext,
    decoder: &mut CudaBatchDecoder,
    group: &j2k::PreparedDeviceOperationGroup,
) -> Result<(Vec<Rect>, Vec<u8>), CudaBatchError> {
    let (width, height) = group.info().dimensions;
    let fmt = group.info().native_pixel_format().expect("native format");
    let len =
        width as usize * height as usize * fmt.bytes_per_pixel() * group.source_indices().len();
    let mut allocation = context.allocate(len).expect("external destination");
    let ptr = allocation.device_ptr();
    let byte_len = allocation.byte_len();
    // SAFETY: the allocation remains live and exclusively owned until the
    // synchronous decode below returns.
    let group_output = unsafe {
        let mut destination = CudaExternalDeviceBufferViewMut::from_raw_parts(
            context,
            ptr,
            byte_len,
            1,
            &mut allocation,
        )
        .expect("external destination view");
        decoder.decode_operation_group_into(group, &mut destination)
    }?;
    let mut samples = vec![0_u8; len];
    allocation
        .copy_to_host(&mut samples)
        .expect("download external operation output");
    Ok((group_output.decoded_rects().to_vec(), samples))
}

#[test]
fn exact_operation_groups_decode_into_external_destinations_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let mut decoder = decoder(&context, BatchLayout::Nhwc);
    let prepared = prepare(deep_textured(3, true, true), BatchLayout::Nhwc);
    let operations = exact_operations(&prepared);
    let expected = cpu_operations(&operations, BatchLayout::Nhwc);
    let device = j2k::prepare_device_operations(&operations, options(BatchLayout::Nhwc))
        .expect("device operations");
    assert!(device.errors().is_empty(), "{:?}", device.errors());
    let mut actual = vec![(Rect::full((0, 0)), Vec::new()); operations.len()];
    for group in device.groups() {
        let (rects, samples) =
            decode_group_externally(&context, &mut decoder, group).expect("external group");
        let per_image = samples.len() / group.source_indices().len();
        for (position, (&index, rect)) in group.source_indices().iter().zip(rects).enumerate() {
            actual[index] = (rect, samples[position * per_image..][..per_image].to_vec());
        }
    }
    assert_operations_match(&actual, &expected, "external");
}

#[test]
fn operation_controls_fail_only_their_own_operations_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let mut decoder = decoder(&context, BatchLayout::Nhwc);
    let prepared = prepare(deep_textured(1, true, true), BatchLayout::Nhwc);
    let image = &prepared.groups()[0].images()[0];
    let full = Rect::full((301, 259));
    // Level 4 of 301 x 259 Gray8 is 19 x 17 samples.
    let output_bytes = 19 * 17;
    let mut fits = J2kDecodeControl::default();
    fits.max_output_bytes = Some(output_bytes);
    let mut too_small = J2kDecodeControl::default();
    too_small.max_output_bytes = Some(output_bytes - 1);
    let mut working = J2kDecodeControl::default();
    working.max_working_memory_bytes = core::num::NonZeroUsize::new(64 << 20);
    let token = J2kCancellationToken::new();
    token.cancel();
    let mut cancelled = J2kDecodeControl::default();
    cancelled.cancellation = Some(token);
    let unbounded = PreparedImageOperation::new(image, full, 4);
    let operations = [
        unbounded.with_control(&fits),
        unbounded.with_control(&too_small),
        unbounded.with_control(&working),
        unbounded.with_control(&cancelled),
        PreparedImageOperation::new(image, full, 6),
    ];
    let result = decoder
        .decode_operations(&operations)
        .expect("CUDA operations");
    assert!(
        result.group_errors().is_empty(),
        "{:?}",
        result.group_errors()
    );
    let errors = result.errors();
    assert_eq!(
        errors.iter().map(|error| error.index).collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
    // Errors are in operation order: positions 0, 1 and 2 hold operations 1, 2 and 3.
    let message = |position: usize| errors[position].source.to_string().to_lowercase();
    assert!(message(0).contains("output"), "{}", message(0));
    assert!(message(1).contains("working-memory"), "{}", message(1));
    assert!(matches!(
        &errors[2].source,
        j2k::BatchItemError::Codec { source, .. } if matches!(**source, j2k::J2kError::Cancelled)
    ));
    let [group] = result.groups() else {
        panic!("one bounded group, got {}", result.groups().len());
    };
    assert_eq!(group.source_indices(), [0]);
    assert_operations_match(
        &cuda_operations(&mut decoder, &[unbounded.with_control(&fits)]),
        &cpu_operations(&[unbounded], BatchLayout::Nhwc),
        "bounded",
    );
}

#[test]
fn a_group_cancelled_after_preparation_is_never_submitted_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let mut decoder = decoder(&context, BatchLayout::Nhwc);
    let prepared = prepare(deep_textured(1, true, true), BatchLayout::Nhwc);
    let image = &prepared.groups()[0].images()[0];
    let token = J2kCancellationToken::new();
    let mut control = J2kDecodeControl::default();
    control.cancellation = Some(token.clone());
    let operations =
        [PreparedImageOperation::new(image, Rect::full((301, 259)), 4).with_control(&control)];
    let device = j2k::prepare_device_operations(&operations, options(BatchLayout::Nhwc))
        .expect("device operations");
    assert!(device.errors().is_empty(), "{:?}", device.errors());
    token.cancel();
    let launches = || context.diagnostics().expect("diagnostics").kernel_launches;
    let before = launches();
    let Err(error) = decode_group_externally(&context, &mut decoder, &device.groups()[0]) else {
        panic!("a cancelled group is not submitted");
    };
    assert!(
        matches!(
            &error,
            CudaBatchError::GroupExecution { source, .. }
                if matches!(**source, j2k_cuda::Error::Decode(j2k::J2kError::Cancelled))
        ),
        "{error}"
    );
    assert_eq!(launches(), before);
    // Resident decoding observes the cancelled token while preparing.
    let result = decoder
        .decode_operations(&operations)
        .expect("CUDA operations");
    assert!(result.groups().is_empty() && result.group_errors().is_empty());
    assert_eq!(result.errors().len(), 1, "cancelled during preparation");
    assert_eq!(launches(), before);
}

/// 4:2:0 classic RGB: offset plans need unit-sampled components.
fn subsampled_rgb() -> Arc<[u8]> {
    let (width, height) = (64_u32, 48_u32);
    let plane = |w: u32, h: u32, seed: u32| -> Vec<u8> {
        (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| u8::try_from((seed + x * 3 + y * 5) & 0xff).expect("masked"))
            })
            .collect()
    };
    let (luma, blue, red) = (
        plane(width, height, 0),
        plane(width / 2, height / 2, 64),
        plane(width / 2, height / 2, 192),
    );
    let planes =
        [(&luma, 1), (&blue, 2), (&red, 2)].map(|(data, rsiz)| j2k::J2kLosslessComponentPlane {
            data,
            x_rsiz: rsiz,
            y_rsiz: rsiz,
        });
    let samples = j2k::J2kLosslessComponentSamples::new(&planes, width, height, 8, false)
        .expect("subsampled component samples");
    j2k::encode_j2k_lossless_components(
        samples,
        &j2k::J2kLosslessEncodeOptions::default()
            .with_backend(j2k::EncodeBackendPreference::CpuOnly)
            .with_block_coding_mode(j2k::J2kBlockCodingMode::Classic)
            .with_reversible_transform(j2k::ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(2)),
    )
    .expect("subsampled encode")
    .codestream
    .into()
}

#[test]
fn subsampled_operations_fail_their_group_instead_of_decoding_wrong_samples_when_runtime_required()
{
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let mut decoder = decoder(&context, BatchLayout::Nhwc);
    let bytes = subsampled_rgb();
    let max_levels = j2k::J2kDecoder::new(&bytes)
        .expect("subsampled decoder")
        .max_reduction_levels()
        .expect("subsampled ladder");
    let prepared = prepare(bytes, BatchLayout::Nhwc);
    let image = &prepared.groups()[0].images()[0];
    let region = Rect {
        x: 5,
        y: 7,
        w: 17,
        h: 13,
    };
    let operations: Vec<_> = (0..=max_levels)
        .flat_map(|levels| {
            [Rect::full((64, 48)), region]
                .map(|roi| PreparedImageOperation::new(image, roi, levels))
        })
        .collect();
    // The CPU decodes every shape; CUDA must not return different samples.
    let expected = cpu_operations(&operations, BatchLayout::Nhwc);
    let result = decoder
        .decode_operations(&operations)
        .expect("CUDA operations");
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    let mut failed = Vec::new();
    for group_error in result.group_errors() {
        assert!(
            matches!(
                group_error.source(),
                j2k_cuda::Error::UnsupportedCudaRequest { .. }
            ),
            "{group_error}"
        );
        failed.extend_from_slice(group_error.source_indices());
    }
    for group in result.groups() {
        let dense = group.dense_output();
        for ((&index, &rect), range) in group
            .source_indices()
            .iter()
            .zip(group.decoded_rects())
            .zip(dense.ranges())
        {
            let mut samples = vec![0_u8; range.len];
            dense
                .buffer()
                .copy_range_to_host(range.offset, &mut samples)
                .expect("download operation output");
            assert_eq!((rect, samples), expected[index], "op={index}");
        }
    }
    failed.sort_unstable();
    assert_eq!(
        failed,
        (0..operations.len()).collect::<Vec<_>>(),
        "every subsampled shape lacks an offset plan (levels 0..={max_levels})"
    );
    // Prepared CUDA batches refuse the same source the same way.
    let legacy = decoder
        .decode_prepared(&prepared)
        .expect("CUDA prepared batch");
    assert!(legacy.groups().is_empty());
    let [group_error] = legacy.group_errors() else {
        panic!("one failed prepared group, got {:?}", legacy.group_errors());
    };
    assert!(
        matches!(
            group_error.source(),
            j2k_cuda::Error::UnsupportedCudaRequest { .. }
        ),
        "{group_error}"
    );
}
