// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact Metal operations against the CPU operation oracle.

use std::sync::Arc;

use j2k::{
    wrap_j2k_codestream, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    EncodedImage, J2kCancellationToken, J2kChannelAssociation, J2kChannelDefinition,
    J2kChannelType, J2kDecodeControl, J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions,
    PreparedBatch, PreparedImageOperation, Rect,
};
use j2k_core::Colorspace;
use j2k_native::{encode, encode_htj2k, EncodeOptions};

use super::MetalBatchDecoder;

fn options() -> BatchDecodeOptions {
    let mut options = BatchDecodeOptions::default();
    options.layout = BatchLayout::Nhwc;
    options
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

fn prepare(bytes: Arc<[u8]>) -> PreparedBatch {
    let prepared =
        j2k::prepare_batch(vec![EncodedImage::full(bytes)], options()).expect("prepare source");
    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    prepared
}

/// Per-operation samples and decoded rectangles from the CPU operation path.
fn cpu_operations(operations: &[PreparedImageOperation<'_>]) -> Vec<(Rect, Vec<u8>)> {
    let result = CpuBatchDecoder::new(options())
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

fn metal_operations(
    decoder: &mut MetalBatchDecoder,
    operations: &[PreparedImageOperation<'_>],
) -> Vec<(Rect, Vec<u8>)> {
    let (groups, errors, group_errors) = decoder
        .decode_operations(operations)
        .expect("Metal operations")
        .into_parts();
    assert!(errors.is_empty(), "{errors:?}");
    assert!(group_errors.is_empty(), "{group_errors:?}");
    let mut images = vec![(Rect::full((0, 0)), Vec::new()); operations.len()];
    for group in groups {
        let (_, indices, rects, _, surfaces) = group.into_parts();
        for ((index, rect), surface) in indices.into_iter().zip(rects).zip(surfaces) {
            images[index] = (
                rect,
                surface.as_bytes().expect("operation surface").into_owned(),
            );
        }
    }
    images
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
fn exact_operations_match_the_cpu_at_deep_reductions_and_edge_regions() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options()).expect("Metal decoder");
    for (components, ht, reversible) in [
        (1, true, true),
        (1, false, true),
        (1, true, false),
        (3, true, true),
        (3, false, false),
        (4, true, true),
    ] {
        let prepared = prepare(deep_textured(components, ht, reversible));
        let operations = exact_operations(&prepared);
        let expected = cpu_operations(&operations);
        let actual = metal_operations(&mut decoder, &operations);
        for (index, ((expected_rect, expected), (rect, pixels))) in
            expected.iter().zip(&actual).enumerate()
        {
            let label =
                format!("components={components} ht={ht} reversible={reversible} op={index}");
            assert_eq!(rect, expected_rect, "{label}");
            assert_eq!(pixels, expected, "{label}");
        }
    }
}

#[test]
fn exact_operations_of_an_odd_origin_image_match_the_cpu_at_every_level() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options()).expect("Metal decoder");
    for bytes in [
        j2k_test_support::OPENJPEG_ORIGIN_23X17_AT_3_5,
        j2k_test_support::OPENJPH_ORIGIN_23X17_AT_3_5,
    ] {
        let max_levels = j2k::J2kDecoder::new(bytes)
            .expect("origin decoder")
            .max_reduction_levels()
            .expect("origin ladder");
        let prepared = prepare(bytes.into());
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
        assert_eq!(
            metal_operations(&mut decoder, &operations),
            cpu_operations(&operations)
        );
    }
}

#[test]
fn operation_plans_never_reuse_the_sources_own_prepared_plan() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options()).expect("Metal decoder");
    let prepared = prepare(deep_textured(1, true, true));
    // Cache the source's full-resolution plan under the same input identity.
    let (groups, errors, group_errors) = decoder
        .decode_prepared(&prepared)
        .expect("decode source")
        .into_parts();
    assert!(errors.is_empty() && group_errors.is_empty());
    assert_eq!(groups[0].info().dimensions, (301, 259));
    let operations = exact_operations(&prepared);
    assert_eq!(
        metal_operations(&mut decoder, &operations),
        cpu_operations(&operations)
    );
    // A repeated call hits the operation plans it just cached.
    assert_eq!(
        metal_operations(&mut decoder, &operations),
        cpu_operations(&operations)
    );
}

#[test]
fn operation_controls_bound_output_and_cancel_without_submitting() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options()).expect("Metal decoder");
    let prepared = prepare(deep_textured(1, true, true));
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
    let before = decoder.submissions().expect("submissions");
    let (groups, errors, group_errors) = decoder
        .decode_operations(&operations)
        .expect("Metal operations")
        .into_parts();
    assert!(group_errors.is_empty(), "{group_errors:?}");
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
    let [group] = groups.as_slice() else {
        panic!("one bounded group, got {}", groups.len());
    };
    assert_eq!(group.source_indices(), [0]);
    assert_eq!(decoder.submissions().expect("submissions"), before + 1);
    assert_eq!(
        metal_operations(&mut decoder, &[unbounded.with_control(&fits)]),
        cpu_operations(&[unbounded])
    );
}

#[test]
fn a_group_cancelled_after_preparation_is_never_committed() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options()).expect("Metal decoder");
    let prepared = prepare(deep_textured(1, true, true));
    let image = &prepared.groups()[0].images()[0];
    let token = J2kCancellationToken::new();
    let mut control = J2kDecodeControl::default();
    control.cancellation = Some(token.clone());
    let operations =
        [PreparedImageOperation::new(image, Rect::full((301, 259)), 4).with_control(&control)];
    let device = j2k::prepare_device_operations(&operations, options()).expect("device operations");
    assert!(device.errors().is_empty(), "{:?}", device.errors());
    let group = &device.groups()[0];
    let (width, height) = group.info().dimensions;
    let fmt = group
        .info()
        .native_pixel_format()
        .expect("native gray format");
    let row_bytes = width as usize * fmt.bytes_per_pixel();
    let output = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        row_bytes * height as usize,
    )
    .expect("operation destination allocation");
    let layout = j2k_metal_support::MetalImageLayout::new_batch(
        0,
        (width, height),
        row_bytes,
        fmt,
        1,
        row_bytes * height as usize,
    )
    .expect("operation destination layout");
    // SAFETY: the fresh allocation has one logical writer until completion.
    let destination = unsafe {
        j2k_metal_support::MetalImageDestination::from_exclusive_buffer(output, layout)
            .expect("operation destination")
    };
    token.cancel();
    let before = decoder.submissions().expect("submissions");
    let Err(error) = decoder.submit_operation_group_into(group, destination) else {
        panic!("a cancelled group is not submitted");
    };
    assert!(
        matches!(error, crate::Error::Decode(j2k::J2kError::Cancelled)),
        "{error}"
    );
    assert_eq!(decoder.submissions().expect("submissions"), before);
}

/// Submits `operations` as one external operation group and checks that every
/// operation decoded its Tier-1 code blocks on the CPU and matches `expected`.
fn assert_external_operation_group_uses_cpu_tier1(
    decoder: &mut MetalBatchDecoder,
    operations: &[PreparedImageOperation<'_>],
    expected: &[(Rect, Vec<u8>)],
) {
    let device = j2k::prepare_device_operations(operations, options()).expect("device operations");
    assert!(device.errors().is_empty(), "{:?}", device.errors());
    assert_eq!(device.groups().len(), 1);
    let group = &device.groups()[0];
    let fmt = group
        .info()
        .native_pixel_format()
        .expect("native RGB format");
    let (width, height) = group.info().dimensions;
    let row_bytes = width as usize * fmt.bytes_per_pixel();
    let image_bytes = row_bytes * height as usize;
    let output = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        image_bytes * group.source_indices().len(),
    )
    .expect("operation destination allocation");
    let layout = j2k_metal_support::MetalImageLayout::new_batch(
        0,
        (width, height),
        row_bytes,
        fmt,
        group.source_indices().len(),
        image_bytes,
    )
    .expect("operation destination layout");
    // SAFETY: the fresh allocation has one logical writer until completion.
    let destination = unsafe {
        j2k_metal_support::MetalImageDestination::from_exclusive_buffer(output.clone(), layout)
            .expect("operation destination")
    };
    let submitted = decoder
        .submit_operation_group_into(group, destination)
        .expect("submit operation group");
    assert_eq!(submitted.cpu_tier1_images_for_test(), operations.len());
    submitted.wait().expect("complete operation group");
    // SAFETY: completion released the destination's exclusive write guard.
    let bytes = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(
            &output,
            0,
            image_bytes * group.source_indices().len(),
        )
        .expect("operation destination bytes")
    };
    for (slot, &index) in group.source_indices().iter().enumerate() {
        assert!(
            bytes[slot * image_bytes..][..image_bytes] == expected[index].1[..],
            "external operation {index} differs from the CPU"
        );
    }
}

/// Eight distinct sampled classic RGB8 sources cross the automatic CPU Tier-1
/// threshold. Whole-image operations at full resolution take that route and
/// still match the CPU; sampled regions keep failing per group with a
/// capability error rather than decoding the wrong samples.
#[test]
fn sampled_classic_operations_use_automatic_cpu_tier1_and_match_the_cpu() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options()).expect("Metal decoder");
    let inputs = (0..8)
        .map(|seed| {
            EncodedImage::full(super::sampled_group_tests::sampled_rgb(
                (130, 128),
                (2, 1),
                8,
                false,
                seed,
            ))
        })
        .collect::<Vec<_>>();
    let prepared = j2k::prepare_batch(inputs, options()).expect("prepare sampled sources");
    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    let images = prepared
        .groups()
        .iter()
        .flat_map(j2k::PreparedBatchGroup::images)
        .collect::<Vec<_>>();
    assert_eq!(images.len(), 8);
    let whole = images
        .iter()
        .map(|image| PreparedImageOperation::new(image, Rect::full((130, 128)), 0))
        .collect::<Vec<_>>();
    let expected = cpu_operations(&whole);
    let (groups, errors, group_errors) = decoder
        .decode_operations(&whole)
        .expect("Metal operations")
        .into_parts();
    assert!(errors.is_empty(), "{errors:?}");
    assert!(group_errors.is_empty(), "{group_errors:?}");
    let mut cpu_tier1_images = 0;
    let mut actual = vec![(Rect::full((0, 0)), Vec::new()); whole.len()];
    for group in groups {
        cpu_tier1_images += group.dispatch_report().cpu_tier1_images;
        let (_, indices, rects, _, surfaces) = group.into_parts();
        for ((index, rect), surface) in indices.into_iter().zip(rects).zip(surfaces) {
            actual[index] = (
                rect,
                surface.as_bytes().expect("operation surface").into_owned(),
            );
        }
    }
    assert_eq!(cpu_tier1_images, whole.len());
    assert_eq!(actual, expected);

    // The external operation-group entry point takes the same route.
    assert_external_operation_group_uses_cpu_tier1(&mut decoder, &whole, &expected);

    let region = Rect {
        x: 17,
        y: 9,
        w: 64,
        h: 48,
    };
    let regions = images
        .iter()
        .map(|image| PreparedImageOperation::new(image, region, 0))
        .collect::<Vec<_>>();
    let (groups, errors, group_errors) = decoder
        .decode_operations(&regions)
        .expect("Metal operations")
        .into_parts();
    assert!(groups.is_empty());
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(group_errors.len(), 1, "{group_errors:?}");
    assert_eq!(group_errors[0].source_indices, (0..8).collect::<Vec<_>>());
    assert!(
        matches!(
            *group_errors[0].source,
            crate::Error::UnsupportedMetalRequest { .. }
        ),
        "{:?}",
        group_errors[0].source
    );
}
