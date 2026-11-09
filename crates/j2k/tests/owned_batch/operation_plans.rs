// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact-operation planning: the plans the C and Python bindings report come
//! from these facade calls, so each plan must describe the decode it precedes.

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    prepare_batch, BatchDecodeOptions, CpuBatchDecoder, CpuBatchSamples, EncodedImage,
    J2kDecodeControl, J2kDecoder, J2kError, J2kSampleInterpretation, J2kScratchPool,
    NativeSampleType, PixelFormat, PreparedImageOperation, Rect,
};

use super::fixtures::htj2k_native_fixture_with_levels;

const SIZE: (u32, u32) = (65, 49);

fn regions() -> [Rect; 2] {
    [
        Rect::full(SIZE),
        Rect {
            x: 9,
            y: 11,
            w: 33,
            h: 25,
        },
    ]
}

fn fixture(components: u16, precision: u8, signed: bool) -> Arc<[u8]> {
    htj2k_native_fixture_with_levels(components, precision, signed, SIZE.0, SIZE.1, 4).into()
}

#[test]
fn planned_operations_describe_the_output_their_decode_returns() {
    let mut control = J2kDecodeControl::default();
    control.max_working_memory_bytes = NonZeroUsize::new(8 << 20);
    for (components, precision, signed) in [(1, 8, false), (3, 12, false), (3, 16, true)] {
        let prepared = prepare_batch(
            vec![EncodedImage::operation_source(fixture(
                components, precision, signed,
            ))],
            BatchDecodeOptions::default(),
        )
        .unwrap();
        assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
        let image = &prepared.groups()[0].images()[0];
        let mut session = CpuBatchDecoder::new(BatchDecodeOptions::default());
        for levels in [0, 2, 4] {
            for roi in regions() {
                for controlled in [false, true] {
                    let operation = PreparedImageOperation::new(image, roi, levels);
                    let operation = if controlled {
                        operation.with_control(&control)
                    } else {
                        operation
                    };
                    let plans = session.plan_operations(&[operation]).unwrap();
                    let [Ok(plan)] = plans.as_slice() else {
                        panic!("plan of {roi:?} at {levels}: {plans:?}");
                    };
                    let decoded = session.decode_operations(&[operation]).unwrap();
                    assert!(decoded.errors().is_empty(), "{:?}", decoded.errors());
                    let [group] = decoded.groups() else {
                        panic!("one group per operation");
                    };
                    assert_eq!(group.info(), &plan.info, "{roi:?} at {levels}");
                    assert_eq!(group.decoded_rects(), [plan.output_rect]);
                    assert_eq!(
                        plan.info.dimensions,
                        (plan.output_rect.w, plan.output_rect.h)
                    );
                    assert!(!plan.used_lenient_metadata_recovery);
                }
            }
        }
    }
}

#[test]
fn planned_operations_report_the_errors_their_decode_reports() {
    let prepared = prepare_batch(
        vec![EncodedImage::operation_source(fixture(3, 8, false))],
        BatchDecodeOptions::default(),
    )
    .unwrap();
    let image = &prepared.groups()[0].images()[0];
    let mut session = CpuBatchDecoder::new(BatchDecodeOptions::default());
    // A region past the image's right edge is invalid; the other operation
    // is valid, and its plan is unaffected by its sibling's error.
    let outside = Rect {
        x: 60,
        y: 3,
        w: 10,
        h: 4,
    };
    let operations = [
        PreparedImageOperation::new(image, outside, 1),
        PreparedImageOperation::new(image, Rect::full(SIZE), 1),
    ];
    let plans = session.plan_operations(&operations).unwrap();
    let decoded = session.decode_operations(&operations).unwrap();
    let [failed] = decoded.errors() else {
        panic!("one failed operation: {:?}", decoded.errors());
    };
    assert_eq!(failed.index, 0);
    let planned = plans[0].as_ref().expect_err("planned failure");
    assert_eq!(planned.to_string(), failed.source.to_string());
    let plan = plans[1].as_ref().expect("valid operation plan");
    assert_eq!(decoded.groups()[0].info(), &plan.info);
}

#[test]
fn native_region_plans_agree_with_session_plans_and_keep_signedness() {
    for (components, precision, signed, format, sample_type) in [
        (1, 8, false, PixelFormat::Gray8, NativeSampleType::U8),
        (3, 12, false, PixelFormat::Rgb16, NativeSampleType::U16),
        (3, 16, true, PixelFormat::Rgb16, NativeSampleType::I16),
    ] {
        let bytes = fixture(components, precision, signed);
        let prepared = prepare_batch(
            vec![EncodedImage::operation_source(Arc::clone(&bytes))],
            BatchDecodeOptions::default(),
        )
        .unwrap();
        let image = &prepared.groups()[0].images()[0];
        let mut session = CpuBatchDecoder::new(BatchDecodeOptions::default());
        for levels in [0, 3] {
            for roi in regions() {
                let mut decoder = J2kDecoder::new(&bytes).unwrap();
                let native = decoder.native_region_decode_plan(roi, levels).unwrap();
                assert_eq!(native.region, {
                    let mut decoder = J2kDecoder::new(&bytes).unwrap();
                    decoder.region_decode_plan(format, roi, levels).unwrap()
                });
                assert_eq!(native.samples.sample_type, sample_type);
                assert_eq!(native.samples.precision, precision);
                assert_eq!(native.samples.signed, signed);
                let plans = session
                    .plan_operations(&[PreparedImageOperation::new(image, roi, levels)])
                    .unwrap();
                let [Ok(plan)] = plans.as_slice() else {
                    panic!("{plans:?}");
                };
                assert_eq!(native.samples, plan.info, "{roi:?} at {levels}");
                assert_eq!(native.region.output_region, plan.output_rect);
            }
        }
    }
}

#[test]
fn native_region_plan_under_control_applies_the_output_limit() {
    let bytes = fixture(3, 8, false);
    let mut control = J2kDecodeControl::default();
    let full = J2kDecoder::new(&bytes)
        .unwrap()
        .native_region_decode_plan(Rect::full(SIZE), 0)
        .unwrap();
    control.max_output_bytes = Some(full.region.buffer_bytes);
    let fits = J2kDecoder::new(&bytes)
        .unwrap()
        .native_region_decode_plan_with_control(Rect::full(SIZE), 0, &control)
        .unwrap();
    assert_eq!(fits, full);
    control.max_output_bytes = Some(full.region.buffer_bytes - 1);
    let refused = J2kDecoder::new(&bytes)
        .unwrap()
        .native_region_decode_plan_with_control(Rect::full(SIZE), 0, &control);
    assert!(
        matches!(
            refused,
            Err(J2kError::Buffer(
                j2k::BufferError::AllocationTooLarge { .. }
            ))
        ),
        "{refused:?}"
    );
}

#[test]
fn planar_srgb8_output_stores_each_channel_as_a_plane() {
    for components in [1, 3] {
        let bytes = fixture(components, 8, false);
        let packed = J2kDecoder::new(&bytes)
            .unwrap()
            .decode_srgb8()
            .unwrap()
            .into_data();
        let planar = J2kDecoder::new(&bytes)
            .unwrap()
            .decode_srgb8()
            .unwrap()
            .into_planar_data_with_control(&J2kDecodeControl::default())
            .unwrap();
        let channels = usize::from(components);
        let expected: Vec<u8> = (0..channels)
            .flat_map(|channel| packed.iter().skip(channel).step_by(channels).copied())
            .collect();
        assert_eq!(planar, expected);
    }
}

#[test]
fn planar_srgb8_output_keeps_both_copies_within_the_working_limit() {
    let bytes = fixture(3, 8, false);
    let image = J2kDecoder::new(&bytes).unwrap().decode_srgb8().unwrap();
    let packed = image.data().len();
    let mut control = J2kDecodeControl::default();
    // The packed samples alone fit; they and their planar copy do not.
    control.max_working_memory_bytes = NonZeroUsize::new(packed + packed / 2);
    let refused = image.into_planar_data_with_control(&control);
    assert!(
        matches!(refused, Err(J2kError::ResourceLimit { .. })),
        "{refused:?}"
    );
}

/// A signed ramp over every value of `precision` bits, one byte per sample.
fn signed_narrow_fixture(precision: u8) -> Arc<[u8]> {
    let span = 1_i32 << precision;
    let count = i32::try_from(SIZE.0 * SIZE.1).unwrap();
    let pixels: Vec<u8> = (0..count)
        .map(|index| {
            let value = (index * 7 + 3) % span - span / 2;
            i8::try_from(value).unwrap().to_le_bytes()[0]
        })
        .collect();
    j2k_native::encode_htj2k(
        &pixels,
        SIZE.0,
        SIZE.1,
        1,
        precision,
        true,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .unwrap()
    .into()
}

/// 16-bit output samples read back as their signed values.
fn signed_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|sample| f32::from(i16::from_le_bytes(*sample)))
        .collect()
}

/// Signed sources of at most eight bits keep their values in 16-bit output,
/// as 16-bit two's complement: a whole-image and a region decode to Gray16
/// match the decoded component planes and the I16 samples of the native
/// batch operations, which the native region plan describes.
#[test]
fn signed_narrow_sources_keep_their_values_in_sixteen_bit_output() {
    for precision in [1, 4, 7, 8] {
        let bytes = signed_narrow_fixture(precision);
        let planes = J2kDecoder::new(&bytes)
            .unwrap()
            .decode_components()
            .unwrap()
            .planes()[0]
            .samples()
            .to_vec();
        assert!(planes.iter().any(|sample| *sample < 0.0));
        let row_bytes = usize::try_from(SIZE.0).unwrap() * 2;
        let mut gray16 = vec![0_u8; row_bytes * usize::try_from(SIZE.1).unwrap()];
        J2kDecoder::new(&bytes)
            .unwrap()
            .decode_into(&mut gray16, row_bytes, PixelFormat::Gray16)
            .unwrap();
        assert_eq!(signed_values(&gray16), planes, "{precision}-bit image");
        let prepared = prepare_batch(
            vec![EncodedImage::operation_source(Arc::clone(&bytes))],
            BatchDecodeOptions::default(),
        )
        .unwrap();
        let image = &prepared.groups()[0].images()[0];
        let mut session = CpuBatchDecoder::new(BatchDecodeOptions::default());
        for roi in regions() {
            let label = format!("{precision}-bit {roi:?}");
            let decoded = session
                .decode_operations(&[PreparedImageOperation::new(image, roi, 0)])
                .unwrap();
            let [group] = decoded.groups() else {
                panic!("{label}: one group");
            };
            let CpuBatchSamples::I16(batch) = group.samples() else {
                panic!("{label}: I16 samples");
            };
            let batch: Vec<f32> = batch.iter().copied().map(f32::from).collect();
            if roi == Rect::full(SIZE) {
                assert_eq!(batch, planes, "{label}");
            }
            let plan = J2kDecoder::new(&bytes)
                .unwrap()
                .native_region_decode_plan(roi, 0)
                .unwrap();
            assert_eq!(plan.samples.sample_type, NativeSampleType::I16, "{label}");
            assert_eq!(plan.region.format, PixelFormat::Gray16, "{label}");
            assert_eq!(
                plan.region.sample_interpretation,
                J2kSampleInterpretation::Native16,
                "{label}"
            );
            let mut region = vec![0_u8; plan.region.buffer_bytes];
            J2kDecoder::new(&bytes)
                .unwrap()
                .decode_region_into(
                    &mut J2kScratchPool::new(),
                    &mut region,
                    plan.region.row_bytes,
                    PixelFormat::Gray16,
                    roi,
                )
                .unwrap();
            assert_eq!(signed_values(&region), batch, "{label}");
        }
    }
}
