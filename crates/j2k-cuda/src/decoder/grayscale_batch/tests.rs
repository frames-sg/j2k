// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    prepare_batch, BatchDecodeOptions, DeviceDecodePlan, DeviceDecodeRequest, EncodedImage,
    PreparationDepth,
};
use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_native::{encode_htj2k, DecodeSettings, EncodeOptions};

use super::completion::{map_grayscale_status_error, GrayscaleJobIdentity};
use super::{prepare_grayscale_batch, prepare_grayscale_batch_with_cap, GrayscaleBatchInput};

#[test]
fn grayscale_kernel_failure_maps_to_responsible_source_index() {
    let identities = [
        GrayscaleJobIdentity {
            source_index: 3,
            original_job_index: 0,
        },
        GrayscaleJobIdentity {
            source_index: 9,
            original_job_index: 1,
        },
    ];
    let mapped = map_grayscale_status_error(
        j2k_cuda_runtime::CudaError::KernelJobStatus {
            kernel: "injected",
            job_index: 1,
            code: 7,
            detail: 11,
        },
        &identities,
    );
    assert!(matches!(
        mapped,
        crate::Error::CudaTier1JobFailed {
            source_index: 9,
            original_job_index: 1,
            ..
        }
    ));
}

#[test]
fn grayscale_batch_rebases_two_plans_into_one_upload_layout() {
    let encoded = gray8_fixture();
    let prepared = prepare_grayscale_batch(
        &[
            GrayscaleBatchInput::full(encoded.as_slice()),
            GrayscaleBatchInput::full(encoded.as_slice()),
        ],
        PixelFormat::Gray8,
        DecodeSettings::strict(),
    )
    .expect("shared grayscale batch plan");

    assert_eq!(prepared.plans.len(), 2);
    assert!(prepared.payload_parts.iter().any(|part| !part.is_empty()));
    assert!(prepared.plans.iter().all(|plan| plan.payload().is_empty()));
    let first_max = prepared.plans[0]
        .code_blocks()
        .iter()
        .map(|block| block.payload_offset + u64::from(block.payload_len))
        .max()
        .expect("first block payload");
    let second_min = prepared.plans[1]
        .code_blocks()
        .iter()
        .map(|block| block.payload_offset)
        .min()
        .expect("second block payload");
    assert!(second_min >= first_max);

    // Both inputs are the same codestream, so each block must resolve to the
    // same bytes once the parts are uploaded back to back.
    let uploaded = prepared
        .payload_parts
        .iter()
        .flat_map(|part| part.iter().copied())
        .collect::<Vec<_>>();
    let bytes = |offset: u64, len: u32| {
        let start = usize::try_from(offset).expect("offset fits usize");
        &uploaded[start..start + len as usize]
    };
    let first = prepared.plans[0].code_blocks();
    let second = prepared.plans[1].code_blocks();
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second) {
        assert_eq!(
            bytes(a.payload_offset, a.payload_len),
            bytes(b.payload_offset, b.payload_len)
        );
    }
}

#[test]
fn grayscale_batch_cap_covers_retained_results_and_one_live_planner() {
    let encoded = gray8_fixture();
    let one = [GrayscaleBatchInput::full(encoded.as_slice())];
    let two = [
        GrayscaleBatchInput::full(encoded.as_slice()),
        GrayscaleBatchInput::full(encoded.as_slice()),
    ];
    let one_cap = minimum_grayscale_batch_cap(&one);
    let two_cap = minimum_grayscale_batch_cap(&two);

    assert!(
        two_cap > one_cap,
        "the second retained plan must consume aggregate headroom"
    );
    let Err(error) = prepare_grayscale_batch_with_cap(
        &two,
        PixelFormat::Gray8,
        DecodeSettings::strict(),
        two_cap - 1,
    ) else {
        panic!("one byte below the discovered aggregate cap must fail");
    };
    assert!(
        error.is_host_planning_limit(),
        "unexpected error: {error:?}"
    );
}

#[test]
fn parallel_grayscale_planning_returns_the_first_input_error() {
    let encoded = gray8_fixture();
    let first_invalid = [0xff_u8, 0x4f, 0x00];
    let later_invalid = [0x00_u8, 0x01];
    let expected = prepare_grayscale_batch(
        &[GrayscaleBatchInput::full(first_invalid.as_slice())],
        PixelFormat::Gray8,
        DecodeSettings::strict(),
    )
    .err()
    .expect("first malformed input must fail");
    let actual = prepare_grayscale_batch(
        &[
            GrayscaleBatchInput::full(encoded.as_slice()),
            GrayscaleBatchInput::full(first_invalid.as_slice()),
            GrayscaleBatchInput::full(later_invalid.as_slice()),
        ],
        PixelFormat::Gray8,
        DecodeSettings::strict(),
    )
    .err()
    .expect("batch containing malformed inputs must fail");

    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

fn minimum_grayscale_batch_cap(inputs: &[GrayscaleBatchInput<'_>]) -> usize {
    crate::decoder::plan::minimum_planning_cap(|cap| {
        prepare_grayscale_batch_with_cap(inputs, PixelFormat::Gray8, DecodeSettings::strict(), cap)
    })
}

#[test]
fn prepared_htj2k_batch_uses_retained_offsets_without_reparsing() {
    let encoded = gray8_fixture();
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(encoded))],
        BatchDecodeOptions::default(),
    )
    .expect("shared retained-plan preparation");
    let [group] = prepared.groups() else {
        panic!("expected one prepared group")
    };
    let [image] = group.images() else {
        panic!("expected one prepared image")
    };
    assert_eq!(image.preparation_depth(), PreparationDepth::Htj2kOffsetPlan);
    let referenced_plan = image.htj2k_plan().expect("retained HTJ2K plan").geometry();
    let settings = DecodeSettings {
        resolve_palette_indices: true,
        strict: group.options().settings.is_strict(),
        target_resolution: None,
    };

    let cuda = prepare_grayscale_batch(
        &[GrayscaleBatchInput {
            source_index: 0,
            bytes: image.bytes(),
            device_plan: Some(image.plan()),
            referenced_plan: Some(referenced_plan),
            referenced_classic_plan: None,
        }],
        PixelFormat::Gray8,
        settings,
    )
    .expect("CUDA retained-plan preparation");

    assert_eq!(cuda.reports[0].parse_us, 0);
    assert_eq!(cuda.reports[0].plan_us, 0);
    assert_eq!(cuda.plans[0].payload(), [] as [u8; 0]);
    assert!(cuda.payload_parts.iter().any(|part| !part.is_empty()));
}

#[test]
fn grayscale_batch_prepares_roi_and_reduced_requests_in_one_payload_arena() {
    let pixels = (0_u16..16 * 16)
        .map(|value| u8::try_from(value).expect("fixture byte"))
        .collect::<Vec<_>>();
    let encoded = encode_htj2k(
        &pixels,
        16,
        16,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("HTJ2K grayscale geometry fixture");
    let roi_plan = DeviceDecodePlan::for_image(
        (16, 16),
        DeviceDecodeRequest::Region {
            roi: Rect {
                x: 3,
                y: 5,
                w: 7,
                h: 6,
            },
        },
    )
    .expect("ROI plan");
    let reduced_plan = DeviceDecodePlan::for_image(
        (16, 16),
        DeviceDecodeRequest::Scaled {
            scale: Downscale::Half,
        },
    )
    .expect("reduced plan");

    let prepared = prepare_grayscale_batch(
        &[
            GrayscaleBatchInput {
                source_index: 0,
                bytes: &encoded,
                device_plan: Some(roi_plan),
                referenced_plan: None,
                referenced_classic_plan: None,
            },
            GrayscaleBatchInput {
                source_index: 1,
                bytes: &encoded,
                device_plan: Some(reduced_plan),
                referenced_plan: None,
                referenced_classic_plan: None,
            },
        ],
        PixelFormat::Gray8,
        DecodeSettings::strict(),
    )
    .expect("prepare geometry batch");

    assert_eq!(prepared.plans[0].dimensions(), roi_plan.output_dims());
    assert_eq!(prepared.plans[1].dimensions(), reduced_plan.output_dims());
    assert!(prepared.plans.iter().all(|plan| plan.payload().is_empty()));
    assert!(prepared.payload_parts.iter().any(|part| !part.is_empty()));
}

fn gray8_fixture() -> Vec<u8> {
    let pixels = (0_u8..64).collect::<Vec<_>>();
    encode_htj2k(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("HTJ2K grayscale fixture")
}
