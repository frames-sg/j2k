// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_core::CodecError;
use j2k_native::{
    HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kDirectIdwtStep, J2kDirectStoreStep, J2kRect,
};

fn one_block_direct_plan(
    cleanup_length: u32,
    refinement_length: u32,
    data: Vec<u8>,
    output_stride: usize,
) -> J2kDirectGrayscalePlan {
    J2kDirectGrayscalePlan {
        dimensions: (1, 1),
        bit_depth: 8,
        steps: vec![
            J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                band_id: 0,
                rect: J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                width: 1,
                height: 1,
                jobs: vec![HtOwnedCodeBlockBatchJob {
                    output_x: 0,
                    output_y: 0,
                    data,
                    cleanup_length,
                    refinement_length,
                    width: 1,
                    height: 1,
                    output_stride,
                    missing_bit_planes: 0,
                    number_of_coding_passes: 1,
                    num_bitplanes: 8,
                    roi_shift: 0,
                    stripe_causal: false,
                    strict: true,
                    dequantization_step: 1.0,
                }],
            }),
            J2kDirectGrayscaleStep::Store(J2kDirectStoreStep {
                input_band_id: 0,
                input_rect: J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                source_x: 0,
                source_y: 0,
                copy_width: 1,
                copy_height: 1,
                output_width: 1,
                output_height: 1,
                output_x: 0,
                output_y: 0,
                addend: 128.0,
            }),
        ],
    }
}

fn one_block_plan(data: Vec<u8>) -> CudaHtj2kDecodePlan {
    let payload_len = u32::try_from(data.len()).expect("fixture payload length");
    let direct = one_block_direct_plan(payload_len, 0, data, 1);
    CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
        .expect("CUDA plan")
}

fn two_block_direct_plan() -> J2kDirectGrayscalePlan {
    J2kDirectGrayscalePlan {
        dimensions: (2, 1),
        bit_depth: 8,
        steps: vec![
            J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                band_id: 0,
                rect: J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 2,
                    y1: 1,
                },
                width: 2,
                height: 1,
                jobs: vec![
                    HtOwnedCodeBlockBatchJob {
                        output_x: 0,
                        output_y: 0,
                        data: vec![1],
                        cleanup_length: 1,
                        refinement_length: 0,
                        width: 1,
                        height: 1,
                        output_stride: 2,
                        missing_bit_planes: 0,
                        number_of_coding_passes: 1,
                        num_bitplanes: 8,
                        roi_shift: 0,
                        stripe_causal: false,
                        strict: true,
                        dequantization_step: 1.0,
                    },
                    HtOwnedCodeBlockBatchJob {
                        output_x: 1,
                        output_y: 0,
                        data: vec![2],
                        cleanup_length: 1,
                        refinement_length: 0,
                        width: 1,
                        height: 1,
                        output_stride: 2,
                        missing_bit_planes: 0,
                        number_of_coding_passes: 1,
                        num_bitplanes: 8,
                        roi_shift: 0,
                        stripe_causal: false,
                        strict: true,
                        dequantization_step: 1.0,
                    },
                ],
            }),
            J2kDirectGrayscaleStep::Store(J2kDirectStoreStep {
                input_band_id: 0,
                input_rect: J2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 2,
                    y1: 1,
                },
                source_x: 0,
                source_y: 0,
                copy_width: 2,
                copy_height: 1,
                output_width: 2,
                output_height: 1,
                output_x: 0,
                output_y: 0,
                addend: 128.0,
            }),
        ],
    }
}

#[test]
fn append_payload_to_shared_offsets_blocks_and_drains_local_payload() {
    let mut first = one_block_plan(vec![1, 2]);
    let mut second = one_block_plan(vec![3, 4, 5]);
    let mut shared = Vec::new();

    first
        .append_payload_to_shared(&mut shared)
        .expect("append first payload");
    second
        .append_payload_to_shared(&mut shared)
        .expect("append second payload");

    assert_eq!(shared, vec![1, 2, 3, 4, 5]);
    assert!(first.payload().is_empty());
    assert!(second.payload().is_empty());
    assert_eq!(first.payload.capacity(), 0);
    assert_eq!(second.payload.capacity(), 0);
    assert_eq!(first.code_blocks()[0].payload_offset, 0);
    assert_eq!(second.code_blocks()[0].payload_offset, 2);
}

#[test]
fn rebase_payload_offsets_preserves_shared_payload_for_larger_batch() {
    let mut plan = one_block_plan(vec![7, 8]);
    let mut shared = Vec::new();
    plan.append_payload_to_shared(&mut shared)
        .expect("append local payload");

    plan.rebase_payload_offsets(4096).expect("rebase payload");

    assert_eq!(shared, vec![7, 8]);
    assert_eq!(plan.code_blocks()[0].payload_offset, 4096);
}

#[test]
fn full_frame_plan_keeps_all_blocks_while_region_plan_prunes() {
    let direct = two_block_direct_plan();
    let full = CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
        .expect("full CUDA plan");
    let mut region_direct = two_block_direct_plan();
    let J2kDirectGrayscaleStep::Store(store) = &mut region_direct.steps[1] else {
        panic!("expected store fixture");
    };
    store.source_x = 1;
    store.copy_width = 1;
    store.output_x = 1;
    let region = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
        &region_direct,
        PixelFormat::Gray8,
        (1, 0),
        (1, 1),
    )
    .expect("region CUDA plan");

    assert_eq!(full.code_blocks().len(), 2);
    assert_eq!(region.code_blocks().len(), 1);
    assert_eq!(region.code_blocks()[0].output_x, 1);
}

#[test]
fn rejects_block_length_mismatch() {
    let direct = one_block_direct_plan(1, 2, vec![0xAA, 0xBB], 1);

    let error =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
            .expect_err("mismatched cleanup/refinement lengths must be rejected");

    assert!(error.is_unsupported());
    assert!(
        error
            .to_string()
            .contains("block lengths do not match payload bytes"),
        "unexpected error: {error}"
    );
}

#[test]
fn rejects_roi_maxshift_jobs() {
    let mut direct = one_block_direct_plan(1, 0, vec![0xAA], 1);
    let J2kDirectGrayscaleStep::HtSubBand(subband) = &mut direct.steps[0] else {
        panic!("fixture starts with one HT sub-band");
    };
    subband.jobs[0].roi_shift = 7;

    let error =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
            .expect_err("ROI maxshift jobs must be rejected");

    assert!(error.is_unsupported());
    assert!(
        error.to_string().contains("ROI maxshift decode"),
        "unexpected error: {error}"
    );
}

#[test]
fn rejects_output_stride_overflow() {
    let direct = one_block_direct_plan(1, 0, vec![0xAA], usize::MAX);

    let error =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
            .expect_err("unrepresentable output stride must be rejected");

    assert!(error.is_unsupported());
}

#[test]
fn rejects_mixed_idwt_transforms() {
    let mut direct = one_block_direct_plan(1, 0, vec![0xAA], 1);
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    direct.steps.insert(
        1,
        J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
            output_band_id: 4,
            rect,
            transform: J2kWaveletTransform::Reversible53,
            ll_band_id: 0,
            ll: rect,
            hl_band_id: 1,
            hl: rect,
            lh_band_id: 2,
            lh: rect,
            hh_band_id: 3,
            hh: rect,
        }),
    );
    direct.steps.insert(
        2,
        J2kDirectGrayscaleStep::Idwt(J2kDirectIdwtStep {
            output_band_id: 8,
            rect,
            transform: J2kWaveletTransform::Irreversible97,
            ll_band_id: 4,
            ll: rect,
            hl_band_id: 5,
            hl: rect,
            lh_band_id: 6,
            lh: rect,
            hh_band_id: 7,
            hh: rect,
        }),
    );

    let error =
        CudaHtj2kDecodePlan::from_grayscale_direct_plan(&direct, PixelFormat::Gray8, (0, 0))
            .expect_err("mixed transforms must be rejected");

    assert!(error.is_unsupported());
    assert!(
        error.to_string().contains("mixed DWT transforms"),
        "unexpected error: {error}"
    );
}

#[test]
fn region_plan_rejects_store_outside_output_rect() {
    let direct = one_block_direct_plan(1, 0, vec![0xAA], 1);

    let error = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
        &direct,
        PixelFormat::Gray8,
        (1, 1),
        (0, 0),
    )
    .expect_err("store outside compact output rectangle must be rejected");

    assert!(error.is_unsupported());
    assert!(
        error
            .to_string()
            .contains("store does not fit the requested output rectangle"),
        "unexpected error: {error}"
    );
}

#[test]
fn referenced_prepared_plan_materializes_only_the_execution_arena() {
    let encoded = std::sync::Arc::<[u8]>::from(
        j2k_native::encode_htj2k(
            &(0_u8..64).collect::<Vec<_>>(),
            8,
            8,
            1,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode referenced-plan fixture"),
    );
    let prepared = j2k::prepare_batch(
        vec![j2k::EncodedImage::full(std::sync::Arc::clone(&encoded))],
        j2k::BatchDecodeOptions::default(),
    )
    .expect("prepare referenced-plan fixture");
    let image = &prepared.groups()[0].images()[0];
    let prepared_plan = image.htj2k_plan().expect("retained HTJ2K plan");
    let referenced = prepared_plan
        .adapter_view()
        .downcast_ref::<j2k_native::J2kReferencedHtj2kPlan>()
        .expect("native referenced HTJ2K plan adapter");
    let tile = &referenced.tiles()[0];
    let geometry = tile
        .grayscale_geometry()
        .expect("grayscale referenced tile geometry");
    let span = tile.payload_records();
    let payload_end = span.end_record().expect("payload span end");
    let tile_payloads = &referenced.payloads()[span.first_record..payload_end];
    let mut expected = Vec::new();
    for payload in referenced.payloads() {
        let cleanup_end = payload.cleanup.end().expect("cleanup range end");
        expected.extend_from_slice(&encoded[payload.cleanup.offset..cleanup_end]);
        if let Some(refinement) = payload.refinement {
            let refinement_end = refinement.end().expect("refinement range end");
            expected.extend_from_slice(&encoded[refinement.offset..refinement_end]);
        }
    }

    let output = image.plan().output_rect();
    let mut shared = Vec::new();
    let mut budget = crate::allocation::HostPhaseBudget::new("referenced CUDA plan test");
    let cuda = CudaHtj2kDecodePlan::from_referenced_tile_grayscale_plan_into_shared(
        geometry,
        tile_payloads,
        &encoded,
        PixelFormat::Gray8,
        (output.x, output.y),
        (output.w, output.h),
        &mut shared,
        &mut budget,
    )
    .expect("flatten referenced CUDA plan");

    assert_eq!(shared, expected);
    assert!(cuda.payload().is_empty());
    assert_eq!(cuda.code_blocks().len(), referenced.payloads().len());
    assert_eq!(cuda.code_blocks()[0].payload_offset, 0);
}
