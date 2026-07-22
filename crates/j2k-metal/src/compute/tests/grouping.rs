// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::abi::J2kHtCleanupBatchJob;
use super::super::direct_prepare::prepare_sub_band_groups;
use super::super::direct_roi::retain_ht_jobs_for_required_region;
use super::super::{
    direct_tier1_input_buffer_prepares_for_test, prepare_direct_grayscale_plan,
    prepared_direct_grayscale_plan_compute_encoder_count,
    prepared_repeated_direct_ht_cleanup_dispatch_count,
    reset_direct_tier1_input_buffer_prepares_for_test,
    supports_stacked_direct_component_plane_batch, DirectTier1Mode, PreparedDirectGrayscaleStep,
    PreparedHtExecutionOwner, PreparedHtPayloadSource, PreparedHtSubBand,
};
use j2k_core::PixelFormat;
use j2k_native::{encode_htj2k, DecodeSettings, DecoderContext, EncodeOptions, Image};
use std::sync::Arc;

fn test_ht_job(output_x: u32, output_y: u32, width: u32, height: u32) -> J2kHtCleanupBatchJob {
    J2kHtCleanupBatchJob {
        coded_offset: output_y
            .checked_mul(32)
            .and_then(|base| base.checked_add(output_x))
            .expect("test coded offset"),
        width,
        height,
        coded_len: 1,
        cleanup_length: 1,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 8,
        roi_shift: 0,
        number_of_coding_passes: 1,
        output_stride: 8,
        output_offset: output_y
            .checked_mul(8)
            .and_then(|base| base.checked_add(output_x))
            .expect("test output offset"),
        dequantization_step: 1.0,
        stripe_causal: 0,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "test helper receives small synthetic band identifiers"
)]
fn test_ht_sub_band(band_id: u32) -> PreparedHtSubBand {
    PreparedHtSubBand {
        band_id,
        width: 8,
        height: 8,
        payload_source: PreparedHtPayloadSource::Contiguous(vec![band_id as u8]),
        jobs: vec![test_ht_job(0, 0, 2, 2)],
        execution_owner: Arc::new(PreparedHtExecutionOwner),
    }
}

fn separator_store_step() -> PreparedDirectGrayscaleStep {
    PreparedDirectGrayscaleStep::Store(j2k_native::J2kDirectStoreStep {
        input_band_id: 99,
        input_rect: j2k_native::J2kRect {
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
        addend: 0.0,
    })
}

#[test]
fn direct_sub_band_grouping_groups_adjacent_ht_runs_without_runtime() {
    let steps = vec![
        PreparedDirectGrayscaleStep::HtSubBand(test_ht_sub_band(1)),
        PreparedDirectGrayscaleStep::HtSubBand(test_ht_sub_band(2)),
        separator_store_step(),
        PreparedDirectGrayscaleStep::HtSubBand(test_ht_sub_band(3)),
        PreparedDirectGrayscaleStep::HtSubBand(test_ht_sub_band(4)),
    ];

    let groups = prepare_sub_band_groups(
        &steps,
        DirectTier1Mode::CpuUpload,
        |step| match step {
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => Some(sub_band),
            _ => None,
        },
        |start, end, sub_bands, _| {
            Ok((
                start,
                end,
                sub_bands
                    .iter()
                    .map(|sub_band| sub_band.band_id)
                    .collect::<Vec<_>>(),
            ))
        },
    )
    .expect("group adjacent HT sub-bands");

    assert_eq!(groups, vec![(0, 2, vec![1, 2]), (3, 5, vec![3, 4])]);
}

#[test]
fn direct_roi_prunes_ht_jobs_without_runtime() {
    let mut jobs = vec![
        test_ht_job(0, 0, 2, 2),
        test_ht_job(6, 0, 2, 2),
        test_ht_job(2, 2, 3, 3),
    ];
    let required =
        j2k_native::J2kRequiredBandRegion::new(0, 0, 4, 4).expect("required test region");

    retain_ht_jobs_for_required_region(&mut jobs, Some(required));

    let retained_offsets = jobs.iter().map(|job| job.output_offset).collect::<Vec<_>>();
    assert_eq!(retained_offsets, vec![0, 18]);

    retain_ht_jobs_for_required_region(&mut jobs, None);
    assert!(jobs.is_empty());
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let ht_subband_steps = plan
        .steps
        .iter()
        .filter(|step| matches!(step, j2k_native::J2kDirectGrayscaleStep::HtSubBand(_)))
        .count();
    assert!(
        ht_subband_steps > 1,
        "fixture must exercise multiple HT sub-band cleanup steps"
    );

    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    assert_eq!(
        prepared.ht_groups.len(),
        1,
        "single-tile HTJ2K direct decode should group adjacent HT sub-bands into one cleanup dispatch"
    );
    assert_eq!(prepared.ht_groups[0].members.len(), ht_subband_steps);
    assert!(matches!(
        prepared.steps[prepared.ht_groups[0].start_step],
        PreparedDirectGrayscaleStep::HtSubBand(_)
    ));
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn grouped_ht_direct_plan_uses_one_group_coded_arena() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");

    reset_direct_tier1_input_buffer_prepares_for_test();
    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    assert_eq!(
        prepared.ht_groups.len(),
        1,
        "fixture must exercise one grouped HT dispatch"
    );
    let group = &prepared.ht_groups[0];
    assert!(matches!(
        &group.payload_source,
        PreparedHtPayloadSource::Contiguous(data) if !data.is_empty()
    ));
    assert_eq!(
        direct_tier1_input_buffer_prepares_for_test(),
        0,
        "HT payload and job buffers must be bounded and uploaded only by the chunk submission"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn prepared_ht_direct_plan_encodes_full_decode_in_one_compute_encoder() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");

    assert_eq!(
        prepared_direct_grayscale_plan_compute_encoder_count(&prepared, PixelFormat::Gray8),
        1,
        "prepared single-tile direct decode should keep cleanup, IDWT, and grayscale store in one compute encoder"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn repeated_prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt() {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("direct grayscale plan");
    let ht_subband_steps = plan
        .steps
        .iter()
        .filter(|step| matches!(step, j2k_native::J2kDirectGrayscaleStep::HtSubBand(_)))
        .count();
    assert!(
        ht_subband_steps > 1,
        "fixture must exercise multiple HT sub-band cleanup steps"
    );

    let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
    assert_eq!(
        prepared_repeated_direct_ht_cleanup_dispatch_count(&prepared),
        1,
        "repeated HTJ2K WSI tile batches should group adjacent sub-band cleanups like the single-tile path"
    );
}

#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn distinct_prepared_ht_direct_plans_support_stacked_component_batch() {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let bytes_a = encode_htj2k(&(0..64).collect::<Vec<u8>>(), 8, 8, 1, 8, false, &options)
        .expect("encode first ht gray8");
    let bytes_b = encode_htj2k(
        &(0..64).rev().collect::<Vec<u8>>(),
        8,
        8,
        1,
        8,
        false,
        &options,
    )
    .expect("encode second ht gray8");
    let image_a = Image::new(&bytes_a, &DecodeSettings::default()).expect("first image");
    let image_b = Image::new(&bytes_b, &DecodeSettings::default()).expect("second image");
    let mut context_a = DecoderContext::default();
    let mut context_b = DecoderContext::default();
    let plan_a = image_a
        .build_direct_grayscale_plan_with_context(&mut context_a)
        .expect("first direct plan");
    let plan_b = image_b
        .build_direct_grayscale_plan_with_context(&mut context_b)
        .expect("second direct plan");
    let prepared_a = prepare_direct_grayscale_plan(&plan_a).expect("first prepared plan");
    let prepared_b = prepare_direct_grayscale_plan(&plan_b).expect("second prepared plan");

    assert!(
        supports_stacked_direct_component_plane_batch(&[&prepared_a, &prepared_b]),
        "distinct same-shape HTJ2K grayscale plans should be eligible for one stacked batch graph"
    );
}
