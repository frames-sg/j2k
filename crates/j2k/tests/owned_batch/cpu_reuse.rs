// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, Downscale, EncodedImage, PreparationDepth, Rect,
};
use j2k_native::{encode, EncodeOptions};

use super::fixtures::{
    classic_gray8_fixture, classic_gray8_fixture_with_tile_size, htj2k_gray8_fixture,
};
use super::oracles::{decoded_samples_for_source, native_request_oracle};
use super::payload_plan::native_prepared_classic_plan;

#[test]
fn cpu_fast_group_uses_one_flattened_arena_and_one_direct_output_allocation() {
    let options = BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    };
    let encoded = Arc::<[u8]>::from(htj2k_gray8_fixture(16, 16));
    let prepared = prepare_batch(
        vec![
            EncodedImage::full(Arc::clone(&encoded)),
            EncodedImage::full(encoded),
        ],
        options,
    )
    .expect("prepare repeated fast group");
    let payloads_per_image = prepared.groups()[0].images()[0]
        .htj2k_plan()
        .expect("HT offset plan")
        .payload_count();
    let mut session = CpuBatchDecoder::new(options);

    let first = session
        .decode_prepared(&prepared)
        .expect("first flattened decode");
    let first_stats = session.workspace_stats();
    let second = session
        .decode_prepared(&prepared)
        .expect("second flattened decode");
    let second_stats = session.workspace_stats();

    assert!(
        first.errors().is_empty(),
        "flattened decode errors: {:?}",
        first.errors()
    );
    assert_eq!(first.groups(), second.groups());
    assert_eq!(first_stats.flattened_group_plans(), 1);
    assert_eq!(
        first_stats.flattened_payload_jobs(),
        2 * payloads_per_image as u64
    );
    assert_eq!(
        first_stats.entropy_job_dispatches(),
        2 * payloads_per_image as u64,
        "every flattened code block must be dispatched as scheduler work"
    );
    assert_eq!(
        first_stats.cross_image_entropy_windows(),
        1,
        "the two images must share one entropy-dispatch window"
    );
    assert_eq!(first_stats.output_group_allocations(), 1);
    assert_eq!(first_stats.output_compaction_copied_samples(), 0);
    assert!(first_stats.retained_compressed_arena_bytes() > 0);
    assert_eq!(second_stats.flattened_group_plans(), 2);
    assert_eq!(
        second_stats.entropy_job_dispatches(),
        4 * payloads_per_image as u64
    );
    assert_eq!(second_stats.cross_image_entropy_windows(), 2);
    assert_eq!(second_stats.output_group_allocations(), 2);
    assert_eq!(second_stats.output_compaction_copied_samples(), 0);
    assert_eq!(second_stats.compressed_arena_reuses(), 1);
    assert_eq!(
        second_stats.retained_compressed_arena_bytes(),
        first_stats.retained_compressed_arena_bytes()
    );
}

#[test]
fn cpu_prepared_ht_plan_decode_avoids_native_reparse() {
    let options = BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(htj2k_gray8_fixture(16, 16)))],
        options,
    )
    .expect("prepare parse-free CPU batch");
    assert_eq!(
        prepared.groups()[0].images()[0].preparation_depth(),
        PreparationDepth::Htj2kOffsetPlan
    );
    let mut session = CpuBatchDecoder::new(options);

    let first = session
        .decode_prepared(&prepared)
        .expect("first parse-free prepared decode");
    let first_stats = session.workspace_stats();
    let second = session
        .decode_prepared(&prepared)
        .expect("second parse-free prepared decode");
    let second_stats = session.workspace_stats();

    assert!(first.errors().is_empty());
    assert_eq!(first.groups(), second.groups());
    assert_eq!(first_stats.prepared_plan_decode_calls(), 1);
    assert_eq!(second_stats.prepared_plan_decode_calls(), 2);
    assert_eq!(first_stats.decode_calls(), 0);
    assert_eq!(second_stats.decode_calls(), 0);
    assert!(first_stats.retained_prepared_plan_ht_workspace_bytes() > 0);
    assert_eq!(
        second_stats.retained_prepared_plan_ht_workspace_bytes(),
        first_stats.retained_prepared_plan_ht_workspace_bytes()
    );
}

#[test]
fn cpu_prepared_classic_plan_decode_avoids_native_reparse() {
    let options = BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(classic_gray8_fixture(16, 16)))],
        options,
    )
    .expect("prepare parse-free classic CPU batch");
    let image = &prepared.groups()[0].images()[0];
    assert_eq!(
        image.preparation_depth(),
        PreparationDepth::ClassicOffsetPlan
    );
    assert!(image
        .classic_plan()
        .is_some_and(j2k::PreparedClassicPlan::is_grayscale));
    let mut session = CpuBatchDecoder::new(options);

    let first = session
        .decode_prepared(&prepared)
        .expect("first parse-free classic prepared decode");
    let first_stats = session.workspace_stats();
    let second = session
        .decode_prepared(&prepared)
        .expect("second parse-free classic prepared decode");
    let second_stats = session.workspace_stats();

    assert!(
        first.errors().is_empty(),
        "classic decode errors: {:?}",
        first.errors()
    );
    assert_eq!(first.groups(), second.groups());
    assert_eq!(first_stats.prepared_plan_decode_calls(), 1);
    assert_eq!(second_stats.prepared_plan_decode_calls(), 2);
    assert_eq!(first_stats.decode_calls(), 0);
    assert_eq!(second_stats.decode_calls(), 0);
    assert!(first_stats.retained_prepared_plan_classic_workspace_bytes() > 0);
    assert_eq!(
        second_stats.retained_prepared_plan_classic_workspace_bytes(),
        first_stats.retained_prepared_plan_classic_workspace_bytes()
    );
}

#[test]
fn prepared_classic_plan_supports_duplicate_full_roi_and_reduced_requests() {
    let encoded = Arc::<[u8]>::from(classic_gray8_fixture(16, 16));
    let roi = Rect {
        x: 3,
        y: 2,
        w: 8,
        h: 6,
    };
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Full,
        DecodeRequest::Region { roi },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi,
            scale: Downscale::Half,
        },
    ];

    for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
        let options = BatchDecodeOptions {
            workers: NonZeroUsize::new(1),
            layout,
            ..BatchDecodeOptions::default()
        };
        let prepared = prepare_batch(
            requests
                .into_iter()
                .map(|request| EncodedImage::new(Arc::clone(&encoded), request))
                .collect(),
            options,
        )
        .expect("prepare classic request matrix");
        assert!(prepared.errors().is_empty());
        assert!(prepared
            .groups()
            .iter()
            .all(|group| group.images().iter().all(|image| {
                image.preparation_depth() == PreparationDepth::ClassicOffsetPlan
                    && image.classic_plan().is_some()
                    && Arc::ptr_eq(image.bytes(), &encoded)
            })));

        let mut decoder = CpuBatchDecoder::new(options);
        let first = decoder
            .decode_prepared(&prepared)
            .expect("first classic request matrix decode");
        let second = decoder
            .decode_prepared(&prepared)
            .expect("second classic request matrix decode");
        assert!(first.errors().is_empty());
        assert_eq!(first.groups(), second.groups());
        assert_eq!(decoder.workspace_stats().decode_calls(), 0);
        assert_eq!(decoder.workspace_stats().prepared_plan_decode_calls(), 10);

        for source_index in 0..requests.len() {
            let prepared_image = prepared
                .groups()
                .iter()
                .flat_map(j2k::PreparedBatchGroup::images)
                .find(|image| image.source_index() == source_index)
                .expect("prepared classic request source");
            assert_eq!(
                decoded_samples_for_source(&first, source_index),
                native_request_oracle(prepared_image, layout),
                "classic {layout:?} source={source_index}"
            );
        }
    }
}

#[test]
fn referenced_classic_whole_plan_executes_all_gray_and_rgb_tiles_exactly() {
    const WIDTH: u32 = 19;
    const HEIGHT: u32 = 13;
    let gray = Arc::<[u8]>::from(classic_gray8_fixture_with_tile_size(
        WIDTH,
        HEIGHT,
        Some((11, 7)),
    ));
    let rgb_samples = (0..WIDTH * HEIGHT * 3)
        .map(|index| ((index * 37 + index / 7) & 0xff) as u8)
        .collect::<Vec<_>>();
    let rgb = Arc::<[u8]>::from(
        encode(
            &rgb_samples,
            WIDTH,
            HEIGHT,
            3,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                tile_size: Some((11, 7)),
                use_mct: false,
                ..EncodeOptions::default()
            },
        )
        .expect("encode multi-tile classic RGB8"),
    );

    for (encoded, component_count) in [(gray, 1usize), (rgb, 3)] {
        let prepared = prepare_batch(
            vec![EncodedImage::full(Arc::clone(&encoded))],
            BatchDecodeOptions::default(),
        )
        .expect("prepare multi-tile classic whole-plan fixture");
        assert!(prepared.errors().is_empty());
        let image = &prepared.groups()[0].images()[0];
        let oracle = native_request_oracle(image, BatchLayout::Nhwc);
        let CpuBatchSamples::U8(oracle) = oracle else {
            panic!("classic Gray/RGB8 oracle must contain u8 samples")
        };
        let plan =
            native_prepared_classic_plan(image.classic_plan().expect("referenced classic plan"));
        assert!(plan.tiles().len() > 1);
        let mut scratch = j2k_native::J2kDirectCpuScratch::new();
        let decoded =
            j2k_native::execute_referenced_classic_plan(plan, &encoded, false, &mut scratch)
                .expect("execute every referenced classic tile");
        assert_eq!(decoded.dimensions(), (WIDTH, HEIGHT));
        assert_eq!(decoded.component_count(), component_count);
        for pixel in 0..(WIDTH * HEIGHT) as usize {
            for component in 0..component_count {
                let expected = if component_count == 1 {
                    oracle[pixel]
                } else {
                    oracle[pixel * component_count + component]
                };
                let actual = decoded.plane(component).expect("decoded plane").samples()[pixel];
                assert_eq!(
                    actual.round().clamp(0.0, 255.0).to_bits(),
                    f32::from(expected).to_bits(),
                    "pixel {pixel}, component {component}"
                );
            }
        }
    }
}

#[test]
fn cpu_session_retained_workspace_stabilizes_during_thousand_batch_soak() {
    let options = BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(
            classic_gray8_fixture_with_tile_size(16, 16, Some((8, 8))),
        ))],
        options,
    )
    .expect("prepare CPU soak batch");
    assert_eq!(
        prepared.groups()[0].images()[0].preparation_depth(),
        PreparationDepth::ClassicOffsetPlan
    );
    let mut session = CpuBatchDecoder::new(options);

    session
        .decode_prepared(&prepared)
        .expect("warm CPU workspace");
    let warmed = session.workspace_stats();
    for _ in 0..1_000 {
        let decoded = session
            .decode_prepared(&prepared)
            .expect("decode CPU soak batch");
        assert!(decoded.errors().is_empty());
    }
    let soaked = session.workspace_stats();

    assert_eq!(warmed.prepared_plan_decode_calls(), 1);
    assert_eq!(soaked.prepared_plan_decode_calls(), 1_001);
    assert_eq!(warmed.decode_calls(), 0);
    assert_eq!(soaked.decode_calls(), 0);
    assert_eq!(warmed.flattened_group_plans(), 1);
    assert_eq!(soaked.flattened_group_plans(), 1_001);
    assert!(warmed.retained_prepared_plan_classic_workspace_bytes() > 0);
    assert_eq!(soaked.scratch_capacity_retries(), 0);
    assert_eq!(
        soaked.retained_prepared_plan_classic_workspace_bytes(),
        warmed.retained_prepared_plan_classic_workspace_bytes()
    );
}
