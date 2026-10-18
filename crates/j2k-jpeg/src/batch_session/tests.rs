// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::decoder::Decoder;
use j2k_test_support::JPEG_BASELINE_420_16X16;

#[test]
fn recent_identical_inputs_share_prepared_decoders_by_pointer_or_content() {
    let mut bytes_a = JPEG_BASELINE_420_16X16.to_vec();
    bytes_a.push(0);
    let mut bytes_b = JPEG_BASELINE_420_16X16.to_vec();
    bytes_b.push(1);
    let bytes_a_clone = bytes_a.clone();
    let mut first = vec![0u8; 16 * 16 * 3];
    let mut second = vec![0u8; 16 * 16 * 3];
    let mut third = vec![0u8; 16 * 16 * 3];
    let mut fourth = vec![0u8; 16 * 16 * 3];
    let mut fifth = vec![0u8; 16 * 16 * 3];
    let mut sixth = vec![0u8; 16 * 16 * 3];
    let jobs = [
        TileDecodeJob {
            input: &bytes_a,
            out: &mut first,
            stride: 16 * 3,
        },
        TileDecodeJob {
            input: &bytes_b,
            out: &mut second,
            stride: 16 * 3,
        },
        TileDecodeJob {
            input: &bytes_a_clone,
            out: &mut third,
            stride: 16 * 3,
        },
        TileDecodeJob {
            input: &bytes_a,
            out: &mut fourth,
            stride: 16 * 3,
        },
        TileDecodeJob {
            input: &bytes_a,
            out: &mut fifth,
            stride: 16 * 3,
        },
        TileDecodeJob {
            input: &bytes_a[..JPEG_BASELINE_420_16X16.len()],
            out: &mut sixth,
            stride: 16 * 3,
        },
    ];
    let mut workers = Vec::new();
    let planned = plan_prepared_regular_jobs(&jobs, 0, &mut workers, |job, ctx, external| {
        prepare_jpeg_tile_decode(
            job.input,
            ctx,
            PixelFormat::Rgb8,
            DecodeOptions::default(),
            external,
        )
    })
    .expect("prepared planning");
    let PreparedRegularPlanning::Prepared(prepared) = planned else {
        panic!("small repeated inputs should retain prepared state");
    };

    assert_eq!(prepared.plans.len(), 6);
    assert_eq!(prepared.prepared.len(), 3);
    assert_eq!(prepared.decoder_indices, [0, 1, 0, 0, 0, 2]);
    let expected_codec_bytes = vec_capacity_bytes(&prepared.prepared).unwrap()
        + prepared
            .prepared
            .iter()
            .map(|prepared| prepared.retained_decoder_bytes)
            .sum::<usize>()
        + prepared
            .planning_context
            .as_ref()
            .expect("first planning context")
            .retained_allocation_bytes();
    assert_eq!(prepared.retained_codec_bytes, expected_codec_bytes);
}

#[test]
fn exact_header_template_preserves_warnings_across_distinct_payloads() {
    let sos = JPEG_BASELINE_420_16X16
        .windows(2)
        .position(|pair| pair == [0xff, 0xda])
        .expect("fixture SOS");
    let app3 = [0xff, 0xe3, 0x00, 0x04, 0xaa, 0xbb];
    let mut first_input = JPEG_BASELINE_420_16X16.to_vec();
    first_input.splice(sos..sos, app3);
    let mut second_input = first_input.clone();
    first_input.push(0);
    second_input.push(1);
    let mut first_output = vec![0u8; 16 * 16 * 3];
    let mut second_output = vec![0u8; 16 * 16 * 3];
    let jobs = [
        TileDecodeJob {
            input: &first_input,
            out: &mut first_output,
            stride: 16 * 3,
        },
        TileDecodeJob {
            input: &second_input,
            out: &mut second_output,
            stride: 16 * 3,
        },
    ];
    let mut workers = Vec::new();
    let planned = plan_prepared_regular_jobs(&jobs, 0, &mut workers, |job, ctx, external| {
        prepare_jpeg_tile_decode(
            job.input,
            ctx,
            PixelFormat::Rgb8,
            DecodeOptions::default(),
            external,
        )
    })
    .expect("prepared warning templates");
    let PreparedRegularPlanning::Prepared(prepared) = planned else {
        panic!("small warning fixtures should retain prepared state");
    };
    let expected = [crate::Warning::UnknownAppMarker {
        marker: 0xe3,
        size: 2,
    }];

    assert_eq!(prepared.prepared.len(), 2);
    for prepared in &prepared.prepared {
        assert_eq!(prepared.decoder.warnings, expected);
    }
}

#[test]
fn exact_header_template_keeps_truncated_tail_errors_input_specific() {
    assert_eq!(
        &JPEG_BASELINE_420_16X16[JPEG_BASELINE_420_16X16.len() - 2..],
        [0xff, 0xd9]
    );
    let missing_eoi = &JPEG_BASELINE_420_16X16[..JPEG_BASELINE_420_16X16.len() - 2];
    let mut first_output = vec![0u8; 16 * 16 * 3];
    let mut second_output = vec![0u8; 16 * 16 * 3];
    let mut jobs = [
        TileDecodeJob {
            input: JPEG_BASELINE_420_16X16,
            out: &mut first_output,
            stride: 16 * 3,
        },
        TileDecodeJob {
            input: missing_eoi,
            out: &mut second_output,
            stride: 16 * 3,
        },
    ];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(1),
    });
    let error = session
        .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
        .expect_err("truncated entropy remains an input-specific decode error");
    let tile = error.tile_error().expect("indexed codec error");

    assert_eq!(tile.index, 1);
    assert!(matches!(
        &tile.source,
        crate::JpegError::HuffmanDecode { .. }
    ));
}

#[test]
fn shared_prepared_decoder_validates_each_jobs_output() {
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(1),
    });
    let mut valid = vec![0x5a; 16 * 16 * 3];
    let mut too_small = [0x5a];
    let mut jobs = [
        TileDecodeJob {
            input: JPEG_BASELINE_420_16X16,
            out: &mut valid,
            stride: 16 * 3,
        },
        TileDecodeJob {
            input: JPEG_BASELINE_420_16X16,
            out: &mut too_small,
            stride: 16 * 3,
        },
    ];
    let error = session
        .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
        .expect_err("second output is too small");

    assert_eq!(error.tile_error().expect("tile error").index, 1);
    assert!(valid.iter().any(|&value| value != 0x5a));
    assert_eq!(too_small, [0x5a]);
}

#[test]
fn repeated_batches_reuse_the_serial_planning_context_decode_cache() {
    use j2k_core::CodecContext;
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(1),
    });
    let mut output = vec![0u8; 16 * 16 * 3];
    let mut expected = None;
    let mut previous_hits = None;
    for sentinel in 0u8..3 {
        output.fill(sentinel);
        let mut jobs = [TileDecodeJob {
            input: JPEG_BASELINE_420_16X16,
            out: &mut output,
            stride: 16 * 3,
        }];
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
            .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(&output, expected);
        } else {
            expected = Some(output.clone());
        }
        // Full-tile preflight is serial and deliberately owns decoder-cache
        // reuse in the first worker context; every worker still retains scratch.
        let stats = session.workers[0]
            .get_mut()
            .unwrap()
            .planning_context()
            .cache_stats();
        if let Some(previous_hits) = previous_hits {
            assert!(
                stats.hits > previous_hits,
                "a later call must reuse the cache"
            );
        } else {
            assert!(
                stats.occupied_slots > 0,
                "the first call must retain its planning cache"
            );
        }
        previous_hits = Some(stats.hits);
    }
}

#[test]
fn one_shot_session_caps_default_workers_for_small_outputs() {
    const JOBS: usize = 64;
    let info = Decoder::inspect(JPEG_BASELINE_420_16X16).expect("fixture inspect");
    let stride = info.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let len = stride * info.dimensions.1 as usize;
    let mut outputs = (0..JOBS).map(|_| vec![0u8; len]).collect::<Vec<_>>();
    let mut session = JpegBatchSession::new_one_shot(TileBatchOptions::default());

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: JPEG_BASELINE_420_16X16,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
            .expect("one-shot session decode")
    };

    let available = available_tile_batch_workers();
    assert_eq!(outcomes.len(), JOBS);
    assert_eq!(
        session.worker_count(),
        available.min(SMALL_OUTPUT_DEFAULT_WORKER_CAP).min(JOBS)
    );
}

#[test]
fn adjacent_inputs_with_the_same_address_but_different_lengths_are_preflighted() {
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });
    let mut first = vec![0x5a; 16 * 16 * 3];
    let mut second = first.clone();
    let mut jobs = [
        TileDecodeJob {
            input: JPEG_BASELINE_420_16X16,
            out: &mut first,
            stride: 16 * 3,
        },
        TileDecodeJob {
            input: &JPEG_BASELINE_420_16X16[..2],
            out: &mut second,
            stride: 16 * 3,
        },
    ];
    let error = session
        .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
        .expect_err("truncated second header");
    assert_eq!(error.tile_error().expect("tile error").index, 1);
    // Every job must pass preflight before any worker can write a destination.
    assert!(first.iter().chain(&second).all(|&value| value == 0x5a));
}
