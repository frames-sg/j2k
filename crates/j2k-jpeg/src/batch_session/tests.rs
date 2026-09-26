// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::decoder::Decoder;
use j2k_test_support::JPEG_BASELINE_420_16X16;

#[test]
fn repeated_batches_reuse_each_workers_decode_cache() {
    use j2k_core::CodecContext;
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });
    let mut outputs = [vec![0u8; 16 * 16 * 3], vec![0u8; 16 * 16 * 3]];
    let mut previous_hits = [0; 2];
    for iteration in 0..3 {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: JPEG_BASELINE_420_16X16,
                out,
                stride: 16 * 3,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
            .unwrap();
        assert_eq!(outputs[0], outputs[1]);
        for (slot, previous) in session.workers.iter_mut().zip(&mut previous_hits) {
            let hits = slot
                .get_mut()
                .unwrap()
                .planning_context()
                .cache_stats()
                .hits;
            if iteration != 0 {
                assert!(
                    hits > *previous,
                    "a warm worker must reuse its cached tables"
                );
            }
            *previous = hits;
        }
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
