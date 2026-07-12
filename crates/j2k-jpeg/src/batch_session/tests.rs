// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::decoder::Decoder;
use j2k_test_support::JPEG_BASELINE_420_16X16;

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
