// SPDX-License-Identifier: MIT OR Apache-2.0

use core::num::NonZeroUsize;

use j2k_core::PixelFormat;

use super::{retained_workspace_bytes, SharedWorkers};
use crate::batch::{TileBatchOptions, TileDecodeJob};
use crate::J2kDecoder;

const SIDE: u32 = 32;
const TILE_BYTES: usize = (SIDE * SIDE * 3) as usize;

/// Distinct 32x32 RGB HTJ2K tiles with two decomposition levels.
fn rgb8_tiles(count: u8) -> Vec<Vec<u8>> {
    (0..count)
        .map(|tile| {
            let pixels = (0..SIDE * SIDE * 3)
                .map(|index| {
                    u8::try_from((index * 7 + u32::from(tile) * 29) & 0xff).expect("masked")
                })
                .collect::<Vec<_>>();
            j2k_native::encode_htj2k(
                &pixels,
                SIDE,
                SIDE,
                3,
                8,
                false,
                &j2k_native::EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 2,
                    ..j2k_native::EncodeOptions::default()
                },
            )
            .expect("encode RGB8 HTJ2K tile")
        })
        .collect()
}

fn single_tile_reference(inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    inputs
        .iter()
        .map(|input| {
            let mut out = core::iter::repeat_n(0_u8, TILE_BYTES).collect::<Vec<_>>();
            J2kDecoder::new(input)
                .expect("reference decoder")
                .decode_into(&mut out, (SIDE * 3) as usize, PixelFormat::Rgb8)
                .expect("reference decode");
            out
        })
        .collect()
}

fn decode_shared(
    shared: &SharedWorkers,
    inputs: &[Vec<u8>],
    workers: usize,
    retention_cap_bytes: usize,
) -> Vec<Vec<u8>> {
    let mut outputs = inputs
        .iter()
        .map(|_| core::iter::repeat_n(0_u8, TILE_BYTES).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut jobs = inputs
        .iter()
        .zip(outputs.iter_mut())
        .map(|(input, out)| TileDecodeJob {
            input,
            out,
            stride: (SIDE * 3) as usize,
        })
        .collect::<Vec<_>>();
    let outcomes = shared
        .decode(
            &mut jobs,
            PixelFormat::Rgb8,
            TileBatchOptions {
                workers: NonZeroUsize::new(workers),
            },
            retention_cap_bytes,
        )
        .expect("shared-worker decode");
    assert_eq!(outcomes.len(), inputs.len());
    outputs
}

/// Sum of a per-worker counter across the retained workers.
fn worker_total(
    shared: &SharedWorkers,
    counter: fn(j2k_native::DecoderWorkspaceStats) -> u64,
) -> u64 {
    shared
        .workers
        .lock()
        .expect("shared workers")
        .iter()
        .map(|worker| counter(worker.native_workspace_stats()))
        .sum()
}

#[test]
fn shared_workers_keep_their_workspaces_across_calls() {
    let shared = SharedWorkers::new();
    let inputs = rgb8_tiles(4);
    let expected = single_tile_reference(&inputs);

    for call in 0..2 {
        assert_eq!(
            decode_shared(&shared, &inputs, 2, usize::MAX),
            expected,
            "call {call}"
        );
    }

    assert_eq!(shared.workers.lock().expect("shared workers").len(), 2);
    // Two workers decode two tiles per call; every decode after a worker's
    // first reuses that worker's component owners.
    assert_eq!(
        worker_total(
            &shared,
            j2k_native::DecoderWorkspaceStats::component_owner_reuses
        ),
        6
    );
    let retained = retained_workspace_bytes(&shared.workers.lock().expect("shared workers"));
    assert!(retained > 0);
    assert_eq!(shared.retained_bytes(), retained);
}

#[test]
fn busy_shared_workers_fall_back_to_per_call_workers() {
    let shared = SharedWorkers::new();
    let inputs = rgb8_tiles(4);
    let expected = single_tile_reference(&inputs);
    let held = shared.workers.lock().expect("hold shared workers");

    assert_eq!(decode_shared(&shared, &inputs, 2, usize::MAX), expected);
    assert!(held.is_empty(), "a busy cache is neither used nor filled");
    drop(held);
}

#[test]
fn shared_workers_over_the_retention_cap_are_released() {
    let shared = SharedWorkers::new();
    let inputs = rgb8_tiles(4);
    let expected = single_tile_reference(&inputs);

    assert_eq!(decode_shared(&shared, &inputs, 2, 0), expected);
    assert!(shared.workers.lock().expect("shared workers").is_empty());
    assert_eq!(shared.retained_bytes(), 0);
}

#[test]
fn poisoned_shared_workers_are_discarded_and_rebuilt() {
    let shared = SharedWorkers::new();
    let inputs = rgb8_tiles(4);
    let expected = single_tile_reference(&inputs);
    decode_shared(&shared, &inputs, 2, usize::MAX);
    let poisoned = std::panic::catch_unwind(|| {
        let _workers = shared.workers.lock().expect("shared workers");
        panic!("poison the shared workers");
    });
    assert!(poisoned.is_err() && shared.workers.is_poisoned());

    assert_eq!(decode_shared(&shared, &inputs, 2, usize::MAX), expected);
    assert!(!shared.workers.is_poisoned());
    assert_eq!(shared.workers.lock().expect("rebuilt workers").len(), 2);
    assert_eq!(
        worker_total(&shared, j2k_native::DecoderWorkspaceStats::decode_calls),
        4,
        "the poisoned workers were replaced"
    );
}

#[test]
fn monitoring_and_release_do_not_block_on_a_running_batch() {
    let shared = SharedWorkers::new();
    let inputs = rgb8_tiles(4);
    let expected = single_tile_reference(&inputs);
    decode_shared(&shared, &inputs, 2, usize::MAX);
    let retained = shared.retained_bytes();
    assert!(retained > 0);

    // A batch holds the workers: both calls return at once instead of waiting.
    let held = shared.workers.lock().expect("hold shared workers");
    assert_eq!(shared.retained_bytes(), retained);
    shared.release();
    drop(held);

    // The deferred release drops the old workers before the next call uses them.
    assert_eq!(decode_shared(&shared, &inputs, 2, usize::MAX), expected);
    assert_eq!(
        worker_total(&shared, j2k_native::DecoderWorkspaceStats::decode_calls),
        4,
        "the workers retained before the release were dropped"
    );
}

#[test]
fn release_frees_idle_workers_immediately() {
    let shared = SharedWorkers::new();
    decode_shared(&shared, &rgb8_tiles(4), 2, usize::MAX);
    assert!(shared.retained_bytes() > 0);

    shared.release();

    assert_eq!(shared.retained_bytes(), 0);
    assert!(shared.workers.lock().expect("shared workers").is_empty());
}
