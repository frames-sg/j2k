// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_test_support::JPEG_GRAYSCALE_8X8;

#[test]
fn concurrent_reservations_never_spend_retained_or_invalid_tile_bytes() {
    let plans = || {
        [
            TilePlan {
                workspace: Ok(80),
                quota: 0,
            },
            TilePlan {
                workspace: Err(JpegToHtj2kError::Unsupported("invalid header")),
                quota: 0,
            },
            TilePlan {
                workspace: Ok(40),
                quota: 0,
            },
        ]
    };
    let mut budget = HostLiveBudget::with_cap(256);
    budget.add_bytes(56).expect("retained output and metadata");
    let mut shared = plans();
    assert!(assign_wave_quotas(
        &mut shared,
        budget.remaining_bytes().unwrap()
    ));
    assert_eq!(shared.iter().map(|plan| plan.quota).sum::<usize>(), 200);
    assert_eq!(
        (shared[0].quota, shared[1].quota, shared[2].quota),
        (120, 0, 80)
    );
    assert!(shared[1].workspace.is_err());

    let mut exact = plans();
    assert!(assign_wave_quotas(&mut exact, 120));
    assert_eq!((exact[0].quota, exact[2].quota), (80, 40));
    let mut over = plans();
    assert!(!assign_wave_quotas(&mut over, 119));
}

#[test]
fn cpu_batch_reserves_all_fixed_owners_before_starting_tiles() {
    let inputs = [JpegTileBatchInput {
        bytes: JPEG_GRAYSCALE_8X8,
    }; 2];
    let retained_scratch = 127;
    let metadata = 2 * size_of::<TileOutcome>()
        + 2 * size_of::<WaveOutcome>()
        + 2 * size_of::<TilePlan>()
        + retained_scratch;
    let options = JpegToHtj2kOptions::lossless_53();
    assert!(matches!(
        transcode_cpu_tiles(&inputs, &options, retained_scratch, metadata - 1, 2),
        Err(JpegToHtj2kError::MemoryCapExceeded { requested, cap })
            if requested == metadata && cap == metadata - 1
    ));
    let batch = transcode_cpu_tiles(&inputs, &options, retained_scratch, metadata, 2)
        .expect("metadata fits, tile budgets are separate failures");
    assert_eq!(batch.report.failed_tiles, 2);
    assert!(batch
        .tiles
        .iter()
        .all(|tile| matches!(tile, Err(JpegToHtj2kError::MemoryCapExceeded { .. }))));
}

#[test]
fn restricted_cpu_batch_keeps_input_order_and_recovers_quota_failures() {
    let inputs = [
        JpegTileBatchInput {
            bytes: JPEG_GRAYSCALE_8X8,
        },
        JpegTileBatchInput {
            bytes: b"not a jpeg",
        },
        JpegTileBatchInput {
            bytes: JPEG_GRAYSCALE_8X8,
        },
    ];
    let options = JpegToHtj2kOptions::lossless_53();
    let expected = crate::jpeg_to_htj2k(JPEG_GRAYSCALE_8X8, &options).expect("small scalar oracle");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    // Small quotas exercise serial admission and retry without allocating a
    // process-sized image. Larger budgets must preserve the same codestream.
    pool.install(|| {
        let workspace = validate_jpeg_transcode_workspace(JPEG_GRAYSCALE_8X8, &options)
            .unwrap()
            .peak_bytes();
        let metadata =
            3 * (size_of::<TileOutcome>() + size_of::<WaveOutcome>() + size_of::<TilePlan>());
        let invalid_error = validate_jpeg_transcode_workspace(inputs[1].bytes, &options)
            .unwrap_err()
            .to_string();
        let mut recovered_budgets = 0;
        // Sweep the small interval around the encoder's admission boundary;
        // its exact metadata size varies with the target's pointer width.
        for available in (4_096..=262_144).step_by(4_096) {
            let cap = metadata + available;
            let mut plans = [
                TilePlan {
                    workspace: Ok(workspace),
                    quota: 0,
                },
                TilePlan {
                    workspace: Err(JpegToHtj2kError::Unsupported("invalid header")),
                    quota: 0,
                },
                TilePlan {
                    workspace: Ok(workspace),
                    quota: 0,
                },
            ];
            let quota_failed = assign_wave_quotas(&mut plans, cap - metadata)
                && matches!(
                    transcode_cpu_tile(JPEG_GRAYSCALE_8X8, &options, plans[0].quota),
                    Err(JpegToHtj2kError::MemoryCapExceeded { .. })
                );
            let batch = transcode_cpu_tiles(&inputs, &options, 0, cap, 3)
                .expect("fixed metadata fits these budgets");
            let mut previous_failed = false;
            for (index, tile) in batch.tiles.iter().enumerate() {
                if index == 1 {
                    assert_eq!(tile.as_ref().unwrap_err().to_string(), invalid_error);
                    continue;
                }
                match tile {
                    Ok(tile) => {
                        assert!(
                            !previous_failed,
                            "later tile cannot displace the earlier output"
                        );
                        assert_eq!(tile.codestream, expected.codestream);
                    }
                    Err(JpegToHtj2kError::MemoryCapExceeded { .. }) => previous_failed = true,
                    Err(error) => panic!("unexpected quota result: {error}"),
                }
            }
            if quota_failed && batch.report.successful_tiles == 2 {
                recovered_budgets += 1;
            }
        }
        assert!(
            recovered_budgets > 0,
            "exercise a real parallel-quota retry"
        );
    });
}
