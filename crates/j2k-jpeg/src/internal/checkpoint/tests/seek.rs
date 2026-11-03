// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;

use super::fixtures::noisy_jpeg;
use crate::decoder::Decoder;
use crate::entropy::sequential::{decode_scan_dct_window_into, sequential_mcu_grid, McuWindow};
use crate::internal::checkpoint::{checkpoint_before_mcu, CpuCheckpointCache};
use crate::JpegSubsampling;

/// Coefficient windows entered through `scan_position_before_mcu` at a
/// decoder resume point equal the same windows entered without one (restart
/// marker scan, or the scan start), before and after the caches fill.
#[test]
fn resume_points_decode_windows_exactly_like_the_unindexed_seek() {
    for restart_interval in [None, Some(5), Some(37)] {
        let bytes = noisy_jpeg(300, 180, JpegSubsampling::Ybr420, restart_interval);
        let decoder = Decoder::new(&bytes).expect("decoder");
        let plan = &decoder.plan;
        let scan_bytes = &decoder.bytes[plan.scan_offset..];
        let (mcus_per_row, mcu_rows) = sequential_mcu_grid(plan);
        let blocks_per_mcu: u32 = plan
            .components
            .iter()
            .map(|component| u32::from(component.h) * u32::from(component.v))
            .sum();
        let cache = CpuCheckpointCache::default();
        let mut resumed_mid_scan = false;
        for pass in 0..3 {
            for row in (0..mcu_rows).step_by(3) {
                let col = (row * 5) % mcus_per_row;
                let window = McuWindow {
                    col,
                    cols: (mcus_per_row - col).min(4),
                    row,
                    rows: (mcu_rows - row).min(2),
                };
                let target = row * mcus_per_row;
                // Restart-coded scans: the decoder's lookup (its restart index
                // from the second seek on). Others: the checkpoint grid, which
                // the decoder itself consults only past a minimum depth.
                let checkpoint = if restart_interval.is_some() {
                    decoder.checkpoint_for_mcu(scan_bytes, target, 0)
                } else {
                    checkpoint_before_mcu(plan, scan_bytes, target, 0, &cache)
                }
                .expect("seek");
                if let Some(checkpoint) = &checkpoint {
                    assert!(checkpoint.mcu_index <= target);
                    resumed_mid_scan |= checkpoint.mcu_index > 0;
                }
                let len = (window.cols * window.rows * blocks_per_mcu) as usize;
                let mut resumed = vec![[0i16; 64]; len];
                let mut unindexed = vec![[0i16; 64]; len];
                decode_scan_dct_window_into(
                    plan,
                    scan_bytes,
                    window,
                    checkpoint.as_ref(),
                    &mut resumed,
                )
                .expect("resumed window");
                decode_scan_dct_window_into(plan, scan_bytes, window, None, &mut unindexed)
                    .expect("unindexed window");
                assert_eq!(
                    resumed, unindexed,
                    "{restart_interval:?} pass {pass} row {row}"
                );
            }
        }
        assert!(resumed_mid_scan, "{restart_interval:?}");
    }
}
