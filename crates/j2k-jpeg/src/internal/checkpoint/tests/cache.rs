// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::fixtures::grayscale_jpeg;
use crate::decoder::Decoder;
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::entropy::sequential::stream_tests::noisy_jpeg;
use crate::entropy::sequential::{skip_mcu, HuffmanPrimaryWidth, PreparedDecodePlan};
use crate::internal::bit_reader::BitReader;
use crate::internal::checkpoint::{
    checkpoint_before_mcu, CheckpointGrid, CompactCheckpoint, CpuCheckpointCache,
};
use crate::JpegSubsampling;

#[test]
fn lazy_non_restart_checkpoint_extends_to_the_grid_slot_before_target() {
    let bytes = grayscale_jpeg(48, 48);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = &decoder.plan;
    let scan_bytes = &decoder.bytes[plan.scan_offset..];
    let grid = CheckpointGrid::for_plan(plan);
    // 48 pixels are 6 MCUs: one span per row, so slots are row starts.
    assert_eq!(grid.spans_per_row, 1);
    let cache = CpuCheckpointCache::default();

    let checkpoint = checkpoint_before_mcu(plan, scan_bytes, 17, 0, &cache)
        .expect("checkpoint lookup")
        .expect("target beyond the first row should produce a checkpoint");
    assert_eq!(checkpoint.mcu_index, 12);
    assert_eq!(cache.published().len(), grid.slot_at_or_before(17) + 1);
}

#[test]
fn checkpoint_grid_places_row_starts_and_spans() {
    let grid = CheckpointGrid {
        mcus_per_row: 40,
        mcu_rows: 3,
        span_mcus: 16,
        spans_per_row: 3,
    };
    assert_eq!(grid.len(), 9);
    assert_eq!(grid.slot_at_or_before(0), 0);
    assert_eq!(grid.slot_at_or_before(15), 0);
    assert_eq!(grid.slot_at_or_before(16), 1);
    assert_eq!(grid.slot_at_or_before(39), 2);
    assert_eq!(grid.slot_at_or_before(40), 3);
    assert_eq!(grid.slot_at_or_before(119), 8);
    assert_eq!(grid.slot_at_or_before(500), 8);
    assert_eq!(
        (0..9).map(|slot| grid.mcu_at(slot)).collect::<Vec<_>>(),
        [0, 16, 32, 40, 56, 72, 80, 96, 112]
    );
}

fn reference_checkpoints(plan: &PreparedDecodePlan, scan_bytes: &[u8]) -> Vec<CompactCheckpoint> {
    fn walk<const N: usize>(
        plan: &PreparedDecodePlan,
        scan_bytes: &[u8],
    ) -> Vec<CompactCheckpoint> {
        let grid = CheckpointGrid::for_plan(plan);
        let wide = plan.prefers_wide_huffman(scan_bytes.len());
        let mut br = BitReader::new(scan_bytes);
        let mut prev_dc = [0i32; 4];
        let mut mcu = 0;
        let mut out = Vec::new();
        for slot in 0..grid.len() {
            while mcu < grid.mcu_at(slot) {
                skip_mcu::<N>(plan, &mut br, &mut prev_dc, wide).expect("reference skip");
                mcu += 1;
            }
            out.push(CompactCheckpoint::at(&br, prev_dc));
        }
        out
    }
    match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => walk::<NARROW_PRIMARY_ENTRIES>(plan, scan_bytes),
        HuffmanPrimaryWidth::Full12 => walk::<FULL_PRIMARY_ENTRIES>(plan, scan_bytes),
    }
}

#[test]
fn published_checkpoints_match_a_direct_skip_at_every_grid_slot() {
    for (width, height, subsampling) in [
        (300, 203, JpegSubsampling::Ybr420),
        (257, 130, JpegSubsampling::Ybr444),
        (523, 61, JpegSubsampling::Ybr422),
    ] {
        let bytes = noisy_jpeg(width, height, subsampling, None, width ^ height);
        let decoder = Decoder::new(&bytes).expect("decoder");
        let plan = &decoder.plan;
        let scan_bytes = &decoder.bytes[plan.scan_offset..];
        let grid = CheckpointGrid::for_plan(plan);
        assert!(
            grid.spans_per_row > 1,
            "{width}x{height} must exercise spans"
        );
        let reference = reference_checkpoints(plan, scan_bytes);
        let cache = CpuCheckpointCache::default();
        // Extend in uneven steps, then look every slot up again.
        for target in [grid.mcu_at(3) + 1, grid.mcu_at(grid.len() / 2), u32::MAX] {
            checkpoint_before_mcu(plan, scan_bytes, target, 0, &cache).expect("extend");
        }
        assert_eq!(cache.published(), reference, "{width}x{height}");
        for (slot, expected) in reference.iter().enumerate().skip(1) {
            let mcu = grid.mcu_at(slot);
            let found = checkpoint_before_mcu(plan, scan_bytes, mcu, 0, &cache)
                .expect("lookup")
                .expect("non-zero slot");
            assert_eq!(found, expected.device(mcu, scan_bytes).expect("device"));
            assert_eq!(found.mcu_index, mcu);
        }
    }
}

#[test]
fn concurrent_lookups_publish_one_consistent_prefix() {
    let bytes = noisy_jpeg(640, 480, JpegSubsampling::Ybr420, None, 864);
    let decoder = Decoder::new(&bytes).expect("decoder");
    let plan = &decoder.plan;
    let scan_bytes = &decoder.bytes[plan.scan_offset..];
    let grid = CheckpointGrid::for_plan(plan);
    let reference = reference_checkpoints(plan, scan_bytes);
    let cache = CpuCheckpointCache::default();
    std::thread::scope(|scope| {
        for worker in 0..8u32 {
            let cache = &cache;
            let reference = &reference;
            scope.spawn(move || {
                for step in 0..20u32 {
                    let target =
                        (worker * 7919 + step * 104_729) % (grid.mcu_at(grid.len() - 1) + 1);
                    let found = checkpoint_before_mcu(plan, scan_bytes, target, 0, cache)
                        .expect("concurrent lookup");
                    let Some(found) = found else { continue };
                    let slot = grid.slot_at_or_before(found.mcu_index);
                    assert_eq!(grid.mcu_at(slot), found.mcu_index);
                    assert!(found.mcu_index <= target);
                    assert_eq!(
                        found,
                        reference[slot]
                            .device(found.mcu_index, scan_bytes)
                            .expect("device")
                    );
                }
            });
        }
    });
    let published = cache.published();
    assert_eq!(published[..], reference[..published.len()]);
}
