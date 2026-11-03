// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lazily extended entropy checkpoints used by CPU region decoding.
//!
//! Checkpoints sit on a fixed grid: every MCU-row start plus every
//! [`CheckpointGrid::span_mcus`] MCUs (about [`CHECKPOINT_SPAN_PX`] pixels)
//! within a row. Each is stored compactly as the canonical data position of
//! the next unread bit (see `BitReader::data_position`) and the DC
//! predictors; its MCU index is implied by its grid slot, and a
//! [`DeviceCheckpoint`] is rebuilt from it on demand. One caller at a time
//! extends the published prefix with the entropy-only skip traversal; it
//! publishes each checkpoint as it is reached and never holds the lock
//! across the traversal, so concurrent lookups keep making progress.

use alloc::vec::Vec;
use std::sync::{Mutex, MutexGuard};

use super::allocation::{checked_live_bytes, element_capacity_bytes};
use super::build::snapshot_checkpoint;
use super::DeviceCheckpoint;
use crate::allocation::{checked_add_allocation_bytes, checked_allocation_bytes};
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::entropy::sequential::{
    sequential_mcu_grid, HuffmanPrimaryWidth, McuSkipper, PreparedDecodePlan,
};
use crate::error::JpegError;
use crate::info::{RestartIndex, RestartSegment};
use crate::internal::bit_reader::BitReader;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

mod allocation;

pub(super) use self::allocation::reserve_checkpoint_capacity;
#[cfg(test)]
pub(super) use self::allocation::{
    checked_checkpoint_reservation_peak, reconcile_actual_checkpoint_capacity,
};

/// Approximate output width between span checkpoints within an MCU row.
pub(crate) const CHECKPOINT_SPAN_PX: u32 = 256;

/// Grid of checkpoint slots for one sequential scan: row starts plus spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CheckpointGrid {
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    /// MCUs between checkpoints within a row.
    pub(crate) span_mcus: u32,
    pub(crate) spans_per_row: u32,
}

impl CheckpointGrid {
    pub(crate) fn for_plan(plan: &PreparedDecodePlan) -> Self {
        let (mcus_per_row, mcu_rows) = sequential_mcu_grid(plan);
        let mcu_width_px = 8 * u32::from(plan.sampling.max_h);
        let span_mcus = CHECKPOINT_SPAN_PX.div_ceil(mcu_width_px).max(1);
        Self {
            mcus_per_row,
            mcu_rows,
            span_mcus,
            spans_per_row: mcus_per_row.div_ceil(span_mcus),
        }
    }

    /// Number of checkpoint slots.
    pub(crate) fn len(self) -> usize {
        self.mcu_rows as usize * self.spans_per_row as usize
    }

    /// Slot of the last checkpoint at or before `mcu` (clamped to the grid).
    pub(crate) fn slot_at_or_before(self, mcu: u32) -> usize {
        let mcus_per_row = self.mcus_per_row.max(1);
        let row = mcu / mcus_per_row;
        if row >= self.mcu_rows {
            return self.len().saturating_sub(1);
        }
        let span = (mcu % mcus_per_row) / self.span_mcus;
        row as usize * self.spans_per_row as usize + span as usize
    }

    /// First MCU of `slot`.
    pub(crate) fn mcu_at(self, slot: usize) -> u32 {
        let spans = self.spans_per_row.max(1) as usize;
        let row = u32::try_from(slot / spans).unwrap_or(u32::MAX);
        let span = u32::try_from(slot % spans).unwrap_or(u32::MAX);
        row.saturating_mul(self.mcus_per_row)
            .saturating_add(span.saturating_mul(self.span_mcus))
    }

    /// Host bytes a fully populated cache of this grid retains.
    pub(crate) fn full_cache_bytes(self) -> Result<usize, JpegError> {
        checked_allocation_bytes::<CompactCheckpoint>(self.len())
    }
}

/// One published checkpoint: canonical data position and DC predictors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompactCheckpoint {
    /// `byte << 3 | bit` of the next unread data bit in the scan.
    pub(crate) position: u64,
    pub(crate) prev_dc: [i32; 4],
}

impl CompactCheckpoint {
    const SCAN_START: Self = Self {
        position: 0,
        prev_dc: [0; 4],
    };

    pub(crate) fn at(br: &BitReader<'_>, prev_dc: [i32; 4]) -> Self {
        let (byte, bit) = br.data_position();
        Self {
            position: (byte as u64) << 3 | u64::from(bit),
            prev_dc,
        }
    }

    pub(crate) fn reader(self, scan_bytes: &[u8]) -> Result<BitReader<'_>, JpegError> {
        let byte = usize::try_from(self.position >> 3).ok();
        let bit = (self.position & 7) as u8;
        byte.and_then(|byte| BitReader::from_data_position(scan_bytes, byte, bit))
            .ok_or(JpegError::InternalInvariant {
                reason: "entropy checkpoint does not name a data bit of the scan",
            })
    }

    /// The [`DeviceCheckpoint`] for this checkpoint at `mcu_index`.
    pub(crate) fn device(
        self,
        mcu_index: u32,
        scan_bytes: &[u8],
    ) -> Result<DeviceCheckpoint, JpegError> {
        Ok(snapshot_checkpoint(
            mcu_index,
            &self.reader(scan_bytes)?,
            self.prev_dc,
            0,
        ))
    }
}

#[derive(Debug, Default)]
struct CacheState {
    /// Published checkpoints for grid slots `0..len`, in MCU order.
    checkpoints: Vec<CompactCheckpoint>,
    /// Whether one caller is extending `checkpoints`.
    extending: bool,
    /// The scan's validated restart-marker index, once built, so restart
    /// seeks stop rescanning the entropy data.
    restart_index: Option<RestartIndex>,
    /// Restart seeks answered without an index. The first seek scans only up
    /// to its marker; a decoder that seeks again builds and keeps the index.
    uncached_restart_seeks: u32,
}

impl CacheState {
    fn restart_index_bytes(&self) -> Result<usize, JpegError> {
        self.restart_index.as_ref().map_or(Ok(0), |index| {
            checked_allocation_bytes::<RestartSegment>(index.segments.capacity())
        })
    }

    fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        checked_add_allocation_bytes(
            element_capacity_bytes::<CompactCheckpoint>(
                self.checkpoints.capacity(),
                DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )?,
            self.restart_index_bytes()?,
        )
    }

    fn reserve(
        &mut self,
        capacity: usize,
        retained_baseline_bytes: usize,
        allocation_cap: usize,
    ) -> Result<(), JpegError> {
        let restart_index_bytes = self.restart_index_bytes()?;
        reserve_checkpoint_capacity(
            &mut self.checkpoints,
            capacity,
            checked_add_allocation_bytes(retained_baseline_bytes, restart_index_bytes)?,
            allocation_cap,
        )
    }
}

/// Per-decoder entropy seek cache: lazily published checkpoints and the
/// restart-marker index. Internally synchronized; `Decoder` is `Sync`.
#[derive(Debug, Default)]
pub(crate) struct CpuCheckpointCache {
    state: Mutex<CacheState>,
}

impl CpuCheckpointCache {
    fn lock(&self) -> Result<MutexGuard<'_, CacheState>, JpegError> {
        self.state.lock().map_err(|_| JpegError::InternalInvariant {
            reason: "CPU entropy checkpoint cache mutex poisoned",
        })
    }

    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        self.lock()?.retained_allocation_bytes()
    }

    /// Grow checkpoint capacity to `capacity` under the decoder baseline and
    /// host cap, counting the old and the replacement storage once each.
    #[cfg(test)]
    pub(crate) fn reserve(
        &self,
        capacity: usize,
        retained_baseline_bytes: usize,
        allocation_cap: usize,
    ) -> Result<(), JpegError> {
        self.lock()?
            .reserve(capacity, retained_baseline_bytes, allocation_cap)
    }

    /// Run `seek` against the cached restart index. Without one, record an
    /// uncached seek and return `Err(build)`, where `build` says whether this
    /// is the decoder's second such seek, the one that should build the
    /// index: the first scans only up to its marker, and after a build that
    /// failed or did not fit, seeks keep scanning.
    pub(crate) fn with_restart_index<R>(
        &self,
        seek: impl FnOnce(&RestartIndex) -> R,
    ) -> Result<Result<R, bool>, JpegError> {
        let mut state = self.lock()?;
        if let Some(index) = &state.restart_index {
            return Ok(Ok(seek(index)));
        }
        state.uncached_restart_seeks = state.uncached_restart_seeks.saturating_add(1);
        Ok(Err(state.uncached_restart_seeks == 2))
    }

    /// Keep `index` when it fits beside the decoder baseline and this cache
    /// under the host cap. An index that does not fit is not retained; its
    /// callers then seek by rescanning the entropy data, as before.
    pub(crate) fn retain_restart_index(
        &self,
        index: RestartIndex,
        retained_baseline_bytes: usize,
    ) -> Result<(), JpegError> {
        let index_bytes = checked_allocation_bytes::<RestartSegment>(index.segments.capacity())?;
        let mut state = self.lock()?;
        let fits = state.restart_index.is_none()
            && checked_live_bytes(
                [
                    retained_baseline_bytes,
                    state.retained_allocation_bytes()?,
                    index_bytes,
                ],
                DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )
            .is_ok();
        if fits {
            state.restart_index = Some(index);
        }
        drop(state);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn published(&self) -> Vec<CompactCheckpoint> {
        self.lock().expect("cache lock").checkpoints.clone()
    }

    #[cfg(test)]
    pub(crate) fn has_restart_index(&self) -> bool {
        self.lock().expect("cache lock").restart_index.is_some()
    }
}

/// Clears the extending flag however the extension ends.
struct ExtendingGuard<'a>(&'a CpuCheckpointCache);

impl Drop for ExtendingGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.0.state.lock() {
            state.extending = false;
        }
    }
}

/// The checkpoint at the last grid slot at or before `target_mcu`, extending
/// the published prefix up to it on first use. `None` means "decode from the
/// scan start". Restart-coded scans seek by restart marker instead.
pub(crate) fn checkpoint_before_mcu(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    target_mcu: u32,
    retained_decoder_baseline_bytes: usize,
    cache: &CpuCheckpointCache,
) -> Result<Option<DeviceCheckpoint>, JpegError> {
    if plan.restart_interval.is_some() || target_mcu == 0 {
        return Ok(None);
    }
    let grid = CheckpointGrid::for_plan(plan);
    let target_slot = grid.slot_at_or_before(target_mcu);
    let device = |slot: usize, checkpoint: CompactCheckpoint| {
        let mcu_index = grid.mcu_at(slot);
        if mcu_index == 0 {
            return Ok(None);
        }
        checkpoint.device(mcu_index, scan_bytes).map(Some)
    };

    let (start_slot, start) = {
        let mut state = cache.lock()?;
        if let Some(&found) = state.checkpoints.get(target_slot) {
            return device(target_slot, found);
        }
        if state.extending {
            // Another caller is publishing further checkpoints: resume from
            // the latest one rather than waiting for it.
            let Some(last_slot) = state.checkpoints.len().checked_sub(1) else {
                return Ok(None);
            };
            return device(last_slot, state.checkpoints[last_slot]);
        }
        state.reserve(
            target_slot + 1,
            retained_decoder_baseline_bytes,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        if state.checkpoints.is_empty() {
            state.checkpoints.push(CompactCheckpoint::SCAN_START);
        }
        state.extending = true;
        let last_slot = state.checkpoints.len() - 1;
        (last_slot, state.checkpoints[last_slot])
    };
    let _extending = ExtendingGuard(cache);
    let reached = match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => extend_checkpoints::<NARROW_PRIMARY_ENTRIES>(
            plan,
            scan_bytes,
            grid,
            start_slot,
            start,
            target_slot,
            cache,
        ),
        HuffmanPrimaryWidth::Full12 => extend_checkpoints::<FULL_PRIMARY_ENTRIES>(
            plan,
            scan_bytes,
            grid,
            start_slot,
            start,
            target_slot,
            cache,
        ),
    }?;
    device(target_slot, reached)
}

/// Checkpoints published per lock acquisition while extending.
const PUBLISH_BATCH: usize = 32;

/// Skip from `start` (grid `start_slot`) to `target_slot`, publishing every
/// grid checkpoint on the way. Returns the target checkpoint.
fn extend_checkpoints<const N: usize>(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    grid: CheckpointGrid,
    start_slot: usize,
    start: CompactCheckpoint,
    target_slot: usize,
    cache: &CpuCheckpointCache,
) -> Result<CompactCheckpoint, JpegError> {
    let skipper = McuSkipper::<N>::new(plan, plan.prefers_wide_huffman(scan_bytes.len()))?;
    let mut br = start.reader(scan_bytes)?;
    let mut prev_dc = start.prev_dc;
    let mut mcu_index = grid.mcu_at(start_slot);
    let mut reached = start;
    let mut pending = [CompactCheckpoint::SCAN_START; PUBLISH_BATCH];
    let mut pending_len = 0;
    let mut first_pending_slot = start_slot + 1;
    for slot in start_slot + 1..=target_slot {
        let slot_mcu = grid.mcu_at(slot);
        while mcu_index < slot_mcu {
            skipper.skip(&mut br, &mut prev_dc)?;
            mcu_index += 1;
        }
        reached = CompactCheckpoint::at(&br, prev_dc);
        pending[pending_len] = reached;
        pending_len += 1;
        if pending_len == PUBLISH_BATCH || slot == target_slot {
            let mut state = cache.lock()?;
            if state.checkpoints.len() == first_pending_slot {
                state.checkpoints.extend_from_slice(&pending[..pending_len]);
            }
            drop(state);
            first_pending_slot += pending_len;
            pending_len = 0;
        }
    }
    Ok(reached)
}
