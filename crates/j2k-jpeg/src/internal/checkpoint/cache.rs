// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lazily extended entropy checkpoints used by CPU region decoding.

use alloc::vec::Vec;

use super::allocation::{checked_live_bytes, checkpoint_allocation_bytes};
use super::build::{skip_one_mcu_with_entries, snapshot_checkpoint};
use super::{total_mcus, DeviceCheckpoint};
use crate::allocation::{checked_add_allocation_bytes, checked_allocation_bytes};
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::entropy::sequential::{HuffmanPrimaryWidth, PreparedDecodePlan};
use crate::error::JpegError;
use crate::info::{RestartIndex, RestartSegment};
use crate::internal::bit_reader::{BitReader, BitReaderSnapshot};
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

mod allocation;

pub(super) use self::allocation::reserve_checkpoint_capacity;
#[cfg(test)]
pub(super) use self::allocation::{
    checked_checkpoint_reservation_peak, reconcile_actual_checkpoint_capacity,
};

#[derive(Debug, Default)]
pub(crate) struct CpuCheckpointCache {
    pub(crate) checkpoints: Vec<DeviceCheckpoint>,
    /// The scan's validated restart-marker index, once built, so restart
    /// seeks stop rescanning the entropy data.
    pub(crate) restart_index: Option<RestartIndex>,
    /// Restart seeks answered without an index. The first seek scans only up
    /// to its marker; a decoder that seeks again builds and keeps the index.
    pub(crate) uncached_restart_seeks: u32,
}

impl CpuCheckpointCache {
    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        let checkpoints = checkpoint_allocation_bytes(
            self.checkpoints.capacity(),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let restart_index = self.restart_index.as_ref().map_or(Ok(0), |index| {
            checked_allocation_bytes::<RestartSegment>(index.segments.capacity())
        })?;
        checked_add_allocation_bytes(checkpoints, restart_index)
    }

    /// Keep `index` when it fits beside the decoder baseline and this cache
    /// under the host cap. An index that does not fit is not retained; its
    /// callers then seek by rescanning the entropy data, as before.
    pub(crate) fn retain_restart_index(
        &mut self,
        index: RestartIndex,
        retained_baseline_bytes: usize,
    ) -> Result<(), JpegError> {
        if self.restart_index.is_some() {
            return Ok(());
        }
        let index_bytes = checked_allocation_bytes::<RestartSegment>(index.segments.capacity())?;
        if checked_live_bytes(
            [
                retained_baseline_bytes,
                self.retained_allocation_bytes()?,
                index_bytes,
            ],
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
        .is_ok()
        {
            self.restart_index = Some(index);
        }
        Ok(())
    }
}

pub(crate) fn checkpoint_before_mcu(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
    target_mcu: u32,
    retained_decoder_baseline_bytes: usize,
    cache: &mut CpuCheckpointCache,
) -> Result<Option<DeviceCheckpoint>, JpegError> {
    if plan.restart_interval.is_some() || target_mcu == 0 {
        return Ok(None);
    }

    let total_mcus = total_mcus(plan);
    let target_mcu = target_mcu.min(total_mcus);
    let cadence_mcus = cadence_mcus.max(1);
    let target_checkpoint_mcu = (target_mcu / cadence_mcus) * cadence_mcus;
    if target_checkpoint_mcu == 0 {
        return Ok(None);
    }

    let required_capacity = usize::try_from(target_checkpoint_mcu / cadence_mcus)
        .ok()
        .and_then(|checkpoint_count| checkpoint_count.checked_add(1))
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    reserve_checkpoint_capacity(
        &mut cache.checkpoints,
        required_capacity,
        retained_decoder_baseline_bytes,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )?;

    if cache.checkpoints.is_empty() {
        cache.checkpoints.push(snapshot_checkpoint(
            0,
            &BitReader::new(scan_bytes),
            [0; 4],
            0,
        ));
    }

    let last_mcu = cache
        .checkpoints
        .last()
        .map_or(0, |checkpoint| checkpoint.mcu_index);
    if last_mcu < target_checkpoint_mcu {
        match plan.huffman_primary_width {
            HuffmanPrimaryWidth::Narrow8 => {
                extend_non_restart_checkpoints::<NARROW_PRIMARY_ENTRIES>(
                    plan,
                    scan_bytes,
                    cadence_mcus,
                    target_checkpoint_mcu,
                    cache,
                )?;
            }
            HuffmanPrimaryWidth::Full12 => extend_non_restart_checkpoints::<FULL_PRIMARY_ENTRIES>(
                plan,
                scan_bytes,
                cadence_mcus,
                target_checkpoint_mcu,
                cache,
            )?,
        }
    }

    Ok(cache
        .checkpoints
        .iter()
        .rev()
        .find(|checkpoint| checkpoint.mcu_index > 0 && checkpoint.mcu_index <= target_mcu)
        .copied())
}

fn extend_non_restart_checkpoints<const N: usize>(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    cadence_mcus: u32,
    target_checkpoint_mcu: u32,
    cache: &mut CpuCheckpointCache,
) -> Result<(), JpegError> {
    let start = cache
        .checkpoints
        .last()
        .copied()
        .unwrap_or_else(|| snapshot_checkpoint(0, &BitReader::new(scan_bytes), [0; 4], 0));
    let mut br = BitReader::from_snapshot(
        scan_bytes,
        BitReaderSnapshot {
            pos: start.scan_offset,
            acc: start.bit_accumulator,
            bits: start.bits_buffered,
        },
    );
    let mut prev_dc = start.prev_dc;
    let mut mcu_index = start.mcu_index;

    while mcu_index < target_checkpoint_mcu {
        skip_one_mcu_with_entries::<N>(plan, &mut br, &mut prev_dc)?;
        mcu_index += 1;
        if mcu_index.is_multiple_of(cadence_mcus) {
            cache
                .checkpoints
                .push(snapshot_checkpoint(mcu_index, &br, prev_dc, 0));
        }
    }
    Ok(())
}
