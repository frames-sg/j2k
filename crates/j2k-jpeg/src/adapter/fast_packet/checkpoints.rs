// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::allocation::try_vec_with_exact_capacity;
use super::error::FastPacketError;
use super::types::JpegEntropyCheckpointV1;
use crate::decoder::Decoder;
use crate::error::JpegError;
use crate::internal::checkpoint::{
    build_checkpoint_plan_mapped_from_validated_with_live_budget, checkpoint_count_summary,
    DeviceCheckpoint, ValidatedScanBytes,
};
use alloc::vec::Vec;

const MAX_NONRESTART_ENTROPY_CHECKPOINTS: u32 = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FastCheckpointLayout {
    pub(super) cadence_mcus: u32,
    pub(super) checkpoint_count: usize,
}

struct DestuffedEntropyCursor<'a> {
    scan_bytes: &'a [u8],
    source_pos: usize,
    destuffed_pos: usize,
}

impl<'a> DestuffedEntropyCursor<'a> {
    fn new(scan_bytes: &'a [u8]) -> Self {
        Self {
            scan_bytes,
            source_pos: 0,
            destuffed_pos: 0,
        }
    }

    fn offset_at(&mut self, target: usize) -> Result<u32, FastPacketError> {
        if target < self.source_pos || target > self.scan_bytes.len() {
            return Err(FastPacketError::TruncatedEntropy);
        }

        while self.source_pos < target {
            if self.scan_bytes[self.source_pos] != 0xff {
                self.source_pos += 1;
                self.destuffed_pos += 1;
                continue;
            }

            let marker = *self
                .scan_bytes
                .get(self.source_pos + 1)
                .ok_or(FastPacketError::TruncatedEntropy)?;
            let marker_end = self
                .source_pos
                .checked_add(2)
                .ok_or(FastPacketError::TruncatedEntropy)?;
            if marker_end > target {
                return Err(FastPacketError::TruncatedEntropy);
            }
            match marker {
                0x00 => {
                    self.source_pos = marker_end;
                    self.destuffed_pos += 1;
                }
                0xd0..=0xd7 | 0xd9 => self.source_pos = marker_end,
                marker => return Err(FastPacketError::EntropyMarkerUnsupported { marker }),
            }
        }

        u32::try_from(self.destuffed_pos).map_err(|_| FastPacketError::TruncatedEntropy)
    }
}

fn nonrestart_entropy_chunk_mcus(total_mcus: u32) -> u32 {
    total_mcus
        .div_ceil(MAX_NONRESTART_ENTROPY_CHECKPOINTS)
        .max(1)
}

pub(super) fn build_fast_entropy_checkpoints(
    decoder: &Decoder<'_>,
    validated_scan: ValidatedScanBytes<'_>,
    layout: FastCheckpointLayout,
    live_bytes: &mut usize,
    allocation_cap: usize,
) -> Result<Vec<JpegEntropyCheckpointV1>, FastPacketError> {
    let mut entropy_offsets = DestuffedEntropyCursor::new(validated_scan.payload());
    let mut previous_mcu = None;
    let packet_checkpoints = build_checkpoint_plan_mapped_from_validated_with_live_budget(
        &decoder.plan,
        validated_scan,
        layout.cadence_mcus,
        live_bytes,
        allocation_cap,
        |checkpoint| -> Result<JpegEntropyCheckpointV1, FastPacketError> {
            if previous_mcu.is_some_and(|mcu| checkpoint.mcu_index <= mcu) {
                return Err(FastPacketError::TruncatedEntropy);
            }
            let entropy_pos = entropy_offsets.offset_at(checkpoint.scan_offset)?;
            previous_mcu = Some(checkpoint.mcu_index);
            Ok(packet_checkpoint_from_device(&checkpoint, entropy_pos))
        },
    )?;
    if packet_checkpoints.len() != layout.checkpoint_count {
        return Err(FastPacketError::Decode(JpegError::InternalInvariant {
            reason: "packet checkpoint count disagrees with fast-packet allocation plan",
        }));
    }
    Ok(packet_checkpoints)
}

pub(super) fn inspect_fast_entropy_checkpoints(
    decoder: &Decoder<'_>,
    total_mcus: u32,
) -> FastCheckpointLayout {
    let cadence_mcus = nonrestart_entropy_chunk_mcus(total_mcus);
    let restart_interval = decoder
        .plan
        .restart_interval
        .filter(|&interval| interval > 0)
        .map(u32::from);
    let checkpoint_count = checkpoint_count_summary(total_mcus, cadence_mcus, restart_interval);
    FastCheckpointLayout {
        cadence_mcus,
        checkpoint_count,
    }
}

#[cfg(test)]
pub(super) fn packet_checkpoints_from_device(
    device_checkpoints: &[DeviceCheckpoint],
    scan_bytes: &[u8],
    allocation_cap: usize,
) -> Result<Vec<JpegEntropyCheckpointV1>, FastPacketError> {
    let mut live_bytes = 0;
    let mut packet_checkpoints =
        try_vec_with_exact_capacity(device_checkpoints.len(), &mut live_bytes, allocation_cap)?;
    let mut entropy_offsets = DestuffedEntropyCursor::new(scan_bytes);
    let mut previous_mcu = None;
    for checkpoint in device_checkpoints {
        if previous_mcu.is_some_and(|mcu| checkpoint.mcu_index <= mcu) {
            return Err(FastPacketError::TruncatedEntropy);
        }
        let entropy_pos = entropy_offsets.offset_at(checkpoint.scan_offset)?;
        packet_checkpoints.push(packet_checkpoint_from_device(checkpoint, entropy_pos));
        previous_mcu = Some(checkpoint.mcu_index);
    }
    Ok(packet_checkpoints)
}

fn packet_checkpoint_from_device(
    checkpoint: &DeviceCheckpoint,
    entropy_pos: u32,
) -> JpegEntropyCheckpointV1 {
    JpegEntropyCheckpointV1 {
        mcu_index: checkpoint.mcu_index,
        entropy_pos,
        bit_acc: checkpoint.bit_accumulator,
        bit_count: u32::from(checkpoint.bits_buffered),
        y_prev_dc: checkpoint.prev_dc[0],
        cb_prev_dc: checkpoint.prev_dc[1],
        cr_prev_dc: checkpoint.prev_dc[2],
        reserved: 0,
    }
}
