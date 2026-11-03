// SPDX-License-Identifier: MIT OR Apache-2.0

use super::PreparedDecodePlan;
use crate::allocation::try_vec_with_capacity;
use crate::entropy::block::skip_block;
use crate::error::{JpegError, Warning};
use crate::internal::bit_reader::{BitReader, BitReaderSnapshot};
use crate::internal::checkpoint::DeviceCheckpoint;
use alloc::vec::Vec;

pub(crate) fn finish_scan(
    br: &mut BitReader<'_>,
    validate_eoi: bool,
) -> Result<Vec<Warning>, JpegError> {
    if !validate_eoi {
        return Ok(Vec::new());
    }

    match br.take_marker() {
        Some(0xD9) => Ok(Vec::new()),
        Some(found) => Err(JpegError::UnexpectedMarker {
            offset: br.position().saturating_sub(2),
            expected: crate::error::MarkerKind::Eoi,
            found,
        }),
        None => {
            let mut warnings = try_vec_with_capacity(1)?;
            warnings.push(Warning::MissingEoi);
            Ok(warnings)
        }
    }
}

fn reader_from_checkpoint<'a>(
    scan_bytes: &'a [u8],
    checkpoint: Option<&DeviceCheckpoint>,
    target_mcu: u32,
) -> (BitReader<'a>, [i32; 4], u32) {
    if let Some(checkpoint) = checkpoint.filter(|checkpoint| checkpoint.mcu_index <= target_mcu) {
        return (
            BitReader::from_snapshot(
                scan_bytes,
                BitReaderSnapshot {
                    pos: checkpoint.scan_offset,
                    acc: checkpoint.bit_accumulator,
                    bits: checkpoint.bits_buffered,
                },
            ),
            checkpoint.prev_dc,
            checkpoint.mcu_index,
        );
    }

    (BitReader::new(scan_bytes), [0; 4], 0)
}

/// Entropy reader positioned at or before an MCU, with the DC predictors and
/// next expected restart marker that hold there. Every sequential route
/// seeks through [`scan_position_before_mcu`] and then skips the rest with
/// [`skip_to_mcu`] or [`skip_to_mcu_with`].
pub(super) struct ScanPosition<'a> {
    pub(super) br: BitReader<'a>,
    pub(super) prev_dc: [i32; 4],
    pub(super) mcu_index: u32,
    pub(super) expected_rst: u8,
}

/// Position a reader at or before `target_mcu`: just after the last restart
/// marker before it when the scan has restart intervals (`restart != 0`),
/// else at `checkpoint` when it does not pass the target, else at the scan
/// start. The caller skips the remaining MCUs.
pub(super) fn scan_position_before_mcu<'a>(
    scan_bytes: &'a [u8],
    restart: u16,
    checkpoint: Option<&DeviceCheckpoint>,
    target_mcu: u32,
) -> ScanPosition<'a> {
    if restart != 0 {
        return restart_seek_for_mcu(scan_bytes, restart, target_mcu).map_or_else(
            || ScanPosition {
                br: BitReader::new(scan_bytes),
                prev_dc: [0; 4],
                mcu_index: 0,
                expected_rst: 0,
            },
            |seek| ScanPosition {
                br: BitReader::new(&scan_bytes[seek.scan_offset..]),
                prev_dc: [0; 4],
                mcu_index: seek.mcu_index,
                expected_rst: seek.expected_rst,
            },
        );
    }
    let (br, prev_dc, mcu_index) = reader_from_checkpoint(scan_bytes, checkpoint, target_mcu);
    ScanPosition {
        br,
        prev_dc,
        mcu_index,
        expected_rst: 0,
    }
}

#[derive(Debug, Clone, Copy)]
struct RestartSeek {
    scan_offset: usize,
    mcu_index: u32,
    expected_rst: u8,
}

fn restart_seek_for_mcu(scan_bytes: &[u8], restart: u16, target_mcu: u32) -> Option<RestartSeek> {
    if restart == 0 {
        return None;
    }
    let restart = u32::from(restart);
    let restart_index = target_mcu / restart;
    if restart_index == 0 {
        return None;
    }
    let marker_ordinal = restart_index - 1;
    let mut seen = 0u32;
    let mut pos = 0usize;
    while pos + 1 < scan_bytes.len() {
        if scan_bytes[pos] != 0xff {
            pos += 1;
            continue;
        }

        let mut marker_pos = pos + 1;
        while marker_pos < scan_bytes.len() && scan_bytes[marker_pos] == 0xff {
            marker_pos += 1;
        }
        if marker_pos >= scan_bytes.len() {
            return None;
        }

        let marker = scan_bytes[marker_pos];
        match marker {
            0x00 => pos = marker_pos + 1,
            0xd0..=0xd7 => {
                if seen == marker_ordinal {
                    return Some(RestartSeek {
                        scan_offset: marker_pos + 1,
                        mcu_index: restart_index * restart,
                        expected_rst: (restart_index & 0x07) as u8,
                    });
                }
                seen += 1;
                pos = marker_pos + 1;
            }
            _ => return None,
        }
    }
    None
}

pub(super) fn skip_mcu<const N: usize>(
    plan: &PreparedDecodePlan,
    br: &mut BitReader<'_>,
    prev_dc: &mut [i32],
    wide_huffman: bool,
) -> Result<(), JpegError> {
    for comp in &plan.components {
        let plane_idx = comp.output_index;
        let dc_table = plan.dc_table_with_entries::<N>(comp)?;
        let ac_table = plan
            .ac_table_with_entries::<N>(comp)?
            .with_wide_lookup(wide_huffman);
        for _ in 0..u32::from(comp.h) * u32::from(comp.v) {
            skip_block(br, dc_table, ac_table, &mut prev_dc[plane_idx])?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct McuSkipTarget {
    pub(super) target_mcu: u32,
    pub(super) total_mcus: u32,
    pub(super) restart: u16,
    pub(super) wide_huffman: bool,
}

pub(super) struct McuSkipState<'a, 'b> {
    pub(super) br: &'a mut BitReader<'b>,
    pub(super) prev_dc: &'a mut [i32],
    pub(super) current_mcu: &'a mut u32,
    pub(super) mcus_since_restart: &'a mut u32,
    pub(super) expected_rst: &'a mut u8,
}

#[derive(Clone, Copy)]
pub(super) struct McuPosition {
    pub(super) current: u32,
    pub(super) total: u32,
}

#[expect(
    clippy::inline_always,
    reason = "the due check runs once per MCU and must inline so callers' local readers stay in registers"
)]
#[inline(always)]
pub(super) fn consume_restart_marker_if_due(
    br: &mut BitReader<'_>,
    restart: u16,
    mcus_since_restart: u32,
    expected_rst: &mut u8,
    position: McuPosition,
) -> Result<bool, JpegError> {
    if restart == 0 || mcus_since_restart != u32::from(restart) {
        return Ok(false);
    }

    let (restarted, next_rst) =
        br.clone()
            .restarted(*expected_rst, position.current, position.total)?;
    *br = restarted;
    *expected_rst = next_rst;
    Ok(true)
}

pub(super) fn skip_to_mcu<const N: usize>(
    plan: &PreparedDecodePlan,
    target: McuSkipTarget,
    state: &mut McuSkipState<'_, '_>,
) -> Result<(), JpegError> {
    skip_to_mcu_with(target, state, |br, prev_dc| {
        skip_mcu::<N>(plan, br, prev_dc, target.wide_huffman)
    })
}

/// [`skip_to_mcu`] with a route's own single-MCU skip, for decoders whose
/// component layout is fixed and whose skip loop is specialized for it.
pub(super) fn skip_to_mcu_with(
    target: McuSkipTarget,
    state: &mut McuSkipState<'_, '_>,
    mut skip_one_mcu: impl FnMut(&mut BitReader<'_>, &mut [i32]) -> Result<(), JpegError>,
) -> Result<(), JpegError> {
    while *state.current_mcu < target.target_mcu {
        if consume_restart_marker_if_due(
            state.br,
            target.restart,
            *state.mcus_since_restart,
            state.expected_rst,
            McuPosition {
                current: *state.current_mcu,
                total: target.total_mcus,
            },
        )? {
            state.prev_dc.fill(0);
            *state.mcus_since_restart = 0;
        }

        skip_one_mcu(state.br, state.prev_dc)?;
        *state.mcus_since_restart += 1;
        *state.current_mcu += 1;
    }
    Ok(())
}
