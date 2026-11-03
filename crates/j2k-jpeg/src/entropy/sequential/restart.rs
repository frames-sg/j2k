// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{PreparedDecodePlan, ResolvedPreparedComponentPlan};
use crate::allocation::try_vec_with_capacity;
use crate::entropy::block::skip_block;
use crate::error::{JpegError, Warning};
use crate::info::RestartIndex;
use crate::internal::bit_reader::{BitReader, BitReaderSnapshot};
use crate::internal::checkpoint::DeviceCheckpoint;
use crate::parse::markers::{next_entropy_marker, EntropyMarker};
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
///
/// A consumer that has a decoder asks `Decoder::checkpoint_for_mcu` for the
/// resume point of the first MCU it needs and hands it to
/// [`scan_position_before_mcu`]: for restart-coded scans the checkpoint at
/// the start of that MCU's restart interval (from the decoder's cached
/// restart index), otherwise the decoder's row-start or span checkpoint at
/// or before it (from its lazily filled checkpoint grid). Without one, the
/// position comes from a scan for the restart marker, or is the scan start.
/// A resumed reader yields the same symbols, values, and errors as one that
/// decoded every earlier MCU.
pub(super) struct ScanPosition<'a> {
    pub(super) br: BitReader<'a>,
    pub(super) prev_dc: [i32; 4],
    pub(super) mcu_index: u32,
    pub(super) expected_rst: u8,
}

/// Position a reader at or before `target_mcu`; the caller skips the
/// remaining MCUs.
///
/// Restart-coded scans (`restart != 0`) resume just after the restart marker
/// that opens the interval holding `target_mcu`: at `checkpoint` when it is
/// that interval's restart-boundary checkpoint (from the decoder's cached
/// restart index, see [`restart_checkpoint_from_index`]), else by scanning
/// the entropy data for that marker. Either way the reader starts on the
/// bytes after the marker, so results and error offsets do not depend on
/// which source answered. Other scans resume at `checkpoint` when it does not
/// pass the target, else at the scan start.
pub(super) fn scan_position_before_mcu<'a>(
    scan_bytes: &'a [u8],
    restart: u16,
    checkpoint: Option<&DeviceCheckpoint>,
    target_mcu: u32,
) -> ScanPosition<'a> {
    if restart != 0 {
        let interval_start = target_mcu / u32::from(restart) * u32::from(restart);
        return checkpoint
            .filter(|checkpoint| {
                interval_start > 0
                    && checkpoint.mcu_index == interval_start
                    && checkpoint.bits_buffered == 0
            })
            .map(|checkpoint| RestartSeek {
                scan_offset: checkpoint.scan_offset,
                mcu_index: checkpoint.mcu_index,
                expected_rst: checkpoint.expected_rst,
            })
            .or_else(|| restart_seek_for_mcu(scan_bytes, restart, target_mcu))
            .map_or_else(
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
    // Lenient by design: the first non-RST marker, or a scan ending inside a
    // fill run, means "no seek"; decoding from the scan start then reports
    // the stream's own error.
    while let Some(EntropyMarker::Marker {
        code_pos,
        code: 0xd0..=0xd7,
        ..
    }) = next_entropy_marker(scan_bytes, pos)
    {
        if seen == marker_ordinal {
            return Some(RestartSeek {
                scan_offset: code_pos + 1,
                mcu_index: restart_index * restart,
                expected_rst: (restart_index & 0x07) as u8,
            });
        }
        seen += 1;
        pos = code_pos + 1;
    }
    None
}

/// The restart-boundary checkpoint that [`scan_position_before_mcu`] resumes
/// from for `target_mcu`, answered from a validated [`RestartIndex`] of the
/// same scan (`scan_data_offset` is where `scan_bytes` starts) instead of by
/// rescanning the entropy data: the byte after the marker that opens the
/// interval, with predictors reset and nothing buffered.
pub(crate) fn restart_checkpoint_from_index(
    index: &RestartIndex,
    target_mcu: u32,
) -> Option<DeviceCheckpoint> {
    let restart_index = target_mcu.checked_div(index.interval_mcus)?;
    if restart_index == 0 {
        return None;
    }
    let segment = index.segments.get(usize::try_from(restart_index).ok()?)?;
    Some(DeviceCheckpoint {
        mcu_index: segment.start_mcu,
        scan_offset: segment.entropy_offset.checked_sub(index.scan_data_offset)?,
        bit_accumulator: 0,
        bits_buffered: 0,
        prev_dc: [0; 4],
        expected_rst: (restart_index & 0x07) as u8,
    })
}

pub(crate) fn skip_mcu<const N: usize>(
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

/// [`skip_mcu`] with every component's Huffman tables resolved once, for
/// loops that skip many MCUs. A single MCU skip should stay on [`skip_mcu`]:
/// resolving four components up front costs more than one MCU's lookups.
#[derive(Clone, Copy)]
pub(crate) struct McuSkipper<'a, const N: usize> {
    /// Tables, plane index, and blocks per MCU of each scan component.
    components: [(ResolvedPreparedComponentPlan<'a, N>, usize, u32); 4],
    component_count: usize,
}

impl<'a, const N: usize> McuSkipper<'a, N> {
    pub(crate) fn new(plan: &'a PreparedDecodePlan, wide_huffman: bool) -> Result<Self, JpegError> {
        let first = plan.resolved_component_with_entries::<N>(0)?;
        let mut components = [(first, 0, 0); 4];
        if plan.components.len() > components.len() {
            return Err(JpegError::InternalInvariant {
                reason: "sequential decode supports at most four components",
            });
        }
        for (slot, comp) in components.iter_mut().zip(&plan.components) {
            *slot = (
                plan.resolve_component_with_entries::<N>(comp)?
                    .with_wide_huffman(wide_huffman),
                comp.output_index,
                u32::from(comp.h) * u32::from(comp.v),
            );
        }
        Ok(Self {
            components,
            component_count: plan.components.len(),
        })
    }

    #[inline]
    pub(crate) fn skip(
        &self,
        br: &mut BitReader<'_>,
        prev_dc: &mut [i32],
    ) -> Result<(), JpegError> {
        for &(tables, plane, blocks) in &self.components[..self.component_count] {
            for _ in 0..blocks {
                skip_block(br, tables.dc_table, tables.ac_table, &mut prev_dc[plane])?;
            }
        }
        Ok(())
    }
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
    if *state.current_mcu >= target.target_mcu {
        return Ok(());
    }
    let skipper = McuSkipper::<N>::new(plan, target.wide_huffman)?;
    skip_to_mcu_with(target, state, |br, prev_dc| skipper.skip(br, prev_dc))
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

#[cfg(test)]
mod tests {
    use super::{restart_checkpoint_from_index, restart_seek_for_mcu, scan_position_before_mcu};
    use crate::decoder::Decoder;

    #[test]
    fn indexed_restart_checkpoints_resume_where_the_marker_scan_does() {
        let bytes = j2k_test_support::restart_coded_grayscale_jpeg(40, 24);
        let decoder = Decoder::new(&bytes).expect("decoder");
        let scan = &decoder.bytes[decoder.plan.scan_offset..];
        let restart = decoder.plan.restart_interval.expect("restart-coded");
        let index = decoder
            .restart_index()
            .expect("index")
            .expect("restart-coded");
        for target in 0..=15 {
            let checkpoint = restart_checkpoint_from_index(&index, target);
            let scanned = restart_seek_for_mcu(scan, restart, target);
            assert_eq!(
                checkpoint.map(|checkpoint| (
                    checkpoint.scan_offset,
                    checkpoint.mcu_index,
                    checkpoint.expected_rst
                )),
                scanned.map(|seek| (seek.scan_offset, seek.mcu_index, seek.expected_rst)),
                "{target}"
            );
            // A checkpoint for another interval is ignored: the scan decides.
            let other = restart_checkpoint_from_index(&index, target + 2);
            let position = scan_position_before_mcu(scan, restart, other.as_ref(), target);
            assert_eq!(
                position.mcu_index,
                scanned.map_or(0, |seek| seek.mcu_index),
                "{target}"
            );
        }
    }
}
