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

/// Position an [`EntropyCursor`] at or before `target_mcu`, and return the
/// MCU it is at; the caller skips the remaining MCUs. Every sequential route
/// seeks through here, inside [`enter_scan_at_mcu`] or
/// [`enter_scan_at_mcu_with`].
///
/// A consumer that has a decoder asks `Decoder::checkpoint_for_mcu` for the
/// resume point of the first MCU it needs and hands it in as `checkpoint`:
/// for restart-coded scans the checkpoint at the start of that MCU's restart
/// interval (from the decoder's cached restart index), otherwise the
/// decoder's row-start or span checkpoint at or before it (from its lazily
/// filled checkpoint grid). Without one, the position comes from a scan for
/// the restart marker, or is the scan start. A resumed cursor yields the same
/// symbols, values, and errors as one that decoded every earlier MCU.
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
) -> (EntropyCursor<'a>, u32) {
    let cursor = |br, prev_dc, expected_rst| EntropyCursor {
        br,
        prev_dc,
        restart: RestartState {
            expected_rst,
            ..RestartState::new(restart)
        },
    };
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
                || (cursor(BitReader::new(scan_bytes), [0; 4], 0), 0),
                |seek| {
                    (
                        cursor(
                            BitReader::new(&scan_bytes[seek.scan_offset..]),
                            [0; 4],
                            seek.expected_rst,
                        ),
                        seek.mcu_index,
                    )
                },
            );
    }
    let (br, prev_dc, mcu_index) = reader_from_checkpoint(scan_bytes, checkpoint, target_mcu);
    (cursor(br, prev_dc, 0), mcu_index)
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
        let mut components = plan
            .resolved_components_with_entries::<N>(wide_huffman)?
            .map(|tables| (tables, 0, 0));
        for ((_, plane, blocks), comp) in components.iter_mut().zip(&plan.components) {
            *plane = comp.output_index;
            *blocks = u32::from(comp.h) * u32::from(comp.v);
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

/// An MCU's index in scan order and the scan's MCU count.
#[derive(Clone, Copy)]
pub(crate) struct McuPosition {
    pub(crate) current: u32,
    pub(crate) total: u32,
}

/// Restart-interval bookkeeping of a scan with MCUs in a fixed order: the
/// sequential routes and the extended 12-bit decoders.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RestartState {
    /// MCUs per restart interval; zero for a scan without restart markers.
    pub(super) interval: u16,
    mcus_since_restart: u32,
    expected_rst: u8,
}

impl RestartState {
    /// Bookkeeping at the start of a scan.
    pub(crate) const fn new(interval: u16) -> Self {
        Self {
            interval,
            mcus_since_restart: 0,
            expected_rst: 0,
        }
    }

    /// Before the MCU at `position`: when a restart interval ends there,
    /// consume its marker and restart the interval count. Returns whether it
    /// did, in which case the caller resets its DC predictors. They stay with
    /// the caller because the fast row kernels keep them in registers; a
    /// slice reset in here measurably defeats that (+1–3 % decode time).
    #[expect(
        clippy::inline_always,
        reason = "the due check runs once per MCU and must inline so callers' local readers stay in registers"
    )]
    #[inline(always)]
    pub(crate) fn begin_mcu(
        &mut self,
        br: &mut BitReader<'_>,
        position: McuPosition,
    ) -> Result<bool, JpegError> {
        if self.interval == 0 || self.mcus_since_restart != u32::from(self.interval) {
            return Ok(false);
        }
        let (restarted, next_rst) =
            br.clone()
                .restarted(self.expected_rst, position.current, position.total)?;
        *br = restarted;
        self.expected_rst = next_rst;
        self.mcus_since_restart = 0;
        Ok(true)
    }

    /// After an MCU, decoded or skipped.
    #[expect(
        clippy::inline_always,
        reason = "runs once per MCU next to the inlined restart check"
    )]
    #[inline(always)]
    pub(crate) fn end_mcu(&mut self) {
        self.mcus_since_restart += 1;
    }
}

/// Entropy state at an MCU boundary: the reader, the DC predictors in output
/// component order, and the restart bookkeeping that hold there.
pub(super) struct EntropyCursor<'a> {
    pub(super) br: BitReader<'a>,
    pub(super) prev_dc: [i32; 4],
    pub(super) restart: RestartState,
}

impl EntropyCursor<'_> {
    /// Skip MCUs `mcus` of a scan of `total_mcus` with `skip_one_mcu`.
    fn skip_mcus(
        &mut self,
        mcus: core::ops::Range<u32>,
        total_mcus: u32,
        mut skip_one_mcu: impl FnMut(&mut BitReader<'_>, &mut [i32; 4]) -> Result<(), JpegError>,
    ) -> Result<(), JpegError> {
        for current in mcus {
            if self.restart.begin_mcu(
                &mut self.br,
                McuPosition {
                    current,
                    total: total_mcus,
                },
            )? {
                self.prev_dc = [0; 4];
            }
            skip_one_mcu(&mut self.br, &mut self.prev_dc)?;
            self.restart.end_mcu();
        }
        Ok(())
    }
}

/// Enter a scan of any component layout at MCU `target.current` of
/// `target.total`: seek through [`scan_position_before_mcu`] (from
/// `checkpoint` when it applies), then skip the MCUs left before the target
/// with an [`McuSkipper`], built only when there is something to skip.
pub(super) fn enter_scan_at_mcu<'a, const N: usize>(
    plan: &PreparedDecodePlan,
    scan_bytes: &'a [u8],
    checkpoint: Option<&DeviceCheckpoint>,
    target: McuPosition,
    wide_huffman: bool,
) -> Result<EntropyCursor<'a>, JpegError> {
    let (mut cursor, first_mcu) = scan_position_before_mcu(
        scan_bytes,
        plan.restart_interval.unwrap_or(0),
        checkpoint,
        target.current,
    );
    if first_mcu < target.current {
        let skipper = McuSkipper::<N>::new(plan, wide_huffman)?;
        cursor.skip_mcus(first_mcu..target.current, target.total, |br, prev_dc| {
            skipper.skip(br, prev_dc)
        })?;
    }
    Ok(cursor)
}

/// [`enter_scan_at_mcu`] with a route's own single-MCU skip, for decoders
/// whose component layout is fixed and whose skip is specialized for it.
pub(super) fn enter_scan_at_mcu_with<'a>(
    plan: &PreparedDecodePlan,
    scan_bytes: &'a [u8],
    checkpoint: Option<&DeviceCheckpoint>,
    target: McuPosition,
    skip_one_mcu: impl FnMut(&mut BitReader<'_>, &mut [i32; 4]) -> Result<(), JpegError>,
) -> Result<EntropyCursor<'a>, JpegError> {
    let (mut cursor, first_mcu) = scan_position_before_mcu(
        scan_bytes,
        plan.restart_interval.unwrap_or(0),
        checkpoint,
        target.current,
    );
    cursor.skip_mcus(first_mcu..target.current, target.total, skip_one_mcu)?;
    Ok(cursor)
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
            let (_, mcu_index) = scan_position_before_mcu(scan, restart, other.as_ref(), target);
            assert_eq!(
                mcu_index,
                scanned.map_or(0, |seek| seek.mcu_index),
                "{target}"
            );
        }
    }
}
