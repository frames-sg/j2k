// SPDX-License-Identifier: MIT OR Apache-2.0

//! Row-at-a-time lossless (SOF3) scan decoding, ITU-T T.81 Annex H.
//!
//! Each MCU row decodes in two passes. The entropy pass reads every coded
//! difference in the row, including MCU padding samples that are coded but
//! never displayed. The reconstruction pass then undifferences each component
//! row with a loop specialized for the scan's predictor. As in libjpeg-turbo's
//! `jdlossls.c`:
//! - reconstruction is modulo 2^16, so differences are kept as wrapping `u16`;
//! - the first row of the scan and of every restart interval uses the 1-D
//!   predictor (Ra), seeded with 2^(P-Pt-1);
//! - every later row starts from Rb, then uses the scan's predictor;
//! - restart intervals must cover whole MCU rows.

use alloc::vec::Vec;

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, try_vec_filled, try_vec_with_capacity,
};
use crate::entropy::huffman::DcHuffmanTable;
use crate::entropy::sequential::finish_scan;
use crate::error::{HuffmanFailure, JpegError, Warning};
use crate::info::SofKind;
use crate::internal::bit_reader::BitReader;

#[cfg(test)]
mod tests;

/// One component of a lossless scan, in scan order.
#[derive(Clone, Copy)]
pub(crate) struct LosslessComponentSpec<'t> {
    /// Horizontal sampling factor from the frame header.
    pub(crate) h: u8,
    /// Vertical sampling factor from the frame header.
    pub(crate) v: u8,
    pub(crate) table: DcHuffmanTable<'t>,
}

/// Frame and scan parameters of a lossless scan.
#[derive(Clone, Copy)]
pub(crate) struct LosslessScanSpec {
    pub(crate) dimensions: (u32, u32),
    /// Predictor selection value `Ss`, 1..=7.
    pub(crate) predictor: u8,
    /// Sample precision `P`, 2..=16.
    pub(crate) precision: u8,
    /// Point transform `Pt` (`Al`), below `P`.
    pub(crate) point_transform: u8,
    pub(crate) restart_interval: Option<u16>,
}

struct ComponentRows<'t> {
    table: DcHuffmanTable<'t>,
    mcu_h: usize,
    mcu_v: usize,
    width: usize,
    height: usize,
    /// Coded samples per row: MCUs per row times `mcu_h`.
    coded_width: usize,
    /// `mcu_v` rows of `coded_width` differences.
    diffs: Vec<u16>,
    /// `mcu_v + 1` rows of `width` samples. Row 0 holds the last row of the
    /// previous MCU row; rows `1..=rows_ready` hold the current MCU row.
    samples: Vec<u16>,
    rows_ready: usize,
    first_row: bool,
}

/// Decoder state for one lossless scan; see the module docs.
pub(crate) struct LosslessScan<'a, 't> {
    br: BitReader<'a>,
    components: Vec<ComponentRows<'t>>,
    predictor: u8,
    initial_predictor: u16,
    /// Largest valid sample, or `u16::MAX` when every value is valid.
    sample_limit: u16,
    point_transform: u8,
    mcus_per_row: usize,
    mcu_rows: usize,
    mcu_row: usize,
    /// Restart interval in MCU rows; zero disables restarts.
    restart_rows: usize,
    rows_to_go: usize,
    expected_rst: u8,
}

/// Buffer layout of one scan component.
struct ComponentLayout {
    /// Samples per MCU; 1x1 in a single-component (non-interleaved) scan.
    mcu_h: usize,
    mcu_v: usize,
    /// Reconstructed extent, `ceil(X * h / Hmax)` by `ceil(Y * v / Vmax)`.
    width: usize,
    height: usize,
    coded_width: usize,
    diff_len: usize,
    sample_len: usize,
}

/// MCU grid of a scan: `(MCUs per row, MCU rows)`, plus each component's
/// layout. A single-component scan is non-interleaved, so its MCU is one sample.
fn scan_layout(
    dimensions: (u32, u32),
    sampling: &[(u8, u8)],
) -> Result<((usize, usize), Vec<ComponentLayout>), JpegError> {
    let max_h = usize::from(sampling.iter().map(|s| s.0).max().unwrap_or(0));
    let max_v = usize::from(sampling.iter().map(|s| s.1).max().unwrap_or(0));
    if max_h == 0 || max_v == 0 {
        return Err(JpegError::NotImplemented {
            sof: SofKind::Lossless,
        });
    }
    let (width, height) = (dimensions.0 as usize, dimensions.1 as usize);
    let interleaved = sampling.len() > 1;
    let grid = if interleaved {
        (width.div_ceil(max_h), height.div_ceil(max_v))
    } else {
        (width, height)
    };
    let mut layouts = try_vec_with_capacity(sampling.len())?;
    for &(h, v) in sampling {
        let (h, v) = (usize::from(h), usize::from(v));
        let (mcu_h, mcu_v) = if interleaved { (h, v) } else { (1, 1) };
        let component_width = (width * h).div_ceil(max_h);
        let coded_width = grid.0.checked_mul(mcu_h).ok_or_else(geometry_overflow)?;
        let diff_len = coded_width
            .checked_mul(mcu_v)
            .ok_or_else(geometry_overflow)?;
        let sample_len = component_width
            .checked_mul(mcu_v + 1)
            .ok_or_else(geometry_overflow)?;
        layouts.push(ComponentLayout {
            mcu_h,
            mcu_v,
            width: component_width,
            height: (height * v).div_ceil(max_v),
            coded_width,
            diff_len,
            sample_len,
        });
    }
    Ok((grid, layouts))
}

fn geometry_overflow() -> JpegError {
    JpegError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

/// Bytes [`LosslessScan::new`] allocates for components with this sampling.
pub(crate) fn lossless_scan_allocation_bytes(
    dimensions: (u32, u32),
    sampling: &[(u8, u8)],
) -> Result<usize, JpegError> {
    let (_, layouts) = scan_layout(dimensions, sampling)?;
    let mut total = checked_allocation_bytes::<ComponentRows<'_>>(layouts.len())?;
    for layout in &layouts {
        total =
            checked_add_allocation_bytes(total, checked_allocation_bytes::<u16>(layout.diff_len)?)?;
        total = checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<u16>(layout.sample_len)?,
        )?;
    }
    Ok(total)
}

impl<'a, 't> LosslessScan<'a, 't> {
    /// Prepare to decode the scan whose entropy data starts at `scan_bytes`.
    ///
    /// # Errors
    /// `UnsupportedPredictor` or `UnsupportedBitDepth` for parameters outside
    /// T.81 Annex H, `NotImplemented` for a restart interval that does not
    /// cover whole MCU rows (libjpeg-turbo rejects those too), or an
    /// allocation error.
    pub(crate) fn new(
        scan_bytes: &'a [u8],
        spec: LosslessScanSpec,
        components: &[LosslessComponentSpec<'t>],
    ) -> Result<Self, JpegError> {
        if !(1..=7).contains(&spec.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: spec.predictor,
            });
        }
        if !(2..=16).contains(&spec.precision) || spec.point_transform >= spec.precision {
            return Err(JpegError::UnsupportedBitDepth {
                depth: spec.precision,
            });
        }
        let mut sampling = [(0u8, 0u8); 4];
        let sampling =
            sampling
                .get_mut(..components.len())
                .ok_or(JpegError::UnsupportedComponentCount {
                    count: u8::try_from(components.len()).unwrap_or(u8::MAX),
                })?;
        for (slot, component) in sampling.iter_mut().zip(components) {
            *slot = (component.h, component.v);
        }
        let ((mcus_per_row, mcu_rows), layouts) = scan_layout(spec.dimensions, sampling)?;
        let restart_mcus = usize::from(spec.restart_interval.unwrap_or(0));
        if restart_mcus != 0 && mcus_per_row != 0 && restart_mcus % mcus_per_row != 0 {
            return Err(JpegError::NotImplemented {
                sof: SofKind::Lossless,
            });
        }
        let restart_rows = restart_mcus.checked_div(mcus_per_row).unwrap_or(0);

        let mut rows = try_vec_with_capacity(components.len())?;
        for (component, layout) in components.iter().zip(layouts) {
            rows.push(ComponentRows {
                table: component.table,
                mcu_h: layout.mcu_h,
                mcu_v: layout.mcu_v,
                width: layout.width,
                height: layout.height,
                coded_width: layout.coded_width,
                diffs: try_vec_filled(layout.diff_len, 0)?,
                samples: try_vec_filled(layout.sample_len, 0)?,
                rows_ready: 0,
                first_row: true,
            });
        }

        let significant_bits = spec.precision - spec.point_transform;
        Ok(Self {
            br: BitReader::new(scan_bytes),
            components: rows,
            predictor: spec.predictor,
            initial_predictor: 1 << (significant_bits - 1),
            sample_limit: if significant_bits >= 16 {
                u16::MAX
            } else {
                (1 << significant_bits) - 1
            },
            point_transform: spec.point_transform,
            mcus_per_row,
            mcu_rows,
            mcu_row: 0,
            restart_rows,
            rows_to_go: restart_rows,
            expected_rst: 0,
        })
    }

    /// Point transform the caller shifts samples left by on output.
    pub(crate) fn point_transform(&self) -> u8 {
        self.point_transform
    }

    /// First component row held by the last decoded MCU row.
    pub(crate) fn row_base(&self, component: usize) -> usize {
        let c = &self.components[component];
        self.mcu_row.saturating_sub(1) * c.mcu_v
    }

    /// Rows of `component` the last decoded MCU row reconstructed.
    pub(crate) fn rows_ready(&self, component: usize) -> usize {
        self.components[component].rows_ready
    }

    /// Row `index` (below [`Self::rows_ready`]) of the last decoded MCU row.
    pub(crate) fn row(&self, component: usize, index: usize) -> &[u16] {
        let c = &self.components[component];
        let start = (index + 1) * c.width;
        &c.samples[start..start + c.width]
    }

    /// Decode and reconstruct the next MCU row. Returns `Ok(false)` once every
    /// MCU row has been decoded.
    ///
    /// # Errors
    /// Entropy, restart-marker, or out-of-range sample errors.
    pub(crate) fn decode_mcu_row(&mut self) -> Result<bool, JpegError> {
        if self.mcu_row == self.mcu_rows {
            return Ok(false);
        }
        if self.restart_rows != 0 {
            if self.rows_to_go == 0 {
                self.restart()?;
            }
            self.rows_to_go -= 1;
        }
        // The reader lives in a local for the whole row so its accumulator
        // stays in registers; it is written back even on error.
        let mut br = self.br.clone();
        let decoded = decode_differences(&mut br, &mut self.components, self.mcus_per_row);
        self.br = br;
        decoded?;

        for component in &mut self.components {
            reconstruct_component(
                component,
                self.mcu_row,
                self.predictor,
                self.initial_predictor,
                self.sample_limit,
            )?;
        }
        self.mcu_row += 1;
        Ok(true)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "MCU indices are bounded by validated u32 image dimensions"
    )]
    fn restart(&mut self) -> Result<(), JpegError> {
        let mcu_at = (self.mcu_row * self.mcus_per_row) as u32;
        let mcu_total = (self.mcu_rows * self.mcus_per_row) as u32;
        self.br.reset_at_restart();
        let (br, next_rst) = self
            .br
            .clone()
            .restarted(self.expected_rst, mcu_at, mcu_total)?;
        self.br = br;
        self.expected_rst = next_rst;
        self.rows_to_go = self.restart_rows;
        for component in &mut self.components {
            component.first_row = true;
        }
        Ok(())
    }

    /// Validate the end of the scan once every MCU row is decoded.
    pub(crate) fn finish(mut self) -> Result<Vec<Warning>, JpegError> {
        finish_scan(&mut self.br, true)
    }
}

/// Entropy pass: every difference of one MCU row, in MCU order.
#[expect(
    clippy::inline_always,
    reason = "keeps the caller's local bit reader in registers across the row"
)]
#[inline(always)]
fn decode_differences(
    br: &mut BitReader<'_>,
    components: &mut [ComponentRows<'_>],
    mcus_per_row: usize,
) -> Result<(), JpegError> {
    match components {
        [only] => {
            let table = only.table;
            for diff in &mut only.diffs[..mcus_per_row] {
                *diff = table.decode_lossless_diff(br)?;
            }
            Ok(())
        }
        [c0, c1, c2] if [c0.mcu_h, c0.mcu_v, c1.mcu_h, c1.mcu_v, c2.mcu_h, c2.mcu_v] == [1; 6] => {
            let (t0, t1, t2) = (c0.table, c1.table, c2.table);
            for ((d0, d1), d2) in c0.diffs[..mcus_per_row]
                .iter_mut()
                .zip(&mut c1.diffs[..mcus_per_row])
                .zip(&mut c2.diffs[..mcus_per_row])
            {
                *d0 = t0.decode_lossless_diff(br)?;
                *d1 = t1.decode_lossless_diff(br)?;
                *d2 = t2.decode_lossless_diff(br)?;
            }
            Ok(())
        }
        _ => decode_sampled_differences(br, components, mcus_per_row),
    }
}

#[expect(
    clippy::inline_always,
    reason = "keeps the caller's local bit reader in registers across the row"
)]
#[inline(always)]
fn decode_sampled_differences(
    br: &mut BitReader<'_>,
    components: &mut [ComponentRows<'_>],
    mcus_per_row: usize,
) -> Result<(), JpegError> {
    for mcu_x in 0..mcus_per_row {
        for component in components.iter_mut() {
            let table = component.table;
            for row in 0..component.mcu_v {
                let start = row * component.coded_width + mcu_x * component.mcu_h;
                for diff in &mut component.diffs[start..start + component.mcu_h] {
                    *diff = table.decode_lossless_diff(br)?;
                }
            }
        }
    }
    Ok(())
}

/// Reconstruction pass for one component's rows of an MCU row.
fn reconstruct_component(
    component: &mut ComponentRows<'_>,
    mcu_row: usize,
    predictor: u8,
    initial_predictor: u16,
    sample_limit: u16,
) -> Result<(), JpegError> {
    let width = component.width;
    if mcu_row > 0 {
        // Carry the last row of the previous MCU row into the predictor slot.
        let last = component.rows_ready * width;
        component.samples.copy_within(last..last + width, 0);
    }
    let first = mcu_row * component.mcu_v;
    let rows = component.mcu_v.min(component.height.saturating_sub(first));
    component.rows_ready = rows;
    for row in 0..rows {
        let diff = &component.diffs[row * component.coded_width..][..width];
        let (above, current) = component.samples.split_at_mut((row + 1) * width);
        let above = &above[row * width..];
        let current = &mut current[..width];
        if component.first_row {
            undifference_first_row(diff, current, initial_predictor);
            component.first_row = false;
        } else {
            undifference_row(predictor, diff, above, current);
        }
        if sample_limit != u16::MAX && current.iter().fold(0, |any, &s| any | s) > sample_limit {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::InvalidSymbol,
            });
        }
    }
    Ok(())
}

/// First row of a scan or restart interval: predictor 1, seeded with
/// 2^(P-Pt-1).
fn undifference_first_row(diff: &[u16], out: &mut [u16], initial_predictor: u16) {
    let mut ra = initial_predictor;
    for (sample, &d) in out.iter_mut().zip(diff) {
        ra = ra.wrapping_add(d);
        *sample = ra;
    }
}

/// Any later row: Rb for the first sample, then the scan's predictor.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "predictions are reduced modulo 2^16 exactly like libjpeg-turbo's `& 0xFFFF`"
)]
fn undifference_row(predictor: u8, diff: &[u16], above: &[u16], out: &mut [u16]) {
    let (Some((&d0, diff)), Some((&rb0, _)), Some((first, rest))) = (
        diff.split_first(),
        above.split_first(),
        out.split_first_mut(),
    ) else {
        return;
    };
    *first = rb0.wrapping_add(d0);
    let pairs = above.windows(2).zip(diff);
    let mut ra = *first;
    match predictor {
        1 => {
            for (sample, &d) in rest.iter_mut().zip(diff) {
                ra = ra.wrapping_add(d);
                *sample = ra;
            }
        }
        2 => {
            for (sample, (above, &d)) in rest.iter_mut().zip(pairs) {
                *sample = above[1].wrapping_add(d);
            }
        }
        3 => {
            for (sample, (above, &d)) in rest.iter_mut().zip(pairs) {
                *sample = above[0].wrapping_add(d);
            }
        }
        4 => {
            for (sample, (above, &d)) in rest.iter_mut().zip(pairs) {
                ra = ra
                    .wrapping_add(above[1])
                    .wrapping_sub(above[0])
                    .wrapping_add(d);
                *sample = ra;
            }
        }
        5 => {
            for (sample, (above, &d)) in rest.iter_mut().zip(pairs) {
                let (rb, rc) = (i32::from(above[1]), i32::from(above[0]));
                let prediction = i32::from(ra) + ((rb - rc) >> 1);
                ra = (prediction as u16).wrapping_add(d);
                *sample = ra;
            }
        }
        6 => {
            for (sample, (above, &d)) in rest.iter_mut().zip(pairs) {
                let (rb, rc) = (i32::from(above[1]), i32::from(above[0]));
                let prediction = rb + ((i32::from(ra) - rc) >> 1);
                ra = (prediction as u16).wrapping_add(d);
                *sample = ra;
            }
        }
        _ => {
            for (sample, (above, &d)) in rest.iter_mut().zip(pairs) {
                let prediction = (u32::from(ra) + u32::from(above[1])) >> 1;
                ra = (prediction as u16).wrapping_add(d);
                *sample = ra;
            }
        }
    }
}
