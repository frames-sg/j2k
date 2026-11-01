// SPDX-License-Identifier: MIT OR Apache-2.0

//! Row-streamed region IDWT.
//!
//! [`super::roi`] reconstructs each level's window as a whole buffer. This
//! computes the same windows with the same kernels, so the output is
//! identical, but holds each level as a ring of rows. A level produces its
//! window's rows top to bottom, pulling the LL rows it needs from the level
//! below, and the caller receives the last level's rows one at a time.
//!
//! Vertical lifting is streamed with the schedule of the banded IDWT (see
//! `banded::lift_arrivals`): when a level's row `n` arrives, step `k`
//! (1-based) runs on row `n - k` if that row has the step's parity. Each
//! step then reads its row's neighbours exactly as the previous step left
//! them, so every read sees the value the whole-window filter would see, and
//! a row is final as soon as the inputs its last step needs have arrived.

use alloc::vec::Vec;

use super::super::codestream::WaveletTransform;
use super::super::decode::DecompositionStorage;
use super::super::rect::IntRect;
use super::banded::{
    horizontal_scratch_elements, lift_ring_arrivals, synthesize_band_row, Lifting, RowGeometry,
    MIN_SIMD_ROW_WIDTH,
};
use super::horizontal::filter_row;
use super::model::CoefficientSource;
use super::roi::interleave_samples_roi;
use super::vertical::{irreversible_row_scale, scale_one_row, vertical_step_count, vertical_steps};
use crate::error::{DecodingError, ValidationError};
use crate::Result;

/// Rows each level keeps: the rows an arrival's steps read, one per lifting
/// step above the arrival, its own row and the row above the deepest step's
/// (see `banded::lift_arrivals`).
pub(crate) const fn ring_rows(transform: WaveletTransform) -> usize {
    vertical_step_count(transform) + 2
}

/// Samples the rings of one component's levels hold together, with the
/// scratch their rows' horizontal synthesis shares.
pub(crate) fn ring_samples(
    windows: impl IntoIterator<Item = IntRect>,
    transform: WaveletTransform,
) -> Result<usize> {
    let mut widest = 0;
    let rings = windows.into_iter().try_fold(0usize, |total, window| {
        widest = widest.max(window.width() as usize);
        level_ring_samples(window, transform).and_then(|samples| {
            total
                .checked_add(samples)
                .ok_or(ValidationError::ImageTooLarge.into())
        })
    })?;
    rings
        .checked_add(synthesis_scratch_samples(widest)?)
        .ok_or(ValidationError::ImageTooLarge.into())
}

/// Samples the ring of a level with this window holds.
fn level_ring_samples(window: IntRect, transform: WaveletTransform) -> Result<usize> {
    (window.width() as usize)
        .checked_mul(ring_rows(transform))
        .ok_or(ValidationError::ImageTooLarge.into())
}

/// Scratch for the horizontal synthesis of a row of the widest level.
fn synthesis_scratch_samples(widest: usize) -> Result<usize> {
    horizontal_scratch_elements(widest)
}

/// Where the streamed IDWT reads the rows of a component's stored sub-bands.
pub(crate) trait SubBandRows {
    /// Make row `y` of `sub_band`, in absolute sub-band coordinates,
    /// readable through [`Self::samples`]. Each sub-band's rows are prepared
    /// in increasing order.
    fn prepare_row(&mut self, sub_band: usize, y: u32) -> Result<()>;

    /// The samples of `sub_band` that hold its last prepared row: row-major
    /// samples, the rectangle they cover and their row stride. Rows outside
    /// the rectangle read as zero.
    fn samples(&self, sub_band: usize) -> (&[f32], IntRect, u32);
}

/// Reads rows straight from sub-bands whose stored rectangles are decoded.
pub(crate) struct StoredRows<'s, 'a> {
    pub(crate) storage: &'s DecompositionStorage<'a>,
}

impl SubBandRows for StoredRows<'_, '_> {
    fn prepare_row(&mut self, _sub_band: usize, _y: u32) -> Result<()> {
        Ok(())
    }

    fn samples(&self, sub_band: usize) -> (&[f32], IntRect, u32) {
        let sub_band = &self.storage.sub_bands[sub_band];
        (
            &self.storage.coefficients[sub_band.coefficients.clone()],
            sub_band.storage_rect,
            sub_band.storage_rect.width(),
        )
    }
}

fn source(rows: &dyn SubBandRows, sub_band: usize) -> CoefficientSource<'_> {
    let (coefficients, rect, stride) = rows.samples(sub_band);
    CoefficientSource::new(coefficients, rect, stride)
}

/// One level of the streamed reconstruction.
struct LevelRows<'r> {
    window: IntRect,
    /// The level's sub-bands, in the order of [`super::super::build::Decomposition`].
    sub_bands: [usize; 3],
    ring: &'r mut [f32],
    /// Rows interleaved, filtered horizontally and scaled.
    fed: usize,
    /// Arrivals whose vertical steps have run: one per fed row, then the
    /// arrivals past the last row that finish the rows above it.
    arrived: usize,
}

impl LevelRows<'_> {
    fn width(&self) -> usize {
        self.window.width() as usize
    }

    fn height(&self) -> usize {
        self.window.height() as usize
    }

    fn slot(&self, row: usize) -> usize {
        row % (self.ring.len() / self.width())
    }

    fn row(&self, row: usize) -> &[f32] {
        let width = self.width();
        &self.ring[self.slot(row) * width..][..width]
    }

    /// Rows that are final. Row `r` takes its last step at arrival `r + k`,
    /// where `k` is that step: the last of the transform's steps for the rows
    /// that step updates, the one before it for the others.
    fn done(&self, transform: WaveletTransform) -> usize {
        let height = self.height();
        if height == 1 {
            return self.fed;
        }
        let Some(last_arrival) = self.arrived.checked_sub(1) else {
            return 0;
        };
        let steps = vertical_step_count(transform);
        let (_, last_step_even) = vertical_steps(transform, self.window.y0)[steps - 1];
        let final_row = |row: usize| {
            let last_step = if row.is_multiple_of(2) == last_step_even {
                steps
            } else {
                steps - 1
            };
            row + last_step <= last_arrival
        };
        // Every row up to `steps` above the last arrival is final, and the
        // next one when its last step is the one before the transform's last.
        let mut done = (last_arrival + 1).saturating_sub(steps).min(height);
        while done < height && final_row(done) {
            done += 1;
        }
        done
    }

    /// The vertical steps of this level's next arrival, which must not have
    /// to wait for a row: every row is fed, or the arrival's row is.
    fn lift_next_arrival(&mut self, transform: WaveletTransform) {
        let (width, arrival) = (self.width(), self.arrived);
        let level = Lifting {
            transform,
            y0: self.window.y0 as usize,
            height: self.height(),
        };
        lift_ring_arrivals(level, width, self.ring, arrival..arrival + 1);
        self.arrived += 1;
    }
}

/// What every level reads besides its own LL input, and the scratch its
/// rows' horizontal synthesis shares.
struct Inputs<'r> {
    rows: &'r mut dyn SubBandRows,
    ll_sub_band: usize,
    transform: WaveletTransform,
    scratch: &'r mut [f32],
}

/// Reconstruct a component's active levels from its stored sub-bands, read
/// through `rows`, one row at a time, and pass each row of the last level
/// inside `copy` to `write_row`, as its absolute row and its samples from
/// `copy.x0` to `copy.x1`. `windows` are the levels' IDWT windows from the
/// ROI plan, `sub_bands` their decompositions' sub-bands, and `ring` holds
/// at least [`ring_samples`] of them. Adds the samples it reconstructs to
/// `output_samples`; rows below the last one any level needs are never
/// computed.
#[expect(
    clippy::too_many_arguments,
    reason = "the streamed IDWT keeps its plan, storage, scratch, and output explicit"
)]
pub(super) fn reconstruct_rows(
    rows: &mut dyn SubBandRows,
    sub_bands: &[[usize; 3]],
    windows: &[IntRect],
    ll_sub_band: usize,
    transform: WaveletTransform,
    ring: &mut [f32],
    copy: IntRect,
    write_row: &mut dyn FnMut(u32, &[f32]) -> Result<()>,
    output_samples: &mut usize,
) -> Result<()> {
    let mut lines =
        LineReconstruction::new(sub_bands, windows, ll_sub_band, transform, ring, copy)?;
    let result = (|| {
        while let Some((y, row)) = lines.next_row(rows)? {
            write_row(y, row)?;
        }
        Ok(())
    })();
    *output_samples = output_samples.saturating_add(lines.output_samples());
    result
}

/// [`reconstruct_rows`] one row at a time, so that several components can
/// advance together.
pub(crate) struct LineReconstruction<'r> {
    levels: Vec<LevelRows<'r>>,
    /// Scratch for the horizontal synthesis of one row at a time.
    scratch: &'r mut [f32],
    ll_sub_band: usize,
    transform: WaveletTransform,
    /// The rows of the last level still to return.
    rows: core::ops::Range<u32>,
    columns: core::ops::Range<usize>,
}

impl<'r> LineReconstruction<'r> {
    /// Plan the reconstruction that [`reconstruct_rows`] describes.
    pub(crate) fn new(
        sub_bands: &[[usize; 3]],
        windows: &[IntRect],
        ll_sub_band: usize,
        transform: WaveletTransform,
        ring: &'r mut [f32],
        copy: IntRect,
    ) -> Result<Self> {
        if windows.is_empty() || windows.len() != sub_bands.len() {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        let mut levels = Vec::new();
        levels
            .try_reserve_exact(windows.len())
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        let mut rest = ring;
        for (window, sub_bands) in windows.iter().zip(sub_bands) {
            let len = level_ring_samples(*window, transform)?;
            if window.is_empty() || rest.len() < len {
                return Err(DecodingError::CodeBlockDecodeFailure.into());
            }
            let (level_ring, tail) = core::mem::take(&mut rest).split_at_mut(len);
            rest = tail;
            levels.push(LevelRows {
                window: *window,
                sub_bands: *sub_bands,
                ring: level_ring,
                fed: 0,
                arrived: 0,
            });
        }
        let widest = levels.iter().map(LevelRows::width).max().unwrap_or(0);
        let scratch_len = synthesis_scratch_samples(widest)?;
        if rest.len() < scratch_len {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        let (scratch, _) = core::mem::take(&mut rest).split_at_mut(scratch_len);

        let window = levels[levels.len() - 1].window;
        let (rows, columns) = if copy.is_empty() {
            (0..0, 0..0)
        } else if copy.x0 < window.x0
            || copy.x1 > window.x1
            || copy.y0 < window.y0
            || copy.y1 > window.y1
        {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        } else {
            (
                copy.y0..copy.y1,
                (copy.x0 - window.x0) as usize..(copy.x1 - window.x0) as usize,
            )
        };
        Ok(Self {
            levels,
            scratch,
            ll_sub_band,
            transform,
            rows,
            columns,
        })
    }

    /// The next row of the last level inside the copy rectangle, as its
    /// absolute row and its samples, or `None` after the last one. The
    /// samples stay valid until the next call.
    pub(crate) fn next_row(&mut self, rows: &mut dyn SubBandRows) -> Result<Option<(u32, &[f32])>> {
        let Some(y) = self.rows.next() else {
            return Ok(None);
        };
        let last = self.levels.len() - 1;
        let row = (y - self.levels[last].window.y0) as usize;
        let mut inputs = Inputs {
            rows,
            ll_sub_band: self.ll_sub_band,
            transform: self.transform,
            scratch: &mut *self.scratch,
        };
        finish_row(&mut self.levels, last, row, &mut inputs)?;
        Ok(Some((y, &self.levels[last].row(row)[self.columns.clone()])))
    }

    /// The samples reconstructed so far, over every level.
    pub(crate) fn output_samples(&self) -> usize {
        self.levels.iter().fold(0usize, |total, level| {
            total.saturating_add(level.fed.saturating_mul(level.width()))
        })
    }
}

/// Advance `levels[level]` until `row` of its window is final. Rows must be
/// requested in increasing order.
fn finish_row(
    levels: &mut [LevelRows<'_>],
    level: usize,
    row: usize,
    inputs: &mut Inputs<'_>,
) -> Result<()> {
    let ring_rows = ring_rows(inputs.transform);
    let steps = vertical_step_count(inputs.transform);
    while levels[level].done(inputs.transform) <= row {
        let current = &levels[level];
        let height = current.height();
        if current.fed < height {
            // Refilling a slot drops the row `ring_rows` above the new one,
            // which the previous arrival's deepest step read last: the ring
            // holds a row per step and two more.
            if current.arrived != current.fed {
                return Err(DecodingError::CodeBlockDecodeFailure.into());
            }
            feed_row(levels, level, inputs)?;
            if height == 1 {
                continue;
            }
        } else if height == 1 || current.arrived >= height + steps {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        levels[level].lift_next_arrival(inputs.transform);
    }
    if levels[level].fed > row + ring_rows {
        // The requested row has already left the ring.
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    Ok(())
}

/// Interleave, filter horizontally and scale the next row of `levels[level]`.
#[expect(
    clippy::too_many_lines,
    reason = "one row's band reads, interleave, horizontal filter and scaling stay together for its whole-row and windowed paths"
)]
fn feed_row(levels: &mut [LevelRows<'_>], level: usize, inputs: &mut Inputs<'_>) -> Result<()> {
    let (lower, upper) = levels.split_at_mut(level);
    let current = &mut upper[0];
    let local = current.fed;
    let y = current.window.y0 + u32::try_from(local).map_err(|_| ValidationError::ImageTooLarge)?;
    let band_y = y / 2;
    let sub_bands = current.sub_bands;

    // Even rows read the LL band and the first high band, odd rows the other
    // two, each at row `band_y`.
    if y.is_multiple_of(2) {
        if lower.is_empty() {
            inputs.rows.prepare_row(inputs.ll_sub_band, band_y)?;
        }
        inputs.rows.prepare_row(sub_bands[0], band_y)?;
    } else {
        inputs.rows.prepare_row(sub_bands[1], band_y)?;
        inputs.rows.prepare_row(sub_bands[2], band_y)?;
    }

    // The LL band is the stored sub-band for the first level, otherwise the
    // level below, whose window may not reach this row.
    let ll = match lower.len().checked_sub(1) {
        None => source(inputs.rows, inputs.ll_sub_band),
        Some(below) => {
            let below_window = lower[below].window;
            if y.is_multiple_of(2) && band_y >= below_window.y0 && band_y < below_window.y1 {
                let below_row = (band_y - below_window.y0) as usize;
                finish_row(lower, below, below_row, inputs)?;
                CoefficientSource::new(
                    lower[below].row(below_row),
                    IntRect::from_ltrb(below_window.x0, band_y, below_window.x1, band_y + 1),
                    below_window.width(),
                )
            } else {
                CoefficientSource::new(&[], IntRect::from_xywh(0, 0, 0, 0), 0)
            }
        }
    };
    let band = |index: usize| source(inputs.rows, index);
    let window = current.window;
    let height = current.height();
    let width = current.width();
    let slot = current.slot(local);
    let target = &mut current.ring[slot * width..][..width];
    // The low band holds the even columns, the high band the odd ones.
    let (low, high) = if y.is_multiple_of(2) {
        (ll, band(sub_bands[0]))
    } else {
        (band(sub_bands[1]), band(sub_bands[2]))
    };
    let low_len = (window.x1.div_ceil(2) - window.x0.div_ceil(2)) as usize;
    // Rows of one sample keep their own scaling, and narrow rows and band
    // rows their rectangles don't hold whole take the windowed path below.
    let band_rows = (height > 1 && width >= MIN_SIMD_ROW_WIDTH)
        .then(|| {
            Some((
                stored_row(low, window.x0.div_ceil(2), low_len, band_y)?,
                stored_row(high, window.x0 / 2, width - low_len, band_y)?,
            ))
        })
        .flatten();
    if let Some((low_row, high_row)) = band_rows {
        let (first, second) = if window.x0.is_multiple_of(2) {
            (low_row, high_row)
        } else {
            (high_row, low_row)
        };
        let row_scale = if inputs.transform == WaveletTransform::Irreversible97 {
            irreversible_row_scale(
                window.y0,
                local.is_multiple_of(2),
                super::OPENJPEG_NORMALIZED_HIGH_PASS_F32,
            )
        } else {
            1.0
        };
        synthesize_band_row(
            RowGeometry {
                width,
                x0: window.x0 as usize,
                transform: inputs.transform,
                high_pass: super::OPENJPEG_NORMALIZED_HIGH_PASS_F32,
            },
            first,
            second,
            target,
            inputs.scratch,
            row_scale,
        );
        current.fed += 1;
        return Ok(());
    }
    interleave_samples_roi(
        ll,
        band(sub_bands[0]),
        band(sub_bands[1]),
        band(sub_bands[2]),
        target,
        IntRect::from_ltrb(window.x0, y, window.x1, y + 1),
    );
    filter_row(
        target,
        width,
        window.x0 as usize,
        inputs.transform,
        super::OPENJPEG_NORMALIZED_HIGH_PASS_F32,
    );
    if height == 1 {
        if !y.is_multiple_of(2) {
            scale_one_row(target, 0.5);
        }
    } else if inputs.transform == WaveletTransform::Irreversible97 {
        scale_one_row(
            target,
            irreversible_row_scale(
                window.y0,
                local.is_multiple_of(2),
                super::OPENJPEG_NORMALIZED_HIGH_PASS_F32,
            ),
        );
    }
    current.fed += 1;
    Ok(())
}

/// Zero samples for band rows that lie wholly outside their stored
/// rectangles, as sub-bands without code-blocks in a row do: as wide as the
/// band rows of a 32768-wide level. A wider zero row takes the scalar path,
/// which made the empty 9000-wide band rows of an 18000-wide image's chroma
/// 1.6 times slower to reconstruct than its luma.
static ZERO_ROW: [f32; 16384] = [0.0; 16384];

/// The `len` samples of row `band_y` of `source` from band column `band_x0`,
/// which read zero outside its rectangle as [`CoefficientSource::get`] does:
/// read in place when the rectangle holds them all, or zeros when it holds
/// none. `None` when it holds only some.
fn stored_row(
    source: CoefficientSource<'_>,
    band_x0: u32,
    len: usize,
    band_y: u32,
) -> Option<&[f32]> {
    let rect = source.rect;
    let band_x1 = u64::from(band_x0) + len as u64;
    if band_y < rect.y0 || band_y >= rect.y1 || band_x1 <= u64::from(rect.x0) || band_x0 >= rect.x1
    {
        return ZERO_ROW.get(..len);
    }
    if band_x0 < rect.x0 || band_x1 > u64::from(rect.x1) {
        return None;
    }
    let start = (band_y - rect.y0) as usize * source.stride as usize + (band_x0 - rect.x0) as usize;
    source.coefficients.get(start..start.checked_add(len)?)
}

#[cfg(test)]
mod tests {
    use super::super::filter_common::periodic_symmetric_extension_left;
    use super::*;

    /// The rows final once `fed` rows of a level have arrived when every
    /// step finishes, in order, each row whose inputs the previous step has
    /// finished: the earliest a row can be final, which the arrival schedule
    /// must match so that no row reaches the output later than that.
    fn earliest_final_rows(
        transform: WaveletTransform,
        y0: u32,
        height: usize,
        fed: usize,
    ) -> usize {
        if height == 1 {
            return fed;
        }
        let steps = vertical_step_count(transform);
        let mut finished = [0; 4];
        for (index, &(_, even_offsets)) in vertical_steps(transform, y0)[..steps].iter().enumerate()
        {
            let available = if index == 0 { fed } else { finished[index - 1] };
            while finished[index] < height {
                let row = finished[index];
                let ready = if row.is_multiple_of(2) == even_offsets {
                    let above = periodic_symmetric_extension_left(row, 1);
                    let below = if row + 1 < height {
                        row + 1
                    } else {
                        height - 2
                    };
                    row.max(above).max(below) < available
                } else {
                    row < available
                };
                if !ready {
                    break;
                }
                finished[index] += 1;
            }
        }
        finished[steps - 1]
    }

    /// A level hands each row over after exactly the rows the earliest
    /// schedule needs: after `fed` rows, or once every row has arrived and
    /// the arrivals past the last one ran, its final rows are those.
    #[test]
    fn rows_are_final_after_the_rows_their_steps_need() {
        for transform in [
            WaveletTransform::Reversible53,
            WaveletTransform::Irreversible97,
        ] {
            let steps = vertical_step_count(transform);
            for y0 in [0, 1, 2, 7] {
                for height in 1..=24 {
                    let mut ring = [];
                    let mut level = LevelRows {
                        window: IntRect::from_xywh(
                            0,
                            y0,
                            1,
                            u32::try_from(height).expect("height"),
                        ),
                        sub_bands: [0; 3],
                        ring: &mut ring,
                        fed: 0,
                        arrived: 0,
                    };
                    for fed in 0..=height {
                        level.fed = fed;
                        level.arrived = if fed == height && height > 1 {
                            height + steps
                        } else {
                            fed
                        };
                        assert_eq!(
                            level.done(transform),
                            earliest_final_rows(transform, y0, height, fed),
                            "{transform:?} y0={y0} height={height} fed={fed}"
                        );
                    }
                }
            }
        }
    }
}
