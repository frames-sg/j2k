// SPDX-License-Identifier: MIT OR Apache-2.0

//! Row-streamed region IDWT.
//!
//! [`super::roi`] reconstructs each level's window as a whole buffer. This
//! computes the same windows with the same kernels, so the output is
//! identical, but holds each level as a ring of rows. A level produces its
//! window's rows top to bottom, pulling the LL rows it needs from the level
//! below, and the caller receives the last level's rows one at a time.
//!
//! Vertical lifting is streamed by keeping, for each step, how many rows the
//! step has finished. A step updates the rows of one parity from the rows
//! above and below them as the previous step left them. It can finish a row
//! once the previous step has finished every row it reads, and the next step
//! reaches a row only after this one has finished both of its neighbours, so
//! every read sees the value the whole-window filter would see.

use alloc::vec::Vec;

use super::super::build::Decomposition;
use super::super::codestream::WaveletTransform;
use super::super::decode::DecompositionStorage;
use super::super::rect::IntRect;
use super::filter_common::periodic_symmetric_extension_left;
use super::horizontal::filter_row;
use super::model::CoefficientSource;
use super::roi::interleave_samples_roi;
use super::vertical::{
    irreversible_row_scale, lift_one_row, scale_one_row, split_rows, vertical_step_count,
    vertical_steps, VerticalStep,
};
use crate::error::{DecodingError, ValidationError};
use crate::Result;

/// Rows each level keeps: one per lifting step, the row above the oldest
/// unfinished one, and the newest row.
pub(crate) const fn ring_rows(transform: WaveletTransform) -> usize {
    vertical_step_count(transform) + 2
}

/// Samples the rings of one component's levels hold together.
pub(crate) fn ring_samples(
    windows: impl IntoIterator<Item = IntRect>,
    transform: WaveletTransform,
) -> Result<usize> {
    windows.into_iter().try_fold(0usize, |total, window| {
        (window.width() as usize)
            .checked_mul(ring_rows(transform))
            .and_then(|samples| total.checked_add(samples))
            .ok_or(ValidationError::ImageTooLarge.into())
    })
}

/// One level of the streamed reconstruction.
struct LevelRows<'r> {
    window: IntRect,
    decomposition: &'r Decomposition,
    ring: &'r mut [f32],
    /// Rows interleaved, filtered horizontally and scaled.
    fed: usize,
    /// Rows each vertical step has finished.
    finished: [usize; 4],
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

    /// Rows that are final.
    fn done(&self, transform: WaveletTransform) -> usize {
        if self.height() == 1 {
            self.fed
        } else {
            self.finished[vertical_step_count(transform) - 1]
        }
    }
}

/// What every level reads besides its own LL input.
struct Inputs<'s, 'a> {
    storage: &'s DecompositionStorage<'a>,
    ll_sub_band: CoefficientSource<'s>,
    transform: WaveletTransform,
}

/// Reconstruct a component's active levels from its stored sub-bands, one
/// row at a time, and pass each row of the last level inside `copy` to
/// `write_row`, as its absolute row and its samples from `copy.x0` to
/// `copy.x1`. `windows` are the levels' IDWT windows from the ROI plan, and
/// `ring` holds at least [`ring_samples`] of them. Adds the samples it
/// reconstructs to `output_samples`; rows below the last one any level needs
/// are never computed.
#[expect(
    clippy::too_many_arguments,
    reason = "the streamed IDWT keeps its plan, storage, scratch, and output explicit"
)]
pub(super) fn reconstruct_rows(
    storage: &DecompositionStorage<'_>,
    decompositions: &[Decomposition],
    windows: &[IntRect],
    ll_sub_band: CoefficientSource<'_>,
    transform: WaveletTransform,
    ring: &mut [f32],
    copy: IntRect,
    write_row: &mut dyn FnMut(u32, &[f32]) -> Result<()>,
    output_samples: &mut usize,
) -> Result<()> {
    if windows.is_empty() || windows.len() != decompositions.len() {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    let mut levels = Vec::new();
    levels
        .try_reserve_exact(windows.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let mut rest = ring;
    for (window, decomposition) in windows.iter().zip(decompositions) {
        let len = ring_samples([*window], transform)?;
        if window.is_empty() || rest.len() < len {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        let (level_ring, tail) = core::mem::take(&mut rest).split_at_mut(len);
        rest = tail;
        levels.push(LevelRows {
            window: *window,
            decomposition,
            ring: level_ring,
            fed: 0,
            finished: [0; 4],
        });
    }

    let last = levels.len() - 1;
    let window = levels[last].window;
    if copy.is_empty() {
        return Ok(());
    }
    if copy.x0 < window.x0 || copy.x1 > window.x1 || copy.y0 < window.y0 || copy.y1 > window.y1 {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    let inputs = Inputs {
        storage,
        ll_sub_band,
        transform,
    };
    let columns = (copy.x0 - window.x0) as usize..(copy.x1 - window.x0) as usize;
    let result = (|| {
        for y in copy.y0..copy.y1 {
            let row = (y - window.y0) as usize;
            finish_row(&mut levels, last, row, &inputs)?;
            write_row(y, &levels[last].row(row)[columns.clone()])?;
        }
        Ok(())
    })();
    for level in &levels {
        *output_samples = output_samples.saturating_add(level.fed.saturating_mul(level.width()));
    }
    result
}

/// Advance `levels[level]` until `row` of its window is final. Rows must be
/// requested in increasing order.
fn finish_row(
    levels: &mut [LevelRows<'_>],
    level: usize,
    row: usize,
    inputs: &Inputs<'_, '_>,
) -> Result<()> {
    let ring_rows = ring_rows(inputs.transform);
    while levels[level].done(inputs.transform) <= row {
        let current = &levels[level];
        // Refilling a slot drops the row `ring_rows` above the new one,
        // which no unfinished step may still read.
        let oldest_read = current.done(inputs.transform).saturating_sub(1);
        if current.fed >= current.height()
            || (current.fed >= ring_rows && current.fed - ring_rows >= oldest_read)
        {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        feed_row(levels, level, inputs)?;
        advance_steps(&mut levels[level], inputs.transform);
    }
    if levels[level].fed > row + ring_rows {
        // The requested row has already left the ring.
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    Ok(())
}

/// Interleave, filter horizontally and scale the next row of `levels[level]`.
fn feed_row(levels: &mut [LevelRows<'_>], level: usize, inputs: &Inputs<'_, '_>) -> Result<()> {
    let (lower, upper) = levels.split_at_mut(level);
    let current = &mut upper[0];
    let local = current.fed;
    let y = current.window.y0 + u32::try_from(local).map_err(|_| ValidationError::ImageTooLarge)?;
    let band_y = y / 2;

    // Even rows read the LL band: the stored sub-band for the first level,
    // otherwise the level below, whose window may not reach this row.
    let ll = match lower.len().checked_sub(1) {
        None => inputs.ll_sub_band,
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
    let storage = inputs.storage;
    let band = |index: usize| CoefficientSource::from_sub_band(&storage.sub_bands[index], storage);
    let sub_bands = current.decomposition.sub_bands;
    let window = current.window;
    let height = current.height();
    let width = current.width();
    let slot = current.slot(local);
    let target = &mut current.ring[slot * width..][..width];
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

/// Let every vertical step finish as many rows as the rows fed so far allow.
fn advance_steps(level: &mut LevelRows<'_>, transform: WaveletTransform) {
    let height = level.height();
    if height == 1 {
        return;
    }
    let steps = vertical_steps(transform, level.window.y0);
    for (index, &(step, even_offsets)) in steps
        .iter()
        .take(vertical_step_count(transform))
        .enumerate()
    {
        let available = if index == 0 {
            level.fed
        } else {
            level.finished[index - 1]
        };
        while level.finished[index] < height {
            let row = level.finished[index];
            if row.is_multiple_of(2) == even_offsets {
                let above = periodic_symmetric_extension_left(row, 1);
                let below = if row + 1 < height {
                    row + 1
                } else {
                    height - 2
                };
                if row.max(above).max(below) >= available {
                    break;
                }
                lift_level_row(level, step, row, above, below);
            } else if row >= available {
                break;
            }
            level.finished[index] += 1;
        }
    }
}

fn lift_level_row(
    level: &mut LevelRows<'_>,
    step: VerticalStep,
    row: usize,
    above: usize,
    below: usize,
) {
    let width = level.width();
    let (target, above, below) = (level.slot(row), level.slot(above), level.slot(below));
    let (row_samples, above, below) = split_rows(level.ring, width, target, above, below);
    lift_one_row(step, row_samples, above, below);
}
