// SPDX-License-Identifier: MIT OR Apache-2.0

//! The parts of an HT coefficient arena that no code-block job writes.
//!
//! A successful HT job writes every sample of its block, so only blocks
//! without coding passes, which have no job, need zeroing before decode.

use super::super::{Error, J2kHtCleanupBatchJob};
use crate::engine::abi::J2kZeroRect;

/// Rows per rectangle, so a large empty region is spread over threadgroups.
const MAX_RECT_ROWS: u32 = 16;

/// Appends the rectangles of one `width` x `height` sub-band, stored row-major
/// at arena word `base`, that none of `jobs` covers.
///
/// Job edges are code-block grid lines, so the cells between consecutive
/// edges are either inside one job or outside every job.
pub(super) fn append_uncovered_rects(
    base: usize,
    width: u32,
    height: u32,
    jobs: &[J2kHtCleanupBatchJob],
    rects: &mut Vec<J2kZeroRect>,
) -> Result<(), Error> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K MetalDirect zero rectangles");
    let mut blocks = budget.try_vec(jobs.len(), "HT zero-rectangle blocks")?;
    for job in jobs {
        if job.width == 0 || job.height == 0 {
            continue;
        }
        let local = (job.output_offset as usize)
            .checked_sub(base)
            .ok_or_else(|| outside("job starts before its sub-band"))?;
        let x = u32::try_from(local % width as usize).map_err(|_| outside("job column"))?;
        let y = u32::try_from(local / width as usize).map_err(|_| outside("job row"))?;
        if job.output_stride != width
            || x.checked_add(job.width).is_none_or(|end| end > width)
            || y.checked_add(job.height).is_none_or(|end| end > height)
        {
            return Err(outside("job does not lie within its sub-band"));
        }
        blocks.push((x, y, job.width, job.height));
    }

    let xs = grid_lines(
        &mut budget,
        width,
        blocks.iter().flat_map(|&(x, _, w, _)| [x, x + w]),
    )?;
    let ys = grid_lines(
        &mut budget,
        height,
        blocks.iter().flat_map(|&(_, y, _, h)| [y, y + h]),
    )?;
    let columns = xs.len() - 1;
    let mut covered = budget.try_vec(columns * (ys.len() - 1), "HT zero-rectangle coverage")?;
    covered.resize(columns * (ys.len() - 1), false);
    for &(x, y, w, h) in &blocks {
        let (cx0, cx1) = (line_index(&xs, x)?, line_index(&xs, x + w)?);
        for cy in line_index(&ys, y)?..line_index(&ys, y + h)? {
            covered[cy * columns + cx0..cy * columns + cx1].fill(true);
        }
    }

    for (cy, rows) in ys.windows(2).enumerate() {
        let mut cx = 0;
        while cx < columns {
            if covered[cy * columns + cx] {
                cx += 1;
                continue;
            }
            let start = cx;
            while cx < columns && !covered[cy * columns + cx] {
                cx += 1;
            }
            push_rect(base, width, (xs[start], xs[cx]), (rows[0], rows[1]), rects)?;
        }
    }
    Ok(())
}

/// Appends `len` contiguous words from `offset` as rectangles of whole rows.
pub(in crate::engine) fn append_span_rects(
    offset: usize,
    len: usize,
    rects: &mut Vec<J2kZeroRect>,
) -> Result<(), Error> {
    const ROW_WORDS: usize = 4096;
    let full_rows = len / ROW_WORDS;
    let mut cursor = offset;
    let mut remaining_rows = full_rows;
    while remaining_rows > 0 {
        let rows = remaining_rows.min(MAX_RECT_ROWS as usize);
        rects.push(rect(cursor, ROW_WORDS, rows, ROW_WORDS)?);
        cursor += rows * ROW_WORDS;
        remaining_rows -= rows;
    }
    let tail = len - full_rows * ROW_WORDS;
    if tail > 0 {
        rects.push(rect(cursor, tail, 1, tail)?);
    }
    Ok(())
}

fn push_rect(
    base: usize,
    stride: u32,
    (x0, x1): (u32, u32),
    (y0, y1): (u32, u32),
    rects: &mut Vec<J2kZeroRect>,
) -> Result<(), Error> {
    let mut y = y0;
    while y < y1 {
        let rows = (y1 - y).min(MAX_RECT_ROWS);
        let offset = base + y as usize * stride as usize + x0 as usize;
        rects.push(rect(
            offset,
            (x1 - x0) as usize,
            rows as usize,
            stride as usize,
        )?);
        y += rows;
    }
    Ok(())
}

fn rect(offset: usize, width: usize, height: usize, stride: usize) -> Result<J2kZeroRect, Error> {
    let field =
        |value: usize| u32::try_from(value).map_err(|_| outside("zero rectangle exceeds u32"));
    Ok(J2kZeroRect {
        offset: field(offset)?,
        width: field(width)?,
        height: field(height)?,
        stride: field(stride)?,
    })
}

/// The sorted, distinct edges `0`, `extent`, and every job edge.
fn grid_lines(
    budget: &mut crate::batch_allocation::BatchMetadataBudget,
    extent: u32,
    edges: impl Iterator<Item = u32>,
) -> Result<Vec<u32>, Error> {
    let mut lines = budget.try_vec(edges.size_hint().0 + 2, "HT zero-rectangle grid lines")?;
    lines.push(0);
    lines.push(extent);
    lines.extend(edges);
    lines.sort_unstable();
    lines.dedup();
    Ok(lines)
}

fn line_index(lines: &[u32], value: u32) -> Result<usize, Error> {
    lines
        .binary_search(&value)
        .map_err(|_| outside("job edge is not a grid line"))
}

fn outside(reason: &'static str) -> Error {
    Error::MetalStateInvariant {
        state: "HTJ2K MetalDirect zero rectangles",
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::{append_span_rects, append_uncovered_rects, J2kZeroRect, MAX_RECT_ROWS};
    use crate::engine::abi::J2kHtCleanupBatchJob;

    fn job(
        base: u32,
        stride: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> J2kHtCleanupBatchJob {
        J2kHtCleanupBatchJob {
            coded_offset: 0,
            width,
            height,
            coded_len: 2,
            cleanup_length: 2,
            refinement_length: 0,
            missing_msbs: 0,
            num_bitplanes: 1,
            roi_shift: 0,
            number_of_coding_passes: 1,
            output_stride: stride,
            output_offset: base + y * stride + x,
            dequantization_step: 1.0,
            stripe_causal: 0,
            irreversible_midpoint: 0,
        }
    }

    /// Marks every word the rectangles cover, failing on any word twice.
    fn coverage(len: usize, rects: &[J2kZeroRect]) -> Vec<bool> {
        let mut words = vec![false; len];
        for rect in rects {
            assert!(rect.height <= MAX_RECT_ROWS);
            for row in 0..rect.height {
                for x in 0..rect.width {
                    let index = (rect.offset + row * rect.stride + x) as usize;
                    assert!(!words[index], "word {index} zeroed twice");
                    words[index] = true;
                }
            }
        }
        words
    }

    #[test]
    fn rectangles_cover_exactly_the_blocks_without_jobs() {
        // A 10 x 40 sub-band at word 100 with 4 x 8 blocks; edge blocks are
        // narrower and shorter, and two blocks have no job.
        let (base, width, height) = (100_u32, 10_u32, 40_u32);
        let mut jobs = Vec::new();
        let mut missing = Vec::new();
        for by in (0..height).step_by(8) {
            for bx in (0..width).step_by(4) {
                let (w, h) = ((width - bx).min(4), (height - by).min(8));
                if (bx, by) == (4, 8) || (bx, by) == (8, 32) {
                    missing.push((bx, by, w, h));
                } else {
                    jobs.push(job(base, width, bx, by, w, h));
                }
            }
        }
        let mut rects = Vec::new();
        append_uncovered_rects(base as usize, width, height, &jobs, &mut rects).unwrap();
        let words = coverage((base + width * height) as usize, &rects);
        for y in 0..height {
            for x in 0..width {
                let expected = missing
                    .iter()
                    .any(|&(bx, by, w, h)| (bx..bx + w).contains(&x) && (by..by + h).contains(&y));
                assert_eq!(
                    words[(base + y * width + x) as usize],
                    expected,
                    "({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn fully_covered_sub_bands_need_no_rectangles() {
        let jobs = [job(0, 8, 0, 0, 8, 4), job(0, 8, 0, 4, 8, 3)];
        let mut rects = Vec::new();
        append_uncovered_rects(0, 8, 7, &jobs, &mut rects).unwrap();
        assert_eq!(rects, Vec::<J2kZeroRect>::new());
    }

    #[test]
    fn sub_bands_without_jobs_are_zeroed_in_short_rectangles() {
        let mut rects = Vec::new();
        append_uncovered_rects(5, 33, 50, &[], &mut rects).unwrap();
        let words = coverage(5 + 33 * 50, &rects);
        assert!(words[5..].iter().all(|&word| word));
        assert!(!words[..5].iter().any(|&word| word));
    }

    #[test]
    fn jobs_outside_their_sub_band_are_rejected() {
        let mut rects = Vec::new();
        assert!(append_uncovered_rects(0, 8, 8, &[job(0, 8, 6, 0, 4, 4)], &mut rects).is_err());
        assert!(append_uncovered_rects(0, 8, 8, &[job(0, 9, 0, 0, 4, 4)], &mut rects).is_err());
    }

    #[test]
    fn spans_split_into_bounded_rectangles() {
        let mut rects = Vec::new();
        let len = 4096 * 40 + 123;
        append_span_rects(7, len, &mut rects).unwrap();
        let words = coverage(7 + len, &rects);
        assert!(words[7..].iter().all(|&word| word));
    }
}
