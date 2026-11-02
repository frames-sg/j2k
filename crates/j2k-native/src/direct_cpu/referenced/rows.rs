// SPDX-License-Identifier: MIT OR Apache-2.0

//! Row-parallel code-block dispatch shared by referenced HT and classic
//! execution: borrowed jobs are grouped into disjoint rows of one sub-band
//! and decoded on the Rayon pool, each row with a pooled task workspace.

use crate::error::{DecodingError, Result, ValidationError};
use crate::{HtOwnedCodeBlockBatchJob, J2kOwnedCodeBlockBatchJob};

use super::super::DirectWorkspaceBudget;

/// Output placement of one owned code-block job inside its sub-band.
pub(in crate::direct_cpu) trait CodeBlockRowGeometry: Sync {
    fn output_x(&self) -> u32;
    fn output_y(&self) -> u32;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}

impl CodeBlockRowGeometry for HtOwnedCodeBlockBatchJob {
    fn output_x(&self) -> u32 {
        self.output_x
    }
    fn output_y(&self) -> u32 {
        self.output_y
    }
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
}

impl CodeBlockRowGeometry for J2kOwnedCodeBlockBatchJob {
    fn output_x(&self) -> u32 {
        self.output_x
    }
    fn output_y(&self) -> u32 {
        self.output_y
    }
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
}

/// One code block of a parallel sub-band decode with its payload `D` and
/// output range in the sub-band. HT borrows each payload as one input slice;
/// classic carries the payload's input fragments for its task to join.
pub(in crate::direct_cpu) struct RowJob<'p, J, D> {
    pub(in crate::direct_cpu) job: &'p J,
    pub(in crate::direct_cpu) data: D,
    pub(in crate::direct_cpu) start: usize,
    pub(in crate::direct_cpu) end: usize,
}

/// A row of code blocks: its output rows, its jobs, and its first sample.
pub(in crate::direct_cpu) type CodeBlockRow<'o, 'j, 'p, J, D> =
    (&'o mut [f32], &'j [RowJob<'p, J, D>], usize);

/// Bytes owned together while one sub-band is dispatched in parallel, with
/// `W` the pooled per-task workspace handle. Planning and execution share
/// this helper so the private dispatch types cannot drift from their
/// accounting.
pub(in crate::direct_cpu) fn row_dispatch_transient_bytes<J, D, W>(
    entry_capacity: usize,
    row_capacity: usize,
    workspace_capacity: usize,
) -> Result<usize> {
    entry_capacity
        .checked_mul(core::mem::size_of::<RowJob<'_, J, D>>())
        .and_then(|bytes| {
            row_capacity
                .checked_mul(core::mem::size_of::<CodeBlockRow<'_, '_, '_, J, D>>())
                .and_then(|rows| bytes.checked_add(rows))
        })
        .and_then(|bytes| {
            workspace_capacity
                .checked_mul(core::mem::size_of::<W>())
                .and_then(|workspaces| bytes.checked_add(workspaces))
        })
        .ok_or_else(|| ValidationError::ImageTooLarge.into())
}

/// Split `output` into one slice per row of code blocks, or `None` when the
/// jobs, sorted by position, do not form disjoint rows of equal-height blocks.
pub(in crate::direct_cpu) fn carve_rows<'o, 'j, 'p, J: CodeBlockRowGeometry, D, W>(
    entries: &'j [RowJob<'p, J, D>],
    output: &'o mut [f32],
    sub_band_width: usize,
    entry_capacity: usize,
    budget: DirectWorkspaceBudget,
) -> Result<Option<alloc::vec::Vec<CodeBlockRow<'o, 'j, 'p, J, D>>>> {
    let mut rows = alloc::vec::Vec::new();
    crate::try_reserve_decode_elements(&mut rows, entries.len())?;
    budget.validate_workspace(row_dispatch_transient_bytes::<J, D, W>(
        entry_capacity,
        rows.capacity(),
        0,
    )?)?;
    let mut rest = output;
    let mut consumed = 0usize;
    let mut index = 0usize;
    while index < entries.len() {
        let first = &entries[index];
        let (y, height) = (first.job.output_y() as usize, first.job.height() as usize);
        let mut end = index;
        let mut next_x = 0u32;
        while end < entries.len() && entries[end].job.output_y() as usize == y {
            let job = entries[end].job;
            if job.height() as usize != height || job.output_x() < next_x {
                return Ok(None);
            }
            next_x = job.output_x() + job.width();
            end += 1;
        }
        let row_start = y * sub_band_width;
        let row_end = (y + height) * sub_band_width;
        if row_start < consumed || row_end > consumed + rest.len() {
            return Ok(None);
        }
        let tail = core::mem::take(&mut rest);
        let (_, tail) = tail.split_at_mut(row_start - consumed);
        let (row, tail) = tail.split_at_mut(row_end - row_start);
        rest = tail;
        consumed = row_end;
        rows.push((row, &entries[index..end], row_start));
        index = end;
    }
    Ok(Some(rows))
}

/// Decode rows of code blocks on the Rayon pool, each with a pooled task
/// workspace taken from `main` and `tasks`.
pub(in crate::direct_cpu) fn decode_rows<J, D, W, F>(
    rows: alloc::vec::Vec<CodeBlockRow<'_, '_, '_, J, D>>,
    entry_capacity: usize,
    main: W,
    tasks: impl ExactSizeIterator<Item = W>,
    budget: DirectWorkspaceBudget,
    decode: F,
) -> Result<()>
where
    J: Sync,
    D: Sync,
    W: Send,
    F: Fn(&RowJob<'_, J, D>, &mut [f32], &mut W) -> Result<()> + Sync,
{
    use rayon::prelude::*;
    use std::sync::Mutex;

    let row_capacity = rows.capacity();
    let workspace_count = tasks
        .len()
        .checked_add(1)
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut workspaces = alloc::vec::Vec::new();
    crate::try_reserve_decode_elements(&mut workspaces, workspace_count)?;
    workspaces.push(main);
    workspaces.extend(tasks);
    budget.validate_workspace(row_dispatch_transient_bytes::<J, D, W>(
        entry_capacity,
        row_capacity,
        workspaces.capacity(),
    )?)?;
    let pool = Mutex::new(workspaces);
    rows.into_par_iter()
        .with_max_len(1)
        .try_for_each(|(row, jobs, row_start)| -> Result<()> {
            // Bind the popped workspace first so the guard drops before the
            // workspace is returned.
            let pooled = pool
                .lock()
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                .pop();
            let mut workspace = pooled.ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let result = jobs.iter().try_for_each(|entry| {
                let output = &mut row[entry.start - row_start..entry.end - row_start];
                decode(entry, output, &mut workspace)
            });
            pool.lock()
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                .push(workspace);
            result
        })
}
