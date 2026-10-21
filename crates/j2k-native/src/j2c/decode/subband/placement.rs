// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where a decoded code-block's samples land in its sub-band's storage.

use alloc::vec::Vec;

use crate::error::{DecodingError, Result, ValidationError};
use crate::j2c::decode::DecodeAllocationBudget;
use crate::j2c::rect::IntRect;

/// The part of a code-block that its sub-band stores.
///
/// Region decode stores only the window of each sub-band that the IDWT reads,
/// so a code-block on the edge of that window keeps only some of its rows and
/// columns. `source_*` is the first kept sample inside the code-block, and
/// `target_*` is where it lands in the stored rectangle. Every other decode
/// stores whole code-blocks, with zero source offsets and the full size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockPlacement {
    pub(crate) source_x: u32,
    pub(crate) source_y: u32,
    pub(crate) target_x: u32,
    pub(crate) target_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl BlockPlacement {
    /// Place `code_block` in a sub-band that stores `stored`.
    pub(crate) fn new(code_block: IntRect, stored: IntRect) -> Self {
        let x0 = code_block.x0.max(stored.x0);
        let y0 = code_block.y0.max(stored.y0);
        let x1 = code_block.x1.min(stored.x1);
        let y1 = code_block.y1.min(stored.y1);
        if x0 >= x1 || y0 >= y1 {
            return Self::whole(0, 0, 0, 0);
        }
        Self {
            source_x: x0 - code_block.x0,
            source_y: y0 - code_block.y0,
            target_x: x0 - stored.x0,
            target_y: y0 - stored.y0,
            width: x1 - x0,
            height: y1 - y0,
        }
    }

    /// A whole `width` x `height` code-block stored at `target`.
    pub(crate) fn whole(target_x: u32, target_y: u32, width: u32, height: u32) -> Self {
        Self {
            source_x: 0,
            source_y: 0,
            target_x,
            target_y,
            width,
            height,
        }
    }

    /// Whether the whole `width` x `height` code-block is stored.
    pub(crate) fn is_whole(self, width: u32, height: u32) -> bool {
        self.source_x == 0 && self.source_y == 0 && self.width == width && self.height == height
    }

    /// Where a whole `width` x `height` code-block lands, for writers that
    /// store whole code-blocks only.
    pub(crate) fn whole_target(self, width: u32, height: u32) -> Result<(u32, u32)> {
        if self.is_whole(width, height) {
            Ok((self.target_x, self.target_y))
        } else {
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }

    /// Index of the first kept sample in a row-major `stride`-wide store.
    pub(crate) fn target_index(self, stride: usize) -> Result<usize> {
        (self.target_y as usize)
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(self.target_x as usize))
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    /// Check that the kept samples fit a `block_width` x `block_height`
    /// code-block and a `stored_width` x `stored_height` store.
    #[cfg_attr(
        not(any(test, feature = "parallel")),
        expect(dead_code, reason = "only the parallel writers validate placements")
    )]
    pub(crate) fn validate(
        self,
        block_width: u32,
        block_height: u32,
        stored_width: u32,
        stored_height: u32,
    ) -> Result<()> {
        let fits = |offset: u32, len: u32, bound: u32| {
            offset.checked_add(len).is_some_and(|end| end <= bound)
        };
        if fits(self.source_x, self.width, block_width)
            && fits(self.source_y, self.height, block_height)
            && fits(self.target_x, self.width, stored_width)
            && fits(self.target_y, self.height, stored_height)
        {
            Ok(())
        } else {
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }

    /// Copy the kept rows of `rows` into `store`, whose rows are `stride`
    /// samples apart. Each item of `rows` is one row of the code-block.
    pub(crate) fn store_rows<'r, S: 'r, T>(
        self,
        rows: impl Iterator<Item = &'r [S]>,
        store: &mut [T],
        stride: usize,
        mut write_row: impl FnMut(&[S], &mut [T]),
    ) -> Result<()> {
        let width = self.width as usize;
        let mut start = self.target_index(stride)?;
        for row in rows.skip(self.source_y as usize).take(self.height as usize) {
            let source = row
                .get(self.source_x as usize..)
                .and_then(|row| row.get(..width))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let target = store
                .get_mut(start..)
                .and_then(|row| row.get_mut(..width))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            write_row(source, target);
            start = start
                .checked_add(stride)
                .ok_or(ValidationError::ImageTooLarge)?;
        }
        Ok(())
    }

    /// Copy the kept samples of a row-major `block_width`-wide code-block.
    #[cfg_attr(
        not(any(test, feature = "simd", feature = "parallel")),
        expect(
            dead_code,
            reason = "only the SIMD and parallel writers decode whole blocks first"
        )
    )]
    pub(crate) fn copy_from_block<T: Copy>(
        self,
        block: &[T],
        block_width: u32,
        store: &mut [T],
        stride: usize,
    ) -> Result<()> {
        if block_width == 0 {
            return Ok(());
        }
        self.store_rows(
            block.chunks_exact(block_width as usize),
            store,
            stride,
            |source, target| target.copy_from_slice(source),
        )
    }
}

/// Zeroed room for a whole `width` x `height` code-block, for writers that
/// decode straight into strided storage and so cannot drop samples a clipped
/// [`BlockPlacement`] leaves out. Grows `scratch` under `budget` when short.
#[cfg_attr(
    not(feature = "simd"),
    expect(dead_code, reason = "only the SIMD HT writer decodes in place")
)]
pub(crate) fn zeroed_block_scratch<'s>(
    scratch: &'s mut Vec<f32>,
    width: u32,
    height: u32,
    budget: &mut DecodeAllocationBudget,
) -> Result<&'s mut [f32]> {
    let len = (width as usize)
        .checked_mul(height as usize)
        .ok_or(ValidationError::ImageTooLarge)?;
    if scratch.len() < len {
        budget.resize_new(scratch, len, 0.0)?;
    }
    let block = &mut scratch[..len];
    block.fill(0.0);
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::{BlockPlacement, IntRect};

    #[test]
    fn a_contained_code_block_is_placed_whole() {
        let placement = BlockPlacement::new(
            IntRect::from_ltrb(8, 4, 12, 8),
            IntRect::from_ltrb(4, 2, 16, 10),
        );
        assert_eq!(placement, BlockPlacement::whole(4, 2, 4, 4));
        assert!(placement.is_whole(4, 4));
    }

    #[test]
    fn an_edge_code_block_keeps_only_its_stored_samples() {
        let placement = BlockPlacement::new(
            IntRect::from_ltrb(0, 0, 4, 4),
            IntRect::from_ltrb(1, 2, 3, 9),
        );
        assert_eq!(
            placement,
            BlockPlacement {
                source_x: 1,
                source_y: 2,
                target_x: 0,
                target_y: 0,
                width: 2,
                height: 2,
            }
        );
        assert!(!placement.is_whole(4, 4));
        assert!(placement.whole_target(4, 4).is_err());

        let block = (0..16).collect::<Vec<i32>>();
        let mut store = vec![-1; 2 * 7];
        placement
            .copy_from_block(&block, 4, &mut store, 2)
            .expect("copy");
        assert_eq!(&store[..4], &[9, 10, 13, 14]);
        assert!(store[4..].iter().all(|&sample| sample == -1));
    }

    #[test]
    fn a_disjoint_code_block_keeps_nothing() {
        let placement = BlockPlacement::new(
            IntRect::from_ltrb(0, 0, 4, 4),
            IntRect::from_ltrb(4, 0, 8, 4),
        );
        assert_eq!((placement.width, placement.height), (0, 0));
    }

    #[test]
    fn validation_rejects_samples_outside_the_block_or_store() {
        let placement = BlockPlacement::new(
            IntRect::from_ltrb(0, 0, 4, 4),
            IntRect::from_ltrb(2, 2, 6, 6),
        );
        assert!(placement.validate(4, 4, 4, 4).is_ok());
        assert!(placement.validate(3, 4, 4, 4).is_err());
        assert!(placement.validate(4, 4, 1, 4).is_err());
    }
}
