// SPDX-License-Identifier: MIT OR Apache-2.0

//! RGB emission for the streaming fast 4:2:0 route.
//!
//! The decoder keeps a single MCU-row stripe. Each plane carries two halo rows
//! above the MCU row's samples holding the previous MCU row's last two rows,
//! so the stripe alone supplies the vertical context of fancy h2v2
//! upsampling: an output row needs its own chroma row and the nearer
//! neighbouring one. Right after MCU row `m` is decoded its pass emits output
//! rows `16m - 2 ..= 16m + 13` as eight row pairs that share a chroma row,
//! while the stripe is still in cache. Rows `16m + 14` and `16m + 15` wait for
//! the next pass, because the lower one needs the next MCU row's first chroma
//! row; the image's last MCU row emits through the bottom edge.
//! Every pixel comes from the same per-row 4:2:0 kernels, window rows, and
//! edge rules a stripe-at-a-time pass uses.

use super::super::emit::{rgb420_chroma_rows, valid_component_rows};
use super::super::layout::CropWindow;
use super::super::{StripeBuffer, StripePlane};
use super::{FAST420_HALO_ROWS as HALO_ROWS, MCU_PX};
use crate::backend::{Backend, Rgb420ChromaRows, Rgb420CroppedRowPair, Rgb420RowPair};
use crate::error::JpegError;
use crate::info::Rect;
use crate::output::InterleavedRgbWriter;

/// Output rows the pass over one MCU row emits.
#[derive(Clone, Copy)]
pub(super) struct Fast420RowPass {
    mcu_y: u32,
    y_start: u32,
    y_end: u32,
    /// First requested output row: the writer's row 0.
    out_y0: u32,
    /// Chroma plane rows below the halo that hold image samples.
    chroma_rows: usize,
}

impl Fast420RowPass {
    /// `halo_valid`: the previous MCU row was decoded, so the halo rows hold
    /// its last rows. `final_row`: this is the image's last MCU row, which
    /// emits through the bottom edge.
    pub(super) fn new(
        roi: Rect,
        image_height: u32,
        mcu_y: u32,
        halo_valid: bool,
        final_row: bool,
    ) -> Result<Self, JpegError> {
        let top = mcu_y * MCU_PX;
        let first = if halo_valid { top - HALO_ROWS } else { top };
        let end = if final_row {
            image_height
        } else {
            top + MCU_PX - HALO_ROWS
        };
        let y_start = first.max(roi.y);
        let y_end = end.min(roi.y + roi.h).max(y_start);
        // The top output row of an MCU row reads the chroma row above it.
        if y_start < y_end && y_start == top && mcu_y > 0 && !halo_valid {
            return Err(JpegError::InternalInvariant {
                reason: "4:2:0 stream emits a row whose upper chroma row was not decoded",
            });
        }
        let luma_rows = image_height.saturating_sub(top).min(MCU_PX) as usize;
        Ok(Self {
            mcu_y,
            y_start,
            y_end,
            out_y0: roi.y,
            chroma_rows: valid_component_rows(luma_rows, 2),
        })
    }

    /// Visit the pass's rows as aligned pairs, which share a chroma row, and
    /// single rows at either end.
    fn for_each_row_group(
        self,
        mut visit: impl FnMut(RowGroup) -> Result<(), JpegError>,
    ) -> Result<(), JpegError> {
        let mut y = self.y_start;
        if y < self.y_end && !y.is_multiple_of(2) {
            visit(RowGroup::Single(y))?;
            y += 1;
        }
        while y + 1 < self.y_end {
            visit(RowGroup::Pair(y))?;
            y += 2;
        }
        if y < self.y_end {
            visit(RowGroup::Single(y))?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum RowGroup {
    /// Output rows `y` (even) and `y + 1`.
    Pair(u32),
    Single(u32),
}

/// Window rows of the output rows on one chroma row: their luma rows and the
/// chroma rows above, at, and below it.
struct RowSources<'s> {
    stripe: &'s StripeBuffer,
    pass: Fast420RowPass,
    row_width: usize,
    chroma: Rgb420ChromaRows<'s>,
}

impl<'s> RowSources<'s> {
    /// The sources of output row `y` and of the other row on its chroma row.
    fn new(stripe: &'s StripeBuffer, crop: CropWindow, pass: Fast420RowPass, y: u32) -> Self {
        let halo = HALO_ROWS as usize;
        // At the image's top edge the halo holds nothing; leaving it out of
        // the plane view makes the first chroma row replicate upwards.
        let dropped = if pass.mcu_y == 0 { halo } else { 0 };
        let row = ((y / 2 + HALO_ROWS - pass.mcu_y * (MCU_PX / 2)) as usize) - dropped;
        let valid_rows = halo - dropped + pass.chroma_rows;
        let view = |plane_idx| {
            let plane = stripe.plane(plane_idx);
            StripePlane {
                data: &plane.data[dropped * plane.stride..],
                stride: plane.stride,
                rows: plane.rows - dropped,
            }
        };
        Self {
            stripe,
            pass,
            row_width: crop.row_width,
            chroma: rgb420_chroma_rows(
                |plane_idx| (None, view(plane_idx), None),
                row,
                valid_rows,
                crop.chroma_width(),
            ),
        }
    }

    fn luma(&self, y: u32) -> &'s [u8] {
        let luma_row = (y + HALO_ROWS - self.pass.mcu_y * MCU_PX) as usize;
        &self.stripe.row(0, luma_row)[..self.row_width]
    }

    /// Both rows of an aligned pair `y`, `y + 1`: the upper reads the chroma
    /// row above, the lower the chroma row below.
    fn pair<'d>(&self, y: u32, dst_top: &'d mut [u8], dst_bottom: &'d mut [u8]) -> Rgb420RowPair<'d>
    where
        's: 'd,
    {
        Rgb420RowPair::new(
            self.luma(y),
            Some(self.luma(y + 1)),
            self.chroma,
            dst_top,
            Some(dst_bottom),
        )
    }

    /// Row `y` on its own. A row-pair kernel computes its upper row from the
    /// current and the "previous" chroma row with the same 3:1 weights it
    /// gives the lower row and the "next" one, so the nearer neighbour of
    /// either kind goes in the previous slot.
    fn single<'d>(&self, y: u32, dst: &'d mut [u8]) -> Rgb420RowPair<'d>
    where
        's: 'd,
    {
        let chroma = self.chroma;
        let (near_cb, near_cr) = if y.is_multiple_of(2) {
            (chroma.prev_cb, chroma.prev_cr)
        } else {
            (chroma.next_cb, chroma.next_cr)
        };
        Rgb420RowPair::new(
            self.luma(y),
            None,
            Rgb420ChromaRows::new(
                near_cb,
                chroma.curr_cb,
                near_cb,
                near_cr,
                chroma.curr_cr,
                near_cr,
            ),
            dst,
            None,
        )
    }
}

/// Convert a decoded pass's rows through `writer`, cropped to the requested
/// window columns.
pub(super) fn emit_pass_rows<W: InterleavedRgbWriter>(
    backend: Backend,
    crop: CropWindow,
    pass: Fast420RowPass,
    stripe: &StripeBuffer,
    writer: &mut W,
) -> Result<(), JpegError> {
    let full_width = crop.is_full_width();
    let fill = |rows: Rgb420RowPair<'_>| {
        if full_width {
            backend.fill_rgb_row_pair_from_420(rows);
        } else {
            backend.fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(rows, crop.crop));
        }
    };
    pass.for_each_row_group(|group| match group {
        RowGroup::Pair(y) => writer.with_rgb_rows(y - pass.out_y0, 2, |dst_top, dst_bottom| {
            let dst_bottom = dst_bottom.ok_or(JpegError::InternalInvariant {
                reason: "row_count=2 supplies bottom row",
            })?;
            fill(RowSources::new(stripe, crop, pass, y).pair(y, dst_top, dst_bottom));
            Ok(())
        }),
        RowGroup::Single(y) => writer.with_rgb_rows(y - pass.out_y0, 1, |dst, _| {
            fill(RowSources::new(stripe, crop, pass, y).single(y, dst));
            Ok(())
        }),
    })
}
