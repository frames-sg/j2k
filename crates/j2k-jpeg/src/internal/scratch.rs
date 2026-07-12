// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable scratch buffers for the decode path.
//!
//! A [`ScratchPool`] owns every `Vec` that the sequential scan decoder
//! would otherwise allocate on each call: the three rolling MCU stripe
//! buffers, the per-component DC predictor, the chroma upsample rows, and
//! the RGB row buffers used by [`j2k_core::RowSink`] drivers.
//!
//! Use [`Decoder::decode_into_with_scratch`](crate::Decoder::decode_into_with_scratch)
//! / [`decode_rows_with_scratch`](crate::Decoder::decode_rows_with_scratch)
//! with a single long-lived pool to pay the allocation cost once across a
//! tile batch. Same-shape and cap-valid smaller requests reuse capacity.
//! Growth discards disposable retained storage first so a reallocation never
//! owns the stale and replacement buffers at the same time.

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, try_reserve_for_len_with_live_budget,
};
use crate::entropy::sequential::{PreparedDecodePlan, StripeBuffer, StripeLayout};
use crate::error::JpegError;
use alloc::vec::Vec;
use j2k_core::ScratchPool as CoreScratchPool;

#[derive(Debug, Default)]
pub(crate) struct YCbCr420Rows {
    pub(crate) cb_top: Vec<u8>,
    pub(crate) cb_bot: Vec<u8>,
    pub(crate) cr_top: Vec<u8>,
    pub(crate) cr_bot: Vec<u8>,
}

#[derive(Debug, Default)]
pub(crate) struct YCbCrGenericRows {
    pub(crate) cb_up: Vec<u8>,
    pub(crate) cr_up: Vec<u8>,
}

#[derive(Debug, Default)]
pub(crate) struct RgbGenericRows {
    pub(crate) r: Vec<u8>,
    pub(crate) g: Vec<u8>,
    pub(crate) b: Vec<u8>,
    pub(crate) k: Vec<u8>,
}

#[derive(Debug, Default)]
pub(crate) struct SinkRows {
    pub(crate) top_row: Vec<u8>,
    pub(crate) bottom_row: Vec<u8>,
}

/// Pool of decoder-internal scratch buffers, reusable across many
/// [`Decoder::decode_into_with_scratch`](crate::Decoder::decode_into_with_scratch)
/// / [`decode_rows_with_scratch`](crate::Decoder::decode_rows_with_scratch)
/// calls.
#[derive(Debug, Default)]
pub struct ScratchPool {
    pub(crate) prev_dc: Vec<i32>,
    pub(crate) stripe_a: StripeBuffer,
    pub(crate) stripe_b: StripeBuffer,
    pub(crate) stripe_c: StripeBuffer,
    pub(crate) ycbcr_420_rows: YCbCr420Rows,
    pub(crate) ycbcr_generic_rows: YCbCrGenericRows,
    pub(crate) rgb_generic_rows: RgbGenericRows,
    pub(crate) lossless_prev_row: Vec<u8>,
    pub(crate) lossless_curr_row: Vec<u8>,
    sink_rows: SinkRows,
    detached_sink_bytes: usize,
}

#[derive(Clone, Copy)]
struct SequentialScratchLayout {
    stripe: StripeLayout,
    component_count: usize,
    row_width: usize,
}

impl SequentialScratchLayout {
    fn for_plan(
        plan: &PreparedDecodePlan,
        mcus_per_row: u32,
        block_size: u32,
    ) -> Result<Self, JpegError> {
        Ok(Self {
            stripe: StripeLayout::for_plan(plan, mcus_per_row, block_size)?,
            component_count: plan.sampling.len(),
            row_width: plan.dimensions.0.div_ceil(8 / block_size.max(1)) as usize,
        })
    }

    fn minimum_bytes(self, detached_sink_bytes: usize) -> Result<usize, JpegError> {
        let mut total = checked_allocation_bytes::<i32>(self.component_count)?;
        let stripe_bytes = self.stripe.allocation_bytes()?;
        for _ in 0..3 {
            total = checked_add_allocation_bytes(total, stripe_bytes)?;
        }
        let row_bytes = self
            .row_width
            .checked_mul(10)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
        total = checked_add_allocation_bytes(total, row_bytes)?;
        checked_add_allocation_bytes(total, detached_sink_bytes)
    }
}

impl ScratchPool {
    /// Create an empty pool. The first decode that uses it pays the full
    /// allocation cost; subsequent decodes at the same-or-smaller shape
    /// reuse the underlying `Vec`s with zero allocations.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grow every internal scratch buffer to the shape required by `plan`
    /// and zero the predictor so each decode starts clean.
    pub(crate) fn prepare_for(
        &mut self,
        plan: &PreparedDecodePlan,
        mcus_per_row: u32,
        block_size: u32,
        max_bytes: usize,
    ) -> Result<(), JpegError> {
        let layout = SequentialScratchLayout::for_plan(plan, mcus_per_row, block_size)?;
        let target_bytes = layout.minimum_bytes(self.detached_sink_bytes)?;
        ensure_request_bytes(target_bytes, max_bytes)?;
        if self.retained_bytes() > target_bytes
            || self.projected_sequential_bytes(layout) > target_bytes
            || self.sequential_requires_growth(layout)
        {
            self.release_retained_allocations();
        }
        ensure_request_bytes(self.projected_sequential_bytes(layout), target_bytes)?;
        if let Err(error) = self.reserve_sequential_storage(layout, target_bytes) {
            self.release_retained_allocations();
            return Err(error);
        }

        self.prev_dc.resize(layout.component_count, 0);
        resize_rows(&mut self.ycbcr_420_rows, layout.row_width);
        resize_generic_rows(&mut self.ycbcr_generic_rows, layout.row_width);
        resize_rgb_rows(&mut self.rgb_generic_rows, layout.row_width);
        for dc in &mut self.prev_dc {
            *dc = 0;
        }
        self.ensure_retained_capacity(target_bytes)?;
        Ok(())
    }

    pub(crate) fn take_sink_rows(
        &mut self,
        row_len: usize,
        max_bytes: usize,
    ) -> Result<SinkRows, JpegError> {
        if self.detached_sink_bytes != 0 {
            return Err(JpegError::InternalInvariant {
                reason: "scratch sink rows are already detached",
            });
        }
        let sink_projected = projected_vec_bytes(&self.sink_rows.top_row, row_len)
            .saturating_add(projected_vec_bytes(&self.sink_rows.bottom_row, row_len));
        let without_sink = self
            .retained_bytes()
            .saturating_sub(vec_bytes(&self.sink_rows.top_row))
            .saturating_sub(vec_bytes(&self.sink_rows.bottom_row));
        if without_sink.saturating_add(sink_projected) > max_bytes
            || self.sink_requires_growth(row_len)
        {
            self.release_retained_allocations();
        }
        let minimum = row_len.checked_mul(2).ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: max_bytes,
        })?;
        ensure_request_bytes(minimum, max_bytes)?;
        let mut live_bytes = self.retained_bytes();
        let reserve_result = (|| {
            try_reserve_for_len_with_live_budget(
                &mut self.sink_rows.top_row,
                row_len,
                &mut live_bytes,
                max_bytes,
            )?;
            try_reserve_for_len_with_live_budget(
                &mut self.sink_rows.bottom_row,
                row_len,
                &mut live_bytes,
                max_bytes,
            )
        })();
        if let Err(error) = reserve_result {
            self.release_retained_allocations();
            return Err(error);
        }
        self.sink_rows.top_row.resize(row_len, 0);
        self.sink_rows.bottom_row.resize(row_len, 0);
        self.ensure_retained_capacity(max_bytes)?;
        let rows = core::mem::take(&mut self.sink_rows);
        self.detached_sink_bytes =
            vec_bytes(&rows.top_row).saturating_add(vec_bytes(&rows.bottom_row));
        Ok(rows)
    }

    pub(crate) fn prepare_lossless_rows(
        &mut self,
        predictor_row_len: usize,
        sink_row_len: usize,
        max_bytes: usize,
    ) -> Result<SinkRows, JpegError> {
        if self.detached_sink_bytes != 0 {
            return Err(JpegError::InternalInvariant {
                reason: "scratch sink rows are already detached",
            });
        }
        if sink_row_len == 0 {
            self.sink_rows = SinkRows::default();
        }
        let minimum = predictor_row_len
            .checked_mul(2)
            .and_then(|bytes| {
                sink_row_len
                    .checked_mul(2)
                    .and_then(|sink| bytes.checked_add(sink))
            })
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: max_bytes,
            })?;
        ensure_request_bytes(minimum, max_bytes)?;
        if self.projected_lossless_bytes(predictor_row_len, sink_row_len) > max_bytes
            || self.lossless_requires_growth(predictor_row_len, sink_row_len)
        {
            self.release_retained_allocations();
        }
        ensure_request_bytes(
            self.projected_lossless_bytes(predictor_row_len, sink_row_len),
            max_bytes,
        )?;
        let mut live_bytes = self.retained_bytes();
        let reserve_result = (|| {
            try_reserve_for_len_with_live_budget(
                &mut self.lossless_prev_row,
                predictor_row_len,
                &mut live_bytes,
                max_bytes,
            )?;
            try_reserve_for_len_with_live_budget(
                &mut self.lossless_curr_row,
                predictor_row_len,
                &mut live_bytes,
                max_bytes,
            )?;
            try_reserve_for_len_with_live_budget(
                &mut self.sink_rows.top_row,
                sink_row_len,
                &mut live_bytes,
                max_bytes,
            )?;
            try_reserve_for_len_with_live_budget(
                &mut self.sink_rows.bottom_row,
                sink_row_len,
                &mut live_bytes,
                max_bytes,
            )
        })();
        if let Err(error) = reserve_result {
            self.release_retained_allocations();
            return Err(error);
        }
        self.lossless_prev_row.resize(predictor_row_len, 0);
        self.lossless_curr_row.resize(predictor_row_len, 0);
        self.sink_rows.top_row.resize(sink_row_len, 0);
        self.sink_rows.bottom_row.resize(sink_row_len, 0);
        self.ensure_retained_capacity(max_bytes)?;
        let rows = core::mem::take(&mut self.sink_rows);
        self.detached_sink_bytes =
            vec_bytes(&rows.top_row).saturating_add(vec_bytes(&rows.bottom_row));
        Ok(rows)
    }

    pub(crate) fn restore_sink_rows(&mut self, rows: SinkRows) {
        self.sink_rows = rows;
        self.detached_sink_bytes = 0;
    }

    pub(crate) const fn detached_sink_bytes(&self) -> usize {
        self.detached_sink_bytes
    }

    pub(crate) fn reconcile_external_workspace(
        &mut self,
        external_bytes: usize,
        max_bytes: usize,
    ) -> Result<(), JpegError> {
        ensure_request_bytes(external_bytes, max_bytes)?;
        if self.retained_bytes().saturating_add(external_bytes) > max_bytes {
            self.release_retained_allocations();
        }
        ensure_request_bytes(
            self.retained_bytes().saturating_add(external_bytes),
            max_bytes,
        )
    }

    pub(crate) fn release_for_external_workspace(
        &mut self,
        external_bytes: usize,
        max_bytes: usize,
    ) -> Result<(), JpegError> {
        ensure_request_bytes(external_bytes, max_bytes)?;
        self.release_retained_allocations();
        ensure_request_bytes(
            self.detached_sink_bytes.saturating_add(external_bytes),
            max_bytes,
        )
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        let mut total = vec_bytes(&self.prev_dc);
        total = total
            .saturating_add(self.stripe_a.retained_bytes())
            .saturating_add(self.stripe_b.retained_bytes())
            .saturating_add(self.stripe_c.retained_bytes())
            .saturating_add(rows_bytes(&self.ycbcr_420_rows))
            .saturating_add(generic_rows_bytes(&self.ycbcr_generic_rows))
            .saturating_add(rgb_rows_bytes(&self.rgb_generic_rows))
            .saturating_add(vec_bytes(&self.lossless_prev_row))
            .saturating_add(vec_bytes(&self.lossless_curr_row))
            .saturating_add(vec_bytes(&self.sink_rows.top_row))
            .saturating_add(vec_bytes(&self.sink_rows.bottom_row))
            .saturating_add(self.detached_sink_bytes);
        total
    }

    fn projected_sequential_bytes(&self, layout: SequentialScratchLayout) -> usize {
        let mut total = projected_vec_bytes(&self.prev_dc, layout.component_count);
        total = total
            .saturating_add(self.stripe_a.projected_bytes(layout.stripe))
            .saturating_add(self.stripe_b.projected_bytes(layout.stripe))
            .saturating_add(self.stripe_c.projected_bytes(layout.stripe))
            .saturating_add(projected_rows_bytes(&self.ycbcr_420_rows, layout.row_width))
            .saturating_add(projected_generic_rows_bytes(
                &self.ycbcr_generic_rows,
                layout.row_width,
            ))
            .saturating_add(projected_rgb_rows_bytes(
                &self.rgb_generic_rows,
                layout.row_width,
            ))
            .saturating_add(vec_bytes(&self.lossless_prev_row))
            .saturating_add(vec_bytes(&self.lossless_curr_row))
            .saturating_add(vec_bytes(&self.sink_rows.top_row))
            .saturating_add(vec_bytes(&self.sink_rows.bottom_row))
            .saturating_add(self.detached_sink_bytes);
        total
    }

    fn projected_lossless_bytes(&self, predictor_row_len: usize, sink_row_len: usize) -> usize {
        self.retained_bytes()
            .saturating_sub(vec_bytes(&self.lossless_prev_row))
            .saturating_sub(vec_bytes(&self.lossless_curr_row))
            .saturating_sub(vec_bytes(&self.sink_rows.top_row))
            .saturating_sub(vec_bytes(&self.sink_rows.bottom_row))
            .saturating_add(projected_vec_bytes(
                &self.lossless_prev_row,
                predictor_row_len,
            ))
            .saturating_add(projected_vec_bytes(
                &self.lossless_curr_row,
                predictor_row_len,
            ))
            .saturating_add(projected_vec_bytes(&self.sink_rows.top_row, sink_row_len))
            .saturating_add(projected_vec_bytes(
                &self.sink_rows.bottom_row,
                sink_row_len,
            ))
    }

    fn reserve_sequential_storage(
        &mut self,
        layout: SequentialScratchLayout,
        cap: usize,
    ) -> Result<(), JpegError> {
        let mut live_bytes = self.retained_bytes();
        try_reserve_for_len_with_live_budget(
            &mut self.prev_dc,
            layout.component_count,
            &mut live_bytes,
            cap,
        )?;
        reserve_rows(
            &mut self.ycbcr_420_rows,
            layout.row_width,
            &mut live_bytes,
            cap,
        )?;
        reserve_generic_rows(
            &mut self.ycbcr_generic_rows,
            layout.row_width,
            &mut live_bytes,
            cap,
        )?;
        reserve_rgb_rows(
            &mut self.rgb_generic_rows,
            layout.row_width,
            &mut live_bytes,
            cap,
        )?;
        self.stripe_a
            .resize_for(layout.stripe, &mut live_bytes, cap)?;
        self.stripe_b
            .resize_for(layout.stripe, &mut live_bytes, cap)?;
        self.stripe_c
            .resize_for(layout.stripe, &mut live_bytes, cap)?;
        ensure_request_bytes(live_bytes, cap)
    }

    fn sequential_requires_growth(&self, layout: SequentialScratchLayout) -> bool {
        self.prev_dc.capacity() < layout.component_count
            || rows_require_growth(&self.ycbcr_420_rows, layout.row_width)
            || generic_rows_require_growth(&self.ycbcr_generic_rows, layout.row_width)
            || rgb_rows_require_growth(&self.rgb_generic_rows, layout.row_width)
            || self.stripe_a.requires_growth(layout.stripe)
            || self.stripe_b.requires_growth(layout.stripe)
            || self.stripe_c.requires_growth(layout.stripe)
    }

    fn sink_requires_growth(&self, row_len: usize) -> bool {
        self.sink_rows.top_row.capacity() < row_len
            || self.sink_rows.bottom_row.capacity() < row_len
    }

    fn lossless_requires_growth(&self, predictor_row_len: usize, sink_row_len: usize) -> bool {
        self.lossless_prev_row.capacity() < predictor_row_len
            || self.lossless_curr_row.capacity() < predictor_row_len
            || self.sink_rows.top_row.capacity() < sink_row_len
            || self.sink_rows.bottom_row.capacity() < sink_row_len
    }

    fn release_retained_allocations(&mut self) {
        self.prev_dc = Vec::new();
        self.stripe_a = StripeBuffer::default();
        self.stripe_b = StripeBuffer::default();
        self.stripe_c = StripeBuffer::default();
        self.ycbcr_420_rows = YCbCr420Rows::default();
        self.ycbcr_generic_rows = YCbCrGenericRows::default();
        self.rgb_generic_rows = RgbGenericRows::default();
        self.lossless_prev_row = Vec::new();
        self.lossless_curr_row = Vec::new();
        self.sink_rows = SinkRows::default();
    }

    fn ensure_retained_capacity(&mut self, cap: usize) -> Result<(), JpegError> {
        let requested = self.retained_bytes();
        if requested > cap {
            self.release_retained_allocations();
            return Err(JpegError::MemoryCapExceeded { requested, cap });
        }
        Ok(())
    }
}

fn ensure_request_bytes(requested: usize, cap: usize) -> Result<(), JpegError> {
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(())
}

fn vec_bytes<T>(vec: &Vec<T>) -> usize {
    vec.capacity().saturating_mul(core::mem::size_of::<T>())
}

fn projected_vec_bytes<T>(vec: &Vec<T>, target_len: usize) -> usize {
    vec.capacity()
        .max(target_len)
        .saturating_mul(core::mem::size_of::<T>())
}

fn reserve_rows(
    rows: &mut YCbCr420Rows,
    width: usize,
    live_bytes: &mut usize,
    cap: usize,
) -> Result<(), JpegError> {
    try_reserve_for_len_with_live_budget(&mut rows.cb_top, width, live_bytes, cap)?;
    try_reserve_for_len_with_live_budget(&mut rows.cb_bot, width, live_bytes, cap)?;
    try_reserve_for_len_with_live_budget(&mut rows.cr_top, width, live_bytes, cap)?;
    try_reserve_for_len_with_live_budget(&mut rows.cr_bot, width, live_bytes, cap)
}

fn resize_rows(rows: &mut YCbCr420Rows, width: usize) {
    rows.cb_top.resize(width, 0);
    rows.cb_bot.resize(width, 0);
    rows.cr_top.resize(width, 0);
    rows.cr_bot.resize(width, 0);
}

fn rows_bytes(rows: &YCbCr420Rows) -> usize {
    vec_bytes(&rows.cb_top)
        .saturating_add(vec_bytes(&rows.cb_bot))
        .saturating_add(vec_bytes(&rows.cr_top))
        .saturating_add(vec_bytes(&rows.cr_bot))
}

fn projected_rows_bytes(rows: &YCbCr420Rows, width: usize) -> usize {
    projected_vec_bytes(&rows.cb_top, width)
        .saturating_add(projected_vec_bytes(&rows.cb_bot, width))
        .saturating_add(projected_vec_bytes(&rows.cr_top, width))
        .saturating_add(projected_vec_bytes(&rows.cr_bot, width))
}

fn rows_require_growth(rows: &YCbCr420Rows, width: usize) -> bool {
    rows.cb_top.capacity() < width
        || rows.cb_bot.capacity() < width
        || rows.cr_top.capacity() < width
        || rows.cr_bot.capacity() < width
}

fn reserve_generic_rows(
    rows: &mut YCbCrGenericRows,
    width: usize,
    live_bytes: &mut usize,
    cap: usize,
) -> Result<(), JpegError> {
    try_reserve_for_len_with_live_budget(&mut rows.cb_up, width, live_bytes, cap)?;
    try_reserve_for_len_with_live_budget(&mut rows.cr_up, width, live_bytes, cap)
}

fn resize_generic_rows(rows: &mut YCbCrGenericRows, width: usize) {
    rows.cb_up.resize(width, 0);
    rows.cr_up.resize(width, 0);
}

fn generic_rows_bytes(rows: &YCbCrGenericRows) -> usize {
    vec_bytes(&rows.cb_up).saturating_add(vec_bytes(&rows.cr_up))
}

fn projected_generic_rows_bytes(rows: &YCbCrGenericRows, width: usize) -> usize {
    projected_vec_bytes(&rows.cb_up, width).saturating_add(projected_vec_bytes(&rows.cr_up, width))
}

fn generic_rows_require_growth(rows: &YCbCrGenericRows, width: usize) -> bool {
    rows.cb_up.capacity() < width || rows.cr_up.capacity() < width
}

fn reserve_rgb_rows(
    rows: &mut RgbGenericRows,
    width: usize,
    live_bytes: &mut usize,
    cap: usize,
) -> Result<(), JpegError> {
    try_reserve_for_len_with_live_budget(&mut rows.r, width, live_bytes, cap)?;
    try_reserve_for_len_with_live_budget(&mut rows.g, width, live_bytes, cap)?;
    try_reserve_for_len_with_live_budget(&mut rows.b, width, live_bytes, cap)?;
    try_reserve_for_len_with_live_budget(&mut rows.k, width, live_bytes, cap)
}

fn resize_rgb_rows(rows: &mut RgbGenericRows, width: usize) {
    rows.r.resize(width, 0);
    rows.g.resize(width, 0);
    rows.b.resize(width, 0);
    rows.k.resize(width, 0);
}

fn rgb_rows_bytes(rows: &RgbGenericRows) -> usize {
    vec_bytes(&rows.r)
        .saturating_add(vec_bytes(&rows.g))
        .saturating_add(vec_bytes(&rows.b))
        .saturating_add(vec_bytes(&rows.k))
}

fn projected_rgb_rows_bytes(rows: &RgbGenericRows, width: usize) -> usize {
    projected_vec_bytes(&rows.r, width)
        .saturating_add(projected_vec_bytes(&rows.g, width))
        .saturating_add(projected_vec_bytes(&rows.b, width))
        .saturating_add(projected_vec_bytes(&rows.k, width))
}

fn rgb_rows_require_growth(rows: &RgbGenericRows, width: usize) -> bool {
    rows.r.capacity() < width
        || rows.g.capacity() < width
        || rows.b.capacity() < width
        || rows.k.capacity() < width
}

#[doc(hidden)]
impl CoreScratchPool for ScratchPool {
    fn bytes_allocated(&self) -> usize {
        self.retained_bytes()
    }

    fn reset(&mut self) {
        fn clear_stripe(stripe: &mut StripeBuffer) {
            for plane in &mut stripe.planes {
                plane.clear();
            }
            stripe.plane_strides.clear();
            stripe.plane_rows.clear();
        }

        self.prev_dc.clear();
        clear_stripe(&mut self.stripe_a);
        clear_stripe(&mut self.stripe_b);
        clear_stripe(&mut self.stripe_c);
        self.ycbcr_420_rows.cb_top.clear();
        self.ycbcr_420_rows.cb_bot.clear();
        self.ycbcr_420_rows.cr_top.clear();
        self.ycbcr_420_rows.cr_bot.clear();
        self.ycbcr_generic_rows.cb_up.clear();
        self.ycbcr_generic_rows.cr_up.clear();
        self.rgb_generic_rows.r.clear();
        self.rgb_generic_rows.g.clear();
        self.rgb_generic_rows.b.clear();
        self.rgb_generic_rows.k.clear();
        self.lossless_prev_row.clear();
        self.lossless_curr_row.clear();
        self.sink_rows.top_row.clear();
        self.sink_rows.bottom_row.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{JpegError, ScratchPool};
    use alloc::vec::Vec;

    #[test]
    fn sink_rows_have_an_exact_aggregate_cap_boundary() {
        let mut pool = ScratchPool::new();
        let rows = pool.take_sink_rows(8, 16).expect("exact boundary");
        assert_eq!(rows.top_row.len(), 8);
        assert_eq!(rows.bottom_row.len(), 8);
        pool.restore_sink_rows(rows);

        let mut pool = ScratchPool::new();
        assert!(matches!(
            pool.take_sink_rows(9, 17),
            Err(JpegError::MemoryCapExceeded {
                requested: 18,
                cap: 17
            })
        ));
    }

    #[test]
    fn same_or_smaller_sink_rows_reuse_capacity_within_cap() {
        let mut pool = ScratchPool::new();
        let rows = pool.take_sink_rows(8, 16).expect("seed sink rows");
        let top_ptr = rows.top_row.as_ptr();
        let bottom_ptr = rows.bottom_row.as_ptr();
        pool.restore_sink_rows(rows);

        let rows = pool.take_sink_rows(4, 16).expect("reuse smaller rows");
        assert_eq!(rows.top_row.as_ptr(), top_ptr);
        assert_eq!(rows.bottom_row.as_ptr(), bottom_ptr);
    }

    #[test]
    fn stale_capacity_is_released_instead_of_rejecting_a_valid_request() {
        let mut pool = ScratchPool::new();
        let rows = pool.take_sink_rows(32, 64).expect("seed retained rows");
        pool.restore_sink_rows(rows);
        assert!(pool.retained_bytes() >= 64);

        pool.reconcile_external_workspace(64, 64)
            .expect("stale capacity must not change request acceptance");
        assert_eq!(pool.retained_bytes(), 0);
    }

    #[test]
    fn detached_rows_remain_part_of_the_live_pool_budget() {
        let mut pool = ScratchPool::new();
        let rows = pool.take_sink_rows(8, 16).expect("detach rows");
        assert!(pool.retained_bytes() >= 16);
        pool.restore_sink_rows(rows);
    }

    #[test]
    fn actual_retained_capacity_is_checked_and_released() {
        let mut pool = ScratchPool::new();
        pool.sink_rows.top_row = Vec::with_capacity(17);
        assert!(matches!(
            pool.ensure_retained_capacity(16),
            Err(JpegError::MemoryCapExceeded {
                requested: 17,
                cap: 16
            })
        ));
        assert_eq!(pool.retained_bytes(), 0);
    }

    #[test]
    fn non_pool_decode_releases_stale_capacity_even_when_it_would_fit() {
        let mut pool = ScratchPool::new();
        pool.sink_rows.top_row = Vec::with_capacity(64);

        pool.release_for_external_workspace(1, 128)
            .expect("external phase fits after stale storage is released");

        assert_eq!(pool.retained_bytes(), 0);
    }
}
