// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stripe layout planning and reusable plane storage.

use alloc::vec::Vec;

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, checked_allocation_len,
    try_reserve_for_len_with_live_budget,
};
use crate::error::JpegError;
use crate::info::SamplingFactors;

use super::PreparedDecodePlan;

#[derive(Debug, Default)]
pub(crate) struct StripeBuffer {
    pub(crate) planes: Vec<Vec<u8>>,
    pub(crate) plane_strides: Vec<usize>,
    pub(crate) plane_rows: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StripeLayout {
    component_count: usize,
    plane_lens: [usize; 4],
    plane_strides: [usize; 4],
    plane_rows: [usize; 4],
}

impl StripeLayout {
    pub(crate) fn for_plan(
        plan: &PreparedDecodePlan,
        mcus_per_row: u32,
        block_size: u32,
    ) -> Result<Self, JpegError> {
        Self::for_sampling(plan.sampling, mcus_per_row, block_size)
    }

    pub(crate) fn for_sampling(
        sampling: SamplingFactors,
        mcus_per_row: u32,
        block_size: u32,
    ) -> Result<Self, JpegError> {
        let component_count = sampling.len();
        if component_count > 4 {
            return Err(JpegError::UnsupportedComponentCount {
                count: u8::try_from(component_count).unwrap_or(u8::MAX),
            });
        }
        let mut plane_lens = [0; 4];
        let mut plane_strides = [0; 4];
        let mut plane_rows = [0; 4];
        for (index, (h, v)) in sampling.iter().enumerate() {
            let stride = checked_allocation_len::<u8>(
                mcus_per_row as usize,
                usize::from(h).checked_mul(block_size as usize).ok_or(
                    JpegError::MemoryCapExceeded {
                        requested: usize::MAX,
                        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                    },
                )?,
            )?;
            let rows = checked_allocation_len::<u8>(usize::from(v), block_size as usize)?;
            plane_lens[index] = checked_allocation_len::<u8>(stride, rows)?;
            plane_strides[index] = stride;
            plane_rows[index] = rows;
        }
        Ok(Self {
            component_count,
            plane_lens,
            plane_strides,
            plane_rows,
        })
    }

    pub(crate) fn allocation_bytes(self) -> Result<usize, JpegError> {
        let mut total = checked_allocation_bytes::<Vec<u8>>(self.component_count)?;
        total = checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<usize>(self.component_count)?,
        )?;
        total = checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<usize>(self.component_count)?,
        )?;
        for &plane_len in &self.plane_lens[..self.component_count] {
            total = checked_add_allocation_bytes(total, plane_len)?;
        }
        Ok(total)
    }
}

#[derive(Clone, Copy)]
pub(super) struct StripePlane<'a> {
    pub(super) data: &'a [u8],
    pub(super) stride: usize,
    pub(super) rows: usize,
}

impl StripeBuffer {
    /// Grow each backing vector to the requested layout without shrinking
    /// retained storage between tiles.
    pub(crate) fn resize_for(
        &mut self,
        layout: StripeLayout,
        live_bytes: &mut usize,
        cap: usize,
    ) -> Result<(), JpegError> {
        let n = layout.component_count;
        try_reserve_for_len_with_live_budget(&mut self.planes, n, live_bytes, cap)?;
        self.planes.resize_with(n, Vec::new);
        try_reserve_for_len_with_live_budget(&mut self.plane_strides, n, live_bytes, cap)?;
        self.plane_strides.resize(n, 0);
        try_reserve_for_len_with_live_budget(&mut self.plane_rows, n, live_bytes, cap)?;
        self.plane_rows.resize(n, 0);
        for (plane, &target_len) in self.planes.iter_mut().zip(&layout.plane_lens[..n]) {
            try_reserve_for_len_with_live_budget(plane, target_len, live_bytes, cap)?;
        }
        for index in 0..n {
            self.planes[index].resize(layout.plane_lens[index], 0);
            self.plane_strides[index] = layout.plane_strides[index];
            self.plane_rows[index] = layout.plane_rows[index];
        }
        Ok(())
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        let mut total = self
            .planes
            .capacity()
            .saturating_mul(core::mem::size_of::<Vec<u8>>())
            .saturating_add(
                self.plane_strides
                    .capacity()
                    .saturating_mul(core::mem::size_of::<usize>()),
            )
            .saturating_add(
                self.plane_rows
                    .capacity()
                    .saturating_mul(core::mem::size_of::<usize>()),
            );
        for plane in &self.planes {
            total = total.saturating_add(plane.capacity());
        }
        total
    }

    pub(crate) fn projected_bytes(&self, layout: StripeLayout) -> usize {
        let n = layout.component_count;
        let mut total = self
            .planes
            .capacity()
            .max(n)
            .saturating_mul(core::mem::size_of::<Vec<u8>>())
            .saturating_add(
                self.plane_strides
                    .capacity()
                    .max(n)
                    .saturating_mul(core::mem::size_of::<usize>()),
            )
            .saturating_add(
                self.plane_rows
                    .capacity()
                    .max(n)
                    .saturating_mul(core::mem::size_of::<usize>()),
            );
        for (index, &target_len) in layout.plane_lens[..n].iter().enumerate() {
            let retained = self.planes.get(index).map_or(0, alloc::vec::Vec::capacity);
            total = total.saturating_add(retained.max(target_len));
        }
        total
    }

    pub(crate) fn requires_growth(&self, layout: StripeLayout) -> bool {
        let n = layout.component_count;
        self.planes.capacity() < n
            || self.plane_strides.capacity() < n
            || self.plane_rows.capacity() < n
            || layout.plane_lens[..n]
                .iter()
                .enumerate()
                .any(|(index, &target_len)| match self.planes.get(index) {
                    Some(plane) => plane.capacity() < target_len,
                    None => target_len != 0,
                })
    }

    pub(super) fn row_count(&self, plane_idx: usize) -> usize {
        self.plane_rows[plane_idx]
    }

    pub(super) fn row(&self, plane_idx: usize, row: usize) -> &[u8] {
        let stride = self.plane_strides[plane_idx];
        let start = row * stride;
        &self.planes[plane_idx][start..start + stride]
    }

    pub(super) fn plane(&self, plane_idx: usize) -> StripePlane<'_> {
        StripePlane {
            data: &self.planes[plane_idx],
            stride: self.plane_strides[plane_idx],
            rows: self.plane_rows[plane_idx],
        }
    }
}
