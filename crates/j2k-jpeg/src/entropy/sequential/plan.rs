// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared baseline component metadata and fast-path shape predicates.

use alloc::vec::Vec;

use crate::allocation::{checked_add_allocation_bytes, checked_allocation_bytes};
use crate::entropy::huffman::{HuffmanTable, PreparedHuffmanTableId, PreparedHuffmanTables};
use crate::error::JpegError;
use crate::info::{ColorSpace, SamplingFactors};

use super::layout::is_ycbcr_420;

mod resolved;
pub(crate) use self::resolved::ResolvedPreparedComponentPlan;

/// Per-component decode context. One entry per component declared in the
/// SOF, in scan order.
#[derive(Debug, Clone)]
pub(crate) struct PreparedComponentPlan {
    pub(crate) h: u8,
    pub(crate) v: u8,
    pub(crate) output_index: usize,
    pub(crate) quant: [u16; 64],
    pub(crate) dc_table: Option<PreparedHuffmanTableId>,
    pub(crate) ac_table: Option<PreparedHuffmanTableId>,
}

#[derive(Debug)]
pub(crate) struct PreparedDecodePlan {
    pub(crate) components: Vec<PreparedComponentPlan>,
    pub(crate) huffman_tables: PreparedHuffmanTables,
    pub(crate) sampling: SamplingFactors,
    pub(crate) color_space: ColorSpace,
    pub(crate) restart_interval: Option<u16>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) scan_offset: usize,
    pub(crate) scratch_bytes: usize,
}

impl PreparedDecodePlan {
    pub(crate) fn allocation_bytes_for_counts(
        component_count: usize,
        huffman_table_count: usize,
    ) -> Result<usize, JpegError> {
        let mut total = checked_allocation_bytes::<PreparedComponentPlan>(component_count)?;
        total = checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<HuffmanTable>(huffman_table_count)?,
        )?;
        Ok(total)
    }

    pub(crate) fn retained_allocation_bytes(&self) -> Result<usize, JpegError> {
        checked_add_allocation_bytes(
            checked_allocation_bytes::<PreparedComponentPlan>(self.components.capacity())?,
            self.huffman_tables.retained_allocation_bytes()?,
        )
    }

    pub(crate) fn matches_fast_tile_shape(&self) -> bool {
        self.restart_interval.is_none()
            && is_ycbcr_420(self)
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 2
            && self.components[0].v == 2
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }

    pub(crate) fn matches_fast_rgb444_shape(&self) -> bool {
        self.color_space == ColorSpace::YCbCr
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 1
            && self.components[0].v == 1
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }

    pub(crate) fn matches_fast_rgb422_shape(&self) -> bool {
        self.color_space == ColorSpace::YCbCr
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 2
            && self.components[0].v == 1
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }
}
