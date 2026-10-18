// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared baseline component metadata and fast-path shape predicates.

use alloc::vec::Vec;

use crate::allocation::{checked_add_allocation_bytes, checked_allocation_bytes};
use crate::entropy::huffman::{
    HuffmanLookupRequirement, PreparedHuffmanTableId, PreparedHuffmanTables,
};
use crate::error::JpegError;
use crate::info::{ColorSpace, SamplingFactors};

use super::layout::is_ycbcr_420;

mod resolved;
pub(crate) use self::resolved::ResolvedPreparedComponentPlan;

/// Primary Huffman width selected once while preparing a sequential plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HuffmanPrimaryWidth {
    Narrow8,
    Full12,
}

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
    pub(crate) huffman_primary_width: HuffmanPrimaryWidth,
    pub(crate) sampling: SamplingFactors,
    pub(crate) color_space: ColorSpace,
    pub(crate) restart_interval: Option<u16>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) scan_offset: usize,
    pub(crate) scratch_bytes: usize,
}

impl PreparedDecodePlan {
    pub(crate) fn huffman_primary_width_for_tables(
        requirement: HuffmanLookupRequirement,
        tables: &PreparedHuffmanTables,
    ) -> HuffmanPrimaryWidth {
        if requirement == HuffmanLookupRequirement::FusedNarrow
            && !tables.all_satisfy_lookup_requirement(HuffmanLookupRequirement::FusedFull)
        {
            HuffmanPrimaryWidth::Narrow8
        } else {
            HuffmanPrimaryWidth::Full12
        }
    }

    fn coded_block_count(
        dimensions: (u32, u32),
        sampling: SamplingFactors,
        blocks_per_mcu: u64,
    ) -> u128 {
        let columns = u64::from(dimensions.0).div_ceil(8 * u64::from(sampling.max_h));
        let rows = u64::from(dimensions.1).div_ceil(8 * u64::from(sampling.max_v));
        u128::from(columns) * u128::from(rows) * u128::from(blocks_per_mcu)
    }

    /// Whether header preparation must request full Huffman lookups for this
    /// coded geometry. Below 512 blocks, a narrow request bounds cold
    /// construction for custom tables; static standard tables and stronger
    /// cached tables still retain their full capability.
    pub(crate) fn can_select_wide_huffman(
        dimensions: (u32, u32),
        sampling: SamplingFactors,
        blocks_per_mcu: u64,
    ) -> bool {
        Self::coded_block_count(dimensions, sampling, blocks_per_mcu) >= 512
    }

    /// Wide fused lookups pay off for densely coded blocks. Sparse scans use
    /// the smaller table to keep the common working set in the L1 data cache.
    /// The estimate includes restart stuffing and any trailer, so it changes
    /// only the strategy, never parsing or decoded values. Evaluate per scan:
    /// prepared plans are cached by headers shared by different image contents.
    pub(crate) fn prefers_wide_huffman(&self, scan_bytes: usize) -> bool {
        let blocks_per_mcu: u64 = self
            .components
            .iter()
            .map(|component| u64::from(component.h) * u64::from(component.v))
            .sum();
        let blocks = Self::coded_block_count(self.dimensions, self.sampling, blocks_per_mcu);
        blocks >= 64 && scan_bytes as u128 >= blocks * 64
    }

    pub(crate) fn allocation_bytes_for_counts(
        component_count: usize,
        huffman_table_count: usize,
    ) -> Result<usize, JpegError> {
        let mut total = checked_allocation_bytes::<PreparedComponentPlan>(component_count)?;
        total = checked_add_allocation_bytes(
            total,
            PreparedHuffmanTables::maximum_allocation_bytes_for_capacity(huffman_table_count)?,
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
        // The CPU fused route cannot cross restart boundaries; accelerator
        // packets retain restart checkpoints and use the shape predicate below.
        self.restart_interval.is_none() && self.matches_fast_420_device_shape()
    }

    /// Coded 4:2:0 YCbCr whose full-size chroma libjpeg-turbo smooths with
    /// `h2v2_fancy_upsample`, the only upsampler the fused 4:2:0 routes
    /// implement. Chroma at most two samples wide (images at most four pixels
    /// wide) is replicated instead and takes the generic route.
    pub(crate) fn matches_fast_420_device_shape(&self) -> bool {
        self.dimensions.0 > 4
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

    /// Coded 4:2:2 YCbCr whose full-size chroma libjpeg-turbo smooths (more
    /// than two chroma samples wide), the shape the device 4:2:2 kernels
    /// implement.
    pub(crate) fn matches_fast_rgb422_shape(&self) -> bool {
        self.dimensions.0 > 4
            && self.color_space == ColorSpace::YCbCr
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

#[cfg(test)]
mod tests {
    use super::{HuffmanPrimaryWidth, PreparedDecodePlan};
    use crate::entropy::huffman::{HuffmanLookupRequirement, HuffmanTable, PreparedHuffmanTables};
    use crate::info::SamplingFactors;
    use crate::parse::tables::{HuffmanTableRole, HuffmanValues, RawHuffmanTable};
    use crate::standard_huffman::{STD_LUMA_DC_BITS, STD_LUMA_DC_VALUES};

    fn arena_with(table: HuffmanTable) -> PreparedHuffmanTables {
        let mut tables = PreparedHuffmanTables::try_with_capacity(1).expect("one table arena");
        tables.push(table).expect("one table");
        tables
    }

    #[test]
    fn wide_huffman_geometry_threshold_is_exactly_512_blocks() {
        let gray = SamplingFactors::from_validated_components(&[(1, 1)]);
        assert!(!PreparedDecodePlan::can_select_wide_huffman(
            (56, 584),
            gray,
            1
        ));
        assert!(PreparedDecodePlan::can_select_wide_huffman(
            (64, 512),
            gray,
            1
        ));

        let standard = RawHuffmanTable {
            bits: STD_LUMA_DC_BITS,
            values: HuffmanValues::from_slice(&STD_LUMA_DC_VALUES),
        };
        let static_full = arena_with(
            HuffmanTable::from_raw(&standard, HuffmanTableRole::Dc).expect("standard table"),
        );
        assert_eq!(
            PreparedDecodePlan::huffman_primary_width_for_tables(
                HuffmanLookupRequirement::FusedNarrow,
                &static_full,
            ),
            HuffmanPrimaryWidth::Full12
        );

        let custom = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0]),
        };
        let mut live_bytes = 0;
        let dynamic_narrow = arena_with(
            HuffmanTable::from_raw_with_requirement_and_live_budget(
                &custom,
                HuffmanTableRole::Dc,
                HuffmanLookupRequirement::FusedNarrow,
                &mut live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )
            .expect("narrow table"),
        );
        assert_eq!(
            PreparedDecodePlan::huffman_primary_width_for_tables(
                HuffmanLookupRequirement::FusedNarrow,
                &dynamic_narrow,
            ),
            HuffmanPrimaryWidth::Narrow8
        );
        assert_eq!(
            PreparedDecodePlan::huffman_primary_width_for_tables(
                HuffmanLookupRequirement::FusedFull,
                &dynamic_narrow,
            ),
            HuffmanPrimaryWidth::Full12
        );
    }
}
