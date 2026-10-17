// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked prepared-table resolution outside entropy hot loops.

use super::{PreparedComponentPlan, PreparedDecodePlan};
use crate::entropy::huffman::{AcHuffmanTable, DcHuffmanTable, FULL_PRIMARY_ENTRIES};
use crate::error::JpegError;

/// Borrowed component metadata with table IDs resolved once before entering an
/// entropy hot loop.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedPreparedComponentPlan<'a, const N: usize = FULL_PRIMARY_ENTRIES> {
    pub(crate) quant: &'a [u16; 64],
    pub(crate) dc_table: DcHuffmanTable<'a, N>,
    pub(crate) ac_table: AcHuffmanTable<'a, N>,
}

impl<const N: usize> ResolvedPreparedComponentPlan<'_, N> {
    pub(crate) fn with_wide_huffman(mut self, wide: bool) -> Self {
        self.ac_table = self.ac_table.with_wide_lookup(wide);
        self
    }
}

impl PreparedDecodePlan {
    pub(crate) fn dc_table(
        &self,
        component: &PreparedComponentPlan,
    ) -> Result<DcHuffmanTable<'_>, JpegError> {
        self.dc_table_with_entries::<FULL_PRIMARY_ENTRIES>(component)
    }

    pub(crate) fn dc_table_with_entries<const N: usize>(
        &self,
        component: &PreparedComponentPlan,
    ) -> Result<DcHuffmanTable<'_, N>, JpegError> {
        self.huffman_tables
            .get_dc_with_entries::<N>(component.dc_table.ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })?)
    }

    pub(crate) fn ac_table_with_entries<const N: usize>(
        &self,
        component: &PreparedComponentPlan,
    ) -> Result<AcHuffmanTable<'_, N>, JpegError> {
        self.huffman_tables
            .get_ac_with_entries::<N>(component.ac_table.ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })?)
    }

    pub(crate) fn resolve_component<'a>(
        &'a self,
        component: &'a PreparedComponentPlan,
    ) -> Result<ResolvedPreparedComponentPlan<'a>, JpegError> {
        self.resolve_component_with_entries::<FULL_PRIMARY_ENTRIES>(component)
    }

    pub(crate) fn resolve_component_with_entries<'a, const N: usize>(
        &'a self,
        component: &'a PreparedComponentPlan,
    ) -> Result<ResolvedPreparedComponentPlan<'a, N>, JpegError> {
        Ok(ResolvedPreparedComponentPlan {
            quant: &component.quant,
            dc_table: self.dc_table_with_entries::<N>(component)?,
            ac_table: self.ac_table_with_entries::<N>(component)?,
        })
    }

    pub(crate) fn resolved_component(
        &self,
        index: usize,
    ) -> Result<ResolvedPreparedComponentPlan<'_>, JpegError> {
        self.resolved_component_with_entries::<FULL_PRIMARY_ENTRIES>(index)
    }

    pub(crate) fn resolved_component_with_entries<const N: usize>(
        &self,
        index: usize,
    ) -> Result<ResolvedPreparedComponentPlan<'_, N>, JpegError> {
        let component = self
            .components
            .get(index)
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared component index is outside the decode plan",
            })?;
        self.resolve_component_with_entries::<N>(component)
    }
}
