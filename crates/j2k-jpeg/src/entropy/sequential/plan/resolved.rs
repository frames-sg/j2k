// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked prepared-table resolution outside entropy hot loops.

use super::{PreparedComponentPlan, PreparedDecodePlan};
use crate::entropy::huffman::{HuffmanTable, PreparedHuffmanTableId};
use crate::error::JpegError;

/// Borrowed component metadata with table IDs resolved once before entering an
/// entropy hot loop.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedPreparedComponentPlan<'a> {
    pub(crate) quant: &'a [u16; 64],
    pub(crate) dc_table: &'a HuffmanTable,
    pub(crate) ac_table: &'a HuffmanTable,
}

impl PreparedDecodePlan {
    pub(crate) fn huffman_table(
        &self,
        id: Option<PreparedHuffmanTableId>,
    ) -> Result<&HuffmanTable, JpegError> {
        id.and_then(|id| self.huffman_tables.get(id))
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared component references a missing Huffman table",
            })
    }

    pub(crate) fn dc_table(
        &self,
        component: &PreparedComponentPlan,
    ) -> Result<&HuffmanTable, JpegError> {
        self.huffman_table(component.dc_table)
    }

    pub(crate) fn ac_table(
        &self,
        component: &PreparedComponentPlan,
    ) -> Result<&HuffmanTable, JpegError> {
        self.huffman_table(component.ac_table)
    }

    pub(crate) fn resolve_component<'a>(
        &'a self,
        component: &'a PreparedComponentPlan,
    ) -> Result<ResolvedPreparedComponentPlan<'a>, JpegError> {
        Ok(ResolvedPreparedComponentPlan {
            quant: &component.quant,
            dc_table: self.dc_table(component)?,
            ac_table: self.ac_table(component)?,
        })
    }

    pub(crate) fn resolved_component(
        &self,
        index: usize,
    ) -> Result<ResolvedPreparedComponentPlan<'_>, JpegError> {
        let component = self
            .components
            .get(index)
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared component index is outside the decode plan",
            })?;
        self.resolve_component(component)
    }
}
