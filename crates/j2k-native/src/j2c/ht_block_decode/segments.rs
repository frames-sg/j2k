// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::CodeBlock;
use super::super::decode::DecompositionStorage;
use crate::error::{bail, DecodingError, Result};

mod owned;

pub(crate) use owned::{
    collect_code_block_data, collect_code_block_data_into, selected_code_block_segment_lengths,
};

pub(crate) struct CombinedCodeBlockData {
    pub(crate) data: Vec<u8>,
    pub(crate) cleanup_length: u32,
    pub(crate) refinement_length: u32,
}

pub(crate) struct HtCodeBlockSegments<'a> {
    pub(crate) cleanup: &'a [u8],
    pub(crate) refinement: &'a [u8],
}

impl<'a> HtCodeBlockSegments<'a> {
    pub(crate) fn from_combined_payload(
        data: &'a [u8],
        cleanup_length: u32,
        refinement_length: u32,
    ) -> Result<Self> {
        let cleanup_len = cleanup_length as usize;
        let refinement_len = refinement_length as usize;
        let total_len = cleanup_len
            .checked_add(refinement_len)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if data.len() < total_len {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        Ok(Self {
            cleanup: &data[..cleanup_len],
            refinement: &data[cleanup_len..total_len],
        })
    }
}

#[cfg(test)]
impl CombinedCodeBlockData {
    pub(crate) fn segments(&self) -> Result<HtCodeBlockSegments<'_>> {
        HtCodeBlockSegments::from_combined_payload(
            &self.data,
            self.cleanup_length,
            self.refinement_length,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HtCodeBlockSegmentKind {
    Cleanup,
    Refinement,
}

pub(crate) fn visit_code_block_segments<'a>(
    code_block: &CodeBlock,
    storage: &'a DecompositionStorage<'a>,
    mut visit: impl FnMut(HtCodeBlockSegmentKind, &'a [u8]) -> Result<()>,
) -> Result<()> {
    let (cleanup_index, refinement_index) = selected_segment_indices(code_block)?;
    let mut cleanup_seen = false;

    for layer in &storage.layers[code_block.layers.start..code_block.layers.end] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };

        for segment in &storage.segments[range] {
            match segment.idx {
                idx if idx == cleanup_index && !cleanup_seen => {
                    cleanup_seen = true;
                    visit(HtCodeBlockSegmentKind::Cleanup, segment.data)?;
                }
                idx if idx == refinement_index && cleanup_seen => {
                    visit(HtCodeBlockSegmentKind::Refinement, segment.data)?;
                }
                idx if idx < cleanup_index => {}
                idx if idx == cleanup_index => bail!(DecodingError::UnsupportedFeature(
                    "fragmented HTJ2K cleanup segment"
                )),
                idx if idx == refinement_index => bail!(DecodingError::UnsupportedFeature(
                    "fragmented HTJ2K refinement segment"
                )),
                _ => bail!(DecodingError::CodeBlockDecodeFailure),
            }
        }
    }

    if !cleanup_seen {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    Ok(())
}

pub(super) fn selected_segment_indices(code_block: &CodeBlock) -> Result<(u8, u8)> {
    let selected_set = code_block
        .ht_selected_set
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let cleanup_index = selected_set
        .checked_mul(2)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let refinement_index = cleanup_index
        .checked_add(1)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    Ok((cleanup_index, refinement_index))
}

#[cfg(test)]
mod tests;
