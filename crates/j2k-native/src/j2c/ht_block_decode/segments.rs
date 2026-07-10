// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::CodeBlock;
use super::super::decode::DecompositionStorage;
use crate::error::{bail, DecodingError, Result};

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

pub(crate) fn collect_code_block_segments<'a>(
    code_block: &CodeBlock,
    storage: &'a DecompositionStorage<'a>,
) -> Result<HtCodeBlockSegments<'a>> {
    let mut cleanup = None;
    let mut refinement = None;

    for layer in &storage.layers[code_block.layers.start..code_block.layers.end] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };

        for segment in &storage.segments[range] {
            match segment.idx {
                0 if cleanup.is_none() => {
                    cleanup = Some(segment.data);
                }
                1 if refinement.is_none() => {
                    refinement = Some(segment.data);
                }
                _ => bail!(DecodingError::UnsupportedFeature(
                    "unexpected HTJ2K segment layout"
                )),
            }
        }
    }

    let Some(cleanup) = cleanup else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };

    Ok(HtCodeBlockSegments {
        cleanup,
        refinement: refinement.unwrap_or(&[]),
    })
}

pub(crate) fn collect_code_block_data<'a>(
    code_block: &CodeBlock,
    storage: &'a DecompositionStorage<'a>,
) -> Result<CombinedCodeBlockData> {
    let segments = collect_code_block_segments(code_block, storage)?;
    let cleanup_length =
        u32::try_from(segments.cleanup.len()).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let refinement_length = u32::try_from(segments.refinement.len())
        .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let mut data = Vec::with_capacity(segments.cleanup.len() + segments.refinement.len());
    data.extend_from_slice(segments.cleanup);
    data.extend_from_slice(segments.refinement);

    Ok(CombinedCodeBlockData {
        data,
        cleanup_length,
        refinement_length,
    })
}
