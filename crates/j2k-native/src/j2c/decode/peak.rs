// SPDX-License-Identifier: MIT OR Apache-2.0

//! Peak decode workspace, sampled at stage boundaries for tests and profiling.

use core::mem::size_of;

use super::{DecompositionStorage, TileDecodeContext};
use crate::Result;

/// Decode workspace bytes live at one sampled point, by owner.
///
/// Covers what the decoder context owns, plus allocations a stage holds
/// outside it that the caller adds in (the windowed IDWT's level buffers and
/// the region precinct selection). The parsed codestream graph and the
/// caller's output buffer are not included.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkspaceBytes {
    /// Sub-band coefficient storage.
    pub(crate) coefficients: usize,
    /// Decoded component planes (`channel_data`).
    pub(crate) channels: usize,
    /// IDWT output, scratch and window buffers.
    pub(crate) idwt: usize,
    /// Tier-1 code-block decode workspaces.
    pub(crate) tier1: usize,
    /// Decomposition, precinct, code-block, layer, segment and tag-tree storage.
    pub(crate) structure: usize,
}

impl WorkspaceBytes {
    pub(crate) fn sample(
        tile_ctx: &TileDecodeContext,
        storage: &DecompositionStorage<'_>,
    ) -> Result<Self> {
        Ok(Self {
            coefficients: capacity_bytes::<f32>(storage.coefficients.capacity())
                .saturating_add(capacity_bytes::<i64>(storage.coefficients_i64.capacity())),
            channels: tile_ctx.retained_channel_bytes()?,
            idwt: tile_ctx.idwt_capacity_bytes()?,
            tier1: tile_ctx.tier1_capacity_bytes()?,
            structure: structure_bytes(storage),
        })
    }

    pub(crate) fn with_idwt(self, bytes: usize) -> Self {
        Self {
            idwt: self.idwt.saturating_add(bytes),
            ..self
        }
    }

    pub(crate) fn with_structure(self, bytes: usize) -> Self {
        Self {
            structure: self.structure.saturating_add(bytes),
            ..self
        }
    }

    pub(crate) fn total(self) -> usize {
        self.coefficients
            .saturating_add(self.channels)
            .saturating_add(self.idwt)
            .saturating_add(self.tier1)
            .saturating_add(self.structure)
    }
}

/// Keeps the sample with the largest total seen during one decode.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkspacePeak(WorkspaceBytes);

impl WorkspacePeak {
    pub(crate) fn record(&mut self, sample: WorkspaceBytes) {
        if sample.total() > self.0.total() {
            self.0 = sample;
        }
    }

    pub(crate) fn bytes(self) -> WorkspaceBytes {
        self.0
    }
}

/// Samples the context and storage, adding `extra_structure_bytes` held
/// outside them, into the decode's peak.
pub(crate) fn record_workspace(
    tile_ctx: &mut TileDecodeContext,
    storage: &DecompositionStorage<'_>,
    extra_structure_bytes: usize,
) -> Result<()> {
    let sample = WorkspaceBytes::sample(tile_ctx, storage)?.with_structure(extra_structure_bytes);
    tile_ctx.debug_counters.workspace_peak.record(sample);
    Ok(())
}

fn structure_bytes(storage: &DecompositionStorage<'_>) -> usize {
    [
        capacity_bytes::<super::Segment<'_>>(storage.segments.capacity()),
        capacity_bytes::<super::Layer>(storage.layers.capacity()),
        capacity_bytes::<super::CodeBlock>(storage.code_blocks.capacity()),
        capacity_bytes::<super::Precinct>(storage.precincts.capacity()),
        capacity_bytes::<super::TagNode>(storage.tag_tree_nodes.capacity()),
        capacity_bytes::<super::SubBand>(storage.sub_bands.capacity()),
        capacity_bytes::<super::Decomposition>(storage.decompositions.capacity()),
        capacity_bytes::<super::TileDecompositions>(storage.tile_decompositions.capacity()),
    ]
    .into_iter()
    .fold(0, usize::saturating_add)
}

fn capacity_bytes<T>(capacity: usize) -> usize {
    capacity.saturating_mul(size_of::<T>())
}
