// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::workspace::JpegTranscodeWorkspace;
use super::{
    validate_jpeg_transcode_workspace, EncodedTranscode, Float97BatchTile,
    Float97PrecomputedBatchRecord, IntegerBatchTile, JpegTileBatchInput, JpegToHtj2kError,
    JpegToHtj2kOptions, PrecomputedHtj2k97Image,
};
use crate::allocation::{checked_add_allocation_bytes, checked_allocation_bytes};

type TileOutcome = Result<EncodedTranscode, JpegToHtj2kError>;
type TileResultSlot = Option<TileOutcome>;
type IndexedEncodedTile = (usize, TileOutcome);
type PreparedResult<T> = (usize, Result<T, JpegToHtj2kError>);

#[derive(Clone, Copy)]
pub(super) enum BatchWorkspaceKind {
    Integer,
    AcceleratedFloat97,
    Individual,
}

pub(super) fn validate_batch_workspace(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    kind: BatchWorkspaceKind,
) -> Result<(), JpegToHtj2kError> {
    let valid_tile_peaks = tiles.iter().filter_map(|tile| {
        validate_jpeg_transcode_workspace(tile.bytes, options)
            .ok()
            .map(JpegTranscodeWorkspace::peak_bytes)
    });
    batch_workspace_bytes(tiles.len(), kind, valid_tile_peaks)?;
    Ok(())
}

fn batch_workspace_bytes(
    tile_count: usize,
    kind: BatchWorkspaceKind,
    valid_tile_peaks: impl Iterator<Item = usize>,
) -> Result<usize, JpegToHtj2kError> {
    let mut peak_bytes = fixed_metadata_peak(tile_count, kind)?;
    for tile_bytes in valid_tile_peaks {
        peak_bytes = checked_add_allocation_bytes(peak_bytes, tile_bytes)?;
    }
    Ok(peak_bytes)
}

fn fixed_metadata_peak(
    tile_count: usize,
    kind: BatchWorkspaceKind,
) -> Result<usize, JpegToHtj2kError> {
    match kind {
        BatchWorkspaceKind::Integer => prepared_batch_peak::<IntegerBatchTile>(tile_count),
        BatchWorkspaceKind::AcceleratedFloat97 => {
            let prepared = prepared_batch_peak::<Float97BatchTile>(tile_count)?;
            let conversion = phase4::<
                Float97BatchTile,
                Float97PrecomputedBatchRecord,
                PrecomputedHtj2k97Image,
                TileResultSlot,
            >(tile_count)?;
            let conversion_output = phase5::<
                Float97PrecomputedBatchRecord,
                PrecomputedHtj2k97Image,
                Vec<u8>,
                IndexedEncodedTile,
                TileResultSlot,
            >(tile_count)?;
            Ok(prepared.max(conversion).max(conversion_output))
        }
        BatchWorkspaceKind::Individual => Ok(checked_allocation_bytes::<TileOutcome>(tile_count)?),
    }
}

fn prepared_batch_peak<T>(tile_count: usize) -> Result<usize, JpegToHtj2kError> {
    let preparation = phase3::<PreparedResult<T>, TileResultSlot, T>(tile_count)?;
    let encoding = phase3::<T, IndexedEncodedTile, TileResultSlot>(tile_count)?;
    let output = phase2::<TileResultSlot, TileOutcome>(tile_count)?;
    Ok(preparation.max(encoding).max(output))
}

fn phase2<A, B>(len: usize) -> Result<usize, JpegToHtj2kError> {
    Ok(checked_add_allocation_bytes(
        checked_allocation_bytes::<A>(len)?,
        checked_allocation_bytes::<B>(len)?,
    )?)
}

fn phase3<A, B, C>(len: usize) -> Result<usize, JpegToHtj2kError> {
    let bytes = phase2::<A, B>(len)?;
    Ok(checked_add_allocation_bytes(
        bytes,
        checked_allocation_bytes::<C>(len)?,
    )?)
}

fn phase4<A, B, C, D>(len: usize) -> Result<usize, JpegToHtj2kError> {
    let bytes = phase3::<A, B, C>(len)?;
    Ok(checked_add_allocation_bytes(
        bytes,
        checked_allocation_bytes::<D>(len)?,
    )?)
}

fn phase5<A, B, C, D, E>(len: usize) -> Result<usize, JpegToHtj2kError> {
    let bytes = phase4::<A, B, C, D>(len)?;
    Ok(checked_add_allocation_bytes(
        bytes,
        checked_allocation_bytes::<E>(len)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::{batch_workspace_bytes, fixed_metadata_peak, BatchWorkspaceKind};
    use crate::JpegToHtj2kError;

    #[test]
    fn fixed_metadata_has_an_exact_cap_boundary_for_both_prepared_batch_types() {
        for kind in [
            BatchWorkspaceKind::Integer,
            BatchWorkspaceKind::AcceleratedFloat97,
        ] {
            let bytes_per_tile = fixed_metadata_peak(1, kind).expect("one metadata record");
            let max_tiles = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES / bytes_per_tile;
            assert_eq!(
                fixed_metadata_peak(max_tiles, kind).expect("exact metadata boundary"),
                max_tiles * bytes_per_tile
            );
            assert!(matches!(
                fixed_metadata_peak(max_tiles + 1, kind),
                Err(JpegToHtj2kError::MemoryCapExceeded { requested, cap })
                    if requested > cap
            ));
        }
    }

    #[test]
    fn all_invalid_tiles_still_pay_fixed_batch_metadata() {
        for kind in [
            BatchWorkspaceKind::Integer,
            BatchWorkspaceKind::AcceleratedFloat97,
        ] {
            let bytes_per_tile = fixed_metadata_peak(1, kind).expect("one metadata record");
            let oversized_count = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES / bytes_per_tile + 1;
            assert!(matches!(
                batch_workspace_bytes(oversized_count, kind, core::iter::empty()),
                Err(JpegToHtj2kError::MemoryCapExceeded { requested, cap })
                    if requested > cap
            ));
        }
    }
}
