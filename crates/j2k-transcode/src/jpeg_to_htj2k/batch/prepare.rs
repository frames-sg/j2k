// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    component_sampling_for_jpeg, decomposition_levels_for_components, extract_dct_blocks,
    DctExtractOptions, Instant, JpegDctImage, JpegToHtj2kError, JpegToHtj2kOptions,
    PrecomputedHtj2k53Component, PrecomputedHtj2k97Component, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97Component, PrequantizedHtj2k97Component, TranscodeComponentReport,
    TranscodeTimingReport,
};

pub(in super::super) struct IntegerBatchTile {
    pub(in super::super) tile_index: usize,
    pub(in super::super) jpeg: JpegDctImage,
    pub(in super::super) component_sampling: Vec<(u8, u8)>,
    pub(in super::super) decomposition_levels: u8,
    pub(in super::super) all_unit_sampled: bool,
    pub(in super::super) component_reports: Vec<TranscodeComponentReport>,
    pub(in super::super) precomputed_components: Vec<Option<PrecomputedHtj2k53Component>>,
    pub(in super::super) float_validation_actual: Vec<i32>,
    pub(in super::super) float_validation_expected: Vec<i32>,
    pub(in super::super) integer_validation_actual: Vec<i32>,
    pub(in super::super) integer_validation_expected: Vec<i32>,
    pub(in super::super) timings: TranscodeTimingReport,
}

pub(in super::super) struct Float97BatchTile {
    pub(in super::super) tile_index: usize,
    pub(in super::super) jpeg: JpegDctImage,
    pub(in super::super) component_sampling: Vec<(u8, u8)>,
    pub(in super::super) decomposition_levels: u8,
    pub(in super::super) all_unit_sampled: bool,
    pub(in super::super) component_reports: Vec<TranscodeComponentReport>,
    pub(in super::super) precomputed_components: Vec<Option<PrecomputedHtj2k97Component>>,
    pub(in super::super) preencoded_compact_payload: Vec<u8>,
    pub(in super::super) preencoded_compact_components:
        Vec<Option<PreencodedHtj2k97CompactComponent>>,
    pub(in super::super) preencoded_components: Vec<Option<PreencodedHtj2k97Component>>,
    pub(in super::super) prequantized_components: Vec<Option<PrequantizedHtj2k97Component>>,
    pub(in super::super) float_validation_actual: Vec<i32>,
    pub(in super::super) float_validation_expected: Vec<i32>,
    pub(in super::super) timings: TranscodeTimingReport,
}

pub(in super::super) struct Float97PrecomputedBatchRecord {
    pub(in super::super) tile_index: usize,
    pub(in super::super) jpeg: JpegDctImage,
    pub(in super::super) decomposition_levels: u8,
    pub(in super::super) all_unit_sampled: bool,
    pub(in super::super) component_reports: Vec<TranscodeComponentReport>,
    pub(in super::super) float_validation_actual: Vec<i32>,
    pub(in super::super) float_validation_expected: Vec<i32>,
    pub(in super::super) timings: TranscodeTimingReport,
}

#[derive(Clone, Copy)]
pub(in super::super) struct BatchComponentRef {
    pub(in super::super) tile_index: usize,
    pub(in super::super) component_index: usize,
}

pub(in super::super) fn prepare_integer_batch_tile(
    tile_index: usize,
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<IntegerBatchTile, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::default())?;
    let timings = TranscodeTimingReport {
        jpeg_dct_extract_us: extract_start.elapsed().as_micros(),
        tile_count: 1,
        ..TranscodeTimingReport::default()
    };
    if jpeg.components.is_empty() || jpeg.components.len() > 4 {
        return Err(JpegToHtj2kError::Unsupported(
            "unsupported JPEG component count for jpeg_to_htj2k",
        ));
    }
    let component_sampling =
        component_sampling_for_jpeg(&jpeg.components, jpeg.width, jpeg.height)?;
    let decomposition_levels = decomposition_levels_for_components(
        &jpeg.components,
        options.encode_options.num_decomposition_levels,
    )?;
    let all_unit_sampled = component_sampling
        .iter()
        .all(|&(x_rsiz, y_rsiz)| x_rsiz == 1 && y_rsiz == 1);
    let component_reports = jpeg
        .components
        .iter()
        .zip(component_sampling.iter().copied())
        .map(|(component, (x_rsiz, y_rsiz))| TranscodeComponentReport {
            component_index: component.component_index,
            width: component.width,
            height: component.height,
            block_cols: component.block_cols,
            block_rows: component.block_rows,
            x_rsiz,
            y_rsiz,
        })
        .collect::<Vec<_>>();
    let precomputed_components = (0..jpeg.components.len()).map(|_| None).collect();

    Ok(IntegerBatchTile {
        tile_index,
        jpeg,
        component_sampling,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        precomputed_components,
        float_validation_actual: Vec::new(),
        float_validation_expected: Vec::new(),
        integer_validation_actual: Vec::new(),
        integer_validation_expected: Vec::new(),
        timings,
    })
}

pub(in super::super) fn prepare_float97_batch_tile(
    tile_index: usize,
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<Float97BatchTile, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::dequantized_only())?;
    let timings = TranscodeTimingReport {
        jpeg_dct_extract_us: extract_start.elapsed().as_micros(),
        tile_count: 1,
        ..TranscodeTimingReport::default()
    };
    if jpeg.components.is_empty() || jpeg.components.len() > 4 {
        return Err(JpegToHtj2kError::Unsupported(
            "unsupported JPEG component count for jpeg_to_htj2k",
        ));
    }
    let component_sampling =
        component_sampling_for_jpeg(&jpeg.components, jpeg.width, jpeg.height)?;
    let decomposition_levels = decomposition_levels_for_components(
        &jpeg.components,
        options.encode_options.num_decomposition_levels,
    )?;
    let all_unit_sampled = component_sampling
        .iter()
        .all(|&(x_rsiz, y_rsiz)| x_rsiz == 1 && y_rsiz == 1);
    let component_reports = jpeg
        .components
        .iter()
        .zip(component_sampling.iter().copied())
        .map(|(component, (x_rsiz, y_rsiz))| TranscodeComponentReport {
            component_index: component.component_index,
            width: component.width,
            height: component.height,
            block_cols: component.block_cols,
            block_rows: component.block_rows,
            x_rsiz,
            y_rsiz,
        })
        .collect::<Vec<_>>();
    let precomputed_components = (0..jpeg.components.len()).map(|_| None).collect();
    let preencoded_compact_components = (0..jpeg.components.len()).map(|_| None).collect();
    let preencoded_components = (0..jpeg.components.len()).map(|_| None).collect();
    let prequantized_components = (0..jpeg.components.len()).map(|_| None).collect();

    Ok(Float97BatchTile {
        tile_index,
        jpeg,
        component_sampling,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        precomputed_components,
        preencoded_compact_payload: Vec::new(),
        preencoded_compact_components,
        preencoded_components,
        prequantized_components,
        float_validation_actual: Vec::new(),
        float_validation_expected: Vec::new(),
        timings,
    })
}

pub(in super::super) fn batch_component_groups(
    tiles: &[IntegerBatchTile],
) -> Vec<Vec<BatchComponentRef>> {
    let mut groups: Vec<Vec<BatchComponentRef>> = Vec::new();

    for (tile_index, tile) in tiles.iter().enumerate() {
        for (component_index, component) in tile.jpeg.components.iter().enumerate() {
            let component_ref = BatchComponentRef {
                tile_index,
                component_index,
            };
            if let Some(group) = groups.iter_mut().find(|group| {
                let first = group[0];
                same_batch_component_key(
                    &tiles[first.tile_index],
                    first.component_index,
                    tile,
                    component_index,
                )
            }) {
                group.push(component_ref);
            } else {
                let _ = component;
                groups.push(vec![component_ref]);
            }
        }
    }

    groups
}

pub(in super::super) fn float97_batch_component_groups(
    tiles: &[Float97BatchTile],
) -> Vec<Vec<BatchComponentRef>> {
    let mut groups: Vec<Vec<BatchComponentRef>> = Vec::new();

    for (tile_index, tile) in tiles.iter().enumerate() {
        for component_index in 0..tile.jpeg.components.len() {
            let component_ref = BatchComponentRef {
                tile_index,
                component_index,
            };
            if let Some(group) = groups.iter_mut().find(|group| {
                let first = group[0];
                same_float97_batch_component_key(
                    &tiles[first.tile_index],
                    first.component_index,
                    tile,
                    component_index,
                )
            }) {
                group.push(component_ref);
            } else {
                groups.push(vec![component_ref]);
            }
        }
    }

    groups
}

pub(in super::super) fn same_batch_component_key(
    left_tile: &IntegerBatchTile,
    left_component_index: usize,
    right_tile: &IntegerBatchTile,
    right_component_index: usize,
) -> bool {
    let left = &left_tile.jpeg.components[left_component_index];
    let right = &right_tile.jpeg.components[right_component_index];
    left.component_index == right.component_index
        && left.width == right.width
        && left.height == right.height
        && left.block_cols == right.block_cols
        && left.block_rows == right.block_rows
        && left_tile.component_sampling[left_component_index]
            == right_tile.component_sampling[right_component_index]
}

pub(in super::super) fn same_float97_batch_component_key(
    left_tile: &Float97BatchTile,
    left_component_index: usize,
    right_tile: &Float97BatchTile,
    right_component_index: usize,
) -> bool {
    let left = &left_tile.jpeg.components[left_component_index];
    let right = &right_tile.jpeg.components[right_component_index];
    left.width == right.width
        && left.height == right.height
        && left.block_cols == right.block_cols
        && left.block_rows == right.block_rows
        && left_tile.component_sampling[left_component_index]
            == right_tile.component_sampling[right_component_index]
}
