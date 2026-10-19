// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned, selected coefficient output before inverse DWT and color conversion.

use alloc::vec::Vec;

use super::build::{self, SubBandType};
use super::codestream::{Header, QuantizationStyle, WaveletTransform as CodecTransform};
use super::decode::{
    decode_selected_wavelet_bands, sub_band_decode_parameters, DecodeAllocationBudget,
    DecoderContext, DecompositionStorage,
};
use super::rect::IntRect;
use super::segment;
use super::tile::{self, Tile};
use crate::error::{DecodingError, Result, ValidationError};
use crate::reader::BitReader;

use j2k_types::{
    WaveletBand, WaveletBandKind, WaveletComponent, WaveletImage, WaveletRect, WaveletSamples,
    WaveletSelection, WaveletTile, WaveletTransform,
};

pub(crate) fn select_component(selection: &WaveletSelection, index: usize) -> bool {
    selection.components.is_empty()
        || selection
            .components
            .iter()
            .any(|&i| usize::from(i) == index)
}

pub(crate) fn select_band(selection: &WaveletSelection, resolution: u8, kind: SubBandType) -> bool {
    (selection.resolutions.is_empty() || selection.resolutions.contains(&resolution))
        && (selection.bands.is_empty() || selection.bands.contains(&band_kind(kind)))
}

fn select_tile(selection: &WaveletSelection, index: u32) -> bool {
    selection.tiles.is_empty() || selection.tiles.contains(&index)
}

pub(crate) fn extract<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    context: &mut DecoderContext<'a>,
    selection: &WaveletSelection,
) -> Result<WaveletImage> {
    // Pixel scratch is not part of a coefficient-only decode's budget. Retire
    // it on entry and retire partial wavelet scratch on both success and error.
    context.release_reusable_allocations();
    let result = extract_inner(data, header, retained_image_bytes, context, selection);
    context.release_reusable_allocations();
    result
}

fn extract_inner<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    context: &mut DecoderContext<'a>,
    selection: &WaveletSelection,
) -> Result<WaveletImage> {
    if header.skipped_resolution_levels != 0 {
        return unsupported("wavelet extraction requires full-resolution header settings; select resolutions explicitly");
    }
    let tiles = tile::parse(&mut BitReader::new(data), header, retained_image_bytes)?;
    for &index in &selection.tiles {
        if !tiles.iter().any(|tile| tile.idx == index) {
            return unsupported("wavelet tile index is out of range");
        }
    }
    let mut budget = DecodeAllocationBudget::from_live_bytes(tiles.structural_workspace_bytes())?;
    let mut output = Vec::new();
    budget.reserve_new(
        &mut output,
        tiles
            .iter()
            .filter(|t| select_tile(selection, t.idx))
            .count(),
    )?;
    let mut matched = false;
    for tile in tiles.iter().filter(|t| select_tile(selection, t.idx)) {
        validate_selection(tile, selection)?;
        context.storage.reset_for_next_tile();
        context.tile_decode_context.channel_data.clear();
        // Allocate the integer slab whenever a selected reversible component
        // needs it, including low-bit-depth sources. Never round through F32.
        context.storage.exact_integer_decode =
            tile.component_infos.iter().enumerate().any(|(i, c)| {
                select_component(selection, i)
                    && c.wavelet_transform() == CodecTransform::Reversible53
            });
        let retained_baseline = budget
            .live_bytes()
            .checked_add(context.tile_decode_context.retained_scratch_bytes()?)
            .ok_or(ValidationError::ImageTooLarge)?;
        build::build(
            tile,
            &mut context.storage,
            retained_baseline,
            false,
            build::BuildWorkspace::CoefficientsOnly,
            None,
        )?;
        segment::parse(tile, header, &mut context.storage)?;
        let parallelism = context.cpu_decode_parallelism();
        decode_selected_wavelet_bands(
            tile,
            &mut context.tile_decode_context,
            &mut context.storage,
            header,
            parallelism,
            selection,
        )?;
        let mut active = DecodeAllocationBudget::for_storage(&context.storage)?;
        let before = active.live_bytes();
        let components = copy_components(tile, &context.storage, selection, &mut active)?;
        matched |= components.iter().any(|c| !c.bands.is_empty());
        budget.include_bytes(
            active
                .live_bytes()
                .checked_sub(before)
                .ok_or(ValidationError::ImageTooLarge)?,
        )?;
        output.push(WaveletTile {
            index: tile.idx,
            rect: rect(tile.rect),
            uses_color_transform: tile.mct,
            components,
        });
    }
    if !matched {
        return unsupported("wavelet selection matches no subbands");
    }
    Ok(WaveletImage {
        rect: WaveletRect {
            x0: header.size_data.image_area_x_offset,
            y0: header.size_data.image_area_y_offset,
            x1: header.size_data.reference_grid_width,
            y1: header.size_data.reference_grid_height,
        },
        tiles: output,
    })
}

fn validate_selection(tile: &Tile<'_>, selection: &WaveletSelection) -> Result<()> {
    if selection
        .components
        .iter()
        .any(|&index| usize::from(index) >= tile.component_infos.len())
    {
        return unsupported("wavelet component index is out of range");
    }
    for (index, component) in tile.component_infos.iter().enumerate() {
        if !select_component(selection, index) {
            continue;
        }
        if selection
            .resolutions
            .iter()
            .any(|&r| r >= component.num_resolution_levels())
        {
            return unsupported(
                "wavelet resolution index is out of range for a selected component",
            );
        }
        if component.wavelet_transform() == CodecTransform::Reversible53
            && component.quantization_info.quantization_style != QuantizationStyle::NoQuantization
        {
            return unsupported("exact reversible wavelet output requires no-quantization coding");
        }
    }
    Ok(())
}

fn copy_components(
    tile: &Tile<'_>,
    storage: &DecompositionStorage<'_>,
    selection: &WaveletSelection,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<WaveletComponent>> {
    let mut components = Vec::new();
    budget.reserve_new(
        &mut components,
        tile.component_infos
            .iter()
            .enumerate()
            .filter(|(i, _)| select_component(selection, *i))
            .count(),
    )?;
    for (index, component_tile) in tile.component_tiles().enumerate() {
        if !select_component(selection, index) {
            continue;
        }
        let info = &tile.component_infos[index];
        let decomposition = &storage.tile_decompositions[index];
        let mut bands = Vec::new();
        let count = (0..info.num_resolution_levels())
            .flat_map(|r| {
                decomposition
                    .sub_band_iter(r, &storage.decompositions)
                    .map(move |i| (r, i))
            })
            .filter(|&(r, i)| select_band(selection, r, storage.sub_bands[i].sub_band_type))
            .count();
        budget.reserve_new(&mut bands, count)?;
        for resolution in 0..info.num_resolution_levels() {
            for band_index in decomposition.sub_band_iter(resolution, &storage.decompositions) {
                let band = &storage.sub_bands[band_index];
                if !select_band(selection, resolution, band.sub_band_type) {
                    continue;
                }
                let step = sub_band_decode_parameters(band, resolution, info)?.dequantization_step;
                let samples = if info.wavelet_transform() == CodecTransform::Reversible53 {
                    let source = &storage.coefficients_i64[band.coefficients.clone()];
                    let mut values = Vec::new();
                    budget.reserve_new(&mut values, source.len())?;
                    values.extend_from_slice(source);
                    WaveletSamples::I64(values)
                } else {
                    let source = &storage.coefficients[band.coefficients.clone()];
                    let mut values = Vec::new();
                    budget.reserve_new(&mut values, source.len())?;
                    values.extend_from_slice(source);
                    WaveletSamples::F32(values)
                };
                bands.push(WaveletBand {
                    kind: band_kind(band.sub_band_type),
                    resolution,
                    rect: rect(band.rect),
                    quantization_step: step,
                    samples,
                });
            }
        }
        components.push(WaveletComponent {
            index: u16::try_from(index).map_err(|_| ValidationError::ImageTooLarge)?,
            rect: rect(component_tile.rect),
            precision: info.size_info.precision,
            signed: info.size_info.signed,
            sampling: (
                info.size_info.horizontal_resolution,
                info.size_info.vertical_resolution,
            ),
            transform: match info.wavelet_transform() {
                CodecTransform::Reversible53 => WaveletTransform::Reversible53,
                CodecTransform::Irreversible97 => WaveletTransform::Irreversible97,
            },
            decomposition_levels: info.num_decomposition_levels(),
            bands,
        });
    }
    Ok(components)
}

fn rect(value: IntRect) -> WaveletRect {
    WaveletRect {
        x0: value.x0,
        y0: value.y0,
        x1: value.x1,
        y1: value.y1,
    }
}

fn band_kind(value: SubBandType) -> WaveletBandKind {
    match value {
        SubBandType::LowLow => WaveletBandKind::LowLow,
        SubBandType::HighLow => WaveletBandKind::HighLow,
        SubBandType::LowHigh => WaveletBandKind::LowHigh,
        SubBandType::HighHigh => WaveletBandKind::HighHigh,
    }
}

fn unsupported<T>(reason: &'static str) -> Result<T> {
    Err(DecodingError::UnsupportedFeature(reason).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_ll_bands_skip_entropy_work_and_never_run_inverse_dwt() {
        let bytes = include_bytes!(
            "../../../j2k-test-support/fixtures/htj2k/openjph_batch/gray_u12_53.j2c"
        );
        let image = crate::Image::new(bytes, &crate::DecodeSettings::strict()).unwrap();
        let mut all = DecoderContext::default();
        let mut selected = DecoderContext::default();
        let full = extract_inner(
            image.codestream,
            &image.header,
            image.retained_metadata_bytes().unwrap(),
            &mut all,
            &WaveletSelection::default(),
        )
        .unwrap();
        let only_ll = extract_inner(
            image.codestream,
            &image.header,
            image.retained_metadata_bytes().unwrap(),
            &mut selected,
            &WaveletSelection {
                resolutions: alloc::vec![0],
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            selected
                .tile_decode_context
                .debug_counters
                .decoded_code_blocks
                > 0
        );
        assert!(
            selected
                .tile_decode_context
                .debug_counters
                .decoded_code_blocks
                < all.tile_decode_context.debug_counters.decoded_code_blocks
        );
        assert_eq!(
            all.tile_decode_context.debug_counters.idwt_output_samples,
            0
        );
        assert_eq!(
            selected
                .tile_decode_context
                .debug_counters
                .idwt_output_samples,
            0
        );
        assert_eq!(
            only_ll.tiles[0].components[0].bands[0].samples,
            full.tiles[0].components[0].bands[0].samples
        );
    }

    #[test]
    fn selected_components_and_tiles_skip_entropy_work() {
        let bytes =
            include_bytes!("../../../j2k-test-support/fixtures/htj2k/openjph_batch/rgb_u8_53.j2c");
        let image = crate::Image::new(bytes, &crate::DecodeSettings::strict()).unwrap();
        let decode = |selection: &WaveletSelection| {
            let mut context = DecoderContext::default();
            let output = extract_inner(
                image.codestream,
                &image.header,
                image.retained_metadata_bytes().unwrap(),
                &mut context,
                selection,
            )
            .unwrap();
            let blocks = context
                .tile_decode_context
                .debug_counters
                .decoded_code_blocks;
            (output, blocks)
        };
        let (full, all_blocks) = decode(&WaveletSelection::default());
        let (one_component, component_blocks) = decode(&WaveletSelection {
            components: alloc::vec![1],
            ..Default::default()
        });
        let (one_tile, tile_blocks) = decode(&WaveletSelection {
            tiles: alloc::vec![0],
            ..Default::default()
        });

        assert!(full.tiles.len() > 1);
        assert!(component_blocks > 0 && component_blocks < all_blocks);
        assert!(tile_blocks > 0 && tile_blocks < all_blocks);
        for (tile, full_tile) in one_component.tiles.iter().zip(&full.tiles) {
            assert_eq!(tile.components.len(), 1);
            assert_eq!(tile.components[0].index, 1);
            let expected = &full_tile.components[1].bands;
            assert_eq!(tile.components[0].bands.len(), expected.len());
            for (band, expected) in tile.components[0].bands.iter().zip(expected) {
                assert_eq!(band.samples, expected.samples);
            }
        }
        assert_eq!(one_tile.tiles.len(), 1);
        assert_eq!(one_tile.tiles[0].index, 0);
    }
}
