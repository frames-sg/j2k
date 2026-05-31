//! Coefficient-domain JPEG 2000 family recode helpers.

use alloc::vec::Vec;

use super::build::{self, Decomposition, SubBand};
use super::codestream::{ComponentInfo, Header, QuantizationStyle, WaveletTransform};
use super::decode::{decode_component_tile_bit_planes, DecoderContext, DecompositionStorage};
use super::progression::progression_iterator;
use super::segment;
use super::tile::{self, Tile};
use crate::error::{bail, DecodingError, Result, TileError};
use crate::reader::BitReader;
use crate::{
    J2kForwardDwt53Level, J2kForwardDwt53Output, PrecomputedHtj2k53Component,
    PrecomputedHtj2k53Image,
};

/// Reversible 5/3 source coefficients ready for HTJ2K code-block recoding.
#[derive(Debug, Clone)]
pub struct Reversible53CoefficientImage {
    /// Precomputed wavelet coefficients in the native HTJ2K encoder shape.
    pub image: PrecomputedHtj2k53Image,
    /// Source COD multi-component transform flag to preserve in output.
    pub use_mct: bool,
    /// Source code-block width exponent minus two.
    pub code_block_width_exp: u8,
    /// Source code-block height exponent minus two.
    pub code_block_height_exp: u8,
    /// Source quantization guard-bit count.
    pub guard_bits: u8,
}

pub(crate) fn extract_reversible_53_coefficients<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    ctx: &mut DecoderContext<'a>,
) -> Result<Reversible53CoefficientImage> {
    validate_header_for_reversible_53_recode(header)?;

    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header)?;
    if tiles.len() != 1 {
        bail!(DecodingError::UnsupportedFeature(
            "coefficient-domain 5/3 recode currently supports single-tile codestreams"
        ));
    }

    let tile = &tiles[0];
    validate_tile_for_reversible_53_recode(tile)?;

    ctx.tile_decode_context.channel_data.clear();
    ctx.storage.reset();

    build::build(tile, &mut ctx.storage)?;
    segment::parse(tile, progression_iterator(tile)?, header, &mut ctx.storage)?;

    let mut no_ht_decoder = None;
    let cpu_decode_parallelism = ctx.cpu_decode_parallelism();
    decode_component_tile_bit_planes(
        tile,
        &mut ctx.tile_decode_context,
        &mut ctx.storage,
        header,
        &mut no_ht_decoder,
        cpu_decode_parallelism,
        false,
    )?;

    let image = precomputed_image_from_storage(header, tile, &ctx.storage)?;
    let first = tile.component_infos.first().ok_or(TileError::Invalid)?;
    let params = &first.coding_style.parameters;
    Ok(Reversible53CoefficientImage {
        image,
        use_mct: tile.mct,
        code_block_width_exp: params.code_block_width.saturating_sub(2),
        code_block_height_exp: params.code_block_height.saturating_sub(2),
        guard_bits: first.quantization_info.guard_bits,
    })
}

fn validate_header_for_reversible_53_recode(header: &Header<'_>) -> Result<()> {
    if header.skipped_resolution_levels != 0 {
        bail!(DecodingError::UnsupportedFeature(
            "coefficient-domain 5/3 recode requires full-resolution decode settings"
        ));
    }
    if header.size_data.num_tiles() != 1 {
        bail!(DecodingError::UnsupportedFeature(
            "coefficient-domain 5/3 recode currently supports single-tile codestreams"
        ));
    }
    if header.size_data.image_area_x_offset != 0
        || header.size_data.image_area_y_offset != 0
        || header.size_data.tile_x_offset != 0
        || header.size_data.tile_y_offset != 0
    {
        bail!(DecodingError::UnsupportedFeature(
            "coefficient-domain 5/3 recode currently requires zero image and tile origins"
        ));
    }
    Ok(())
}

fn validate_tile_for_reversible_53_recode(tile: &Tile<'_>) -> Result<()> {
    if !matches!(tile.component_infos.len(), 1 | 3) {
        bail!(DecodingError::UnsupportedFeature(
            "coefficient-domain 5/3 recode supports only grayscale or RGB codestreams"
        ));
    }
    if tile.mct && tile.component_infos.len() != 3 {
        bail!(DecodingError::UnsupportedFeature(
            "reversible color transform requires three components"
        ));
    }

    let first = tile.component_infos.first().ok_or(TileError::Invalid)?;
    let first_params = &first.coding_style.parameters;
    let first_bit_depth = first.size_info.precision;
    let first_guard_bits = first.quantization_info.guard_bits;

    for component in &tile.component_infos {
        validate_component_for_reversible_53_recode(component)?;
        if component.size_info.precision != first_bit_depth {
            bail!(DecodingError::UnsupportedFeature(
                "coefficient-domain 5/3 recode requires equal component bit depths"
            ));
        }
        if component.quantization_info.guard_bits != first_guard_bits {
            bail!(DecodingError::UnsupportedFeature(
                "coefficient-domain 5/3 recode requires equal component guard bits"
            ));
        }
        let params = &component.coding_style.parameters;
        if params.num_decomposition_levels != first_params.num_decomposition_levels
            || params.code_block_width != first_params.code_block_width
            || params.code_block_height != first_params.code_block_height
        {
            bail!(DecodingError::UnsupportedFeature(
                "coefficient-domain 5/3 recode requires matching component coding geometry"
            ));
        }
    }

    Ok(())
}

fn validate_component_for_reversible_53_recode(component: &ComponentInfo) -> Result<()> {
    if component.wavelet_transform() != WaveletTransform::Reversible53 {
        bail!(DecodingError::UnsupportedFeature(
            "coefficient-domain lossless recode currently supports only reversible 5/3 sources"
        ));
    }
    if component.num_decomposition_levels() == 0 {
        bail!(DecodingError::UnsupportedFeature(
            "coefficient-domain 5/3 recode requires at least one decomposition level"
        ));
    }
    if component.quantization_info.quantization_style != QuantizationStyle::NoQuantization {
        bail!(DecodingError::UnsupportedFeature(
            "coefficient-domain 5/3 recode requires no-quantization QCD/QCC"
        ));
    }
    if component
        .coding_style
        .parameters
        .code_block_style
        .uses_high_throughput_block_coding()
    {
        bail!(DecodingError::UnsupportedFeature(
            "source already uses HT block coding"
        ));
    }
    Ok(())
}

fn precomputed_image_from_storage(
    header: &Header<'_>,
    tile: &Tile<'_>,
    storage: &DecompositionStorage<'_>,
) -> Result<PrecomputedHtj2k53Image> {
    let mut components = Vec::with_capacity(tile.component_infos.len());
    for (component_index, component_info) in tile.component_infos.iter().enumerate() {
        let tile_decomposition = storage
            .tile_decompositions
            .get(component_index)
            .ok_or(TileError::Invalid)?;
        components.push(PrecomputedHtj2k53Component {
            x_rsiz: component_info.size_info.horizontal_resolution,
            y_rsiz: component_info.size_info.vertical_resolution,
            dwt: component_dwt_from_storage(tile_decomposition, storage)?,
        });
    }

    let first = tile.component_infos.first().ok_or(TileError::Invalid)?;
    Ok(PrecomputedHtj2k53Image {
        width: header.size_data.image_width(),
        height: header.size_data.image_height(),
        bit_depth: first.size_info.precision,
        signed: false,
        components,
    })
}

fn component_dwt_from_storage(
    tile_decomposition: &super::decode::TileDecompositions,
    storage: &DecompositionStorage<'_>,
) -> Result<J2kForwardDwt53Output> {
    let ll = storage
        .sub_bands
        .get(tile_decomposition.first_ll_sub_band)
        .ok_or(TileError::Invalid)?;

    let mut levels = Vec::with_capacity(tile_decomposition.decompositions.len());
    for idx in tile_decomposition.decompositions.clone() {
        let decomposition = storage.decompositions.get(idx).ok_or(TileError::Invalid)?;
        levels.push(level_from_decomposition(decomposition, storage));
    }

    Ok(J2kForwardDwt53Output {
        ll: subband_coefficients(ll, storage),
        ll_width: ll.rect.width(),
        ll_height: ll.rect.height(),
        levels,
    })
}

fn level_from_decomposition(
    decomposition: &Decomposition,
    storage: &DecompositionStorage<'_>,
) -> J2kForwardDwt53Level {
    let hl = &storage.sub_bands[decomposition.sub_bands[0]];
    let lh = &storage.sub_bands[decomposition.sub_bands[1]];
    let hh = &storage.sub_bands[decomposition.sub_bands[2]];
    J2kForwardDwt53Level {
        hl: subband_coefficients(hl, storage),
        lh: subband_coefficients(lh, storage),
        hh: subband_coefficients(hh, storage),
        width: decomposition.rect.width(),
        height: decomposition.rect.height(),
        low_width: lh.rect.width(),
        low_height: hl.rect.height(),
        high_width: hl.rect.width(),
        high_height: lh.rect.height(),
    }
}

fn subband_coefficients(subband: &SubBand, storage: &DecompositionStorage<'_>) -> Vec<f32> {
    storage.coefficients[subband.coefficients.clone()].to_vec()
}
