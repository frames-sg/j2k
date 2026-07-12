//! Coefficient-domain JPEG 2000 family recode helpers.

use alloc::vec::Vec;

use super::build::{self, Decomposition, SubBand};
use super::codestream::{ComponentInfo, Header, QuantizationStyle, WaveletTransform};
use super::decode::{
    decode_component_tile_bit_planes_budgeted, DecodeAllocationBudget, DecoderContext,
    DecompositionStorage,
};
use super::encode::allocation::{checked_add_bytes, checked_element_bytes};
use super::progression::progression_iterator;
use super::segment;
use super::tile::{self, Tile};
use crate::error::{bail, DecodingError, Result, TileError, ValidationError};
use crate::reader::BitReader;
use crate::{
    try_reserve_decode_elements, EncodeOptions, EncodeResult, J2kForwardDwt53Level,
    J2kForwardDwt53Output, PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
};

#[cfg(test)]
mod tests;

/// Reversible 5/3 source coefficients ready for HTJ2K code-block recoding.
#[derive(Debug)]
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

impl Reversible53CoefficientImage {
    /// Encode this complete coefficient owner as HTJ2K under one retained-input budget.
    ///
    /// # Errors
    ///
    /// Returns a typed error when retained capacity arithmetic, encode
    /// planning, allocation, packetization, or codestream assembly fails.
    #[doc(hidden)]
    pub fn encode_htj2k(&self, options: &EncodeOptions) -> EncodeResult<Vec<u8>> {
        let retained_bytes = self.checked_retained_capacity_bytes()?;
        super::encode::encode_precomputed_htj2k_53_with_mct_and_retained_owner(
            &self.image,
            options,
            self.use_mct,
            self,
            retained_bytes,
        )
    }

    /// Return allocator-capacity bytes retained while this coefficient image is encoded.
    fn checked_retained_capacity_bytes(&self) -> EncodeResult<usize> {
        let mut bytes = checked_element_bytes::<PrecomputedHtj2k53Component>(
            self.image.components.capacity(),
            "reversible coefficient component capacity",
        )?;
        for component in &self.image.components {
            bytes = checked_add_bytes(
                bytes,
                checked_element_bytes::<f32>(
                    component.dwt.ll.capacity(),
                    "reversible coefficient LL capacity",
                )?,
                "reversible coefficient image capacity",
            )?;
            bytes = checked_add_bytes(
                bytes,
                checked_element_bytes::<J2kForwardDwt53Level>(
                    component.dwt.levels.capacity(),
                    "reversible coefficient level capacity",
                )?,
                "reversible coefficient image capacity",
            )?;
            for level in &component.dwt.levels {
                for band in [&level.hl, &level.lh, &level.hh] {
                    bytes = checked_add_bytes(
                        bytes,
                        checked_element_bytes::<f32>(
                            band.capacity(),
                            "reversible coefficient detail-band capacity",
                        )?,
                        "reversible coefficient image capacity",
                    )?;
                }
            }
        }
        Ok(bytes)
    }
}

pub(crate) fn extract_reversible_53_coefficients<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<Reversible53CoefficientImage> {
    ctx.release_reusable_allocations();
    let result = extract_reversible_53_coefficients_inner(data, header, retained_image_bytes, ctx);
    ctx.release_reusable_allocations();
    result
}

fn extract_reversible_53_coefficients_inner<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<Reversible53CoefficientImage> {
    validate_header_for_reversible_53_recode(header)?;

    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header, retained_image_bytes)?;
    if tiles.len() != 1 {
        bail!(DecodingError::UnsupportedFeature(
            "coefficient-domain 5/3 recode currently supports single-tile codestreams"
        ));
    }

    let tile = &tiles[0];
    validate_tile_for_reversible_53_recode(tile)?;

    ctx.tile_decode_context.channel_data.clear();
    ctx.storage.reset_for_next_tile();

    build::build(
        tile,
        &mut ctx.storage,
        tiles.structural_workspace_bytes(),
        false,
        build::BuildWorkspace::CoefficientsOnly,
    )?;
    segment::parse(tile, progression_iterator(tile)?, header, &mut ctx.storage)?;

    let mut no_ht_decoder = None;
    let cpu_decode_parallelism = ctx.cpu_decode_parallelism();
    decode_component_tile_bit_planes_budgeted(
        tile,
        &mut ctx.tile_decode_context,
        &mut ctx.storage,
        header,
        &mut no_ht_decoder,
        cpu_decode_parallelism,
        false,
    )?;

    let output_plan = RecodeOutputPlan::for_storage(tile, &ctx.storage)?;
    let mut budget = DecodeAllocationBudget::for_storage(&ctx.storage)?;
    output_plan.include_in(&mut budget)?;
    let image =
        precomputed_image_from_storage(header, tile, &ctx.storage, &output_plan, &mut budget)?;
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

struct RecodeOutputPlan {
    components: usize,
    levels: usize,
    coefficients: usize,
}

impl RecodeOutputPlan {
    fn for_storage(tile: &Tile<'_>, storage: &DecompositionStorage<'_>) -> Result<Self> {
        let level_count =
            storage
                .tile_decompositions
                .iter()
                .try_fold(0usize, |count, decomposition| {
                    count
                        .checked_add(decomposition.decompositions.len())
                        .ok_or(ValidationError::ImageTooLarge)
                })?;
        Ok(Self {
            components: tile.component_infos.len(),
            levels: level_count,
            coefficients: storage.coefficients.len(),
        })
    }

    fn include_in(&self, budget: &mut DecodeAllocationBudget) -> Result<()> {
        budget.include_elements::<PrecomputedHtj2k53Component>(self.components)?;
        budget.include_elements::<J2kForwardDwt53Level>(self.levels)?;
        budget.include_elements::<f32>(self.coefficients)
    }
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
    output_plan: &RecodeOutputPlan,
    budget: &mut DecodeAllocationBudget,
) -> Result<PrecomputedHtj2k53Image> {
    let mut components = Vec::new();
    try_reserve_decode_elements(&mut components, output_plan.components)?;
    budget.include_capacity_overage::<PrecomputedHtj2k53Component>(
        output_plan.components,
        components.capacity(),
    )?;
    for (component_index, component_info) in tile.component_infos.iter().enumerate() {
        let tile_decomposition = storage
            .tile_decompositions
            .get(component_index)
            .ok_or(TileError::Invalid)?;
        components.push(PrecomputedHtj2k53Component {
            x_rsiz: component_info.size_info.horizontal_resolution,
            y_rsiz: component_info.size_info.vertical_resolution,
            dwt: component_dwt_from_storage(tile_decomposition, storage, budget)?,
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
    budget: &mut DecodeAllocationBudget,
) -> Result<J2kForwardDwt53Output> {
    let ll = storage
        .sub_bands
        .get(tile_decomposition.first_ll_sub_band)
        .ok_or(TileError::Invalid)?;

    let level_count = tile_decomposition.decompositions.len();
    let mut levels = Vec::new();
    try_reserve_decode_elements(&mut levels, level_count)?;
    budget.include_capacity_overage::<J2kForwardDwt53Level>(level_count, levels.capacity())?;
    for idx in tile_decomposition.decompositions.clone() {
        let decomposition = storage.decompositions.get(idx).ok_or(TileError::Invalid)?;
        levels.push(level_from_decomposition(decomposition, storage, budget)?);
    }

    Ok(J2kForwardDwt53Output {
        ll: subband_coefficients(ll, storage, budget)?,
        ll_width: ll.rect.width(),
        ll_height: ll.rect.height(),
        levels,
    })
}

fn level_from_decomposition(
    decomposition: &Decomposition,
    storage: &DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<J2kForwardDwt53Level> {
    let hl = storage
        .sub_bands
        .get(decomposition.sub_bands[0])
        .ok_or(TileError::Invalid)?;
    let lh = storage
        .sub_bands
        .get(decomposition.sub_bands[1])
        .ok_or(TileError::Invalid)?;
    let hh = storage
        .sub_bands
        .get(decomposition.sub_bands[2])
        .ok_or(TileError::Invalid)?;
    Ok(J2kForwardDwt53Level {
        hl: subband_coefficients(hl, storage, budget)?,
        lh: subband_coefficients(lh, storage, budget)?,
        hh: subband_coefficients(hh, storage, budget)?,
        width: decomposition.rect.width(),
        height: decomposition.rect.height(),
        low_width: lh.rect.width(),
        low_height: hl.rect.height(),
        high_width: hl.rect.width(),
        high_height: lh.rect.height(),
    })
}

fn subband_coefficients(
    subband: &SubBand,
    storage: &DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<f32>> {
    let coefficients = storage
        .coefficients
        .get(subband.coefficients.clone())
        .ok_or(TileError::Invalid)?;
    let mut copied = Vec::new();
    try_reserve_decode_elements(&mut copied, coefficients.len())?;
    budget.include_capacity_overage::<f32>(coefficients.len(), copied.capacity())?;
    copied.extend_from_slice(coefficients);
    Ok(copied)
}
