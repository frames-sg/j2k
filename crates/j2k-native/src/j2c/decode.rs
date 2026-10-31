//! Decoding JPEG2000 code streams.
//!
//! This is the "core" module of the crate that orchestrates all
//! stages in such a way that a given codestream is decoded into its
//! component channels.

use alloc::vec::Vec;

use super::bitplane::{BitPlaneDecodeBuffers, BitPlaneDecodeContext};
use super::build::{CodeBlock, Decomposition, Layer, Precinct, Segment, SubBand, SubBandType};
use super::codestream::{ComponentInfo, Header, QuantizationStyle, WaveletTransform};
use super::ht_block_decode::{self, HtBlockDecodeContext};
use super::idwt::IDWTOutput;
use super::roi::{roi_required_output_margin, tile_intersects_output_region, RoiPlan};
use super::tag_tree::TagNode;
use super::tile::{ComponentTile, ParsedTiles, ResolutionTile, Tile};
use super::{bitplane, build, idwt, mct, segment, tile, ComponentData};
use crate::error::{
    bail, ColorError, DecodeError, DecodingError, DirectPlanUnsupportedReason, Result, TileError,
    ValidationError,
};
use crate::j2c::segment::MAX_BITPLANE_COUNT;
use crate::profile;
use crate::reader::BitReader;
use crate::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_i64, checked_decode_sample_count,
    decode_j2k_code_block_scalar_with_workspace,
    decode_j2k_code_block_scalar_with_workspace_midpoint, ComponentRowSink, HtCodeBlockBatchJob,
    HtCodeBlockDecodeJob, HtCodeBlockDecoder, HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan,
    HtSubBandDecodeJob, J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace,
    J2kCodeBlockStyle, J2kDirectBandId, J2kDirectColorPlan, J2kDirectGrayscalePlan,
    J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep, J2kOwnedCodeBlockBatchJob,
    J2kOwnedSubBandPlan, J2kRect, J2kStoreComponentJob, J2kSubBandDecodeJob, J2kSubBandType,
    J2kWaveletTransform,
};
use core::mem::size_of;
use core::ops::Range;

mod allocation;
mod banded_tier1;
mod direct_plan;
#[cfg(feature = "parallel")]
mod parallel_bands;
mod peak;
mod reuse;
mod store;
mod subband;
#[cfg(all(test, feature = "parallel"))]
pub(crate) use self::subband::MIN_BAND_WORK;
mod subband_params;
mod tier1;
#[cfg(feature = "parallel")]
mod tile_workers;
mod workspace;
pub(crate) use self::allocation::DecodeAllocationBudget;
use self::direct_plan::collect_classic_code_block_data;
pub(crate) use self::direct_plan::{
    build_component_grid_color_plan, build_direct_color_plan, build_direct_grayscale_plan,
    build_referenced_classic_color_plan, build_referenced_classic_grayscale_plan,
    build_referenced_classic_rgba_plan, build_referenced_htj2k_color_plan,
    build_referenced_htj2k_grayscale_plan, build_referenced_htj2k_rgba_plan,
};
pub(crate) use self::peak::{record_workspace, WorkspaceBytes, WorkspacePeak};
#[cfg(feature = "std")]
pub(crate) use self::reuse::is_capacity_error;
use self::store::{
    apply_inverse_mct_per_tile, apply_sign_shift_after_mct, component_unsigned_level_shift,
    fused_last_level, store, store_last_level_bands,
};
#[cfg(feature = "parallel")]
use self::store::{store_rows, TileRows};
use self::subband::{
    code_block_required_by_index, count_classic_code_blocks, count_ht_code_blocks,
    decode_component_tile_bit_planes,
};
#[cfg(all(test, feature = "parallel"))]
use self::subband::{
    copy_decoded_classic_blocks_to_sub_band, copy_decoded_ht_blocks_to_sub_band, DecodedBlock,
};
#[cfg(test)]
pub(crate) use self::subband::{
    should_decode_classic_sub_band_in_parallel, should_decode_ht_sub_band_in_parallel,
};
pub(crate) use self::subband_params::sub_band_decode_parameters;
use self::subband_params::{
    classic_decode_job_parameters, ht_code_block_has_decodable_passes, SubBandDecodeParameters,
};
pub use self::workspace::{
    CpuDecodeParallelism, DecoderContext, DecoderWorkspace, DecoderWorkspaceStats,
};

#[cfg(test)]
pub(crate) fn decode_with_capacity_retry<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    #[cfg(feature = "std")] directory: Option<
        &alloc::sync::Arc<super::segment::directory::PacketDirectory>,
    >,
) -> Result<()> {
    decode_rows_with_capacity_retry(
        data,
        header,
        retained_image_bytes,
        ctx,
        ht_decoder,
        &mut None,
        #[cfg(feature = "std")]
        directory,
    )
}

/// [`decode_with_capacity_retry`] that hands a region decode's samples to
/// `row_sink` as they are finished, when every tile can stream them (see
/// [`streams_rows`]). Otherwise the samples land in the component planes as
/// usual, and `ctx` records which happened.
pub(crate) fn decode_rows_with_capacity_retry<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    row_sink: &mut Option<&mut dyn ComponentRowSink>,
    #[cfg(feature = "std")] directory: Option<
        &alloc::sync::Arc<super::segment::directory::PacketDirectory>,
    >,
) -> Result<()> {
    run_decode_with_capacity_retry(ctx, |ctx| {
        decode(
            data,
            header,
            retained_image_bytes,
            ctx,
            ht_decoder,
            row_sink,
            #[cfg(feature = "std")]
            directory,
        )
    })
}

fn run_decode_with_capacity_retry<'a>(
    ctx: &mut DecoderContext<'a>,
    mut decode: impl FnMut(&mut DecoderContext<'a>) -> Result<()>,
) -> Result<()> {
    ctx.storage.release_all_allocations();
    let retained_component_bytes = ctx.tile_decode_context.retained_channel_bytes()?;
    let retained_tier1_bytes = ctx.tile_decode_context.tier1_capacity_bytes()?;
    let retained_idwt_bytes = ctx.tile_decode_context.idwt_capacity_bytes()?;
    let retained_scratch_bytes = retained_tier1_bytes
        .checked_add(retained_idwt_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    ctx.record_decode_start(
        retained_component_bytes,
        retained_tier1_bytes,
        retained_idwt_bytes,
    );

    ctx.tile_decode_context.streamed_samples = 0;
    let first_result = decode(ctx);
    let result = ctx.retry_without_retained_scratch_on_capacity(
        retained_scratch_bytes,
        first_result,
        &mut decode,
    );

    ctx.storage.release_all_allocations();
    let retained_component_bytes = ctx.tile_decode_context.retained_channel_bytes()?;
    let retained_tier1_bytes = ctx.tile_decode_context.tier1_capacity_bytes()?;
    let retained_idwt_bytes = ctx.tile_decode_context.idwt_capacity_bytes()?;
    ctx.record_decode_complete(
        retained_component_bytes,
        retained_tier1_bytes,
        retained_idwt_bytes,
    );
    result
}

pub(crate) fn prepare_region_tiles<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<ParsedTiles<'a>> {
    // The parsed graph remains live for the complete region session. Start it
    // from a fresh lifetime-free workspace so its baseline is exact and later
    // stripe allocations can account the retained graph explicitly.
    ctx.release_reusable_allocations();
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header, retained_image_bytes)?;
    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }
    Ok(tiles)
}

pub(crate) fn decode_preparsed_with_capacity_retry<'a>(
    header: &Header<'a>,
    retained_image_bytes: usize,
    tiles: &ParsedTiles<'a>,
    ctx: &mut DecoderContext<'a>,
) -> Result<()> {
    run_decode_with_capacity_retry(ctx, |ctx| {
        decode_preparsed(header, retained_image_bytes, tiles, ctx)
    })
}

fn decode_preparsed<'a>(
    header: &Header<'a>,
    retained_image_bytes: usize,
    tiles: &ParsedTiles<'a>,
    ctx: &mut DecoderContext<'a>,
) -> Result<()> {
    let reused_baseline = ctx.prepare_reused_decode_baseline(retained_image_bytes)?;
    let structural_workspace_bytes = reused_baseline
        .parser_live
        .checked_add(tiles.metadata_owner_bytes())
        .ok_or(ValidationError::ImageTooLarge)?;
    if structural_workspace_bytes > crate::DEFAULT_MAX_DECODE_BYTES {
        bail!(ValidationError::ImageTooLarge);
    }
    let profile_enabled = profile::profile_stages_enabled();
    decode_parsed_tiles(
        header,
        tiles,
        structural_workspace_bytes,
        reused_baseline,
        ctx,
        &mut None,
        &mut None,
        profile_enabled,
        DecodeProfileTimings::default(),
        profile::profile_now(profile_enabled),
    )
}

fn decode<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    row_sink: &mut Option<&mut dyn ComponentRowSink>,
    #[cfg(feature = "std")] directory: Option<
        &alloc::sync::Arc<super::segment::directory::PacketDirectory>,
    >,
) -> Result<()> {
    let mut reused_baseline = ctx.prepare_reused_decode_baseline(retained_image_bytes)?;
    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);
    let mut profile_timings = DecodeProfileTimings::default();
    let stage_start = profile::profile_now(profile_enabled);
    let mut reader = BitReader::new(data);
    let mut parsed_tiles = tile::parse(&mut reader, header, reused_baseline.parser_live);
    if reused_baseline.scratch_capacity == 0
        && reused_baseline.channel_capacity != 0
        && matches!(
            parsed_tiles.as_ref(),
            Err(error) if reuse::is_capacity_error(error)
        )
    {
        // Retained component buffers are an optimization, not a reason for an
        // otherwise valid decode to fail its aggregate allocation cap.
        ctx.discard_reused_channels();
        reused_baseline = reuse::ReusedDecodeBaseline {
            parser_live: retained_image_bytes,
            channel_capacity: 0,
            scratch_capacity: 0,
        };
        reader = BitReader::new(data);
        parsed_tiles = tile::parse(&mut reader, header, reused_baseline.parser_live);
    }
    let tiles = parsed_tiles?;
    #[cfg(feature = "std")]
    let tiles = {
        let mut tiles = tiles;
        tiles.set_packet_directory(directory.cloned());
        tiles
    };
    profile_timings.parse_tiles_us += profile::elapsed_us(stage_start);

    decode_parsed_tiles(
        header,
        &tiles,
        tiles.structural_workspace_bytes(),
        reused_baseline,
        ctx,
        ht_decoder,
        row_sink,
        profile_enabled,
        profile_timings,
        total_start,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the parsed decode boundary keeps graph ownership, allocation baseline, adapter, and profiling state explicit"
)]
fn decode_parsed_tiles<'a>(
    header: &Header<'a>,
    tiles: &ParsedTiles<'a>,
    structural_workspace_bytes: usize,
    reused_baseline: reuse::ReusedDecodeBaseline,
    ctx: &mut DecoderContext<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    row_sink: &mut Option<&mut dyn ComponentRowSink>,
    profile_enabled: bool,
    mut profile_timings: DecodeProfileTimings,
    total_start: Option<profile::ProfileInstant>,
) -> Result<()> {
    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }
    ensure_tiles_share_sample_type(tiles)?;
    let (structural_workspace_bytes, reused_baseline) = choose_region_idwt(
        header,
        tiles,
        ctx,
        (ht_decoder.is_some(), row_sink.is_some()),
        structural_workspace_bytes,
        reused_baseline,
    )?;
    let stream_rows = ctx.tile_decode_context.stream_rows;
    // The row-streamed IDWT writes rows straight into the planes, so they
    // need no padding.
    let last_idwt_level_padding = last_idwt_level_padding(
        header,
        tiles,
        ctx.tile_decode_context.output_region,
        ht_decoder.is_some(),
    )
    .filter(|_| !ctx.tile_decode_context.line_region_idwt);
    #[cfg(test)]
    let last_idwt_level_padding =
        last_idwt_level_padding.filter(|_| !ctx.tile_decode_context.separate_last_idwt_level);
    let retained_decode_baseline = ctx.reset(
        header,
        &tiles[0],
        structural_workspace_bytes,
        reused_baseline.channel_capacity,
        ht_decoder.is_some(),
        last_idwt_level_padding,
    )?;
    let retained_decode_base_without_scratch = retained_decode_baseline
        .checked_sub(reused_baseline.scratch_capacity)
        .ok_or(ValidationError::ImageTooLarge)?;
    let cpu_decode_parallelism = ctx.cpu_decode_parallelism;
    let (tile_ctx, storage) = (&mut ctx.tile_decode_context, &mut ctx.storage);

    // Under Auto, the tiles may decode at once on worker contexts. On any
    // failure the loop below decodes every tile again, in order.
    #[cfg(feature = "parallel")]
    let decoded_by_workers = cpu_decode_parallelism == CpuDecodeParallelism::Auto
        && ht_decoder.is_none()
        && !profile_enabled
        && tile_workers::decode_tiles(header, tiles, tile_ctx, retained_decode_baseline).is_some();
    #[cfg(not(feature = "parallel"))]
    let decoded_by_workers = false;

    tile_ctx.fused_last_level = None;
    for tile in tiles.iter().filter(|_| !decoded_by_workers) {
        ltrace!(
            "tile {} rect [{},{} {}x{}]",
            tile.idx,
            tile.rect.x0,
            tile.rect.y0,
            tile.rect.width(),
            tile.rect.height(),
        );

        if let Some(output_region) = tile_ctx.output_region {
            if !tile_intersects_output_region(tile.rect, &header.size_data, output_region) {
                ltrace!("tile {} outside output region, skipped", tile.idx);
                continue;
            }
        }

        let output = match row_sink.as_deref_mut() {
            Some(sink) if stream_rows => TileOutput::Sink(sink),
            _ => TileOutput::Planes,
        };
        decode_tile(
            tile,
            header,
            tile_ctx,
            storage,
            ht_decoder,
            cpu_decode_parallelism,
            profile_enabled,
            &mut profile_timings,
            retained_decode_base_without_scratch,
            output,
        )?;
    }

    ensure_every_sample_streamed(header, tile_ctx)?;
    tile_ctx.ensure_unzeroed_planes_overwritten()?;

    tile_ctx.deferred_mct = None;
    // Streamed rows were transformed on the way to the sink.
    if tiles.iter().any(|tile| tile.mct) && !stream_rows {
        let stage_start = profile::profile_now(profile_enabled);
        let parallel = cpu_decode_parallelism == CpuDecodeParallelism::Auto;
        apply_inverse_mct(tile_ctx, header, tiles, ht_decoder, parallel)?;
        profile_timings.mct_us += profile::elapsed_us(stage_start);
    }
    // Planes holding a last level's sub-bands need the deferred MCT's pack.
    if tile_ctx.fused_last_level.take().is_some() {
        bail!(DecodingError::IncompletePlaneStore);
    }

    if profile_enabled {
        emit_decode_profile_row(tile_ctx, &profile_timings, total_start);
    }

    // The returned image only borrows channel data. Parsed tile and packet
    // graphs borrow the codestream and must be released. Tier-1 and IDWT
    // owners are lifetime-free and remain available to the next image.
    storage.release_all_allocations();

    Ok(())
}

/// The planes hold integers or floats for the whole image, as the first tile
/// needs, so every tile must need the same.
fn ensure_tiles_share_sample_type(tiles: &[Tile<'_>]) -> Result<()> {
    let exact_integer_decode = tile_requires_exact_integer_decode(&tiles[0]);
    if tiles
        .iter()
        .any(|tile| tile_requires_exact_integer_decode(tile) != exact_integer_decode)
    {
        bail!(DecodingError::UnsupportedFeature(
            "tiles that differ in needing exact integer decode"
        ));
    }
    Ok(())
}

/// A decode that streamed its rows has no planes: the sink must have
/// received every sample of every component, transformed, once.
fn ensure_every_sample_streamed(header: &Header<'_>, tile_ctx: &TileDecodeContext) -> Result<()> {
    if !tile_ctx.stream_rows {
        return Ok(());
    }
    let expected = tile_ctx
        .output_region
        .map(OutputRegion::dimensions)
        .and_then(|(width, height)| checked_decode_sample_count(width, height).ok())
        .and_then(|samples| samples.checked_mul(header.component_infos.len()))
        .ok_or(ValidationError::ImageTooLarge)?;
    if tile_ctx.streamed_samples != expected {
        bail!(DecodingError::IncompletePlaneStore);
    }
    Ok(())
}

/// The inverse MCT of the decoded tiles. One transform of the whole planes
/// serves when every tile has the first tile's MCT and wavelets. Tile-part
/// COD and COC can set them per tile, and then each tile with an MCT
/// transforms its own samples.
fn apply_inverse_mct(
    tile_ctx: &mut TileDecodeContext,
    header: &Header<'_>,
    tiles: &[Tile<'_>],
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    parallel: bool,
) -> Result<()> {
    if !mct_matches_first_tile(tiles) {
        return apply_inverse_mct_per_tile(tile_ctx, header, tiles);
    }
    mct::apply_inverse(
        tile_ctx,
        &tiles[0].component_infos,
        header,
        ht_decoder,
        parallel,
    )?;
    apply_sign_shift_after_mct(tile_ctx, &tiles[0].component_infos);
    Ok(())
}

/// Whether every tile has the first tile's MCT and, with an MCT, its
/// wavelets, which also round the components after the first three.
fn mct_matches_first_tile(tiles: &[Tile<'_>]) -> bool {
    let first = &tiles[0];
    tiles.iter().all(|tile| {
        tile.mct == first.mct
            && (!tile.mct
                || tile.component_infos.iter().zip(&first.component_infos).all(
                    |(component, first)| component.wavelet_transform() == first.wavelet_transform(),
                ))
    })
}

/// The padded size each component plane reserves so that a region decode can
/// write its last IDWT level straight into the plane: the region expanded by
/// the IDWT margin on every side, within the image. `None` when it cannot or
/// would not save memory. A hook may store IDWT output itself, exact-integer
/// and subsampled decodes take other paths, and when several tiles meet the
/// region no one tile owns the plane. Every plane holds its padding for the
/// whole decode while the padding replaces one component's window at a time,
/// so it pays off only when all planes' padding together is smaller than one
/// padded window.
fn last_idwt_level_padding(
    header: &Header<'_>,
    tiles: &ParsedTiles<'_>,
    output_region: Option<OutputRegion>,
    hooked: bool,
) -> Option<(u32, u32)> {
    if !region_decode_without_hook_or_special_samples(tiles, output_region, hooked) {
        return None;
    }
    let output_region = output_region?;
    let mut covering = tiles
        .iter()
        .filter(|tile| tile_intersects_output_region(tile.rect, &header.size_data, output_region));
    let components = &covering.next()?.component_infos;
    if covering.next().is_some() {
        return None;
    }
    let margin = components
        .iter()
        .map(|component| roi_required_output_margin(component.wavelet_transform()))
        .max()?;
    let padded = |start: u32, len: u32, bound: u32| {
        let end = start.checked_add(len)?;
        let after = bound.checked_sub(end)?;
        len.checked_add(margin.min(start))?
            .checked_add(margin.min(after))
    };
    let width = padded(
        output_region.x,
        output_region.width,
        header.size_data.image_width(),
    )?;
    let height = padded(
        output_region.y,
        output_region.height,
        header.size_data.image_height(),
    )?;
    let window = u64::from(width) * u64::from(height);
    let padding = window - u64::from(output_region.width) * u64::from(output_region.height);
    let planes = u64::try_from(components.len()).ok()?;
    (planes.checked_mul(padding)? < window).then_some((width, height))
}

/// Whether a decode is a region decode without a code-block decoder hook,
/// whose components need neither exact-integer samples nor subsampling: the
/// decodes whose region IDWT may write into `channel_data` itself.
fn region_decode_without_hook_or_special_samples(
    tiles: &ParsedTiles<'_>,
    output_region: Option<OutputRegion>,
    hooked: bool,
) -> bool {
    output_region.is_some()
        && !hooked
        && tiles.first().is_some_and(|tile| {
            tile.component_infos.iter().all(|component| {
                !component.requires_exact_integer_decode()
                    && component.size_info.horizontal_resolution == 1
                    && component.size_info.vertical_resolution == 1
            })
        })
}

/// Record in `ctx` whether a region decode runs the row-streamed IDWT and,
/// if it `wants_rows`, whether it streams them (see [`streams_rows`]). A
/// decode that streams gives up the context's retained capacity first (see
/// `release_retained_before_streaming`), and the decode goes on with the
/// returned structural bytes and baseline.
fn choose_region_idwt(
    header: &Header<'_>,
    tiles: &ParsedTiles<'_>,
    ctx: &mut DecoderContext<'_>,
    (hooked, wants_rows): (bool, bool),
    structural_workspace_bytes: usize,
    reused_baseline: reuse::ReusedDecodeBaseline,
) -> Result<(usize, reuse::ReusedDecodeBaseline)> {
    let line_region_idwt = ctx.line_region_idwt
        && ctx.tile_decode_context.line_region_idwt_permitted
        && region_decode_without_hook_or_special_samples(
            tiles,
            ctx.tile_decode_context.output_region,
            hooked,
        );
    ctx.tile_decode_context.line_region_idwt = line_region_idwt;
    let stream_rows = wants_rows
        && line_region_idwt
        && streams_rows(
            header,
            tiles,
            &ctx.tile_decode_context,
            ctx.cpu_decode_parallelism,
        );
    ctx.tile_decode_context.stream_rows = stream_rows;
    ctx.tile_decode_context.streamed_samples = 0;
    if stream_rows {
        ctx.release_retained_before_streaming(structural_workspace_bytes, reused_baseline)
    } else {
        Ok((structural_workspace_bytes, reused_baseline))
    }
}

/// Whether a region decode on the row-streamed IDWT can hand its samples to
/// a row sink instead of component planes: every tile decodes its Tier-1 in
/// bands (see [`tile_bands`]), so all components can advance together; the
/// image and tiles have no offsets; there are one to four components; and
/// each tile with an inverse MCT has three components or more, the first
/// three with one wavelet, as `apply_inverse_mct_per_tile` requires.
/// `ensure_every_sample_streamed` checks afterwards that the rows covered
/// the region.
fn streams_rows(
    header: &Header<'_>,
    tiles: &ParsedTiles<'_>,
    tile_ctx: &TileDecodeContext,
    parallelism: CpuDecodeParallelism,
) -> bool {
    let size = &header.size_data;
    let components = header.component_infos.len();
    size.image_area_x_offset == 0
        && size.image_area_y_offset == 0
        && size.tile_x_offset == 0
        && size.tile_y_offset == 0
        && (1..=4).contains(&components)
        && tiles.iter().all(|tile| {
            let infos = &tile.component_infos;
            infos.len() == components
                // The row-streamed IDWT already excludes hooks.
                && tile_bands(header, tile, tile_ctx, false, parallelism)
                && infos
                    .iter()
                    .all(|component| !component.requires_exact_integer_decode())
                // Each tile's rows take its own MCT, as the planes do. A tile
                // whose MCT the planes reject decodes into them instead, so
                // the decode reports their error.
                && (!tile.mct
                    || (components >= 3
                        && infos[1..3].iter().all(|component| {
                            component.wavelet_transform() == infos[0].wavelet_transform()
                        })))
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputRegion {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl OutputRegion {
    pub(crate) fn from_tuple(region: (u32, u32, u32, u32)) -> Self {
        let (x, y, width, height) = region;
        Self {
            x,
            y,
            width,
            height,
        }
    }

    fn dimensions(self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodeDebugCounters {
    #[cfg(all(test, feature = "std"))]
    pub(crate) packet_headers: usize,
    #[cfg(all(test, feature = "std"))]
    pub(crate) directory_skips: usize,
    pub(crate) decoded_code_blocks: usize,
    pub(crate) skipped_code_blocks: usize,
    pub(crate) idwt_output_samples: usize,
    /// Sub-band coefficients held in storage, summed over decoded tiles.
    pub(crate) stored_coefficients: usize,
    /// Code-blocks built, and the layer records built for them, summed over
    /// decoded tiles. Region decodes give layers only to code-blocks they read.
    pub(crate) built_code_blocks: usize,
    pub(crate) built_layer_records: usize,
    /// Precinct records built, summed over decoded tiles. Region decodes
    /// build none for precincts they never parse.
    pub(crate) built_precincts: usize,
    /// Progression positions region precinct selection walked, summed over
    /// decoded tiles.
    pub(crate) selection_scanned_packets: usize,
    /// Empty decomposition capacity discarded before region selection.
    pub(crate) selection_released_storage_bytes: usize,
    /// Components whose bands decoded on the worker pool ahead of the
    /// streamed IDWT, summed over decoded tiles.
    pub(crate) pool_band_components: usize,
    /// Components that gave up the pool for whole windows because a band
    /// was left running with no worker to finish it.
    pub(crate) pool_band_fallbacks: usize,
    /// The bytes the decode's budget starts from after `reset`, retained
    /// capacity included.
    pub(crate) decode_baseline_bytes: usize,
    /// The largest decode workspace sampled at stage boundaries, by owner.
    pub(crate) workspace_peak: WorkspacePeak,
    pub(crate) ht_phase_stats: ht_block_decode::HtBlockDecodeStats,
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) ht_parallel_tasks: usize,
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) ht_task_workspace_growths: usize,
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) classic_task_workspace_growths: usize,
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) parallel_coefficients: ParallelCoefficientStats,
}

#[cfg(all(test, feature = "parallel"))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParallelCoefficientStats {
    pub(crate) allocations: usize,
    pub(crate) allocated_bytes: usize,
    pub(crate) peak_live_bytes: usize,
    pub(crate) scatter_bytes: usize,
}

#[cfg(all(test, feature = "parallel"))]
impl ParallelCoefficientStats {
    fn record_allocation(&mut self, bytes: usize, peak_live_bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.allocations = self.allocations.saturating_add(1);
        self.allocated_bytes = self.allocated_bytes.saturating_add(bytes);
        self.peak_live_bytes = self.peak_live_bytes.max(peak_live_bytes);
    }

    fn record_retained(&mut self, bytes: usize) {
        self.peak_live_bytes = self.peak_live_bytes.max(bytes);
    }

    fn record_scatter(&mut self, bytes: usize) {
        self.scatter_bytes = self.scatter_bytes.saturating_add(bytes);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
fn decode_tile<'a>(
    tile: &Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
    retained_decode_baseline: usize,
    output: TileOutput<'_, '_>,
) -> Result<()> {
    storage.reset_for_next_tile();
    let retained_decode_baseline = retained_decode_baseline
        .checked_add(tile_ctx.retained_scratch_bytes()?)
        .ok_or(ValidationError::ImageTooLarge)?;
    if retained_decode_baseline > crate::DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    storage.exact_integer_decode = tile_requires_exact_integer_decode(tile);
    if storage.exact_integer_decode {
        validate_exact_integer_decode_tile(tile)?;
        if tile_ctx.output_region.is_some() {
            bail!(DecodingError::UnsupportedFeature(
                "25-38 bit region decode requires exact integer region IDWT support"
            ));
        }
    }

    // This is the method that orchestrates all steps.

    // First, we build the decompositions, including their sub-bands, precincts
    // and code blocks.
    let stage_start = profile::profile_now(profile_enabled);
    // Region decode builds its ROI plan first so that precincts the decode
    // will never parse get no code-blocks or tag trees.
    let selection = match tile_ctx.output_region {
        Some(output_region) => build::select_region_precincts(
            tile,
            header,
            storage,
            output_region,
            retained_decode_baseline,
        )?,
        None => None,
    };
    // The selection stays live through packet parsing, so its bytes are part
    // of the baseline every later phase budgets against until it is dropped.
    let selection_bytes = selection
        .as_ref()
        .map_or(0, build::PrecinctSelection::owned_bytes);
    tile_ctx.debug_counters.selection_scanned_packets += selection
        .as_ref()
        .map_or(0, build::PrecinctSelection::scanned_packets);
    tile_ctx.debug_counters.selection_released_storage_bytes += selection
        .as_ref()
        .map_or(0, build::PrecinctSelection::released_storage_bytes);
    build::build(
        tile,
        storage,
        retained_decode_baseline
            .checked_add(selection_bytes)
            .ok_or(ValidationError::ImageTooLarge)?,
        tile_ctx.output_region.is_some(),
        build::BuildWorkspace::DecodePixels {
            skipped_resolution_levels: header.skipped_resolution_levels,
            // A code-block decoder hook may rely on every component's
            // coefficients being live together, so it keeps the tile layout,
            // as do streamed rows, which decode the components together.
            coefficient_layout: if ht_decoder.is_some() || tile_ctx.stream_rows {
                build::CoefficientLayout::Tile
            } else {
                build::CoefficientLayout::Component
            },
        },
        selection.as_ref(),
    )?;
    if tile_ctx.output_region.is_some() {
        prepare_region_storage(
            tile,
            header,
            tile_ctx,
            storage,
            ht_decoder.is_some(),
            cpu_decode_parallelism,
        )?;
    }
    tile_ctx.debug_counters.stored_coefficients += storage.coefficients.len();
    tile_ctx.debug_counters.built_code_blocks += storage.code_blocks.len();
    tile_ctx.debug_counters.built_layer_records += storage.layers.len();
    tile_ctx.debug_counters.built_precincts += storage.precincts.len();
    record_workspace(tile_ctx, storage, selection_bytes)?;
    profile_timings.build_us += profile::elapsed_us(stage_start);
    // Next, we parse the layers/segments for each code block.
    let stage_start = profile::profile_now(profile_enabled);
    let packet_stats = segment::parse_selected(tile, header, storage, selection.as_ref())?;
    #[cfg(not(all(test, feature = "std")))]
    let _ = packet_stats;
    #[cfg(all(test, feature = "std"))]
    {
        tile_ctx.debug_counters.packet_headers += packet_stats.parsed_headers;
        tile_ctx.debug_counters.directory_skips += packet_stats.skipped_packets;
    }
    record_workspace(tile_ctx, storage, selection_bytes)?;
    drop(selection);
    storage.structural_workspace_bytes = storage
        .structural_workspace_bytes
        .checked_sub(selection_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    profile_timings.segment_us += profile::elapsed_us(stage_start);
    decode_components(
        tile,
        header,
        tile_ctx,
        storage,
        ht_decoder,
        cpu_decode_parallelism,
        profile_enabled,
        profile_timings,
        output,
    )?;
    Ok(())
}

/// After `build`, choose how a region decode's IDWT and Tier-1 run and size
/// its coefficient storage to match.
fn prepare_region_storage(
    tile: &Tile<'_>,
    header: &Header<'_>,
    tile_ctx: &TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    hooked: bool,
    cpu_decode_parallelism: CpuDecodeParallelism,
) -> Result<()> {
    if storage.roi_plan.is_none() {
        build::release_unused_roi_workspace(storage, tile.component_infos.len())?;
    }
    storage.line_region_idwt = tile_ctx.line_region_idwt && storage.roi_plan.is_some();
    storage.last_idwt_level_in_channels =
        !storage.line_region_idwt && idwt::last_level_fits_channels(storage, tile_ctx, header)?;
    // Region decode sizes coefficient storage to the code-blocks the ROI
    // plan keeps, clipped to the windows the IDWT reads unless a hook
    // writes whole code-blocks, or to the whole tile when there is no plan.
    let clip_to_idwt_windows = clips_to_idwt_windows(tile_ctx, hooked);
    storage.banded_tier1 = storage.line_region_idwt
        && tile_bands(header, tile, tile_ctx, hooked, cpu_decode_parallelism);
    storage.lockstep_lines = storage.banded_tier1 && tile_ctx.stream_rows;
    #[cfg(feature = "parallel")]
    {
        storage.parallel_bands = storage.line_region_idwt
            && !storage.banded_tier1
            && parallel_bands::applies(header, tile, tile_ctx, hooked, cpu_decode_parallelism);
    }
    build::allocate_deferred_coefficients(
        storage,
        header.skipped_resolution_levels,
        clip_to_idwt_windows,
    )?;
    Ok(())
}

/// Whether a tile clips its stored windows to the ones the IDWT reads: a hook
/// writes whole code-blocks instead.
fn clips_to_idwt_windows(tile_ctx: &TileDecodeContext, hooked: bool) -> bool {
    #[cfg(test)]
    return !hooked && !tile_ctx.store_whole_code_blocks;
    #[cfg(not(test))]
    {
        let _ = tile_ctx;
        !hooked
    }
}

/// Whether `tile` decodes its Tier-1 in bands on this decode's row-streamed
/// IDWT (see [`banded_tier1::applies`]). The tile's layout and
/// [`streams_rows`], which needs every tile to band, share this check.
fn tile_bands(
    header: &Header<'_>,
    tile: &Tile<'_>,
    tile_ctx: &TileDecodeContext,
    hooked: bool,
    parallelism: CpuDecodeParallelism,
) -> bool {
    let bands = tile_ctx.line_region_idwt
        && banded_tier1::applies(
            header,
            tile,
            parallelism,
            clips_to_idwt_windows(tile_ctx, hooked),
        );
    #[cfg(test)]
    let bands = bands && !tile_ctx.whole_window_tier1;
    bands
}

/// Where a decoded tile's samples go.
enum TileOutput<'o, 'p> {
    /// The decode's component planes.
    Planes,
    /// A tile worker's own rows of the planes.
    #[cfg(feature = "parallel")]
    Rows(&'o mut TileRows<'p>),
    /// A row sink, as the streamed IDWT finishes them; see [`streams_rows`].
    Sink(&'o mut (dyn ComponentRowSink + 'p)),
}

/// Runs Tier-1, the IDWT and store for each component of a parsed tile.
///
/// With [`build::CoefficientLayout::Component`], every component decodes into
/// the same coefficient buffer, so each finishes all three stages before the
/// next is decoded. Otherwise Tier-1 decodes the whole tile first. With
/// [`TileOutput::Rows`], a tile worker stores into its own rows of the
/// planes, and with [`TileOutput::Sink`], the tile's rows go to the sink.
#[expect(
    clippy::too_many_arguments,
    reason = "the per-component stages share decode_tile's state, adapter, and profiling explicitly"
)]
fn decode_components<'a>(
    tile: &Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
    mut output: TileOutput<'_, '_>,
) -> Result<()> {
    if let TileOutput::Sink(sink) = output {
        return banded_tier1::decode_tile_rows(
            tile,
            header,
            tile_ctx,
            storage,
            sink,
            profile_enabled,
            profile_timings,
        );
    }
    let per_component = storage.coefficient_layout == build::CoefficientLayout::Component;
    if !per_component {
        decode_tier1_stage(
            tile,
            0..tile.component_infos.len(),
            header,
            tile_ctx,
            storage,
            ht_decoder,
            cpu_decode_parallelism,
            profile_enabled,
            profile_timings,
        )?;
    }

    // Unlike before, we interleave the apply_idwt and store stages
    // for each component tile so we can reuse allocations better.
    tile_ctx.fused_last_level = matches!(output, TileOutput::Planes)
        .then(|| fused_last_level(tile, header, tile_ctx, storage, ht_decoder.is_some()))
        .flatten();
    let fused = tile_ctx.fused_last_level.is_some();
    if !fused {
        tile_ctx.prepare_deferred_plane()?;
    }
    // A tile-part COD or COC can give the tile its own coding style.
    for (idx, component_info) in tile.component_infos.iter().enumerate() {
        if decode_component_bands(
            tile,
            idx,
            header,
            tile_ctx,
            storage,
            component_info.wavelet_transform(),
            profile_enabled,
            profile_timings,
        )? {
            record_workspace(tile_ctx, storage, 0)?;
        } else {
            if per_component {
                // Code-blocks the decode skips leave their samples untouched, so
                // Tier-1 clears the previous component's coefficients first.
                storage.stale_coefficients = idx > 0;
                decode_tier1_stage(
                    tile,
                    idx..idx + 1,
                    header,
                    tile_ctx,
                    storage,
                    ht_decoder,
                    cpu_decode_parallelism,
                    profile_enabled,
                    profile_timings,
                )?;
            }
            // Next, we apply the inverse discrete wavelet transform. A last level
            // left to the pack ends at its LL band.
            let stage_start = profile::profile_now(profile_enabled);
            let idwt_result = idwt::apply(
                storage,
                tile_ctx,
                idx,
                header,
                component_info.wavelet_transform(),
                ht_decoder,
                cpu_decode_parallelism == CpuDecodeParallelism::Auto,
            );
            idwt_result?;
            record_workspace(tile_ctx, storage, 0)?;
            profile_timings.idwt_us += profile::elapsed_us(stage_start);
        }
        // Finally, we store the raw samples for the tile area in the correct
        // location. Note that in case we have MCT, we are not applying it yet.
        // It will be applied in the very end once all tiles have been processed.
        // The reason we do this is that applying MCT requires access to the
        // data from _all_ components. If we didn't defer this until the end
        // we would have to collect the IDWT outputs of all components before
        // applying it. By not applying MCT here, we can get away with doing
        // IDWT and store on a per-component basis. Thus, we only need to
        // store one IDWT output at a time, allowing for better reuse of
        // allocations.
        let stage_start = profile::profile_now(profile_enabled);
        let parallel = cpu_decode_parallelism == CpuDecodeParallelism::Auto;
        if let Some(level) = tile_ctx.fused_last_level {
            store_last_level_bands(tile_ctx, storage, idx, &level)?;
        } else {
            match &mut output {
                #[cfg(feature = "parallel")]
                TileOutput::Rows(rows) => {
                    store_rows(rows, tile, header, tile_ctx, component_info, idx, parallel)?;
                }
                _ => store(
                    tile,
                    header,
                    tile_ctx,
                    component_info,
                    idx,
                    ht_decoder,
                    parallel,
                )?,
            }
        }
        record_workspace(tile_ctx, storage, 0)?;
        profile_timings.store_us += profile::elapsed_us(stage_start);
    }

    Ok(())
}

/// Decode component `component_idx` of a tile that decodes in bands as the
/// streamed IDWT reads them, on the pool or serially. Returns `false` when it
/// takes the whole-window path instead.
#[expect(
    clippy::too_many_arguments,
    reason = "the banded stage shares decode_tile's state and profiling explicitly"
)]
fn decode_component_bands<'a>(
    tile: &Tile<'a>,
    component_idx: usize,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    transform: WaveletTransform,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
) -> Result<bool> {
    // Unless the route declines and the sub-bands hold whole windows again.
    #[cfg(feature = "parallel")]
    if storage.parallel_bands {
        return parallel_bands::decode_component(
            tile,
            component_idx,
            header,
            tile_ctx,
            storage,
            transform,
            profile_enabled,
            profile_timings,
        );
    }
    if !storage.banded_tier1 {
        return Ok(false);
    }
    banded_tier1::decode_component(
        tile,
        component_idx,
        header,
        tile_ctx,
        storage,
        transform,
        profile_enabled,
        profile_timings,
    )?;
    Ok(true)
}

/// Decode the code-blocks of `components` and sample the workspace after.
#[expect(
    clippy::too_many_arguments,
    reason = "the Tier-1 stage shares decode_tile's state, adapter, and profiling explicitly"
)]
fn decode_tier1_stage<'a>(
    tile: &Tile<'a>,
    components: Range<usize>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
) -> Result<()> {
    // We decode the bitplanes of each code block, yielding the (possibly
    // dequantized) coefficients of each code block.
    let stage_start = profile::profile_now(profile_enabled);
    decode_component_tile_bit_planes_budgeted(
        tile,
        components,
        tile_ctx,
        storage,
        header,
        ht_decoder,
        cpu_decode_parallelism,
        profile_enabled,
    )?;
    record_workspace(tile_ctx, storage, 0)?;
    profile_timings.codeblock_us += profile::elapsed_us(stage_start);
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub(crate) fn decode_component_tile_bit_planes_budgeted<'a>(
    tile: &Tile<'a>,
    components: Range<usize>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
) -> Result<()> {
    let tier1_workspace_bytes =
        tier1::prepare_tier1_workspace(tile, components.clone(), header, tile_ctx, storage)?;
    let decode_result = decode_component_tile_bit_planes(
        tile,
        components,
        tile_ctx,
        storage,
        header,
        ht_decoder,
        cpu_decode_parallelism,
        profile_enabled,
    );
    tier1::release_tier1_workspace(tile_ctx, storage, &tier1_workspace_bytes)?;
    decode_result
}

fn tile_requires_exact_integer_decode(tile: &Tile<'_>) -> bool {
    tile.component_infos
        .iter()
        .any(ComponentInfo::requires_exact_integer_decode)
}

pub(crate) fn decode_selected_wavelet_bands<'a>(
    tile: &Tile<'a>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
    parallelism: CpuDecodeParallelism,
    selection: &j2k_types::WaveletSelection,
) -> Result<()> {
    let workspace = tier1::prepare_tier1_workspace(
        tile,
        0..tile.component_infos.len(),
        header,
        tile_ctx,
        storage,
    )?;
    let result = (|| {
        for (component_index, info) in tile.component_infos.iter().enumerate() {
            if !super::wavelets::select_component(selection, component_index) {
                continue;
            }
            storage.exact_integer_decode =
                info.wavelet_transform() == WaveletTransform::Reversible53;
            for resolution in 0..info.num_resolution_levels() {
                let composition = &storage.tile_decompositions[component_index];
                for index in composition.sub_band_iter(resolution, &storage.decompositions) {
                    if !super::wavelets::select_band(
                        selection,
                        resolution,
                        storage.sub_bands[index].sub_band_type,
                    ) {
                        continue;
                    }
                    subband::decode_sub_band_bitplanes(
                        index,
                        resolution,
                        info,
                        tile_ctx,
                        storage,
                        header,
                        &mut None,
                        parallelism,
                        false,
                    )?;
                }
            }
        }
        Ok(())
    })();
    tier1::release_tier1_workspace(tile_ctx, storage, &workspace)?;
    result
}

fn validate_exact_integer_decode_tile(tile: &Tile<'_>) -> Result<()> {
    for component in &tile.component_infos {
        if component.size_info.precision > 38 {
            bail!(DecodingError::UnsupportedFeature(
                "JPEG 2000 Part 1 component precision is limited to 38 bits"
            ));
        }
        if component.wavelet_transform() != WaveletTransform::Reversible53 {
            bail!(DecodingError::UnsupportedFeature(
                "25-38 bit decode currently requires reversible 5/3 coding"
            ));
        }
        if component.quantization_info.quantization_style != QuantizationStyle::NoQuantization {
            bail!(DecodingError::UnsupportedFeature(
                "25-38 bit decode currently requires reversible no-quantization coding"
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
#[expect(
    clippy::struct_field_names,
    reason = "the repeated _us suffix makes every profiling unit explicit"
)]
struct DecodeProfileTimings {
    parse_tiles_us: u128,
    build_us: u128,
    segment_us: u128,
    codeblock_us: u128,
    idwt_us: u128,
    store_us: u128,
    mct_us: u128,
}

#[cold]
#[inline(never)]
fn emit_decode_profile_row(
    tile_ctx: &TileDecodeContext,
    profile_timings: &DecodeProfileTimings,
    total_start: Option<profile::ProfileInstant>,
) {
    let peak = tile_ctx.debug_counters.workspace_peak.bytes();
    let widen = |bytes: usize| bytes as u128;
    profile::emit_profile_row(
        "decode",
        "cpu",
        &[
            ("parse_tiles_us", profile_timings.parse_tiles_us),
            ("build_us", profile_timings.build_us),
            ("segment_us", profile_timings.segment_us),
            ("codeblock_us", profile_timings.codeblock_us),
            ("ht_blocks", tile_ctx.debug_counters.ht_phase_stats.blocks),
            (
                "ht_refinement_blocks",
                tile_ctx.debug_counters.ht_phase_stats.refinement_blocks,
            ),
            (
                "ht_cleanup_bytes",
                tile_ctx.debug_counters.ht_phase_stats.cleanup_bytes,
            ),
            (
                "ht_refinement_bytes",
                tile_ctx.debug_counters.ht_phase_stats.refinement_bytes,
            ),
            (
                "ht_cleanup_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_cleanup_us,
            ),
            (
                "ht_mag_sgn_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_mag_sgn_us,
            ),
            (
                "ht_sigma_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_sigma_us,
            ),
            (
                "ht_sigprop_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_sigprop_us,
            ),
            (
                "ht_magref_us",
                tile_ctx.debug_counters.ht_phase_stats.ht_magref_us,
            ),
            ("idwt_us", profile_timings.idwt_us),
            ("store_us", profile_timings.store_us),
            ("mct_us", profile_timings.mct_us),
            ("total_us", profile::elapsed_us(total_start)),
            ("peak_workspace_bytes", widen(peak.total())),
            ("peak_coefficient_bytes", widen(peak.coefficients)),
            ("peak_channel_bytes", widen(peak.channels)),
            ("peak_idwt_bytes", widen(peak.idwt)),
            ("peak_tier1_bytes", widen(peak.tier1)),
            ("peak_structure_bytes", widen(peak.structure)),
        ],
    );
}

/// All decompositions for a single tile.
#[derive(Clone)]
pub(crate) struct TileDecompositions {
    pub(crate) first_ll_sub_band: usize,
    pub(crate) decompositions: Range<usize>,
}

impl TileDecompositions {
    pub(crate) fn sub_band_iter(
        &self,
        resolution: u8,
        decompositions: &[Decomposition],
    ) -> SubBandIter {
        let indices = if resolution == 0 {
            [
                self.first_ll_sub_band,
                self.first_ll_sub_band,
                self.first_ll_sub_band,
            ]
        } else {
            decompositions[self.decompositions.clone()][resolution as usize - 1].sub_bands
        };

        SubBandIter {
            next_idx: 0,
            indices,
            resolution,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SubBandIter {
    resolution: u8,
    next_idx: usize,
    indices: [usize; 3],
}

impl Iterator for SubBandIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.resolution == 0 {
            if self.next_idx > 0 {
                None
            } else {
                Some(self.indices[0])
            }
        } else if self.next_idx >= self.indices.len() {
            None
        } else {
            Some(self.indices[self.next_idx])
        };

        self.next_idx += 1;

        value
    }
}

/// Owned decomposition workspace for one active tile.
///
/// Within an image, reset retains capacities and the next build counts them in
/// its live baseline. Crossing to a new image drops every retained allocation.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the integer, stale-plane, in-channel last level, row-streamed IDWT and banded Tier-1 switches are independent"
)]
pub(crate) struct DecompositionStorage<'a> {
    pub(crate) segments: Vec<Segment<'a>>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) code_blocks: Vec<CodeBlock>,
    pub(crate) precincts: Vec<Precinct>,
    pub(crate) tag_tree_nodes: Vec<TagNode>,
    pub(crate) coefficients: Vec<f32>,
    pub(crate) coefficients_i64: Vec<i64>,
    pub(crate) sub_bands: Vec<SubBand>,
    pub(crate) decompositions: Vec<Decomposition>,
    pub(crate) tile_decompositions: Vec<TileDecompositions>,
    pub(crate) roi_plan: Option<RoiPlan>,
    pub(crate) exact_integer_decode: bool,
    /// How the built sub-bands' coefficient ranges share `coefficients`.
    pub(crate) coefficient_layout: build::CoefficientLayout,
    /// Whether `coefficients` still holds the previous component's samples.
    /// Tier-1 must zero what it does not decode before the IDWT reads it.
    pub(crate) stale_coefficients: bool,
    /// Whether the region IDWT writes each component's last level into the
    /// padding of its `channel_data` plane, for `store` to compact in place.
    pub(crate) last_idwt_level_in_channels: bool,
    /// Whether the region IDWT streams this tile's levels a row at a time.
    pub(crate) line_region_idwt: bool,
    /// Whether Tier-1 decodes each component in bands of code-block rows as
    /// the streamed IDWT reaches them. Each sub-band's `coefficients` then
    /// hold one band of its stored rectangle's rows.
    pub(crate) banded_tier1: bool,
    /// Whether Tier-1 decodes each component's bands ahead of the streamed
    /// IDWT on the worker pool (see [`parallel_bands`]). Each sub-band's
    /// `coefficients` then hold a ring of bands.
    pub(crate) parallel_bands: bool,
    /// Whether every component's IDWT rings, and a scratch row each, are
    /// live at once, because the tile streams its rows to a sink.
    pub(crate) lockstep_lines: bool,
    /// The f32 samples of IDWT workspace the ROI plan charged for this tile.
    pub(crate) roi_idwt_charge_elements: usize,
    /// While a band is decoded, the first row of its code-blocks: Tier-1
    /// decodes only the code-blocks that start there, into the rows the
    /// sub-band's `storage_rect` then names.
    pub(crate) tier1_band: Option<u32>,
    /// Planned non-segment live bytes for the active tile and future decode scratch.
    pub(crate) structural_workspace_bytes: usize,
    /// Set when region decode deferred coefficient storage until the ROI
    /// plan is known. Holds the whole-plane IDWT workspace the plan left out,
    /// which must be charged if decode falls back to the whole tile.
    pub(crate) deferred_idwt_workspace_elements: Option<usize>,
    /// Allocation/cap failure raised from the option-based packet parser.
    pub(crate) packet_workspace_error: Option<DecodeError>,
    /// The scratch precincts parsed-only precincts borrow during parsing.
    pub(crate) parse_scratch: Option<build::ParseScratch>,
}

impl DecompositionStorage<'_> {
    pub(crate) fn reset_for_next_tile(&mut self) {
        self.segments.clear();
        self.layers.clear();
        self.code_blocks.clear();
        self.precincts.clear();
        self.tag_tree_nodes.clear();
        self.coefficients.clear();
        self.coefficients_i64.clear();
        self.sub_bands.clear();
        self.decompositions.clear();
        self.tile_decompositions.clear();
        self.roi_plan = None;
        self.exact_integer_decode = false;
        self.coefficient_layout = build::CoefficientLayout::Tile;
        self.stale_coefficients = false;
        self.last_idwt_level_in_channels = false;
        self.line_region_idwt = false;
        self.banded_tier1 = false;
        self.parallel_bands = false;
        self.lockstep_lines = false;
        self.roi_idwt_charge_elements = 0;
        self.tier1_band = None;
        self.structural_workspace_bytes = 0;
        self.deferred_idwt_workspace_elements = None;
        self.packet_workspace_error = None;
        self.parse_scratch = None;
    }

    pub(crate) fn release_all_allocations(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn retained_capacity_bytes(&self) -> Result<usize> {
        let mut bytes = 0_usize;
        include_capacity::<Segment<'_>>(&mut bytes, self.segments.capacity())?;
        include_capacity::<Layer>(&mut bytes, self.layers.capacity())?;
        include_capacity::<CodeBlock>(&mut bytes, self.code_blocks.capacity())?;
        include_capacity::<Precinct>(&mut bytes, self.precincts.capacity())?;
        include_capacity::<TagNode>(&mut bytes, self.tag_tree_nodes.capacity())?;
        include_capacity::<f32>(&mut bytes, self.coefficients.capacity())?;
        include_capacity::<i64>(&mut bytes, self.coefficients_i64.capacity())?;
        include_capacity::<SubBand>(&mut bytes, self.sub_bands.capacity())?;
        include_capacity::<Decomposition>(&mut bytes, self.decompositions.capacity())?;
        include_capacity::<TileDecompositions>(&mut bytes, self.tile_decompositions.capacity())?;
        Ok(bytes)
    }
}

fn include_capacity<T>(bytes: &mut usize, capacity: usize) -> Result<()> {
    let additional = capacity
        .checked_mul(size_of::<T>())
        .ok_or(ValidationError::ImageTooLarge)?;
    *bytes = bytes
        .checked_add(additional)
        .ok_or(ValidationError::ImageTooLarge)?;
    if *bytes > crate::DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(())
}

/// A reusable context used during the decoding of a single tile.
///
/// Some of the fields are temporary in nature and reset after moving on to the
/// next tile, some contain global state.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the rounding, deferred-MCT and row-streamed IDWT switches are independent, and test builds add switches that pin the per-sub-band HT, whole-code-block storage, separate last-level IDWT and staged RGB8 pack paths"
)]
pub(crate) struct TileDecodeContext {
    /// A reusable buffer for the IDWT output.
    pub(crate) idwt_output: IDWTOutput,
    /// A scratch buffer used during IDWT.
    pub(crate) idwt_scratch_buffer: Vec<f32>,
    /// Per-worker band buffers for the row-banded IDWT.
    pub(crate) idwt_band_scratch: Vec<f32>,
    /// A scratch buffer used during exact reversible integer IDWT.
    pub(crate) idwt_scratch_buffer_i64: Vec<i64>,
    /// The rings of rows the row-streamed region IDWT keeps for each level.
    pub(crate) idwt_line_rings: Vec<f32>,
    /// A reusable context for decoding code blocks.
    pub(crate) bit_plane_decode_context: BitPlaneDecodeContext,
    /// Reusable buffers for decoding bitplanes.
    pub(crate) bit_plane_decode_buffers: BitPlaneDecodeBuffers,
    /// A reusable context for decoding HTJ2K code blocks.
    pub(crate) ht_block_decode_context: HtBlockDecodeContext,
    #[cfg(feature = "parallel")]
    pub(in crate::j2c::decode) ht_task_workspaces: Vec<workspace::HtTaskWorkspace>,
    /// One classic workspace per worker of the tile-wide route.
    #[cfg(feature = "parallel")]
    pub(in crate::j2c::decode) classic_task_workspaces:
        Vec<workspace::TaskWorkspace<J2kCodeBlockDecodeWorkspace>>,
    #[cfg(feature = "parallel")]
    pub(in crate::j2c::decode) parallel_coefficients: Vec<f32>,
    /// Test-only switch that keeps tiles off the tile-wide route, on the
    /// per-sub-band path, so its slab and bank behavior stays covered.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) force_per_sub_band: bool,
    /// One-shot fault injection after a tile route has written coefficients.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) tile_route_post_decode: Option<fn(&mut DecompositionStorage<'_>) -> Result<()>>,
    /// Test-only switch that stores whole code-blocks in region decode, as a
    /// hooked decode does, so clipped storage can be compared against it.
    #[cfg(test)]
    pub(crate) store_whole_code_blocks: bool,
    /// Test-only switch that decodes Tier-1 over whole windows before the
    /// streamed IDWT, as a parallel decode does, so banded Tier-1 can be
    /// compared against it.
    #[cfg(test)]
    pub(crate) whole_window_tier1: bool,
    /// Test-only: the first pool worker to start a band abandons it, so the
    /// band is left running with no worker to finish it.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) strand_pool_band: bool,
    /// Test-only: the average work per band below which a component's
    /// bands decode over whole windows instead of on the pool; zero, so
    /// unit tests' tiny fixtures take the pool, unless a test sets it.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) band_work_cutoff: usize,
    /// Test-only capacity failure when building the optional band row index.
    #[cfg(test)]
    pub(crate) reject_band_index: bool,
    /// Test-only switch that keeps the region IDWT's last level in its own
    /// window, as a hooked decode does, so writing it into `channel_data`
    /// can be compared against it.
    #[cfg(test)]
    pub(crate) separate_last_idwt_level: bool,
    /// Test-only switch that keeps an RGB8 decode's last IDWT level and its
    /// pack apart, so the fused pass can be compared against them.
    #[cfg(test)]
    pub(crate) staged_rgb8_pack: bool,
    /// The raw, decoded samples for each channel.
    pub(crate) channel_data: Vec<ComponentData>,
    /// Optional output window for region-local decode storage.
    pub(crate) output_region: Option<OutputRegion>,
    /// Whether irreversible samples are being decoded for an integer output.
    pub(crate) round_irreversible_output: bool,
    /// Whether the inverse MCT is left to the caller's RGB8 pack, set
    /// around a decode like `round_irreversible_output`.
    pub(crate) defer_mct: bool,
    /// The inverse MCT this decode left to the caller.
    pub(crate) deferred_mct: Option<super::mct::DeferredMct>,
    /// The last IDWT level this decode leaves to the RGB8 pack with the
    /// deferred MCT: the IDWT stops one level short, at the level's LL band,
    /// and each plane receives the level's other sub-bands instead of samples.
    pub(crate) fused_last_level: Option<idwt::LastLevel>,
    /// Samples, over all components, that `store` must write because the
    /// planes were reused without zeroing; zero when the planes were zeroed.
    pub(crate) unzeroed_plane_samples: usize,
    /// Component-plane samples `store` wrote during the current decode.
    pub(crate) stored_plane_samples: usize,
    /// The sample count of the last component's plane when `reset` left it
    /// unallocated for a fused RGB8 decode to fill with its coefficients.
    pub(crate) deferred_plane: Option<usize>,
    /// Set for a region decode whose component planes reserve room for this
    /// padded size, so a tile covering the whole region can write its last
    /// IDWT level into them.
    pub(crate) last_idwt_level_padding: Option<(u32, u32)>,
    /// Whether the current call may use the row-streamed region IDWT. Set
    /// around a decode by the outputs that allow it, like `output_region`.
    pub(crate) line_region_idwt_permitted: bool,
    /// Whether this decode streams the region IDWT a row at a time.
    pub(crate) line_region_idwt: bool,
    /// Whether this decode hands the region's samples to a row sink as the
    /// streamed IDWT finishes them, with no component planes.
    pub(crate) stream_rows: bool,
    /// Samples, over all components, handed to the row sink this decode.
    pub(crate) streamed_samples: usize,
    /// Debug counters for tests and ROI instrumentation.
    pub(crate) debug_counters: DecodeDebugCounters,
}

impl TileDecodeContext {
    fn release_all_allocations(&mut self) {
        let output_region = self.output_region;
        let round_irreversible_output = self.round_irreversible_output;
        let defer_mct = self.defer_mct;
        let line_region_idwt_permitted = self.line_region_idwt_permitted;
        #[cfg(all(test, feature = "parallel"))]
        let force_per_sub_band = self.force_per_sub_band;
        #[cfg(all(test, feature = "parallel"))]
        let tile_route_post_decode = self.tile_route_post_decode;
        #[cfg(test)]
        let store_whole_code_blocks = self.store_whole_code_blocks;
        #[cfg(test)]
        let whole_window_tier1 = self.whole_window_tier1;
        #[cfg(test)]
        let reject_band_index = self.reject_band_index;
        #[cfg(test)]
        let separate_last_idwt_level = self.separate_last_idwt_level;
        #[cfg(test)]
        let staged_rgb8_pack = self.staged_rgb8_pack;
        *self = Self::default();
        self.output_region = output_region;
        self.round_irreversible_output = round_irreversible_output;
        self.defer_mct = defer_mct;
        self.line_region_idwt_permitted = line_region_idwt_permitted;
        #[cfg(all(test, feature = "parallel"))]
        {
            self.force_per_sub_band = force_per_sub_band;
            self.tile_route_post_decode = tile_route_post_decode;
        }
        #[cfg(test)]
        {
            self.store_whole_code_blocks = store_whole_code_blocks;
            self.whole_window_tier1 = whole_window_tier1;
            self.reject_band_index = reject_band_index;
            self.separate_last_idwt_level = separate_last_idwt_level;
            self.staged_rgb8_pack = staged_rgb8_pack;
        }
    }

    fn release_tile_scratch_allocations(&mut self) {
        self.release_idwt_allocations();
        self.release_tier1_allocations();
    }

    fn release_tier1_allocations(&mut self) {
        self.bit_plane_decode_context = BitPlaneDecodeContext::default();
        self.bit_plane_decode_buffers = BitPlaneDecodeBuffers::default();
        self.ht_block_decode_context = HtBlockDecodeContext::default();
        #[cfg(feature = "parallel")]
        {
            self.ht_task_workspaces = Vec::new();
            self.classic_task_workspaces = Vec::new();
            self.parallel_coefficients = Vec::new();
        }
    }

    pub(crate) fn serial_tier1_capacity_bytes(&self) -> Result<usize> {
        let classic_bytes = self.bit_plane_decode_context.allocated_bytes()?;
        let buffer_bytes = self.bit_plane_decode_buffers.allocated_bytes()?;
        let ht_bytes = self.ht_block_decode_context.allocated_bytes()?;
        classic_bytes
            .checked_add(buffer_bytes)
            .and_then(|bytes| bytes.checked_add(ht_bytes))
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    pub(crate) fn tier1_capacity_bytes(&self) -> Result<usize> {
        let serial_bytes = self.serial_tier1_capacity_bytes()?;
        #[cfg(feature = "parallel")]
        {
            let classic_bytes = workspace::task_workspace_bytes(&self.classic_task_workspaces)?;
            serial_bytes
                .checked_add(workspace::task_workspace_bytes(&self.ht_task_workspaces)?)
                .and_then(|bytes| bytes.checked_add(classic_bytes))
                .and_then(|bytes| {
                    self.parallel_coefficients
                        .capacity()
                        .checked_mul(size_of::<f32>())
                        .and_then(|coefficient_bytes| bytes.checked_add(coefficient_bytes))
                })
                .ok_or(ValidationError::ImageTooLarge.into())
        }
        #[cfg(not(feature = "parallel"))]
        {
            Ok(serial_bytes)
        }
    }

    #[cfg(all(test, feature = "parallel"))]
    pub(crate) fn ht_parallel_workspace_count(&self) -> usize {
        self.ht_task_workspaces.len()
    }

    #[cfg(all(test, feature = "parallel"))]
    pub(crate) fn ht_parallel_workspace_bytes(&self) -> Result<usize> {
        workspace::task_workspace_bytes(&self.ht_task_workspaces)
    }

    pub(crate) fn idwt_capacity_bytes(&self) -> Result<usize> {
        let mut bytes = 0_usize;
        include_capacity::<f32>(&mut bytes, self.idwt_output.coefficients.capacity())?;
        include_capacity::<i64>(&mut bytes, self.idwt_output.coefficients_i64.capacity())?;
        include_capacity::<f32>(&mut bytes, self.idwt_scratch_buffer.capacity())?;
        include_capacity::<f32>(&mut bytes, self.idwt_band_scratch.capacity())?;
        include_capacity::<i64>(&mut bytes, self.idwt_scratch_buffer_i64.capacity())?;
        include_capacity::<f32>(&mut bytes, self.idwt_line_rings.capacity())?;
        Ok(bytes)
    }

    pub(crate) fn retained_scratch_bytes(&self) -> Result<usize> {
        self.tier1_capacity_bytes()?
            .checked_add(self.idwt_capacity_bytes()?)
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    fn release_idwt_allocations(&mut self) {
        self.idwt_output = IDWTOutput::default();
        self.idwt_scratch_buffer = Vec::new();
        self.idwt_band_scratch = Vec::new();
        self.idwt_scratch_buffer_i64 = Vec::new();
        self.idwt_line_rings = Vec::new();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        collect_classic_code_block_data, CodeBlock, DecodeAllocationBudget, DecoderContext,
        DecompositionStorage, Layer, Segment,
    };
    use crate::error::DecodingError;
    #[cfg(feature = "parallel")]
    use crate::j2c::build::{SubBand, SubBandType};
    use crate::j2c::codestream::CodeBlockStyle;
    use crate::j2c::rect::IntRect;
    use alloc::vec;

    fn classic_test_style() -> CodeBlockStyle {
        CodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: true,
            vertically_causal_context: false,
            segmentation_symbols: false,
            high_throughput_block_coding: false,
            mixed_block_coding: false,
        }
    }

    fn classic_test_code_block() -> CodeBlock {
        CodeBlock {
            rect: IntRect::from_xywh(0, 0, 1, 1),
            x_idx: 0,
            y_idx: 0,
            layers: 0..1,
            has_been_included: true,
            missing_bit_planes: 0,
            number_of_coding_passes: 3,
            ht_total_coding_passes: 0,
            ht_first_cleanup_pass: None,
            ht_selected_set: None,
            coding: Some(crate::j2c::build::CodeBlockCoding::Classic),
            l_block: 3,
            non_empty_layer_count: 1,
        }
    }

    #[test]
    fn cross_image_release_drops_reusable_decode_capacities() {
        let mut ctx = DecoderContext::default();
        ctx.storage.coefficients.reserve(64);
        ctx.storage.segments.reserve(16);
        ctx.tile_decode_context.idwt_scratch_buffer.reserve(64);
        ctx.tile_decode_context
            .bit_plane_decode_context
            .reserve_coefficients_for_test(64);

        ctx.release_reusable_allocations();

        assert_eq!(ctx.storage.coefficients.capacity(), 0);
        assert_eq!(ctx.storage.segments.capacity(), 0);
        assert_eq!(ctx.tile_decode_context.idwt_scratch_buffer.capacity(), 0);
        assert_eq!(
            ctx.tile_decode_context
                .bit_plane_decode_context
                .coefficient_capacity_for_test(),
            0
        );
    }

    #[test]
    fn next_tile_reset_preserves_storage_capacity_for_accounted_reuse() {
        let mut storage = DecompositionStorage::default();
        storage.coefficients.reserve(64);
        storage.layers.reserve(16);
        let coefficient_capacity = storage.coefficients.capacity();
        let layer_capacity = storage.layers.capacity();

        storage.reset_for_next_tile();

        assert_eq!(storage.coefficients.capacity(), coefficient_capacity);
        assert_eq!(storage.layers.capacity(), layer_capacity);
    }

    #[test]
    fn collect_classic_code_block_data_preserves_zero_length_segments() {
        let mut storage = DecompositionStorage::default();
        storage.layers.push(Layer {
            segments: Some(0..3),
        });
        storage.segments.push(Segment {
            idx: 0,
            coding_pases: 1,
            data_length: 1,
            data: &[0xAA],
        });
        storage.segments.push(Segment {
            idx: 1,
            coding_pases: 1,
            data_length: 0,
            data: &[],
        });
        storage.segments.push(Segment {
            idx: 2,
            coding_pases: 1,
            data_length: 1,
            data: &[0xBB],
        });

        let mut budget = DecodeAllocationBudget::for_storage(&storage).expect("budget baseline");
        let (combined_data, segments) = collect_classic_code_block_data(
            &classic_test_code_block(),
            &classic_test_style(),
            &storage,
            &mut budget,
        )
        .expect("collect classic segments");

        assert_eq!(combined_data, vec![0xAA, 0xBB]);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].data_offset, 0);
        assert_eq!(segments[0].data_length, 1);
        assert_eq!(segments[0].start_coding_pass, 0);
        assert_eq!(segments[0].end_coding_pass, 1);
        assert_eq!(segments[1].data_offset, 1);
        assert_eq!(segments[1].data_length, 0);
        assert_eq!(segments[1].start_coding_pass, 1);
        assert_eq!(segments[1].end_coding_pass, 2);
        assert_eq!(segments[2].data_offset, 1);
        assert_eq!(segments[2].data_length, 1);
        assert_eq!(segments[2].start_coding_pass, 2);
        assert_eq!(segments[2].end_coding_pass, 3);
    }

    #[test]
    fn collect_classic_code_block_data_rejects_non_contiguous_segment_indices() {
        let mut storage = DecompositionStorage::default();
        storage.layers.push(Layer {
            segments: Some(0..2),
        });
        storage.segments.push(Segment {
            idx: 0,
            coding_pases: 1,
            data_length: 1,
            data: &[0xAA],
        });
        storage.segments.push(Segment {
            idx: 2,
            coding_pases: 2,
            data_length: 1,
            data: &[0xBB],
        });

        let mut budget = DecodeAllocationBudget::for_storage(&storage).expect("budget baseline");
        let error = collect_classic_code_block_data(
            &classic_test_code_block(),
            &classic_test_style(),
            &storage,
            &mut budget,
        )
        .expect_err("non-contiguous segment indices must fail");

        assert_eq!(error, DecodingError::CodeBlockDecodeFailure.into());
    }

    #[cfg(feature = "parallel")]
    fn copyback_test_sub_band(width: u32, height: u32) -> (SubBand, DecompositionStorage<'static>) {
        let len = (width * height) as usize;
        let storage = DecompositionStorage {
            coefficients: vec![-1.0; len],
            ..DecompositionStorage::default()
        };
        let sub_band = SubBand {
            sub_band_type: SubBandType::LowLow,
            rect: IntRect::from_xywh(0, 0, width, height),
            precincts: 0..0,
            coefficients: 0..len,
            storage_rect: IntRect::from_xywh(0, 0, width, height),
        };
        (sub_band, storage)
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decoded_classic_block_copyback_covers_full_block() {
        let (sub_band, mut storage) = copyback_test_sub_band(4, 3);
        let mut coefficient_stats = super::ParallelCoefficientStats::default();
        let mut coefficients = (0..12)
            .map(|value| f32::from(u8::try_from(value).expect("test value fits u8")))
            .collect::<Vec<_>>();
        let block = super::DecodedBlock {
            placement: super::subband::BlockPlacement::whole(0, 0, 4, 3),
            width: 4,
            height: 3,
            coefficients: &mut coefficients,
        };

        super::copy_decoded_classic_blocks_to_sub_band(
            &[block],
            &sub_band,
            &mut storage,
            &mut coefficient_stats,
        )
        .expect("full classic block copyback");

        assert_eq!(
            storage.coefficients,
            (0..12)
                .map(|value| f32::from(u8::try_from(value).expect("test value fits u8")))
                .collect::<Vec<_>>()
        );
        assert_eq!(coefficient_stats.scatter_bytes, 12 * size_of::<f32>());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decoded_ht_block_copyback_covers_partial_edge_block() {
        let (sub_band, mut storage) = copyback_test_sub_band(5, 3);
        let mut coefficient_stats = super::ParallelCoefficientStats::default();
        let mut coefficients = vec![1.0, 2.0, 3.0, 4.0];
        let block = super::DecodedBlock {
            placement: super::subband::BlockPlacement::whole(3, 1, 2, 2),
            width: 2,
            height: 2,
            coefficients: &mut coefficients,
        };

        super::copy_decoded_ht_blocks_to_sub_band(
            &[block],
            &sub_band,
            &mut storage,
            &mut coefficient_stats,
        )
        .expect("partial HT block copyback");

        assert_eq!(
            storage.coefficients,
            vec![
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, -1.0, -1.0, -1.0, 3.0,
                4.0,
            ]
        );
        assert_eq!(coefficient_stats.scatter_bytes, 4 * size_of::<f32>());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decoded_block_copyback_rejects_out_of_bounds_blocks() {
        let (sub_band, mut storage) = copyback_test_sub_band(5, 3);
        let mut coefficient_stats = super::ParallelCoefficientStats::default();
        let mut coefficients = vec![1.0, 2.0];
        let block = super::DecodedBlock {
            placement: super::subband::BlockPlacement::whole(4, 1, 2, 1),
            width: 2,
            height: 1,
            coefficients: &mut coefficients,
        };

        let error = super::copy_decoded_classic_blocks_to_sub_band(
            &[block],
            &sub_band,
            &mut storage,
            &mut coefficient_stats,
        )
        .expect_err("out-of-bounds block must fail");

        assert_eq!(error, DecodingError::CodeBlockDecodeFailure.into());
        assert_eq!(coefficient_stats.scatter_bytes, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn auto_cpu_parallelism_enables_sub_band_parallel_branches() {
        let branches: [fn(super::CpuDecodeParallelism, usize) -> bool; 2] = [
            super::should_decode_ht_sub_band_in_parallel,
            super::should_decode_classic_sub_band_in_parallel,
        ];
        for threads in [1, 2] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for (branch, blocks) in branches
                    .iter()
                    .flat_map(|branch| [0, 3, 4, 16, usize::MAX].map(|blocks| (branch, blocks)))
                {
                    assert_eq!(
                        branch(super::CpuDecodeParallelism::Auto, blocks),
                        threads > 1 && blocks >= 4
                    );
                }
            });
        }
    }

    #[test]
    fn serial_cpu_parallelism_disables_ht_sub_band_parallel_branch() {
        assert!(!super::should_decode_ht_sub_band_in_parallel(
            super::CpuDecodeParallelism::Serial,
            16
        ));
    }
}
