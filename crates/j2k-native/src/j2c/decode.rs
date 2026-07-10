//! Decoding JPEG2000 code streams.
//!
//! This is the "core" module of the crate that orchestrates all
//! stages in such a way that a given codestream is decoded into its
//! component channels.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use super::bitplane::{BitPlaneDecodeBuffers, BitPlaneDecodeContext};
use super::build::{CodeBlock, Decomposition, Layer, Precinct, Segment, SubBand, SubBandType};
use super::codestream::{ComponentInfo, Header, QuantizationStyle, WaveletTransform};
use super::ht_block_decode::{self, HtBlockDecodeContext};
use super::idwt::IDWTOutput;
use super::progression::{progression_iterator, ProgressionData};
use super::roi::RoiPlan;
use super::tag_tree::TagNode;
use super::tile::{ComponentTile, ResolutionTile, Tile};
use super::{bitplane, build, idwt, mct, segment, tile, ComponentData};
use crate::error::{
    bail, ColorError, DecodingError, DirectPlanUnsupportedReason, Result, TileError,
};
use crate::j2c::segment::MAX_BITPLANE_COUNT;
use crate::math::SimdBuffer;
use crate::profile;
use crate::reader::BitReader;
use crate::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_i32, apply_roi_maxshift_inverse_i64,
    checked_decode_byte_len3, checked_decode_sample_count, decode_j2k_code_block_scalar,
    HtCodeBlockBatchJob, HtCodeBlockDecodeJob, HtCodeBlockDecoder, HtOwnedCodeBlockBatchJob,
    HtOwnedSubBandPlan, HtSubBandDecodeJob, J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob,
    J2kCodeBlockSegment, J2kCodeBlockStyle, J2kDirectBandId, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan, J2kRect, J2kStoreComponentJob,
    J2kSubBandDecodeJob, J2kSubBandType, J2kWaveletTransform,
};
#[cfg(feature = "parallel")]
use crate::{decode_ht_code_block_scalar_with_workspace, HtCodeBlockDecodeWorkspace};
use core::mem::size_of;
use core::ops::{DerefMut, Range};

mod direct_plan;
mod store;
mod subband;
mod subband_params;
use self::direct_plan::collect_classic_code_block_data;
pub(crate) use self::direct_plan::{build_direct_color_plan, build_direct_grayscale_plan};
use self::store::{apply_sign_shift, component_unsigned_level_shift, store};
use self::subband::code_block_required_by_index;
pub(crate) use self::subband::decode_component_tile_bit_planes;
#[cfg(all(test, feature = "parallel"))]
use self::subband::{
    copy_decoded_classic_blocks_to_sub_band, copy_decoded_ht_blocks_to_sub_band,
    DecodedClassicBlock, DecodedHtBlock,
};
#[cfg(test)]
pub(crate) use self::subband::{
    should_decode_classic_sub_band_in_parallel, should_decode_ht_sub_band_in_parallel,
};
use self::subband_params::{
    classic_decode_job_parameters, ht_code_block_has_decodable_passes, sub_band_decode_parameters,
    SubBandDecodeParameters,
};

pub(crate) fn decode<'a>(
    data: &'a [u8],
    header: &Header<'a>,
    ctx: &mut DecoderContext<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
) -> Result<()> {
    let mut reader = BitReader::new(data);
    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);
    let mut profile_timings = DecodeProfileTimings::default();
    let stage_start = profile::profile_now(profile_enabled);
    let tiles = tile::parse(&mut reader, header)?;
    profile_timings.parse_tiles_us += profile::elapsed_us(stage_start);

    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }

    ctx.reset(header, &tiles[0])?;
    let cpu_decode_parallelism = ctx.cpu_decode_parallelism;
    let (tile_ctx, storage) = (&mut ctx.tile_decode_context, &mut ctx.storage);

    for tile in &tiles {
        ltrace!(
            "tile {} rect [{},{} {}x{}]",
            tile.idx,
            tile.rect.x0,
            tile.rect.y0,
            tile.rect.width(),
            tile.rect.height(),
        );

        decode_tile(
            tile,
            header,
            progression_iterator(tile)?,
            tile_ctx,
            storage,
            ht_decoder,
            cpu_decode_parallelism,
            profile_enabled,
            &mut profile_timings,
        )?;
    }

    // Note that this assumes that either all tiles have MCT or none of them.
    // In theory, only some could have it... But hopefully no such cursed
    // images exist!
    if tiles[0].mct {
        let stage_start = profile::profile_now(profile_enabled);
        mct::apply_inverse(tile_ctx, &tiles[0].component_infos, header, ht_decoder)?;
        apply_sign_shift(tile_ctx, &header.component_infos);
        profile_timings.mct_us += profile::elapsed_us(stage_start);
    }

    if profile_enabled {
        emit_decode_profile_row(tile_ctx, &profile_timings, total_start);
    }

    Ok(())
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
    pub(crate) decoded_code_blocks: usize,
    pub(crate) skipped_code_blocks: usize,
    pub(crate) idwt_output_samples: usize,
    pub(crate) ht_phase_stats: ht_block_decode::HtBlockDecodeStats,
}

/// CPU parallelism policy for native JPEG 2000 decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CpuDecodeParallelism {
    /// Allow a single tile decode to use internal code-block parallelism.
    #[default]
    Auto,
    /// Keep code-block decode serial for callers that already parallelize tiles.
    Serial,
}

/// A decoder context for decoding JPEG2000 images.
pub struct DecoderContext<'a> {
    pub(crate) tile_decode_context: TileDecodeContext,
    pub(crate) storage: DecompositionStorage<'a>,
    cpu_decode_parallelism: CpuDecodeParallelism,
}

impl Default for DecoderContext<'_> {
    fn default() -> Self {
        Self {
            tile_decode_context: TileDecodeContext::default(),
            storage: DecompositionStorage::default(),
            cpu_decode_parallelism: CpuDecodeParallelism::Auto,
        }
    }
}

impl DecoderContext<'_> {
    fn reset(&mut self, header: &Header<'_>, initial_tile: &Tile<'_>) -> Result<()> {
        self.tile_decode_context.reset(header, initial_tile)?;
        self.storage.reset();
        Ok(())
    }

    pub(crate) fn set_output_region(&mut self, output_region: Option<(u32, u32, u32, u32)>) {
        self.tile_decode_context.output_region = output_region.map(OutputRegion::from_tuple);
    }

    /// Return the native CPU decode parallelism policy.
    pub fn cpu_decode_parallelism(&self) -> CpuDecodeParallelism {
        self.cpu_decode_parallelism
    }

    /// Set the native CPU decode parallelism policy.
    pub fn set_cpu_decode_parallelism(&mut self, parallelism: CpuDecodeParallelism) {
        self.cpu_decode_parallelism = parallelism;
    }
}

fn decode_tile<'a, 'b>(
    tile: &'b Tile<'a>,
    header: &Header<'_>,
    progression_iterator: Box<dyn Iterator<Item = ProgressionData> + '_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    ht_decoder: &mut Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
) -> Result<()> {
    storage.reset();
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
    build::build(tile, storage)?;
    if let Some(output_region) = tile_ctx.output_region {
        storage.roi_plan = RoiPlan::build(tile, header, storage, output_region);
        if storage.roi_plan.is_some() {
            storage.coefficients.fill(0.0);
            if storage.exact_integer_decode {
                storage.coefficients_i64.fill(0);
            }
        }
    }
    profile_timings.build_us += profile::elapsed_us(stage_start);
    // Next, we parse the layers/segments for each code block.
    let stage_start = profile::profile_now(profile_enabled);
    segment::parse(tile, progression_iterator, header, storage)?;
    profile_timings.segment_us += profile::elapsed_us(stage_start);
    // We then decode the bitplanes of each code block, yielding the
    // (possibly dequantized) coefficients of each code block.
    let stage_start = profile::profile_now(profile_enabled);
    decode_component_tile_bit_planes(
        tile,
        tile_ctx,
        storage,
        header,
        ht_decoder,
        cpu_decode_parallelism,
        profile_enabled,
    )?;
    profile_timings.codeblock_us += profile::elapsed_us(stage_start);

    // Unlike before, we interleave the apply_idwt and store stages
    // for each component tile so we can reuse allocations better.
    for (idx, component_info) in header.component_infos.iter().enumerate() {
        // Next, we apply the inverse discrete wavelet transform.
        let stage_start = profile::profile_now(profile_enabled);
        idwt::apply(
            storage,
            tile_ctx,
            idx,
            header,
            component_info.wavelet_transform(),
            ht_decoder,
        )?;
        profile_timings.idwt_us += profile::elapsed_us(stage_start);
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
        store(tile, header, tile_ctx, component_info, idx, ht_decoder)?;
        profile_timings.store_us += profile::elapsed_us(stage_start);
    }

    Ok(())
}

fn tile_requires_exact_integer_decode(tile: &Tile<'_>) -> bool {
    tile.component_infos
        .iter()
        .any(ComponentInfo::requires_exact_integer_decode)
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

/// A buffer so that we can reuse allocations for layers/code blocks/etc.
/// across different tiles.
#[derive(Default)]
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
}

impl DecompositionStorage<'_> {
    pub(crate) fn reset(&mut self) {
        self.segments.clear();
        self.layers.clear();
        self.code_blocks.clear();
        // No need to clear the coefficients, as they will be resized
        // and then overridden.
        // self.coefficients.clear();
        self.precincts.clear();
        self.sub_bands.clear();
        self.decompositions.clear();
        self.tile_decompositions.clear();
        self.tag_tree_nodes.clear();
        self.roi_plan = None;
        self.exact_integer_decode = false;
    }
}

/// A reusable context used during the decoding of a single tile.
///
/// Some of the fields are temporary in nature and reset after moving on to the
/// next tile, some contain global state.
#[derive(Default)]
pub(crate) struct TileDecodeContext {
    /// A reusable buffer for the IDWT output.
    pub(crate) idwt_output: IDWTOutput,
    /// A scratch buffer used during IDWT.
    pub(crate) idwt_scratch_buffer: Vec<f32>,
    /// A scratch buffer used during exact reversible integer IDWT.
    pub(crate) idwt_scratch_buffer_i64: Vec<i64>,
    /// A reusable context for decoding code blocks.
    pub(crate) bit_plane_decode_context: BitPlaneDecodeContext,
    /// Reusable buffers for decoding bitplanes.
    pub(crate) bit_plane_decode_buffers: BitPlaneDecodeBuffers,
    /// A reusable context for decoding HTJ2K code blocks.
    pub(crate) ht_block_decode_context: HtBlockDecodeContext,
    /// The raw, decoded samples for each channel.
    pub(crate) channel_data: Vec<ComponentData>,
    /// Optional output window for region-local decode storage.
    pub(crate) output_region: Option<OutputRegion>,
    /// Debug counters for tests and ROI instrumentation.
    pub(crate) debug_counters: DecodeDebugCounters,
}

impl TileDecodeContext {
    /// Reset the context for processing a new image.
    fn reset(&mut self, header: &Header<'_>, initial_tile: &Tile<'_>) -> Result<()> {
        // Bitplane decode context and buffers will be reset in the
        // corresponding methods. IDWT output and scratch buffer will be
        // overridden on demand, so those don't need to be reset either.
        self.channel_data.clear();
        self.debug_counters = DecodeDebugCounters::default();

        let (output_width, output_height) =
            self.output_region.map(OutputRegion::dimensions).unwrap_or((
                header.size_data.image_width(),
                header.size_data.image_height(),
            ));

        let sample_count = checked_decode_sample_count(output_width, output_height)?;
        checked_decode_byte_len3(
            sample_count,
            initial_tile.component_infos.len(),
            size_of::<f32>(),
        )?;
        let exact_integer_decode = initial_tile
            .component_infos
            .iter()
            .any(ComponentInfo::requires_exact_integer_decode);
        if exact_integer_decode {
            checked_decode_byte_len3(
                sample_count,
                initial_tile.component_infos.len(),
                size_of::<i64>(),
            )?;
        }

        // Allocate per component here; the surrounding context reuses the
        // higher-level vectors while `SimdBuffer` owns its initialized storage.
        for info in &initial_tile.component_infos {
            self.channel_data.push(ComponentData {
                container: SimdBuffer::zeros(sample_count),
                integer_container: exact_integer_decode.then(|| vec![0; sample_count]),
                bit_depth: info.size_info.precision,
                signed: info.size_info.signed,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{collect_classic_code_block_data, CodeBlock, DecompositionStorage, Layer, Segment};
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
            l_block: 3,
            non_empty_layer_count: 1,
        }
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

        let (combined_data, segments) = collect_classic_code_block_data(
            &classic_test_code_block(),
            &classic_test_style(),
            &storage,
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

        let error = collect_classic_code_block_data(
            &classic_test_code_block(),
            &classic_test_style(),
            &storage,
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
        };
        (sub_band, storage)
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decoded_classic_block_copyback_covers_full_block() {
        let (sub_band, mut storage) = copyback_test_sub_band(4, 3);
        let block = super::DecodedClassicBlock {
            output_x: 0,
            output_y: 0,
            width: 4,
            height: 3,
            coefficients: (0..12).map(|value| value as f32).collect(),
        };

        super::copy_decoded_classic_blocks_to_sub_band(&[block], &sub_band, &mut storage)
            .expect("full classic block copyback");

        assert_eq!(
            storage.coefficients,
            (0..12).map(|value| value as f32).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decoded_ht_block_copyback_covers_partial_edge_block() {
        let (sub_band, mut storage) = copyback_test_sub_band(5, 3);
        let block = super::DecodedHtBlock {
            output_x: 3,
            output_y: 1,
            width: 2,
            height: 2,
            coefficients: vec![1.0, 2.0, 3.0, 4.0],
        };

        super::copy_decoded_ht_blocks_to_sub_band(&[block], &sub_band, &mut storage)
            .expect("partial HT block copyback");

        assert_eq!(
            storage.coefficients,
            vec![
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, -1.0, -1.0, -1.0, 3.0,
                4.0,
            ]
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn decoded_block_copyback_rejects_out_of_bounds_blocks() {
        let (sub_band, mut storage) = copyback_test_sub_band(5, 3);
        let block = super::DecodedClassicBlock {
            output_x: 4,
            output_y: 1,
            width: 2,
            height: 1,
            coefficients: vec![1.0, 2.0],
        };

        let error =
            super::copy_decoded_classic_blocks_to_sub_band(&[block], &sub_band, &mut storage)
                .expect_err("out-of-bounds block must fail");

        assert_eq!(error, DecodingError::CodeBlockDecodeFailure.into());
    }

    #[test]
    fn auto_cpu_parallelism_enables_ht_sub_band_parallel_branch() {
        assert!(super::should_decode_ht_sub_band_in_parallel(
            super::CpuDecodeParallelism::Auto,
            16
        ));
    }

    #[test]
    fn serial_cpu_parallelism_disables_ht_sub_band_parallel_branch() {
        assert!(!super::should_decode_ht_sub_band_in_parallel(
            super::CpuDecodeParallelism::Serial,
            16
        ));
    }
}
