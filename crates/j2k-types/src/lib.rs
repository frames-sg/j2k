//! Shared JPEG 2000 encode-stage contract types for j2k.
//!
//! This crate is the neutral public contract between the `j2k`
//! adapter surface and the `j2k-native` codec engine: job, output,
//! and report types cross the boundary here so neither crate mirrors the
//! other's definitions. It intentionally contains plain data types only -
//! codec behavior stays in `j2k-native` and adapter traits stay in
//! their owning crates.

#![no_std]
#![forbid(unsafe_code)]
#![forbid(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;
use core::ops::Range;

/// Adapter classic J2K sub-band kind for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kSubBandType {
    /// Low-low sub-band.
    LowLow,
    /// High-low sub-band.
    HighLow,
    /// Low-high sub-band.
    LowHigh,
    /// High-high sub-band.
    HighHigh,
}

/// Adapter classic J2K code-block style for backend experimentation.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // models the five independent COD code-block style flags
pub struct J2kCodeBlockStyle {
    /// Selective arithmetic coding bypass was enabled.
    pub selective_arithmetic_coding_bypass: bool,
    /// Context probabilities reset after each pass.
    pub reset_context_probabilities: bool,
    /// Coding terminated after each pass.
    pub termination_on_each_pass: bool,
    /// Vertically causal context was enabled.
    pub vertically_causal_context: bool,
    /// Segmentation symbols were enabled.
    pub segmentation_symbols: bool,
}

/// Adapter classic J2K coded segment for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kCodeBlockSegment {
    /// Byte offset of this segment within the combined payload.
    pub data_offset: u32,
    /// Segment payload length in bytes.
    pub data_length: u32,
    /// First coding pass covered by this segment.
    pub start_coding_pass: u8,
    /// One-past-last coding pass covered by this segment.
    pub end_coding_pass: u8,
    /// Whether this segment is decoded through the arithmetic path.
    pub use_arithmetic: bool,
}

/// Adapter encoded classic J2K code-block payload for backend experimentation.
#[derive(Debug, Clone)]
pub struct EncodedJ2kCodeBlock {
    /// Combined payload bytes for all coded segments in this code block.
    pub data: Vec<u8>,
    /// Coded segments for the code block.
    pub segments: Vec<J2kCodeBlockSegment>,
    /// Number of coding passes present for this code block.
    pub number_of_coding_passes: u8,
    /// Missing most-significant bit planes for this code block.
    pub missing_bit_planes: u8,
}

/// Adapter encoded HTJ2K cleanup code-block payload for backend experimentation.
#[derive(Debug, Clone)]
pub struct EncodedHtJ2kCodeBlock {
    /// Combined cleanup/refinement bytes for this code block.
    pub data: Vec<u8>,
    /// Cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// Refinement segment length in bytes.
    pub refinement_length: u32,
    /// Number of coding passes present for this code block.
    pub num_coding_passes: u8,
    /// Number of zero most-significant bitplanes before first inclusion.
    pub num_zero_bitplanes: u8,
}

/// Adapter pixel deinterleave/level-shift job for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kDeinterleaveToF32Job<'a> {
    /// Interleaved source pixel bytes.
    pub pixels: &'a [u8],
    /// Number of pixels to convert.
    pub num_pixels: usize,
    /// Number of interleaved components per pixel.
    pub num_components: u8,
    /// Source sample bit depth.
    pub bit_depth: u8,
    /// Whether source samples are signed.
    pub signed: bool,
}

/// Adapter forward RCT job for backend experimentation.
#[derive(Debug)]
pub struct J2kForwardRctJob<'a> {
    /// First component plane, updated in place.
    pub plane0: &'a mut [f32],
    /// Second component plane, updated in place.
    pub plane1: &'a mut [f32],
    /// Third component plane, updated in place.
    pub plane2: &'a mut [f32],
}

/// Adapter forward ICT job for backend experimentation.
#[derive(Debug)]
pub struct J2kForwardIctJob<'a> {
    /// First component plane, updated in place.
    pub plane0: &'a mut [f32],
    /// Second component plane, updated in place.
    pub plane1: &'a mut [f32],
    /// Third component plane, updated in place.
    pub plane2: &'a mut [f32],
}

/// Adapter forward 5/3 DWT job for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kForwardDwt53Job<'a> {
    /// Source samples in row-major order.
    pub samples: &'a [f32],
    /// Source width in samples.
    pub width: u32,
    /// Source height in samples.
    pub height: u32,
    /// Number of decomposition levels requested.
    pub num_levels: u8,
}

/// Adapter forward 5/3 DWT output for backend experimentation.
#[derive(Debug, Clone)]
pub struct J2kForwardDwt53Output {
    /// LL subband coefficients from the lowest decomposition level.
    pub ll: Vec<f32>,
    /// LL subband width.
    pub ll_width: u32,
    /// LL subband height.
    pub ll_height: u32,
    /// Higher resolution detail levels, ordered from lowest to highest.
    pub levels: Vec<J2kForwardDwt53Level>,
}

/// Adapter forward 5/3 DWT detail level for backend experimentation.
#[derive(Debug, Clone)]
pub struct J2kForwardDwt53Level {
    /// HL subband coefficients.
    pub hl: Vec<f32>,
    /// LH subband coefficients.
    pub lh: Vec<f32>,
    /// HH subband coefficients.
    pub hh: Vec<f32>,
    /// Full-resolution width represented by this level.
    pub width: u32,
    /// Full-resolution height represented by this level.
    pub height: u32,
    /// Low-pass width at this level.
    pub low_width: u32,
    /// Low-pass height at this level.
    pub low_height: u32,
    /// High-pass width at this level.
    pub high_width: u32,
    /// High-pass height at this level.
    pub high_height: u32,
}

/// Adapter forward irreversible 9/7 DWT job for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kForwardDwt97Job<'a> {
    /// Source samples in row-major order.
    pub samples: &'a [f32],
    /// Source width in samples.
    pub width: u32,
    /// Source height in samples.
    pub height: u32,
    /// Number of decomposition levels requested.
    pub num_levels: u8,
}

/// Adapter forward 9/7 DWT output for backend experimentation.
#[derive(Debug, Clone)]
pub struct J2kForwardDwt97Output {
    /// LL subband coefficients from the lowest decomposition level.
    pub ll: Vec<f32>,
    /// LL subband width.
    pub ll_width: u32,
    /// LL subband height.
    pub ll_height: u32,
    /// Higher resolution detail levels, ordered from lowest to highest.
    pub levels: Vec<J2kForwardDwt97Level>,
}

/// Adapter forward 9/7 DWT detail level for backend experimentation.
#[derive(Debug, Clone)]
pub struct J2kForwardDwt97Level {
    /// HL subband coefficients.
    pub hl: Vec<f32>,
    /// LH subband coefficients.
    pub lh: Vec<f32>,
    /// HH subband coefficients.
    pub hh: Vec<f32>,
    /// Full-resolution width represented by this level.
    pub width: u32,
    /// Full-resolution height represented by this level.
    pub height: u32,
    /// Low-pass width at this level.
    pub low_width: u32,
    /// Low-pass height at this level.
    pub low_height: u32,
    /// High-pass width at this level.
    pub high_width: u32,
    /// High-pass height at this level.
    pub high_height: u32,
}

/// Adapter sub-band quantization job for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kQuantizeSubbandJob<'a> {
    /// Source sub-band coefficients in row-major order.
    pub coefficients: &'a [f32],
    /// Quantization step-size exponent.
    pub step_exponent: u16,
    /// Quantization step-size mantissa.
    pub step_mantissa: u16,
    /// Nominal range bits for this sub-band.
    pub range_bits: u8,
    /// Whether to use reversible integer quantization.
    pub reversible: bool,
}

/// Adapter Tier-1 classic J2K code-block encode job for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kTier1CodeBlockEncodeJob<'a> {
    /// Quantized coefficients in row-major order.
    pub coefficients: &'a [i32],
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Subband kind containing this code-block.
    pub sub_band_type: J2kSubBandType,
    /// Total bitplanes for this subband/code-block.
    pub total_bitplanes: u8,
    /// Classic J2K code-block style flags.
    pub style: J2kCodeBlockStyle,
}

/// Adapter HTJ2K code-block encode job for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kHtCodeBlockEncodeJob<'a> {
    /// Quantized coefficients in row-major order.
    pub coefficients: &'a [i32],
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Total bitplanes for this subband/code-block.
    pub total_bitplanes: u8,
    /// Requested HT coding passes for this contribution.
    ///
    /// `1` is cleanup-only. Higher values require an accelerator that can
    /// encode those passes and must not be silently reduced by CPU fallback.
    pub target_coding_passes: u8,
}

/// Adapter HTJ2K cleanup encode job for one unquantized sub-band.
#[derive(Debug, Clone, Copy)]
pub struct J2kHtSubbandEncodeJob<'a> {
    /// Source sub-band coefficients in row-major order.
    pub coefficients: &'a [f32],
    /// Sub-band width in samples.
    pub width: u32,
    /// Sub-band height in samples.
    pub height: u32,
    /// Quantization step-size exponent.
    pub step_exponent: u16,
    /// Quantization step-size mantissa.
    pub step_mantissa: u16,
    /// Nominal range bits for this sub-band.
    pub range_bits: u8,
    /// Whether to use reversible integer quantization.
    pub reversible: bool,
    /// Code-block width in samples.
    pub code_block_width: u32,
    /// Code-block height in samples.
    pub code_block_height: u32,
    /// Total coded bitplanes for this sub-band.
    pub total_bitplanes: u8,
}

/// Adapter HTJ2K tile-body encode job for backend-resident full-tile paths.
#[derive(Debug, Clone, Copy)]
pub struct J2kHtj2kTileEncodeJob<'a> {
    /// Interleaved source pixel bytes.
    pub pixels: &'a [u8],
    /// Tile/image width in samples.
    pub width: u32,
    /// Tile/image height in samples.
    pub height: u32,
    /// Number of interleaved image components.
    pub num_components: u8,
    /// Source component bit depth.
    pub bit_depth: u8,
    /// Whether source samples are signed.
    pub signed: bool,
    /// Number of DWT decomposition levels.
    pub num_decomposition_levels: u8,
    /// Whether the codestream uses reversible coding.
    pub reversible: bool,
    /// Whether a multi-component transform should be applied.
    pub use_mct: bool,
    /// JPEG 2000 guard bits used to derive total coded bitplanes.
    pub guard_bits: u8,
    /// Code-block width in samples.
    pub code_block_width: u32,
    /// Code-block height in samples.
    pub code_block_height: u32,
    /// Packet progression order to emit.
    pub progression_order: J2kPacketizationProgressionOrder,
    /// Per-component sampling factors, as `(x_rsiz, y_rsiz)`.
    pub component_sampling: &'a [(u8, u8)],
    /// Quantization step sizes, as `(exponent, mantissa)`, in codestream order.
    pub quantization_steps: &'a [(u16, u16)],
}

/// Adapter LRCP packetization code-block contribution for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kPacketizationCodeBlock<'a> {
    /// Encoded Tier-1 bitstream bytes for this packet contribution.
    pub data: &'a [u8],
    /// HTJ2K cleanup segment length in bytes when using high-throughput coding.
    pub ht_cleanup_length: u32,
    /// HTJ2K refinement segment length in bytes when using high-throughput coding.
    pub ht_refinement_length: u32,
    /// Number of coding passes in this contribution.
    pub num_coding_passes: u8,
    /// Number of zero most-significant bitplanes before first inclusion.
    pub num_zero_bitplanes: u8,
    /// Whether this code-block was included in a previous packet.
    pub previously_included: bool,
    /// L-block value used for segment length coding.
    pub l_block: u32,
    /// Block coder used for this contribution.
    pub block_coding_mode: J2kPacketizationBlockCodingMode,
}

/// Adapter packetization block coding mode for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kPacketizationBlockCodingMode {
    /// Classic JPEG 2000 Part 1 EBCOT block coding.
    Classic,
    /// High-throughput JPEG 2000 Part 15 block coding.
    HighThroughput,
}

/// Adapter packet progression order for backend packetization experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kPacketizationProgressionOrder {
    /// Layer-resolution-component-position progression.
    Lrcp,
    /// Resolution-layer-component-position progression.
    Rlcp,
    /// Resolution-position-component-layer progression.
    Rpcl,
    /// Position-component-resolution-layer progression.
    Pcrl,
    /// Component-position-resolution-layer progression.
    Cprl,
}

/// Adapter LRCP packetization subband precinct for backend experimentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kPacketizationSubband<'a> {
    /// Code-block contributions in row-major order.
    pub code_blocks: Vec<J2kPacketizationCodeBlock<'a>>,
    /// Number of code-blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub num_cbs_y: u32,
}

/// Adapter LRCP packetization resolution packet for backend experimentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct J2kPacketizationResolution<'a> {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<J2kPacketizationSubband<'a>>,
}

/// Adapter explicit packet descriptor for backend packetization experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kPacketizationPacketDescriptor {
    /// Index into the packet contribution array.
    pub packet_index: u32,
    /// Persistent packet-state index for repeated layer/precinct packets.
    pub state_index: u32,
    /// Quality layer for inclusion tag-tree thresholds.
    pub layer: u8,
    /// Resolution index in the output progression.
    pub resolution: u32,
    /// Component index in the output progression.
    pub component: u8,
    /// Precinct index in the output progression.
    pub precinct: u64,
}

/// Adapter LRCP packetization job for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kPacketizationEncodeJob<'a> {
    /// Number of resolution packets prepared for packetization.
    pub resolution_count: u32,
    /// Number of layers to write.
    pub num_layers: u8,
    /// Number of image components.
    pub num_components: u8,
    /// Total number of code-block contributions.
    pub code_block_count: u32,
    /// Packet progression order to emit.
    pub progression_order: J2kPacketizationProgressionOrder,
    /// Explicit packet descriptors in output progression order.
    pub packet_descriptors: &'a [J2kPacketizationPacketDescriptor],
    /// Packet payload prepared by Tier-1, in LRCP packet order.
    pub resolutions: &'a [J2kPacketizationResolution<'a>],
}

/// Adapter encode-stage dispatch counters for backend experimentation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct J2kEncodeDispatchReport {
    /// Pixel deinterleave/level-shift dispatch count.
    pub deinterleave: usize,
    /// Forward RCT kernel dispatch count.
    pub forward_rct: usize,
    /// Forward ICT kernel dispatch count.
    pub forward_ict: usize,
    /// Forward reversible 5/3 DWT kernel dispatch count.
    pub forward_dwt53: usize,
    /// Forward irreversible 9/7 DWT kernel dispatch count.
    pub forward_dwt97: usize,
    /// Sub-band quantization dispatch count.
    pub quantize_subband: usize,
    /// Tier-1 code-block encode dispatch count.
    pub tier1_code_block: usize,
    /// HTJ2K code-block encode dispatch count.
    pub ht_code_block: usize,
    /// Packetization dispatch count.
    pub packetization: usize,
}

impl J2kEncodeDispatchReport {
    /// Return the saturating per-stage delta from `before` to `self`.
    #[must_use]
    pub fn saturating_delta(self, before: Self) -> Self {
        Self {
            deinterleave: self.deinterleave.saturating_sub(before.deinterleave),
            forward_rct: self.forward_rct.saturating_sub(before.forward_rct),
            forward_ict: self.forward_ict.saturating_sub(before.forward_ict),
            forward_dwt53: self.forward_dwt53.saturating_sub(before.forward_dwt53),
            forward_dwt97: self.forward_dwt97.saturating_sub(before.forward_dwt97),
            quantize_subband: self
                .quantize_subband
                .saturating_sub(before.quantize_subband),
            tier1_code_block: self
                .tier1_code_block
                .saturating_sub(before.tier1_code_block),
            ht_code_block: self.ht_code_block.saturating_sub(before.ht_code_block),
            packetization: self.packetization.saturating_sub(before.packetization),
        }
    }

    /// Return total dispatches across all encode stages.
    #[must_use]
    pub fn total(self) -> usize {
        self.forward_rct
            .saturating_add(self.deinterleave)
            .saturating_add(self.forward_ict)
            .saturating_add(self.forward_dwt53)
            .saturating_add(self.forward_dwt97)
            .saturating_add(self.quantize_subband)
            .saturating_add(self.tier1_code_block)
            .saturating_add(self.ht_code_block)
            .saturating_add(self.packetization)
    }

    /// Return whether at least one encode stage dispatched.
    #[must_use]
    pub fn any(self) -> bool {
        self.total() > 0
    }
}

/// Adapter CPU-only encode accelerator that always falls back to native stages.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuOnlyJ2kEncodeStageAccelerator;

/// Multipliers applied to irreversible 9/7 quantization step sizes by subband.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrreversibleQuantizationSubbandScales {
    /// Multiplier for the LL subband.
    pub low_low: f32,
    /// Multiplier for HL subbands.
    pub high_low: f32,
    /// Multiplier for LH subbands.
    pub low_high: f32,
    /// Multiplier for HH subbands.
    pub high_high: f32,
}

/// Public JPEG 2000 irreversible quantization step-size tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrreversibleQuantizationStep {
    /// Quantization step-size exponent.
    pub exponent: u8,
    /// Quantization step-size mantissa.
    pub mantissa: u16,
}

impl Default for IrreversibleQuantizationSubbandScales {
    fn default() -> Self {
        Self {
            low_low: 1.0,
            high_low: 1.0,
            low_high: 1.0,
            high_high: 1.0,
        }
    }
}

/// Precomputed reversible 5/3 wavelet coefficients for one component.
#[derive(Debug, Clone)]
pub struct PrecomputedHtj2k53Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Forward 5/3 DWT output, ordered as the encoder expects.
    pub dwt: J2kForwardDwt53Output,
}

/// Precomputed reversible 5/3 wavelet image.
#[derive(Debug, Clone)]
pub struct PrecomputedHtj2k53Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PrecomputedHtj2k53Component>,
}

/// Precomputed irreversible 9/7 wavelet coefficients for one component.
#[derive(Debug, Clone)]
pub struct PrecomputedHtj2k97Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Forward 9/7 DWT output, ordered as the encoder expects.
    pub dwt: J2kForwardDwt97Output,
}

/// Precomputed irreversible 9/7 wavelet image.
#[derive(Debug, Clone)]
pub struct PrecomputedHtj2k97Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PrecomputedHtj2k97Component>,
}

/// Prequantized irreversible 9/7 HTJ2K code-block image.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PrequantizedHtj2k97Component>,
}

/// Prequantized irreversible 9/7 HTJ2K component.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Resolution packets for this component, ordered from lowest to highest.
    pub resolutions: Vec<PrequantizedHtj2k97Resolution>,
}

/// One component resolution's prequantized HTJ2K subbands.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97Resolution {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<PrequantizedHtj2k97Subband>,
}

/// One prequantized HTJ2K subband split into code-blocks.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97Subband {
    /// Subband kind.
    pub sub_band_type: J2kSubBandType,
    /// Number of code-blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub num_cbs_y: u32,
    /// Total bitplanes declared for every code-block in this subband.
    pub total_bitplanes: u8,
    /// Code-block coefficients in row-major code-block order.
    pub code_blocks: Vec<PrequantizedHtj2k97CodeBlock>,
}

/// One prequantized HTJ2K code-block.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97CodeBlock {
    /// Quantized coefficients in row-major order.
    pub coefficients: Vec<i32>,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
}

/// Preencoded irreversible 9/7 HTJ2K code-block image.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PreencodedHtj2k97Component>,
}

/// Preencoded irreversible 9/7 HTJ2K component.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Resolution packets for this component, ordered from lowest to highest.
    pub resolutions: Vec<PreencodedHtj2k97Resolution>,
}

/// One component resolution's preencoded HTJ2K subbands.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97Resolution {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<PreencodedHtj2k97Subband>,
}

/// One preencoded HTJ2K subband split into code-blocks.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97Subband {
    /// Subband kind.
    pub sub_band_type: J2kSubBandType,
    /// Number of code-blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub num_cbs_y: u32,
    /// Total bitplanes declared for every code-block in this subband.
    pub total_bitplanes: u8,
    /// Encoded code-block payloads in row-major code-block order.
    pub code_blocks: Vec<PreencodedHtj2k97CodeBlock>,
}

/// One preencoded HTJ2K code-block.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97CodeBlock {
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Encoded cleanup/refinement payload and packet metadata.
    pub encoded: EncodedHtJ2kCodeBlock,
}

/// Preencoded irreversible 9/7 HTJ2K code-block image backed by one compact
/// payload buffer.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97CompactImage {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Contiguous encoded code-block payload bytes.
    pub payload: Vec<u8>,
    /// Components at their native resolution.
    pub components: Vec<PreencodedHtj2k97CompactComponent>,
}

/// Preencoded compact irreversible 9/7 HTJ2K component.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97CompactComponent {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Resolution packets for this component, ordered from lowest to highest.
    pub resolutions: Vec<PreencodedHtj2k97CompactResolution>,
}

/// One component resolution's compact preencoded HTJ2K subbands.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97CompactResolution {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<PreencodedHtj2k97CompactSubband>,
}

/// One compact preencoded HTJ2K subband split into code-blocks.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97CompactSubband {
    /// Subband kind.
    pub sub_band_type: J2kSubBandType,
    /// Number of code-blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub num_cbs_y: u32,
    /// Total bitplanes declared for every code-block in this subband.
    pub total_bitplanes: u8,
    /// Code-block metadata in row-major code-block order.
    pub code_blocks: Vec<PreencodedHtj2k97CompactCodeBlock>,
}

/// One compact preencoded HTJ2K code-block.
#[derive(Debug, Clone)]
pub struct PreencodedHtj2k97CompactCodeBlock {
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Byte range into the image-level compact payload.
    pub payload_range: Range<usize>,
    /// HTJ2K cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// HTJ2K refinement segment length in bytes.
    pub refinement_length: u32,
    /// Number of coding passes in the encoded payload.
    pub num_coding_passes: u8,
    /// Number of missing most-significant bitplanes.
    pub num_zero_bitplanes: u8,
}
