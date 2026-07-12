// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use j2k_core::BackendKind;

use crate::J2kEncodeDispatchReport;

pub(super) use j2k_types::{
    MAX_JPEG2000_PART1_COMPONENTS,
    MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH as MAX_PART1_SAMPLE_BIT_DEPTH,
};
pub(super) const MAX_RAW_PIXEL_ENCODE_BIT_DEPTH: u8 = 24;
pub(super) const MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES: u16 = 37;
pub(super) const MAX_HTJ2K_ENCODE_BITPLANES: u16 = 31;

macro_rules! define_encoded_j2k {
    (
        $(#[$attr:meta])*
        pub struct $name:ident {
            $($extra_fields:tt)*
        }
    ) => {
        $(#[$attr])*
        pub struct $name {
            /// Raw JPEG 2000 codestream bytes.
            pub codestream: Vec<u8>,
            /// Backend that satisfied the encode contract.
            pub backend: BackendKind,
            /// Encode-stage dispatches observed while producing this codestream.
            ///
            /// This can be nonzero even when [`Self::backend`] is [`BackendKind::Cpu`]
            /// for Auto routes that used one or more device stages but did not satisfy
            /// every stage required for a fully device-backed encode contract.
            pub dispatch_report: J2kEncodeDispatchReport,
            /// Encoded image width in pixels.
            pub width: u32,
            /// Encoded image height in pixels.
            pub height: u32,
            /// Encoded component count.
            pub components: u16,
            /// Encoded significant bits per sample.
            pub bit_depth: u8,
            /// Whether encoded samples are signed.
            pub signed: bool,
            $($extra_fields)*
        }
    };
}

/// Backend preference for JPEG 2000 lossless encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EncodeBackendPreference {
    /// Pick the fastest safe backend exposed by the caller, falling back to CPU.
    #[default]
    Auto,
    /// Require the pure Rust CPU encoder.
    CpuOnly,
    /// Require a device encoder and fail if unavailable or unsupported.
    RequireDevice,
}

/// Supported JPEG 2000 progression orders for the lossless encode facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum J2kProgressionOrder {
    /// Layer-resolution-component-position progression.
    #[default]
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

/// Supported code-block coding modes for the lossless encode facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum J2kBlockCodingMode {
    /// Classic JPEG 2000 Part 1 EBCOT block coding.
    #[default]
    Classic,
    /// High-throughput JPEG 2000 Part 15 block coding.
    HighThroughput,
}

/// Reversible transform profile for lossless JPEG 2000 output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReversibleTransform {
    /// Reversible color transform with 5/3 wavelet transform.
    #[default]
    Rct53,
    /// No color transform with 5/3 wavelet transform.
    None53,
}

/// Validation policy for the lossless encode facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum J2kEncodeValidation {
    /// Decode the produced codestream with the native CPU decoder and compare
    /// decoded samples before returning.
    #[default]
    CpuRoundTrip,
    /// Skip facade validation because the caller performs equivalent external
    /// validation, for example by decoding on a device backend.
    External,
}

/// Options controlling JPEG 2000 lossless encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the public options type exposes independent compatibility switches"
)]
#[non_exhaustive]
pub struct J2kLosslessEncodeOptions {
    /// Backend preference for encode stages.
    pub backend: EncodeBackendPreference,
    /// Code-block coding mode for the codestream.
    pub block_coding_mode: J2kBlockCodingMode,
    /// Packet progression order.
    pub progression: J2kProgressionOrder,
    /// Optional explicit lossless decomposition level request.
    ///
    /// Requests are clamped to the geometry-safe maximum for the tile.
    pub max_decomposition_levels: Option<u8>,
    /// Optional tile width and height.
    pub tile_size: Option<(u32, u32)>,
    /// Optional maximum number of complete packets to place in each tile-part.
    pub tile_part_packet_limit: Option<u16>,
    /// Number of lossless quality layers to write.
    ///
    /// HTJ2K uses this to request cleanup plus refinement coding passes from
    /// the native block encoder. Values outside 1..=32 are rejected by the
    /// native encoder.
    pub quality_layers: u8,
    /// Write a TLM marker segment.
    pub write_tlm: bool,
    /// Write PLT packet-length marker segments.
    pub write_plt: bool,
    /// Write PLM packet-length marker segments.
    pub write_plm: bool,
    /// Write PPM packed packet-header marker segments.
    pub write_ppm: bool,
    /// Write PPT packed packet-header marker segments.
    pub write_ppt: bool,
    /// Write SOP packet marker segments.
    pub write_sop: bool,
    /// Write EPH packet header termination markers.
    pub write_eph: bool,
    /// Reversible transform profile.
    pub reversible_transform: ReversibleTransform,
    /// Validation policy applied before returning encoded bytes.
    pub validation: J2kEncodeValidation,
}

impl Default for J2kLosslessEncodeOptions {
    fn default() -> Self {
        Self {
            backend: EncodeBackendPreference::Auto,
            block_coding_mode: J2kBlockCodingMode::Classic,
            progression: J2kProgressionOrder::Lrcp,
            max_decomposition_levels: None,
            tile_size: None,
            tile_part_packet_limit: None,
            quality_layers: 1,
            write_tlm: false,
            write_plt: false,
            write_plm: false,
            write_ppm: false,
            write_ppt: false,
            write_sop: false,
            write_eph: false,
            reversible_transform: ReversibleTransform::Rct53,
            validation: J2kEncodeValidation::CpuRoundTrip,
        }
    }
}

impl J2kLosslessEncodeOptions {
    /// Create JPEG 2000 lossless encode options.
    pub const fn new(
        backend: EncodeBackendPreference,
        block_coding_mode: J2kBlockCodingMode,
        progression: J2kProgressionOrder,
        max_decomposition_levels: Option<u8>,
        reversible_transform: ReversibleTransform,
        validation: J2kEncodeValidation,
    ) -> Self {
        Self {
            backend,
            block_coding_mode,
            progression,
            max_decomposition_levels,
            tile_size: None,
            tile_part_packet_limit: None,
            quality_layers: 1,
            write_tlm: false,
            write_plt: false,
            write_plm: false,
            write_ppm: false,
            write_ppt: false,
            write_sop: false,
            write_eph: false,
            reversible_transform,
            validation,
        }
    }

    /// Return options with a different backend preference.
    #[must_use]
    pub const fn with_backend(mut self, backend: EncodeBackendPreference) -> Self {
        self.backend = backend;
        self
    }

    /// Return options using adaptive accelerated routing.
    #[must_use]
    pub const fn with_accelerated_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::Auto)
    }

    /// Return options using the portable CPU route.
    #[must_use]
    pub const fn with_cpu_only_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::CpuOnly)
    }

    /// Return options requiring a strict device route.
    #[must_use]
    pub const fn with_strict_device_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::RequireDevice)
    }

    /// Return options with a different code-block coding mode.
    #[must_use]
    pub const fn with_block_coding_mode(mut self, block_coding_mode: J2kBlockCodingMode) -> Self {
        self.block_coding_mode = block_coding_mode;
        self
    }

    /// Return options with a different packet progression order.
    #[must_use]
    pub const fn with_progression(mut self, progression: J2kProgressionOrder) -> Self {
        self.progression = progression;
        self
    }

    /// Return options with a different maximum decomposition-level request.
    #[must_use]
    pub const fn with_max_decomposition_levels(
        mut self,
        max_decomposition_levels: Option<u8>,
    ) -> Self {
        self.max_decomposition_levels = max_decomposition_levels;
        self
    }

    /// Return options with a different tile size.
    #[must_use]
    pub const fn with_tile_size(mut self, tile_size: Option<(u32, u32)>) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Return options with a different tile-part packet limit.
    #[must_use]
    pub const fn with_tile_part_packet_limit(
        mut self,
        tile_part_packet_limit: Option<u16>,
    ) -> Self {
        self.tile_part_packet_limit = tile_part_packet_limit;
        self
    }

    /// Return options with a different lossless quality-layer count.
    #[must_use]
    pub const fn with_quality_layers(mut self, quality_layers: u8) -> Self {
        self.quality_layers = quality_layers;
        self
    }

    /// Return options with explicit marker segment requests.
    #[must_use]
    pub fn with_marker_segments(mut self, marker_segments: &[J2kMarkerSegment]) -> Self {
        self.write_tlm = false;
        self.write_plt = false;
        self.write_plm = false;
        self.write_ppm = false;
        self.write_ppt = false;
        self.write_sop = false;
        self.write_eph = false;
        for marker in marker_segments {
            match marker {
                J2kMarkerSegment::Sop => self.write_sop = true,
                J2kMarkerSegment::Eph => self.write_eph = true,
                J2kMarkerSegment::Tlm => self.write_tlm = true,
                J2kMarkerSegment::Plt => self.write_plt = true,
                J2kMarkerSegment::Plm => self.write_plm = true,
                J2kMarkerSegment::Ppm => self.write_ppm = true,
                J2kMarkerSegment::Ppt => self.write_ppt = true,
            }
        }
        self
    }

    /// Return options with a different reversible transform.
    #[must_use]
    pub const fn with_reversible_transform(
        mut self,
        reversible_transform: ReversibleTransform,
    ) -> Self {
        self.reversible_transform = reversible_transform;
        self
    }

    /// Return options with a different validation policy.
    #[must_use]
    pub const fn with_validation(mut self, validation: J2kEncodeValidation) -> Self {
        self.validation = validation;
        self
    }
}

/// Rate target for stable lossy JPEG 2000 encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum J2kRateTarget {
    /// Target total codestream bits per image pixel.
    BitsPerPixel(f64),
    /// Target total codestream byte size.
    Bytes(u64),
    /// Target decoded peak signal-to-noise ratio in dB.
    PsnrDb(f64),
}

/// One cumulative lossy quality layer request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct J2kQualityLayer {
    /// Cumulative target for this quality layer.
    pub target: J2kRateTarget,
}

impl J2kQualityLayer {
    /// Create a cumulative lossy quality layer target.
    #[must_use]
    pub const fn new(target: J2kRateTarget) -> Self {
        Self { target }
    }
}

/// Optional JPEG 2000 marker segment requested for lossy encode output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kMarkerSegment {
    /// SOP packet marker segments.
    Sop,
    /// EPH packet header termination markers.
    Eph,
    /// TLM tile-part length marker segment.
    Tlm,
    /// PLT packet length marker segments.
    Plt,
    /// PLM packet length marker segments.
    Plm,
    /// PPM packed packet-header marker segments.
    Ppm,
    /// PPT packed packet-header marker segments.
    Ppt,
}

/// Options controlling stable lossy JPEG 2000 encoding.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct J2kLossyEncodeOptions {
    /// Backend preference for encode stages.
    pub backend: EncodeBackendPreference,
    /// Code-block coding mode for the codestream.
    pub block_coding_mode: J2kBlockCodingMode,
    /// Packet progression order.
    pub progression: J2kProgressionOrder,
    /// Optional explicit lossy decomposition level request.
    pub max_decomposition_levels: Option<u8>,
    /// Single codestream rate target.
    pub rate_target: Option<J2kRateTarget>,
    /// Cumulative quality layer targets.
    pub quality_layers: Vec<J2kQualityLayer>,
    /// Optional tile width and height.
    pub tile_size: Option<(u32, u32)>,
    /// Optional maximum number of complete packets to place in each tile-part.
    pub tile_part_packet_limit: Option<u16>,
    /// Optional precinct exponents in COD/COC order.
    pub precinct_exponents: Vec<(u8, u8)>,
    /// Optional marker segments requested for the codestream.
    pub marker_segments: Vec<J2kMarkerSegment>,
    /// Allowed PSNR target tolerance in dB.
    pub psnr_tolerance_db: f64,
    /// Iteration budget for lossy target searches.
    pub psnr_iteration_budget: u8,
    /// Validation policy applied before returning encoded bytes.
    pub validation: J2kEncodeValidation,
}

impl Default for J2kLossyEncodeOptions {
    fn default() -> Self {
        Self {
            backend: EncodeBackendPreference::Auto,
            block_coding_mode: J2kBlockCodingMode::Classic,
            progression: J2kProgressionOrder::Lrcp,
            max_decomposition_levels: None,
            rate_target: None,
            quality_layers: Vec::new(),
            tile_size: None,
            tile_part_packet_limit: None,
            precinct_exponents: Vec::new(),
            marker_segments: Vec::new(),
            psnr_tolerance_db: 0.25,
            psnr_iteration_budget: 8,
            validation: J2kEncodeValidation::CpuRoundTrip,
        }
    }
}

impl J2kLossyEncodeOptions {
    /// Return options with a different backend preference.
    #[must_use]
    pub fn with_backend(mut self, backend: EncodeBackendPreference) -> Self {
        self.backend = backend;
        self
    }

    /// Return options using adaptive accelerated routing.
    #[must_use]
    pub fn with_accelerated_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::Auto)
    }

    /// Return options using the portable CPU route.
    #[must_use]
    pub fn with_cpu_only_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::CpuOnly)
    }

    /// Return options requiring a strict device route.
    #[must_use]
    pub fn with_strict_device_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::RequireDevice)
    }

    /// Return options with a different code-block coding mode.
    #[must_use]
    pub fn with_block_coding_mode(mut self, block_coding_mode: J2kBlockCodingMode) -> Self {
        self.block_coding_mode = block_coding_mode;
        self
    }

    /// Return options with a different packet progression order.
    #[must_use]
    pub fn with_progression(mut self, progression: J2kProgressionOrder) -> Self {
        self.progression = progression;
        self
    }

    /// Return options with a different maximum decomposition-level request.
    #[must_use]
    pub fn with_max_decomposition_levels(mut self, max_decomposition_levels: Option<u8>) -> Self {
        self.max_decomposition_levels = max_decomposition_levels;
        self
    }

    /// Return options with a different single codestream rate target.
    #[must_use]
    pub fn with_rate_target(mut self, rate_target: Option<J2kRateTarget>) -> Self {
        self.rate_target = rate_target;
        self
    }

    /// Return options with different cumulative quality layer targets.
    #[must_use]
    pub fn with_quality_layers(mut self, quality_layers: Vec<J2kQualityLayer>) -> Self {
        self.quality_layers = quality_layers;
        self
    }

    /// Return options with a different tile size.
    #[must_use]
    pub fn with_tile_size(mut self, tile_size: Option<(u32, u32)>) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Return options with a different tile-part packet limit.
    #[must_use]
    pub fn with_tile_part_packet_limit(mut self, tile_part_packet_limit: Option<u16>) -> Self {
        self.tile_part_packet_limit = tile_part_packet_limit;
        self
    }

    /// Return options with different optional marker segment requests.
    #[must_use]
    pub fn with_marker_segments(mut self, marker_segments: Vec<J2kMarkerSegment>) -> Self {
        self.marker_segments = marker_segments;
        self
    }

    /// Return options with a different validation policy.
    #[must_use]
    pub fn with_validation(mut self, validation: J2kEncodeValidation) -> Self {
        self.validation = validation;
        self
    }
}

define_encoded_j2k! {
    /// Encoded JPEG 2000 lossless codestream and encode metadata.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EncodedJ2k {
    }
}

/// Metrics reported by stable lossy JPEG 2000 encoding.
#[derive(Debug, Clone, PartialEq)]
pub struct J2kLossyEncodeReport {
    /// Requested effective rate target.
    pub target: Option<J2kRateTarget>,
    /// Number of cumulative quality layers emitted.
    pub quality_layers: u16,
    /// Final native irreversible quantization scale.
    pub quantization_scale: f32,
    /// Total encoded codestream bytes.
    pub actual_bytes: u64,
    /// Total codestream bits per image pixel.
    pub actual_bits_per_pixel: f64,
    /// Decoded PSNR in dB when CPU validation was requested.
    pub psnr_db: Option<f64>,
    /// HTJ2K rate granularity in bytes when HT block coding is used.
    pub ht_rate_granularity_bytes: Option<u64>,
}

define_encoded_j2k! {
    /// Encoded JPEG 2000 lossy codestream and encode metadata.
    #[derive(Debug, Clone, PartialEq)]
    pub struct EncodedLossyJ2k {
        /// Lossy encode metrics.
        pub report: J2kLossyEncodeReport,
    }
}
