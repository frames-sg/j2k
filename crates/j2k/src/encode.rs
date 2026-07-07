// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use j2k_core::{BackendKind, Unsupported};
use j2k_native::{
    DecodeSettings, EncodeComponentPlane as NativeEncodeComponentPlane, EncodeOptions,
    EncodeProgressionOrder, EncodeRoiRegion as NativeEncodeRoiRegion,
    EncodeTypedComponentPlane as NativeEncodeTypedComponentPlane, Image,
};

use crate::{
    J2kError, {J2kEncodeDispatchReport, J2kEncodeStageAccelerator},
};

const MAX_JPEG2000_PART1_COMPONENTS: u16 = 16_384;
const MAX_RAW_PIXEL_ENCODE_BIT_DEPTH: u8 = 24;
const MAX_PART1_SAMPLE_BIT_DEPTH: u8 = 38;
const MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES: u16 = 37;
const MAX_HTJ2K_ENCODE_BITPLANES: u16 = 31;

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
#[allow(clippy::struct_excessive_bools)]
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

/// Borrowed interleaved samples and image geometry for lossless encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessSamples<'a> {
    /// Interleaved sample bytes.
    pub data: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Component count. Component counts beyond four are encoded as independent
    /// component planes without a multi-component transform.
    pub components: u16,
    /// Significant bits per component sample.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampleGeometry {
    expected_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct SampleGeometryRequest<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    max_bit_depth: u8,
    component_what: &'static str,
    bit_depth_what: &'static str,
}

fn validate_sample_geometry(
    request: SampleGeometryRequest<'_>,
) -> Result<SampleGeometry, J2kError> {
    let SampleGeometryRequest {
        data,
        width,
        height,
        components,
        bit_depth,
        max_bit_depth,
        component_what,
        bit_depth_what,
    } = request;
    if width == 0 || height == 0 {
        return Err(J2kError::InvalidSamples {
            what: "dimensions must be non-zero".to_string(),
        });
    }
    if components == 0 || components > MAX_JPEG2000_PART1_COMPONENTS {
        return Err(J2kError::Unsupported(Unsupported {
            what: component_what,
        }));
    }
    if bit_depth == 0 || bit_depth > max_bit_depth {
        return Err(J2kError::Unsupported(Unsupported {
            what: bit_depth_what,
        }));
    }
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth);
    let expected_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(usize::from(components)))
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow { width, height })?;
    if data.len() != expected_bytes {
        let what = if data.len() < expected_bytes {
            format!(
                "pixel data too short: expected {expected_bytes} bytes, got {}",
                data.len()
            )
        } else {
            format!(
                "pixel data has trailing bytes: expected {expected_bytes} bytes, got {}",
                data.len()
            )
        };
        return Err(J2kError::InvalidSamples { what });
    }
    Ok(SampleGeometry { expected_bytes })
}

impl<'a> J2kLosslessSamples<'a> {
    /// Validate and construct a sample descriptor.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        components: u16,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kError> {
        let geometry = validate_sample_geometry(SampleGeometryRequest {
            data,
            width,
            height,
            components,
            bit_depth,
            max_bit_depth: MAX_PART1_SAMPLE_BIT_DEPTH,
            component_what: "JPEG 2000 lossless encode supports 1-16384 component samples",
            bit_depth_what: "JPEG 2000 lossless encode supports 1-38 bits per sample for classic reversible codestreams",
        })?;
        debug_assert_eq!(geometry.expected_bytes, data.len());
        Ok(Self {
            data,
            width,
            height,
            components,
            bit_depth,
            signed,
        })
    }
}

/// Rectangular region-of-interest request for lossless JPEG 2000 maxshift
/// encoding.
///
/// The rectangle is expressed in full-resolution image pixels. All regions for
/// one component must use the same non-zero `shift`, because JPEG 2000 stores
/// one RGN maxshift value per component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kRoiRegion {
    /// Component index to which the ROI applies.
    pub component: u16,
    /// Left edge in image pixels.
    pub x: u32,
    /// Top edge in image pixels.
    pub y: u32,
    /// Width in image pixels.
    pub width: u32,
    /// Height in image pixels.
    pub height: u32,
    /// Maxshift value to write for this component.
    pub shift: u8,
}

/// Borrowed samples for one lossless component plane.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessComponentPlane<'a> {
    /// Row-major little-endian samples for this component's own SIZ grid.
    pub data: &'a [u8],
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
}

/// Borrowed component-plane samples and reference-grid image geometry for
/// lossless encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessComponentSamples<'a> {
    /// Component planes in codestream order.
    pub planes: &'a [J2kLosslessComponentPlane<'a>],
    /// Reference-grid image width in pixels.
    pub width: u32,
    /// Reference-grid image height in pixels.
    pub height: u32,
    /// Significant bits per component sample. Mixed component bit depths are
    /// not yet supported by the encode facade.
    pub bit_depth: u8,
    /// Whether every component sample is signed. Mixed signedness is not yet
    /// supported by the encode facade.
    pub signed: bool,
}

impl<'a> J2kLosslessComponentSamples<'a> {
    /// Validate and construct a component-plane sample descriptor.
    pub fn new(
        planes: &'a [J2kLosslessComponentPlane<'a>],
        width: u32,
        height: u32,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kError> {
        if width == 0 || height == 0 {
            return Err(J2kError::InvalidSamples {
                what: "dimensions must be non-zero".to_string(),
            });
        }
        if planes.is_empty() || planes.len() > usize::from(MAX_JPEG2000_PART1_COMPONENTS) {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless component-plane encode supports 1-16384 components",
            }));
        }
        if bit_depth == 0 || bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless component-plane encode supports 1-38 bits per sample",
            }));
        }
        for (index, plane) in planes.iter().enumerate() {
            validate_component_plane_geometry(plane, width, height, bit_depth, index)?;
        }
        Ok(Self {
            planes,
            width,
            height,
            bit_depth,
            signed,
        })
    }

    /// Return the component count.
    #[must_use]
    pub fn components(&self) -> u16 {
        u16::try_from(self.planes.len()).unwrap_or(MAX_JPEG2000_PART1_COMPONENTS)
    }
}

/// Borrowed samples for one typed lossless component plane.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessTypedComponentPlane<'a> {
    /// Row-major little-endian samples for this component's own SIZ grid.
    pub data: &'a [u8],
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Significant bits per sample for this component.
    pub bit_depth: u8,
    /// Whether samples in this component are signed.
    pub signed: bool,
}

/// Borrowed typed component-plane samples and reference-grid image geometry for
/// lossless encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessTypedComponentSamples<'a> {
    /// Component planes in codestream order.
    pub planes: &'a [J2kLosslessTypedComponentPlane<'a>],
    /// Reference-grid image width in pixels.
    pub width: u32,
    /// Reference-grid image height in pixels.
    pub height: u32,
}

impl<'a> J2kLosslessTypedComponentSamples<'a> {
    /// Validate and construct a typed component-plane sample descriptor.
    pub fn new(
        planes: &'a [J2kLosslessTypedComponentPlane<'a>],
        width: u32,
        height: u32,
    ) -> Result<Self, J2kError> {
        if width == 0 || height == 0 {
            return Err(J2kError::InvalidSamples {
                what: "dimensions must be non-zero".to_string(),
            });
        }
        if planes.is_empty() || planes.len() > usize::from(MAX_JPEG2000_PART1_COMPONENTS) {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless typed component-plane encode supports 1-16384 components",
            }));
        }
        for (index, plane) in planes.iter().enumerate() {
            validate_typed_component_plane_geometry(plane, width, height, index)?;
        }
        Ok(Self {
            planes,
            width,
            height,
        })
    }

    /// Return the component count.
    #[must_use]
    pub fn components(&self) -> u16 {
        u16::try_from(self.planes.len()).unwrap_or(MAX_JPEG2000_PART1_COMPONENTS)
    }

    /// Return the maximum significant bit depth across all components.
    #[must_use]
    pub fn max_bit_depth(&self) -> u8 {
        self.planes
            .iter()
            .map(|plane| plane.bit_depth)
            .max()
            .unwrap_or(0)
    }

    /// Return whether every component is signed.
    #[must_use]
    pub fn all_components_signed(&self) -> bool {
        self.planes.iter().all(|plane| plane.signed)
    }
}

fn validate_component_plane_geometry(
    plane: &J2kLosslessComponentPlane<'_>,
    width: u32,
    height: u32,
    bit_depth: u8,
    index: usize,
) -> Result<(), J2kError> {
    if plane.x_rsiz == 0 || plane.y_rsiz == 0 {
        return Err(J2kError::InvalidSamples {
            what: format!("component plane {index} sampling factors must be non-zero"),
        });
    }
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth);
    let component_width = width.div_ceil(u32::from(plane.x_rsiz));
    let component_height = height.div_ceil(u32::from(plane.y_rsiz));
    let expected_bytes = (component_width as usize)
        .checked_mul(component_height as usize)
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow { width, height })?;
    if plane.data.len() != expected_bytes {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "component plane {index} data length mismatch: expected {expected_bytes} bytes, got {}",
                plane.data.len()
            ),
        });
    }
    Ok(())
}

fn validate_typed_component_plane_geometry(
    plane: &J2kLosslessTypedComponentPlane<'_>,
    width: u32,
    height: u32,
    index: usize,
) -> Result<(), J2kError> {
    if plane.x_rsiz == 0 || plane.y_rsiz == 0 {
        return Err(J2kError::InvalidSamples {
            what: format!("component plane {index} sampling factors must be non-zero"),
        });
    }
    if plane.bit_depth == 0 || plane.bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossless typed component-plane encode supports 1-38 bits per sample",
        }));
    }
    let bytes_per_sample = raw_pixel_bytes_per_sample(plane.bit_depth);
    let component_width = width.div_ceil(u32::from(plane.x_rsiz));
    let component_height = height.div_ceil(u32::from(plane.y_rsiz));
    let expected_bytes = (component_width as usize)
        .checked_mul(component_height as usize)
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow { width, height })?;
    if plane.data.len() != expected_bytes {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "component plane {index} data length mismatch: expected {expected_bytes} bytes, got {}",
                plane.data.len()
            ),
        });
    }
    Ok(())
}

/// Borrowed interleaved samples and image geometry for lossy encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLossySamples<'a> {
    /// Interleaved sample bytes.
    pub data: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Component count. Component counts beyond four are encoded as independent
    /// component planes without a multi-component transform.
    pub components: u16,
    /// Significant bits per component sample.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
}

impl<'a> J2kLossySamples<'a> {
    /// Validate and construct a lossy sample descriptor.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        components: u16,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kError> {
        let geometry = validate_sample_geometry(SampleGeometryRequest {
            data,
            width,
            height,
            components,
            bit_depth,
            max_bit_depth: MAX_PART1_SAMPLE_BIT_DEPTH,
            component_what: "JPEG 2000 lossy encode supports 1-16384 component samples",
            bit_depth_what: "JPEG 2000 lossy encode supports 1-38 bits per sample",
        })?;
        debug_assert_eq!(geometry.expected_bytes, data.len());
        Ok(Self {
            data,
            width,
            height,
            components,
            bit_depth,
            signed,
        })
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

/// Encode interleaved samples into a raw JPEG 2000 lossless codestream.
pub fn encode_j2k_lossless(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    validate_lossless_high_bit_options(samples, options)?;
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu(samples, *options)?;
    validate_lossless_roundtrip(samples, &codestream, options.validation)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

/// Encode interleaved samples into a raw lossless JPEG 2000 codestream with
/// rectangular ROI maxshift.
///
/// ROI encode currently uses the native CPU encoder. The produced codestream
/// is validated with the same policy as [`encode_j2k_lossless`].
pub fn encode_j2k_lossless_with_roi_regions(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
    roi_regions: &[J2kRoiRegion],
) -> Result<EncodedJ2k, J2kError> {
    validate_lossless_high_bit_options(samples, options)?;
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu_with_roi_regions(samples, *options, roi_regions)?;
    validate_lossless_roundtrip(samples, &codestream, options.validation)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

/// Encode component-plane samples into a raw JPEG 2000 lossless codestream.
///
/// This is the lossless encode entry point for images whose component grids
/// cannot be represented as one interleaved full-resolution sample stream, such
/// as codestreams with component sampling. Components are encoded without a
/// reversible color transform.
pub fn encode_j2k_lossless_components(
    samples: J2kLosslessComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    if samples.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return encode_j2k_lossless_components_high_bit(samples, options);
    }
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu_components(samples, *options)?;
    validate_lossless_component_roundtrip(samples, &codestream, options.validation)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components(),
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

fn encode_j2k_lossless_components_high_bit(
    samples: J2kLosslessComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    if samples
        .planes
        .iter()
        .any(|plane| plane.x_rsiz != 1 || plane.y_rsiz != 1)
    {
        return encode_j2k_lossless_sampled_components_high_bit(samples, options);
    }

    let interleaved = interleave_component_planes(samples)?;
    let raw_samples = J2kLosslessSamples::new(
        &interleaved,
        samples.width,
        samples.height,
        samples.components(),
        samples.bit_depth,
        samples.signed,
    )?;
    let raw_options = (*options)
        .with_reversible_transform(ReversibleTransform::None53)
        .with_validation(J2kEncodeValidation::External);
    let encoded = encode_j2k_lossless(raw_samples, &raw_options)?;
    validate_lossless_high_bit_component_roundtrip(
        samples,
        &encoded.codestream,
        options.validation,
    )?;
    Ok(encoded)
}

fn encode_j2k_lossless_sampled_components_high_bit(
    samples: J2kLosslessComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    let typed_planes = samples
        .planes
        .iter()
        .map(|plane| J2kLosslessTypedComponentPlane {
            data: plane.data,
            x_rsiz: plane.x_rsiz,
            y_rsiz: plane.y_rsiz,
            bit_depth: samples.bit_depth,
            signed: samples.signed,
        })
        .collect::<Vec<_>>();
    let typed_samples =
        J2kLosslessTypedComponentSamples::new(&typed_planes, samples.width, samples.height)?;
    encode_j2k_lossless_typed_components(typed_samples, options)
}

/// Encode typed component-plane samples into a raw JPEG 2000 lossless
/// codestream.
///
/// This is the lossless encode entry point for codestreams whose components
/// have different precision or signedness. Components are encoded without a
/// reversible color transform.
pub fn encode_j2k_lossless_typed_components(
    samples: J2kLosslessTypedComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu_typed_components(samples, *options)?;
    validate_lossless_typed_component_roundtrip(samples, &codestream, options.validation)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components(),
        bit_depth: samples.max_bit_depth(),
        signed: samples.all_components_signed(),
    })
}

/// Encode interleaved samples with an optional device encode-stage accelerator.
///
/// Accelerators return CPU fallback by reporting no dispatch. `Auto` accepts
/// that fallback; `RequireDevice` requires at least one dispatch. Any
/// accelerator error or codestream validation error is returned to the caller.
pub fn encode_j2k_lossless_with_accelerator(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
    accelerated_backend: BackendKind,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<EncodedJ2k, J2kError> {
    validate_lossless_high_bit_options(samples, options)?;
    if samples.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit lossless encode currently uses the CPU classic reversible path only",
        }));
    }
    if options.backend == EncodeBackendPreference::CpuOnly {
        return encode_j2k_lossless(samples, options);
    }

    let before = accelerator.dispatch_report();
    let required_stages = required_encode_stages(samples, *options, accelerated_backend);
    let codestream = encode_with_native_accelerator(samples, *options, accelerator)?;
    let dispatch = accelerator.dispatch_report().saturating_delta(before);
    validate_lossless_roundtrip(samples, &codestream, options.validation)?;

    let backend = resolve_accelerated_encode_backend(
        options.backend,
        accelerated_backend,
        dispatch,
        required_stages,
    )?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: dispatch,
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

/// Encode interleaved samples into a raw JPEG 2000 lossy codestream.
pub fn encode_j2k_lossy(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<EncodedLossyJ2k, J2kError> {
    validate_lossy_options(options)?;
    validate_lossy_high_bit_options(samples, options)?;
    let target = effective_lossy_target(options)?;
    let attempt = encode_lossy_targeted(samples, options, target, |scale| {
        encode_cpu_lossy(samples, options, scale)
    })?;
    let report = lossy_report(samples, options, target, &attempt)?;
    Ok(EncodedLossyJ2k {
        codestream: attempt.codestream,
        backend: resolve_encode_backend(options.backend)?,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
        report,
    })
}

/// Encode interleaved samples into a raw lossy JPEG 2000 codestream with
/// rectangular ROI maxshift.
///
/// ROI encode currently uses the native CPU encoder and preserves the normal
/// lossy rate/PSNR reporting behavior.
pub fn encode_j2k_lossy_with_roi_regions(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    roi_regions: &[J2kRoiRegion],
) -> Result<EncodedLossyJ2k, J2kError> {
    validate_lossy_options(options)?;
    validate_lossy_high_bit_options(samples, options)?;
    let native_roi_regions = native_roi_regions_for_samples(
        samples.width,
        samples.height,
        samples.components,
        roi_regions,
    )?;
    let target = effective_lossy_target(options)?;
    let attempt = encode_lossy_targeted(samples, options, target, |scale| {
        encode_cpu_lossy_with_roi_regions(samples, options, scale, &native_roi_regions)
    })?;
    let report = lossy_report(samples, options, target, &attempt)?;
    Ok(EncodedLossyJ2k {
        codestream: attempt.codestream,
        backend: resolve_encode_backend(options.backend)?,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
        report,
    })
}

/// Encode interleaved lossy samples with an optional device encode-stage accelerator.
pub fn encode_j2k_lossy_with_accelerator(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    accelerated_backend: BackendKind,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<EncodedLossyJ2k, J2kError> {
    if options.backend == EncodeBackendPreference::CpuOnly {
        return encode_j2k_lossy(samples, options);
    }

    validate_lossy_options(options)?;
    validate_lossy_high_bit_options(samples, options)?;
    let target = effective_lossy_target(options)?;
    let before = accelerator.dispatch_report();
    let required_stages = required_lossy_encode_stages(samples, options, accelerated_backend);
    let attempt = encode_lossy_targeted(samples, options, target, |scale| {
        encode_lossy_with_native_accelerator(samples, options, scale, accelerator)
    })?;
    let dispatch = accelerator.dispatch_report().saturating_delta(before);
    let backend = resolve_accelerated_encode_backend(
        options.backend,
        accelerated_backend,
        dispatch,
        required_stages,
    )?;
    let report = lossy_report(samples, options, target, &attempt)?;
    Ok(EncodedLossyJ2k {
        codestream: attempt.codestream,
        backend,
        dispatch_report: dispatch,
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
        report,
    })
}

fn resolve_encode_backend(preference: EncodeBackendPreference) -> Result<BackendKind, J2kError> {
    match preference {
        EncodeBackendPreference::Auto | EncodeBackendPreference::CpuOnly => Ok(BackendKind::Cpu),
        EncodeBackendPreference::RequireDevice => Err(J2kError::Unsupported(Unsupported {
            what: "device JPEG 2000 lossless encode backend is unavailable",
        })),
    }
}

fn resolve_accelerated_encode_backend(
    preference: EncodeBackendPreference,
    accelerated_backend: BackendKind,
    dispatch: J2kEncodeDispatchReport,
    required_stages: RequiredEncodeStages,
) -> Result<BackendKind, J2kError> {
    if required_stages.satisfied_by(dispatch) {
        return Ok(accelerated_backend);
    }
    match preference {
        EncodeBackendPreference::RequireDevice => Err(J2kError::Unsupported(Unsupported {
            what: required_stages.missing_message(dispatch),
        })),
        EncodeBackendPreference::Auto | EncodeBackendPreference::CpuOnly => Ok(BackendKind::Cpu),
    }
}

fn validate_lossless_high_bit_options(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<(), J2kError> {
    if samples.bit_depth <= MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Ok(());
    }
    let decomposition_levels = j2k_lossless_decomposition_levels_for_options(samples, *options);
    let reversible_gain = if decomposition_levels == 0 { 0 } else { 2 };
    let coded_bitplanes = u16::from(samples.bit_depth) + reversible_gain;
    if options.block_coding_mode == J2kBlockCodingMode::HighThroughput && decomposition_levels > 0 {
        return Err(J2kError::Unsupported(Unsupported {
            what: "HTJ2K high-bit lossless encode with DWT remains blocked by the current HT integer coefficient path",
        }));
    }
    if options.block_coding_mode == J2kBlockCodingMode::HighThroughput
        && coded_bitplanes > MAX_HTJ2K_ENCODE_BITPLANES
    {
        return Err(J2kError::Unsupported(Unsupported {
            what: "HTJ2K high-bit lossless encode exceeds the current HT block bitplane limit",
        }));
    }
    if options.block_coding_mode == J2kBlockCodingMode::Classic
        && coded_bitplanes > MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES
    {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit classic lossless encode exceeds the current no-quantization guard/exponent signaling limit",
        }));
    }
    if !matches!(
        options.block_coding_mode,
        J2kBlockCodingMode::Classic | J2kBlockCodingMode::HighThroughput
    ) {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit lossless encode currently requires classic J2K or HTJ2K block coding",
        }));
    }
    if options.backend == EncodeBackendPreference::RequireDevice {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit lossless encode currently uses the CPU reversible path only",
        }));
    }
    Ok(())
}

fn validate_lossy_high_bit_options(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<(), J2kError> {
    if samples.bit_depth <= MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Ok(());
    }
    if options.block_coding_mode == J2kBlockCodingMode::HighThroughput {
        return Err(J2kError::Unsupported(Unsupported {
            what: "HTJ2K high-bit lossy encode remains blocked by the current HT integer coefficient path",
        }));
    }
    if options.backend == EncodeBackendPreference::RequireDevice {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit lossy encode currently uses the CPU irreversible path only",
        }));
    }
    Ok(())
}

fn encode_cpu(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossless_options(samples, options);
    j2k_native::encode(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
    )
    .map_err(|err| J2kError::backend(format!("JPEG 2000 lossless encode failed: {err}")))
}

fn encode_cpu_with_roi_regions(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
    roi_regions: &[J2kRoiRegion],
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossless_options(samples, options);
    let native_roi_regions = native_roi_regions_for_lossless_samples(samples, roi_regions)?;
    j2k_native::encode_with_roi_regions(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        &native_roi_regions,
    )
    .map_err(map_native_lossless_roi_encode_error)
}

fn map_native_lossless_roi_encode_error(err: &'static str) -> J2kError {
    match err {
        "ROI maxshift exceeds supported coded bitplane count" => {
            J2kError::Unsupported(Unsupported { what: err })
        }
        _ => J2kError::backend(format!("JPEG 2000 lossless ROI encode failed: {err}")),
    }
}

fn native_roi_regions_for_lossless_samples(
    samples: J2kLosslessSamples<'_>,
    roi_regions: &[J2kRoiRegion],
) -> Result<Vec<NativeEncodeRoiRegion>, J2kError> {
    native_roi_regions_for_samples(
        samples.width,
        samples.height,
        samples.components,
        roi_regions,
    )
}

fn native_roi_regions_for_samples(
    width: u32,
    height: u32,
    components: u16,
    roi_regions: &[J2kRoiRegion],
) -> Result<Vec<NativeEncodeRoiRegion>, J2kError> {
    roi_regions
        .iter()
        .map(|region| {
            if region.component >= components {
                return Err(J2kError::InvalidSamples {
                    what: "ROI region component index out of range".to_string(),
                });
            }
            if region.width == 0 || region.height == 0 {
                return Err(J2kError::InvalidSamples {
                    what: "ROI region dimensions must be non-zero".to_string(),
                });
            }
            if region.shift == 0 {
                return Err(J2kError::InvalidSamples {
                    what: "ROI region maxshift must be non-zero".to_string(),
                });
            }
            let x1 =
                region
                    .x
                    .checked_add(region.width)
                    .ok_or_else(|| J2kError::InvalidSamples {
                        what: "ROI region bounds overflow".to_string(),
                    })?;
            let y1 =
                region
                    .y
                    .checked_add(region.height)
                    .ok_or_else(|| J2kError::InvalidSamples {
                        what: "ROI region bounds overflow".to_string(),
                    })?;
            if region.x >= width || region.y >= height || x1 > width || y1 > height {
                return Err(J2kError::InvalidSamples {
                    what: "ROI region must be inside image bounds".to_string(),
                });
            }
            Ok(NativeEncodeRoiRegion {
                component: region.component,
                x: region.x,
                y: region.y,
                width: region.width,
                height: region.height,
                shift: region.shift,
            })
        })
        .collect()
}

fn encode_cpu_components(
    samples: J2kLosslessComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> Result<Vec<u8>, J2kError> {
    let native_options = native_lossless_component_options(samples, options);
    let planes = samples
        .planes
        .iter()
        .map(|plane| NativeEncodeComponentPlane {
            data: plane.data,
            x_rsiz: plane.x_rsiz,
            y_rsiz: plane.y_rsiz,
        })
        .collect::<Vec<_>>();
    j2k_native::encode_component_planes_53(
        &planes,
        samples.width,
        samples.height,
        samples.bit_depth,
        samples.signed,
        &native_options,
    )
    .map_err(|err| {
        J2kError::backend(format!(
            "JPEG 2000 lossless component-plane encode failed: {err}"
        ))
    })
}

fn interleave_component_planes(
    samples: J2kLosslessComponentSamples<'_>,
) -> Result<Vec<u8>, J2kError> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(samples.bit_depth);
    let pixel_count = (samples.width as usize)
        .checked_mul(samples.height as usize)
        .ok_or(J2kError::DimensionOverflow {
            width: samples.width,
            height: samples.height,
        })?;
    let capacity = pixel_count
        .checked_mul(samples.planes.len())
        .and_then(|sample_count| sample_count.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow {
            width: samples.width,
            height: samples.height,
        })?;
    let mut interleaved = Vec::with_capacity(capacity);
    for sample_idx in 0..pixel_count {
        let start =
            sample_idx
                .checked_mul(bytes_per_sample)
                .ok_or(J2kError::DimensionOverflow {
                    width: samples.width,
                    height: samples.height,
                })?;
        let end = start + bytes_per_sample;
        for plane in samples.planes {
            interleaved.extend_from_slice(&plane.data[start..end]);
        }
    }
    Ok(interleaved)
}

fn encode_cpu_typed_components(
    samples: J2kLosslessTypedComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> Result<Vec<u8>, J2kError> {
    let native_options = native_lossless_typed_component_options(samples, options);
    let planes = samples
        .planes
        .iter()
        .map(|plane| NativeEncodeTypedComponentPlane {
            data: plane.data,
            x_rsiz: plane.x_rsiz,
            y_rsiz: plane.y_rsiz,
            bit_depth: plane.bit_depth,
            signed: plane.signed,
        })
        .collect::<Vec<_>>();
    j2k_native::encode_typed_component_planes_53(
        &planes,
        samples.width,
        samples.height,
        &native_options,
    )
    .map_err(|err| {
        J2kError::backend(format!(
            "JPEG 2000 lossless typed component-plane encode failed: {err}"
        ))
    })
}

fn encode_with_native_accelerator(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossless_options(samples, options);
    j2k_native::encode_with_accelerator(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        accelerator,
    )
    .map_err(|err| J2kError::backend(format!("JPEG 2000 lossless encode failed: {err}")))
}

struct LossyAttempt {
    codestream: Vec<u8>,
    quantization_scale: f32,
}

fn encode_cpu_lossy(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossy_options(samples, options, quantization_scale)?;
    j2k_native::encode(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
    )
    .map_err(|err| J2kError::backend(format!("JPEG 2000 lossy encode failed: {err}")))
}

fn encode_cpu_lossy_with_roi_regions(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
    roi_regions: &[NativeEncodeRoiRegion],
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossy_options(samples, options, quantization_scale)?;
    j2k_native::encode_with_roi_regions(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        roi_regions,
    )
    .map_err(|err| J2kError::backend(format!("JPEG 2000 lossy ROI encode failed: {err}")))
}

fn encode_lossy_with_native_accelerator(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossy_options(samples, options, quantization_scale)?;
    j2k_native::encode_with_accelerator(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        accelerator,
    )
    .map_err(|err| J2kError::backend(format!("JPEG 2000 lossy encode failed: {err}")))
}

fn encode_lossy_targeted(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target: Option<J2kRateTarget>,
    mut encode_at_scale: impl FnMut(f32) -> Result<Vec<u8>, J2kError>,
) -> Result<LossyAttempt, J2kError> {
    match target {
        None => {
            let codestream = encode_at_scale(1.0)?;
            Ok(LossyAttempt {
                codestream,
                quantization_scale: 1.0,
            })
        }
        Some(J2kRateTarget::Bytes(bytes)) => {
            encode_lossy_to_byte_target(samples, options, bytes, encode_at_scale)
        }
        Some(J2kRateTarget::BitsPerPixel(bits_per_pixel)) => {
            let target_bytes = target_bytes_for_bpp(samples, bits_per_pixel)?;
            encode_lossy_to_byte_target(samples, options, target_bytes, encode_at_scale)
        }
        Some(J2kRateTarget::PsnrDb(psnr_db)) => {
            encode_lossy_to_psnr_target(samples, options, psnr_db, encode_at_scale)
        }
    }
}

fn encode_lossy_to_byte_target(
    _samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target_bytes: u64,
    mut encode_at_scale: impl FnMut(f32) -> Result<Vec<u8>, J2kError>,
) -> Result<LossyAttempt, J2kError> {
    let tolerance = byte_target_tolerance(target_bytes);
    let mut low = 1.0f32;
    let mut high = 1.0f32;
    let mut best = LossyAttempt {
        codestream: encode_at_scale(high)?,
        quantization_scale: high,
    };
    let mut best_diff = byte_target_diff(best.codestream.len() as u64, target_bytes);

    while best.codestream.len() as u64 > target_bytes.saturating_add(tolerance)
        && high < 1_048_576.0
    {
        low = high;
        high *= 2.0;
        let codestream = encode_at_scale(high)?;
        let diff = byte_target_diff(codestream.len() as u64, target_bytes);
        if diff < best_diff {
            best = LossyAttempt {
                codestream,
                quantization_scale: high,
            };
            best_diff = diff;
        }
    }

    if best.codestream.len() as u64 > target_bytes.saturating_add(tolerance) {
        return Err(J2kError::RateTargetUnreachable {
            target: format!("{target_bytes} bytes"),
            best: format!("{} bytes", best.codestream.len()),
        });
    }

    for _ in 0..options.psnr_iteration_budget.max(1) {
        let mid = (low + high) * 0.5;
        let codestream = encode_at_scale(mid)?;
        let len = codestream.len() as u64;
        let diff = byte_target_diff(len, target_bytes);
        if diff < best_diff {
            best = LossyAttempt {
                codestream,
                quantization_scale: mid,
            };
            best_diff = diff;
        }
        if len > target_bytes {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok(best)
}

fn encode_lossy_to_psnr_target(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target_psnr_db: f64,
    mut encode_at_scale: impl FnMut(f32) -> Result<Vec<u8>, J2kError>,
) -> Result<LossyAttempt, J2kError> {
    let tolerance = options.psnr_tolerance_db;
    let mut low = 1.0f32;
    let mut high = 1.0f32;
    let mut best = LossyAttempt {
        codestream: encode_at_scale(high)?,
        quantization_scale: high,
    };
    let mut best_psnr = decoded_psnr(samples, &best.codestream)?;
    if best_psnr + tolerance < target_psnr_db {
        return Err(J2kError::RateTargetUnreachable {
            target: format!("{target_psnr_db:.3} dB"),
            best: format!("{best_psnr:.3} dB"),
        });
    }

    for _ in 0..options.psnr_iteration_budget.max(1) {
        high *= 2.0;
        let codestream = encode_at_scale(high)?;
        let psnr = decoded_psnr(samples, &codestream)?;
        if psnr + tolerance >= target_psnr_db {
            best = LossyAttempt {
                codestream,
                quantization_scale: high,
            };
            best_psnr = psnr;
            low = high;
        } else {
            break;
        }
    }

    for _ in 0..options.psnr_iteration_budget.max(1) {
        let mid = (low + high) * 0.5;
        let codestream = encode_at_scale(mid)?;
        let psnr = decoded_psnr(samples, &codestream)?;
        if psnr + tolerance >= target_psnr_db {
            best = LossyAttempt {
                codestream,
                quantization_scale: mid,
            };
            best_psnr = psnr;
            low = mid;
        } else {
            high = mid;
        }
    }

    let _ = best_psnr;
    Ok(best)
}

fn native_lossless_options(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> EncodeOptions {
    let progression_order = native_progression_order(options.progression);
    EncodeOptions {
        reversible: true,
        num_decomposition_levels: j2k_lossless_decomposition_levels_for_options(samples, options),
        use_ht_block_coding: options.block_coding_mode == J2kBlockCodingMode::HighThroughput,
        progression_order,
        write_tlm: options.write_tlm || options.progression == J2kProgressionOrder::Rpcl,
        write_plt: options.write_plt,
        write_plm: options.write_plm,
        write_ppm: options.write_ppm,
        write_ppt: options.write_ppt,
        write_sop: options.write_sop,
        write_eph: options.write_eph,
        use_mct: options.reversible_transform == ReversibleTransform::Rct53
            && matches!(samples.components, 3 | 4),
        tile_size: options.tile_size,
        tile_part_packet_limit: options.tile_part_packet_limit,
        num_layers: options.quality_layers,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}

fn native_lossless_component_options(
    samples: J2kLosslessComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> EncodeOptions {
    let interleaved_shape = J2kLosslessSamples {
        data: &[],
        width: samples.width,
        height: samples.height,
        components: samples.components(),
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    };
    let mut native = native_lossless_options(interleaved_shape, options);
    native.use_mct = false;
    native
}

fn native_lossless_typed_component_options(
    samples: J2kLosslessTypedComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> EncodeOptions {
    let interleaved_shape = J2kLosslessSamples {
        data: &[],
        width: samples.width,
        height: samples.height,
        components: samples.components(),
        bit_depth: samples.max_bit_depth(),
        signed: samples.all_components_signed(),
    };
    let mut native = native_lossless_options(interleaved_shape, options);
    native.use_mct = false;
    native
}

fn native_lossy_options(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
) -> Result<EncodeOptions, J2kError> {
    let num_layers = lossy_quality_layer_count(options);
    Ok(EncodeOptions {
        reversible: false,
        num_decomposition_levels: j2k_lossy_decomposition_levels_for_options(samples, options),
        use_ht_block_coding: options.block_coding_mode == J2kBlockCodingMode::HighThroughput,
        progression_order: native_progression_order(options.progression),
        write_tlm: options.marker_segments.contains(&J2kMarkerSegment::Tlm),
        write_plt: options.marker_segments.contains(&J2kMarkerSegment::Plt),
        write_plm: options.marker_segments.contains(&J2kMarkerSegment::Plm),
        write_ppm: options.marker_segments.contains(&J2kMarkerSegment::Ppm),
        write_ppt: options.marker_segments.contains(&J2kMarkerSegment::Ppt),
        write_sop: options.marker_segments.contains(&J2kMarkerSegment::Sop),
        write_eph: options.marker_segments.contains(&J2kMarkerSegment::Eph),
        use_mct: matches!(samples.components, 3 | 4),
        num_layers,
        quality_layer_byte_targets: lossy_quality_layer_byte_targets(samples, options)?,
        tile_size: options.tile_size,
        tile_part_packet_limit: options.tile_part_packet_limit,
        precinct_exponents: options.precinct_exponents.clone(),
        validate_high_throughput_codestream: false,
        irreversible_quantization_scale: quantization_scale,
        ..EncodeOptions::default()
    })
}

fn lossy_quality_layer_byte_targets(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<Vec<u64>, J2kError> {
    if options.quality_layers.len() <= 1 {
        return Ok(Vec::new());
    }

    let mut targets = Vec::with_capacity(options.quality_layers.len());
    for layer in &options.quality_layers {
        match layer.target {
            J2kRateTarget::Bytes(bytes) => targets.push(bytes),
            J2kRateTarget::BitsPerPixel(bits_per_pixel) => {
                targets.push(target_bytes_for_bpp(samples, bits_per_pixel)?);
            }
            J2kRateTarget::PsnrDb(_) => return Ok(Vec::new()),
        }
    }
    if targets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy quality layer targets must be cumulative and monotonic",
        }));
    }
    Ok(targets)
}

pub(crate) fn native_progression_order(progression: J2kProgressionOrder) -> EncodeProgressionOrder {
    match progression {
        J2kProgressionOrder::Lrcp => EncodeProgressionOrder::Lrcp,
        J2kProgressionOrder::Rlcp => EncodeProgressionOrder::Rlcp,
        J2kProgressionOrder::Rpcl => EncodeProgressionOrder::Rpcl,
        J2kProgressionOrder::Pcrl => EncodeProgressionOrder::Pcrl,
        J2kProgressionOrder::Cprl => EncodeProgressionOrder::Cprl,
    }
}

const MIN_LOSSLESS_DWT_DIMENSION: u32 = 64;

/// Return the default lossless decomposition level policy used by the facade.
pub fn j2k_lossless_decomposition_levels(samples: J2kLosslessSamples<'_>) -> u8 {
    j2k_lossless_decomposition_levels_for_progression(samples, J2kProgressionOrder::Lrcp)
}

/// Return the default lossless decomposition level policy for a progression.
pub fn j2k_lossless_decomposition_levels_for_progression(
    samples: J2kLosslessSamples<'_>,
    progression: J2kProgressionOrder,
) -> u8 {
    if matches!(
        progression,
        J2kProgressionOrder::Rpcl | J2kProgressionOrder::Pcrl | J2kProgressionOrder::Cprl
    ) {
        return j2k_rpcl_lossless_decomposition_levels(samples);
    }

    if samples.width.min(samples.height) < MIN_LOSSLESS_DWT_DIMENSION {
        return 0;
    }

    1
}

fn j2k_lossy_decomposition_levels_for_options(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> u8 {
    let levels = if matches!(
        options.progression,
        J2kProgressionOrder::Rpcl | J2kProgressionOrder::Pcrl | J2kProgressionOrder::Cprl
    ) {
        j2k_lossy_position_progression_decomposition_levels(samples)
    } else {
        u8::from(samples.width.min(samples.height) >= MIN_LOSSLESS_DWT_DIMENSION)
    };
    options.max_decomposition_levels.map_or(levels, |max| {
        levels
            .min(max)
            .min(max_decomposition_levels(samples.width, samples.height))
    })
}

fn j2k_lossy_position_progression_decomposition_levels(samples: J2kLossySamples<'_>) -> u8 {
    j2k_rpcl_lossless_decomposition_levels(J2kLosslessSamples {
        data: samples.data,
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

/// Return the effective lossless decomposition level policy for encode options.
pub fn j2k_lossless_decomposition_levels_for_options(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> u8 {
    let levels = j2k_lossless_decomposition_levels_for_progression(samples, options.progression);
    options
        .max_decomposition_levels
        .map_or(levels, |requested| {
            if samples.width.min(samples.height) < MIN_LOSSLESS_DWT_DIMENSION {
                return 0;
            }
            requested.min(max_decomposition_levels(samples.width, samples.height))
        })
}

fn j2k_rpcl_lossless_decomposition_levels(samples: J2kLosslessSamples<'_>) -> u8 {
    let mut levels = 0u8;
    let mut width = samples.width;
    let mut height = samples.height;
    let max_levels = max_decomposition_levels(samples.width, samples.height);

    while width.min(height) > MIN_LOSSLESS_DWT_DIMENSION && levels < max_levels {
        width = width.div_ceil(2);
        height = height.div_ceil(2);
        levels += 1;
    }

    levels
}

fn max_decomposition_levels(width: u32, height: u32) -> u8 {
    let min_dim = width.min(height);
    if min_dim <= 1 {
        return 0;
    }
    min_dim.ilog2() as u8
}

#[derive(Debug, Clone, Copy)]
struct RequiredEncodeStages {
    bits: u16,
}

impl RequiredEncodeStages {
    const DEINTERLEAVE: u16 = 1 << 0;
    const FORWARD_RCT: u16 = 1 << 1;
    const FORWARD_DWT53: u16 = 1 << 2;
    const TIER1_CODE_BLOCK: u16 = 1 << 3;
    const HT_CODE_BLOCK: u16 = 1 << 4;
    const PACKETIZATION: u16 = 1 << 5;
    const QUANTIZE_SUBBAND: u16 = 1 << 6;
    const FORWARD_ICT: u16 = 1 << 7;
    const FORWARD_DWT97: u16 = 1 << 8;

    fn satisfied_by(self, dispatch: J2kEncodeDispatchReport) -> bool {
        self.missing_stage(dispatch).is_none()
    }

    fn missing_message(self, dispatch: J2kEncodeDispatchReport) -> &'static str {
        match self.missing_stage(dispatch) {
            Some("deinterleave") => {
                "requested JPEG 2000 device encode backend did not dispatch deinterleave"
            }
            Some("forward_rct") => {
                "requested JPEG 2000 device encode backend did not dispatch forward_rct"
            }
            Some("forward_ict") => {
                "requested JPEG 2000 device encode backend did not dispatch forward_ict"
            }
            Some("forward_dwt53") => {
                "requested JPEG 2000 device encode backend did not dispatch forward_dwt53"
            }
            Some("forward_dwt97") => {
                "requested JPEG 2000 device encode backend did not dispatch forward_dwt97"
            }
            Some("tier1_code_block") => {
                "requested JPEG 2000 device encode backend did not dispatch tier1_code_block"
            }
            Some("ht_code_block") => {
                "requested JPEG 2000 device encode backend did not dispatch ht_code_block"
            }
            Some("quantize_subband") => {
                "requested JPEG 2000 device encode backend did not dispatch quantize_subband"
            }
            Some("packetization") => {
                "requested JPEG 2000 device encode backend did not dispatch packetization"
            }
            _ => "requested JPEG 2000 device encode backend did not dispatch",
        }
    }

    fn missing_stage(self, dispatch: J2kEncodeDispatchReport) -> Option<&'static str> {
        if self.contains(Self::DEINTERLEAVE) && dispatch.deinterleave == 0 {
            return Some("deinterleave");
        }
        if self.contains(Self::FORWARD_RCT) && dispatch.forward_rct == 0 {
            return Some("forward_rct");
        }
        if self.contains(Self::FORWARD_ICT) && dispatch.forward_ict == 0 {
            return Some("forward_ict");
        }
        if self.contains(Self::FORWARD_DWT53) && dispatch.forward_dwt53 == 0 {
            return Some("forward_dwt53");
        }
        if self.contains(Self::FORWARD_DWT97) && dispatch.forward_dwt97 == 0 {
            return Some("forward_dwt97");
        }
        if self.contains(Self::TIER1_CODE_BLOCK) && dispatch.tier1_code_block == 0 {
            return Some("tier1_code_block");
        }
        if self.contains(Self::HT_CODE_BLOCK) && dispatch.ht_code_block == 0 {
            return Some("ht_code_block");
        }
        if self.contains(Self::QUANTIZE_SUBBAND) && dispatch.quantize_subband == 0 {
            return Some("quantize_subband");
        }
        if self.contains(Self::PACKETIZATION) && dispatch.packetization == 0 {
            return Some("packetization");
        }
        None
    }

    fn contains(self, stage: u16) -> bool {
        self.bits & stage != 0
    }
}

fn required_encode_stages(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
    accelerated_backend: BackendKind,
) -> RequiredEncodeStages {
    let decomposition_levels = j2k_lossless_decomposition_levels_for_options(samples, options);
    let high_throughput = options.block_coding_mode == J2kBlockCodingMode::HighThroughput;

    let mut bits = RequiredEncodeStages::PACKETIZATION;
    if matches!(accelerated_backend, BackendKind::Cuda | BackendKind::Metal) {
        bits |= RequiredEncodeStages::DEINTERLEAVE | RequiredEncodeStages::QUANTIZE_SUBBAND;
    }
    if matches!(samples.components, 3 | 4)
        && options.reversible_transform == ReversibleTransform::Rct53
    {
        bits |= RequiredEncodeStages::FORWARD_RCT;
    }
    if decomposition_levels > 0 {
        bits |= RequiredEncodeStages::FORWARD_DWT53;
    }
    if high_throughput {
        bits |= RequiredEncodeStages::HT_CODE_BLOCK;
    } else {
        bits |= RequiredEncodeStages::TIER1_CODE_BLOCK;
    }

    RequiredEncodeStages { bits }
}

fn required_lossy_encode_stages(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    accelerated_backend: BackendKind,
) -> RequiredEncodeStages {
    let decomposition_levels = j2k_lossy_decomposition_levels_for_options(samples, options);
    let high_throughput = options.block_coding_mode == J2kBlockCodingMode::HighThroughput;

    let scalar_packetization_required = lossy_quality_layer_count(options) > 1
        || options.marker_segments.contains(&J2kMarkerSegment::Plt)
        || options.marker_segments.contains(&J2kMarkerSegment::Plm)
        || options.marker_segments.contains(&J2kMarkerSegment::Sop)
        || options.marker_segments.contains(&J2kMarkerSegment::Eph);
    let mut bits = 0;
    if !scalar_packetization_required || accelerated_backend == BackendKind::Metal {
        bits |= RequiredEncodeStages::PACKETIZATION;
    }
    if matches!(accelerated_backend, BackendKind::Cuda | BackendKind::Metal) {
        bits |= RequiredEncodeStages::DEINTERLEAVE | RequiredEncodeStages::QUANTIZE_SUBBAND;
        if matches!(samples.components, 3 | 4) {
            bits |= RequiredEncodeStages::FORWARD_ICT;
        }
        if decomposition_levels > 0 {
            bits |= RequiredEncodeStages::FORWARD_DWT97;
        }
    }
    if high_throughput {
        bits |= RequiredEncodeStages::HT_CODE_BLOCK;
    } else {
        bits |= RequiredEncodeStages::TIER1_CODE_BLOCK;
    }

    RequiredEncodeStages { bits }
}

fn validate_lossy_options(options: &J2kLossyEncodeOptions) -> Result<(), J2kError> {
    if options.quality_layers.len() > 32 {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy encode supports 1-32 quality layers",
        }));
    }
    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width == 0 || tile_height == 0 {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossy tile dimensions must be non-zero",
            }));
        }
    }
    if options
        .precinct_exponents
        .iter()
        .any(|&(ppx, ppy)| ppx > 15 || ppy > 15)
    {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy precinct exponents must be 0-15",
        }));
    }
    if !(options.psnr_tolerance_db.is_finite() && options.psnr_tolerance_db >= 0.0) {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy PSNR tolerance must be finite and non-negative",
        }));
    }
    if options.psnr_iteration_budget == 0 {
        return Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy PSNR iteration budget must be greater than zero",
        }));
    }
    validate_rate_target(options.rate_target)?;
    for layer in &options.quality_layers {
        validate_rate_target(Some(layer.target))?;
    }
    Ok(())
}

fn effective_lossy_target(
    options: &J2kLossyEncodeOptions,
) -> Result<Option<J2kRateTarget>, J2kError> {
    match (options.rate_target, options.quality_layers.as_slice()) {
        (target, []) => Ok(target),
        (None, [layer]) => Ok(Some(layer.target)),
        (Some(target), [layer]) if target == layer.target => Ok(Some(target)),
        (Some(_), [_]) => Err(J2kError::Unsupported(Unsupported {
            what:
                "specify either a JPEG 2000 lossy rate target or one quality layer target, not both",
        })),
        (None, layers) => Ok(layers.last().map(|layer| layer.target)),
        (Some(target), layers) if layers.last().is_some_and(|layer| layer.target == target) => {
            Ok(Some(target))
        }
        (Some(_), _) => Err(J2kError::Unsupported(Unsupported {
            what: "when multiple JPEG 2000 quality layers are specified, the single rate target must match the final cumulative layer target",
        })),
    }
}

fn validate_rate_target(target: Option<J2kRateTarget>) -> Result<(), J2kError> {
    match target {
        None => Ok(()),
        Some(J2kRateTarget::BitsPerPixel(bits_per_pixel))
            if bits_per_pixel.is_finite() && bits_per_pixel > 0.0 =>
        {
            Ok(())
        }
        Some(J2kRateTarget::Bytes(bytes)) if bytes > 0 => Ok(()),
        Some(J2kRateTarget::PsnrDb(psnr_db)) if psnr_db.is_finite() && psnr_db > 0.0 => Ok(()),
        Some(J2kRateTarget::BitsPerPixel(_)) => Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy bits-per-pixel target must be finite and greater than zero",
        })),
        Some(J2kRateTarget::Bytes(_)) => Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy byte target must be greater than zero",
        })),
        Some(J2kRateTarget::PsnrDb(_)) => Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy PSNR target must be finite and greater than zero",
        })),
    }
}

fn lossy_report(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    target: Option<J2kRateTarget>,
    attempt: &LossyAttempt,
) -> Result<J2kLossyEncodeReport, J2kError> {
    let actual_bytes = attempt.codestream.len() as u64;
    Ok(J2kLossyEncodeReport {
        target,
        quality_layers: u16::from(lossy_quality_layer_count(options)),
        quantization_scale: attempt.quantization_scale,
        actual_bytes,
        actual_bits_per_pixel: bits_per_pixel(samples, actual_bytes),
        psnr_db: validate_lossy_roundtrip(samples, &attempt.codestream, options.validation)?,
        ht_rate_granularity_bytes: (options.block_coding_mode
            == J2kBlockCodingMode::HighThroughput)
            .then_some(actual_bytes),
    })
}

fn lossy_quality_layer_count(options: &J2kLossyEncodeOptions) -> u8 {
    u8::try_from(options.quality_layers.len().max(1)).unwrap_or(32)
}

fn validate_lossy_roundtrip(
    samples: J2kLossySamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<Option<f64>, J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(None);
    }

    let decoded = Image::new(codestream, &DecodeSettings::default())
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?
        .decode_native()
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;

    if decoded.width != samples.width
        || decoded.height != samples.height
        || decoded.num_components != samples.components
        || decoded.bit_depth != samples.bit_depth
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossy encode failed round-trip geometry validation".to_string(),
        });
    }

    Ok(Some(psnr_from_decoded(samples, &decoded.data)?))
}

fn decoded_psnr(samples: J2kLossySamples<'_>, codestream: &[u8]) -> Result<f64, J2kError> {
    let decoded = Image::new(codestream, &DecodeSettings::default())
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?
        .decode_native()
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;
    psnr_from_decoded(samples, &decoded.data)
}

#[allow(clippy::cast_precision_loss)]
fn psnr_from_decoded(samples: J2kLossySamples<'_>, decoded: &[u8]) -> Result<f64, J2kError> {
    if decoded.len() != samples.data.len() {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "JPEG 2000 lossy encode validation length mismatch: expected {} bytes, got {} bytes",
                samples.data.len(),
                decoded.len()
            ),
        });
    }
    let bytes_per_sample = raw_pixel_bytes_per_sample(samples.bit_depth);
    let sample_count = samples.data.len() / bytes_per_sample;
    let mut squared_error = 0.0f64;
    for sample_idx in 0..sample_count {
        let original = sample_value(samples.data, sample_idx, samples.bit_depth, samples.signed);
        let decoded = sample_value(decoded, sample_idx, samples.bit_depth, samples.signed);
        let error = original - decoded;
        squared_error += error * error;
    }
    if squared_error == 0.0 {
        return Ok(f64::INFINITY);
    }
    let mse = squared_error / usize_to_f64(sample_count);
    let peak = ((1_u64 << u32::from(samples.bit_depth)) - 1) as f64;
    Ok(10.0 * ((peak * peak) / mse).log10())
}

#[allow(clippy::cast_precision_loss)]
fn sample_value(data: &[u8], sample_idx: usize, bit_depth: u8, signed: bool) -> f64 {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth);
    let byte_idx = sample_idx * bytes_per_sample;
    let raw = read_le_sample_value(&data[byte_idx..byte_idx + bytes_per_sample], bit_depth);
    if signed {
        sign_extend_sample(raw, bit_depth) as f64
    } else {
        raw as f64
    }
}

fn raw_pixel_bytes_per_sample(bit_depth: u8) -> usize {
    usize::from(bit_depth).div_ceil(8).max(1)
}

fn read_le_sample_value(bytes: &[u8], bit_depth: u8) -> u64 {
    let mut raw = 0_u64;
    for (shift, byte) in bytes.iter().enumerate() {
        raw |= u64::from(*byte) << (shift * 8);
    }
    let mask = (1_u64 << bit_depth) - 1;
    raw & mask
}

fn sign_extend_sample(raw: u64, bit_depth: u8) -> i64 {
    let shift = 64 - u32::from(bit_depth);
    ((raw << shift) as i64) >> shift
}

fn target_bytes_for_bpp(
    samples: J2kLossySamples<'_>,
    bits_per_pixel: f64,
) -> Result<u64, J2kError> {
    let pixels = f64::from(samples.width) * f64::from(samples.height);
    let bytes = (pixels * bits_per_pixel / 8.0).ceil();
    if bytes.is_finite() && bytes > 0.0 && bytes <= 18_446_744_073_709_551_615.0 {
        Ok(bytes as u64)
    } else {
        Err(J2kError::Unsupported(Unsupported {
            what: "JPEG 2000 lossy bits-per-pixel target overflows byte target",
        }))
    }
}

fn byte_target_tolerance(target_bytes: u64) -> u64 {
    target_bytes.div_ceil(100).max(512)
}

fn byte_target_diff(actual: u64, target: u64) -> u64 {
    actual.abs_diff(target)
}

fn bits_per_pixel(samples: J2kLossySamples<'_>, bytes: u64) -> f64 {
    (u64_to_f64(bytes) * 8.0) / (f64::from(samples.width) * f64::from(samples.height))
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn u64_to_f64(value: u64) -> f64 {
    value as f64
}

fn validate_lossless_roundtrip(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }
    if samples.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        let header = j2k_native::inspect_j2k_codestream_header(codestream).map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;
        if header.dimensions != (samples.width, samples.height)
            || header.components != samples.components
            || header.bit_depth != samples.bit_depth
            || header
                .component_info
                .iter()
                .any(|component| component.signed != samples.signed)
        {
            return Err(J2kError::InvalidSamples {
                what: "JPEG 2000 high-bit lossless encode failed metadata validation".to_string(),
            });
        }
        return Ok(());
    }

    let decoded = Image::new(codestream, &DecodeSettings::default())
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?
        .decode_native()
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;

    if decoded.width != samples.width
        || decoded.height != samples.height
        || decoded.num_components != samples.components
        || decoded.bit_depth != samples.bit_depth
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless encode failed round-trip geometry validation".to_string(),
        });
    }
    if decoded.data != samples.data {
        let mismatch = decoded
            .data
            .iter()
            .zip(samples.data.iter())
            .position(|(actual, expected)| actual != expected);
        return Err(J2kError::InvalidSamples {
            what: match mismatch {
            Some(index) => format!(
                "JPEG 2000 lossless encode failed round-trip validation at byte {index}: expected {}, got {}",
                samples.data[index], decoded.data[index]
            ),
            None => format!(
                "JPEG 2000 lossless encode failed round-trip validation: expected {} bytes, got {} bytes",
                samples.data.len(),
                decoded.data.len()
            ),
        }});
    }
    Ok(())
}

fn validate_lossless_component_roundtrip(
    samples: J2kLosslessComponentSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }

    let image = Image::new(codestream, &DecodeSettings::default()).map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;
    let mut context = j2k_native::DecoderContext::default();
    let decoded = image
        .decode_components_with_context(&mut context)
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;

    if decoded.dimensions() != (samples.width, samples.height)
        || decoded.planes().len() != samples.planes.len()
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless component-plane encode failed round-trip geometry validation"
                .to_string(),
        });
    }

    for (index, (expected, actual)) in samples
        .planes
        .iter()
        .zip(decoded.planes().iter())
        .enumerate()
    {
        let expected_sampling = (expected.x_rsiz, expected.y_rsiz);
        if actual.bit_depth() != samples.bit_depth
            || actual.signed() != samples.signed
            || actual.sampling() != expected_sampling
        {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless component-plane encode failed metadata validation for component {index}"
                ),
            });
        }
        if expected_sampling == (1, 1) {
            validate_full_resolution_component_samples(samples, expected, actual.samples(), index)?;
        }
    }
    Ok(())
}

fn validate_lossless_high_bit_component_roundtrip(
    samples: J2kLosslessComponentSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }

    let image = Image::new(codestream, &DecodeSettings::default()).map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;
    let decoded = image.decode_native_components().map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;

    if decoded.dimensions() != (samples.width, samples.height)
        || decoded.planes().len() != samples.planes.len()
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless high-bit component-plane encode failed round-trip geometry validation"
                .to_string(),
        });
    }

    for (index, (expected, actual)) in samples
        .planes
        .iter()
        .zip(decoded.planes().iter())
        .enumerate()
    {
        if actual.bit_depth() != samples.bit_depth
            || actual.signed() != samples.signed
            || actual.sampling() != (expected.x_rsiz, expected.y_rsiz)
            || actual.data() != expected.data
        {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless high-bit component-plane encode failed validation for component {index}"
                ),
            });
        }
    }
    Ok(())
}

fn validate_lossless_typed_component_roundtrip(
    samples: J2kLosslessTypedComponentSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }
    if samples.max_bit_depth() > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return validate_lossless_high_bit_typed_component_roundtrip(
            samples, codestream, validation,
        );
    }

    let image = Image::new(codestream, &DecodeSettings::default()).map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;
    let mut context = j2k_native::DecoderContext::default();
    let decoded = image
        .decode_components_with_context(&mut context)
        .map_err(|err| {
            J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
        })?;

    if decoded.dimensions() != (samples.width, samples.height)
        || decoded.planes().len() != samples.planes.len()
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless typed component-plane encode failed round-trip geometry validation"
                .to_string(),
        });
    }

    for (index, (expected, actual)) in samples
        .planes
        .iter()
        .zip(decoded.planes().iter())
        .enumerate()
    {
        let expected_sampling = (expected.x_rsiz, expected.y_rsiz);
        if actual.bit_depth() != expected.bit_depth
            || actual.signed() != expected.signed
            || actual.sampling() != expected_sampling
        {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless typed component-plane encode failed metadata validation for component {index}"
                ),
            });
        }
        if expected_sampling == (1, 1) {
            validate_full_resolution_typed_component_samples(expected, actual.samples(), index)?;
        }
    }
    Ok(())
}

fn validate_lossless_high_bit_typed_component_roundtrip(
    samples: J2kLosslessTypedComponentSamples<'_>,
    codestream: &[u8],
    validation: J2kEncodeValidation,
) -> Result<(), J2kError> {
    if validation == J2kEncodeValidation::External {
        return Ok(());
    }

    let image = Image::new(codestream, &DecodeSettings::default()).map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;
    let decoded = image.decode_native_components().map_err(|err| {
        J2kError::validation_backend(format!("encoded codestream validation failed: {err}"))
    })?;

    if decoded.dimensions() != (samples.width, samples.height)
        || decoded.planes().len() != samples.planes.len()
    {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 lossless high-bit typed component-plane encode failed round-trip geometry validation"
                .to_string(),
        });
    }

    for (index, (expected, actual)) in samples
        .planes
        .iter()
        .zip(decoded.planes().iter())
        .enumerate()
    {
        let expected_data = canonical_native_typed_component_bytes_for_reference_grid(
            expected,
            samples.width,
            samples.height,
        )?;
        if actual.bit_depth() != expected.bit_depth
            || actual.signed() != expected.signed
            || actual.sampling() != (expected.x_rsiz, expected.y_rsiz)
            || actual.data() != expected_data.as_slice()
        {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless high-bit typed component-plane encode failed validation for component {index}"
                ),
            });
        }
    }
    Ok(())
}

fn canonical_native_typed_component_bytes_for_reference_grid(
    plane: &J2kLosslessTypedComponentPlane<'_>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, J2kError> {
    let component_bytes = canonical_native_typed_component_bytes(plane)?;
    if (plane.x_rsiz, plane.y_rsiz) == (1, 1) {
        return Ok(component_bytes);
    }

    let bytes_per_sample = raw_pixel_bytes_per_sample(plane.bit_depth);
    let component_width = width.div_ceil(u32::from(plane.x_rsiz)) as usize;
    let component_height = height.div_ceil(u32::from(plane.y_rsiz)) as usize;
    let output_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|sample_count| sample_count.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow { width, height })?;
    let mut out = Vec::with_capacity(output_len);

    for y in 0..height as usize {
        let component_y = (y / usize::from(plane.y_rsiz)).min(component_height.saturating_sub(1));
        for x in 0..width as usize {
            let component_x =
                (x / usize::from(plane.x_rsiz)).min(component_width.saturating_sub(1));
            let component_idx = component_y
                .checked_mul(component_width)
                .and_then(|offset| offset.checked_add(component_x))
                .ok_or(J2kError::DimensionOverflow { width, height })?;
            let start = component_idx
                .checked_mul(bytes_per_sample)
                .ok_or(J2kError::DimensionOverflow { width, height })?;
            let end = start
                .checked_add(bytes_per_sample)
                .ok_or(J2kError::DimensionOverflow { width, height })?;
            out.extend_from_slice(component_bytes.get(start..end).ok_or_else(|| {
                J2kError::InvalidSamples {
                    what: "JPEG 2000 typed component-plane canonicalization length mismatch"
                        .to_string(),
                }
            })?);
        }
    }

    Ok(out)
}

fn canonical_native_typed_component_bytes(
    plane: &J2kLosslessTypedComponentPlane<'_>,
) -> Result<Vec<u8>, J2kError> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(plane.bit_depth);
    let mut out = Vec::with_capacity(plane.data.len());
    for sample in plane.data.chunks_exact(bytes_per_sample) {
        let raw = read_le_sample_value(sample, plane.bit_depth);
        if plane.signed {
            let value = sign_extend_sample(raw, plane.bit_depth);
            if plane.bit_depth <= 8 {
                out.push((value as i8) as u8);
            } else if plane.bit_depth <= 16 {
                out.extend_from_slice(&(value as i16).to_le_bytes());
            } else {
                let bytes = value.to_le_bytes();
                out.extend_from_slice(&bytes[..bytes_per_sample]);
            }
        } else if plane.bit_depth <= 8 {
            out.push(raw as u8);
        } else if plane.bit_depth <= 16 {
            out.extend_from_slice(&(raw as u16).to_le_bytes());
        } else {
            let bytes = raw.to_le_bytes();
            out.extend_from_slice(&bytes[..bytes_per_sample]);
        }
    }
    if out.len() != plane.data.len() {
        return Err(J2kError::InvalidSamples {
            what: "JPEG 2000 typed component-plane canonicalization length mismatch".to_string(),
        });
    }
    Ok(out)
}

fn validate_full_resolution_component_samples(
    samples: J2kLosslessComponentSamples<'_>,
    expected: &J2kLosslessComponentPlane<'_>,
    actual: &[f32],
    component_index: usize,
) -> Result<(), J2kError> {
    let expected_samples = (samples.width as usize)
        .checked_mul(samples.height as usize)
        .ok_or(J2kError::DimensionOverflow {
            width: samples.width,
            height: samples.height,
        })?;
    if actual.len() < expected_samples {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "JPEG 2000 lossless component-plane encode failed validation for component {component_index}: expected {expected_samples} samples, got {}",
                actual.len()
            ),
        });
    }
    for (sample_index, actual_sample) in actual.iter().take(expected_samples).enumerate() {
        let expected_sample = sample_value(
            expected.data,
            sample_index,
            samples.bit_depth,
            samples.signed,
        );
        if (f64::from(actual_sample.round()) - expected_sample).abs() > f64::EPSILON {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless component-plane encode failed validation for component {component_index} sample {sample_index}: expected {expected_sample}, got {}",
                    actual_sample.round()
                ),
            });
        }
    }
    Ok(())
}

fn validate_full_resolution_typed_component_samples(
    expected: &J2kLosslessTypedComponentPlane<'_>,
    actual: &[f32],
    component_index: usize,
) -> Result<(), J2kError> {
    let expected_samples = expected.data.len() / raw_pixel_bytes_per_sample(expected.bit_depth);
    if actual.len() < expected_samples {
        return Err(J2kError::InvalidSamples {
            what: format!(
                "JPEG 2000 lossless typed component-plane encode failed validation for component {component_index}: expected {expected_samples} samples, got {}",
                actual.len()
            ),
        });
    }
    for (sample_index, actual_sample) in actual.iter().take(expected_samples).enumerate() {
        let expected_sample = sample_value(
            expected.data,
            sample_index,
            expected.bit_depth,
            expected.signed,
        );
        if (f64::from(actual_sample.round()) - expected_sample).abs() > f64::EPSILON {
            return Err(J2kError::InvalidSamples {
                what: format!(
                    "JPEG 2000 lossless typed component-plane encode failed validation for component {component_index} sample {sample_index}: expected {expected_sample}, got {}",
                    actual_sample.round()
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        encode_j2k_lossless, j2k_lossless_decomposition_levels_for_options,
        native_lossless_options, DecodeSettings, EncodeBackendPreference, Image,
        J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
        J2kProgressionOrder, ReversibleTransform,
    };

    fn cod_mct(codestream: &[u8]) -> u8 {
        let cod_offset = codestream
            .windows(2)
            .position(|window| window == [0xFF, 0x52])
            .expect("COD marker");
        codestream[cod_offset + 8]
    }

    #[test]
    fn lossless_encode_can_disable_component_transform() {
        let pixels: Vec<u8> = (0..4 * 4 * 3)
            .map(|value| ((value * 17) & 0xFF) as u8)
            .collect();
        let samples = J2kLosslessSamples::new(&pixels, 4, 4, 3, 8, false).unwrap();
        let encoded = encode_j2k_lossless(
            samples,
            &J2kLosslessEncodeOptions {
                block_coding_mode: J2kBlockCodingMode::Classic,
                progression: J2kProgressionOrder::Lrcp,
                max_decomposition_levels: Some(0),
                reversible_transform: ReversibleTransform::None53,
                validation: J2kEncodeValidation::CpuRoundTrip,
                ..J2kLosslessEncodeOptions::default()
            },
        )
        .unwrap();

        assert_eq!(cod_mct(&encoded.codestream), 0);
    }

    #[test]
    fn explicit_decomposition_levels_override_default_lrcp_policy() {
        let pixels = vec![0; 128 * 128];
        let samples = J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).unwrap();

        let levels = j2k_lossless_decomposition_levels_for_options(
            samples,
            J2kLosslessEncodeOptions {
                block_coding_mode: J2kBlockCodingMode::Classic,
                progression: J2kProgressionOrder::Lrcp,
                max_decomposition_levels: Some(5),
                ..J2kLosslessEncodeOptions::default()
            },
        );

        assert_eq!(levels, 5);
    }

    #[test]
    fn facade_native_options_skip_internal_ht_validation_for_external_validation() {
        let pixels = vec![0; 64 * 64];
        let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

        let external = native_lossless_options(
            samples,
            J2kLosslessEncodeOptions {
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                validation: J2kEncodeValidation::External,
                ..J2kLosslessEncodeOptions::default()
            },
        );
        let roundtrip = native_lossless_options(
            samples,
            J2kLosslessEncodeOptions {
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                validation: J2kEncodeValidation::CpuRoundTrip,
                ..J2kLosslessEncodeOptions::default()
            },
        );

        assert!(!external.validate_high_throughput_codestream);
        assert!(!roundtrip.validate_high_throughput_codestream);
    }

    #[test]
    fn lossless_facade_roundtrips_four_component_via_public_api() {
        let width: u32 = 32;
        let height: u32 = 24;
        let components: u16 = 4;

        // Deterministic 4-component (RGBA/CMYK) 8-bit input, distinct per plane.
        let mut pixels = Vec::with_capacity((width * height * u32::from(components)) as usize);
        for y in 0..height {
            for x in 0..width {
                for c in 0..u32::from(components) {
                    let value = (x.wrapping_mul(7) ^ y.wrapping_mul(13)).wrapping_add(c * 41);
                    pixels.push((value & 0xFF) as u8);
                }
            }
        }

        // MUST go through the real public constructor.
        let samples = J2kLosslessSamples::new(&pixels, width, height, components, 8, false)
            .expect("4-component samples must be accepted by the public constructor");

        // Encode via the public CPU lossless entry.
        let encoded = encode_j2k_lossless(
            samples,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::CpuOnly,
                validation: J2kEncodeValidation::CpuRoundTrip,
                ..J2kLosslessEncodeOptions::default()
            },
        )
        .expect("4-component CPU lossless encode must succeed");

        assert_eq!(encoded.components, components);

        // Decode the bytes with the native decoder and assert an exact round-trip.
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("native decode of 4-component codestream must construct")
            .decode_native()
            .expect("native decode of 4-component codestream must succeed");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.num_components, components);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(
            decoded.data, pixels,
            "4-component pixels must round-trip exactly"
        );

        // 2-component is accepted and handled as independent channels without MCT.
        let two_component = vec![0u8; (width * height * 2) as usize];
        let two_component = J2kLosslessSamples::new(&two_component, width, height, 2, 8, false)
            .expect("2-component samples must be accepted by the public constructor");
        assert_eq!(two_component.components, 2);
    }
}
