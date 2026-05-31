// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;

use signinum_core::{BackendKind, Unsupported};
use signinum_j2k_native::{
    DecodeSettings, EncodeOptions, EncodeProgressionOrder, Image, J2kEncodeDispatchReport,
    J2kEncodeStageAccelerator,
};

use crate::J2kError;

/// Backend preference for JPEG 2000 lossless encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EncodeBackendPreference {
    /// Pick the fastest safe backend exposed by the caller, falling back to CPU.
    #[default]
    Auto,
    /// Require the pure Rust CPU encoder.
    CpuOnly,
    /// Legacy name for adaptive accelerated routing.
    ///
    /// Prefer [`EncodeBackendPreference::ACCELERATED`] or
    /// [`J2kLosslessEncodeOptions::with_accelerated_backend`].
    PreferDevice,
    /// Require a device encoder and fail if unavailable or unsupported.
    RequireDevice,
}

impl EncodeBackendPreference {
    /// Adaptive accelerated route: CPU-only stages stay on CPU and device-shaped
    /// stages run on Metal/CUDA only after benchmark gates approve that shape.
    pub const ACCELERATED: Self = Self::Auto;
    /// Explicit portable CPU route.
    pub const CPU_ONLY: Self = Self::CpuOnly;
    /// Strict device diagnostic/conformance route.
    pub const STRICT_DEVICE: Self = Self::RequireDevice;
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
        self.with_backend(EncodeBackendPreference::ACCELERATED)
    }

    /// Return options using the portable CPU route.
    #[must_use]
    pub const fn with_cpu_only_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::CPU_ONLY)
    }

    /// Return options requiring a strict device route.
    #[must_use]
    pub const fn with_strict_device_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::STRICT_DEVICE)
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
        self.with_backend(EncodeBackendPreference::ACCELERATED)
    }

    /// Return options using the portable CPU route.
    #[must_use]
    pub fn with_cpu_only_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::CPU_ONLY)
    }

    /// Return options requiring a strict device route.
    #[must_use]
    pub fn with_strict_device_backend(self) -> Self {
        self.with_backend(EncodeBackendPreference::STRICT_DEVICE)
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
    /// Component count. The stable facade accepts 1-4 independent component
    /// samples. Two-component output is written without MCT.
    pub components: u8,
    /// Significant bits per component sample.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
}

impl<'a> J2kLosslessSamples<'a> {
    /// Validate and construct a sample descriptor.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kError> {
        if width == 0 || height == 0 {
            return Err(J2kError::Backend("invalid dimensions".to_string()));
        }
        if !(1..=4).contains(&components) {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless encode supports 1-4 component samples",
            }));
        }
        if bit_depth == 0 || bit_depth > 16 {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless encode supports 1-16 bits per sample",
            }));
        }
        let bytes_per_sample = if bit_depth <= 8 { 1usize } else { 2usize };
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(components as usize))
            .and_then(|samples| samples.checked_mul(bytes_per_sample))
            .ok_or(J2kError::DimensionOverflow { width, height })?;
        if data.len() != expected {
            return Err(J2kError::Backend(format!(
                "pixel data too short: expected {expected} bytes, got {}",
                data.len()
            )));
        }
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

/// Borrowed interleaved samples and image geometry for lossy encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLossySamples<'a> {
    /// Interleaved sample bytes.
    pub data: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Component count. The stable facade accepts 1-4 component samples.
    pub components: u8,
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
        components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kError> {
        if width == 0 || height == 0 {
            return Err(J2kError::Backend("invalid dimensions".to_string()));
        }
        if !(1..=4).contains(&components) {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossy encode supports 1-4 component samples",
            }));
        }
        if bit_depth == 0 || bit_depth > 16 {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossy encode supports 1-16 bits per sample; 17-38 bit encode is not supported",
            }));
        }
        let bytes_per_sample = if bit_depth <= 8 { 1usize } else { 2usize };
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(components as usize))
            .and_then(|samples| samples.checked_mul(bytes_per_sample))
            .ok_or(J2kError::DimensionOverflow { width, height })?;
        if data.len() != expected {
            return Err(J2kError::Backend(format!(
                "pixel data too short: expected {expected} bytes, got {}",
                data.len()
            )));
        }
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

/// Encoded JPEG 2000 lossless codestream and encode metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedJ2k {
    /// Raw JPEG 2000 codestream bytes.
    pub codestream: Vec<u8>,
    /// Backend that satisfied the encode contract.
    pub backend: BackendKind,
    /// Encoded image width in pixels.
    pub width: u32,
    /// Encoded image height in pixels.
    pub height: u32,
    /// Encoded component count.
    pub components: u8,
    /// Encoded significant bits per sample.
    pub bit_depth: u8,
    /// Whether encoded samples are signed.
    pub signed: bool,
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

/// Encoded JPEG 2000 lossy codestream and encode metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedLossyJ2k {
    /// Raw JPEG 2000 codestream bytes.
    pub codestream: Vec<u8>,
    /// Backend that satisfied the encode contract.
    pub backend: BackendKind,
    /// Encoded image width in pixels.
    pub width: u32,
    /// Encoded image height in pixels.
    pub height: u32,
    /// Encoded component count.
    pub components: u8,
    /// Encoded significant bits per sample.
    pub bit_depth: u8,
    /// Whether encoded samples are signed.
    pub signed: bool,
    /// Lossy encode metrics.
    pub report: J2kLossyEncodeReport,
}

/// Encode interleaved samples into a raw JPEG 2000 lossless codestream.
pub fn encode_j2k_lossless(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu(samples, *options)?;
    validate_lossless_roundtrip(samples, &codestream, options.validation)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

/// Encode interleaved samples with an optional device encode-stage accelerator.
///
/// Accelerators return CPU fallback by reporting no dispatch. `Auto` and
/// `PreferDevice` accept that fallback; `RequireDevice` requires at least one
/// dispatch. Any accelerator error or codestream validation error is returned to
/// the caller.
pub fn encode_j2k_lossless_with_accelerator(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
    accelerated_backend: BackendKind,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<EncodedJ2k, J2kError> {
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
    let target = effective_lossy_target(options)?;
    let attempt = encode_lossy_targeted(samples, options, target, |scale| {
        encode_cpu_lossy(samples, options, scale)
    })?;
    let report = lossy_report(samples, options, target, &attempt)?;
    Ok(EncodedLossyJ2k {
        codestream: attempt.codestream,
        backend: resolve_encode_backend(options.backend)?,
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
        EncodeBackendPreference::Auto
        | EncodeBackendPreference::CpuOnly
        | EncodeBackendPreference::PreferDevice => Ok(BackendKind::Cpu),
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
        EncodeBackendPreference::Auto
        | EncodeBackendPreference::CpuOnly
        | EncodeBackendPreference::PreferDevice => Ok(BackendKind::Cpu),
    }
}

fn encode_cpu(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossless_options(samples, options);
    signinum_j2k_native::encode(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
    )
    .map_err(|err| J2kError::Backend(format!("JPEG 2000 lossless encode failed: {err}")))
}

fn encode_with_native_accelerator(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossless_options(samples, options);
    signinum_j2k_native::encode_with_accelerator(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        accelerator,
    )
    .map_err(|err| J2kError::Backend(format!("JPEG 2000 lossless encode failed: {err}")))
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
    signinum_j2k_native::encode(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
    )
    .map_err(|err| J2kError::Backend(format!("JPEG 2000 lossy encode failed: {err}")))
}

fn encode_lossy_with_native_accelerator(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossy_options(samples, options, quantization_scale)?;
    signinum_j2k_native::encode_with_accelerator(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        accelerator,
    )
    .map_err(|err| J2kError::Backend(format!("JPEG 2000 lossy encode failed: {err}")))
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
        return Err(J2kError::Backend(format!(
            "JPEG 2000 lossy rate target unreachable: target {target_bytes} bytes, best {} bytes",
            best.codestream.len()
        )));
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
        return Err(J2kError::Backend(format!(
            "JPEG 2000 lossy PSNR target unreachable: target {target_psnr_db:.3} dB, best {best_psnr:.3} dB"
        )));
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
        write_tlm: options.progression == J2kProgressionOrder::Rpcl,
        use_mct: options.reversible_transform == ReversibleTransform::Rct53,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
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
        write_sop: options.marker_segments.contains(&J2kMarkerSegment::Sop),
        write_eph: options.marker_segments.contains(&J2kMarkerSegment::Eph),
        use_mct: samples.components >= 3,
        num_layers,
        quality_layer_byte_targets: lossy_quality_layer_byte_targets(samples, options)?,
        tile_size: options.tile_size,
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

fn native_progression_order(progression: J2kProgressionOrder) -> EncodeProgressionOrder {
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
    if accelerated_backend == BackendKind::Cuda {
        bits |= RequiredEncodeStages::DEINTERLEAVE | RequiredEncodeStages::QUANTIZE_SUBBAND;
    }
    if samples.components >= 3 && options.reversible_transform == ReversibleTransform::Rct53 {
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
    if !scalar_packetization_required {
        bits |= RequiredEncodeStages::PACKETIZATION;
    }
    if accelerated_backend == BackendKind::Cuda {
        bits |= RequiredEncodeStages::DEINTERLEAVE | RequiredEncodeStages::QUANTIZE_SUBBAND;
        if samples.components >= 3 {
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
        .map_err(|err| J2kError::Backend(format!("encoded codestream validation failed: {err}")))?
        .decode_native()
        .map_err(|err| J2kError::Backend(format!("encoded codestream validation failed: {err}")))?;

    if decoded.width != samples.width
        || decoded.height != samples.height
        || decoded.num_components != samples.components
        || decoded.bit_depth != samples.bit_depth
    {
        return Err(J2kError::Backend(
            "JPEG 2000 lossy encode failed round-trip geometry validation".to_string(),
        ));
    }

    Ok(Some(psnr_from_decoded(samples, &decoded.data)?))
}

fn decoded_psnr(samples: J2kLossySamples<'_>, codestream: &[u8]) -> Result<f64, J2kError> {
    let decoded = Image::new(codestream, &DecodeSettings::default())
        .map_err(|err| J2kError::Backend(format!("encoded codestream validation failed: {err}")))?
        .decode_native()
        .map_err(|err| J2kError::Backend(format!("encoded codestream validation failed: {err}")))?;
    psnr_from_decoded(samples, &decoded.data)
}

fn psnr_from_decoded(samples: J2kLossySamples<'_>, decoded: &[u8]) -> Result<f64, J2kError> {
    if decoded.len() != samples.data.len() {
        return Err(J2kError::Backend(format!(
            "JPEG 2000 lossy encode validation length mismatch: expected {} bytes, got {} bytes",
            samples.data.len(),
            decoded.len()
        )));
    }
    let bytes_per_sample = if samples.bit_depth <= 8 {
        1usize
    } else {
        2usize
    };
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
    let peak = f64::from((1u32 << u32::from(samples.bit_depth)) - 1);
    Ok(10.0 * ((peak * peak) / mse).log10())
}

fn sample_value(data: &[u8], sample_idx: usize, bit_depth: u8, signed: bool) -> f64 {
    if bit_depth <= 8 {
        if signed {
            f64::from(data[sample_idx] as i8)
        } else {
            f64::from(data[sample_idx])
        }
    } else {
        let byte_idx = sample_idx * 2;
        let bytes = [data[byte_idx], data[byte_idx + 1]];
        if signed {
            f64::from(i16::from_le_bytes(bytes))
        } else {
            f64::from(u16::from_le_bytes(bytes))
        }
    }
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

    let decoded = Image::new(codestream, &DecodeSettings::default())
        .map_err(|err| J2kError::Backend(format!("encoded codestream validation failed: {err}")))?
        .decode_native()
        .map_err(|err| J2kError::Backend(format!("encoded codestream validation failed: {err}")))?;

    if decoded.width != samples.width
        || decoded.height != samples.height
        || decoded.num_components != samples.components
        || decoded.bit_depth != samples.bit_depth
    {
        return Err(J2kError::Backend(
            "JPEG 2000 lossless encode failed round-trip geometry validation".to_string(),
        ));
    }
    if decoded.data != samples.data {
        let mismatch = decoded
            .data
            .iter()
            .zip(samples.data.iter())
            .position(|(actual, expected)| actual != expected);
        return Err(J2kError::Backend(match mismatch {
            Some(index) => format!(
                "JPEG 2000 lossless encode failed round-trip validation at byte {index}: expected {}, got {}",
                samples.data[index], decoded.data[index]
            ),
            None => format!(
                "JPEG 2000 lossless encode failed round-trip validation: expected {} bytes, got {} bytes",
                samples.data.len(),
                decoded.data.len()
            ),
        }));
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
        let components: u8 = 4;

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
