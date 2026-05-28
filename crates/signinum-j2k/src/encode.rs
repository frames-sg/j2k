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
    /// Prefer a device encoder, but fall back to CPU when unavailable.
    PreferDevice,
    /// Require a device encoder and fail if unavailable or unsupported.
    RequireDevice,
}

/// Supported JPEG 2000 progression orders for the lossless encode facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum J2kProgressionOrder {
    /// Layer-resolution-component-position progression.
    #[default]
    Lrcp,
    /// Resolution-position-component-layer progression.
    Rpcl,
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

/// Borrowed interleaved samples and image geometry for lossless encoding.
#[derive(Debug, Clone, Copy)]
pub struct J2kLosslessSamples<'a> {
    /// Interleaved sample bytes.
    pub data: &'a [u8],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Component count. The stable facade accepts 1 or 3.
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
        if !matches!(components, 1 | 3) {
            return Err(J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 lossless encode supports only grayscale or RGB samples",
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

fn native_progression_order(progression: J2kProgressionOrder) -> EncodeProgressionOrder {
    match progression {
        J2kProgressionOrder::Lrcp => EncodeProgressionOrder::Lrcp,
        J2kProgressionOrder::Rpcl => EncodeProgressionOrder::Rpcl,
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
    if progression == J2kProgressionOrder::Rpcl {
        return j2k_rpcl_lossless_decomposition_levels(samples);
    }

    if samples.width.min(samples.height) < MIN_LOSSLESS_DWT_DIMENSION {
        return 0;
    }

    1
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
    bits: u8,
}

impl RequiredEncodeStages {
    const DEINTERLEAVE: u8 = 1 << 0;
    const FORWARD_RCT: u8 = 1 << 1;
    const FORWARD_DWT53: u8 = 1 << 2;
    const TIER1_CODE_BLOCK: u8 = 1 << 3;
    const HT_CODE_BLOCK: u8 = 1 << 4;
    const PACKETIZATION: u8 = 1 << 5;
    const QUANTIZE_SUBBAND: u8 = 1 << 6;

    fn satisfied_by(self, dispatch: J2kEncodeDispatchReport) -> bool {
        self.missing_stage(dispatch).is_none()
    }

    fn missing_message(self, dispatch: J2kEncodeDispatchReport) -> &'static str {
        match self.missing_stage(dispatch) {
            Some("deinterleave") => {
                "requested JPEG 2000 lossless device encode backend did not dispatch deinterleave"
            }
            Some("forward_rct") => {
                "requested JPEG 2000 lossless device encode backend did not dispatch forward_rct"
            }
            Some("forward_dwt53") => {
                "requested JPEG 2000 lossless device encode backend did not dispatch forward_dwt53"
            }
            Some("tier1_code_block") => {
                "requested JPEG 2000 lossless device encode backend did not dispatch tier1_code_block"
            }
            Some("ht_code_block") => {
                "requested JPEG 2000 lossless device encode backend did not dispatch ht_code_block"
            }
            Some("quantize_subband") => {
                "requested JPEG 2000 lossless device encode backend did not dispatch quantize_subband"
            }
            Some("packetization") => {
                "requested JPEG 2000 lossless device encode backend did not dispatch packetization"
            }
            _ => "requested JPEG 2000 lossless device encode backend did not dispatch",
        }
    }

    fn missing_stage(self, dispatch: J2kEncodeDispatchReport) -> Option<&'static str> {
        if self.contains(Self::DEINTERLEAVE) && dispatch.deinterleave == 0 {
            return Some("deinterleave");
        }
        if self.contains(Self::FORWARD_RCT) && dispatch.forward_rct == 0 {
            return Some("forward_rct");
        }
        if self.contains(Self::FORWARD_DWT53) && dispatch.forward_dwt53 == 0 {
            return Some("forward_dwt53");
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

    fn contains(self, stage: u8) -> bool {
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
        native_lossless_options, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
        J2kLosslessSamples, J2kProgressionOrder, ReversibleTransform,
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
}
