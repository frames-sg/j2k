// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless encode orchestration and result construction.

use super::accelerator::resolve_encode_backend;
use super::contracts::{EncodedJ2k, J2kLosslessEncodeOptions, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH};
use super::cpu::{encode_cpu, encode_cpu_components, encode_cpu_typed_components};
use super::high_bit;
use super::samples::{
    J2kLosslessComponentSamples, J2kLosslessSamples, J2kLosslessTypedComponentSamples,
};
use super::validation::{
    validate_lossless_component_roundtrip, validate_lossless_roundtrip,
    validate_lossless_typed_component_roundtrip,
};
use crate::{J2kEncodeDispatchReport, J2kError};

pub(super) fn encode(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    high_bit::validate_lossless_options(samples, options)?;
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

pub(super) fn encode_components(
    samples: J2kLosslessComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    if samples.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return high_bit::encode_components(samples, options);
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

pub(super) fn encode_typed_components(
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
