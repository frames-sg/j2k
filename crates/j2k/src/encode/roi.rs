// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless and lossy ROI encode orchestration.

use alloc::{string::ToString, vec::Vec};
use j2k_native::EncodeRoiRegion as NativeEncodeRoiRegion;

use super::accelerator::resolve_encode_backend;
use super::allocation::try_collect_results_exact;
use super::contracts::{
    EncodedJ2k, EncodedLossyJ2k, J2kLosslessEncodeOptions, J2kLossyEncodeOptions,
};
use super::cpu::encode_cpu_with_roi_regions;
use super::high_bit;
use super::lossy::{
    effective_lossy_target, encode_cpu_lossy_with_roi_regions, encode_lossy_targeted, lossy_report,
    validate_lossy_options,
};
use super::samples::{J2kLosslessSamples, J2kLossySamples, J2kRoiRegion};
use super::validation::validate_lossless_roundtrip;
use crate::{J2kEncodeDispatchReport, J2kError};

pub(super) fn native_roi_regions_for_lossless_samples(
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

pub(super) fn native_roi_regions_for_samples(
    width: u32,
    height: u32,
    components: u16,
    roi_regions: &[J2kRoiRegion],
) -> Result<Vec<NativeEncodeRoiRegion>, J2kError> {
    try_collect_results_exact(
        roi_regions.iter().map(|region| {
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
        }),
        "native ROI descriptors",
    )
}

pub(super) fn encode_lossless(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
    roi_regions: &[J2kRoiRegion],
) -> Result<EncodedJ2k, J2kError> {
    high_bit::validate_lossless_options(samples, options)?;
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

pub(super) fn encode_lossy(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    roi_regions: &[J2kRoiRegion],
) -> Result<EncodedLossyJ2k, J2kError> {
    validate_lossy_options(options)?;
    high_bit::validate_lossy_options(samples, options)?;
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
