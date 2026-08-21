// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stable facade encode entry points.

use j2k_core::BackendKind;

use super::{
    accelerator, lossless, lossy, roi, EncodedJ2k, EncodedLossyJ2k, J2kLosslessComponentSamples,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kLosslessTypedComponentSamples,
    J2kLossyEncodeOptions, J2kLossySamples, J2kRoiRegion,
};
use crate::{J2kEncodeStageAccelerator, J2kError};

/// Encode interleaved samples into a raw JPEG 2000 lossless codestream.
pub fn encode_j2k_lossless(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    lossless::encode(samples, options)
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
    roi::encode_lossless(samples, options, roi_regions)
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
    lossless::encode_components(samples, options)
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
    lossless::encode_typed_components(samples, options)
}

/// Encode interleaved samples with an optional device encode-stage accelerator.
///
/// Accelerators return CPU fallback by reporting no dispatch. `Auto` accepts
/// that fallback and routes CPU-only sample precisions directly to the native
/// encoder; `RequireDevice` requires every stage needed by the request. Any
/// accelerator execution error or codestream validation error is returned to
/// the caller without a second encode attempt.
pub fn encode_j2k_lossless_with_accelerator(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
    accelerated_backend: BackendKind,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<EncodedJ2k, J2kError> {
    accelerator::encode_lossless(samples, options, accelerated_backend, accelerator)
}

/// Encode interleaved samples into a raw JPEG 2000 lossy codestream.
pub fn encode_j2k_lossy(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<EncodedLossyJ2k, J2kError> {
    lossy::encode(samples, options)
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
    roi::encode_lossy(samples, options, roi_regions)
}

/// Encode interleaved lossy samples with an optional device encode-stage accelerator.
pub fn encode_j2k_lossy_with_accelerator(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    accelerated_backend: BackendKind,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<EncodedLossyJ2k, J2kError> {
    accelerator::encode_lossy(samples, options, accelerated_backend, accelerator)
}
