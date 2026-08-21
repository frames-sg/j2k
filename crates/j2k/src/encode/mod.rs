// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 encode facade and focused orchestration owners.

mod accelerator;
mod allocation;
mod api;
mod contracts;
mod cpu;
mod geometry;
mod high_bit;
mod lossless;
mod lossy;
mod resident;
mod roi;
mod samples;
mod validation;

pub use self::api::{
    encode_j2k_lossless, encode_j2k_lossless_components, encode_j2k_lossless_typed_components,
    encode_j2k_lossless_with_accelerator, encode_j2k_lossless_with_roi_regions, encode_j2k_lossy,
    encode_j2k_lossy_with_accelerator, encode_j2k_lossy_with_roi_regions,
};
pub use self::contracts::{
    EncodeBackendPreference, EncodedJ2k, EncodedLossyJ2k, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLossyEncodeOptions, J2kLossyEncodeReport, J2kMarkerSegment,
    J2kProgressionOrder, J2kQualityLayer, J2kRateTarget, ReversibleTransform,
};
pub use self::geometry::{
    j2k_lossless_decomposition_levels, j2k_lossless_decomposition_levels_for_options,
    j2k_lossless_decomposition_levels_for_progression,
};
pub use self::samples::{
    J2kLosslessComponentPlane, J2kLosslessComponentSamples, J2kLosslessSamples,
    J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples, J2kLossySamples,
    J2kRoiRegion,
};

pub(crate) use self::cpu::native_progression_order;
#[doc(hidden)]
pub use self::resident::encode_j2k_lossless_resident_with_accelerator;

#[cfg(test)]
use self::cpu::native_lossless_options;
#[cfg(test)]
use j2k_native::{DecodeSettings, Image};

#[cfg(test)]
mod tests;
