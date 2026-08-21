// SPDX-License-Identifier: MIT OR Apache-2.0

//! Facade decomposition-level policy.

use super::{
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kLossyEncodeOptions, J2kLossySamples,
    J2kProgressionOrder,
};
use j2k_types::encode_geometry::{
    lossless_decomposition_levels as shared_lossless_decomposition_levels,
    maximum_decomposition_levels,
};

/// Return the default lossless decomposition level policy used by the facade.
pub fn j2k_lossless_decomposition_levels(samples: J2kLosslessSamples<'_>) -> u8 {
    j2k_lossless_decomposition_levels_for_progression(samples, J2kProgressionOrder::Lrcp)
}

/// Return the default lossless decomposition level policy for a progression.
pub fn j2k_lossless_decomposition_levels_for_progression(
    samples: J2kLosslessSamples<'_>,
    progression: J2kProgressionOrder,
) -> u8 {
    shared_lossless_decomposition_levels(
        samples.width,
        samples.height,
        progression.packetization_order(),
        None,
    )
}

pub(super) fn j2k_lossy_decomposition_levels_for_options(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> u8 {
    let levels = shared_lossless_decomposition_levels(
        samples.width,
        samples.height,
        options.progression.packetization_order(),
        None,
    );
    options.max_decomposition_levels.map_or(levels, |max| {
        levels
            .min(max)
            .min(maximum_decomposition_levels(samples.width, samples.height))
    })
}

/// Return the effective lossless decomposition level policy for encode options.
pub fn j2k_lossless_decomposition_levels_for_options(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> u8 {
    j2k_lossless_decomposition_levels_for_resident_geometry(samples.width, samples.height, options)
}

pub(super) fn j2k_lossless_decomposition_levels_for_resident_geometry(
    width: u32,
    height: u32,
    options: J2kLosslessEncodeOptions,
) -> u8 {
    shared_lossless_decomposition_levels(
        width,
        height,
        options.progression.packetization_order(),
        options.max_decomposition_levels,
    )
}
