// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained HTJ2K and classic color-plan construction.

mod classic;
mod ht;

use j2k_native::{J2kDirectGrayscalePlan, J2kWaveletTransform};

type ReferencedTileColorGeometry<'a> = (
    (u32, u32),
    [u8; 4],
    bool,
    J2kWaveletTransform,
    &'a [J2kDirectGrayscalePlan],
);

pub(in crate::decoder) use self::{
    classic::build_cuda_classic_color_plans_from_referenced_with_profile,
    ht::build_cuda_htj2k_color_plans_from_referenced_with_profile,
};
