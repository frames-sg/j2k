// SPDX-License-Identifier: MIT OR Apache-2.0

//! Baseline sequential scan decoder. Focused submodules own generic,
//! DCT-plane, fast 4:2:0, and fast 4:4:4 execution paths.

mod dct;
mod deposit;
mod emit;
mod fast420;
mod generic;
mod layout;
mod output_scratch;
mod plan;
mod profile;
mod restart;
mod rgb444;
mod stripe;

pub(crate) use self::dct::{
    decode_scan_dct_blocks, DecodedDctBlocks, SequentialDctLifecycleMetadata,
};
#[cfg(feature = "bench-internals")]
pub(crate) use self::fast420::decode_scan_fast_tile_rgb_profiled;
pub(crate) use self::fast420::{
    decode_scan_fast_tile_rgb, decode_scan_fast_tile_rgb_region,
    decode_scan_fast_tile_rgb_region_scaled, FastTileRegionScaledRequest,
};
pub(crate) use self::generic::{decode_scan_baseline, decode_scan_baseline_rgb};
pub(crate) use self::layout::{fast_tile_region_first_decode_mcu, stripe_region_layout};
use self::layout::{is_ycbcr_420, scaled_dimensions, Fast420RegionLayout};
use self::output_scratch::{OutputScratch, RgbOutputScratch};
pub(crate) use self::plan::{
    PreparedComponentPlan, PreparedDecodePlan, ResolvedPreparedComponentPlan,
};
pub(crate) use self::restart::finish_scan;
pub(crate) use self::rgb444::decode_scan_fast_rgb_444;
use self::stripe::StripePlane;
pub(crate) use self::stripe::{StripeBuffer, StripeLayout};

#[cfg(test)]
use self::deposit::{
    deposit_block, deposit_dc_block, FastTile420Components, FastTile420DcState,
    FastTile420EntropyState, FastTile420Window,
};
#[cfg(test)]
use self::emit::{component_row_triplet, emit_stripe_rgb_444, should_use_direct_420_crop};
#[cfg(test)]
use self::fast420::decode_mcu_row_fast_tile_420;
#[cfg(test)]
use self::layout::{fast420_decode_mcu_row_end, fast420_first_decode_mcu_row};
#[cfg(test)]
use self::profile::NoopFast420Profiler;
#[cfg(test)]
use crate::backend::Backend;
#[cfg(all(test, feature = "bench-internals"))]
use crate::bench_support::BenchFast420Profile;
#[cfg(test)]
use crate::entropy::block::CoefficientBlock;
#[cfg(test)]
use crate::info::{DownscaleFactor, Rect};
#[cfg(test)]
use crate::internal::bit_reader::BitReader;
#[cfg(test)]
use crate::internal::scratch::ScratchPool;

#[cfg(test)]
mod tests;
