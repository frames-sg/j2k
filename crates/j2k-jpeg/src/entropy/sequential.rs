// SPDX-License-Identifier: MIT OR Apache-2.0

//! Baseline sequential scan decoder. Focused submodules own generic,
//! DCT-plane, fast 4:2:0, and fast 4:4:4 execution paths.

use crate::entropy::huffman::HuffmanTable;
use crate::info::{ColorSpace, SamplingFactors};
use crate::internal::scratch::{RgbGenericRows, YCbCr420Rows, YCbCrGenericRows};
use alloc::sync::Arc;
use alloc::vec::Vec;

mod dct;
mod deposit;
mod emit;
mod fast420;
mod generic;
mod layout;
mod profile;
mod restart;
mod rgb444;

pub(crate) use self::dct::{decode_scan_dct_blocks, DecodedDctBlocks};
#[cfg(feature = "bench-internals")]
pub(crate) use self::fast420::decode_scan_fast_tile_rgb_profiled;
pub(crate) use self::fast420::{
    decode_scan_fast_tile_rgb, decode_scan_fast_tile_rgb_region,
    decode_scan_fast_tile_rgb_region_scaled, FastTileRegionScaledRequest,
};
pub(crate) use self::generic::{decode_scan_baseline, decode_scan_baseline_rgb};
pub(crate) use self::layout::{fast_tile_region_first_decode_mcu, stripe_region_layout};
use self::layout::{is_ycbcr_420, scaled_dimensions, Fast420RegionLayout};
pub(crate) use self::restart::finish_scan;
pub(crate) use self::rgb444::decode_scan_fast_rgb_444;

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

/// Per-component decode context. One entry per component declared in the
/// SOF, in scan order.
#[derive(Debug, Clone)]
pub(crate) struct PreparedComponentPlan {
    pub(crate) h: u8,
    pub(crate) v: u8,
    pub(crate) output_index: usize,
    pub(crate) quant: Arc<[u16; 64]>,
    pub(crate) dc_table: Arc<HuffmanTable>,
    pub(crate) ac_table: Arc<HuffmanTable>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedDecodePlan {
    pub(crate) components: Vec<PreparedComponentPlan>,
    pub(crate) sampling: SamplingFactors,
    pub(crate) color_space: ColorSpace,
    pub(crate) restart_interval: Option<u16>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) scan_offset: usize,
    pub(crate) scratch_bytes: usize,
}

impl PreparedDecodePlan {
    pub(crate) fn matches_fast_tile_shape(&self) -> bool {
        self.restart_interval.is_none()
            && is_ycbcr_420(self)
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 2
            && self.components[0].v == 2
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }

    pub(crate) fn matches_fast_rgb444_shape(&self) -> bool {
        self.color_space == ColorSpace::YCbCr
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 1
            && self.components[0].v == 1
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }

    pub(crate) fn matches_fast_rgb422_shape(&self) -> bool {
        self.color_space == ColorSpace::YCbCr
            && self.components.len() == 3
            && self.components[0].output_index == 0
            && self.components[0].h == 2
            && self.components[0].v == 1
            && self.components[1].output_index == 1
            && self.components[1].h == 1
            && self.components[1].v == 1
            && self.components[2].output_index == 2
            && self.components[2].h == 1
            && self.components[2].v == 1
    }
}

enum OutputScratch<'a> {
    Grayscale,
    YCbCr420(&'a mut YCbCr420Rows),
    YCbCrGeneric(&'a mut YCbCrGenericRows),
    RgbGeneric(&'a mut RgbGenericRows),
}

enum RgbOutputScratch<'a> {
    None,
    YCbCr420,
    YCbCrGeneric(&'a mut YCbCrGenericRows),
    RgbGeneric(&'a mut RgbGenericRows),
}

#[derive(Debug, Default)]
pub(crate) struct StripeBuffer {
    pub(crate) planes: Vec<Vec<u8>>,
    pub(crate) plane_strides: Vec<usize>,
    pub(crate) plane_rows: Vec<usize>,
}

#[derive(Clone, Copy)]
struct StripePlane<'a> {
    data: &'a [u8],
    stride: usize,
    rows: usize,
}

impl StripeBuffer {
    /// Grow each plane's backing Vec to the size required by `plan` and
    /// `mcus_per_row`. Never shrinks the allocation — a monotonic
    /// tile-batch workload pays the allocation cost exactly once.
    pub(crate) fn resize_for(
        &mut self,
        plan: &PreparedDecodePlan,
        mcus_per_row: u32,
        block_size: u32,
    ) {
        let n = plan.sampling.len();
        self.planes.resize_with(n, Vec::new);
        self.plane_strides.resize(n, 0);
        self.plane_rows.resize(n, 0);
        for (i, (h, v)) in plan.sampling.iter().enumerate() {
            let cols = (mcus_per_row as usize) * (h as usize) * (block_size as usize);
            let rows = (v as usize) * (block_size as usize);
            let bytes = cols * rows;
            if self.planes[i].len() < bytes {
                self.planes[i].resize(bytes, 0);
            }
            self.plane_strides[i] = cols;
            self.plane_rows[i] = rows;
        }
    }

    fn row_count(&self, plane_idx: usize) -> usize {
        self.plane_rows[plane_idx]
    }

    fn row(&self, plane_idx: usize, row: usize) -> &[u8] {
        let stride = self.plane_strides[plane_idx];
        let start = row * stride;
        &self.planes[plane_idx][start..start + stride]
    }

    fn plane(&self, plane_idx: usize) -> StripePlane<'_> {
        StripePlane {
            data: &self.planes[plane_idx],
            stride: self.plane_strides[plane_idx],
            rows: self.plane_rows[plane_idx],
        }
    }
}

#[cfg(test)]
mod tests;
