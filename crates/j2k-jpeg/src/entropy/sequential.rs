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
    decode_scan_dct_blocks, decode_scan_dct_blocks_into, decode_scan_dct_window_into,
    sequential_mcu_grid, sequential_mcu_size, DecodedDctBlocks, McuWindow,
    SequentialDctLifecycleMetadata,
};
#[cfg(feature = "bench-internals")]
pub(crate) use self::fast420::decode_scan_fast_tile_rgb_profiled;
pub(crate) use self::generic::{decode_scan_baseline, decode_scan_baseline_rgb};
pub(crate) use self::layout::stripe_region_layout;
use self::layout::{scaled_dimensions, uses_fancy_420_emit, CropWindow, McuColumns};
use self::output_scratch::{OutputScratch, RgbOutputScratch};
pub(crate) use self::plan::{
    HuffmanPrimaryWidth, PreparedComponentPlan, PreparedDecodePlan, ResolvedPreparedComponentPlan,
};
pub(crate) use self::restart::{
    finish_scan, restart_checkpoint_from_index, skip_mcu, McuPosition, McuSkipper, RestartState,
};
use self::stripe::StripePlane;
pub(crate) use self::stripe::{StripeBuffer, StripeLayout};
use crate::adapter::JpegFastPacketFamily;
use crate::backend::Backend;
use crate::error::{JpegError, Warning};
use crate::info::{DownscaleFactor, Rect};
use crate::internal::checkpoint::DeviceCheckpoint;
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

#[cfg(test)]
use self::deposit::{
    deposit_block, deposit_dc_block, FastRowState, FastTile420Components, FastTile420Window,
};
#[cfg(test)]
use self::emit::{component_row_triplet, FastRgbRowsEmitter};
#[cfg(test)]
use self::fast420::{decode_mcu_row_fast_tile_420, FAST420_HALO_ROWS};
#[cfg(test)]
use self::layout::{fast420_decode_mcu_row_end, fast420_first_decode_mcu_row};
#[cfg(test)]
use self::profile::NoopFast420Profiler;
#[cfg(all(test, feature = "bench-internals"))]
use crate::bench_support::BenchFast420Profile;
#[cfg(test)]
use crate::entropy::block::CoefficientBlock;
#[cfg(test)]
use crate::internal::bit_reader::BitReader;

/// The part of a full-size scan a fast RGB route writes (the writer's row 0
/// is `roi.y`), and an optional entropy checkpoint to start nearer to it.
#[derive(Clone, Copy)]
pub(crate) struct FastRgbRequest<'a> {
    pub(crate) roi: Rect,
    pub(crate) checkpoint: Option<&'a DeviceCheckpoint>,
}

impl FastRgbRequest<'_> {
    /// The MCU columns of `plan`'s scan a decode of `roi` reads, and the
    /// columns of them it writes.
    fn columns(self, plan: &PreparedDecodePlan) -> (McuColumns, CropWindow) {
        let region = stripe_region_layout(plan, DownscaleFactor::Full, self.roi);
        (
            McuColumns::for_region(sequential_mcu_grid(plan), region),
            CropWindow::new(region, self.roi),
        )
    }
}

/// The streaming CPU routes that decode a full-size sequential scan of a
/// fast shape (see [`PreparedDecodePlan::fast_shape_family`]) to interleaved
/// RGB: fancy-upsampled 4:2:0 streams through a two-row halo; 4:2:2 and
/// 4:4:4 (YCbCr or identity RGB) need no vertical upsampling context.
impl JpegFastPacketFamily {
    /// First MCU row a decode of `roi` reads: fancy 4:2:0 also needs the
    /// chroma row above an MCU-aligned top edge.
    pub(crate) fn first_decode_row(self, plan: &PreparedDecodePlan, roi: Rect) -> u32 {
        let (_, mcu_height_px) = sequential_mcu_size(plan);
        match self {
            Self::Fast420 => layout::fast420_first_decode_mcu_row(roi, mcu_height_px),
            Self::Fast422 | Self::Fast444 => layout::first_mcu_row_for_rect(roi, mcu_height_px),
        }
    }

    /// First MCU in scan order a decode of `roi` reads, where an entropy
    /// checkpoint helps most.
    pub(crate) fn first_decode_mcu(self, plan: &PreparedDecodePlan, roi: Rect) -> u32 {
        self.first_decode_row(plan, roi) * sequential_mcu_grid(plan).0
    }

    /// Decode `request.roi` of the scan into `writer`.
    pub(crate) fn decode<W: OutputWriter + InterleavedRgbWriter>(
        self,
        plan: &PreparedDecodePlan,
        backend: Backend,
        scan_bytes: &[u8],
        pool: &mut ScratchPool,
        writer: &mut W,
        request: FastRgbRequest<'_>,
    ) -> Result<Vec<Warning>, JpegError> {
        match self {
            Self::Fast420 => {
                fast420::decode_scan_fast_tile_rgb(plan, backend, scan_bytes, pool, writer, request)
            }
            Self::Fast422 => rgb444::decode_scan_fast_rgb_rows::<2, W>(
                plan, backend, scan_bytes, pool, writer, request,
            ),
            Self::Fast444 => rgb444::decode_scan_fast_rgb_rows::<1, W>(
                plan, backend, scan_bytes, pool, writer, request,
            ),
        }
    }
}

#[cfg(test)]
pub(crate) mod stream_tests;
#[cfg(test)]
mod tests;
