// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared geometry, restart seek, and rolling-stripe scan orchestration.

use super::super::emit::StripeEmit;
use super::super::layout::{
    decode_mcu_row_end_for_rect, expanded_output_rect, fast420_decode_mcu_row_end,
    fast420_first_decode_mcu_row, first_decode_mcu_row_for_rect, is_ycbcr_420,
    last_mcu_row_for_rect, mcu_row_intersects_rect, scaled_dimensions, stripe_region_layout,
    StripeRegionLayout,
};
use super::super::restart::{
    finish_scan, restart_seek_for_mcu, skip_to_mcu, McuSkipState, McuSkipTarget,
};
use super::super::{PreparedDecodePlan, StripeBuffer};
use super::row::{decode_mcu_row, McuRowContext, McuRowState};
use crate::backend::Backend;
use crate::entropy::block::CoefficientBlock;
use crate::error::{JpegError, Warning};
use crate::info::{DownscaleFactor, Rect};
use crate::internal::bit_reader::BitReader;
use crate::internal::scratch::ScratchPool;
use alloc::vec::Vec;

#[derive(Clone, Copy)]
pub(super) enum ScanOutputMode {
    ComponentRows,
    InterleavedRgb,
}

#[derive(Clone, Copy)]
pub(super) struct ScanSetup {
    block_size: u32,
    mcu_height_px: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    region: StripeRegionLayout,
    expanded_rect: Rect,
    emit_rect: Rect,
    full_output_rect: bool,
    first_decode_mcu_row: u32,
    decode_mcu_row_end: u32,
    last_output_mcu_row: u32,
    restart: u16,
}

impl ScanSetup {
    pub(super) fn new(
        plan: &PreparedDecodePlan,
        downscale: DownscaleFactor,
        output_rect: Rect,
        mode: ScanOutputMode,
    ) -> Self {
        let (width, height) = scaled_dimensions(plan.dimensions, downscale);
        let block_size = downscale.output_block_size();
        let mcu_width_px = block_size * u32::from(plan.sampling.max_h);
        let mcu_height_px = block_size * u32::from(plan.sampling.max_v);
        let mcus_per_row = width.div_ceil(mcu_width_px);
        let mcu_rows = height.div_ceil(mcu_height_px);
        let expanded_rect = expanded_output_rect(output_rect, width, height);
        let full_output_rect = expanded_rect == Rect::full((width, height));
        let use_420_context_window = matches!(mode, ScanOutputMode::InterleavedRgb)
            && !full_output_rect
            && is_ycbcr_420(plan);
        let emit_rect = if use_420_context_window {
            output_rect
        } else {
            expanded_rect
        };
        let first_decode_mcu_row = if use_420_context_window {
            fast420_first_decode_mcu_row(output_rect, mcu_height_px)
        } else {
            first_decode_mcu_row_for_rect(full_output_rect, expanded_rect, mcu_height_px)
        };
        let decode_mcu_row_end = if use_420_context_window {
            fast420_decode_mcu_row_end(output_rect, mcu_height_px, mcu_rows)
        } else {
            decode_mcu_row_end_for_rect(full_output_rect, expanded_rect, mcu_height_px, mcu_rows)
        };

        Self {
            block_size,
            mcu_height_px,
            mcus_per_row,
            mcu_rows,
            region: stripe_region_layout(plan, downscale, output_rect),
            expanded_rect,
            emit_rect,
            full_output_rect,
            first_decode_mcu_row,
            decode_mcu_row_end,
            last_output_mcu_row: last_mcu_row_for_rect(emit_rect, mcu_height_px, mcu_rows),
            restart: plan.restart_interval.unwrap_or(0),
        }
    }

    pub(super) fn prepare_pool(
        self,
        plan: &PreparedDecodePlan,
        pool: &mut ScratchPool,
    ) -> Result<(), JpegError> {
        pool.prepare_for(
            plan,
            self.region.stripe_mcus_per_row,
            self.block_size,
            plan.scratch_bytes,
        )
    }

    fn should_emit(self, mcu_row: u32) -> bool {
        self.full_output_rect
            || mcu_row_intersects_rect(mcu_row, self.mcu_height_px, self.emit_rect)
    }
}

pub(super) struct ScanBuffers<'a> {
    pub(super) prev_dc: &'a mut [i32],
    pub(super) stripe_a: &'a mut StripeBuffer,
    pub(super) stripe_b: &'a mut StripeBuffer,
    pub(super) stripe_c: &'a mut StripeBuffer,
}

pub(super) trait StripeEmitter {
    fn emit(&mut self, stripe: StripeEmit<'_>) -> Result<(), JpegError>;
}

pub(super) fn decode_scan_rows<E: StripeEmitter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    downscale: DownscaleFactor,
    setup: ScanSetup,
    buffers: ScanBuffers<'_>,
    emitter: &mut E,
) -> Result<Vec<Warning>, JpegError> {
    let ScanBuffers {
        prev_dc,
        stripe_a,
        stripe_b,
        stripe_c,
    } = buffers;
    let mut br = BitReader::new(scan_bytes);
    let mut coeff = CoefficientBlock::default();
    let mut pixels = [0u8; 64];
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let total_mcus = setup.mcu_rows * setup.mcus_per_row;
    let first_decode_mcu = setup.first_decode_mcu_row * setup.mcus_per_row;
    let mut current_mcu = 0u32;
    if let Some(seek) = restart_seek_for_mcu(scan_bytes, setup.restart, first_decode_mcu) {
        br = BitReader::new(&scan_bytes[seek.scan_offset..]);
        current_mcu = seek.mcu_index;
        expected_rst = seek.expected_rst;
    }
    skip_to_mcu(
        plan,
        McuSkipTarget {
            target_mcu: first_decode_mcu,
            total_mcus,
            restart: setup.restart,
        },
        &mut McuSkipState {
            br: &mut br,
            prev_dc,
            current_mcu: &mut current_mcu,
            mcus_since_restart: &mut mcus_since_restart,
            expected_rst: &mut expected_rst,
        },
    )?;

    let row_context = McuRowContext {
        plan,
        backend,
        downscale,
        output_rect: setup.expanded_rect,
        full_output_rect: setup.full_output_rect,
        stripe_mcu_start: setup.region.stripe_mcu_start,
        stripe_mcus_per_row: setup.region.stripe_mcus_per_row,
        mcus_per_row: setup.mcus_per_row,
        mcu_rows: setup.mcu_rows,
        restart: setup.restart,
    };
    let mut prev_stripe = stripe_a;
    let mut curr_stripe = stripe_b;
    let mut next_stripe = stripe_c;
    let mut has_prev = false;
    {
        let mut row_state = McuRowState {
            br: &mut br,
            prev_dc,
            coeff: &mut coeff,
            pixels: &mut pixels,
            mcus_since_restart: &mut mcus_since_restart,
            expected_rst: &mut expected_rst,
        };
        decode_mcu_row(
            &row_context,
            &mut row_state,
            setup.first_decode_mcu_row,
            curr_stripe,
        )?;

        for mcu_row in setup.first_decode_mcu_row + 1..setup.decode_mcu_row_end {
            decode_mcu_row(&row_context, &mut row_state, mcu_row, next_stripe)?;
            if setup.should_emit(mcu_row - 1) {
                emitter.emit(StripeEmit {
                    prev: has_prev.then_some(&*prev_stripe),
                    curr: curr_stripe,
                    next: Some(&*next_stripe),
                    stripe_index: mcu_row - 1,
                    source_width: setup.region.source_width_usize(),
                    downscale,
                })?;
            }
            core::mem::swap(&mut prev_stripe, &mut curr_stripe);
            core::mem::swap(&mut curr_stripe, &mut next_stripe);
            has_prev = true;
        }
    }

    let current_mcu_row = setup.decode_mcu_row_end - 1;
    if current_mcu_row <= setup.last_output_mcu_row && setup.should_emit(current_mcu_row) {
        emitter.emit(StripeEmit {
            prev: has_prev.then_some(&*prev_stripe),
            curr: curr_stripe,
            next: None,
            stripe_index: current_mcu_row,
            source_width: setup.region.source_width_usize(),
            downscale,
        })?;
    }
    finish_scan(&mut br, setup.decode_mcu_row_end == setup.mcu_rows)
}
