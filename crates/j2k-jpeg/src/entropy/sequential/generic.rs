// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic baseline sequential scan entry points and typed output dispatch.

mod driver;
mod row;

use self::driver::{decode_scan_rows, ScanBuffers, ScanOutputMode, ScanSetup, StripeEmitter};
use super::emit::{emit_stripe, emit_stripe_rgb, StripeEmit};
use super::{OutputScratch, PreparedDecodePlan, RgbOutputScratch};
use crate::backend::Backend;
use crate::error::{JpegError, Warning};
use crate::info::{ColorSpace, DownscaleFactor, Rect};
use crate::internal::scratch::ScratchPool;
use crate::output::{InterleavedRgbWriter, OutputWriter};
use alloc::vec::Vec;

struct ComponentStripeEmitter<'plan, 'writer, 'scratch, W> {
    plan: &'plan PreparedDecodePlan,
    writer: &'writer mut W,
    scratch: OutputScratch<'scratch>,
}

impl<W: OutputWriter> StripeEmitter for ComponentStripeEmitter<'_, '_, '_, W> {
    #[inline]
    fn emit(&mut self, stripe: StripeEmit<'_>) -> Result<(), JpegError> {
        emit_stripe(self.plan, self.writer, &mut self.scratch, stripe)
    }
}

struct RgbStripeEmitter<'plan, 'writer, 'scratch, W> {
    plan: &'plan PreparedDecodePlan,
    backend: Backend,
    writer: &'writer mut W,
    scratch: RgbOutputScratch<'scratch>,
}

impl<W: OutputWriter + InterleavedRgbWriter> StripeEmitter for RgbStripeEmitter<'_, '_, '_, W> {
    #[inline]
    fn emit(&mut self, stripe: StripeEmit<'_>) -> Result<(), JpegError> {
        emit_stripe_rgb(
            self.plan,
            self.backend,
            self.writer,
            &mut self.scratch,
            stripe,
        )
    }
}

pub(crate) fn decode_scan_baseline<W: OutputWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    downscale: DownscaleFactor,
    output_rect: Rect,
) -> Result<Vec<Warning>, JpegError> {
    let setup = ScanSetup::new(plan, downscale, output_rect, ScanOutputMode::ComponentRows);
    setup.prepare_pool(plan, pool)?;
    let ScratchPool {
        prev_dc,
        stripe_a,
        stripe_b,
        stripe_c,
        ycbcr_420_rows,
        ycbcr_generic_rows,
        rgb_generic_rows,
        ..
    } = pool;
    let scratch = match plan.color_space {
        ColorSpace::Grayscale => OutputScratch::Grayscale,
        ColorSpace::YCbCr if super::is_ycbcr_420(plan) => OutputScratch::YCbCr420(ycbcr_420_rows),
        ColorSpace::YCbCr => OutputScratch::YCbCrGeneric(ycbcr_generic_rows),
        ColorSpace::Rgb | ColorSpace::Cmyk | ColorSpace::Ycck => {
            OutputScratch::RgbGeneric(rgb_generic_rows)
        }
    };
    let mut emitter = ComponentStripeEmitter {
        plan,
        writer,
        scratch,
    };
    decode_scan_rows(
        plan,
        backend,
        scan_bytes,
        downscale,
        setup,
        ScanBuffers {
            prev_dc: prev_dc.as_mut_slice(),
            stripe_a,
            stripe_b,
            stripe_c,
        },
        &mut emitter,
    )
}

pub(crate) fn decode_scan_baseline_rgb<W: OutputWriter + InterleavedRgbWriter>(
    plan: &PreparedDecodePlan,
    backend: Backend,
    scan_bytes: &[u8],
    pool: &mut ScratchPool,
    writer: &mut W,
    downscale: DownscaleFactor,
    output_rect: Rect,
) -> Result<Vec<Warning>, JpegError> {
    let setup = ScanSetup::new(plan, downscale, output_rect, ScanOutputMode::InterleavedRgb);
    setup.prepare_pool(plan, pool)?;
    let ScratchPool {
        prev_dc,
        stripe_a,
        stripe_b,
        stripe_c,
        ycbcr_generic_rows,
        rgb_generic_rows,
        ..
    } = pool;
    let scratch = match plan.color_space {
        ColorSpace::Grayscale => RgbOutputScratch::None,
        ColorSpace::YCbCr if super::is_ycbcr_420(plan) => RgbOutputScratch::YCbCr420,
        ColorSpace::YCbCr => RgbOutputScratch::YCbCrGeneric(ycbcr_generic_rows),
        ColorSpace::Rgb | ColorSpace::Cmyk | ColorSpace::Ycck => {
            RgbOutputScratch::RgbGeneric(rgb_generic_rows)
        }
    };
    let mut emitter = RgbStripeEmitter {
        plan,
        backend,
        writer,
        scratch,
    };
    decode_scan_rows(
        plan,
        backend,
        scan_bytes,
        downscale,
        setup,
        ScanBuffers {
            prev_dc: prev_dc.as_mut_slice(),
            stripe_a,
            stripe_b,
            stripe_c,
        },
        &mut emitter,
    )
}
