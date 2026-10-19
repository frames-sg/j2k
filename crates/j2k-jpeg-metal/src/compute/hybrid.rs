// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hybrid Metal decode for JPEG classes outside the fast-packet kernels.
//!
//! The CPU runs only the serial entropy stage, writing dequantized DCT blocks
//! (or, for lossless JPEG, predictor differences) straight into shared Metal
//! storage. Metal then runs each component's IDCT at the CPU decoder's
//! per-component reduced size (or reconstructs lossless samples), followed by
//! passes that upsample, color-convert, and pack each requested rectangle.
//! The result is byte-identical to CPU decode.

mod batch;
mod lossless;
mod plan;

pub(crate) use self::batch::{
    decode_hybrid_rgb8_batch_into_output, decode_hybrid_rgb8_batch_into_textures,
};

use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_jpeg::adapter::{
    decode_device_dct_blocks_into, decode_device_dct_window_into, device_component_geometry,
    device_dct_layout, device_dct_mcu_window, device_dct_window_layout, JpegDeviceMcuWindow,
};
use j2k_jpeg::{JpegError, SofKind};
use j2k_metal_support::with_checked_buffer_slice_mut;

use crate::abi::JpegHybridPackParams;
use crate::batch_allocation::BatchMetadataBudget;
use crate::metal_types::prelude::*;
use crate::metal_types::{Buffer, BufferRef};

use self::plan::HybridDctPlan;
use super::{
    commit_and_wait_jpeg, dispatch_2d_pipeline, new_command_buffer, new_compute_command_encoder,
    with_runtime, with_runtime_for_session, CpuDecoder, Error, MetalRuntime, Surface,
};
use crate::batch::BatchOp;

const PLANE_KEYS: [&str; 4] = [
    "jpeg_hybrid_plane0",
    "jpeg_hybrid_plane1",
    "jpeg_hybrid_plane2",
    "jpeg_hybrid_plane3",
];

/// One rectangle of the decoded image written into the output buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HybridPack {
    /// Source rectangle in full-resolution image coordinates.
    pub(crate) source_roi: Rect,
    /// Destination origin in output pixels.
    pub(crate) dest: (u32, u32),
}

/// Hybrid-decode `op` on the calling thread's default Metal runtime.
pub(crate) fn decode_hybrid_op_to_surface(
    decoder: &CpuDecoder<'_>,
    fmt: PixelFormat,
    op: BatchOp,
) -> Result<Surface, Error> {
    let (roi, scale) = op_geometry(decoder, op);
    with_runtime(|runtime| decode_hybrid_to_surface(runtime, decoder, fmt, roi, scale))
}

/// Hybrid-decode `op` on a caller-owned Metal session.
pub(crate) fn decode_hybrid_op_to_surface_with_session(
    decoder: &CpuDecoder<'_>,
    fmt: PixelFormat,
    op: BatchOp,
    session: &crate::MetalBackendSession,
) -> Result<Surface, Error> {
    let (roi, scale) = op_geometry(decoder, op);
    with_runtime_for_session(session, |runtime| {
        decode_hybrid_to_surface(runtime, decoder, fmt, roi, scale)
    })
}

/// Compose an RGB8 viewport from source regions of one JPEG on the default
/// Metal runtime: entropy decode once, then pack each region in order onto a
/// zeroed viewport, as the CPU viewport compositor does.
pub(crate) fn compose_hybrid_rgb8_viewport(
    decoder: &CpuDecoder<'_>,
    scale: Downscale,
    viewport_dims: (u32, u32),
    packs: &[HybridPack],
) -> Result<Surface, Error> {
    let fmt = PixelFormat::Rgb8;
    with_runtime(|runtime| {
        let bytes = output_bytes(viewport_dims, fmt)?;
        let output = runtime.surface_output_buffer(bytes)?;
        crate::buffers::checked_fill_buffer_u8(&output, bytes, 0, "hybrid viewport clear")?;
        decode_hybrid_into(runtime, decoder, fmt, scale, packs, &output, viewport_dims)?;
        Surface::from_metal_buffer(output, viewport_dims, fmt)
    })
}

pub(super) fn op_geometry(decoder: &CpuDecoder<'_>, op: BatchOp) -> (Rect, Downscale) {
    let full = Rect::full(decoder.info().dimensions);
    match op {
        BatchOp::Full => (full, Downscale::None),
        BatchOp::Region(roi) => (roi, Downscale::None),
        BatchOp::Scaled(scale) => (full, scale),
        BatchOp::RegionScaled { roi, scale } => (roi, scale),
    }
}

/// Decode `roi` at `scale` into a Metal-resident surface of `fmt`.
pub(crate) fn decode_hybrid_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<Surface, Error> {
    let scaled = roi.scaled_covering(scale);
    let dims = (scaled.w, scaled.h);
    let output = runtime.surface_output_buffer(output_bytes(dims, fmt)?)?;
    decode_hybrid_into(
        runtime,
        decoder,
        fmt,
        scale,
        &[HybridPack {
            source_roi: roi,
            dest: (0, 0),
        }],
        &output,
        dims,
    )?;
    Surface::from_metal_buffer(output, dims, fmt)
}

fn output_bytes(dims: (u32, u32), fmt: PixelFormat) -> Result<usize, Error> {
    if dims.0 == 0 || dims.1 == 0 {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_operation(
                "JPEG Metal decode requires a non-empty output region",
            ),
        ));
    }
    (dims.0 as usize)
        .checked_mul(dims.1 as usize)
        .and_then(|pixels| pixels.checked_mul(fmt.bytes_per_pixel()))
        .ok_or_else(|| plan::invariant("output size overflowed"))
}

/// Entropy-decode `decoder` once and write every pack rectangle into
/// `output`, a tightly packed `output_dims` buffer of `fmt` pixels.
fn decode_hybrid_into(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    fmt: PixelFormat,
    scale: Downscale,
    packs: &[HybridPack],
    output: &BufferRef,
    output_dims: (u32, u32),
) -> Result<(), Error> {
    let info = decoder.info();
    for pack in packs {
        if !pack.source_roi.is_within(info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: pack.source_roi.into(),
                width: info.dimensions.0,
                height: info.dimensions.1,
            }
            .into());
        }
    }
    if info.sof_kind == SofKind::Lossless {
        return lossless::decode_lossless_into(
            runtime,
            decoder,
            fmt,
            scale,
            packs,
            output,
            output_dims,
        );
    }
    validate_dct_output_format(info.bit_depth, info.color_space, fmt)?;
    let window = bounding_rect(packs, scale)?;
    let geometry = device_component_geometry(info, scale)?;
    // Regions of 8-bit sequential scans decode only the MCUs they need.
    let blocks = plan::component_block_windows(&geometry, window)?;
    let mcu_window = device_dct_mcu_window(decoder, &blocks[..geometry.len()]);
    let layout = match mcu_window {
        Some(mcu_window) => device_dct_window_layout(decoder, mcu_window)?,
        None => device_dct_layout(decoder)?,
    };
    let plan = HybridDctPlan::new(info.color_space, &layout, &geometry, window, fmt)?;
    let mut pack_params = BatchMetadataBudget::new("JPEG Metal hybrid pack plan")
        .try_vec(packs.len(), "JPEG Metal hybrid pack rectangles")?;
    for pack in packs {
        let source = pack.source_roi.scaled_covering(scale);
        pack_params.push(plan.pack_params(source, pack.dest, output_dims, fmt)?);
    }

    let mut scratch = runtime.batch_scratch()?;
    let coefficients = stage_coefficients(runtime, &mut scratch, decoder, &layout, mcu_window)?;
    let sample_bytes = if layout.precision == 12 { 2 } else { 1 };
    let mut planes: [Option<Buffer>; 4] = Default::default();
    for (index, &samples) in plan.plane_samples[..plan.component_count]
        .iter()
        .enumerate()
    {
        planes[index] = Some(scratch.private_buffer(
            &runtime.device,
            PLANE_KEYS[index],
            samples * sample_bytes,
        )?);
    }
    encode_dct_passes(
        runtime,
        &DctPasses {
            twelve_bit: layout.precision == 12,
            coefficients: &coefficients,
            planes: &planes,
            plan: &plan,
            pack_params: &pack_params,
            output,
        },
    )?;
    drop(scratch);
    Ok(())
}

/// Entropy-decode every block straight into leased shared Metal storage.
fn stage_coefficients(
    runtime: &MetalRuntime,
    scratch: &mut crate::buffers::MetalBatchScratch,
    decoder: &CpuDecoder<'_>,
    layout: &j2k_jpeg::adapter::JpegDeviceDctLayout,
    mcu_window: Option<JpegDeviceMcuWindow>,
) -> Result<Buffer, Error> {
    let total_blocks = layout.total_blocks()?;
    let coefficient_bytes = total_blocks
        .checked_mul(core::mem::size_of::<[i16; 64]>())
        .ok_or_else(|| plan::invariant("coefficient storage size overflowed"))?;
    let coefficients = scratch.shared_buffer(
        &runtime.device,
        "jpeg_hybrid_coefficients",
        coefficient_bytes,
    )?;
    // SAFETY: the coefficient buffer is exclusively leased from this
    // runtime's scratch slot and is not referenced by any submitted command;
    // the CPU writes finish before the IDCT pass is encoded.
    unsafe {
        with_checked_buffer_slice_mut::<[i16; 64], _>(&coefficients, 0, total_blocks, |blocks| {
            match mcu_window {
                Some(window) => decode_device_dct_window_into(decoder, window, blocks),
                None => decode_device_dct_blocks_into(decoder, blocks).map(drop),
            }
        })
    }
    .map_err(|error| {
        crate::error::metal_kernel_support_error(
            format!("JPEG Metal hybrid coefficient staging failed: {error}"),
            error,
        )
    })??;
    Ok(coefficients)
}

struct DctPasses<'a> {
    twelve_bit: bool,
    coefficients: &'a BufferRef,
    planes: &'a [Option<Buffer>; 4],
    plan: &'a HybridDctPlan,
    pack_params: &'a [JpegHybridPackParams],
    output: &'a BufferRef,
}

/// Encode every component's IDCT and every pack rectangle in one serial
/// compute encoder, then wait for completion.
fn encode_dct_passes(runtime: &MetalRuntime, passes: &DctPasses<'_>) -> Result<(), Error> {
    let (idct_pipeline, pack_pipeline) = if passes.twelve_bit {
        (
            &runtime.pipelines.hybrid_idct12,
            &runtime.pipelines.hybrid_pack12,
        )
    } else {
        (
            &runtime.pipelines.hybrid_idct8,
            &runtime.pipelines.hybrid_pack8,
        )
    };
    let first_plane = passes.planes[0]
        .as_ref()
        .ok_or_else(|| plan::invariant("first component plane is missing"))?;
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    // Serial dispatches: each pack pass observes every IDCT write, and later
    // pack rectangles overwrite earlier ones where they overlap.
    encoder.setComputePipelineState(idct_pipeline);
    encoder.bind_buffer(0, Some(passes.coefficients), 0);
    for (params, plane) in passes.plan.idct[..passes.plan.component_count]
        .iter()
        .zip(passes.planes.iter().flatten())
    {
        encoder.bind_buffer(1, Some(plane), 0);
        encoder.bind_bytes(2, params);
        dispatch_2d_pipeline(
            &encoder,
            idct_pipeline,
            (params.block_x_count, params.block_y_count),
        );
    }
    encoder.setComputePipelineState(pack_pipeline);
    for (slot, plane) in (0_u64..).zip(passes.planes) {
        encoder.bind_buffer(slot, Some(plane.as_ref().unwrap_or(first_plane)), 0);
    }
    encoder.bind_buffer(4, Some(passes.output), 0);
    for params in passes.pack_params {
        encoder.bind_bytes(5, params);
        dispatch_2d_pipeline(&encoder, pack_pipeline, (params.width, params.height));
    }
    encoder.endEncoding();
    commit_and_wait_jpeg(&command_buffer)
}

/// Union of every pack's scaled source rectangle.
fn bounding_rect(packs: &[HybridPack], scale: Downscale) -> Result<Rect, Error> {
    let mut rects = packs
        .iter()
        .map(|pack| pack.source_roi.scaled_covering(scale));
    let first = rects
        .next()
        .ok_or_else(|| plan::invariant("no output rectangles"))?;
    let (mut x0, mut y0) = (first.x, first.y);
    let (mut x1, mut y1) = (first.x + first.w, first.y + first.h);
    for rect in rects {
        x0 = x0.min(rect.x);
        y0 = y0.min(rect.y);
        x1 = x1.max(rect.x + rect.w);
        y1 = y1.max(rect.y + rect.h);
    }
    Ok(Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    })
}

/// The CPU decoder's precision/output pairing: 8-bit DCT JPEG decodes to
/// 8-bit formats, 12-bit DCT JPEG to native-range 16-bit formats, and Gray16
/// only from grayscale.
fn validate_dct_output_format(
    precision: u8,
    color_space: j2k_jpeg::ColorSpace,
    fmt: PixelFormat,
) -> Result<(), Error> {
    let supported = match (precision, fmt) {
        (8, PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8)
        | (12, PixelFormat::Rgb16 | PixelFormat::Rgba16) => true,
        (12, PixelFormat::Gray16) => color_space == j2k_jpeg::ColorSpace::Grayscale,
        _ => false,
    };
    if supported {
        Ok(())
    } else {
        Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_format(
                "JPEG Metal decode supports Gray8/Rgb8/Rgba8 for 8-bit JPEG and Rgb16/Rgba16 (and Gray16 for grayscale) for 12-bit JPEG, matching CPU decode",
            ),
        ))
    }
}
