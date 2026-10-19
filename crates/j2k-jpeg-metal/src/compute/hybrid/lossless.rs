// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless (SOF3) hybrid decode: the CPU entropy-decodes every difference
//! into shared Metal storage; Metal reconstructs samples with each
//! component's predictor, then replicates subsampled components,
//! color-converts, and packs each rectangle.

use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_jpeg::adapter::{
    decode_device_lossless_differences_into, device_lossless_layout, JpegDeviceLosslessLayout,
};
use j2k_jpeg::{ColorSpace as JpegColorSpace, DecodeRequest};
use j2k_metal_support::with_checked_buffer_slice_mut;
use objc2_metal::{MTLComputePipelineState as _, MTLSize};

use crate::abi::{
    JpegLosslessComponentParams, JpegLosslessPackParams, JpegLosslessReconstructParams,
    JpegLosslessRowsParams, HYBRID_COLOR_GRAY, HYBRID_COLOR_RGB, HYBRID_COLOR_YCBCR,
};
use crate::metal_types::prelude::*;
use crate::metal_types::{Buffer, BufferRef, ComputeCommandEncoderRef};

use super::super::{
    checked_u32, commit_and_wait_jpeg, dispatch_2d_pipeline, new_command_buffer,
    new_compute_command_encoder, CpuDecoder, Error, MetalRuntime,
};
use super::plan::{invariant, out_format_code};
use super::HybridPack;

/// Entropy-decode `decoder` once and write every pack rectangle into
/// `output`, a tightly packed `output_dims` buffer of `fmt` pixels.
pub(super) fn decode_lossless_into(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    fmt: PixelFormat,
    scale: Downscale,
    packs: &[HybridPack],
    output: &BufferRef,
    output_dims: (u32, u32),
) -> Result<(), Error> {
    let layout = device_lossless_layout(decoder, fmt)?;
    let reconstruct = reconstruct_params(&layout)?;
    let template = pack_template(decoder, &layout, fmt, scale)?;
    let mut pack_params =
        crate::batch_allocation::BatchMetadataBudget::new("JPEG Metal lossless pack plan")
            .try_vec(packs.len(), "JPEG Metal lossless pack rectangles")?;
    for pack in packs {
        let source = pack.source_roi.scaled_covering(scale);
        pack_params.push(pack_params_for(
            template,
            source,
            pack.dest,
            output_dims,
            fmt,
        )?);
    }

    let mut scratch = runtime.batch_scratch()?;
    let staged = stage_differences(runtime, &mut scratch, decoder, fmt, &layout)?;
    let total = layout.total_samples();
    let samples = scratch.private_buffer(&runtime.device, "jpeg_lossless_samples", total * 2)?;
    let status = scratch.shared_zeroed_buffer(&runtime.device, "jpeg_lossless_status", 4)?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    // Serial dispatches: packing observes every reconstructed sample.
    encode_reconstruction(
        runtime,
        &encoder,
        &layout,
        &reconstruct,
        [&staged.diffs, &samples, &status],
        staged.columns.as_deref(),
    )?;
    // Samples above 8 bits pack to 16-bit output.
    let pack_pipeline = if layout.precision > 8 {
        &runtime.pipelines.lossless_pack16
    } else {
        &runtime.pipelines.lossless_pack8
    };
    encoder.setComputePipelineState(pack_pipeline);
    encoder.bind_buffer(0, Some(&samples), 0);
    encoder.bind_buffer(1, Some(output), 0);
    encoder.bind_bytes(3, &reconstruct);
    for params in &pack_params {
        encoder.bind_bytes(2, params);
        dispatch_2d_pipeline(
            &encoder,
            pack_pipeline,
            (params.out_width, params.out_height),
        );
    }
    encoder.endEncoding();
    commit_and_wait_jpeg(&command_buffer)?;
    let out_of_range = crate::buffers::checked_buffer_read::<u32>(&status, "lossless status")?;
    drop(scratch);
    if out_of_range != 0 {
        return Err(cpu_decode_error(decoder, fmt));
    }
    Ok(())
}

struct StagedDifferences {
    diffs: Buffer,
    /// Each row's first sample of every predictor-1 component, for
    /// row-parallel reconstruction.
    columns: Option<Buffer>,
}

/// Entropy-decode every difference straight into leased shared Metal
/// storage. For predictor-1 components, also derive each row's first sample.
fn stage_differences(
    runtime: &MetalRuntime,
    scratch: &mut crate::buffers::MetalBatchScratch,
    decoder: &CpuDecoder<'_>,
    fmt: PixelFormat,
    layout: &JpegDeviceLosslessLayout,
) -> Result<StagedDifferences, Error> {
    let total = layout.total_samples();
    checked_u32(total, "lossless sample count")?;
    let diffs = scratch.shared_buffer(&runtime.device, "jpeg_lossless_differences", total * 2)?;
    let staging_error = |error: j2k_metal_support::MetalSupportError| {
        crate::error::metal_kernel_support_error(
            format!("JPEG Metal lossless difference staging failed: {error}"),
            error,
        )
    };
    // SAFETY: the difference buffer is exclusively leased from this runtime's
    // scratch slot and is not referenced by any submitted command; the CPU
    // writes finish before reconstruction is encoded.
    let staged = unsafe {
        with_checked_buffer_slice_mut::<u16, _>(&diffs, 0, total, |storage| {
            decode_device_lossless_differences_into(decoder, fmt, storage)
        })
    }
    .map_err(staging_error)?;
    if staged.is_err() {
        return Err(cpu_decode_error(decoder, fmt));
    }
    let rows: usize = layout
        .components()
        .iter()
        .filter(|component| component.predictor == 1)
        .map(|component| component.height as usize)
        .sum();
    if rows == 0 {
        return Ok(StagedDifferences {
            diffs,
            columns: None,
        });
    }
    let columns = scratch.shared_buffer(&runtime.device, "jpeg_lossless_columns", rows * 2)?;
    // SAFETY: both buffers are exclusively leased and not yet submitted; the
    // CPU only reads the differences it just wrote.
    unsafe {
        with_checked_buffer_slice_mut::<u16, _>(&diffs, 0, total, |diffs| {
            with_checked_buffer_slice_mut::<u16, _>(&columns, 0, rows, |columns| {
                first_samples_of_rows(layout, diffs, columns);
            })
        })
    }
    .map_err(staging_error)?
    .map_err(staging_error)?;
    Ok(StagedDifferences {
        diffs,
        columns: Some(columns),
    })
}

/// Each row's first sample, predictor-1 component after component: the
/// first row of the scan and of every restart interval seeds the 1-D
/// predictor, every later row predicts from the sample above. Modulo 2^16,
/// as reconstruction is.
fn first_samples_of_rows(layout: &JpegDeviceLosslessLayout, diffs: &[u16], columns: &mut [u16]) {
    let mut out = columns.iter_mut();
    for (component, base) in layout.components().iter().zip(layout.sample_bases()) {
        if component.predictor != 1 {
            continue;
        }
        let (width, restart) = (component.width as usize, component.restart_rows as usize);
        let mut previous = 0_u16;
        for row in 0..component.height as usize {
            let first_row = row == 0 || (restart != 0 && row % restart == 0);
            let seed = if first_row {
                component.initial_predictor
            } else {
                previous
            };
            previous = seed.wrapping_add(diffs[base + row * width]);
            if let Some(slot) = out.next() {
                *slot = previous;
            }
        }
    }
}

/// Encode sample reconstruction: row-parallel prefix sums for predictor-1
/// components (given each row's first sample), then one anti-diagonal
/// wavefront threadgroup per remaining component.
fn encode_reconstruction(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    layout: &JpegDeviceLosslessLayout,
    reconstruct: &JpegLosslessReconstructParams,
    [diffs, samples, status]: [&BufferRef; 3],
    columns: Option<&BufferRef>,
) -> Result<(), Error> {
    if let Some(columns) = columns {
        let pipeline = &runtime.pipelines.lossless_reconstruct_rows;
        encoder.setComputePipelineState(pipeline);
        encoder.bind_buffer(0, Some(diffs), 0);
        encoder.bind_buffer(1, Some(samples), 0);
        encoder.bind_buffer(2, Some(status), 0);
        encoder.bind_bytes(3, reconstruct);
        encoder.bind_buffer(4, Some(columns), 0);
        let mut column_base = 0_usize;
        for (index, component) in layout.components().iter().enumerate() {
            if component.predictor != 1 {
                continue;
            }
            encoder.bind_bytes(
                5,
                &JpegLosslessRowsParams {
                    component: checked_u32(index, "lossless component index")?,
                    column_base: checked_u32(column_base, "lossless column base")?,
                },
            );
            // Eight or more samples per thread keep the prefix-sum overhead small.
            let threads = (component.width.saturating_sub(1) as usize)
                .div_ceil(8)
                .next_multiple_of(32)
                .clamp(32, 256)
                .min(pipeline.maxTotalThreadsPerThreadgroup());
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: component.height as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
            column_base += component.height as usize;
        }
    }
    if reconstruct.wavefront_mask == 0 {
        return Ok(());
    }
    let pipeline = &runtime.pipelines.lossless_reconstruct;
    let tallest = layout
        .components()
        .iter()
        .filter(|component| component.predictor != 1)
        .map(|component| component.height as usize)
        .max()
        .unwrap_or(1);
    encoder.setComputePipelineState(pipeline);
    encoder.bind_buffer(0, Some(diffs), 0);
    encoder.bind_buffer(1, Some(samples), 0);
    encoder.bind_buffer(2, Some(status), 0);
    encoder.bind_bytes(3, reconstruct);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: layout.components().len(),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: pipeline.maxTotalThreadsPerThreadgroup().min(tallest).max(1),
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// The error CPU decode reports for a stream the device stages rejected.
///
/// CPU decode reconstructs each MCU row right after entropy-decoding it, so
/// for a malformed stream it may report an out-of-range sample before an
/// entropy error later in the scan. Reproducing that order on the device
/// would serialize both stages; malformed streams instead take CPU decode's
/// own error. No pixels are produced on this path.
fn cpu_decode_error(decoder: &CpuDecoder<'_>, fmt: PixelFormat) -> Error {
    match decoder.decode_request(DecodeRequest::full(fmt)) {
        Err(error) => error.into(),
        Ok(_) => Error::MetalKernel {
            message: "JPEG Metal lossless reconstruction rejected a stream CPU decode accepts"
                .to_string(),
        },
    }
}

fn reconstruct_params(
    layout: &JpegDeviceLosslessLayout,
) -> Result<JpegLosslessReconstructParams, Error> {
    let bases = layout.sample_bases();
    let mut components = [JpegLosslessComponentParams::default(); 3];
    let mut wavefront_mask = 0_u32;
    for (index, component) in layout.components().iter().enumerate() {
        components[index] = JpegLosslessComponentParams {
            width: component.width,
            height: component.height,
            restart_rows: component.restart_rows,
            base: checked_u32(bases[index], "lossless plane base")?,
            predictor: u32::from(component.predictor),
            initial_predictor: u32::from(component.initial_predictor),
            sample_limit: u32::from(component.sample_limit),
            point_transform: u32::from(component.point_transform),
            h_expand: u32::from(component.h_expand),
            v_expand: u32::from(component.v_expand),
        };
        if component.predictor != 1 {
            wavefront_mask |= 1 << index;
        }
    }
    Ok(JpegLosslessReconstructParams {
        components,
        wavefront_mask,
        component_count: checked_u32(layout.components().len(), "lossless component count")?,
    })
}

fn pack_template(
    decoder: &CpuDecoder<'_>,
    layout: &JpegDeviceLosslessLayout,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<JpegLosslessPackParams, Error> {
    let (width, height) = decoder.info().dimensions;
    Ok(JpegLosslessPackParams {
        width,
        height,
        color_space: match layout.color_space {
            JpegColorSpace::YCbCr => HYBRID_COLOR_YCBCR,
            JpegColorSpace::Grayscale => HYBRID_COLOR_GRAY,
            _ => HYBRID_COLOR_RGB,
        },
        precision: u32::from(layout.precision),
        out_format: out_format_code(fmt)?,
        denominator: scale.denominator(),
        ..JpegLosslessPackParams::default()
    })
}

fn pack_params_for(
    template: JpegLosslessPackParams,
    source: Rect,
    dest: (u32, u32),
    dest_dims: (u32, u32),
    fmt: PixelFormat,
) -> Result<JpegLosslessPackParams, Error> {
    let inside_dest = dest
        .0
        .checked_add(source.w)
        .is_some_and(|end| end <= dest_dims.0)
        && dest
            .1
            .checked_add(source.h)
            .is_some_and(|end| end <= dest_dims.1);
    if !inside_dest || source.w == 0 || source.h == 0 {
        return Err(invariant(
            "lossless pack rectangle lies outside its destination",
        ));
    }
    let output_samples = (dest_dims.0 as usize)
        .checked_mul(dest_dims.1 as usize)
        .and_then(|pixels| pixels.checked_mul(fmt.channels()))
        .ok_or_else(|| invariant("output size overflowed"))?;
    checked_u32(output_samples, "lossless output size")?;
    Ok(JpegLosslessPackParams {
        src_x: source.x,
        src_y: source.y,
        out_width: source.w,
        out_height: source.h,
        dest_x: dest.0,
        dest_y: dest.1,
        dest_stride: dest_dims.0,
        ..template
    })
}
