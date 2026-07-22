// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::BatchLayout;
use j2k_core::PixelFormat;
use j2k_cuda_runtime::{CudaJ2kStoreRgbNativeTarget, CudaJ2kStoreRgbaNativeTarget};

use super::jobs::{
    native_level_shift, native_rgba_store_job, native_store_job, transform_selector,
};
use crate::decoder::color_batch::{
    pooled_cuda_buffer, validate_color_stores, CudaDecodedComponent, CudaHtj2kColorDecodePlans,
    CudaHtj2kProfileReport,
};
use crate::decoder::{Error, CUDA_HTJ2K_KERNELS_NOT_READY};

pub(super) enum NativeColorStoreTargets<'a> {
    Rgb(Vec<CudaJ2kStoreRgbNativeTarget<'a>>),
    Rgba(Vec<CudaJ2kStoreRgbaNativeTarget<'a>>),
}

pub(super) fn build_store_targets(
    colors: Vec<CudaHtj2kColorDecodePlans>,
    decoded: &[CudaDecodedComponent],
    fmt: PixelFormat,
    layout: BatchLayout,
) -> Result<(NativeColorStoreTargets<'_>, Vec<CudaHtj2kProfileReport>), Error> {
    let mut reports = try_reports(colors.len())?;
    let rgba = matches!(
        fmt,
        PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaI16
    );
    let mut three_channel_targets = if rgba {
        Vec::new()
    } else {
        try_rgb_targets(colors.len())?
    };
    let mut four_channel_targets = if rgba {
        try_rgba_targets(colors.len())?
    } else {
        Vec::new()
    };
    let mut component_offset = 0usize;
    for (image_index, mut color) in colors.into_iter().enumerate() {
        let components =
            decoded_color_components(decoded, &mut component_offset, color.components.len())?;
        if rgba {
            four_channel_targets.push(rgba_store_target(&color, components, fmt, layout)?);
        } else {
            three_channel_targets.push(rgb_store_target(&color, components, fmt, layout)?);
        }
        update_color_report(&mut color, components, image_index);
        reports.push(color.report);
    }
    if rgba {
        Ok((NativeColorStoreTargets::Rgba(four_channel_targets), reports))
    } else {
        Ok((NativeColorStoreTargets::Rgb(three_channel_targets), reports))
    }
}

fn try_reports(capacity: usize) -> Result<Vec<CudaHtj2kProfileReport>, Error> {
    let mut reports = Vec::new();
    reports
        .try_reserve_exact(capacity)
        .map_err(|_| Error::HostAllocationFailed {
            bytes: capacity.saturating_mul(std::mem::size_of::<CudaHtj2kProfileReport>()),
            what: "j2k CUDA exact RGB reports",
        })?;
    Ok(reports)
}

fn try_rgb_targets<'a>(capacity: usize) -> Result<Vec<CudaJ2kStoreRgbNativeTarget<'a>>, Error> {
    let mut targets = Vec::new();
    targets
        .try_reserve_exact(capacity)
        .map_err(|_| Error::HostAllocationFailed {
            bytes: capacity.saturating_mul(std::mem::size_of::<CudaJ2kStoreRgbNativeTarget<'a>>()),
            what: "j2k CUDA exact RGB store targets",
        })?;
    Ok(targets)
}

fn try_rgba_targets<'a>(capacity: usize) -> Result<Vec<CudaJ2kStoreRgbaNativeTarget<'a>>, Error> {
    let mut targets = Vec::new();
    targets
        .try_reserve_exact(capacity)
        .map_err(|_| Error::HostAllocationFailed {
            bytes: capacity.saturating_mul(std::mem::size_of::<CudaJ2kStoreRgbaNativeTarget<'a>>()),
            what: "j2k CUDA exact RGBA store targets",
        })?;
    Ok(targets)
}

fn decoded_color_components<'a>(
    decoded: &'a [CudaDecodedComponent],
    component_offset: &mut usize,
    component_count: usize,
) -> Result<&'a [CudaDecodedComponent], Error> {
    let component_end =
        component_offset
            .checked_add(component_count)
            .ok_or(Error::HostAllocationFailed {
                bytes: usize::MAX,
                what: "j2k CUDA exact color component range",
            })?;
    let components =
        decoded
            .get(*component_offset..component_end)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            })?;
    *component_offset = component_end;
    Ok(components)
}

fn rgb_store_target<'a>(
    color: &CudaHtj2kColorDecodePlans,
    components: &'a [CudaDecodedComponent],
    fmt: PixelFormat,
    layout: BatchLayout,
) -> Result<CudaJ2kStoreRgbNativeTarget<'a>, Error> {
    let stores = [
        &components[0].store,
        &components[1].store,
        &components[2].store,
    ];
    validate_color_stores(stores, color.dimensions)?;
    let bit_depths = color.rgb_bit_depths();
    let addends = if color.mct && fmt != PixelFormat::RgbI16 {
        bit_depths.map(native_level_shift)
    } else {
        stores.map(|store| store.addend)
    };
    Ok(CudaJ2kStoreRgbNativeTarget {
        output_index: color.output_index,
        plane0: pooled_cuda_buffer(&components[0].buffer)?,
        plane1: pooled_cuda_buffer(&components[1].buffer)?,
        plane2: pooled_cuda_buffer(&components[2].buffer)?,
        job: native_store_job(
            stores,
            bit_depths,
            addends,
            layout,
            transform_selector(color.mct, color.transform),
        ),
    })
}

fn rgba_store_target<'a>(
    color: &CudaHtj2kColorDecodePlans,
    components: &'a [CudaDecodedComponent],
    fmt: PixelFormat,
    layout: BatchLayout,
) -> Result<CudaJ2kStoreRgbaNativeTarget<'a>, Error> {
    let stores = [
        &components[0].store,
        &components[1].store,
        &components[2].store,
        &components[3].store,
    ];
    validate_color_stores(stores, color.dimensions)?;
    let addends = if color.mct && fmt != PixelFormat::RgbaI16 {
        let mut addends = color.bit_depths.map(native_level_shift);
        addends[3] = stores[3].addend;
        addends
    } else {
        stores.map(|store| store.addend)
    };
    Ok(CudaJ2kStoreRgbaNativeTarget {
        output_index: color.output_index,
        plane0: pooled_cuda_buffer(&components[0].buffer)?,
        plane1: pooled_cuda_buffer(&components[1].buffer)?,
        plane2: pooled_cuda_buffer(&components[2].buffer)?,
        plane3: pooled_cuda_buffer(&components[3].buffer)?,
        job: native_rgba_store_job(
            stores,
            color.bit_depths,
            addends,
            layout,
            transform_selector(color.mct, color.transform),
        ),
    })
}

fn update_color_report(
    color: &mut CudaHtj2kColorDecodePlans,
    components: &[CudaDecodedComponent],
    image_index: usize,
) {
    color.report.dispatch_count = components
        .iter()
        .map(|component| component.dispatches)
        .sum::<usize>()
        + usize::from(image_index == 0);
    for component in components {
        component.timings.add_to_report(&mut color.report);
    }
}
