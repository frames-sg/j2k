// SPDX-License-Identifier: MIT OR Apache-2.0

mod jobs;
mod submission;

use j2k::BatchLayout;
use j2k_core::PixelFormat;
use j2k_cuda_runtime::{
    CudaExternalDeviceBufferViewMut, CudaJ2kStoreRgbNativeTarget, CudaJ2kStoreRgbaNativeTarget,
};

use jobs::{native_level_shift, native_rgba_store_job, native_store_job, transform_selector};
use submission::store_targets;

use super::StoredNativeColorBatch;
use crate::decoder::color_batch::{
    finish_cuda_component_decode, pooled_cuda_buffer, validate_color_stores,
    CudaComponentDecodeWork, CudaDecodedComponent, CudaHtj2kColorDecodePlans,
    CudaHtj2kProfileReport,
};
use crate::decoder::{Error, CUDA_HTJ2K_KERNELS_NOT_READY};

enum NativeColorStoreTargets<'a> {
    Rgb(Vec<CudaJ2kStoreRgbNativeTarget<'a>>),
    Rgba(Vec<CudaJ2kStoreRgbaNativeTarget<'a>>),
}

pub(super) fn finish_and_store_native_color(
    context: &j2k_cuda_runtime::CudaContext,
    colors: Vec<CudaHtj2kColorDecodePlans>,
    component_work: Vec<CudaComponentDecodeWork>,
    fmt: PixelFormat,
    layout: BatchLayout,
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
) -> Result<
    (
        StoredNativeColorBatch,
        Vec<CudaHtj2kProfileReport>,
        Vec<CudaDecodedComponent>,
    ),
    Error,
> {
    let expected_components = colors.iter().try_fold(0usize, |count, color| {
        count.checked_add(color.components.len())
    });
    let decoded = finish_components(
        component_work,
        expected_components.ok_or(Error::HostAllocationFailed {
            bytes: usize::MAX,
            what: "j2k CUDA exact color decoded component count",
        })?,
    )?;
    let (targets, reports) = build_store_targets(colors, &decoded, fmt, layout)?;
    let stored = store_targets(context, fmt, external, enqueue_external, &targets)?;
    Ok((stored, reports, decoded))
}

fn finish_components(
    component_work: Vec<CudaComponentDecodeWork>,
    expected_components: usize,
) -> Result<Vec<CudaDecodedComponent>, Error> {
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(component_work.len())
        .map_err(|_| Error::HostAllocationFailed {
            bytes: component_work
                .len()
                .saturating_mul(std::mem::size_of::<CudaDecodedComponent>()),
            what: "j2k CUDA exact RGB decoded components",
        })?;
    for work in component_work {
        decoded.push(finish_cuda_component_decode(work)?);
    }
    if decoded.len() != expected_components {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }
    Ok(decoded)
}

#[expect(
    clippy::too_many_lines,
    reason = "one target builder keeps RGB/RGBA plane ownership and exact final-store metadata aligned"
)]
fn build_store_targets(
    colors: Vec<CudaHtj2kColorDecodePlans>,
    decoded: &[CudaDecodedComponent],
    fmt: PixelFormat,
    layout: BatchLayout,
) -> Result<(NativeColorStoreTargets<'_>, Vec<CudaHtj2kProfileReport>), Error> {
    let mut reports = Vec::new();
    reports
        .try_reserve_exact(colors.len())
        .map_err(|_| Error::HostAllocationFailed {
            bytes: colors
                .len()
                .saturating_mul(std::mem::size_of::<CudaHtj2kProfileReport>()),
            what: "j2k CUDA exact RGB reports",
        })?;
    let rgba = matches!(
        fmt,
        PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaI16
    );
    let mut three_channel_targets = Vec::new();
    let mut four_channel_targets = Vec::new();
    if rgba {
        four_channel_targets
            .try_reserve_exact(colors.len())
            .map_err(|_| Error::HostAllocationFailed {
                bytes: colors
                    .len()
                    .saturating_mul(std::mem::size_of::<CudaJ2kStoreRgbaNativeTarget<'_>>()),
                what: "j2k CUDA exact RGBA store targets",
            })?;
    } else {
        three_channel_targets
            .try_reserve_exact(colors.len())
            .map_err(|_| Error::HostAllocationFailed {
                bytes: colors
                    .len()
                    .saturating_mul(std::mem::size_of::<CudaJ2kStoreRgbNativeTarget<'_>>()),
                what: "j2k CUDA exact RGB store targets",
            })?;
    }
    let mut component_offset = 0usize;
    for (image_index, mut color) in colors.into_iter().enumerate() {
        let component_count = color.components.len();
        let component_end =
            component_offset
                .checked_add(component_count)
                .ok_or(Error::HostAllocationFailed {
                    bytes: usize::MAX,
                    what: "j2k CUDA exact color component range",
                })?;
        let components =
            decoded
                .get(component_offset..component_end)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        component_offset = component_end;
        let transform = transform_selector(color.mct, color.transform);
        if rgba {
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
            four_channel_targets.push(CudaJ2kStoreRgbaNativeTarget {
                output_index: color.output_index,
                plane0: pooled_cuda_buffer(&components[0].buffer)?,
                plane1: pooled_cuda_buffer(&components[1].buffer)?,
                plane2: pooled_cuda_buffer(&components[2].buffer)?,
                plane3: pooled_cuda_buffer(&components[3].buffer)?,
                job: native_rgba_store_job(stores, color.bit_depths, addends, layout, transform),
            });
        } else {
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
            three_channel_targets.push(CudaJ2kStoreRgbNativeTarget {
                output_index: color.output_index,
                plane0: pooled_cuda_buffer(&components[0].buffer)?,
                plane1: pooled_cuda_buffer(&components[1].buffer)?,
                plane2: pooled_cuda_buffer(&components[2].buffer)?,
                job: native_store_job(stores, bit_depths, addends, layout, transform),
            });
        }
        color.report.dispatch_count = components
            .iter()
            .map(|component| component.dispatches)
            .sum::<usize>()
            + usize::from(image_index == 0);
        for component in components {
            component.timings.add_to_report(&mut color.report);
        }
        reports.push(color.report);
    }
    let targets = if rgba {
        NativeColorStoreTargets::Rgba(four_channel_targets)
    } else {
        NativeColorStoreTargets::Rgb(three_channel_targets)
    };
    Ok((targets, reports))
}
