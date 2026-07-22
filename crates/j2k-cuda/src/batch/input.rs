// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared-input validation and CUDA adapter mapping.

#[cfg(feature = "cuda-runtime")]
use super::{
    cuda_range_storage, Arc, BackendKind, CudaResidentBatchBuffer, CudaSurfaceStats,
    PreparedBatchGroup, Surface, SurfaceResidency,
};
use super::{
    BatchColor, BatchDecodeOptions, BatchGroupInfo, BatchLayout, Error, J2kDecodeWarning,
    PixelFormat,
};

pub(super) fn group_pixel_format(info: &BatchGroupInfo) -> Result<PixelFormat, Error> {
    info.native_pixel_format()
        .ok_or(Error::UnsupportedCudaRequest {
            reason: "CUDA batch output color/sample type is unsupported",
        })
}

pub(super) fn validate_layout(info: &BatchGroupInfo) -> Result<(), Error> {
    if !matches!(info.layout, BatchLayout::Nchw | BatchLayout::Nhwc) {
        return Err(Error::UnsupportedCudaRequest {
            reason: "CUDA batch output layout is unsupported",
        });
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn native_decode_settings(settings: j2k::DecodeSettings) -> j2k_native::DecodeSettings {
    j2k_native::DecodeSettings {
        resolve_palette_indices: true,
        strict: settings.is_strict(),
        target_resolution: None,
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn native_referenced_htj2k_plan(
    plan: &j2k::PreparedHtj2kPlan,
) -> Result<&j2k_native::J2kReferencedHtj2kPlan, Error> {
    plan.adapter_view()
        .downcast_ref::<j2k_native::J2kReferencedHtj2kPlan>()
        .ok_or(Error::UnsupportedCudaRequest {
            reason: "prepared HTJ2K plan is not compatible with the CUDA adapter",
        })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn native_referenced_classic_plan(
    plan: &j2k::PreparedClassicPlan,
) -> Result<&j2k_native::J2kReferencedClassicPlan, Error> {
    plan.adapter_view()
        .downcast_ref::<j2k_native::J2kReferencedClassicPlan>()
        .ok_or(Error::UnsupportedCudaRequest {
            reason: "prepared classic plan is not compatible with the CUDA adapter",
        })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn native_color_inputs(
    group: &PreparedBatchGroup,
) -> Result<Vec<crate::decoder::NativeColorBatchInput<'_>>, Error> {
    group
        .images()
        .iter()
        .zip(group.source_indices().iter().copied())
        .map(|(image, source_index)| {
            let referenced_plan = match image.htj2k_plan() {
                Some(prepared_plan)
                    if group.info().color == BatchColor::Rgb && prepared_plan.is_color() =>
                {
                    Some(native_referenced_htj2k_plan(prepared_plan)?)
                }
                Some(prepared_plan)
                    if group.info().color == BatchColor::Rgba && prepared_plan.is_rgba() =>
                {
                    Some(native_referenced_htj2k_plan(prepared_plan)?)
                }
                Some(_) => {
                    return Err(Error::UnsupportedCudaRequest {
                        reason: "exact CUDA color batch received incompatible prepared geometry",
                    })
                }
                None => None,
            };
            let referenced_classic_plan = match image.classic_plan() {
                Some(prepared_plan)
                    if group.info().color == BatchColor::Rgb && prepared_plan.is_color() =>
                {
                    Some(native_referenced_classic_plan(prepared_plan)?)
                }
                Some(prepared_plan)
                    if group.info().color == BatchColor::Rgba && prepared_plan.is_rgba() =>
                {
                    Some(native_referenced_classic_plan(prepared_plan)?)
                }
                Some(_) => {
                    return Err(Error::UnsupportedCudaRequest {
                        reason: "exact CUDA color batch received incompatible classic geometry",
                    });
                }
                None => None,
            };
            if referenced_plan.is_none() && referenced_classic_plan.is_none() {
                return Err(Error::UnsupportedCudaRequest {
                    reason: "exact CUDA color batch requires a supported prepared device plan",
                });
            }
            Ok(crate::decoder::NativeColorBatchInput {
                source_index,
                bytes: image.bytes().as_ref(),
                device_plan: image.plan(),
                referenced_plan,
                referenced_classic_plan,
                settings: native_decode_settings(group.options().settings),
            })
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn native_color_group_storage(
    info: &BatchGroupInfo,
    fmt: PixelFormat,
    output: crate::decoder::NativeColorOwnedBatch,
) -> (Vec<Surface>, CudaResidentBatchBuffer) {
    let crate::decoder::NativeColorOwnedBatch {
        buffer,
        ranges,
        execution,
    } = output;
    let shared = Arc::new(buffer);
    let surfaces = if info.layout == BatchLayout::Nhwc {
        ranges
            .iter()
            .map(|range| Surface {
                backend: BackendKind::Cuda,
                residency: SurfaceResidency::CudaResidentDecode,
                dimensions: info.dimensions,
                fmt,
                pitch_bytes: info.dimensions.0 as usize * fmt.bytes_per_pixel(),
                stats: CudaSurfaceStats {
                    total: execution.kernel_dispatches(),
                    copy: execution.copy_kernel_dispatches(),
                    decode: execution.decode_kernel_dispatches(),
                },
                storage: cuda_range_storage(shared.clone(), range.offset, range.len),
            })
            .collect()
    } else {
        Vec::new()
    };
    (
        surfaces,
        CudaResidentBatchBuffer {
            buffer: shared,
            ranges,
        },
    )
}

pub(super) fn decode_warnings(
    options: BatchDecodeOptions,
    count: usize,
) -> Vec<Vec<J2kDecodeWarning>> {
    (0..count)
        .map(|_| {
            if options.settings.lenient_tolerance_enabled() {
                vec![J2kDecodeWarning::LenientDecodeMode]
            } else {
                Vec::new()
            }
        })
        .collect()
}
