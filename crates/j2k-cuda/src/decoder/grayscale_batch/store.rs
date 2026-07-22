// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed grayscale final-store materialization.

use super::{
    cuda_error, cuda_range_storage, pooled_cuda_buffer, Arc, BackendKind, CudaDecodedComponent,
    CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut, CudaHtj2kDecodePlan,
    CudaJ2kStoreGray16Job, CudaJ2kStoreGray16Target, CudaJ2kStoreGray8Job, CudaJ2kStoreGray8Target,
    CudaJ2kStoreGrayI16Target, CudaSurfaceStats, Error, GrayscaleBatchOutput, GrayscaleOwnedBatch,
    PixelFormat, StoredGrayscaleBatch, Surface, SurfaceResidency,
};

fn gray_store_job(component: &CudaDecodedComponent) -> (u32, super::super::CudaHtj2kStoreStep) {
    let store = component.store;
    let input_width = store.input_rect.x1.saturating_sub(store.input_rect.x0);
    (input_width, store)
}

pub(super) fn store_gray8_batch(
    context: &j2k_cuda_runtime::CudaContext,
    plans: &[CudaHtj2kDecodePlan],
    output_indices: &[usize],
    output_dimensions: &[(u32, u32)],
    decoded: &[CudaDecodedComponent],
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
) -> Result<StoredGrayscaleBatch, Error> {
    let targets = plans
        .iter()
        .zip(decoded)
        .zip(output_indices.iter().copied())
        .map(|((plan, component), output_index)| {
            let (input_width, store) = gray_store_job(component);
            Ok(CudaJ2kStoreGray8Target {
                output_index,
                input: pooled_cuda_buffer(&component.buffer)?,
                job: CudaJ2kStoreGray8Job {
                    input_width,
                    source_x: store.source_x,
                    source_y: store.source_y,
                    copy_width: store.copy_width,
                    copy_height: store.copy_height,
                    output_width: store.output_width,
                    output_height: store.output_height,
                    output_x: store.output_x,
                    output_y: store.output_y,
                    addend: store.addend,
                    bit_depth: u32::from(plan.bit_depth()),
                },
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    materialize_gray8_output(
        context,
        output_dimensions,
        &targets,
        external,
        enqueue_external,
    )
}

fn materialize_gray8_output(
    context: &j2k_cuda_runtime::CudaContext,
    output_dimensions: &[(u32, u32)],
    targets: &[CudaJ2kStoreGray8Target<'_>],
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
) -> Result<StoredGrayscaleBatch, Error> {
    if let Some(destination) = external {
        if enqueue_external {
            // SAFETY: the pending high-level owner retains every decoded
            // coefficient buffer and the caller guarantees the external
            // allocation lifetime until that owner is retired.
            let (ranges, queued) = unsafe {
                context.j2k_store_gray8_batch_into_external_device_enqueue(targets, destination)
            }
            .map_err(cuda_error)?;
            return Ok(StoredGrayscaleBatch {
                output: GrayscaleBatchOutput::External(ranges),
                queued: Some(queued),
            });
        }
        // SAFETY: this synchronous path retains all decoded coefficient
        // owners and the caller-owned destination through the completion
        // boundary. Its public caller carries the quarantine-on-error contract.
        let (ranges, _) =
            unsafe { context.j2k_store_gray8_batch_into_external_device(targets, destination) }
                .map_err(cuda_error)?;
        return Ok(StoredGrayscaleBatch {
            output: GrayscaleBatchOutput::External(ranges),
            queued: None,
        });
    }
    if enqueue_external {
        let (buffer, ranges, queued) = context
            .j2k_store_gray8_batch_contiguous_device_enqueue(targets)
            .map_err(cuda_error)?;
        let stats = queued.execution();
        return Ok(StoredGrayscaleBatch {
            output: GrayscaleBatchOutput::Owned(owned_batch_from_contiguous(
                output_dimensions,
                PixelFormat::Gray8,
                buffer,
                ranges,
                stats,
            )),
            queued: Some(queued),
        });
    }
    let output = context
        .j2k_store_gray8_batch_contiguous_device(targets)
        .map_err(cuda_error)?;
    let (buffer, ranges, stats) = output.into_parts();
    Ok(StoredGrayscaleBatch {
        output: GrayscaleBatchOutput::Owned(owned_batch_from_contiguous(
            output_dimensions,
            PixelFormat::Gray8,
            buffer,
            ranges,
            stats,
        )),
        queued: None,
    })
}

pub(super) fn store_gray16_batch(
    context: &j2k_cuda_runtime::CudaContext,
    plans: &[CudaHtj2kDecodePlan],
    output_indices: &[usize],
    output_dimensions: &[(u32, u32)],
    decoded: &[CudaDecodedComponent],
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
) -> Result<StoredGrayscaleBatch, Error> {
    let targets = plans
        .iter()
        .zip(decoded)
        .zip(output_indices.iter().copied())
        .map(|((plan, component), output_index)| {
            let (input_width, store) = gray_store_job(component);
            Ok(CudaJ2kStoreGray16Target {
                output_index,
                input: pooled_cuda_buffer(&component.buffer)?,
                job: CudaJ2kStoreGray16Job {
                    input_width,
                    source_x: store.source_x,
                    source_y: store.source_y,
                    copy_width: store.copy_width,
                    copy_height: store.copy_height,
                    output_width: store.output_width,
                    output_height: store.output_height,
                    output_x: store.output_x,
                    output_y: store.output_y,
                    addend: store.addend,
                    bit_depth: u32::from(plan.bit_depth()),
                },
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if let Some(destination) = external {
        if enqueue_external {
            // SAFETY: the pending high-level owner retains every decoded
            // coefficient buffer and the caller guarantees the external
            // allocation lifetime until that owner is retired.
            let (ranges, queued) = unsafe {
                context.j2k_store_gray16_batch_into_external_device_enqueue(&targets, destination)
            }
            .map_err(cuda_error)?;
            return Ok(StoredGrayscaleBatch {
                output: GrayscaleBatchOutput::External(ranges),
                queued: Some(queued),
            });
        }
        // SAFETY: this synchronous path retains all decoded coefficient
        // owners and the caller-owned destination through the completion
        // boundary. Its public caller carries the quarantine-on-error contract.
        let (ranges, _) =
            unsafe { context.j2k_store_gray16_batch_into_external_device(&targets, destination) }
                .map_err(cuda_error)?;
        return Ok(StoredGrayscaleBatch {
            output: GrayscaleBatchOutput::External(ranges),
            queued: None,
        });
    }
    if enqueue_external {
        let (buffer, ranges, queued) = context
            .j2k_store_gray16_batch_contiguous_device_enqueue(&targets)
            .map_err(cuda_error)?;
        let stats = queued.execution();
        return Ok(StoredGrayscaleBatch {
            output: GrayscaleBatchOutput::Owned(owned_batch_from_contiguous(
                output_dimensions,
                PixelFormat::Gray16,
                buffer,
                ranges,
                stats,
            )),
            queued: Some(queued),
        });
    }
    let output = context
        .j2k_store_gray16_batch_contiguous_device(&targets)
        .map_err(cuda_error)?;
    let (buffer, ranges, stats) = output.into_parts();
    Ok(StoredGrayscaleBatch {
        output: GrayscaleBatchOutput::Owned(owned_batch_from_contiguous(
            output_dimensions,
            PixelFormat::Gray16,
            buffer,
            ranges,
            stats,
        )),
        queued: None,
    })
}

pub(super) fn store_grayi16_batch(
    context: &j2k_cuda_runtime::CudaContext,
    plans: &[CudaHtj2kDecodePlan],
    output_indices: &[usize],
    output_dimensions: &[(u32, u32)],
    decoded: &[CudaDecodedComponent],
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
) -> Result<StoredGrayscaleBatch, Error> {
    let targets = plans
        .iter()
        .zip(decoded)
        .zip(output_indices.iter().copied())
        .map(|((plan, component), output_index)| {
            let (input_width, store) = gray_store_job(component);
            Ok(CudaJ2kStoreGrayI16Target {
                output_index,
                input: pooled_cuda_buffer(&component.buffer)?,
                job: CudaJ2kStoreGray16Job {
                    input_width,
                    source_x: store.source_x,
                    source_y: store.source_y,
                    copy_width: store.copy_width,
                    copy_height: store.copy_height,
                    output_width: store.output_width,
                    output_height: store.output_height,
                    output_x: store.output_x,
                    output_y: store.output_y,
                    addend: store.addend,
                    bit_depth: u32::from(plan.bit_depth()),
                },
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if let Some(destination) = external {
        if enqueue_external {
            // SAFETY: the pending high-level owner retains every decoded
            // coefficient buffer and the caller guarantees the external
            // allocation lifetime until that owner is retired.
            let (ranges, queued) = unsafe {
                context.j2k_store_grayi16_batch_into_external_device_enqueue(&targets, destination)
            }
            .map_err(cuda_error)?;
            return Ok(StoredGrayscaleBatch {
                output: GrayscaleBatchOutput::External(ranges),
                queued: Some(queued),
            });
        }
        // SAFETY: this synchronous path retains all decoded coefficient
        // owners and the caller-owned destination through the completion
        // boundary. Its public caller carries the quarantine-on-error contract.
        let (ranges, _) =
            unsafe { context.j2k_store_grayi16_batch_into_external_device(&targets, destination) }
                .map_err(cuda_error)?;
        return Ok(StoredGrayscaleBatch {
            output: GrayscaleBatchOutput::External(ranges),
            queued: None,
        });
    }
    if enqueue_external {
        let (buffer, ranges, queued) = context
            .j2k_store_grayi16_batch_contiguous_device_enqueue(&targets)
            .map_err(cuda_error)?;
        let stats = queued.execution();
        return Ok(StoredGrayscaleBatch {
            output: GrayscaleBatchOutput::Owned(owned_batch_from_contiguous(
                output_dimensions,
                PixelFormat::GrayI16,
                buffer,
                ranges,
                stats,
            )),
            queued: Some(queued),
        });
    }
    let output = context
        .j2k_store_grayi16_batch_contiguous_device(&targets)
        .map_err(cuda_error)?;
    let (buffer, ranges, stats) = output.into_parts();
    Ok(StoredGrayscaleBatch {
        output: GrayscaleBatchOutput::Owned(owned_batch_from_contiguous(
            output_dimensions,
            PixelFormat::GrayI16,
            buffer,
            ranges,
            stats,
        )),
        queued: None,
    })
}

fn owned_batch_from_contiguous(
    output_dimensions: &[(u32, u32)],
    fmt: PixelFormat,
    buffer: j2k_cuda_runtime::CudaDeviceBuffer,
    ranges: Vec<CudaDeviceBufferRange>,
    stats: j2k_cuda_runtime::CudaExecutionStats,
) -> GrayscaleOwnedBatch {
    let shared = Arc::new(buffer);
    let surfaces = output_dimensions
        .iter()
        .zip(ranges.iter().copied())
        .map(|(&dimensions, range)| Surface {
            backend: BackendKind::Cuda,
            residency: SurfaceResidency::CudaResidentDecode,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            stats: CudaSurfaceStats {
                total: stats.kernel_dispatches(),
                copy: stats.copy_kernel_dispatches(),
                decode: stats.decode_kernel_dispatches(),
            },
            storage: cuda_range_storage(shared.clone(), range.offset, range.len),
        })
        .collect();
    GrayscaleOwnedBatch {
        surfaces,
        buffer: shared,
        ranges,
    }
}
