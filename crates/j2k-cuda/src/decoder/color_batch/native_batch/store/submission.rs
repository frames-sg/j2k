// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;
use j2k_cuda_runtime::CudaExternalDeviceBufferViewMut;

use super::NativeColorStoreTargets;
use crate::decoder::color_batch::cuda_error;
use crate::decoder::color_batch::native_batch::{
    NativeColorBatchOutput, NativeColorOwnedBatch, StoredNativeColorBatch,
};
use crate::decoder::{Error, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED};

#[expect(
    clippy::too_many_lines,
    reason = "the exhaustive dtype/channel/ownership dispatch matrix is one auditable final-store boundary"
)]
pub(super) fn store_targets(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
    targets: &NativeColorStoreTargets<'_>,
) -> Result<StoredNativeColorBatch, Error> {
    match (fmt, external, targets) {
        (PixelFormat::Rgb8, Some(destination), NativeColorStoreTargets::Rgb(targets))
            if enqueue_external =>
        {
            // SAFETY: the returned high-level owner retains decoded planes,
            // uploaded store metadata, and the caller-owned destination.
            let (ranges, queued) = unsafe {
                context
                    .j2k_store_rgb8_native_batch_into_external_device_enqueue(targets, destination)
            }
            .map_err(cuda_error)?;
            Ok(StoredNativeColorBatch {
                output: NativeColorBatchOutput::External(ranges),
                queued: Some(queued),
            })
        }
        (PixelFormat::Rgb16, Some(destination), NativeColorStoreTargets::Rgb(targets))
            if enqueue_external =>
        {
            // SAFETY: same retained ownership as the U8 branch.
            let (ranges, queued) = unsafe {
                context
                    .j2k_store_rgb16_native_batch_into_external_device_enqueue(targets, destination)
            }
            .map_err(cuda_error)?;
            Ok(StoredNativeColorBatch {
                output: NativeColorBatchOutput::External(ranges),
                queued: Some(queued),
            })
        }
        (PixelFormat::RgbI16, Some(destination), NativeColorStoreTargets::Rgb(targets))
            if enqueue_external =>
        {
            // SAFETY: same retained ownership as the unsigned branches.
            let (ranges, queued) = unsafe {
                context.j2k_store_rgbi16_native_batch_into_external_device_enqueue(
                    targets,
                    destination,
                )
            }
            .map_err(cuda_error)?;
            Ok(StoredNativeColorBatch {
                output: NativeColorBatchOutput::External(ranges),
                queued: Some(queued),
            })
        }
        (PixelFormat::Rgba8, Some(destination), NativeColorStoreTargets::Rgba(targets))
            if enqueue_external =>
        {
            // SAFETY: the pending owner retains decoded planes and metadata.
            let (ranges, queued) = unsafe {
                context
                    .j2k_store_rgba8_native_batch_into_external_device_enqueue(targets, destination)
            }
            .map_err(cuda_error)?;
            Ok(StoredNativeColorBatch {
                output: NativeColorBatchOutput::External(ranges),
                queued: Some(queued),
            })
        }
        (PixelFormat::Rgba16, Some(destination), NativeColorStoreTargets::Rgba(targets))
            if enqueue_external =>
        {
            // SAFETY: the pending owner retains decoded planes and metadata.
            let (ranges, queued) = unsafe {
                context.j2k_store_rgba16_native_batch_into_external_device_enqueue(
                    targets,
                    destination,
                )
            }
            .map_err(cuda_error)?;
            Ok(StoredNativeColorBatch {
                output: NativeColorBatchOutput::External(ranges),
                queued: Some(queued),
            })
        }
        (PixelFormat::RgbaI16, Some(destination), NativeColorStoreTargets::Rgba(targets))
            if enqueue_external =>
        {
            // SAFETY: the pending owner retains decoded planes and metadata.
            let (ranges, queued) = unsafe {
                context.j2k_store_rgbai16_native_batch_into_external_device_enqueue(
                    targets,
                    destination,
                )
            }
            .map_err(cuda_error)?;
            Ok(StoredNativeColorBatch {
                output: NativeColorBatchOutput::External(ranges),
                queued: Some(queued),
            })
        }
        (PixelFormat::Rgb8, None, NativeColorStoreTargets::Rgb(targets)) if enqueue_external => {
            context
                .j2k_store_rgb8_native_batch_contiguous_device_enqueue(targets)
                .map_err(cuda_error)
                .map(queued_owned_store)
        }
        (PixelFormat::Rgb16, None, NativeColorStoreTargets::Rgb(targets)) if enqueue_external => {
            context
                .j2k_store_rgb16_native_batch_contiguous_device_enqueue(targets)
                .map_err(cuda_error)
                .map(queued_owned_store)
        }
        (PixelFormat::RgbI16, None, NativeColorStoreTargets::Rgb(targets)) if enqueue_external => {
            context
                .j2k_store_rgbi16_native_batch_contiguous_device_enqueue(targets)
                .map_err(cuda_error)
                .map(queued_owned_store)
        }
        (PixelFormat::Rgba8, None, NativeColorStoreTargets::Rgba(targets)) if enqueue_external => {
            context
                .j2k_store_rgba8_native_batch_contiguous_device_enqueue(targets)
                .map_err(cuda_error)
                .map(queued_owned_store)
        }
        (PixelFormat::Rgba16, None, NativeColorStoreTargets::Rgba(targets)) if enqueue_external => {
            context
                .j2k_store_rgba16_native_batch_contiguous_device_enqueue(targets)
                .map_err(cuda_error)
                .map(queued_owned_store)
        }
        (PixelFormat::RgbaI16, None, NativeColorStoreTargets::Rgba(targets))
            if enqueue_external =>
        {
            context
                .j2k_store_rgbai16_native_batch_contiguous_device_enqueue(targets)
                .map_err(cuda_error)
                .map(queued_owned_store)
        }
        (PixelFormat::Rgb8, None, NativeColorStoreTargets::Rgb(targets)) => context
            .j2k_store_rgb8_native_batch_contiguous_device(targets)
            .map_err(cuda_error)
            .map(owned_store),
        (PixelFormat::Rgb16, None, NativeColorStoreTargets::Rgb(targets)) => context
            .j2k_store_rgb16_native_batch_contiguous_device(targets)
            .map_err(cuda_error)
            .map(owned_store),
        (PixelFormat::RgbI16, None, NativeColorStoreTargets::Rgb(targets)) => context
            .j2k_store_rgbi16_native_batch_contiguous_device(targets)
            .map_err(cuda_error)
            .map(owned_store),
        (PixelFormat::Rgba8, None, NativeColorStoreTargets::Rgba(targets)) => context
            .j2k_store_rgba8_native_batch_contiguous_device(targets)
            .map_err(cuda_error)
            .map(owned_store),
        (PixelFormat::Rgba16, None, NativeColorStoreTargets::Rgba(targets)) => context
            .j2k_store_rgba16_native_batch_contiguous_device(targets)
            .map_err(cuda_error)
            .map(owned_store),
        (PixelFormat::RgbaI16, None, NativeColorStoreTargets::Rgba(targets)) => context
            .j2k_store_rgbai16_native_batch_contiguous_device(targets)
            .map_err(cuda_error)
            .map(owned_store),
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

fn queued_owned_store(
    (buffer, ranges, queued): (
        j2k_cuda_runtime::CudaDeviceBuffer,
        Vec<j2k_cuda_runtime::CudaDeviceBufferRange>,
        j2k_cuda_runtime::CudaQueuedJ2kStoreBatch,
    ),
) -> StoredNativeColorBatch {
    let execution = queued.execution();
    StoredNativeColorBatch {
        output: NativeColorBatchOutput::Owned(NativeColorOwnedBatch {
            buffer,
            ranges,
            execution,
        }),
        queued: Some(queued),
    }
}

fn owned_store(
    output: j2k_cuda_runtime::CudaKernelContiguousBatchOutput,
) -> StoredNativeColorBatch {
    let (buffer, ranges, execution) = output.into_parts();
    StoredNativeColorBatch {
        output: NativeColorBatchOutput::Owned(NativeColorOwnedBatch {
            buffer,
            ranges,
            execution,
        }),
        queued: None,
    }
}
