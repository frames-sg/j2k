// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;
use j2k_cuda_runtime::{CudaJ2kStoreRgbNativeTarget, CudaJ2kStoreRgbaNativeTarget};

use super::super::targets::NativeColorStoreTargets;
use crate::decoder::color_batch::cuda_error;
use crate::decoder::color_batch::native_batch::{
    NativeColorBatchOutput, NativeColorOwnedBatch, StoredNativeColorBatch,
};
use crate::decoder::{Error, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED};

pub(super) fn enqueue_owned_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    targets: &NativeColorStoreTargets<'_>,
) -> Result<StoredNativeColorBatch, Error> {
    match targets {
        NativeColorStoreTargets::Rgb(targets) => enqueue_owned_rgb_store(context, fmt, targets),
        NativeColorStoreTargets::Rgba(targets) => enqueue_owned_rgba_store(context, fmt, targets),
    }
}

fn enqueue_owned_rgb_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
) -> Result<StoredNativeColorBatch, Error> {
    match fmt {
        PixelFormat::Rgb8 => context.j2k_store_rgb8_native_batch_contiguous_device_enqueue(targets),
        PixelFormat::Rgb16 => {
            context.j2k_store_rgb16_native_batch_contiguous_device_enqueue(targets)
        }
        PixelFormat::RgbI16 => {
            context.j2k_store_rgbi16_native_batch_contiguous_device_enqueue(targets)
        }
        _ => return Err(unsupported_store()),
    }
    .map_err(cuda_error)
    .map(queued_owned_store)
}

fn enqueue_owned_rgba_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
) -> Result<StoredNativeColorBatch, Error> {
    match fmt {
        PixelFormat::Rgba8 => {
            context.j2k_store_rgba8_native_batch_contiguous_device_enqueue(targets)
        }
        PixelFormat::Rgba16 => {
            context.j2k_store_rgba16_native_batch_contiguous_device_enqueue(targets)
        }
        PixelFormat::RgbaI16 => {
            context.j2k_store_rgbai16_native_batch_contiguous_device_enqueue(targets)
        }
        _ => return Err(unsupported_store()),
    }
    .map_err(cuda_error)
    .map(queued_owned_store)
}

pub(super) fn finish_owned_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    targets: &NativeColorStoreTargets<'_>,
) -> Result<StoredNativeColorBatch, Error> {
    match targets {
        NativeColorStoreTargets::Rgb(targets) => finish_owned_rgb_store(context, fmt, targets),
        NativeColorStoreTargets::Rgba(targets) => finish_owned_rgba_store(context, fmt, targets),
    }
}

fn finish_owned_rgb_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
) -> Result<StoredNativeColorBatch, Error> {
    match fmt {
        PixelFormat::Rgb8 => context.j2k_store_rgb8_native_batch_contiguous_device(targets),
        PixelFormat::Rgb16 => context.j2k_store_rgb16_native_batch_contiguous_device(targets),
        PixelFormat::RgbI16 => context.j2k_store_rgbi16_native_batch_contiguous_device(targets),
        _ => return Err(unsupported_store()),
    }
    .map_err(cuda_error)
    .map(owned_store)
}

fn finish_owned_rgba_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
) -> Result<StoredNativeColorBatch, Error> {
    match fmt {
        PixelFormat::Rgba8 => context.j2k_store_rgba8_native_batch_contiguous_device(targets),
        PixelFormat::Rgba16 => context.j2k_store_rgba16_native_batch_contiguous_device(targets),
        PixelFormat::RgbaI16 => context.j2k_store_rgbai16_native_batch_contiguous_device(targets),
        _ => return Err(unsupported_store()),
    }
    .map_err(cuda_error)
    .map(owned_store)
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

const fn unsupported_store() -> Error {
    Error::UnsupportedCudaRequest {
        reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
    }
}
