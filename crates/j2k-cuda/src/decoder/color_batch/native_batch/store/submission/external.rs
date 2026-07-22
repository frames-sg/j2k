// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;
use j2k_cuda_runtime::{
    CudaExternalDeviceBufferViewMut, CudaJ2kStoreRgbNativeTarget, CudaJ2kStoreRgbaNativeTarget,
};

use super::super::targets::NativeColorStoreTargets;
use crate::decoder::color_batch::cuda_error;
use crate::decoder::color_batch::native_batch::{NativeColorBatchOutput, StoredNativeColorBatch};
use crate::decoder::{Error, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED};

pub(super) fn enqueue_external_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    targets: &NativeColorStoreTargets<'_>,
) -> Result<StoredNativeColorBatch, Error> {
    match targets {
        NativeColorStoreTargets::Rgb(targets) => {
            enqueue_external_rgb_store(context, fmt, destination, targets)
        }
        NativeColorStoreTargets::Rgba(targets) => {
            enqueue_external_rgba_store(context, fmt, destination, targets)
        }
    }
}

fn enqueue_external_rgb_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
) -> Result<StoredNativeColorBatch, Error> {
    // SAFETY: the returned high-level owner retains decoded planes, uploaded
    // store metadata, and the caller-owned destination through completion.
    let result = unsafe {
        match fmt {
            PixelFormat::Rgb8 => context
                .j2k_store_rgb8_native_batch_into_external_device_enqueue(targets, destination),
            PixelFormat::Rgb16 => context
                .j2k_store_rgb16_native_batch_into_external_device_enqueue(targets, destination),
            PixelFormat::RgbI16 => context
                .j2k_store_rgbi16_native_batch_into_external_device_enqueue(targets, destination),
            _ => return Err(unsupported_store()),
        }
    };
    let (ranges, queued) = result.map_err(cuda_error)?;
    Ok(StoredNativeColorBatch {
        output: NativeColorBatchOutput::External(ranges),
        queued: Some(queued),
    })
}

fn enqueue_external_rgba_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
) -> Result<StoredNativeColorBatch, Error> {
    // SAFETY: the pending owner retains decoded planes, uploaded store
    // metadata, and the caller-owned destination through completion.
    let result = unsafe {
        match fmt {
            PixelFormat::Rgba8 => context
                .j2k_store_rgba8_native_batch_into_external_device_enqueue(targets, destination),
            PixelFormat::Rgba16 => context
                .j2k_store_rgba16_native_batch_into_external_device_enqueue(targets, destination),
            PixelFormat::RgbaI16 => context
                .j2k_store_rgbai16_native_batch_into_external_device_enqueue(targets, destination),
            _ => return Err(unsupported_store()),
        }
    };
    let (ranges, queued) = result.map_err(cuda_error)?;
    Ok(StoredNativeColorBatch {
        output: NativeColorBatchOutput::External(ranges),
        queued: Some(queued),
    })
}

const fn unsupported_store() -> Error {
    Error::UnsupportedCudaRequest {
        reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
    }
}
