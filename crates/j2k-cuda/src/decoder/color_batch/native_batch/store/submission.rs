// SPDX-License-Identifier: MIT OR Apache-2.0

mod external;
mod owned;

use j2k_core::PixelFormat;
use j2k_cuda_runtime::CudaExternalDeviceBufferViewMut;

use self::external::enqueue_external_store;
use self::owned::{enqueue_owned_store, finish_owned_store};
use super::targets::NativeColorStoreTargets;
use crate::decoder::color_batch::native_batch::StoredNativeColorBatch;
use crate::decoder::{Error, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED};

pub(super) fn store_targets(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    external: Option<&mut CudaExternalDeviceBufferViewMut<'_>>,
    enqueue_external: bool,
    targets: &NativeColorStoreTargets<'_>,
) -> Result<StoredNativeColorBatch, Error> {
    match (external, enqueue_external) {
        (Some(destination), true) => enqueue_external_store(context, fmt, destination, targets),
        (None, true) => enqueue_owned_store(context, fmt, targets),
        (None, false) => finish_owned_store(context, fmt, targets),
        (Some(_), false) => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}
