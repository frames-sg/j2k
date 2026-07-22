// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::PreparedBatchGroup;
use j2k_cuda::{CudaBatchDecoder, SubmittedCudaExternalBatch};
use j2k_cuda_runtime::{CudaContext, CudaDeviceBuffer, CudaExternalDeviceBufferViewMut};

pub(super) unsafe fn submit_external_for_test(
    decoder: &mut CudaBatchDecoder,
    group: &PreparedBatchGroup,
    context: &CudaContext,
    allocation: &mut CudaDeviceBuffer,
) -> SubmittedCudaExternalBatch {
    let ptr = allocation.device_ptr();
    let len = allocation.byte_len();
    // SAFETY: the caller guarantees `allocation` stays live and inaccessible
    // until the returned completion owner is retired.
    let mut destination = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(
            context,
            ptr,
            len,
            std::mem::align_of::<u8>(),
            allocation,
        )
    }
    .expect("external destination view");
    // SAFETY: forwarded from this helper's caller; the view exclusively
    // borrows the live allocation for submission.
    unsafe { decoder.submit_batch_into(group, &mut destination) }
        .expect("submit external test batch")
}
