// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{CudaError, Error};

#[cfg(feature = "cuda-runtime")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "conversion consumes the owned plan error to preserve its message"
)]
pub(super) fn cuda_invalid_decode_plan(error: Error) -> CudaError {
    CudaError::InvalidArgument {
        message: error.to_string(),
    }
}
