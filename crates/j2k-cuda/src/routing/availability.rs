// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CudaSession, Error};

#[cfg_attr(
    not(feature = "cuda-runtime"),
    expect(
        clippy::unnecessary_wraps,
        reason = "feature-disabled shim preserves the runtime-enabled fallible signature"
    )
)]
pub(crate) fn auto_cuda_available(session: &mut CudaSession) -> Result<bool, Error> {
    #[cfg(feature = "cuda-runtime")]
    {
        match session.cuda_context() {
            Ok(_) => Ok(true),
            Err(Error::CudaUnavailable) => Ok(false),
            Err(error) => Err(error),
        }
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        let _ = session;
        Ok(false)
    }
}
