// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::select_pinned_upload_result;
use crate::CudaError;

#[test]
fn upload_and_recycle_failures_preserve_primary_and_release_sources() {
    let error = select_pinned_upload_result::<()>(
        Err(CudaError::InvalidArgument {
            message: "upload failed".to_string(),
        }),
        Err(CudaError::StatePoisoned {
            message: "recycle failed".to_string(),
        }),
    )
    .expect_err("both failures must be returned");
    let CudaError::ResourceReleaseFailed { primary, release } = error else {
        panic!("both sources must use the compound release error");
    };
    assert!(matches!(*primary, CudaError::InvalidArgument { .. }));
    assert!(matches!(*release, CudaError::StatePoisoned { .. }));
}
