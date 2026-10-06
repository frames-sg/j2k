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

#[test]
fn parallel_staging_fill_matches_serial_concatenation() {
    let parts: [&[u8]; 6] = [&[1, 2, 3], &[], &[4], &[5, 6, 7, 8, 9], &[10, 11], &[12]];
    let expected = parts.concat();
    for workers in 2..=super::MAX_STAGING_FILL_WORKERS {
        let mut target = vec![0u8; expected.len()];
        super::copy_parts_in_parallel(&mut target, &parts, workers);
        assert_eq!(target, expected, "workers = {workers}");
    }
}
