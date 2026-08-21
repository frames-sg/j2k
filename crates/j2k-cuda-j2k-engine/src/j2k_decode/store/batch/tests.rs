// SPDX-License-Identifier: MIT OR Apache-2.0

use super::ensure_internal_count;
use crate::error::CudaError;

#[test]
fn batch_materialization_mismatch_is_a_typed_error() {
    assert!(ensure_internal_count(3, 3, "matching counts").is_ok());
    assert!(matches!(
        ensure_internal_count(2, 3, "fixture mismatch"),
        Err(CudaError::InternalInvariant {
            what: "fixture mismatch"
        })
    ));
}
