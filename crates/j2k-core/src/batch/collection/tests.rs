// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{format, string::ToString};
use core::{convert::Infallible, error::Error};

use super::{ensure_collection_live_bytes, try_ordered_vec};
use crate::{BatchDecodeError, BatchInfrastructureError};

#[test]
fn collection_byte_overflow_is_a_saturated_cap_error() {
    let cap = 37;
    let error = ensure_collection_live_bytes::<Infallible>(
        0,
        usize::MAX,
        1,
        0,
        usize::MAX,
        cap,
        "ordered batch collection",
    )
    .expect_err("collection byte overflow must fail before allocation");

    assert!(error.tile_error().is_none());
    assert_eq!(
        error.infrastructure_error(),
        Some(&BatchInfrastructureError::AllocationTooLarge {
            what: "ordered batch collection",
            requested: usize::MAX,
            cap,
        })
    );
    assert_eq!(
        error.to_string(),
        format!(
            "ordered batch collection is too large: requested {} bytes, cap {cap}",
            usize::MAX
        )
    );
    let source = Error::source(&error).expect("infrastructure source");
    assert_eq!(
        source.downcast_ref::<BatchInfrastructureError>(),
        error.infrastructure_error()
    );
    assert!(Error::source(
        source
            .downcast_ref::<BatchInfrastructureError>()
            .expect("typed infrastructure source")
    )
    .is_none());
}

#[test]
fn impossible_ordered_capacity_is_an_allocator_error_not_a_cap_error() {
    let error = try_ordered_vec::<u8, Infallible>(usize::MAX, "ordered batch results")
        .expect_err("impossible vector capacity must fail fallibly");

    assert!(error.tile_error().is_none());
    assert_eq!(
        error.infrastructure_error(),
        Some(&BatchInfrastructureError::HostAllocationFailed {
            what: "ordered batch results",
            bytes: usize::MAX,
        })
    );
    assert_eq!(
        error.to_string(),
        format!(
            "host allocation failed for {} bytes while allocating ordered batch results",
            usize::MAX
        )
    );
    assert!(!matches!(
        error,
        BatchDecodeError::Infrastructure(BatchInfrastructureError::AllocationTooLarge { .. })
    ));
    let source = Error::source(&error).expect("infrastructure source");
    assert!(source
        .downcast_ref::<BatchInfrastructureError>()
        .is_some_and(|error| matches!(
            error,
            BatchInfrastructureError::HostAllocationFailed {
                what: "ordered batch results",
                bytes: usize::MAX,
            }
        )));
}
