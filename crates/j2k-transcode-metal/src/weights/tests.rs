// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Dwt53WeightRows, Dwt97WeightRows, SparseDwt53WeightRows, SparseDwt97WeightRows,
    SparseWeightRowsError,
};
use core::mem::size_of;

#[test]
fn sparse_weight_rows_reject_impossible_axes_before_allocation() {
    assert_eq!(
        SparseDwt53WeightRows::for_len(usize::MAX),
        Err(SparseWeightRowsError::SizeOverflow)
    );
    assert_eq!(
        SparseDwt97WeightRows::for_len(usize::MAX),
        Err(SparseWeightRowsError::SizeOverflow)
    );
    assert_eq!(
        Dwt53WeightRows::for_len(usize::MAX),
        Err(SparseWeightRowsError::SizeOverflow)
    );
    assert_eq!(
        Dwt97WeightRows::for_len(usize::MAX),
        Err(SparseWeightRowsError::SizeOverflow)
    );
    assert!(matches!(
        Dwt97WeightRows::for_len(20_000),
        Err(SparseWeightRowsError::AllocationTooLarge { .. })
    ));
    let over_cap_sparse_axis = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
        / (size_of::<super::SparseWeightRow>() + 16 * size_of::<super::SparseWeightTap>())
        + 1;
    assert!(matches!(
        SparseDwt97WeightRows::for_len(over_cap_sparse_axis),
        Err(SparseWeightRowsError::AllocationTooLarge { .. })
    ));
}

#[test]
fn sparse_weight_rows_handle_large_axes_without_dense_workspace() {
    let axis = 16_384;
    let rows53 = SparseDwt53WeightRows::for_len(axis).expect("bounded large 5/3 axis");
    let rows97 = SparseDwt97WeightRows::for_len(axis).expect("bounded large 9/7 axis");

    assert_eq!(rows53.low.len() + rows53.high.len(), axis);
    assert_eq!(rows97.low.len() + rows97.high.len(), axis);
    assert!(rows53.max_taps_per_row() <= 5);
    assert!(rows97.max_taps_per_row() <= 16);
}
