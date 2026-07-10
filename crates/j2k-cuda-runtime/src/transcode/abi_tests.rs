// SPDX-License-Identifier: MIT OR Apache-2.0

use super::types::Dwt97ColumnLiftQuantizeCodeblocksParams;

#[test]
fn fused_column_quantize_cuda_abi_layout_remains_stable() {
    use core::mem::{align_of, offset_of, size_of};

    assert_eq!(size_of::<Dwt97ColumnLiftQuantizeCodeblocksParams>(), 16);
    assert_eq!(align_of::<Dwt97ColumnLiftQuantizeCodeblocksParams>(), 4);
    assert_eq!(
        offset_of!(Dwt97ColumnLiftQuantizeCodeblocksParams, cb_height),
        4
    );
    assert_eq!(
        offset_of!(Dwt97ColumnLiftQuantizeCodeblocksParams, inv_delta_low),
        8
    );
    assert_eq!(
        offset_of!(Dwt97ColumnLiftQuantizeCodeblocksParams, inv_delta_high),
        12
    );
}
