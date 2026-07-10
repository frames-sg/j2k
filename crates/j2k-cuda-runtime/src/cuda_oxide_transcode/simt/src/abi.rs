// SPDX-License-Identifier: MIT OR Apache-2.0

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Dwt97ColumnLiftQuantizeCodeblocksParams {
    pub(crate) cb_width: i32,
    pub(crate) cb_height: i32,
    pub(crate) inv_delta_low: f32,
    pub(crate) inv_delta_high: f32,
}
