// SPDX-License-Identifier: MIT OR Apache-2.0

mod runtime;
mod shape;

pub(super) use self::runtime::prepared_direct_color_plan_supports_runtime;
pub(super) use self::shape::{
    classic_group_shapes_match, classic_sub_band_shapes_match, ht_group_shapes_match,
    ht_sub_band_shapes_match, idwt_shapes_match, store_shapes_match,
};

use crate::Error;

pub(super) fn direct_preflight_invariant(message: &'static str) -> Error {
    Error::MetalKernel {
        message: format!("internal J2K Metal direct preflight error: {message}"),
    }
}
