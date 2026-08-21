// SPDX-License-Identifier: MIT OR Apache-2.0

use super::rejection::AutoCudaRejection;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct AutoRouteObservation {
    pub(super) promoted: bool,
    pub(super) rejection: Option<AutoCudaRejection>,
}

pub(super) const fn observe(
    promoted: bool,
    rejection: Option<AutoCudaRejection>,
) -> AutoRouteObservation {
    AutoRouteObservation {
        promoted,
        rejection,
    }
}
