// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) const fn metal_is_compiled() -> bool {
    cfg!(target_os = "macos")
}
