// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::Error;

#[cfg(target_os = "macos")]
pub(in crate::compute) fn checked_u32(value: usize, label: &str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::MetalKernel {
        message: format!("JPEG Metal {label} does not fit in u32"),
    })
}
