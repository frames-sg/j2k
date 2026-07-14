// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Surface;
use crate::Error;
#[cfg(target_os = "macos")]
use metal::foreign_types::ForeignTypeRef;

/// Read a group of completed Metal-resident surfaces into one tightly packed
/// host allocation using a single Metal staging buffer.
///
/// Surface order is preserved. Every surface must have been produced on the
/// supplied session's device; host surfaces are rejected rather than copied.
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn download_surfaces_packed(
    session: &crate::MetalBackendSession,
    surfaces: &[&Surface],
) -> Result<Vec<u8>, Error> {
    use j2k_core::DeviceSurface;
    use j2k_metal_support::{
        checked_blit_command_encoder, checked_buffer_read_vec, checked_command_buffer,
        checked_command_queue, checked_shared_buffer, commit_and_wait,
    };

    let total = surfaces.iter().try_fold(0usize, |total, surface| {
        total
            .checked_add(surface.byte_len())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal packed surface readback size overflow".to_string(),
            })
    })?;
    let map_support = |error: j2k_metal_support::MetalSupportError| Error::MetalKernel {
        message: format!("J2K Metal packed surface readback failed: {error}"),
    };
    let staging = checked_shared_buffer(session.device(), total).map_err(map_support)?;
    let queue = checked_command_queue(session.device()).map_err(map_support)?;
    let command = checked_command_buffer(&queue).map_err(map_support)?;
    let blit = checked_blit_command_encoder(&command).map_err(map_support)?;
    let mut destination_offset = 0usize;
    for surface in surfaces {
        let (buffer, source_offset) =
            surface
                .metal_buffer_trusted()
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal packed surface readback received a host surface"
                        .to_string(),
                })?;
        if buffer.device().as_ptr() != session.device().as_ptr() {
            return Err(Error::MetalKernel {
                message: "J2K Metal packed surface belongs to a different device".to_string(),
            });
        }
        let len = surface.byte_len();
        blit.copy_from_buffer(
            buffer,
            u64::try_from(source_offset).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packed source offset exceeds u64".to_string(),
            })?,
            &staging,
            u64::try_from(destination_offset).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packed destination offset exceeds u64".to_string(),
            })?,
            u64::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packed surface length exceeds u64".to_string(),
            })?,
        );
        destination_offset =
            destination_offset
                .checked_add(len)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal packed destination offset overflow".to_string(),
                })?;
    }
    blit.end_encoding();
    commit_and_wait(&command).map_err(map_support)?;
    // SAFETY: the blit completed above and the local staging buffer has no
    // overlapping writer for the duration of this owned copy.
    unsafe { checked_buffer_read_vec::<u8>(&staging, 0, total) }.map_err(map_support)
}

/// Return `MetalUnavailable` on platforms without Metal support.
#[cfg(not(target_os = "macos"))]
#[doc(hidden)]
pub fn download_surfaces_packed(
    _session: &crate::MetalBackendSession,
    _surfaces: &[&Surface],
) -> Result<Vec<u8>, Error> {
    Err(Error::MetalUnavailable)
}
