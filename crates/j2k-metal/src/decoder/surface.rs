// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    checked_surface_len, BackendKind, BackendRequest, PixelFormat,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
#[cfg(target_os = "macos")]
use metal::MTLResourceOptions;

use crate::{Error, Storage, Surface, SurfaceResidency};

#[cfg(target_os = "macos")]
pub(super) const CPU_STAGED_METAL_REQUIRES_EXPLICIT_API: &str =
    "CPU-staged Metal upload requires the explicit CPU-staged API; BackendRequest::Metal only accepts resident Metal decode";

pub(super) fn allocate_cpu_surface(
    dims: (u32, u32),
    fmt: PixelFormat,
) -> Result<(Vec<u8>, usize), Error> {
    let (stride, len) = checked_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        "j2k Metal CPU fallback surface",
    )?;
    Ok((vec![0u8; len], stride))
}

pub(super) fn upload_surface(
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    match backend {
        BackendRequest::Cpu | BackendRequest::Auto => Ok(Surface {
            backend: BackendKind::Cpu,
            residency: SurfaceResidency::Host,
            dimensions,
            fmt,
            pitch_bytes,
            byte_offset: 0,
            storage: Storage::Host(bytes),
        }),
        BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                let _ = bytes;
                Err(Error::UnsupportedMetalRequest {
                    reason: CPU_STAGED_METAL_REQUIRES_EXPLICIT_API,
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = bytes;
                Err(Error::MetalUnavailable)
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

#[cfg(target_os = "macos")]
pub(super) fn upload_surface_to_metal_with_device(
    bytes: &[u8],
    dimensions: (u32, u32),
    fmt: PixelFormat,
    device: &metal::DeviceRef,
) -> Surface {
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    let buffer = device.new_buffer_with_data(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    Surface {
        backend: BackendKind::Metal,
        residency: SurfaceResidency::CpuStagedMetalUpload,
        dimensions,
        fmt,
        pitch_bytes,
        byte_offset: 0,
        storage: Storage::Metal(buffer),
    }
}
