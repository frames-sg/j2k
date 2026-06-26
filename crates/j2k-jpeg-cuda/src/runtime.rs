// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendKind, BackendRequest, PixelFormat};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaError;

use crate::surface::Storage;
use crate::{CudaSession, CudaSurfaceStats, Error, Surface};

pub(crate) fn wrap_surface(
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    backend: BackendRequest,
    session: &mut CudaSession,
) -> Result<Surface, Error> {
    validate_surface_request(backend)?;
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    match backend {
        BackendRequest::Cpu | BackendRequest::Auto => {
            if j2k_profile::gpu_route_profile_enabled() {
                let request_s = format!("{backend:?}");
                let fmt_s = format!("{fmt:?}");
                let width_s = dimensions.0.to_string();
                let height_s = dimensions.1.to_string();
                j2k_profile::emit_gpu_route_profile(
                    "jpeg",
                    "cuda",
                    &[
                        ("op", "wrap_surface"),
                        ("request", request_s.as_str()),
                        ("fmt", fmt_s.as_str()),
                        ("width", width_s.as_str()),
                        ("height", height_s.as_str()),
                        ("decision", "host_surface"),
                    ],
                );
            }
            Ok(Surface {
                backend: BackendKind::Cpu,
                dimensions,
                fmt,
                pitch_bytes,
                stats: CudaSurfaceStats::default(),
                storage: Storage::Host(bytes),
            })
        }
        BackendRequest::Cuda => wrap_cuda_surface(&bytes, dimensions, fmt, pitch_bytes, session),
        BackendRequest::Metal => Err(Error::UnsupportedBackend { request: backend }),
    }
}

pub(crate) fn validate_surface_request(backend: BackendRequest) -> Result<(), Error> {
    j2k_core::validate_cuda_surface_backend_request(backend)
        .map_err(|request| Error::UnsupportedBackend { request })
}

#[cfg(feature = "cuda-runtime")]
fn wrap_cuda_surface(
    bytes: &[u8],
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    session: &mut CudaSession,
) -> Result<Surface, Error> {
    let context = session.cuda_context()?;
    let output = context.copy_with_kernel(bytes).map_err(cuda_error)?;
    let (buffer, stats) = output.into_parts();
    if j2k_profile::gpu_route_profile_enabled() {
        let fmt_s = format!("{fmt:?}");
        let width_s = dimensions.0.to_string();
        let height_s = dimensions.1.to_string();
        let kernel_dispatches_s = stats.kernel_dispatches().to_string();
        let copy_dispatches_s = stats.copy_kernel_dispatches().to_string();
        j2k_profile::emit_gpu_route_profile(
            "jpeg",
            "cuda",
            &[
                ("op", "wrap_surface"),
                ("request", "Cuda"),
                ("fmt", fmt_s.as_str()),
                ("width", width_s.as_str()),
                ("height", height_s.as_str()),
                ("decision", "cuda_upload"),
                ("kernel_dispatches", kernel_dispatches_s.as_str()),
                ("copy_kernel_dispatches", copy_dispatches_s.as_str()),
            ],
        );
    }
    Ok(Surface {
        backend: BackendKind::Cuda,
        dimensions,
        fmt,
        pitch_bytes,
        stats: CudaSurfaceStats {
            kernel_dispatches: stats.kernel_dispatches(),
            copy_kernel_dispatches: stats.copy_kernel_dispatches(),
            decode_kernel_dispatches: stats.decode_kernel_dispatches(),
            hardware_decode: stats.used_hardware_decode(),
            decode_path: crate::surface::CudaJpegDecodePath::None,
        },
        storage: Storage::Cuda(buffer),
    })
}

#[cfg(not(feature = "cuda-runtime"))]
fn wrap_cuda_surface(
    _bytes: &[u8],
    dimensions: (u32, u32),
    fmt: PixelFormat,
    _pitch_bytes: usize,
    _session: &mut CudaSession,
) -> Result<Surface, Error> {
    if j2k_profile::gpu_route_profile_enabled() {
        let fmt_s = format!("{fmt:?}");
        let width_s = dimensions.0.to_string();
        let height_s = dimensions.1.to_string();
        j2k_profile::emit_gpu_route_profile(
            "jpeg",
            "cuda",
            &[
                ("op", "wrap_surface"),
                ("request", "Cuda"),
                ("fmt", fmt_s.as_str()),
                ("width", width_s.as_str()),
                ("height", height_s.as_str()),
                ("decision", "cuda_unavailable"),
            ],
        );
    }
    Err(Error::CudaUnavailable)
}

#[cfg(feature = "cuda-runtime")]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn cuda_error(error: CudaError) -> Error {
    if error.is_unavailable() {
        Error::CudaUnavailable
    } else {
        Error::CudaRuntime {
            message: error.to_string(),
        }
    }
}
