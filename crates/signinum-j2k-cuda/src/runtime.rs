// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BackendKind, BackendRequest, PixelFormat};
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::CudaError;

use crate::surface::Storage;
use crate::{CudaSession, CudaSurfaceStats, Error, Surface, SurfaceResidency};

const CPU_STAGED_CUDA_REQUIRES_EXPLICIT_API: &str =
    "CPU-staged CUDA upload requires the explicit CPU-staged API; BackendRequest::Cuda only accepts resident CUDA HTJ2K decode";

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
            if signinum_profile::gpu_route_profile_enabled() {
                let request_s = format!("{backend:?}");
                let fmt_s = format!("{fmt:?}");
                let width_s = dimensions.0.to_string();
                let height_s = dimensions.1.to_string();
                signinum_profile::emit_gpu_route_profile(
                    "j2k",
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
                residency: SurfaceResidency::Host,
                dimensions,
                fmt,
                pitch_bytes,
                stats: CudaSurfaceStats::default(),
                storage: Storage::Host(bytes),
            })
        }
        BackendRequest::Cuda => {
            let _ = (bytes, session);
            Err(Error::UnsupportedCudaRequest {
                reason: CPU_STAGED_CUDA_REQUIRES_EXPLICIT_API,
            })
        }
        BackendRequest::Metal => Err(Error::UnsupportedBackend { request: backend }),
    }
}

pub(crate) fn wrap_cpu_staged_cuda_surface(
    bytes: &[u8],
    dimensions: (u32, u32),
    fmt: PixelFormat,
    session: &mut CudaSession,
) -> Result<Surface, Error> {
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    wrap_cuda_surface(bytes, dimensions, fmt, pitch_bytes, session)
}

pub(crate) fn validate_surface_request(backend: BackendRequest) -> Result<(), Error> {
    signinum_core::validate_cuda_surface_backend_request(backend)
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
    if signinum_profile::gpu_route_profile_enabled() {
        let fmt_s = format!("{fmt:?}");
        let width_s = dimensions.0.to_string();
        let height_s = dimensions.1.to_string();
        let kernel_dispatches_s = stats.kernel_dispatches().to_string();
        signinum_profile::emit_gpu_route_profile(
            "j2k",
            "cuda",
            &[
                ("op", "wrap_surface"),
                ("request", "Cuda"),
                ("fmt", fmt_s.as_str()),
                ("width", width_s.as_str()),
                ("height", height_s.as_str()),
                ("decision", "cuda_upload"),
                ("kernel_dispatches", kernel_dispatches_s.as_str()),
            ],
        );
    }
    Ok(Surface {
        backend: BackendKind::Cuda,
        residency: SurfaceResidency::CpuStagedCudaUpload,
        dimensions,
        fmt,
        pitch_bytes,
        stats: CudaSurfaceStats {
            total: stats.kernel_dispatches(),
            copy: stats.copy_kernel_dispatches(),
            decode: stats.decode_kernel_dispatches(),
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
    if signinum_profile::gpu_route_profile_enabled() {
        let fmt_s = format!("{fmt:?}");
        let width_s = dimensions.0.to_string();
        let height_s = dimensions.1.to_string();
        signinum_profile::emit_gpu_route_profile(
            "j2k",
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
