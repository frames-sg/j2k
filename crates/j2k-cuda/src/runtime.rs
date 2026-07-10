// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendKind, BackendRequest, PixelFormat};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaError;

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
            j2k_profile::emit_gpu_route_surface_profile(
                ("j2k", "cuda"),
                (
                    "wrap_surface",
                    format_args!("{backend:?}"),
                    format_args!("{fmt:?}"),
                    "host_surface",
                ),
                dimensions,
                [],
            );
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
    j2k_profile::emit_gpu_route_surface_profile(
        ("j2k", "cuda"),
        (
            "wrap_surface",
            "Cuda",
            format_args!("{fmt:?}"),
            "cuda_upload",
        ),
        dimensions,
        [j2k_profile::ProfileField::metric(
            "kernel_dispatches",
            stats.kernel_dispatches(),
        )],
    );
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
    j2k_profile::emit_gpu_route_surface_profile(
        ("j2k", "cuda"),
        (
            "wrap_surface",
            "Cuda",
            format_args!("{fmt:?}"),
            "cuda_unavailable",
        ),
        dimensions,
        [],
    );
    Err(Error::CudaUnavailable)
}

#[cfg(feature = "cuda-runtime")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "adapter consumes the runtime error while translating its owned message"
)]
pub(crate) fn cuda_error(error: CudaError) -> Error {
    if error.is_unavailable() {
        Error::CudaUnavailable
    } else {
        Error::CudaRuntime {
            message: error.to_string(),
        }
    }
}
