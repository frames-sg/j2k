// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendKind, BackendRequest, PixelFormat};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaError;

use crate::surface::{HostStorage, Storage};
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
            j2k_profile::emit_gpu_route_surface_profile(
                ("jpeg", "cuda"),
                (
                    "wrap_surface",
                    format_args!("{backend:?}"),
                    format_args!("{fmt:?}"),
                    "host_surface",
                ),
                dimensions,
                [],
            );
            #[cfg(feature = "cuda-runtime")]
            let storage = {
                let retained_bytes = j2k_core::host_capacity_bytes::<u8>(bytes.capacity());
                let lease = session.reserve_existing_host_owner(retained_bytes)?;
                Storage::Host(HostStorage::new(bytes, lease))
            };
            #[cfg(not(feature = "cuda-runtime"))]
            let storage = Storage::Host(HostStorage::new(bytes));
            Ok(Surface {
                backend: BackendKind::Cpu,
                dimensions,
                fmt,
                pitch_bytes,
                stats: CudaSurfaceStats::default(),
                storage,
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
    crate::profile::emit_optional_gpu_route_fields(
        "jpeg_cuda_wrap_surface_fields",
        || {
            Ok([
                j2k_profile::ProfileField::metric("kernel_dispatches", stats.kernel_dispatches())?,
                j2k_profile::ProfileField::metric(
                    "copy_kernel_dispatches",
                    stats.copy_kernel_dispatches(),
                )?,
            ])
        },
        |fields| {
            j2k_profile::emit_gpu_route_surface_profile(
                ("jpeg", "cuda"),
                (
                    "wrap_surface",
                    "Cuda",
                    format_args!("{fmt:?}"),
                    "cuda_upload",
                ),
                dimensions,
                fields,
            );
        },
    );
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
    j2k_profile::emit_gpu_route_surface_profile(
        ("jpeg", "cuda"),
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
pub(crate) fn cuda_error(error: CudaError) -> Error {
    if error.is_unavailable() {
        return Error::CudaUnavailable;
    }
    Error::CudaRuntime { source: error }
}

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests {
    use super::{cuda_error, wrap_surface};
    use crate::CudaSession;
    use crate::Error;
    use j2k_core::{BackendRequest, PixelFormat};
    use j2k_cuda_runtime::CudaError;

    #[test]
    fn runtime_allocation_failure_keeps_typed_runtime_source_and_size() {
        assert!(matches!(
            cuda_error(CudaError::HostAllocationFailed { bytes: 4096 }),
            Error::CudaRuntime {
                source: CudaError::HostAllocationFailed { bytes: 4096 }
            }
        ));
    }

    #[test]
    fn runtime_failure_keeps_typed_source() {
        let error = cuda_error(CudaError::KernelStatus {
            kernel: "jpeg_test_kernel",
            code: 5,
            detail: 13,
        });

        assert!(matches!(
            error,
            Error::CudaRuntime {
                source: CudaError::KernelStatus {
                    kernel: "jpeg_test_kernel",
                    code: 5,
                    detail: 13
                }
            }
        ));
    }

    #[test]
    fn runtime_host_cap_and_allocator_failures_remain_distinguishable() {
        let cap_error = cuda_error(CudaError::HostAllocationTooLarge {
            requested: 8193,
            cap: 8192,
            what: "JPEG baseline batch encode output",
        });
        assert!(matches!(
            cap_error,
            Error::CudaRuntime {
                source: CudaError::HostAllocationTooLarge {
                    requested: 8193,
                    cap: 8192,
                    what: "JPEG baseline batch encode output",
                }
            }
        ));

        assert!(matches!(
            cuda_error(CudaError::HostAllocationFailed { bytes: 8193 }),
            Error::CudaRuntime {
                source: CudaError::HostAllocationFailed { bytes: 8193 },
            }
        ));
    }

    #[test]
    fn host_surface_capacity_remains_leased_until_the_surface_drops() {
        let mut session = CudaSession::default();
        let bytes = Vec::from([1_u8, 2, 3, 4, 5, 6]);
        let expected = j2k_core::host_capacity_bytes::<u8>(bytes.capacity());
        let surface = wrap_surface(
            bytes,
            (2, 1),
            PixelFormat::Rgb8,
            BackendRequest::Cpu,
            &mut session,
        )
        .unwrap();

        assert_eq!(
            session
                .owned_cuda_host_memory_diagnostics()
                .unwrap()
                .active_owner_bytes,
            expected
        );
        drop(surface);
        assert_eq!(
            session
                .owned_cuda_host_memory_diagnostics()
                .unwrap()
                .active_owner_bytes,
            0
        );
    }
}
