// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_runtime::{
    cuda_kernel_param, CudaContext, CudaKernelParam, CudaKernelSpec, CudaLaunchGeometry,
};

#[repr(C)]
struct ExampleKernelParam {
    value: u32,
}

// SAFETY: the example has a stable C layout containing one CUDA-compatible
// scalar and remains alive while its parameter pointer is used.
unsafe impl CudaKernelParam for ExampleKernelParam {}

fn validate_disjoint_buffers(
    context: &CudaContext,
    buffers: &[&j2k_cuda_runtime::CudaDeviceBuffer],
) -> Result<(), j2k_cuda_runtime::CudaError> {
    context.validate_disjoint_device_buffers(buffers.iter().copied())
}

type CudaResult<T> = Result<T, j2k_cuda_runtime::CudaError>;
type LaunchFn = unsafe fn(
    &CudaContext,
    CudaKernelSpec,
    CudaLaunchGeometry,
    &mut [*mut std::ffi::c_void],
) -> CudaResult<()>;
type DeviceBufferOperationFn<T> =
    fn(&CudaContext, &j2k_cuda_runtime::CudaDeviceBuffer, T, usize) -> CudaResult<()>;
type ValidatePointerFn = fn(&CudaContext, u64) -> CudaResult<u64>;
type RecordStatusCopyFn = fn(&CudaContext, usize);
type BufferOwnershipFn<T> = fn(&T, &CudaContext) -> bool;
type ValidateDisjointFn =
    fn(&CudaContext, &[&j2k_cuda_runtime::CudaDeviceBuffer]) -> CudaResult<()>;
type QueuedLaunchFn = unsafe fn(
    &CudaContext,
    CudaKernelSpec,
    CudaLaunchGeometry,
    &mut [*mut std::ffi::c_void],
    &j2k_cuda_runtime::CudaBufferPool,
    Vec<j2k_cuda_runtime::CudaPooledDeviceBuffer>,
    j2k_cuda_runtime::CudaExecutionStats,
) -> CudaResult<j2k_cuda_runtime::CudaQueuedExecution>;
type FinishedResources = (
    Vec<j2k_cuda_runtime::CudaPooledDeviceBuffer>,
    j2k_cuda_runtime::CudaExecutionStats,
);
type FinishFn = fn(j2k_cuda_runtime::CudaQueuedExecution) -> CudaResult<FinishedResources>;
type FinishAfterCompletionFn =
    unsafe fn(j2k_cuda_runtime::CudaQueuedExecution) -> CudaResult<FinishedResources>;
type PrepareOperationFn = fn(&CudaContext) -> CudaResult<()>;
type SynchronizeReleaseFn = fn(&CudaContext) -> j2k_cuda_runtime::CudaSynchronizationOutcome;
type LifetimesPoisonedFn = fn(&CudaContext) -> bool;
type DeferReuseFn =
    fn(&j2k_cuda_runtime::CudaBufferPool) -> CudaResult<j2k_cuda_runtime::CudaBufferPoolReuseGuard>;

#[test]
fn kernel_specs_validate_static_module_and_entrypoint_contracts() {
    const PTX: &[u8] = b".version 8.0\n\0";
    let spec = CudaKernelSpec::new("jpeg-decode", PTX, b"decode_rgb8\0")
        .expect("well-formed static kernel spec");
    assert_eq!(spec.module_id(), "jpeg-decode");
    assert_eq!(spec.ptx(), PTX);
    assert_eq!(spec.entrypoint(), b"decode_rgb8\0");

    assert!(CudaKernelSpec::new("", PTX, b"decode_rgb8\0").is_err());
    assert!(CudaKernelSpec::new("jpeg-decode", b"not-terminated", b"decode_rgb8\0").is_err());
    assert!(CudaKernelSpec::new("jpeg-decode", PTX, b"not-terminated").is_err());
    assert!(CudaKernelSpec::new("jpeg-decode", PTX, b"\0").is_err());
}

#[test]
fn launch_geometry_is_a_checked_low_level_value() {
    let geometry =
        CudaLaunchGeometry::new((4, 3, 2), (32, 2, 1)).expect("valid CUDA launch geometry");
    assert_eq!(geometry.grid(), (4, 3, 2));
    assert_eq!(geometry.block(), (32, 2, 1));
    assert!(CudaLaunchGeometry::new((0, 1, 1), (1, 1, 1)).is_none());
    assert!(CudaLaunchGeometry::new((1, 1, 1), (1024, 2, 1)).is_none());
}

#[test]
fn external_engines_can_build_typed_parameter_arrays_for_the_launch_primitive() {
    let mut value = ExampleKernelParam { value: 7 };
    let pointer = cuda_kernel_param(&mut value);
    assert_eq!(pointer, std::ptr::from_mut(&mut value).cast());

    let _: LaunchFn = CudaContext::launch_compiled_kernel;
    let _: LaunchFn = CudaContext::launch_compiled_kernel_async;
    let _: DeviceBufferOperationFn<u8> = CudaContext::memset_d8;
    let _: ValidatePointerFn = CudaContext::validate_device_pointer;
    let _: DeviceBufferOperationFn<u32> = CudaContext::memset_d32_async;
    let _: DeviceBufferOperationFn<u32> = CudaContext::memset_d32;
    let _: RecordStatusCopyFn = CudaContext::record_status_device_to_host_copy;
    let _: BufferOwnershipFn<j2k_cuda_runtime::CudaDeviceBuffer> =
        j2k_cuda_runtime::CudaDeviceBuffer::is_owned_by;
    let _: BufferOwnershipFn<j2k_cuda_runtime::CudaBufferPool> =
        j2k_cuda_runtime::CudaBufferPool::is_owned_by;
    let _: ValidateDisjointFn = validate_disjoint_buffers;
    let _: QueuedLaunchFn = CudaContext::launch_compiled_kernel_queued;
    let _: FinishFn = j2k_cuda_runtime::CudaQueuedExecution::finish_with_resources;
    let _: FinishAfterCompletionFn =
        j2k_cuda_runtime::CudaQueuedExecution::finish_with_resources_after_completion;
    let _: PrepareOperationFn = CudaContext::prepare_operation;
    let _: SynchronizeReleaseFn = CudaContext::synchronize_for_resource_release;
    let _: LifetimesPoisonedFn = CudaContext::resource_lifetimes_poisoned;
    let _: DeferReuseFn = j2k_cuda_runtime::CudaBufferPool::defer_reuse;
}
