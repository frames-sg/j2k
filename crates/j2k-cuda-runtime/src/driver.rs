use crate::{build_flags::CUDA_SUCCESS, error::CudaError};
use libloading::Library;
#[cfg(feature = "cuda-profiling")]
use std::sync::OnceLock;
use std::{
    ffi::{c_void, CStr},
    os::raw::{c_char, c_int, c_uint},
};

pub(crate) type CuResult = c_int;

pub(crate) type CuDevice = c_int;

pub(crate) type CuContext = *mut c_void;

pub(crate) type CuDevicePtr = u64;

pub(crate) type CuModule = *mut c_void;

pub(crate) type CuFunction = *mut c_void;

pub(crate) type CuStream = *mut c_void;

pub(crate) type CuEvent = *mut c_void;

pub(crate) type CuInit = unsafe extern "C" fn(c_uint) -> CuResult;

pub(crate) type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;

pub(crate) type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;

pub(crate) type CuCtxCreate = unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;

pub(crate) type CuCtxDestroy = unsafe extern "C" fn(CuContext) -> CuResult;

pub(crate) type CuDevicePrimaryCtxRetain =
    unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult;

pub(crate) type CuDevicePrimaryCtxRelease = unsafe extern "C" fn(CuDevice) -> CuResult;

pub(crate) type CuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;

pub(crate) type CuPointerGetAttribute =
    unsafe extern "C" fn(*mut c_void, c_int, CuDevicePtr) -> CuResult;

pub(crate) type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;

pub(crate) type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;

pub(crate) type CuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CuResult;

pub(crate) type CuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CuResult;

pub(crate) type CuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;

pub(crate) type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;

pub(crate) type CuMemsetD8 = unsafe extern "C" fn(CuDevicePtr, u8, usize) -> CuResult;

pub(crate) type CuMemsetD32 = unsafe extern "C" fn(CuDevicePtr, c_uint, usize) -> CuResult;

pub(crate) type CuGetErrorName = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;

pub(crate) type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;

pub(crate) type CuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;

pub(crate) type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;

pub(crate) type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;

pub(crate) type CuCtxSynchronize = unsafe extern "C" fn() -> CuResult;

#[cfg(test)]
pub(crate) type CuStreamCreate = unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult;

#[cfg(test)]
pub(crate) type CuStreamDestroy = unsafe extern "C" fn(CuStream) -> CuResult;

pub(crate) type CuEventCreate = unsafe extern "C" fn(*mut CuEvent, c_uint) -> CuResult;

pub(crate) type CuEventDestroy = unsafe extern "C" fn(CuEvent) -> CuResult;

pub(crate) type CuEventRecord = unsafe extern "C" fn(CuEvent, CuStream) -> CuResult;

pub(crate) type CuEventSynchronize = unsafe extern "C" fn(CuEvent) -> CuResult;

pub(crate) type CuEventElapsedTime = unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult;

#[cfg(feature = "cuda-profiling")]
pub(crate) type NvtxRangePushA = unsafe extern "C" fn(*const c_char) -> c_int;

#[cfg(feature = "cuda-profiling")]
pub(crate) type NvtxRangePop = unsafe extern "C" fn() -> c_int;

pub(crate) struct Driver {
    pub(crate) _library: Library,
    pub(crate) cu_init: CuInit,
    pub(crate) cu_device_get_count: CuDeviceGetCount,
    pub(crate) cu_device_get: CuDeviceGet,
    pub(crate) cu_ctx_create: CuCtxCreate,
    pub(crate) cu_ctx_destroy: CuCtxDestroy,
    pub(crate) cu_device_primary_ctx_retain: CuDevicePrimaryCtxRetain,
    pub(crate) cu_device_primary_ctx_release: CuDevicePrimaryCtxRelease,
    pub(crate) cu_ctx_set_current: CuCtxSetCurrent,
    pub(crate) cu_pointer_get_attribute: CuPointerGetAttribute,
    pub(crate) cu_mem_alloc: CuMemAlloc,
    pub(crate) cu_mem_free: CuMemFree,
    pub(crate) cu_mem_host_alloc: CuMemHostAlloc,
    pub(crate) cu_mem_free_host: CuMemFreeHost,
    pub(crate) cu_memcpy_htod: CuMemcpyHtoD,
    pub(crate) cu_memcpy_dtoh: CuMemcpyDtoH,
    pub(crate) cu_memset_d8: CuMemsetD8,
    pub(crate) cu_memset_d32: CuMemsetD32,
    pub(crate) cu_get_error_name: CuGetErrorName,
    #[cfg_attr(
        not(j2k_cuda_oxide_enabled),
        expect(
            dead_code,
            reason = "module loading is used only by CUDA Oxide kernels"
        )
    )]
    pub(crate) cu_module_load_data: CuModuleLoadData,
    pub(crate) cu_module_unload: CuModuleUnload,
    #[cfg_attr(
        not(j2k_cuda_oxide_enabled),
        expect(dead_code, reason = "kernel lookup is used only by CUDA Oxide modules")
    )]
    pub(crate) cu_module_get_function: CuModuleGetFunction,
    pub(crate) cu_launch_kernel: CuLaunchKernel,
    pub(crate) cu_ctx_synchronize: CuCtxSynchronize,
    #[cfg(test)]
    pub(crate) cu_stream_create: CuStreamCreate,
    #[cfg(test)]
    pub(crate) cu_stream_destroy: CuStreamDestroy,
    pub(crate) cu_event_create: CuEventCreate,
    pub(crate) cu_event_destroy: CuEventDestroy,
    pub(crate) cu_event_record: CuEventRecord,
    pub(crate) cu_event_synchronize: CuEventSynchronize,
    pub(crate) cu_event_elapsed_time: CuEventElapsedTime,
}

impl Driver {
    pub(crate) fn load() -> Result<Self, CudaError> {
        #[cfg(target_os = "linux")]
        const LIBRARY_CANDIDATES: &[&str] = &["libcuda.so.1", "libcuda.so"];
        #[cfg(target_os = "windows")]
        const LIBRARY_CANDIDATES: &[&str] = &["nvcuda.dll"];
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        const LIBRARY_CANDIDATES: &[&str] = &[];

        let mut last_error = None;
        for candidate in LIBRARY_CANDIDATES {
            // SAFETY: Loading the CUDA driver library is required before symbol
            // lookup. The resulting Library is owned by Driver and outlives all
            // copied function pointers.
            match unsafe { Library::new(*candidate) } {
                Ok(library) => return Self::from_library(library),
                Err(error) => last_error = Some(error.to_string()),
            }
        }

        Err(CudaError::Unavailable {
            message: last_error.unwrap_or_else(|| "unsupported CUDA host platform".to_string()),
        })
    }

    pub(crate) fn from_library(library: Library) -> Result<Self, CudaError> {
        Ok(Self {
            cu_init: load_symbol(&library, b"cuInit\0")?,
            cu_device_get_count: load_symbol(&library, b"cuDeviceGetCount\0")?,
            cu_device_get: load_symbol(&library, b"cuDeviceGet\0")?,
            cu_ctx_create: load_symbol(&library, b"cuCtxCreate_v2\0")?,
            cu_ctx_destroy: load_symbol(&library, b"cuCtxDestroy_v2\0")?,
            cu_device_primary_ctx_retain: load_symbol(&library, b"cuDevicePrimaryCtxRetain\0")?,
            cu_device_primary_ctx_release: load_symbol(
                &library,
                b"cuDevicePrimaryCtxRelease_v2\0",
            )?,
            cu_ctx_set_current: load_symbol(&library, b"cuCtxSetCurrent\0")?,
            cu_pointer_get_attribute: load_symbol(&library, b"cuPointerGetAttribute\0")?,
            cu_mem_alloc: load_symbol(&library, b"cuMemAlloc_v2\0")?,
            cu_mem_free: load_symbol(&library, b"cuMemFree_v2\0")?,
            cu_mem_host_alloc: load_symbol(&library, b"cuMemHostAlloc\0")?,
            cu_mem_free_host: load_symbol(&library, b"cuMemFreeHost\0")?,
            cu_memcpy_htod: load_symbol(&library, b"cuMemcpyHtoD_v2\0")?,
            cu_memcpy_dtoh: load_symbol(&library, b"cuMemcpyDtoH_v2\0")?,
            cu_memset_d8: load_symbol(&library, b"cuMemsetD8_v2\0")?,
            cu_memset_d32: load_symbol(&library, b"cuMemsetD32_v2\0")?,
            cu_get_error_name: load_symbol(&library, b"cuGetErrorName\0")?,
            cu_module_load_data: load_symbol(&library, b"cuModuleLoadData\0")?,
            cu_module_unload: load_symbol(&library, b"cuModuleUnload\0")?,
            cu_module_get_function: load_symbol(&library, b"cuModuleGetFunction\0")?,
            cu_launch_kernel: load_symbol(&library, b"cuLaunchKernel\0")?,
            cu_ctx_synchronize: load_symbol(&library, b"cuCtxSynchronize\0")?,
            #[cfg(test)]
            cu_stream_create: load_symbol(&library, b"cuStreamCreate\0")?,
            #[cfg(test)]
            cu_stream_destroy: load_symbol(&library, b"cuStreamDestroy_v2\0")?,
            cu_event_create: load_symbol(&library, b"cuEventCreate\0")?,
            cu_event_destroy: load_symbol(&library, b"cuEventDestroy_v2\0")?,
            cu_event_record: load_symbol(&library, b"cuEventRecord\0")?,
            cu_event_synchronize: load_symbol(&library, b"cuEventSynchronize\0")?,
            cu_event_elapsed_time: load_symbol(&library, b"cuEventElapsedTime\0")?,
            _library: library,
        })
    }

    pub(crate) fn check(&self, operation: &'static str, result: CuResult) -> Result<(), CudaError> {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CudaError::Driver {
                operation,
                code: result,
                name: self.error_name(result),
            })
        }
    }

    pub(crate) fn error_name(&self, result: CuResult) -> String {
        let mut name = std::ptr::null();
        // SAFETY: cuGetErrorName writes a borrowed static C string pointer for
        // a CUDA result code. A failure here is non-critical for diagnostics.
        let status = unsafe { (self.cu_get_error_name)(result, &raw mut name) };
        if status == CUDA_SUCCESS && !name.is_null() {
            // SAFETY: CUDA returns a NUL-terminated static string on success.
            let cstr = unsafe { CStr::from_ptr(name) };
            format!(" ({})", cstr.to_string_lossy())
        } else {
            String::new()
        }
    }
}

pub(crate) fn load_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Result<T, CudaError> {
    // SAFETY: Symbol names are NUL-terminated CUDA Driver API entry points. The
    // symbol value is copied, and Driver keeps the Library alive.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| CudaError::Unavailable {
            message: format!(
                "missing CUDA driver symbol {}: {error}",
                String::from_utf8_lossy(name)
            ),
        })
}

pub(crate) struct CudaNvtxRange {
    #[cfg(feature = "cuda-profiling")]
    pub(crate) active: bool,
}

impl CudaNvtxRange {
    pub(crate) fn push(name: &str) -> Self {
        #[cfg(feature = "cuda-profiling")]
        {
            let Some(api) = nvtx_api() else {
                return Self { active: false };
            };
            let Ok(name) = std::ffi::CString::new(name) else {
                return Self { active: false };
            };
            // SAFETY: `name` is a NUL-terminated C string that lives for the
            // duration of the call. The NVTX function pointer is loaded from a
            // live library stored in NvtxApi.
            let depth = unsafe { (api.range_push_a)(name.as_ptr()) };
            Self { active: depth >= 0 }
        }
        #[cfg(not(feature = "cuda-profiling"))]
        {
            let _ = name;
            Self {}
        }
    }
}

impl Drop for CudaNvtxRange {
    fn drop(&mut self) {
        #[cfg(feature = "cuda-profiling")]
        if self.active {
            if let Some(api) = nvtx_api() {
                // SAFETY: Matching pop for a successful nvtxRangePushA in this
                // thread. NVTX returns a depth value that is not needed here.
                let _ = unsafe { (api.range_pop)() };
            }
        }
    }
}

#[cfg(feature = "cuda-profiling")]
pub(crate) struct NvtxApi {
    pub(crate) _library: Library,
    pub(crate) range_push_a: NvtxRangePushA,
    pub(crate) range_pop: NvtxRangePop,
}

#[cfg(feature = "cuda-profiling")]
pub(crate) fn nvtx_api() -> Option<&'static NvtxApi> {
    static API: OnceLock<Option<NvtxApi>> = OnceLock::new();
    API.get_or_init(load_optional_nvtx).as_ref()
}

#[cfg(feature = "cuda-profiling")]
pub(crate) fn load_optional_nvtx() -> Option<NvtxApi> {
    #[cfg(target_os = "linux")]
    const LIBRARY_CANDIDATES: &[&str] = &["libnvToolsExt.so.1", "libnvToolsExt.so"];
    #[cfg(target_os = "windows")]
    const LIBRARY_CANDIDATES: &[&str] = &["nvToolsExt64_1.dll", "nvToolsExt64_64_1.dll"];
    #[cfg(target_os = "macos")]
    const LIBRARY_CANDIDATES: &[&str] = &["libnvToolsExt.dylib"];
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    const LIBRARY_CANDIDATES: &[&str] = &[];

    for candidate in LIBRARY_CANDIDATES {
        // SAFETY: This optional profiling path only copies immutable NVTX
        // function pointers and stores the Library in NvtxApi for their
        // lifetime. Failure to load simply disables NVTX ranges.
        let Ok(library) = (unsafe { Library::new(*candidate) }) else {
            continue;
        };
        let Ok(range_push_a) = load_symbol(&library, b"nvtxRangePushA\0") else {
            continue;
        };
        let Ok(range_pop) = load_symbol(&library, b"nvtxRangePop\0") else {
            continue;
        };
        return Some(NvtxApi {
            _library: library,
            range_push_a,
            range_pop,
        });
    }
    None
}

// SAFETY: CUDA Driver API handles are process resources guarded by the driver.
// The struct stores copied function pointers and owns the loaded library.
unsafe impl Send for Driver {}

// SAFETY: Driver entry points are immutable function pointers, and mutable CUDA
// state is always addressed through explicit CUDA context calls.
unsafe impl Sync for Driver {}
