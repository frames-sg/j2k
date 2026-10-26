// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ownership transfer through the DLPack C ABI. No libtorch link dependency.

use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex, Weak},
};

#[cfg(feature = "cuda")]
use pyo3::types::PyDict;
use pyo3::{
    exceptions::{PyBufferError, PyRuntimeError, PyValueError},
    ffi,
    prelude::*,
    types::{PyCapsule, PyTuple},
};

pub(crate) enum Samples {
    U8(Vec<u8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    #[cfg(feature = "cuda")]
    Torch(Py<PyAny>),
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    Metal(crate::metal::MetalSamples),
}

impl Samples {
    fn pointer(&mut self) -> *mut c_void {
        match self {
            Self::U8(v) => v.as_mut_ptr().cast(),
            Self::U16(v) => v.as_mut_ptr().cast(),
            Self::I16(v) => v.as_mut_ptr().cast(),
            Self::I64(v) => v.as_mut_ptr().cast(),
            Self::F32(v) => v.as_mut_ptr().cast(),
            #[cfg(feature = "cuda")]
            Self::Torch(_) => unreachable!("framework allocations are exported by their owner"),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::Metal(buffer) => buffer.pointer(),
        }
    }

    fn len(&self) -> Option<usize> {
        Some(match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F32(v) => v.len(),
            #[cfg(feature = "cuda")]
            Self::Torch(_) => return None,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            Self::Metal(buffer) => buffer.len(),
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DType {
    pub code: u8,
    pub bits: u8,
    pub name: &'static str,
}

impl DType {
    pub const U8: Self = Self {
        code: 1,
        bits: 8,
        name: "uint8",
    };
    pub const U16: Self = Self {
        code: 1,
        bits: 16,
        name: "uint16",
    };
    pub const I16: Self = Self {
        code: 0,
        bits: 16,
        name: "int16",
    };
    pub const I64: Self = Self {
        code: 0,
        bits: 64,
        name: "int64",
    };
    pub const F32: Self = Self {
        code: 2,
        bits: 32,
        name: "float32",
    };
}

/// A contiguous decoded tensor whose storage can be exported exactly once.
///
/// Use to_torch(), to_numpy(), or a framework's from_dlpack(). Ownership then
/// belongs to that framework; the decoder can be reused or destroyed safely.
/// Make subsequent views or copies through the returned framework tensor.
#[pyclass(name = "Tensor", module = "j2k", frozen)]
pub(crate) struct Tensor {
    shape: Vec<usize>,
    dtype: DType,
    device: (i32, i32),
    storage: Arc<Mutex<Option<Samples>>>,
}

impl Tensor {
    pub fn new(
        samples: Samples,
        shape: Vec<usize>,
        dtype: DType,
        device: (i32, i32),
    ) -> PyResult<Self> {
        let len = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| PyValueError::new_err("tensor shape overflow"))?;
        if shape.iter().any(|&d| i64::try_from(d).is_err())
            || samples.len().is_some_and(|n| n != len)
        {
            return Err(PyValueError::new_err(
                "tensor shape does not match its owned allocation",
            ));
        }
        Ok(Self {
            shape,
            dtype,
            device,
            storage: Arc::new(Mutex::new(Some(samples))),
        })
    }

    pub fn remove_batch_axis(&mut self) -> PyResult<()> {
        if self.shape.first() != Some(&1) {
            return Err(PyValueError::new_err(
                "single-image decode produced a non-singleton batch",
            ));
        }
        self.shape.remove(0);
        Ok(())
    }
}

#[pymethods]
impl Tensor {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.name
    }

    #[getter]
    fn device(&self) -> (i32, i32) {
        self.device
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        self.device
    }

    /// Export once. CPU output shares the Rust allocation; it is never a borrowed Rust slice.
    #[pyo3(signature = (stream=None, *, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "copy=True is unsupported; clone the imported tensor explicitly",
            ));
        }
        if dl_device.is_some_and(|d| d != self.device) {
            return Err(PyBufferError::new_err(
                "device conversion is not part of DLPack export",
            ));
        }
        if matches!(self.device.0, 1 | 8) && stream.is_some() {
            return Err(PyValueError::new_err(
                "CPU and Metal DLPack export do not accept a stream",
            ));
        }
        let shape = crate::owned_items(
            self.shape
                .iter()
                .map(|&n| i64::try_from(n).expect("validated dimension")),
        )?;
        let ndim = i32::try_from(self.shape.len())
            .map_err(|_| PyValueError::new_err("tensor rank overflow"))?;
        let samples = self
            .storage
            .lock()
            .map_err(|_| PyRuntimeError::new_err("tensor owner is poisoned"))?
            .take()
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "tensor ownership was already transferred; use the returned framework tensor",
                )
            })?;
        let _ = max_version; // A legacy capsule is valid for every advertised maximum version.
        let mut owner = Box::new(ManagedOwner {
            managed: DLManagedTensor {
                tensor: DLTensor {
                    data: std::ptr::null_mut(),
                    device: DLDevice {
                        kind: self.device.0,
                        id: self.device.1,
                    },
                    ndim,
                    dtype: DLDataType {
                        code: self.dtype.code,
                        bits: self.dtype.bits,
                        lanes: 1,
                    },
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                context: std::ptr::null_mut(),
                deleter: Some(delete_managed),
            },
            samples: Some(samples),
            shape,
            return_to: Arc::downgrade(&self.storage),
            consumed: false,
            #[cfg(feature = "cuda")]
            foreign: None,
        });
        owner.prepare_data(py, stream)?;
        owner.managed.context = (&raw mut *owner).cast();
        let raw = Box::into_raw(owner);
        // SAFETY: ManagedOwner is repr(C) with the DLPack managed tensor first.
        // The Box and its allocations stay live until exactly one of the
        // unconsumed-capsule destructor or consumer's DLPack deleter runs.
        let result = unsafe {
            PyCapsule::new_with_pointer_and_destructor(
                py,
                NonNull::new_unchecked(raw.cast()),
                c"dltensor",
                Some(delete_capsule),
            )
        };
        match result {
            Ok(capsule) => Ok(capsule.into_any().unbind()),
            Err(error) => {
                // SAFETY: capsule construction failed; no consumer obtained this unique owner.
                unsafe { drop(Box::from_raw(raw)) };
                Err(error)
            }
        }
    }

    /// Transfer storage into a torch.Tensor without converting its dtype or layout.
    fn to_torch(slf: Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(slf
            .py()
            .import("torch")?
            .call_method1("from_dlpack", (slf,))?
            .unbind())
    }

    /// Transfer CPU storage into a NumPy array using NumPy's DLPack consumer.
    fn to_numpy(slf: Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        if slf.get().device.0 != 1 {
            return Err(PyBufferError::new_err(
                "NumPy requires CPU storage; use to_torch().cpu().numpy() for GPU tensors",
            ));
        }
        Ok(slf
            .py()
            .import("numpy")?
            .call_method1("from_dlpack", (slf,))?
            .unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "j2k.Tensor(shape={:?}, dtype={}, device={:?})",
            self.shape, self.dtype.name, self.device
        )
    }
}

// DLPack's legacy ABI is intentionally kept independent of any framework ABI.
#[derive(Clone, Copy)]
#[repr(C)]
struct DLDevice {
    kind: i32,
    id: i32,
}
#[derive(Clone, Copy)]
#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}
#[derive(Clone, Copy)]
#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}
#[repr(C)]
struct DLManagedTensor {
    tensor: DLTensor,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}
#[repr(C)]
struct ManagedOwner {
    managed: DLManagedTensor,
    samples: Option<Samples>,
    shape: Vec<i64>,
    return_to: Weak<Mutex<Option<Samples>>>,
    consumed: bool,
    #[cfg(feature = "cuda")]
    foreign: Option<NonNull<DLManagedTensor>>,
}

impl ManagedOwner {
    fn prepare_data(&mut self, py: Python<'_>, stream: Option<i64>) -> PyResult<()> {
        #[cfg(feature = "cuda")]
        if let Some(Samples::Torch(tensor)) = &self.samples {
            let kwargs = PyDict::new(py);
            kwargs.set_item("stream", stream)?;
            // Let PyTorch establish the consumer-stream dependency first. The
            // inner DLPack owner keeps its shape and allocation live while our
            // outer capsule decides whether the consumer accepted the export.
            let capsule = tensor
                .bind(py)
                .call_method("__dlpack__", (), Some(&kwargs))?;
            // SAFETY: PyCapsule_GetPointer validates the legacy capsule name.
            let pointer =
                unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), c"dltensor".as_ptr()) };
            let pointer =
                NonNull::new(pointer.cast::<DLManagedTensor>()).ok_or_else(|| PyErr::fetch(py))?;
            // SAFETY: the live capsule is uniquely borrowed under the GIL. After
            // renaming it, only this owner invokes its managed-tensor deleter.
            if unsafe { ffi::PyCapsule_SetName(capsule.as_ptr(), c"used_dltensor".as_ptr()) } != 0 {
                return Err(PyErr::fetch(py));
            }
            self.foreign = Some(pointer);
            // SAFETY: the foreign owner remains live until this owner's Drop.
            let foreign = unsafe { &pointer.as_ref().tensor };
            if foreign.device.kind != self.managed.tensor.device.kind
                || foreign.device.id != self.managed.tensor.device.id
                || foreign.dtype.code != self.managed.tensor.dtype.code
                || foreign.dtype.bits != self.managed.tensor.dtype.bits
                || foreign.dtype.lanes != 1
            {
                return Err(PyRuntimeError::new_err(
                    "framework DLPack metadata disagrees with owned output",
                ));
            }
            self.managed.tensor.data = foreign.data;
            self.managed.tensor.byte_offset = foreign.byte_offset;
            // CUDA output is allocated with torch.empty and is contiguous.
            // Keep our logical shape: single ImageRequest decode removes the
            // batch axis without changing the underlying framework allocation.
            self.managed.tensor.shape = self.shape.as_mut_ptr();
            return Ok(());
        }
        let _ = (py, stream);
        self.managed.tensor.data = self
            .samples
            .as_mut()
            .expect("live export samples")
            .pointer();
        self.managed.tensor.shape = self.shape.as_mut_ptr();
        Ok(())
    }
}

impl Drop for ManagedOwner {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if let Some(pointer) = self.foreign.take() {
            // SAFETY: the inner capsule was consumed by prepare_data. Its
            // DLPack deleter releases only its own framework allocation owner.
            unsafe {
                if let Some(deleter) = pointer.as_ref().deleter {
                    deleter(pointer.as_ptr());
                }
            }
        }
        if !self.consumed {
            if let Some(storage) = self.return_to.upgrade() {
                let mut storage = storage.lock().unwrap_or_else(|error| error.into_inner());
                if storage.is_none() {
                    *storage = self.samples.take();
                }
            }
        }
    }
}

unsafe extern "C" fn delete_managed(tensor: *mut DLManagedTensor) {
    if tensor.is_null() {
        return;
    }
    // SAFETY: a consumer calls this exactly once for the live managed tensor.
    // CPU, retained Metal and framework DLPack owners may be released on any
    // thread. Marking the export consumed prevents producer-side restoration.
    let mut owner = unsafe { Box::from_raw((*tensor).context.cast::<ManagedOwner>()) };
    owner.consumed = true;
}

unsafe extern "C" fn delete_capsule(capsule: *mut ffi::PyObject) {
    // SAFETY: Python calls this with a live capsule under the GIL. A consumed
    // capsule has been renamed and belongs solely to the consumer's deleter.
    if unsafe { ffi::PyCapsule_IsValid(capsule, c"dltensor".as_ptr()) } != 0 {
        // SAFETY: the valid unconsumed capsule still uniquely owns this pointer.
        let tensor = unsafe {
            ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr()).cast::<DLManagedTensor>()
        };
        // SAFETY: dropping an unconsumed owner restores storage if its producer
        // is alive; otherwise it releases storage without retaining the producer.
        unsafe { drop(Box::from_raw((*tensor).context.cast::<ManagedOwner>())) };
    }
}
