// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ownership transfer through the DLPack C ABI. No libtorch link dependency.

use std::{ffi::c_void, ptr::NonNull, sync::Mutex};

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
    storage: Mutex<Option<Samples>>,
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
            storage: Mutex::new(Some(samples)),
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
        if self.device.0 == 1 && stream.is_some() {
            return Err(PyValueError::new_err(
                "CPU DLPack export does not accept a stream",
            ));
        }
        let mut samples = self
            .storage
            .lock()
            .map_err(|_| PyRuntimeError::new_err("tensor owner is poisoned"))?
            .take()
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "tensor ownership was already transferred; use the returned framework tensor",
                )
            })?;
        #[cfg(feature = "cuda")]
        if let Samples::Torch(tensor) = samples {
            let kwargs = PyDict::new(py);
            kwargs.set_item("stream", stream)?;
            // The underlying PyTorch object owns allocation and consumer-stream
            // lifetimes. Export it using its own protocol implementation.
            return Ok(tensor
                .bind(py)
                .call_method("__dlpack__", (), Some(&kwargs))?
                .unbind());
        }
        let _ = max_version; // A legacy capsule is valid for every advertised maximum version.
        let mut owner = Box::new(ManagedOwner {
            managed: DLManagedTensor {
                tensor: DLTensor {
                    data: samples.pointer(),
                    device: DLDevice {
                        kind: self.device.0,
                        id: self.device.1,
                    },
                    ndim: i32::try_from(self.shape.len())
                        .map_err(|_| PyValueError::new_err("tensor rank overflow"))?,
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
            _samples: samples,
            shape: self
                .shape
                .iter()
                .map(|&n| i64::try_from(n).expect("validated dimension"))
                .collect(),
        });
        owner.managed.tensor.shape = owner.shape.as_mut_ptr();
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
                unsafe { delete_managed(raw.cast()) };
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
#[repr(C)]
struct DLDevice {
    kind: i32,
    id: i32,
}
#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}
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
    _samples: Samples,
    shape: Vec<i64>,
}

unsafe extern "C" fn delete_managed(tensor: *mut DLManagedTensor) {
    if tensor.is_null() {
        return;
    }
    // SAFETY: this is called once by the DLPack consumer or by our unconsumed
    // capsule destructor, with the live pointer produced above. CPU Vec owners
    // may be freed on any thread, and no Rust access survives export.
    unsafe { drop(Box::from_raw((*tensor).context.cast::<ManagedOwner>())) };
}

unsafe extern "C" fn delete_capsule(capsule: *mut ffi::PyObject) {
    // SAFETY: Python calls this with a live capsule. Consumers rename it to
    // used_dltensor on taking ownership; only an unconsumed capsule frees here.
    if unsafe { ffi::PyCapsule_IsValid(capsule, c"dltensor".as_ptr()) } != 0 {
        // SAFETY: name validity was checked and the pointer is our live managed tensor.
        unsafe { delete_managed(ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr()).cast()) };
    }
}
