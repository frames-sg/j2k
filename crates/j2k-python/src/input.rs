// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoded input from Python bytes-like objects.
//!
//! Every decode, inspection and planning entry point accepts `bytes` or any
//! one-dimensional, C-contiguous buffer of unsigned bytes (`bytearray`,
//! `memoryview`, `mmap.mmap`, ...). A `bytes` object is borrowed: it is
//! immutable, and the call's argument keeps it alive while the interpreter is
//! detached. Other buffers are copied once, under the interpreter, before the
//! codec runs, so later writes to them cannot reach bytes the codec is reading.

use std::sync::Arc;

use pyo3::{
    exceptions::{PyBufferError, PyTypeError},
    prelude::*,
    types::{PyBytes, PyMemoryView},
};

/// Complete encoded bytes taken from one Python bytes-like object.
pub(crate) struct EncodedInput<'py> {
    bytes: Bound<'py, PyBytes>,
}

impl<'py> EncodedInput<'py> {
    /// Borrow `bytes`, or copy any other one-dimensional, C-contiguous
    /// unsigned-byte buffer. Other objects raise `TypeError`; other buffer
    /// layouts raise `BufferError`.
    pub(crate) fn extract(data: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(bytes) = data.cast::<PyBytes>() {
            return Ok(Self {
                bytes: bytes.clone(),
            });
        }
        let view = PyMemoryView::from(data).map_err(|_| not_bytes_like(data))?;
        check_layout(&view)?;
        let bytes = view.call_method0("tobytes")?.cast_into::<PyBytes>()?;
        Ok(Self { bytes })
    }

    /// The encoded bytes, valid while this input is held.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.bytes.as_bytes()
    }

    /// An owned copy for the batch decoder, whose inputs own their bytes.
    pub(crate) fn to_shared(&self) -> Arc<[u8]> {
        Arc::from(self.as_bytes())
    }
}

fn not_bytes_like(data: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "data must be bytes or a bytes-like object such as bytearray, memoryview or mmap, not {}",
        type_name(data)
    ))
}

fn type_name(data: &Bound<'_, PyAny>) -> String {
    data.get_type()
        .name()
        .map_or_else(|_| String::from("object"), |name| name.to_string())
}

/// Accept one dimension of unsigned bytes in C order, as the codec reads them.
fn check_layout(view: &Bound<'_, PyMemoryView>) -> PyResult<()> {
    let format = view.getattr("format")?.extract::<String>()?;
    if !matches!(format.as_str(), "B" | "c") || view.getattr("itemsize")?.extract::<usize>()? != 1 {
        return Err(PyTypeError::new_err(format!(
            "data must be a buffer of unsigned bytes (format 'B'), not format {format:?}"
        )));
    }
    if view.getattr("ndim")?.extract::<usize>()? != 1
        || !view.getattr("c_contiguous")?.extract::<bool>()?
    {
        return Err(PyBufferError::new_err(
            "data must be a one-dimensional C-contiguous buffer; pass bytes(data) to copy it",
        ));
    }
    Ok(())
}
