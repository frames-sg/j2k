// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoded input from Python bytes-like objects.
//!
//! Every decode, inspection and planning entry point accepts `bytes` or any
//! one-dimensional, C-contiguous buffer of unsigned bytes (`bytearray`,
//! `memoryview`, `mmap.mmap`, ...). The codec reads the bytes with the
//! interpreter detached, so it may only read memory that nothing can change
//! while it does:
//!
//! - `bytes`, and a read-only view whose exporting object is `bytes` or a
//!   read-only (`ACCESS_READ`) `mmap.mmap`, is borrowed in place; the buffer
//!   export is held for as long as the borrow lives;
//! - every other buffer, including a read-only view of a writable exporter
//!   such as `memoryview(bytearray).toreadonly()`, is copied once with the
//!   interpreter attached, before the codec runs.
//!
//! A read-only mmap is immutable only through that mmap object: see
//! [`EncodedInput::as_bytes`] for what another writer to the file can do.

use std::{ptr::NonNull, sync::Arc};

use pyo3::{
    buffer::{PyBuffer, PyUntypedBuffer},
    exceptions::{PyBufferError, PyTypeError},
    prelude::*,
    sync::PyOnceLock,
    types::{PyBytes, PyMemoryView, PyType},
};

/// Complete encoded bytes taken from one Python bytes-like object.
pub(crate) struct EncodedInput<'py> {
    source: Source<'py>,
}

enum Source<'py> {
    /// A `bytes` object, which the call's argument keeps alive.
    Bytes(Bound<'py, PyBytes>),
    /// A held export of memory that this process cannot change, and the
    /// contiguous bytes it describes.
    View {
        bytes: NonNull<[u8]>,
        _export: PyBuffer<u8>,
    },
    /// A copy of a buffer whose memory can change.
    Copied(Vec<u8>),
}

impl<'py> EncodedInput<'py> {
    /// Borrow `bytes` or an immutable buffer, or copy any other
    /// one-dimensional, C-contiguous unsigned-byte buffer. Other objects
    /// raise `TypeError`; other buffer layouts raise `BufferError`.
    pub(crate) fn extract(data: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(bytes) = data.cast::<PyBytes>() {
            return Ok(Self {
                source: Source::Bytes(bytes.clone()),
            });
        }
        let export = PyUntypedBuffer::get(data).map_err(|_| not_bytes_like(data))?;
        let format = export.format().to_string_lossy().into_owned();
        if !matches!(format.as_str(), "B" | "c") || export.item_size() != 1 {
            return Err(PyTypeError::new_err(format!(
                "data must be a buffer of unsigned bytes (format 'B'), not format {format:?}"
            )));
        }
        let export = export.into_typed::<u8>()?;
        let bytes = export
            .as_slice_ptr()
            .filter(|_| export.dimensions() == 1)
            .ok_or_else(|| {
                PyBufferError::new_err(
                    "data must be a one-dimensional C-contiguous buffer; pass bytes(data) to copy it",
                )
            })?;
        let source = if exporter_is_immutable(data, &export)? {
            Source::View {
                bytes,
                _export: export,
            }
        } else {
            Source::Copied(export.to_vec(data.py())?)
        };
        Ok(Self { source })
    }

    /// The encoded bytes, valid while this input is held.
    ///
    /// A borrowed read-only `mmap` still reflects its file. If another
    /// process, or another writable mapping, writes the file while the codec
    /// reads it, the bytes change under a shared borrow: the codec
    /// bounds-checks every read against the unchanged length, so the usual
    /// result is a decode error or wrong pixels, but Rust gives no guarantee
    /// for memory that changes under a shared reference. If the file is
    /// truncated, reading the vanished pages raises `SIGBUS`, as any read of
    /// that map would, including a copy. Callers that cannot keep a mapped
    /// file unchanged and untruncated for the call should pass `bytes`.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match &self.source {
            Source::Bytes(bytes) => bytes.as_bytes(),
            Source::Copied(bytes) => bytes,
            // SAFETY: `bytes` is the C-contiguous, one-dimensional, one-byte
            // item region of `_export`, which this input holds: the exporter
            // keeps that memory allocated and mapped until the export is
            // released (bytes objects stay referenced; an exported mmap refuses
            // close() and resize()). `exporter_is_immutable` admitted only
            // exporters that cannot be written through while the export is
            // held: bytes, and read-only mmaps whose own export is read-only.
            // Writes to a mapped file through anything else (another mapping
            // or file descriptor, in this process or another) are the
            // documented caller contract above.
            Source::View { bytes, .. } => unsafe { bytes.as_ref() },
        }
    }

    /// An owned copy for the batch decoder, whose inputs own their bytes.
    pub(crate) fn to_shared(&self) -> Arc<[u8]> {
        Arc::from(self.as_bytes())
    }
}

/// Whether nothing in this process can change `export`'s memory while it is
/// held: its exporting object is `bytes`, or an `mmap` whose own export is
/// read-only. The view's own read-only flag is not enough: a read-only view
/// of a `bytearray` can still change through the `bytearray`.
fn exporter_is_immutable(data: &Bound<'_, PyAny>, export: &PyBuffer<u8>) -> PyResult<bool> {
    if !export.readonly() {
        return Ok(false);
    }
    let exporter = match data.cast::<PyMemoryView>() {
        Ok(view) => view.getattr("obj")?,
        Err(_) => data.clone(),
    };
    if exporter.is_instance_of::<PyBytes>() {
        return Ok(true);
    }
    if !exporter.is_instance(mmap_type(data.py())?)? {
        return Ok(false);
    }
    Ok(PyUntypedBuffer::get(&exporter)?.readonly())
}

fn mmap_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static MMAP: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    MMAP.import(py, "mmap", "mmap")
}

fn not_bytes_like(data: &Bound<'_, PyAny>) -> PyErr {
    let kind = data
        .get_type()
        .name()
        .map_or_else(|_| String::from("object"), |name| name.to_string());
    PyTypeError::new_err(format!(
        "data must be bytes or a bytes-like object such as bytearray, memoryview or mmap, not {kind}"
    ))
}
