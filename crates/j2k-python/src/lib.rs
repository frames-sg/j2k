// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional Python extension. Codec and tensor ownership are implemented in Rust.

#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::{
    exceptions::{PyMemoryError, PyValueError},
    types::{PyBytes, PyDict},
};

#[cfg(feature = "cuda")]
mod cuda;
mod decode;
mod encode;
mod tensor;
mod transforms;
mod wavelets;

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

// Codec allocations are budgeted by the engine. Adapter metadata also uses
// fallible allocation so a failed reservation becomes a Python exception.
fn owned_items<T>(items: impl ExactSizeIterator<Item = T>) -> PyResult<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(items.len())
        .map_err(|error| PyMemoryError::new_err(error.to_string()))?;
    output.extend(items);
    Ok(output)
}

/// Inspect JPEG 2000/HTJ2K metadata without decoding pixels.
#[pyfunction]
fn inspect<'py>(py: Python<'py>, data: &Bound<'_, PyBytes>) -> PyResult<Bound<'py, PyDict>> {
    let bytes = data.as_bytes();
    let support = py.detach(|| j2k::J2kDecoder::inspect_support(bytes).map_err(value_error))?;
    let out = PyDict::new(py);
    out.set_item("width", support.info.dimensions.0)?;
    out.set_item("height", support.info.dimensions.1)?;
    out.set_item("components", support.info.components)?;
    out.set_item("bit_depth", support.info.bit_depth)?;
    out.set_item("signed", support.has_signed_components())?;
    out.set_item(
        "component_precision",
        support
            .components
            .iter()
            .map(|c| c.bit_depth)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "component_signed",
        support
            .components
            .iter()
            .map(|c| c.signed)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "component_sampling",
        support
            .components
            .iter()
            .map(|c| (c.x_rsiz, c.y_rsiz))
            .collect::<Vec<_>>(),
    )?;
    out.set_item("colorspace", format!("{:?}", support.info.colorspace))?;
    out.set_item("transfer_syntax", format!("{:?}", support.transfer_syntax))?;
    Ok(out)
}

#[pymodule]
#[pyo3(name = "j2k")]
fn j2k_python(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("cuda_enabled", cfg!(feature = "cuda"))?;
    module.add_class::<decode::Decoder>()?;
    module.add_class::<decode::ImageRequest>()?;
    module.add_class::<decode::PreparedBatch>()?;
    module.add_class::<decode::Batch>()?;
    module.add_class::<decode::BatchGroup>()?;
    module.add_class::<tensor::Tensor>()?;
    module.add_class::<wavelets::Wavelets>()?;
    module.add_class::<wavelets::Component>()?;
    module.add_class::<wavelets::Band>()?;
    module.add_function(wrap_pyfunction!(inspect, module)?)?;
    module.add_function(wrap_pyfunction!(encode::encode_rgb, module)?)?;
    module.add_function(wrap_pyfunction!(encode::decode_rgb, module)?)?;
    module.add_function(wrap_pyfunction!(wavelets::decode_wavelets, module)?)?;
    module.add_function(wrap_pyfunction!(transforms::forward_rgb97, module)?)?;
    module.add_function(wrap_pyfunction!(transforms::inverse_rgb97, module)?)?;
    Ok(())
}
