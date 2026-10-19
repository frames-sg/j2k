// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    forward_dwt97_reference, forward_ict_reference, inverse_dwt97_reference, inverse_ict_reference,
    J2kForwardDwt97Level, J2kForwardDwt97Output,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

use crate::{owned_items, value_error};

type OutputBand = (u8, u8, &'static str, u32, u32, Py<PyBytes>);
type InputBand = (u8, u8, String, Vec<u8>);

fn validate_geometry(width: u32, height: u32, levels: u8, color: &str) -> PyResult<usize> {
    let pixels = u64::from(width) * u64::from(height);
    if width == 0 || height == 0 || pixels > 4_194_304 {
        return Err(PyValueError::new_err(
            "float transforms require 1 to 4194304 pixels",
        ));
    }
    if levels > width.min(height).ilog2() as u8 {
        return Err(PyValueError::new_err(
            "decomposition depth exceeds the image geometry",
        ));
    }
    if color != "rgb" && color != "ict" {
        return Err(PyValueError::new_err("color must be rgb or ict"));
    }
    Ok(pixels as usize)
}

fn floats(bytes: &[u8], count: usize) -> PyResult<Vec<f32>> {
    if bytes.len() != count * 4 {
        return Err(PyValueError::new_err(
            "float32 byte length does not match geometry",
        ));
    }
    let values = owned_items(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v)),
    )?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("float samples must be finite"));
    }
    Ok(values)
}

fn bytes<'py>(py: Python<'py>, values: &[f32]) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, values.len() * 4, |output| {
        for (value, target) in values.iter().zip(output.as_chunks_mut::<4>().0.iter_mut()) {
            target.copy_from_slice(&value.to_le_bytes());
        }
        Ok(())
    })
}

/// Analyze float32 little-endian HWC RGB; detail level 1 is finest, 0 is LL.
#[pyfunction]
#[pyo3(signature = (data, width, height, *, levels=5, color="ict"))]
pub(crate) fn forward_rgb97(
    py: Python<'_>,
    data: &Bound<'_, PyBytes>,
    width: u32,
    height: u32,
    levels: u8,
    color: &str,
) -> PyResult<Vec<OutputBand>> {
    let count = validate_geometry(width, height, levels, color)?;
    let pixels = floats(data.as_bytes(), count * 3)?;
    let trees = py.detach(|| {
        let mut planes =
            owned_items((0..3).map(|c| owned_items((0..count).map(|i| pixels[3 * i + c]))))?
                .into_iter()
                .collect::<PyResult<Vec<_>>>()?;
        if color == "ict" {
            planes = forward_ict_reference(planes);
        }
        planes
            .iter()
            .map(|p| forward_dwt97_reference(p, width, height, levels).map_err(value_error))
            .collect::<PyResult<Vec<_>>>()
    })?;
    let mut output = Vec::new();
    for (component, tree) in trees.into_iter().enumerate() {
        output.push((
            component as u8,
            0,
            "LL",
            tree.ll_width,
            tree.ll_height,
            bytes(py, &tree.ll)?.unbind(),
        ));
        for (index, level) in tree.levels.into_iter().enumerate() {
            let fine_level = levels - index as u8;
            for (orientation, values, w, h) in [
                ("HL", level.hl, level.high_width, level.low_height),
                ("LH", level.lh, level.low_width, level.high_height),
                ("HH", level.hh, level.high_width, level.high_height),
            ] {
                output.push((
                    component as u8,
                    fine_level,
                    orientation,
                    w,
                    h,
                    bytes(py, &values)?.unbind(),
                ));
            }
        }
    }
    Ok(output)
}

fn take_band(
    bands: &mut Vec<InputBand>,
    component: u8,
    level: u8,
    orientation: &str,
    width: u32,
    height: u32,
) -> PyResult<Vec<f32>> {
    let matching: Vec<_> = bands
        .iter()
        .enumerate()
        .filter(|(_, b)| b.0 == component && b.1 == level && b.2 == orientation)
        .map(|(i, _)| i)
        .collect();
    if matching.len() != 1 {
        return Err(PyValueError::new_err(
            "each component, level and orientation must occur once",
        ));
    }
    let (_, _, _, raw) = bands.swap_remove(matching[0]);
    floats(&raw, width as usize * height as usize)
}

fn component_tree(
    bands: &mut Vec<InputBand>,
    component: u8,
    width: u32,
    height: u32,
    depth: u8,
) -> PyResult<J2kForwardDwt97Output> {
    let divisor = 1_u32 << depth;
    let ll_width = width.div_ceil(divisor);
    let ll_height = height.div_ceil(divisor);
    let ll = take_band(bands, component, 0, "LL", ll_width, ll_height)?;
    let mut levels = Vec::new();
    for fine_level in (1..=depth).rev() {
        let factor = 1_u32 << (fine_level - 1);
        let w = width.div_ceil(factor);
        let h = height.div_ceil(factor);
        let (lw, lh, hw, hh) = (w.div_ceil(2), h.div_ceil(2), w / 2, h / 2);
        levels.push(J2kForwardDwt97Level {
            hl: take_band(bands, component, fine_level, "HL", hw, lh)?,
            lh: take_band(bands, component, fine_level, "LH", lw, hh)?,
            hh: take_band(bands, component, fine_level, "HH", hw, hh)?,
            width: w,
            height: h,
            low_width: lw,
            low_height: lh,
            high_width: hw,
            high_height: hh,
        });
    }
    Ok(J2kForwardDwt97Output {
        ll,
        ll_width,
        ll_height,
        levels,
    })
}

/// Reconstruct edited coefficients into unclipped float32 little-endian HWC RGB.
#[pyfunction]
#[pyo3(signature = (bands, width, height, *, levels=5, color="ict"))]
pub(crate) fn inverse_rgb97<'py>(
    py: Python<'py>,
    mut bands: Vec<InputBand>,
    width: u32,
    height: u32,
    levels: u8,
    color: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let count = validate_geometry(width, height, levels, color)?;
    if bands.len() != 3 * (1 + 3 * usize::from(levels)) {
        return Err(PyValueError::new_err(
            "coefficient band count does not match decomposition depth",
        ));
    }
    let output = py.detach(|| {
        let mut planes = Vec::new();
        for component in 0..3 {
            let tree = component_tree(&mut bands, component, width, height, levels)?;
            planes.push(inverse_dwt97_reference(&tree).map_err(value_error)?);
        }
        if color == "ict" {
            planes = inverse_ict_reference(planes).map_err(value_error)?;
        }
        owned_items((0..count * 3).map(|i| planes[i % 3][i / 3]))
    })?;
    if output.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "reconstruction produced nonfinite pixels",
        ));
    }
    bytes(py, &output)
}
