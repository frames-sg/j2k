// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    owned_items,
    tensor::{DType, Samples, Tensor},
    value_error,
};
use j2k::{J2kDecoder, WaveletBandKind, WaveletSamples, WaveletSelection, WaveletTransform};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

#[pyclass(name = "WaveletBand", module = "j2k", frozen)]
pub(crate) struct Band {
    #[pyo3(get)]
    tensor: Py<Tensor>,
    #[pyo3(get)]
    kind: &'static str,
    #[pyo3(get)]
    resolution: u8,
    #[pyo3(get)]
    rect: (u32, u32, u32, u32),
    #[pyo3(get)]
    quantization_step: f32,
}

#[pyclass(name = "WaveletComponent", module = "j2k", frozen)]
pub(crate) struct Component {
    #[pyo3(get)]
    tile_index: u32,
    #[pyo3(get)]
    tile_rect: (u32, u32, u32, u32),
    #[pyo3(get)]
    index: u16,
    #[pyo3(get)]
    rect: (u32, u32, u32, u32),
    #[pyo3(get)]
    precision: u8,
    #[pyo3(get)]
    signed: bool,
    #[pyo3(get)]
    sampling: (u8, u8),
    #[pyo3(get)]
    uses_color_transform: bool,
    #[pyo3(get)]
    transform: &'static str,
    #[pyo3(get)]
    decomposition_levels: u8,
    #[pyo3(get)]
    bands: Vec<Py<Band>>,
}

/// Tile/component/band output. Coefficients remain in the native transform domain.
#[pyclass(name = "WaveletImage", module = "j2k", frozen)]
pub(crate) struct Wavelets {
    #[pyo3(get)]
    rect: (u32, u32, u32, u32),
    #[pyo3(get)]
    components: Vec<Py<Component>>,
}

/// Decode selected bands on CPU. Empty selectors mean all; resolution 0 is LL.
/// Returns exact I64 reversible coefficients or dequantized F32 irreversible
/// coefficients. Pixel/ICC/channel conversion and model normalization are not applied.
#[pyfunction]
#[pyo3(signature = (data, *, tiles=None, components=None, resolutions=None, bands=None))]
pub(crate) fn decode_wavelets(
    py: Python<'_>,
    data: &Bound<'_, PyBytes>,
    tiles: Option<Vec<u32>>,
    components: Option<Vec<u16>>,
    resolutions: Option<Vec<u8>>,
    bands: Option<Vec<String>>,
) -> PyResult<Wavelets> {
    let selection = WaveletSelection {
        tiles: tiles.unwrap_or_default(),
        components: components.unwrap_or_default(),
        resolutions: resolutions.unwrap_or_default(),
        bands: bands
            .unwrap_or_default()
            .iter()
            .map(|s| match s.as_str() {
                "LL" => Ok(WaveletBandKind::LowLow),
                "HL" => Ok(WaveletBandKind::HighLow),
                "LH" => Ok(WaveletBandKind::LowHigh),
                "HH" => Ok(WaveletBandKind::HighHigh),
                _ => Err(PyValueError::new_err("bands must be LL, HL, LH, or HH")),
            })
            .collect::<PyResult<_>>()?,
    };
    // PyBytes is immutable and the borrowed Python argument retains its owner
    // for the entire call, including while the interpreter is detached.
    let bytes = data.as_bytes();
    let output = py.detach(|| {
        J2kDecoder::new(bytes)
            .and_then(|mut d| d.decode_wavelets(&selection))
            .map_err(value_error)
    })?;
    let mut components = Vec::new();
    for tile in output.tiles {
        for component in tile.components {
            let bands = component
                .bands
                .into_iter()
                .map(|band| {
                    let (samples, dtype) = match band.samples {
                        WaveletSamples::I64(v) => (Samples::I64(v), DType::I64),
                        WaveletSamples::F32(v) => (Samples::F32(v), DType::F32),
                    };
                    let tensor = Tensor::new(
                        samples,
                        owned_items(
                            [band.rect.height() as usize, band.rect.width() as usize].into_iter(),
                        )?,
                        dtype,
                        (1, 0),
                    )?;
                    Py::new(
                        py,
                        Band {
                            tensor: Py::new(py, tensor)?,
                            kind: match band.kind {
                                WaveletBandKind::LowLow => "LL",
                                WaveletBandKind::HighLow => "HL",
                                WaveletBandKind::LowHigh => "LH",
                                WaveletBandKind::HighHigh => "HH",
                            },
                            resolution: band.resolution,
                            rect: rect(band.rect),
                            quantization_step: band.quantization_step,
                        },
                    )
                })
                .collect::<PyResult<_>>()?;
            components.push(Py::new(
                py,
                Component {
                    tile_index: tile.index,
                    tile_rect: rect(tile.rect),
                    index: component.index,
                    rect: rect(component.rect),
                    precision: component.precision,
                    signed: component.signed,
                    sampling: component.sampling,
                    uses_color_transform: tile.uses_color_transform,
                    transform: match component.transform {
                        WaveletTransform::Reversible53 => "5/3",
                        WaveletTransform::Irreversible97 => "9/7",
                    },
                    decomposition_levels: component.decomposition_levels,
                    bands,
                },
            )?);
        }
    }
    Ok(Wavelets {
        rect: rect(output.rect),
        components,
    })
}

fn rect(r: j2k::WaveletRect) -> (u32, u32, u32, u32) {
    (r.x0, r.y0, r.width(), r.height())
}
