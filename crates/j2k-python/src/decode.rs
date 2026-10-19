// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use j2k::{
    BatchDecodeOptions, BatchGroupInfo, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, DecodeSettings, Downscale, EncodedImage, Rect,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::PyBytes,
};

use crate::{
    owned_items,
    tensor::{DType, Samples, Tensor},
    value_error,
};

/// One encoded image with its own full/region/reduced request.
#[pyclass(name = "ImageRequest", module = "j2k", frozen)]
pub(crate) struct ImageRequest {
    pub encoded: EncodedImage,
}

#[pymethods]
impl ImageRequest {
    #[new]
    #[pyo3(signature = (data, *, region=None, reduction=0))]
    fn new(
        data: &Bound<'_, PyBytes>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
    ) -> PyResult<Self> {
        Ok(Self {
            encoded: EncodedImage::new(Arc::from(data.as_bytes()), request(region, reduction)?),
        })
    }
}

/// Reusable codec preparation, including per-input errors and source order.
#[pyclass(name = "PreparedBatch", module = "j2k", frozen)]
pub(crate) struct PreparedBatch {
    pub inner: j2k::PreparedBatch,
}

#[pymethods]
impl PreparedBatch {
    #[getter]
    fn group_count(&self) -> usize {
        self.inner.groups().len()
    }
    #[getter]
    fn errors(&self) -> Vec<(usize, String)> {
        self.inner
            .errors()
            .iter()
            .map(|e| (e.index, e.source.to_string()))
            .collect()
    }
}

/// One homogeneous tensor group. source_indices aligns labels with tensor rows.
#[pyclass(name = "BatchGroup", module = "j2k", frozen)]
pub(crate) struct BatchGroup {
    #[pyo3(get)]
    pub tensor: Py<Tensor>,
    #[pyo3(get)]
    pub source_indices: Vec<usize>,
    #[pyo3(get)]
    pub decoded_rects: Vec<(u32, u32, u32, u32)>,
    #[pyo3(get)]
    pub warnings: Vec<Vec<String>>,
    #[pyo3(get)]
    pub precision: u8,
    #[pyo3(get)]
    pub signed: bool,
    #[pyo3(get)]
    pub color: String,
    #[pyo3(get)]
    pub alpha: String,
    #[pyo3(get)]
    pub layout: String,
}

/// Successful groups plus per-input and per-group errors. No implicit padding.
#[pyclass(name = "Batch", module = "j2k", frozen)]
pub(crate) struct Batch {
    #[pyo3(get)]
    pub groups: Vec<Py<BatchGroup>>,
    #[pyo3(get)]
    pub errors: Vec<(usize, String)>,
    #[pyo3(get)]
    pub group_errors: Vec<(Vec<usize>, String)>,
}

pub(crate) struct MaterializedGroup {
    pub info: BatchGroupInfo,
    pub indices: Vec<usize>,
    pub rects: Vec<Rect>,
    pub warnings: Vec<Vec<String>>,
    pub tensor: Tensor,
}

impl MaterializedGroup {
    pub fn into_python(self, py: Python<'_>) -> PyResult<Py<BatchGroup>> {
        Py::new(
            py,
            BatchGroup {
                tensor: Py::new(py, self.tensor)?,
                source_indices: self.indices,
                decoded_rects: self
                    .rects
                    .into_iter()
                    .map(|r| (r.x, r.y, r.w, r.h))
                    .collect(),
                warnings: self.warnings,
                precision: self.info.precision,
                signed: self.info.signed,
                color: format!("{:?}", self.info.color),
                alpha: format!("{:?}", self.info.alpha),
                layout: match self.info.layout {
                    BatchLayout::Nchw => "nchw",
                    _ => "nhwc",
                }
                .into(),
            },
        )
    }
}

/// Persistent decoder. Construct sessions inside DataLoader workers, not before forking.
#[pyclass(name = "Decoder", module = "j2k", frozen)]
pub(crate) struct Decoder {
    process_id: u32,
    options: BatchDecodeOptions,
    cpu: Mutex<CpuBatchDecoder>,
    #[cfg(feature = "cuda")]
    cuda: Option<crate::cuda::CudaDecoder>,
}

#[pymethods]
impl Decoder {
    #[new]
    #[pyo3(signature = (*, layout="nchw", workers=None, strict=true, device="cpu"))]
    fn new(layout: &str, workers: Option<usize>, strict: bool, device: &str) -> PyResult<Self> {
        let layout = match layout {
            "nchw" => BatchLayout::Nchw,
            "nhwc" => BatchLayout::Nhwc,
            _ => return Err(PyValueError::new_err("layout must be nchw or nhwc")),
        };
        let workers = workers
            .map(|n| {
                NonZeroUsize::new(n)
                    .ok_or_else(|| PyValueError::new_err("workers must be positive"))
            })
            .transpose()?;
        let options = BatchDecodeOptions {
            layout,
            workers,
            settings: if strict {
                DecodeSettings::strict()
            } else {
                DecodeSettings::lenient()
            },
        };
        #[cfg(feature = "cuda")]
        let cuda = if device == "cpu" {
            None
        } else {
            Some(crate::cuda::CudaDecoder::new(device, options)?)
        };
        #[cfg(not(feature = "cuda"))]
        if device != "cpu" {
            return Err(PyValueError::new_err(
                "this build supports device='cpu'; CUDA requires the optional cuda build",
            ));
        }
        Ok(Self {
            process_id: std::process::id(),
            options,
            cpu: Mutex::new(CpuBatchDecoder::new(options)),
            #[cfg(feature = "cuda")]
            cuda,
        })
    }

    /// Prepare bytes or ImageRequest objects once for repeated batch decode.
    #[pyo3(signature = (inputs, *, region=None, reduction=0))]
    fn prepare(
        &self,
        py: Python<'_>,
        inputs: Vec<Bound<'_, PyAny>>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
    ) -> PyResult<PreparedBatch> {
        self.check_process()?;
        let inputs = encoded_inputs(inputs, request(region, reduction)?)?;
        py.detach(|| {
            self.cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?
                .prepare(inputs)
                .map(|inner| PreparedBatch { inner })
                .map_err(value_error)
        })
    }

    /// Decode bytes or per-input requests, grouping compatible tensor shapes.
    #[pyo3(signature = (inputs, *, region=None, reduction=0))]
    fn decode_batch(
        &self,
        py: Python<'_>,
        inputs: Vec<Bound<'_, PyAny>>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
    ) -> PyResult<Batch> {
        let prepared = self.prepare(py, inputs, region, reduction)?;
        self.decode_prepared(py, &prepared)
    }

    /// Reuse preparation while retaining source ordering and all item failures.
    fn decode_prepared(&self, py: Python<'_>, prepared: &PreparedBatch) -> PyResult<Batch> {
        self.check_process()?;
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            return cuda.decode(py, &prepared.inner);
        }
        let decoded = py.detach(|| {
            self.cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?
                .decode_prepared(&prepared.inner)
                .map_err(value_error)
        })?;
        let (groups, errors) = decoded.into_parts();
        let groups = groups
            .into_iter()
            .map(|group| cpu_group(group)?.into_python(py))
            .collect::<PyResult<_>>()?;
        Ok(Batch {
            groups,
            errors: errors
                .into_iter()
                .map(|e| (e.index, e.source.to_string()))
                .collect(),
            group_errors: Vec::new(),
        })
    }

    /// Decode one image to CHW or HWC native integer samples, with no batch axis.
    #[pyo3(signature = (data, *, region=None, reduction=0))]
    fn decode(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
    ) -> PyResult<Tensor> {
        self.check_process()?;
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            return cuda.decode_one(py, data.as_bytes(), request(region, reduction)?);
        }
        let input = EncodedImage::new(Arc::from(data.as_bytes()), request(region, reduction)?);
        let decoded = py.detach(|| {
            self.cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?
                .decode(owned_items([input].into_iter())?)
                .map_err(value_error)
        })?;
        let (mut groups, errors) = decoded.into_parts();
        if !errors.is_empty() {
            return Err(value_error(&errors[0].source));
        }
        if groups.len() != 1 {
            return Err(PyRuntimeError::new_err(
                "single-image decode did not return exactly one group",
            ));
        }
        let mut tensor = cpu_group(groups.remove(0))?.tensor;
        tensor.remove_batch_axis()?;
        Ok(tensor)
    }

    fn __repr__(&self) -> String {
        format!("j2k.Decoder(layout={:?})", self.options.layout)
    }
}

impl Decoder {
    fn check_process(&self) -> PyResult<()> {
        if self.process_id != std::process::id() {
            return Err(PyRuntimeError::new_err("decoder was inherited across fork; construct a new Decoder inside each DataLoader worker"));
        }
        Ok(())
    }
}

fn request(region: Option<(u32, u32, u32, u32)>, reduction: u8) -> PyResult<DecodeRequest> {
    let scale = match reduction {
        0 => Downscale::None,
        1 => Downscale::Half,
        2 => Downscale::Quarter,
        3 => Downscale::Eighth,
        _ => {
            return Err(PyValueError::new_err(
                "reduction must be 0, 1, 2, or 3 (full, half, quarter, eighth)",
            ))
        }
    };
    let roi = region.map(|(x, y, w, h)| Rect { x, y, w, h });
    Ok(match (roi, reduction) {
        (None, 0) => DecodeRequest::Full,
        (Some(roi), 0) => DecodeRequest::Region { roi },
        (None, _) => DecodeRequest::Reduced { scale },
        (Some(roi), _) => DecodeRequest::RegionReduced { roi, scale },
    })
}

fn encoded_inputs(
    inputs: Vec<Bound<'_, PyAny>>,
    default_request: DecodeRequest,
) -> PyResult<Vec<EncodedImage>> {
    inputs
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            if let Ok(bytes) = item.cast::<PyBytes>() {
                return Ok(EncodedImage::new(
                    Arc::from(bytes.as_bytes()),
                    default_request,
                ));
            }
            let request = item.extract::<PyRef<'_, ImageRequest>>().map_err(|_| {
                let kind = item
                    .get_type()
                    .name()
                    .map_or_else(|_| String::from("object"), |name| name.to_string());
                PyTypeError::new_err(format!(
                    "input {index}: expected bytes or ImageRequest, not {kind}"
                ))
            })?;
            if default_request != DecodeRequest::Full {
                return Err(PyValueError::new_err(format!(
                    "input {index}: use per-image ImageRequest options or batch options, not both"
                )));
            }
            Ok(request.encoded.clone())
        })
        .collect()
}

pub(crate) fn shape(info: &BatchGroupInfo, batch: usize) -> PyResult<Vec<usize>> {
    let (w, h) = info.dimensions;
    let c = info.color.channels();
    match info.layout {
        BatchLayout::Nchw => owned_items([batch, c, h as usize, w as usize].into_iter()),
        BatchLayout::Nhwc => owned_items([batch, h as usize, w as usize, c].into_iter()),
        _ => Err(PyValueError::new_err("unsupported codec tensor layout")),
    }
}

fn cpu_group(group: j2k::CpuBatchGroup) -> PyResult<MaterializedGroup> {
    let (info, indices, rects, warnings, values) = group.into_parts();
    let dimensions = shape(&info, indices.len())?;
    let (samples, dtype) = match values {
        CpuBatchSamples::U8(v) => (Samples::U8(v), DType::U8),
        CpuBatchSamples::U16(v) => (Samples::U16(v), DType::U16),
        CpuBatchSamples::I16(v) => (Samples::I16(v), DType::I16),
        _ => return Err(PyValueError::new_err("unsupported codec sample type")),
    };
    Ok(MaterializedGroup {
        info,
        indices,
        rects,
        warnings: warnings
            .into_iter()
            .map(|w| w.into_iter().map(|v| v.to_string()).collect())
            .collect(),
        tensor: Tensor::new(samples, dimensions, dtype, (1, 0))?,
    })
}
