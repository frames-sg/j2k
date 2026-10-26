// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use j2k::{
    BatchAlpha, BatchColor, BatchDecodeOptions, BatchGroupInfo, BatchLayout, CpuBatchDecoder,
    CpuBatchSamples, DecodeRequest, DecodeSettings, Downscale, EncodedImage, Rect,
};
use j2k_jpeg::{
    ColorTransform, DecodeOptions as JpegOptions, JpegBatchDecoder, JpegBatchGroup, JpegBatchInput,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::PyBytes,
};

use crate::{
    is_jpeg, owned_items,
    tensor::{DType, Samples, Tensor},
    value_error,
};

const JPEG_FULL_ONLY: &str = "JPEG inputs support full-image decode only (no region or reduction)";

/// One encoded image with its own full/region/reduced request.
#[pyclass(name = "ImageRequest", module = "j2k", frozen)]
#[derive(Clone)]
pub(crate) struct ImageRequest {
    pub encoded: EncodedImage,
    jpeg_options: JpegOptions,
}

#[pymethods]
impl ImageRequest {
    #[new]
    #[pyo3(signature = (data, *, region=None, reduction=0, color_transform="auto"))]
    fn new(
        data: &Bound<'_, PyBytes>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
        color_transform: &str,
    ) -> PyResult<Self> {
        let transform = match color_transform {
            "auto" => ColorTransform::Auto,
            "ycbcr" => ColorTransform::ForceYCbCr,
            "none" => ColorTransform::ForceRgb,
            _ => {
                return Err(PyValueError::new_err(
                    "color_transform must be auto, ycbcr, or none",
                ))
            }
        };
        if transform != ColorTransform::Auto && !is_jpeg(data.as_bytes()) {
            return Err(PyValueError::new_err(
                "color_transform overrides are supported only for JPEG inputs",
            ));
        }
        Ok(Self {
            encoded: EncodedImage::new(Arc::from(data.as_bytes()), request(region, reduction)?),
            jpeg_options: JpegOptions::default().with_color_transform(transform),
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
    pub recovered: Vec<bool>,
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

/// Python-facing description shared by every image in a group.
pub(crate) struct GroupMeta {
    pub precision: u8,
    pub signed: bool,
    pub color: BatchColor,
    pub alpha: BatchAlpha,
    pub layout: BatchLayout,
}

impl GroupMeta {
    pub fn of(info: &BatchGroupInfo) -> Self {
        Self {
            precision: info.precision,
            signed: info.signed,
            color: info.color,
            alpha: info.alpha,
            layout: info.layout,
        }
    }
}

pub(crate) struct MaterializedGroup {
    pub meta: GroupMeta,
    pub indices: Vec<usize>,
    pub rects: Vec<Rect>,
    pub warnings: Vec<Vec<String>>,
    pub recovered: Vec<bool>,
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
                recovered: self.recovered,
                precision: self.meta.precision,
                signed: self.meta.signed,
                color: match self.meta.color {
                    BatchColor::Rgb => "Rgb",
                    BatchColor::Gray => "Gray",
                    BatchColor::Rgba => "Rgba",
                    _ => return Err(PyValueError::new_err("unsupported output color")),
                }
                .into(),
                alpha: match self.meta.alpha {
                    BatchAlpha::None => "None",
                    BatchAlpha::Straight => "Straight",
                    BatchAlpha::Premultiplied => "Premultiplied",
                    _ => return Err(PyValueError::new_err("unsupported output alpha")),
                }
                .into(),
                layout: match self.meta.layout {
                    BatchLayout::Nchw => "nchw",
                    _ => "nhwc",
                }
                .into(),
            },
        )
    }
}

/// Decoded groups and failures before they become Python objects.
#[derive(Default)]
pub(crate) struct Decoded {
    pub groups: Vec<MaterializedGroup>,
    pub errors: Vec<(usize, String)>,
    pub group_errors: Vec<(Vec<usize>, String)>,
}

impl Decoded {
    /// Append a sub-batch result, mapping its indices to caller positions.
    fn extend_from(&mut self, other: Self, positions: &[usize]) {
        let remap = |index: &mut usize| *index = positions[*index];
        for mut group in other.groups {
            group.indices.iter_mut().for_each(remap);
            self.groups.push(group);
        }
        for (mut index, error) in other.errors {
            remap(&mut index);
            self.errors.push((index, error));
        }
        for (mut indices, error) in other.group_errors {
            indices.iter_mut().for_each(remap);
            self.group_errors.push((indices, error));
        }
    }

    fn into_batch(mut self, py: Python<'_>) -> PyResult<Batch> {
        self.errors.sort_unstable_by_key(|(index, _)| *index);
        Ok(Batch {
            groups: self
                .groups
                .into_iter()
                .map(|group| group.into_python(py))
                .collect::<PyResult<_>>()?,
            errors: self.errors,
            group_errors: self.group_errors,
        })
    }
}

/// Persistent decoder. Construct sessions inside DataLoader workers, not before forking.
#[pyclass(name = "Decoder", module = "j2k", frozen)]
pub(crate) struct Decoder {
    process_id: u32,
    options: BatchDecodeOptions,
    cpu: Mutex<CpuBatchDecoder>,
    jpeg: Mutex<JpegBatchDecoder>,
    #[cfg(feature = "cuda")]
    cuda: Option<crate::cuda::CudaDecoder>,
}

#[pymethods]
impl Decoder {
    #[new]
    #[pyo3(signature = (*, layout="nchw", workers=None, strict=true, device="cpu"))]
    fn new(
        py: Python<'_>,
        layout: &str,
        workers: Option<usize>,
        strict: bool,
        device: &str,
    ) -> PyResult<Self> {
        #[cfg(not(feature = "cuda"))]
        let _ = py;
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
        let options = {
            let mut options = BatchDecodeOptions::default();
            options.layout = layout;
            options.workers = workers;
            options.settings = if strict {
                DecodeSettings::strict()
            } else {
                DecodeSettings::lenient()
            };
            options
        };
        let is_cuda = device == "cuda" || device.starts_with("cuda:");
        match device {
            "cpu" => {}
            _ if is_cuda && cfg!(feature = "cuda") => {}
            _ if is_cuda => {
                return Err(PyValueError::new_err(
                    "CUDA requires the optional cuda build",
                ))
            }
            _ => {
                return Err(PyValueError::new_err(
                    "device must be cpu, mps, cuda, or cuda:N",
                ))
            }
        }
        #[cfg(feature = "cuda")]
        let cuda = is_cuda
            .then(|| crate::cuda::CudaDecoder::new(py, device, options))
            .transpose()?;
        Ok(Self {
            process_id: std::process::id(),
            options,
            cpu: Mutex::new(CpuBatchDecoder::new(options)),
            jpeg: Mutex::new(JpegBatchDecoder::new({
                let mut jpeg = j2k_jpeg::JpegBatchDecodeOptions::default();
                jpeg.layout = layout;
                jpeg.workers = workers;
                jpeg
            })),
            #[cfg(feature = "cuda")]
            cuda,
        })
    }

    /// Prepare JPEG 2000 bytes or ImageRequest objects once for repeated batch decode.
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
        self.prepare_encoded(
            py,
            owned_items(inputs.into_iter().map(|input| input.encoded))?,
        )
    }

    /// Decode JPEG or JPEG 2000 bytes or per-input requests, grouping
    /// compatible tensor shapes.
    #[pyo3(signature = (inputs, *, region=None, reduction=0))]
    fn decode_batch(
        &self,
        py: Python<'_>,
        inputs: Vec<Bound<'_, PyAny>>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
    ) -> PyResult<Batch> {
        self.check_process()?;
        let inputs = encoded_inputs(inputs, request(region, reduction)?)?;
        self.decode_encoded(py, inputs)?.into_batch(py)
    }

    /// Reuse preparation while retaining source ordering and all item failures.
    fn decode_prepared(&self, py: Python<'_>, prepared: &PreparedBatch) -> PyResult<Batch> {
        self.check_process()?;
        self.decode_j2k(py, &prepared.inner)?.into_batch(py)
    }

    /// Decode one image to CHW or HWC native integer samples, with no batch axis.
    #[pyo3(signature = (data, *, region=None, reduction=0))]
    fn decode(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
    ) -> PyResult<Tensor> {
        self.check_process()?;
        let default_request = request(region, reduction)?;
        let mut decoded = if let Ok(data) = data.cast::<PyBytes>() {
            #[cfg(feature = "cuda")]
            if let Some(cuda) = &self.cuda {
                if !is_jpeg(data.as_bytes()) {
                    return cuda.decode_one(py, data.as_bytes(), default_request);
                }
            }
            let encoded = EncodedImage::new(Arc::from(data.as_bytes()), default_request);
            if is_jpeg(&encoded.bytes) {
                self.decode_encoded(
                    py,
                    owned_items(
                        [ImageRequest {
                            encoded,
                            jpeg_options: JpegOptions::default(),
                        }]
                        .into_iter(),
                    )?,
                )?
            } else {
                self.decode_owned_j2k(py, owned_items([encoded].into_iter())?)?
            }
        } else {
            let inputs = encoded_inputs(owned_items([data.clone()].into_iter())?, default_request)?;
            self.decode_encoded(py, inputs)?
        };
        if let Some((_, error)) = decoded.errors.pop() {
            return Err(PyValueError::new_err(error));
        }
        if let Some((_, error)) = decoded.group_errors.pop() {
            return Err(PyValueError::new_err(error));
        }
        let (Some(group), true) = (decoded.groups.pop(), decoded.groups.is_empty()) else {
            return Err(PyRuntimeError::new_err(
                "single-image decode did not return exactly one group",
            ));
        };
        if group.recovered.iter().any(|&recovered| recovered) {
            py.import("warnings")?.call_method1("warn", (
                "JPEG decode required recovery or accepted unverified completion; use decode_batch for per-row status",
                py.get_type::<crate::RecoveredImageWarning>(),
            ))?;
        }
        let mut tensor = group.tensor;
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

    /// Decode JPEG 2000 and JPEG inputs, with indices relative to `inputs`.
    fn decode_encoded(&self, py: Python<'_>, inputs: Vec<ImageRequest>) -> PyResult<Decoded> {
        let mut decoded = Decoded::default();
        let (mut j2k, mut j2k_positions) = (Vec::new(), Vec::new());
        let (mut jpeg, mut jpeg_positions) = (Vec::new(), Vec::new());
        for (position, input) in inputs.into_iter().enumerate() {
            if !is_jpeg(&input.encoded.bytes) {
                j2k.push(input.encoded);
                j2k_positions.push(position);
            } else if input.encoded.request == DecodeRequest::Full {
                jpeg.push(input);
                jpeg_positions.push(position);
            } else {
                decoded.errors.push((position, JPEG_FULL_ONLY.into()));
            }
        }
        if !j2k.is_empty() {
            decoded.extend_from(self.decode_owned_j2k(py, j2k)?, &j2k_positions);
        }
        if !jpeg.is_empty() {
            decoded.extend_from(self.decode_jpeg(py, &jpeg)?, &jpeg_positions);
        }
        Ok(decoded)
    }

    // Keep CPU preparation and decoding inside one interpreter handoff, as in
    // the original JPEG 2000 adapter. A second handoff delays concurrent callers.
    fn decode_owned_j2k(&self, py: Python<'_>, inputs: Vec<EncodedImage>) -> PyResult<Decoded> {
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() {
            let prepared = self.prepare_encoded(py, inputs)?;
            return self.decode_j2k(py, &prepared.inner);
        }
        let decoded = py.detach(|| {
            self.cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?
                .decode(inputs)
                .map_err(value_error)
        })?;
        cpu_decoded(decoded)
    }

    fn prepare_encoded(
        &self,
        py: Python<'_>,
        inputs: Vec<EncodedImage>,
    ) -> PyResult<PreparedBatch> {
        py.detach(|| {
            self.cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?
                .prepare(inputs)
                .map(|inner| PreparedBatch { inner })
                .map_err(value_error)
        })
    }

    fn decode_j2k(&self, py: Python<'_>, prepared: &j2k::PreparedBatch) -> PyResult<Decoded> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            return cuda.decode(py, prepared);
        }
        let decoded = py.detach(|| {
            self.cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?
                .decode_prepared(prepared)
                .map_err(value_error)
        })?;
        cpu_decoded(decoded)
    }

    fn decode_jpeg(&self, py: Python<'_>, inputs: &[ImageRequest]) -> PyResult<Decoded> {
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() {
            // An explicit GPU request never silently decodes on the CPU.
            return Ok(Decoded {
                errors: (0..inputs.len())
                    .map(|index| (index, "JPEG decode on CUDA is not supported yet".into()))
                    .collect(),
                ..Decoded::default()
            });
        }
        let decoded = py.detach(|| {
            let batch = owned_items(inputs.iter().map(|input| {
                JpegBatchInput::new(&input.encoded.bytes).with_options(
                    input
                        .jpeg_options
                        .with_strict(self.options.settings.is_strict()),
                )
            }))?;
            self.jpeg
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?
                .decode(&batch)
                .map_err(value_error)
        })?;
        let (groups, errors) = decoded.into_parts();
        Ok(Decoded {
            groups: groups
                .into_iter()
                .map(jpeg_group)
                .collect::<PyResult<_>>()?,
            errors: errors
                .into_iter()
                .map(|error| (error.index, error.source.to_string()))
                .collect(),
            group_errors: Vec::new(),
        })
    }
}

fn cpu_decoded(decoded: j2k::CpuBatchDecodeResult) -> PyResult<Decoded> {
    let (groups, errors) = decoded.into_parts();
    Ok(Decoded {
        groups: groups.into_iter().map(cpu_group).collect::<PyResult<_>>()?,
        errors: errors
            .into_iter()
            .map(|e| (e.index, e.source.to_string()))
            .collect(),
        group_errors: Vec::new(),
    })
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
) -> PyResult<Vec<ImageRequest>> {
    inputs
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            if let Ok(bytes) = item.cast::<PyBytes>() {
                return Ok(ImageRequest {
                    encoded: EncodedImage::new(Arc::from(bytes.as_bytes()), default_request),
                    jpeg_options: JpegOptions::default(),
                });
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
            Ok((*request).clone())
        })
        .collect()
}

pub(crate) fn shape(info: &BatchGroupInfo, batch: usize) -> PyResult<Vec<usize>> {
    dense_shape(info.layout, batch, info.color.channels(), info.dimensions)
}

fn dense_shape(
    layout: BatchLayout,
    batch: usize,
    channels: usize,
    (w, h): (u32, u32),
) -> PyResult<Vec<usize>> {
    match layout {
        BatchLayout::Nchw => owned_items([batch, channels, h as usize, w as usize].into_iter()),
        BatchLayout::Nhwc => owned_items([batch, h as usize, w as usize, channels].into_iter()),
        _ => Err(PyValueError::new_err("unsupported codec tensor layout")),
    }
}

fn jpeg_group(group: JpegBatchGroup) -> PyResult<MaterializedGroup> {
    let (dimensions, layout) = (group.dimensions(), group.layout());
    let (indices, warnings, recovered, samples) = group.into_parts();
    let color = BatchColor::Rgb;
    let shape = dense_shape(layout, indices.len(), color.channels(), dimensions)?;
    Ok(MaterializedGroup {
        meta: GroupMeta {
            precision: 8,
            signed: false,
            color,
            alpha: BatchAlpha::None,
            layout,
        },
        recovered,
        rects: indices.iter().map(|_| Rect::full(dimensions)).collect(),
        indices,
        warnings: warnings
            .into_iter()
            .map(|w| w.iter().map(ToString::to_string).collect())
            .collect(),
        tensor: Tensor::new(Samples::U8(samples), shape, DType::U8, (1, 0))?,
    })
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
        meta: GroupMeta::of(&info),
        recovered: owned_items((0..indices.len()).map(|_| false))?,
        indices,
        rects,
        warnings: warnings
            .into_iter()
            .map(|w| w.into_iter().map(|v| v.to_string()).collect())
            .collect(),
        tensor: Tensor::new(samples, dimensions, dtype, (1, 0))?,
    })
}
