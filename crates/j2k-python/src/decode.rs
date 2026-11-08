// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use j2k::{
    BatchAlpha, BatchColor, BatchDecodeOptions, BatchGroupInfo, BatchLayout, CpuBatchDecoder,
    CpuBatchSamples, DecodeSettings, EncodedImage, Rect,
};
use j2k_jpeg::{
    ColorTransform, DecodeOptions as JpegOptions, JpegBatchDecoder, JpegBatchGroup, JpegBatchInput,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};

use crate::{
    is_jpeg,
    operations::{self, DecodeControl, PreparedState, RegionRequest},
    owned_items,
    tensor::{DType, Samples, Tensor},
    value_error,
};

const JPEG_FULL_ONLY: &str = "JPEG inputs support full-image decode only (no region or reduction)";

/// One encoded image with its own full/region/reduced request.
#[pyclass(name = "ImageRequest", module = "j2k", frozen)]
#[derive(Clone)]
pub(crate) struct ImageRequest {
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) request: RegionRequest,
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
            bytes: Arc::from(data.as_bytes()),
            request: RegionRequest::new(region, reduction)?,
            jpeg_options: JpegOptions::default().with_color_transform(transform),
        })
    }
}

/// Reusable codec preparation, including per-input errors and source order.
#[pyclass(name = "PreparedBatch", module = "j2k", frozen)]
pub(crate) struct PreparedBatch {
    pub(crate) inner: PreparedState,
}

#[pymethods]
impl PreparedBatch {
    #[getter]
    fn group_count(&self) -> usize {
        self.inner.group_count()
    }
    #[getter]
    fn errors(&self) -> Vec<(usize, String)> {
        self.inner.errors()
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
                alpha: alpha_name(self.meta.alpha)?.into(),
                layout: match self.meta.layout {
                    BatchLayout::Nchw => "nchw",
                    _ => "nhwc",
                }
                .into(),
            },
        )
    }
}

pub(crate) fn alpha_name(alpha: BatchAlpha) -> PyResult<&'static str> {
    match alpha {
        BatchAlpha::None => Ok("None"),
        BatchAlpha::Straight => Ok("Straight"),
        BatchAlpha::Premultiplied => Ok("Premultiplied"),
        _ => Err(PyValueError::new_err("unsupported output alpha")),
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
    pub(crate) fn extend_from(&mut self, other: Self, positions: &[usize]) {
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
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    metal: Option<crate::metal::MetalDecoder>,
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
        let device = if device == "mps:0" { "mps" } else { device };
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
            "mps"
                if cfg!(all(
                    feature = "metal",
                    target_os = "macos",
                    target_arch = "aarch64"
                )) => {}
            "mps" => {
                return Err(PyValueError::new_err(
                    "MPS requires the optional metal build on Apple Silicon macOS",
                ))
            }
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
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        let metal = (device == "mps")
            .then(|| crate::metal::MetalDecoder::new(options))
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
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            metal,
        })
    }

    /// Prepare JPEG 2000 bytes or ImageRequest objects once for repeated batch decode.
    ///
    /// A CPU decoder prepares under `control` when it is given: inputs are
    /// prepared one at a time, the working limit bounds each input's parsing
    /// and planning beside the plans already kept for earlier inputs, the
    /// output limit rejects inputs whose decoded output would exceed it, and
    /// cancellation stops at the next checkpoint. Failing inputs are reported
    /// in the prepared batch's errors.
    #[pyo3(signature = (inputs, *, region=None, reduction=0, control=None))]
    fn prepare(
        &self,
        py: Python<'_>,
        inputs: Vec<Bound<'_, PyAny>>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
        control: Option<&DecodeControl>,
    ) -> PyResult<PreparedBatch> {
        self.check_process()?;
        if control.is_some() && self.has_device_backend() {
            return Err(value_error("preparation controls require a CPU decoder"));
        }
        let inputs = encoded_inputs(inputs, RegionRequest::new(region, reduction)?)?;
        let control = control.map(|control| control.inner.clone());
        self.prepare_requests(py, inputs, control.as_ref())
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
        let inputs = encoded_inputs(inputs, RegionRequest::new(region, reduction)?)?;
        self.decode_encoded(py, inputs)?.into_batch(py)
    }

    /// Reuse preparation while retaining source ordering and all item failures.
    #[pyo3(signature = (prepared, *, region=None, reduction=None, control=None))]
    fn decode_prepared(
        &self,
        py: Python<'_>,
        prepared: &PreparedBatch,
        region: Option<(u32, u32, u32, u32)>,
        reduction: Option<u8>,
        control: Option<&DecodeControl>,
    ) -> PyResult<Batch> {
        self.check_process()?;
        let overrides = region.is_some() || reduction.is_some() || control.is_some();
        if !overrides {
            if let PreparedState::Legacy(batch) = &prepared.inner {
                return self.decode_j2k(py, batch)?.into_batch(py);
            }
        }
        if self.options.layout != prepared.inner.layout() {
            return Err(value_error(
                "exact prepared operations require a Decoder with the preparation's output layout",
            ));
        }
        let region = region.map(|(x, y, w, h)| Rect { x, y, w, h });
        let control = control.map(|control| control.inner.clone());
        if let Some(decoded) =
            self.decode_on_device(py, &prepared.inner, region, reduction, control.as_ref())
        {
            return decoded?.into_batch(py);
        }
        let result = py.detach(|| {
            let mut cpu = self
                .cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?;
            operations::decode_operations(
                &mut cpu,
                &prepared.inner,
                region,
                reduction,
                control.as_ref(),
            )
        })?;
        result.into_batch(py)
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
        let default_request = RegionRequest::new(region, reduction)?;
        let mut decoded = if let Ok(data) = data.cast::<PyBytes>() {
            #[cfg(feature = "cuda")]
            if let Some(cuda) = &self.cuda {
                if !is_jpeg(data.as_bytes()) {
                    if let Some(request) = default_request.legacy() {
                        return cuda.decode_one(py, data.as_bytes(), request);
                    }
                }
            }
            self.decode_encoded(
                py,
                owned_items(
                    [ImageRequest {
                        bytes: Arc::from(data.as_bytes()),
                        request: default_request,
                        jpeg_options: JpegOptions::default(),
                    }]
                    .into_iter(),
                )?,
            )?
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

    /// Plan exact native or explicit sRGB region output. This input-inspection
    /// step is outside operation controls and does not allocate pixel output.
    #[pyo3(signature = (data, *, region=None, reduction=0, interpretation="native"))]
    fn plan_region<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyBytes>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
        interpretation: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.check_process()?;
        if self.has_device_backend() {
            return Err(value_error("exact region planning requires a CPU decoder"));
        }
        operations::plan_region(
            py,
            data.as_bytes(),
            self.options,
            RegionRequest::new(region, reduction)?,
            interpretation,
        )
    }

    /// Decode explicit straight-alpha sRGB through bounded native reconstruction.
    /// The input is parsed under `control` as well as decoded under it; opaque
    /// ICC conversion rejects a working limit.
    #[pyo3(signature = (data, *, region=None, reduction=0, control=None))]
    fn decode_srgb8(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        region: Option<(u32, u32, u32, u32)>,
        reduction: u8,
        control: Option<&DecodeControl>,
    ) -> PyResult<Tensor> {
        self.check_process()?;
        if self.has_device_backend() {
            return Err(value_error("explicit sRGB output requires a CPU decoder"));
        }
        if is_jpeg(data.as_bytes()) {
            return Err(value_error(
                "explicit sRGB output supports JPEG 2000 inputs only",
            ));
        }
        let request = RegionRequest::new(region, reduction)?;
        let control = operations::checked_normalized_control(control)?;
        let bytes = data.as_bytes();
        let (tensor, warnings) = py.detach(|| {
            let mut codec =
                j2k::J2kDecoder::new_with_control(bytes, self.options.settings, &control)
                    .map_err(value_error)?;
            if self
                .options
                .workers
                .is_some_and(|workers| workers.get() == 1)
            {
                codec.set_cpu_decode_parallelism(j2k::CpuDecodeParallelism::Serial);
            }
            let region = request.source_region(codec.info().dimensions);
            let image = codec
                .decode_region_srgb8_at_reduction_with_control(region, request.levels, &control)
                .map_err(value_error)?;
            // The image owns its output. Release native planes/transforms before
            // a second packed buffer is allocated for the requested layout.
            drop(codec);
            let warnings = image
                .warnings()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let dimensions = image.dimensions();
            let layout = image.layout();
            let data = image.into_data();
            let channels = match layout {
                j2k::J2kSrgb8Layout::Gray => 1,
                j2k::J2kSrgb8Layout::Rgb => 3,
                j2k::J2kSrgb8Layout::Rgba => 4,
                _ => return Err(value_error("unsupported sRGB output layout")),
            };
            let data = if self.options.layout == BatchLayout::Nchw && channels > 1 {
                // The packed output and its planar copy are both codec-owned
                // while the layout converts, within the caller's working limit.
                let budget = j2k_native::DecodeOperationControl::new(
                    control
                        .max_working_memory_bytes
                        .map(core::num::NonZeroUsize::get),
                    None,
                )
                .beside_retained_bytes(data.capacity())
                .map_err(value_error)?;
                budget
                    .clone()
                    .beside_retained_bytes(data.len())
                    .map_err(value_error)?;
                let mut planar = crate::owned_capacity(data.len())?;
                budget
                    .beside_retained_bytes(planar.capacity())
                    .map_err(value_error)?;
                for channel in 0..channels {
                    planar.extend(data.iter().skip(channel).step_by(channels).copied());
                }
                planar
            } else {
                data
            };
            let mut tensor = Tensor::new(
                Samples::U8(data),
                dense_shape(self.options.layout, 1, channels, dimensions)?,
                DType::U8,
                (1, 0),
            )?;
            tensor.remove_batch_axis()?;
            Ok::<_, PyErr>((tensor, warnings))
        })?;
        for warning in warnings {
            py.import("warnings")?.call_method1(
                "warn",
                (warning, py.get_type::<crate::RecoveredImageWarning>()),
            )?;
        }
        Ok(tensor)
    }

    /// Bound only the session's derived exact-operation cache; zero disables it.
    fn set_operation_cache_limit(&self, py: Python<'_>, bytes: usize) -> PyResult<()> {
        self.check_process()?;
        if self.has_device_backend() {
            return Err(value_error("operation cache limits require a CPU decoder"));
        }
        py.detach(|| {
            self.cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?
                .set_operation_cache_limit_bytes(bytes);
            Ok(())
        })
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
            if !is_jpeg(&input.bytes) {
                j2k.push(input);
                j2k_positions.push(position);
            } else if input.request == RegionRequest::default() {
                jpeg.push(input);
                jpeg_positions.push(position);
            } else {
                decoded.errors.push((position, JPEG_FULL_ONLY.into()));
            }
        }
        if !j2k.is_empty() {
            decoded.extend_from(self.decode_owned_requests(py, j2k)?, &j2k_positions);
        }
        if !jpeg.is_empty() {
            decoded.extend_from(self.decode_jpeg(py, &jpeg)?, &jpeg_positions);
        }
        Ok(decoded)
    }

    fn prepare_requests(
        &self,
        py: Python<'_>,
        inputs: Vec<ImageRequest>,
        control: Option<&j2k::J2kDecodeControl>,
    ) -> PyResult<PreparedBatch> {
        if inputs.iter().all(|input| input.request.legacy().is_some()) {
            return self.prepare_encoded(py, legacy_inputs(inputs)?, control);
        }
        if self.has_device_backend() {
            return py.detach(|| {
                operations::prepare_exact_for_device(inputs, self.options).map(|batch| {
                    PreparedBatch {
                        inner: PreparedState::Exact(batch),
                    }
                })
            });
        }
        py.detach(|| {
            let cpu = self
                .cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?;
            operations::prepare_exact(&cpu, inputs, self.options, control).map(|batch| {
                PreparedBatch {
                    inner: PreparedState::Exact(batch),
                }
            })
        })
    }

    fn decode_owned_requests(
        &self,
        py: Python<'_>,
        inputs: Vec<ImageRequest>,
    ) -> PyResult<Decoded> {
        if inputs.iter().all(|input| input.request.legacy().is_some()) {
            return self.decode_owned_j2k(py, legacy_inputs(inputs)?);
        }
        if self.has_device_backend() {
            return self.decode_mixed_on_device(py, inputs);
        }
        py.detach(|| {
            let mut cpu = self
                .cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?;
            let prepared =
                PreparedState::Exact(operations::prepare_exact(&cpu, inputs, self.options, None)?);
            let decoded = operations::decode_operations(&mut cpu, &prepared, None, None, None);
            drop(cpu);
            decoded
        })
    }

    /// Decode requests on a device: those the legacy prepared route
    /// describes decode on it, as they do in batches without deeper requests,
    /// and the rest decode as exact device operations.
    fn decode_mixed_on_device(
        &self,
        py: Python<'_>,
        inputs: Vec<ImageRequest>,
    ) -> PyResult<Decoded> {
        let mut legacy = crate::owned_capacity(inputs.len())?;
        let mut legacy_positions = crate::owned_capacity(inputs.len())?;
        let mut exact = crate::owned_capacity(inputs.len())?;
        let mut exact_positions = crate::owned_capacity(inputs.len())?;
        for (index, input) in inputs.into_iter().enumerate() {
            if let Some(request) = input.request.legacy() {
                legacy.push(EncodedImage::new(input.bytes, request));
                legacy_positions.push(index);
            } else {
                exact.push(input);
                exact_positions.push(index);
            }
        }
        let mut decoded = Decoded::default();
        if !legacy.is_empty() {
            decoded.extend_from(self.decode_owned_j2k(py, legacy)?, &legacy_positions);
        }
        let prepared = py.detach(|| {
            operations::prepare_exact_for_device(exact, self.options).map(PreparedState::Exact)
        })?;
        let exact_decoded = self
            .decode_on_device(py, &prepared, None, None, None)
            .unwrap_or_else(|| Err(PyRuntimeError::new_err("device decoder is unavailable")))?;
        decoded.extend_from(exact_decoded, &exact_positions);
        Ok(decoded)
    }

    // Keep CPU preparation and decoding inside one interpreter handoff, as in
    // the original JPEG 2000 adapter. A second handoff delays concurrent callers.
    fn decode_owned_j2k(&self, py: Python<'_>, inputs: Vec<EncodedImage>) -> PyResult<Decoded> {
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() {
            let prepared = self.prepare_encoded(py, inputs, None)?;
            let PreparedState::Legacy(batch) = &prepared.inner else {
                return Err(value_error(
                    "device preparation did not retain a legacy plan",
                ));
            };
            return self.decode_j2k(py, batch);
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if self.metal.is_some() {
            let prepared = self.prepare_encoded(py, inputs, None)?;
            let PreparedState::Legacy(batch) = &prepared.inner else {
                return Err(value_error(
                    "device preparation did not retain a legacy plan",
                ));
            };
            return self.decode_j2k(py, batch);
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

    /// Prepare `inputs` for this decoder's backend. A device backend prepares
    /// as its own decoder does, with the shared preparation: the CPU
    /// decoder's admission rejects images whose CPU execution would pass the
    /// CPU decode bound, which a device decoder decodes on its own terms.
    fn prepare_encoded(
        &self,
        py: Python<'_>,
        inputs: Vec<EncodedImage>,
        control: Option<&j2k::J2kDecodeControl>,
    ) -> PyResult<PreparedBatch> {
        if self.has_device_backend() {
            return py.detach(|| {
                j2k::prepare_batch(inputs, self.options)
                    .map(|inner| PreparedBatch {
                        inner: PreparedState::Legacy(inner),
                    })
                    .map_err(value_error)
            });
        }
        py.detach(|| {
            let cpu = self
                .cpu
                .lock()
                .map_err(|_| PyRuntimeError::new_err("decoder is poisoned"))?;
            match control {
                None => cpu.prepare(inputs),
                Some(control) => cpu.prepare_with_control(inputs, control),
            }
            .map(|inner| PreparedBatch {
                inner: PreparedState::Legacy(inner),
            })
            .map_err(value_error)
        })
    }

    /// Whether batches decode on a CUDA or Metal device instead of the CPU.
    #[cfg_attr(
        not(any(
            feature = "cuda",
            all(feature = "metal", target_os = "macos", target_arch = "aarch64")
        )),
        expect(
            clippy::unused_self,
            reason = "builds without a device backend have no device to check"
        )
    )]
    fn has_device_backend(&self) -> bool {
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() {
            return true;
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if self.metal.is_some() {
            return true;
        }
        false
    }

    /// Decode exact operations on this decoder's CUDA or Metal device, or
    /// return `None` for a CPU decoder.
    #[cfg_attr(
        not(any(
            feature = "cuda",
            all(feature = "metal", target_os = "macos", target_arch = "aarch64")
        )),
        expect(
            clippy::unused_self,
            reason = "builds without a device backend have no device to decode on"
        )
    )]
    fn decode_on_device(
        &self,
        py: Python<'_>,
        prepared: &PreparedState,
        region: Option<Rect>,
        reduction: Option<u8>,
        control: Option<&j2k::J2kDecodeControl>,
    ) -> Option<PyResult<Decoded>> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            return Some(operations::decode_device_operations(
                prepared,
                region,
                reduction,
                control,
                |requests| cuda.decode_operations(py, requests, self.options),
            ));
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(metal) = &self.metal {
            return Some(py.detach(|| {
                operations::decode_device_operations(
                    prepared,
                    region,
                    reduction,
                    control,
                    |requests| metal.decode_operations(requests, self.options),
                )
            }));
        }
        let _ = (py, prepared, region, reduction, control);
        None
    }

    fn decode_j2k(&self, py: Python<'_>, prepared: &j2k::PreparedBatch) -> PyResult<Decoded> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            return cuda.decode(py, prepared);
        }
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if let Some(metal) = &self.metal {
            return py.detach(|| metal.decode(prepared));
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
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if self.metal.is_some() {
            return Ok(Decoded {
                errors: (0..inputs.len())
                    .map(|index| (index, "JPEG decode on MPS is not supported yet".into()))
                    .collect(),
                ..Decoded::default()
            });
        }
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
                JpegBatchInput::new(&input.bytes).with_options(
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

pub(crate) fn cpu_decoded(decoded: j2k::CpuBatchDecodeResult) -> PyResult<Decoded> {
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

fn legacy_inputs(inputs: Vec<ImageRequest>) -> PyResult<Vec<EncodedImage>> {
    let mut output = crate::owned_capacity(inputs.len())?;
    for input in inputs {
        let request = input
            .request
            .legacy()
            .ok_or_else(|| value_error("deep reduction requires exact CPU preparation"))?;
        output.push(EncodedImage::new(input.bytes, request));
    }
    Ok(output)
}

fn encoded_inputs(
    inputs: Vec<Bound<'_, PyAny>>,
    default_request: RegionRequest,
) -> PyResult<Vec<ImageRequest>> {
    inputs
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            if let Ok(bytes) = item.cast::<PyBytes>() {
                return Ok(ImageRequest {
                    bytes: Arc::from(bytes.as_bytes()),
                    request: default_request,
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
            if default_request != RegionRequest::default() {
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
