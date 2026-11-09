// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact request facts and thin bindings to existing native operation controls.

use crate::decode::{alpha_name, cpu_decoded, Decoded, ImageRequest};
use crate::{owned_capacity, owned_items, value_error};
use j2k::{
    BatchDecodeOptions, BatchGroupInfo, BatchLayout, CpuBatchDecoder, DecodeRequest, Downscale,
    EncodedImage, J2kCancellationToken, J2kDecodeControl, J2kDecoder, PreparedImage,
    PreparedImageOperation, Rect,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use std::num::NonZeroUsize;

#[pyclass(name = "CancellationToken", module = "j2k", frozen)]
pub(crate) struct CancellationToken {
    inner: J2kCancellationToken,
}

#[pymethods]
impl CancellationToken {
    #[new]
    fn new() -> Self {
        Self {
            inner: J2kCancellationToken::new(),
        }
    }
    fn cancel(&self) {
        self.inner.cancel();
    }
    #[getter]
    fn cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

#[pyclass(name = "DecodeControl", module = "j2k", frozen)]
pub(crate) struct DecodeControl {
    pub(crate) inner: J2kDecodeControl,
}

#[pymethods]
impl DecodeControl {
    #[new]
    #[pyo3(signature = (*, max_working_memory_bytes=None, max_output_bytes=None, cancellation=None))]
    fn new(
        max_working_memory_bytes: Option<usize>,
        max_output_bytes: Option<usize>,
        cancellation: Option<&CancellationToken>,
    ) -> PyResult<Self> {
        let mut inner = J2kDecodeControl::default();
        inner.max_working_memory_bytes = max_working_memory_bytes
            .map(|bytes| {
                NonZeroUsize::new(bytes)
                    .ok_or_else(|| value_error("max_working_memory_bytes must be positive"))
            })
            .transpose()?;
        inner.max_output_bytes = max_output_bytes;
        inner.cancellation = cancellation.map(|token| token.inner.clone());
        Ok(Self { inner })
    }
    #[getter]
    fn max_working_memory_bytes(&self) -> Option<usize> {
        self.inner.max_working_memory_bytes.map(NonZeroUsize::get)
    }
    #[getter]
    fn max_output_bytes(&self) -> Option<usize> {
        self.inner.max_output_bytes
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RegionRequest {
    pub(crate) region: Option<Rect>,
    pub(crate) levels: u8,
}

impl RegionRequest {
    pub(crate) fn new(region: Option<(u32, u32, u32, u32)>, levels: u8) -> PyResult<Self> {
        if levels > 31 {
            return Err(value_error("reduction must be between 0 and 31"));
        }
        Ok(Self {
            region: region.map(|(x, y, w, h)| Rect { x, y, w, h }),
            levels,
        })
    }

    pub(crate) fn legacy(self) -> Option<DecodeRequest> {
        let scale = match self.levels {
            0 => Downscale::None,
            1 => Downscale::Half,
            2 => Downscale::Quarter,
            3 => Downscale::Eighth,
            _ => return None,
        };
        Some(match (self.region, self.levels) {
            (None, 0) => DecodeRequest::Full,
            (Some(roi), 0) => DecodeRequest::Region { roi },
            (None, _) => DecodeRequest::Reduced { scale },
            (Some(roi), _) => DecodeRequest::RegionReduced { roi, scale },
        })
    }

    pub(crate) fn source_region(self, dimensions: (u32, u32)) -> Rect {
        self.region.unwrap_or_else(|| Rect::full(dimensions))
    }
}

pub(crate) enum PreparedState {
    Legacy(j2k::PreparedBatch),
    Exact(ExactPrepared),
}

pub(crate) struct ExactPrepared {
    operations: Vec<ExactPreparedOperation>,
    errors: Vec<(usize, String)>,
    group_count: usize,
    layout: BatchLayout,
}

struct ExactPreparedOperation {
    image: PreparedImage,
    roi: Rect,
    levels: u8,
}

impl PreparedState {
    pub(crate) fn group_count(&self) -> usize {
        match self {
            Self::Legacy(batch) => batch.groups().len(),
            Self::Exact(batch) => batch.group_count,
        }
    }
    pub(crate) fn errors(&self) -> Vec<(usize, String)> {
        match self {
            Self::Legacy(batch) => batch
                .errors()
                .iter()
                .map(|error| (error.index, error.source.to_string()))
                .collect(),
            Self::Exact(batch) => batch.errors.clone(),
        }
    }
    pub(crate) fn layout(&self) -> BatchLayout {
        match self {
            Self::Legacy(batch) => batch.options().layout,
            Self::Exact(batch) => batch.layout,
        }
    }
}

/// Prepare exact requests on the CPU, under `control`'s working limit,
/// output limit and cancellation when it is given.
pub(crate) fn prepare_exact(
    cpu: &CpuBatchDecoder,
    inputs: Vec<ImageRequest>,
    options: BatchDecodeOptions,
    control: Option<&J2kDecodeControl>,
) -> PyResult<ExactPrepared> {
    prepare_exact_with(inputs, options, control, |sources| {
        match control {
            None => cpu.prepare(sources),
            Some(control) => cpu.prepare_with_control(sources, control),
        }
        .map_err(value_error)
    })
}

/// [`prepare_exact`] with a device backend's source preparation, which does
/// not apply the CPU session's execution admission.
pub(crate) fn prepare_exact_for_device(
    inputs: Vec<ImageRequest>,
    options: BatchDecodeOptions,
) -> PyResult<ExactPrepared> {
    prepare_exact_with(inputs, options, None, |sources| {
        j2k::prepare_batch(sources, options).map_err(value_error)
    })
}

fn prepare_exact_with(
    inputs: Vec<ImageRequest>,
    options: BatchDecodeOptions,
    control: Option<&J2kDecodeControl>,
    prepare_sources: impl FnOnce(Vec<EncodedImage>) -> PyResult<j2k::PreparedBatch>,
) -> PyResult<ExactPrepared> {
    let count = inputs.len();
    let requests = owned_items(inputs.iter().map(|input| input.request))?;
    let inputs = owned_items(
        inputs
            .into_iter()
            .map(|input| EncodedImage::operation_source(input.bytes)),
    )?;
    let sources = prepare_sources(inputs)?;
    let mut operations = owned_capacity(count)?;
    let mut errors = owned_capacity(count)?;
    errors.extend(
        sources
            .errors()
            .iter()
            .map(|error| (error.index, error.source.to_string())),
    );
    let mut groups: Vec<BatchGroupInfo> = owned_capacity(count)?;
    for group in sources.groups() {
        for image in group.images() {
            let request = requests[image.source_index()];
            let grid = image.reference_grid();
            let roi = request.source_region(grid.image_size);
            let plan = match control {
                None => J2kDecoder::new_with_settings(image.bytes(), image.decode_settings())
                    .and_then(|mut decoder| decoder.native_region_decode_plan(roi, request.levels)),
                Some(control) => {
                    J2kDecoder::new_with_control(image.bytes(), image.decode_settings(), control)
                        .and_then(|mut decoder| {
                            decoder.native_region_decode_plan_with_control(
                                roi,
                                request.levels,
                                control,
                            )
                        })
                }
            };
            match plan {
                Ok(plan) => {
                    if !groups.contains(&plan.samples) {
                        groups.push(plan.samples);
                    }
                    operations.push(ExactPreparedOperation {
                        image: image.clone(),
                        roi,
                        levels: request.levels,
                    });
                }
                Err(error) => errors.push((image.source_index(), error.to_string())),
            }
        }
    }
    errors.sort_unstable_by_key(|(index, _)| *index);
    Ok(ExactPrepared {
        operations,
        errors,
        group_count: groups.len(),
        layout: options.layout,
    })
}

pub(crate) fn decode_operations(
    cpu: &mut CpuBatchDecoder,
    prepared: &PreparedState,
    region: Option<Rect>,
    reduction: Option<u8>,
    control: Option<&J2kDecodeControl>,
) -> PyResult<Decoded> {
    let (requests, positions) = exact_requests(prepared, region, reduction, control)?;
    let result = cpu.decode_operations(&requests).map_err(value_error)?;
    let mut decoded = Decoded {
        errors: prepared.errors(),
        ..Decoded::default()
    };
    decoded.extend_from(cpu_decoded(result)?, &positions);
    Ok(decoded)
}

/// [`decode_operations`] on a device backend. `decode` decodes the exact
/// operations, with result indices referring to positions in its slice.
#[cfg(any(
    feature = "cuda",
    all(feature = "metal", target_os = "macos", target_arch = "aarch64")
))]
pub(crate) fn decode_device_operations(
    prepared: &PreparedState,
    region: Option<Rect>,
    reduction: Option<u8>,
    control: Option<&J2kDecodeControl>,
    decode: impl FnOnce(&[PreparedImageOperation<'_>]) -> PyResult<Decoded>,
) -> PyResult<Decoded> {
    let (requests, positions) = exact_requests(prepared, region, reduction, control)?;
    let mut decoded = Decoded {
        errors: prepared.errors(),
        ..Decoded::default()
    };
    decoded.extend_from(decode(&requests)?, &positions);
    Ok(decoded)
}

/// One exact operation per prepared image, applying any overrides, and each
/// operation's original source index.
fn exact_requests<'a>(
    prepared: &'a PreparedState,
    region: Option<Rect>,
    reduction: Option<u8>,
    control: Option<&'a J2kDecodeControl>,
) -> PyResult<(Vec<PreparedImageOperation<'a>>, Vec<usize>)> {
    if reduction.is_some_and(|levels| levels > 31) {
        return Err(value_error("reduction must be between 0 and 31"));
    }
    let count = match prepared {
        PreparedState::Legacy(batch) => batch
            .groups()
            .iter()
            .map(|group| group.images().len())
            .sum(),
        PreparedState::Exact(batch) => batch.operations.len(),
    };
    let mut requests = owned_capacity(count)?;
    let mut positions = owned_capacity(count)?;
    let mut push = |image, roi, levels| {
        let request =
            PreparedImageOperation::new(image, region.unwrap_or(roi), reduction.unwrap_or(levels));
        requests.push(control.map_or(request, |control| request.with_control(control)));
        positions.push(image.source_index());
    };
    match prepared {
        PreparedState::Legacy(batch) => {
            for group in batch.groups() {
                for image in group.images() {
                    let plan = image.plan();
                    let levels = u8::try_from(plan.scale().denominator().trailing_zeros())
                        .map_err(value_error)?;
                    push(image, plan.source_rect(), levels);
                }
            }
        }
        PreparedState::Exact(batch) => {
            for operation in &batch.operations {
                push(&operation.image, operation.roi, operation.levels);
            }
        }
    }
    Ok((requests, positions))
}

pub(crate) fn plan_region<'py>(
    py: Python<'py>,
    data: &[u8],
    options: BatchDecodeOptions,
    request: RegionRequest,
    interpretation: &str,
) -> PyResult<Bound<'py, PyDict>> {
    if crate::is_jpeg(data) {
        return Err(value_error(
            "region planning supports JPEG 2000 inputs only",
        ));
    }
    if !matches!(interpretation, "native" | "srgb8") {
        return Err(value_error("interpretation must be native or srgb8"));
    }
    let (plan, max_reduction, precision, signed, alpha, sample_type) = py.detach(|| {
        let view =
            j2k::J2kView::parse_with_settings(data, options.settings).map_err(value_error)?;
        let roi = request.source_region(view.info().dimensions);
        if interpretation == "srgb8" {
            let mut decoder = J2kDecoder::from_view(view).map_err(value_error)?;
            let plan = decoder
                .srgb8_region_decode_plan(roi, request.levels)
                .map_err(value_error)?;
            let alpha = alpha_name(plan.alpha)?;
            return Ok((
                plan,
                decoder.max_reduction_levels().map_err(value_error)?,
                8,
                false,
                alpha,
                "uint8",
            ));
        }
        let mut decoder = J2kDecoder::from_view(view).map_err(value_error)?;
        let j2k::J2kNativeRegionDecodePlan {
            region: plan,
            samples: info,
            ..
        } = decoder
            .native_region_decode_plan(roi, request.levels)
            .map_err(value_error)?;
        let sample_type = match info.sample_type {
            j2k::NativeSampleType::U8 => "uint8",
            j2k::NativeSampleType::U16 => "uint16",
            j2k::NativeSampleType::I16 => "int16",
            _ => return Err(value_error("unsupported native sample type")),
        };
        let alpha = alpha_name(plan.alpha)?;
        Ok((
            plan,
            decoder.max_reduction_levels().map_err(value_error)?,
            info.precision,
            info.signed,
            alpha,
            sample_type,
        ))
    })?;
    let result = PyDict::new(py);
    result.set_item(
        "source_region",
        (
            plan.source_region.x,
            plan.source_region.y,
            plan.source_region.w,
            plan.source_region.h,
        ),
    )?;
    result.set_item(
        "output_region",
        (
            plan.output_region.x,
            plan.output_region.y,
            plan.output_region.w,
            plan.output_region.h,
        ),
    )?;
    result.set_item("width", plan.output_region.w)?;
    result.set_item("height", plan.output_region.h)?;
    result.set_item("image_origin", plan.reference_grid.image_origin)?;
    result.set_item("source_dimensions", plan.reference_grid.image_size)?;
    result.set_item("reduction", plan.reduction_levels)?;
    result.set_item("max_reduction", max_reduction)?;
    let sample_bytes = if sample_type == "uint8" { 1 } else { 2 };
    let channels = plan.format.bytes_per_pixel() / sample_bytes;
    let width = plan.output_region.w as usize;
    let height = plan.output_region.h as usize;
    let (shape, strides, row_bytes, layout) = if options.layout == BatchLayout::Nchw {
        (
            (channels, height, width),
            (
                height * width * sample_bytes,
                width * sample_bytes,
                sample_bytes,
            ),
            width * sample_bytes,
            "nchw",
        )
    } else {
        (
            (height, width, channels),
            (plan.row_bytes, channels * sample_bytes, sample_bytes),
            plan.row_bytes,
            "nhwc",
        )
    };
    result.set_item("shape", shape)?;
    result.set_item("strides", strides)?;
    result.set_item("layout", layout)?;
    result.set_item("row_bytes", row_bytes)?;
    result.set_item("buffer_bytes", plan.buffer_bytes)?;
    result.set_item("tile_origin", plan.reference_grid.tile_origin)?;
    result.set_item("tile_size", plan.reference_grid.tile_size)?;
    result.set_item("precision", precision)?;
    result.set_item("signed", signed)?;
    result.set_item("alpha", alpha)?;
    result.set_item("sample_type", sample_type)?;
    result.set_item("interpretation", interpretation)?;
    result.set_item(
        "route",
        if interpretation == "native" {
            "native_cpu"
        } else {
            "region_planes"
        },
    )?;
    Ok(result)
}

/// The control of one raw sRGB call, refusing a token already cancelled
/// before any input is parsed.
pub(crate) fn checked_normalized_control(
    control: Option<&DecodeControl>,
) -> PyResult<J2kDecodeControl> {
    let control = control.map_or_else(J2kDecodeControl::default, |control| control.inner.clone());
    if control
        .cancellation
        .as_ref()
        .is_some_and(J2kCancellationToken::is_cancelled)
    {
        return Err(PyValueError::new_err("decode operation was cancelled"));
    }
    Ok(control)
}
