// SPDX-License-Identifier: MIT OR Apache-2.0

//! Strict J2K Metal decode followed by a compact staged Burn Metal upload.

// The non-macOS Metal session is intentionally a zero-sized compatibility
// stub. Keep these private helper signatures identical across targets.
#![cfg_attr(
    not(target_os = "macos"),
    allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "the non-macOS Metal compatibility session is intentionally zero-sized"
    )
)]

use burn_core::tensor::{DType, FloatDType, Int, Tensor};
use burn_wgpu::{Wgpu, WgpuDevice};
use j2k::{
    BackendRequest, DeviceDecodePlan, DeviceDecodeRequest, J2kDecodeWarning, J2kDecoder, Rect,
};
#[cfg(target_os = "macos")]
use j2k_metal::download_surfaces_packed;
use j2k_metal::{
    J2kDecoder as MetalDecoder, MetalBackendSession, MetalDecodeRequest, Surface, SurfaceResidency,
};

use crate::cpu::{
    ensure_dtype, integer_tensor_3_from_bytes, integer_tensor_4_from_bytes, normalize_3,
    normalize_4, pixel_format, planned_shape, selected_channels, validate_normalization_channels,
    validate_normalization_values, SampleWidth,
};
use crate::{
    TensorBatchDecode, TensorDecode, TensorDecodeError, TensorDecodeOptions, TensorInput,
    TensorRoute,
};

type MetalTensor<const D: usize> = Tensor<Wgpu, D>;
type MetalIntTensor<const D: usize> = Tensor<Wgpu, D, Int>;

struct MetalImage {
    surface: Surface,
    shape: [usize; 3],
    sample_width: SampleWidth,
    decoded: Rect,
    warnings: Vec<J2kDecodeWarning>,
}

/// Decode one image into a rank-3 U8 Burn Metal tensor.
pub fn decode_u8(
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    device: &WgpuDevice,
) -> Result<TensorDecode<MetalIntTensor<3>>, TensorDecodeError> {
    ensure_dtype::<Wgpu>(device, DType::U8)?;
    let session = metal_session()?;
    let image = decode_surface(&session, input, options, SampleWidth::U8)?;
    let bytes = packed_readback(&session, &[(&image.surface, image_byte_len(&image)?)])?;
    let tensor =
        integer_tensor_3_from_bytes::<Wgpu>(bytes, image.shape, options.layout, device, DType::U8);
    Ok(single_result(tensor, image))
}

/// Decode one image into a rank-3 U16 Burn Metal tensor.
pub fn decode_u16(
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    device: &WgpuDevice,
) -> Result<TensorDecode<MetalIntTensor<3>>, TensorDecodeError> {
    ensure_dtype::<Wgpu>(device, DType::U16)?;
    let session = metal_session()?;
    let image = decode_surface(&session, input, options, SampleWidth::U16)?;
    let bytes = packed_readback(&session, &[(&image.surface, image_byte_len(&image)?)])?;
    let tensor =
        integer_tensor_3_from_bytes::<Wgpu>(bytes, image.shape, options.layout, device, DType::U16);
    Ok(single_result(tensor, image))
}

/// Decode one image into a rank-3 F32 Burn Metal tensor.
pub fn decode_float(
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    device: &WgpuDevice,
) -> Result<TensorDecode<MetalTensor<3>>, TensorDecodeError> {
    validate_normalization_values(&options.normalization)?;
    ensure_dtype::<Wgpu>(device, DType::F32)?;
    let info = J2kDecoder::inspect(input.encoded)?;
    let width = sample_width(info.bit_depth);
    ensure_dtype::<Wgpu>(device, width.dtype())?;
    validate_normalization_channels(
        &options.normalization,
        selected_channels(options.channels, info.components),
    )?;
    let session = metal_session()?;
    let image = decode_surface(&session, input, options, width)?;
    let bytes = packed_readback(&session, &[(&image.surface, image_byte_len(&image)?)])?;
    let tensor = integer_tensor_3_from_bytes::<Wgpu>(
        bytes,
        image.shape,
        options.layout,
        device,
        width.dtype(),
    )
    .cast(FloatDType::F32);
    let tensor = normalize_3(
        tensor,
        &options.normalization,
        options.layout,
        image.shape[2],
        width,
        device,
    );
    Ok(single_result(tensor, image))
}

/// Decode a batch into one rank-4 U8 Burn Metal tensor.
pub fn decode_u8_batch(
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    device: &WgpuDevice,
) -> Result<TensorBatchDecode<MetalIntTensor<4>>, TensorDecodeError> {
    ensure_dtype::<Wgpu>(device, DType::U8)?;
    let session = metal_session()?;
    let images = decode_surfaces(&session, inputs, options, SampleWidth::U8)?;
    let (bytes, shape) = readback_batch(&session, &images)?;
    let tensor = integer_tensor_4_from_bytes::<Wgpu>(
        bytes,
        images.len(),
        shape,
        options.layout,
        device,
        DType::U8,
    );
    Ok(batch_result(tensor, images))
}

/// Decode a batch into one rank-4 U16 Burn Metal tensor.
pub fn decode_u16_batch(
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    device: &WgpuDevice,
) -> Result<TensorBatchDecode<MetalIntTensor<4>>, TensorDecodeError> {
    ensure_dtype::<Wgpu>(device, DType::U16)?;
    let session = metal_session()?;
    let images = decode_surfaces(&session, inputs, options, SampleWidth::U16)?;
    let (bytes, shape) = readback_batch(&session, &images)?;
    let tensor = integer_tensor_4_from_bytes::<Wgpu>(
        bytes,
        images.len(),
        shape,
        options.layout,
        device,
        DType::U16,
    );
    Ok(batch_result(tensor, images))
}

/// Decode a batch into one rank-4 F32 Burn Metal tensor.
pub fn decode_float_batch(
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    device: &WgpuDevice,
) -> Result<TensorBatchDecode<MetalTensor<4>>, TensorDecodeError> {
    validate_normalization_values(&options.normalization)?;
    ensure_dtype::<Wgpu>(device, DType::F32)?;
    let first = inputs.first().ok_or(TensorDecodeError::EmptyBatch)?;
    let first_info =
        J2kDecoder::inspect(first.encoded).map_err(|error| indexed(0, error.into()))?;
    let width = sample_width(first_info.bit_depth);
    ensure_dtype::<Wgpu>(device, width.dtype())?;
    validate_normalization_channels(
        &options.normalization,
        selected_channels(options.channels, first_info.components),
    )
    .map_err(|error| indexed(0, error))?;
    for (index, input) in inputs.iter().enumerate().skip(1) {
        let item_info =
            J2kDecoder::inspect(input.encoded).map_err(|error| indexed(index, error.into()))?;
        let item_width = sample_width(item_info.bit_depth);
        if item_width.dtype() != width.dtype() {
            return Err(indexed(
                index,
                strict(format!(
                    "mixed canonical integer widths are unsupported: first item uses {:?}, item uses {:?}",
                    width.dtype(),
                    item_width.dtype()
                )),
            ));
        }
        validate_normalization_channels(
            &options.normalization,
            selected_channels(options.channels, item_info.components),
        )
        .map_err(|error| indexed(index, error))?;
    }
    let session = metal_session()?;
    let images = decode_surfaces(&session, inputs, options, width)?;
    let (bytes, shape) = readback_batch(&session, &images)?;
    let tensor = integer_tensor_4_from_bytes::<Wgpu>(
        bytes,
        images.len(),
        shape,
        options.layout,
        device,
        width.dtype(),
    )
    .cast(FloatDType::F32);
    let tensor = normalize_4(
        tensor,
        &options.normalization,
        options.layout,
        shape[2],
        width,
        device,
    );
    Ok(batch_result(tensor, images))
}

fn sample_width(bit_depth: u8) -> SampleWidth {
    if bit_depth <= 8 {
        SampleWidth::U8
    } else {
        SampleWidth::U16
    }
}

fn metal_session() -> Result<MetalBackendSession, TensorDecodeError> {
    MetalBackendSession::system_default().map_err(|error| strict(error.to_string()))
}

fn decode_surfaces(
    session: &MetalBackendSession,
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    width: SampleWidth,
) -> Result<Vec<MetalImage>, TensorDecodeError> {
    let first = inputs.first().ok_or(TensorDecodeError::EmptyBatch)?;
    let expected_shape =
        planned_shape(*first, options.channels).map_err(|error| indexed(0, error))?;
    for (index, input) in inputs.iter().enumerate().skip(1) {
        let actual =
            planned_shape(*input, options.channels).map_err(|error| indexed(index, error))?;
        if actual != expected_shape {
            return Err(TensorDecodeError::BatchShapeMismatch {
                index,
                expected: expected_shape,
                actual,
            });
        }
    }
    let mut images: Vec<MetalImage> = Vec::new();
    images
        .try_reserve_exact(inputs.len())
        .map_err(|_| TensorDecodeError::SizeOverflow)?;
    for (index, input) in inputs.iter().enumerate() {
        let image = decode_surface(session, *input, options, width)
            .map_err(|error| indexed(index, error))?;
        if let Some(first) = images.first() {
            if image.shape != first.shape {
                return Err(TensorDecodeError::BatchShapeMismatch {
                    index,
                    expected: first.shape,
                    actual: image.shape,
                });
            }
        }
        images.push(image);
    }
    Ok(images)
}

fn decode_surface(
    session: &MetalBackendSession,
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    width: SampleWidth,
) -> Result<MetalImage, TensorDecodeError> {
    let mut decoder =
        MetalDecoder::new(input.encoded).map_err(|error| strict(error.to_string()))?;
    let info = decoder.inner().info();
    let plan = DeviceDecodePlan::for_image(info.dimensions, input.request)?;
    let channels = selected_channels(options.channels, info.components);
    let format = pixel_format(channels, width);
    let request = metal_request(input.request, format);
    let surface = decoder
        .decode_request_to_device_with_session(request, session)
        .map_err(|error| strict(error.to_string()))?;
    if surface.residency() != SurfaceResidency::MetalResidentDecode {
        return Err(strict(format!(
            "resident decode returned unexpected residency {:?}",
            surface.residency()
        )));
    }
    let (output_width, output_height) = plan.output_dims();
    Ok(MetalImage {
        surface,
        shape: [
            usize::try_from(output_height).map_err(|_| TensorDecodeError::SizeOverflow)?,
            usize::try_from(output_width).map_err(|_| TensorDecodeError::SizeOverflow)?,
            channels,
        ],
        sample_width: width,
        decoded: plan.output_rect(),
        warnings: vec![J2kDecodeWarning::LenientDecodeMode],
    })
}

fn metal_request(request: DeviceDecodeRequest, format: j2k::PixelFormat) -> MetalDecodeRequest {
    match request {
        DeviceDecodeRequest::Full => MetalDecodeRequest::full(format, BackendRequest::Metal),
        DeviceDecodeRequest::Region { roi } => {
            MetalDecodeRequest::region(format, roi, BackendRequest::Metal)
        }
        DeviceDecodeRequest::Scaled { scale } => {
            MetalDecodeRequest::scaled(format, scale, BackendRequest::Metal)
        }
        DeviceDecodeRequest::RegionScaled { roi, scale } => {
            MetalDecodeRequest::region_scaled(format, roi, scale, BackendRequest::Metal)
        }
    }
}

fn readback_batch(
    session: &MetalBackendSession,
    images: &[MetalImage],
) -> Result<(Vec<u8>, [usize; 3]), TensorDecodeError> {
    let first = images.first().ok_or(TensorDecodeError::EmptyBatch)?;
    let mut surfaces = Vec::new();
    surfaces
        .try_reserve_exact(images.len())
        .map_err(|_| TensorDecodeError::SizeOverflow)?;
    for image in images {
        surfaces.push((&image.surface, image_byte_len(image)?));
    }
    Ok((packed_readback(session, &surfaces)?, first.shape))
}

fn image_byte_len(image: &MetalImage) -> Result<usize, TensorDecodeError> {
    let expected_pitch = image.shape[1]
        .checked_mul(image.shape[2])
        .and_then(|samples| samples.checked_mul(image.sample_width.bytes()))
        .ok_or(TensorDecodeError::SizeOverflow)?;
    if image.surface.pitch_bytes() != expected_pitch {
        return Err(strict(format!(
            "Metal surface pitch {} does not match compact pitch {expected_pitch}",
            image.surface.pitch_bytes()
        )));
    }
    expected_pitch
        .checked_mul(image.shape[0])
        .ok_or(TensorDecodeError::SizeOverflow)
}

#[cfg(target_os = "macos")]
fn packed_readback(
    session: &MetalBackendSession,
    surfaces: &[(&Surface, usize)],
) -> Result<Vec<u8>, TensorDecodeError> {
    let mut surface_refs = Vec::new();
    surface_refs
        .try_reserve_exact(surfaces.len())
        .map_err(|_| TensorDecodeError::SizeOverflow)?;
    surface_refs.extend(surfaces.iter().map(|(surface, _)| *surface));
    download_surfaces_packed(session, &surface_refs).map_err(|error| strict(error.to_string()))
}

#[cfg(not(target_os = "macos"))]
fn packed_readback(
    _session: &MetalBackendSession,
    _surfaces: &[(&Surface, usize)],
) -> Result<Vec<u8>, TensorDecodeError> {
    Err(strict("Metal is unavailable on this platform"))
}

fn strict(message: impl Into<String>) -> TensorDecodeError {
    TensorDecodeError::StrictRoute {
        route: TensorRoute::MetalStaged,
        message: message.into(),
    }
}

fn indexed(index: usize, source: TensorDecodeError) -> TensorDecodeError {
    TensorDecodeError::BatchItem {
        index,
        source: Box::new(source),
    }
}

fn single_result<T>(tensor: T, image: MetalImage) -> TensorDecode<T> {
    TensorDecode {
        tensor,
        decoded: image.decoded,
        warnings: image.warnings,
        route: TensorRoute::MetalStaged,
    }
}

fn batch_result<T>(tensor: T, images: Vec<MetalImage>) -> TensorBatchDecode<T> {
    let mut decoded = Vec::with_capacity(images.len());
    let mut warnings = Vec::with_capacity(images.len());
    for image in images {
        decoded.push(image.decoded);
        warnings.push(image.warnings);
    }
    TensorBatchDecode {
        tensor,
        decoded,
        warnings,
        route: TensorRoute::MetalStaged,
    }
}
