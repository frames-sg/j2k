// SPDX-License-Identifier: MIT OR Apache-2.0

//! Strict decode directly into Burn's default fused CUDA backend.

use burn_core::tensor::{DType, Int, Tensor};
use burn_cuda::{Cuda, CudaDevice};
use j2k::{DeviceDecodePlan, J2kDecodeWarning, Rect};
use j2k_cuda::{CudaSession, J2kDecoder as CudaDecoder, Surface};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};

use crate::cpu::{
    ensure_dtype, pixel_format, selected_channels, validate_normalization_channels,
    validate_normalization_values, SampleWidth,
};
use crate::{
    TensorBatchDecode, TensorDecode, TensorDecodeError, TensorDecodeOptions, TensorInput,
    TensorRoute,
};

mod config;
mod interop;

use config::{kernel_config, tensor_shape_3, tensor_shape_4};
use interop::{fill_float_tensor, fill_int_tensor};

type CudaTensor<const D: usize> = Tensor<Cuda, D>;
type CudaIntTensor<const D: usize> = Tensor<Cuda, D, Int>;

#[derive(Debug, Clone)]
struct PlannedImage<'a> {
    input: TensorInput<'a>,
    pub(super) shape: [usize; 3],
    decoded: Rect,
}

/// Decode one image into a rank-3 U8 tensor without decoded-pixel host transfer.
pub fn decode_u8(
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    device: &CudaDevice,
) -> Result<TensorDecode<CudaIntTensor<3>>, TensorDecodeError> {
    ensure_dtype::<Cuda>(device, DType::U8)?;
    let plan = plan_image(input, options, SampleWidth::U8)?;
    let shape = tensor_shape_3(plan.shape, options.layout);
    let context =
        CudaContext::retain_primary(device.index).map_err(|error| cuda_runtime_error(&error))?;
    let tensor = fill_int_tensor::<3>(shape, DType::U8, device, &context, |destination| {
        decode_plans_into(
            std::slice::from_ref(&plan),
            options,
            SampleWidth::U8,
            true,
            false,
            &context,
            destination,
        )
    })?;
    Ok(single_result(tensor, &plan))
}

/// Decode one image into a rank-3 U16 tensor without decoded-pixel host transfer.
pub fn decode_u16(
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    device: &CudaDevice,
) -> Result<TensorDecode<CudaIntTensor<3>>, TensorDecodeError> {
    ensure_dtype::<Cuda>(device, DType::U16)?;
    let plan = plan_image(input, options, SampleWidth::U16)?;
    let shape = tensor_shape_3(plan.shape, options.layout);
    let context =
        CudaContext::retain_primary(device.index).map_err(|error| cuda_runtime_error(&error))?;
    let tensor = fill_int_tensor::<3>(shape, DType::U16, device, &context, |destination| {
        decode_plans_into(
            std::slice::from_ref(&plan),
            options,
            SampleWidth::U16,
            true,
            false,
            &context,
            destination,
        )
    })?;
    Ok(single_result(tensor, &plan))
}

/// Decode one image into a rank-3 F32 tensor without decoded-pixel host transfer.
pub fn decode_float(
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    device: &CudaDevice,
) -> Result<TensorDecode<CudaTensor<3>>, TensorDecodeError> {
    validate_normalization_values(&options.normalization)?;
    ensure_dtype::<Cuda>(device, DType::F32)?;
    let width = canonical_width(input.encoded)?;
    let plan = plan_image(input, options, width)?;
    validate_normalization_channels(&options.normalization, plan.shape[2])?;
    let shape = tensor_shape_3(plan.shape, options.layout);
    let context =
        CudaContext::retain_primary(device.index).map_err(|error| cuda_runtime_error(&error))?;
    let tensor = fill_float_tensor::<3>(shape, device, &context, |destination| {
        decode_plans_into(
            std::slice::from_ref(&plan),
            options,
            width,
            false,
            false,
            &context,
            destination,
        )
    })?;
    Ok(single_result(tensor, &plan))
}

/// Decode a batch into one rank-4 U8 tensor without decoded-pixel host transfer.
pub fn decode_u8_batch(
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    device: &CudaDevice,
) -> Result<TensorBatchDecode<CudaIntTensor<4>>, TensorDecodeError> {
    ensure_dtype::<Cuda>(device, DType::U8)?;
    let plans = plan_batch(inputs, options, SampleWidth::U8)?;
    let shape = tensor_shape_4(plans.len(), plans[0].shape, options.layout);
    let context =
        CudaContext::retain_primary(device.index).map_err(|error| cuda_runtime_error(&error))?;
    let tensor = fill_int_tensor::<4>(shape, DType::U8, device, &context, |destination| {
        decode_plans_into(
            &plans,
            options,
            SampleWidth::U8,
            true,
            true,
            &context,
            destination,
        )
    })?;
    Ok(batch_result(tensor, &plans))
}

/// Decode a batch into one rank-4 U16 tensor without decoded-pixel host transfer.
pub fn decode_u16_batch(
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    device: &CudaDevice,
) -> Result<TensorBatchDecode<CudaIntTensor<4>>, TensorDecodeError> {
    ensure_dtype::<Cuda>(device, DType::U16)?;
    let plans = plan_batch(inputs, options, SampleWidth::U16)?;
    let shape = tensor_shape_4(plans.len(), plans[0].shape, options.layout);
    let context =
        CudaContext::retain_primary(device.index).map_err(|error| cuda_runtime_error(&error))?;
    let tensor = fill_int_tensor::<4>(shape, DType::U16, device, &context, |destination| {
        decode_plans_into(
            &plans,
            options,
            SampleWidth::U16,
            true,
            true,
            &context,
            destination,
        )
    })?;
    Ok(batch_result(tensor, &plans))
}

/// Decode a batch into one rank-4 F32 tensor without decoded-pixel host transfer.
pub fn decode_float_batch(
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    device: &CudaDevice,
) -> Result<TensorBatchDecode<CudaTensor<4>>, TensorDecodeError> {
    validate_normalization_values(&options.normalization)?;
    ensure_dtype::<Cuda>(device, DType::F32)?;
    let first = inputs.first().ok_or(TensorDecodeError::EmptyBatch)?;
    let width = canonical_width(first.encoded).map_err(|error| indexed(0, error))?;
    for (index, input) in inputs.iter().enumerate().skip(1) {
        let item_width = canonical_width(input.encoded).map_err(|error| indexed(index, error))?;
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
    }
    let plans = plan_batch(inputs, options, width)?;
    validate_normalization_channels(&options.normalization, plans[0].shape[2])?;
    let shape = tensor_shape_4(plans.len(), plans[0].shape, options.layout);
    let context =
        CudaContext::retain_primary(device.index).map_err(|error| cuda_runtime_error(&error))?;
    let tensor = fill_float_tensor::<4>(shape, device, &context, |destination| {
        decode_plans_into(&plans, options, width, false, true, &context, destination)
    })?;
    Ok(batch_result(tensor, &plans))
}

fn canonical_width(encoded: &[u8]) -> Result<SampleWidth, TensorDecodeError> {
    Ok(if j2k::J2kDecoder::inspect(encoded)?.bit_depth <= 8 {
        SampleWidth::U8
    } else {
        SampleWidth::U16
    })
}

fn plan_image<'a>(
    input: TensorInput<'a>,
    options: &TensorDecodeOptions,
    width: SampleWidth,
) -> Result<PlannedImage<'a>, TensorDecodeError> {
    let info = j2k::J2kDecoder::inspect(input.encoded)?;
    let plan = DeviceDecodePlan::for_image(info.dimensions, input.request)?;
    let channels = selected_channels(options.channels, info.components);
    let (width_px, height_px) = plan.output_dims();
    let _ = pixel_format(channels, width);
    Ok(PlannedImage {
        input,
        shape: [
            usize::try_from(height_px).map_err(|_| TensorDecodeError::SizeOverflow)?,
            usize::try_from(width_px).map_err(|_| TensorDecodeError::SizeOverflow)?,
            channels,
        ],
        decoded: plan.output_rect(),
    })
}

fn plan_batch<'a>(
    inputs: &[TensorInput<'a>],
    options: &TensorDecodeOptions,
    width: SampleWidth,
) -> Result<Vec<PlannedImage<'a>>, TensorDecodeError> {
    if inputs.is_empty() {
        return Err(TensorDecodeError::EmptyBatch);
    }
    let mut plans: Vec<PlannedImage<'a>> = Vec::new();
    plans
        .try_reserve_exact(inputs.len())
        .map_err(|_| TensorDecodeError::SizeOverflow)?;
    for (index, input) in inputs.iter().enumerate() {
        let plan = plan_image(*input, options, width).map_err(|error| indexed(index, error))?;
        if let Some(first) = plans.first() {
            if plan.shape != first.shape {
                return Err(TensorDecodeError::BatchShapeMismatch {
                    index,
                    expected: first.shape,
                    actual: plan.shape,
                });
            }
        }
        plans.push(plan);
    }
    Ok(plans)
}

fn decode_plans_into(
    plans: &[PlannedImage<'_>],
    options: &TensorDecodeOptions,
    width: SampleWidth,
    integer_output: bool,
    index_errors: bool,
    context: &CudaContext,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
) -> Result<(), TensorDecodeError> {
    let mut session = CudaSession::with_context(context.clone());
    let item_elements = plans[0]
        .shape
        .iter()
        .try_fold(1usize, |size, dim| size.checked_mul(*dim))
        .ok_or(TensorDecodeError::SizeOverflow)?;
    for (index, plan) in plans.iter().enumerate() {
        let mut decoder = CudaDecoder::new(plan.input.encoded)
            .map_err(|error| route_item_error(index_errors, index, strict(error.to_string())))?;
        let format = pixel_format(plan.shape[2], width);
        let surface = decoder
            .decode_request_to_device_with_session(format, plan.input.request, &mut session)
            .map_err(|error| route_item_error(index_errors, index, strict(error.to_string())))?;
        validate_surface(&surface, plan.shape, width)
            .map_err(|error| route_item_error(index_errors, index, error))?;
        let source = surface.cuda_surface().ok_or_else(|| {
            route_item_error(
                index_errors,
                index,
                strict("strict CUDA decode returned a host surface"),
            )
        })?;
        let source_len = item_elements
            .checked_mul(match width {
                SampleWidth::U8 => 1,
                SampleWidth::U16 => 2,
            })
            .ok_or(TensorDecodeError::SizeOverflow)?;
        context
            .j2k_ml_convert_into_external(
                source.device_ptr(),
                source_len,
                destination,
                kernel_config(plan, options, width, integer_output, index, item_elements)?,
            )
            .map_err(|error| route_item_error(index_errors, index, cuda_runtime_error(&error)))?;
    }
    Ok(())
}

fn validate_surface(
    surface: &Surface,
    shape: [usize; 3],
    width: SampleWidth,
) -> Result<(), TensorDecodeError> {
    let sample_bytes = match width {
        SampleWidth::U8 => 1,
        SampleWidth::U16 => 2,
    };
    let expected_pitch = shape[1]
        .checked_mul(shape[2])
        .and_then(|value| value.checked_mul(sample_bytes))
        .ok_or(TensorDecodeError::SizeOverflow)?;
    if surface.pitch_bytes() != expected_pitch {
        return Err(strict(format!(
            "CUDA surface pitch {} does not match compact pitch {expected_pitch}",
            surface.pitch_bytes()
        )));
    }
    Ok(())
}

pub(super) fn strict(message: impl Into<String>) -> TensorDecodeError {
    TensorDecodeError::StrictRoute {
        route: TensorRoute::CudaDirect,
        message: message.into(),
    }
}

pub(super) fn cuda_runtime_error(error: &j2k_cuda_runtime::CudaError) -> TensorDecodeError {
    strict(error.to_string())
}

fn indexed(index: usize, source: TensorDecodeError) -> TensorDecodeError {
    TensorDecodeError::BatchItem {
        index,
        source: Box::new(source),
    }
}

fn route_item_error(is_batch: bool, index: usize, source: TensorDecodeError) -> TensorDecodeError {
    if is_batch {
        indexed(index, source)
    } else {
        source
    }
}

fn single_result<T>(tensor: T, plan: &PlannedImage<'_>) -> TensorDecode<T> {
    TensorDecode {
        tensor,
        decoded: plan.decoded,
        warnings: vec![J2kDecodeWarning::LenientDecodeMode],
        route: TensorRoute::CudaDirect,
    }
}

fn batch_result<T>(tensor: T, plans: &[PlannedImage<'_>]) -> TensorBatchDecode<T> {
    TensorBatchDecode {
        tensor,
        decoded: plans.iter().map(|plan| plan.decoded).collect(),
        warnings: plans
            .iter()
            .map(|_| vec![J2kDecodeWarning::LenientDecodeMode])
            .collect(),
        route: TensorRoute::CudaDirect,
    }
}
