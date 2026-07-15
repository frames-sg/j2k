// SPDX-License-Identifier: MIT OR Apache-2.0

//! Strict decode directly into Burn's default fused CUDA backend.

use burn_core::tensor::{DType, Int, Shape, Tensor, TensorPrimitive};
use burn_cubecl::cubecl::{cuda::CudaRuntime, Runtime};
use burn_cubecl::{ops::numeric::empty_device_contiguous_dtype, CubeBackend};
use burn_cuda::{Cuda, CudaDevice};
use burn_fusion::{get_client, stream::OperationStreams, NoOp};
use burn_ir::{BackendIr, InitOperationIr, OperationIr};
use j2k::{DeviceDecodePlan, J2kDecodeWarning, Rect};
use j2k_cuda::{CudaSession, J2kDecoder as CudaDecoder, Surface};
use j2k_cuda_runtime::{
    CudaContext, CudaExternalDeviceBufferViewMut, CudaJ2kMlKernelConfig, CudaJ2kMlLayout,
    CudaJ2kMlNormalization, CudaJ2kMlSample,
};

use crate::cpu::{
    ensure_dtype, pixel_format, selected_channels, validate_normalization_channels,
    validate_normalization_values, SampleWidth,
};
use crate::{
    FloatNormalization, TensorBatchDecode, TensorDecode, TensorDecodeError, TensorDecodeOptions,
    TensorInput, TensorLayout, TensorRoute,
};

type InnerCuda = CubeBackend<CudaRuntime, f32, i32, u8>;
type CudaTensor<const D: usize> = Tensor<Cuda, D>;
type CudaIntTensor<const D: usize> = Tensor<Cuda, D, Int>;

#[derive(Debug, Clone)]
struct PlannedImage<'a> {
    input: TensorInput<'a>,
    shape: [usize; 3],
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

fn kernel_config(
    plan: &PlannedImage<'_>,
    options: &TensorDecodeOptions,
    width: SampleWidth,
    integer_output: bool,
    index: usize,
    item_elements: usize,
) -> Result<CudaJ2kMlKernelConfig, TensorDecodeError> {
    let sample = match width {
        SampleWidth::U8 => CudaJ2kMlSample::U8,
        SampleWidth::U16 => CudaJ2kMlSample::U16,
    };
    let layout = match options.layout {
        TensorLayout::ChannelsFirst => CudaJ2kMlLayout::ChannelsFirst,
        TensorLayout::ChannelsLast => CudaJ2kMlLayout::ChannelsLast,
    };
    let normalization = if integer_output {
        CudaJ2kMlNormalization::Integer
    } else {
        match &options.normalization {
            FloatNormalization::Raw => CudaJ2kMlNormalization::Raw,
            FloatNormalization::Unit => CudaJ2kMlNormalization::Unit,
            FloatNormalization::MeanStd { mean, std } => {
                let mut means = [0.0; 4];
                let mut deviations = [1.0; 4];
                means[..plan.shape[2]].copy_from_slice(mean);
                deviations[..plan.shape[2]].copy_from_slice(std);
                CudaJ2kMlNormalization::MeanStd {
                    mean: means,
                    std: deviations,
                }
            }
        }
    };
    Ok(CudaJ2kMlKernelConfig {
        width: u32::try_from(plan.shape[1]).map_err(|_| TensorDecodeError::SizeOverflow)?,
        height: u32::try_from(plan.shape[0]).map_err(|_| TensorDecodeError::SizeOverflow)?,
        channels: u32::try_from(plan.shape[2]).map_err(|_| TensorDecodeError::SizeOverflow)?,
        sample,
        layout,
        destination_offset_elements: index
            .checked_mul(item_elements)
            .ok_or(TensorDecodeError::SizeOverflow)?,
        normalization,
    })
}

fn fill_int_tensor<const D: usize>(
    shape: [usize; D],
    dtype: DType,
    device: &CudaDevice,
    context: &CudaContext,
    fill: impl FnOnce(&mut CudaExternalDeviceBufferViewMut<'_>) -> Result<(), TensorDecodeError>,
) -> Result<Tensor<Cuda, D, Int>, TensorDecodeError> {
    let cube = fill_cube_tensor(shape, dtype, device, context, fill)?;
    Ok(register_int_tensor(
        cube,
        Shape::from(shape.to_vec()),
        dtype,
        device,
    ))
}

fn fill_float_tensor<const D: usize>(
    shape: [usize; D],
    device: &CudaDevice,
    context: &CudaContext,
    fill: impl FnOnce(&mut CudaExternalDeviceBufferViewMut<'_>) -> Result<(), TensorDecodeError>,
) -> Result<Tensor<Cuda, D>, TensorDecodeError> {
    let cube = fill_cube_tensor(shape, DType::F32, device, context, fill)?;
    Ok(register_float_tensor(
        cube,
        Shape::from(shape.to_vec()),
        device,
    ))
}

fn fill_cube_tensor<const D: usize>(
    shape: [usize; D],
    dtype: DType,
    device: &CudaDevice,
    context: &CudaContext,
    fill: impl FnOnce(&mut CudaExternalDeviceBufferViewMut<'_>) -> Result<(), TensorDecodeError>,
) -> Result<burn_cubecl::tensor::CubeTensor<CudaRuntime>, TensorDecodeError> {
    let logical_len = tensor_byte_len(&shape, dtype)?;
    let shape = Shape::from(shape.to_vec());
    let client = CudaRuntime::client(device);
    let cube = empty_device_contiguous_dtype(client, device.clone(), shape, dtype);
    // CubeCL 0.10 allocates on a nonblocking stream with cuMemAllocAsync,
    // while j2k launches its conversion on the CUDA default stream. CubeCL
    // does not currently expose a stream or event interop primitive, so its
    // stream must complete before j2k may safely write the allocation.
    burn_cubecl::cubecl::future::block_on(cube.client.sync())
        .map_err(|error| strict(format!("CubeCL CUDA allocation handoff failed: {error}")))?;
    let handle_len =
        usize::try_from(cube.handle.size_in_used()).map_err(|_| TensorDecodeError::SizeOverflow)?;
    if handle_len != logical_len {
        return Err(strict(format!(
            "CubeCL CUDA tensor handle exposes {handle_len} bytes; expected {logical_len}"
        )));
    }
    let mut resource = cube
        .client
        .get_resource(cube.handle.clone())
        .map_err(|error| strict(format!("CubeCL CUDA resource access failed: {error}")))?;
    let raw = resource.resource();
    let available = usize::try_from(raw.size).map_err(|_| TensorDecodeError::SizeOverflow)?;
    if logical_len > available {
        return Err(strict(format!(
            "CubeCL CUDA resource exposes {available} bytes for a {logical_len}-byte tensor"
        )));
    }
    let pointer = raw.ptr;
    // SAFETY: `resource` is CubeCL's managed allocation guard for `pointer`.
    // Its exclusive borrow is retained by the non-owning view through `fill`,
    // and the context/length/alignment are validated by the constructor.
    let mut destination = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(
            context,
            pointer,
            logical_len,
            dtype.size(),
            &mut resource,
        )
    }
    .map_err(|error| cuda_runtime_error(&error))?;
    fill(&mut destination)?;
    drop(destination);
    drop(resource);
    Ok(cube)
}

fn tensor_byte_len(shape: &[usize], dtype: DType) -> Result<usize, TensorDecodeError> {
    shape
        .iter()
        .try_fold(dtype.size(), |size, dim| size.checked_mul(*dim))
        .ok_or(TensorDecodeError::SizeOverflow)
}

fn register_int_tensor<const D: usize>(
    cube: burn_cubecl::tensor::CubeTensor<CudaRuntime>,
    shape: Shape,
    dtype: DType,
    device: &CudaDevice,
) -> Tensor<Cuda, D, Int> {
    let fusion = get_client::<InnerCuda>(device);
    let handle = <InnerCuda as BackendIr>::int_tensor_handle(cube);
    let desc = InitOperationIr::create(shape, dtype, || fusion.register_tensor_handle(handle));
    let primitive = fusion
        .register(
            OperationStreams::default(),
            OperationIr::Init(desc),
            NoOp::<InnerCuda>::new(),
        )
        .remove(0);
    Tensor::<Cuda, D, Int>::from_primitive(primitive)
}

fn register_float_tensor<const D: usize>(
    cube: burn_cubecl::tensor::CubeTensor<CudaRuntime>,
    shape: Shape,
    device: &CudaDevice,
) -> Tensor<Cuda, D> {
    let fusion = get_client::<InnerCuda>(device);
    let handle = <InnerCuda as BackendIr>::float_tensor_handle(cube);
    let desc = InitOperationIr::create(shape, DType::F32, || fusion.register_tensor_handle(handle));
    let primitive = fusion
        .register(
            OperationStreams::default(),
            OperationIr::Init(desc),
            NoOp::<InnerCuda>::new(),
        )
        .remove(0);
    Tensor::<Cuda, D>::from_primitive(TensorPrimitive::Float(primitive))
}

fn tensor_shape_3(shape: [usize; 3], layout: TensorLayout) -> [usize; 3] {
    match layout {
        TensorLayout::ChannelsFirst => [shape[2], shape[0], shape[1]],
        TensorLayout::ChannelsLast => shape,
    }
}

fn tensor_shape_4(batch: usize, shape: [usize; 3], layout: TensorLayout) -> [usize; 4] {
    match layout {
        TensorLayout::ChannelsFirst => [batch, shape[2], shape[0], shape[1]],
        TensorLayout::ChannelsLast => [batch, shape[0], shape[1], shape[2]],
    }
}

fn strict(message: impl Into<String>) -> TensorDecodeError {
    TensorDecodeError::StrictRoute {
        route: TensorRoute::CudaDirect,
        message: message.into(),
    }
}

fn cuda_runtime_error(error: &j2k_cuda_runtime::CudaError) -> TensorDecodeError {
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

#[cfg(test)]
mod tests {
    use burn_core::tensor::DType;

    use super::tensor_byte_len;
    use crate::TensorDecodeError;

    #[test]
    fn tensor_byte_length_is_exact_and_overflow_checked_before_cubecl_allocation() {
        assert_eq!(tensor_byte_len(&[2, 3, 4], DType::U16).unwrap(), 48);
        assert!(matches!(
            tensor_byte_len(&[usize::MAX, 2], DType::F32),
            Err(TensorDecodeError::SizeOverflow)
        ));
    }
}
