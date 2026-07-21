// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::{DType, Int, Shape, Tensor, TensorPrimitive};
use burn_cubecl::cubecl::{cuda::CudaRuntime, Runtime};
use burn_cubecl::{ops::numeric::empty_device_contiguous_dtype, CubeBackend};
use burn_cuda::{Cuda, CudaDevice};
use burn_fusion::{get_client, stream::OperationStreams, NoOp};
use burn_ir::{BackendIr, InitOperationIr, OperationIr};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};

use super::{cuda_runtime_error, strict};
use crate::TensorDecodeError;

type InnerCuda = CubeBackend<CudaRuntime, f32, i32, u8>;

pub(super) fn fill_int_tensor<const D: usize>(
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

pub(super) fn fill_float_tensor<const D: usize>(
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
    // CubeCL allocates asynchronously. Its stream must finish before j2k writes
    // through the default CUDA stream because no event interop is exposed.
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
    // Its exclusive borrow spans `fill`; context, extent, and alignment are
    // validated by the non-owning view constructor.
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

pub(super) fn tensor_byte_len(shape: &[usize], dtype: DType) -> Result<usize, TensorDecodeError> {
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
