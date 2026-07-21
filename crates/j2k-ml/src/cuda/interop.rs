// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::{DType, Int, Shape, Tensor};
use burn_cubecl::cubecl::{cuda::CudaRuntime, Runtime};
use burn_cubecl::{ops::numeric::empty_device_contiguous_dtype, CubeBackend};
use burn_cuda::{Cuda, CudaDevice};
use burn_fusion::{get_client, stream::OperationStreams, NoOp};
use burn_ir::{BackendIr, InitOperationIr, OperationIr};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};

use crate::BurnDecodeError;

type InnerCuda = CubeBackend<CudaRuntime, f32, i32, u8>;

/// Fresh Burn allocation plus an asynchronously submitted codec owner.
///
/// `payload` is declared first so dropping an unfinished value retires codec
/// work before the `CubeCL` allocation is released.
pub(super) struct SubmittedBatchIntTensor<R, const D: usize> {
    payload: R,
    cube: burn_cubecl::tensor::CubeTensor<CudaRuntime>,
    shape: Shape,
    dtype: DType,
    device: CudaDevice,
}

impl<R, const D: usize> SubmittedBatchIntTensor<R, D> {
    pub(super) fn payload(&self) -> &R {
        &self.payload
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        burn_cubecl::tensor::CubeTensor<CudaRuntime>,
        Shape,
        DType,
        CudaDevice,
        R,
    ) {
        let Self {
            payload,
            cube,
            shape,
            dtype,
            device,
        } = self;
        (cube, shape, dtype, device, payload)
    }
}

pub(super) fn fill_batch_int_tensor<R, const D: usize>(
    shape: [usize; D],
    dtype: DType,
    device: &CudaDevice,
    context: &CudaContext,
    fill: impl FnOnce(&mut CudaExternalDeviceBufferViewMut<'_>) -> Result<R, BurnDecodeError>,
) -> Result<SubmittedBatchIntTensor<R, D>, BurnDecodeError> {
    let logical_len = tensor_byte_len(&shape, dtype)?;
    let burn_shape = Shape::from(shape.to_vec());
    let client = CudaRuntime::client(device);
    let cube = empty_device_contiguous_dtype(client, device.clone(), burn_shape.clone(), dtype);
    let handle_len =
        usize::try_from(cube.handle.size_in_used()).map_err(|_| BurnDecodeError::SizeOverflow)?;
    if handle_len != logical_len {
        return Err(interop(format!(
            "CubeCL tensor handle exposes {handle_len} bytes; expected {logical_len}"
        )));
    }

    // SAFETY: the stream token stays inside this audited CUDA bridge. The
    // client, allocation, managed resource, and codec completion owner remain
    // live until both cross-stream event dependencies have been registered.
    let raw_stream = unsafe { cube.client.external_write_stream(&cube.handle) }
        .map_err(|error| interop(format!("CubeCL stream handoff failed: {error}")))?;
    let mut stream_owner = cube.client.clone();
    let mut resource = cube
        .client
        .get_resource(cube.handle.clone())
        .map_err(|error| interop(format!("CubeCL resource access failed: {error}")))?;
    let raw = resource.resource();
    let available = usize::try_from(raw.size).map_err(|_| BurnDecodeError::SizeOverflow)?;
    if logical_len > available {
        return Err(interop(format!(
            "CubeCL resource exposes {available} bytes for a {logical_len}-byte tensor"
        )));
    }

    // SAFETY: this is a fresh, unregistered CubeCL tensor. Its managed
    // resource is exclusively borrowed for the complete codec submission;
    // the view validates retained-primary-context identity, extent, and
    // alignment. `with_primary_stream_ordering` orders allocation before the
    // codec and the Burn stream after the codec without a CPU synchronization.
    let submission = unsafe {
        let mut destination = CudaExternalDeviceBufferViewMut::from_raw_parts(
            context,
            raw.ptr,
            logical_len,
            dtype.size(),
            &mut resource,
        )
        .map_err(|error| interop(error.to_string()))?;
        let mut work_invoked = false;
        let ordered = context.with_primary_stream_ordering(raw_stream, &mut stream_owner, || {
            work_invoked = true;
            fill(&mut destination)
        });
        drop(destination);
        match ordered {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(error)) => {
                let quarantine = burn_error_completion_is_uncertain(&error);
                Err((error, quarantine))
            }
            Err(error) => {
                let quarantine = work_invoked && error.completion_is_uncertain();
                Err((
                    interop(format!("CUDA stream event handoff failed: {error}")),
                    quarantine,
                ))
            }
        }
    };
    drop(resource);
    let output = match submission {
        Ok(output) => output,
        Err((error, quarantine)) => {
            if quarantine {
                // CUDA could not prove that the external allocation is no
                // longer referenced. Quarantine it rather than letting CubeCL
                // recycle or free storage still reachable by the driver.
                std::mem::forget(cube);
            }
            return Err(error);
        }
    };

    Ok(SubmittedBatchIntTensor {
        payload: output,
        cube,
        shape: burn_shape,
        dtype,
        device: device.clone(),
    })
}

fn tensor_byte_len(shape: &[usize], dtype: DType) -> Result<usize, BurnDecodeError> {
    shape
        .iter()
        .try_fold(dtype.size(), |size, dim| size.checked_mul(*dim))
        .ok_or(BurnDecodeError::SizeOverflow)
}

pub(super) fn register_int_tensor<const D: usize>(
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

fn interop(message: impl Into<String>) -> BurnDecodeError {
    BurnDecodeError::AcceleratorInterop {
        backend: "CUDA",
        message: message.into(),
    }
}

fn burn_error_completion_is_uncertain(error: &BurnDecodeError) -> bool {
    matches!(error, BurnDecodeError::Cuda(source) if source.completion_is_uncertain())
}

#[cfg(test)]
mod tests {
    use burn_core::tensor::DType;

    use super::{burn_error_completion_is_uncertain, tensor_byte_len};
    use crate::BurnDecodeError;

    #[test]
    fn tensor_byte_length_is_exact_and_overflow_checked_before_allocation() {
        assert_eq!(tensor_byte_len(&[2, 3, 4], DType::U16).unwrap(), 48);
        assert!(matches!(
            tensor_byte_len(&[usize::MAX, 2], DType::U16),
            Err(BurnDecodeError::SizeOverflow)
        ));
    }

    #[test]
    fn only_uncertain_cuda_completion_requires_allocation_quarantine() {
        let validation = BurnDecodeError::Cuda(j2k_cuda::CudaBatchError::GroupExecution {
            source_indices: vec![0],
            source: Box::new(j2k_cuda::Error::UnsupportedCudaRequest {
                reason: "pre-submit validation",
            }),
        });
        assert!(!burn_error_completion_is_uncertain(&validation));

        let uncertain = BurnDecodeError::Cuda(j2k_cuda::CudaBatchError::GroupExecution {
            source_indices: vec![0],
            source: Box::new(j2k_cuda::Error::CudaRuntime {
                source: j2k_cuda_runtime::CudaError::StatePoisoned {
                    message: "completion unknown".to_string(),
                },
            }),
        });
        assert!(burn_error_completion_is_uncertain(&uncertain));
    }
}
