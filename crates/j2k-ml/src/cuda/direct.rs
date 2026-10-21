// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synchronous decode directly into Burn-owned CUDA storage.

use burn_core::tensor::{backend::Backend, DType, Int, Shape, Tensor};
use burn_cubecl::{
    cubecl::{cuda::CudaRuntime, Runtime},
    tensor::CubeTensor,
    CubeBackend,
};
use burn_cuda::{Cuda, CudaDevice};
use burn_fusion::{get_client, stream::OperationStreams, NoOp};
use burn_ir::{InitOperationIr, OperationIr, OperationOutput};
use j2k::{
    BatchDecodeOptions, EncodedImage, NativeSampleType, PreparedBatch, PreparedBatchGroup,
    PreparedImage,
};
use j2k_cuda::{CudaBatchDecoder as CodecDecoder, CudaBatchError};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};

use crate::batch_contract::{dtype, tensor_shape};
use crate::{
    BurnBatchDecode, BurnBatchGroup, BurnBatchGroupError, BurnBatchTensor, BurnDecodeError,
};

type InnerCuda = CubeBackend<CudaRuntime, f32, i32, u8>;
type BurnCuda = Cuda<f32, i32>;

/// Persistent CUDA decoder that writes pixels directly into Burn-owned storage.
///
/// Each group is synchronous: the adapter allocates an ordinary Burn CUDA
/// buffer, waits for that allocation, decodes directly into it, validates codec
/// completion, and only then publishes the buffer as a fused [`Cuda`] tensor.
/// Decoded pixels are not copied through host memory or through a second CUDA
/// allocation.
#[derive(Debug)]
pub struct CudaDirectBurnDecoder {
    codec: CodecDecoder,
    device: CudaDevice,
}

impl CudaDirectBurnDecoder {
    /// Create a direct-destination decoder for one Burn CUDA device.
    #[must_use]
    pub fn new(device: CudaDevice, options: BatchDecodeOptions) -> Self {
        Self {
            codec: CodecDecoder::with_options(options),
            device,
        }
    }

    /// Borrow the persistent CUDA codec session.
    #[must_use]
    pub const fn codec(&self) -> &CodecDecoder {
        &self.codec
    }

    /// Borrow the Burn CUDA device that owns destination allocations.
    #[must_use]
    pub const fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Parse and group owned inputs once for repeated direct decode calls.
    pub fn prepare(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, BurnDecodeError> {
        Ok(self.codec.prepare(inputs)?)
    }

    /// Regroup caller-supplied prepared images without reparsing codestream bytes.
    pub fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, BurnDecodeError> {
        Ok(self.codec.prepare_prepared_images(images)?)
    }

    /// Prepare and decode into Burn-owned CUDA allocations.
    pub fn decode(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<BurnBatchDecode<BurnCuda>, BurnDecodeError> {
        let prepared = self.prepare(inputs)?;
        self.decode_prepared(&prepared)
    }

    /// Regroup prepared images and decode into Burn-owned CUDA allocations.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<BurnBatchDecode<BurnCuda>, BurnDecodeError> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a reusable preparation directly into Burn-owned CUDA allocations.
    ///
    /// Groups are submitted serially because Burn 0.21 does not expose its CUDA
    /// stream handle for an asynchronous cross-runtime completion bridge.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<BurnBatchDecode<BurnCuda>, BurnDecodeError> {
        ensure_dtypes(prepared, &self.device)?;
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(prepared.groups().len())
            .map_err(|_| BurnDecodeError::SizeOverflow)?;
        let mut group_errors = Vec::new();
        group_errors
            .try_reserve_exact(prepared.groups().len())
            .map_err(|_| BurnDecodeError::SizeOverflow)?;

        for group in prepared.groups() {
            match self.decode_group(group) {
                Ok(output) => groups.push(output),
                Err(error) if direct_error_is_fatal(&error) => return Err(error),
                Err(error) => group_errors.push(BurnBatchGroupError::new(
                    group.source_indices().to_vec(),
                    error,
                )),
            }
        }

        Ok(BurnBatchDecode {
            groups,
            errors: prepared.errors().to_vec(),
            group_errors,
        })
    }

    fn decode_group(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<BurnBatchGroup<BurnCuda>, BurnDecodeError> {
        let shape = tensor_shape(group.images().len(), group.info())?;
        let dtype = dtype(group.info().sample_type)?;
        let byte_len = crate::staging::byte_len(group.images().len(), group.info())?;
        let client = CudaRuntime::client(&self.device);
        let handle = client.empty(byte_len);

        <InnerCuda as Backend>::sync(&self.device).map_err(|source| {
            BurnDecodeError::AcceleratorInterop {
                backend: "Burn CUDA direct allocation",
                message: source.to_string(),
            }
        })?;

        let resource = client.get_resource(handle.clone()).map_err(|source| {
            BurnDecodeError::AcceleratorInterop {
                backend: "Burn CUDA direct allocation",
                message: source.to_string(),
            }
        })?;
        let context = self
            .codec
            .session_mut()
            .context_for_device_interop(self.device.index)
            .map_err(|source| CudaBatchError::GroupExecution {
                source_indices: group.source_indices().to_vec(),
                source: Box::new(source),
            })
            .map_err(BurnDecodeError::Cuda)?;
        let mut allocation = DirectAllocationGuard::new(context.clone(), handle, resource);
        let (ptr, exposed_size) = {
            let raw = allocation.resource()?.resource();
            (raw.ptr, raw.size)
        };
        if exposed_size < u64::try_from(byte_len).unwrap_or(u64::MAX) {
            return Err(BurnDecodeError::AcceleratorInterop {
                backend: "Burn CUDA direct allocation",
                message: format!(
                    "Burn allocation exposes {exposed_size} bytes; codec requires {byte_len}"
                ),
            });
        }

        // From this point onward an unwind must establish context completion
        // before CubeCL is allowed to recycle the destination allocation.
        allocation.arm();
        let result = {
            let resource = allocation.resource_mut()?;
            // SAFETY: `handle` and `resource` jointly keep this fresh CubeCL
            // allocation live. The inner-backend sync above completed its
            // stream-ordered allocation. `resource` is exclusively borrowed
            // through the view, and the synchronous codec call retires all
            // submitted work before the view is dropped.
            let mut destination = unsafe {
                CudaExternalDeviceBufferViewMut::from_raw_parts(
                    &context,
                    ptr,
                    byte_len,
                    dtype.size(),
                    resource,
                )
            }
            .map_err(|source| BurnDecodeError::AcceleratorInterop {
                backend: "Burn CUDA direct destination",
                message: source.to_string(),
            })?;
            // SAFETY: the unpublished Burn allocation remains exclusively
            // borrowed until this synchronous completion and status check.
            unsafe { self.codec.decode_batch_into(group, &mut destination) }
                .map_err(BurnDecodeError::Cuda)
        };
        let metadata = result?;
        let (handle, resource) = allocation.into_parts()?;
        drop(resource);

        let tensor = register_fused_tensor(client, handle, shape, dtype, &self.device);
        let tensor = match group.info().sample_type {
            NativeSampleType::U8 => BurnBatchTensor::U8(tensor),
            NativeSampleType::U16 => BurnBatchTensor::U16(tensor),
            NativeSampleType::I16 => BurnBatchTensor::I16(tensor),
            _ => return Err(BurnDecodeError::UnsupportedCodecContract),
        };
        Ok(BurnBatchGroup {
            tensor,
            info: metadata.info().clone(),
            source_indices: metadata.source_indices().to_vec(),
            decoded_rects: metadata.decoded_rects().to_vec(),
            warnings: metadata.warnings().to_vec(),
        })
    }
}

struct DirectAllocationGuard<H, R> {
    context: CudaContext,
    handle: Option<H>,
    resource: Option<R>,
    armed: bool,
}

impl<H, R> DirectAllocationGuard<H, R> {
    fn new(context: CudaContext, handle: H, resource: R) -> Self {
        Self {
            context,
            handle: Some(handle),
            resource: Some(resource),
            armed: false,
        }
    }

    fn resource(&self) -> Result<&R, BurnDecodeError> {
        self.resource.as_ref().ok_or_else(released_allocation_error)
    }

    fn resource_mut(&mut self) -> Result<&mut R, BurnDecodeError> {
        self.resource.as_mut().ok_or_else(released_allocation_error)
    }

    const fn arm(&mut self) {
        self.armed = true;
    }

    // Only called after the synchronous decode completed, so disarming is sound.
    fn into_parts(mut self) -> Result<(H, R), BurnDecodeError> {
        self.armed = false;
        match (self.handle.take(), self.resource.take()) {
            (Some(handle), Some(resource)) => Ok((handle, resource)),
            _ => Err(released_allocation_error()),
        }
    }
}

fn released_allocation_error() -> BurnDecodeError {
    BurnDecodeError::AcceleratorInterop {
        backend: "Burn CUDA direct allocation",
        message: "direct allocation owner was already released".to_owned(),
    }
}

impl<H, R> Drop for DirectAllocationGuard<H, R> {
    fn drop(&mut self) {
        if self.armed
            && !self
                .context
                .synchronize_for_resource_release()
                .completion_established()
        {
            // CUDA could not prove that queued work stopped using the Burn
            // allocation. Preserve both CubeCL lifetime owners indefinitely
            // instead of returning potentially live storage to its pool.
            if let Some(resource) = self.resource.take() {
                core::mem::forget(resource);
            }
            if let Some(handle) = self.handle.take() {
                core::mem::forget(handle);
            }
        }
    }
}

fn ensure_dtypes(prepared: &PreparedBatch, device: &CudaDevice) -> Result<(), BurnDecodeError> {
    for group in prepared.groups() {
        let dtype = dtype(group.info().sample_type)?;
        if !BurnCuda::supports_dtype(device, dtype) {
            return Err(BurnDecodeError::UnsupportedDType { dtype });
        }
    }
    Ok(())
}

fn register_fused_tensor(
    client: burn_cubecl::cubecl::client::ComputeClient<CudaRuntime>,
    handle: burn_cubecl::cubecl::server::Handle,
    shape: [usize; 4],
    dtype: DType,
    device: &CudaDevice,
) -> Tensor<BurnCuda, 4, Int> {
    let shape = Shape::from(shape);
    let tensor = CubeTensor::new_contiguous(client, device.clone(), shape.clone(), handle, dtype);
    let fusion_handle = tensor.into();
    let fusion_client = get_client::<InnerCuda>(device);
    let operation = InitOperationIr::create(shape, dtype, || {
        fusion_client.register_tensor_handle(fusion_handle)
    });
    let primitive = fusion_client
        .register(
            OperationStreams::default(),
            OperationIr::Init(operation),
            NoOp::<InnerCuda>::new(),
        )
        .output();
    Tensor::from_primitive(primitive)
}

fn direct_error_is_fatal(error: &BurnDecodeError) -> bool {
    match error {
        BurnDecodeError::Infrastructure(_) | BurnDecodeError::AcceleratorInterop { .. } => true,
        BurnDecodeError::Cuda(source) => source.session_is_unusable(),
        BurnDecodeError::UnsupportedDType { .. }
        | BurnDecodeError::SampleTypeMismatch
        | BurnDecodeError::SizeOverflow
        | BurnDecodeError::UnsupportedCodecContract => false,
        #[cfg(feature = "metal")]
        BurnDecodeError::Metal(source) => source.session_is_unusable(),
    }
}
