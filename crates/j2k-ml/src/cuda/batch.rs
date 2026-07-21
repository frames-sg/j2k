// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::backend::Backend;
use burn_cuda::{Cuda, CudaDevice};
use j2k::{
    BatchDecodeOptions, EncodedImage, IndexedBatchError, NativeSampleType, PreparedBatch,
    PreparedBatchGroup, PreparedImage,
};
use j2k_cuda::{CudaBatchDecoder as CodecDecoder, SubmittedCudaExternalBatch};

use super::interop::{fill_batch_int_tensor, register_int_tensor, SubmittedBatchIntTensor};
use crate::batch_contract::{dtype, tensor_shape};
use crate::{
    BurnBatchDecode, BurnBatchGroup, BurnBatchGroupError, BurnBatchTensor, BurnDecodeError,
};

/// Pending CUDA decode whose output allocation is not exposed to Burn until
/// group-level codec status validation succeeds.
#[must_use = "submitted CUDA Burn batches must be waited or dropped"]
pub struct SubmittedCudaBurnBatch {
    groups: Vec<SubmittedCudaBurnGroup>,
    errors: Vec<IndexedBatchError>,
    group_errors: Vec<BurnBatchGroupError>,
}

impl core::fmt::Debug for SubmittedCudaBurnBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubmittedCudaBurnBatch")
            .field("groups", &self.groups.len())
            .field("errors", &self.errors)
            .field("group_errors", &self.group_errors)
            .finish_non_exhaustive()
    }
}

impl SubmittedCudaBurnBatch {
    /// Number of asynchronously submitted homogeneous groups.
    #[must_use]
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Whether no accelerator work was submitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Wait once per group, validate its batched status, and only then expose
    /// ordinary rank-4 Burn tensors.
    pub fn wait(self) -> Result<BurnBatchDecode<Cuda>, BurnDecodeError> {
        let Self {
            groups,
            errors,
            group_errors,
        } = self;
        let (groups, group_errors) =
            crate::completion::finish_submitted_groups(groups, group_errors, |group| {
                let source_indices = group.source_indices.clone();
                (source_indices, group.wait())
            })?;
        Ok(BurnBatchDecode {
            groups,
            errors,
            group_errors,
        })
    }
}

struct SubmittedCudaBurnGroup {
    sample_type: NativeSampleType,
    source_indices: Vec<usize>,
    allocation: SubmittedBatchIntTensor<SubmittedCudaExternalBatch, 4>,
}

impl SubmittedCudaBurnGroup {
    fn wait(self) -> Result<BurnBatchGroup<Cuda>, BurnDecodeError> {
        let sample_type = self.sample_type;
        let (cube, shape, dtype, device, pending) = self.allocation.into_parts();
        let decoded = match pending.wait() {
            Ok(decoded) => decoded,
            Err(source) => {
                if source.completion_is_uncertain() {
                    std::mem::forget(cube);
                }
                return Err(BurnDecodeError::Cuda(source));
            }
        };
        let tensor = register_int_tensor(cube, shape, dtype, &device);
        let tensor = match sample_type {
            NativeSampleType::U8 => BurnBatchTensor::U8(tensor),
            NativeSampleType::U16 => BurnBatchTensor::U16(tensor),
            NativeSampleType::I16 => BurnBatchTensor::I16(tensor),
            _ => return Err(BurnDecodeError::UnsupportedCodecContract),
        };
        Ok(BurnBatchGroup {
            tensor,
            info: decoded.info().clone(),
            source_indices: decoded.source_indices().to_vec(),
            decoded_rects: decoded.decoded_rects().to_vec(),
            warnings: decoded.warnings().to_vec(),
        })
    }
}

/// Persistent CUDA codec session writing directly into Burn-owned allocations.
#[derive(Debug)]
pub struct CudaBurnDecoder {
    codec: CodecDecoder,
    device: CudaDevice,
}

impl CudaBurnDecoder {
    /// Create a decoder for one Burn CUDA device. The retained primary context
    /// is initialized lazily when the first nonempty valid group is submitted.
    #[must_use]
    pub fn new(device: CudaDevice, options: BatchDecodeOptions) -> Self {
        Self {
            codec: CodecDecoder::with_options(options),
            device,
        }
    }

    /// Borrow the retained codec session.
    #[must_use]
    pub const fn codec(&self) -> &CodecDecoder {
        &self.codec
    }

    /// Burn CUDA device receiving decoded tensors.
    #[must_use]
    pub const fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Parse and group owned inputs once for repeated decode calls.
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

    /// Prepare and synchronously finish one owned batch. Use [`Self::submit`]
    /// to overlap caller work with GPU execution.
    pub fn decode(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<BurnBatchDecode<Cuda>, BurnDecodeError> {
        self.submit(inputs)?.wait()
    }

    /// Regroup and synchronously decode caller-supplied prepared images.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<BurnBatchDecode<Cuda>, BurnDecodeError> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Finish a reusable codec preparation and return validated Burn tensors.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<BurnBatchDecode<Cuda>, BurnDecodeError> {
        self.submit_prepared(prepared)?.wait()
    }

    /// Prepare and asynchronously submit one owned batch without exposing
    /// partially written tensor storage.
    pub fn submit(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<SubmittedCudaBurnBatch, BurnDecodeError> {
        let prepared = self.prepare(inputs)?;
        self.submit_prepared(&prepared)
    }

    /// Asynchronously submit a reusable preparation. The returned guard owns
    /// every fresh Burn allocation and codec completion resource until
    /// [`SubmittedCudaBurnBatch::wait`] validates the whole group.
    pub fn submit_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedCudaBurnBatch, BurnDecodeError> {
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(prepared.groups().len())
            .map_err(|_| BurnDecodeError::SizeOverflow)?;
        let mut group_errors = Vec::new();
        group_errors
            .try_reserve_exact(prepared.groups().len())
            .map_err(|_| BurnDecodeError::SizeOverflow)?;
        for group in prepared.groups() {
            match self.submit_group(group) {
                Ok(submitted) => groups.push(submitted),
                Err(source) if crate::completion::burn_group_error_is_fatal(&source) => {
                    return Err(source);
                }
                Err(source) => group_errors.push(BurnBatchGroupError::new(
                    group.source_indices().to_vec(),
                    source,
                )),
            }
        }
        Ok(SubmittedCudaBurnBatch {
            groups,
            errors: prepared.errors().to_vec(),
            group_errors,
        })
    }

    fn submit_group(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<SubmittedCudaBurnGroup, BurnDecodeError> {
        let sample_type = group.info().sample_type;
        let dtype = dtype(sample_type)?;
        if !<Cuda<f32, i32> as Backend>::supports_dtype(&self.device, dtype) {
            return Err(BurnDecodeError::UnsupportedDType { dtype });
        }
        let shape = tensor_shape(group.source_indices().len(), group.info())?;
        let device = self.device.clone();
        let context = self
            .codec
            .session_mut()
            .context_for_device_interop(self.device.index)
            .map_err(|error| BurnDecodeError::AcceleratorInterop {
                backend: "CUDA",
                message: error.to_string(),
            })?;
        let allocation = fill_batch_int_tensor(shape, dtype, &device, &context, |destination| {
            // SAFETY: the interop owner keeps the unique Burn allocation live,
            // establishes CubeCL-to-codec-to-CubeCL event ordering, and stores
            // the returned completion guard ahead of that allocation.
            Ok(unsafe { self.codec.submit_batch_into(group, destination) }?)
        })?;
        if allocation.payload().group().info().sample_type != sample_type {
            return Err(BurnDecodeError::SampleTypeMismatch);
        }
        Ok(SubmittedCudaBurnGroup {
            sample_type,
            source_indices: group.source_indices().to_vec(),
            allocation,
        })
    }
}
