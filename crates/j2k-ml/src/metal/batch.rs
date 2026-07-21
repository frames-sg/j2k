// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};
use j2k::{
    BatchDecodeOptions, BatchGroupInfo, EncodedImage, IndexedBatchError, NativeSampleType,
    PreparedBatch, PreparedBatchGroup, PreparedImage,
};
use j2k_metal::MetalBatchDecoder as CodecDecoder;

use crate::batch_contract::{dtype, tensor_shape};
use crate::{
    BurnBatchDecode, BurnBatchGroup, BurnBatchGroupError, BurnBatchTensor, BurnDecodeError,
};

#[cfg(target_os = "macos")]
use super::interop::{
    fill_batch_int_tensor, paired_metal_runtime, register_int_tensor, SubmittedBatchIntTensor,
};
#[cfg(target_os = "macos")]
use j2k_metal::SubmittedMetalGroupDecodeInto;

/// Pending Metal decode whose registered Burn tensors remain private until
/// group-level command and codec-status validation succeeds.
#[cfg(target_os = "macos")]
#[must_use = "submitted Metal Burn batches must be waited or dropped"]
pub struct SubmittedMetalBurnBatch {
    groups: Vec<SubmittedMetalBurnGroup>,
    errors: Vec<IndexedBatchError>,
    group_errors: Vec<BurnBatchGroupError>,
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for SubmittedMetalBurnBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubmittedMetalBurnBatch")
            .field("groups", &self.groups.len())
            .field("errors", &self.errors)
            .field("group_errors", &self.group_errors)
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "macos")]
impl SubmittedMetalBurnBatch {
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
    pub fn wait(self) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
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

#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
pub struct SubmittedMetalBurnBatch {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl SubmittedMetalBurnBatch {
    /// No Metal groups can be submitted on this host.
    #[must_use]
    pub const fn len(&self) -> usize {
        0
    }

    /// No Metal groups can be submitted on this host.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        true
    }

    /// Return the typed Metal-unavailable error.
    pub fn wait(self) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
        Err(unavailable())
    }
}

#[cfg(target_os = "macos")]
struct SubmittedMetalBurnGroup {
    sample_type: NativeSampleType,
    info: BatchGroupInfo,
    source_indices: Vec<usize>,
    allocation: SubmittedBatchIntTensor<SubmittedMetalGroupDecodeInto, 4>,
}

#[cfg(target_os = "macos")]
impl SubmittedMetalBurnGroup {
    fn wait(self) -> Result<BurnBatchGroup<Wgpu>, BurnDecodeError> {
        let Self {
            sample_type,
            info,
            source_indices,
            allocation,
        } = self;
        let (cube, shape, dtype, device, pending) = allocation.into_parts();
        let (decoded_rects, warnings) = pending.wait()?.into_parts();
        let tensor = register_int_tensor(cube, shape, dtype, &device);
        let tensor = match sample_type {
            NativeSampleType::U8 => BurnBatchTensor::U8(tensor),
            NativeSampleType::U16 => BurnBatchTensor::U16(tensor),
            NativeSampleType::I16 => BurnBatchTensor::I16(tensor),
            _ => return Err(BurnDecodeError::UnsupportedCodecContract),
        };
        Ok(BurnBatchGroup {
            tensor,
            info,
            source_indices,
            decoded_rects,
            warnings,
        })
    }
}

/// Persistent Metal codec session writing directly into Burn-owned allocations.
pub struct MetalBurnDecoder {
    codec: CodecDecoder,
    device: WgpuDevice,
    #[cfg(target_os = "macos")]
    consumer_queue: metal::CommandQueue,
}

impl core::fmt::Debug for MetalBurnDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalBurnDecoder")
            .field("codec", &self.codec)
            .field("device", &self.device)
            .field("options", &self.codec.options())
            .finish_non_exhaustive()
    }
}

impl MetalBurnDecoder {
    /// Create paired J2K and Burn sessions from the exact same Metal device and
    /// retain Burn's exact consumer command queue for implicit queue ordering.
    #[cfg(target_os = "macos")]
    pub fn system_default(options: BatchDecodeOptions) -> Result<Self, BurnDecodeError> {
        let (backend, device, consumer_queue) = paired_metal_runtime()?;
        Ok(Self {
            codec: CodecDecoder::with_backend_session_and_options(backend, options),
            device,
            consumer_queue,
        })
    }

    /// Return a typed unavailable error on hosts without Metal.
    #[cfg(not(target_os = "macos"))]
    pub fn system_default(_options: BatchDecodeOptions) -> Result<Self, BurnDecodeError> {
        Err(unavailable())
    }

    /// Borrow the retained codec session.
    #[must_use]
    pub const fn codec(&self) -> &CodecDecoder {
        &self.codec
    }

    /// Burn wgpu device paired with the codec's underlying Metal device.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Parse and group owned inputs once for repeated decode calls.
    pub fn prepare(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, BurnDecodeError> {
        Ok(self.codec.prepare(inputs)?)
    }

    /// Regroup caller-supplied prepared images without reparsing encoded bytes.
    ///
    /// Source indices in the returned batch are positions in `images`; each
    /// [`PreparedImage::source_index`] remains its original preparation index.
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
    ) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
        self.submit(inputs)?.wait()
    }

    /// Finish a reusable codec preparation and return validated Burn tensors.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
        self.submit_prepared(prepared)?.wait()
    }

    /// Regroup, decode, and materialize caller-supplied prepared images.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Prepare and asynchronously submit one owned batch without exposing
    /// partially written tensor storage.
    #[cfg(target_os = "macos")]
    pub fn submit(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<SubmittedMetalBurnBatch, BurnDecodeError> {
        let prepared = self.prepare(inputs)?;
        self.submit_prepared(&prepared)
    }

    /// Return a typed unavailable error on hosts without Metal.
    #[cfg(not(target_os = "macos"))]
    pub fn submit(
        &mut self,
        _inputs: Vec<EncodedImage>,
    ) -> Result<SubmittedMetalBurnBatch, BurnDecodeError> {
        Err(unavailable())
    }

    /// Asynchronously submit a reusable preparation. Same-queue ordering is
    /// implicit; a GPU event dependency is registered only when queues differ.
    #[cfg(target_os = "macos")]
    pub fn submit_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedMetalBurnBatch, BurnDecodeError> {
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
        Ok(SubmittedMetalBurnBatch {
            groups,
            errors: prepared.errors().to_vec(),
            group_errors,
        })
    }

    /// Return a typed unavailable error on hosts without Metal.
    #[cfg(not(target_os = "macos"))]
    pub fn submit_prepared(
        &mut self,
        _prepared: &PreparedBatch,
    ) -> Result<SubmittedMetalBurnBatch, BurnDecodeError> {
        Err(unavailable())
    }

    #[cfg(target_os = "macos")]
    fn submit_group(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<SubmittedMetalBurnGroup, BurnDecodeError> {
        let sample_type = group.info().sample_type;
        let dtype = dtype(sample_type)?;
        if !<Wgpu<f32, i32, u32> as Backend>::supports_dtype(&self.device, dtype) {
            return Err(BurnDecodeError::UnsupportedDType { dtype });
        }
        let shape = tensor_shape(group.source_indices().len(), group.info())?;
        let device = self.device.clone();
        let consumer_queue = self.consumer_queue.clone();
        let allocation =
            fill_batch_int_tensor(shape, dtype, group.info(), &device, |destination| {
                Ok(self.codec.submit_prepared_group_into_for_consumer_queue(
                    group,
                    destination,
                    &consumer_queue,
                )?)
            })?;
        Ok(SubmittedMetalBurnGroup {
            sample_type,
            info: group.info().clone(),
            source_indices: group.source_indices().to_vec(),
            allocation,
        })
    }
}

#[cfg(not(target_os = "macos"))]
fn unavailable() -> BurnDecodeError {
    BurnDecodeError::AcceleratorInterop {
        backend: "Metal",
        message: "Metal is unavailable on this platform".to_string(),
    }
}
