// SPDX-License-Identifier: MIT OR Apache-2.0

use core::marker::PhantomData;

use burn_core::tensor::{backend::Backend, DType, Tensor, TensorData};
use j2k::{
    BatchDecodeOptions, CpuBatchDecodeResult, CpuBatchSamples, EncodedImage, NativeSampleType,
    PreparedBatch, PreparedImage,
};

use crate::batch_contract::{dtype, tensor_shape};
use crate::{BurnBatchDecode, BurnBatchGroup, BurnBatchTensor, BurnDecodeError};

/// Persistent CPU codec session followed by one Burn materialization per output group.
#[derive(Debug)]
pub struct CpuBurnDecoder<B: Backend> {
    codec: j2k::CpuBatchDecoder,
    device: B::Device,
    backend: PhantomData<B>,
}

impl<B: Backend> CpuBurnDecoder<B> {
    /// Construct a CPU codec session for one Burn device.
    #[must_use]
    pub fn new(device: B::Device, options: BatchDecodeOptions) -> Self {
        Self {
            codec: j2k::CpuBatchDecoder::new(options),
            device,
            backend: PhantomData,
        }
    }

    /// Borrow the retained codec session.
    #[must_use]
    pub const fn codec(&self) -> &j2k::CpuBatchDecoder {
        &self.codec
    }

    /// Borrow the Burn device receiving decoded tensors.
    #[must_use]
    pub const fn device(&self) -> &B::Device {
        &self.device
    }

    /// Parse and group owned inputs once for repeated training epochs.
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

    /// Prepare, decode, and materialize one owned batch.
    pub fn decode(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<BurnBatchDecode<B>, BurnDecodeError> {
        let prepared = self.prepare(inputs)?;
        self.decode_prepared(&prepared)
    }

    /// Regroup, decode, and materialize caller-supplied prepared images.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<BurnBatchDecode<B>, BurnDecodeError> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a reusable prepared batch into ordinary rank-4 Burn integer tensors.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<BurnBatchDecode<B>, BurnDecodeError> {
        self.ensure_prepared_dtypes(prepared)?;
        let decoded = self.codec.decode_prepared(prepared)?;
        materialize::<B>(decoded, &self.device)
    }

    fn ensure_prepared_dtypes(&self, prepared: &PreparedBatch) -> Result<(), BurnDecodeError> {
        for group in prepared.groups() {
            let dtype = dtype(group.info().sample_type)?;
            if !B::supports_dtype(&self.device, dtype) {
                return Err(BurnDecodeError::UnsupportedDType { dtype });
            }
        }
        Ok(())
    }
}

fn materialize<B: Backend>(
    decoded: CpuBatchDecodeResult,
    device: &B::Device,
) -> Result<BurnBatchDecode<B>, BurnDecodeError> {
    let (codec_groups, errors) = decoded.into_parts();
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(codec_groups.len())
        .map_err(|_| BurnDecodeError::SizeOverflow)?;

    for group in codec_groups {
        let (info, source_indices, decoded_rects, warnings, samples) = group.into_parts();
        let shape = tensor_shape(source_indices.len(), &info)?;
        let tensor = match (info.sample_type, samples) {
            (NativeSampleType::U8, CpuBatchSamples::U8(samples)) => BurnBatchTensor::U8(
                Tensor::from_data(TensorData::new(samples, shape), (device, DType::U8)),
            ),
            (NativeSampleType::U16, CpuBatchSamples::U16(samples)) => BurnBatchTensor::U16(
                Tensor::from_data(TensorData::new(samples, shape), (device, DType::U16)),
            ),
            (NativeSampleType::I16, CpuBatchSamples::I16(samples)) => BurnBatchTensor::I16(
                Tensor::from_data(TensorData::new(samples, shape), (device, DType::I16)),
            ),
            _ => return Err(BurnDecodeError::SampleTypeMismatch),
        };
        groups.push(BurnBatchGroup {
            tensor,
            info,
            source_indices,
            decoded_rects,
            warnings,
        });
    }

    Ok(BurnBatchDecode {
        groups,
        errors,
        group_errors: Vec::new(),
    })
}
