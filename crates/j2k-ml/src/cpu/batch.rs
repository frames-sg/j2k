// SPDX-License-Identifier: MIT OR Apache-2.0

use core::marker::PhantomData;

use burn_core::tensor::{backend::Backend, DType, Tensor, TensorData};
use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecodeResult, CpuBatchSamples, EncodedImage,
    NativeSampleType, PreparedBatch, PreparedImage,
};
use j2k_jpeg::{JpegBatchDecodeOptions, JpegBatchDecoder};

use crate::batch_contract::{dtype, tensor_shape};
use crate::{BurnBatchDecode, BurnBatchGroup, BurnBatchTensor, BurnDecodeError};

/// Persistent CPU codec session followed by one Burn materialization per output group.
#[derive(Debug)]
pub struct CpuBurnDecoder<B: Backend> {
    codec: j2k::CpuBatchDecoder,
    jpeg: JpegBatchDecoder,
    device: B::Device,
    backend: PhantomData<B>,
}

impl<B: Backend> CpuBurnDecoder<B> {
    /// Construct a CPU codec session for one Burn device.
    #[must_use]
    pub fn new(device: B::Device, options: BatchDecodeOptions) -> Self {
        Self {
            codec: j2k::CpuBatchDecoder::new(options),
            jpeg: JpegBatchDecoder::new({
                let mut jpeg = JpegBatchDecodeOptions::default();
                jpeg.layout = options.layout;
                jpeg.workers = options.workers;
                jpeg
            }),
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

    /// Decode mixed JPEG, JPEG 2000 and HTJ2K inputs using one result model.
    ///
    /// Session strictness is authoritative for both codec families. JPEG color
    /// hints are preserved. JPEG output is RGB8 with grayscale expanded; native
    /// grayscale JPEG 2000 and HTJ2K output stays one channel. Group order is
    /// unspecified; use `source_indices` to align tensor rows with inputs.
    pub fn decode_batch(
        &mut self,
        inputs: &[crate::ImageInput<'_>],
    ) -> Result<BurnBatchDecode<B>, BurnDecodeError> {
        let mut budget = j2k_core::BatchAllocationBudget::new("Burn mixed input metadata");
        let mut jpeg = budget.try_vec(inputs.len(), "Burn JPEG inputs")?;
        let mut jpeg_indices = budget.try_vec(inputs.len(), "Burn JPEG indices")?;
        let mut j2k = budget.try_vec(inputs.len(), "Burn JPEG 2000 inputs")?;
        let mut j2k_indices = budget.try_vec(inputs.len(), "Burn JPEG 2000 indices")?;
        for (index, input) in inputs.iter().enumerate() {
            match input {
                crate::ImageInput::Jpeg(input) => {
                    jpeg.push(
                        input.with_options(
                            input
                                .options()
                                .with_strict(self.codec.options().settings.is_strict()),
                        ),
                    );
                    jpeg_indices.push(index);
                }
                crate::ImageInput::Jpeg2000(input) => {
                    j2k.push((*input).clone());
                    j2k_indices.push(index);
                }
            }
        }
        let mut result = self.decode(j2k)?;
        for group in &mut result.groups {
            for index in &mut group.source_indices {
                *index = j2k_indices[*index];
            }
        }
        for error in &mut result.errors {
            error.index = j2k_indices[error.index];
        }
        if !jpeg.is_empty() && !B::supports_dtype(&self.device, DType::U8) {
            return Err(BurnDecodeError::UnsupportedDType { dtype: DType::U8 });
        }
        let (groups, errors) = self.jpeg.decode(&jpeg)?.into_parts();
        result
            .groups
            .try_reserve_exact(groups.len())
            .map_err(|_| BurnDecodeError::SizeOverflow)?;
        result
            .errors
            .try_reserve_exact(errors.len())
            .map_err(|_| BurnDecodeError::SizeOverflow)?;
        for group in groups {
            let (dimensions, layout) = (group.dimensions(), group.layout());
            let (mut source_indices, warnings, recovered, samples) = group.into_parts();
            for index in &mut source_indices {
                *index = jpeg_indices[*index];
            }
            let shape = match layout {
                BatchLayout::Nchw => [
                    source_indices.len(),
                    3,
                    dimensions.1 as usize,
                    dimensions.0 as usize,
                ],
                BatchLayout::Nhwc => [
                    source_indices.len(),
                    dimensions.1 as usize,
                    dimensions.0 as usize,
                    3,
                ],
                _ => return Err(BurnDecodeError::UnsupportedCodecContract),
            };
            let mut row_budget = j2k_core::BatchAllocationBudget::new("Burn JPEG result metadata");
            let mut rects = row_budget.try_vec(source_indices.len(), "Burn JPEG rectangles")?;
            rects.resize(source_indices.len(), j2k::Rect::full(dimensions));
            result.groups.push(BurnBatchGroup {
                tensor: BurnBatchTensor::U8(Tensor::from_data(
                    TensorData::new(samples, shape),
                    (&self.device, DType::U8),
                )),
                info: crate::GroupInfo::Jpeg { dimensions, layout },
                source_indices,
                decoded_rects: rects,
                warnings: crate::tagged_warnings(warnings, crate::ImageDecodeWarning::Jpeg)?,
                recovered,
            });
        }
        result
            .errors
            .extend(errors.into_iter().map(|error| j2k_core::TileBatchError {
                index: jpeg_indices[error.index],
                source: crate::ImageDecodeError::Jpeg(error.source),
            }));
        result.errors.sort_unstable_by_key(|error| error.index);
        Ok(result)
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
            info: crate::GroupInfo::Jpeg2000(info),
            recovered: crate::unrecovered(source_indices.len())?,
            source_indices,
            decoded_rects,
            warnings: crate::tagged_warnings(warnings, crate::ImageDecodeWarning::Jpeg2000)?,
        });
    }

    Ok(BurnBatchDecode {
        groups,
        errors: crate::j2k_errors(errors)?,
        group_errors: Vec::new(),
    })
}
