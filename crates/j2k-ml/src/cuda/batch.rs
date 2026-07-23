// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::backend::Backend;
use burn_cuda::{Cuda, CudaDevice};
use j2k::{BatchDecodeOptions, EncodedImage, PreparedBatch, PreparedImage};
use j2k_cuda::{
    CudaBatchDecodeResult, CudaBatchDecoder as CodecDecoder, SubmittedCudaResidentBatch,
};

use crate::{BurnBatchDecode, BurnBatchGroupError, BurnDecodeError};

/// Pending CUDA codec decode whose completed pixels will be staged through
/// host memory and uploaded with Burn's ordinary tensor API.
#[must_use = "submitted CUDA upload batches must be waited or dropped"]
pub struct SubmittedCudaUploadBurnBatch {
    pending: SubmittedCudaResidentBatch,
    device: CudaDevice,
}

impl core::fmt::Debug for SubmittedCudaUploadBurnBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubmittedCudaUploadBurnBatch")
            .field("pending", &self.pending)
            .field("device", &self.device)
            .finish()
    }
}

impl SubmittedCudaUploadBurnBatch {
    /// Number of successfully submitted homogeneous codec groups.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.pending_group_count()
    }

    /// Whether no accelerator codec work was submitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wait for CUDA codec completion, copy each completed group to host
    /// staging, and upload it into an ordinary Burn CUDA tensor.
    pub fn wait(self) -> Result<BurnBatchDecode<Cuda>, BurnDecodeError> {
        materialize(self.pending.wait()?, &self.device)
    }
}

/// Persistent CUDA codec session followed by an explicit staged Burn upload.
///
/// JPEG 2000 decoding executes on CUDA. Completed codec-owned device output is
/// copied to host staging and then uploaded through [`burn_core::tensor::Tensor::from_data`].
/// This type does not provide direct-destination or zero-copy behavior.
#[derive(Debug)]
pub struct CudaUploadBurnDecoder {
    codec: CodecDecoder,
    device: CudaDevice,
}

impl CudaUploadBurnDecoder {
    /// Create a staged decoder for one Burn CUDA device.
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

    /// Borrow the Burn CUDA device receiving the staged upload.
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

    /// Decode on CUDA, stage completed pixels through host memory, and upload to Burn.
    pub fn decode(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<BurnBatchDecode<Cuda>, BurnDecodeError> {
        self.submit(inputs)?.wait()
    }

    /// Regroup prepared images, decode on CUDA, and perform the staged upload.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<BurnBatchDecode<Cuda>, BurnDecodeError> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a reusable preparation on CUDA and perform the staged upload.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<BurnBatchDecode<Cuda>, BurnDecodeError> {
        self.submit_prepared(prepared)?.wait()
    }

    /// Submit CUDA codec work. The later [`SubmittedCudaUploadBurnBatch::wait`]
    /// performs a synchronous device-to-host copy and ordinary Burn upload.
    pub fn submit(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<SubmittedCudaUploadBurnBatch, BurnDecodeError> {
        let prepared = self.prepare(inputs)?;
        self.submit_prepared(&prepared)
    }

    /// Submit a reusable preparation to codec-owned CUDA output.
    pub fn submit_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedCudaUploadBurnBatch, BurnDecodeError> {
        ensure_dtypes::<Cuda>(prepared, &self.device)?;
        Ok(SubmittedCudaUploadBurnBatch {
            pending: self.codec.submit_prepared(prepared)?,
            device: self.device.clone(),
        })
    }
}

fn ensure_dtypes<B: Backend>(
    prepared: &PreparedBatch,
    device: &B::Device,
) -> Result<(), BurnDecodeError> {
    for group in prepared.groups() {
        let dtype = crate::batch_contract::dtype(group.info().sample_type)?;
        if !B::supports_dtype(device, dtype) {
            return Err(BurnDecodeError::UnsupportedDType { dtype });
        }
    }
    Ok(())
}

fn materialize(
    decoded: CudaBatchDecodeResult,
    device: &CudaDevice,
) -> Result<BurnBatchDecode<Cuda>, BurnDecodeError> {
    let (codec_groups, errors, codec_group_errors) = decoded.into_parts();
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(codec_groups.len())
        .map_err(|_| BurnDecodeError::SizeOverflow)?;
    for group in codec_groups {
        let (info, source_indices, decoded_rects, warnings, _surfaces, dense) = group.into_parts();
        let mut bytes = vec![0; crate::staging::byte_len(source_indices.len(), &info)?];
        dense.buffer().copy_to_host(&mut bytes)?;
        groups.push(crate::staging::materialize(
            info,
            source_indices,
            decoded_rects,
            warnings,
            bytes,
            device,
        )?);
    }
    let group_errors = codec_group_errors
        .into_iter()
        .map(|error| {
            let (source_indices, source) = error.into_parts();
            BurnBatchGroupError::new(source_indices, BurnDecodeError::CudaCodec(source))
        })
        .collect();
    Ok(BurnBatchDecode {
        groups,
        errors,
        group_errors,
    })
}
