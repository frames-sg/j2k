// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use burn_core::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};
use j2k::{BatchDecodeOptions, EncodedImage, PreparedBatch, PreparedImage};
use j2k_metal::MetalBatchDecoder as CodecDecoder;
#[cfg(target_os = "macos")]
use j2k_metal::{MetalBatchDecodeResult, SubmittedMetalPreparedBatch};

#[cfg(target_os = "macos")]
use crate::BurnBatchGroupError;
use crate::{BurnBatchDecode, BurnDecodeError};

/// Pending Metal codec decode whose completed pixels will be staged through
/// host memory and uploaded with Burn's ordinary tensor API.
#[cfg(target_os = "macos")]
#[must_use = "submitted Metal upload batches must be waited or dropped"]
pub struct SubmittedMetalUploadBurnBatch {
    pending: SubmittedMetalPreparedBatch,
    device: WgpuDevice,
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for SubmittedMetalUploadBurnBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubmittedMetalUploadBurnBatch")
            .field("pending", &self.pending)
            .field("device", &self.device)
            .finish()
    }
}

#[cfg(target_os = "macos")]
impl SubmittedMetalUploadBurnBatch {
    /// Number of successfully submitted homogeneous codec groups.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether no accelerator codec work was submitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Wait for Metal codec completion, copy each completed group to host
    /// staging, and upload it into an ordinary Burn wgpu tensor.
    pub fn wait(self) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
        materialize(self.pending.wait()?, &self.device)
    }
}

/// Uninhabited pending Metal upload returned only for cross-platform API compatibility.
#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
pub struct SubmittedMetalUploadBurnBatch {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl SubmittedMetalUploadBurnBatch {
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

/// Persistent Metal codec session followed by an explicit staged Burn upload.
///
/// JPEG 2000 decoding executes on Metal. Completed codec-owned device output is
/// copied to host staging and then uploaded through [`burn_core::tensor::Tensor::from_data`].
/// This type does not provide direct-destination or zero-copy behavior.
pub struct MetalUploadBurnDecoder {
    codec: CodecDecoder,
    device: WgpuDevice,
}

impl core::fmt::Debug for MetalUploadBurnDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalUploadBurnDecoder")
            .field("codec", &self.codec)
            .field("device", &self.device)
            .field("options", &self.codec.options())
            .finish_non_exhaustive()
    }
}

impl MetalUploadBurnDecoder {
    /// Create a Metal codec session and target Burn's default wgpu device.
    #[cfg(target_os = "macos")]
    pub fn system_default(options: BatchDecodeOptions) -> Result<Self, BurnDecodeError> {
        Ok(Self {
            codec: CodecDecoder::system_default_with_options(options)?,
            device: WgpuDevice::DefaultDevice,
        })
    }

    /// Return a typed unavailable error on hosts without Metal.
    #[cfg(not(target_os = "macos"))]
    pub fn system_default(_options: BatchDecodeOptions) -> Result<Self, BurnDecodeError> {
        Err(unavailable())
    }

    /// Borrow the persistent Metal codec session.
    #[must_use]
    pub const fn codec(&self) -> &CodecDecoder {
        &self.codec
    }

    /// Borrow the Burn wgpu device receiving the staged upload.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
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

    /// Decode on Metal, stage completed pixels through host memory, and upload to Burn.
    pub fn decode(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
        self.submit(inputs)?.wait()
    }

    /// Regroup prepared images, decode on Metal, and perform the staged upload.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a reusable preparation on Metal and perform the staged upload.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
        self.submit_prepared(prepared)?.wait()
    }

    /// Submit Metal codec work. The later [`SubmittedMetalUploadBurnBatch::wait`]
    /// performs a synchronous device-to-host copy and ordinary Burn upload.
    #[cfg(target_os = "macos")]
    pub fn submit(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<SubmittedMetalUploadBurnBatch, BurnDecodeError> {
        let prepared = self.prepare(inputs)?;
        self.submit_prepared(&prepared)
    }

    /// Return a typed unavailable error on hosts without Metal.
    #[cfg(not(target_os = "macos"))]
    pub fn submit(
        &mut self,
        _inputs: Vec<EncodedImage>,
    ) -> Result<SubmittedMetalUploadBurnBatch, BurnDecodeError> {
        Err(unavailable())
    }

    /// Submit a reusable preparation to codec-owned Metal output.
    #[cfg(target_os = "macos")]
    pub fn submit_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedMetalUploadBurnBatch, BurnDecodeError> {
        ensure_dtypes::<Wgpu>(prepared, &self.device)?;
        Ok(SubmittedMetalUploadBurnBatch {
            pending: self.codec.submit_prepared(prepared)?,
            device: self.device.clone(),
        })
    }

    /// Return a typed unavailable error on hosts without Metal.
    #[cfg(not(target_os = "macos"))]
    pub fn submit_prepared(
        &mut self,
        _prepared: &PreparedBatch,
    ) -> Result<SubmittedMetalUploadBurnBatch, BurnDecodeError> {
        Err(unavailable())
    }
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn materialize(
    decoded: MetalBatchDecodeResult,
    device: &WgpuDevice,
) -> Result<BurnBatchDecode<Wgpu>, BurnDecodeError> {
    let (codec_groups, errors, codec_group_errors) = decoded.into_parts();
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(codec_groups.len())
        .map_err(|_| BurnDecodeError::SizeOverflow)?;
    for group in codec_groups {
        let resident = group
            .resident_batch()
            .ok_or(BurnDecodeError::UnsupportedCodecContract)?;
        // SAFETY: codec completion was established before this immutable read,
        // and `checked_buffer_read_vec` copies the validated resident range.
        let bytes = unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(
                resident.metal_buffer(),
                resident.byte_offset(),
                resident.byte_len(),
            )
        }
        .map_err(|source| BurnDecodeError::AcceleratorInterop {
            backend: "Metal staged readback",
            message: source.to_string(),
        })?;
        let (info, source_indices, decoded_rects, warnings, _surfaces) = group.into_parts();
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
            BurnBatchGroupError::new(source_indices, BurnDecodeError::Metal(source))
        })
        .collect();
    Ok(BurnBatchDecode {
        groups,
        errors,
        group_errors,
    })
}

#[cfg(not(target_os = "macos"))]
fn unavailable() -> BurnDecodeError {
    BurnDecodeError::AcceleratorInterop {
        backend: "Metal",
        message: "Metal is unavailable on this platform".to_string(),
    }
}
