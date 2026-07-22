// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistent CUDA batch decoder facade.

#[cfg(feature = "cuda-runtime")]
use super::{
    decode_warnings, group_pixel_format, native_color_inputs, native_decode_settings,
    native_referenced_classic_plan, native_referenced_htj2k_plan, validate_layout,
    CudaExternalBatchGroup, PixelFormat, PreparedBatchGroup, SubmittedCudaCodecBatch,
    SubmittedCudaExternalBatch,
};
use super::{
    prepare_batch, prepare_batch_from_images, BatchDecodeOptions, BatchDecoder,
    BatchInfrastructureError, CudaBatchDecodeResult, CudaBatchError, CudaSession, EncodedImage,
    Error, PreparedBatch, PreparedImage,
};

/// Persistent CUDA batch decoder that reuses one [`CudaSession`].
#[derive(Clone, Debug, Default)]
pub struct CudaBatchDecoder {
    pub(super) session: CudaSession,
    pub(super) options: BatchDecodeOptions,
}

impl CudaBatchDecoder {
    /// Create a decoder with a lazily initialized CUDA session and strict
    /// shared batch options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a decoder with explicit shared preparation options.
    #[must_use]
    pub fn with_options(options: BatchDecodeOptions) -> Self {
        Self {
            session: CudaSession::default(),
            options,
        }
    }

    /// Create a decoder around an existing CUDA session.
    #[must_use]
    pub fn with_session(session: CudaSession) -> Self {
        Self {
            session,
            options: BatchDecodeOptions::default(),
        }
    }

    /// Create a decoder around an existing session and preparation policy.
    #[must_use]
    pub const fn with_session_and_options(
        session: CudaSession,
        options: BatchDecodeOptions,
    ) -> Self {
        Self { session, options }
    }

    /// Shared preparation options used by [`Self::prepare`] and
    /// [`Self::decode_batch`].
    #[must_use]
    pub const fn options(&self) -> BatchDecodeOptions {
        self.options
    }

    /// Borrow the persistent session for diagnostics.
    #[must_use]
    pub const fn session(&self) -> &CudaSession {
        &self.session
    }

    /// Snapshot the persistent session's two private decode buffer pools.
    #[cfg(feature = "cuda-runtime")]
    pub fn decode_pool_diagnostics(&self) -> Result<crate::CudaDecodePoolDiagnostics, Error> {
        self.session.decode_pool_diagnostics()
    }

    /// Snapshot CUDA transfer/event counters and retained decode-pool memory.
    #[cfg(feature = "cuda-runtime")]
    pub fn diagnostics(&self) -> Result<crate::CudaSessionDiagnostics, Error> {
        self.session.diagnostics()
    }

    /// Mutably borrow the persistent session for advanced configuration.
    #[must_use]
    pub fn session_mut(&mut self) -> &mut CudaSession {
        &mut self.session
    }

    /// Inspect and group owned inputs without copying their compressed bytes.
    pub fn prepare(
        &self,
        inputs: Vec<EncodedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        prepare_batch(inputs, self.options)
    }

    /// Regroup caller-supplied prepared images without reparsing codestream bytes.
    ///
    /// Returned source indices are positions in `images`; each image retains
    /// its original [`PreparedImage::source_index`] for provenance.
    pub fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        prepare_batch_from_images(images, self.options)
    }

    /// Prepare and strictly decode one owned batch to CUDA-resident groups.
    pub fn decode_batch(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<CudaBatchDecodeResult, CudaBatchError> {
        let prepared = self.prepare(inputs)?;
        self.decode_prepared(&prepared)
    }

    /// Regroup and decode prepared images without reparsing their encoded bytes.
    pub fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<CudaBatchDecodeResult, CudaBatchError> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Strictly decode a reusable shared codec batch to CUDA-resident groups.
    ///
    /// Explicit CUDA execution never falls back to CPU-decoded pixels. A CUDA
    /// execution error discards the entire affected dense group.
    pub fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<CudaBatchDecodeResult, CudaBatchError> {
        #[cfg(feature = "cuda-runtime")]
        {
            self.submit_prepared(prepared)?.wait()
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            if let Some(group) = prepared.groups().first() {
                return Err(CudaBatchError::group(group, Error::CudaUnavailable));
            }
            Ok(CudaBatchDecodeResult {
                groups: Vec::new(),
                errors: prepared.errors().to_vec(),
                group_errors: Vec::new(),
            })
        }
    }

    /// Decode one prepared exact-native Gray/RGB/RGBA group directly into a
    /// validated caller-owned CUDA allocation range.
    ///
    /// The destination must belong to this decoder's CUDA context and cover
    /// the tightly concatenated group output. Decoded pixels are never staged
    /// through host memory or copied through an intermediate device output.
    ///
    /// # Safety
    ///
    /// The destination allocation must remain live until this method returns
    /// success. If CUDA completion cannot be proven and an error is returned,
    /// the caller must quarantine the allocation rather than free or reuse it.
    #[cfg(feature = "cuda-runtime")]
    pub unsafe fn decode_batch_into(
        &mut self,
        group: &PreparedBatchGroup,
        destination: &mut j2k_cuda_runtime::CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<CudaExternalBatchGroup, CudaBatchError> {
        // SAFETY: this synchronous convenience immediately waits while the
        // caller's external view and exclusive managed-owner borrow are live.
        unsafe { self.submit_batch_into(group, destination) }?.wait()
    }

    /// Submit one prepared exact-native Gray/RGB/RGBA group into CUDA storage
    /// without a host completion wait.
    ///
    /// Callers integrating another CUDA runtime must use
    /// [`j2k_cuda_runtime::CudaContext::with_primary_stream_ordering`] around
    /// this submission, then retain the returned value alongside the tensor so
    /// codec-internal resources outlive the ordered final store.
    ///
    /// # Safety
    ///
    /// The destination allocation must remain live and may only be consumed on
    /// a CUDA stream ordered after codec completion until this value is waited
    /// or dropped. No unordered host or device access may overlap the decode.
    /// Stream completion alone does not validate entropy status: callers must
    /// not expose the destination as decoded pixels until
    /// [`SubmittedCudaExternalBatch::wait`] succeeds. If CUDA completion
    /// cannot be proven, the external allocation must be quarantined rather
    /// than freed or reused.
    #[cfg(feature = "cuda-runtime")]
    pub unsafe fn submit_batch_into(
        &mut self,
        group: &PreparedBatchGroup,
        destination: &mut j2k_cuda_runtime::CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<SubmittedCudaExternalBatch, CudaBatchError> {
        let fmt = group_pixel_format(group.info())
            .and_then(|fmt| {
                validate_layout(group.info())?;
                Ok(fmt)
            })
            .map_err(|source| CudaBatchError::group(group, source))?;
        let pending = if matches!(
            fmt,
            PixelFormat::Rgb8
                | PixelFormat::Rgb16
                | PixelFormat::RgbI16
                | PixelFormat::Rgba8
                | PixelFormat::Rgba16
                | PixelFormat::RgbaI16
        ) {
            let inputs = native_color_inputs(group)
                .map_err(|source| CudaBatchError::group(group, source))?;
            SubmittedCudaCodecBatch::Color(
                crate::decoder::submit_native_color_resident_prepared_batch_into(
                    &inputs,
                    &mut self.session,
                    fmt,
                    group.info().layout,
                    destination,
                )
                .map_err(|source| CudaBatchError::group(group, source))?,
            )
        } else if matches!(
            fmt,
            PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayI16
        ) {
            let inputs = group
                .images()
                .iter()
                .zip(group.source_indices().iter().copied())
                .map(|(image, source_index)| {
                    let referenced_plan = image
                        .htj2k_plan()
                        .map(native_referenced_htj2k_plan)
                        .transpose()?;
                    let referenced_classic_plan = image
                        .classic_plan()
                        .map(native_referenced_classic_plan)
                        .transpose()?;
                    Ok(crate::decoder::grayscale_batch::GrayscaleBatchInput {
                        source_index,
                        bytes: image.bytes().as_ref(),
                        device_plan: Some(image.plan()),
                        referenced_plan,
                        referenced_classic_plan,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()
                .map_err(|source| CudaBatchError::group(group, source))?;
            SubmittedCudaCodecBatch::Grayscale(
                crate::decoder::grayscale_batch::submit_grayscale_cuda_resident_prepared_batch_into(
                    &inputs,
                    native_decode_settings(group.options().settings),
                    &mut self.session,
                    fmt,
                    destination,
                )
                .map_err(|source| CudaBatchError::group(group, source))?,
            )
        } else {
            return Err(CudaBatchError::group(
                group,
                Error::UnsupportedCudaRequest {
                    reason:
                        "direct external CUDA batch decode requires exact Gray, RGB, or RGBA output",
                },
            ));
        };
        let external_group = CudaExternalBatchGroup {
            info: group.info().clone(),
            source_indices: group.source_indices().to_vec(),
            decoded_rects: group
                .images()
                .iter()
                .map(|image| image.plan().output_rect())
                .collect(),
            warnings: decode_warnings(group.options(), group.images().len()),
            ranges: pending.ranges().to_vec(),
        };
        Ok(SubmittedCudaExternalBatch {
            group: external_group,
            pending,
        })
    }
}

impl BatchDecoder for CudaBatchDecoder {
    type Output = CudaBatchDecodeResult;
    type Error = CudaBatchError;

    fn decode_prepared(&mut self, prepared: &PreparedBatch) -> Result<Self::Output, Self::Error> {
        CudaBatchDecoder::decode_prepared(self, prepared)
    }

    fn options(&self) -> BatchDecodeOptions {
        self.options
    }
}
