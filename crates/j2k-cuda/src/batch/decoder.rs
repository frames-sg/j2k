// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistent CUDA batch decoder facade.

#[cfg(not(feature = "cuda-runtime"))]
use super::IndexedBatchError;
#[cfg(feature = "cuda-runtime")]
use super::{
    decode_warnings, grayscale_inputs, group_pixel_format, native_color_inputs, validate_layout,
    CudaExternalBatchGroup, CudaGroupView, PixelFormat, PreparedBatchGroup,
    SubmittedCudaCodecBatch, SubmittedCudaExternalBatch,
};
use super::{
    prepare_batch, prepare_batch_from_images, BatchDecodeOptions, BatchDecoder,
    BatchInfrastructureError, CudaBatchDecodeResult, CudaBatchError, CudaSession, EncodedImage,
    Error, PreparedBatch, PreparedImage, PreparedImageOperation,
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
            unavailable(
                prepared
                    .groups()
                    .first()
                    .map(j2k::PreparedBatchGroup::source_indices),
                prepared.errors(),
            )
        }
    }

    /// Decode regions at exact wavelet reductions of existing prepared
    /// sources into CUDA-resident groups.
    ///
    /// Output geometry, native sample values and grouping match
    /// [`j2k::CpuBatchDecoder::decode_operations`], and source indices refer to
    /// positions in `operations`. Exact operations decode only through their
    /// prepared offset plans; a shape the prepared CUDA route cannot describe
    /// fails its group with a capability error.
    ///
    /// A [`j2k::J2kDecodeControl`]'s output limit applies to each operation's
    /// output, and its cancellation token is checked during preparation and
    /// before each group is planned and submitted; submitted CUDA work runs to
    /// completion. Operations share a group only with operations carrying the
    /// same token, so a cancelled group fails only its own operations. A
    /// working-memory limit fails the operation as unsupported: CUDA working
    /// memory is not accounted against a host allowance.
    ///
    /// # Errors
    /// Returns [`CudaBatchError`] when operation metadata cannot be allocated
    /// or the CUDA session becomes unusable; other failures are reported per
    /// operation or group in the result.
    pub fn decode_operations(
        &mut self,
        operations: &[PreparedImageOperation<'_>],
    ) -> Result<CudaBatchDecodeResult, CudaBatchError> {
        #[cfg(feature = "cuda-runtime")]
        {
            self.submit_operations(operations)?.wait()
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            let prepared = j2k::prepare_device_operations(operations, self.options)?;
            unavailable(
                prepared
                    .groups()
                    .first()
                    .map(j2k::PreparedDeviceOperationGroup::source_indices),
                prepared.errors(),
            )
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

    /// Submit one prepared exact-native Gray/RGB/RGBA group into CUDA storage,
    /// leaving its final store pending.
    ///
    /// RGB and RGBA groups larger than 32 images decode in successive 32-image
    /// chunks into consecutive ranges of `destination`. To keep decode scratch
    /// within the session cache, submission blocks the host until every chunk
    /// except the last has completed; only the final chunk remains pending.
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
        self.submit_group_view_into(group.into(), destination)
    }

    /// Decode one group of exact operations, prepared by
    /// [`j2k::prepare_device_operations`], directly into a validated
    /// caller-owned CUDA allocation range. See [`Self::decode_operations`]
    /// and [`Self::decode_batch_into`].
    ///
    /// # Safety
    ///
    /// As [`Self::decode_batch_into`].
    #[cfg(feature = "cuda-runtime")]
    #[doc(hidden)]
    pub unsafe fn decode_operation_group_into(
        &mut self,
        group: &j2k::PreparedDeviceOperationGroup,
        destination: &mut j2k_cuda_runtime::CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<CudaExternalBatchGroup, CudaBatchError> {
        // SAFETY: this synchronous convenience immediately waits while the
        // caller's external view and exclusive managed-owner borrow are live.
        unsafe { self.submit_operation_group_into(group, destination) }?.wait()
    }

    /// Submit one group of exact operations into CUDA storage, leaving its
    /// final store pending. A cancelled group is not submitted.
    ///
    /// # Safety
    ///
    /// As [`Self::submit_batch_into`].
    #[cfg(feature = "cuda-runtime")]
    #[doc(hidden)]
    pub unsafe fn submit_operation_group_into(
        &mut self,
        group: &j2k::PreparedDeviceOperationGroup,
        destination: &mut j2k_cuda_runtime::CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<SubmittedCudaExternalBatch, CudaBatchError> {
        self.submit_group_view_into(group.into(), destination)
    }

    #[cfg(feature = "cuda-runtime")]
    fn submit_group_view_into(
        &mut self,
        group: CudaGroupView<'_>,
        destination: &mut j2k_cuda_runtime::CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<SubmittedCudaExternalBatch, CudaBatchError> {
        let group_error = |source| CudaBatchError::group(group.source_indices, source);
        group.check_cancelled().map_err(group_error)?;
        let fmt = group_pixel_format(group.info)
            .and_then(|fmt| {
                validate_layout(group.info)?;
                Ok(fmt)
            })
            .map_err(group_error)?;
        let pending = if matches!(
            fmt,
            PixelFormat::Rgb8
                | PixelFormat::Rgb16
                | PixelFormat::RgbI16
                | PixelFormat::Rgba8
                | PixelFormat::Rgba16
                | PixelFormat::RgbaI16
        ) {
            let inputs = native_color_inputs(group).map_err(group_error)?;
            SubmittedCudaCodecBatch::Color(
                crate::decoder::submit_native_color_resident_prepared_batch_into(
                    &inputs,
                    &mut self.session,
                    fmt,
                    group.info.layout,
                    destination,
                )
                .map_err(group_error)?,
            )
        } else if matches!(
            fmt,
            PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayI16
        ) {
            SubmittedCudaCodecBatch::Grayscale(
                crate::decoder::grayscale_batch::submit_grayscale_cuda_resident_prepared_batch_into(
                    &grayscale_inputs(group),
                    group.fallback_settings(),
                    &mut self.session,
                    fmt,
                    destination,
                )
                .map_err(group_error)?,
            )
        } else {
            return Err(group_error(Error::capability_rejected(
                j2k_core::CapabilityRejection::unsupported_format(
                    "direct external CUDA batch decode requires exact Gray, RGB, or RGBA output",
                ),
            )));
        };
        let mut metadata_budget =
            crate::allocation::HostPhaseBudget::new("CUDA external batch result metadata");
        let source_indices = metadata_budget
            .try_clone_slice(group.source_indices)
            .map_err(|source| group_error(source.into()))?;
        let decoded_rects = metadata_budget
            .try_collect_exact(group.images().map(super::input::GroupImage::output_rect))
            .map_err(|source| group_error(source.into()))?;
        let warnings = decode_warnings(group).map_err(group_error)?;
        let ranges = metadata_budget
            .try_clone_slice(pending.ranges())
            .map_err(|source| group_error(source.into()))?;
        let external_group = CudaExternalBatchGroup {
            info: group.info.clone(),
            source_indices,
            decoded_rects,
            warnings,
            ranges,
        };
        Ok(SubmittedCudaExternalBatch {
            group: external_group,
            pending,
        })
    }
}

/// A build without the CUDA runtime fails its first group and reports
/// preparation errors unchanged.
#[cfg(not(feature = "cuda-runtime"))]
fn unavailable(
    first_group: Option<&[usize]>,
    preparation_errors: &[IndexedBatchError],
) -> Result<CudaBatchDecodeResult, CudaBatchError> {
    if let Some(source_indices) = first_group {
        return Err(CudaBatchError::group(
            source_indices,
            Error::CudaUnavailable,
        ));
    }
    let mut errors = Vec::new();
    errors
        .try_reserve_exact(preparation_errors.len())
        .map_err(|_| BatchInfrastructureError::HostAllocationFailed {
            what: "CUDA stub indexed errors",
            bytes: preparation_errors
                .len()
                .saturating_mul(core::mem::size_of::<IndexedBatchError>()),
        })?;
    errors.extend_from_slice(preparation_errors);
    Ok(CudaBatchDecodeResult {
        groups: Vec::new(),
        errors,
        group_errors: Vec::new(),
    })
}

impl BatchDecoder for CudaBatchDecoder {
    type Output = CudaBatchDecodeResult;
    type Error = CudaBatchError;

    fn decode_prepared(&mut self, prepared: &PreparedBatch) -> Result<Self::Output, Self::Error> {
        Self::decode_prepared(self, prepared)
    }

    fn options(&self) -> BatchDecodeOptions {
        self.options
    }
}
