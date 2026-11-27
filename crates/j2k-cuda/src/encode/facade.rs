// SPDX-License-Identifier: MIT OR Apache-2.0

use core::{cell::Cell, marker::PhantomData};

use j2k::{EncodeBackendPreference, EncodedJ2k, J2kLosslessEncodeOptions, J2kLosslessSamples};
use j2k_core::BackendKind;

use super::CudaEncodeStageAccelerator;

/// Reason an `Auto` lossless encode that was offered to CUDA completed on the
/// CPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CudaEncodeFallbackReason {
    /// CUDA support was not compiled or no usable CUDA runtime/device was found.
    DeviceUnavailable,
    /// CUDA was available, but it did not implement every stage required by the request.
    DeviceRouteIncomplete,
}

/// Whether `Auto` offers a host-input lossless encode to CUDA: only for a
/// shape whose CUDA route was measured faster than the CPU.
fn auto_uses_cuda(samples: J2kLosslessSamples<'_>) -> bool {
    crate::generated::promotion::auto_host_output_encode_qualifies(
        samples.components,
        samples.width,
        samples.height,
    )
}

/// Opaque result from [`CudaLosslessEncoder`].
///
/// The requested backend is retained separately from the backend that actually
/// satisfied the request. A CPU result is a fallback only when an
/// [`EncodeBackendPreference::Auto`] request was offered to CUDA; when `Auto`
/// keeps a shape on the CPU because no measured-faster CUDA route covers it,
/// the CPU is the chosen route and no fallback is reported.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaLosslessEncodeResult {
    requested_backend: EncodeBackendPreference,
    fallback_reason: Option<CudaEncodeFallbackReason>,
    encoded: EncodedJ2k,
}

impl CudaLosslessEncodeResult {
    /// `device_unavailable` is `None` when the encode was not offered to CUDA.
    fn new(
        requested_backend: EncodeBackendPreference,
        device_unavailable: Option<bool>,
        encoded: EncodedJ2k,
    ) -> Self {
        let fell_back = requested_backend == EncodeBackendPreference::Auto
            && encoded.backend == BackendKind::Cpu;
        let fallback_reason = match device_unavailable {
            Some(true) if fell_back => Some(CudaEncodeFallbackReason::DeviceUnavailable),
            Some(false) if fell_back => Some(CudaEncodeFallbackReason::DeviceRouteIncomplete),
            _ => None,
        };
        Self {
            requested_backend,
            fallback_reason,
            encoded,
        }
    }

    /// Backend preference supplied for this encode job.
    #[must_use]
    pub const fn requested_backend(&self) -> EncodeBackendPreference {
        self.requested_backend
    }

    /// Backend that satisfied the encode contract.
    #[must_use]
    pub const fn actual_backend(&self) -> BackendKind {
        self.encoded.backend
    }

    /// Why an `Auto` request that was offered to CUDA completed on the CPU.
    ///
    /// `None` when CUDA completed the encode, when the request was not `Auto`,
    /// and when `Auto` kept the encode on the CPU because no measured-faster
    /// CUDA route covers its shape.
    #[must_use]
    pub const fn fallback_reason(&self) -> Option<CudaEncodeFallbackReason> {
        self.fallback_reason
    }

    /// Encode-stage dispatches observed while producing the codestream.
    #[must_use]
    pub const fn dispatch_report(&self) -> j2k::J2kEncodeDispatchReport {
        self.encoded.dispatch_report
    }

    /// Borrow the encoded codestream and its image metadata.
    #[must_use]
    pub const fn encoded(&self) -> &EncodedJ2k {
        &self.encoded
    }

    /// Consume the route report and return the encoded codestream and metadata.
    #[must_use]
    pub fn into_encoded(self) -> EncodedJ2k {
        self.encoded
    }
}

/// Reusable CUDA-aware lossless JPEG 2000 encoder.
///
/// [`Self::encode`] honors each job's [`EncodeBackendPreference`]:
///
/// - `CpuOnly` does not initialize or submit CUDA work.
/// - `Auto` uses CUDA only for a shape whose CUDA route was measured faster
///   than the CPU, as listed in the routing evidence. No encode shape is
///   listed today, so `Auto` encodes on the CPU and reports no fallback. For
///   a listed shape it returns a CPU result, with a
///   [`CudaEncodeFallbackReason`], when CUDA is unavailable or does not cover
///   every required stage.
/// - `RequireDevice` returns an error unless CUDA satisfies every required stage.
///
/// `Auto` does not retry on the CPU after a CUDA execution error. Such errors
/// can indicate uncertain device state and are returned to the caller. The
/// encoder discards its cached accelerator state after any error, so a later
/// job can safely retry initialization or select `CpuOnly`.
///
/// The encoder is `Send` but intentionally not `Sync`; encoding requires
/// exclusive `&mut self` access. Move one encoder to each worker, or put it
/// behind a mutex when jobs must share it.
///
/// ```compile_fail
/// fn require_sync<T: Sync>() {}
/// require_sync::<j2k_cuda::CudaLosslessEncoder>();
/// ```
#[derive(Debug)]
pub struct CudaLosslessEncoder {
    accelerator: CudaEncodeStageAccelerator,
    not_sync: PhantomData<Cell<()>>,
}

impl Default for CudaLosslessEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaLosslessEncoder {
    /// Create a reusable encoder with lazily initialized CUDA state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            accelerator: CudaEncodeStageAccelerator::default(),
            not_sync: PhantomData,
        }
    }

    /// Encode one job according to its backend preference.
    ///
    /// Device unavailability and incomplete device-stage coverage are recoverable
    /// only for `Auto`. Input, allocation, validation, and CUDA execution errors
    /// are returned without a second CPU encode attempt.
    pub fn encode(
        &mut self,
        samples: J2kLosslessSamples<'_>,
        options: &J2kLosslessEncodeOptions,
    ) -> Result<CudaLosslessEncodeResult, crate::Error> {
        self.encode_with_options(samples, *options)
    }

    /// Encode one job with a strict CUDA contract.
    ///
    /// This method ignores the job's stored backend preference and behaves as
    /// [`EncodeBackendPreference::RequireDevice`]. It exists alongside
    /// [`Self::encode`] so CUDA-specific callers can require CUDA (with no CPU
    /// fallback) without changing reusable per-job option templates.
    pub fn encode_strict_cuda(
        &mut self,
        samples: J2kLosslessSamples<'_>,
        options: &J2kLosslessEncodeOptions,
    ) -> Result<CudaLosslessEncodeResult, crate::Error> {
        self.encode_with_options(
            samples,
            options.with_backend(EncodeBackendPreference::RequireDevice),
        )
    }

    fn encode_with_options(
        &mut self,
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
    ) -> Result<CudaLosslessEncodeResult, crate::Error> {
        let offer_device = match options.backend {
            EncodeBackendPreference::CpuOnly => false,
            EncodeBackendPreference::Auto => auto_uses_cuda(samples),
            EncodeBackendPreference::RequireDevice => true,
        };
        self.encode_routed(samples, options, offer_device)
    }

    /// Encode one job whose routing decision is already made: `offer_device`
    /// says whether CUDA is offered the encode.
    pub(super) fn encode_routed(
        &mut self,
        samples: J2kLosslessSamples<'_>,
        options: J2kLosslessEncodeOptions,
        offer_device: bool,
    ) -> Result<CudaLosslessEncodeResult, crate::Error> {
        self.accelerator.begin_encode_attempt();
        let requested_backend = options.backend;
        let encoded = if offer_device {
            j2k::encode_j2k_lossless_with_accelerator(
                samples,
                &options,
                BackendKind::Cuda,
                &mut self.accelerator,
            )
        } else {
            j2k::encode_j2k_lossless(
                samples,
                &options.with_backend(EncodeBackendPreference::CpuOnly),
            )
        };

        match encoded {
            Ok(encoded) => Ok(CudaLosslessEncodeResult::new(
                requested_backend,
                offer_device.then(|| self.accelerator.device_unavailable_observed()),
                encoded,
            )),
            Err(error) => {
                self.accelerator = CudaEncodeStageAccelerator::default();
                Err(error.into())
            }
        }
    }
}
