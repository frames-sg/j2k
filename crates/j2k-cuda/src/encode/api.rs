// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::BackendKind;

use crate::profile;

use super::CudaEncodeStageAccelerator;

/// Encode lossless JPEG 2000/HTJ2K samples through the CUDA encode-stage adapter.
///
/// This CUDA-named API is strict: every caller-provided backend preference is
/// treated as `EncodeBackendPreference::RequireDevice`, so unsupported stage
/// coverage returns an error instead of a CPU fallback codestream.
pub fn encode_j2k_lossless_with_cuda(
    samples: j2k::J2kLosslessSamples<'_>,
    options: &j2k::J2kLosslessEncodeOptions,
) -> Result<j2k::EncodedJ2k, crate::Error> {
    let strict_options = strict_cuda_encode_options(*options);
    let profile_enabled = profile::profile_stages_enabled();
    let mut accelerator = CudaEncodeStageAccelerator::with_profile_collection(profile_enabled);
    let total_start = profile::profile_now(profile_enabled);
    let encoded = j2k::encode_j2k_lossless_with_accelerator(
        samples,
        &strict_options,
        BackendKind::Cuda,
        &mut accelerator,
    )?;
    reject_non_cuda_encode_backend(&encoded)?;
    if profile_enabled {
        accelerator
            .encode_profile_report(
                &encoded,
                samples.data.len(),
                profile::elapsed_us(total_start),
            )
            .emit("encode");
    }
    Ok(encoded)
}

/// Encode lossless JPEG 2000/HTJ2K samples through CUDA and return stage timings.
#[doc(hidden)]
pub fn encode_j2k_lossless_with_cuda_and_profile(
    samples: j2k::J2kLosslessSamples<'_>,
    options: &j2k::J2kLosslessEncodeOptions,
) -> Result<(j2k::EncodedJ2k, profile::CudaHtj2kEncodeProfileReport), crate::Error> {
    let input_bytes = samples.data.len();
    let strict_options = strict_cuda_encode_options(*options);
    let mut accelerator = CudaEncodeStageAccelerator::with_profile_collection(true);
    let total_start = profile::profile_now(true);
    let encoded = j2k::encode_j2k_lossless_with_accelerator(
        samples,
        &strict_options,
        BackendKind::Cuda,
        &mut accelerator,
    )?;
    reject_non_cuda_encode_backend(&encoded)?;
    let report =
        accelerator.encode_profile_report(&encoded, input_bytes, profile::elapsed_us(total_start));
    report.emit("encode");
    Ok((encoded, report))
}

pub(super) fn strict_cuda_encode_options(
    options: j2k::J2kLosslessEncodeOptions,
) -> j2k::J2kLosslessEncodeOptions {
    options.with_backend(j2k::EncodeBackendPreference::RequireDevice)
}

pub(super) fn reject_non_cuda_encode_backend(
    encoded: &j2k::EncodedJ2k,
) -> Result<(), crate::Error> {
    if encoded.backend == BackendKind::Cuda {
        Ok(())
    } else {
        Err(crate::Error::UnsupportedCudaRequest {
            reason: "strict CUDA HTJ2K encode did not dispatch all required stages",
        })
    }
}
