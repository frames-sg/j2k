// SPDX-License-Identifier: MIT OR Apache-2.0

mod facade;
mod htj2k;
mod packetization;
mod resident;
mod resident_session;
mod resident_tiles;
mod routing;
mod sample_precision;
mod transforms;

#[cfg(feature = "cuda-runtime")]
use super::cuda_htj2k_encode_tables;
use super::packetization::{
    cuda_ht_segment_lengths, flatten_cuda_htj2k_packetization_job, CudaHtj2kPacketizationPlanError,
    CudaHtj2kPacketizationPlanTagNodeState,
};
use super::stage::cuda_packetization_plan_fallback_reason;
#[cfg(feature = "cuda-runtime")]
use super::stage::{cuda_dwt53_output_to_j2k, cuda_dwt97_output_to_j2k};
#[cfg(not(feature = "cuda-runtime"))]
use super::CudaEncodeFallbackReason;
#[cfg(feature = "cuda-runtime")]
use super::{
    cuda_resident_input_error, encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report,
    encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report, CudaLosslessEncodeTile,
};
use super::{
    encode_j2k_lossless_with_cuda, encode_j2k_lossless_with_cuda_and_profile,
    CudaEncodeStageAccelerator, CudaLosslessEncoder,
};
#[cfg(feature = "cuda-runtime")]
use crate::CudaSession;
#[cfg(feature = "cuda-runtime")]
use j2k::{encode_j2k_lossy_with_accelerator, J2kLossyEncodeOptions, J2kLossySamples};
use j2k::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLosslessSamples,
};
#[cfg(feature = "cuda-runtime")]
use j2k::{
    J2kDeinterleaveMctToF32Job, J2kDeinterleaveToF32Job, J2kHtCodeBlockEncodeJob,
    J2kHtCodeBlockSetEncodeJob, J2kResidentEncodeInputError,
};
use j2k::{
    J2kEncodeStageAccelerator, J2kEncodeStageError, J2kHtSubbandEncodeJob,
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder, J2kPacketizationResolution,
    J2kPacketizationSubband, J2kQuantizeSubbandJob,
};
#[cfg(feature = "cuda-runtime")]
use j2k_core::PixelFormat;
use j2k_core::{BackendKind, CodecError};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_j2k_engine::{
    CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob, CudaJ2kQuantizeJob,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::CudaContext;
use j2k_native::{
    encode_with_accelerator as encode_with_native_accelerator, DecodeSettings, EncodeOptions,
    EncodeResult, Image,
};
#[cfg(feature = "cuda-runtime")]
use j2k_native::{
    forward_dwt53_reference, forward_dwt97_reference, forward_ict_reference,
    try_deinterleave_reference,
};

fn assert_strict_cuda_classic_tier1_error<E: CodecError + ?Sized>(err: &E, context: &str) {
    // A CUDA host whose build skipped cuda-oxide fails before tier-1.
    if j2k_test_support::cuda_oxide_ptx_missing_is_skip(err, context) {
        return;
    }
    assert!(err.is_unsupported());
    let message = err.to_string();
    assert!(
        message.contains("tier1_code_block") || message.contains("deinterleave"),
        "expected {context} error to mention either the missing classic tier-1 stage or unavailable CUDA deinterleave, got {message}"
	        );
}

#[cfg(feature = "cuda-runtime")]
fn strict_cuda_resident_lossless_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(0))
        .with_validation(J2kEncodeValidation::External)
}

struct CudaTestEncodeRequest<'a> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
    signed: bool,
    options: &'a EncodeOptions,
    accelerator: &'a mut CudaEncodeStageAccelerator,
}

fn encode_with_cuda_test_accelerator(request: CudaTestEncodeRequest<'_>) -> EncodeResult<Vec<u8>> {
    let CudaTestEncodeRequest {
        pixels,
        width,
        height,
        components,
        bit_depth,
        signed,
        options,
        accelerator,
    } = request;
    encode_with_native_accelerator(
        pixels,
        width,
        height,
        u16::from(components),
        bit_depth,
        signed,
        options,
        accelerator,
    )
}

/// RGB8 fixture for the CUDA lossy parity tests: the PNM named by
/// `J2K_CUDA_LOSSY_PARITY_PNM` when set, else a generated 640x480 image, so
/// the tests always run under `J2K_REQUIRE_CUDA_RUNTIME`.
#[cfg(feature = "cuda-runtime")]
fn lossy_parity_fixture() -> (Vec<u8>, u32, u32, u16) {
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;
    if let Some(path) = std::env::var_os("J2K_CUDA_LOSSY_PARITY_PNM") {
        let image = j2k_test_support::read_pnm_image(path).expect("read parity PNM fixture");
        return (
            image.pixels,
            image.width,
            image.height,
            u16::try_from(image.channels).expect("PNM component count fits u16"),
        );
    }
    let pixels = (0u32..WIDTH * HEIGHT)
        .flat_map(|index| {
            let x = index % WIDTH;
            let y = index / WIDTH;
            [
                u8::try_from((x * 17 + y * 31 + index / 7) & 0xFF)
                    .expect("masked red sample fits u8"),
                u8::try_from((x * 11 + y * 47 + index / 13) & 0xFF)
                    .expect("masked green sample fits u8"),
                u8::try_from((x * 43 + y * 5 + index / 29) & 0xFF)
                    .expect("masked blue sample fits u8"),
            ]
        })
        .collect();
    (pixels, WIDTH, HEIGHT, 3)
}
