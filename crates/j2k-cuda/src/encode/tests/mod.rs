// SPDX-License-Identifier: MIT OR Apache-2.0

mod htj2k;
mod packetization;
mod resident;
mod resident_session;
mod resident_tiles;
mod routing;
mod transforms;

#[cfg(feature = "cuda-runtime")]
use super::cuda_htj2k_encode_tables;
use super::packetization::{
    cuda_ht_segment_lengths, flatten_cuda_htj2k_packetization_job, CudaHtj2kPacketizationPlanError,
    CudaHtj2kPacketizationPlanTagNodeState,
};
#[cfg(feature = "cuda-runtime")]
use super::stage::cuda_dwt53_output_to_j2k;
use super::stage::cuda_packetization_plan_fallback_reason;
#[cfg(feature = "cuda-runtime")]
use super::{
    cuda_resident_input_error, encode_lossless_from_cuda_buffer_to_cuda_buffer_with_report,
    encode_lossless_from_cuda_buffers_to_cuda_buffers_with_report, CudaLosslessEncodeTile,
};
use super::{
    encode_j2k_lossless_with_cuda, encode_j2k_lossless_with_cuda_and_profile,
    CudaEncodeStageAccelerator,
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
use j2k::{J2kDeinterleaveToF32Job, J2kHtCodeBlockEncodeJob, J2kResidentEncodeInputError};
use j2k::{
    J2kEncodeStageAccelerator, J2kEncodeStageError, J2kHtSubbandEncodeJob,
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder, J2kPacketizationResolution,
    J2kPacketizationSubband, J2kQuantizeSubbandJob,
};
use j2k_core::CodecError;
#[cfg(feature = "cuda-runtime")]
use j2k_core::{BackendKind, PixelFormat};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{
    CudaContext, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob, CudaJ2kQuantizeJob,
};
#[cfg(feature = "cuda-runtime")]
use j2k_native::forward_dwt53_reference;
use j2k_native::{
    encode_with_accelerator as encode_with_native_accelerator, DecodeSettings, EncodeOptions,
    EncodeResult, Image,
};

fn assert_strict_cuda_classic_tier1_error<E: CodecError + ?Sized>(err: &E, context: &str) {
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
