// SPDX-License-Identifier: Apache-2.0

use signinum_j2k_cuda::{CudaEncodeStageAccelerator, CudaEncodeStageTimings};
use signinum_j2k_native::{
    J2kEncodeStageAccelerator, J2kPacketizationEncodeJob, J2kPacketizationProgressionOrder,
};

#[cfg(feature = "cuda-runtime")]
use signinum_core::{CodecError, PixelFormat};
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::CudaContext;
#[cfg(feature = "cuda-runtime")]
use signinum_j2k::{
    encode_j2k_lossless, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLosslessSamples,
};
#[cfg(feature = "cuda-runtime")]
use signinum_j2k_cuda::{
    encode_lossless_from_cuda_buffer, encode_lossless_from_cuda_buffer_with_report,
    encode_lossless_from_cuda_buffers, submit_lossless_from_cuda_buffer, CudaLosslessEncodeTile,
    CudaSession,
};

#[test]
fn cuda_encode_stage_timings_are_publicly_readable_and_resettable() {
    let mut accelerator = CudaEncodeStageAccelerator::with_profile_collection(true);

    assert_eq!(
        accelerator.collected_stage_timings(),
        CudaEncodeStageTimings::default()
    );

    accelerator.reset_collected_stage_timings();
    assert_eq!(
        accelerator.collected_stage_timings(),
        CudaEncodeStageTimings::default()
    );
}

#[test]
fn cuda_encode_stage_can_prefer_cpu_packetization() {
    let mut accelerator = CudaEncodeStageAccelerator::default().prefer_cpu_packetization(true);
    let job = J2kPacketizationEncodeJob {
        resolution_count: 0,
        num_layers: 1,
        num_components: 1,
        code_block_count: 0,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[],
        resolutions: &[],
    };

    assert!(accelerator.encode_packetization(job).unwrap().is_none());
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 0);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_device_buffer_api_shapes_are_public() {
    fn assert_single_fn(
        _f: for<'tile, 'options, 'session> fn(
            CudaLosslessEncodeTile<'tile>,
            &'options J2kLosslessEncodeOptions,
            &'session mut CudaSession,
        ) -> Result<
            signinum_j2k::EncodedJ2k,
            signinum_j2k_cuda::Error,
        >,
    ) {
    }
    fn assert_single_report_fn(
        _f: for<'tile, 'options, 'session> fn(
            CudaLosslessEncodeTile<'tile>,
            &'options J2kLosslessEncodeOptions,
            &'session mut CudaSession,
        ) -> Result<
            signinum_j2k_cuda::CudaLosslessEncodeOutcome,
            signinum_j2k_cuda::Error,
        >,
    ) {
    }
    fn assert_submit_fn(
        _f: for<'tile, 'options, 'session> fn(
            CudaLosslessEncodeTile<'tile>,
            &'options J2kLosslessEncodeOptions,
            &'session mut CudaSession,
        ) -> Result<
            signinum_j2k_cuda::SubmittedJ2kLosslessCudaEncode,
            signinum_j2k_cuda::Error,
        >,
    ) {
    }
    type BatchEncodeFn = for<'slice, 'tile, 'options, 'session> fn(
        &'slice [CudaLosslessEncodeTile<'tile>],
        &'options J2kLosslessEncodeOptions,
        &'session mut CudaSession,
    ) -> Result<
        Vec<signinum_j2k::EncodedJ2k>,
        signinum_j2k_cuda::Error,
    >;
    fn assert_batch_fn(_f: BatchEncodeFn) {}

    assert_single_fn(encode_lossless_from_cuda_buffer);
    assert_single_report_fn(encode_lossless_from_cuda_buffer_with_report);
    assert_submit_fn(submit_lossless_from_cuda_buffer);
    assert_batch_fn(encode_lossless_from_cuda_buffers);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_device_buffer_empty_batch_fails_clearly() {
    let mut session = CudaSession::default();
    let options = J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_validation(J2kEncodeValidation::External);

    let error = encode_lossless_from_cuda_buffers(&[], &options, &mut session)
        .expect_err("empty CUDA encode batch should fail");

    assert!(error.is_unsupported());
    assert!(
        error.to_string().contains("empty"),
        "expected an empty-batch error, got {error}"
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_lossless_device_buffer_encode_matches_host_htj2k_when_required() {
    if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
        return;
    }

    let width = 8;
    let height = 8;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| u8::try_from((i * 17 + 11) % 251).expect("sample fits"))
        .collect();
    let context = CudaContext::system_default().expect("CUDA context");
    let buffer = context.upload(&pixels).expect("upload source pixels");
    let tile = CudaLosslessEncodeTile {
        buffer: &buffer,
        byte_offset: 0,
        width,
        height,
        pitch_bytes: width as usize * PixelFormat::Rgb8.bytes_per_pixel(),
        output_width: width,
        output_height: height,
        format: PixelFormat::Rgb8,
    };
    let options = J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_validation(J2kEncodeValidation::External);
    let mut session = CudaSession::default();

    let device = encode_lossless_from_cuda_buffer_with_report(tile, &options, &mut session)
        .expect("CUDA device-buffer HTJ2K encode");
    let host = encode_j2k_lossless(
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("host samples"),
        &options,
    )
    .expect("host HTJ2K encode");

    assert_eq!(device.encoded.width, host.width);
    assert_eq!(device.encoded.height, host.height);
    assert_eq!(device.encoded.components, host.components);
    assert_eq!(device.encoded.bit_depth, host.bit_depth);
    assert!(device.resident.coefficient_prep_used);
    assert!(device.resident.packetization_used);
    assert!(!device.input_copy_used);
    assert!(!device.encoded.codestream.is_empty());
}
