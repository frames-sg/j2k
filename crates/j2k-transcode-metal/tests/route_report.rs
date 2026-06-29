// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendKind, BackendRequest};
use j2k_test_support::JPEG_GRAYSCALE_8X8;
use j2k_transcode::accelerator::TranscodeStageError;
use j2k_transcode::{JpegTileBatchInput, JpegToHtj2kError, JpegToHtj2kOptions};
use j2k_transcode_metal::{
    jpeg_to_htj2k_batch_with_metal_route, jpeg_to_htj2k_with_metal_route,
    MetalTranscodeFallbackReason,
};

#[cfg(target_os = "macos")]
#[test]
fn metal_encoded_codestream_exports_resident_handoff_descriptor() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let codestream_buffer = device.new_buffer(512, metal::MTLResourceOptions::StorageModeShared);
    let encoded = j2k_metal::MetalEncodedJ2k {
        codestream_buffer,
        byte_offset: 16,
        byte_len: 128,
        capacity: 256,
        width: 64,
        height: 64,
        components: 1,
        bit_depth: 8,
        signed: false,
    };

    let descriptor =
        j2k_transcode_metal::resident_codestream_buffer_from_metal_encoded_j2k(&encoded)
            .expect("valid resident codestream descriptor");

    assert_eq!(descriptor.buffer.backend(), BackendKind::Metal);
    assert_eq!(descriptor.buffer.memory_range().offset, 16);
    assert_eq!(descriptor.buffer.byte_len(), 256);
    assert_eq!(descriptor.byte_len, 128);
    assert_eq!(descriptor.capacity, 256);
}

#[test]
fn cpu_transcode_route_reports_explicit_cpu_request() {
    let routed = jpeg_to_htj2k_with_metal_route(
        JPEG_GRAYSCALE_8X8,
        &JpegToHtj2kOptions::lossless_53(),
        BackendRequest::Cpu,
    )
    .expect("CPU-routed transcode succeeds");

    assert!(!routed.encoded.codestream.is_empty());
    assert_eq!(routed.route.request, BackendRequest::Cpu);
    assert_eq!(routed.route.selected_transform_backend, BackendKind::Cpu);
    assert_eq!(routed.route.output_backend, BackendKind::Cpu);
    assert_eq!(
        routed.route.fallback_reason,
        Some(MetalTranscodeFallbackReason::CpuRequested)
    );
    assert_eq!(routed.route.pipeline_map.stages.len(), 6);
}

#[test]
fn auto_transcode_route_reports_structured_cpu_fallback() {
    let routed = jpeg_to_htj2k_with_metal_route(
        JPEG_GRAYSCALE_8X8,
        &JpegToHtj2kOptions::lossless_53(),
        BackendRequest::Auto,
    )
    .expect("Auto-routed transcode succeeds");

    assert_eq!(routed.route.request, BackendRequest::Auto);
    assert_eq!(routed.route.selected_transform_backend, BackendKind::Cpu);
    assert_eq!(
        routed.route.fallback_reason,
        Some(MetalTranscodeFallbackReason::AutoAllTransformJobsFellBackToCpu)
    );
    assert!(
        routed
            .route
            .pipeline_map
            .debug_report()
            .contains("transfer_count="),
        "route report should expose transfer count fields"
    );
}

#[test]
fn batch_auto_transcode_route_reports_aggregate_fallback() {
    let inputs = [
        JpegTileBatchInput {
            bytes: JPEG_GRAYSCALE_8X8,
        },
        JpegTileBatchInput {
            bytes: JPEG_GRAYSCALE_8X8,
        },
    ];

    let routed = jpeg_to_htj2k_batch_with_metal_route(
        &inputs,
        &JpegToHtj2kOptions::lossless_53(),
        BackendRequest::Auto,
    )
    .expect("Auto-routed batch transcode succeeds");

    assert_eq!(routed.batch.report.tile_count, inputs.len());
    assert_eq!(routed.batch.report.successful_tiles, inputs.len());
    assert_eq!(routed.route.selected_transform_backend, BackendKind::Cpu);
    assert_eq!(
        routed.route.fallback_reason,
        Some(MetalTranscodeFallbackReason::AutoAllTransformJobsFellBackToCpu)
    );
}

#[test]
fn strict_metal_transcode_never_silently_returns_cpu_route() {
    let result = jpeg_to_htj2k_with_metal_route(
        JPEG_GRAYSCALE_8X8,
        &JpegToHtj2kOptions::lossless_53(),
        BackendRequest::Metal,
    );

    match result {
        Ok(routed) => {
            assert_eq!(routed.route.request, BackendRequest::Metal);
            assert_eq!(routed.route.selected_transform_backend, BackendKind::Metal);
            assert_eq!(routed.route.fallback_reason, None);
        }
        Err(JpegToHtj2kError::Accelerator(
            TranscodeStageError::DeviceUnavailable
            | TranscodeStageError::Unsupported("strict Metal transcode produced no Metal dispatch"),
        )) => {}
        Err(error) => panic!("strict Metal route returned unexpected error: {error}"),
    }
}

#[test]
fn cuda_request_through_metal_route_is_explicitly_rejected() {
    let Err(error) = jpeg_to_htj2k_with_metal_route(
        JPEG_GRAYSCALE_8X8,
        &JpegToHtj2kOptions::lossless_53(),
        BackendRequest::Cuda,
    ) else {
        panic!("CUDA request through Metal facade should be unsupported");
    };

    assert!(matches!(
        error,
        JpegToHtj2kError::Unsupported("CUDA transcode requested through Metal adapter")
    ));
}
