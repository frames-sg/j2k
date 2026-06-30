use j2k_core::{
    BackendRequest, CodecError, DecoderContext, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice,
};
use j2k_jpeg_cuda::{Codec, CudaSession, Decoder, Error};
use j2k_test_support::{cuda_jpeg_hardware_decode_required, cuda_runtime_required};

const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");
const OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA: u8 = 2;

#[test]
fn auto_falls_back_to_cpu_surface() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
}

#[test]
fn explicit_cuda_request_returns_cuda_surface_or_clear_unavailable_error() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    match decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda) {
        Ok(surface) => {
            assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
            assert_eq!(surface.as_host_bytes(), None);
            #[cfg(feature = "cuda-runtime")]
            assert_ne!(
                surface.cuda_surface().expect("cuda surface").device_ptr(),
                0
            );
        }
        Err(error) => assert!(error.is_unsupported()),
    }
}

#[test]
fn explicit_cuda_request_validates_decode_before_upload() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_to_device(PixelFormat::Rgba16, BackendRequest::Cuda)
        .expect_err("unsupported decode");
    assert!(error.is_unsupported());
    assert!(!matches!(error, Error::CudaUnavailable));
}

#[test]
fn explicit_cuda_gray8_request_fails_without_cpu_upload() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cuda)
        .expect_err("strict CUDA Gray8 decode should be unsupported");
    assert!(error.is_unsupported());
    assert!(!matches!(error, Error::CudaUnavailable));
}

#[test]
fn explicit_cuda_request_returns_cuda_surface_when_cuda_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (16, 16));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode(PixelFormat::Rgb8)
        .expect("host decode");
    assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
}

#[test]
fn explicit_cuda_region_scaled_surface_fails_without_owned_cuda_path() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let error = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Cuda)
        .expect_err("strict CUDA region+scaled decode should be unsupported");
    assert!(error.is_unsupported());
}

#[test]
fn explicit_cuda_region_surface_fails_without_owned_cuda_path() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Cuda)
        .expect_err("strict CUDA region decode should be unsupported");

    assert!(error.is_unsupported());
    assert!(error.to_string().contains("region output"));
}

#[test]
fn explicit_cuda_scaled_surface_fails_without_owned_cuda_path() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Cuda)
        .expect_err("strict CUDA scaled decode should be unsupported");

    assert!(error.is_unsupported());
    assert!(error.to_string().contains("scaled output"));
}

#[test]
fn explicit_cuda_download_respects_padded_stride_when_cuda_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_cuda_surface(&surface);
    let row_bytes = surface.pitch_bytes();
    let stride = row_bytes + 5;
    let mut downloaded = vec![0xCD; stride * surface.dimensions().1 as usize];
    surface
        .download_into(&mut downloaded, stride)
        .expect("download cuda surface");

    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode(PixelFormat::Rgb8)
        .expect("host decode");
    for (row, expected_row) in expected.chunks(row_bytes).enumerate() {
        let start = row * stride;
        assert_surface_bytes_match_or_are_close(
            &surface,
            &downloaded[start..start + row_bytes],
            expected_row,
        );
        assert_eq!(&downloaded[start + row_bytes..start + stride], &[0xCD; 5]);
    }
}

#[test]
fn explicit_cuda_full_frame_uses_owned_decode_when_required() {
    if !cuda_jpeg_hardware_decode_required() {
        return;
    }

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    let cuda = surface.cuda_surface().expect("cuda surface");
    let stats = cuda.stats();
    assert!(
        stats.used_owned_cuda_decode(),
        "explicit full-frame RGB8 CUDA decode must use the J2K-owned CUDA JPEG path when required"
    );
    assert!(
        !stats.used_hardware_decode(),
        "strict J2K-owned CUDA JPEG decode must not report external hardware decode"
    );
    assert!(
        stats.decode_kernel_dispatches() > 0,
        "owned CUDA decode path must report decode kernel dispatches"
    );
    assert_eq!(
        stats.copy_kernel_dispatches(),
        0,
        "owned CUDA decode path should not be reported as the CPU decode plus copy fallback"
    );
}

#[test]
fn explicit_cuda_full_frame_422_uses_owned_decode_when_required() {
    assert_full_frame_owned_cuda_decode_when_required(BASELINE_422, (16, 8));
}

#[test]
fn explicit_cuda_full_frame_444_uses_owned_decode_when_required() {
    assert_full_frame_owned_cuda_decode_when_required(BASELINE_444, (8, 8));
}

#[test]
fn cuda_session_owned_decode_cache_starts_empty() {
    let session = CudaSession::default();

    assert_eq!(session.owned_cuda_packet_cache_len(), 0);
}

#[cfg(feature = "cuda-runtime")]
fn generated_rgb_jpeg(subsampling: j2k_jpeg::JpegSubsampling, width: u32, height: u32) -> Vec<u8> {
    let rgb = j2k_test_support::gpu_bench_rgb8(width, height);
    j2k_jpeg::encode_jpeg_baseline(
        j2k_jpeg::JpegSamples::Rgb8 {
            data: &rgb,
            width,
            height,
        },
        j2k_jpeg::JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval: None,
            backend: j2k_jpeg::JpegBackend::Cpu,
        },
    )
    .expect("generated JPEG")
    .data
}

fn assert_cuda_surface(surface: &j2k_jpeg_cuda::Surface) {
    let cuda = surface.cuda_surface().expect("cuda surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert!(cuda.stats().kernel_dispatches() > 0);
}

fn assert_surface_bytes_match_or_are_close(
    surface: &j2k_jpeg_cuda::Surface,
    actual: &[u8],
    expected: &[u8],
) {
    assert_eq!(actual.len(), expected.len());
    let stats = surface.cuda_surface().expect("cuda surface").stats();
    if stats.used_owned_cuda_decode() {
        let max_delta = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .unwrap_or(0);
        assert!(
            max_delta <= OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA,
            "J2K-owned CUDA decode differed from the CPU reference by max channel delta {max_delta}"
        );
        return;
    }
    assert_eq!(actual, expected);
}

fn assert_full_frame_owned_cuda_decode_when_required(input: &[u8], dimensions: (u32, u32)) {
    if !cuda_jpeg_hardware_decode_required() {
        return;
    }

    let mut decoder = Decoder::new(input).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.dimensions(), dimensions);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cuda_surface(&surface);
    let stats = surface.cuda_surface().expect("cuda surface").stats();
    assert!(stats.used_owned_cuda_decode());
    assert!(!stats.used_hardware_decode());
    assert_eq!(stats.copy_kernel_dispatches(), 0);

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");
    let (expected, _) = j2k_jpeg::Decoder::new(input)
        .expect("host decoder")
        .decode(PixelFormat::Rgb8)
        .expect("host decode");
    assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
}

#[test]
fn submit_to_device_auto_falls_back_to_cpu_surface() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("submission")
    .wait()
    .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
    assert!(session.submissions() >= 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn submit_to_device_auto_does_not_initialize_cuda_runtime() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("submission")
    .wait()
    .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert_eq!(session.submissions(), 1);
    assert!(!session.is_runtime_initialized());
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_submissions_reuse_session_runtime_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let mut session = CudaSession::default();
    assert!(!session.is_runtime_initialized());

    let mut first = Decoder::new(BASELINE_420).expect("decoder");
    let first_surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut first,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("first submission")
    .wait()
    .expect("first surface");
    assert_eq!(first_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_cuda_surface(&first_surface);
    assert!(session.is_runtime_initialized());

    let mut second = Decoder::new(BASELINE_420).expect("decoder");
    let second_surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut second,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("second submission")
    .wait()
    .expect("second surface");
    assert_eq!(second_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_cuda_surface(&second_surface);
    assert_eq!(session.submissions(), 2);
    assert!(session.is_runtime_initialized());
}

#[test]
fn auto_region_scaled_surface_matches_host_decode() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut host_decoder = Decoder::new(BASELINE_420).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize * 3];
    host_decoder
        .decode_region_scaled_into(
            &mut j2k_jpeg::ScratchPool::new(),
            &mut host,
            scaled.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn tile_batch_region_scaled_auto_surface_matches_host_decode() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        scale,
        BackendRequest::Auto,
    )
    .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
}

#[test]
fn tile_batch_region_scaled_cuda_surface_fails_without_owned_cuda_path() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let error = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        scale,
        BackendRequest::Cuda,
    )
    .expect_err("strict CUDA tile-batch region+scaled decode should be unsupported");
    assert!(error.is_unsupported());
}

#[test]
fn tile_batch_region_cuda_surface_fails_without_owned_cuda_path() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();

    let error = Codec::decode_tile_region_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        BackendRequest::Cuda,
    )
    .expect_err("strict CUDA tile-batch region decode should be unsupported");

    assert!(error.is_unsupported());
    assert!(error.to_string().contains("region output"));
}

#[test]
fn tile_batch_scaled_cuda_surface_fails_without_owned_cuda_path() {
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();

    let error = Codec::decode_tile_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        Downscale::Half,
        BackendRequest::Cuda,
    )
    .expect_err("strict CUDA tile-batch scaled decode should be unsupported");

    assert!(error.is_unsupported());
    assert!(error.to_string().contains("scaled output"));
}

#[test]
fn decode_tiles_to_device_auto_preserves_order_and_matches_host_bytes() {
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode(PixelFormat::Rgb8)
        .expect("host decode");
    for surface in surfaces {
        assert_eq!(surface.dimensions(), (16, 16));
        match surface.backend_kind() {
            j2k_core::BackendKind::Cpu => {
                assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
            }
            j2k_core::BackendKind::Cuda => {
                let mut downloaded = vec![0u8; surface.byte_len()];
                surface
                    .download_into(&mut downloaded, surface.pitch_bytes())
                    .expect("download cuda surface");
                assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
            }
            j2k_core::BackendKind::Metal => panic!("JPEG CUDA batch returned Metal surface"),
        }
    }
}

#[test]
fn decode_tiles_to_device_with_session_auto_preserves_order_and_matches_host_bytes() {
    let inputs = [BASELINE_420, BASELINE_420];
    let mut session = CudaSession::default();

    let surfaces = Codec::decode_tiles_to_device_with_session(
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
        &mut session,
    )
    .expect("session-backed batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode(PixelFormat::Rgb8)
        .expect("host decode");
    for surface in surfaces {
        assert_eq!(surface.dimensions(), (16, 16));
        match surface.backend_kind() {
            j2k_core::BackendKind::Cpu => {
                assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
            }
            j2k_core::BackendKind::Cuda => {
                let mut downloaded = vec![0u8; surface.byte_len()];
                surface
                    .download_into(&mut downloaded, surface.pitch_bytes())
                    .expect("download cuda surface");
                assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
            }
            j2k_core::BackendKind::Metal => panic!("JPEG CUDA batch returned Metal surface"),
        }
    }
}

#[test]
fn decode_tiles_to_device_explicit_cuda_returns_cuda_surfaces_or_clear_unavailable_error() {
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    match Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    ) {
        Ok(surfaces) => {
            assert_eq!(surfaces.len(), inputs.len());
            for surface in surfaces {
                assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
                assert_eq!(surface.as_host_bytes(), None);
                assert_cuda_surface(&surface);
            }
        }
        Err(error) => assert!(error.is_unsupported()),
    }
}

#[test]
fn decode_tiles_to_device_explicit_cuda_gray8_fails_without_cpu_upload() {
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    let error = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Gray8,
        BackendRequest::Cuda,
    )
    .expect_err("strict CUDA Gray8 batch decode should be unsupported");
    assert!(error.is_unsupported());
    assert!(!matches!(error, Error::CudaUnavailable));
}

#[test]
fn decode_tiles_to_device_explicit_cuda_uses_owned_decode_when_required() {
    if !cuda_jpeg_hardware_decode_required() {
        return;
    }

    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("cuda batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    for surface in surfaces {
        let stats = surface.cuda_surface().expect("cuda surface").stats();
        assert!(
            stats.used_owned_cuda_decode(),
            "explicit full-tile RGB8 CUDA batch decode must use the J2K-owned CUDA path when required"
        );
        assert!(
            stats.decode_kernel_dispatches() > 0,
            "owned CUDA batch decode path must report decode dispatches"
        );
        assert_eq!(
            stats.copy_kernel_dispatches(),
            0,
            "owned CUDA batch decode path should not be reported as CPU decode plus copy"
        );
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn generated_420_chunked_entropy_diagnostic_runs_when_cuda_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let input = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 256, 256);
    let mut session = CudaSession::default();
    let report = Codec::diagnose_tile_rgb8_chunked_entropy_with_session(
        &input,
        j2k_cuda_runtime::CudaJpegChunkedEntropyConfig {
            subsequence_words: 64,
            sequence_len: 32,
            max_overflow_subsequences: 4,
        },
        &mut session,
    )
    .expect("chunked entropy diagnostic");

    assert!(report.subsequence_count() > 0);
    assert_eq!(report.failed_state_count(), 0);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn generated_422_chunked_entropy_diagnostic_returns_diagnostic_420_only_error() {
    let input = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr422, 256, 256);
    let mut session = CudaSession::default();
    let error = Codec::diagnose_tile_rgb8_chunked_entropy_with_session(
        &input,
        j2k_cuda_runtime::CudaJpegChunkedEntropyConfig {
            subsequence_words: 64,
            sequence_len: 32,
            max_overflow_subsequences: 4,
        },
        &mut session,
    )
    .expect_err("4:2:2 input should be rejected before diagnostic runtime");

    assert!(error.is_unsupported());
    match error {
        Error::UnsupportedCudaRequest { reason } => {
            assert!(reason.contains("chunked entropy diagnostic"));
            assert!(reason.contains("4:2:0"));
        }
        other => panic!("expected unsupported CUDA diagnostic error, got {other:?}"),
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn generated_420_chunked_entropy_diagnostic_rejects_invalid_config_before_runtime() {
    let input = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 256, 256);
    let mut session = CudaSession::default();
    let error = Codec::diagnose_tile_rgb8_chunked_entropy_with_session(
        &input,
        j2k_cuda_runtime::CudaJpegChunkedEntropyConfig {
            subsequence_words: 0,
            sequence_len: 32,
            max_overflow_subsequences: 4,
        },
        &mut session,
    )
    .expect_err("invalid diagnostic config should be rejected before runtime");

    assert!(error.is_unsupported());
    match error {
        Error::UnsupportedCudaRequest { reason } => {
            assert!(reason.contains("chunked entropy diagnostic"));
            assert!(reason.contains("config"));
        }
        other => panic!("expected unsupported CUDA diagnostic config error, got {other:?}"),
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_session_batch_records_owned_packet_cache_when_required() {
    if !cuda_jpeg_hardware_decode_required() {
        return;
    }

    let inputs = [BASELINE_420, BASELINE_420];
    let mut session = CudaSession::default();
    let surfaces = Codec::decode_tiles_to_device_with_session(
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
        &mut session,
    )
    .expect("cuda session batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    assert_eq!(session.owned_cuda_packet_cache_len(), 1);
    for surface in surfaces {
        let stats = surface.cuda_surface().expect("cuda surface").stats();
        assert!(stats.used_owned_cuda_decode());
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_decodes_into_caller_owned_buffer_when_required() {
    if !cuda_jpeg_hardware_decode_required() {
        return;
    }

    let mut session = CudaSession::default();
    let pitch = 16 * PixelFormat::Rgb8.bytes_per_pixel();
    let byte_len = pitch * 16;
    let buffer = session
        .take_owned_cuda_output_buffer(byte_len)
        .expect("device output buffer");

    let stats = Codec::decode_tile_rgb8_into_cuda_buffer_with_session(
        BASELINE_420,
        &buffer,
        pitch,
        &mut session,
    )
    .expect("direct owned CUDA decode");

    assert!(stats.used_owned_cuda_decode());
    assert_eq!(session.owned_cuda_packet_cache_len(), 1);

    let mut downloaded = vec![0u8; byte_len];
    buffer
        .copy_to_host(&mut downloaded)
        .expect("download buffer");
    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode(PixelFormat::Rgb8)
        .expect("host decode");
    let max_delta = downloaded
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.abs_diff(expected))
        .max()
        .unwrap_or(0);
    assert!(
        max_delta <= OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA,
        "direct J2K-owned CUDA decode differed from the CPU reference by max channel delta {max_delta}"
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_decodes_422_and_444_into_caller_owned_buffers_when_required() {
    if !cuda_jpeg_hardware_decode_required() {
        return;
    }

    for (input, dimensions) in [
        (BASELINE_422, (16_u32, 8_u32)),
        (BASELINE_444, (8_u32, 8_u32)),
    ] {
        let mut session = CudaSession::default();
        let pitch = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let byte_len = pitch * dimensions.1 as usize;
        let buffer = session
            .take_owned_cuda_output_buffer(byte_len)
            .expect("device output buffer");

        let stats = Codec::decode_tile_rgb8_into_cuda_buffer_with_session(
            input,
            &buffer,
            pitch,
            &mut session,
        )
        .expect("direct owned CUDA decode");

        assert!(stats.used_owned_cuda_decode());
        assert_eq!(session.owned_cuda_packet_cache_len(), 1);

        let mut downloaded = vec![0u8; byte_len];
        buffer
            .copy_to_host(&mut downloaded)
            .expect("download buffer");
        let (expected, _) = j2k_jpeg::Decoder::new(input)
            .expect("host decoder")
            .decode(PixelFormat::Rgb8)
            .expect("host decode");
        let max_delta = downloaded
            .iter()
            .zip(expected)
            .map(|(actual, expected)| actual.abs_diff(expected))
            .max()
            .unwrap_or(0);
        assert!(
            max_delta <= OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA,
            "direct J2K-owned CUDA decode differed from the CPU reference by max channel delta {max_delta}"
        );
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_decodes_batch_into_caller_owned_buffers_when_required() {
    if !cuda_jpeg_hardware_decode_required() {
        return;
    }

    let cases = [
        (BASELINE_420, (16_u32, 16_u32)),
        (BASELINE_422, (16_u32, 8_u32)),
    ];
    let mut session = CudaSession::default();
    let buffers = cases
        .iter()
        .map(|(_, dimensions)| {
            let pitch = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            session
                .take_owned_cuda_output_buffer(pitch * dimensions.1 as usize)
                .expect("device output buffer")
        })
        .collect::<Vec<_>>();
    let tiles = cases
        .iter()
        .zip(buffers.iter())
        .map(
            |((input, dimensions), buffer)| j2k_jpeg_cuda::CudaJpegDecodeOutputTile {
                input,
                output: buffer,
                pitch_bytes: dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel(),
            },
        )
        .collect::<Vec<_>>();

    let stats = Codec::decode_tiles_rgb8_into_cuda_buffers_with_session(&tiles, &mut session)
        .expect("direct owned CUDA batch decode");

    assert_eq!(stats.len(), cases.len());
    assert_eq!(session.owned_cuda_packet_cache_len(), cases.len());
    for ((input, dimensions), (buffer, stats)) in cases.iter().zip(buffers.iter().zip(stats)) {
        assert!(stats.used_owned_cuda_decode());
        let pitch = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let mut downloaded = vec![0u8; pitch * dimensions.1 as usize];
        buffer
            .copy_to_host(&mut downloaded)
            .expect("download buffer");
        let (expected, _) = j2k_jpeg::Decoder::new(input)
            .expect("host decoder")
            .decode(PixelFormat::Rgb8)
            .expect("host decode");
        let max_delta = downloaded
            .iter()
            .zip(expected)
            .map(|(actual, expected)| actual.abs_diff(expected))
            .max()
            .unwrap_or(0);
        assert!(
            max_delta <= OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA,
            "direct J2K-owned CUDA batch decode differed from the CPU reference by max channel delta {max_delta}"
        );
    }
}
