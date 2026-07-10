use j2k_core::CodecError;
#[cfg(feature = "cuda-runtime")]
use j2k_core::PixelFormat;
use j2k_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSamples,
    JpegSubsampling,
};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::{DecodeRequest, Decoder};

struct EncodeClassificationCase {
    error: JpegEncodeError,
    unsupported: bool,
    buffer_error: bool,
}

fn encode_classification_cases() -> Vec<EncodeClassificationCase> {
    vec![
        EncodeClassificationCase {
            error: JpegEncodeError::EmptyDimensions,
            unsupported: false,
            buffer_error: true,
        },
        EncodeClassificationCase {
            error: JpegEncodeError::DimensionsTooLarge {
                width: 70_000,
                height: 8,
            },
            unsupported: false,
            buffer_error: true,
        },
        EncodeClassificationCase {
            error: JpegEncodeError::SampleLength {
                expected: 12,
                actual: 8,
            },
            unsupported: false,
            buffer_error: true,
        },
        EncodeClassificationCase {
            error: JpegEncodeError::IncompatibleSubsampling {
                subsampling: JpegSubsampling::Ybr420,
                samples: "Gray8",
            },
            unsupported: true,
            buffer_error: false,
        },
        EncodeClassificationCase {
            error: JpegEncodeError::InvalidRestartInterval,
            unsupported: false,
            buffer_error: false,
        },
        EncodeClassificationCase {
            error: JpegEncodeError::UnsupportedBackend {
                backend: JpegBackend::Cpu,
            },
            unsupported: true,
            buffer_error: false,
        },
        EncodeClassificationCase {
            error: JpegEncodeError::SegmentTooLarge { name: "APP0" },
            unsupported: false,
            buffer_error: false,
        },
        EncodeClassificationCase {
            error: JpegEncodeError::MissingHuffmanCode { symbol: 17 },
            unsupported: false,
            buffer_error: false,
        },
        EncodeClassificationCase {
            error: JpegEncodeError::Internal("entropy overflow".to_string()),
            unsupported: false,
            buffer_error: false,
        },
    ]
}

#[test]
fn cuda_encode_errors_match_adapter_classification_contract() {
    for case in encode_classification_cases() {
        let err = j2k_jpeg_cuda::Error::from(case.error);
        assert_eq!(err.is_unsupported(), case.unsupported, "{err:?}");
        assert_eq!(err.is_buffer_error(), case.buffer_error, "{err:?}");
        assert!(!err.is_truncated(), "{err:?}");
        assert!(!err.is_not_implemented(), "{err:?}");
    }
}

#[test]
fn cpu_jpeg_encoder_rejects_cuda_backend() {
    let pixels = patterned_rgb(8, 8);
    let error = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &pixels,
            width: 8,
            height: 8,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cuda,
        },
    )
    .expect_err("CPU crate must not silently encode CUDA requests");
    assert!(matches!(
        error,
        JpegEncodeError::UnsupportedBackend {
            backend: JpegBackend::Cuda
        }
    ));
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_resident_rgb8_encode_round_trips_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let width = 16;
    let height = 16;
    let pixels = patterned_rgb(width, height);
    let context = j2k_cuda_runtime::CudaContext::system_default().expect("CUDA context");
    let buffer = context.upload(&pixels).expect("upload rgb pixels");
    let mut session = j2k_jpeg_cuda::CudaSession::default();
    let encoded = j2k_jpeg_cuda::encode_jpeg_baseline_from_cuda_buffer(
        j2k_jpeg_cuda::JpegBaselineCudaEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize * 3,
            output_width: width,
            output_height: height,
            format: PixelFormat::Rgb8,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cuda,
        },
        &mut session,
    )
    .expect("CUDA resident JPEG encode");

    assert_eq!(encoded.backend, JpegBackend::Cuda);
    assert!(encoded.data.len() > 64);
    let (decoded, outcome) = Decoder::new(&encoded.data)
        .expect("decode CUDA JPEG")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("decode CUDA JPEG RGB8");
    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    assert_rgb_close(&decoded, &pixels, 40);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_resident_batch_encode_preserves_order_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let width = 16;
    let height = 16;
    let first = patterned_rgb(width, height);
    let second = shifted_rgb(width, height);
    let mut combined = first.clone();
    let second_offset = combined.len();
    combined.extend_from_slice(&second);
    let context = j2k_cuda_runtime::CudaContext::system_default().expect("CUDA context");
    let buffer = context.upload(&combined).expect("upload rgb pixels");
    let mut session = j2k_jpeg_cuda::CudaSession::default();
    let options = JpegEncodeOptions {
        quality: 90,
        subsampling: JpegSubsampling::Ybr444,
        restart_interval: Some(4),
        backend: JpegBackend::Cuda,
    };
    let tiles = [
        j2k_jpeg_cuda::JpegBaselineCudaEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize * 3,
            output_width: width,
            output_height: height,
            format: PixelFormat::Rgb8,
        },
        j2k_jpeg_cuda::JpegBaselineCudaEncodeTile {
            buffer: &buffer,
            byte_offset: second_offset,
            width,
            height,
            pitch_bytes: width as usize * 3,
            output_width: width,
            output_height: height,
            format: PixelFormat::Rgb8,
        },
    ];
    let encoded =
        j2k_jpeg_cuda::encode_jpeg_baseline_batch_from_cuda_buffers(&tiles, options, &mut session)
            .expect("CUDA resident JPEG batch encode");

    assert_eq!(encoded.len(), 2);
    for frame in &encoded {
        assert_eq!(frame.backend, JpegBackend::Cuda);
    }
    let (decoded_first, _) = Decoder::new(&encoded[0].data)
        .expect("decode first")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("decode first RGB8");
    let (decoded_second, _) = Decoder::new(&encoded[1].data)
        .expect("decode second")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("decode second RGB8");
    assert_rgb_close(&decoded_first, &first, 40);
    assert_rgb_close(&decoded_second, &second, 40);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_resident_encode_rejects_cpu_backend_without_fallback() {
    let width = 8;
    let height = 8;
    let pixels = patterned_rgb(width, height);
    let Ok(context) = j2k_cuda_runtime::CudaContext::system_default() else {
        return;
    };
    let buffer = context.upload(&pixels).expect("upload rgb pixels");
    let mut session = j2k_jpeg_cuda::CudaSession::default();
    let error = j2k_jpeg_cuda::encode_jpeg_baseline_from_cuda_buffer(
        j2k_jpeg_cuda::JpegBaselineCudaEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize * 3,
            output_width: width,
            output_height: height,
            format: PixelFormat::Rgb8,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
        &mut session,
    )
    .expect_err("explicit CPU backend must not fall back from CUDA resident encode");
    assert!(error.is_unsupported());
    assert!(!matches!(error, j2k_jpeg_cuda::Error::CudaUnavailable));
}

fn patterned_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 9 + y * 3 + 20) & 0xff) as u8);
            pixels.push(((x * 5 + y * 7 + 50) & 0xff) as u8);
            pixels.push(((x * 3 + y * 11 + 90) & 0xff) as u8);
        }
    }
    pixels
}

#[cfg(feature = "cuda-runtime")]
fn shifted_rgb(width: u32, height: u32) -> Vec<u8> {
    patterned_rgb(width, height)
        .into_iter()
        .enumerate()
        .map(|(idx, value)| {
            value.wrapping_add(u8::try_from(idx % 17).expect("modulo output fits in u8"))
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn assert_rgb_close(actual: &[u8], expected: &[u8], max_delta: u8) {
    assert_eq!(actual.len(), expected.len());
    let observed = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .max()
        .unwrap_or(0);
    assert!(
        observed <= max_delta,
        "decoded CUDA JPEG differed from source by max channel delta {observed}"
    );
}
