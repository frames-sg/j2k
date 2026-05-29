use signinum_core::{
    BackendKind, BackendRequest, CodecError, DecoderContext, DeviceSubmission, DeviceSurface,
    Downscale, PixelFormat, Rect, TileBatchDecodeDevice, TileBatchDecodeSubmit,
};
use signinum_jpeg::{Decoder as CpuDecoder, DecoderContext as JpegDecoderContext};
#[cfg(target_os = "macos")]
use signinum_jpeg_metal::JpegTileBatch;
use signinum_jpeg_metal::{Codec, MetalSession, ScratchPool};

const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const BASELINE_420_RESTART: &[u8] =
    include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");
const GRAYSCALE: &[u8] = include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg");

#[cfg(target_os = "macos")]
fn same_length_distinct_baseline_420() -> Vec<u8> {
    let (baseline, _) = CpuDecoder::new(BASELINE_420)
        .expect("baseline decoder")
        .decode(PixelFormat::Rgb8)
        .expect("baseline decode");

    let scan_start = BASELINE_420
        .windows(2)
        .position(|window| window == [0xFF, 0xDA])
        .expect("SOS marker")
        + 2;
    let entropy_start = scan_start
        + 2
        + u16::from_be_bytes([BASELINE_420[scan_start], BASELINE_420[scan_start + 1]]) as usize;
    let entropy_end = BASELINE_420
        .windows(2)
        .rposition(|window| window == [0xFF, 0xD9])
        .expect("EOI marker");

    for index in entropy_start..entropy_end {
        for mask in [0x01_u8, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40] {
            let mut candidate = BASELINE_420.to_vec();
            candidate[index] ^= mask;
            let Ok(candidate_decoder) = CpuDecoder::new(&candidate) else {
                continue;
            };
            let Ok((candidate_pixels, _)) = candidate_decoder.decode(PixelFormat::Rgb8) else {
                continue;
            };
            if candidate_pixels != baseline {
                return candidate;
            }
        }
    }

    panic!("could not construct same-length distinct JPEG fixture");
}

#[cfg(target_os = "macos")]
#[test]
fn tile_device_decode_matches_host_tile_decode() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let surface = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("tile device decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download");
    assert_eq!(downloaded, surface.as_bytes());
    assert_eq!(downloaded, expected);
}

#[cfg(target_os = "macos")]
#[test]
fn tile_scaled_device_decode_has_expected_dimensions() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_scaled(PixelFormat::Rgb8, Downscale::Quarter)
        .expect("cpu scaled decode");
    let surface = Codec::decode_tile_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        Downscale::Quarter,
        BackendRequest::Metal,
    )
    .expect("tile scaled device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn tile_region_device_decode_has_expected_dimensions() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_region_scaled(
            PixelFormat::Rgb8,
            signinum_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            Downscale::None,
        )
        .expect("cpu region decode");
    let surface = Codec::decode_tile_region_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        BackendRequest::Metal,
    )
    .expect("tile region device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (8, 8));
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn compatible_tile_submits_flush_once() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let submissions = (0..4)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    assert_eq!(session.submissions(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn reused_mutable_input_slice_is_copied_per_submission() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let mut input = BASELINE_420.to_vec();
    let second_input = same_length_distinct_baseline_420();

    let (expected_first, _) = CpuDecoder::new(&input)
        .expect("first decoder")
        .decode(PixelFormat::Rgb8)
        .expect("first decode");
    let (expected_second, _) = CpuDecoder::new(&second_input)
        .expect("second decoder")
        .decode(PixelFormat::Rgb8)
        .expect("second decode");
    assert_ne!(expected_first, expected_second);

    let first = <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        &input,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("first submit");

    input.copy_from_slice(&second_input);
    let second = <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        &input,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("second submit");

    let second_surface = second.wait().expect("second wait");
    let first_surface = first.wait().expect("first wait");

    assert_eq!(first_surface.as_bytes(), expected_first.as_slice());
    assert_eq!(second_surface.as_bytes(), expected_second.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn jpeg_tile_batch_api_decodes_full_tiles_in_submission_order() {
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let mut batch = JpegTileBatch::with_capacity(2);

    assert!(batch.is_empty());
    assert_eq!(
        batch
            .push_tile(BASELINE_420, PixelFormat::Rgb8, BackendRequest::Metal)
            .expect("first push"),
        0
    );
    assert_eq!(
        batch
            .push_tile(BASELINE_420, PixelFormat::Rgb8, BackendRequest::Metal)
            .expect("second push"),
        1
    );
    assert_eq!(batch.len(), 2);
    assert_eq!(batch.submissions(), 0);

    let surfaces = batch.decode_all().expect("decode JPEG tile batch");

    assert_eq!(surfaces.len(), 2);
    for surface in surfaces {
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (16, 16));
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn auto_small_restart_tile_batch_stays_cpu_surface() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let (expected, _) = CpuDecoder::new(BASELINE_420_RESTART)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let submissions = (0..7)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420_RESTART,
                PixelFormat::Rgb8,
                BackendRequest::Auto,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Cpu);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    assert_eq!(session.submissions(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_restart_wsi_tile_batch_uses_metal_at_threshold() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let (expected, _) = CpuDecoder::new(BASELINE_420_RESTART)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let submissions = (0..8)
        .map(|_| {
            <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420_RESTART,
                PixelFormat::Rgb8,
                BackendRequest::Auto,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    assert_eq!(session.submissions(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn compatible_region_scaled_tile_submits_flush_once() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let scale = Downscale::Quarter;
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_region_scaled(
            PixelFormat::Rgb8,
            signinum_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("cpu region scaled");

    let submissions = (0..4)
        .map(|_| {
            Codec::submit_tile_region_scaled_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420,
                PixelFormat::Rgb8,
                roi,
                scale,
                BackendRequest::Metal,
            )
            .expect("submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (2, 2));
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    assert_eq!(session.submissions(), 1);
}

#[test]
fn auto_tile_region_scaled_unsupported_metal_shape_returns_cpu_surface() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };

    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
        BackendRequest::Auto,
    )
    .expect("auto tile region scaled surface");

    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (2, 2));
    #[cfg(target_os = "macos")]
    assert!(surface.metal_buffer().is_none());
}

#[test]
fn explicit_metal_tile_unsupported_shape_is_rejected() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let result = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        GRAYSCALE,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    );

    match result {
        Err(signinum_jpeg_metal::Error::UnsupportedMetalRequest { reason }) => {
            assert!(reason.contains("JPEG Metal"));
        }
        Err(other) => panic!("unexpected explicit Metal tile error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal tile request must not fall back; got {:?}",
            surface.backend_kind()
        ),
    }
}

#[test]
fn explicit_metal_tile_unsupported_error_is_codec_unsupported() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let err = match Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        GRAYSCALE,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    ) {
        Err(err) => err,
        Ok(surface) => panic!(
            "explicit Metal tile request must not fall back; got {:?}",
            surface.backend_kind()
        ),
    };

    assert!(err.is_unsupported());
}

#[test]
fn cuda_tile_request_remains_unsupported_backend() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let result = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    );

    match result {
        Err(signinum_jpeg_metal::Error::UnsupportedBackend {
            request: BackendRequest::Cuda,
        }) => {}
        Err(signinum_jpeg_metal::Error::UnsupportedMetalRequest { reason }) => {
            panic!("CUDA must not be reported as a Metal request: {reason}")
        }
        Err(other) => panic!("unexpected CUDA tile error: {other:?}"),
        Ok(surface) => panic!(
            "CUDA tile request unexpectedly returned {:?}",
            surface.backend_kind()
        ),
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn non_macos_explicit_metal_tile_decode_is_unavailable() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let result = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    );

    assert!(matches!(
        result,
        Err(signinum_jpeg_metal::Error::MetalUnavailable)
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn incompatible_shapes_split_batches() {
    let mut ctx = DecoderContext::<JpegDecoderContext>::new();
    let mut pool = ScratchPool::new();
    let mut session = MetalSession::default();

    let full = <Codec as TileBatchDecodeSubmit>::submit_tile_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("full");
    let scaled = <Codec as TileBatchDecodeSubmit>::submit_tile_scaled_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        Downscale::Quarter,
        BackendRequest::Metal,
    )
    .expect("scaled");

    let _ = full.wait().expect("full wait");
    let _ = scaled.wait().expect("scaled wait");

    assert_eq!(session.submissions(), 2);
}
