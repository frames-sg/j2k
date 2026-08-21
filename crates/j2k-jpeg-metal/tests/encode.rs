use j2k_core::CodecError;
#[cfg(target_os = "macos")]
use j2k_jpeg::DecodeRequest;
use j2k_jpeg::{JpegBackend, JpegEncodeError, JpegSubsampling};

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
            error: JpegEncodeError::InternalInvariant {
                reason: "entropy overflow",
            },
            unsupported: false,
            buffer_error: false,
        },
    ]
}

#[cfg(target_os = "macos")]
fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

#[test]
fn metal_encode_errors_match_cuda_classification_contract() {
    for case in encode_classification_cases() {
        let err = j2k_jpeg_metal::Error::from(case.error);
        assert_eq!(err.is_unsupported(), case.unsupported, "{err:?}");
        assert_eq!(err.is_buffer_error(), case.buffer_error, "{err:?}");
        assert!(!err.is_truncated(), "{err:?}");
        assert!(!err.is_not_implemented(), "{err:?}");
    }
}

#[cfg(target_os = "macos")]
fn assert_independent_decoder_accepts(
    encoded: &[u8],
    width: u32,
    height: u32,
    expected_format: jpeg_decoder::PixelFormat,
) {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(encoded));
    let pixels = decoder.decode().expect("jpeg-decoder accepts Metal JPEG");
    let info = decoder.info().expect("jpeg-decoder exposes frame info");
    assert_eq!(
        (u32::from(info.width), u32::from(info.height)),
        (width, height)
    );
    assert_eq!(info.pixel_format, expected_format);
    let expected_components = match expected_format {
        jpeg_decoder::PixelFormat::L8 => 1usize,
        jpeg_decoder::PixelFormat::RGB24 => 3usize,
        jpeg_decoder::PixelFormat::CMYK32 => 4usize,
        jpeg_decoder::PixelFormat::L16 => 2usize,
    };
    assert_eq!(
        pixels.len(),
        width as usize * height as usize * expected_components
    );
}

#[cfg(target_os = "macos")]
fn assert_entropy_byte_stuffing_and_restart_markers(encoded: &[u8]) {
    let sos = encoded
        .windows(2)
        .position(|bytes| bytes == [0xff, 0xda])
        .expect("JPEG contains SOS");
    let header_len = usize::from(u16::from_be_bytes([encoded[sos + 2], encoded[sos + 3]]));
    let mut index = sos + 2 + header_len;
    while index + 1 < encoded.len() {
        if encoded[index] != 0xff {
            index += 1;
            continue;
        }
        let marker = encoded[index + 1];
        assert!(
            marker == 0x00 || (0xd0..=0xd7).contains(&marker) || marker == 0xd9,
            "entropy 0xff byte must be stuffed or introduce restart/EOI marker, got 0x{marker:02x}"
        );
        if marker == 0xd9 {
            return;
        }
        index += 2;
    }
    panic!("JPEG entropy scan did not reach EOI");
}

#[cfg(target_os = "macos")]
#[test]
fn metal_baseline_encoder_round_trips_rgb_422() {
    use j2k_core::PixelFormat;
    use j2k_jpeg::{Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use j2k_jpeg_metal::{
        encode_jpeg_baseline_from_metal_buffer, JpegBaselineMetalEncodeTile, MetalBackendSession,
    };

    if !should_run_metal_runtime() {
        return;
    }

    let width = 19u32;
    let height = 17u32;
    let rgb = j2k_test_support::patterned_rgb8(width, height);

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let buffer = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &rgb)
        .expect("upload test RGB pixels");
    let layout = j2k_metal_support::MetalImageLayout::new(
        0,
        (width, height),
        width as usize * 3,
        PixelFormat::Rgb8,
    )
    .expect("valid resident JPEG layout");
    // SAFETY: the upload completed synchronously and the raw buffer is moved
    // into the immutable resident owner without a surviving writable alias.
    let resident =
        unsafe { j2k_metal_support::ResidentMetalImage::from_completed_buffer(buffer, layout) }
            .expect("resident JPEG input");
    let tile = JpegBaselineMetalEncodeTile::from_resident(&resident, (width, height));
    let encoded = encode_jpeg_baseline_from_metal_buffer(
        tile,
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Metal,
        },
        &session,
    )
    .expect("Metal JPEG baseline encode");

    assert_eq!(encoded.backend, JpegBackend::Metal);
    assert!(encoded.data.starts_with(&[0xff, 0xd8]));
    assert!(encoded.data.ends_with(&[0xff, 0xd9]));

    let decoder = Decoder::new(&encoded.data).expect("parse Metal-encoded JPEG");
    let (pixels, outcome) = decoder
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("decode Metal-encoded JPEG");

    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    assert_eq!(pixels.len(), rgb.len());
    assert_independent_decoder_accepts(
        &encoded.data,
        width,
        height,
        jpeg_decoder::PixelFormat::RGB24,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_baseline_encoder_round_trips_all_rgb_subsampling_modes() {
    use j2k_core::PixelFormat;
    use j2k_jpeg::{Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use j2k_jpeg_metal::{
        encode_jpeg_baseline_from_metal_buffer, JpegBaselineMetalEncodeTile, MetalBackendSession,
    };

    if !should_run_metal_runtime() {
        return;
    }

    let width = 23u32;
    let height = 19u32;
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.push(((x * 29 + y * 3 + 11) & 0xff) as u8);
            rgb.push(((x * 7 + y * 17 + 40) & 0xff) as u8);
            rgb.push(((x * 13 + y * 5 + 90) & 0xff) as u8);
        }
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let buffer = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &rgb)
        .expect("upload test RGB pixels");

    for subsampling in [
        JpegSubsampling::Ybr444,
        JpegSubsampling::Ybr422,
        JpegSubsampling::Ybr420,
    ] {
        // SAFETY: the buffer was initialized before tile construction and no
        // CPU or GPU writer accesses it while the tile is alive.
        let tile = unsafe {
            JpegBaselineMetalEncodeTile::new(
                &buffer,
                0,
                (width, height),
                width as usize * 3,
                (width, height),
                PixelFormat::Rgb8,
            )
        };
        let encoded = encode_jpeg_baseline_from_metal_buffer(
            tile,
            JpegEncodeOptions {
                quality: 88,
                subsampling,
                restart_interval: Some(5),
                backend: JpegBackend::Metal,
            },
            &session,
        )
        .expect("Metal JPEG baseline encode");

        assert_eq!(encoded.backend, JpegBackend::Metal);
        let decoder = Decoder::new(&encoded.data).expect("parse Metal-encoded JPEG");
        let (pixels, outcome) = decoder
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("decode Metal-encoded JPEG");

        assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
        assert_eq!(pixels.len(), rgb.len());
        assert_independent_decoder_accepts(
            &encoded.data,
            width,
            height,
            jpeg_decoder::PixelFormat::RGB24,
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_baseline_encoder_round_trips_gray_with_padded_output() {
    use j2k_core::PixelFormat;
    use j2k_jpeg::{Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use j2k_jpeg_metal::{
        encode_jpeg_baseline_from_metal_buffer, JpegBaselineMetalEncodeTile, MetalBackendSession,
    };

    if !should_run_metal_runtime() {
        return;
    }

    let width = 7u32;
    let height = 5u32;
    let output_width = 13u32;
    let output_height = 11u32;
    let gray = j2k_test_support::patterned_gray8(width, height);

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let buffer = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &gray)
        .expect("upload test grayscale pixels");

    // SAFETY: the buffer was initialized before tile construction and no CPU
    // or GPU writer accesses it while the tile is alive.
    let tile = unsafe {
        JpegBaselineMetalEncodeTile::new(
            &buffer,
            0,
            (width, height),
            width as usize,
            (output_width, output_height),
            PixelFormat::Gray8,
        )
    };
    let encoded = encode_jpeg_baseline_from_metal_buffer(
        tile,
        JpegEncodeOptions {
            quality: 85,
            subsampling: JpegSubsampling::Gray,
            restart_interval: Some(3),
            backend: JpegBackend::Metal,
        },
        &session,
    )
    .expect("Metal JPEG baseline encode");

    assert_eq!(encoded.backend, JpegBackend::Metal);
    let decoder = Decoder::new(&encoded.data).expect("parse Metal-encoded gray JPEG");
    let (pixels, outcome) = decoder
        .decode_request(DecodeRequest::full(PixelFormat::Gray8))
        .expect("decode Metal-encoded gray JPEG");

    assert_eq!(
        (outcome.decoded.w, outcome.decoded.h),
        (output_width, output_height)
    );
    assert_eq!(pixels.len(), output_width as usize * output_height as usize);
    assert_independent_decoder_accepts(
        &encoded.data,
        output_width,
        output_height,
        jpeg_decoder::PixelFormat::L8,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_baseline_batch_encoder_round_trips_multiple_rgb_tiles() {
    use j2k_core::PixelFormat;
    use j2k_jpeg::{Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use j2k_jpeg_metal::{
        encode_jpeg_baseline_batch_from_metal_buffers, JpegBaselineMetalEncodeTile,
        MetalBackendSession,
    };

    if !should_run_metal_runtime() {
        return;
    }

    let width = 32u32;
    let height = 24u32;
    let tile_count_u32 = 3u32;
    let tile_count = tile_count_u32 as usize;
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3 * tile_count);
    for tile in 0..tile_count_u32 {
        for y in 0..height {
            for x in 0..width {
                rgb.push(((x * 11 + y * 7 + tile * 31) & 0xff) as u8);
                rgb.push(((x * 5 + y * 17 + tile * 19) & 0xff) as u8);
                rgb.push(((x * 23 + y * 3 + tile * 13) & 0xff) as u8);
            }
        }
    }

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let buffer = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &rgb)
        .expect("upload test RGB pixels");
    let tile_bytes = width as usize * height as usize * 3;
    // SAFETY: the buffer was initialized before tile construction and no CPU
    // or GPU writer accesses any described range while the tiles are alive.
    let tiles: Vec<_> = (0..tile_count)
        .map(|tile| unsafe {
            JpegBaselineMetalEncodeTile::new(
                &buffer,
                tile * tile_bytes,
                (width, height),
                width as usize * 3,
                (width, height),
                PixelFormat::Rgb8,
            )
        })
        .collect();

    let encoded = encode_jpeg_baseline_batch_from_metal_buffers(
        &tiles,
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: Some(4),
            backend: JpegBackend::Metal,
        },
        &session,
    )
    .expect("Metal JPEG baseline batch encode");

    assert_eq!(encoded.len(), tile_count);
    for frame in encoded {
        assert_eq!(frame.backend, JpegBackend::Metal);
        let decoder = Decoder::new(&frame.data).expect("parse Metal batch JPEG");
        let (pixels, outcome) = decoder
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("decode Metal batch JPEG");
        assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
        assert_eq!(pixels.len(), tile_bytes);
        assert_independent_decoder_accepts(
            &frame.data,
            width,
            height,
            jpeg_decoder::PixelFormat::RGB24,
        );
    }
}

#[cfg(target_os = "macos")]
fn update_p18_rgb_matrix_digest(
    session: &j2k_jpeg_metal::MetalBackendSession,
    digest: &mut sha2::Sha256,
) {
    use j2k_core::PixelFormat;
    use j2k_jpeg::{JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use j2k_jpeg_metal::{
        encode_jpeg_baseline_batch_from_metal_buffers, JpegBaselineMetalEncodeTile,
    };
    use sha2::Digest;

    let rgb_size = (23u32, 19u32);
    let rgb_tile_count = 2usize;
    let rgb_tile_bytes = rgb_size.0 as usize * rgb_size.1 as usize * 3;
    let rgb = j2k_test_support::patterned_rgb8_tiles(rgb_size.0, rgb_size.1, rgb_tile_count);
    let rgb_buffer = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &rgb)
        .expect("upload P18 RGB differential matrix");
    let rgb_tiles = (0..rgb_tile_count)
        .map(|tile| {
            // SAFETY: each descriptor names an initialized, disjoint immutable tile range
            // retained by `rgb_buffer` through every synchronous encode below.
            unsafe {
                JpegBaselineMetalEncodeTile::new(
                    &rgb_buffer,
                    tile * rgb_tile_bytes,
                    rgb_size,
                    rgb_size.0 as usize * 3,
                    rgb_size,
                    PixelFormat::Rgb8,
                )
            }
        })
        .collect::<Vec<_>>();
    for (subsampling, restart_interval, quality) in [
        (JpegSubsampling::Ybr444, None, 1u8),
        (JpegSubsampling::Ybr444, Some(5), 90),
        (JpegSubsampling::Ybr422, None, 90),
        (JpegSubsampling::Ybr422, Some(5), 100),
        (JpegSubsampling::Ybr420, None, 100),
        (JpegSubsampling::Ybr420, Some(5), 1),
    ] {
        let options = JpegEncodeOptions {
            quality,
            subsampling,
            restart_interval,
            backend: JpegBackend::Metal,
        };
        let frames = encode_jpeg_baseline_batch_from_metal_buffers(&rgb_tiles, options, session)
            .expect("P18 RGB differential matrix encode");
        let repeat = encode_jpeg_baseline_batch_from_metal_buffers(&rgb_tiles, options, session)
            .expect("repeat P18 RGB differential matrix encode");
        assert_eq!(
            frames.iter().map(|frame| &frame.data).collect::<Vec<_>>(),
            repeat.iter().map(|frame| &frame.data).collect::<Vec<_>>(),
            "P18 RGB route must be deterministic for {subsampling:?}, restart={restart_interval:?}, quality={quality}"
        );
        for frame in frames {
            assert_entropy_byte_stuffing_and_restart_markers(&frame.data);
            assert_independent_decoder_accepts(
                &frame.data,
                rgb_size.0,
                rgb_size.1,
                jpeg_decoder::PixelFormat::RGB24,
            );
            digest.update((frame.data.len() as u64).to_le_bytes());
            digest.update(&frame.data);
        }
    }
}

#[cfg(target_os = "macos")]
fn update_p18_gray_matrix_digest(
    session: &j2k_jpeg_metal::MetalBackendSession,
    digest: &mut sha2::Sha256,
) {
    use j2k_core::PixelFormat;
    use j2k_jpeg::{JpegBackend, JpegEncodeOptions, JpegSubsampling};
    use j2k_jpeg_metal::{
        encode_jpeg_baseline_batch_from_metal_buffers, JpegBaselineMetalEncodeTile,
    };
    use sha2::Digest;

    let gray_input = (7u32, 5u32);
    let gray_output = (13u32, 11u32);
    let gray_tile_count = 2usize;
    let gray_tile_bytes = gray_input.0 as usize * gray_input.1 as usize;
    let mut gray = Vec::new();
    for salt in 0..gray_tile_count {
        let salt = u8::try_from(salt).expect("P18 grayscale tile salt fits u8");
        gray.extend(
            j2k_test_support::patterned_gray8(gray_input.0, gray_input.1)
                .into_iter()
                .map(|sample| sample.wrapping_add(salt)),
        );
    }
    let gray_buffer = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &gray)
        .expect("upload P18 grayscale differential matrix");
    let gray_tiles = (0..gray_tile_count)
        .map(|tile| {
            // SAFETY: each descriptor names an initialized, disjoint immutable tile range
            // retained by `gray_buffer` through every synchronous encode below.
            unsafe {
                JpegBaselineMetalEncodeTile::new(
                    &gray_buffer,
                    tile * gray_tile_bytes,
                    gray_input,
                    gray_input.0 as usize,
                    gray_output,
                    PixelFormat::Gray8,
                )
            }
        })
        .collect::<Vec<_>>();
    for (quality, restart_interval) in [(1u8, None), (100, Some(3))] {
        let options = JpegEncodeOptions {
            quality,
            subsampling: JpegSubsampling::Gray,
            restart_interval,
            backend: JpegBackend::Metal,
        };
        let frames = encode_jpeg_baseline_batch_from_metal_buffers(&gray_tiles, options, session)
            .expect("P18 grayscale differential matrix encode");
        let repeat = encode_jpeg_baseline_batch_from_metal_buffers(&gray_tiles, options, session)
            .expect("repeat P18 grayscale differential matrix encode");
        assert_eq!(
            frames.iter().map(|frame| &frame.data).collect::<Vec<_>>(),
            repeat.iter().map(|frame| &frame.data).collect::<Vec<_>>(),
            "P18 grayscale route must be deterministic for restart={restart_interval:?}, quality={quality}"
        );
        for frame in frames {
            assert_entropy_byte_stuffing_and_restart_markers(&frame.data);
            assert_independent_decoder_accepts(
                &frame.data,
                gray_output.0,
                gray_output.1,
                jpeg_decoder::PixelFormat::L8,
            );
            digest.update((frame.data.len() as u64).to_le_bytes());
            digest.update(&frame.data);
        }
    }
}

#[cfg(target_os = "macos")]
fn p18_promoted_route_matrix_digest() -> String {
    use sha2::{Digest, Sha256};

    let session =
        j2k_jpeg_metal::MetalBackendSession::system_default().expect("Metal backend session");
    let mut digest = Sha256::new();
    update_p18_rgb_matrix_digest(&session, &mut digest);
    update_p18_gray_matrix_digest(&session, &mut digest);

    format!("{:x}", digest.finalize())
}

#[cfg(target_os = "macos")]
#[test]
fn promoted_staged_metal_encoder_matches_reviewed_p18_matrix_hash() {
    if !should_run_metal_runtime() {
        return;
    }
    assert_eq!(
        p18_promoted_route_matrix_digest(),
        "b9af03a1a522926f3bee386fd554f146db773cf8287cef91d9001335c88c3a26",
        "promoted staged route must match the reviewed fused/staged P18 matrix hash"
    );
}
