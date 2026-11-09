use j2k_jpeg::{
    encode_jpeg_baseline, encode_jpeg_baseline_optimized, DecodeRequest, Decoder, EncodedJpeg,
    JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSamples, JpegSubsampling, PixelFormat,
};
use j2k_test_support::{
    jpeg_encode_matrix_cases, jpeg_encode_matrix_digest, jpeg_entropy_segment,
    jpeg_entropy_stage_cases, jpeg_entropy_stage_digest, jpeg_standard_tables, patterned_gray8,
    patterned_rgb8, JpegEncodeMatrixCase, JpegEncodeMatrixSampling, JPEG_ENCODE_MATRIX_SHA256,
    JPEG_ENTROPY_STAGE_SHA256,
};
use std::io::Cursor;

fn encode_rgb(subsampling: JpegSubsampling) -> EncodedJpeg {
    let width = 19;
    let height = 17;
    let rgb = patterned_rgb8(width, height);
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode RGB baseline JPEG")
}

fn assert_independent_decoder_accepts(
    encoded: &[u8],
    width: u32,
    height: u32,
    expected_format: jpeg_decoder::PixelFormat,
) {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(encoded));
    let decoded = decoder.decode().expect("jpeg-decoder accepts encoded JPEG");
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
        decoded.len(),
        width as usize * height as usize * expected_components
    );
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn matrix_subsampling(sampling: JpegEncodeMatrixSampling) -> JpegSubsampling {
    match sampling {
        JpegEncodeMatrixSampling::Gray => JpegSubsampling::Gray,
        JpegEncodeMatrixSampling::Ybr444 => JpegSubsampling::Ybr444,
        JpegEncodeMatrixSampling::Ybr422 => JpegSubsampling::Ybr422,
        JpegEncodeMatrixSampling::Ybr420 => JpegSubsampling::Ybr420,
    }
}

fn encode_matrix_case(case: JpegEncodeMatrixCase) -> Vec<u8> {
    let data = case.samples();
    let (width, height) = (case.width, case.height);
    let samples = if case.sampling == JpegEncodeMatrixSampling::Gray {
        JpegSamples::Gray8 {
            data: &data,
            width,
            height,
        }
    } else {
        JpegSamples::Rgb8 {
            data: &data,
            width,
            height,
        }
    };
    let options = JpegEncodeOptions {
        quality: case.quality,
        subsampling: matrix_subsampling(case.sampling),
        restart_interval: case.restart_interval,
        backend: JpegBackend::Cpu,
    };
    encode_jpeg_baseline(samples, options)
        .unwrap_or_else(|error| panic!("{}: {error}", case.label()))
        .data
}

/// The CPU encoder produces the bytes every baseline backend produces. Its
/// SIMD and scalar transforms all yield the integer coefficients of
/// `j2k_codec_math::jpeg::fdct`, so the golden holds on every target.
#[test]
fn baseline_encoder_matches_the_cross_backend_golden() {
    let frames = jpeg_encode_matrix_cases()
        .into_iter()
        .map(encode_matrix_case)
        .collect::<Vec<_>>();
    let digest = jpeg_encode_matrix_digest(frames.iter().map(Vec::as_slice));
    assert_eq!(
        digest, JPEG_ENCODE_MATRIX_SHA256,
        "baseline JPEG codestream bytes changed"
    );
}

/// The GPU entropy-stage tests that cannot call the CPU encoder take their
/// tables from `jpeg_standard_tables` and check `JPEG_ENTROPY_STAGE_SHA256`;
/// both are the CPU encoder's.
#[test]
fn entropy_stage_tables_and_golden_are_the_cpu_encoders() {
    for quality in [20, 100] {
        let tables = j2k_jpeg::adapter::baseline_encode_tables(JpegEncodeOptions {
            quality,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        })
        .expect("standard tables");
        let standard = jpeg_standard_tables(quality);
        assert_eq!(tables.q_luma, standard.q_luma, "quality {quality}");
        assert_eq!(tables.q_chroma, standard.q_chroma, "quality {quality}");
        let huffman = [
            tables.huff_dc_luma,
            tables.huff_ac_luma,
            tables.huff_dc_chroma,
            tables.huff_ac_chroma,
        ]
        .map(|table| (table.codes, table.lens));
        assert_eq!(huffman, standard.huffman, "quality {quality}");
    }
    let frames = jpeg_entropy_stage_cases()
        .into_iter()
        .map(encode_matrix_case)
        .collect::<Vec<_>>();
    assert_eq!(
        jpeg_entropy_stage_digest(frames.iter().map(|frame| jpeg_entropy_segment(frame))),
        JPEG_ENTROPY_STAGE_SHA256,
        "entropy-stage segments changed"
    );
}

#[test]
fn cpu_encoder_round_trips_rgb_444_422_420() {
    for subsampling in [
        JpegSubsampling::Ybr444,
        JpegSubsampling::Ybr422,
        JpegSubsampling::Ybr420,
    ] {
        let encoded = encode_rgb(subsampling);
        assert_eq!(encoded.backend, JpegBackend::Cpu);
        assert!(encoded.data.starts_with(&[0xFF, 0xD8]));
        assert!(encoded.data.ends_with(&[0xFF, 0xD9]));

        let decoder = Decoder::new(&encoded.data).expect("parse encoded JPEG");
        let (decoded, outcome) = decoder
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("decode RGB JPEG");

        assert_eq!((outcome.decoded.w, outcome.decoded.h), (19, 17));
        assert_eq!(decoded.len(), 19 * 17 * 3);
        assert_independent_decoder_accepts(&encoded.data, 19, 17, jpeg_decoder::PixelFormat::RGB24);
    }
}

#[test]
fn cpu_encoder_round_trips_gray_and_writes_required_markers() {
    let width = 13;
    let height = 11;
    let gray = patterned_gray8(width, height);
    let encoded = encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &gray,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 85,
            subsampling: JpegSubsampling::Gray,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode gray JPEG");

    for marker in [
        [0xFF, 0xDB],
        [0xFF, 0xC4],
        [0xFF, 0xC0],
        [0xFF, 0xDA],
        [0xFF, 0xDD],
    ] {
        assert!(
            encoded.data.windows(2).any(|window| window == marker),
            "missing marker {:02X}{:02X}",
            marker[0],
            marker[1]
        );
    }
    assert!(
        !encoded
            .data
            .windows(3)
            .any(|window| window[0] == 0xFF && window[1] == 0xFF && window[2] != 0x00),
        "entropy/header should not contain unstuffed fill-marker pairs"
    );

    let decoder = Decoder::new(&encoded.data).expect("parse encoded gray JPEG");
    let (decoded, outcome) = decoder
        .decode_request(DecodeRequest::full(PixelFormat::Gray8))
        .expect("decode gray JPEG");

    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    assert_eq!(decoded.len(), width as usize * height as usize);
    assert_independent_decoder_accepts(&encoded.data, width, height, jpeg_decoder::PixelFormat::L8);
}

fn decode(encoded: &[u8], fmt: PixelFormat) -> Vec<u8> {
    Decoder::new(encoded)
        .expect("decoder")
        .decode_request(DecodeRequest::full(fmt))
        .expect("decode")
        .0
}

fn count_restart_markers(encoded: &[u8]) -> usize {
    encoded
        .windows(2)
        .filter(|pair| pair[0] == 0xFF && (0xD0..=0xD7).contains(&pair[1]))
        .count()
}

#[test]
fn optimized_huffman_keeps_pixels_and_shrinks_output() {
    let (width, height) = (61, 43);
    let rgb = patterned_rgb8(width, height);
    let gray = patterned_gray8(width, height);
    // Pinned codestream hashes. Their DHT segments and entropy-coded data
    // were checked equal to `jpegtran -optimize -restart <N>B` (libjpeg-turbo
    // 3.1.3) applied to the standard-table encode of the same samples, after
    // the encoder adopted the integer forward DCT of
    // `j2k_codec_math::jpeg::fdct`.
    let cases = [
        (JpegSubsampling::Gray, None, 0xb411_8f53_ae46_300d_u64),
        (JpegSubsampling::Gray, Some(3), 0x824b_be39_a63d_cea2),
        (JpegSubsampling::Ybr444, None, 0xb865_e7b1_0ce3_2da2),
        (JpegSubsampling::Ybr422, Some(2), 0x10d1_13f9_13dc_8069),
        (JpegSubsampling::Ybr420, None, 0x3e0f_3daa_28c4_8563),
        (JpegSubsampling::Ybr420, Some(5), 0x4234_561f_a09c_c5c5),
    ];
    let mut hashes = Vec::new();
    for (subsampling, restart_interval, golden) in cases {
        let (samples, fmt) = if subsampling == JpegSubsampling::Gray {
            (
                JpegSamples::Gray8 {
                    data: &gray,
                    width,
                    height,
                },
                PixelFormat::Gray8,
            )
        } else {
            (
                JpegSamples::Rgb8 {
                    data: &rgb,
                    width,
                    height,
                },
                PixelFormat::Rgb8,
            )
        };
        let options = JpegEncodeOptions {
            quality: 85,
            subsampling,
            restart_interval,
            backend: JpegBackend::Cpu,
        };
        let standard = encode_jpeg_baseline(samples, options).expect("standard encode");
        let optimized = encode_jpeg_baseline_optimized(samples, options).expect("optimized encode");
        let label = format!("{subsampling:?} restart {restart_interval:?}");
        assert_eq!(optimized.backend, JpegBackend::Cpu);
        hashes.push((fnv1a64(&optimized.data), golden));
        assert_eq!(
            decode(&optimized.data, fmt),
            decode(&standard.data, fmt),
            "{label}"
        );
        assert!(
            optimized.data.len() < standard.data.len(),
            "{label}: {} >= {}",
            optimized.data.len(),
            standard.data.len()
        );
        assert_eq!(
            count_restart_markers(&optimized.data),
            count_restart_markers(&standard.data),
            "{label}"
        );
        let expected_format = if fmt == PixelFormat::Gray8 {
            jpeg_decoder::PixelFormat::L8
        } else {
            jpeg_decoder::PixelFormat::RGB24
        };
        assert_independent_decoder_accepts(&optimized.data, width, height, expected_format);
    }
    let (actual, expected): (Vec<_>, Vec<_>) = hashes.into_iter().unzip();
    assert_eq!(actual, expected, "optimized codestreams changed");
}

#[test]
fn optimized_huffman_rejects_gpu_backends() {
    let gray = patterned_gray8(8, 8);
    let result = encode_jpeg_baseline_optimized(
        JpegSamples::Gray8 {
            data: &gray,
            width: 8,
            height: 8,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Gray,
            restart_interval: None,
            backend: JpegBackend::Metal,
        },
    );
    assert!(matches!(
        result,
        Err(JpegEncodeError::UnsupportedBackend {
            backend: JpegBackend::Metal
        })
    ));
}
