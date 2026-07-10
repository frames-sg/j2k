use j2k_jpeg::{
    encode_jpeg_baseline, DecodeRequest, Decoder, EncodedJpeg, JpegBackend, JpegEncodeOptions,
    JpegSamples, JpegSubsampling, PixelFormat,
};
use j2k_test_support::{patterned_gray8, patterned_rgb8};
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

#[test]
fn baseline_codestream_bytes_match_goldens() {
    let gray = patterned_gray8(13, 11);
    let gray = encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &gray,
            width: 13,
            height: 11,
        },
        JpegEncodeOptions {
            quality: 85,
            subsampling: JpegSubsampling::Gray,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode grayscale golden");
    let actual = [
        ("gray-restart", gray.data.len(), fnv1a64(&gray.data)),
        {
            let encoded = encode_rgb(JpegSubsampling::Ybr444);
            ("rgb-444", encoded.data.len(), fnv1a64(&encoded.data))
        },
        {
            let encoded = encode_rgb(JpegSubsampling::Ybr422);
            ("rgb-422", encoded.data.len(), fnv1a64(&encoded.data))
        },
        {
            let encoded = encode_rgb(JpegSubsampling::Ybr420);
            ("rgb-420", encoded.data.len(), fnv1a64(&encoded.data))
        },
    ];
    assert_eq!(
        actual,
        [
            ("gray-restart", 450, 0xb49e_4d37_8d96_2fa0),
            ("rgb-444", 1094, 0xe3db_ce49_8363_0bf8),
            ("rgb-422", 1000, 0x0837_394c_3c14_12b3),
            ("rgb-420", 977, 0xc966_bef1_7b8d_b2a8),
        ],
        "baseline JPEG codestream bytes changed"
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
