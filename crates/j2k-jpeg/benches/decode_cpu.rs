// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use j2k_jpeg::{
    encode_jpeg_baseline, Decoder, JpegBackend, JpegEncodeOptions, JpegError, JpegSamples,
    JpegSubsampling, PixelFormat, RowSink,
};
use j2k_test_support::{patterned_gray8, patterned_rgb8};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn zeroed_bytes(len: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .expect("reserve deterministic benchmark output");
    bytes.resize(len, 0);
    bytes
}

#[derive(Clone, Copy)]
enum DecodeMode {
    Buffer(PixelFormat),
    Rows,
}

struct DecodeCase {
    name: &'static str,
    width: u32,
    height: u32,
    bytes: Vec<u8>,
    mode: DecodeMode,
    expected_checksum: u64,
}

impl DecodeCase {
    fn new(name: &'static str, width: u32, height: u32, bytes: Vec<u8>, mode: DecodeMode) -> Self {
        let decoder = Decoder::new(&bytes).expect("generated benchmark JPEG must parse");
        assert_eq!(decoder.info().dimensions, (width, height));
        let expected_checksum = decode_checksum(&decoder, mode);
        assert_ne!(expected_checksum, FNV_OFFSET_BASIS);
        Self {
            name,
            width,
            height,
            bytes,
            mode,
            expected_checksum,
        }
    }

    fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}

fn bytes_per_pixel(format: PixelFormat) -> usize {
    match format {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb8 => 3,
        _ => panic!("decode benchmark only supports Gray8 and Rgb8"),
    }
}

fn decode_buffer_checksum(decoder: &Decoder<'_>, format: PixelFormat) -> u64 {
    let (width, height) = decoder.info().dimensions;
    let stride = width as usize * bytes_per_pixel(format);
    let mut output = zeroed_bytes(stride * height as usize);
    let outcome = decoder
        .decode_into(&mut output, stride, format)
        .expect("generated benchmark JPEG must decode");
    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    fnv1a_update(FNV_OFFSET_BASIS, &output)
}

struct ChecksumSink {
    hash: u64,
    expected_y: u32,
    row_bytes: usize,
}

impl ChecksumSink {
    fn new(row_bytes: usize) -> Self {
        Self {
            hash: FNV_OFFSET_BASIS,
            expected_y: 0,
            row_bytes,
        }
    }
}

impl RowSink<u8> for ChecksumSink {
    type Error = JpegError;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), Self::Error> {
        assert_eq!(y, self.expected_y);
        assert_eq!(row.len(), self.row_bytes);
        self.hash = fnv1a_update(self.hash, row);
        self.expected_y += 1;
        Ok(())
    }
}

fn decode_rows_checksum(decoder: &Decoder<'_>) -> u64 {
    let (width, height) = decoder.info().dimensions;
    let mut sink = ChecksumSink::new(width as usize * 3);
    let outcome = decoder
        .decode_rows(&mut sink)
        .expect("generated benchmark JPEG row decode must succeed");
    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    assert_eq!(sink.expected_y, height);
    sink.hash
}

fn decode_checksum(decoder: &Decoder<'_>, mode: DecodeMode) -> u64 {
    match mode {
        DecodeMode::Buffer(format) => decode_buffer_checksum(decoder, format),
        DecodeMode::Rows => decode_rows_checksum(decoder),
    }
}

fn encode_gray(width: u32, height: u32) -> Vec<u8> {
    let pixels = patterned_gray8(width, height);
    encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &pixels,
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Gray,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode deterministic grayscale benchmark JPEG")
    .data
}

fn encode_rgb(width: u32, height: u32, subsampling: JpegSubsampling) -> Vec<u8> {
    let pixels = patterned_rgb8(width, height);
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &pixels,
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
    .expect("encode deterministic RGB benchmark JPEG")
    .data
}

fn decode_cases() -> Vec<DecodeCase> {
    let rgb_420 = encode_rgb(512, 512, JpegSubsampling::Ybr420);
    let mut cases = Vec::new();
    cases
        .try_reserve_exact(6)
        .expect("reserve deterministic benchmark cases");
    cases.push(DecodeCase::new(
        "gray8_512",
        512,
        512,
        encode_gray(512, 512),
        DecodeMode::Buffer(PixelFormat::Gray8),
    ));
    cases.push(DecodeCase::new(
        "rgb8_512_444",
        512,
        512,
        encode_rgb(512, 512, JpegSubsampling::Ybr444),
        DecodeMode::Buffer(PixelFormat::Rgb8),
    ));
    cases.push(DecodeCase::new(
        "rgb8_512_422",
        512,
        512,
        encode_rgb(512, 512, JpegSubsampling::Ybr422),
        DecodeMode::Buffer(PixelFormat::Rgb8),
    ));
    cases.push(DecodeCase::new(
        "rgb8_512_420",
        512,
        512,
        rgb_420.clone(),
        DecodeMode::Buffer(PixelFormat::Rgb8),
    ));
    cases.push(DecodeCase::new(
        "rgb8_257x263_420",
        257,
        263,
        encode_rgb(257, 263, JpegSubsampling::Ybr420),
        DecodeMode::Buffer(PixelFormat::Rgb8),
    ));
    cases.push(DecodeCase::new(
        "rgb8_512_420_rows",
        512,
        512,
        rgb_420,
        DecodeMode::Rows,
    ));
    cases
}

fn bench_decode_cpu(c: &mut Criterion) {
    let cases = decode_cases();
    let mut group = c.benchmark_group("jpeg_cpu_decode_runtime");
    for case in &cases {
        let decoder = Decoder::new(&case.bytes).expect("benchmark JPEG must parse");
        let expected_checksum = case.expected_checksum;
        group.throughput(Throughput::Elements(case.pixels()));
        match case.mode {
            DecodeMode::Buffer(format) => {
                let stride = case.width as usize * bytes_per_pixel(format);
                let mut output = zeroed_bytes(stride * case.height as usize);
                group.bench_function(case.name, |b| {
                    b.iter(|| {
                        let outcome = decoder
                            .decode_into(&mut output, stride, format)
                            .expect("benchmark decode must succeed");
                        std::hint::black_box(outcome);
                        let checksum = fnv1a_update(FNV_OFFSET_BASIS, &output);
                        debug_assert_eq!(checksum, expected_checksum);
                        std::hint::black_box(checksum);
                    });
                });
            }
            DecodeMode::Rows => {
                group.bench_function(case.name, |b| {
                    b.iter(|| {
                        let mut sink = ChecksumSink::new(case.width as usize * 3);
                        let outcome = decoder
                            .decode_rows(&mut sink)
                            .expect("benchmark row decode must succeed");
                        debug_assert_eq!(sink.hash, expected_checksum);
                        std::hint::black_box((outcome, sink.hash));
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group! {
    name = decode_cpu_benches;
    config = Criterion::default()
        .confidence_level(0.95)
        .sample_size(50)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10));
    targets = bench_decode_cpu
}
criterion_main!(decode_cpu_benches);
