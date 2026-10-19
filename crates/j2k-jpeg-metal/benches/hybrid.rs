// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hybrid Metal decode against CPU decode for JPEG classes outside the
//! fast-packet kernels: progressive, grayscale, 4:4:0 sampling, and CMYK,
//! plus any JPEG listed in `J2K_BENCH_INPUTS` (12-bit and lossless files, for
//! example). Inputs CPU decode rejects are reported and skipped.
//!
//! Each iteration parses the JPEG and decodes it to a full-size surface, so
//! both sides pay the same header and entropy costs; the Metal side also
//! waits for the GPU. Output bytes are checked once per input before timing.
//!
//! ```sh
//! J2K_REQUIRE_METAL_BENCH=1 cargo bench --locked -p j2k-jpeg-metal --bench hybrid \
//!     --profile release-bench -- --sample-size 10 --noplot
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use j2k_core::{BackendRequest, PixelFormat};
use j2k_jpeg::{DecodeRequest, Decoder as CpuDecoder};
use j2k_jpeg_metal::{Decoder, MetalDecodeRequest};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};

struct HybridInput {
    name: String,
    bytes: Vec<u8>,
    fmt: PixelFormat,
}

fn encode(
    name: &str,
    (width, height): (u16, u16),
    color: ColorType,
    sampling: SamplingFactor,
    progressive: bool,
) -> HybridInput {
    let rgb = j2k_test_support::gpu_bench_rgb8(u32::from(width), u32::from(height));
    let (pixels, fmt) = match color {
        ColorType::Luma => (
            rgb.as_chunks::<3>()
                .0
                .iter()
                .map(|px| {
                    ((u16::from(px[0]) * 77 + u16::from(px[1]) * 150 + u16::from(px[2]) * 29) >> 8)
                        as u8
                })
                .collect::<Vec<_>>(),
            PixelFormat::Gray8,
        ),
        ColorType::Cmyk => (
            rgb.as_chunks::<3>()
                .0
                .iter()
                .flat_map(|px| [255 - px[0], 255 - px[1], 255 - px[2], 32])
                .collect(),
            PixelFormat::Rgb8,
        ),
        _ => (rgb, PixelFormat::Rgb8),
    };
    let mut bytes = Vec::new();
    let mut encoder = Encoder::new(&mut bytes, 85);
    encoder.set_sampling_factor(sampling);
    encoder.set_progressive(progressive);
    encoder
        .encode(&pixels, width, height, color)
        .expect("encode hybrid bench input");
    HybridInput {
        name: name.to_string(),
        bytes,
        fmt,
    }
}

fn synthetic_inputs() -> Vec<HybridInput> {
    vec![
        encode(
            "progressive_420_2048x1536",
            (2048, 1536),
            ColorType::Rgb,
            SamplingFactor::F_2_2,
            true,
        ),
        encode(
            "progressive_444_1024x1024",
            (1024, 1024),
            ColorType::Rgb,
            SamplingFactor::F_1_1,
            true,
        ),
        encode(
            "gray_baseline_2048x2048",
            (2048, 2048),
            ColorType::Luma,
            SamplingFactor::F_1_1,
            false,
        ),
        encode(
            "gray_progressive_1024x1024",
            (1024, 1024),
            ColorType::Luma,
            SamplingFactor::F_1_1,
            true,
        ),
        encode(
            "ycbcr440_1600x1200",
            (1600, 1200),
            ColorType::Rgb,
            SamplingFactor::F_1_2,
            false,
        ),
        encode(
            "cmyk_1024x1024",
            (1024, 1024),
            ColorType::Cmyk,
            SamplingFactor::F_1_1,
            false,
        ),
    ]
}

/// Files from `J2K_BENCH_INPUTS`, decoded to the widest RGB format CPU
/// decode accepts for their precision.
fn external_inputs() -> Vec<HybridInput> {
    let mut inputs = Vec::new();
    for path in j2k_test_support::paths_from_env("J2K_BENCH_INPUTS") {
        for file in j2k_test_support::collect_jpeg_paths(&path) {
            let Ok(bytes) = std::fs::read(&file) else {
                continue;
            };
            let Ok(decoder) = CpuDecoder::new(&bytes) else {
                continue;
            };
            let info = decoder.info();
            let gray = info.color_space == j2k_jpeg::ColorSpace::Grayscale;
            let fmt = match (info.bit_depth > 8, gray) {
                (false, true) => PixelFormat::Gray8,
                (false, false) => PixelFormat::Rgb8,
                (true, true) => PixelFormat::Gray16,
                (true, false) => PixelFormat::Rgb16,
            };
            if decoder.decode_request(DecodeRequest::full(fmt)).is_err() {
                continue;
            }
            let name = file.file_stem().map_or_else(
                || "input".into(),
                |stem| stem.to_string_lossy().into_owned(),
            );
            inputs.push(HybridInput { name, bytes, fmt });
        }
    }
    inputs
}

fn decode_cpu(input: &HybridInput) -> Vec<u8> {
    CpuDecoder::new(&input.bytes)
        .expect("CPU decoder")
        .decode_request(DecodeRequest::full(input.fmt))
        .expect("CPU decode")
        .0
}

fn decode_metal(input: &HybridInput) -> j2k_jpeg_metal::Surface {
    Decoder::new(&input.bytes)
        .expect("Metal decoder")
        .decode_request_to_device(MetalDecodeRequest::full(input.fmt, BackendRequest::Metal))
        .expect("Metal decode")
}

fn metal_available() -> bool {
    let required = std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_some();
    #[cfg(target_os = "macos")]
    {
        let available = j2k_metal_support::system_default_device().is_ok();
        assert!(
            !required || available,
            "J2K_REQUIRE_METAL_BENCH is set but no default Metal device is available"
        );
        available
    }
    #[cfg(not(target_os = "macos"))]
    {
        assert!(
            !required,
            "J2K_REQUIRE_METAL_BENCH is set but this is not a Metal host"
        );
        false
    }
}

fn bench_hybrid_decode(c: &mut Criterion) {
    let has_metal = metal_available();
    let mut inputs = synthetic_inputs();
    inputs.extend(external_inputs());
    let mut group = c.benchmark_group("jpeg_metal_hybrid_decode");
    for input in &inputs {
        if let Err(err) = CpuDecoder::new(&input.bytes)
            .and_then(|decoder| decoder.decode_request(DecodeRequest::full(input.fmt)))
        {
            eprintln!("skipping {}: CPU decode rejects it: {err}", input.name);
            continue;
        }
        let expected = decode_cpu(input);
        group.bench_function(BenchmarkId::new(&input.name, "cpu"), |b| {
            b.iter(|| decode_cpu(input));
        });
        if has_metal {
            let surface = decode_metal(input);
            assert_eq!(
                surface.as_bytes().expect("surface bytes").as_ref(),
                expected,
                "{}: Metal output differs from CPU decode",
                input.name
            );
            group.bench_function(BenchmarkId::new(&input.name, "metal"), |b| {
                b.iter(|| decode_metal(input));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_hybrid_decode);
criterion_main!(benches);
