// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG decode kernel harness.
//!
//! Correctness: every Metal consumer of the shared `decode_block` and IDCT
//! routines must match the CPU decoder bit for bit on entropy-stress fixtures:
//! all three samplings, quality 50 to 100, noise that forces long Huffman
//! codes, large magnitudes and 0xFF stuffing, odd dimensions, restart
//! intervals, batch 1 and 16, and every batch operation and output kind.
//!
//! Profiling (`#[ignore]`d): GPU time per decode stage for the buffer-output
//! batch kernels. Probe variants replace `decode_idct_deposit_block` with a
//! body that stops after Huffman decoding, coefficient materialization, or the
//! IDCT, so successive differences split the fused kernel's time. Setting
//! `J2K_JPEG_HARNESS_BASELINE_SHADERS` to a directory holding another copy of
//! the shader files A/Bs those kernels against the in-tree ones in one process.
//!
//! ```sh
//! J2K_REQUIRE_METAL_RUNTIME=1 cargo test --profile gpu-quick -p j2k-jpeg-metal --lib \
//!     -- decode_kernel_harness --include-ignored --nocapture --test-threads=1
//! ```

use super::*;
use crate::compute::command::gpu_time;
use crate::compute::pipeline_registry::SHADER_SOURCE;
use j2k_core::{DeviceSubmission, Downscale, ImageDecodeSubmit, Rect};
use std::time::Instant;

/// Shader files in `SHADER_SOURCE` concatenation order.
const SHADER_FILES: [&str; 9] = [
    "shaders_shared.metal",
    "shaders_encode.metal",
    "shaders_encode_staged.metal",
    "shaders_decode_helpers.metal",
    "shaders_pack_444.metal",
    "shaders_decode_fast420.metal",
    "shaders_decode_fast422_regions.metal",
    "shaders_decode_fast444.metal",
    "shaders_pack_subsampled.metal",
];

const BASELINE_SHADERS_ENV: &str = "J2K_JPEG_HARNESS_BASELINE_SHADERS";
const PROFILE_BATCH: usize = 16;
const PROFILE_WARMUP: usize = 3;
const PROFILE_SAMPLES: usize = 21;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sampling {
    S420,
    S422,
    S444,
}

impl Sampling {
    fn factor(self) -> jpeg_encoder::SamplingFactor {
        match self {
            Self::S420 => jpeg_encoder::SamplingFactor::F_2_2,
            Self::S422 => jpeg_encoder::SamplingFactor::F_2_1,
            Self::S444 => jpeg_encoder::SamplingFactor::F_1_1,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::S420 => "420",
            Self::S422 => "422",
            Self::S444 => "444",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Content {
    /// The shared GPU benchmark image (sawtooth ramps, sharp edges).
    Bench,
    /// Smooth gradients plus ±16 noise: photographic high-frequency detail.
    Textured,
    /// Uniform noise: long AC codes, large magnitudes, frequent 0xFF stuffing.
    Noise,
}

impl Content {
    fn label(self) -> &'static str {
        match self {
            Self::Bench => "bench",
            Self::Textured => "textured",
            Self::Noise => "noise",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FixtureSpec {
    sampling: Sampling,
    content: Content,
    quality: u8,
    width: u16,
    height: u16,
    restart: Option<u16>,
}

impl FixtureSpec {
    fn label(&self) -> String {
        let restart = self
            .restart
            .map_or(String::new(), |interval| format!("_rst{interval}"));
        format!(
            "{}_{}_q{}_{}x{}{restart}",
            self.sampling.label(),
            self.content.label(),
            self.quality,
            self.width,
            self.height
        )
    }

    fn dimensions(&self) -> (u32, u32) {
        (u32::from(self.width), u32::from(self.height))
    }
}

struct XorShift(u32);

impl XorShift {
    fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9) | 1)
    }

    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

fn rgb_pixels(content: Content, width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut rng = XorShift::new(seed);
    match content {
        Content::Bench => {
            let mut rgb = j2k_test_support::gpu_bench_rgb8(width, height);
            // Distinct tiles need distinct entropy streams.
            let shift = (seed as usize * 3 * 7) % rgb.len().max(1);
            rgb.rotate_left(shift);
            rgb
        }
        Content::Textured => {
            let span = (width + height).max(1);
            let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
            for y in 0..height {
                for x in 0..width {
                    let base = [
                        (x + y) * 255 / span,
                        (x * 2 + seed * 17) % 256,
                        (y * 3 + (x * y) / 64) % 256,
                    ];
                    for channel in base {
                        let noise = i32::try_from(rng.next() % 33).expect("noise fits i32") - 16;
                        let channel = i32::try_from(channel).expect("channel fits i32");
                        let sample = (channel + noise).clamp(0, 255);
                        rgb.push(u8::try_from(sample).expect("clamped sample fits u8"));
                    }
                }
            }
            rgb
        }
        Content::Noise => (0..width as usize * height as usize * 3)
            .map(|_| rng.next().to_le_bytes()[1])
            .collect(),
    }
}

fn encode_jpeg(spec: &FixtureSpec, seed: u32) -> Vec<u8> {
    let (width, height) = spec.dimensions();
    let rgb = rgb_pixels(spec.content, width, height, seed);
    let mut jpeg = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut jpeg, spec.quality);
    encoder.set_sampling_factor(spec.sampling.factor());
    if let Some(interval) = spec.restart {
        encoder.set_restart_interval(interval);
    }
    encoder
        .encode(&rgb, spec.width, spec.height, jpeg_encoder::ColorType::Rgb)
        .unwrap_or_else(|error| panic!("encode {}: {error}", spec.label()));
    jpeg
}

fn encode_batch(spec: &FixtureSpec, count: usize) -> Vec<Arc<[u8]>> {
    (0..count)
        .map(|seed| {
            let seed = u32::try_from(seed).expect("fixture seed fits u32");
            Arc::<[u8]>::from(encode_jpeg(spec, seed + 1))
        })
        .collect()
}

fn correctness_specs() -> Vec<FixtureSpec> {
    let mut specs = Vec::new();
    for sampling in [Sampling::S420, Sampling::S422, Sampling::S444] {
        let spec = |content, quality, width, height, restart| FixtureSpec {
            sampling,
            content,
            quality,
            width,
            height,
            restart,
        };
        specs.extend([
            spec(Content::Bench, 90, 128, 128, None),
            spec(Content::Textured, 75, 128, 128, None),
            spec(Content::Textured, 95, 133, 77, None),
            spec(Content::Noise, 100, 96, 64, None),
            spec(Content::Noise, 50, 131, 67, None),
            spec(Content::Noise, 90, 130, 66, None),
            spec(Content::Textured, 90, 96, 64, Some(3)),
        ]);
    }
    specs
}

fn queued_requests(
    jpegs: &[Arc<[u8]>],
    sampling: Sampling,
    op: batch::BatchOp,
) -> Vec<batch::QueuedRequest> {
    jpegs
        .iter()
        .map(|jpeg| {
            let bytes = jpeg.as_ref();
            let (p444, p422, p420) = match sampling {
                Sampling::S444 => (
                    Some(Arc::new(
                        j2k_jpeg::adapter::build_fast444_packet(bytes).expect("fast444 packet"),
                    )),
                    None,
                    None,
                ),
                Sampling::S422 => (
                    None,
                    Some(Arc::new(
                        j2k_jpeg::adapter::build_fast422_packet(bytes).expect("fast422 packet"),
                    )),
                    None,
                ),
                Sampling::S420 => (
                    None,
                    None,
                    Some(Arc::new(
                        j2k_jpeg::adapter::build_fast420_packet(bytes).expect("fast420 packet"),
                    )),
                ),
            };
            batch::QueuedRequest::new(
                Arc::clone(jpeg),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                op,
                p444,
                p422,
                p420,
            )
        })
        .collect()
}

fn jpeg_rect(rect: Rect) -> j2k_jpeg::Rect {
    j2k_jpeg::Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

fn cpu_request(op: batch::BatchOp) -> DecodeRequest {
    match op {
        batch::BatchOp::Full => DecodeRequest::full(PixelFormat::Rgb8),
        batch::BatchOp::Region(roi) => DecodeRequest::region(PixelFormat::Rgb8, jpeg_rect(roi)),
        batch::BatchOp::Scaled(scale) => DecodeRequest::scaled(PixelFormat::Rgb8, scale),
        batch::BatchOp::RegionScaled { roi, scale } => {
            DecodeRequest::region_scaled(PixelFormat::Rgb8, jpeg_rect(roi), scale)
        }
    }
}

fn cpu_rgb(jpeg: &[u8], op: batch::BatchOp) -> Vec<u8> {
    CpuDecoder::new(jpeg)
        .expect("CPU decoder")
        .decode_request(cpu_request(op))
        .expect("CPU oracle decode")
        .0
}

/// Describes how `actual` differs from `expected` without dumping megabytes:
/// the mismatch count, the rows involved, and the first differing sample.
fn pixel_mismatch(
    label: &str,
    actual: &[u8],
    expected: &[u8],
    width: u32,
    channels: usize,
) -> Option<String> {
    if actual.len() != expected.len() {
        return Some(format!(
            "{label}: {} output bytes, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let first = actual.iter().zip(expected).position(|(a, e)| a != e)?;
    let row_len = width as usize * channels;
    let rows = actual
        .chunks(row_len)
        .zip(expected.chunks(row_len))
        .enumerate()
        .filter(|(_, (a, e))| a != e)
        .map(|(row, _)| row)
        .collect::<Vec<_>>();
    let mismatches = actual.iter().zip(expected).filter(|(a, e)| a != e).count();
    let pixel = first / channels;
    Some(format!(
        "{label}: {mismatches} samples differ in rows {rows:?} of {}; first x={} y={} c={} (GPU {} vs CPU {})",
        actual.len() / row_len.max(1),
        pixel % width as usize,
        pixel / width as usize,
        first % channels,
        actual[first],
        expected[first]
    ))
}

fn assert_no_mismatches(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} mismatching outputs:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn output_width(spec: &FixtureSpec, op: batch::BatchOp) -> u32 {
    let scaled = |width: u32, scale: Downscale| width.div_ceil(scale.denominator());
    match op {
        batch::BatchOp::Full => spec.dimensions().0,
        batch::BatchOp::Region(roi) => roi.w,
        batch::BatchOp::Scaled(scale) => scaled(spec.dimensions().0, scale),
        batch::BatchOp::RegionScaled { roi, scale } => scaled(roi.w, scale),
    }
}

fn batch_ops(spec: &FixtureSpec) -> Vec<batch::BatchOp> {
    let (width, height) = spec.dimensions();
    // An odd window that straddles MCU boundaries on both axes.
    let roi = Rect {
        x: 5,
        y: 3,
        w: width - 5 - 7,
        h: height - 3 - 6,
    };
    // The same columns down to the bottom edge, where 4:2:0 upsampling must
    // replicate the last real chroma row rather than read MCU padding.
    let bottom_roi = Rect {
        y: height - 11,
        h: 11,
        ..roi
    };
    vec![
        batch::BatchOp::Full,
        batch::BatchOp::Scaled(Downscale::Half),
        batch::BatchOp::Scaled(Downscale::Quarter),
        batch::BatchOp::Scaled(Downscale::Eighth),
        batch::BatchOp::Region(roi),
        batch::BatchOp::RegionScaled {
            roi,
            scale: Downscale::Half,
        },
        batch::BatchOp::Region(bottom_roi),
        batch::BatchOp::RegionScaled {
            roi: bottom_roi,
            scale: Downscale::Half,
        },
    ]
}

#[test]
fn shader_file_order_matches_production_source() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    assert_eq!(read_shader_dir(std::path::Path::new(dir)), SHADER_SOURCE);
}

#[test]
fn batch_buffer_decodes_match_cpu_on_entropy_stress_fixtures() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    for spec in correctness_specs() {
        let jpegs = encode_batch(&spec, PROFILE_BATCH);
        for op in batch_ops(&spec) {
            let expected = jpegs
                .iter()
                .map(|jpeg| cpu_rgb(jpeg, op))
                .collect::<Vec<_>>();
            for batch_len in [1, PROFILE_BATCH] {
                let requests = queued_requests(&jpegs[..batch_len], spec.sampling, op);
                let results = decode_full_batch_to_surfaces(&requests)
                    .expect("Metal batch decode")
                    .unwrap_or_else(|| panic!("{}: {op:?} left the Metal path", spec.label()));
                assert_eq!(results.len(), batch_len);
                for (index, result) in results.into_iter().enumerate() {
                    let surface = result.expect("tile surface");
                    let label = format!("{} {op:?} batch{batch_len} tile{index}", spec.label());
                    failures.extend(pixel_mismatch(
                        &label,
                        &surface.as_bytes().expect("surface bytes"),
                        &expected[index],
                        output_width(&spec, op),
                        3,
                    ));
                }
            }
        }
    }
    assert_no_mismatches(&failures);
}

#[test]
fn batch_texture_decodes_match_cpu_on_entropy_stress_fixtures() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let mut failures = Vec::new();
    for spec in correctness_specs() {
        let jpegs = encode_batch(&spec, PROFILE_BATCH);
        if spec.restart.is_some() && spec.sampling != Sampling::S420 {
            // Documented limitation: resident texture batches reject
            // restart-coded full-tile 4:2:2 and 4:4:4 with a typed error.
            continue;
        }
        let decoders = jpegs
            .iter()
            .map(|jpeg| crate::Decoder::new(jpeg.as_ref()).expect("Metal decoder"))
            .collect::<Vec<_>>();
        let decoder_refs = decoders.iter().collect::<Vec<_>>();
        let ops = [
            (crate::Rgb8MetalBatchOp::Full, batch::BatchOp::Full),
            (
                crate::Rgb8MetalBatchOp::Scaled(Downscale::Half),
                batch::BatchOp::Scaled(Downscale::Half),
            ),
        ];
        for (public_op, op) in ops {
            let mut output = crate::MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1)
                .expect("texture output");
            let tiles = crate::Codec::decode_rgb8_batch_into_textures_with_session(
                crate::Rgb8MetalBatchRequest {
                    source: crate::Rgb8MetalBatchSource::Decoders(&decoder_refs),
                    op: public_op,
                },
                crate::MetalTextureBatchTarget::Resizable(&mut output),
                &session,
            )
            .unwrap_or_else(|error| panic!("{} {op:?} textures: {error}", spec.label()));
            for (index, tile) in tiles.into_iter().enumerate() {
                let tile = tile.expect("texture tile");
                let actual = crate::tests::download_rgba8_texture(
                    &session,
                    tile.texture_trusted(),
                    tile.dimensions(),
                );
                let expected = crate::tests::rgb_to_rgba_opaque(&cpu_rgb(&jpegs[index], op));
                let label = format!("{} {op:?} texture tile{index}", spec.label());
                failures.extend(pixel_mismatch(
                    &label,
                    &actual,
                    &expected,
                    tile.dimensions().0,
                    4,
                ));
            }
        }
    }
    assert_no_mismatches(&failures);
}

#[test]
fn single_decodes_match_cpu_on_entropy_stress_fixtures() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    for spec in correctness_specs() {
        let jpeg = encode_jpeg(&spec, 1);
        let mut decoder = crate::Decoder::new(&jpeg).expect("Metal decoder");
        let mut session = crate::MetalSession::default();
        let surface = <crate::Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
            &mut decoder,
            &mut session,
            PixelFormat::Rgb8,
            BackendRequest::Metal,
        )
        .expect("single submit")
        .wait()
        .expect("single surface");
        assert!(
            matches!(surface.storage, crate::Storage::Metal { .. }),
            "{}: single decode left the Metal path",
            spec.label()
        );
        failures.extend(pixel_mismatch(
            &format!("{} single full", spec.label()),
            &surface.as_bytes().expect("surface bytes"),
            &cpu_rgb(&jpeg, batch::BatchOp::Full),
            spec.dimensions().0,
            3,
        ));
    }
    assert_no_mismatches(&failures);
}

/// Marks a 3-component JPEG as RGB with an Adobe APP14 segment (transform 0),
/// so its components decode without the YCbCr conversion.
fn with_adobe_rgb_transform(jpeg: &[u8]) -> Vec<u8> {
    const APP14_ADOBE_RGB: [u8; 16] = [
        0xFF, 0xEE, 0x00, 0x0E, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];
    let mut out = Vec::with_capacity(jpeg.len() + APP14_ADOBE_RGB.len());
    out.extend_from_slice(&jpeg[..2]);
    out.extend_from_slice(&APP14_ADOBE_RGB);
    out.extend_from_slice(&jpeg[2..]);
    out
}

#[test]
fn rgb_colorspace_444_batches_match_cpu() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let spec = FixtureSpec {
        sampling: Sampling::S444,
        content: Content::Textured,
        quality: 90,
        width: 133,
        height: 77,
        restart: None,
    };
    let jpegs = encode_batch(&spec, PROFILE_BATCH)
        .iter()
        .map(|jpeg| Arc::<[u8]>::from(with_adobe_rgb_transform(jpeg)))
        .collect::<Vec<_>>();
    assert_eq!(
        CpuDecoder::new(&jpegs[0])
            .expect("CPU decoder")
            .info()
            .color_space,
        j2k_jpeg::ColorSpace::Rgb
    );
    let op = batch::BatchOp::Full;
    let expected = jpegs
        .iter()
        .map(|jpeg| cpu_rgb(jpeg, op))
        .collect::<Vec<_>>();
    let mut failures = Vec::new();

    let requests = queued_requests(&jpegs, spec.sampling, op);
    let surfaces = decode_full_batch_to_surfaces(&requests)
        .expect("Metal batch decode")
        .expect("RGB 4:4:4 buffer batch stays on Metal");
    for (index, surface) in surfaces.into_iter().enumerate() {
        failures.extend(pixel_mismatch(
            &format!("rgb444 buffer tile{index}"),
            &surface.expect("surface").as_bytes().expect("surface bytes"),
            &expected[index],
            spec.dimensions().0,
            3,
        ));
    }

    let inputs = jpegs.iter().map(AsRef::as_ref).collect::<Vec<&[u8]>>();
    let mut output = crate::MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1)
        .expect("texture output");
    let tiles = crate::Codec::decode_rgb8_batch_into_textures_with_session(
        crate::Rgb8MetalBatchRequest {
            source: crate::Rgb8MetalBatchSource::Bytes(&inputs),
            op: crate::Rgb8MetalBatchOp::Full,
        },
        crate::MetalTextureBatchTarget::Resizable(&mut output),
        &session,
    )
    .expect("RGB 4:4:4 texture batch");
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        let actual = crate::tests::download_rgba8_texture(
            &session,
            tile.texture_trusted(),
            tile.dimensions(),
        );
        failures.extend(pixel_mismatch(
            &format!("rgb444 texture tile{index}"),
            &actual,
            &crate::tests::rgb_to_rgba_opaque(&expected[index]),
            tile.dimensions().0,
            4,
        ));
    }
    assert_no_mismatches(&failures);
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Evidence for the P33 record: input and output SHA-256 of the
/// `wsi_tile_batch_rgba_textures` 4:4:4 benchmark workloads through the
/// component-plane route and the direct texture kernels, plus the CPU oracle.
#[test]
#[ignore = "P33 evidence; run explicitly with --include-ignored --nocapture"]
fn p33_fast444_texture_route_output_hashes() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let generated = generated_bench_jpeg(jpeg_encoder::SamplingFactor::F_1_1, None);
    let fixture = include_bytes!("../../../fixtures/jpeg/baseline_444_8x8.jpg").to_vec();
    let mut corpus = generated.clone();
    corpus.extend_from_slice(&fixture);
    println!("input_corpus_sha256 {}", sha256_hex(&corpus));
    for (name, bytes) in [
        ("generated/fast444_256x256", &generated),
        ("repo/baseline_444_8x8", &fixture),
    ] {
        let (rgb, _) = CpuDecoder::new(bytes)
            .expect("CPU decoder")
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("CPU decode");
        let cpu_tile = crate::tests::rgb_to_rgba_opaque(&rgb);
        for batch_size in [16, 64] {
            let decoders = (0..batch_size)
                .map(|_| crate::Decoder::new(bytes).expect("Metal decoder"))
                .collect::<Vec<_>>();
            let decoder_refs = decoders.iter().collect::<Vec<_>>();
            let mut output = crate::MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1)
                .expect("texture output");
            let expected = cpu_tile.repeat(batch_size);
            let mut hashes = Vec::new();
            for planes in [false, true] {
                let tiles = crate::compute::texture_tuning::with_component_planes(planes, || {
                    crate::Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
                        &decoder_refs,
                        &mut output,
                        &session,
                    )
                })
                .expect("texture batch");
                let mut actual = Vec::with_capacity(expected.len());
                for tile in tiles {
                    let tile = tile.expect("texture tile");
                    actual.extend(crate::tests::download_rgba8_texture(
                        &session,
                        tile.texture_trusted(),
                        tile.dimensions(),
                    ));
                }
                assert!(
                    actual == expected,
                    "{name} batch{batch_size} planes={planes} differs from CPU"
                );
                hashes.push(sha256_hex(&actual));
            }
            println!(
                "{name} batch{batch_size} input_sha256 {} direct {} planes {} cpu {}",
                sha256_hex(bytes),
                hashes[0],
                hashes[1],
                sha256_hex(&expected)
            );
        }
    }
}

/// A generated 256x256 benchmark JPEG, as `benches/compare.rs` encodes it.
fn generated_bench_jpeg(
    sampling: jpeg_encoder::SamplingFactor,
    restart_interval: Option<u16>,
) -> Vec<u8> {
    let rgb = j2k_test_support::gpu_bench_rgb8(256, 256);
    let mut jpeg = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut jpeg, 90);
    encoder.set_sampling_factor(sampling);
    if let Some(interval) = restart_interval {
        encoder.set_restart_interval(interval);
    }
    encoder
        .encode(&rgb, 256, 256, jpeg_encoder::ColorType::Rgb)
        .expect("encode generated benchmark JPEG");
    jpeg
}

/// Decodes 64 tiles of `bytes` three times on one session, as the
/// `wsi_tile_batch_rgb` one-shot and retained-session rows do, and returns the
/// concatenated RGB output of each batch.
fn retained_tile_batches(bytes: &[u8]) -> Vec<Vec<u8>> {
    use j2k_core::TileBatchDecodeSubmit as _;

    let mut ctx = j2k_jpeg::DecoderContext::default();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let mut session = crate::MetalSession::default();
    (0..3)
        .map(|_| {
            let submissions = (0..64)
                .map(|_| {
                    crate::Codec::submit_tile_to_device(
                        &mut ctx,
                        &mut session,
                        &mut pool,
                        bytes,
                        PixelFormat::Rgb8,
                        BackendRequest::Metal,
                    )
                    .expect("submit tile")
                })
                .collect::<Vec<_>>();
            let mut output = Vec::new();
            for submission in submissions {
                let surface = submission.wait().expect("tile surface");
                output.extend_from_slice(surface.as_bytes().expect("surface bytes").as_ref());
            }
            output
        })
        .collect()
}

/// Decodes 16 resident RGBA textures of `bytes`, as the batch-16
/// `wsi_tile_batch_rgba_textures` rows do, and reads them back.
fn texture_batch16(session: &crate::MetalBackendSession, bytes: &[u8]) -> Vec<u8> {
    let decoders = (0..16)
        .map(|_| crate::Decoder::new(bytes).expect("Metal decoder"))
        .collect::<Vec<_>>();
    let decoder_refs = decoders.iter().collect::<Vec<_>>();
    let mut output = crate::MetalBatchTextureOutput::new_rgba8_tiles(session, (256, 256), 16)
        .expect("texture output");
    let tiles = crate::Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        &decoder_refs,
        &mut output,
        session,
    )
    .expect("texture batch");
    let mut textures = Vec::new();
    for tile in tiles {
        let tile = tile.expect("texture tile");
        textures.extend(crate::tests::download_rgba8_texture(
            session,
            tile.texture_trusted(),
            tile.dimensions(),
        ));
    }
    textures
}

/// Evidence for the P32 record: input and output SHA-256 of the generated
/// 256x256 `wsi_tile_batch_rgb` workloads (64 tiles; three batches on one
/// session, as in the one-shot and retained-session rows) and the batch-16
/// `wsi_tile_batch_rgba_textures` workloads, each checked against the CPU
/// decoder. The P32 A/B runs it once per arm.
#[test]
#[ignore = "P32 evidence; run explicitly with --include-ignored --nocapture"]
fn p32_tile_and_texture_batch_output_hashes() {
    use jpeg_encoder::SamplingFactor;

    if !should_run_metal_runtime() {
        return;
    }
    let inputs = [
        ("generated/fast420_256x256", SamplingFactor::F_2_2, None),
        (
            "generated/fast420_restart2_256x256",
            SamplingFactor::F_2_2,
            Some(2),
        ),
        ("generated/fast422_256x256", SamplingFactor::F_2_1, None),
        ("generated/fast444_256x256", SamplingFactor::F_1_1, None),
    ];
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let mut corpus = Vec::new();
    for (name, sampling, restart_interval) in inputs {
        let bytes = generated_bench_jpeg(sampling, restart_interval);
        corpus.extend_from_slice(&bytes);
        let (rgb, _) = CpuDecoder::new(&bytes)
            .expect("CPU decoder")
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("CPU decode");

        let expected_rgb = rgb.repeat(64);
        for (iteration, output) in retained_tile_batches(&bytes).iter().enumerate() {
            assert!(
                *output == expected_rgb,
                "{name} tile batch iteration {iteration} differs from CPU"
            );
        }
        let textures = texture_batch16(&session, &bytes);
        assert!(
            textures == crate::tests::rgb_to_rgba_opaque(&rgb).repeat(16),
            "{name} texture batch differs from CPU"
        );
        println!(
            "{name} input_sha256 {} rgb_batch64_sha256 {} rgba_textures_batch16_sha256 {}",
            sha256_hex(&bytes),
            sha256_hex(&expected_rgb),
            sha256_hex(&textures)
        );
    }
    println!("input_corpus_sha256 {}", sha256_hex(&corpus));
}

// ---------------------------------------------------------------------------
// Stage profile
// ---------------------------------------------------------------------------

fn read_shader_dir(dir: &std::path::Path) -> String {
    SHADER_FILES
        .iter()
        .map(|name| {
            std::fs::read_to_string(dir.join(name))
                .unwrap_or_else(|error| panic!("read {}: {error}", dir.join(name).display()))
        })
        .collect()
}

const PRODUCTION_HELPER: &str = "inline bool decode_idct_deposit_block(";

/// Replaces `decode_idct_deposit_block` for the buffer-output kernels. Every
/// stage consumes the same bits as production, except stage 0, which decodes
/// nothing and measures launch, checkpoint setup and the pack pass. Stages 2
/// and 3 fold their whole result into one stored byte so it cannot be elided.
const PROBE_HELPER: &str = r"
inline bool decode_idct_deposit_block(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    TABLE_SPACE PreparedHuffman &dc_table,
    TABLE_SPACE PreparedHuffman &ac_table,
    constant ushort *quant,
    thread int &prev_dc,
    device JpegDecodeStatus *status,
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    thread short coeffs[64]
) {
#if JPEG_HARNESS_PROBE_STAGE == 0
    return true;
#elif JPEG_HARNESS_PROBE_STAGE == 1
    return decode_block_skip(br, bytes, len, dc_table, ac_table, prev_dc, status);
#else
    bool dc_only = false;
    if (!decode_block(br, bytes, len, dc_table, ac_table, quant, prev_dc, status, coeffs, dc_only)) {
        return false;
    }
    uint sink = dc_only ? 1u : 0u;
#if JPEG_HARNESS_PROBE_STAGE == 2
    for (uint i = 0; i < 64; ++i) {
        sink ^= uint(ushort(coeffs[i])) << (i & 15u);
    }
#else
    if (!dc_only) {
        thread uchar pixels[64];
        idct_islow(coeffs, pixels);
        for (uint i = 0; i < 64; ++i) {
            sink ^= uint(pixels[i]) << (i & 23u);
        }
    }
#endif
    if (x < width && y < height) {
        plane[y * stride + x] = uchar(sink ^ (sink >> 8) ^ (sink >> 16) ^ (sink >> 24));
    }
    return true;
#endif
}
";

/// Pack kernels of the profiled buffer paths. Probes stub them so every probe
/// column is decode-kernel time alone; `full - decode` is the pack pass.
const PROFILED_PACK_KERNELS: [&str; 2] = ["jpeg_pack_420_rgb_batch", "jpeg_pack_422_rgb_batch"];

/// Stage of a probe build. `Decode` keeps the production decode kernel.
#[derive(Clone, Copy)]
enum ProbeStage {
    Launch,
    Huffman,
    Coefficients,
    Idct,
    Decode,
}

impl ProbeStage {
    const ALL: [Self; 5] = [
        Self::Launch,
        Self::Huffman,
        Self::Coefficients,
        Self::Idct,
        Self::Decode,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Launch => "launch",
            Self::Huffman => "+huffman",
            Self::Coefficients => "+coeffs",
            Self::Idct => "+idct",
            Self::Decode => "+store",
        }
    }
}

fn stub_kernel(source: &str, kernel: &str) -> String {
    let at = source
        .find(&format!("kernel void {kernel}("))
        .unwrap_or_else(|| panic!("shader source defines {kernel}"));
    let body = at + source[at..].find(") {").expect("kernel body") + 3;
    format!("{}\n    return;{}", &source[..body], &source[body..])
}

fn probe_source(source: &str, stage: ProbeStage) -> String {
    let mut source = PROFILED_PACK_KERNELS
        .iter()
        .fold(source.to_string(), |source, kernel| {
            stub_kernel(&source, kernel)
        });
    let stage = match stage {
        ProbeStage::Decode => return source,
        ProbeStage::Launch => 0,
        ProbeStage::Huffman => 1,
        ProbeStage::Coefficients => 2,
        ProbeStage::Idct => 3,
    };
    let at = source
        .find(PRODUCTION_HELPER)
        .expect("shader source defines decode_idct_deposit_block");
    // Match the production helper's Huffman table address space.
    let table_param = &source[at..][..source[at..]
        .find("PreparedHuffman &dc_table")
        .expect("helper takes a DC table")];
    let table_space = table_param[table_param.rfind('\n').expect("parameter line") + 1..].trim();
    let helper = PROBE_HELPER.replace("TABLE_SPACE", table_space);
    let tail = source.split_off(at).replacen(
        PRODUCTION_HELPER,
        "inline bool decode_idct_deposit_block_production(",
        1,
    );
    format!("{source}#define JPEG_HARNESS_PROBE_STAGE {stage}\n{helper}\n{tail}")
}

fn run_full_rgb_batch(
    runtime: &MetalRuntime,
    sampling: Sampling,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Vec<Result<Surface, Error>> {
    let results = match sampling {
        Sampling::S420 => try_decode_fast_subsampled_full_rgb_batch_to_surfaces::<
            JpegFast420PacketV1,
        >(runtime, requests, packets),
        Sampling::S422 => try_decode_fast_subsampled_full_rgb_batch_to_surfaces::<
            JpegFast422PacketV1,
        >(runtime, requests, packets),
        Sampling::S444 => unreachable!("4:4:4 full batches use the region-scaled kernels"),
    };
    results
        .expect("profiled batch decode")
        .expect("profiled batch stays on the fused Metal path")
}

#[derive(Clone, Copy)]
struct StageTime {
    gpu_ms: f64,
    wall_ms: f64,
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Times `run` on every runtime. Samples are taken round-robin so GPU clock
/// ramping affects all variants alike; the first `PROFILE_WARMUP` rounds are
/// discarded.
fn profile_runtimes(
    runtimes: &[&MetalRuntime],
    mut run: impl FnMut(&MetalRuntime),
) -> Vec<StageTime> {
    profile_variants(runtimes.len(), |index| run(runtimes[index]))
}

/// Samples `variants` round-robin so clock ramping affects each alike.
fn profile_variants(variants: usize, mut run: impl FnMut(usize)) -> Vec<StageTime> {
    let mut gpu = vec![Vec::with_capacity(PROFILE_SAMPLES); variants];
    let mut wall = vec![Vec::with_capacity(PROFILE_SAMPLES); variants];
    for round in 0..PROFILE_WARMUP + PROFILE_SAMPLES {
        for index in 0..variants {
            let _ = gpu_time::take_seconds();
            let start = Instant::now();
            run(index);
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            let gpu_ms = gpu_time::take_seconds() * 1e3;
            if round >= PROFILE_WARMUP {
                gpu[index].push(gpu_ms);
                wall[index].push(elapsed);
            }
        }
    }
    gpu.into_iter()
        .zip(wall)
        .map(|(gpu, wall)| StageTime {
            gpu_ms: median(gpu),
            wall_ms: median(wall),
        })
        .collect()
}

fn assert_buffer_batch_matches(
    runtime: &MetalRuntime,
    sampling: Sampling,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    expected: &[Vec<u8>],
) {
    let results = run_full_rgb_batch(runtime, sampling, requests, packets);
    for (index, result) in results.into_iter().enumerate() {
        let surface = result.expect("profiled tile surface");
        let bytes = surface.as_bytes().expect("surface bytes");
        assert!(
            bytes.as_ref() == expected[index].as_slice(),
            "profiled output differs from CPU at tile {index}"
        );
    }
}

fn run_texture_batch(
    runtime: &MetalRuntime,
    sampling: Sampling,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Vec<Result<crate::MetalTextureTile, Error>> {
    let _access = output
        .lock_for_safe_access()
        .expect("texture output access");
    let results = match sampling {
        Sampling::S420 => {
            try_decode_fast_subsampled_full_rgba_batch_to_textures::<JpegFast420PacketV1>(
                runtime,
                requests,
                packets,
                output,
                FastBatchDecodeMode::Fused,
            )
        }
        Sampling::S422 => {
            try_decode_fast_subsampled_full_rgba_batch_to_textures::<JpegFast422PacketV1>(
                runtime,
                requests,
                packets,
                output,
                FastBatchDecodeMode::Fused,
            )
        }
        Sampling::S444 => {
            try_decode_fast444_full_rgba_batch_to_textures(runtime, requests, packets, output)
        }
    };
    results
        .expect("profiled texture batch")
        .expect("profiled texture batch stays on Metal")
}

fn print_full_and_baseline(times: &[StageTime], jpegs: &[Arc<[u8]>]) {
    let full = times[0];
    let base = times.get(1);
    let jpeg_kib = jpegs.iter().map(|jpeg| jpeg.len()).sum::<usize>() / 1024;
    println!(
        " {:>9.3} {:>9} {:>9.3} {:>9}   ({jpeg_kib} KiB JPEG)",
        full.gpu_ms,
        base.map_or("-".to_string(), |time| format!("{:.3}", time.gpu_ms)),
        full.wall_ms,
        base.map_or("-".to_string(), |time| format!("{:.3}", time.wall_ms)),
    );
}

/// Resident texture batches for every sampling.
fn texture_profile_specs() -> Vec<FixtureSpec> {
    let spec = |sampling, side| FixtureSpec {
        sampling,
        content: Content::Textured,
        quality: 90,
        width: side,
        height: side,
        restart: None,
    };
    vec![
        spec(Sampling::S420, 256),
        spec(Sampling::S420, 512),
        spec(Sampling::S422, 512),
        spec(Sampling::S444, 256),
        spec(Sampling::S444, 512),
    ]
}

/// Keeps the GPU busy long enough to leave its idle clock state, so the first
/// profiled workload is not measured while clocks ramp.
fn warm_up_gpu(runtime: &MetalRuntime) {
    let spec = profile_specs()[0];
    let jpegs = encode_batch(&spec, PROFILE_BATCH);
    let requests = queued_requests(&jpegs, spec.sampling, batch::BatchOp::Full);
    let packets = batched_fast_packets(&requests)
        .expect("packet lookup")
        .expect("fast packets");
    let start = Instant::now();
    while start.elapsed().as_millis() < 500 {
        drop(run_full_rgb_batch(
            runtime,
            spec.sampling,
            &requests,
            &packets,
        ));
    }
}

/// Fused 4:2:0 and 4:2:2 buffer batches: the kernels the probes instrument.
fn profile_specs() -> Vec<FixtureSpec> {
    let spec = |sampling, content, quality, side| FixtureSpec {
        sampling,
        content,
        quality,
        width: side,
        height: side,
        restart: None,
    };
    vec![
        spec(Sampling::S420, Content::Bench, 90, 512),
        spec(Sampling::S420, Content::Textured, 90, 512),
        spec(Sampling::S420, Content::Textured, 75, 512),
        spec(Sampling::S420, Content::Textured, 90, 256),
        spec(Sampling::S420, Content::Textured, 90, 1024),
        spec(Sampling::S422, Content::Textured, 90, 512),
        spec(Sampling::S420, Content::Noise, 95, 512),
    ]
}

/// Runs one buffer batch in a loop for a CPU sampling profiler, for example
/// `xcrun xctrace record --template 'Time Profiler' --launch -- <test binary>
/// host_overhead_loop --include-ignored --exact`.
#[test]
#[ignore = "host-overhead profiling loop; run under a sampling profiler"]
fn host_overhead_loop() {
    if !should_run_metal_runtime() {
        return;
    }
    let spec = profile_specs()[1];
    let jpegs = encode_batch(&spec, PROFILE_BATCH);
    let requests = queued_requests(&jpegs, spec.sampling, batch::BatchOp::Full);
    let packets = batched_fast_packets(&requests)
        .expect("packet lookup")
        .expect("fast packets");
    let runtime = MetalRuntime::new_with_shader_source(SHADER_SOURCE).expect("runtime");
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let output =
        crate::MetalBatchOutputBuffer::new_rgb8_tiles(&session, spec.dimensions(), PROFILE_BATCH)
            .expect("reusable output buffer");
    for reuse_output in [false, true] {
        let _ = (gpu_time::take_seconds(), gpu_time::take_submit_seconds());
        let start = Instant::now();
        let mut iterations = 0_u32;
        while start.elapsed().as_secs() < 4 {
            let results = if reuse_output {
                let _access = output.lock_for_safe_access().expect("output access");
                try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output::<
                    JpegFast420PacketV1,
                >(&runtime, &requests, &packets, &output)
                .expect("batch decode")
                .expect("stays on Metal")
            } else {
                run_full_rgb_batch(&runtime, spec.sampling, &requests, &packets)
            };
            for result in results {
                result.expect("batch tile");
            }
            iterations += 1;
        }
        let per_iteration = |seconds: f64| seconds * 1e3 / f64::from(iterations);
        let (scheduling, queued) = gpu_time::take_submit_seconds();
        println!(
            "{} reuse_output={reuse_output}: {iterations} iterations, {:.3} ms wall each: \
             scheduling {:.3}, queued {:.3}, GPU {:.3}",
            spec.label(),
            per_iteration(start.elapsed().as_secs_f64()),
            per_iteration(scheduling),
            per_iteration(queued),
            per_iteration(gpu_time::take_seconds()),
        );
    }
}

#[test]
#[ignore = "GPU stage profile; run explicitly with --include-ignored --nocapture"]
fn decode_kernel_stage_profile() {
    if !should_run_metal_runtime() {
        return;
    }
    let probes = ProbeStage::ALL
        .iter()
        .map(|&stage| {
            MetalRuntime::new_with_shader_source(&probe_source(SHADER_SOURCE, stage))
                .expect("probe runtime")
        })
        .collect::<Vec<_>>();
    let current = MetalRuntime::new_with_shader_source(SHADER_SOURCE).expect("current runtime");
    let baseline = std::env::var_os(BASELINE_SHADERS_ENV).map(|dir| {
        let source = read_shader_dir(std::path::Path::new(&dir));
        MetalRuntime::new_with_shader_source(&source).expect("baseline runtime")
    });

    warm_up_gpu(&current);
    profile_buffer_batches(&probes, &current, baseline.as_ref());
    profile_texture_batches(&current, baseline.as_ref());
}

fn profile_buffer_batches(
    probes: &[MetalRuntime],
    current: &MetalRuntime,
    baseline: Option<&MetalRuntime>,
) {
    println!(
        "GPU ms, median of {PROFILE_SAMPLES}, batch {PROFILE_BATCH}. Probe columns time the decode \
         kernel alone (pack stubbed) and are cumulative; `full` adds the pack pass."
    );
    print!("{:<28}", "workload");
    for stage in ProbeStage::ALL {
        print!(" {:>9}", stage.label());
    }
    println!(
        " {:>9} {:>9} {:>9} {:>9}",
        "full", "baseline", "wall", "base wall"
    );
    for spec in profile_specs() {
        let jpegs = encode_batch(&spec, PROFILE_BATCH);
        let requests = queued_requests(&jpegs, spec.sampling, batch::BatchOp::Full);
        let packets = batched_fast_packets(&requests)
            .expect("packet lookup")
            .expect("fast packets");
        let expected = jpegs
            .iter()
            .map(|jpeg| cpu_rgb(jpeg, batch::BatchOp::Full))
            .collect::<Vec<_>>();
        let mut runtimes = probes.iter().collect::<Vec<_>>();
        runtimes.push(current);
        runtimes.extend(baseline);
        for runtime in &runtimes[probes.len()..] {
            assert_buffer_batch_matches(runtime, spec.sampling, &requests, &packets, &expected);
        }
        let times = profile_runtimes(&runtimes, |runtime| {
            for result in run_full_rgb_batch(runtime, spec.sampling, &requests, &packets) {
                result.expect("probe status must stay OK");
            }
        });
        print!("{:<28}", spec.label());
        for time in &times[..probes.len()] {
            print!(" {:>9.3}", time.gpu_ms);
        }
        print_full_and_baseline(&times[probes.len()..], &jpegs);
    }
}

fn profile_texture_batches(current: &MetalRuntime, baseline: Option<&MetalRuntime>) {
    println!(
        "\nResident RGBA texture batches, batch {PROFILE_BATCH}: whole command buffer. \
         `direct` forces the direct texture kernels instead of component planes."
    );
    println!(
        "{:<28} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "workload", "full", "direct", "baseline", "wall", "dir wall", "base wall"
    );
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    for spec in texture_profile_specs() {
        let jpegs = encode_batch(&spec, PROFILE_BATCH);
        let requests = queued_requests(&jpegs, spec.sampling, batch::BatchOp::Full);
        let packets = batched_fast_packets(&requests)
            .expect("packet lookup")
            .expect("fast packets");
        let output = crate::MetalBatchTextureOutput::new_rgba8_tiles(
            &session,
            spec.dimensions(),
            PROFILE_BATCH,
        )
        .expect("texture output");
        // Variants: component planes, direct kernels, then the baseline shaders.
        let mut variants = vec![(current, true), (current, false)];
        variants.extend(baseline.map(|runtime| (runtime, true)));
        let run_variant = |(runtime, planes): (&MetalRuntime, bool)| {
            crate::compute::texture_tuning::with_component_planes(planes, || {
                run_texture_batch(runtime, spec.sampling, &requests, &packets, &output)
            })
        };
        for &variant in &variants {
            for (index, tile) in run_variant(variant).into_iter().enumerate() {
                let tile = tile.expect("texture tile");
                let actual = crate::tests::download_rgba8_texture(
                    &session,
                    tile.texture_trusted(),
                    tile.dimensions(),
                );
                let expected =
                    crate::tests::rgb_to_rgba_opaque(&cpu_rgb(&jpegs[index], batch::BatchOp::Full));
                assert!(
                    actual == expected,
                    "{} texture tile {index} differs from CPU",
                    spec.label()
                );
            }
        }
        let times = profile_variants(variants.len(), |index| {
            for tile in run_variant(variants[index]) {
                tile.expect("texture tile");
            }
        });
        let cell = |time: Option<&StageTime>, wall: bool| {
            time.map_or("-".to_string(), |time| {
                format!("{:.3}", if wall { time.wall_ms } else { time.gpu_ms })
            })
        };
        println!(
            "{:<28} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}   ({} KiB JPEG)",
            spec.label(),
            cell(times.first(), false),
            cell(times.get(1), false),
            cell(times.get(2), false),
            cell(times.first(), true),
            cell(times.get(1), true),
            cell(times.get(2), true),
            jpegs.iter().map(|jpeg| jpeg.len()).sum::<usize>() / 1024,
        );
    }
}
