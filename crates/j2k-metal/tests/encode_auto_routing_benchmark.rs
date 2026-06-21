// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use j2k::adapter::encode_stage::J2kEncodeDispatchReport;
use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, encode_j2k_lossy,
    encode_j2k_lossy_with_accelerator, EncodeBackendPreference, EncodedJ2k, EncodedLossyJ2k,
    J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kLossyEncodeOptions, J2kLossySamples, J2kRateTarget,
};
use j2k_core::BackendKind;
use j2k_metal::MetalEncodeStageAccelerator;
use j2k_test_support::{patterned_gray8, patterned_rgb8};

const DIMS: &[u32] = &[128, 512, 1024];
const ITERS: usize = 5;
const AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS: u64 = 1024 * 1024;

#[test]
#[ignore = "benchmark harness; run explicitly with --ignored --nocapture"]
fn encode_auto_routing_benchmark() {
    for &dim in DIMS {
        run_lossless_case(Codec::Classic, Components::Gray8, dim);
        run_lossless_case(Codec::Htj2k, Components::Rgb8, dim);
        run_lossy_case(Codec::Classic, Components::Gray8, dim);
        run_lossy_case(Codec::Htj2k, Components::Gray8, dim);
    }
}

#[derive(Clone, Copy)]
enum Codec {
    Classic,
    Htj2k,
}

impl Codec {
    const fn block_coding_mode(self) -> J2kBlockCodingMode {
        match self {
            Self::Classic => J2kBlockCodingMode::Classic,
            Self::Htj2k => J2kBlockCodingMode::HighThroughput,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Htj2k => "htj2k",
        }
    }
}

#[derive(Clone, Copy)]
enum Components {
    Gray8,
    Rgb8,
}

impl Components {
    const fn count(self) -> u8 {
        match self {
            Self::Gray8 => 1,
            Self::Rgb8 => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Gray8 => "gray8",
            Self::Rgb8 => "rgb8",
        }
    }

    fn pixels(self, width: u32, height: u32) -> Vec<u8> {
        match self {
            Self::Gray8 => patterned_gray8(width, height),
            Self::Rgb8 => patterned_rgb8(width, height),
        }
    }
}

fn run_lossless_case(codec: Codec, components: Components, dim: u32) {
    let pixels = components.pixels(dim, dim);
    let auto_probe = probe_lossless_auto(&pixels, dim, codec, components);
    emit_probe("lossless", codec, components, dim, &auto_probe);
    let cpu = measure(|| {
        let samples = lossless_samples(std::hint::black_box(pixels.as_slice()), dim, components);
        let options = lossless_options(codec, EncodeBackendPreference::CpuOnly);
        let encoded = encode_j2k_lossless(samples, &options).expect("CPU lossless encode");
        assert_eq!(encoded.backend, BackendKind::Cpu);
        encoded.codestream.len()
    });
    let expected_dispatch = expected_lossless_auto_dispatch(codec, components, dim);
    let auto = should_bench_auto(&auto_probe, expected_dispatch).then(|| {
        measure(|| {
            let samples =
                lossless_samples(std::hint::black_box(pixels.as_slice()), dim, components);
            let options = lossless_options(codec, EncodeBackendPreference::Auto);
            let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();
            let encoded = encode_j2k_lossless_with_accelerator(
                samples,
                &options,
                BackendKind::Metal,
                &mut accelerator,
            )
            .expect("Auto Metal lossless encode");
            encoded.codestream.len()
        })
    });
    emit_timing("lossless", codec, components, dim, cpu, auto);
}

fn run_lossy_case(codec: Codec, components: Components, dim: u32) {
    let pixels = components.pixels(dim, dim);
    let auto_probe = probe_lossy_auto(&pixels, dim, codec, components);
    emit_probe("lossy", codec, components, dim, &auto_probe);
    let cpu = measure(|| {
        let samples = lossy_samples(std::hint::black_box(pixels.as_slice()), dim, components);
        let options = lossy_options(codec, EncodeBackendPreference::CpuOnly);
        let encoded = encode_j2k_lossy(samples, &options).expect("CPU lossy encode");
        assert_eq!(encoded.backend, BackendKind::Cpu);
        encoded.codestream.len()
    });
    let auto = should_bench_auto(&auto_probe, false).then(|| {
        measure(|| {
            let samples = lossy_samples(std::hint::black_box(pixels.as_slice()), dim, components);
            let options = lossy_options(codec, EncodeBackendPreference::Auto);
            let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();
            let encoded = encode_j2k_lossy_with_accelerator(
                samples,
                &options,
                BackendKind::Metal,
                &mut accelerator,
            )
            .expect("Auto Metal lossy encode");
            encoded.codestream.len()
        })
    });
    emit_timing("lossy", codec, components, dim, cpu, auto);
}

fn measure(mut run: impl FnMut() -> usize) -> Duration {
    std::hint::black_box(run());
    let mut durations = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let started = Instant::now();
        std::hint::black_box(run());
        durations.push(started.elapsed());
    }
    durations.sort_unstable();
    durations[durations.len() / 2]
}

fn lossless_samples(pixels: &[u8], dim: u32, components: Components) -> J2kLosslessSamples<'_> {
    J2kLosslessSamples::new(pixels, dim, dim, components.count(), 8, false)
        .expect("valid lossless samples")
}

fn lossy_samples(pixels: &[u8], dim: u32, components: Components) -> J2kLossySamples<'_> {
    J2kLossySamples::new(pixels, dim, dim, components.count(), 8, false)
        .expect("valid lossy samples")
}

fn lossless_options(codec: Codec, backend: EncodeBackendPreference) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(backend)
        .with_block_coding_mode(codec.block_coding_mode())
        .with_max_decomposition_levels(Some(1))
        .with_validation(J2kEncodeValidation::External)
}

fn lossy_options(codec: Codec, backend: EncodeBackendPreference) -> J2kLossyEncodeOptions {
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(backend)
        .with_block_coding_mode(codec.block_coding_mode())
        .with_max_decomposition_levels(Some(1))
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(8.0)))
        .with_validation(J2kEncodeValidation::External);
    options.psnr_iteration_budget = 1;
    options
}

fn expected_lossless_auto_dispatch(codec: Codec, components: Components, dim: u32) -> bool {
    matches!(codec, Codec::Htj2k)
        && matches!(components, Components::Rgb8)
        && u64::from(dim).saturating_mul(u64::from(dim)) >= AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS
}

fn probe_lossless_auto(
    pixels: &[u8],
    dim: u32,
    codec: Codec,
    components: Components,
) -> Result<J2kEncodeDispatchReport, String> {
    let samples = lossless_samples(pixels, dim, components);
    let options = lossless_options(codec, EncodeBackendPreference::Auto);
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();
    encode_j2k_lossless_with_accelerator(samples, &options, BackendKind::Metal, &mut accelerator)
        .map(|encoded: EncodedJ2k| encoded.dispatch_report)
        .map_err(|error| error.to_string())
}

fn probe_lossy_auto(
    pixels: &[u8],
    dim: u32,
    codec: Codec,
    components: Components,
) -> Result<J2kEncodeDispatchReport, String> {
    let samples = lossy_samples(pixels, dim, components);
    let options = lossy_options(codec, EncodeBackendPreference::Auto);
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();
    encode_j2k_lossy_with_accelerator(samples, &options, BackendKind::Metal, &mut accelerator)
        .map(|encoded: EncodedLossyJ2k| encoded.dispatch_report)
        .map_err(|error| error.to_string())
}

fn should_bench_auto(
    probe: &Result<J2kEncodeDispatchReport, String>,
    expected_dispatch: bool,
) -> bool {
    match probe {
        Ok(dispatch) if !expected_dispatch || *dispatch != J2kEncodeDispatchReport::default() => {
            true
        }
        Ok(_) if std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_some() => {
            panic!("J2K_REQUIRE_METAL_BENCH is set but Auto Metal encode did not dispatch")
        }
        Ok(_) => {
            eprintln!("skipping Auto Metal encode bench: route did not dispatch");
            false
        }
        Err(error) if std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_some() => {
            panic!("J2K_REQUIRE_METAL_BENCH is set but Auto Metal encode failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping Auto Metal encode bench: {error}");
            false
        }
    }
}

fn emit_probe(
    mode: &str,
    codec: Codec,
    components: Components,
    dim: u32,
    probe: &Result<J2kEncodeDispatchReport, String>,
) {
    match probe {
        Ok(dispatch) => println!(
            "j2k_metal_encode_auto_probe mode={mode} codec={} components={} size={}x{} dispatch={dispatch:?}",
            codec.label(),
            components.label(),
            dim,
            dim
        ),
        Err(error) => println!(
            "j2k_metal_encode_auto_probe mode={mode} codec={} components={} size={}x{} error={error}",
            codec.label(),
            components.label(),
            dim,
            dim
        ),
    }
}

fn emit_timing(
    mode: &str,
    codec: Codec,
    components: Components,
    dim: u32,
    cpu: Duration,
    auto: Option<Duration>,
) {
    let auto_ms = auto.map_or_else(
        || "skipped".to_string(),
        |duration| format!("{:.3}", duration.as_secs_f64() * 1_000.0),
    );
    println!(
        "j2k_metal_encode_auto_bench mode={mode} codec={} components={} size={}x{} cpu_ms={:.3} auto_ms={auto_ms}",
        codec.label(),
        components.label(),
        dim,
        dim,
        cpu.as_secs_f64() * 1_000.0
    );
}
