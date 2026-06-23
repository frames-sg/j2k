// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)]

use std::time::{Duration, Instant};

use j2k::adapter::encode_stage::{
    J2kDeinterleaveToF32Job, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    J2kForwardDwt53Job, J2kForwardDwt97Job, J2kForwardIctJob, J2kForwardRctJob,
    J2kQuantizeSubbandJob,
};
use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_with_accelerator, encode_j2k_lossy,
    encode_j2k_lossy_with_accelerator, EncodeBackendPreference, EncodedJ2k, EncodedLossyJ2k,
    J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kLossyEncodeOptions, J2kLossySamples, J2kRateTarget,
};
use j2k_core::BackendKind;
use j2k_metal::MetalEncodeStageAccelerator;
use j2k_native::{
    deinterleave_reference, forward_dwt53_reference, forward_dwt97_reference,
    forward_ict_reference, forward_rct_reference, quantize_subband_reference,
};
use j2k_test_support::{patterned_gray8, patterned_rgb8};

const DIMS: &[u32] = &[128, 512, 1024];
const ITERS: usize = 5;
const AUTO_STAGE_MIN_PIXELS: u64 = 512 * 512;
const AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS: u64 = 1024 * 1024;

#[test]
#[ignore = "benchmark harness; run explicitly with --ignored --nocapture"]
fn encode_auto_routing_benchmark() {
    run_stage_benchmarks();
    for &dim in DIMS {
        run_lossless_case(Codec::Classic, Components::Gray8, dim);
        run_lossless_case(Codec::Classic, Components::Rgb8, dim);
        run_lossless_case(Codec::Htj2k, Components::Rgb8, dim);
        run_lossy_case(Codec::Classic, Components::Gray8, dim);
        run_lossy_case(Codec::Htj2k, Components::Gray8, dim);
        run_lossy_case(Codec::Htj2k, Components::Rgb8, dim);
    }
}

fn run_stage_benchmarks() {
    for &dim in DIMS {
        run_deinterleave_stage_benchmark(dim);
        run_forward_rct_stage_benchmark(dim);
        run_forward_ict_stage_benchmark(dim);
        run_forward_dwt53_stage_benchmark(dim);
        run_forward_dwt97_stage_benchmark(dim);
        run_quantize_subband_stage_benchmark(dim);
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
        let options = lossy_options(codec, components, EncodeBackendPreference::CpuOnly);
        let encoded = encode_j2k_lossy(samples, &options).expect("CPU lossy encode");
        assert_eq!(encoded.backend, BackendKind::Cpu);
        encoded.codestream.len()
    });
    let auto = should_bench_auto(&auto_probe, false).then(|| {
        measure(|| {
            let samples = lossy_samples(std::hint::black_box(pixels.as_slice()), dim, components);
            let options = lossy_options(codec, components, EncodeBackendPreference::Auto);
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

fn run_deinterleave_stage_benchmark(dim: u32) {
    let pixels = Components::Rgb8.pixels(dim, dim);
    let num_pixels = usize::try_from(u64::from(dim) * u64::from(dim)).expect("dim fits usize");
    let cpu = measure(|| {
        let planes = deinterleave_reference(
            std::hint::black_box(pixels.as_slice()),
            num_pixels,
            3,
            8,
            false,
        );
        plane_len(&planes)
    });
    let metal = probe_deinterleave_stage(&pixels, num_pixels).map(|dispatch| {
        let timing = measure(|| {
            let mut accelerator = MetalEncodeStageAccelerator::default();
            let planes = accelerator
                .encode_deinterleave(J2kDeinterleaveToF32Job {
                    pixels: std::hint::black_box(pixels.as_slice()),
                    num_pixels,
                    num_components: 3,
                    bit_depth: 8,
                    signed: false,
                })
                .expect("Metal deinterleave stage")
                .expect("Metal deinterleave dispatch");
            plane_len(&planes)
        });
        (timing, dispatch)
    });
    emit_stage_timing("deinterleave", dim, cpu, metal);
}

fn run_forward_rct_stage_benchmark(dim: u32) {
    let planes = stage_planes(dim);
    let cpu = measure(|| {
        let transformed = forward_rct_reference(std::hint::black_box(planes.clone()));
        plane_len(&transformed)
    });
    let metal = probe_forward_rct_stage(&planes).map(|dispatch| {
        let timing = measure(|| {
            let mut stage_planes = planes.clone();
            let mut accelerator = MetalEncodeStageAccelerator::default();
            let (plane0, plane1, plane2) = split_three_planes(&mut stage_planes);
            let dispatched = accelerator
                .encode_forward_rct(J2kForwardRctJob {
                    plane0,
                    plane1,
                    plane2,
                })
                .expect("Metal forward RCT stage");
            assert!(dispatched);
            plane_len(&stage_planes)
        });
        (timing, dispatch)
    });
    emit_stage_timing("forward_rct", dim, cpu, metal);
}

fn run_forward_ict_stage_benchmark(dim: u32) {
    let planes = stage_planes(dim);
    let cpu = measure(|| {
        let transformed = forward_ict_reference(std::hint::black_box(planes.clone()));
        plane_len(&transformed)
    });
    let metal = probe_forward_ict_stage(&planes).map(|dispatch| {
        let timing = measure(|| {
            let mut stage_planes = planes.clone();
            let mut accelerator = MetalEncodeStageAccelerator::default();
            let (plane0, plane1, plane2) = split_three_planes(&mut stage_planes);
            let dispatched = accelerator
                .encode_forward_ict(J2kForwardIctJob {
                    plane0,
                    plane1,
                    plane2,
                })
                .expect("Metal forward ICT stage");
            assert!(dispatched);
            plane_len(&stage_planes)
        });
        (timing, dispatch)
    });
    emit_stage_timing("forward_ict", dim, cpu, metal);
}

fn run_forward_dwt53_stage_benchmark(dim: u32) {
    let samples = stage_samples(dim);
    let cpu = measure(|| {
        let output = forward_dwt53_reference(std::hint::black_box(samples.as_slice()), dim, dim, 1);
        dwt53_len(&output)
    });
    let metal = probe_forward_dwt53_stage(&samples, dim).map(|dispatch| {
        let timing = measure(|| {
            let mut accelerator = MetalEncodeStageAccelerator::default();
            let output = accelerator
                .encode_forward_dwt53(J2kForwardDwt53Job {
                    samples: std::hint::black_box(samples.as_slice()),
                    width: dim,
                    height: dim,
                    num_levels: 1,
                })
                .expect("Metal forward DWT 5/3 stage")
                .expect("Metal forward DWT 5/3 dispatch");
            dwt53_len(&output)
        });
        (timing, dispatch)
    });
    emit_stage_timing("forward_dwt53", dim, cpu, metal);
}

fn run_forward_dwt97_stage_benchmark(dim: u32) {
    let samples = stage_samples(dim);
    let cpu = measure(|| {
        let output = forward_dwt97_reference(std::hint::black_box(samples.as_slice()), dim, dim, 1);
        dwt97_len(&output)
    });
    let metal = probe_forward_dwt97_stage(&samples, dim).map(|dispatch| {
        let timing = measure(|| {
            let mut accelerator = MetalEncodeStageAccelerator::default();
            let output = accelerator
                .encode_forward_dwt97(J2kForwardDwt97Job {
                    samples: std::hint::black_box(samples.as_slice()),
                    width: dim,
                    height: dim,
                    num_levels: 1,
                })
                .expect("Metal forward DWT 9/7 stage")
                .expect("Metal forward DWT 9/7 dispatch");
            dwt97_len(&output)
        });
        (timing, dispatch)
    });
    emit_stage_timing("forward_dwt97", dim, cpu, metal);
}

fn run_quantize_subband_stage_benchmark(dim: u32) {
    let coefficients = stage_samples(dim);
    let cpu = measure(|| {
        let quantized = quantize_subband_reference(
            std::hint::black_box(coefficients.as_slice()),
            8,
            256,
            8,
            false,
        );
        quantized.len()
    });
    let metal = probe_quantize_subband_stage(&coefficients).map(|dispatch| {
        let timing = measure(|| {
            let mut accelerator = MetalEncodeStageAccelerator::default();
            let quantized = accelerator
                .encode_quantize_subband(J2kQuantizeSubbandJob {
                    coefficients: std::hint::black_box(coefficients.as_slice()),
                    step_exponent: 8,
                    step_mantissa: 256,
                    range_bits: 8,
                    reversible: false,
                })
                .expect("Metal quantize_subband stage")
                .expect("Metal quantize_subband dispatch");
            quantized.len()
        });
        (timing, dispatch)
    });
    emit_stage_timing("quantize_subband", dim, cpu, metal);
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

fn lossy_options(
    codec: Codec,
    components: Components,
    backend: EncodeBackendPreference,
) -> J2kLossyEncodeOptions {
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(backend)
        .with_block_coding_mode(codec.block_coding_mode())
        .with_max_decomposition_levels(Some(1))
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(
            8.0 * f64::from(components.count()),
        )))
        .with_validation(J2kEncodeValidation::External);
    options.psnr_iteration_budget = 1;
    options
}

fn stage_samples(dim: u32) -> Vec<f32> {
    let len = usize::try_from(u64::from(dim) * u64::from(dim)).expect("dim fits usize");
    (0..len)
        .map(|idx| stage_sample_value(idx * 37 + idx / 11 + 17))
        .collect()
}

fn stage_planes(dim: u32) -> Vec<Vec<f32>> {
    let len = usize::try_from(u64::from(dim) * u64::from(dim)).expect("dim fits usize");
    (0..3)
        .map(|component| {
            (0..len)
                .map(|idx| {
                    stage_sample_value(idx * (31 + component * 6) + idx / 7 + component * 19)
                })
                .collect()
        })
        .collect()
}

fn stage_sample_value(value: usize) -> f32 {
    f32::from(u8::try_from(value & 0xff).expect("masked stage sample fits u8")) - 128.0
}

fn split_three_planes(planes: &mut [Vec<f32>]) -> (&mut [f32], &mut [f32], &mut [f32]) {
    assert!(planes.len() >= 3);
    let (plane0, rest) = planes.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    (&mut plane0[0], &mut plane1[0], &mut plane2[0])
}

fn plane_len(planes: &[Vec<f32>]) -> usize {
    planes.iter().map(Vec::len).sum()
}

fn dwt53_len(output: &j2k::adapter::encode_stage::J2kForwardDwt53Output) -> usize {
    output.ll.len()
        + output
            .levels
            .iter()
            .map(|level| level.hl.len() + level.lh.len() + level.hh.len())
            .sum::<usize>()
}

fn dwt97_len(output: &j2k::adapter::encode_stage::J2kForwardDwt97Output) -> usize {
    output.ll.len()
        + output
            .levels
            .iter()
            .map(|level| level.hl.len() + level.lh.len() + level.hh.len())
            .sum::<usize>()
}

fn expected_lossless_auto_dispatch(codec: Codec, components: Components, dim: u32) -> bool {
    let pixels = u64::from(dim).saturating_mul(u64::from(dim));
    let resident_htj2k_rgb = matches!(codec, Codec::Htj2k)
        && matches!(components, Components::Rgb8)
        && pixels >= AUTO_HTJ2K_HOST_RESIDENT_MIN_PIXELS;
    let stage_gated_classic = matches!(codec, Codec::Classic) && pixels >= AUTO_STAGE_MIN_PIXELS;
    resident_htj2k_rgb || stage_gated_classic
}

fn probe_deinterleave_stage(
    pixels: &[u8],
    num_pixels: usize,
) -> Result<J2kEncodeDispatchReport, String> {
    let mut accelerator = MetalEncodeStageAccelerator::default();
    let components = accelerator
        .encode_deinterleave(J2kDeinterleaveToF32Job {
            pixels,
            num_pixels,
            num_components: 3,
            bit_depth: 8,
            signed: false,
        })
        .map_err(str::to_string)?;
    if components.is_none() {
        return Err("Metal deinterleave stage did not dispatch".to_string());
    }
    Ok(accelerator.dispatch_report())
}

fn probe_forward_rct_stage(planes: &[Vec<f32>]) -> Result<J2kEncodeDispatchReport, String> {
    let mut stage_planes = planes.to_vec();
    let mut accelerator = MetalEncodeStageAccelerator::default();
    let (plane0, plane1, plane2) = split_three_planes(&mut stage_planes);
    let dispatched = accelerator
        .encode_forward_rct(J2kForwardRctJob {
            plane0,
            plane1,
            plane2,
        })
        .map_err(str::to_string)?;
    if !dispatched {
        return Err("Metal forward RCT stage did not dispatch".to_string());
    }
    Ok(accelerator.dispatch_report())
}

fn probe_forward_ict_stage(planes: &[Vec<f32>]) -> Result<J2kEncodeDispatchReport, String> {
    let mut stage_planes = planes.to_vec();
    let mut accelerator = MetalEncodeStageAccelerator::default();
    let (plane0, plane1, plane2) = split_three_planes(&mut stage_planes);
    let dispatched = accelerator
        .encode_forward_ict(J2kForwardIctJob {
            plane0,
            plane1,
            plane2,
        })
        .map_err(str::to_string)?;
    if !dispatched {
        return Err("Metal forward ICT stage did not dispatch".to_string());
    }
    Ok(accelerator.dispatch_report())
}

fn probe_forward_dwt53_stage(samples: &[f32], dim: u32) -> Result<J2kEncodeDispatchReport, String> {
    let mut accelerator = MetalEncodeStageAccelerator::default();
    let output = accelerator
        .encode_forward_dwt53(J2kForwardDwt53Job {
            samples,
            width: dim,
            height: dim,
            num_levels: 1,
        })
        .map_err(str::to_string)?;
    if output.is_none() {
        return Err("Metal forward DWT 5/3 stage did not dispatch".to_string());
    }
    Ok(accelerator.dispatch_report())
}

fn probe_forward_dwt97_stage(samples: &[f32], dim: u32) -> Result<J2kEncodeDispatchReport, String> {
    let mut accelerator = MetalEncodeStageAccelerator::default();
    let output = accelerator
        .encode_forward_dwt97(J2kForwardDwt97Job {
            samples,
            width: dim,
            height: dim,
            num_levels: 1,
        })
        .map_err(str::to_string)?;
    if output.is_none() {
        return Err("Metal forward DWT 9/7 stage did not dispatch".to_string());
    }
    Ok(accelerator.dispatch_report())
}

fn probe_quantize_subband_stage(coefficients: &[f32]) -> Result<J2kEncodeDispatchReport, String> {
    let mut accelerator = MetalEncodeStageAccelerator::default();
    let quantized = accelerator
        .encode_quantize_subband(J2kQuantizeSubbandJob {
            coefficients,
            step_exponent: 8,
            step_mantissa: 256,
            range_bits: 8,
            reversible: false,
        })
        .map_err(str::to_string)?;
    if quantized.is_none() {
        return Err("Metal quantize_subband stage did not dispatch".to_string());
    }
    Ok(accelerator.dispatch_report())
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
    let options = lossy_options(codec, components, EncodeBackendPreference::Auto);
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

fn emit_stage_timing(
    stage: &str,
    dim: u32,
    cpu: Duration,
    metal: Result<(Duration, J2kEncodeDispatchReport), String>,
) {
    match metal {
        Ok((metal, dispatch)) => println!(
            "j2k_metal_encode_stage_bench stage={stage} size={}x{} cpu_ms={:.3} metal_ms={:.3} dispatch={dispatch:?}",
            dim,
            dim,
            cpu.as_secs_f64() * 1_000.0,
            metal.as_secs_f64() * 1_000.0,
        ),
        Err(error) if std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_some() => {
            panic!("J2K_REQUIRE_METAL_BENCH is set but Metal stage bench failed: {error}");
        }
        Err(error) => println!(
            "j2k_metal_encode_stage_bench stage={stage} size={}x{} cpu_ms={:.3} metal_ms=skipped error={error}",
            dim,
            dim,
            cpu.as_secs_f64() * 1_000.0,
        ),
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
