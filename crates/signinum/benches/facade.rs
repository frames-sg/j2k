// SPDX-License-Identifier: Apache-2.0

use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(any(feature = "metal", feature = "cuda"))]
use signinum::j2k::J2kEncodeStageAccelerator;
use signinum::j2k::{
    encode_j2k_lossless as facade_encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
};
#[cfg(any(feature = "metal", feature = "cuda"))]
use signinum::j2k::{encode_j2k_lossless_with_accelerator, BackendKind};
use signinum_test_support::patterned_rgb8;

const TILE_SIDE: u32 = 128;
const MATRIX_SIDE: u32 = 512;

fn patterned_rgba8(width: u32, height: u32) -> Vec<u8> {
    let rgb = patterned_rgb8(width, height);
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.extend_from_slice(chunk);
        rgba.push(255);
    }
    rgba
}

struct FacadeMatrixCase {
    label: &'static str,
    width: u32,
    height: u32,
    components: u8,
    pixels: Vec<u8>,
}

fn bench_encode_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_validation(J2kEncodeValidation::External)
}

fn matrix_encode_options(
    backend: EncodeBackendPreference,
    block_coding_mode: J2kBlockCodingMode,
) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(backend)
        .with_validation(J2kEncodeValidation::External)
        .with_block_coding_mode(block_coding_mode)
}

fn bench_facade_j2k_encode(c: &mut Criterion) {
    let pixels = patterned_rgb8(TILE_SIDE, TILE_SIDE);
    let options = bench_encode_options();

    let mut group = c.benchmark_group("facade_j2k_lossless_encode");
    group.bench_function("facade_cpu_only_rgb8_128x128", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                TILE_SIDE,
                TILE_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                facade_encode_j2k_lossless(samples, &options).expect("facade cpu-only encode");
            black_box(encoded.codestream.len());
        });
    });

    group.bench_function("direct_cpu_only_rgb8_128x128", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                TILE_SIDE,
                TILE_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                signinum_j2k::encode_j2k_lossless(samples, &options).expect("direct cpu encode");
            black_box(encoded.codestream.len());
        });
    });
    group.finish();
}

fn bench_facade_cpu_matrix(c: &mut Criterion) {
    let pixels = patterned_rgb8(MATRIX_SIDE, MATRIX_SIDE);
    let classic_options = matrix_encode_options(
        EncodeBackendPreference::CpuOnly,
        J2kBlockCodingMode::Classic,
    );
    let htj2k_options = matrix_encode_options(
        EncodeBackendPreference::CpuOnly,
        J2kBlockCodingMode::HighThroughput,
    );

    let mut group = c.benchmark_group("facade_j2k_lossless_encode_cpu_matrix");
    group.bench_function("cpu_only_rgb8_512_classic_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                MATRIX_SIDE,
                MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                facade_encode_j2k_lossless(samples, &classic_options).expect("classic CPU encode");
            black_box(encoded.codestream.len());
        });
    });

    group.bench_function("cpu_only_rgb8_512_htj2k_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                MATRIX_SIDE,
                MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                facade_encode_j2k_lossless(samples, &htj2k_options).expect("HTJ2K CPU encode");
            black_box(encoded.codestream.len());
        });
    });
    group.finish();
}

fn bench_facade_adaptive_matrix(c: &mut Criterion) {
    let pixels = patterned_rgb8(MATRIX_SIDE, MATRIX_SIDE);
    let auto_classic = matrix_encode_options(
        EncodeBackendPreference::ACCELERATED,
        J2kBlockCodingMode::Classic,
    );
    let auto_htj2k = matrix_encode_options(
        EncodeBackendPreference::ACCELERATED,
        J2kBlockCodingMode::HighThroughput,
    );

    let mut group = c.benchmark_group("facade_j2k_lossless_encode_adaptive_matrix");
    group.bench_function("adaptive_rgb8_512_classic_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                MATRIX_SIDE,
                MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded =
                facade_encode_j2k_lossless(samples, &auto_classic).expect("facade adaptive encode");
            black_box((encoded.backend, encoded.codestream.len()));
        });
    });

    group.bench_function("adaptive_rgb8_512_htj2k_external", |b| {
        b.iter(|| {
            let samples = J2kLosslessSamples::new(
                black_box(pixels.as_slice()),
                MATRIX_SIDE,
                MATRIX_SIDE,
                3,
                8,
                false,
            )
            .expect("valid rgb8 samples");
            let encoded = facade_encode_j2k_lossless(samples, &auto_htj2k)
                .expect("facade adaptive HTJ2K encode");
            black_box((encoded.backend, encoded.codestream.len()));
        });
    });

    #[cfg(feature = "metal")]
    {
        group.bench_function("direct_metal_auto_stage_rgb8_512_classic_external", |b| {
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    black_box(pixels.as_slice()),
                    MATRIX_SIDE,
                    MATRIX_SIDE,
                    3,
                    8,
                    false,
                )
                .expect("valid rgb8 samples");
                let mut accelerator =
                    signinum::j2k::metal::MetalEncodeStageAccelerator::for_auto_host_output();
                let encoded = encode_j2k_lossless_with_accelerator(
                    samples,
                    &auto_classic,
                    BackendKind::Metal,
                    &mut accelerator,
                )
                .expect("direct Metal-stage classic encode");
                black_box((encoded.backend, encoded.codestream.len()));
            });
        });

        group.bench_function("direct_metal_cpu_rct_stage_rgb8_512_htj2k_external", |b| {
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    black_box(pixels.as_slice()),
                    MATRIX_SIDE,
                    MATRIX_SIDE,
                    3,
                    8,
                    false,
                )
                .expect("valid rgb8 samples");
                let mut accelerator =
                    signinum::j2k::metal::MetalEncodeStageAccelerator::with_cpu_forward_rct();
                let encoded = encode_j2k_lossless_with_accelerator(
                    samples,
                    &auto_htj2k,
                    BackendKind::Metal,
                    &mut accelerator,
                )
                .expect("direct Metal-stage HTJ2K encode");
                black_box((encoded.backend, encoded.codestream.len()));
            });
        });
    }

    group.finish();
}

fn bench_facade_backend_speed_matrix(c: &mut Criterion) {
    let cases = [
        FacadeMatrixCase {
            label: "rgb8_512",
            width: 512,
            height: 512,
            components: 3,
            pixels: patterned_rgb8(512, 512),
        },
        FacadeMatrixCase {
            label: "rgb8_1024",
            width: 1024,
            height: 1024,
            components: 3,
            pixels: patterned_rgb8(1024, 1024),
        },
        FacadeMatrixCase {
            label: "rgba8_512",
            width: 512,
            height: 512,
            components: 4,
            pixels: patterned_rgba8(512, 512),
        },
        FacadeMatrixCase {
            label: "rgba8_1024",
            width: 1024,
            height: 1024,
            components: 4,
            pixels: patterned_rgba8(1024, 1024),
        },
    ];
    let cpu_options = matrix_encode_options(
        EncodeBackendPreference::CPU_ONLY,
        J2kBlockCodingMode::HighThroughput,
    );
    let adaptive_options = matrix_encode_options(
        EncodeBackendPreference::ACCELERATED,
        J2kBlockCodingMode::HighThroughput,
    );
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let strict_options = matrix_encode_options(
        EncodeBackendPreference::STRICT_DEVICE,
        J2kBlockCodingMode::HighThroughput,
    );

    let mut group = c.benchmark_group("facade_j2k_htj2k_encode_backend_speed_matrix");
    for case in &cases {
        let cpu_name = format!("cpu_{}_htj2k_external", case.label);
        group.bench_function(cpu_name.as_str(), |b| {
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    black_box(case.pixels.as_slice()),
                    case.width,
                    case.height,
                    case.components,
                    8,
                    false,
                )
                .expect("valid matrix samples");
                let encoded =
                    facade_encode_j2k_lossless(samples, &cpu_options).expect("CPU HTJ2K encode");
                black_box(encoded.codestream.len());
            });
        });

        let adaptive_name = format!("adaptive_{}_htj2k_perf_gate_external", case.label);
        group.bench_function(adaptive_name.as_str(), |b| {
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    black_box(case.pixels.as_slice()),
                    case.width,
                    case.height,
                    case.components,
                    8,
                    false,
                )
                .expect("valid matrix samples");
                let encoded = facade_encode_j2k_lossless(samples, &adaptive_options)
                    .expect("adaptive HTJ2K encode");
                black_box((encoded.backend, encoded.codestream.len()));
            });
        });

        #[cfg(feature = "metal")]
        if metal_htj2k_encode_available(case, strict_options) {
            let metal_name = format!("strict_metal_{}_htj2k_external", case.label);
            group.bench_function(metal_name.as_str(), |b| {
                b.iter(|| {
                    let samples = J2kLosslessSamples::new(
                        black_box(case.pixels.as_slice()),
                        case.width,
                        case.height,
                        case.components,
                        8,
                        false,
                    )
                    .expect("valid matrix samples");
                    let mut accelerator =
                        signinum::j2k::metal::MetalEncodeStageAccelerator::default();
                    let encoded = encode_j2k_lossless_with_accelerator(
                        samples,
                        &strict_options,
                        BackendKind::Metal,
                        &mut accelerator,
                    )
                    .expect("strict Metal HTJ2K encode");
                    assert_eq!(
                        encoded.backend,
                        BackendKind::Metal,
                        "Metal speed bench must report a strict Metal backend"
                    );
                    black_box((encoded.backend, encoded.codestream.len()));
                });
            });
        }

        #[cfg(feature = "cuda")]
        if cuda_htj2k_encode_available(case, strict_options) {
            let cuda_name = format!("strict_cuda_{}_htj2k_external", case.label);
            group.bench_function(cuda_name.as_str(), |b| {
                b.iter(|| {
                    let samples = J2kLosslessSamples::new(
                        black_box(case.pixels.as_slice()),
                        case.width,
                        case.height,
                        case.components,
                        8,
                        false,
                    )
                    .expect("valid matrix samples");
                    let mut accelerator =
                        signinum::j2k::cuda::CudaEncodeStageAccelerator::default();
                    let encoded = encode_j2k_lossless_with_accelerator(
                        samples,
                        &strict_options,
                        BackendKind::Cuda,
                        &mut accelerator,
                    )
                    .expect("CUDA HTJ2K encode");
                    assert_eq!(
                        encoded.backend,
                        BackendKind::Cuda,
                        "CUDA speed bench must report a strict CUDA backend"
                    );
                    assert!(
                        accelerator.dispatch_report().any(),
                        "CUDA speed bench must dispatch at least one CUDA stage"
                    );
                    black_box((encoded.backend, encoded.codestream.len()));
                });
            });
        }
    }

    group.finish();
}

#[cfg(feature = "metal")]
fn metal_htj2k_encode_available(
    case: &FacadeMatrixCase,
    options: J2kLosslessEncodeOptions,
) -> bool {
    let samples = J2kLosslessSamples::new(
        case.pixels.as_slice(),
        case.width,
        case.height,
        case.components,
        8,
        false,
    )
    .expect("samples");
    let case_context = format!(
        "{} {}x{} components={}",
        case.label, case.width, case.height, case.components
    );
    let mut accelerator = signinum::j2k::metal::MetalEncodeStageAccelerator::default();
    match encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    ) {
        Ok(encoded)
            if encoded.backend == BackendKind::Metal && accelerator.dispatch_report().any() =>
        {
            true
        }
        Ok(_) if std::env::var_os("SIGNINUM_REQUIRE_METAL_BENCH").is_some() => {
            panic!(
                "SIGNINUM_REQUIRE_METAL_BENCH is set but strict Metal encode was not available for {case_context}"
            )
        }
        Ok(_) => {
            eprintln!(
                "skipping Metal encode speed bench for {case_context}: strict Metal encode was not available"
            );
            false
        }
        Err(error) if std::env::var_os("SIGNINUM_REQUIRE_METAL_BENCH").is_some() => {
            panic!(
                "SIGNINUM_REQUIRE_METAL_BENCH is set but Metal encode probe failed for {case_context}: {error}"
            )
        }
        Err(error) => {
            eprintln!("skipping Metal encode speed bench for {case_context}: {error}");
            false
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_htj2k_encode_available(case: &FacadeMatrixCase, options: J2kLosslessEncodeOptions) -> bool {
    let samples = J2kLosslessSamples::new(
        case.pixels.as_slice(),
        case.width,
        case.height,
        case.components,
        8,
        false,
    )
    .expect("samples");
    let case_context = format!(
        "{} {}x{} components={}",
        case.label, case.width, case.height, case.components
    );
    let mut accelerator = signinum::j2k::cuda::CudaEncodeStageAccelerator::default();
    match encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Cuda,
        &mut accelerator,
    ) {
        Ok(encoded)
            if encoded.backend == BackendKind::Cuda && accelerator.dispatch_report().any() =>
        {
            true
        }
        Ok(_) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!(
                "SIGNINUM_REQUIRE_CUDA_BENCH is set but strict CUDA encode was not available for {case_context}"
            )
        }
        Ok(_) => {
            eprintln!(
                "skipping CUDA encode speed bench for {case_context}: strict CUDA encode was not available"
            );
            false
        }
        Err(error) if std::env::var_os("SIGNINUM_REQUIRE_CUDA_BENCH").is_some() => {
            panic!(
                "SIGNINUM_REQUIRE_CUDA_BENCH is set but CUDA encode probe failed for {case_context}: {error}"
            )
        }
        Err(error) => {
            eprintln!("skipping CUDA encode speed bench for {case_context}: {error}");
            false
        }
    }
}

criterion_group!(
    benches,
    bench_facade_j2k_encode,
    bench_facade_cpu_matrix,
    bench_facade_adaptive_matrix,
    bench_facade_backend_speed_matrix
);
criterion_main!(benches);
