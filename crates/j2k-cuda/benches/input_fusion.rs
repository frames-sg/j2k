// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fmt::Write, hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use j2k::{
    encode_j2k_lossless_with_accelerator, encode_j2k_lossy_with_accelerator,
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeDispatchReport, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kLossyEncodeOptions, J2kLossySamples,
    J2kRateTarget,
};
use j2k_core::BackendKind;
use j2k_cuda::CudaEncodeStageAccelerator;
use j2k_cuda_j2k_engine::J2kCudaEngine;
use j2k_cuda_runtime::CudaContext;
use j2k_native::{
    forward_ict_reference, forward_rct_reference, try_deinterleave_reference, DecodeSettings, Image,
};
use sha2::{Digest, Sha256};

const DIMENSION: usize = 512;
const DIMENSION_U32: u32 = 512;
const PRODUCT_LEVELS: u8 = 3;
const SAMPLE_SIZE: usize = 10;
const WARM_UP: Duration = Duration::from_secs(1);
const MEASUREMENT: Duration = Duration::from_secs(3);
struct RouteRun {
    components: Vec<Vec<f32>>,
    deinterleave_dispatches: usize,
    mct_dispatches: usize,
}

impl RouteRun {
    const fn physical_dispatches(&self) -> usize {
        self.deinterleave_dispatches + self.mct_dispatches
    }
}

fn bench_cuda_input_fusion(c: &mut Criterion) {
    let Some(context) = cuda_context() else {
        return;
    };
    let pixels = rgb8_fixture(DIMENSION, DIMENSION);
    probe_route(&context, &pixels, true);
    probe_route(&context, &pixels, false);

    let mut group = c.benchmark_group("j2k_cuda_p16_input_fusion_rgb8_512");
    group.bench_with_input(BenchmarkId::new("rct", DIMENSION), &pixels, |b, pixels| {
        b.iter(|| black_box(run_input_route(&context, pixels, true)));
    });
    group.bench_with_input(BenchmarkId::new("ict", DIMENSION), &pixels, |b, pixels| {
        b.iter(|| black_box(run_input_route(&context, pixels, false)));
    });
    group.finish();
}

#[derive(Clone, Copy)]
enum ProductCell {
    RctLossless,
    IctLossy,
}

impl ProductCell {
    const fn workload(self) -> &'static str {
        match self {
            Self::RctLossless => "rct_lossless_htj2k_rgb8_512x512",
            Self::IctLossy => "ict_lossy_htj2k_rgb8_512x512",
        }
    }

    const fn transform(self) -> &'static str {
        match self {
            Self::RctLossless => "rct",
            Self::IctLossy => "ict",
        }
    }
}

struct ProductOptions {
    lossless: J2kLosslessEncodeOptions,
    lossy: J2kLossyEncodeOptions,
}

struct ProductRun {
    codestream: Vec<u8>,
    dispatch: J2kEncodeDispatchReport,
}

struct ProductRouteCounters {
    deinterleave: usize,
    mct: usize,
    physical_input: usize,
    total: usize,
}

fn product_options() -> ProductOptions {
    let lossless = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(PRODUCT_LEVELS))
        .with_validation(J2kEncodeValidation::External);
    let mut lossy = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(PRODUCT_LEVELS))
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(4.0)))
        .with_validation(J2kEncodeValidation::External);
    lossy.psnr_iteration_budget = 1;
    ProductOptions { lossless, lossy }
}

fn run_product(
    accelerator: &mut CudaEncodeStageAccelerator,
    pixels: &[u8],
    cell: ProductCell,
    options: &ProductOptions,
) -> ProductRun {
    match cell {
        ProductCell::RctLossless => {
            let samples =
                J2kLosslessSamples::new(pixels, DIMENSION_U32, DIMENSION_U32, 3, 8, false)
                    .expect("valid P16 lossless RGB8 product samples");
            let encoded = encode_j2k_lossless_with_accelerator(
                samples,
                &options.lossless,
                BackendKind::Cuda,
                accelerator,
            )
            .expect("P16 lossless CUDA HTJ2K product encode");
            assert_eq!(encoded.backend, BackendKind::Cuda);
            ProductRun {
                codestream: encoded.codestream,
                dispatch: encoded.dispatch_report,
            }
        }
        ProductCell::IctLossy => {
            let samples = J2kLossySamples::new(pixels, DIMENSION_U32, DIMENSION_U32, 3, 8, false)
                .expect("valid P16 lossy RGB8 product samples");
            let encoded = encode_j2k_lossy_with_accelerator(
                samples,
                &options.lossy,
                BackendKind::Cuda,
                accelerator,
            )
            .expect("P16 lossy CUDA HTJ2K product encode");
            assert_eq!(encoded.backend, BackendKind::Cuda);
            ProductRun {
                codestream: encoded.codestream,
                dispatch: encoded.dispatch_report,
            }
        }
    }
}

fn product_route_counters(
    dispatch: J2kEncodeDispatchReport,
    cell: ProductCell,
) -> ProductRouteCounters {
    assert_eq!(dispatch.deinterleave, 1, "P16 product deinterleave route");
    let mct = match cell {
        ProductCell::RctLossless => {
            assert_eq!(dispatch.forward_ict, 0, "P16 lossless ICT dispatches");
            dispatch.forward_rct
        }
        ProductCell::IctLossy => {
            assert_eq!(dispatch.forward_rct, 0, "P16 lossy RCT dispatches");
            dispatch.forward_ict
        }
    };
    assert_eq!(mct, 1, "P16 product MCT route");
    ProductRouteCounters {
        deinterleave: dispatch.deinterleave,
        mct,
        physical_input: dispatch.deinterleave + mct,
        total: dispatch.total(),
    }
}

fn decode_product(codestream: &[u8], pixels: &[u8], cell: ProductCell) -> (String, f64) {
    let decoded = Image::new(codestream, &DecodeSettings::strict())
        .expect("P16 product codestream parses")
        .decode_native()
        .expect("P16 product codestream decodes");
    assert_eq!(decoded.width, DIMENSION_U32);
    assert_eq!(decoded.height, DIMENSION_U32);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bit_depth, 8);
    assert!(!decoded.signed);
    assert_eq!(decoded.data.len(), pixels.len());

    let psnr_db = match cell {
        ProductCell::RctLossless => {
            assert_eq!(decoded.data, pixels, "P16 lossless product decoded pixels");
            f64::INFINITY
        }
        ProductCell::IctLossy => {
            let squared_error = decoded
                .data
                .iter()
                .zip(pixels)
                .map(|(&actual, &expected)| {
                    let error = f64::from(actual) - f64::from(expected);
                    error * error
                })
                .sum::<f64>();
            let sample_count =
                u32::try_from(pixels.len()).expect("P16 RGB8 product sample count fits u32");
            let mse = squared_error / f64::from(sample_count);
            let psnr_db = 10.0 * (255.0_f64 * 255.0 / mse).log10();
            assert!(
                psnr_db.is_finite() && psnr_db > 0.0,
                "P16 lossy product PSNR must be finite and positive, got {psnr_db}"
            );
            psnr_db
        }
    };
    (
        byte_slice_sha256(b"j2k-cuda-p16-decoded-rgb8-v1", &decoded.data),
        psnr_db,
    )
}

fn emit_product_probe(pixels: &[u8], cell: ProductCell, options: &ProductOptions) {
    let first = run_product(
        &mut CudaEncodeStageAccelerator::default(),
        pixels,
        cell,
        options,
    );
    let second = run_product(
        &mut CudaEncodeStageAccelerator::default(),
        pixels,
        cell,
        options,
    );
    let first_validation = decode_product(&first.codestream, pixels, cell);
    let second_validation = decode_product(&second.codestream, pixels, cell);
    assert_eq!(
        first.codestream, second.codestream,
        "P16 product determinism"
    );
    assert_eq!(first_validation.0, second_validation.0);
    assert_eq!(first_validation.1.to_bits(), second_validation.1.to_bits());

    let first_counters = product_route_counters(first.dispatch, cell);
    let second_counters = product_route_counters(second.dispatch, cell);
    assert_eq!(
        (
            first_counters.deinterleave,
            first_counters.mct,
            first_counters.physical_input,
            first_counters.total,
        ),
        (
            second_counters.deinterleave,
            second_counters.mct,
            second_counters.physical_input,
            second_counters.total,
        ),
        "P16 repeated product route contract"
    );

    let rate_target = match cell {
        ProductCell::RctLossless => "lossless",
        ProductCell::IctLossy => "4.0_bpp",
    };
    let codestream_sha256 =
        byte_slice_sha256(b"j2k-cuda-p16-complete-codestream-v1", &first.codestream);
    println!(
        "j2k_cuda_p16_product_probe workload={} product_route=separate_baseline transform={} coding=htj2k rate_target={} width=512 height=512 components=3 bit_depth=8 signed=false levels={} input_sha256={} output_sha256={} codestream_sha256={} decoded_sha256={} psnr_db={} exact_parity=true deterministic=true product_deinterleave_dispatches={} product_mct_dispatches={} product_physical_input_dispatches={} product_total_dispatches={}",
        cell.workload(),
        cell.transform(),
        rate_target,
        PRODUCT_LEVELS,
        input_sha256(pixels),
        codestream_sha256,
        codestream_sha256,
        first_validation.0,
        first_validation.1,
        first_counters.deinterleave,
        first_counters.mct,
        first_counters.physical_input,
        first_counters.total,
    );
}

fn bench_cuda_input_fusion_product(c: &mut Criterion) {
    if cuda_context().is_none() {
        return;
    }
    let pixels = rgb8_fixture(DIMENSION, DIMENSION);
    let options = product_options();
    emit_product_probe(&pixels, ProductCell::RctLossless, &options);
    emit_product_probe(&pixels, ProductCell::IctLossy, &options);

    let mut group = c.benchmark_group("j2k_cuda_p16_product_encode");
    group.throughput(Throughput::Elements(
        u64::from(DIMENSION_U32) * u64::from(DIMENSION_U32),
    ));
    let mut rct_accelerator = CudaEncodeStageAccelerator::default();
    group.bench_with_input(
        BenchmarkId::new("rct_lossless_htj2k", DIMENSION),
        &pixels,
        |bencher, pixels| {
            bencher.iter(|| {
                black_box(run_product(
                    &mut rct_accelerator,
                    black_box(pixels),
                    ProductCell::RctLossless,
                    &options,
                ))
            });
        },
    );
    let mut ict_accelerator = CudaEncodeStageAccelerator::default();
    group.bench_with_input(
        BenchmarkId::new("ict_lossy_htj2k", DIMENSION),
        &pixels,
        |bencher, pixels| {
            bencher.iter(|| {
                black_box(run_product(
                    &mut ict_accelerator,
                    black_box(pixels),
                    ProductCell::IctLossy,
                    &options,
                ))
            });
        },
    );
    group.finish();
}

fn probe_route(context: &CudaContext, pixels: &[u8], reversible: bool) {
    let first = run_input_route(context, pixels, reversible);
    let second = run_input_route(context, pixels, reversible);
    assert_components_exact(&first.components, &second.components, "P16 determinism");

    let separate = try_deinterleave_reference(pixels, DIMENSION * DIMENSION, 3, 8, false)
        .expect("native RGB8 deinterleave reference");
    let expected = if reversible {
        forward_rct_reference(separate)
    } else {
        forward_ict_reference(separate)
    };
    assert_components_exact(&first.components, &expected, "P16 exact parity");

    let expected_counts = (1, 1, 2);
    assert_eq!(
        (
            first.deinterleave_dispatches,
            first.mct_dispatches,
            first.physical_dispatches(),
        ),
        expected_counts,
        "P16 route dispatch contract"
    );
    assert_eq!(
        (
            second.deinterleave_dispatches,
            second.mct_dispatches,
            second.physical_dispatches(),
        ),
        expected_counts,
        "P16 repeated route dispatch contract"
    );

    let input_sha256 = input_sha256(pixels);
    let output_digest = output_sha256(&first.components);
    assert_eq!(output_digest, output_sha256(&second.components));
    let transform = if reversible { "rct" } else { "ict" };
    eprintln!(
        "p16_input_fusion_probe route=separate_baseline transform={transform} width=512 height=512 bit_depth=8 signed=false hash_framing=le64_length_prefixed input_sha256={input_sha256} output_sha256={output_digest} exact_parity=true deterministic=true deinterleave_dispatches={} mct_dispatches={} physical_dispatches={}",
        first.deinterleave_dispatches,
        first.mct_dispatches,
        first.physical_dispatches(),
    );
}

fn run_input_route(context: &CudaContext, pixels: &[u8], reversible: bool) -> RouteRun {
    let engine = J2kCudaEngine::new(context);
    let output = engine
        .j2k_deinterleave_to_f32(pixels, DIMENSION * DIMENSION, 3, 8, false)
        .expect("CUDA P16 separate deinterleave");
    let deinterleave_dispatches = output.execution().kernel_dispatches();
    let mut components = output.into_components();
    let (plane0, rest) = components.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    let execution = if reversible {
        engine.j2k_forward_rct(&mut plane0[0], &mut plane1[0], &mut plane2[0])
    } else {
        engine.j2k_forward_ict(&mut plane0[0], &mut plane1[0], &mut plane2[0])
    }
    .expect("CUDA P16 separate MCT");
    RouteRun {
        components,
        deinterleave_dispatches,
        mct_dispatches: execution.kernel_dispatches(),
    }
}

fn cuda_context() -> Option<CudaContext> {
    match CudaContext::system_default() {
        Ok(context) => Some(context),
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA initialization failed: {error}")
        }
        Err(error) if error.is_unavailable() => {
            eprintln!("skipping CUDA P16 benchmark: CUDA runtime is unavailable");
            None
        }
        Err(error) => panic!("CUDA P16 benchmark initialization failed: {error}"),
    }
}

fn assert_components_exact(actual: &[Vec<f32>], expected: &[Vec<f32>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context} component count");
    for (component, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{context} plane {component} length"
        );
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{context}: component {component} sample {index}"
            );
        }
    }
}

fn input_sha256(pixels: &[u8]) -> String {
    let mut hasher = Sha256::new();
    update_frame(&mut hasher, b"j2k-cuda-p16-input-v1");
    update_frame(&mut hasher, &usize_u64(DIMENSION).to_le_bytes());
    update_frame(&mut hasher, &usize_u64(DIMENSION).to_le_bytes());
    update_frame(&mut hasher, &[3, 8, 0]);
    update_frame(&mut hasher, pixels);
    digest_hex(hasher.finalize())
}

fn output_sha256(components: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    update_frame(&mut hasher, b"j2k-cuda-p16-output-f32le-v1");
    update_frame(&mut hasher, &usize_u64(components.len()).to_le_bytes());
    for component in components {
        update_frame(&mut hasher, &usize_u64(component.len()).to_le_bytes());
        for sample in component {
            hasher.update(sample.to_bits().to_le_bytes());
        }
    }
    digest_hex(hasher.finalize())
}

fn byte_slice_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    update_frame(&mut hasher, domain);
    update_frame(&mut hasher, bytes);
    digest_hex(hasher.finalize())
}

fn update_frame(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(usize_u64(bytes.len()).to_le_bytes());
    hasher.update(bytes);
}

fn usize_u64(value: usize) -> u64 {
    u64::try_from(value).expect("P16 framed hash length fits u64")
}

fn digest_hex(digest: impl AsRef<[u8]>) -> String {
    let mut output = String::new();
    output
        .try_reserve_exact(digest.as_ref().len() * 2)
        .expect("allocate P16 SHA-256 hex");
    for byte in digest.as_ref() {
        write!(&mut output, "{byte:02x}").expect("write P16 SHA-256 hex");
    }
    output
}

fn rgb8_fixture(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(width * height * 3)
        .expect("allocate P16 benchmark fixture");
    for index in 0..width * height {
        let index = u32::try_from(index).expect("P16 fixture index fits u32");
        pixels.push(index.wrapping_mul(13).wrapping_add(17).to_le_bytes()[0]);
        pixels.push(index.wrapping_mul(29).wrapping_add(71).to_le_bytes()[0]);
        pixels.push(index.wrapping_mul(47).wrapping_add(103).to_le_bytes()[0]);
    }
    pixels
}

fn p16_criterion() -> Criterion {
    Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(WARM_UP)
        .measurement_time(MEASUREMENT)
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = p16_criterion();
    targets = bench_cuda_input_fusion, bench_cuda_input_fusion_product
}
criterion_main!(benches);
