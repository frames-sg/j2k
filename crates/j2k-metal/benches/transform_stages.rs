// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::time::Duration;

#[cfg(target_os = "macos")]
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
#[cfg(target_os = "macos")]
use j2k::{J2kEncodeStageAccelerator, J2kForwardDwt97Job, J2kForwardDwt97Output};
#[cfg(target_os = "macos")]
use j2k_metal::MetalEncodeStageAccelerator;
#[cfg(target_os = "macos")]
use j2k_native::EncodeOptions;

#[cfg(target_os = "macos")]
const STAGE_WIDTH: u32 = 1024;
#[cfg(target_os = "macos")]
const STAGE_HEIGHT: u32 = 768;
#[cfg(target_os = "macos")]
const ENCODE_DIMENSION: u32 = 512;

#[cfg(not(target_os = "macos"))]
fn main() {
    assert!(
        std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_none(),
        "J2K Metal transform benchmark requires macOS"
    );
    eprintln!("J2K Metal transform benchmark skipped outside macOS");
}

#[cfg(target_os = "macos")]
fn stage_samples() -> Vec<f32> {
    (0..STAGE_WIDTH * STAGE_HEIGHT)
        .map(|index| {
            let value = index.wrapping_mul(43).wrapping_add(index / 7 + 91) & 0xff;
            f32::from(u8::try_from(value).expect("masked stage sample fits u8")) * 0.587 - 74.472_99
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn encode_pixels() -> Vec<u8> {
    (0..ENCODE_DIMENSION * ENCODE_DIMENSION)
        .map(|index| {
            u8::try_from(index.wrapping_mul(29).wrapping_add(index / 5 + 17) & 0xff)
                .expect("masked encode sample fits u8")
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn encode_rgb_pixels() -> Vec<u8> {
    j2k_test_support::patterned_rgb8(ENCODE_DIMENSION, ENCODE_DIMENSION)
}

#[cfg(target_os = "macos")]
fn stage_signature(output: &J2kForwardDwt97Output) -> u64 {
    stage_coefficients(output).fold(0xcbf2_9ce4_8422_2325, |hash, sample| {
        (hash ^ u64::from(sample.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(target_os = "macos")]
fn stage_coefficients(output: &J2kForwardDwt97Output) -> impl Iterator<Item = &f32> {
    output.ll.iter().chain(
        output
            .levels
            .iter()
            .flat_map(|level| level.hl.iter().chain(&level.lh).chain(&level.hh)),
    )
}

#[cfg(target_os = "macos")]
fn stage_sha256(samples: &[f32]) -> String {
    let mut accelerator = MetalEncodeStageAccelerator::default();
    let output = accelerator
        .encode_forward_dwt97(J2kForwardDwt97Job {
            samples,
            width: STAGE_WIDTH,
            height: STAGE_HEIGHT,
            num_levels: 3,
        })
        .expect("Metal FDWT97 SHA probe succeeds")
        .expect("Metal FDWT97 SHA probe dispatches");
    let bytes = stage_coefficients(&output)
        .flat_map(|sample| sample.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    j2k_test_support::auto_routing_sha256(&bytes)
}

#[cfg(target_os = "macos")]
fn run_stage(samples: &[f32], accelerator: &mut MetalEncodeStageAccelerator) -> u64 {
    let output = accelerator
        .encode_forward_dwt97(J2kForwardDwt97Job {
            samples,
            width: STAGE_WIDTH,
            height: STAGE_HEIGHT,
            num_levels: 3,
        })
        .expect("Metal FDWT97 stage succeeds")
        .expect("Metal FDWT97 stage dispatches");
    stage_signature(&output)
}

#[cfg(target_os = "macos")]
fn run_encode(pixels: &[u8], accelerator: &mut MetalEncodeStageAccelerator) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: false,
        num_decomposition_levels: 3,
        guard_bits: 2,
        use_ht_block_coding: true,
        ..EncodeOptions::default()
    };
    j2k_native::encode_with_accelerator(
        pixels,
        ENCODE_DIMENSION,
        ENCODE_DIMENSION,
        1,
        8,
        false,
        &options,
        accelerator,
    )
    .expect("Metal-assisted irreversible encode succeeds")
}

#[cfg(target_os = "macos")]
fn run_rgb_encode(
    pixels: &[u8],
    reversible: bool,
    use_ht_block_coding: bool,
    accelerator: &mut MetalEncodeStageAccelerator,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: 3,
        guard_bits: if reversible { 1 } else { 2 },
        use_ht_block_coding,
        use_mct: true,
        ..EncodeOptions::default()
    };
    j2k_native::encode_with_accelerator(
        pixels,
        ENCODE_DIMENSION,
        ENCODE_DIMENSION,
        3,
        8,
        false,
        &options,
        accelerator,
    )
    .expect("Metal-assisted RGB encode succeeds")
}

#[cfg(target_os = "macos")]
fn bench_transform_stages(criterion: &mut Criterion) {
    let samples = stage_samples();
    let pixels = encode_pixels();
    let rgb_pixels = encode_rgb_pixels();
    let mut probe = MetalEncodeStageAccelerator::default();
    let stage_hash = run_stage(&samples, &mut probe);
    let codestream = run_encode(&pixels, &mut probe);
    let reversible_rgb = run_rgb_encode(&rgb_pixels, true, true, &mut probe);
    let irreversible_rgb = run_rgb_encode(&rgb_pixels, false, true, &mut probe);
    let reversible_rgb_classic = run_rgb_encode(&rgb_pixels, true, false, &mut probe);
    let irreversible_rgb_classic = run_rgb_encode(&rgb_pixels, false, false, &mut probe);
    eprintln!(
        "j2k_metal_fdwt97_probe stage_fnv1a64={stage_hash:016x} stage_sha256={} output_sha256={} output_bytes={}",
        stage_sha256(&samples),
        j2k_test_support::auto_routing_sha256(&codestream),
        codestream.len()
    );
    eprintln!(
        "j2k_metal_input_mct_probe ht_rct_sha256={} ht_ict_sha256={} classic_rct_sha256={} classic_ict_sha256={}",
        j2k_test_support::auto_routing_sha256(&reversible_rgb),
        j2k_test_support::auto_routing_sha256(&irreversible_rgb),
        j2k_test_support::auto_routing_sha256(&reversible_rgb_classic),
        j2k_test_support::auto_routing_sha256(&irreversible_rgb_classic),
    );

    let mut group = criterion.benchmark_group("metal_fdwt97");
    group.throughput(Throughput::Elements(
        u64::from(STAGE_WIDTH) * u64::from(STAGE_HEIGHT),
    ));
    let mut stage_accelerator = MetalEncodeStageAccelerator::default();
    group.bench_function("stage_1024x768_l3", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(run_stage(
                std::hint::black_box(&samples),
                &mut stage_accelerator,
            ))
        });
    });
    let mut encode_accelerator = MetalEncodeStageAccelerator::default();
    group.bench_function("encode_gray8_512_l3", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(run_encode(
                std::hint::black_box(&pixels),
                &mut encode_accelerator,
            ))
        });
    });
    let mut rct_accelerator = MetalEncodeStageAccelerator::default();
    group.bench_function("encode_rgb8_512_rct_l3", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(run_rgb_encode(
                std::hint::black_box(&rgb_pixels),
                true,
                true,
                &mut rct_accelerator,
            ))
        });
    });
    let mut ict_accelerator = MetalEncodeStageAccelerator::default();
    group.bench_function("encode_rgb8_512_ict_l3", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(run_rgb_encode(
                std::hint::black_box(&rgb_pixels),
                false,
                true,
                &mut ict_accelerator,
            ))
        });
    });
    let mut classic_reversible_accelerator = MetalEncodeStageAccelerator::default();
    group.bench_function("encode_rgb8_512_classic_rct_l3", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(run_rgb_encode(
                std::hint::black_box(&rgb_pixels),
                true,
                false,
                &mut classic_reversible_accelerator,
            ))
        });
    });
    let mut classic_irreversible_accelerator = MetalEncodeStageAccelerator::default();
    group.bench_function("encode_rgb8_512_classic_ict_l3", |bencher| {
        bencher.iter(|| {
            std::hint::black_box(run_rgb_encode(
                std::hint::black_box(&rgb_pixels),
                false,
                false,
                &mut classic_irreversible_accelerator,
            ))
        });
    });
    group.finish();
}

#[cfg(target_os = "macos")]
fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(5))
}

#[cfg(target_os = "macos")]
criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_transform_stages
}
#[cfg(target_os = "macos")]
criterion_main!(benches);
