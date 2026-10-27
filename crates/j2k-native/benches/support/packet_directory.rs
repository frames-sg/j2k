// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repeated regions through the existing component-output API.

use criterion::{BatchSize, Criterion, Throughput};
use j2k_native::{
    encode, CpuDecodeParallelism, DecodeSettings, DecodedComponents, DecoderContext, EncodeOptions,
    EncodeProgressionOrder, Image,
};
#[cfg(feature = "parallel")]
use rayon::{prelude::*, ThreadPoolBuilder};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

type Region = (u32, u32, u32, u32);

fn source_bytes() -> Vec<u8> {
    if let Some(path) = std::env::var_os("J2K_PACKET_DIRECTORY_SOURCE") {
        eprintln!("packet directory: source {}", path.to_string_lossy());
        return std::fs::read(path).expect("read region source");
    }
    eprintln!("packet directory: generated 1536x1024 approximation");
    let (w, h) = (1536, 1024);
    encode(
        &super::patterned_rgb8(w, h),
        w,
        h,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            progression_order: EncodeProgressionOrder::Rpcl,
            num_layers: 7,
            precinct_exponents: vec![(7, 7), (7, 7), (7, 7), (7, 7), (8, 8), (8, 8)],
            ..EncodeOptions::default()
        },
    )
    .unwrap()
}

fn context<'a>(mode: CpuDecodeParallelism) -> DecoderContext<'a> {
    let mut ctx = DecoderContext::default();
    ctx.set_cpu_decode_parallelism(mode);
    ctx
}

// Each control Image is constructed outside the timer and used once, so it
// follows the existing packet walk without charging header setup to decoding.
fn control<'a, 'ctx>(
    image: &Image<'a>,
    roi: Region,
    ctx: &'ctx mut DecoderContext<'a>,
) -> DecodedComponents<'ctx> {
    image
        .decode_region_components_for_integer_output_with_context(roi, ctx)
        .unwrap()
}

pub(super) fn bench(c: &mut Criterion) {
    let bytes = source_bytes();
    let image = Image::new(&bytes, &DecodeSettings::default()).unwrap();
    let original_metadata = image.retained_allocation_bytes().unwrap();
    let (w, h) = (image.width(), image.height());
    let side = std::env::var("J2K_PACKET_DIRECTORY_REGION_SIDE")
        .map_or(256, |s| s.parse::<u32>().unwrap());
    let threads =
        std::env::var("J2K_PACKET_DIRECTORY_THREADS").map_or(4, |s| s.parse::<usize>().unwrap());
    assert!(side > 0 && side <= w && side <= h && threads > 0);
    let label = std::env::var_os("J2K_PACKET_DIRECTORY_SOURCE").map_or_else(
        || "generated".to_owned(),
        |p| {
            std::path::Path::new(&p)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        },
    );
    let mut group = c.benchmark_group(format!(
        "packet_directory/{label}_{w}x{h}_roi{side}_pool{threads}"
    ));
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("image_setup", |b| {
        b.iter(|| black_box(Image::new(&bytes, &DecodeSettings::default()).unwrap()))
    });
    let regions = [
        ("middle", ((w - side) / 2, (h - side) / 2, side, side)),
        ("edge", (w - side, h - side, side, side)),
    ];
    for (mode_name, mode) in [
        ("serial", CpuDecodeParallelism::Serial),
        ("auto", CpuDecodeParallelism::Auto),
    ] {
        #[cfg(feature = "parallel")]
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let install = |f: &mut (dyn FnMut() + Send)| {
            #[cfg(feature = "parallel")]
            pool.install(f);
            #[cfg(not(feature = "parallel"))]
            f();
        };
        for (region_name, roi) in regions {
            let case = format!("{mode_name}/{region_name}");
            if std::env::var("J2K_PACKET_DIRECTORY_CASE").is_ok_and(|selected| selected != case) {
                continue;
            }
            let mut a_ctx = context(mode);
            let mut b_ctx = context(mode);
            for call in 0..2 {
                let control_image = Image::new(&bytes, &DecodeSettings::default()).unwrap();
                let start = Instant::now();
                install(&mut || {
                    let expected = control(&control_image, roi, &mut a_ctx);
                    let actual = image
                        .decode_region_components_for_integer_output_with_context(roi, &mut b_ctx)
                        .unwrap();
                    assert_eq!(actual.dimensions(), expected.dimensions());
                    assert_eq!(actual.planes().len(), expected.planes().len());
                    for (a, b) in actual.planes().iter().zip(expected.planes()) {
                        assert!(a
                            .samples()
                            .iter()
                            .map(|v| v.to_bits())
                            .eq(b.samples().iter().map(|v| v.to_bits())));
                    }
                });
                eprintln!(
                    "region {mode_name}/{region_name} parity call={call} pair_ms={:.3}",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            let mut aa = Vec::new();
            let mut a = Vec::new();
            let mut b = Vec::new();
            for run in 0..33 {
                for (route, samples) in [(0, &mut aa), (0, &mut a), (1, &mut b)] {
                    let control_image = (route == 0)
                        .then(|| Image::new(&bytes, &DecodeSettings::default()).unwrap());
                    let start = Instant::now();
                    install(&mut || {
                        if route == 0 {
                            black_box(control(control_image.as_ref().unwrap(), roi, &mut a_ctx));
                        } else {
                            black_box(
                                image
                                    .decode_region_components_for_integer_output_with_context(
                                        roi, &mut b_ctx,
                                    )
                                    .unwrap(),
                            );
                        }
                    });
                    if run > 0 {
                        samples.push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
            for (name, samples) in [("A1", &mut aa), ("A2", &mut a), ("B", &mut b)] {
                samples.sort_by(f64::total_cmp);
                eprintln!(
                    "region {mode_name}/{region_name} {name} p50_ms={:.3} p95_ms={:.3}",
                    samples[16], samples[30]
                );
            }
            eprintln!(
                "region {mode_name}/{region_name} cache_bytes={} retained_context_bytes={}",
                image.retained_allocation_bytes().unwrap() - original_metadata,
                b_ctx.workspace_stats().retained_component_bytes()
                    + b_ctx.workspace_stats().retained_scratch_bytes()
            );
            #[cfg(feature = "parallel")]
            {
                let mut control_contexts: Vec<_> = (0..8).map(|_| context(mode)).collect();
                let mut automatic_contexts: Vec<_> = (0..8).map(|_| context(mode)).collect();
                let mut batches = [Vec::new(), Vec::new(), Vec::new()];
                for round in 0..17 {
                    let order = if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
                    for route in order {
                        let control_images: Vec<_> = if route == 2 {
                            Vec::new()
                        } else {
                            (0..8)
                                .map(|_| Image::new(&bytes, &DecodeSettings::default()).unwrap())
                                .collect()
                        };
                        let start = Instant::now();
                        pool.install(|| {
                            if route == 2 {
                                automatic_contexts.par_iter_mut().for_each(|ctx| {
                                    black_box(image.decode_region_components_for_integer_output_with_context(roi, ctx).unwrap());
                                });
                            } else {
                                control_contexts.par_iter_mut().zip(&control_images).for_each(|(ctx, control_image)| {
                                    black_box(control(control_image, roi, ctx));
                                });
                            }
                        });
                        if round > 0 {
                            batches[route].push(start.elapsed().as_secs_f64());
                        }
                    }
                }
                for (name, samples) in ["A1", "A2", "B"].into_iter().zip(&mut batches) {
                    samples.sort_by(f64::total_cmp);
                    eprintln!(
                        "outer8 {case} {name} p50_ms={:.3} requests_per_s={:.3}",
                        samples[8] * 1000.0,
                        8.0 / samples[8]
                    );
                }
            }
            for automatic in [false, true] {
                let route = if automatic { "automatic" } else { "control" };
                let mut reused = context(mode);
                group.throughput(Throughput::Elements(1));
                for fresh in [false, true] {
                    group.bench_function(
                        format!("{case}/{route}_{}", if fresh { "fresh" } else { "reuse" }),
                        |b| b.iter_batched_ref(
                            || (!automatic).then(|| Image::new(&bytes, &DecodeSettings::default()).unwrap()),
                            |control_image| install(&mut || {
                                let mut new_ctx = context(mode);
                                let ctx = if fresh { &mut new_ctx } else { &mut reused };
                                if automatic {
                                    black_box(image.decode_region_components_for_integer_output_with_context(roi, ctx).unwrap());
                                } else {
                                    black_box(control(control_image.as_ref().unwrap(), roi, ctx));
                                }
                            }),
                            BatchSize::PerIteration,
                        ),
                    );
                }
                #[cfg(feature = "parallel")]
                {
                    let mut contexts: Vec<_> = (0..8).map(|_| context(mode)).collect();
                    group.throughput(Throughput::Elements(8));
                    group.bench_function(format!("{case}/{route}_outer8"), |b| {
                        b.iter_batched_ref(
                            || if automatic { Vec::new() } else {
                                (0..8).map(|_| Image::new(&bytes, &DecodeSettings::default()).unwrap()).collect::<Vec<_>>()
                            },
                            |control_images| pool.install(|| {
                                contexts.par_iter_mut().enumerate().for_each(|(index, ctx)| {
                                    if automatic {
                                        black_box(image.decode_region_components_for_integer_output_with_context(roi, ctx).unwrap());
                                    } else {
                                        black_box(control(&control_images[index], roi, ctx));
                                    }
                                });
                            }),
                            BatchSize::PerIteration,
                        );
                    });
                }
            }
        }
    }
    group.finish();
}
