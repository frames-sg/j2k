// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, EncodedImage,
    PreparationDepth, Rect,
};
use j2k_metal_support::{MetalImageDestination, MetalImageLayout};
use j2k_native::{encode_component_planes_53, EncodeComponentPlane, EncodeOptions};

use super::MetalBatchDecoder;
use crate::Error;

/// RGB codestream without MCT whose second and third components use `sampling`.
pub(super) fn sampled_rgb(
    dimensions: (u32, u32),
    sampling: (u8, u8),
    precision: u8,
    ht: bool,
    seed: u32,
) -> Arc<[u8]> {
    sampled_rgb_with_factors(
        dimensions,
        [(1, 1), sampling, sampling],
        precision,
        ht,
        seed,
    )
}

/// RGB codestream without MCT whose components use `factors`.
fn sampled_rgb_with_factors(
    dimensions: (u32, u32),
    factors: [(u8, u8); 3],
    precision: u8,
    ht: bool,
    seed: u32,
) -> Arc<[u8]> {
    let (width, height) = dimensions;
    let max = (1_u32 << precision) - 1;
    let data: Vec<Vec<u8>> = factors
        .iter()
        .enumerate()
        .map(|(component, &(sx, sy))| {
            let samples = width.div_ceil(u32::from(sx)) * height.div_ceil(u32::from(sy));
            (0..samples)
                .flat_map(|index| {
                    let component = u32::try_from(component).expect("component index");
                    let sample = (index * 37 + component * 101 + seed * 13) % (max + 1);
                    if precision <= 8 {
                        vec![u8::try_from(sample).expect("8-bit sample")]
                    } else {
                        u16::try_from(sample)
                            .expect("16-bit sample")
                            .to_le_bytes()
                            .to_vec()
                    }
                })
                .collect()
        })
        .collect();
    let planes: Vec<_> = data
        .iter()
        .zip(factors)
        .map(|(data, (x_rsiz, y_rsiz))| EncodeComponentPlane {
            data,
            x_rsiz,
            y_rsiz,
        })
        .collect();
    encode_component_planes_53(
        &planes,
        width,
        height,
        precision,
        false,
        &EncodeOptions {
            use_ht_block_coding: ht,
            use_mct: false,
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode sampled RGB fixture")
    .into()
}

fn options(layout: BatchLayout) -> BatchDecodeOptions {
    let mut options = BatchDecodeOptions::default();
    options.layout = layout;
    options
}

fn cpu_bytes(samples: &CpuBatchSamples) -> Vec<u8> {
    match samples {
        CpuBatchSamples::U8(samples) => samples.clone(),
        CpuBatchSamples::U16(samples) => samples.iter().flat_map(|s| s.to_le_bytes()).collect(),
        _ => panic!("sampled fixtures are unsigned"),
    }
}

struct ResidentRouteReport {
    command_buffers: usize,
    cpu_tier1_images: usize,
}

/// Decodes `inputs` through the codec-owned resident path, cooperatively
/// when `cpu_workers` is set, and returns its route observations.
fn assert_resident_matches_cpu(
    inputs: &[Arc<[u8]>],
    case: &str,
    cpu_workers: Option<NonZeroUsize>,
) -> ResidentRouteReport {
    assert_resident_matches_cpu_with_options(inputs, case, options(BatchLayout::Nhwc), cpu_workers)
}

fn assert_resident_matches_cpu_with_options(
    inputs: &[Arc<[u8]>],
    case: &str,
    options: BatchDecodeOptions,
    cooperative_workers: Option<NonZeroUsize>,
) -> ResidentRouteReport {
    let encoded = inputs
        .iter()
        .cloned()
        .map(EncodedImage::full)
        .collect::<Vec<_>>();
    let expected = CpuBatchDecoder::new(options)
        .decode(encoded.clone())
        .expect("CPU sampled oracle");
    assert!(
        expected.errors().is_empty(),
        "{case}: {:?}",
        expected.errors()
    );

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder.prepare(encoded).expect("prepare sampled batch");
    assert!(
        prepared.errors().is_empty(),
        "{case}: {:?}",
        prepared.errors()
    );
    crate::engine::reset_metal_command_buffers_for_test();
    let result = match cooperative_workers {
        Some(workers) => decoder.decode_prepared_cooperative(&prepared, workers),
        None => decoder.decode_prepared(&prepared),
    }
    .expect("decode sampled batch");
    let command_buffers = crate::engine::metal_command_buffers_for_test();
    let (groups, errors, group_errors) = result.into_parts();
    assert!(errors.is_empty(), "{case}: {errors:?}");
    assert!(group_errors.is_empty(), "{case}: {group_errors:?}");
    assert_eq!(groups.len(), expected.groups().len(), "{case}");
    for group in &groups {
        let expected_group = expected
            .groups()
            .iter()
            .find(|expected| expected.source_indices() == group.source_indices())
            .expect("matching CPU group");
        let expected = cpu_bytes(expected_group.samples());
        let image_bytes = expected.len() / group.surfaces().len();
        for (image, surface) in group.surfaces().iter().enumerate() {
            let actual = surface.as_bytes().expect("resident surface bytes");
            assert!(
                actual.as_ref() == &expected[image * image_bytes..][..image_bytes],
                "{case}: image {image} of sources {:?} differs from CPU",
                group.source_indices()
            );
        }
    }
    let cpu_tier1_images = groups
        .iter()
        .map(|group| group.dispatch_report().cpu_tier1_images)
        .sum();
    ResidentRouteReport {
        command_buffers,
        cpu_tier1_images,
    }
}

/// Decodes one prepared group into caller destinations with `layout`, through
/// the submitting and the synchronous entry points, and returns how many images
/// the submitted group decoded with CPU Tier-1.
fn assert_external_matches_cpu(inputs: &[Arc<[u8]>], layout: BatchLayout, case: &str) -> usize {
    let options = options(layout);
    let encoded = inputs
        .iter()
        .cloned()
        .map(EncodedImage::full)
        .collect::<Vec<_>>();
    let expected = CpuBatchDecoder::new(options)
        .decode(encoded.clone())
        .expect("CPU sampled oracle");
    assert_eq!(expected.groups().len(), 1, "{case}");
    let expected = cpu_bytes(expected.groups()[0].samples());

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder.prepare(encoded).expect("prepare sampled group");
    assert!(
        prepared.errors().is_empty(),
        "{case}: {:?}",
        prepared.errors()
    );
    let group = &prepared.groups()[0];
    let fmt = group
        .info()
        .native_pixel_format()
        .expect("native RGB format");
    let (width, height) = group.info().dimensions;
    let row_bytes = width as usize * fmt.bytes_per_pixel();
    let image_bytes = row_bytes * height as usize;
    let destination = || {
        let output = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
            decoder.backend_session().device(),
            image_bytes * inputs.len(),
        )
        .expect("sampled destination allocation");
        let destination_layout = MetalImageLayout::new_batch(
            0,
            (width, height),
            row_bytes,
            fmt,
            inputs.len(),
            image_bytes,
        )
        .expect("sampled destination layout");
        // SAFETY: the fresh allocation has one logical writer until completion.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(output.clone(), destination_layout)
                .expect("sampled destination")
        };
        (output, destination)
    };
    let read = |output: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    >| {
        // SAFETY: completion released the destination's exclusive write guard.
        unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(output, 0, image_bytes * inputs.len())
                .expect("sampled destination bytes")
        }
    };

    let (submitted_output, submitted_destination) = destination();
    let (sync_output, sync_destination) = destination();
    let submitted = decoder
        .submit_prepared_group_into(group, submitted_destination)
        .unwrap_or_else(|error| panic!("{case}: submit sampled group: {error}"));
    let cpu_tier1_images = submitted.cpu_tier1_images_for_test();
    submitted
        .wait()
        .unwrap_or_else(|error| panic!("{case}: complete sampled group: {error}"));
    assert!(
        read(&submitted_output) == expected,
        "{case}: submitted external destination differs from CPU"
    );
    decoder
        .decode_prepared_group_into(group, &sync_destination)
        .unwrap_or_else(|error| panic!("{case}: decode sampled group: {error}"));
    drop(sync_destination);
    assert!(
        read(&sync_output) == expected,
        "{case}: synchronous external destination differs from CPU"
    );
    cpu_tier1_images
}

#[test]
fn sampled_groups_decode_on_metal_in_one_command_buffer() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    for ht in [false, true] {
        for factors in [
            [(1, 1), (2, 1), (2, 1)],
            [(1, 1), (2, 2), (2, 2)],
            [(1, 1), (1, 2), (1, 2)],
            [(1, 1), (3, 2), (3, 2)],
            [(1, 2), (2, 1), (2, 1)],
        ] {
            for precision in [8, 12] {
                let [a, b, c] = [1, 2, 3]
                    .map(|seed| sampled_rgb_with_factors((33, 31), factors, precision, ht, seed));
                for (shape, inputs) in [
                    ("single", vec![a.clone()]),
                    ("distinct", vec![a.clone(), b.clone(), c.clone()]),
                    ("repeated", vec![a.clone(), a.clone()]),
                    ("partially repeated", vec![a.clone(), b.clone(), a.clone()]),
                ] {
                    let case = format!("ht={ht} factors={factors:?} precision={precision} {shape}");
                    let report = assert_resident_matches_cpu(&inputs, &case, None);
                    assert_eq!(report.command_buffers, 1, "{case}");
                    assert_eq!(report.cpu_tier1_images, 0, "{case}");
                    for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
                        let cpu_tier1_images = assert_external_matches_cpu(
                            &inputs,
                            layout,
                            &format!("{case} {layout:?}"),
                        );
                        assert_eq!(cpu_tier1_images, 0, "{case} {layout:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn sampled_and_unit_sampled_groups_decode_together() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let unit = sampled_rgb((24, 16), (1, 1), 8, false, 4);
    let sampled = sampled_rgb((24, 16), (2, 1), 8, false, 5);
    let other = sampled_rgb((24, 16), (2, 1), 8, false, 6);
    assert_resident_matches_cpu(
        &[unit.clone(), sampled, unit, other],
        "unit and sampled tiles in one batch",
        None,
    );
}

#[test]
fn sampled_classic_batch_uses_automatic_cpu_tier1_and_matches_cpu() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    // Eight distinct RGB8 classic inputs cross the automatic route threshold.
    let inputs = (0..8)
        .map(|seed| sampled_rgb((130, 128), (2, 1), 8, false, seed))
        .collect::<Vec<_>>();
    let report = assert_resident_matches_cpu(&inputs, "sampled CPU Tier-1 batch", None);
    assert_eq!(report.command_buffers, 1);
    assert_eq!(report.cpu_tier1_images, inputs.len());
    // The single-group external entry points take the same route.
    for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
        assert_eq!(
            assert_external_matches_cpu(
                &inputs,
                layout,
                &format!("sampled CPU Tier-1 external batch {layout:?}"),
            ),
            inputs.len(),
            "{layout:?}"
        );
    }
}

#[test]
fn sampled_automatic_cpu_tier1_honors_the_batch_worker_budget() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let inputs = (0..8)
        .map(|seed| sampled_rgb((130, 128), (2, 1), 8, false, seed))
        .collect::<Vec<_>>();

    let mut single_worker = options(BatchLayout::Nhwc);
    single_worker.workers = NonZeroUsize::new(1);
    let report = assert_resident_matches_cpu_with_options(
        &inputs,
        "single-worker sampled batch",
        single_worker,
        None,
    );
    assert_eq!(report.command_buffers, 1);
    assert_eq!(report.cpu_tier1_images, 0);

    let mut two_workers = options(BatchLayout::Nhwc);
    two_workers.workers = NonZeroUsize::new(2);
    let report = assert_resident_matches_cpu_with_options(
        &inputs,
        "two-worker sampled batch",
        two_workers,
        None,
    );
    assert_eq!(report.command_buffers, 1);
    assert_eq!(report.cpu_tier1_images, inputs.len());
}

#[test]
fn sampled_cooperative_fallback_disables_automatic_cpu_tier1() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let inputs = (0..8)
        .map(|seed| sampled_rgb((130, 128), (2, 1), 8, false, seed))
        .collect::<Vec<_>>();

    for workers in [1, 2] {
        let case = format!("sampled cooperative fallback with {workers} worker(s)");
        let report = assert_resident_matches_cpu(&inputs, &case, NonZeroUsize::new(workers));
        assert_eq!(report.command_buffers, 1, "{case}");
        assert_eq!(report.cpu_tier1_images, 0, "{case}");
    }
}

#[test]
fn sampled_decode_inside_rayon_keeps_cpu_tier1_off_the_busy_pool() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let inputs = (0..8)
        .map(|seed| sampled_rgb((130, 128), (2, 1), 8, false, seed))
        .collect::<Vec<_>>();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("two-thread pool");
    let report =
        pool.install(|| assert_resident_matches_cpu(&inputs, "nested sampled batch", None));
    assert_eq!(report.command_buffers, 1);
    assert_eq!(report.cpu_tier1_images, 0);
}

#[test]
fn sampled_region_requests_remain_a_metal_capability_rejection() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let encoded = sampled_rgb((33, 31), (2, 1), 8, false, 7);
    let mut decoder = MetalBatchDecoder::system_default_with_options(options(BatchLayout::Nhwc))
        .expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![EncodedImage::new(
            encoded,
            DecodeRequest::Region {
                roi: Rect {
                    x: 2,
                    y: 3,
                    w: 16,
                    h: 9,
                },
            },
        )])
        .expect("prepare sampled region");
    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    assert_eq!(
        prepared.groups()[0].images()[0].preparation_depth(),
        PreparationDepth::MetadataOnly
    );
    let (groups, _, group_errors) = decoder
        .decode_prepared(&prepared)
        .expect("decode sampled region")
        .into_parts();
    assert!(groups.is_empty());
    assert_eq!(group_errors.len(), 1);
    let (indices, error) = group_errors.into_iter().next().unwrap().into_parts();
    assert_eq!(indices, [0]);
    assert!(
        matches!(error, Error::UnsupportedMetalRequest { .. }),
        "callers fall back on capability rejections: {error}"
    );
}
