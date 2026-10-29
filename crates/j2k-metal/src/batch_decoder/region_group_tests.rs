// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, Downscale,
    EncodedImage, Rect,
};
use j2k_core::PixelFormat;
use j2k_native::{encode, encode_htj2k, EncodeOptions};

use super::MetalBatchDecoder;

fn options() -> BatchDecodeOptions {
    let mut options = BatchDecodeOptions::default();
    options.layout = BatchLayout::Nhwc;
    options
}

fn request(roi: Rect, scale: Downscale) -> DecodeRequest {
    if scale == Downscale::None {
        DecodeRequest::Region { roi }
    } else {
        DecodeRequest::RegionReduced { roi, scale }
    }
}

/// Textured image with 16 x 16 code-blocks, so a small region keeps only a
/// few blocks of each sub-band.
fn textured(components: u16, ht: bool, reversible: bool) -> Arc<[u8]> {
    let (width, height) = (300_u32, 260_u32);
    let pixels: Vec<u8> = (0..width * height * u32::from(components))
        .map(|index| {
            let pixel = index / u32::from(components);
            let (x, y) = (pixel % width, pixel / width);
            u8::try_from((x * 7 + y * 13 + (x * y) / 5 + index % 3 * 41) & 0xff).expect("masked")
        })
        .collect();
    let options = EncodeOptions {
        reversible,
        use_mct: components == 3,
        num_decomposition_levels: 3,
        code_block_width_exp: 4,
        code_block_height_exp: 4,
        ..EncodeOptions::default()
    };
    let encoder = if ht { encode_htj2k } else { encode };
    encoder(&pixels, width, height, components, 8, false, &options)
        .expect("encode textured fixture")
        .into()
}

fn cpu_images(inputs: &[EncodedImage]) -> Vec<Vec<u8>> {
    let result = CpuBatchDecoder::new(options())
        .decode(inputs.to_vec())
        .expect("CPU region oracle");
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    let mut images = vec![Vec::new(); inputs.len()];
    for group in result.groups() {
        let CpuBatchSamples::U8(samples) = group.samples() else {
            panic!("8-bit fixtures decode to U8");
        };
        let per_image = samples.len() / group.source_indices().len();
        for (position, &index) in group.source_indices().iter().enumerate() {
            images[index] = samples[position * per_image..][..per_image].to_vec();
        }
    }
    images
}

fn metal_images(decoder: &mut MetalBatchDecoder, inputs: &[EncodedImage]) -> Vec<Vec<u8>> {
    let prepared = decoder
        .prepare(inputs.to_vec())
        .expect("prepare region batch");
    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    let (groups, errors, group_errors) = decoder
        .decode_prepared(&prepared)
        .expect("decode region batch")
        .into_parts();
    assert!(errors.is_empty(), "{errors:?}");
    assert!(group_errors.is_empty(), "{group_errors:?}");
    let mut images = vec![Vec::new(); inputs.len()];
    for group in groups {
        let (_, indices, _, _, surfaces) = group.into_parts();
        for (index, surface) in indices.into_iter().zip(surfaces) {
            images[index] = surface.as_bytes().expect("region surface").into_owned();
        }
    }
    images
}

#[test]
fn large_region_batches_stay_within_the_metal_buffer_cap() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    // Whole-level coefficient buffers at reductions 0 and 1 exceed the 512 MiB
    // cap on an 18000 x 12000 RGB image; each tile needs only its window.
    for ht in [false, true] {
        let bytes: Arc<[u8]> = crate::engine::tests::roi::empty_large_rgb_codestream(ht).into();
        let mut decoder =
            MetalBatchDecoder::system_default_with_options(options()).expect("Metal decoder");
        for scale in [Downscale::None, Downscale::Half, Downscale::Quarter] {
            let side = 1024 * scale.denominator();
            let inputs: Vec<_> = [(0, 0), (8192, 6144), (18000 - side, 12000 - side)]
                .into_iter()
                .map(|(x, y)| {
                    let roi = Rect {
                        x,
                        y,
                        w: side,
                        h: side,
                    };
                    EncodedImage::new(bytes.clone(), request(roi, scale))
                })
                .collect();
            let prepared = decoder
                .prepare(inputs.clone())
                .expect("prepare large region batch");
            for group in prepared.groups() {
                for plan in decoder
                    .prepared_color_group_plans(group, PixelFormat::Rgb8)
                    .expect("large region plans")
                {
                    let (_, coefficients) = plan.geometry_for_test();
                    assert!(
                        coefficients.iter().all(|&count| count < 4 * 1024 * 1024),
                        "ht={ht} {scale:?}: a 1024 px tile retained {coefficients:?} coefficients"
                    );
                }
            }
            for image in metal_images(&mut decoder, &inputs) {
                assert_eq!(image, vec![128; 1024 * 1024 * 3], "ht={ht} {scale:?}");
            }
        }
    }
}

#[test]
fn pruned_region_batches_match_the_cpu_exactly() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let regions = [
        (
            Rect {
                x: 0,
                y: 0,
                w: 40,
                h: 36,
            },
            Downscale::None,
        ),
        (
            Rect {
                x: 131,
                y: 97,
                w: 23,
                h: 41,
            },
            Downscale::None,
        ),
        (
            Rect {
                x: 251,
                y: 213,
                w: 49,
                h: 47,
            },
            Downscale::None,
        ),
        (
            Rect {
                x: 100,
                y: 64,
                w: 96,
                h: 80,
            },
            Downscale::Half,
        ),
        (
            Rect {
                x: 37,
                y: 150,
                w: 120,
                h: 110,
            },
            Downscale::Quarter,
        ),
    ];
    for components in [1, 3] {
        for ht in [false, true] {
            for reversible in [true, false] {
                let bytes = textured(components, ht, reversible);
                let inputs: Vec<_> = regions
                    .iter()
                    .map(|&(roi, scale)| EncodedImage::new(bytes.clone(), request(roi, scale)))
                    .collect();
                let mut decoder =
                    MetalBatchDecoder::system_default_with_options(options()).expect("decoder");
                let expected = cpu_images(&inputs);
                let actual = metal_images(&mut decoder, &inputs);
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        actual == expected,
                        "components={components} ht={ht} reversible={reversible} region {index} differs from CPU"
                    );
                }
            }
        }
    }
}

#[test]
fn cropping_a_referenced_plan_drops_blocks_and_keeps_pixels() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    // Batch region plans already reference only their blocks, so pruning them
    // drops none. Cropping a whole-image referenced plan does drop blocks,
    // which exercises the HT path that keeps payload ranges in the input.
    let roi = Rect {
        x: 131,
        y: 97,
        w: 23,
        h: 41,
    };
    for ht in [false, true] {
        let bytes = textured(3, ht, true);
        let decoder = MetalBatchDecoder::system_default_with_options(options()).expect("decoder");
        let prepared = decoder
            .prepare(vec![EncodedImage::full(bytes.clone())])
            .expect("prepare whole image");
        let image = &prepared.groups()[0].images()[0];
        let surface = crate::engine::with_runtime_for_session(decoder.backend_session(), |_| {
            let mut plan = match image.htj2k_plan() {
                Some(plan) => {
                    crate::engine::prepare_referenced_htj2k_color_plan(plan.geometry(), &bytes)
                }
                None => crate::engine::prepare_referenced_classic_color_plan(
                    image.classic_plan().expect("classic plan").geometry(),
                    &bytes,
                ),
            }?;
            let (whole_jobs, _) = plan.jobs_for_test();
            crate::engine::crop_prepared_direct_color_plan_to_output_region(&mut plan, roi)?;
            let (cropped_jobs, _) = plan.jobs_for_test();
            assert!(
                cropped_jobs < whole_jobs,
                "ht={ht}: crop kept {cropped_jobs} of {whole_jobs} jobs"
            );
            crate::engine::execute_prepared_direct_color_plan(Arc::new(plan), PixelFormat::Rgb8)
        })
        .unwrap_or_else(|error| panic!("ht={ht}: cropped referenced decode: {error}"));
        let expected = cpu_images(&[EncodedImage::new(bytes, DecodeRequest::Region { roi })]);
        assert!(
            surface.as_bytes().expect("cropped surface").as_ref() == expected[0].as_slice(),
            "ht={ht}: cropped referenced plan differs from CPU"
        );
    }
}

// Local timing harness: one round of four 1024 px tiles per level of a large
// codestream, through the batch decoder and through serial CPU region
// decodes run in parallel. The generated tests above cover the assertions.
#[cfg(not(debug_assertions))]
#[test]
#[ignore = "release large-region batch characterization; requires J2K_LARGE_REGION_INPUT"]
fn local_large_region_batch_characterization() {
    use j2k::{CpuDecodeParallelism, J2kDecoder, J2kScratchPool};

    assert!(j2k_test_support::metal_runtime_gate(module_path!()));
    let path = std::env::var_os("J2K_LARGE_REGION_INPUT").expect("trusted large codestream");
    let bytes: Arc<[u8]> = std::fs::read(path).expect("large codestream").into();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    for (reduction, scale) in [Downscale::None, Downscale::Half, Downscale::Quarter]
        .into_iter()
        .enumerate()
    {
        let side = 1024 * scale.denominator();
        let rois: Vec<_> = (0..4)
            .map(|index| Rect {
                x: 2048 + (index % 2) * side,
                y: 1024 + (index / 2) * side,
                w: side,
                h: side,
            })
            .collect();
        let inputs: Vec<_> = rois
            .iter()
            .map(|&roi| EncodedImage::new(bytes.clone(), request(roi, scale)))
            .collect();
        for sample in 0..7 {
            let mut decoder =
                MetalBatchDecoder::with_backend_session_and_options(session.clone(), options());
            let started = std::time::Instant::now();
            let prepared = decoder.prepare(inputs.clone()).expect("prepare");
            let (_, errors, group_errors) = decoder
                .decode_prepared(&prepared)
                .expect("decode")
                .into_parts();
            let metal_ms = started.elapsed().as_secs_f64() * 1000.0;
            assert!(
                errors.is_empty() && group_errors.is_empty(),
                "{group_errors:?}"
            );
            let started = std::time::Instant::now();
            std::thread::scope(|scope| {
                for &roi in &rois {
                    let bytes = &bytes;
                    scope.spawn(move || {
                        let mut decoder = J2kDecoder::new(bytes).expect("CPU decoder");
                        decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
                        let mut rgb = vec![0; 1024 * 1024 * 3];
                        decoder
                            .decode_region_scaled_pow2_into(
                                &mut J2kScratchPool::default(),
                                &mut rgb,
                                1024 * 3,
                                PixelFormat::Rgb8,
                                roi,
                                u8::try_from(reduction).expect("reduction"),
                            )
                            .expect("CPU region decode");
                    });
                }
            });
            let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "large_region level={reduction} sample={sample} tiles=4 metal_batch_ms={metal_ms:.3} cpu_tiles_ms={cpu_ms:.3}"
            );
        }
    }
}
