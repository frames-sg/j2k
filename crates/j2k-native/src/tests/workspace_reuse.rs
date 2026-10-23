// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{fixture_gray, fixture_ht_gray};
use crate::{
    encode, encode_typed_component_planes_53, DecodeSettings, DecoderContext, EncodeOptions,
    EncodeTypedComponentPlane, Image,
};

#[test]
fn decoder_workspace_reuses_component_owners_across_distinct_input_lifetimes() {
    let mut workspace = crate::DecoderWorkspace::default();
    let first_pixels = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("first image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("first workspace decode");
        let pixels = decoded.data.clone();
        drop(decoded);
        workspace = context.into_workspace();
        pixels
    };

    let second_pixels = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second workspace decode");
        let pixels = decoded.data.clone();
        drop(decoded);
        workspace = context.into_workspace();
        pixels
    };

    assert_eq!(second_pixels, first_pixels);
    assert_eq!(workspace.stats().decode_calls(), 2);
    assert_eq!(workspace.stats().component_owner_reuses(), 1);
    assert!(workspace.stats().retained_component_bytes() > 0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScratchCapacitySnapshot {
    classic_coefficients: (*const crate::j2c::bitplane::Coefficient, usize),
    classic_payload: (*const u8, usize),
    ht_coefficients: (*const u32, usize),
    idwt_output: (*const f32, usize),
    idwt_scratch: (*const f32, usize),
    idwt_output_i64: (*const i64, usize),
    idwt_scratch_i64: (*const i64, usize),
}

fn scratch_capacity_snapshot(context: &DecoderContext<'_>) -> ScratchCapacitySnapshot {
    let tile = &context.tile_decode_context;
    ScratchCapacitySnapshot {
        classic_coefficients: (
            tile.bit_plane_decode_context.coefficient_ptr_for_test(),
            tile.bit_plane_decode_context
                .coefficient_capacity_for_test(),
        ),
        classic_payload: tile
            .bit_plane_decode_buffers
            .combined_layers_owner_for_test(),
        ht_coefficients: tile.ht_block_decode_context.coefficient_owner_for_test(),
        idwt_output: (
            tile.idwt_output.coefficients.as_ptr(),
            tile.idwt_output.coefficients.capacity(),
        ),
        idwt_scratch: (
            tile.idwt_scratch_buffer.as_ptr(),
            tile.idwt_scratch_buffer.capacity(),
        ),
        idwt_output_i64: (
            tile.idwt_output.coefficients_i64.as_ptr(),
            tile.idwt_output.coefficients_i64.capacity(),
        ),
        idwt_scratch_i64: (
            tile.idwt_scratch_buffer_i64.as_ptr(),
            tile.idwt_scratch_buffer_i64.capacity(),
        ),
    }
}

#[test]
fn decoder_workspace_reuses_classic_tier1_and_idwt_owners_across_input_lifetimes() {
    let mut workspace = crate::DecoderWorkspace::default();
    let first = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("first image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("first classic decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let second = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second classic decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };

    assert!(first.classic_coefficients.1 > 0);
    assert!(first.classic_payload.1 > 0);
    assert!(first.idwt_output.1 > 0);
    assert_eq!(second, first);
    assert_eq!(workspace.stats().tier1_owner_reuses(), 1);
    assert_eq!(workspace.stats().idwt_owner_reuses(), 1);
    assert!(workspace.stats().retained_tier1_bytes() > 0);
    assert!(workspace.stats().retained_idwt_bytes() > 0);
}

#[test]
fn decoder_workspace_reuses_ht_tier1_owner_across_input_lifetimes() {
    let mut workspace = crate::DecoderWorkspace::default();
    let first = {
        let bytes = fixture_ht_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("first HT image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("first HT decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let second = {
        let bytes = fixture_ht_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("second HT image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("second HT decode");
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };

    assert!(first.ht_coefficients.1 > 0);
    assert!(first.idwt_output.1 > 0);
    assert_eq!(second, first);
    assert_eq!(workspace.stats().tier1_owner_reuses(), 1);
    assert_eq!(workspace.stats().idwt_owner_reuses(), 1);
}

#[test]
fn decoder_workspace_reuses_scratch_across_alternating_shapes_and_precision() {
    let large_pixels = (0..16_u16 * 16)
        .map(|sample| (sample & 0xff) as u8)
        .collect::<Vec<_>>();
    let large_bytes = encode(
        &large_pixels,
        16,
        16,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode large classic fixture");
    let exact_samples = [0_u32, 1, (1_u32 << 28) + 7, (1_u32 << 29) - 1];
    let exact_pixels = exact_samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let exact_planes = [EncodeTypedComponentPlane {
        data: &exact_pixels,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 29,
        signed: false,
    }];
    let exact_bytes = encode_typed_component_planes_53(
        &exact_planes,
        2,
        2,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode exact fixture");

    let mut workspace = crate::DecoderWorkspace::default();
    let large_scratch = {
        let image = Image::new(&large_bytes, &DecodeSettings::default()).expect("large image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_native_with_context(&mut context)
            .expect("large decode");
        assert_eq!(decoded.data, large_pixels);
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let exact_scratch = {
        let image = Image::new(&exact_bytes, &DecodeSettings::default()).expect("exact image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_native_with_context(&mut context)
            .expect("exact decode");
        assert_eq!(decoded.data, exact_pixels);
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };
    let final_scratch = {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("small image");
        let mut context = DecoderContext::from_workspace(workspace);
        let decoded = image
            .decode_with_context(&mut context)
            .expect("small decode");
        assert_eq!(decoded.data, (0_u8..16).collect::<Vec<_>>());
        drop(decoded);
        let scratch = scratch_capacity_snapshot(&context);
        workspace = context.into_workspace();
        scratch
    };

    assert_eq!(
        final_scratch.classic_coefficients,
        large_scratch.classic_coefficients
    );
    assert_eq!(final_scratch.classic_payload, large_scratch.classic_payload);
    assert_eq!(final_scratch.idwt_output, large_scratch.idwt_output);
    assert_eq!(final_scratch.idwt_scratch, large_scratch.idwt_scratch);
    assert!(exact_scratch.idwt_output_i64.1 > 0);
    assert_eq!(final_scratch.idwt_output_i64, exact_scratch.idwt_output_i64);
    assert_eq!(workspace.stats().decode_calls(), 3);
    assert_eq!(workspace.stats().tier1_owner_reuses(), 2);
    assert_eq!(workspace.stats().idwt_owner_reuses(), 2);
    assert_eq!(workspace.stats().scratch_capacity_retries(), 0);
}

#[cfg(feature = "parallel")]
fn parallel_gray_fixture(width: u32, height: u32, ht: bool, reversible: bool) -> Vec<u8> {
    let pixels = super::gradient_pixels(width, height, 1);
    encode(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            reversible,
            guard_bits: if reversible { 1 } else { 2 },
            num_decomposition_levels: 3,
            use_ht_block_coding: ht,
            ..EncodeOptions::default()
        },
    )
    .expect("encode parallel workspace fixture")
}

#[cfg(feature = "parallel")]
fn parallel_workspace_measurements(context: &DecoderContext<'_>) -> (usize, usize, usize, usize) {
    let tile = &context.tile_decode_context;
    let counters = tile.debug_counters;
    (
        tile.ht_parallel_workspace_count(),
        tile.ht_parallel_workspace_bytes().expect("HT bank bytes"),
        counters.ht_parallel_tasks,
        counters.ht_task_workspace_growths,
    )
}

#[cfg(feature = "parallel")]
#[test]
fn ht_task_workspaces_match_serial_and_reuse_capacity() {
    for (reversible, per_sub_band) in [(false, false), (true, false), (false, true), (true, true)] {
        let bytes = parallel_gray_fixture(257, 263, true, reversible);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("odd image");
        let mut serial = DecoderContext::default();
        serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
        let expected = image
            .decode_with_context(&mut serial)
            .expect("serial baseline");
        if reversible {
            assert_eq!(expected.data, super::gradient_pixels(257, 263, 1));
        }
        for threads in [2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test pool");
            pool.install(|| {
                let mut context = DecoderContext::default();
                context.tile_decode_context.force_per_sub_band = per_sub_band;
                let first = image
                    .decode_with_context(&mut context)
                    .expect("first decode");
                assert_eq!(first.data, expected.data, "reversible={reversible}");
                assert_eq!((first.width, first.height), (257, 263));
                let (slots, bytes, tasks, growths) = parallel_workspace_measurements(&context);
                assert!(
                    tasks > 0 && tasks <= threads.saturating_mul(2),
                    "tasks={tasks}, pool={threads}"
                );
                assert!(slots > 0 && slots <= tasks);
                assert!(
                    slots
                        < context
                            .tile_decode_context
                            .debug_counters
                            .decoded_code_blocks
                );
                assert!(bytes > 0 && growths > 0);

                let second = image
                    .decode_with_context(&mut context)
                    .expect("warm decode");
                assert_eq!(second.data, expected.data);
                let (warm_slots, warm_bytes, warm_tasks, warm_growths) =
                    parallel_workspace_measurements(&context);
                assert_eq!((warm_slots, warm_bytes, warm_tasks), (slots, bytes, tasks));
                assert_eq!(warm_growths, 0, "unchanged workspaces must not grow");
            });
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn ht_task_workspaces_survive_input_lifetimes_and_region_errors() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test pool");
    for per_sub_band in [false, true] {
        pool.install(|| {
            let (workspace, retained_bank_bytes) = {
                let bytes = parallel_gray_fixture(257, 263, true, true);
                let image = Image::new(&bytes, &DecodeSettings::default()).expect("first image");
                let mut context = DecoderContext::default();
                context.tile_decode_context.force_per_sub_band = per_sub_band;
                let decoded = image
                    .decode_with_context(&mut context)
                    .expect("first decode");
                assert_eq!(decoded.data, super::gradient_pixels(257, 263, 1));
                let (_, bank_bytes, tasks, _) = parallel_workspace_measurements(&context);
                assert!(tasks > 0);
                (context.into_workspace(), bank_bytes)
            };
            // The original codestream and all borrowed parsing state are gone.
            let bytes = parallel_gray_fixture(241, 127, true, true);
            let image = Image::new(&bytes, &DecodeSettings::default()).expect("second image");
            let mut context = DecoderContext::from_workspace(workspace);
            context.tile_decode_context.force_per_sub_band = per_sub_band;
            let decoded = image
                .decode_with_context(&mut context)
                .expect("second decode");
            assert_eq!(decoded.data, super::gradient_pixels(241, 127, 1));
            let (_, bank_bytes, _, growths) = parallel_workspace_measurements(&context);
            assert!(bank_bytes <= retained_bank_bytes);
            assert_eq!(growths, 0);

            let roi = (7, 9, 53, 47);
            let region = image
                .decode_region_with_context(roi, &mut context)
                .expect("ROI");
            assert_eq!(
                region.data,
                super::crop_interleaved(&decoded.data, 241, 1, roi)
            );
            assert!(image
                .decode_region_with_context((242, 0, 1, 1), &mut context)
                .is_err());
            let recovered = image
                .decode_with_context(&mut context)
                .expect("reuse after error");
            assert_eq!(recovered.data, decoded.data);

            let reduced = Image::new(
                &bytes,
                &DecodeSettings {
                    target_resolution: Some((121, 64)),
                    ..DecodeSettings::default()
                },
            )
            .expect("reduced image");
            let mut serial = DecoderContext::default();
            serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
            let expected = reduced
                .decode_with_context(&mut serial)
                .expect("serial reduced");
            let actual = reduced
                .decode_with_context(&mut context)
                .expect("reused reduced");
            assert_eq!(actual.data, expected.data);
            assert_eq!(
                (actual.width, actual.height),
                (expected.width, expected.height)
            );
            assert_eq!(parallel_workspace_measurements(&context).3, 0);
        });
    }
}

#[cfg(feature = "parallel")]
#[test]
fn ht_task_workspaces_remain_reusable_across_classic_decode() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test pool");
    for (reversible, per_sub_band) in [(true, false), (false, false), (true, true), (false, true)] {
        let ht_bytes = parallel_gray_fixture(257, 263, true, reversible);
        let classic_bytes = parallel_gray_fixture(241, 127, false, reversible);
        let ht = Image::new(&ht_bytes, &DecodeSettings::default()).expect("HT image");
        let classic =
            Image::new(&classic_bytes, &DecodeSettings::default()).expect("classic image");
        pool.install(|| {
            let mut context = DecoderContext::default();
            context.tile_decode_context.force_per_sub_band = per_sub_band;
            let first = ht
                .decode_with_context(&mut context)
                .expect("first HT decode");
            let (slots, bank_bytes, _, growths) = parallel_workspace_measurements(&context);
            assert!(slots > 0 && bank_bytes > 0 && growths > 0);

            let mut serial = DecoderContext::default();
            serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
            let expected = classic
                .decode_with_context(&mut serial)
                .expect("classic oracle");
            let actual = classic
                .decode_with_context(&mut context)
                .expect("classic with HT scratch");
            assert_eq!(actual.data, expected.data);
            assert_eq!(
                (actual.width, actual.height),
                (expected.width, expected.height)
            );
            let (classic_slots, classic_bank_bytes, tasks, growths) =
                parallel_workspace_measurements(&context);
            assert_eq!((classic_slots, classic_bank_bytes), (slots, bank_bytes));
            assert_eq!((tasks, growths), (0, 0));

            let repeated = ht
                .decode_with_context(&mut context)
                .expect("HT after classic");
            assert_eq!(repeated.data, first.data);
            let (warm_slots, warm_bytes, tasks, growths) =
                parallel_workspace_measurements(&context);
            assert_eq!((warm_slots, warm_bytes), (slots, bank_bytes));
            assert!(tasks > 0);
            assert_eq!(growths, 0);
            assert_eq!(context.workspace_stats().scratch_capacity_retries(), 0);
        });
    }
}

#[cfg(feature = "parallel")]
#[test]
fn classic_tile_route_matches_serial_decode() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test pool");
    let gradient = |components: u8, num_layers: u8| {
        let pixels = super::gradient_pixels(257, 263, components);
        let options = EncodeOptions {
            reversible: true,
            guard_bits: 1,
            num_decomposition_levels: 3,
            num_layers,
            ..EncodeOptions::default()
        };
        encode(&pixels, 257, 263, components.into(), 8, false, &options)
            .expect("encode classic fixture")
    };
    // (case, codestream, decoded in place on the tile route). The later
    // components of a colour image decode into the reused coefficient buffer,
    // most blocks of the three-layer OpenJPEG fixture gather their payloads,
    // and the crate's own layered classic encode terminates every pass, which
    // keeps it on the per-sub-band path.
    for (case, bytes, in_place) in [
        ("lossless colour", gradient(3, 1), true),
        (
            "three layers",
            include_bytes!("../../fixtures/quality_layers/openjpeg_rgb_96x80_3layers_97.j2k")
                .to_vec(),
            true,
        ),
        (
            "ROI shift",
            include_bytes!(
                "../../fixtures/reversible_early_stop/openjpeg_53_rate3_roi2_gray8_19x13.j2k"
            )
            .to_vec(),
            true,
        ),
        ("pass termination", gradient(1, 3), false),
    ] {
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("classic image");
        let mut serial = DecoderContext::default();
        serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
        let expected = image
            .decode_with_context(&mut serial)
            .expect("serial oracle");
        pool.install(|| {
            let mut context = DecoderContext::default();
            let actual = image
                .decode_with_context(&mut context)
                .expect("Auto decode");
            assert_eq!(actual.data, expected.data, "{case}");
            let staged = context
                .tile_decode_context
                .debug_counters
                .parallel_coefficients
                .allocations;
            assert_eq!(staged == 0, in_place, "{case}");
            // The route keeps one classic workspace per worker over all the
            // components, and for the next decode with the context.
            let counters = &context.tile_decode_context.debug_counters;
            let growths = counters.classic_task_workspace_growths;
            assert_eq!(growths > 0 && growths <= 2, in_place, "{case}");
            let warm = image
                .decode_with_context(&mut context)
                .expect("warm Auto decode");
            assert_eq!(warm.data, expected.data, "{case}");
            let counters = &context.tile_decode_context.debug_counters;
            assert_eq!(counters.classic_task_workspace_growths, 0, "{case}");
        });
    }
}

#[cfg(feature = "parallel")]
#[test]
#[ignore = "coefficient allocation measurement; run explicitly with --ignored --nocapture"]
fn parallel_coefficient_allocation_report() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test pool");
    for ht in [false, true] {
        for reversible in [false, true] {
            let bytes = parallel_gray_fixture(257, 263, ht, reversible);
            let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
            pool.install(|| {
                let mut context = DecoderContext::default();
                context.tile_decode_context.force_per_sub_band = true;
                let cold = image.decode_with_context(&mut context).expect("cold decode");
                let cold_stats = context.tile_decode_context.debug_counters.parallel_coefficients;
                let cold_retained = context.tile_decode_context.tier1_capacity_bytes().expect("retained bytes");
                let warm = image.decode_with_context(&mut context).expect("warm decode");
                let warm_stats = context.tile_decode_context.debug_counters.parallel_coefficients;
                let warm_retained = context.tile_decode_context.tier1_capacity_bytes().expect("retained bytes");
                assert_eq!(warm.data, cold.data);
                assert!(cold_stats.allocations > 0);
                assert!(cold_stats.scatter_bytes > 0);
                assert_eq!(warm_stats.scatter_bytes, cold_stats.scatter_bytes);
                println!("parallel_coefficients ht={ht} reversible={reversible} width=257 height=263 pool=2 cold={cold_stats:?} warm={warm_stats:?} cold_retained_tier1_bytes={cold_retained} warm_retained_tier1_bytes={warm_retained}");
            });
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_coefficient_buffers_reuse_capacity_across_fitting_images() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test pool");
    for ht in [false, true] {
        for reversible in [false, true] {
            let first_bytes = parallel_gray_fixture(257, 263, ht, reversible);
            let second_bytes = parallel_gray_fixture(241, 127, ht, reversible);
            let first = Image::new(&first_bytes, &DecodeSettings::default()).expect("first image");
            let second =
                Image::new(&second_bytes, &DecodeSettings::default()).expect("second image");
            pool.install(|| {
                let mut context = DecoderContext::default();
                // The slab belongs to the per-sub-band path; tile-wide
                // decode writes code blocks in place.
                context.tile_decode_context.force_per_sub_band = true;
                first.decode_with_context(&mut context).expect("reserve coefficient buffers");
                assert!(context.tile_decode_context.debug_counters.parallel_coefficients.allocations > 0);
                for image in [&first, &second, &first] {
                    let mut serial = DecoderContext::default();
                    serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
                    let expected = image.decode_with_context(&mut serial).expect("serial oracle");
                    let actual = image.decode_with_context(&mut context).expect("reused decode");
                    assert_eq!(actual.data, expected.data, "ht={ht}, reversible={reversible}");
                    assert_eq!((actual.width, actual.height), (expected.width, expected.height));
                    assert_eq!(context.tile_decode_context.debug_counters.parallel_coefficients.allocations, 0,
                        "fitting coefficient buffers must not allocate again: ht={ht}, reversible={reversible}");
                }
            });
        }
    }
}

#[cfg(feature = "parallel")]
#[path = "../../benches/support/scheduling_fixtures.rs"]
mod scheduling_fixtures;

#[cfg(feature = "parallel")]
#[test]
#[ignore = "scheduling fixture route report; run explicitly with --ignored --nocapture"]
fn scheduling_fixture_route_report() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test pool");
    for ht in [false, true] {
        for (side, sparse) in [(512, false), (256, false), (256, true)] {
            let pixels = if sparse {
                scheduling_fixtures::sparse_gray8(side, side, 3)
            } else {
                scheduling_fixtures::dense_gray8(side, side, 3)
            };
            let bytes = encode(
                &pixels,
                side,
                side,
                1,
                8,
                false,
                &EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 5,
                    use_ht_block_coding: ht,
                    ..EncodeOptions::default()
                },
            )
            .expect("encode scheduling fixture");
            let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
            pool.install(|| {
                let mut serial = DecoderContext::default();
                serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
                let expected = image.decode_with_context(&mut serial).expect("serial oracle");
                assert_eq!(expected.data, pixels);
                assert_eq!(serial.tile_decode_context.debug_counters.parallel_coefficients.scatter_bytes, 0);
                let mut context = DecoderContext::default();
                let actual = image.decode_with_context(&mut context).expect("Auto decode");
                assert_eq!(actual.data, expected.data);
                assert_eq!((actual.width, actual.height), (side, side));
                let scatter = context.tile_decode_context.debug_counters.parallel_coefficients.scatter_bytes;
                assert!(scatter > 0, "fixture must exercise parallel staging: ht={ht}, side={side}, sparse={sparse}");
                println!("scheduling_fixture ht={ht} side={side} sparse={sparse} encoded_bytes={} parallel_scatter_bytes={scatter}", bytes.len());
            });
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn ht_auto_single_worker_matches_serial_without_parallel_staging() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single worker pool");
    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("parallel warmup pool");
    for reversible in [false, true] {
        let bytes = parallel_gray_fixture(257, 263, true, reversible);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("HT image");
        let mut serial = DecoderContext::default();
        serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
        let expected = image
            .decode_with_context(&mut serial)
            .expect("serial oracle");
        let mut context = DecoderContext::default();
        pool.install(|| {
            for _ in 0..2 {
                let actual = image
                    .decode_with_context(&mut context)
                    .expect("one-worker Auto decode");
                assert_eq!(actual.data, expected.data);
                assert_eq!((actual.width, actual.height), (257, 263));
                assert_eq!(parallel_workspace_measurements(&context), (0, 0, 0, 0));
                let counters = context.tile_decode_context.debug_counters;
                assert!(counters.decoded_code_blocks > 0);
                assert_eq!(counters.parallel_coefficients.allocations, 0);
                assert_eq!(counters.parallel_coefficients.scatter_bytes, 0);
            }
        });
        parallel_pool.install(|| {
            image
                .decode_with_context(&mut context)
                .expect("populate parallel bank");
        });
        let retained = parallel_workspace_measurements(&context);
        assert!(retained.0 > 0 && retained.2 > 0);
        pool.install(|| {
            let actual = image
                .decode_with_context(&mut context)
                .expect("one worker after parallel reuse");
            assert_eq!(actual.data, expected.data);
            let current = parallel_workspace_measurements(&context);
            assert_eq!((current.0, current.1), (retained.0, retained.1));
            assert_eq!((current.2, current.3), (0, 0));
            let coefficients = context
                .tile_decode_context
                .debug_counters
                .parallel_coefficients;
            assert_eq!(
                (coefficients.allocations, coefficients.scatter_bytes),
                (0, 0)
            );
        });
    }
}

#[cfg(feature = "parallel")]
#[test]
fn ht_compact_region_with_empty_trailing_blocks_matches_source() {
    let mut pixels = vec![128_u8; 128 * 256];
    let mut state = 0x2545_F491_u32;
    for pixel in &mut pixels[128 * 64..128 * 192] {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *pixel = state.to_le_bytes()[0];
    }
    let bytes = encode(
        &pixels,
        128,
        256,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        },
    )
    .expect("encode compact-region fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("HT image");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("parallel pool");
    pool.install(|| {
        for policy in [
            crate::CpuDecodeParallelism::Serial,
            crate::CpuDecodeParallelism::Auto,
        ] {
            let mut context = DecoderContext::default();
            context.set_cpu_decode_parallelism(policy);
            let actual = image
                .decode_region_with_context((0, 64, 128, 192), &mut context)
                .expect("compact region decode");
            assert_eq!((actual.width, actual.height), (128, 192));
            assert!(
                actual.data.as_slice() == &pixels[128 * 64..],
                "{policy:?} ROI pixels differ"
            );
        }
    });
}

#[cfg(feature = "parallel")]
#[test]
fn ht_refinement_across_precincts_matches_serial_decode() {
    let pixels: Vec<u8> = (0..256)
        .map(|index| u8::try_from(index * 73 % 251).expect("sample fits u8"))
        .collect();
    let options = EncodeOptions {
        reversible: false,
        use_ht_block_coding: true,
        num_layers: 3,
        num_decomposition_levels: 0,
        code_block_width_exp: 0,
        code_block_height_exp: 0,
        precinct_exponents: vec![(3, 3)],
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 16, 16, 1, 8, false, &options)
        .expect("encode multiple-precinct refinement fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("HT image");
    let mut serial = DecoderContext::default();
    serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
    let expected = image
        .decode_with_context(&mut serial)
        .expect("serial oracle");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("parallel pool");
    pool.install(|| {
        let mut context = DecoderContext::default();
        let actual = image
            .decode_with_context(&mut context)
            .expect("parallel multiple-precinct refinement decode");
        assert_eq!(actual.data, expected.data);
        assert!(
            context
                .tile_decode_context
                .debug_counters
                .ht_phase_stats
                .refinement_blocks
                > 0
        );
    });
}

#[cfg(feature = "parallel")]
#[test]
fn ht_workers_preserve_entropy_error_and_context_recovery() {
    let valid_bytes = super::fixture_ht_multi_block();
    let mut malformed_bytes = valid_bytes.clone();
    assert!(malformed_bytes.ends_with(&[0xff, 0xd9]));
    // This fixture has four cleanup-only blocks in one packet. Its last
    // cleanup suffix precedes EOC; 0xff makes Scup exceed the maximum 4079.
    let suffix = malformed_bytes.len() - 3;
    malformed_bytes[suffix] = 0xff;
    let malformed =
        Image::new(&malformed_bytes, &DecodeSettings::default()).expect("header still parses");
    let valid = Image::new(&valid_bytes, &DecodeSettings::default()).expect("valid image");
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("decode worker pool");
        pool.install(|| {
            for policy in [
                crate::CpuDecodeParallelism::Auto,
                crate::CpuDecodeParallelism::Serial,
            ] {
                let mut context = DecoderContext::default();
                context.set_cpu_decode_parallelism(policy);
                let error = malformed
                    .decode_with_context(&mut context)
                    .err()
                    .expect("malformed entropy must fail");
                assert_eq!(
                    error,
                    crate::DecodeError::Decoding(crate::DecodingError::CodeBlockDecodeFailure)
                );
                if threads == 1 || policy == crate::CpuDecodeParallelism::Serial {
                    assert!(
                        context
                            .tile_decode_context
                            .debug_counters
                            .decoded_code_blocks
                            > 0
                    );
                    assert_eq!(parallel_workspace_measurements(&context), (0, 0, 0, 0));
                } else {
                    assert!(parallel_workspace_measurements(&context).0 > 0);
                }
                let coefficients = context
                    .tile_decode_context
                    .debug_counters
                    .parallel_coefficients;
                assert_eq!(
                    (coefficients.allocations, coefficients.scatter_bytes),
                    (0, 0)
                );
                let recovered = valid
                    .decode_with_context(&mut context)
                    .expect("reuse after entropy error");
                assert_eq!(recovered.data, (0u8..64).collect::<Vec<_>>());
            }
        });
    }
}

/// IDWT level and band-scratch samples zero-initialized while serially decoding
/// one 256x256, five-level HTJ2K tile with `components` components.
fn idwt_zero_initialized_samples(components: u8) -> usize {
    let (width, height) = (256, 256);
    let pixels = super::gradient_pixels(width, height, components);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 5,
        ..EncodeOptions::default()
    };
    let bytes = crate::encode_htj2k(
        &pixels,
        width,
        height,
        components.into(),
        8,
        false,
        &options,
    )
    .expect("encode IDWT fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("IDWT fixture");
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
    // The fused last level keeps an LL band per component; see
    // `fused_last_level`.
    context.tile_decode_context.staged_rgb8_pack = true;
    crate::j2c::idwt::reset_zero_initialized_samples();
    let decoded = image
        .decode_with_context(&mut context)
        .expect("decode IDWT fixture");
    assert_eq!(decoded.data, pixels);
    crate::j2c::idwt::zero_initialized_samples()
}

#[test]
fn idwt_buffers_are_zero_initialized_once_per_tile_not_once_per_component() {
    let gray = idwt_zero_initialized_samples(1);
    let rgb = idwt_zero_initialized_samples(3);

    assert!(gray > 0);
    // Every level overwrites the samples it uses, so later components reuse
    // the samples the first component already initialized.
    assert_eq!(rgb, gray);
}

/// Decode `bytes` twice with one context, serially. Before the second decode
/// every retained component-plane sample is poisoned with a value no decode
/// produces, so any sample `store` does not overwrite shows up in the output.
/// Returns (first output, second output, retained plane samples the second
/// decode zeroed).
fn decode_twice_with_poisoned_planes(
    bytes: &[u8],
    settings: &DecodeSettings,
) -> (Vec<u8>, Vec<u8>, usize) {
    let image = Image::new(bytes, settings).expect("plane reuse fixture");
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
    let first = image
        .decode_with_context(&mut context)
        .expect("first decode")
        .data;
    for component in &mut context.tile_decode_context.channel_data {
        component.container.fill(77.0);
    }
    crate::math::reset_retained_zeroed_samples();
    let second = image
        .decode_with_context(&mut context)
        .expect("second decode")
        .data;
    (first, second, crate::math::retained_zeroed_samples())
}

fn rgb_plane_fixture(width: u32, height: u32, tile_size: Option<(u32, u32)>) -> Vec<u8> {
    let pixels = super::gradient_pixels(width, height, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        tile_size,
        ..EncodeOptions::default()
    };
    crate::encode_htj2k(&pixels, width, height, 3, 8, false, &options)
        .expect("encode RGB plane fixture")
}

#[test]
fn retained_planes_are_not_rezeroed_when_store_overwrites_every_sample() {
    for (name, bytes) in [
        ("single tile", rgb_plane_fixture(64, 64, None)),
        (
            "partial edge tiles",
            rgb_plane_fixture(64, 64, Some((24, 24))),
        ),
    ] {
        let (first, second, zeroed) =
            decode_twice_with_poisoned_planes(&bytes, &DecodeSettings::default());
        assert_eq!(second, first, "{name}");
        assert_eq!(
            zeroed, 0,
            "{name}: 64x64 planes have no SIMD padding to zero"
        );
    }
}

#[test]
fn retained_planes_store_does_not_fully_cover_are_still_zeroed() {
    let reduced = DecodeSettings {
        target_resolution: Some((16, 16)),
        ..DecodeSettings::default()
    };
    let full = super::gradient_pixels(32, 32, 1);
    let chroma = super::gradient_pixels(16, 16, 1);
    let subsampled = encode_typed_component_planes_53(
        &[
            EncodeTypedComponentPlane {
                data: &full,
                x_rsiz: 1,
                y_rsiz: 1,
                bit_depth: 8,
                signed: false,
            },
            EncodeTypedComponentPlane {
                data: &chroma,
                x_rsiz: 2,
                y_rsiz: 2,
                bit_depth: 8,
                signed: false,
            },
            EncodeTypedComponentPlane {
                data: &chroma,
                x_rsiz: 2,
                y_rsiz: 2,
                bit_depth: 8,
                signed: false,
            },
        ],
        32,
        32,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            use_mct: false,
            ..EncodeOptions::default()
        },
    )
    .expect("encode subsampled fixture");
    for (name, bytes, settings) in [
        (
            "reduced resolution",
            rgb_plane_fixture(64, 64, Some((24, 24))),
            reduced,
        ),
        ("mixed subsampling", subsampled, DecodeSettings::default()),
    ] {
        let (first, second, zeroed) = decode_twice_with_poisoned_planes(&bytes, &settings);
        assert_eq!(second, first, "{name}");
        assert!(
            zeroed > 0,
            "{name}: planes store may not cover must be zeroed"
        );
    }
}

#[test]
fn planes_left_unzeroed_count_every_stored_sample() {
    let reduced = DecodeSettings {
        target_resolution: Some((16, 16)),
        ..DecodeSettings::default()
    };
    for (name, tile_size, settings, unzeroed) in [
        ("single tile", None, DecodeSettings::default(), 3 * 64 * 64),
        (
            "partial edge tiles",
            Some((24, 24)),
            DecodeSettings::default(),
            3 * 64 * 64,
        ),
        (
            "reduced resolution zeroes its planes",
            Some((24, 24)),
            reduced,
            0,
        ),
    ] {
        let bytes = rgb_plane_fixture(64, 64, tile_size);
        let image = Image::new(&bytes, &settings).expect("plane store fixture");
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
        image
            .decode_with_context(&mut context)
            .expect("plane store decode");

        let tile_context = &context.tile_decode_context;
        assert_eq!(tile_context.unzeroed_plane_samples, unzeroed, "{name}");
        if unzeroed != 0 {
            assert_eq!(tile_context.stored_plane_samples, unzeroed, "{name}");
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn malformed_classic_tile_route_matches_serial_error() {
    let mut state = 0x1234_5678_u32;
    let pixels: Vec<u8> = (0..257 * 263)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect();
    let bytes = encode(
        &pixels,
        257,
        263,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            guard_bits: 1,
            num_decomposition_levels: 3,
            ..EncodeOptions::default()
        },
    )
    .expect("classic fixture");
    let qcd = bytes
        .windows(2)
        .position(|w| w == [0xff, 0x5c])
        .expect("QCD");
    assert_eq!(bytes[qcd + 4] & 31, 0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test pool");
    // A later invalid band must not hide the earlier coding-pass error.
    // Keeping that later band valid also exercises failure during execution.
    for next_exponent in [0, bytes[qcd + 6]] {
        let mut malformed = bytes.clone();
        malformed[qcd + 4] = 0;
        malformed[qcd + 5] = 2 << 3;
        malformed[qcd + 6] = next_exponent;
        let image = Image::new(&malformed, &DecodeSettings::default()).expect("malformed image");
        pool.install(|| {
            for mode in [
                crate::CpuDecodeParallelism::Serial,
                crate::CpuDecodeParallelism::Auto,
            ] {
                let mut context = DecoderContext::default();
                context.set_cpu_decode_parallelism(mode);
                let error = image
                    .decode_with_context(&mut context)
                    .err()
                    .expect("coding-pass error");
                assert_eq!(
                    error,
                    crate::DecodeError::Decoding(crate::DecodingError::TooManyCodingPasses)
                );
            }
        });
    }
}

#[cfg(feature = "parallel")]
#[test]
fn failed_tile_route_clears_coefficients_before_serial_retry() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("test pool");
    for ht in [false, true] {
        for components in [1_u8, 3] {
            let pixels: Vec<u8> = (0..257_u32 * 263)
                .flat_map(|i| {
                    let value = if i % 257 < 128 { (i % 251) as u8 } else { 128 };
                    core::iter::repeat_n(value, usize::from(components))
                })
                .collect();
            let bytes = encode(
                &pixels,
                257,
                263,
                components.into(),
                8,
                false,
                &EncodeOptions {
                    reversible: true,
                    use_ht_block_coding: ht,
                    num_decomposition_levels: 0,
                    ..EncodeOptions::default()
                },
            )
            .expect("fixture with coded and empty blocks");
            for strict in [false, true] {
                let image = Image::new(
                    &bytes,
                    &DecodeSettings {
                        strict,
                        ..DecodeSettings::default()
                    },
                )
                .expect("image");
                pool.install(|| {
                    let mut serial = DecoderContext::default();
                    serial.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
                    let expected = image
                        .decode_with_context(&mut serial)
                        .expect("serial oracle");
                    let mut context = DecoderContext::default();
                    context.tile_decode_context.tile_route_post_decode = Some(|storage| {
                        assert!(storage.coefficients.iter().any(|value| *value != 0.0));
                        assert!(storage
                            .code_blocks
                            .iter()
                            .any(|block| block.coding.is_none()));
                        // Simulate a partially written failed block, including
                        // samples the serial retry skips because it has no passes.
                        storage.coefficients.fill(123.0);
                        Err(crate::DecodingError::CodeBlockDecodeFailure.into())
                    });
                    let actual = image
                        .decode_with_context(&mut context)
                        .expect("serial retry");
                    assert!(
                        context.tile_decode_context.tile_route_post_decode.is_none(),
                        "route must run"
                    );
                    assert_eq!(
                        actual.data, expected.data,
                        "ht={ht} components={components} strict={strict}"
                    );
                });
            }
        }
    }
}
