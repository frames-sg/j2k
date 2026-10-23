// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic Tier-1 shared by the GPU and the CPU pool.

use std::{sync::Arc, time::Duration};

use super::super::{
    current_runtime, execute_prepared_direct_color_plan_batch, metal_command_buffers_for_test,
    reset_metal_command_buffers_for_test, with_isolated_runtime_for_device_for_test,
    PreparedDirectColorPlan,
};
use super::runtime::should_run_metal_runtime;
use super::shared_tier1::{cpu_rgb8, prepared_plan};
use crate::engine::abi::J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
use crate::{Error, J2kDecoder, MetalDecodeRequest};
use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};
use j2k_native::{encode, encode_htj2k, EncodeOptions, J2kWaveletTransform};

const LARGE: (u32, u32) = (1024, 512);
const SMALL: (u32, u32) = (128, 128);

fn transform(reversible: bool) -> J2kWaveletTransform {
    if reversible {
        J2kWaveletTransform::Reversible53
    } else {
        J2kWaveletTransform::Irreversible97
    }
}

fn encoded_rgb((width, height): (u32, u32), reversible: bool, precincts: bool) -> Vec<u8> {
    let options = EncodeOptions {
        reversible,
        code_block_width_exp: 3,
        code_block_height_exp: 3,
        // 128-sample precincts hold 2 x 2 of the 32 x 32 code blocks of each
        // sub-band below the top level, so a group's jobs are out of order.
        precinct_exponents: if precincts {
            vec![(7, 7); 6]
        } else {
            Vec::new()
        },
        ..EncodeOptions::default()
    };
    let pixels = j2k_test_support::patterned_rgb8(width, height);
    encode(&pixels, width, height, 3, 8, false, &options).expect("encode classic RGB8")
}

/// Records splits balanced at `share` until the runtime's share is about it.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "fixture byte counts"
)]
fn set_cpu_share(transform: J2kWaveletTransform, share: f64) -> Result<(), Error> {
    let runtime = current_runtime()?;
    let bytes = |share: f64| (share * 1e6) as u64;
    for _ in 0..8 {
        let time = Duration::from_millis(1);
        runtime.record_classic_split(transform, (bytes(share), time), (bytes(1.0 - share), time));
    }
    Ok(())
}

/// Decodes `plans` to RGB8 and returns their bytes and the command buffers
/// the decode created.
fn metal_rgb8(plans: &[Arc<PreparedDirectColorPlan>]) -> Result<(Vec<u8>, usize), Error> {
    reset_metal_command_buffers_for_test();
    let surfaces = execute_prepared_direct_color_plan_batch(plans, PixelFormat::Rgb8)?;
    let bytes = (surfaces.iter())
        .flat_map(|surface| surface.as_bytes().expect("surface bytes").into_owned())
        .collect();
    Ok((bytes, metal_command_buffers_for_test()))
}

/// A single-image classic RGB decode, small or large, shares Tier-1 with the
/// CPU pool, in one extra command buffer, at any share; a precinct-partitioned
/// image and a batch (of both transforms, which the stacked batch route
/// declines) decode on the GPU alone. Each matches the host.
#[test]
fn classic_rgb_decodes_split_tier1_when_eligible_and_match_host() {
    if !should_run_metal_runtime() {
        return;
    }
    let device = j2k_metal_support::system_default_device().expect("Metal device");
    // (row, size, reversible, precincts, share, batch, split)
    let rows = [
        ("5/3 low share", LARGE, true, false, 0.05, false, true),
        ("5/3 even share", LARGE, true, false, 0.5, false, true),
        ("5/3 high share", LARGE, true, false, 0.95, false, true),
        ("9/7 low share", LARGE, false, false, 0.05, false, true),
        ("9/7 even share", LARGE, false, false, 0.5, false, true),
        ("9/7 high share", LARGE, false, false, 0.95, false, true),
        ("precinct-partitioned", LARGE, true, true, 0.5, false, false),
        ("small image", SMALL, true, false, 0.5, false, true),
        ("two-image batch", LARGE, true, false, 0.5, true, false),
    ];
    for (row, size, reversible, precincts, share, batch, split) in rows {
        with_isolated_runtime_for_device_for_test(&device, || {
            let transforms: &[bool] = if batch {
                &[reversible, !reversible]
            } else {
                &[reversible]
            };
            let images: Vec<_> = (transforms.iter())
                .map(|&reversible| encoded_rgb(size, reversible, precincts))
                .collect();
            let plans: Vec<_> = images
                .iter()
                .map(|bytes| Arc::new(prepared_plan(bytes)))
                .collect();
            let ordered = plans[0].component_plans.iter().all(|plan| {
                let group = plan.classic_group_starting_at(0).expect("classic group");
                group.jobs.is_sorted_by_key(|job| job.output_offset)
            });
            assert_eq!(ordered, !precincts, "{row}");
            set_cpu_share(transform(reversible), share)?;
            let (actual, command_buffers) = metal_rgb8(&plans)?;
            assert_eq!(command_buffers, 1 + usize::from(split), "{row}");
            let expected: Vec<u8> = images
                .iter()
                .flat_map(|bytes| cpu_rgb8(bytes, size))
                .collect();
            assert_eq!(actual, expected, "{row}");
            Ok(())
        })
        .expect(row);
    }
}

/// A strict decode failure in a block the CPU pool decodes fails the decode
/// with the GPU's error, and the next decode on the runtime matches the host.
#[test]
fn split_classic_tier1_failure_reports_the_gpu_error_and_keeps_the_runtime() {
    if !should_run_metal_runtime() {
        return;
    }
    let device = j2k_metal_support::system_default_device().expect("Metal device");
    with_isolated_runtime_for_device_for_test(&device, || {
        let bytes = encoded_rgb(LARGE, true, false);
        set_cpu_share(transform(true), 0.95)?;
        let mut broken = prepared_plan(&bytes);
        // Claim segmentation symbols the coded data does not contain, in
        // blocks of the finest sub-band, which the CPU decodes at this share.
        let jobs = broken.component_plans[0].classic_groups[0].jobs.iter_mut();
        let coded = jobs
            .rev()
            .filter(|job| job.coded_len != 0 && job.number_of_coding_passes > 3);
        let mut broken_jobs = 0;
        for job in coded.take(16) {
            job.style_flags |= J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
            job.strict = 1;
            broken_jobs += 1;
        }
        assert_eq!(broken_jobs, 16);

        let error = metal_rgb8(&[Arc::new(broken)]).expect_err("the broken block fails");
        let message = error.to_string();
        assert!(
            message.contains("classic J2K Metal kernel decode failure"),
            "{message}"
        );
        let (actual, _) = metal_rgb8(&[Arc::new(prepared_plan(&bytes))])?;
        assert_eq!(actual, cpu_rgb8(&bytes, LARGE));
        Ok(())
    })
    .expect("isolated runtime");
}

/// Region and region-scaled colour decodes run Tier-1 on the Metal route, a
/// classic one split with the CPU pool, and match the host's region decode.
#[test]
fn colour_region_decodes_run_metal_tier1_and_match_host() {
    if !should_run_metal_runtime() {
        return;
    }
    let device = j2k_metal_support::system_default_device().expect("Metal device");
    let roi = Rect {
        x: 200,
        y: 100,
        w: 384,
        h: 256,
    };
    // (row, HT, reversible, scale, split)
    let rows = [
        ("classic 5/3 region", false, true, Downscale::None, true),
        (
            "classic 9/7 half-scale region",
            false,
            false,
            Downscale::Half,
            true,
        ),
        ("HT 9/7 region", true, false, Downscale::None, false),
    ];
    for (row, ht, reversible, scale, split) in rows {
        with_isolated_runtime_for_device_for_test(&device, || {
            let bytes = if ht {
                let pixels = j2k_test_support::patterned_rgb8(LARGE.0, LARGE.1);
                let options = EncodeOptions {
                    reversible,
                    ..EncodeOptions::default()
                };
                encode_htj2k(&pixels, LARGE.0, LARGE.1, 3, 8, false, &options).expect("encode HT")
            } else {
                encoded_rgb(LARGE, reversible, false)
            };
            reset_metal_command_buffers_for_test();
            let request = MetalDecodeRequest::region_scaled(
                PixelFormat::Rgb8,
                roi,
                scale,
                BackendRequest::Metal,
            );
            let surface = J2kDecoder::new(&bytes)?.decode_request_to_device(request)?;
            assert_eq!(
                metal_command_buffers_for_test(),
                1 + usize::from(split),
                "{row}"
            );
            let out = roi.scaled_covering(scale);
            let mut expected = vec![0; out.w as usize * out.h as usize * 3];
            j2k::J2kDecoder::new(&bytes)
                .expect("host decoder")
                .decode_region_scaled_into(
                    &mut j2k::J2kScratchPool::default(),
                    &mut expected,
                    out.w as usize * 3,
                    PixelFormat::Rgb8,
                    roi,
                    scale,
                )
                .expect("host region decode");
            let actual = surface.as_bytes().expect("surface bytes");
            assert_eq!(actual.as_ref(), expected.as_slice(), "{row}");
            Ok(())
        })
        .expect(row);
    }
}
