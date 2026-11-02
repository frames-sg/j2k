// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::{
    encode, execute_referenced_classic_plan, DecodeSettings, DecoderContext, EncodeOptions, Image,
    J2kDirectCpuScratch,
};

#[cfg(feature = "parallel")]
fn rgb_fixture(width: u32, height: u32, reversible: bool) -> Vec<u8> {
    rgb_fixture_with_layers(width, height, reversible, 1)
}

fn rgb_fixture_with_layers(width: u32, height: u32, reversible: bool, num_layers: u8) -> Vec<u8> {
    let samples = (0..width * height * 3)
        .map(|index| {
            let (pixel, channel) = (index / 3, index % 3);
            let (x, y) = (pixel % width, pixel / width);
            u8::try_from((x * 5 + y * 3 + channel * 41 + (x ^ y) % 37) & 0xff)
                .expect("masked sample")
        })
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible,
        num_decomposition_levels: 4,
        num_layers,
        ..EncodeOptions::default()
    };
    encode(&samples, width, height, 3, 8, false, &options).expect("encode classic RGB fixture")
}

/// Reconstructed planes as raw bits, the retained classic workspace bytes,
/// and each extra task's payload buffer (length, capacity) after one
/// execution.
struct Execution {
    planes: Vec<Vec<u32>>,
    workspace_bytes: usize,
    task_payloads: Vec<(usize, usize)>,
}

type Region = (u32, u32, u32, u32);

fn whole_image(bytes: &[u8]) -> Region {
    let image = Image::new(bytes, &DecodeSettings::default()).expect("classic image");
    (0, 0, image.width(), image.height())
}

/// The largest payload split over several input fragments, if any.
#[cfg(feature = "parallel")]
fn largest_fragmented_payload(bytes: &[u8], region: Region) -> Option<usize> {
    let image = Image::new(bytes, &DecodeSettings::default()).expect("classic image");
    let plan = image
        .build_referenced_classic_plan_region_with_context(&mut DecoderContext::default(), region)
        .expect("classic region plan");
    plan.payloads()
        .iter()
        .filter(|payload| payload.range_count > 1)
        .map(|payload| payload.combined_length)
        .max()
}

fn execute_region(bytes: &[u8], region: Region, parallel: bool) -> Execution {
    let image = Image::new(bytes, &DecodeSettings::default()).expect("classic image");
    let plan = image
        .build_referenced_classic_plan_region_with_context(&mut DecoderContext::default(), region)
        .expect("classic region plan");
    let mut scratch = J2kDirectCpuScratch::new();
    scratch.set_parallel_decode(parallel);
    let decoded = execute_referenced_classic_plan(&plan, bytes, false, &mut scratch)
        .expect("execute referenced classic plan");
    let planes = (0..3)
        .map(|component| {
            decoded
                .plane(component)
                .expect("RGB plane")
                .samples()
                .iter()
                .map(|sample| sample.to_bits())
                .collect()
        })
        .collect();
    Execution {
        planes,
        workspace_bytes: scratch.retained_classic_workspace_bytes(),
        task_payloads: scratch.classic_task_payloads(),
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_referenced_classic_execution_matches_serial_bits() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("four-thread pool");
    for reversible in [true, false] {
        // Several rows of 64 x 64 code blocks per high-resolution sub-band.
        let bytes = rgb_fixture(300, 260, reversible);
        let region = whole_image(&bytes);
        assert_eq!(largest_fragmented_payload(&bytes, region), None);
        let serial = execute_region(&bytes, region, false);
        let parallel = pool.install(|| execute_region(&bytes, region, true));
        assert_eq!(serial.planes, parallel.planes, "reversible={reversible}");
        // The tallest sub-band (130 rows) holds three rows of 64-high
        // blocks, so of the four workers only two extra ones can take a row
        // and keep their own prepared workspace.
        assert_eq!(
            parallel.workspace_bytes,
            serial.workspace_bytes * 3,
            "reversible={reversible}"
        );
        // Single-fragment payloads are borrowed from the input, so the extra
        // tasks keep no payload buffer.
        assert_eq!(
            parallel.task_payloads,
            [(0, 0); 2],
            "reversible={reversible}"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn fragmented_payloads_decode_in_parallel_with_per_task_buffers() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("four-thread pool");
    for (reversible, layers) in [(false, 4), (true, 8), (false, 8)] {
        let bytes = rgb_fixture_with_layers(300, 260, reversible, layers);
        let crop = (37, 21, 230, 211);
        for region in [whole_image(&bytes), crop] {
            let case = format!("reversible={reversible} layers={layers} region={region:?}");
            let largest = largest_fragmented_payload(&bytes, region)
                .unwrap_or_else(|| panic!("{case}: layers split payloads over fragments"));
            let serial = execute_region(&bytes, region, false);
            let parallel = pool.install(|| execute_region(&bytes, region, true));
            assert_eq!(serial.planes, parallel.planes, "{case}");
            // One prepared workspace per extra task, each beside its buffer.
            let tasks = parallel.task_payloads.len();
            assert!(tasks >= 1, "{case}: no extra task");
            assert_eq!(
                parallel.workspace_bytes,
                serial.workspace_bytes * (tasks + 1),
                "{case}"
            );
            for &(_, capacity) in &parallel.task_payloads {
                assert!(
                    capacity >= largest,
                    "{case}: task buffer {capacity} < {largest}"
                );
            }
            // The pool hands out the extra tasks first, so the parallel rows
            // joined fragments in their own buffers.
            assert!(
                parallel.task_payloads.iter().any(|&(len, _)| len > 0),
                "{case}: no extra task joined a fragmented payload"
            );
        }
    }
}

#[test]
fn parallel_request_on_the_global_pool_matches_serial_bits() {
    for layers in [1, 6] {
        let bytes = rgb_fixture_with_layers(200, 136, false, layers);
        let region = whole_image(&bytes);
        let serial = execute_region(&bytes, region, false);
        let parallel = execute_region(&bytes, region, true);
        assert_eq!(serial.planes, parallel.planes, "layers={layers}");
        assert_eq!(
            serial.task_payloads,
            Vec::<(usize, usize)>::new(),
            "layers={layers}"
        );
        // However many workers the global pool has, each extra task keeps
        // one prepared workspace.
        assert_eq!(
            parallel.workspace_bytes,
            serial.workspace_bytes * (parallel.task_payloads.len() + 1),
            "layers={layers}"
        );
    }
}
