// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::{
    encode, execute_referenced_classic_plan, DecodeSettings, DecoderContext, EncodeOptions, Image,
    J2kDirectCpuScratch,
};

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

/// Every reconstructed plane as raw bits, plus the retained classic
/// workspace bytes after the execution.
fn execute_whole_plan(bytes: &[u8], parallel: bool) -> (Vec<Vec<u32>>, usize) {
    let image = Image::new(bytes, &DecodeSettings::default()).expect("classic image");
    let region = (0, 0, image.width(), image.height());
    let plan = image
        .build_referenced_classic_plan_region_with_context(&mut DecoderContext::default(), region)
        .expect("classic whole-image plan");
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
    (planes, scratch.retained_classic_workspace_bytes())
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
        let (serial, serial_workspace_bytes) = execute_whole_plan(&bytes, false);
        let (parallel, parallel_workspace_bytes) =
            pool.install(|| execute_whole_plan(&bytes, true));
        assert_eq!(serial, parallel, "reversible={reversible}");
        // The tallest sub-band (130 rows) holds three rows of 64-high
        // blocks, so of the four workers only two extra ones can take a row
        // and keep their own prepared workspace.
        assert_eq!(
            parallel_workspace_bytes,
            serial_workspace_bytes * 3,
            "reversible={reversible}"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn fragmented_payloads_decode_serially_without_extra_workspaces() {
    let bytes = rgb_fixture_with_layers(300, 260, false, 4);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("classic image");
    let plan = image
        .build_referenced_classic_plan_region_with_context(
            &mut DecoderContext::default(),
            (0, 0, image.width(), image.height()),
        )
        .expect("classic whole-image plan");
    assert!(
        plan.payloads()
            .iter()
            .any(|payload| payload.range_count > 1),
        "four quality layers split code-block payloads over fragments"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("four-thread pool");
    let (serial, serial_workspace_bytes) = execute_whole_plan(&bytes, false);
    let (parallel, parallel_workspace_bytes) = pool.install(|| execute_whole_plan(&bytes, true));
    assert_eq!(serial, parallel);
    assert_eq!(parallel_workspace_bytes, serial_workspace_bytes);
}

#[test]
fn parallel_request_on_the_global_pool_matches_serial_bits() {
    let bytes = rgb_fixture(200, 136, false);
    let (serial, _) = execute_whole_plan(&bytes, false);
    let (parallel, _) = execute_whole_plan(&bytes, true);
    assert_eq!(serial, parallel);
}
