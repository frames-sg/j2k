// SPDX-License-Identifier: MIT OR Apache-2.0

//! Irreversible (9/7) RGB without a multi-component transform must match the
//! CPU sample for sample. The CPU rounds each centered sample ties-to-even and
//! only then adds the unsigned level shift; the exact-native CUDA store once
//! derived that rounding from `mct`, so 9/7 without MCT rounded half away from
//! zero after the shift (centered -9.5 stored 119 where the CPU stores 118).
//!
//! Without decomposition levels and with a unit LL step (scale 4 against the
//! two guard bits), every non-zero coefficient reconstructs to `q ± 0.5`, so
//! each sample is a tie that distinguishes the two rounding orders.

use std::sync::Arc;

use j2k::{
    prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, Downscale, EncodedImage, NativeSampleType,
};
use j2k_cuda::{CudaBatchDecoder, CudaSession};
use j2k_cuda_runtime::CudaContext;

use super::support::submit_external_for_test;

const SIZE: u32 = 64;

#[derive(Clone, Copy, Debug)]
struct Fixture {
    bit_depth: u8,
    signed: bool,
    sample_type: NativeSampleType,
}

const FIXTURES: [Fixture; 4] = [
    Fixture {
        bit_depth: 8,
        signed: false,
        sample_type: NativeSampleType::U8,
    },
    Fixture {
        bit_depth: 16,
        signed: false,
        sample_type: NativeSampleType::U16,
    },
    Fixture {
        bit_depth: 8,
        signed: true,
        sample_type: NativeSampleType::I16,
    },
    Fixture {
        bit_depth: 16,
        signed: true,
        sample_type: NativeSampleType::I16,
    },
];

#[expect(
    clippy::cast_possible_truncation,
    reason = "samples are masked to the fixture precision before narrowing"
)]
fn fixture_bytes(fixture: Fixture) -> Vec<u8> {
    let mut state = 0x9e37_79b9_u32;
    let span = 1_u32 << fixture.bit_depth;
    let samples = (0..SIZE * SIZE * 3).map(move |index| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let (pixel, component) = (index / 3, index % 3);
        let (x, y) = (pixel % SIZE, pixel / SIZE);
        let ramp = (x * 7 + y * 11 + component * 29) * (span / 256).max(1);
        (ramp + state % (span / 8)) & (span - 1)
    });
    // Signed fixtures store the same codes as two's complement around zero.
    let centered = move |code: u32| code.wrapping_sub(span / 2);
    match (fixture.bit_depth, fixture.signed) {
        (8, false) => samples.map(|code| code as u8).collect(),
        (8, true) => samples.map(|code| centered(code) as u8).collect(),
        (_, false) => samples
            .flat_map(|code| (code as u16).to_ne_bytes())
            .collect(),
        (_, true) => samples
            .flat_map(|code| (centered(code) as u16).to_ne_bytes())
            .collect(),
    }
}

fn encode_fixture(fixture: Fixture, levels: u8) -> Arc<[u8]> {
    Arc::from(
        j2k_native::encode_htj2k(
            &fixture_bytes(fixture),
            SIZE,
            SIZE,
            3,
            fixture.bit_depth,
            fixture.signed,
            &j2k_native::EncodeOptions {
                reversible: false,
                use_mct: false,
                num_decomposition_levels: levels,
                irreversible_quantization_scale: 4.0,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("{fixture:?}: encode 9/7 RGB without MCT: {error}")),
    )
}

/// Centered CPU samples whose half-away rounding after the level shift
/// differs from ties-to-even rounding before it: the inputs that detect the
/// rounding order this test pins.
#[expect(
    clippy::float_cmp,
    reason = "rounding agreement is an exact property of the f32 values"
)]
fn order_sensitive_samples(bytes: &[u8], fixture: Fixture) -> usize {
    let image = j2k_native::Image::new(bytes, &j2k_native::DecodeSettings::default())
        .expect("inspect 9/7 fixture");
    let mut context = j2k_native::DecoderContext::default();
    let components = image
        .decode_components_with_context(&mut context)
        .expect("CPU float component decode");
    let shift = if fixture.signed {
        0.0
    } else {
        f32::from(1_u16 << (fixture.bit_depth - 1))
    };
    components
        .planes()
        .iter()
        .flat_map(j2k_native::ComponentPlane::samples)
        .filter(|&&shifted| {
            let centered = shifted - shift;
            centered.round_ties_even() + shift != (centered + shift).round()
        })
        .count()
}

fn cpu_bytes(samples: &CpuBatchSamples) -> Vec<u8> {
    match samples {
        CpuBatchSamples::U8(samples) => samples.clone(),
        CpuBatchSamples::U16(samples) => samples
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect(),
        CpuBatchSamples::I16(samples) => samples
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect(),
        other => panic!("unexpected CPU oracle samples: {other:?}"),
    }
}

fn first_mismatch(actual: &[u8], expected: &[u8]) -> Option<(usize, u8, u8)> {
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
        .map(|(index, (&actual, &expected))| (index, actual, expected))
}

#[test]
fn irreversible_rgb_without_mct_matches_cpu_exactly_for_u8_u16_and_i16_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let mut failures = Vec::new();
    for (fixture, levels) in FIXTURES
        .into_iter()
        .flat_map(|fixture| [(fixture, 0_u8), (fixture, 2)])
    {
        let encoded = encode_fixture(fixture, levels);
        let requests: &[DecodeRequest] = if levels == 0 {
            assert!(
                order_sensitive_samples(&encoded, fixture) > 0,
                "{fixture:?}: no sample distinguishes the rounding order"
            );
            &[DecodeRequest::Full]
        } else {
            &[
                DecodeRequest::Full,
                DecodeRequest::Reduced {
                    scale: Downscale::Half,
                },
            ]
        };
        for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
            let mut options = BatchDecodeOptions::default();
            options.layout = layout;
            let mut cpu = CpuBatchDecoder::new(options);
            let session = CudaSession::with_context(context.clone());
            let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
            for &request in requests {
                let label = format!("{fixture:?} levels={levels} {layout:?} {request:?}");
                let prepared = prepare_batch(
                    vec![EncodedImage::new(Arc::clone(&encoded), request)],
                    options,
                )
                .unwrap_or_else(|error| panic!("{label}: prepare: {error}"));
                assert!(prepared.errors().is_empty(), "{label}");
                let [group] = prepared.groups() else {
                    panic!("{label}: expected one group")
                };
                assert_eq!(group.info().sample_type, fixture.sample_type, "{label}");
                assert!(group.images()[0].htj2k_plan().is_some(), "{label}");
                let oracle = cpu
                    .decode_prepared(&prepared)
                    .unwrap_or_else(|error| panic!("{label}: CPU oracle: {error}"));
                let expected = cpu_bytes(oracle.groups()[0].samples());

                let mut allocation = context
                    .allocate(expected.len())
                    .expect("external 9/7 destination");
                // SAFETY: the allocation stays live and unread until the
                // returned completion owner is waited below.
                let submitted = unsafe {
                    submit_external_for_test(&mut decoder, group, &context, &mut allocation)
                };
                submitted
                    .wait()
                    .unwrap_or_else(|error| panic!("{label}: wait external: {error}"));
                let mut external = vec![0_u8; expected.len()];
                allocation
                    .copy_to_host(&mut external)
                    .expect("download external 9/7 output");
                if let Some(mismatch) = first_mismatch(&external, &expected) {
                    failures.push(format!("{label} external: (byte, CUDA, CPU) {mismatch:?}"));
                }

                let resident = decoder
                    .decode_prepared(&prepared)
                    .unwrap_or_else(|error| panic!("{label}: resident: {error}"));
                let [resident_group] = resident.groups() else {
                    panic!("{label}: expected one resident group")
                };
                let dense = resident_group.dense_output();
                let mut resident_bytes = vec![0_u8; expected.len()];
                dense
                    .buffer()
                    .copy_range_to_host(dense.ranges()[0].offset, &mut resident_bytes)
                    .expect("download resident 9/7 output");
                if let Some(mismatch) = first_mismatch(&resident_bytes, &expected) {
                    failures.push(format!("{label} resident: (byte, CUDA, CPU) {mismatch:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
