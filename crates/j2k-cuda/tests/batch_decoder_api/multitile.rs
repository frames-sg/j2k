// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{prepare_batch, BatchDecodeOptions, BatchLayout, EncodedImage};
use j2k_cuda::{CudaBatchDecoder, CudaSession, Surface};
use j2k_cuda_runtime::CudaContext;

use super::support::submit_external_for_test;

#[test]
fn independent_openjph_multitile_rgb_retains_every_prepared_tile_without_cuda() {
    let encoded = openjph_multitile_rgb_u12();
    let prepared = prepare_batch(
        vec![EncodedImage::full(encoded)],
        BatchDecodeOptions::default(),
    )
    .expect("prepare independent OpenJPH RGB fixture");
    let [group] = prepared.groups() else {
        panic!("expected one OpenJPH RGB group")
    };
    let native = group.images()[0]
        .htj2k_plan()
        .expect("prepared multi-tile HTJ2K plan")
        .adapter_view()
        .downcast_ref::<j2k_native::J2kReferencedHtj2kPlan>()
        .expect("native referenced HTJ2K adapter");
    assert_eq!(native.tiles().len(), 4);
}

#[test]
fn independent_openjph_multitile_gray_and_rgb_are_resident_and_external_bit_exact_when_runtime_required(
) {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let fixtures = j2k_test_support::openjph_batch_fixtures();
    let rgb = fixtures
        .iter()
        .find(|fixture| fixture.name == "openjph-rgb-u12-53-raw")
        .expect("independent multi-tile RGB12 fixture");
    let gray = fixtures
        .iter()
        .find(|fixture| fixture.name == "openjph-gray-u12-53-raw")
        .expect("independent multi-tile Gray12 fixture");
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![
            EncodedImage::full(Arc::from(rgb.encoded)),
            EncodedImage::full(Arc::from(gray.encoded)),
        ],
        options,
    )
    .expect("prepare heterogeneous multi-tile CUDA batch");
    assert_eq!(prepared.groups().len(), 2);

    let context = CudaContext::system_default().expect("CUDA context");
    let session = CudaSession::with_context(context.clone());
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    for group in prepared.groups() {
        let expected = match group.source_indices() {
            [0] => rgb.oracle,
            [1] => gray.oracle,
            other => panic!("unexpected source indices: {other:?}"),
        };
        let mut allocation = context
            .allocate(expected.len())
            .expect("multi-tile external destination");
        // SAFETY: `allocation` remains live and inaccessible until `wait`.
        let submitted =
            unsafe { submit_external_for_test(&mut decoder, group, &context, &mut allocation) };
        let external = submitted.wait().expect("wait multi-tile external decode");
        assert_eq!(external.ranges().len(), 1);
        let mut actual = vec![0_u8; expected.len()];
        allocation
            .copy_to_host(&mut actual)
            .expect("download multi-tile external output");
        assert_eq!(actual, expected);
    }

    let output = decoder
        .decode_prepared(&prepared)
        .expect("decode multi-tile resident batch");
    assert!(output.group_errors().is_empty());
    assert_eq!(output.groups().len(), 2);
    for group in output.groups() {
        let (expected, is_color) = match group.source_indices() {
            [0] => (rgb.oracle, true),
            [1] => (gray.oracle, false),
            other => panic!("unexpected resident source indices: {other:?}"),
        };
        let dense = group.dense_output();
        assert_eq!(dense.ranges().len(), 1);
        let mut actual = vec![0_u8; expected.len()];
        dense
            .buffer()
            .copy_range_to_host(dense.ranges()[0].offset, &mut actual)
            .expect("download multi-tile resident output");
        assert_eq!(actual, expected);
        if !is_color {
            assert_eq!(group.surfaces().len(), 1);
            let surface_actual = Surface::download_batch_tight(group.surfaces())
                .expect("download multi-tile resident grayscale output");
            assert_eq!(surface_actual, expected);
        }
    }
}

fn openjph_multitile_rgb_u12() -> Arc<[u8]> {
    Arc::from(
        j2k_test_support::openjph_batch_fixtures()
            .iter()
            .find(|fixture| fixture.name == "openjph-rgb-u12-53-raw")
            .expect("checked-in OpenJPH RGB12 fixture")
            .encoded,
    )
}
