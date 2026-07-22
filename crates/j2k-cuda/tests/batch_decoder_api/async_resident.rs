// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{prepare_batch, BatchDecodeOptions, BatchLayout, EncodedImage};
use j2k_cuda::{CudaBatchDecoder, CudaSession, Surface};
use j2k_cuda_runtime::CudaContext;

#[test]
fn dropped_resident_submission_retires_work_and_preserves_session_reuse_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let fixture = j2k_test_support::openjph_batch_fixtures()
        .iter()
        .find(|fixture| fixture.name == "openjph-rgb-s12-53-single-raw")
        .expect("independent signed RGB12 fixture");
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(fixture.encoded))],
        options,
    )
    .expect("prepare independent signed RGB12 fixture");
    let context = CudaContext::system_default().expect("CUDA context");
    let session = CudaSession::with_context(context);
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);

    let dropped = decoder
        .submit_prepared(&prepared)
        .expect("submit first resident batch");
    assert_eq!(dropped.pending_group_count(), 1);
    drop(dropped);

    let completed = decoder
        .submit_prepared(&prepared)
        .expect("reuse session after dropped submission")
        .wait()
        .expect("wait reused resident submission");
    let [group] = completed.groups() else {
        panic!("expected one completed signed RGB group")
    };
    assert_eq!(group.source_indices(), [0]);
    let dense = group.dense_output();
    let mut actual = vec![0_u8; fixture.oracle.len()];
    dense
        .buffer()
        .copy_range_to_host(dense.ranges()[0].offset, &mut actual)
        .expect("download completed signed RGB output");
    assert_eq!(actual, fixture.oracle);
}

#[test]
fn asynchronous_multitile_rgb_and_grayscale_groups_complete_together_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let fixtures = j2k_test_support::openjph_batch_fixtures();
    let rgb = fixtures
        .iter()
        .find(|fixture| fixture.name == "openjph-rgb-u12-53-raw")
        .expect("independent multi-tile RGB fixture");
    let gray_pixels = (0_u8..64).collect::<Vec<_>>();
    let supported_gray = Arc::<[u8]>::from(
        j2k_native::encode_htj2k(
            &gray_pixels,
            8,
            8,
            1,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode supported grayscale fixture"),
    );
    // The independent OpenJPH oracle is stored in interleaved sample order.
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![
            EncodedImage::full(Arc::from(rgb.encoded)),
            EncodedImage::full(supported_gray),
        ],
        options,
    )
    .expect("prepare heterogeneous async batch");
    assert_eq!(prepared.groups().len(), 2);

    let context = CudaContext::system_default().expect("CUDA context");
    let session = CudaSession::with_context(context);
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    let submitted = decoder
        .submit_prepared(&prepared)
        .expect("submit heterogeneous async batch");
    assert_eq!(submitted.pending_group_count(), 2);

    let completed = submitted.wait().expect("wait heterogeneous async batch");
    assert!(completed.group_errors().is_empty());
    assert_eq!(completed.groups().len(), 2);
    let rgb_group = completed
        .groups()
        .iter()
        .find(|group| group.source_indices() == [0])
        .expect("completed multi-tile RGB group");
    let dense = rgb_group.dense_output();
    let mut actual_rgb = vec![0_u8; rgb.oracle.len()];
    dense
        .buffer()
        .copy_range_to_host(dense.ranges()[0].offset, &mut actual_rgb)
        .expect("download completed RGB output");
    assert_eq!(actual_rgb, rgb.oracle);
    let gray = completed
        .groups()
        .iter()
        .find(|group| group.source_indices() == [1])
        .expect("completed grayscale group");
    assert_eq!(
        Surface::download_batch_tight(gray.surfaces()).expect("download grayscale output"),
        gray_pixels
    );
}
