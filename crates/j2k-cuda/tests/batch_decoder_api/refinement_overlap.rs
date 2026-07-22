// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{prepare_batch, BatchDecodeOptions, BatchLayout, EncodedImage};
use j2k_cuda::{CudaBatchDecoder, CudaSession};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};

#[test]
fn independent_sigprop_magref_overlap_matches_openhtj2k_for_resident_and_external_output() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let expected = j2k_test_support::openhtj2k_sigprop_overlap_pixels();
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(
            j2k_test_support::openhtj2k_sigprop_overlap_fixture(),
        ))],
        options,
    )
    .expect("prepare independent refinement-overlap fixture");
    let [group] = prepared.groups() else {
        panic!("expected one refinement-overlap group")
    };
    assert!(group.images()[0].htj2k_plan().is_some());

    let context = CudaContext::system_default().expect("CUDA context");
    let session = CudaSession::with_context(context.clone());
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    let mut allocation = context
        .allocate(expected.len())
        .expect("refinement external destination");
    let submitted = {
        let ptr = allocation.device_ptr();
        let len = allocation.byte_len();
        // SAFETY: destination is exclusive and retained through completion.
        let mut destination = unsafe {
            CudaExternalDeviceBufferViewMut::from_raw_parts(&context, ptr, len, 1, &mut allocation)
        }
        .expect("refinement external view");
        // SAFETY: allocation is not read until the pending owner is waited.
        unsafe { decoder.submit_batch_into(group, &mut destination) }
            .expect("submit refinement external decode")
    };
    submitted.wait().expect("wait refinement external decode");
    let mut external = vec![0_u8; expected.len()];
    allocation
        .copy_to_host(&mut external)
        .expect("download refinement external output");
    assert_within_one_lsb(&external, expected, "external");

    let resident = decoder
        .decode_prepared(&prepared)
        .expect("decode refinement resident output");
    let dense = resident.groups()[0].dense_output();
    let mut actual = vec![0_u8; expected.len()];
    dense
        .buffer()
        .copy_range_to_host(dense.ranges()[0].offset, &mut actual)
        .expect("download refinement resident output");
    assert_within_one_lsb(&actual, expected, "resident");
}

fn assert_within_one_lsb(actual: &[u8], expected: &[u8], route: &str) {
    assert_eq!(actual.len(), expected.len(), "{route} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.abs_diff(expected) <= 1,
            "{route} sample {index}: actual={actual}, expected={expected}"
        );
    }
}
