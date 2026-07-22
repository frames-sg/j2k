// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, Downscale, EncodedImage, Rect,
};
use j2k_cuda::{CudaBatchDecoder, CudaSession, Surface};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};

#[test]
fn exact_rgb_external_and_resident_batches_match_cpu_for_geometry_and_layout_when_runtime_required()
{
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region {
            roi: Rect {
                x: 2,
                y: 3,
                w: 9,
                h: 7,
            },
        },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi: Rect {
                x: 2,
                y: 4,
                w: 10,
                h: 8,
            },
            scale: Downscale::Half,
        },
    ];
    let context = CudaContext::system_default().expect("CUDA context");

    for (encoded, expected_u16) in exact_rgb_fixtures() {
        for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
            for request in requests {
                assert_exact_rgb_case(&context, &encoded, expected_u16, layout, request);
            }
        }
    }
}

fn exact_rgb_fixtures() -> [(Arc<[u8]>, bool); 2] {
    let encoded_u8 = Arc::<[u8]>::from(
        j2k_native::encode_htj2k(
            &(0_u32..16 * 16 * 3)
                .map(|value| ((value * 29 + 7) & 0x7f) as u8)
                .collect::<Vec<_>>(),
            16,
            16,
            3,
            7,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode RGB7 HTJ2K"),
    );
    let samples_u16 = (0_u32..16 * 16 * 3)
        .map(|value| ((value * 977 + 31) & 0x0fff) as u16)
        .collect::<Vec<_>>();
    // SAFETY: the encoder copies this native integer fixture immediately.
    let samples_u16_bytes = unsafe {
        std::slice::from_raw_parts(
            samples_u16.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples_u16.as_slice()),
        )
    };
    let encoded_u16 = Arc::<[u8]>::from(
        j2k_native::encode_htj2k(
            samples_u16_bytes,
            16,
            16,
            3,
            12,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode RGB12 HTJ2K"),
    );
    [(encoded_u8, false), (encoded_u16, true)]
}

fn assert_exact_rgb_case(
    context: &CudaContext,
    encoded: &Arc<[u8]>,
    expected_u16: bool,
    layout: BatchLayout,
    request: DecodeRequest,
) {
    let options = BatchDecodeOptions {
        layout,
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![EncodedImage::new(Arc::clone(encoded), request)],
        options,
    )
    .expect("prepare exact RGB group");
    assert!(prepared.errors().is_empty());
    let [prepared_group] = prepared.groups() else {
        panic!("expected one exact RGB group")
    };
    assert!(prepared_group.images()[0].htj2k_plan().is_some());

    let mut cpu = CpuBatchDecoder::new(options);
    let oracle = cpu
        .decode_prepared(&prepared)
        .expect("CPU exact RGB oracle");
    let expected = match oracle.groups()[0].samples() {
        CpuBatchSamples::U8(samples) => samples.clone(),
        CpuBatchSamples::U16(samples) => samples
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect(),
        other => panic!("unexpected exact RGB oracle type: {other:?}"),
    };
    assert_eq!(
        expected_u16,
        prepared_group.info().sample_type == j2k::NativeSampleType::U16
    );

    let session = CudaSession::with_context(context.clone());
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    assert_exact_external(
        context,
        &mut decoder,
        prepared_group,
        &expected,
        expected_u16,
        layout,
        request,
    );
    assert_exact_resident(&mut decoder, &prepared, &expected, layout, request);
}

fn assert_exact_external(
    context: &CudaContext,
    decoder: &mut CudaBatchDecoder,
    group: &j2k::PreparedBatchGroup,
    expected: &[u8],
    expected_u16: bool,
    layout: BatchLayout,
    request: DecodeRequest,
) {
    let mut allocation = context
        .allocate(expected.len())
        .expect("external exact RGB destination");
    let ptr = allocation.device_ptr();
    let len = allocation.byte_len();
    // SAFETY: the allocation remains live and exclusively owned until the
    // returned completion owner is waited below.
    let submitted = unsafe {
        let mut destination = CudaExternalDeviceBufferViewMut::from_raw_parts(
            context,
            ptr,
            len,
            if expected_u16 { 2 } else { 1 },
            &mut allocation,
        )
        .expect("exact RGB external view");
        decoder
            .submit_batch_into(group, &mut destination)
            .expect("submit exact RGB external batch")
    };
    let external = submitted.wait().expect("wait exact RGB external batch");
    assert_eq!(external.ranges().len(), 1);
    let mut actual = vec![0_u8; expected.len()];
    allocation
        .copy_to_host(&mut actual)
        .expect("download exact RGB external output");
    assert_eq!(actual, expected, "{layout:?} {request:?}");
}

fn assert_exact_resident(
    decoder: &mut CudaBatchDecoder,
    prepared: &j2k::PreparedBatch,
    expected: &[u8],
    layout: BatchLayout,
    request: DecodeRequest,
) {
    let resident = decoder
        .decode_prepared(prepared)
        .expect("decode exact RGB resident batch");
    let [resident_group] = resident.groups() else {
        panic!("expected one resident RGB group")
    };
    let dense = resident_group.dense_output();
    assert_eq!(dense.ranges().len(), 1);
    let mut dense_bytes = vec![0_u8; expected.len()];
    dense
        .buffer()
        .copy_range_to_host(dense.ranges()[0].offset, &mut dense_bytes)
        .expect("download dense exact RGB output");
    assert_eq!(dense_bytes, expected, "dense {layout:?} {request:?}");
    if layout == BatchLayout::Nhwc {
        assert_eq!(resident_group.surfaces().len(), 1);
        assert_eq!(
            Surface::download_batch_tight(resident_group.surfaces())
                .expect("download NHWC surface"),
            expected
        );
    } else {
        assert!(resident_group.surfaces().is_empty());
    }
}

#[test]
fn irreversible_rgb8_external_and_resident_outputs_match_cpu_within_one_lsb_when_runtime_required()
{
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let pixels = (0_u32..16 * 16 * 3)
        .map(|value| ((value * 47 + value / 7 + 13) & 0xff) as u8)
        .collect::<Vec<_>>();
    let encoded = Arc::<[u8]>::from(
        j2k_native::encode_htj2k(
            &pixels,
            16,
            16,
            3,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: false,
                num_decomposition_levels: 2,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode irreversible RGB8 HTJ2K fixture"),
    );
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(vec![EncodedImage::full(encoded)], options)
        .expect("prepare irreversible RGB8 CUDA group");
    let [group] = prepared.groups() else {
        panic!("expected one irreversible RGB8 group")
    };
    assert!(group.images()[0].htj2k_plan().is_some());

    let mut cpu = CpuBatchDecoder::new(options);
    let oracle = cpu
        .decode_prepared(&prepared)
        .expect("decode irreversible RGB8 CPU oracle");
    let expected = match oracle.groups()[0].samples() {
        CpuBatchSamples::U8(samples) => samples,
        other => panic!("expected U8 CPU oracle, got {other:?}"),
    };

    let context = CudaContext::system_default().expect("CUDA context");
    let session = CudaSession::with_context(context.clone());
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    let mut allocation = context
        .allocate(expected.len())
        .expect("irreversible external destination");
    let ptr = allocation.device_ptr();
    let len = allocation.byte_len();
    // SAFETY: the allocation remains live and inaccessible until completion.
    let submitted = unsafe {
        let mut destination = CudaExternalDeviceBufferViewMut::from_raw_parts(
            &context,
            ptr,
            len,
            std::mem::align_of::<u8>(),
            &mut allocation,
        )
        .expect("irreversible external view");
        decoder
            .submit_batch_into(group, &mut destination)
            .expect("submit irreversible RGB8 external batch")
    };
    submitted
        .wait()
        .expect("wait irreversible RGB8 external batch");
    let mut external = vec![0_u8; expected.len()];
    allocation
        .copy_to_host(&mut external)
        .expect("download irreversible RGB8 external output");
    assert_u8_within_one_lsb(&external, expected, "external");

    let resident = decoder
        .decode_prepared(&prepared)
        .expect("decode irreversible RGB8 resident batch");
    let [resident_group] = resident.groups() else {
        panic!("expected one irreversible resident group")
    };
    let dense = resident_group.dense_output();
    let mut resident_bytes = vec![0_u8; expected.len()];
    dense
        .buffer()
        .copy_range_to_host(dense.ranges()[0].offset, &mut resident_bytes)
        .expect("download irreversible RGB8 resident output");
    assert_u8_within_one_lsb(&resident_bytes, expected, "resident");
}

fn assert_u8_within_one_lsb(actual: &[u8], expected: &[u8], route: &str) {
    assert_eq!(actual.len(), expected.len(), "{route} output length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.abs_diff(expected) <= 1,
            "{route} sample {index}: actual={actual}, expected={expected}"
        );
    }
}
