// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    prepare_batch, BatchColor, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, EncodedImage, NativeSampleType, Rect,
};
use j2k_core::Downscale;
use j2k_cuda::{CudaBatchDecoder, CudaBatchGroup, CudaSession, Surface};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};

#[test]
fn prepared_classic_gray_and_rgb_native_batches_match_cpu_for_all_requests_and_layouts() {
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
    for channels in [1, 3] {
        for encoded in classic_fixtures(channels) {
            for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
                for request in requests {
                    assert_classic_case(&context, &encoded, layout, request);
                }
            }
        }
    }
}

#[test]
fn prepared_classic_multitile_gray_and_rgb_are_resident_and_external_bit_exact() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let requests = [
        DecodeRequest::Full,
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
    for channels in [1, 3] {
        let encoded = encode_classic_multitile(channels);
        let prepared = prepare_batch(
            vec![EncodedImage::full(Arc::clone(&encoded))],
            BatchDecodeOptions::default(),
        )
        .expect("prepare classic multi-tile fixture");
        let native = prepared.groups()[0].images()[0]
            .classic_plan()
            .expect("prepared classic plan")
            .adapter_view()
            .downcast_ref::<j2k_native::J2kReferencedClassicPlan>()
            .expect("native referenced classic adapter");
        assert!(native.tiles().len() > 1);

        for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
            for request in requests {
                assert_classic_case(&context, &encoded, layout, request);
            }
        }
    }
}

#[test]
fn classic_irreversible_gray_and_rgb_match_cpu_within_one_lsb_for_all_requests_and_layouts() {
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
    for channels in [1, 3] {
        let encoded = classic_irreversible_fixture(channels);
        for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
            for request in requests {
                assert_classic_irreversible_case(&context, &encoded, layout, request);
            }
        }
    }
}

fn classic_fixtures(channels: u16) -> [Arc<[u8]>; 3] {
    let sample_count = 16 * 16 * channels as usize;
    let u8_samples = (0..sample_count)
        .map(|index| u8::try_from((index * 29 + 7) & 0xff).expect("masked sample fits u8"))
        .collect::<Vec<_>>();
    let u16_samples = (0..sample_count)
        .map(|index| u16::try_from((index * 977 + 31) & 0x0fff).expect("masked sample fits u16"))
        .collect::<Vec<_>>();
    let i16_samples = (0..sample_count)
        .map(|index| {
            let index = i32::try_from(index).expect("fixture index fits i32");
            i16::try_from((index * 811 + 19) % 65_536 - 32_768).expect("bounded sample fits i16")
        })
        .collect::<Vec<_>>();
    [
        encode_classic(&u8_samples, channels, 8, false),
        encode_classic(native_bytes(&u16_samples), channels, 12, false),
        encode_classic(native_bytes(&i16_samples), channels, 16, true),
    ]
}

fn encode_classic(
    samples: impl AsRef<[u8]>,
    channels: u16,
    bit_depth: u8,
    signed: bool,
) -> Arc<[u8]> {
    Arc::from(
        j2k_native::encode(
            samples.as_ref(),
            16,
            16,
            channels,
            bit_depth,
            signed,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                use_mct: channels == 3,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode classic native fixture"),
    )
}

fn encode_classic_multitile(channels: u16) -> Arc<[u8]> {
    let sample_count = 16 * 16 * channels as usize;
    let samples = (0..sample_count)
        .map(|index| u16::try_from((index * 977 + 31) & 0x0fff).expect("masked sample fits u16"))
        .collect::<Vec<_>>();
    Arc::from(
        j2k_native::encode(
            native_bytes(&samples).as_slice(),
            16,
            16,
            channels,
            12,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                use_mct: channels == 3,
                tile_size: Some((9, 7)),
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode classic multi-tile fixture"),
    )
}

fn classic_irreversible_fixture(channels: u16) -> Arc<[u8]> {
    let sample_count = 16 * 16 * channels as usize;
    let samples = (0..sample_count)
        .map(|index| {
            u8::try_from((index * 47 + index / 7 + 13) & 0xff).expect("masked sample fits u8")
        })
        .collect::<Vec<_>>();
    Arc::from(
        j2k_native::encode(
            &samples,
            16,
            16,
            channels,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: false,
                num_decomposition_levels: 2,
                use_mct: channels == 3,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode classic irreversible fixture"),
    )
}

fn native_bytes<T: Copy>(samples: &[T]) -> Vec<u8> {
    // SAFETY: integers have no invalid bit patterns and the returned bytes are copied.
    unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
    .to_vec()
}

fn assert_classic_case(
    context: &CudaContext,
    encoded: &Arc<[u8]>,
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
    .expect("prepare classic native fixture");
    assert!(prepared.errors().is_empty());
    let [group] = prepared.groups() else {
        panic!("expected one classic native group")
    };
    assert!(group.images()[0].classic_plan().is_some());
    assert!(group.images()[0].htj2k_plan().is_none());

    let mut cpu = CpuBatchDecoder::new(options);
    let oracle = cpu
        .decode_prepared(&prepared)
        .expect("CPU classic native oracle");
    let expected = samples_as_bytes(oracle.groups()[0].samples());
    let alignment = match group.info().sample_type {
        NativeSampleType::U8 => 1,
        NativeSampleType::U16 | NativeSampleType::I16 => 2,
        _ => panic!("unsupported native sample type"),
    };

    let session = CudaSession::with_context(context.clone());
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    let mut allocation = context
        .allocate(expected.len())
        .expect("classic external destination");
    let submitted = {
        let ptr = allocation.device_ptr();
        let len = allocation.byte_len();
        // SAFETY: destination is fresh, exclusive, and retained until wait.
        let mut destination = unsafe {
            CudaExternalDeviceBufferViewMut::from_raw_parts(
                context,
                ptr,
                len,
                alignment,
                &mut allocation,
            )
        }
        .expect("classic external view");
        // SAFETY: allocation is not accessed while submitted work is pending.
        unsafe { decoder.submit_batch_into(group, &mut destination) }
            .expect("submit classic external batch")
    };
    submitted.wait().expect("wait classic external batch");
    let mut external = vec![0_u8; expected.len()];
    allocation
        .copy_to_host(&mut external)
        .expect("download classic external output");
    assert_eq!(external, expected, "external {layout:?} {request:?}");

    let resident = decoder
        .decode_prepared(&prepared)
        .expect("decode classic resident batch");
    assert!(resident.group_errors().is_empty());
    let actual = download_resident_bytes(&resident.groups()[0], expected.len());
    assert_eq!(actual, expected, "resident {layout:?} {request:?}");
}

fn assert_classic_irreversible_case(
    context: &CudaContext,
    encoded: &Arc<[u8]>,
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
    .expect("prepare classic irreversible fixture");
    assert!(prepared.errors().is_empty());
    let [group] = prepared.groups() else {
        panic!("expected one classic irreversible group")
    };
    assert!(group.images()[0].classic_plan().is_some());

    let mut cpu = CpuBatchDecoder::new(options);
    let oracle = cpu
        .decode_prepared(&prepared)
        .expect("CPU classic irreversible oracle");
    let CpuBatchSamples::U8(expected) = oracle.groups()[0].samples() else {
        panic!("classic irreversible fixture must use U8 storage")
    };
    let session = CudaSession::with_context(context.clone());
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    let mut allocation = context
        .allocate(expected.len())
        .expect("classic irreversible external destination");
    let submitted = {
        let ptr = allocation.device_ptr();
        let len = allocation.byte_len();
        // SAFETY: destination is fresh, exclusive, and retained until wait.
        let mut destination = unsafe {
            CudaExternalDeviceBufferViewMut::from_raw_parts(context, ptr, len, 1, &mut allocation)
        }
        .expect("classic irreversible external view");
        // SAFETY: allocation is not accessed while submitted work is pending.
        unsafe { decoder.submit_batch_into(group, &mut destination) }
            .expect("submit classic irreversible external batch")
    };
    submitted
        .wait()
        .expect("wait classic irreversible external batch");
    let mut external = vec![0_u8; expected.len()];
    allocation
        .copy_to_host(&mut external)
        .expect("download classic irreversible external output");
    assert_within_one_lsb(&external, expected, "external", layout, request);

    let resident = decoder
        .decode_prepared(&prepared)
        .expect("decode classic irreversible resident batch");
    let actual = download_resident_bytes(&resident.groups()[0], expected.len());
    assert_within_one_lsb(&actual, expected, "resident", layout, request);
}

fn download_resident_bytes(group: &CudaBatchGroup, expected_len: usize) -> Vec<u8> {
    if group.info().color == BatchColor::Gray {
        return Surface::download_batch_tight(group.surfaces())
            .expect("download classic grayscale resident surfaces");
    }
    let dense = group.dense_output();
    let mut actual = vec![0_u8; expected_len];
    dense
        .buffer()
        .copy_range_to_host(dense.ranges()[0].offset, &mut actual)
        .expect("download classic color resident output");
    actual
}

fn assert_within_one_lsb(
    actual: &[u8],
    expected: &[u8],
    route: &str,
    layout: BatchLayout,
    request: DecodeRequest,
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.abs_diff(expected) <= 1,
            "{route} {layout:?} {request:?} sample {index}: actual={actual}, expected={expected}"
        );
    }
}

fn samples_as_bytes(samples: &CpuBatchSamples) -> Vec<u8> {
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
        other => panic!("unsupported classic oracle type: {other:?}"),
    }
}
