// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn distinct_classic_inputs() -> Vec<EncodedImage> {
    (0..8u8)
        .map(|seed| {
            let pixels = (0..128 * 128 * 3)
                .map(|index| {
                    u8::try_from((index * (usize::from(seed) + 3) + index / 128) & 255)
                        .expect("masked fixture sample fits u8")
                })
                .collect::<Vec<_>>();
            let encoded = encode(
                &pixels,
                128,
                128,
                3,
                8,
                false,
                &EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 3,
                    ..EncodeOptions::default()
                },
            )
            .expect("encode distinct classic RGB");
            EncodedImage::full(Arc::from(encoded))
        })
        .collect()
}

#[test]
fn cooperative_classic_batch_preserves_order_layout_and_reuse() {
    if !should_run_metal_runtime() {
        return;
    }
    let inputs = distinct_classic_inputs();
    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        let expected = CpuBatchDecoder::new(options)
            .decode(inputs.clone())
            .expect("CPU oracle");
        let CpuBatchSamples::U8(expected_bytes) = expected.groups()[0].samples() else {
            panic!("RGB8 oracle");
        };
        let mut decoder = MetalBatchDecoder::system_default_with_options(options).unwrap();
        let prepared = decoder.prepare(inputs.clone()).unwrap();
        for _ in 0..2 {
            let result = decoder
                .decode_prepared_cooperative(&prepared, std::num::NonZeroUsize::new(12).unwrap())
                .expect("cooperative decode");
            assert_eq!(result.errors(), []);
            assert!(
                result.group_errors().is_empty(),
                "{:?}",
                result.group_errors()
            );
            assert_eq!(result.groups().len(), 1);
            let group = &result.groups()[0];
            assert_eq!(group.source_indices(), &[0, 1, 2, 3, 4, 5, 6, 7]);
            assert!(group.dispatch_report().cpu_tier1_images > 0);
            assert!(group.dispatch_report().classic_tier1 > 0);
            assert_eq!(&completed_resident_batch_bytes(group), expected_bytes);
        }
        let strict = decoder
            .decode_prepared(&prepared)
            .expect("strict GPU route remains reusable");
        assert_eq!(strict.groups()[0].dispatch_report().cpu_tier1_images, 0);
        for workers in [1, 4, 8] {
            let limited = decoder
                .decode_prepared_cooperative(
                    &prepared,
                    std::num::NonZeroUsize::new(workers).unwrap(),
                )
                .expect("partial CPU budget");
            assert_eq!(limited.errors(), []);
            assert!(limited.group_errors().is_empty());
            if workers == 1 {
                assert_eq!(limited.groups()[0].dispatch_report().cpu_tier1_images, 0);
            }
            assert_eq!(
                &completed_resident_batch_bytes(&limited.groups()[0]),
                expected_bytes
            );
        }
    }
}

#[test]
fn cooperative_mixed_batch_retains_indexed_and_group_errors() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut inputs = distinct_classic_inputs();
    inputs.insert(1, EncodedImage::full(Arc::from(fixture_gray12())));
    inputs.insert(
        3,
        EncodedImage::full(Arc::from(fixture_ht_u8_unsupported_direct_width(3))),
    );
    inputs.insert(5, EncodedImage::full(Arc::from(&b"invalid codestream"[..])));
    inputs.push(EncodedImage::full(Arc::from(fixture_rgb12())));
    let mut decoder = MetalBatchDecoder::system_default().unwrap();
    let prepared = decoder.prepare(inputs).unwrap();
    let strict = decoder.decode_prepared(&prepared).unwrap();
    let cooperative = decoder
        .decode_prepared_cooperative(&prepared, std::num::NonZeroUsize::new(12).unwrap())
        .unwrap();
    assert_eq!(cooperative.errors(), strict.errors());
    assert_eq!(cooperative.errors().len(), 1);
    assert_eq!(cooperative.group_errors().len(), 1);
    assert_eq!(cooperative.group_errors()[0].source_indices(), &[3]);
    assert_eq!(cooperative.groups().len(), 3);
    assert!(cooperative
        .groups()
        .iter()
        .any(|group| group.dispatch_report().cpu_tier1_images > 0));
    for (actual, expected) in cooperative.groups().iter().zip(strict.groups()) {
        assert_eq!(actual.source_indices(), expected.source_indices());
        assert_eq!(actual.decoded_rects(), expected.decoded_rects());
        assert_eq!(actual.warnings(), expected.warnings());
        assert_eq!(
            completed_resident_batch_bytes(actual),
            completed_resident_batch_bytes(expected)
        );
        if actual.info().precision > 8 {
            assert_eq!(actual.dispatch_report().cpu_tier1_images, 0);
        }
    }
    decoder
        .decode_prepared_cooperative(&prepared, std::num::NonZeroUsize::new(12).unwrap())
        .expect("session remains usable after nonfatal group error");
}

#[test]
fn cooperative_small_and_repeated_batches_keep_gpu_tier1() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut decoder = MetalBatchDecoder::system_default().unwrap();
    let repeated = EncodedImage::full(Arc::from(fixture_rgb8_sized(128, 128)));
    for inputs in [
        vec![repeated.clone(); 8],
        vec![repeated],
        (0..8)
            .map(|_| EncodedImage::full(Arc::from(fixture_rgb8())))
            .collect(),
    ] {
        let prepared = decoder.prepare(inputs).unwrap();
        let strict = decoder.decode_prepared(&prepared).unwrap();
        let result = decoder
            .decode_prepared_cooperative(&prepared, std::num::NonZeroUsize::new(12).unwrap())
            .unwrap();
        assert!(result.group_errors().is_empty());
        assert_eq!(result.groups()[0].dispatch_report().cpu_tier1_images, 0);
        assert_eq!(
            completed_resident_batch_bytes(&result.groups()[0]),
            completed_resident_batch_bytes(&strict.groups()[0])
        );
    }
}
