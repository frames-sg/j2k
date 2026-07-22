// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn direct_metal_burn_region_reduced_matches_cpu_oracle() {
    if !metal_runtime_gate("j2k-ml direct Metal ROI-reduced batch") {
        return;
    }
    let encoded = Arc::<[u8]>::from(openhtj2k_refinement_odd_fixture());
    let request = DecodeRequest::RegionReduced {
        roi: Rect {
            x: 3,
            y: 5,
            w: 9,
            h: 21,
        },
        scale: Downscale::Half,
    };
    let options = BatchDecodeOptions::default();
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode(vec![EncodedImage::new(encoded.clone(), request)])
        .expect("CPU ROI-reduced oracle");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("odd OpenHT fixture must decode to U8")
    };

    let mut decoder = MetalBurnDecoder::system_default(options).expect("paired Metal session");
    let output = decoder
        .decode(vec![EncodedImage::new(encoded, request)])
        .expect("direct Metal ROI-reduced decode");
    let BurnBatchTensor::U8(tensor) = output.groups.into_iter().next().unwrap().tensor else {
        panic!("expected native U8 Metal tensor")
    };
    let actual = tensor.into_data().into_vec::<u8>().expect("U8 data");
    assert_eq!(actual.as_slice(), expected.as_slice());
}

#[test]
fn direct_metal_odd_u8_tensor_tails_are_not_zero_initialized() {
    if !metal_runtime_gate("j2k-ml odd direct Metal tensor tails") {
        return;
    }
    let mut decoder = MetalBurnDecoder::system_default(BatchDecodeOptions::default())
        .expect("paired J2K/Burn Metal session");
    for (width, height) in [(5_u32, 1_u32), (11, 3)] {
        let pixels = (0..width * height)
            .map(|index| index.to_le_bytes()[0].wrapping_mul(17).wrapping_add(1))
            .collect::<Vec<_>>();
        let encoded = encode_gray8(&pixels, width, height);
        let output = decoder
            .decode(vec![EncodedImage::full(Arc::from(encoded))])
            .expect("direct odd-size Metal decode");
        let BurnBatchTensor::U8(tensor) = output.groups.into_iter().next().unwrap().tensor else {
            panic!("expected native U8 Metal tensor")
        };
        assert_eq!(
            tensor.into_data().into_vec::<u8>().expect("U8 data"),
            pixels
        );
    }
}

#[test]
fn direct_metal_burn_rgb_u8_region_reduced_is_exact_for_nhwc_and_nchw() {
    if !metal_runtime_gate("j2k-ml exact RGB U8 Metal batch") {
        return;
    }
    let first = Arc::<[u8]>::from(encode_rgb_u8(8, 8, 0));
    let second = Arc::<[u8]>::from(encode_rgb_u8(8, 8, 17));
    let request = DecodeRequest::RegionReduced {
        roi: Rect {
            x: 2,
            y: 2,
            w: 4,
            h: 4,
        },
        scale: Downscale::Half,
    };

    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        let inputs = vec![
            EncodedImage::new(first.clone(), request),
            EncodedImage::new(second.clone(), request),
        ];
        let mut cpu = CpuBatchDecoder::new(options);
        let expected = cpu.decode(inputs.clone()).expect("CPU RGB U8 oracle");
        let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
            panic!("six-bit RGB must use U8 batch storage")
        };

        let mut decoder = MetalBurnDecoder::system_default(options).expect("paired Metal session");
        let prepared = decoder
            .prepare(inputs)
            .expect("prepare reusable RGB U8 batch");
        for _ in 0..2 {
            let output = decoder
                .decode_prepared(&prepared)
                .expect("direct exact RGB U8 Burn decode");
            let group = output.groups.into_iter().next().expect("RGB U8 group");
            let BurnBatchTensor::U8(tensor) = group.tensor else {
                panic!("expected native RGB U8 Metal tensor")
            };
            assert_eq!(tensor.dtype(), DType::U8);
            assert_eq!(
                tensor.dims(),
                match layout {
                    BatchLayout::Nhwc => [2, 2, 2, 3],
                    BatchLayout::Nchw => [2, 3, 2, 2],
                    _ => unreachable!(),
                }
            );
            let actual = tensor.into_data().into_vec::<u8>().expect("RGB U8 data");
            assert_eq!(actual.as_slice(), expected.as_slice(), "{layout:?}");
            assert!(actual.iter().all(|sample| *sample <= 0x3f));
        }
    }
}

#[test]
fn direct_metal_burn_rgb_u16_is_exact_and_drop_safe() {
    if !metal_runtime_gate("j2k-ml exact RGB U16 Metal batch") {
        return;
    }
    let encoded = Arc::<[u8]>::from(encode_rgb_u16(8, 8, 257));
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let inputs = vec![EncodedImage::full(encoded.clone())];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU RGB U16 oracle");
    let CpuBatchSamples::U16(expected) = expected.groups()[0].samples() else {
        panic!("twelve-bit RGB must use U16 batch storage")
    };

    let mut decoder = MetalBurnDecoder::system_default(options).expect("paired Metal session");
    let prepared = decoder
        .prepare(inputs)
        .expect("prepare reusable RGB U16 batch");
    drop(
        decoder
            .submit_prepared(&prepared)
            .expect("submit disposable RGB U16 Burn batch"),
    );
    let output = decoder
        .decode_prepared(&prepared)
        .expect("reuse RGB U16 Burn decoder after pending drop");
    let BurnBatchTensor::U16(tensor) = output.groups.into_iter().next().unwrap().tensor else {
        panic!("expected native RGB U16 Metal tensor")
    };
    assert_eq!(tensor.dtype(), DType::U16);
    let actual = tensor.into_data().into_vec::<u16>().expect("RGB U16 data");
    assert_eq!(actual.as_slice(), expected.as_slice());
    assert!(actual.iter().all(|sample| *sample <= 0x0fff));
}

#[test]
fn direct_metal_burn_classic_rgb_u16_is_exact_for_both_layouts() {
    if !metal_runtime_gate("j2k-ml exact classic RGB U16 Metal batch") {
        return;
    }
    let encoded = Arc::<[u8]>::from(encode_classic_rgb_u16(8, 8, 257));
    for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        let inputs = vec![
            EncodedImage::full(encoded.clone()),
            EncodedImage::full(encoded.clone()),
        ];
        let mut cpu = CpuBatchDecoder::new(options);
        let expected = cpu.decode(inputs.clone()).expect("CPU classic RGB oracle");
        let CpuBatchSamples::U16(expected) = expected.groups()[0].samples() else {
            panic!("classic twelve-bit RGB must use U16 batch storage")
        };

        let mut decoder = MetalBurnDecoder::system_default(options).expect("paired Metal session");
        let prepared = decoder
            .prepare(inputs)
            .expect("prepare classic RGB Burn batch");
        let output = decoder
            .decode_prepared(&prepared)
            .expect("direct classic RGB Burn decode");
        assert!(output.errors.is_empty(), "{layout:?}");
        assert!(output.group_errors.is_empty(), "{layout:?}");
        let group = output.groups.into_iter().next().expect("classic RGB group");
        assert_eq!(group.source_indices, [0, 1]);
        let BurnBatchTensor::U16(tensor) = group.tensor else {
            panic!("expected native classic RGB U16 Metal tensor")
        };
        assert_eq!(tensor.dtype(), DType::U16);
        let actual = tensor.into_data().into_vec::<u16>().expect("RGB U16 data");
        assert_eq!(actual.as_slice(), expected.as_slice(), "{layout:?}");
    }
}

#[test]
fn direct_metal_burn_irreversible_rgb_u8_is_within_one_lsb_of_cpu() {
    if !metal_runtime_gate("j2k-ml irreversible RGB U8 Metal batch") {
        return;
    }
    let encoded = Arc::<[u8]>::from(htj2k_rgb8_97_fixture(8, 8));
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let inputs = vec![EncodedImage::full(encoded)];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode(inputs.clone())
        .expect("CPU irreversible RGB oracle");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("irreversible RGB8 must use U8 batch storage")
    };

    let mut decoder = MetalBurnDecoder::system_default(options).expect("paired Metal session");
    let output = decoder
        .decode(inputs)
        .expect("direct irreversible RGB Burn decode");
    let BurnBatchTensor::U8(tensor) = output.groups.into_iter().next().unwrap().tensor else {
        panic!("expected native irreversible RGB U8 Metal tensor")
    };
    let actual = tensor.into_data().into_vec::<u8>().expect("RGB U8 data");
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected.iter())
            .all(|(metal, cpu)| metal.abs_diff(*cpu) <= 1),
        "irreversible 9/7 Metal reconstruction must stay within one integer LSB of CPU"
    );
}
