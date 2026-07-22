// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn independent_openjph_signed_rgb_is_exact_in_burn_for_all_requests_and_layouts() {
    if !metal_runtime_gate("j2k-ml exact signed RGB Metal batch") {
        return;
    }

    let roi = Rect {
        x: 3,
        y: 2,
        w: 9,
        h: 7,
    };
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region { roi },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi,
            scale: Downscale::Half,
        },
    ];
    let fixtures = j2k_test_support::openjph_batch_fixtures()
        .iter()
        .filter(|fixture| {
            matches!(
                fixture.name,
                "openjph-rgb-s8-53-single-raw"
                    | "openjph-rgb-s12-53-single-raw"
                    | "openjph-rgb-s16-53-single-raw"
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(fixtures.len(), 3);

    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        let mut decoder = MetalBurnDecoder::system_default(options).expect("paired Metal session");
        for fixture in &fixtures {
            let encoded = Arc::<[u8]>::from(fixture.encoded);
            for request in requests {
                let inputs = vec![
                    EncodedImage::new(encoded.clone(), request),
                    EncodedImage::new(encoded.clone(), request),
                ];
                let mut cpu = CpuBatchDecoder::new(options);
                let expected = cpu
                    .decode(inputs.clone())
                    .unwrap_or_else(|error| panic!("{} CPU oracle: {error}", fixture.name));
                let CpuBatchSamples::I16(expected) = expected.groups()[0].samples() else {
                    panic!("{} must use I16 batch storage", fixture.name)
                };

                let prepared = decoder
                    .prepare(inputs)
                    .unwrap_or_else(|error| panic!("{} prepare: {error}", fixture.name));
                let output = decoder
                    .decode_prepared(&prepared)
                    .unwrap_or_else(|error| panic!("{} decode: {error}", fixture.name));
                assert!(output.errors.is_empty(), "{}", fixture.name);
                assert!(output.group_errors.is_empty(), "{}", fixture.name);
                assert_eq!(output.groups.len(), 1, "{}", fixture.name);
                assert_eq!(output.groups[0].source_indices, [0, 1]);
                let BurnBatchTensor::I16(tensor) = output.groups.into_iter().next().unwrap().tensor
                else {
                    panic!("{} must produce an I16 Burn tensor", fixture.name)
                };
                assert_eq!(tensor.dtype(), DType::I16);
                let actual = tensor.into_data().into_vec::<i16>().expect("I16 data");
                assert_eq!(
                    actual.as_slice(),
                    expected.as_slice(),
                    "{} {layout:?} {request:?}",
                    fixture.name
                );
            }
        }
    }
}

#[test]
fn direct_metal_burn_rgba_is_exact_for_native_types_and_layouts() {
    if !metal_runtime_gate("j2k-ml exact RGBA Metal batch") {
        return;
    }

    for profile in [
        Htj2kRgbaSampleProfile::U8Rct,
        Htj2kRgbaSampleProfile::U12,
        Htj2kRgbaSampleProfile::I16,
    ] {
        let fixture = generated_htj2k_rgba_fixture(profile, Htj2kRgbaAlpha::Straight);
        let encoded = Arc::<[u8]>::from(wrap_rgba_jph(&fixture.encoded, fixture.alpha));
        for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
            let options = BatchDecodeOptions {
                layout,
                ..BatchDecodeOptions::default()
            };
            let inputs = vec![
                EncodedImage::full(encoded.clone()),
                EncodedImage::full(encoded.clone()),
            ];
            let mut cpu = CpuBatchDecoder::new(options);
            let expected = cpu.decode(inputs.clone()).expect("CPU RGBA oracle");
            assert!(
                expected.errors().is_empty(),
                "{profile:?} {layout:?}: {:?}",
                expected.errors()
            );
            assert_eq!(expected.groups().len(), 1);

            let mut decoder =
                MetalBurnDecoder::system_default(options).expect("paired Metal session");
            let prepared = decoder.prepare(inputs).expect("prepare RGBA Burn batch");
            let output = decoder
                .decode_prepared(&prepared)
                .expect("direct RGBA Burn decode");
            assert!(output.errors.is_empty(), "{profile:?} {layout:?}");
            assert!(output.group_errors.is_empty(), "{profile:?} {layout:?}");
            assert_eq!(output.groups.len(), 1, "{profile:?} {layout:?}");
            let group = output.groups.into_iter().next().expect("RGBA group");
            assert_eq!(group.source_indices, [0, 1]);
            let shape = match layout {
                BatchLayout::Nhwc => [
                    2,
                    usize::try_from(fixture.height).expect("RGBA height fits usize"),
                    usize::try_from(fixture.width).expect("RGBA width fits usize"),
                    4,
                ],
                BatchLayout::Nchw => [
                    2,
                    4,
                    usize::try_from(fixture.height).expect("RGBA height fits usize"),
                    usize::try_from(fixture.width).expect("RGBA width fits usize"),
                ],
                _ => unreachable!(),
            };
            match (expected.groups()[0].samples(), group.tensor) {
                (CpuBatchSamples::U8(expected), BurnBatchTensor::U8(tensor)) => {
                    assert_eq!(tensor.dtype(), DType::U8);
                    assert_eq!(tensor.dims(), shape);
                    assert_eq!(
                        tensor.into_data().into_vec::<u8>().expect("RGBA U8 data"),
                        *expected,
                        "{profile:?} {layout:?}"
                    );
                }
                (CpuBatchSamples::U16(expected), BurnBatchTensor::U16(tensor)) => {
                    assert_eq!(tensor.dtype(), DType::U16);
                    assert_eq!(tensor.dims(), shape);
                    assert_eq!(
                        tensor
                            .into_data()
                            .into_vec::<u16>()
                            .expect("RGBA U16 data"),
                        *expected,
                        "{profile:?} {layout:?}"
                    );
                }
                (CpuBatchSamples::I16(expected), BurnBatchTensor::I16(tensor)) => {
                    assert_eq!(tensor.dtype(), DType::I16);
                    assert_eq!(tensor.dims(), shape);
                    assert_eq!(
                        tensor
                            .into_data()
                            .into_vec::<i16>()
                            .expect("RGBA I16 data"),
                        *expected,
                        "{profile:?} {layout:?}"
                    );
                }
                (expected, actual) => panic!(
                    "unexpected RGBA storage for {profile:?} {layout:?}: expected {expected:?}, got {actual:?}"
                ),
            }
        }
    }
}

#[test]
fn direct_metal_batch_preserves_native_u16_samples() {
    if !metal_runtime_gate("j2k-ml direct Metal U16 batch") {
        return;
    }
    let samples = [0_u16, 1, 2048, 4095];
    let encoded = encode_gray12(&samples);
    let mut decoder = MetalBurnDecoder::system_default(BatchDecodeOptions::default())
        .expect("paired J2K/Burn Metal session");
    let output = decoder
        .decode(vec![EncodedImage::full(Arc::from(encoded))])
        .expect("direct Metal U16 decode");

    let BurnBatchTensor::U16(tensor) = output.groups.into_iter().next().unwrap().tensor else {
        panic!("expected native U16 Metal tensor")
    };
    assert_eq!(tensor.dtype(), DType::U16);
    assert_eq!(
        tensor.into_data().into_vec::<u16>().expect("U16 data"),
        samples
    );
}

#[test]
fn direct_metal_batch_preserves_native_signed_i16_samples() {
    if !metal_runtime_gate("j2k-ml direct Metal signed I16 batch") {
        return;
    }
    let samples = [-2048_i16, -1, 0, 2047];
    let encoded = Arc::<[u8]>::from(encode_signed_gray12(&samples));
    let options = BatchDecodeOptions::default();
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode(vec![EncodedImage::full(Arc::clone(&encoded))])
        .expect("CPU signed Gray12 oracle");
    let CpuBatchSamples::I16(expected) = expected.groups()[0].samples() else {
        panic!("signed Gray12 CPU oracle must be I16")
    };

    let mut decoder = MetalBurnDecoder::system_default(options).expect("paired Metal session");
    let output = decoder
        .decode(vec![EncodedImage::full(encoded)])
        .expect("direct Metal signed Gray12 decode");
    let BurnBatchTensor::I16(tensor) = output.groups.into_iter().next().unwrap().tensor else {
        panic!("expected native I16 Metal tensor")
    };
    assert_eq!(tensor.dtype(), DType::I16);
    assert_eq!(
        tensor.into_data().into_vec::<i16>().expect("I16 data"),
        *expected
    );
}
