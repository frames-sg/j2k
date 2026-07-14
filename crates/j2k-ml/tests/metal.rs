// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(feature = "metal", target_os = "macos"))]

use burn_flex::{Flex, FlexDevice};
use burn_wgpu::WgpuDevice;
use j2k::{DeviceDecodeRequest, Downscale, Rect};
use j2k_ml::{
    cpu, metal, FloatNormalization, TensorDecodeOptions, TensorInput, TensorLayout, TensorRoute,
};
use j2k_test_support::{
    classic_j2k_gray8_fixture, metal_runtime_gate, openhtj2k_refinement_fixture,
};

#[test]
fn strict_metal_staged_decode_reports_route_and_pixels() {
    if !metal_runtime_gate("j2k-ml strict Metal staged decode") {
        return;
    }
    let encoded = classic_j2k_gray8_fixture(4, 3);
    let decoded = metal::decode_u8(
        TensorInput::full(&encoded),
        &TensorDecodeOptions::default(),
        &WgpuDevice::DefaultDevice,
    )
    .expect("strict Metal tensor decode");

    assert_eq!(decoded.route, TensorRoute::MetalStaged);
    assert_eq!(decoded.tensor.dims(), [1, 3, 4]);
    assert_eq!(
        decoded
            .tensor
            .into_data()
            .into_vec::<u8>()
            .expect("u8 data"),
        (0..12).collect::<Vec<_>>()
    );
}

#[test]
fn metal_batch_returns_one_rank_four_tensor() {
    if !metal_runtime_gate("j2k-ml Metal staged batch") {
        return;
    }
    let first = classic_j2k_gray8_fixture(2, 2);
    let second = classic_j2k_gray8_fixture(2, 2);
    let decoded = metal::decode_u8_batch(
        &[TensorInput::full(&first), TensorInput::full(&second)],
        &TensorDecodeOptions::default(),
        &WgpuDevice::DefaultDevice,
    )
    .expect("strict Metal batch");
    assert_eq!(decoded.route, TensorRoute::MetalStaged);
    assert_eq!(decoded.tensor.dims(), [2, 1, 2, 2]);
}

#[test]
fn metal_u16_and_float_match_portable_decode() {
    if !metal_runtime_gate("j2k-ml Metal u16 and float parity") {
        return;
    }
    let encoded = openhtj2k_refinement_fixture();
    let expected_u16 = cpu::decode_u16::<Flex>(
        TensorInput::full(encoded),
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect("portable u16")
    .tensor
    .into_data()
    .into_vec::<u16>()
    .expect("portable u16 data");
    let actual_u16 = metal::decode_u16(
        TensorInput::full(encoded),
        &TensorDecodeOptions::default(),
        &WgpuDevice::DefaultDevice,
    )
    .expect("Metal u16")
    .tensor
    .into_data()
    .into_vec::<u16>()
    .expect("Metal u16 data");
    assert_eq!(actual_u16, expected_u16);

    for layout in [TensorLayout::ChannelsFirst, TensorLayout::ChannelsLast] {
        for normalization in [
            FloatNormalization::Raw,
            FloatNormalization::Unit,
            FloatNormalization::MeanStd {
                mean: vec![0.25],
                std: vec![0.5],
            },
        ] {
            let options = TensorDecodeOptions {
                layout,
                normalization,
                ..TensorDecodeOptions::default()
            };
            let expected =
                cpu::decode_float::<Flex>(TensorInput::full(encoded), &options, &FlexDevice)
                    .expect("portable float")
                    .tensor
                    .into_data()
                    .into_vec::<f32>()
                    .expect("portable float data");
            let actual = metal::decode_float(
                TensorInput::full(encoded),
                &options,
                &WgpuDevice::DefaultDevice,
            )
            .expect("Metal float")
            .tensor
            .into_data()
            .into_vec::<f32>()
            .expect("Metal float data");
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1.0e-6);
            }
        }
    }
}

#[test]
fn metal_roi_scale_and_all_batch_output_modes_report_staged_route() {
    if !metal_runtime_gate("j2k-ml Metal ROI and batch modes") {
        return;
    }
    let encoded = classic_j2k_gray8_fixture(8, 8);
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let input = TensorInput {
        encoded: &encoded,
        request: DeviceDecodeRequest::RegionScaled {
            roi,
            scale: Downscale::Half,
        },
    };
    let decoded = metal::decode_float(
        input,
        &TensorDecodeOptions::default(),
        &WgpuDevice::DefaultDevice,
    )
    .expect("Metal ROI-scaled float");
    assert_eq!(decoded.route, TensorRoute::MetalStaged);
    assert_eq!(decoded.tensor.dims(), [1, 2, 2]);
    let expected_roi =
        cpu::decode_float::<Flex>(input, &TensorDecodeOptions::default(), &FlexDevice)
            .expect("portable ROI-scaled float")
            .tensor
            .into_data()
            .into_vec::<f32>()
            .expect("portable ROI data");
    assert_eq!(
        decoded
            .tensor
            .into_data()
            .into_vec::<f32>()
            .expect("Metal ROI data"),
        expected_roi
    );

    let encoded_u16 = openhtj2k_refinement_fixture();
    let u16_batch = metal::decode_u16_batch(
        &[
            TensorInput::full(encoded_u16),
            TensorInput::full(encoded_u16),
        ],
        &TensorDecodeOptions::default(),
        &WgpuDevice::DefaultDevice,
    )
    .expect("Metal u16 batch");
    assert_eq!(u16_batch.route, TensorRoute::MetalStaged);
    assert_eq!(u16_batch.tensor.dims()[0], 2);
    let u16_values = u16_batch
        .tensor
        .into_data()
        .into_vec::<u16>()
        .expect("Metal u16 batch data");
    let midpoint = u16_values.len() / 2;
    assert_eq!(&u16_values[..midpoint], &u16_values[midpoint..]);

    let float_batch = metal::decode_float_batch(
        &[TensorInput::full(&encoded), TensorInput::full(&encoded)],
        &TensorDecodeOptions::default(),
        &WgpuDevice::DefaultDevice,
    )
    .expect("Metal float batch");
    assert_eq!(float_batch.route, TensorRoute::MetalStaged);
    assert_eq!(float_batch.tensor.dims(), [2, 1, 8, 8]);
    let float_values = float_batch
        .tensor
        .into_data()
        .into_vec::<f32>()
        .expect("Metal float batch data");
    let midpoint = float_values.len() / 2;
    assert_eq!(&float_values[..midpoint], &float_values[midpoint..]);
}
