// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(feature = "cuda", not(target_os = "macos")))]

use burn_autodiff::Autodiff;
use burn_core::tensor::Tensor;
use burn_cuda::{Cuda, CudaDevice};
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
use burn_flex::{Flex, FlexDevice};
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
use burn_ndarray::NdArrayDevice::Cpu as FlexDevice;
use j2k::{DeviceDecodeRequest, Downscale, Rect};
use j2k_ml::{
    cpu, cuda, FloatNormalization, TensorDecodeError, TensorDecodeOptions, TensorInput,
    TensorLayout, TensorRoute,
};
use j2k_test_support::{
    cuda_runtime_and_strict_oxide_gate, htj2k_gray8_fixture, htj2k_gray8_large_fixture,
    openhtj2k_refinement_fixture,
};

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
type Flex = burn_ndarray::NdArray<f32, i64, i8>;

#[test]
fn direct_cuda_decode_reports_route_and_exact_pixels() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA direct") {
        return;
    }
    let encoded = htj2k_gray8_fixture(4, 3);
    let decoded = cuda::decode_u8(
        TensorInput::full(&encoded),
        &TensorDecodeOptions::default(),
        &CudaDevice::default(),
    )
    .expect("CUDA direct tensor decode");
    assert_eq!(decoded.route, TensorRoute::CudaDirect);
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
fn retained_primary_context_matches_cubecl_device_context() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA primary context") {
        return;
    }
    let first = j2k_cuda_runtime::CudaContext::retain_primary(0).expect("retain primary");
    let second = j2k_cuda_runtime::CudaContext::retain_primary(0).expect("retain primary again");
    assert!(first.is_same_context(&second));
    assert_eq!(first.device_ordinal(), 0);
}

#[test]
fn direct_cuda_u16_matches_portable_and_batches() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA u16 tensor parity") {
        return;
    }
    let device = CudaDevice::default();

    let encoded_u16 = openhtj2k_refinement_fixture();
    let expected_u16 = cpu::decode_u16::<Flex>(
        TensorInput::full(encoded_u16),
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect("portable u16 decode")
    .tensor
    .into_data()
    .into_vec::<u16>()
    .expect("portable u16 data");
    let actual_u16 = cuda::decode_u16(
        TensorInput::full(encoded_u16),
        &TensorDecodeOptions::default(),
        &device,
    )
    .expect("direct u16 decode")
    .tensor
    .into_data()
    .into_vec::<u16>()
    .expect("direct u16 data");
    assert_eq!(actual_u16, expected_u16);

    let u16_batch = cuda::decode_u16_batch(
        &[
            TensorInput::full(encoded_u16),
            TensorInput::full(encoded_u16),
        ],
        &TensorDecodeOptions::default(),
        &device,
    )
    .expect("direct u16 batch");
    assert_eq!(u16_batch.tensor.dims()[0], 2);
    let expected_u16_batch = expected_u16
        .iter()
        .chain(&expected_u16)
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(
        u16_batch
            .tensor
            .into_data()
            .into_vec::<u16>()
            .expect("direct u16 batch data"),
        expected_u16_batch
    );
}

#[test]
fn direct_cuda_float_matches_portable_full_batch_and_roi() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA float tensor parity") {
        return;
    }
    let device = CudaDevice::default();
    let encoded = htj2k_gray8_fixture(4, 3);
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
                cpu::decode_float::<Flex>(TensorInput::full(&encoded), &options, &FlexDevice)
                    .expect("portable float decode")
                    .tensor
                    .into_data()
                    .into_vec::<f32>()
                    .expect("portable float data");
            let actual = cuda::decode_float(TensorInput::full(&encoded), &options, &device)
                .expect("direct float decode")
                .tensor
                .into_data()
                .into_vec::<f32>()
                .expect("direct float data");
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1.0e-6);
            }
        }
    }

    let options = TensorDecodeOptions {
        layout: TensorLayout::ChannelsLast,
        normalization: FloatNormalization::MeanStd {
            mean: vec![0.25],
            std: vec![0.5],
        },
        ..TensorDecodeOptions::default()
    };
    let batch = cuda::decode_float_batch(
        &[TensorInput::full(&encoded), TensorInput::full(&encoded)],
        &options,
        &device,
    )
    .expect("direct float batch");
    assert_eq!(batch.tensor.dims(), [2, 3, 4, 1]);

    let large = htj2k_gray8_large_fixture(64, 64);
    let roi_input = TensorInput {
        encoded: &large,
        request: DeviceDecodeRequest::RegionScaled {
            roi: Rect {
                x: 8,
                y: 8,
                w: 32,
                h: 32,
            },
            scale: Downscale::Half,
        },
    };
    let expected_roi = cpu::decode_float::<Flex>(roi_input, &options, &FlexDevice)
        .expect("portable ROI decode")
        .tensor
        .into_data()
        .into_vec::<f32>()
        .expect("portable ROI data");
    let direct_roi = cuda::decode_float(roi_input, &options, &device).expect("direct ROI decode");
    assert_eq!(direct_roi.tensor.dims(), [16, 16, 1]);
    for (actual, expected) in direct_roi
        .tensor
        .into_data()
        .into_vec::<f32>()
        .expect("direct ROI data")
        .iter()
        .zip(expected_roi)
    {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
}

#[test]
fn direct_cuda_reports_batch_mismatch_and_lifts_to_autodiff() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA tensor contracts") {
        return;
    }
    let device = CudaDevice::default();
    let encoded = htj2k_gray8_fixture(4, 3);
    let options = TensorDecodeOptions {
        layout: TensorLayout::ChannelsLast,
        normalization: FloatNormalization::MeanStd {
            mean: vec![0.25],
            std: vec![0.5],
        },
        ..TensorDecodeOptions::default()
    };
    let mismatch = htj2k_gray8_fixture(5, 3);
    let error = cuda::decode_u8_batch(
        &[TensorInput::full(&encoded), TensorInput::full(&mismatch)],
        &TensorDecodeOptions::default(),
        &device,
    )
    .expect_err("direct batch shape mismatch");
    assert!(matches!(
        error,
        TensorDecodeError::BatchShapeMismatch { index: 1, .. }
    ));

    let inner = cuda::decode_float(TensorInput::full(&encoded), &options, &device)
        .expect("direct float for autodiff")
        .tensor;
    let autodiff = Tensor::<Autodiff<Cuda>, 3>::from_inner(inner).require_grad();
    assert_eq!(autodiff.dims(), [3, 4, 1]);
}
