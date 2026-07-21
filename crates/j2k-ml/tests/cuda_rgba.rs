// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(feature = "cuda", not(target_os = "macos")))]

use std::sync::Arc;

use burn_core::tensor::DType;
use burn_cuda::CudaDevice;
use j2k::{
    encode_j2k_lossless, wrap_j2k_codestream, BatchDecodeOptions, BatchLayout, CpuBatchDecoder,
    CpuBatchSamples, DecodeRequest, Downscale, EncodedImage, J2kBlockCodingMode,
    J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kEncodeValidation,
    J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions, J2kLosslessEncodeOptions,
    J2kLosslessSamples, Rect, ReversibleTransform,
};
use j2k_core::Colorspace;
use j2k_ml::{BurnBatchTensor, CudaBurnDecoder};
use j2k_test_support::{
    cuda_runtime_and_strict_oxide_gate, generated_htj2k_rgba_fixture, Htj2kRgbaAlpha,
    Htj2kRgbaFixture, Htj2kRgbaSampleProfile, Htj2kRgbaSamples,
};

#[test]
fn direct_cuda_burn_rgba_matches_cpu_across_codecs_types_geometry_and_layouts() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml exact RGBA CUDA batch") {
        return;
    }
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region {
            roi: Rect {
                x: 1,
                y: 2,
                w: 5,
                h: 4,
            },
        },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi: Rect {
                x: 1,
                y: 2,
                w: 5,
                h: 4,
            },
            scale: Downscale::Half,
        },
    ];

    for profile in [
        Htj2kRgbaSampleProfile::U8Rct,
        Htj2kRgbaSampleProfile::U12,
        Htj2kRgbaSampleProfile::I16,
    ] {
        let fixture = generated_htj2k_rgba_fixture(profile, Htj2kRgbaAlpha::Straight);
        let encodings = [
            ("HTJ2K", Arc::<[u8]>::from(wrap_rgba_jph(&fixture))),
            (
                "classic JPEG 2000",
                Arc::<[u8]>::from(wrap_classic_rgba_jp2(&fixture)),
            ),
        ];
        for (codec, encoded) in encodings {
            for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
                for request in requests {
                    assert_burn_case(codec, &encoded, layout, request);
                }
            }
        }
    }
}

fn assert_burn_case(codec: &str, encoded: &Arc<[u8]>, layout: BatchLayout, request: DecodeRequest) {
    let options = BatchDecodeOptions {
        layout,
        ..BatchDecodeOptions::default()
    };
    let inputs = vec![
        EncodedImage::new(Arc::clone(encoded), request),
        EncodedImage::new(Arc::clone(encoded), request),
    ];
    let mut decoder = CudaBurnDecoder::new(CudaDevice::default(), options)
        .unwrap_or_else(|error| panic!("create {codec} RGBA CUDA adapter: {error}"));
    let prepared = decoder
        .prepare(inputs.clone())
        .unwrap_or_else(|error| panic!("prepare {codec} RGBA Burn batch: {error}"));
    assert!(prepared.errors().is_empty(), "{codec} preflight errors");
    let [prepared_group] = prepared.groups() else {
        panic!("expected one {codec} RGBA group")
    };
    if codec == "HTJ2K" {
        assert!(prepared_group.images()[0].htj2k_plan().is_some());
    } else {
        assert!(prepared_group.images()[0].classic_plan().is_some());
    }

    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode_prepared(&prepared)
        .unwrap_or_else(|error| panic!("CPU {codec} RGBA oracle: {error}"));
    let output = decoder
        .decode_prepared(&prepared)
        .unwrap_or_else(|error| panic!("direct {codec} RGBA Burn decode: {error}"));
    assert!(output.errors.is_empty(), "{codec} {layout:?} {request:?}");
    assert!(
        output.group_errors.is_empty(),
        "{codec} {layout:?} {request:?}: {:?}",
        output.group_errors
    );
    let group = output.groups.into_iter().next().expect("RGBA Burn group");
    assert_eq!(group.source_indices, [0, 1]);
    let dimensions = prepared_group.info().dimensions;
    let shape = match layout {
        BatchLayout::Nhwc => [2, dimensions.1 as usize, dimensions.0 as usize, 4],
        BatchLayout::Nchw => [2, 4, dimensions.1 as usize, dimensions.0 as usize],
        _ => unreachable!("test only covers public dense layouts"),
    };
    assert_tensor_matches_cpu(
        expected.groups()[0].samples(),
        group.tensor,
        shape,
        &format!("{codec} {layout:?} {request:?}"),
    );
}

fn assert_tensor_matches_cpu(
    expected: &CpuBatchSamples,
    tensor: BurnBatchTensor<burn_cuda::Cuda>,
    shape: [usize; 4],
    case: &str,
) {
    match (expected, tensor) {
        (CpuBatchSamples::U8(expected), BurnBatchTensor::U8(tensor)) => {
            assert_eq!(tensor.dtype(), DType::U8);
            assert_eq!(tensor.dims(), shape);
            assert_eq!(
                tensor.into_data().into_vec::<u8>().expect("RGBA U8 data"),
                *expected,
                "{case}"
            );
        }
        (CpuBatchSamples::U16(expected), BurnBatchTensor::U16(tensor)) => {
            assert_eq!(tensor.dtype(), DType::U16);
            assert_eq!(tensor.dims(), shape);
            assert_eq!(
                tensor.into_data().into_vec::<u16>().expect("RGBA U16 data"),
                *expected,
                "{case}"
            );
        }
        (CpuBatchSamples::I16(expected), BurnBatchTensor::I16(tensor)) => {
            assert_eq!(tensor.dtype(), DType::I16);
            assert_eq!(tensor.dims(), shape);
            assert_eq!(
                tensor.into_data().into_vec::<i16>().expect("RGBA I16 data"),
                *expected,
                "{case}"
            );
        }
        (expected, actual) => {
            panic!("unexpected RGBA storage for {case}: expected {expected:?}, got {actual:?}")
        }
    }
}

fn wrap_rgba_jph(fixture: &Htj2kRgbaFixture) -> Vec<u8> {
    wrap_rgba_container(&fixture.encoded, fixture.alpha, J2kFileWrapOptions::jph())
}

fn wrap_classic_rgba_jp2(fixture: &Htj2kRgbaFixture) -> Vec<u8> {
    let bytes = samples_as_encoded_bytes(&fixture.samples);
    let samples = J2kLosslessSamples::new(
        &bytes,
        fixture.width,
        fixture.height,
        4,
        fixture.bit_depth,
        fixture.signed,
    )
    .expect("valid classic RGBA samples");
    let transform = if fixture.use_mct {
        ReversibleTransform::Rct53
    } else {
        ReversibleTransform::None53
    };
    let codestream = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(transform)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode classic RGBA fixture")
    .codestream;
    wrap_rgba_container(&codestream, fixture.alpha, J2kFileWrapOptions::jp2())
}

fn samples_as_encoded_bytes(samples: &Htj2kRgbaSamples) -> Vec<u8> {
    match samples {
        Htj2kRgbaSamples::U8(samples) => samples.clone(),
        Htj2kRgbaSamples::U16(samples) => samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect(),
        Htj2kRgbaSamples::I16(samples) => samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect(),
    }
}

fn wrap_rgba_container(
    codestream: &[u8],
    alpha: Htj2kRgbaAlpha,
    options: J2kFileWrapOptions<'_>,
) -> Vec<u8> {
    let alpha_type = match alpha {
        Htj2kRgbaAlpha::Straight => J2kChannelType::Opacity,
        Htj2kRgbaAlpha::Premultiplied => J2kChannelType::PremultipliedOpacity,
    };
    let definitions = [
        J2kChannelDefinition {
            channel_index: 0,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 1 },
        },
        J2kChannelDefinition {
            channel_index: 1,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 2 },
        },
        J2kChannelDefinition {
            channel_index: 2,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 3 },
        },
        J2kChannelDefinition {
            channel_index: 3,
            channel_type: alpha_type,
            association: J2kChannelAssociation::WholeImage,
        },
    ];
    wrap_j2k_codestream(
        codestream,
        options
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &definitions,
            }),
    )
    .expect("wrap RGBA container")
}
