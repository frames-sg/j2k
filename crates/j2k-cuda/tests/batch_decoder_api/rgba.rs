// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    prepare_batch, wrap_j2k_codestream, BatchDecodeOptions, BatchLayout, CpuBatchDecoder,
    CpuBatchSamples, DecodeRequest, EncodedImage, J2kChannelAssociation, J2kChannelDefinition,
    J2kChannelType, J2kFileBoxMetadata, J2kFileColorSpec, J2kFileWrapOptions, NativeSampleType,
    Rect,
};
use j2k_core::{Colorspace, Downscale};
use j2k_cuda::{CudaBatchDecoder, CudaSession};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};
use j2k_test_support::{
    generated_htj2k_rgba_fixture, Htj2kRgbaAlpha, Htj2kRgbaFixture, Htj2kRgbaSampleProfile,
    Htj2kRgbaSamples,
};

#[test]
fn exact_rgba_resident_and_external_match_cpu_across_codecs_types_geometry_and_layouts() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
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
    let context = CudaContext::system_default().expect("CUDA context");
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
                    assert_one_case(&context, codec, &encoded, layout, request);
                }
            }
        }
    }
}

fn assert_one_case(
    context: &CudaContext,
    codec: &str,
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
    .unwrap_or_else(|error| panic!("prepare {codec} RGBA: {error}"));
    assert!(prepared.errors().is_empty(), "{codec} preflight errors");
    let [group] = prepared.groups() else {
        panic!("expected one {codec} RGBA group")
    };
    if codec == "HTJ2K" {
        assert!(group.images()[0].htj2k_plan().is_some());
    } else {
        assert!(group.images()[0].classic_plan().is_some());
    }

    let mut cpu = CpuBatchDecoder::new(options);
    let oracle = cpu
        .decode_prepared(&prepared)
        .unwrap_or_else(|error| panic!("CPU {codec} RGBA oracle: {error}"));
    let expected = samples_as_bytes(oracle.groups()[0].samples());

    let session = CudaSession::with_context(context.clone());
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    let mut allocation = context
        .allocate(expected.len())
        .expect("RGBA external destination");
    let submitted = {
        let ptr = allocation.device_ptr();
        let len = allocation.byte_len();
        let alignment = match group.info().sample_type {
            NativeSampleType::U8 => 1,
            NativeSampleType::U16 | NativeSampleType::I16 => 2,
            _ => panic!("unsupported RGBA sample type"),
        };
        // SAFETY: the allocation is exclusively borrowed until submission
        // completion and is not read while codec work is pending.
        let mut destination = unsafe {
            CudaExternalDeviceBufferViewMut::from_raw_parts(
                context,
                ptr,
                len,
                alignment,
                &mut allocation,
            )
        }
        .expect("RGBA external view");
        // SAFETY: the destination owner remains live and inaccessible until wait.
        unsafe { decoder.submit_batch_into(group, &mut destination) }
            .unwrap_or_else(|error| panic!("submit external {codec} RGBA: {error}"))
    };
    submitted
        .wait()
        .unwrap_or_else(|error| panic!("wait external {codec} RGBA: {error}"));
    let mut external = vec![0_u8; expected.len()];
    allocation
        .copy_to_host(&mut external)
        .expect("download external RGBA");
    assert_eq!(
        external, expected,
        "external {codec} {layout:?} {request:?}"
    );

    let resident = decoder
        .decode_prepared(&prepared)
        .unwrap_or_else(|error| panic!("decode resident {codec} RGBA: {error}"));
    assert!(
        resident.group_errors().is_empty(),
        "{codec} resident errors"
    );
    let dense = resident.groups()[0].dense_output();
    let mut actual = vec![0_u8; expected.len()];
    dense
        .buffer()
        .copy_range_to_host(dense.ranges()[0].offset, &mut actual)
        .expect("download resident RGBA");
    assert_eq!(actual, expected, "resident {codec} {layout:?} {request:?}");
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
        other => panic!("unsupported RGBA oracle type: {other:?}"),
    }
}

fn wrap_rgba_jph(fixture: &Htj2kRgbaFixture) -> Vec<u8> {
    wrap_rgba_container(&fixture.encoded, fixture.alpha, J2kFileWrapOptions::jph())
}

fn wrap_classic_rgba_jp2(fixture: &Htj2kRgbaFixture) -> Vec<u8> {
    let pixels = samples_as_encoded_bytes(&fixture.samples);
    let codestream = j2k_native::encode(
        &pixels,
        fixture.width,
        fixture.height,
        4,
        fixture.bit_depth,
        fixture.signed,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            use_mct: fixture.use_mct,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode classic RGBA fixture");
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
