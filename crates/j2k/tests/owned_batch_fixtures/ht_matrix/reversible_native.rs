// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    BatchDecodeOptions, BatchLayout, BatchWaveletTransform, CpuBatchDecoder, EncodedImage,
    NativeSampleType,
};

use super::super::{encode_case, CodingRoute};

#[test]
fn reversible_batch_matrix_preserves_native_gray_and_rgb_samples() {
    let mut cases = Vec::new();
    for route in [CodingRoute::Classic, CodingRoute::Htj2k] {
        for components in [1, 3] {
            for signed in [false, true] {
                for precision in [8, 12, 16] {
                    cases.push(encode_case(route, components, precision, signed));
                }
            }
        }
    }
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(
            cases
                .iter()
                .map(|case| EncodedImage::full(Arc::clone(&case.encoded)))
                .collect(),
        )
        .expect("decode native sample matrix");

    assert!(
        result.errors().is_empty(),
        "matrix errors: {:?}",
        result.errors()
    );
    assert_eq!(result.groups().len(), cases.len());
    for (source_index, case) in cases.iter().enumerate() {
        let group = result
            .groups()
            .iter()
            .find(|group| group.source_indices() == [source_index])
            .unwrap_or_else(|| panic!("{}: output group", case.name));
        assert_eq!(group.info().precision, case.precision, "{}", case.name);
        assert_eq!(group.info().signed, case.signed, "{}", case.name);
        assert_eq!(
            group.info().color.channels(),
            case.components,
            "{}",
            case.name
        );
        assert_eq!(group.info().route, case.route, "{}", case.name);
        assert_eq!(
            group.info().transform,
            BatchWaveletTransform::Reversible53,
            "{}",
            case.name
        );
        assert_eq!(
            group.info().sample_type,
            if case.signed {
                NativeSampleType::I16
            } else if case.precision <= 8 {
                NativeSampleType::U8
            } else {
                NativeSampleType::U16
            },
            "{}",
            case.name
        );
        case.oracle.assert_samples(group.samples(), &case.name);
    }
}
