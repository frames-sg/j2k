// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every CPU integer output of irreversible (9/7) data follows one rounding
//! contract: the centered sample is rounded ties-to-even before the unsigned
//! level shift is added, as the full decode and `OpenJPEG` do. Region, row, and
//! batch decodes must therefore equal the matching crop of the full decode,
//! including samples whose shifted value is an exact tie.

use std::sync::Arc;

use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, EncodedImage,
    J2kDecoder, J2kError, J2kScratchPool,
};
use j2k_core::{ImageDecodeRows, PixelFormat, Rect, RowSink};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::{crop_interleaved_bytes, PixelRect};

const SIZE: u32 = 512;

#[derive(Clone, Copy, Debug)]
struct LossyCase {
    components: u16,
    ht: bool,
    use_mct: bool,
}

const CASES: [LossyCase; 3] = [
    LossyCase {
        components: 3,
        ht: true,
        use_mct: true,
    },
    LossyCase {
        components: 3,
        ht: false,
        use_mct: false,
    },
    LossyCase {
        components: 1,
        ht: true,
        use_mct: false,
    },
];

#[derive(Default)]
struct CollectRows(Vec<u8>);

impl RowSink<u8> for CollectRows {
    type Error = J2kError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), Self::Error> {
        self.0.extend_from_slice(row);
        Ok(())
    }
}

fn lossy_fixture(case: LossyCase) -> Vec<u8> {
    let components = u32::from(case.components);
    let mut state = 0x1234_5678_u32;
    let pixels: Vec<u8> = (0..SIZE * SIZE * components)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let (pixel, component) = (index / components, index % components);
            let (x, y) = (pixel % SIZE, pixel / SIZE);
            let base = (x * 3 + y * 5 + component * 40) / 9 % 200 + 20;
            u8::try_from(base + state % 17).expect("sample fits u8")
        })
        .collect();
    let options = EncodeOptions {
        reversible: false,
        use_mct: case.use_mct,
        irreversible_quantization_scale: 4.0,
        ..EncodeOptions::default()
    };
    if case.ht {
        encode_htj2k(&pixels, SIZE, SIZE, case.components, 8, false, &options)
    } else {
        encode(&pixels, SIZE, SIZE, case.components, 8, false, &options)
    }
    .expect("encode lossy 9/7 fixture")
}

/// Samples whose unrounded output lands exactly on `k + 0.5` after the level
/// shift: the inputs on which rounding after the shift breaks the contract.
#[expect(
    clippy::float_cmp,
    reason = "a tie is an exact property of the f32 value"
)]
fn shifted_tie_count(bytes: &[u8]) -> usize {
    let image = j2k_native::Image::new(bytes, &j2k_native::DecodeSettings::default())
        .expect("inspect lossy fixture");
    let mut context = j2k_native::DecoderContext::default();
    let components = image
        .decode_components_with_context(&mut context)
        .expect("float component decode");
    components
        .planes()
        .iter()
        .flat_map(j2k_native::ComponentPlane::samples)
        .filter(|sample| {
            let doubled = **sample * 2.0;
            doubled == doubled.floor() && doubled.rem_euclid(2.0) == 1.0
        })
        .count()
}

fn format_for(case: LossyCase) -> (PixelFormat, usize) {
    if case.components == 1 {
        (PixelFormat::Gray8, 1)
    } else {
        (PixelFormat::Rgb8, 3)
    }
}

fn batch_u8(bytes: &[u8], request: DecodeRequest) -> Vec<u8> {
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let result = CpuBatchDecoder::new(options)
        .decode(Vec::from([EncodedImage::new(Arc::from(bytes), request)]))
        .expect("CPU batch decode");
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    let CpuBatchSamples::U8(samples) = result.groups()[0].samples() else {
        panic!("8-bit lossy fixture must use U8 batch storage")
    };
    samples.clone()
}

fn mismatch_summary(label: &str, expected: &[u8], actual: &[u8]) -> Option<String> {
    if expected.len() != actual.len() {
        return Some(format!(
            "{label}: lengths {} / {}",
            expected.len(),
            actual.len()
        ));
    }
    let mismatches = expected.iter().zip(actual).filter(|(a, b)| a != b).count();
    (mismatches > 0).then(|| format!("{label}: {mismatches} bytes differ from the full decode"))
}

#[test]
fn irreversible_region_row_and_batch_outputs_match_the_full_decode() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: SIZE - 1,
        h: SIZE - 1,
    };
    let mut failures = Vec::new();
    for case in CASES {
        let bytes = lossy_fixture(case);
        assert!(
            shifted_tie_count(&bytes) > 0,
            "{case:?}: fixture has no shifted ties, so it cannot detect the rounding order"
        );
        let (fmt, channels) = format_for(case);
        let width = SIZE as usize;

        let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
        let mut full: Vec<u8> = std::iter::repeat_n(0, width * width * channels).collect();
        decoder
            .decode_into(&mut full, width * channels, fmt)
            .expect("full decode");
        let expected_region = crop_interleaved_bytes(
            &full,
            width,
            channels,
            PixelRect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
        );

        let mut region: Vec<u8> = std::iter::repeat_n(0, expected_region.len()).collect();
        decoder
            .decode_region_into(
                &mut J2kScratchPool::new(),
                &mut region,
                roi.w as usize * channels,
                fmt,
                roi,
            )
            .expect("region decode");

        let mut rows = CollectRows::default();
        <J2kDecoder<'_> as ImageDecodeRows<'_, u8>>::decode_rows(&mut decoder, &mut rows)
            .expect("row decode");

        let checks = [
            ("region", &expected_region, region),
            ("rows", &full, rows.0),
            ("batch full", &full, batch_u8(&bytes, DecodeRequest::Full)),
            (
                "batch region",
                &expected_region,
                batch_u8(&bytes, DecodeRequest::Region { roi }),
            ),
        ];
        for (label, expected, actual) in checks {
            failures.extend(mismatch_summary(
                &format!("{case:?} {label}"),
                expected,
                &actual,
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
