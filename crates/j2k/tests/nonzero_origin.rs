// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression for frames-sg/j2k#328.
//!
//! The seed is a 53x2 Gray8 reversible 5/3 codestream at origin (0, 14).
//! It contains tile 0 of four tile columns, so only tile 0 is complete.
//! Reversible decoding must return integers, and every decode route must
//! agree on tile 0.

use std::num::NonZeroUsize;
use std::sync::Arc;

use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, EncodedImage,
    J2kDecoder, J2kScratchPool, PixelFormat, Rect,
};

const SEED: &[u8] = include_bytes!(
    "../../j2k-test-support/fixtures/nonzero_origin/classic_gray8_nonzero_origin.j2c"
);
const WIDTH: u32 = 53;
const HEIGHT: u32 = 2;
const TILE_WIDTH: u32 = 16;

const TILE0: [[u8; TILE_WIDTH as usize]; HEIGHT as usize] = [
    [
        169, 104, 24, 127, 144, 125, 118, 153, 134, 105, 130, 126, 116, 156, 89, 197,
    ],
    [
        121, 177, 253, 129, 119, 117, 127, 151, 122, 103, 138, 128, 112, 151, 84, 192,
    ],
];

fn full_roi() -> Rect {
    Rect {
        x: 0,
        y: 0,
        w: WIDTH,
        h: HEIGHT,
    }
}

fn decode_core(planes_budget: Option<usize>) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(SEED).expect("parse the seed");
    if let Some(budget) = planes_budget {
        decoder.set_full_decode_planes_budget(budget);
    }
    let (width, height) = decoder.info().dimensions;
    assert_eq!((width, height), (WIDTH, HEIGHT), "decoded size");
    let mut samples = vec![0; (width * height) as usize];
    decoder
        .decode_into(&mut samples, width as usize, PixelFormat::Gray8)
        .expect("whole-image decode");
    samples
}

fn decode_region() -> Vec<u8> {
    let roi = full_roi();
    let mut decoder = J2kDecoder::new(SEED).expect("parse the seed");
    let mut samples = vec![0; (roi.w * roi.h) as usize];
    decoder
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut samples,
            roi.w as usize,
            PixelFormat::Gray8,
            roi,
        )
        .expect("region decode");
    samples
}

fn decode_batch(request: DecodeRequest) -> Vec<u8> {
    let mut options = BatchDecodeOptions::default();
    options.layout = BatchLayout::Nhwc;
    options.workers = NonZeroUsize::new(1);
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(vec![EncodedImage::new(Arc::from(SEED), request)])
        .expect("batch decode");
    assert!(
        result.errors().is_empty(),
        "batch decode errors: {:?}",
        result.errors()
    );
    match result.groups()[0].samples() {
        CpuBatchSamples::U8(samples) => samples.clone(),
        _ => panic!("expected 8-bit batch samples"),
    }
}

fn assert_tile0(name: &str, samples: &[u8]) {
    assert_eq!(
        samples.len(),
        (WIDTH * HEIGHT) as usize,
        "{name} output size"
    );
    for y in 0..HEIGHT as usize {
        for x in 0..TILE_WIDTH as usize {
            assert_eq!(
                samples[y * WIDTH as usize + x],
                TILE0[y][x],
                "{name} sample at ({x}, {y})"
            );
        }
    }
}

#[test]
fn reversible_53_nonzero_origin_decodes_exactly() {
    let routes = [
        ("default", decode_core(None)),
        ("planes", decode_core(Some(usize::MAX))),
        ("streamed", decode_core(Some(0))),
        ("region", decode_region()),
        ("batch-full", decode_batch(DecodeRequest::Full)),
        (
            "batch-region",
            decode_batch(DecodeRequest::Region { roi: full_roi() }),
        ),
    ];
    for (name, samples) in &routes {
        assert_tile0(name, samples);
    }
    for (name, samples) in &routes[1..] {
        assert_eq!(
            samples, &routes[0].1,
            "{name} differs from the default route"
        );
    }

    let image =
        j2k_native::Image::new(SEED, &j2k_native::DecodeSettings::default()).expect("parse seed");
    let mut context = j2k_native::DecoderContext::default();
    let components = image
        .decode_components_with_context(&mut context)
        .expect("float component decode");
    let samples = components.planes()[0].samples();
    assert_eq!(samples.len(), (WIDTH * HEIGHT) as usize, "plane size");
    let fractional: Vec<(usize, f32)> = samples
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, sample)| sample.fract() != 0.0)
        .collect();
    assert!(
        fractional.is_empty(),
        "fractional f32 samples: {fractional:?}"
    );
}
