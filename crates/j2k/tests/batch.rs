// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    decode_tiles_into, decode_tiles_region_into, decode_tiles_region_scaled_into,
    decode_tiles_scaled_into, wrap_j2k_codestream, Downscale, J2kDecoder, J2kFileWrapOptions,
    PixelFormat, Rect, TileBatchOptions, TileDecodeJob, TileRegionDecodeJob,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use std::num::NonZeroUsize;

fn encode_codestream(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(
        pixels,
        width,
        height,
        components.into(),
        bit_depth,
        false,
        &options,
    )
    .expect("encode")
}

fn encode_ht_codestream(
    pixels: &[u8],
    width: u32,
    height: u32,
    components: u8,
    bit_depth: u8,
) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    encode_htj2k(
        pixels,
        width,
        height,
        components.into(),
        bit_depth,
        false,
        &options,
    )
    .expect("encode HTJ2K")
}

fn rgb_fixture() -> Vec<u8> {
    let pixels = (0_u8..48).collect::<Vec<_>>();
    encode_codestream(&pixels, 4, 4, 3, 8)
}

fn ht_rgb_fixture() -> Vec<u8> {
    let pixels = (0..16 * 16 * 3)
        .map(|idx| u8::try_from((idx * 13 + idx / 3) & 0xff).expect("masked fixture byte"))
        .collect::<Vec<_>>();
    encode_ht_codestream(&pixels, 16, 16, 3, 8)
}

fn classic_rgb16_fixture() -> Vec<u8> {
    let pixels = (0..16 * 16 * 3)
        .map(|idx| u8::try_from((idx * 7 + idx / 5) & 0xff).expect("masked fixture byte"))
        .collect::<Vec<_>>();
    encode_codestream(&pixels, 16, 16, 3, 8)
}

fn ht_rgb_jph_fixture() -> Vec<u8> {
    wrap_j2k_codestream(&ht_rgb_fixture(), J2kFileWrapOptions::jph()).expect("wrap JPH")
}

fn decode_rgb8_reference(bytes: &[u8]) -> (Vec<u8>, usize) {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let (width, height) = decoder.info().dimensions;
    let stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut out = vec![0_u8; stride * height as usize];
    decoder
        .decode_into(&mut out, stride, PixelFormat::Rgb8)
        .expect("decode reference");
    (out, stride)
}

fn assert_region_scaled_batch_matches_single_decode(bytes: &[u8], fmt: PixelFormat) {
    const JOBS: usize = 8;
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);
    let stride = scaled_roi.w as usize * fmt.bytes_per_pixel();

    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut expected = vec![0_u8; stride * scaled_roi.h as usize];
    decoder
        .decode_region_scaled_into(&mut pool, &mut expected, stride, fmt, roi, scale)
        .expect("decode reference");

    let mut outputs = (0..JOBS)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions::new(NonZeroUsize::new(4));

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: bytes,
                out: out.as_mut_slice(),
                stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, fmt, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for outcome in &outcomes {
        assert_eq!(outcome.decoded, scaled_roi);
    }
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_decode_empty_input_succeeds() {
    let mut jobs: Vec<TileDecodeJob<'_, '_>> = Vec::new();

    let outcomes = decode_tiles_into(&mut jobs, PixelFormat::Rgb8, TileBatchOptions::default())
        .expect("empty batch succeeds");

    assert!(outcomes.is_empty());
}

#[test]
fn production_batch_decode_worker_one_matches_single_tile_decode() {
    let codestream = rgb_fixture();
    let (expected, stride) = decode_rgb8_reference(&codestream);
    let mut actual = vec![0_u8; expected.len()];
    let options = TileBatchOptions::new(NonZeroUsize::new(1));

    let outcomes = {
        let mut jobs = vec![TileDecodeJob {
            input: &codestream,
            out: actual.as_mut_slice(),
            stride,
        }];
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), 1);
    assert_eq!(actual, expected);
}

#[test]
fn production_batch_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 16;
    let codestream = rgb_fixture();
    let (expected, stride) = decode_rgb8_reference(&codestream);
    let mut outputs = (0..JOBS)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions::new(NonZeroUsize::new(4));

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: codestream.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_decode_matches_repeated_single_tile_decodes() {
    let inputs = [
        rgb_fixture(),
        encode_codestream(&(48_u8..96).collect::<Vec<_>>(), 4, 4, 3, 8),
        encode_codestream(&(96_u8..144).collect::<Vec<_>>(), 4, 4, 3, 8),
    ];
    let expected = inputs
        .iter()
        .map(|input| decode_rgb8_reference(input).0)
        .collect::<Vec<_>>();
    let stride = 4 * PixelFormat::Rgb8.bytes_per_pixel();
    let mut outputs = expected
        .iter()
        .map(|tile| vec![0_u8; tile.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions::new(NonZeroUsize::new(2));

    let outcomes = {
        let mut jobs = inputs
            .iter()
            .zip(outputs.iter_mut())
            .map(|(input, out)| TileDecodeJob {
                input: input.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), inputs.len());
    assert_eq!(outputs, expected);
}

#[test]
fn production_batch_region_scaled_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 12;
    let codestream = rgb_fixture();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut expected = vec![0_u8; stride * scaled_roi.h as usize];
    decoder
        .decode_region_scaled_into(
            &mut pool,
            &mut expected,
            stride,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("decode reference");

    let mut outputs = (0..JOBS)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions::new(NonZeroUsize::new(3));

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: codestream.as_slice(),
                out: out.as_mut_slice(),
                stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("batch decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for outcome in &outcomes {
        assert_eq!(outcome.decoded, scaled_roi);
    }
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_region_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 12;
    let codestream = rgb_fixture();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let stride = roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut expected = vec![0_u8; stride * roi.h as usize];
    decoder
        .decode_region_into(&mut pool, &mut expected, stride, PixelFormat::Rgb8, roi)
        .expect("decode reference");

    let mut outputs = (0..JOBS)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions::new(NonZeroUsize::new(3));

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileRegionDecodeJob {
                input: codestream.as_slice(),
                out: out.as_mut_slice(),
                stride,
                roi,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for outcome in &outcomes {
        assert_eq!(outcome.decoded, roi);
    }
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_scaled_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 12;
    let codestream = ht_rgb_fixture();
    let scale = Downscale::Half;
    let scaled = Rect::full((16, 16)).scaled_covering(scale);
    let stride = scaled.w as usize * PixelFormat::Rgb8.bytes_per_pixel();

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut expected = vec![0_u8; stride * scaled.h as usize];
    decoder
        .decode_scaled_into(&mut pool, &mut expected, stride, PixelFormat::Rgb8, scale)
        .expect("decode reference");

    let mut outputs = (0..JOBS)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions::new(NonZeroUsize::new(3));

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileScaledDecodeJob {
                input: codestream.as_slice(),
                out: out.as_mut_slice(),
                stride,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for outcome in &outcomes {
        assert_eq!(outcome.decoded, scaled);
    }
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_region_scaled_htj2k_rgb_matches_single_decode() {
    const JOBS: usize = 12;
    let codestream = ht_rgb_fixture();
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();

    let mut decoder = J2kDecoder::new(&codestream).expect("decoder");
    let mut pool = j2k::J2kScratchPool::new();
    let mut expected = vec![0_u8; stride * scaled_roi.h as usize];
    decoder
        .decode_region_scaled_into(
            &mut pool,
            &mut expected,
            stride,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("decode reference");

    let mut outputs = (0..JOBS)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions::new(NonZeroUsize::new(8));

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: codestream.as_slice(),
                out: out.as_mut_slice(),
                stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("batch decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for outcome in &outcomes {
        assert_eq!(outcome.decoded, scaled_roi);
    }
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn region_scaled_batch_mixes_direct_and_full_claim_fallback_without_deadlock() {
    const JOBS: usize = 12;
    let direct = ht_rgb_fixture();
    let fallback = classic_rgb16_fixture();
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let scale = Downscale::Half;
    let scaled_roi = roi.scaled_covering(scale);
    let stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();

    let expected = [&direct, &fallback].map(|input| {
        let mut decoder = J2kDecoder::new(input).expect("decoder");
        let mut pool = j2k::J2kScratchPool::new();
        let mut out = vec![0_u8; stride * scaled_roi.h as usize];
        decoder
            .decode_region_scaled_into(&mut pool, &mut out, stride, PixelFormat::Rgb8, roi, scale)
            .expect("reference decode");
        out
    });
    let inputs = (0..JOBS)
        .map(|index| {
            if index % 2 == 0 {
                direct.as_slice()
            } else {
                fallback.as_slice()
            }
        })
        .collect::<Vec<_>>();
    let mut outputs = (0..JOBS)
        .map(|_| vec![0_u8; stride * scaled_roi.h as usize])
        .collect::<Vec<_>>();
    let mut jobs = inputs
        .iter()
        .zip(&mut outputs)
        .map(|(input, out)| TileRegionScaledDecodeJob {
            input,
            out,
            stride,
            roi,
            scale,
        })
        .collect::<Vec<_>>();

    let outcomes = decode_tiles_region_scaled_into(
        &mut jobs,
        PixelFormat::Rgb8,
        TileBatchOptions::new(NonZeroUsize::new(8)),
    )
    .expect("mixed direct/fallback batch");

    assert_eq!(outcomes.len(), JOBS);
    for (index, output) in outputs.iter().enumerate() {
        assert_eq!(output, &expected[index % 2], "tile {index} output diverged");
    }
}

#[test]
fn production_batch_region_scaled_htj2k_jph_rgb_matches_single_decode() {
    let jph = ht_rgb_jph_fixture();

    assert_region_scaled_batch_matches_single_decode(&jph, PixelFormat::Rgb8);
}

#[test]
fn production_batch_region_scaled_htj2k_jph_rgba_matches_single_decode() {
    let jph = ht_rgb_jph_fixture();

    assert_region_scaled_batch_matches_single_decode(&jph, PixelFormat::Rgba8);
}

#[test]
fn production_batch_decode_reports_first_failing_tile_index() {
    let codestream = rgb_fixture();
    let (expected, stride) = decode_rgb8_reference(&codestream);
    let mut outputs = (0..3)
        .map(|_| vec![0_u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions::new(NonZeroUsize::new(2));

    let err = {
        let inputs: [&[u8]; 3] = [codestream.as_slice(), b"not j2k", codestream.as_slice()];
        let mut jobs = inputs
            .into_iter()
            .zip(outputs.iter_mut())
            .map(|(input, out)| TileDecodeJob {
                input,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect_err("bad tile fails")
    };

    match &err {
        j2k::TileBatchError::Tile(indexed) => assert_eq!(indexed.index, 1),
        j2k::TileBatchError::Infrastructure(error) => {
            panic!("codec failure became infrastructure error: {error}")
        }
        _ => panic!("codec failure returned an unknown batch error variant"),
    }
    assert!(err.infrastructure_error().is_none());

    let _: &j2k_core::BatchDecodeError<j2k::J2kError> = &err;
}
