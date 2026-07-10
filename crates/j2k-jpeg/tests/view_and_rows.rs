// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for the parsed-view API and row-streaming decode surface.

use j2k_jpeg::{
    ComponentRowWriter, DecodeRequest, Decoder, Downscale, JpegError, JpegView, PixelFormat, Rect,
    RowSink, ScratchPool,
};
use j2k_test_support::restart_coded_grayscale_jpeg;

use fixtures::{
    baseline_422_16x8_jpeg, baseline_444_8x8_jpeg, cmyk_8x8_jpeg, grayscale_8x8_jpeg,
    lossless_predictor_grayscale_16bit_3x3_jpeg, lossless_predictor_grayscale_3x3_jpeg,
    lossless_predictor_rgb_16bit_3x3_jpeg, lossless_predictor_rgb_3x3_jpeg,
    lossless_predictor_ycbcr_16bit_3x3_jpeg, lossless_predictor_ycbcr_3x3_jpeg,
    lossless_restart_predictor_grayscale_16bit_3x3_jpeg,
    lossless_restart_predictor_grayscale_3x3_jpeg, lossless_restart_predictor_rgb_16bit_3x3_jpeg,
    lossless_restart_predictor_rgb_3x3_jpeg, lossless_restart_predictor_ycbcr_16bit_3x3_jpeg,
    lossless_restart_predictor_ycbcr_3x3_jpeg, lossless_ycbcr_16bit_3x3_rgb16,
    lossless_ycbcr_3x3_rgb8, minimal_baseline_420_jpeg, rgb_app14_8x8_jpeg, ycck_8x8_jpeg,
    LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS, LOSSLESS_GRAYSCALE_3X3_PIXELS,
    LOSSLESS_RGB_16BIT_3X3_PIXELS, LOSSLESS_RGB_3X3_PIXELS,
};
use j2k_test_support as fixtures;

#[derive(Default)]
struct CollectRows {
    rows: Vec<(u32, Vec<u8>)>,
}

impl RowSink<u8> for CollectRows {
    type Error = JpegError;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), JpegError> {
        self.rows.push((y, row.to_vec()));
        Ok(())
    }
}

#[derive(Default)]
struct CollectGrayComponentRows {
    rows: Vec<(u32, Vec<u8>)>,
}

impl ComponentRowWriter for CollectGrayComponentRows {
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
        self.rows.push((y, gray_row.to_vec()));
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        _y: u32,
        _y_row: &[u8],
        _cb_row: &[u8],
        _cr_row: &[u8],
    ) -> Result<(), JpegError> {
        unreachable!("grayscale test writer should not receive ycbcr rows");
    }

    fn write_rgb_row(
        &mut self,
        _y: u32,
        _r_row: &[u8],
        _g_row: &[u8],
        _b_row: &[u8],
    ) -> Result<(), JpegError> {
        unreachable!("grayscale test writer should not receive rgb rows");
    }
}

#[derive(Default)]
struct CollectRgbComponentRows {
    rows: Vec<(u32, Vec<u8>)>,
}

impl ComponentRowWriter for CollectRgbComponentRows {
    fn write_gray_row(&mut self, _y: u32, _gray_row: &[u8]) -> Result<(), JpegError> {
        unreachable!("RGB component-row test writer should not receive gray rows");
    }

    fn write_ycbcr_row(
        &mut self,
        _y: u32,
        _y_row: &[u8],
        _cb_row: &[u8],
        _cr_row: &[u8],
    ) -> Result<(), JpegError> {
        unreachable!("RGB component-row test writer should not receive ycbcr rows");
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError> {
        let mut row = Vec::with_capacity(r_row.len() * 3);
        for ((r, g), b) in r_row.iter().zip(g_row).zip(b_row) {
            row.extend_from_slice(&[*r, *g, *b]);
        }
        self.rows.push((y, row));
        Ok(())
    }
}

#[test]
fn jpeg_view_parse_matches_decoder_inspect() {
    let bytes = minimal_baseline_420_jpeg();
    let view = JpegView::parse(&bytes).expect("parsed view must construct");
    let info = Decoder::inspect(&bytes).expect("inspect must succeed");
    assert_eq!(view.info(), &info);
}

#[test]
fn decoder_from_view_matches_decoder_new_rgb_output() {
    let bytes = rgb_app14_8x8_jpeg();
    let dec_from_new = Decoder::new(&bytes).expect("decoder::new must succeed");
    let dec_from_view = Decoder::from_view(JpegView::parse(&bytes).unwrap())
        .expect("decoder::from_view must succeed");

    let (w, h) = dec_from_new.info().dimensions;
    let stride = (w * 3) as usize;
    let mut new_out = vec![0u8; stride * h as usize];
    let mut view_out = vec![0u8; stride * h as usize];

    dec_from_new
        .decode_scaled_into(&mut new_out, stride, PixelFormat::Rgb8, Downscale::None)
        .unwrap();
    dec_from_view
        .decode_scaled_into(&mut view_out, stride, PixelFormat::Rgb8, Downscale::None)
        .unwrap();

    assert_eq!(view_out, new_out);
}

#[test]
fn decode_rows_matches_decode_into_rgb8_for_ycbcr_sampling_modes() {
    for (route, bytes) in [
        ("4:2:0", minimal_baseline_420_jpeg()),
        ("4:2:2", baseline_422_16x8_jpeg()),
        ("4:4:4", baseline_444_8x8_jpeg()),
    ] {
        let dec = Decoder::new(&bytes).expect("decoder::new must succeed");
        let (w, h) = dec.info().dimensions;
        let stride = (w * 3) as usize;

        let mut expected = vec![0u8; stride * h as usize];
        dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, Downscale::None)
            .unwrap();

        let mut sink = CollectRows::default();
        dec.decode_rows(&mut sink)
            .expect("decode_rows must succeed");

        assert_eq!(sink.rows.len(), h as usize, "{route} row count");
        for (row_idx, (y, row)) in sink.rows.iter().enumerate() {
            assert_eq!(*y as usize, row_idx, "{route} row index");
            assert_eq!(row.len(), stride, "{route} row length");
            assert_eq!(
                row.as_slice(),
                &expected[row_idx * stride..(row_idx + 1) * stride],
                "{route} row {row_idx}"
            );
        }
    }
}

#[test]
fn decode_rows_matches_decode_into_rgb8_for_grayscale_input() {
    let bytes = grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("decoder::new must succeed");
    let (w, h) = dec.info().dimensions;
    let stride = (w * 3) as usize;

    let mut expected = vec![0u8; stride * h as usize];
    dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, Downscale::None)
        .unwrap();

    let mut sink = CollectRows::default();
    dec.decode_rows(&mut sink)
        .expect("decode_rows must succeed");

    assert_eq!(sink.rows.len(), h as usize);
    for (row_idx, (y, row)) in sink.rows.iter().enumerate() {
        assert_eq!(*y as usize, row_idx);
        assert_eq!(
            row.as_slice(),
            &expected[row_idx * stride..(row_idx + 1) * stride]
        );
    }
}

#[test]
fn decode_rows_matches_decode_into_rgb8_for_cmyk_and_ycck() {
    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("CMYK/YCCK decoder must construct");
        let (w, h) = dec.info().dimensions;
        let stride = (w * 3) as usize;

        let mut expected = vec![0u8; stride * h as usize];
        dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, Downscale::None)
            .expect("full CMYK/YCCK RGB8 decode must succeed");

        let mut sink = CollectRows::default();
        dec.decode_rows(&mut sink)
            .expect("CMYK/YCCK decode_rows must succeed");

        assert_eq!(sink.rows.len(), h as usize);
        for (row_idx, (y, row)) in sink.rows.iter().enumerate() {
            assert_eq!(*y as usize, row_idx);
            assert_eq!(row.len(), stride);
            assert_eq!(
                row.as_slice(),
                &expected[row_idx * stride..(row_idx + 1) * stride]
            );
        }
    }
}

#[test]
fn decode_rows_expands_lossless_gray8_common_predictors() {
    let expected = expand_gray_to_rgb(&LOSSLESS_GRAYSCALE_3X3_PIXELS);
    for predictor in 1..=7 {
        for bytes in [
            lossless_predictor_grayscale_3x3_jpeg(predictor),
            lossless_restart_predictor_grayscale_3x3_jpeg(predictor),
        ] {
            let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
                panic!("SOF3 grayscale predictor-{predictor} decoder must construct: {err}")
            });
            let mut sink = CollectRows::default();

            dec.decode_rows(&mut sink).unwrap_or_else(|err| {
                panic!("SOF3 grayscale predictor-{predictor} decode_rows must succeed: {err}")
            });

            assert_eq!(flatten_rows(&sink.rows, 3 * 3), expected);
        }
    }
}

#[test]
fn decode_rows_streams_lossless_gray16_common_predictors() {
    let expected = gray16_samples_to_le_bytes(&LOSSLESS_GRAYSCALE_16BIT_3X3_PIXELS);
    for predictor in 1..=7 {
        for bytes in [
            lossless_predictor_grayscale_16bit_3x3_jpeg(predictor),
            lossless_restart_predictor_grayscale_16bit_3x3_jpeg(predictor),
        ] {
            let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
                panic!("SOF3 Gray16 predictor-{predictor} decoder must construct: {err}")
            });
            let mut sink = CollectRows::default();

            dec.decode_rows(&mut sink).unwrap_or_else(|err| {
                panic!("SOF3 Gray16 predictor-{predictor} decode_rows must succeed: {err}")
            });

            assert_eq!(flatten_rows(&sink.rows, 3 * 2), expected);
        }
    }
}

#[test]
fn decode_rows_matches_lossless_app14_rgb_common_predictors() {
    for predictor in 1..=7 {
        for bytes in [
            lossless_predictor_rgb_3x3_jpeg(predictor),
            lossless_restart_predictor_rgb_3x3_jpeg(predictor),
        ] {
            let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
                panic!("SOF3 APP14 RGB predictor-{predictor} decoder must construct: {err}")
            });
            let mut sink = CollectRows::default();

            dec.decode_rows(&mut sink).unwrap_or_else(|err| {
                panic!("SOF3 APP14 RGB predictor-{predictor} decode_rows must succeed: {err}")
            });

            assert_eq!(flatten_rows(&sink.rows, 3 * 3), LOSSLESS_RGB_3X3_PIXELS);
        }
    }
}

#[test]
fn decode_rows_matches_lossless_ycbcr_rgb8_common_predictors() {
    let expected = lossless_ycbcr_3x3_rgb8();
    for predictor in 1..=7 {
        for bytes in [
            lossless_predictor_ycbcr_3x3_jpeg(predictor),
            lossless_restart_predictor_ycbcr_3x3_jpeg(predictor),
        ] {
            let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
                panic!("SOF3 YCbCr predictor-{predictor} decoder must construct: {err}")
            });
            let mut sink = CollectRows::default();

            dec.decode_rows(&mut sink).unwrap_or_else(|err| {
                panic!("SOF3 YCbCr predictor-{predictor} decode_rows must succeed: {err}")
            });

            assert_eq!(flatten_rows(&sink.rows, 3 * 3), expected);
        }
    }
}

#[test]
fn decode_rows_matches_lossless_app14_rgb16_common_predictors() {
    let expected = rgb16_samples_to_le_bytes(&LOSSLESS_RGB_16BIT_3X3_PIXELS);
    for predictor in 1..=7 {
        for bytes in [
            lossless_predictor_rgb_16bit_3x3_jpeg(predictor),
            lossless_restart_predictor_rgb_16bit_3x3_jpeg(predictor),
        ] {
            let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
                panic!("SOF3 APP14 RGB16 predictor-{predictor} decoder must construct: {err}")
            });
            let mut sink = CollectRows::default();

            dec.decode_rows(&mut sink).unwrap_or_else(|err| {
                panic!("SOF3 APP14 RGB16 predictor-{predictor} decode_rows must succeed: {err}")
            });

            assert_eq!(flatten_rows(&sink.rows, 3 * 6), expected);
        }
    }
}

#[test]
fn decode_rows_matches_lossless_ycbcr16_rgb16_common_predictors() {
    let expected = lossless_ycbcr_16bit_3x3_rgb16();
    for predictor in 1..=7 {
        for bytes in [
            lossless_predictor_ycbcr_16bit_3x3_jpeg(predictor),
            lossless_restart_predictor_ycbcr_16bit_3x3_jpeg(predictor),
        ] {
            let dec = Decoder::new(&bytes).unwrap_or_else(|err| {
                panic!("SOF3 YCbCr16 predictor-{predictor} decoder must construct: {err}")
            });
            let mut sink = CollectRows::default();

            dec.decode_rows(&mut sink).unwrap_or_else(|err| {
                panic!("SOF3 YCbCr16 predictor-{predictor} decode_rows must succeed: {err}")
            });

            assert_eq!(flatten_rows(&sink.rows, 3 * 6), expected);
        }
    }
}

#[test]
fn decode_rows_matches_decode_into_rgb8_for_restart_coded_grayscale_wsi_shape() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let dec = Decoder::new(&bytes).expect("restart-coded grayscale fixture must parse");
    let (w, h) = dec.info().dimensions;
    let stride = (w * 3) as usize;

    let mut expected = vec![0u8; stride * h as usize];
    dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, Downscale::None)
        .expect("full decode must succeed");

    let mut sink = CollectRows::default();
    dec.decode_rows(&mut sink)
        .expect("decode_rows must succeed on restart-coded input");

    assert_eq!(sink.rows.len(), h as usize);
    for (row_idx, (y, row)) in sink.rows.iter().enumerate() {
        assert_eq!(*y as usize, row_idx);
        assert_eq!(row.len(), stride);
        assert_eq!(
            row.as_slice(),
            &expected[row_idx * stride..(row_idx + 1) * stride]
        );
    }
}

fn flatten_rows(rows: &[(u32, Vec<u8>)], stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() * stride);
    for (row_idx, (y, row)) in rows.iter().enumerate() {
        assert_eq!(*y as usize, row_idx);
        assert_eq!(row.len(), stride);
        out.extend_from_slice(row);
    }
    out
}

fn expand_gray_to_rgb(gray: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(gray.len() * 3);
    for &sample in gray {
        out.extend_from_slice(&[sample, sample, sample]);
    }
    out
}

fn gray16_samples_to_le_bytes(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

fn rgb16_samples_to_le_bytes(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

#[test]
fn region_component_rows_scaled_matches_gray_region_decode_for_restart_fixture() {
    let bytes = restart_coded_grayscale_jpeg(24, 24);
    let dec = Decoder::new(&bytes).expect("restart-coded grayscale fixture must parse");
    let roi = Rect {
        x: 5,
        y: 6,
        w: 11,
        h: 10,
    };

    let mut pool = ScratchPool::new();
    let mut sink = CollectGrayComponentRows::default();
    dec.decode_region_component_rows_with_scratch(&mut pool, &mut sink, roi, Downscale::Half)
        .expect("scaled region component rows must decode");

    let expected = dec
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Gray8,
            roi,
            Downscale::Half,
        ))
        .expect("scaled region decode must succeed")
        .0;

    let mut collected = Vec::new();
    for (row_idx, (y, row)) in sink.rows.iter().enumerate() {
        assert_eq!(*y as usize, row_idx);
        collected.extend_from_slice(row);
    }

    assert_eq!(collected, expected);
}

#[test]
fn region_component_rows_scaled_match_cmyk_ycck_region_decode() {
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };

    for bytes in [cmyk_8x8_jpeg(), ycck_8x8_jpeg()] {
        let dec = Decoder::new(&bytes).expect("CMYK/YCCK decoder must construct");
        let mut pool = ScratchPool::new();
        let mut sink = CollectRgbComponentRows::default();
        dec.decode_region_component_rows_with_scratch(&mut pool, &mut sink, roi, Downscale::Half)
            .expect("CMYK/YCCK scaled region component rows must decode");

        let expected = dec
            .decode_request(DecodeRequest::region_scaled(
                PixelFormat::Rgb8,
                roi,
                Downscale::Half,
            ))
            .expect("CMYK/YCCK scaled region decode must succeed")
            .0;

        let mut collected = Vec::new();
        for (row_idx, (y, row)) in sink.rows.iter().enumerate() {
            assert_eq!(*y as usize, row_idx);
            collected.extend_from_slice(row);
        }

        assert_eq!(collected, expected);
    }
}
