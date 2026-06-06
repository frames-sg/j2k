// SPDX-License-Identifier: Apache-2.0

//! Batch decode via [`Decoder::decode_tile`]: sequential output must match
//! parallel output byte-for-byte across a worker pool. Validates the
//! Phase 5 tile primitive under `std::thread::scope`.

use signinum_jpeg::{
    decode_tile_into_in_context, decode_tile_region_scaled_into_in_context, decode_tiles_into,
    decode_tiles_into_with_options, decode_tiles_region_scaled_into, decode_tiles_scaled_into,
    decode_tiles_scaled_into_with_options, ColorTransform, DecodeOptions, Decoder, DecoderContext,
    Downscale, JpegBatchSession, JpegOutputBuffer, PixelFormat, Rect, RowSink, ScratchPool,
    TileBatchOptions, TileDecodeJob, TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
mod fixtures;
use fixtures::{
    cmyk_8x8_jpeg, extended_12bit_rgb_8x8_jpeg, extended_12bit_rgb_8x8_rgb16,
    extended_12bit_ycbcr_420_32x32_jpeg, extended_12bit_ycbcr_420_32x32_rgb16,
    extended_12bit_ycbcr_420_restart_32x32_jpeg, extended_12bit_ycbcr_420_restart_32x32_rgb16,
    extended_12bit_ycbcr_422_32x8_jpeg, extended_12bit_ycbcr_422_32x8_rgb16,
    extended_12bit_ycbcr_8x8_jpeg, extended_12bit_ycbcr_8x8_rgb16, four_component_8x8_rgb,
    lossless_predictor_rgb_16bit_3x3_jpeg, lossless_restart_predictor_rgb_16bit_3x3_jpeg,
    progressive_12bit_grayscale_8x8_jpeg, progressive_12bit_rgb_8x8_jpeg,
    progressive_12bit_ycbcr_420_32x32_jpeg, progressive_12bit_ycbcr_422_32x8_jpeg,
    progressive_12bit_ycbcr_8x8_jpeg, progressive_8x8_jpeg, ycck_8x8_jpeg,
    LOSSLESS_RGB_16BIT_3X3_PIXELS,
};
use std::num::NonZeroUsize;
use std::thread;

const BASELINE_420: &[u8] = include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");

const BATCH_SIZE: usize = 100;

#[derive(Default)]
struct CollectRows {
    rows: Vec<u8>,
}

impl RowSink<u8> for CollectRows {
    type Error = signinum_jpeg::JpegError;

    fn write_row(&mut self, _y: u32, row: &[u8]) -> Result<(), signinum_jpeg::JpegError> {
        self.rows.extend_from_slice(row);
        Ok(())
    }
}

fn decode_tile_bytes(bytes: &[u8], ctx: &mut DecoderContext, pool: &mut ScratchPool) -> Vec<u8> {
    let mut sink = CollectRows::default();
    Decoder::decode_tile(bytes, ctx, pool, &mut sink).expect("Decoder::decode_tile");
    sink.rows
}

fn decode_tile_rgb8_reference(bytes: &[u8]) -> (Vec<u8>, usize) {
    let dec = Decoder::new(bytes).expect("fixture decoder");
    let (width, height) = dec.info().dimensions;
    let stride = width as usize * 3;
    let mut out = vec![0u8; stride * height as usize];
    dec.decode_into(&mut out, stride, PixelFormat::Rgb8)
        .expect("fixture decode_into");
    (out, stride)
}

fn rgb16_samples_to_le_bytes(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
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
    let (expected, stride) = decode_tile_rgb8_reference(BASELINE_420);
    let mut actual = vec![0u8; expected.len()];
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(1),
    };

    let outcomes = {
        let mut jobs = vec![TileDecodeJob {
            input: BASELINE_420,
            out: actual.as_mut_slice(),
            stride,
        }];
        decode_tiles_into(&mut jobs, PixelFormat::Rgb8, options).expect("batch decode")
    };

    assert_eq!(outcomes.len(), 1);
    assert_eq!(actual, expected);
}

#[test]
fn production_batch_decode_progressive8_matches_single_tile_decode() {
    let bytes = progressive_8x8_jpeg();
    let (expected, stride) = decode_tile_rgb8_reference(&bytes);
    let mut actual = vec![0u8; expected.len()];

    let outcomes = {
        let mut jobs = vec![TileDecodeJob {
            input: &bytes,
            out: actual.as_mut_slice(),
            stride,
        }];
        decode_tiles_into(
            &mut jobs,
            PixelFormat::Rgb8,
            TileBatchOptions {
                workers: NonZeroUsize::new(1),
            },
        )
        .expect("progressive batch decode")
    };

    assert_eq!(outcomes.len(), 1);
    assert_eq!(actual, expected);
}

#[test]
fn session_batch_scaled_and_region_scaled_progressive8_matches_single_tile_decode() {
    let bytes = progressive_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("progressive decoder");
    let scale = Downscale::Half;
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_full = (
        dec.info().dimensions.0.div_ceil(2),
        dec.info().dimensions.1.div_ceil(2),
    );
    let scaled_roi = Rect {
        x: roi.x / 2,
        y: roi.y / 2,
        w: (roi.x + roi.w).div_ceil(2) - roi.x / 2,
        h: (roi.y + roi.h).div_ceil(2) - roi.y / 2,
    };
    let scaled_stride = scaled_full.0 as usize * 3;
    let region_stride = scaled_roi.w as usize * 3;
    let mut expected_scaled = vec![0u8; scaled_stride * scaled_full.1 as usize];
    let mut expected_region = vec![0u8; region_stride * scaled_roi.h as usize];
    dec.decode_scaled_into(
        &mut expected_scaled,
        scaled_stride,
        PixelFormat::Rgb8,
        scale,
    )
    .expect("progressive scaled reference");
    dec.decode_region_scaled_into(
        &mut expected_region,
        region_stride,
        PixelFormat::Rgb8,
        roi,
        scale,
    )
    .expect("progressive region-scaled reference");

    let mut actual_scaled = vec![0u8; expected_scaled.len()];
    let mut actual_region = vec![0u8; expected_region.len()];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    {
        let mut jobs = vec![TileScaledDecodeJob {
            input: &bytes,
            out: actual_scaled.as_mut_slice(),
            stride: scaled_stride,
            scale,
        }];
        session
            .decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8)
            .expect("progressive session scaled batch");
    }
    {
        let mut jobs = vec![TileRegionScaledDecodeJob {
            input: &bytes,
            out: actual_region.as_mut_slice(),
            stride: region_stride,
            roi,
            scale,
        }];
        session
            .decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8)
            .expect("progressive session region-scaled batch");
    }

    assert_eq!(actual_scaled, expected_scaled);
    assert_eq!(actual_region, expected_region);
}

#[test]
fn session_batch_scaled_and_region_scaled_progressive12_matches_single_tile_decode() {
    let bytes = progressive_12bit_grayscale_8x8_jpeg();
    let dec = Decoder::new(&bytes).expect("12-bit progressive decoder");
    let scale = Downscale::Half;
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_full = (
        dec.info().dimensions.0.div_ceil(2),
        dec.info().dimensions.1.div_ceil(2),
    );
    let scaled_roi = Rect {
        x: roi.x / 2,
        y: roi.y / 2,
        w: (roi.x + roi.w).div_ceil(2) - roi.x / 2,
        h: (roi.y + roi.h).div_ceil(2) - roi.y / 2,
    };
    let scaled_stride = scaled_full.0 as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let region_stride = scaled_roi.w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut expected_scaled = vec![0u8; scaled_stride * scaled_full.1 as usize];
    let mut expected_region = vec![0u8; region_stride * scaled_roi.h as usize];
    dec.decode_scaled_into(
        &mut expected_scaled,
        scaled_stride,
        PixelFormat::Rgb16,
        scale,
    )
    .expect("12-bit progressive scaled reference");
    dec.decode_region_scaled_into(
        &mut expected_region,
        region_stride,
        PixelFormat::Rgb16,
        roi,
        scale,
    )
    .expect("12-bit progressive region-scaled reference");

    let mut actual_scaled = vec![0u8; expected_scaled.len()];
    let mut actual_region = vec![0u8; expected_region.len()];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    {
        let mut jobs = vec![TileScaledDecodeJob {
            input: &bytes,
            out: actual_scaled.as_mut_slice(),
            stride: scaled_stride,
            scale,
        }];
        session
            .decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit progressive session scaled batch");
    }
    {
        let mut jobs = vec![TileRegionScaledDecodeJob {
            input: &bytes,
            out: actual_region.as_mut_slice(),
            stride: region_stride,
            roi,
            scale,
        }];
        session
            .decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit progressive session region-scaled batch");
    }

    assert_eq!(actual_scaled, expected_scaled);
    assert_eq!(actual_region, expected_region);
}

#[test]
fn session_batch_decode_extended12_app14_rgb_matches_single_tile_decode() {
    let bytes = extended_12bit_rgb_8x8_jpeg();
    let expected = extended_12bit_rgb_8x8_rgb16();
    let stride = 8 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit APP14 RGB session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_decode_progressive12_app14_rgb_matches_single_tile_decode() {
    let bytes = progressive_12bit_rgb_8x8_jpeg();
    let expected = extended_12bit_rgb_8x8_rgb16();
    let stride = 8 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit progressive APP14 RGB session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_decode_lossless_app14_rgb16_matches_single_tile_decode() {
    let expected = rgb16_samples_to_le_bytes(&LOSSLESS_RGB_16BIT_3X3_PIXELS);
    let stride = 3 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    for bytes in [
        lossless_predictor_rgb_16bit_3x3_jpeg(1),
        lossless_restart_predictor_rgb_16bit_3x3_jpeg(1),
    ] {
        let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
        let outcomes = {
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileDecodeJob {
                    input: bytes.as_slice(),
                    out: out.as_mut_slice(),
                    stride,
                })
                .collect::<Vec<_>>();
            session
                .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
                .expect("lossless SOF3 APP14 RGB16 session batch decode")
        };

        assert_eq!(outcomes.len(), 2);
        for output in outputs {
            assert_eq!(output, expected);
        }
    }
}

#[test]
fn session_batch_decode_extended12_ycbcr444_matches_single_tile_decode() {
    let bytes = extended_12bit_ycbcr_8x8_jpeg();
    let expected = extended_12bit_ycbcr_8x8_rgb16();
    let stride = 8 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit YCbCr session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_decode_progressive12_ycbcr444_matches_single_tile_decode() {
    let bytes = progressive_12bit_ycbcr_8x8_jpeg();
    let expected = extended_12bit_ycbcr_8x8_rgb16();
    let stride = 8 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit progressive YCbCr session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_decode_extended12_ycbcr422_matches_single_tile_decode() {
    let bytes = extended_12bit_ycbcr_422_32x8_jpeg();
    let expected = extended_12bit_ycbcr_422_32x8_rgb16();
    let stride = 32 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit YCbCr 4:2:2 session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_decode_progressive12_ycbcr422_matches_single_tile_decode() {
    let bytes = progressive_12bit_ycbcr_422_32x8_jpeg();
    let expected = extended_12bit_ycbcr_422_32x8_rgb16();
    let stride = 32 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit progressive YCbCr 4:2:2 session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_decode_extended12_ycbcr420_matches_single_tile_decode() {
    let bytes = extended_12bit_ycbcr_420_32x32_jpeg();
    let expected = extended_12bit_ycbcr_420_32x32_rgb16();
    let stride = 32 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit YCbCr 4:2:0 session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_decode_extended12_restart_ycbcr420_matches_single_tile_decode() {
    let bytes = extended_12bit_ycbcr_420_restart_32x32_jpeg();
    let expected = extended_12bit_ycbcr_420_restart_32x32_rgb16();
    let stride = 32 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit restart YCbCr 4:2:0 session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_decode_progressive12_ycbcr420_matches_single_tile_decode() {
    let bytes = progressive_12bit_ycbcr_420_32x32_jpeg();
    let expected = extended_12bit_ycbcr_420_32x32_rgb16();
    let stride = 32 * PixelFormat::Rgb16.bytes_per_pixel();
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: bytes.as_slice(),
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb16)
            .expect("12-bit progressive YCbCr 4:2:0 session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_decode_converts_cmyk_and_ycck() {
    let inputs = [cmyk_8x8_jpeg(), ycck_8x8_jpeg()];
    let expected = four_component_8x8_rgb();
    let stride = 8 * 3;
    let mut outputs = vec![vec![0u8; expected.len()], vec![0u8; expected.len()]];
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let outcomes = {
        let mut jobs = inputs
            .iter()
            .zip(outputs.iter_mut())
            .map(|(input, out)| TileDecodeJob {
                input,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
            .expect("CMYK/YCCK session batch decode")
    };

    assert_eq!(outcomes.len(), 2);
    for output in outputs {
        assert_eq!(output, expected);
    }
}

#[test]
fn session_batch_scaled_and_region_scaled_cmyk_ycck_match_free_batch() {
    let inputs = [cmyk_8x8_jpeg(), ycck_8x8_jpeg()];
    let scale = Downscale::Half;
    let roi = Rect {
        x: 1,
        y: 1,
        w: 6,
        h: 6,
    };
    let scaled_full = (4, 4);
    let scaled_roi = Rect {
        x: roi.x / 2,
        y: roi.y / 2,
        w: (roi.x + roi.w).div_ceil(2) - roi.x / 2,
        h: (roi.y + roi.h).div_ceil(2) - roi.y / 2,
    };
    let scaled_stride = scaled_full.0 * PixelFormat::Rgb8.bytes_per_pixel();
    let region_stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut scaled_outputs = inputs
        .iter()
        .map(|_| vec![0u8; scaled_stride * scaled_full.1])
        .collect::<Vec<_>>();
    let mut scaled_expected = scaled_outputs.clone();
    let mut region_outputs = inputs
        .iter()
        .map(|_| vec![0u8; region_stride * scaled_roi.h as usize])
        .collect::<Vec<_>>();
    let mut region_expected = region_outputs.clone();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(2),
    };
    let mut session = JpegBatchSession::new(options);

    {
        let mut jobs = inputs
            .iter()
            .zip(scaled_expected.iter_mut())
            .map(|(input, out)| TileScaledDecodeJob {
                input,
                out: out.as_mut_slice(),
                stride: scaled_stride,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("free CMYK/YCCK scaled batch decode");
    }
    {
        let mut jobs = inputs
            .iter()
            .zip(scaled_outputs.iter_mut())
            .map(|(input, out)| TileScaledDecodeJob {
                input,
                out: out.as_mut_slice(),
                stride: scaled_stride,
                scale,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8)
            .expect("session CMYK/YCCK scaled batch decode");
    }
    {
        let mut jobs = inputs
            .iter()
            .zip(region_expected.iter_mut())
            .map(|(input, out)| TileRegionScaledDecodeJob {
                input,
                out: out.as_mut_slice(),
                stride: region_stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("free CMYK/YCCK region-scaled batch decode");
    }
    {
        let mut jobs = inputs
            .iter()
            .zip(region_outputs.iter_mut())
            .map(|(input, out)| TileRegionScaledDecodeJob {
                input,
                out: out.as_mut_slice(),
                stride: region_stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8)
            .expect("session CMYK/YCCK region-scaled batch decode");
    }

    assert_eq!(scaled_outputs, scaled_expected);
    assert_eq!(region_outputs, region_expected);
}

#[test]
fn production_batch_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 32;
    let (expected, stride) = decode_tile_rgb8_reference(BASELINE_420);
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(4),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: BASELINE_420,
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
fn session_batch_decode_reuses_worker_state_across_calls_and_matches_free_batch() {
    const JOBS: usize = 32;
    let (expected, stride) = decode_tile_rgb8_reference(BASELINE_420);
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(4),
    };
    let mut session = JpegBatchSession::new(options);
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();

    for pass in 0..3 {
        for output in &mut outputs {
            output.fill(pass as u8);
        }
        let outcomes = {
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileDecodeJob {
                    input: BASELINE_420,
                    out: out.as_mut_slice(),
                    stride,
                })
                .collect::<Vec<_>>();
            session
                .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
                .expect("session batch decode")
        };

        assert_eq!(outcomes.len(), JOBS);
        assert_eq!(session.worker_count(), 4);
        for (index, out) in outputs.iter().enumerate() {
            assert_eq!(out, &expected, "pass {pass} tile {index} output diverged");
        }
    }
}

#[test]
fn session_batch_decode_reports_first_failing_tile_index() {
    let (expected, stride) = decode_tile_rgb8_reference(BASELINE_420);
    let mut outputs = (0..3)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let mut session = JpegBatchSession::new(TileBatchOptions {
        workers: NonZeroUsize::new(2),
    });

    let err = {
        let inputs: [&[u8]; 3] = [BASELINE_420, b"not a jpeg", BASELINE_420];
        let mut jobs = inputs
            .into_iter()
            .zip(outputs.iter_mut())
            .map(|(input, out)| TileDecodeJob {
                input,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
            .expect_err("bad tile fails")
    };

    assert_eq!(err.index, 1);
}

#[test]
fn session_batch_scaled_and_region_scaled_decode_match_existing_batch_api() {
    const JOBS: usize = 16;
    let dec = Decoder::new(BASELINE_420).expect("fixture decoder");
    let scale = Downscale::Quarter;
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let scaled_full = (
        dec.info().dimensions.0.div_ceil(4),
        dec.info().dimensions.1.div_ceil(4),
    );
    let scaled_roi = Rect {
        x: roi.x / 4,
        y: roi.y / 4,
        w: (roi.x + roi.w).div_ceil(4) - roi.x / 4,
        h: (roi.y + roi.h).div_ceil(4) - roi.y / 4,
    };
    let scaled_stride = scaled_full.0 as usize * 3;
    let region_stride = scaled_roi.w as usize * 3;
    let mut scaled_outputs = (0..JOBS)
        .map(|_| vec![0u8; scaled_stride * scaled_full.1 as usize])
        .collect::<Vec<_>>();
    let mut scaled_expected = scaled_outputs.clone();
    let mut region_outputs = (0..JOBS)
        .map(|_| vec![0u8; region_stride * scaled_roi.h as usize])
        .collect::<Vec<_>>();
    let mut region_expected = region_outputs.clone();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(4),
    };
    let mut session = JpegBatchSession::new(options);

    {
        let mut jobs = scaled_expected
            .iter_mut()
            .map(|out| TileScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride: scaled_stride,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("free scaled batch decode");
    }
    {
        let mut jobs = scaled_outputs
            .iter_mut()
            .map(|out| TileScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride: scaled_stride,
                scale,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8)
            .expect("session scaled batch decode");
    }
    {
        let mut jobs = region_expected
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride: region_stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("free region-scaled batch decode");
    }
    {
        let mut jobs = region_outputs
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride: region_stride,
                roi,
                scale,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8)
            .expect("session region-scaled batch decode");
    }

    assert_eq!(scaled_outputs, scaled_expected);
    assert_eq!(region_outputs, region_expected);
}

#[test]
fn jpeg_output_buffer_resizes_without_reallocating_for_same_or_smaller_shape() {
    let mut buffer = JpegOutputBuffer::new((16, 16), PixelFormat::Rgb8).expect("output buffer");
    let initial_capacity = buffer.capacity();
    assert_eq!(buffer.dimensions(), (16, 16));
    assert_eq!(buffer.stride(), 16 * 3);
    assert_eq!(buffer.as_mut_slice().len(), 16 * 16 * 3);

    buffer
        .resize((8, 8), PixelFormat::Rgb8)
        .expect("smaller resize");

    assert_eq!(buffer.capacity(), initial_capacity);
    assert_eq!(buffer.dimensions(), (8, 8));
    assert_eq!(buffer.as_mut_slice().len(), 8 * 8 * 3);
}

#[test]
fn production_batch_decode_with_options_preserves_forced_color_transform() {
    const JOBS: usize = 8;
    let decode_options = DecodeOptions::default().with_color_transform(ColorTransform::ForceRgb);
    let dec = Decoder::new_with_options(BASELINE_420, decode_options).expect("fixture decoder");
    let (width, height) = dec.info().dimensions;
    let stride = width as usize * 3;
    let mut expected = vec![0u8; stride * height as usize];
    dec.decode_into(&mut expected, stride, PixelFormat::Rgb8)
        .expect("reference forced-RGB decode");
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(2),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride,
            })
            .collect::<Vec<_>>();
        decode_tiles_into_with_options(&mut jobs, PixelFormat::Rgb8, decode_options, options)
            .expect("batch decode with options")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_decode_reports_first_failing_tile_index() {
    let (expected, stride) = decode_tile_rgb8_reference(BASELINE_420);
    let mut outputs = (0..3)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(2),
    };

    let err = {
        let inputs: [&[u8]; 3] = [BASELINE_420, b"not a jpeg", BASELINE_420];
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

    assert_eq!(err.index, 1);
}

#[test]
fn sequential_and_parallel_batch_produce_identical_output() {
    let tiles: Vec<&[u8]> = (0..BATCH_SIZE).map(|_| BASELINE_420).collect();

    let sequential: Vec<Vec<u8>> = {
        let mut pool = ScratchPool::new();
        let mut ctx = DecoderContext::new();
        tiles
            .iter()
            .map(|bytes| decode_tile_bytes(bytes, &mut ctx, &mut pool))
            .collect()
    };

    let parallel: Vec<Vec<u8>> = thread::scope(|scope| {
        const WORKERS: usize = 4;
        let chunk_size = tiles.len().div_ceil(WORKERS);
        let handles: Vec<_> = tiles
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(|| {
                    let mut pool = ScratchPool::new();
                    let mut ctx = DecoderContext::new();
                    chunk
                        .iter()
                        .map(|bytes| decode_tile_bytes(bytes, &mut ctx, &mut pool))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker panicked"))
            .collect()
    });

    assert_eq!(sequential.len(), parallel.len());
    for (i, (seq, par)) in sequential.iter().zip(parallel.iter()).enumerate() {
        assert_eq!(
            seq, par,
            "tile {i} diverged between sequential and parallel"
        );
    }
}

#[test]
fn pool_reuse_across_batch_matches_fresh_pool() {
    let mut reused_pool = ScratchPool::new();
    let mut reused_ctx = DecoderContext::new();
    let reused_outputs: Vec<Vec<u8>> = (0..BATCH_SIZE)
        .map(|_| decode_tile_bytes(BASELINE_420, &mut reused_ctx, &mut reused_pool))
        .collect();

    let fresh_outputs: Vec<Vec<u8>> = (0..BATCH_SIZE)
        .map(|_| {
            let mut pool = ScratchPool::new();
            let mut ctx = DecoderContext::new();
            decode_tile_bytes(BASELINE_420, &mut ctx, &mut pool)
        })
        .collect();

    for (i, (reused, fresh)) in reused_outputs.iter().zip(fresh_outputs.iter()).enumerate() {
        assert_eq!(reused, fresh, "iter {i} reused-pool output diverged");
    }
}

#[test]
fn tile_buffer_decode_matches_decoder_decode_into() {
    let dec = Decoder::new(BASELINE_420).expect("fixture decoder");
    let (width, height) = dec.info().dimensions;
    let stride = width as usize * 3;
    let mut expected = vec![0u8; stride * height as usize];
    let mut actual = vec![0u8; expected.len()];
    dec.decode_into(&mut expected, stride, PixelFormat::Rgb8)
        .expect("baseline decode_into");

    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    decode_tile_into_in_context(
        BASELINE_420,
        &mut ctx,
        &mut pool,
        &mut actual,
        stride,
        PixelFormat::Rgb8,
    )
    .expect("tile decode_into_in_context");

    assert_eq!(actual, expected);
}

#[test]
fn tile_region_scaled_decode_matches_decoder_region_decode() {
    let dec = Decoder::new(BASELINE_420).expect("fixture decoder");
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let denom = 4;
    let scaled_w = (roi.x + roi.w).div_ceil(denom) - roi.x / denom;
    let scaled_h = (roi.y + roi.h).div_ceil(denom) - roi.y / denom;
    let stride = scaled_w as usize * 3;
    let mut expected = vec![0u8; stride * scaled_h as usize];
    let mut actual = vec![0u8; expected.len()];
    dec.decode_region_scaled_into(
        &mut expected,
        stride,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
    )
    .expect("core region-scaled decode");

    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    decode_tile_region_scaled_into_in_context(
        BASELINE_420,
        &mut ctx,
        &mut pool,
        &mut actual,
        stride,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
    )
    .expect("tile region decode_into_in_context");

    assert_eq!(actual, expected);
}

#[test]
fn production_batch_region_scaled_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 32;
    let dec = Decoder::new(BASELINE_420).expect("fixture decoder");
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let denom = 4;
    let scaled_w = (roi.x + roi.w).div_ceil(denom) - roi.x / denom;
    let scaled_h = (roi.y + roi.h).div_ceil(denom) - roi.y / denom;
    let stride = scaled_w as usize * 3;
    let mut expected = vec![0u8; stride * scaled_h as usize];
    dec.decode_region_scaled_into(
        &mut expected,
        stride,
        PixelFormat::Rgb8,
        roi,
        Downscale::Quarter,
    )
    .expect("reference region-scaled decode");
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(4),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileRegionScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride,
                roi,
                scale: Downscale::Quarter,
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("batch region-scaled decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_scaled_decode_parallel_preserves_order_and_output() {
    const JOBS: usize = 32;
    let dec = Decoder::new(BASELINE_420).expect("fixture decoder");
    let scale = Downscale::Quarter;
    let denom = 4;
    let (width, height) = dec.info().dimensions;
    let scaled_w = width.div_ceil(denom);
    let scaled_h = height.div_ceil(denom);
    let stride = scaled_w as usize * 3;
    let mut expected = vec![0u8; stride * scaled_h as usize];
    dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, scale)
        .expect("reference scaled decode");
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(4),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8, options)
            .expect("batch scaled decode")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}

#[test]
fn production_batch_scaled_decode_with_options_preserves_forced_color_transform() {
    const JOBS: usize = 8;
    let decode_options = DecodeOptions::default().with_color_transform(ColorTransform::ForceRgb);
    let dec = Decoder::new_with_options(BASELINE_420, decode_options).expect("fixture decoder");
    let scale = Downscale::Quarter;
    let denom = 4;
    let (width, height) = dec.info().dimensions;
    let scaled_w = width.div_ceil(denom);
    let scaled_h = height.div_ceil(denom);
    let stride = scaled_w as usize * 3;
    let mut expected = vec![0u8; stride * scaled_h as usize];
    dec.decode_scaled_into(&mut expected, stride, PixelFormat::Rgb8, scale)
        .expect("reference scaled forced-RGB decode");
    let mut outputs = (0..JOBS)
        .map(|_| vec![0u8; expected.len()])
        .collect::<Vec<_>>();
    let options = TileBatchOptions {
        workers: NonZeroUsize::new(2),
    };

    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .map(|out| TileScaledDecodeJob {
                input: BASELINE_420,
                out: out.as_mut_slice(),
                stride,
                scale,
            })
            .collect::<Vec<_>>();
        decode_tiles_scaled_into_with_options(&mut jobs, PixelFormat::Rgb8, decode_options, options)
            .expect("batch scaled decode with options")
    };

    assert_eq!(outcomes.len(), JOBS);
    for (index, out) in outputs.iter().enumerate() {
        assert_eq!(out, &expected, "tile {index} output diverged");
    }
}
