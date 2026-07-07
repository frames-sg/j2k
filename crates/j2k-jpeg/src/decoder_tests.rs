// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Warning;
    use crate::output::OutputWriter;
    use alloc::vec;
    use alloc::vec::Vec;

    fn minimal_baseline_jpeg() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]);
        v.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
        v.extend(core::iter::repeat_n(1u8, 64));
        v.extend_from_slice(&[
            0xFF,
            0xC0,
            0x00,
            17,
            8,
            0,
            16,
            0,
            16,
            3,
            1,
            (2 << 4) | 2,
            0,
            2,
            (1 << 4) | 1,
            0,
            3,
            (1 << 4) | 1,
            0,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAA,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xBB,
        ]);
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);
        v.extend_from_slice(&[0x00, 0xFF, 0xD9]);
        v
    }

    fn baseline_jpeg_with_dimensions(width: u16, height: u16) -> Vec<u8> {
        let mut bytes = minimal_baseline_jpeg();
        let sof = bytes
            .windows(2)
            .position(|w| w == [0xFF, 0xC0])
            .expect("SOF0 marker");
        bytes[sof + 5..sof + 7].copy_from_slice(&height.to_be_bytes());
        bytes[sof + 7..sof + 9].copy_from_slice(&width.to_be_bytes());
        bytes
    }

    fn dc_only_420_jpeg(width: u16, height: u16) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]);
        v.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
        v.extend(core::iter::repeat_n(1u8, 64));
        v.extend_from_slice(&[
            0xFF,
            0xC0,
            0x00,
            17,
            8,
            (height >> 8) as u8,
            height as u8,
            (width >> 8) as u8,
            width as u8,
            3,
            1,
            (2 << 4) | 2,
            0,
            2,
            (1 << 4) | 1,
            0,
            3,
            (1 << 4) | 1,
            0,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);

        let mcus_per_row = u32::from(width).div_ceil(16);
        let mcu_rows = u32::from(height).div_ceil(16);
        let entropy_bits = mcus_per_row * mcu_rows * 12;
        let entropy_bytes = (entropy_bits as usize).div_ceil(8) + 8;
        v.extend(core::iter::repeat_n(0u8, entropy_bytes));
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    #[test]
    fn decoder_new_succeeds_on_baseline_stream() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        assert_eq!(dec.info().dimensions, (16, 16));
    }

    #[test]
    fn owned_baseline_decode_with_huge_dimensions_errors_before_allocating() {
        let bytes = baseline_jpeg_with_dimensions(65_500, 65_500);
        let dec = Decoder::new(&bytes).expect("huge baseline header should parse");

        let err = dec.decode_request(DecodeRequest::full(PixelFormat::Rgb8)).unwrap_err();

        assert!(
            matches!(err, JpegError::MemoryCapExceeded { .. }),
            "expected MemoryCapExceeded before owned output allocation, got {err:?}"
        );
    }

    #[test]
    fn owned_baseline_region_decode_with_huge_dimensions_errors_before_allocating() {
        let bytes = baseline_jpeg_with_dimensions(65_500, 65_500);
        let dec = Decoder::new(&bytes).expect("huge baseline header should parse");

        let err = dec
            .decode_request(DecodeRequest::region(PixelFormat::Rgb8, Rect::full(dec.info().dimensions)))
            .unwrap_err();

        assert!(
            matches!(err, JpegError::MemoryCapExceeded { .. }),
            "expected MemoryCapExceeded before region output allocation, got {err:?}"
        );
    }

    fn minimal_lossless_jpeg(
        width: u16,
        height: u16,
        precision: u8,
        sampling_420: bool,
    ) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]);
        let first_sampling = if sampling_420 {
            (2 << 4) | 2
        } else {
            (1 << 4) | 1
        };
        v.extend_from_slice(&[
            0xFF,
            0xC3,
            0x00,
            17,
            precision,
            (height >> 8) as u8,
            height as u8,
            (width >> 8) as u8,
            width as u8,
            3,
            1,
            first_sampling,
            0,
            2,
            (1 << 4) | 1,
            0,
            3,
            (1 << 4) | 1,
            0,
        ]);
        v.extend_from_slice(&[
            0xFF, 0xC4, 0x00, 20, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
        ]);
        // SOS: predictor 1 (ss=1), point transform 0.
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 12, 3, 1, 0x00, 2, 0x00, 3, 0x00, 1, 0, 0]);
        v.extend_from_slice(&[0x00, 0x00, 0xFF, 0xD9]);
        v
    }

    #[test]
    fn lossless_sampled_plan_with_huge_dimensions_is_rejected_by_memory_cap() {
        // 65500x65500 lossless 16-bit 4:2:0: the sampled-color decode path
        // would allocate ~12.9 GB of full-frame planes. Plan build must refuse.
        let bytes = minimal_lossless_jpeg(65_500, 65_500, 16, true);
        match Decoder::new(&bytes) {
            Err(JpegError::MemoryCapExceeded { .. }) => {}
            other => panic!("expected MemoryCapExceeded at plan build, got {other:?}"),
        }
    }

    #[test]
    fn lossless_sampled_plan_with_small_dimensions_still_parses() {
        let bytes = minimal_lossless_jpeg(16, 16, 16, true);
        let dec = Decoder::new(&bytes).expect("small lossless stream should parse");
        assert_eq!(dec.info().dimensions, (16, 16));
        assert!(
            dec.plan.scratch_bytes > 0,
            "sampled lossless plan must report scratch"
        );
    }

    #[test]
    fn extended12_plan_with_huge_dimensions_is_rejected_by_memory_cap() {
        // 65500x65500 Extended12 4:2:0: the sequential 12-bit path allocates
        // full-frame u16 planes (~13 GB). Plan build must refuse.
        let mut bytes = minimal_baseline_jpeg();
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[p + 1] = 0xC1;
        bytes[p + 4] = 12;
        bytes[p + 5] = (65_500u16 >> 8) as u8;
        bytes[p + 6] = 65_500u16 as u8;
        bytes[p + 7] = (65_500u16 >> 8) as u8;
        bytes[p + 8] = 65_500u16 as u8;
        match Decoder::new(&bytes) {
            Err(JpegError::MemoryCapExceeded { .. }) => {}
            other => panic!("expected MemoryCapExceeded at plan build, got {other:?}"),
        }
    }

    #[test]
    fn lossless_gray16_region_scaled_with_huge_dimensions_errors_before_allocating() {
        // Grayscale direct decode streams to the caller's buffer, so plan
        // scratch stays 0 — but the region/scaled path materializes the full
        // frame (~8.6 GB here) and must be stopped at the allocation site.
        let mut bytes = minimal_lossless_jpeg(65_500, 65_500, 16, false);
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xC3]).unwrap();
        // Rewrite SOF to a single grayscale component.
        bytes[p + 3] = 11;
        bytes[p + 9] = 1;
        bytes[p + 10] = 1;
        bytes[p + 11] = (1 << 4) | 1;
        bytes[p + 12] = 0;
        bytes.splice(p + 13..p + 19, core::iter::empty());
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        bytes.splice(
            p..p + 14,
            [0xFF, 0xDA, 0x00, 8, 1, 1, 0x00, 1, 0, 0].iter().copied(),
        );
        let dec = Decoder::new(&bytes).expect("huge lossless gray stream should parse");
        let roi = Rect {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
        };
        let mut out = vec![0u8; 16 * 16 * 2];
        let err = dec
            .decode_region_scaled_into(&mut out, 16 * 2, PixelFormat::Gray16, roi, Downscale::Half)
            .unwrap_err();
        assert!(
            matches!(err, JpegError::MemoryCapExceeded { .. }),
            "expected MemoryCapExceeded, got {err:?}"
        );
    }

    #[test]
    fn decode_into_rejects_unsupported_extended12_ycbcr_sampling_with_not_implemented() {
        let mut bytes = minimal_baseline_jpeg();
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[p + 1] = 0xC1;
        bytes[p + 4] = 12;
        bytes[p + 11] = (1 << 4) | 2;
        let dec = Decoder::new(&bytes).expect("unsupported Extended12 YCbCr sampling should parse");
        let stride = dec.info().dimensions.0 as usize * PixelFormat::Rgb16.bytes_per_pixel();
        let mut out = vec![0u8; stride * dec.info().dimensions.1 as usize];

        let err = dec
            .decode_into(&mut out, stride, PixelFormat::Rgb16)
            .unwrap_err();

        assert!(err.is_not_implemented());
    }

    #[test]
    fn decoder_new_rejects_arithmetic_as_unsupported() {
        let mut bytes = minimal_baseline_jpeg();
        let p = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[p + 1] = 0xC9;
        let err = Decoder::new(&bytes).unwrap_err();
        assert!(err.is_unsupported());
    }

    #[test]
    fn decode_outcome_carries_rect_and_warnings() {
        let outcome = DecodeOutcome {
            decoded: Rect {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
            },
            warnings: vec![Warning::MissingEoi],
        };
        assert_eq!(outcome.decoded.w, 16);
        assert_eq!(outcome.warnings.len(), 1);
    }

    #[test]
    fn decode_into_rejects_undersized_buffer() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        let mut buf = vec![0u8; 4];
        let err = dec
            .decode_into(&mut buf, 48, PixelFormat::Rgb8)
            .unwrap_err();
        assert!(matches!(err, JpegError::OutputBufferTooSmall { .. }));
    }

    #[test]
    fn decode_into_rejects_invalid_stride() {
        let bytes = minimal_baseline_jpeg();
        let dec = Decoder::new(&bytes).unwrap();
        let mut buf = vec![0u8; 16 * 16 * 3];
        let err = dec
            .decode_into(&mut buf, 10, PixelFormat::Rgb8)
            .unwrap_err();
        assert!(matches!(err, JpegError::InvalidStride { .. }));
    }

    #[test]
    fn decode_into_output_format_writes_custom_rgba_alpha() {
        let bytes = dc_only_420_jpeg(16, 16);
        let dec = Decoder::new(&bytes).unwrap();
        let (w, h) = dec.info().dimensions;
        let mut pool = ScratchPool::new();
        let mut buf = vec![0u8; (w * h * 4) as usize];

        dec.decode_into_output_format_with_scratch(
            &mut pool,
            &mut buf,
            (w * 4) as usize,
            OutputFormat::Rgba8 { alpha: 200 },
        )
        .unwrap();

        for y in 0..h as usize {
            for x in 0..w as usize {
                let idx = (y * w as usize + x) * 4;
                assert_eq!(buf[idx + 3], 200, "pixel ({x},{y}) alpha");
            }
        }
    }

    #[test]
    fn decode_region_output_format_writes_custom_rgba_alpha() {
        let bytes = dc_only_420_jpeg(16, 16);
        let dec = Decoder::new(&bytes).unwrap();
        let roi = Rect {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
        };
        let mut pool = ScratchPool::new();
        let mut buf = vec![0u8; (roi.w * roi.h * 4) as usize];

        dec.decode_region_into_output_format_with_scratch(
            &mut pool,
            &mut buf,
            (roi.w * 4) as usize,
            OutputFormat::Rgba8 { alpha: 123 },
            roi,
        )
        .unwrap();

        for y in 0..roi.h as usize {
            for x in 0..roi.w as usize {
                let idx = (y * roi.w as usize + x) * 4;
                assert_eq!(buf[idx + 3], 123, "pixel ({x},{y}) alpha");
            }
        }
    }

    #[test]
    fn large_fast_420_region_decode_populates_cpu_entropy_checkpoints() {
        let bytes = dc_only_420_jpeg(1024, 2048);
        let dec = Decoder::new(&bytes).expect("decoder");
        assert!(dec.plan.matches_fast_tile_shape());

        let roi = Rect {
            x: 64,
            y: 1536,
            w: 64,
            h: 64,
        };
        let mut out = vec![0u8; roi.w as usize * roi.h as usize * 3];
        let mut pool = ScratchPool::new();
        dec.decode_region_into_with_scratch(
            &mut pool,
            &mut out,
            roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("deep ROI decode");

        let cache = dec
            .cpu_entropy_checkpoints
            .lock()
            .expect("checkpoint cache mutex");
        assert!(cache
            .checkpoints
            .iter()
            .any(|checkpoint| checkpoint.mcu_index >= CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS));
    }

    #[derive(Default)]
    struct GrayRows {
        rows: Vec<(u32, Vec<u8>)>,
    }

    impl OutputWriter for GrayRows {
        fn write_rgb_row(
            &mut self,
            _y: u32,
            _r_row: &[u8],
            _g_row: &[u8],
            _b_row: &[u8],
        ) -> Result<(), JpegError> {
            unreachable!("gray test writer should not receive rgb rows");
        }

        fn write_ycbcr_row(
            &mut self,
            _y: u32,
            _y_row: &[u8],
            _cb_row: &[u8],
            _cr_row: &[u8],
        ) -> Result<(), JpegError> {
            unreachable!("gray test writer should not receive ycbcr rows");
        }

        fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError> {
            self.rows.push((y, gray_row.to_vec()));
            Ok(())
        }
    }

    #[test]
    fn cropped_writer_honors_source_window_origin() {
        let inner = GrayRows::default();
        let rect = Rect {
            x: 6,
            y: 1,
            w: 2,
            h: 1,
        };
        let mut writer = CroppedWriter::new(inner, rect, 4, 4);

        writer
            .write_gray_row(1, &[10, 20, 30, 40])
            .expect("crop write must succeed");

        assert_eq!(writer.inner.rows, vec![(0, vec![30, 40])]);
    }
}
