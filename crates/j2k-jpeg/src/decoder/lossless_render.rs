// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless grayscale and color rendering paths.

use super::{
    allocate_output_buffer, checked_scratch_len, convert_ycbcr16_to_rgb16_in_place,
    convert_ycbcr8_to_rgb8_in_place, copy_gray16_scaled_rect, copy_gray8_scaled_rect,
    copy_ycbcr16_row_to_rgb16, copy_ycbcr8_row_to_rgb8, decode_lossless_color_sample,
    decode_lossless_sampled_color_mcu, finish_scan, lossless_color_sampling,
    lossless_predictor_gray_rows, lossless_predictor_value, lossless_predictor_value_u16,
    merged_warnings, scaled_rect_covering, validate_lossless_color_plan,
    write_lossless_color16_sampled_output, write_lossless_color8_sampled_output, BitReader,
    ColorSpace, DecodeOutcome, Decoder, DownscaleFactor, JpegError, LosslessColorIntoSample,
    LosslessColorPlanes, LosslessColorRowSample, LosslessColorSampling, LosslessRestartTracker,
    LosslessSample, LosslessSampledColorPlanesMut, LosslessSampledMcu, Rect, RowSink, Vec,
};

impl Decoder<'_> {
    pub(super) fn decode_lossless_gray8_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if plan.bit_depth != 8 {
            return Err(JpegError::UnsupportedBitDepth {
                depth: plan.bit_depth,
            });
        }
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }

        let (width, height) = plan.dimensions;
        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_samples = width.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_samples);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let sample_index = y as u32 * width + x as u32;
                let restart_first_sample = restart_tracker.begin_unit(&mut br, sample_index)?;
                let predictor = if restart_first_sample {
                    128
                } else {
                    lossless_predictor_value(plan.predictor, out, stride, x, y)
                };
                let diff = plan.dc_table.decode_fast_dc(&mut br)?;
                let sample = <u8 as LosslessSample>::from_i32(predictor + diff)?;
                out[y * stride + x] = sample;
                restart_tracker.finish_unit();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    pub(super) fn decode_lossless_gray8_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        if roi == Rect::full(self.info.dimensions) && downscale == DownscaleFactor::Full {
            return self.decode_lossless_gray8_into(out, stride);
        }

        let (width, height) = self.info.dimensions;
        let full_stride = width as usize;
        let mut full =
            allocate_output_buffer(checked_scratch_len(&[full_stride, height as usize])?);
        let mut outcome = self.decode_lossless_gray8_into(&mut full, full_stride)?;
        let output_rect = scaled_rect_covering(roi, downscale)?;
        copy_gray8_scaled_rect(
            &full,
            (width, height),
            output_rect,
            downscale.denominator(),
            out,
            stride,
        );
        outcome.decoded = roi;
        Ok(outcome)
    }

    pub(super) fn decode_lossless_gray_rows<P, S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        mut emit_row: impl FnMut(&mut S, u32, &[u8]) -> Result<(), JpegError>,
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
        S: RowSink<u8, Error = JpegError>,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if plan.bit_depth != P::BIT_DEPTH {
            return Err(JpegError::UnsupportedBitDepth {
                depth: plan.bit_depth,
            });
        }
        if self.info.color_space != ColorSpace::Grayscale {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }

        let (width, height) = plan.dimensions;
        let width = width as usize;
        let row_len = width.saturating_mul(P::BYTES);
        prev_row.resize(row_len, 0);
        curr_row.resize(row_len, 0);

        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_samples = plan.dimensions.0.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_samples);
        for y in 0..height as usize {
            for x in 0..width {
                let sample_index = y as u32 * plan.dimensions.0 + x as u32;
                let restart_first_sample = restart_tracker.begin_unit(&mut br, sample_index)?;
                let predictor = if restart_first_sample {
                    P::RESTART_PREDICTOR
                } else {
                    lossless_predictor_gray_rows::<P>(plan.predictor, curr_row, prev_row, x, y)
                };
                let diff = plan.dc_table.decode_fast_dc(&mut br)?;
                let sample = P::from_i32(predictor + diff)?;
                sample.write_le(&mut curr_row[x * P::BYTES..]);
                restart_tracker.finish_unit();
            }
            emit_row(sink, y as u32, &curr_row[..row_len])?;
            core::mem::swap(prev_row, curr_row);
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    pub(super) fn decode_lossless_gray8_rows<S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        rgb_row: &mut [u8],
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_gray_rows::<u8, S>(sink, prev_row, curr_row, |sink, y, gray_row| {
            let rgb_len = gray_row.len().saturating_mul(3);
            if rgb_row.len() < rgb_len {
                return Err(JpegError::OutputBufferTooSmall {
                    required: rgb_len,
                    provided: rgb_row.len(),
                });
            }
            for (pixel, &sample) in rgb_row[..rgb_len].chunks_exact_mut(3).zip(gray_row.iter()) {
                pixel.copy_from_slice(&[sample, sample, sample]);
            }
            sink.write_row(y, &rgb_row[..rgb_len])
        })
    }

    pub(super) fn decode_lossless_rgb8_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color8_output_into(out, stride, ColorSpace::Rgb)
    }

    pub(super) fn decode_lossless_color8_output_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        match lossless_color_sampling(&self.info) {
            Some(LosslessColorSampling::S444) => {
                let outcome =
                    self.decode_lossless_color8_components_into(out, stride, color_space)?;
                if color_space == ColorSpace::YCbCr {
                    convert_ycbcr8_to_rgb8_in_place(out, stride, self.info.dimensions);
                }
                Ok(outcome)
            }
            Some(LosslessColorSampling::S422 | LosslessColorSampling::S420) => {
                self.decode_lossless_color8_sampled_into(out, stride, color_space)
            }
            None => Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            }),
        }
    }

    pub(super) fn decode_lossless_color_components_into<P>(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;

        let (width, height) = plan.dimensions;
        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_pixels = width.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_pixels);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let pixel_index = y as u32 * width + x as u32;
                let restart_first_pixel = restart_tracker.begin_unit(&mut br, pixel_index)?;
                decode_lossless_color_sample::<P, _>(
                    &mut br,
                    &self.plan.components,
                    plan.predictor,
                    restart_first_pixel,
                    &mut LosslessColorIntoSample {
                        out: &mut *out,
                        stride,
                        x,
                        y,
                    },
                )?;
                restart_tracker.finish_unit();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    pub(super) fn decode_lossless_color8_components_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_components_into::<u8>(out, stride, color_space)
    }

    pub(super) fn decode_lossless_color_sampled_into<P>(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
        write_output: impl FnOnce(
            &mut [u8],
            usize,
            ColorSpace,
            LosslessColorSampling,
            (usize, usize),
            LosslessColorPlanes<'_, P>,
        ),
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        let sampling = lossless_color_sampling(&self.info).ok_or(JpegError::NotImplemented {
            sof: self.info.sof_kind,
        })?;
        if !matches!(
            sampling,
            LosslessColorSampling::S422 | LosslessColorSampling::S420
        ) {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;

        let (width, height) = plan.dimensions;
        let width = width as usize;
        let height = height as usize;
        let chroma_width = width.div_ceil(self.info.sampling.max_h as usize);
        let chroma_height = height.div_ceil(self.info.sampling.max_v as usize);
        let mut c0 = vec![P::default(); width * height];
        let mut c1 = vec![P::default(); chroma_width * chroma_height];
        let mut c2 = vec![P::default(); chroma_width * chroma_height];
        let mut planes = LosslessSampledColorPlanesMut {
            c0: &mut c0,
            c1: &mut c1,
            c2: &mut c2,
            dimensions: (width, height),
            chroma_dimensions: (chroma_width, chroma_height),
        };

        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_mcus = (chroma_width * chroma_height) as u32;
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_mcus);
        for mcu_y in 0..chroma_height {
            for mcu_x in 0..chroma_width {
                let mcu_index = (mcu_y * chroma_width + mcu_x) as u32;
                let restart_first_mcu = restart_tracker.begin_unit(&mut br, mcu_index)?;
                decode_lossless_sampled_color_mcu::<P>(
                    &mut br,
                    &self.plan.components,
                    plan.predictor,
                    LosslessSampledMcu {
                        x: mcu_x,
                        y: mcu_y,
                        restart_first_mcu,
                    },
                    &mut planes,
                )?;
                restart_tracker.finish_unit();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        let LosslessSampledColorPlanesMut { c0, c1, c2, .. } = planes;
        write_output(
            out,
            stride,
            color_space,
            sampling,
            (width, height),
            LosslessColorPlanes { c0, c1, c2 },
        );
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    pub(super) fn decode_lossless_color8_sampled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_sampled_into::<u8>(
            out,
            stride,
            color_space,
            write_lossless_color8_sampled_output,
        )
    }

    pub(super) fn decode_lossless_ycbcr8_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color8_output_into(out, stride, ColorSpace::YCbCr)
    }

    pub(super) fn decode_lossless_color_rows<P, S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        conversion_row: Option<&mut [u8]>,
        color_space: ColorSpace,
        convert_row: impl Fn(&[u8], &mut [u8]),
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
        S: RowSink<u8, Error = JpegError>,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;

        let (width, height) = plan.dimensions;
        let width = width as usize;
        let row_len = width.saturating_mul(3 * P::BYTES);
        prev_row.resize(row_len, 0);
        curr_row.resize(row_len, 0);

        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_pixels = plan.dimensions.0.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_pixels);
        let mut conversion_row = conversion_row;
        for y in 0..height as usize {
            for x in 0..width {
                let pixel_index = y as u32 * plan.dimensions.0 + x as u32;
                let restart_first_pixel = restart_tracker.begin_unit(&mut br, pixel_index)?;
                decode_lossless_color_sample::<P, _>(
                    &mut br,
                    &self.plan.components,
                    plan.predictor,
                    restart_first_pixel,
                    &mut LosslessColorRowSample {
                        curr_row: &mut *curr_row,
                        prev_row: &*prev_row,
                        x,
                        y,
                    },
                )?;
                restart_tracker.finish_unit();
            }
            let row = if color_space == ColorSpace::YCbCr {
                let row = conversion_row
                    .as_deref_mut()
                    .ok_or(JpegError::OutputBufferTooSmall {
                        required: row_len,
                        provided: 0,
                    })?;
                if row.len() < row_len {
                    return Err(JpegError::OutputBufferTooSmall {
                        required: row_len,
                        provided: row.len(),
                    });
                }
                convert_row(&curr_row[..row_len], &mut row[..row_len]);
                &row[..row_len]
            } else {
                &curr_row[..row_len]
            };
            sink.write_row(y as u32, row)?;
            core::mem::swap(prev_row, curr_row);
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    pub(super) fn decode_lossless_color8_rows<S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        conversion_row: Option<&mut [u8]>,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_color_rows::<u8, S>(
            sink,
            prev_row,
            curr_row,
            conversion_row,
            color_space,
            copy_ycbcr8_row_to_rgb8,
        )
    }

    pub(super) fn decode_lossless_gray16_rows<S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_gray_rows::<u16, S>(sink, prev_row, curr_row, |sink, y, row| {
            sink.write_row(y, row)
        })
    }

    pub(super) fn decode_lossless_color16_rows<S>(
        &self,
        sink: &mut S,
        prev_row: &mut Vec<u8>,
        curr_row: &mut Vec<u8>,
        conversion_row: Option<&mut [u8]>,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_color_rows::<u16, S>(
            sink,
            prev_row,
            curr_row,
            conversion_row,
            color_space,
            copy_ycbcr16_row_to_rgb16,
        )
    }

    pub(super) fn decode_lossless_rgb16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color16_output_into(out, stride, ColorSpace::Rgb)
    }

    pub(super) fn decode_lossless_color16_output_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        match lossless_color_sampling(&self.info) {
            Some(LosslessColorSampling::S444) => {
                let outcome =
                    self.decode_lossless_color16_components_into(out, stride, color_space)?;
                if color_space == ColorSpace::YCbCr {
                    convert_ycbcr16_to_rgb16_in_place(out, stride, self.info.dimensions);
                }
                Ok(outcome)
            }
            Some(LosslessColorSampling::S422 | LosslessColorSampling::S420) => {
                self.decode_lossless_color16_sampled_into(out, stride, color_space)
            }
            None => Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            }),
        }
    }

    pub(super) fn decode_lossless_color16_components_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_components_into::<u16>(out, stride, color_space)
    }

    pub(super) fn decode_lossless_color16_sampled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_sampled_into::<u16>(
            out,
            stride,
            color_space,
            write_lossless_color16_sampled_output,
        )
    }

    pub(super) fn decode_lossless_ycbcr16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color16_output_into(out, stride, ColorSpace::YCbCr)
    }

    pub(super) fn decode_lossless_gray16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if plan.bit_depth != 16 {
            return Err(JpegError::UnsupportedBitDepth {
                depth: plan.bit_depth,
            });
        }
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }

        let (width, height) = plan.dimensions;
        let scan_bytes = &self.bytes[plan.scan_offset..];
        let mut br = BitReader::new(scan_bytes);
        let total_samples = width.saturating_mul(height);
        let mut restart_tracker =
            LosslessRestartTracker::new(self.plan.restart_interval, total_samples);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let sample_index = y as u32 * width + x as u32;
                let restart_first_sample = restart_tracker.begin_unit(&mut br, sample_index)?;
                let predictor = if restart_first_sample {
                    32768
                } else {
                    lossless_predictor_value_u16(plan.predictor, out, stride, x, y)
                };
                let diff = plan.dc_table.decode_fast_dc(&mut br)?;
                let sample = <u16 as LosslessSample>::from_i32(predictor + diff)?;
                let offset = y * stride + x * 2;
                sample.write_le(&mut out[offset..offset + 2]);
                restart_tracker.finish_unit();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    pub(super) fn decode_lossless_gray16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        if roi == Rect::full(self.info.dimensions) && downscale == DownscaleFactor::Full {
            return self.decode_lossless_gray16_into(out, stride);
        }

        let (width, height) = self.info.dimensions;
        let full_stride = width as usize * 2;
        let mut full =
            allocate_output_buffer(checked_scratch_len(&[full_stride, height as usize])?);
        let mut outcome = self.decode_lossless_gray16_into(&mut full, full_stride)?;
        let output_rect = scaled_rect_covering(roi, downscale)?;
        copy_gray16_scaled_rect(
            &full,
            (width, height),
            output_rect,
            downscale.denominator(),
            out,
            stride,
        );
        outcome.decoded = roi;
        Ok(outcome)
    }
}
