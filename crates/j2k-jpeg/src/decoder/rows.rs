// SPDX-License-Identifier: MIT OR Apache-2.0

//! Row-oriented decode entry points and component-row routing.

use super::{
    jpeg_downscale, scaled_rect_covering, ColorSpace, ComponentRowWriter, CroppedWriter,
    DecodeOutcome, Decoder, Downscale, DownscaleFactor, JpegError, Rect, RowSink, ScratchPool,
    SinkWriter, DEFAULT_SCRATCH,
};

impl Decoder<'_> {
    /// Decode the full image into rows delivered to `sink`.
    ///
    /// DCT-backed and 8-bit lossless color paths emit interleaved RGB8 rows.
    /// Lossless 16-bit grayscale SOF3 emits little-endian Gray16 rows, and
    /// supported lossless 16-bit color SOF3 emits little-endian Rgb16 rows.
    pub fn decode_rows<S>(&self, sink: &mut S) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        DEFAULT_SCRATCH.with(|pool| self.decode_rows_with_scratch(&mut pool.borrow_mut(), sink))
    }

    /// [`Self::decode_rows`] with caller-owned scratch. See
    /// [`Self::decode_into_with_scratch`] for the reuse contract.
    pub fn decode_rows_with_scratch<S>(
        &self,
        pool: &mut ScratchPool,
        sink: &mut S,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        if self.lossless_plan.is_some() {
            return self.decode_lossless_rows_with_scratch(pool, sink);
        }
        let width = self.info.dimensions.0 as usize;
        let rows = pool.take_sink_rows(width);
        let mut writer = SinkWriter::new(sink, rows, self.backend);
        let result = self.decode_rgb_with_writer(
            pool,
            &mut writer,
            DownscaleFactor::Full,
            Rect::full(self.info.dimensions),
        );
        pool.restore_sink_rows(writer.into_rows());
        result
    }

    fn decode_lossless_rows_with_scratch<S>(
        &self,
        pool: &mut ScratchPool,
        sink: &mut S,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }

        let width = self.info.dimensions.0 as usize;
        match (self.info.color_space, plan.bit_depth) {
            (ColorSpace::Grayscale, 8) => {
                let mut rows = pool.take_sink_rows(width);
                let result = self.decode_lossless_gray8_rows(
                    sink,
                    &mut pool.lossless_prev_row,
                    &mut pool.lossless_curr_row,
                    &mut rows.top_row,
                );
                pool.restore_sink_rows(rows);
                result
            }
            (ColorSpace::Grayscale, 16) => self.decode_lossless_gray16_rows(
                sink,
                &mut pool.lossless_prev_row,
                &mut pool.lossless_curr_row,
            ),
            (ColorSpace::Rgb, 8) => self.decode_lossless_color8_rows(
                sink,
                &mut pool.lossless_prev_row,
                &mut pool.lossless_curr_row,
                None,
                ColorSpace::Rgb,
            ),
            (ColorSpace::YCbCr, 8) => {
                let mut rows = pool.take_sink_rows(width);
                let result = self.decode_lossless_color8_rows(
                    sink,
                    &mut pool.lossless_prev_row,
                    &mut pool.lossless_curr_row,
                    Some(&mut rows.top_row),
                    ColorSpace::YCbCr,
                );
                pool.restore_sink_rows(rows);
                result
            }
            (ColorSpace::Rgb, 16) => self.decode_lossless_color16_rows(
                sink,
                &mut pool.lossless_prev_row,
                &mut pool.lossless_curr_row,
                None,
                ColorSpace::Rgb,
            ),
            (ColorSpace::YCbCr, 16) => {
                let mut rows = pool.take_sink_rows(width);
                rows.top_row.resize(width.saturating_mul(6), 0);
                let result = self.decode_lossless_color16_rows(
                    sink,
                    &mut pool.lossless_prev_row,
                    &mut pool.lossless_curr_row,
                    Some(&mut rows.top_row),
                    ColorSpace::YCbCr,
                );
                pool.restore_sink_rows(rows);
                result
            }
            (_, depth) if depth != 8 && depth != 16 => {
                Err(JpegError::UnsupportedBitDepth { depth })
            }
            _ => Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            }),
        }
    }

    /// Decode the full image into component rows.
    pub fn decode_component_rows_with_scratch<W>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
    ) -> Result<DecodeOutcome, JpegError>
    where
        W: ComponentRowWriter,
    {
        self.decode_region_component_rows_with_scratch(
            pool,
            writer,
            Rect::full(self.info.dimensions),
            Downscale::None,
        )
    }

    /// Decode `roi` into component rows, optionally at a reduced scale.
    pub fn decode_region_component_rows_with_scratch<W>(
        &self,
        pool: &mut ScratchPool,
        mut writer: &mut W,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError>
    where
        W: ComponentRowWriter,
    {
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }

        let downscale = jpeg_downscale(scale);
        let scaled_roi = scaled_rect_covering(roi, downscale)?;

        if roi == Rect::full(self.info.dimensions) {
            self.decode_with_writer(pool, &mut writer, downscale, roi)
        } else {
            let (source_x0, source_width) =
                self.source_window_for_output_rect(downscale, scaled_roi);
            let mut cropped = CroppedWriter::new(writer, scaled_roi, source_x0, source_width);
            self.decode_with_writer(pool, &mut cropped, downscale, roi)
        }
    }
}
