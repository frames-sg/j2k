// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structured decode profiling kept out of output dispatch.

use super::super::{
    downscale_profile_name, emit_jpeg_profile_fields, output_format_profile_name, DecodeOutcome,
    DownscaleFactor, Instant, OutputFormat, ProfileField, Rect,
};

pub(super) struct DecodeProfileRecord {
    pub(super) total_start: Instant,
    pub(super) decode_start: Instant,
    pub(super) source_dimensions: (u32, u32),
    pub(super) output_rect: Rect,
    pub(super) stride: usize,
    pub(super) bytes_per_pixel: usize,
    pub(super) scratch_bytes: usize,
    pub(super) fmt: OutputFormat,
    pub(super) downscale: DownscaleFactor,
    pub(super) source_roi: Option<Rect>,
}

impl DecodeProfileRecord {
    pub(super) fn emit(self, outcome: &DecodeOutcome) {
        match self.source_roi {
            Some(roi) => self.emit_region(roi, outcome),
            None => self.emit_full(outcome),
        }
    }

    fn emit_full(self, outcome: &DecodeOutcome) {
        let output_bytes = self.stride.saturating_mul(self.output_rect.h as usize);
        emit_jpeg_profile_fields("jpeg_decode_full_fields", "decode", "cpu", || {
            Ok([
                ProfileField::label("mode", "full")?,
                ProfileField::label("fmt", output_format_profile_name(self.fmt))?,
                ProfileField::label("downscale", downscale_profile_name(self.downscale))?,
                ProfileField::metric_with_summary("source_width", self.source_dimensions.0, false)?,
                ProfileField::metric_with_summary(
                    "source_height",
                    self.source_dimensions.1,
                    false,
                )?,
                ProfileField::metric_with_summary("output_width", self.output_rect.w, false)?,
                ProfileField::metric_with_summary("output_height", self.output_rect.h, false)?,
                ProfileField::metric_with_summary("stride", self.stride, false)?,
                ProfileField::metric_with_summary("bpp", self.bytes_per_pixel, false)?,
                ProfileField::metric_with_summary("scratch_bytes", self.scratch_bytes, false)?,
                ProfileField::metric_with_summary("output_bytes", output_bytes, false)?,
                ProfileField::metric("decode_us", self.decode_start.elapsed().as_micros())?,
                ProfileField::metric("total_us", self.total_start.elapsed().as_micros())?,
                ProfileField::metric_with_summary("warnings", outcome.warnings.len(), false)?,
            ])
        });
    }

    fn emit_region(self, roi: Rect, outcome: &DecodeOutcome) {
        let output_bytes = self.stride.saturating_mul(self.output_rect.h as usize);
        let mode = if self.downscale == DownscaleFactor::Full {
            "region"
        } else {
            "region_scaled"
        };
        emit_jpeg_profile_fields("jpeg_decode_region_fields", "decode", "cpu", || {
            Ok([
                ProfileField::label("mode", mode)?,
                ProfileField::label("fmt", output_format_profile_name(self.fmt))?,
                ProfileField::label("downscale", downscale_profile_name(self.downscale))?,
                ProfileField::metric_with_summary("source_width", self.source_dimensions.0, false)?,
                ProfileField::metric_with_summary(
                    "source_height",
                    self.source_dimensions.1,
                    false,
                )?,
                ProfileField::metric_with_summary("roi_x", roi.x, false)?,
                ProfileField::metric_with_summary("roi_y", roi.y, false)?,
                ProfileField::metric_with_summary("roi_w", roi.w, false)?,
                ProfileField::metric_with_summary("roi_h", roi.h, false)?,
                ProfileField::metric_with_summary("output_width", self.output_rect.w, false)?,
                ProfileField::metric_with_summary("output_height", self.output_rect.h, false)?,
                ProfileField::metric_with_summary("stride", self.stride, false)?,
                ProfileField::metric_with_summary("bpp", self.bytes_per_pixel, false)?,
                ProfileField::metric_with_summary("scratch_bytes", self.scratch_bytes, false)?,
                ProfileField::metric_with_summary("output_bytes", output_bytes, false)?,
                ProfileField::metric("decode_us", self.decode_start.elapsed().as_micros())?,
                ProfileField::metric("total_us", self.total_start.elapsed().as_micros())?,
                ProfileField::metric_with_summary("warnings", outcome.warnings.len(), false)?,
            ])
        });
    }
}
