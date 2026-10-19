// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless grayscale and color rendering paths. Every path pulls whole MCU
//! rows from one [`LosslessScan`] and stores them into its output.

use super::lossless_planes::{lossless_color_uses_planes, lossless_plane_layout};
use super::{
    allocate_output_buffer_with_live_budget, checked_scratch_len,
    convert_ycbcr16_to_rgb16_in_place, convert_ycbcr8_to_rgb8_in_place, copy_gray16_scaled_rect,
    copy_gray8_scaled_rect, copy_ycbcr16_row_to_rgb16, copy_ycbcr8_row_to_rgb8,
    lossless_color_sampling, lossless_output_order, merged_warnings, scaled_rect_covering,
    validate_lossless_color_plan, ColorSpace, DecodeOutcome, Decoder, DownscaleFactor, JpegError,
    LosslessColorSampling, LosslessSample, PreparedComponentPlan, PreparedLosslessPlan, Rect,
    RowSink, Vec,
};
use crate::allocation::{try_resize_filled, try_vec_with_capacity};
use crate::lossless::scan::{LosslessComponentSpec, LosslessScan, LosslessScanSpec};

#[cfg(test)]
mod tests;

/// Color output steps that depend on the sample container.
trait LosslessColorOutput: LosslessSample {
    fn convert_ycbcr_in_place(out: &mut [u8], stride: usize, dimensions: (u32, u32));
}

impl LosslessColorOutput for u8 {
    fn convert_ycbcr_in_place(out: &mut [u8], stride: usize, dimensions: (u32, u32)) {
        convert_ycbcr8_to_rgb8_in_place(out, stride, dimensions);
    }
}

impl LosslessColorOutput for u16 {
    fn convert_ycbcr_in_place(out: &mut [u8], stride: usize, dimensions: (u32, u32)) {
        convert_ycbcr16_to_rgb16_in_place(out, stride, dimensions);
    }
}

/// One scan of a lossless frame: its components (in scan order, with the
/// scan's tables) and parameters.
#[derive(Clone, Copy)]
pub(super) struct LosslessScanView<'p> {
    pub(super) components: &'p [PreparedComponentPlan],
    predictor: u8,
    point_transform: u8,
    restart_interval: Option<u16>,
    entropy_offset: usize,
    /// Whether EOI must follow: false for a split frame's earlier scans and
    /// for a final scan the header walk saw cut off at EOF.
    pub(super) validate_eoi: bool,
}

impl Decoder<'_> {
    /// The lossless plan, checked against the `P` output container.
    pub(super) fn lossless_plan_for<P: LosslessSample>(
        &self,
    ) -> Result<&PreparedLosslessPlan, JpegError> {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if !P::holds_precision(plan.bit_depth) {
            return Err(JpegError::UnsupportedBitDepth {
                depth: plan.bit_depth,
            });
        }
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }
        Ok(plan)
    }

    /// Scan `index` of the frame.
    pub(super) fn lossless_scan_view<'p>(
        &'p self,
        plan: &'p PreparedLosslessPlan,
        index: usize,
    ) -> Result<LosslessScanView<'p>, JpegError> {
        let invalid = || JpegError::InternalInvariant {
            reason: "lossless scan view is outside the prepared scans",
        };
        if plan.split_scans.is_empty() {
            if index != 0 {
                return Err(invalid());
            }
            return Ok(LosslessScanView {
                components: &self.plan.components,
                predictor: plan.predictor,
                point_transform: plan.point_transform,
                restart_interval: self.plan.restart_interval,
                entropy_offset: plan.scan_offset,
                validate_eoi: true,
            });
        }
        let scan = plan.split_scans.get(index).ok_or_else(invalid)?;
        let components = self
            .plan
            .components
            .get(scan.component_start..scan.component_start + scan.component_len)
            .ok_or_else(invalid)?;
        Ok(LosslessScanView {
            components,
            predictor: scan.predictor,
            point_transform: scan.point_transform,
            restart_interval: scan.restart_interval,
            entropy_offset: scan.entropy_offset,
            validate_eoi: scan.terminal_code == 0xD9,
        })
    }

    /// Start decoding the scan `view` describes.
    pub(super) fn open_lossless_scan(
        &self,
        plan: &PreparedLosslessPlan,
        view: LosslessScanView<'_>,
    ) -> Result<LosslessScan<'_, '_>, JpegError> {
        let mut components = try_vec_with_capacity(view.components.len())?;
        for component in view.components {
            components.push(LosslessComponentSpec {
                h: component.h,
                v: component.v,
                table: self.plan.dc_table(component)?,
            });
        }
        let scan_bytes = self
            .bytes
            .get(view.entropy_offset..)
            .ok_or(JpegError::Truncated {
                offset: view.entropy_offset,
                expected: 1,
            })?;
        LosslessScan::new(
            scan_bytes,
            LosslessScanSpec {
                dimensions: plan.dimensions,
                predictor: view.predictor,
                precision: plan.bit_depth,
                point_transform: view.point_transform,
                restart_interval: view.restart_interval,
                frame_max_sampling: (self.info.sampling.max_h, self.info.sampling.max_v),
            },
            &components,
        )
    }

    /// Start the frame's only scan.
    fn lossless_scan(
        &self,
        plan: &PreparedLosslessPlan,
    ) -> Result<LosslessScan<'_, '_>, JpegError> {
        if !plan.split_scans.is_empty() {
            return Err(JpegError::InternalInvariant {
                reason: "single-scan lossless path reached a split frame",
            });
        }
        self.open_lossless_scan(plan, self.lossless_scan_view(plan, 0)?)
    }

    fn lossless_gray_scan<P: LosslessSample>(&self) -> Result<LosslessScan<'_, '_>, JpegError> {
        let plan = self.lossless_plan_for::<P>()?;
        if self.plan.components.len() != 1 {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        self.lossless_scan(plan)
    }

    fn finish_lossless_scan(&self, scan: LosslessScan<'_, '_>) -> Result<DecodeOutcome, JpegError> {
        let scan_warnings = scan.finish()?;
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings)?,
        })
    }

    fn decode_lossless_gray_into<P: LosslessSample>(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        let mut scan = self.lossless_gray_scan::<P>()?;
        let shift = scan.point_transform();
        let row_bytes = checked_scratch_len(&[self.info.dimensions.0 as usize, P::BYTES])?;
        while scan.decode_mcu_row()? {
            let y = scan.row_base(0);
            P::store_row(scan.row(0, 0), shift, &mut out[y * stride..][..row_bytes]);
        }
        self.finish_lossless_scan(scan)
    }

    pub(super) fn decode_lossless_gray8_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_gray_into::<u8>(out, stride)
    }

    pub(super) fn decode_lossless_gray8_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        external_live_bytes: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        if roi == Rect::full(self.info.dimensions) && downscale == DownscaleFactor::Full {
            return self.decode_lossless_gray8_into(out, stride);
        }

        let (width, height) = self.info.dimensions;
        let full_stride = width as usize;
        let full_len = checked_scratch_len(&[full_stride, height as usize])?;
        let (mut live_bytes, workspace_cap) = self.decode_phase_live_bytes(external_live_bytes)?;
        let mut full =
            allocate_output_buffer_with_live_budget(full_len, &mut live_bytes, workspace_cap)?;
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

    /// Stream grayscale rows through `emit_row`, assembling each in `row`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "row indices are bounded by validated u32 image dimensions"
    )]
    fn decode_lossless_gray_rows<P, S>(
        &self,
        sink: &mut S,
        row: &mut Vec<u8>,
        mut emit_row: impl FnMut(&mut S, u32, &[u8]) -> Result<(), JpegError>,
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
        S: RowSink<u8, Error = JpegError>,
    {
        if self.info.color_space != ColorSpace::Grayscale {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        let mut scan = self.lossless_gray_scan::<P>()?;
        let shift = scan.point_transform();
        let row_len = checked_scratch_len(&[self.info.dimensions.0 as usize, P::BYTES])?;
        try_resize_filled(row, row_len, 0)?;
        while scan.decode_mcu_row()? {
            P::store_row(scan.row(0, 0), shift, &mut row[..row_len]);
            emit_row(sink, scan.row_base(0) as u32, &row[..row_len])?;
        }
        self.finish_lossless_scan(scan)
    }

    pub(super) fn decode_lossless_gray8_rows<S>(
        &self,
        sink: &mut S,
        row: &mut Vec<u8>,
        rgb_row: &mut [u8],
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_gray_rows::<u8, S>(sink, row, |sink, y, gray_row| {
            let rgb_len = gray_row.len().saturating_mul(3);
            if rgb_row.len() < rgb_len {
                return Err(JpegError::OutputBufferTooSmall {
                    required: rgb_len,
                    provided: rgb_row.len(),
                });
            }
            for (pixel, &sample) in rgb_row[..rgb_len]
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(gray_row.iter())
            {
                pixel.copy_from_slice(&[sample, sample, sample]);
            }
            sink.write_row(y, &rgb_row[..rgb_len])
        })
    }

    pub(super) fn decode_lossless_gray16_rows<S>(
        &self,
        sink: &mut S,
        row: &mut Vec<u8>,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_gray_rows::<u16, S>(sink, row, S::write_row)
    }

    pub(super) fn decode_lossless_gray16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_gray_into::<u16>(out, stride)
    }

    pub(super) fn decode_lossless_gray16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        external_live_bytes: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        if roi == Rect::full(self.info.dimensions) && downscale == DownscaleFactor::Full {
            return self.decode_lossless_gray16_into(out, stride);
        }

        let (width, height) = self.info.dimensions;
        let full_stride = width as usize * 2;
        let full_len = checked_scratch_len(&[full_stride, height as usize])?;
        let (mut live_bytes, workspace_cap) = self.decode_phase_live_bytes(external_live_bytes)?;
        let mut full =
            allocate_output_buffer_with_live_budget(full_len, &mut live_bytes, workspace_cap)?;
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

    fn decode_lossless_color_output_into<P: LosslessColorOutput>(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self.lossless_plan_for::<P>()?;
        if lossless_color_uses_planes(&self.info, plan) {
            return self.decode_lossless_color_planes_into::<P>(out, stride, color_space);
        }
        match lossless_color_sampling(&self.info) {
            Some(LosslessColorSampling::S444) => {
                let outcome =
                    self.decode_lossless_color_components_into::<P>(out, stride, color_space)?;
                if color_space == ColorSpace::YCbCr {
                    P::convert_ycbcr_in_place(out, stride, self.info.dimensions);
                }
                Ok(outcome)
            }
            Some(LosslessColorSampling::Sampled) | None => Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            }),
        }
    }

    /// 4:4:4 color: each MCU row is one pixel row of all three components.
    fn decode_lossless_color_components_into<P: LosslessSample>(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self.lossless_plan_for::<P>()?;
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;
        let [r, g, b] = lossless_output_order(&self.plan)?;
        let mut scan = self.lossless_scan(plan)?;
        let shift = scan.point_transform();
        let row_bytes = checked_scratch_len(&[self.info.dimensions.0 as usize, 3, P::BYTES])?;
        while scan.decode_mcu_row()? {
            let y = scan.row_base(0);
            P::store_interleaved(
                [scan.row(r, 0), scan.row(g, 0), scan.row(b, 0)],
                shift,
                &mut out[y * stride..][..row_bytes],
            );
        }
        self.finish_lossless_scan(scan)
    }

    pub(super) fn decode_lossless_rgb8_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_output_into::<u8>(out, stride, ColorSpace::Rgb)
    }

    pub(super) fn decode_lossless_ycbcr8_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_output_into::<u8>(out, stride, ColorSpace::YCbCr)
    }

    pub(super) fn decode_lossless_rgb16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_output_into::<u16>(out, stride, ColorSpace::Rgb)
    }

    pub(super) fn decode_lossless_ycbcr16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_lossless_color_output_into::<u16>(out, stride, ColorSpace::YCbCr)
    }

    /// Stream 4:4:4 color rows, assembling each in `row` and converting
    /// YCbCr through `conversion_row`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "row indices are bounded by validated u32 image dimensions"
    )]
    fn decode_lossless_color_rows<P, S>(
        &self,
        sink: &mut S,
        row: &mut Vec<u8>,
        mut conversion_row: Option<&mut [u8]>,
        color_space: ColorSpace,
        convert_row: impl Fn(&[u8], &mut [u8]),
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
        S: RowSink<u8, Error = JpegError>,
    {
        let plan = self.lossless_plan_for::<P>()?;
        if lossless_color_uses_planes(&self.info, plan) {
            return self.decode_lossless_color_plane_rows::<P, S>(sink, row, color_space);
        }
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;
        if lossless_color_sampling(&self.info) != Some(LosslessColorSampling::S444) {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        let [r, g, b] = lossless_output_order(&self.plan)?;
        let row_len = checked_scratch_len(&[plan.dimensions.0 as usize, 3, P::BYTES])?;
        try_resize_filled(row, row_len, 0)?;

        let mut scan = self.lossless_scan(plan)?;
        let shift = scan.point_transform();
        while scan.decode_mcu_row()? {
            P::store_interleaved(
                [scan.row(r, 0), scan.row(g, 0), scan.row(b, 0)],
                shift,
                &mut row[..row_len],
            );
            let emitted = if color_space == ColorSpace::YCbCr {
                let converted =
                    conversion_row
                        .as_deref_mut()
                        .ok_or(JpegError::OutputBufferTooSmall {
                            required: row_len,
                            provided: 0,
                        })?;
                let provided = converted.len();
                let converted =
                    converted
                        .get_mut(..row_len)
                        .ok_or(JpegError::OutputBufferTooSmall {
                            required: row_len,
                            provided,
                        })?;
                convert_row(&row[..row_len], converted);
                &*converted
            } else {
                &row[..row_len]
            };
            sink.write_row(scan.row_base(0) as u32, emitted)?;
        }
        self.finish_lossless_scan(scan)
    }

    pub(super) fn decode_lossless_color8_rows<S>(
        &self,
        sink: &mut S,
        row: &mut Vec<u8>,
        conversion_row: Option<&mut [u8]>,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_color_rows::<u8, S>(
            sink,
            row,
            conversion_row,
            color_space,
            copy_ycbcr8_row_to_rgb8,
        )
    }

    pub(super) fn decode_lossless_color16_rows<S>(
        &self,
        sink: &mut S,
        row: &mut Vec<u8>,
        conversion_row: Option<&mut [u8]>,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError>
    where
        S: RowSink<u8, Error = JpegError>,
    {
        self.decode_lossless_color_rows::<u16, S>(
            sink,
            row,
            conversion_row,
            color_space,
            copy_ycbcr16_row_to_rgb16,
        )
    }
}

/// One scan of a validated device lossless decode.
pub(crate) struct DeviceLosslessScan<'d> {
    pub(crate) scan: LosslessScan<'d, 'd>,
    /// Frame (`output_index`) position of each scan component, in scan order.
    pub(crate) frame_components: [usize; 3],
    pub(crate) component_count: usize,
    /// Whether EOI must follow the scan.
    pub(crate) validate_eoi: bool,
}

impl Decoder<'_> {
    /// Validate a device decode to `fmt` exactly as CPU decode validates
    /// that output, and return how many scans it decodes.
    pub(crate) fn device_lossless_scan_count(
        &self,
        fmt: j2k_core::PixelFormat,
    ) -> Result<usize, JpegError> {
        use j2k_core::PixelFormat;
        let plan = match fmt {
            PixelFormat::Gray8 => self.device_lossless_gray_plan::<u8>()?,
            PixelFormat::Gray16 => self.device_lossless_gray_plan::<u16>()?,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 => self.device_lossless_color_plan::<u8>()?,
            PixelFormat::Rgb16 | PixelFormat::Rgba16 => self.device_lossless_color_plan::<u16>()?,
            _ => {
                return Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                })
            }
        };
        Ok(plan.scan_count())
    }

    /// Open scan `index` of a device decode validated by
    /// [`Self::device_lossless_scan_count`].
    pub(crate) fn device_lossless_scan(
        &self,
        index: usize,
    ) -> Result<DeviceLosslessScan<'_>, JpegError> {
        let plan = self
            .lossless_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        let view = self.lossless_scan_view(plan, index)?;
        let mut frame_components = [0; 3];
        if view.components.len() > frame_components.len() {
            return Err(JpegError::UnsupportedComponentCount {
                count: u8::try_from(view.components.len()).unwrap_or(u8::MAX),
            });
        }
        for (slot, component) in frame_components.iter_mut().zip(view.components) {
            *slot = component.output_index;
        }
        Ok(DeviceLosslessScan {
            scan: self.open_lossless_scan(plan, view)?,
            frame_components,
            component_count: view.components.len(),
            validate_eoi: view.validate_eoi,
        })
    }

    fn device_lossless_gray_plan<P: LosslessSample>(
        &self,
    ) -> Result<&PreparedLosslessPlan, JpegError> {
        let plan = self.lossless_plan_for::<P>()?;
        if self.plan.components.len() != 1 {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        Ok(plan)
    }

    fn device_lossless_color_plan<P: LosslessSample>(
        &self,
    ) -> Result<&PreparedLosslessPlan, JpegError> {
        let color_space = if self.info.color_space == ColorSpace::YCbCr {
            ColorSpace::YCbCr
        } else {
            ColorSpace::Rgb
        };
        let plan = self.lossless_plan_for::<P>()?;
        if lossless_color_uses_planes(&self.info, plan) {
            if self.info.color_space != color_space {
                return Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                });
            }
            let layout = lossless_plane_layout(&self.info, self.plan.scratch_bytes)?.ok_or(
                JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                },
            )?;
            if layout.total_bytes > self.plan.scratch_bytes {
                return Err(JpegError::MemoryCapExceeded {
                    requested: layout.total_bytes,
                    cap: self.plan.scratch_bytes,
                });
            }
            return Ok(plan);
        }
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;
        if lossless_color_sampling(&self.info) != Some(LosslessColorSampling::S444) {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        Ok(plan)
    }
}
