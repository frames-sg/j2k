// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless grayscale and color rendering paths. Every path pulls whole MCU
//! rows from one [`LosslessScan`] and stores them into its output.

use super::{
    allocate_output_buffer_with_live_budget, checked_scratch_len,
    convert_ycbcr16_to_rgb16_in_place, convert_ycbcr8_to_rgb8_in_place, copy_gray16_scaled_rect,
    copy_gray8_scaled_rect, copy_ycbcr16_row_to_rgb16, copy_ycbcr8_row_to_rgb8,
    lossless_color_sampling, lossless_output_order, lossless_sampled_plane_layout, merged_warnings,
    scaled_rect_covering, validate_lossless_color_plan, write_lossless_color16_sampled_output,
    write_lossless_color8_sampled_output, ColorSpace, DecodeOutcome, Decoder, DownscaleFactor,
    JpegError, LosslessColorPlanes, LosslessColorSampling, LosslessSample,
    LosslessSampledPlaneLayout, PreparedLosslessPlan, Rect, RowSink, Vec,
};
use crate::allocation::{
    try_reserve_for_len_with_live_budget, try_resize_filled, try_vec_with_capacity,
};
use crate::lossless::scan::{LosslessComponentSpec, LosslessScan, LosslessScanSpec};

#[cfg(test)]
mod tests;

struct OwnedLosslessSampledPlanes<P> {
    c0: Vec<P>,
    c1: Vec<P>,
    c2: Vec<P>,
}

fn allocate_lossless_sampled_planes<P: LosslessSample>(
    layout: LosslessSampledPlaneLayout,
    cap: usize,
) -> Result<OwnedLosslessSampledPlanes<P>, JpegError> {
    let mut live_bytes = 0;
    let mut c0 = Vec::new();
    try_reserve_for_len_with_live_budget(&mut c0, layout.luma_len, &mut live_bytes, cap)?;
    c0.resize(layout.luma_len, P::default());
    let mut c1 = Vec::new();
    try_reserve_for_len_with_live_budget(&mut c1, layout.chroma_len, &mut live_bytes, cap)?;
    c1.resize(layout.chroma_len, P::default());
    let mut c2 = Vec::new();
    try_reserve_for_len_with_live_budget(&mut c2, layout.chroma_len, &mut live_bytes, cap)?;
    c2.resize(layout.chroma_len, P::default());
    Ok(OwnedLosslessSampledPlanes { c0, c1, c2 })
}

/// Color output steps that depend on the sample container.
trait LosslessColorOutput: LosslessSample {
    fn convert_ycbcr_in_place(out: &mut [u8], stride: usize, dimensions: (u32, u32));

    fn write_sampled(
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
        sampling: LosslessColorSampling,
        dimensions: (usize, usize),
        planes: LosslessColorPlanes<'_, Self>,
    );
}

impl LosslessColorOutput for u8 {
    fn convert_ycbcr_in_place(out: &mut [u8], stride: usize, dimensions: (u32, u32)) {
        convert_ycbcr8_to_rgb8_in_place(out, stride, dimensions);
    }

    fn write_sampled(
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
        sampling: LosslessColorSampling,
        dimensions: (usize, usize),
        planes: LosslessColorPlanes<'_, Self>,
    ) {
        write_lossless_color8_sampled_output(
            out,
            stride,
            color_space,
            sampling,
            dimensions,
            planes,
        );
    }
}

impl LosslessColorOutput for u16 {
    fn convert_ycbcr_in_place(out: &mut [u8], stride: usize, dimensions: (u32, u32)) {
        convert_ycbcr16_to_rgb16_in_place(out, stride, dimensions);
    }

    fn write_sampled(
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
        sampling: LosslessColorSampling,
        dimensions: (usize, usize),
        planes: LosslessColorPlanes<'_, Self>,
    ) {
        write_lossless_color16_sampled_output(
            out,
            stride,
            color_space,
            sampling,
            dimensions,
            planes,
        );
    }
}

impl Decoder<'_> {
    /// The lossless plan, checked against the `P` output container.
    fn lossless_plan_for<P: LosslessSample>(&self) -> Result<&PreparedLosslessPlan, JpegError> {
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
        if !(1..=7).contains(&plan.predictor) {
            return Err(JpegError::UnsupportedPredictor {
                predictor: plan.predictor,
            });
        }
        Ok(plan)
    }

    /// Start the scan over every plan component, in scan order.
    fn lossless_scan(
        &self,
        plan: &PreparedLosslessPlan,
    ) -> Result<LosslessScan<'_, '_>, JpegError> {
        let mut components = try_vec_with_capacity(self.plan.components.len())?;
        for component in &self.plan.components {
            components.push(LosslessComponentSpec {
                h: component.h,
                v: component.v,
                table: self.plan.dc_table(component)?,
            });
        }
        LosslessScan::new(
            &self.bytes[plan.scan_offset..],
            LosslessScanSpec {
                dimensions: plan.dimensions,
                predictor: plan.predictor,
                precision: plan.bit_depth,
                point_transform: plan.point_transform,
                restart_interval: self.plan.restart_interval,
            },
            &components,
        )
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
            for (pixel, &sample) in rgb_row[..rgb_len].chunks_exact_mut(3).zip(gray_row.iter()) {
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
        match lossless_color_sampling(&self.info) {
            Some(LosslessColorSampling::S444) => {
                let outcome =
                    self.decode_lossless_color_components_into::<P>(out, stride, color_space)?;
                if color_space == ColorSpace::YCbCr {
                    P::convert_ycbcr_in_place(out, stride, self.info.dimensions);
                }
                Ok(outcome)
            }
            Some(sampling @ (LosslessColorSampling::S422 | LosslessColorSampling::S420)) => {
                self.decode_lossless_color_sampled_into::<P>(out, stride, color_space, sampling)
            }
            None => Err(JpegError::NotImplemented {
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

    /// 4:2:2 and 4:2:0 color: reconstruct full component planes, then
    /// upsample and convert them into `out`.
    fn decode_lossless_color_sampled_into<P: LosslessColorOutput>(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
        sampling: LosslessColorSampling,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self.lossless_plan_for::<P>()?;
        validate_lossless_color_plan::<P>(plan, &self.plan, &self.info, color_space)?;
        let order = lossless_output_order(&self.plan)?;
        let (width, height) = (plan.dimensions.0 as usize, plan.dimensions.1 as usize);
        let layout = lossless_sampled_plane_layout(&self.info, super::DEFAULT_MAX_DECODE_BYTES)?
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        let expected_plane_bytes = layout
            .chroma_len
            .checked_mul(2)
            .and_then(|chroma_bytes| layout.luma_len.checked_add(chroma_bytes))
            .and_then(|samples| samples.checked_mul(core::mem::size_of::<P>()))
            .ok_or(JpegError::InternalInvariant {
                reason: "lossless sampled-plane layout arithmetic diverged after validation",
            })?;
        if layout.total_bytes != expected_plane_bytes {
            return Err(JpegError::InternalInvariant {
                reason: "lossless sampled-plane layout total disagrees with its planes",
            });
        }
        if layout.total_bytes > self.plan.scratch_bytes {
            return Err(JpegError::MemoryCapExceeded {
                requested: layout.total_bytes,
                cap: self.plan.scratch_bytes,
            });
        }
        let chroma_width = layout.chroma_dimensions.0;
        let OwnedLosslessSampledPlanes {
            mut c0,
            mut c1,
            mut c2,
        } = allocate_lossless_sampled_planes::<P>(layout, self.plan.scratch_bytes)?;

        let mut scan = self.lossless_scan(plan)?;
        let shift = scan.point_transform();
        while scan.decode_mcu_row()? {
            for (output, &component) in order.iter().enumerate() {
                let (plane, plane_width) = match output {
                    0 => (&mut c0, width),
                    1 => (&mut c1, chroma_width),
                    _ => (&mut c2, chroma_width),
                };
                let base = scan.row_base(component);
                for index in 0..scan.rows_ready(component) {
                    let dst = &mut plane[(base + index) * plane_width..][..plane_width];
                    for (dst, &sample) in dst.iter_mut().zip(scan.row(component, index)) {
                        *dst = P::from_sample(sample << shift);
                    }
                }
            }
        }
        let outcome = self.finish_lossless_scan(scan)?;
        P::write_sampled(
            out,
            stride,
            color_space,
            sampling,
            (width, height),
            LosslessColorPlanes {
                c0: &c0,
                c1: &c1,
                c2: &c2,
            },
        );
        Ok(outcome)
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
