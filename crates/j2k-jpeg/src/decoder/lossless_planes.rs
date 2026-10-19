// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plane-rendered lossless color: subsampled layouts, split (multi-scan)
//! frames, and YCbCr at precisions other than 8 and 16 bits. Every scan
//! reconstructs its components into full-size planes in frame order. Output
//! then replicates each component by its sampling ratio, as libjpeg-turbo's
//! lossless upsampler (`int_upsample`) does, and converts YCbCr at the frame
//! precision.

use super::{
    checked_scratch_len, checked_usize_product, lossless_color_sampling, merged_warnings,
    ColorSpace, DecodeOutcome, Decoder, Info, JpegError, LosslessColorSampling, LosslessSample,
    PreparedLosslessPlan, Rect, RowSink, Vec,
};
use crate::allocation::{try_reserve_for_len_with_live_budget, try_resize_filled};
use crate::color::ycbcr::{ycbcr16_to_rgb16, ycbcr_to_rgb, ycbcr_to_rgb_with_precision};

#[cfg(test)]
mod tests;

/// One component plane of a plane-rendered frame, in samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct LosslessPlane {
    pub(super) offset: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    /// Output pixels each sample covers: `Hmax / h` by `Vmax / v`.
    pub(super) h_expand: usize,
    pub(super) v_expand: usize,
}

/// Frame-order component planes stored back to back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LosslessPlaneLayout {
    pub(super) planes: [LosslessPlane; 3],
    pub(super) total_samples: usize,
    pub(super) total_bytes: usize,
}

/// Plane layout of a three-component frame whose factors divide the maximum
/// factors, or `None` for any other frame.
pub(super) fn lossless_plane_layout(
    info: &Info,
    cap: usize,
) -> Result<Option<LosslessPlaneLayout>, JpegError> {
    if lossless_color_sampling(info).is_none() {
        return Ok(None);
    }
    let (width, height) = (info.dimensions.0 as usize, info.dimensions.1 as usize);
    let (max_h, max_v) = (
        usize::from(info.sampling.max_h),
        usize::from(info.sampling.max_v),
    );
    let mut planes = [LosslessPlane::default(); 3];
    let mut total_samples = 0usize;
    for (plane, (h, v)) in planes.iter_mut().zip(info.sampling.iter()) {
        let (h, v) = (usize::from(h), usize::from(v));
        let plane_width = checked_usize_product(&[width, h], cap)?.div_ceil(max_h);
        let plane_height = checked_usize_product(&[height, v], cap)?.div_ceil(max_v);
        *plane = LosslessPlane {
            offset: total_samples,
            width: plane_width,
            height: plane_height,
            h_expand: max_h / h,
            v_expand: max_v / v,
        };
        total_samples = total_samples
            .checked_add(checked_usize_product(&[plane_width, plane_height], cap)?)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    }
    let bytes_per_sample = if info.bit_depth > 8 { 2 } else { 1 };
    let total_bytes = checked_usize_product(&[total_samples, bytes_per_sample], cap)?;
    if total_bytes > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total_bytes,
            cap,
        });
    }
    Ok(Some(LosslessPlaneLayout {
        planes,
        total_samples,
        total_bytes,
    }))
}

/// Whether lossless color renders through planes instead of streaming one
/// interleaved 4:4:4 scan row by row.
pub(super) fn lossless_color_uses_planes(info: &Info, plan: &PreparedLosslessPlan) -> bool {
    !plan.split_scans.is_empty()
        || lossless_color_sampling(info) == Some(LosslessColorSampling::Sampled)
        || (info.color_space == ColorSpace::YCbCr && !matches!(plan.bit_depth, 8 | 16))
}

/// YCbCr to RGB at `precision`, with the 8- and 16-bit converters there.
#[expect(
    clippy::cast_possible_truncation,
    reason = "8-bit lossless samples fit u8"
)]
fn lossless_ycbcr_to_rgb(y: u16, cb: u16, cr: u16, precision: u8) -> (u16, u16, u16) {
    match precision {
        8 => {
            let (r, g, b) = ycbcr_to_rgb(y as u8, cb as u8, cr as u8);
            (r.into(), g.into(), b.into())
        }
        16 => ycbcr16_to_rgb16(y, cb, cr),
        _ => ycbcr_to_rgb_with_precision(y, cb, cr, precision),
    }
}

/// Fill output row `y` (`width` interleaved RGB pixels of `P`) from the
/// planes: every component's sample covering each pixel, color converted.
pub(super) fn fill_lossless_plane_row<P: LosslessSample>(
    row: &mut [u8],
    y: usize,
    width: usize,
    source: (&[P], &LosslessPlaneLayout),
    color: (ColorSpace, u8),
) {
    let (samples, layout) = source;
    let (color_space, precision) = color;
    let rows = layout.planes.map(|plane| {
        let start = plane.offset + (y / plane.v_expand) * plane.width;
        (&samples[start..start + plane.width], plane.h_expand)
    });
    for (x, pixel) in row.chunks_exact_mut(3 * P::BYTES).take(width).enumerate() {
        let [c0, c1, c2] = rows.map(|(plane_row, h_expand)| plane_row[x / h_expand].to_sample());
        let (r, g, b) = if color_space == ColorSpace::YCbCr {
            lossless_ycbcr_to_rgb(c0, c1, c2, precision)
        } else {
            (c0, c1, c2)
        };
        let (first, rest) = pixel.split_at_mut(P::BYTES);
        let (second, third) = rest.split_at_mut(P::BYTES);
        P::from_sample(r).write_le(first);
        P::from_sample(g).write_le(second);
        P::from_sample(b).write_le(third);
    }
}

impl Decoder<'_> {
    /// Reconstruct every scan of a color frame into frame-order planes,
    /// shifted by each scan's point transform.
    fn decode_lossless_planes<P: LosslessSample>(
        &self,
        plan: &PreparedLosslessPlan,
        layout: &LosslessPlaneLayout,
    ) -> Result<(Vec<P>, Vec<crate::error::Warning>), JpegError> {
        if layout.total_bytes > self.plan.scratch_bytes {
            return Err(JpegError::MemoryCapExceeded {
                requested: layout.total_bytes,
                cap: self.plan.scratch_bytes,
            });
        }
        let mut live_bytes = 0;
        let mut samples = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut samples,
            layout.total_samples,
            &mut live_bytes,
            self.plan.scratch_bytes,
        )?;
        samples.resize(layout.total_samples, P::default());

        let mut scan_warnings = Vec::new();
        for index in 0..plan.scan_count() {
            let view = self.lossless_scan_view(plan, index)?;
            let mut scan = self.open_lossless_scan(plan, view)?;
            let shift = scan.point_transform();
            while scan.decode_mcu_row()? {
                for (slot, component) in view.components.iter().enumerate() {
                    let plane = *layout.planes.get(component.output_index).ok_or(
                        JpegError::InternalInvariant {
                            reason: "lossless scan component exceeds the frame planes",
                        },
                    )?;
                    let base = scan.row_base(slot);
                    for row in 0..scan.rows_ready(slot) {
                        let start = plane.offset + (base + row) * plane.width;
                        for (dst, &sample) in samples[start..start + plane.width]
                            .iter_mut()
                            .zip(scan.row(slot, row))
                        {
                            *dst = P::from_sample(sample << shift);
                        }
                    }
                }
            }
            scan_warnings = scan.finish_with_eoi(view.validate_eoi)?;
        }
        Ok((samples, scan_warnings))
    }

    fn lossless_planes_for<P: LosslessSample>(
        &self,
        color_space: ColorSpace,
    ) -> Result<(&PreparedLosslessPlan, LosslessPlaneLayout), JpegError> {
        let plan = self.lossless_plan_for::<P>()?;
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
        Ok((plan, layout))
    }

    /// Plane-rendered color into `out`.
    pub(super) fn decode_lossless_color_planes_into<P: LosslessSample>(
        &self,
        out: &mut [u8],
        stride: usize,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError> {
        let (plan, layout) = self.lossless_planes_for::<P>(color_space)?;
        let (samples, scan_warnings) = self.decode_lossless_planes::<P>(plan, &layout)?;
        let width = plan.dimensions.0 as usize;
        let row_bytes = checked_scratch_len(&[width, 3, P::BYTES])?;
        for y in 0..plan.dimensions.1 as usize {
            fill_lossless_plane_row::<P>(
                &mut out[y * stride..][..row_bytes],
                y,
                width,
                (&samples, &layout),
                (color_space, plan.bit_depth),
            );
        }
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings)?,
        })
    }

    /// Plane-rendered color streamed through `sink`, assembling each row in
    /// `row`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "row indices are bounded by validated u32 image dimensions"
    )]
    pub(super) fn decode_lossless_color_plane_rows<P, S>(
        &self,
        sink: &mut S,
        row: &mut Vec<u8>,
        color_space: ColorSpace,
    ) -> Result<DecodeOutcome, JpegError>
    where
        P: LosslessSample,
        S: RowSink<u8, Error = JpegError>,
    {
        let (plan, layout) = self.lossless_planes_for::<P>(color_space)?;
        let (samples, scan_warnings) = self.decode_lossless_planes::<P>(plan, &layout)?;
        let width = plan.dimensions.0 as usize;
        let row_len = checked_scratch_len(&[width, 3, P::BYTES])?;
        try_resize_filled(row, row_len, 0)?;
        for y in 0..plan.dimensions.1 as usize {
            fill_lossless_plane_row::<P>(
                &mut row[..row_len],
                y,
                width,
                (&samples, &layout),
                (color_space, plan.bit_depth),
            );
            sink.write_row(y as u32, &row[..row_len])?;
        }
        Ok(DecodeOutcome {
            decoded: Rect::full(self.info.dimensions),
            warnings: merged_warnings(&self.warnings, scan_warnings)?,
        })
    }
}
