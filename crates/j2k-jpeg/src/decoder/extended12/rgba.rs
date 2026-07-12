// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared RGB16-to-RGBA16 projection for 12-bit sequential/progressive decode.

use super::super::{
    allocate_output_buffer_with_live_budget, checked_scratch_len, copy_rgb16_to_rgba16,
    scaled_rect_covering, DecodeOutcome, Decoder, DownscaleFactor, JpegError, Rect, SofKind,
};

impl Decoder<'_> {
    pub(in crate::decoder) fn decode_12bit_rgba16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        alpha: u16,
        external_live_bytes: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        let output_rect = scaled_rect_covering(roi, downscale)?;
        let rgb_stride = output_rect.w as usize * 6;
        let rgb_len = checked_scratch_len(&[rgb_stride, output_rect.h as usize])?;
        let (mut live_bytes, workspace_cap) = self.decode_phase_live_bytes(external_live_bytes)?;
        let mut rgb =
            allocate_output_buffer_with_live_budget(rgb_len, &mut live_bytes, workspace_cap)?;
        let outcome = if self.info.sof_kind == SofKind::Progressive12 {
            self.decode_progressive12_rgb16_region_scaled_into(&mut rgb, rgb_stride, roi, downscale)
        } else {
            self.decode_extended12_rgb16_region_scaled_into(&mut rgb, rgb_stride, roi, downscale)
        }?;
        copy_rgb16_to_rgba16(
            &rgb,
            rgb_stride,
            output_rect.w,
            output_rect.h,
            out,
            stride,
            alpha,
        );
        Ok(outcome)
    }
}
