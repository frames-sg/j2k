// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::borrow::Cow;
use alloc::vec::Vec;

use crate::adapter::JpegBaselineSampling;
use crate::allocation::try_new_vec_with_live_budget;

use super::planning::{checked_sample_byte_len, map_allocation_budget_error};
use super::{JpegEncodeError, JpegSamples, JpegSubsampling};

impl JpegSamples<'_> {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Gray8 { .. } => "Gray8",
            Self::Rgb8 { .. } => "Rgb8",
        }
    }

    pub(super) fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Gray8 { width, height, .. } | Self::Rgb8 { width, height, .. } => (width, height),
        }
    }

    pub(super) fn data_len(self) -> usize {
        match self {
            Self::Gray8 { data, .. } | Self::Rgb8 { data, .. } => data.len(),
        }
    }
}

pub(super) fn validate_sample_layout(
    samples: JpegSamples<'_>,
    subsampling: JpegSubsampling,
) -> Result<usize, JpegEncodeError> {
    let (width, height, components, name) = match samples {
        JpegSamples::Gray8 { width, height, .. } => (width, height, 1usize, "Gray8"),
        JpegSamples::Rgb8 { width, height, .. } => (width, height, 3usize, "Rgb8"),
    };
    let expected = checked_sample_byte_len(width, height, components)?;
    match (name, subsampling) {
        ("Gray8", JpegSubsampling::Gray)
        | ("Rgb8", JpegSubsampling::Ybr444 | JpegSubsampling::Ybr422 | JpegSubsampling::Ybr420) => {
            Ok(expected)
        }
        _ => Err(JpegEncodeError::IncompatibleSubsampling {
            subsampling,
            samples: name,
        }),
    }
}

pub(super) fn component_planes(
    samples: JpegSamples<'_>,
    subsampling: JpegSubsampling,
    plane_capacity_limit: usize,
) -> Result<Vec<Cow<'_, [u8]>>, JpegEncodeError> {
    let mut live_bytes = 0;
    match samples {
        JpegSamples::Gray8 {
            data,
            width,
            height,
        } => {
            checked_sample_byte_len(width, height, 1)?;
            let mut planes = try_new_vec_with_live_budget(1, &mut live_bytes, plane_capacity_limit)
                .map_err(map_allocation_budget_error)?;
            planes.push(Cow::Borrowed(data));
            Ok(planes)
        }
        JpegSamples::Rgb8 {
            data,
            width,
            height,
        } => {
            if subsampling == JpegSubsampling::Gray {
                return Err(JpegEncodeError::IncompatibleSubsampling {
                    subsampling,
                    samples: "Rgb8",
                });
            }
            let sample_bytes = checked_sample_byte_len(width, height, 3)?;
            let pixels = sample_bytes / 3;
            let logical_plane_bytes = core::mem::size_of::<Cow<'_, [u8]>>()
                .checked_mul(3)
                .and_then(|metadata| metadata.checked_add(sample_bytes))
                .ok_or(JpegEncodeError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: plane_capacity_limit,
                })?;
            if logical_plane_bytes > plane_capacity_limit {
                return Err(JpegEncodeError::MemoryCapExceeded {
                    requested: logical_plane_bytes,
                    cap: plane_capacity_limit,
                });
            }
            let mut planes = try_new_vec_with_live_budget(3, &mut live_bytes, plane_capacity_limit)
                .map_err(map_allocation_budget_error)?;
            let mut y_plane =
                try_new_vec_with_live_budget(pixels, &mut live_bytes, plane_capacity_limit)
                    .map_err(map_allocation_budget_error)?;
            let mut cb_plane =
                try_new_vec_with_live_budget(pixels, &mut live_bytes, plane_capacity_limit)
                    .map_err(map_allocation_budget_error)?;
            let mut cr_plane =
                try_new_vec_with_live_budget(pixels, &mut live_bytes, plane_capacity_limit)
                    .map_err(map_allocation_budget_error)?;
            for rgb in data.chunks_exact(3) {
                let (y, cb, cr) = rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
                y_plane.push(y);
                cb_plane.push(cb);
                cr_plane.push(cr);
            }
            planes.push(Cow::Owned(y_plane));
            planes.push(Cow::Owned(cb_plane));
            planes.push(Cow::Owned(cr_plane));
            Ok(planes)
        }
    }
}

fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = i32::from(r);
    let g = i32::from(g);
    let b = i32::from(b);
    let y = (19_595 * r + 38_470 * g + 7_471 * b + 32_768) >> 16;
    let cb = (-11_059 * r - 21_709 * g + 32_768 * b + 8_421_376) >> 16;
    let cr = (32_768 * r - 27_439 * g - 5_329 * b + 8_421_376) >> 16;
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

#[expect(
    clippy::cast_sign_loss,
    reason = "RGB-to-YCbCr arithmetic is clamped to the u8 sample range before conversion"
)]
fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

#[expect(
    clippy::too_many_arguments,
    reason = "private JPEG sample hot path keeps scalar arguments for optimized codegen"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "edge-replicated source coordinates address validated u8 sample planes"
)]
pub(super) fn sample_block(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    component: usize,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u8,
    block_y: u8,
) -> [u8; 64] {
    let mut out = [0u8; 64];
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let comp_h = u32::from(sampling.h[component]);
    let comp_v = u32::from(sampling.v[component]);
    let x_scale = max_h / comp_h;
    let y_scale = max_v / comp_v;
    let mcu_origin_x = mcu_x * max_h * 8;
    let mcu_origin_y = mcu_y * max_v * 8;
    for y in 0..8u32 {
        for x in 0..8u32 {
            let value = if component == 0 {
                let sx = (mcu_origin_x + u32::from(block_x) * 8 + x).min(width - 1);
                let sy = (mcu_origin_y + u32::from(block_y) * 8 + y).min(height - 1);
                planes[component][(sy as usize * width as usize) + sx as usize]
            } else {
                let mut sum = 0u32;
                for dy in 0..y_scale {
                    for dx in 0..x_scale {
                        let sx = (mcu_origin_x + (u32::from(block_x) * 8 + x) * x_scale + dx)
                            .min(width - 1);
                        let sy = (mcu_origin_y + (u32::from(block_y) * 8 + y) * y_scale + dy)
                            .min(height - 1);
                        sum += u32::from(
                            planes[component][sy as usize * width as usize + sx as usize],
                        );
                    }
                }
                (sum / (x_scale * y_scale)) as u8
            };
            out[(y * 8 + x) as usize] = value;
        }
    }
    out
}
