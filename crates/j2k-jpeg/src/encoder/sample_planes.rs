// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use crate::adapter::JpegBaselineSampling;
use crate::allocation::try_new_vec_with_live_budget;

use super::planning::{checked_sample_byte_len, map_allocation_budget_error};
use super::{JpegEncodeError, JpegSamples, JpegSubsampling};

#[cfg(target_arch = "aarch64")]
#[path = "sample_planes_neon.rs"]
mod neon;

#[derive(Clone, Copy, Debug)]
pub(super) struct PrecomputedChromaLayout {
    width: usize,
    height: usize,
    x_scale: usize,
    y_scale: usize,
}

impl PrecomputedChromaLayout {
    fn len(self) -> Option<usize> {
        self.width.checked_mul(self.height)
    }
}

fn precomputed_chroma_layout(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    component: usize,
) -> Option<PrecomputedChromaLayout> {
    if component == 0 || component >= usize::from(sampling.components) {
        return None;
    }
    let (x_scale, y_scale) = match sampling {
        JpegBaselineSampling {
            components: 3,
            h: [2, 1, 1],
            v: [1, 1, 1],
            max_h: 2,
            max_v: 1,
        } => (2usize, 1usize),
        JpegBaselineSampling {
            components: 3,
            h: [2, 1, 1],
            v: [2, 1, 1],
            max_h: 2,
            max_v: 2,
        } => (2usize, 2usize),
        _ => return None,
    };
    let mcu_columns = width.div_ceil(u32::from(sampling.max_h) * 8);
    let mcu_rows = height.div_ceil(u32::from(sampling.max_v) * 8);
    let plane_width = usize::try_from(mcu_columns)
        .ok()?
        .checked_mul(usize::from(sampling.h[component]))?
        .checked_mul(8)?;
    let plane_height = usize::try_from(mcu_rows)
        .ok()?
        .checked_mul(usize::from(sampling.v[component]))?
        .checked_mul(8)?;
    let layout = PrecomputedChromaLayout {
        width: plane_width,
        height: plane_height,
        x_scale,
        y_scale,
    };
    let full_resolution_len = usize::try_from(width)
        .ok()?
        .checked_mul(usize::try_from(height).ok()?)?;
    (layout.len()? <= full_resolution_len).then_some(layout)
}

// The RGB producer and block sampler share this private contiguous-8x8 layout.
#[expect(
    clippy::manual_is_multiple_of,
    reason = "explicit alignment divisors document the block-plane layout contract"
)]
pub(super) fn uses_block_plane_layout(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
) -> bool {
    width != 0
        && height != 0
        && width % 16 == 0
        && height % 8 == 0
        && matches!(
            sampling,
            JpegBaselineSampling {
                components: 3,
                h: [1, 1, 1],
                v: [1, 1, 1],
                max_h: 1,
                max_v: 1,
            }
        )
}

#[expect(
    clippy::manual_is_multiple_of,
    reason = "explicit alignment divisors document the 4:2:2 block-plane layout contract"
)]
pub(super) fn uses_422_block_plane_layout(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
) -> bool {
    width != 0
        && height != 0
        && width % 16 == 0
        && height % 8 == 0
        && matches!(
            sampling,
            JpegBaselineSampling {
                components: 3,
                h: [2, 1, 1],
                v: [1, 1, 1],
                max_h: 2,
                max_v: 1,
            }
        )
}

#[expect(
    clippy::manual_is_multiple_of,
    reason = "explicit alignment divisors document the 4:2:0 block-plane layout contract"
)]
pub(super) fn uses_420_block_plane_layout(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
) -> bool {
    width != 0
        && height != 0
        && width % 16 == 0
        && height % 16 == 0
        && matches!(
            sampling,
            JpegBaselineSampling {
                components: 3,
                h: [2, 1, 1],
                v: [2, 1, 1],
                max_h: 2,
                max_v: 2,
            }
        )
}

pub(super) fn uses_compact_block_plane_layout(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
) -> bool {
    if width < 16
        || height < 8
        || uses_block_plane_layout(width, height, sampling)
        || uses_422_block_plane_layout(width, height, sampling)
        || uses_420_block_plane_layout(width, height, sampling)
    {
        return false;
    }
    match sampling {
        JpegBaselineSampling {
            components: 3,
            h: [1, 1, 1],
            v: [1, 1, 1],
            max_h: 1,
            max_v: 1,
        } => true,
        JpegBaselineSampling {
            components: 3,
            h: [2, 1, 1],
            v: [1, 1, 1],
            max_h: 2,
            max_v: 1,
        }
        | JpegBaselineSampling {
            components: 3,
            h: [2, 1, 1],
            v: [2, 1, 1],
            max_h: 2,
            max_v: 2,
        } => precomputed_chroma_layout(width, height, sampling, 1).is_some(),
        _ => false,
    }
}

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

#[expect(
    clippy::too_many_lines,
    reason = "the central plane constructor preserves measured architecture dispatch and shared allocation ownership"
)]
#[cfg_attr(
    target_arch = "aarch64",
    expect(
        clippy::if_not_else,
        reason = "the NEON branch keeps the enabled architecture path first"
    )
)]
pub(super) fn component_planes(
    samples: JpegSamples<'_>,
    subsampling: JpegSubsampling,
    sampling: JpegBaselineSampling,
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
            let block_layout = uses_block_plane_layout(width, height, sampling);
            let block_layout_422 = uses_422_block_plane_layout(width, height, sampling);
            let block_layout_420 = uses_420_block_plane_layout(width, height, sampling);
            let compact_block_layout = uses_compact_block_plane_layout(width, height, sampling);
            let chroma_layout = precomputed_chroma_layout(width, height, sampling, 1);
            let chroma_pixels = chroma_layout
                .and_then(PrecomputedChromaLayout::len)
                .unwrap_or(pixels);
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
                try_new_vec_with_live_budget(chroma_pixels, &mut live_bytes, plane_capacity_limit)
                    .map_err(map_allocation_budget_error)?;
            let mut cr_plane =
                try_new_vec_with_live_budget(chroma_pixels, &mut live_bytes, plane_capacity_limit)
                    .map_err(map_allocation_budget_error)?;
            if block_layout_422 {
                #[cfg(target_arch = "aarch64")]
                if cfg!(feature = "scalar-only") {
                    fill_rgb_to_ycbcr_422_block_planes_scalar(
                        data,
                        width as usize,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                } else if let Some(neon) = fearless_simd::Level::new().as_neon() {
                    neon::rgb_to_ycbcr_422_block_planes(
                        neon,
                        data,
                        width as usize,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                } else {
                    fill_rgb_to_ycbcr_422_block_planes_scalar(
                        data,
                        width as usize,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                fill_rgb_to_ycbcr_422_block_planes_scalar(
                    data,
                    width as usize,
                    &mut y_plane,
                    &mut cb_plane,
                    &mut cr_plane,
                );
            } else if block_layout_420 {
                #[cfg(target_arch = "aarch64")]
                if cfg!(feature = "scalar-only") {
                    fill_rgb_to_ycbcr_420_block_planes_scalar(
                        data,
                        width as usize,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                } else if let Some(neon) = fearless_simd::Level::new().as_neon() {
                    neon::rgb_to_ycbcr_420_block_planes(
                        neon,
                        data,
                        width as usize,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                } else {
                    fill_rgb_to_ycbcr_420_block_planes_scalar(
                        data,
                        width as usize,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                fill_rgb_to_ycbcr_420_block_planes_scalar(
                    data,
                    width as usize,
                    &mut y_plane,
                    &mut cb_plane,
                    &mut cr_plane,
                );
            } else if compact_block_layout {
                if let Some(layout) = chroma_layout {
                    #[cfg(target_arch = "aarch64")]
                    if cfg!(feature = "scalar-only") {
                        fill_rgb_to_ycbcr_compact_subsampled_scalar(
                            data,
                            width as usize,
                            height as usize,
                            layout,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    } else if let Some(neon) = fearless_simd::Level::new().as_neon() {
                        neon::rgb_to_ycbcr_compact_subsampled_planes(
                            neon,
                            data,
                            width as usize,
                            height as usize,
                            layout,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    } else {
                        fill_rgb_to_ycbcr_compact_subsampled_scalar(
                            data,
                            width as usize,
                            height as usize,
                            layout,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    fill_rgb_to_ycbcr_compact_subsampled_scalar(
                        data,
                        width as usize,
                        height as usize,
                        layout,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                } else {
                    #[cfg(target_arch = "aarch64")]
                    if cfg!(feature = "scalar-only") {
                        fill_rgb_to_ycbcr_compact_block_planes_scalar(
                            data,
                            width as usize,
                            height as usize,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    } else if let Some(neon) = fearless_simd::Level::new().as_neon() {
                        neon::rgb_to_ycbcr_compact_block_planes(
                            neon,
                            data,
                            width as usize,
                            height as usize,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    } else {
                        fill_rgb_to_ycbcr_compact_block_planes_scalar(
                            data,
                            width as usize,
                            height as usize,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    fill_rgb_to_ycbcr_compact_block_planes_scalar(
                        data,
                        width as usize,
                        height as usize,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                }
            } else if let Some(layout) = chroma_layout {
                #[cfg(target_arch = "aarch64")]
                if !cfg!(feature = "scalar-only") {
                    if let Some(neon) = fearless_simd::Level::new().as_neon() {
                        neon::rgb_to_ycbcr_subsampled_planes(
                            neon,
                            data,
                            width as usize,
                            height as usize,
                            layout,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    } else {
                        fill_rgb_to_ycbcr_subsampled_scalar(
                            data,
                            width as usize,
                            height as usize,
                            layout,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    }
                } else {
                    fill_rgb_to_ycbcr_subsampled_scalar(
                        data,
                        width as usize,
                        height as usize,
                        layout,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                }
                #[cfg(not(target_arch = "aarch64"))]
                fill_rgb_to_ycbcr_subsampled_scalar(
                    data,
                    width as usize,
                    height as usize,
                    layout,
                    &mut y_plane,
                    &mut cb_plane,
                    &mut cr_plane,
                );
            } else {
                #[cfg(target_arch = "aarch64")]
                if cfg!(feature = "scalar-only") {
                    if block_layout {
                        fill_rgb_to_ycbcr_block_planes_scalar(
                            data,
                            width as usize,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    } else {
                        fill_rgb_to_ycbcr_scalar(data, &mut y_plane, &mut cb_plane, &mut cr_plane);
                    }
                } else if let Some(neon) = fearless_simd::Level::new().as_neon() {
                    if block_layout {
                        neon::rgb_to_ycbcr_block_planes(
                            neon,
                            data,
                            width as usize,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    } else {
                        neon::rgb_to_ycbcr_planes(
                            neon,
                            data,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    }
                } else {
                    if block_layout {
                        fill_rgb_to_ycbcr_block_planes_scalar(
                            data,
                            width as usize,
                            &mut y_plane,
                            &mut cb_plane,
                            &mut cr_plane,
                        );
                    } else {
                        fill_rgb_to_ycbcr_scalar(data, &mut y_plane, &mut cb_plane, &mut cr_plane);
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                if block_layout {
                    fill_rgb_to_ycbcr_block_planes_scalar(
                        data,
                        width as usize,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                    );
                } else {
                    fill_rgb_to_ycbcr_scalar(data, &mut y_plane, &mut cb_plane, &mut cr_plane);
                }
            }
            planes.push(Cow::Owned(y_plane));
            planes.push(Cow::Owned(cb_plane));
            planes.push(Cow::Owned(cr_plane));
            Ok(planes)
        }
    }
}

fn fill_rgb_to_ycbcr_subsampled_scalar(
    data: &[u8],
    width: usize,
    height: usize,
    layout: PrecomputedChromaLayout,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_eq!(layout.x_scale, 2);
    assert!(matches!(layout.y_scale, 1 | 2));
    let logical_chroma_width = width.div_ceil(layout.x_scale);
    let logical_chroma_height = height.div_ceil(layout.y_scale);
    assert!(layout.width >= logical_chroma_width);
    assert!(layout.height >= logical_chroma_height);
    let y_required = width * height;
    let chroma_required = layout.width * layout.height;
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    {
        let y_spare = y_plane.spare_capacity_mut();
        let cb_spare = cb_plane.spare_capacity_mut();
        let cr_spare = cr_plane.spare_capacity_mut();
        assert!(y_spare.len() >= y_required);
        assert!(cb_spare.len() >= chroma_required);
        assert!(cr_spare.len() >= chroma_required);
        let y_out = &mut y_spare[..y_required];
        let cb_out = &mut cb_spare[..chroma_required];
        let cr_out = &mut cr_spare[..chroma_required];
        for cy in 0..logical_chroma_height {
            let top_y = cy * layout.y_scale;
            let bottom_y = (top_y + layout.y_scale - 1).min(height - 1);
            fill_subsampled_chroma_row_scalar(
                data,
                width,
                top_y,
                bottom_y,
                layout,
                Some(&mut *y_out),
                &mut cb_out[cy * layout.width..(cy + 1) * layout.width],
                &mut cr_out[cy * layout.width..(cy + 1) * layout.width],
            );
        }
        if logical_chroma_height < layout.height {
            let edge = logical_chroma_height * layout.width;
            fill_subsampled_chroma_row_scalar(
                data,
                width,
                height - 1,
                height - 1,
                layout,
                None,
                &mut cb_out[edge..edge + layout.width],
                &mut cr_out[edge..edge + layout.width],
            );
            for cy in logical_chroma_height + 1..layout.height {
                cb_out.copy_within(edge..edge + layout.width, cy * layout.width);
                cr_out.copy_within(edge..edge + layout.width, cy * layout.width);
            }
        }
    }
    // SAFETY: the row conversion initializes each real Y sample and every
    // padded Cb/Cr cell in the capacity slices exposed above.
    unsafe {
        y_plane.set_len(y_len + y_required);
        cb_plane.set_len(cb_len + chroma_required);
        cr_plane.set_len(cr_len + chroma_required);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    reason = "fused row conversion writes Y and bounded floor averages for both chroma rows together"
)]
fn fill_subsampled_chroma_row_scalar(
    data: &[u8],
    width: usize,
    top_y: usize,
    bottom_y: usize,
    layout: PrecomputedChromaLayout,
    mut y_plane: Option<&mut [MaybeUninit<u8>]>,
    cb_row: &mut [MaybeUninit<u8>],
    cr_row: &mut [MaybeUninit<u8>],
) {
    let logical_chroma_width = width.div_ceil(layout.x_scale);
    for cx in 0..layout.width {
        let left_x = (cx * layout.x_scale).min(width - 1);
        let right_x = (left_x + layout.x_scale - 1).min(width - 1);
        let top_left = rgb_at(data, width, left_x, top_y);
        let top_right = rgb_at(data, width, right_x, top_y);
        let (y00, cb00, cr00) = rgb_to_ycbcr(top_left[0], top_left[1], top_left[2]);
        let (y01, cb01, cr01) = rgb_to_ycbcr(top_right[0], top_right[1], top_right[2]);
        let (cb_sum, cr_sum, sample_count) = if layout.y_scale == 2 {
            let bottom_left = rgb_at(data, width, left_x, bottom_y);
            let bottom_right = rgb_at(data, width, right_x, bottom_y);
            let (y10, cb10, cr10) = rgb_to_ycbcr(bottom_left[0], bottom_left[1], bottom_left[2]);
            let (y11, cb11, cr11) = rgb_to_ycbcr(bottom_right[0], bottom_right[1], bottom_right[2]);
            if cx < logical_chroma_width && bottom_y != top_y {
                if let Some(y_plane) = y_plane.as_deref_mut() {
                    y_plane[bottom_y * width + left_x].write(y10);
                    if right_x != left_x {
                        y_plane[bottom_y * width + right_x].write(y11);
                    }
                }
            }
            (
                u16::from(cb00) + u16::from(cb01) + u16::from(cb10) + u16::from(cb11),
                u16::from(cr00) + u16::from(cr01) + u16::from(cr10) + u16::from(cr11),
                4,
            )
        } else {
            (
                u16::from(cb00) + u16::from(cb01),
                u16::from(cr00) + u16::from(cr01),
                2,
            )
        };
        if cx < logical_chroma_width {
            if let Some(y_plane) = y_plane.as_deref_mut() {
                y_plane[top_y * width + left_x].write(y00);
                if right_x != left_x {
                    y_plane[top_y * width + right_x].write(y01);
                }
            }
        }
        cb_row[cx].write((cb_sum / sample_count) as u8);
        cr_row[cx].write((cr_sum / sample_count) as u8);
    }
}

fn rgb_at(data: &[u8], width: usize, x: usize, y: usize) -> &[u8; 3] {
    data[y * width * 3 + x * 3..]
        .first_chunk()
        .expect("validated RGB sample coordinates")
}

fn fill_rgb_to_ycbcr_scalar(
    data: &[u8],
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    for rgb in data.as_chunks::<3>().0 {
        let (y, cb, cr) = rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
        y_plane.push(y);
        cb_plane.push(cb);
        cr_plane.push(cr);
    }
}

fn fill_rgb_to_ycbcr_compact_block_planes_scalar(
    data: &[u8],
    width: usize,
    height: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    let required = width
        .checked_mul(height)
        .expect("compact 4:4:4 plane length fits usize");
    assert_eq!(data.len(), required * 3);
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    {
        let y_spare = y_plane.spare_capacity_mut();
        let cb_spare = cb_plane.spare_capacity_mut();
        let cr_spare = cr_plane.spare_capacity_mut();
        assert!(y_spare.len() >= required);
        assert!(cb_spare.len() >= required);
        assert!(cr_spare.len() >= required);
        for y in 0..height {
            for x in 0..width {
                let rgb = &data[(y * width + x) * 3..][..3];
                let (luma, cb, cr) = rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
                let offset = compact_plane_sample_offset(width, height, x, y);
                y_spare[offset].write(luma);
                cb_spare[offset].write(cb);
                cr_spare[offset].write(cr);
            }
        }
    }
    // SAFETY: every real component sample initializes its unique compact-band
    // position, and the layout is a permutation of the width-by-height plane.
    unsafe {
        y_plane.set_len(y_len + required);
        cb_plane.set_len(cb_len + required);
        cr_plane.set_len(cr_len + required);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "floor averages of at most four u8 chroma samples remain in the u8 range"
)]
fn fill_rgb_to_ycbcr_compact_subsampled_scalar(
    data: &[u8],
    width: usize,
    height: usize,
    layout: PrecomputedChromaLayout,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_eq!(layout.x_scale, 2);
    assert!(matches!(layout.y_scale, 1 | 2));
    assert_eq!(layout.width % 8, 0);
    assert_eq!(layout.height % 8, 0);
    let logical_chroma_width = width.div_ceil(layout.x_scale);
    let logical_chroma_height = height.div_ceil(layout.y_scale);
    assert!(layout.width >= logical_chroma_width);
    assert!(layout.height >= logical_chroma_height);
    let y_required = width * height;
    let chroma_required = layout.width * layout.height;
    assert_eq!(data.len(), y_required * 3);
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    {
        let y_spare = y_plane.spare_capacity_mut();
        let cb_spare = cb_plane.spare_capacity_mut();
        let cr_spare = cr_plane.spare_capacity_mut();
        assert!(y_spare.len() >= y_required);
        assert!(cb_spare.len() >= chroma_required);
        assert!(cr_spare.len() >= chroma_required);
        for cy in 0..layout.height {
            let top_y = (cy * layout.y_scale).min(height - 1);
            let bottom_y = (top_y + layout.y_scale - 1).min(height - 1);
            let write_y = cy < logical_chroma_height;
            for cx in 0..layout.width {
                let left_x = (cx * layout.x_scale).min(width - 1);
                let right_x = (left_x + layout.x_scale - 1).min(width - 1);
                let top_left = rgb_at(data, width, left_x, top_y);
                let top_right = rgb_at(data, width, right_x, top_y);
                let (y00, cb00, cr00) = rgb_to_ycbcr(top_left[0], top_left[1], top_left[2]);
                let (y01, cb01, cr01) = rgb_to_ycbcr(top_right[0], top_right[1], top_right[2]);
                let (cb_sum, cr_sum, sample_count) = if layout.y_scale == 2 {
                    let bottom_left = rgb_at(data, width, left_x, bottom_y);
                    let bottom_right = rgb_at(data, width, right_x, bottom_y);
                    let (y10, cb10, cr10) =
                        rgb_to_ycbcr(bottom_left[0], bottom_left[1], bottom_left[2]);
                    let (y11, cb11, cr11) =
                        rgb_to_ycbcr(bottom_right[0], bottom_right[1], bottom_right[2]);
                    if write_y && cx < logical_chroma_width && bottom_y != top_y {
                        y_spare[compact_plane_sample_offset(width, height, left_x, bottom_y)]
                            .write(y10);
                        if right_x != left_x {
                            y_spare[compact_plane_sample_offset(width, height, right_x, bottom_y)]
                                .write(y11);
                        }
                    }
                    (
                        u16::from(cb00) + u16::from(cb01) + u16::from(cb10) + u16::from(cb11),
                        u16::from(cr00) + u16::from(cr01) + u16::from(cr10) + u16::from(cr11),
                        4,
                    )
                } else {
                    (
                        u16::from(cb00) + u16::from(cb01),
                        u16::from(cr00) + u16::from(cr01),
                        2,
                    )
                };
                if write_y && cx < logical_chroma_width {
                    y_spare[compact_plane_sample_offset(width, height, left_x, top_y)].write(y00);
                    if right_x != left_x {
                        y_spare[compact_plane_sample_offset(width, height, right_x, top_y)]
                            .write(y01);
                    }
                }
                let chroma_offset = block_plane_sample_offset(layout.width, layout.height, cx, cy);
                cb_spare[chroma_offset].write((cb_sum / sample_count) as u8);
                cr_spare[chroma_offset].write((cr_sum / sample_count) as u8);
            }
        }
    }
    // SAFETY: logical source coordinates initialize every compact Y position;
    // clamped padded chroma coordinates initialize both complete block planes.
    unsafe {
        y_plane.set_len(y_len + y_required);
        cb_plane.set_len(cb_len + chroma_required);
        cr_plane.set_len(cr_len + chroma_required);
    }
}

fn fill_rgb_to_ycbcr_block_planes_scalar(
    data: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_ne!(width, 0);
    assert_eq!(width % 16, 0);
    let row_bytes = width
        .checked_mul(3)
        .expect("4:4:4 block-plane row bytes fit usize");
    let block_row_bytes = row_bytes
        .checked_mul(8)
        .expect("4:4:4 block-plane row group fits usize");
    assert_eq!(data.len() % block_row_bytes, 0);
    let required = data.len() / 3;
    let blocks_per_row = width / 8;
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    {
        let y_spare = y_plane.spare_capacity_mut();
        let cb_spare = cb_plane.spare_capacity_mut();
        let cr_spare = cr_plane.spare_capacity_mut();
        assert!(y_spare.len() >= required);
        assert!(cb_spare.len() >= required);
        assert!(cr_spare.len() >= required);
        for (block_row, rows) in data.chunks_exact(block_row_bytes).enumerate() {
            for (row_in_block, row) in rows.chunks_exact(row_bytes).enumerate() {
                for (x, rgb) in row.as_chunks::<3>().0.iter().enumerate() {
                    let block = block_row * blocks_per_row + x / 8;
                    let offset = block * 64 + row_in_block * 8 + x % 8;
                    let (y, cb, cr) = rgb_to_ycbcr(rgb[0], rgb[1], rgb[2]);
                    y_spare[offset].write(y);
                    cb_spare[offset].write(cb);
                    cr_spare[offset].write(cr);
                }
            }
        }
    }
    // SAFETY: each source pixel initializes its unique position in all three
    // complete block-major plane prefixes.
    unsafe {
        y_plane.set_len(y_len + required);
        cb_plane.set_len(cb_len + required);
        cr_plane.set_len(cr_len + required);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::manual_midpoint,
    reason = "the explicit widened floor average cannot overflow and preserves the measured scalar loop"
)]
fn fill_rgb_to_ycbcr_422_block_planes_scalar(
    data: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_ne!(width, 0);
    assert_eq!(width % 16, 0);
    let row_bytes = width
        .checked_mul(3)
        .expect("4:2:2 block-plane row bytes fit usize");
    let block_row_bytes = row_bytes
        .checked_mul(8)
        .expect("4:2:2 block-plane row group fits usize");
    assert_eq!(data.len() % block_row_bytes, 0);
    let y_required = data.len() / 3;
    let chroma_required = y_required / 2;
    let mcus_per_row = width / 16;
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    {
        let y_spare = y_plane.spare_capacity_mut();
        let cb_spare = cb_plane.spare_capacity_mut();
        let cr_spare = cr_plane.spare_capacity_mut();
        assert!(y_spare.len() >= y_required);
        assert!(cb_spare.len() >= chroma_required);
        assert!(cr_spare.len() >= chroma_required);
        for (block_row, rows) in data.chunks_exact(block_row_bytes).enumerate() {
            for (row_in_block, row) in rows.chunks_exact(row_bytes).enumerate() {
                for (mcu_x, rgb) in row.as_chunks::<48>().0.iter().enumerate() {
                    let mcu = block_row * mcus_per_row + mcu_x;
                    let y_offset = mcu * 128 + row_in_block * 8;
                    let chroma_offset = mcu * 64 + row_in_block * 8;
                    for (pair_x, pair) in rgb.as_chunks::<6>().0.iter().enumerate() {
                        let (y0, cb0, cr0) = rgb_to_ycbcr(pair[0], pair[1], pair[2]);
                        let (y1, cb1, cr1) = rgb_to_ycbcr(pair[3], pair[4], pair[5]);
                        let pair_y_offset = y_offset + (pair_x / 4) * 64 + (pair_x % 4) * 2;
                        y_spare[pair_y_offset].write(y0);
                        y_spare[pair_y_offset + 1].write(y1);
                        cb_spare[chroma_offset + pair_x]
                            .write(((u16::from(cb0) + u16::from(cb1)) / 2) as u8);
                        cr_spare[chroma_offset + pair_x]
                            .write(((u16::from(cr0) + u16::from(cr1)) / 2) as u8);
                    }
                }
            }
        }
    }
    // SAFETY: every aligned MCU initializes two Y blocks and one block in
    // each chroma plane, covering each reserved prefix exactly once.
    unsafe {
        y_plane.set_len(y_len + y_required);
        cb_plane.set_len(cb_len + chroma_required);
        cr_plane.set_len(cr_len + chroma_required);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the floor average of four u8 chroma samples remains in the u8 range"
)]
fn fill_rgb_to_ycbcr_420_block_planes_scalar(
    data: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    cb_plane: &mut Vec<u8>,
    cr_plane: &mut Vec<u8>,
) {
    assert_ne!(width, 0);
    assert_eq!(width % 16, 0);
    let row_bytes = width
        .checked_mul(3)
        .expect("4:2:0 block-plane row bytes fit usize");
    let block_row_bytes = row_bytes
        .checked_mul(16)
        .expect("4:2:0 block-plane row group fits usize");
    assert_eq!(data.len() % block_row_bytes, 0);
    let y_required = data.len() / 3;
    let chroma_required = y_required / 4;
    let mcus_per_row = width / 16;
    let y_len = y_plane.len();
    let cb_len = cb_plane.len();
    let cr_len = cr_plane.len();
    {
        let y_spare = y_plane.spare_capacity_mut();
        let cb_spare = cb_plane.spare_capacity_mut();
        let cr_spare = cr_plane.spare_capacity_mut();
        assert!(y_spare.len() >= y_required);
        assert!(cb_spare.len() >= chroma_required);
        assert!(cr_spare.len() >= chroma_required);
        for (block_row, rows) in data.chunks_exact(block_row_bytes).enumerate() {
            for (row_pair, rows) in rows.chunks_exact(row_bytes * 2).enumerate() {
                let (top, bottom) = rows.split_at(row_bytes);
                for (mcu_x, (top_rgb, bottom_rgb)) in top
                    .as_chunks::<48>()
                    .0
                    .iter()
                    .zip(bottom.as_chunks::<48>().0.iter())
                    .enumerate()
                {
                    let mcu = block_row * mcus_per_row + mcu_x;
                    let y_offset = mcu * 256 + (row_pair / 4) * 128 + (row_pair % 4) * 16;
                    let chroma_offset = mcu * 64 + row_pair * 8;
                    for (pair_x, (top_pair, bottom_pair)) in top_rgb
                        .as_chunks::<6>()
                        .0
                        .iter()
                        .zip(bottom_rgb.as_chunks::<6>().0.iter())
                        .enumerate()
                    {
                        let (y00, cb00, cr00) = rgb_to_ycbcr(top_pair[0], top_pair[1], top_pair[2]);
                        let (y01, cb01, cr01) = rgb_to_ycbcr(top_pair[3], top_pair[4], top_pair[5]);
                        let (y10, cb10, cr10) =
                            rgb_to_ycbcr(bottom_pair[0], bottom_pair[1], bottom_pair[2]);
                        let (y11, cb11, cr11) =
                            rgb_to_ycbcr(bottom_pair[3], bottom_pair[4], bottom_pair[5]);
                        let pair_y_offset = y_offset + (pair_x / 4) * 64 + (pair_x % 4) * 2;
                        y_spare[pair_y_offset].write(y00);
                        y_spare[pair_y_offset + 1].write(y01);
                        y_spare[pair_y_offset + 8].write(y10);
                        y_spare[pair_y_offset + 9].write(y11);
                        cb_spare[chroma_offset + pair_x].write(
                            ((u16::from(cb00)
                                + u16::from(cb01)
                                + u16::from(cb10)
                                + u16::from(cb11))
                                / 4) as u8,
                        );
                        cr_spare[chroma_offset + pair_x].write(
                            ((u16::from(cr00)
                                + u16::from(cr01)
                                + u16::from(cr10)
                                + u16::from(cr11))
                                / 4) as u8,
                        );
                    }
                }
            }
        }
    }
    // SAFETY: every aligned MCU initializes four Y blocks and one block in
    // each chroma plane, covering each reserved prefix exactly once.
    unsafe {
        y_plane.set_len(y_len + y_required);
        cb_plane.set_len(cb_len + chroma_required);
        cr_plane.set_len(cr_len + chroma_required);
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
    clippy::inline_always,
    reason = "compact-plane address arithmetic is forced into measured producer and sampler loops"
)]
#[inline(always)]
fn compact_plane_sample_offset(width: usize, height: usize, x: usize, y: usize) -> usize {
    debug_assert!(x < width && y < height);
    let complete_height = height / 8 * 8;
    if y >= complete_height {
        return y * width + x;
    }
    let complete_width = width / 8 * 8;
    let band_offset = y / 8 * 8 * width;
    let row_in_band = y % 8;
    if x < complete_width {
        band_offset + x / 8 * 64 + row_in_band * 8 + x % 8
    } else {
        band_offset + complete_width * 8 + row_in_band * (width - complete_width) + x
            - complete_width
    }
}

#[expect(
    clippy::inline_always,
    reason = "block-plane address arithmetic is forced into measured producer and sampler loops"
)]
#[inline(always)]
fn block_plane_sample_offset(width: usize, height: usize, x: usize, y: usize) -> usize {
    debug_assert_eq!(width % 8, 0);
    debug_assert_eq!(height % 8, 0);
    debug_assert!(x < width && y < height);
    (y / 8 * (width / 8) + x / 8) * 64 + (y % 8) * 8 + x % 8
}

#[expect(
    clippy::inline_always,
    reason = "fixed block-plane lookup remains folded into the measured entropy loop"
)]
#[inline(always)]
pub(super) fn sample_block_from_block_planes<'a>(
    planes: &'a [Cow<'_, [u8]>],
    component: usize,
    block_index: u32,
) -> &'a [u8; 64] {
    let block_index = block_index as usize;
    planes[component][block_index * 64..]
        .first_chunk()
        .expect("aligned JPEG block is present")
}

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "compact block lookup mirrors and remains folded into the private entropy block geometry"
)]
#[inline(always)]
pub(super) fn sample_block_from_compact_planes<'a>(
    planes: &'a [Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    component: usize,
    mcu_index: u32,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u8,
    block_y: u8,
) -> Option<&'a [u8; 64]> {
    if component != 0 && sampling.max_h == 2 {
        return Some(
            planes[component][mcu_index as usize * 64..]
                .first_chunk()
                .expect("compact chroma block is present"),
        );
    }
    let block_column = mcu_x as usize * usize::from(sampling.h[component]) + usize::from(block_x);
    let block_row = mcu_y as usize * usize::from(sampling.v[component]) + usize::from(block_y);
    let origin_x = block_column * 8;
    let origin_y = block_row * 8;
    let width = width as usize;
    let height = height as usize;
    if origin_x + 8 > width || origin_y + 8 > height {
        return None;
    }
    let offset = origin_y * width + block_column * 64;
    Some(
        planes[component][offset..]
            .first_chunk()
            .expect("compact full-resolution block is present"),
    )
}

#[cfg(all(target_arch = "aarch64", not(feature = "scalar-only")))]
#[expect(
    clippy::inline_always,
    reason = "the fixed repeated-edge sampler remains folded into its measured thin-edge transform caller"
)]
#[inline(always)]
pub(super) fn sample_compact_repeated_edge_axis<const AXIS: u8>(
    plane: &[u8],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
) -> [u8; 8] {
    debug_assert!(AXIS == 1 || AXIS == 2);
    debug_assert!(width != 0 && height != 0);
    debug_assert!((AXIS == 1 && origin_y >= height - 1) || (AXIS == 2 && origin_x >= width - 1));
    let last_x = width - 1;
    let last_y = height - 1;
    let mut samples = [0u8; 8];
    for (index, sample) in samples.iter_mut().enumerate() {
        let (source_x, source_y) = match AXIS {
            1 => ((origin_x + index).min(last_x), origin_y.min(last_y)),
            2 => (origin_x.min(last_x), (origin_y + index).min(last_y)),
            _ => unreachable!("repeated compact edge axis is rows or columns"),
        };
        *sample = plane[compact_plane_sample_offset(width, height, source_x, source_y)];
    }
    samples
}

#[expect(
    clippy::too_many_arguments,
    reason = "compact edge sampling mirrors the private entropy block geometry"
)]
#[inline(never)]
pub(super) fn sample_block_from_compact_edge(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    component: usize,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u8,
    block_y: u8,
    out: &mut [u8; 64],
) {
    debug_assert!(component == 0 || sampling.max_h == 1);
    let block_column = mcu_x as usize * usize::from(sampling.h[component]) + usize::from(block_x);
    let block_row = mcu_y as usize * usize::from(sampling.v[component]) + usize::from(block_y);
    let origin_x = block_column * 8;
    let origin_y = block_row * 8;
    let width = width as usize;
    let height = height as usize;
    for y in 0..8 {
        let source_y = (origin_y + y).min(height - 1);
        for x in 0..8 {
            let source_x = (origin_x + x).min(width - 1);
            let source = compact_plane_sample_offset(width, height, source_x, source_y);
            out[y * 8 + x] = planes[component][source];
        }
    }
}

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "private JPEG sample hot path keeps scalar arguments and forced inlining for measured codegen"
)]
#[inline(always)]
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
    out: &mut [u8; 64],
) {
    if let Some(layout) = precomputed_chroma_layout(width, height, sampling, component) {
        debug_assert_eq!(planes[component].len(), layout.len().unwrap_or_default());
        let origin_x =
            (mcu_x as usize * usize::from(sampling.h[component]) + usize::from(block_x)) * 8;
        let origin_y =
            (mcu_y as usize * usize::from(sampling.v[component]) + usize::from(block_y)) * 8;
        for y in 0..8 {
            let start = (origin_y + y) * layout.width + origin_x;
            out[y * 8..y * 8 + 8].copy_from_slice(&planes[component][start..start + 8]);
        }
        return;
    }
    let (block_origin_x, block_origin_y, x_scale, y_scale) =
        sample_block_geometry(sampling, component, mcu_x, mcu_y, block_x, block_y);
    if block_origin_x + 8 * x_scale <= width && block_origin_y + 8 * y_scale <= height {
        #[cfg(target_arch = "aarch64")]
        if sampling.max_v == 1 && x_scale == 2 && y_scale == 1 && !cfg!(feature = "scalar-only") {
            if let Some(neon) = fearless_simd::Level::new().as_neon() {
                neon::sample_422_block(
                    neon,
                    &planes[component],
                    width as usize,
                    block_origin_x as usize,
                    block_origin_y as usize,
                    out,
                );
                return;
            }
        }
        sample_block_interior(
            &planes[component],
            width as usize,
            block_origin_x as usize,
            block_origin_y as usize,
            x_scale as usize,
            y_scale as usize,
            out,
        );
        return;
    }
    sample_block_edge(
        &planes[component],
        width as usize,
        height as usize,
        component,
        block_origin_x as usize,
        block_origin_y as usize,
        x_scale as usize,
        y_scale as usize,
        out,
    );
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "private thin-image sampler mirrors the complete block geometry and stays in its measured caller"
)]
#[inline(always)]
pub(super) fn sample_repeated_axis<const AXIS: u8>(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    component: usize,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u8,
    block_y: u8,
) -> [u8; 8] {
    let (origin_x, origin_y, x_scale, y_scale) =
        sample_block_geometry(sampling, component, mcu_x, mcu_y, block_x, block_y);
    match AXIS {
        1 => {
            debug_assert_eq!(height, 1);
            sample_single_row_axis(
                &planes[component],
                width as usize,
                component,
                origin_x as usize,
                x_scale as usize,
                y_scale as usize,
            )
        }
        2 => {
            debug_assert_eq!(width, 1);
            sample_single_column_axis(
                &planes[component],
                height as usize,
                origin_y as usize,
                y_scale as usize,
            )
        }
        _ => unreachable!("repeated JPEG sample axis is rows or columns"),
    }
}

#[expect(
    clippy::inline_always,
    reason = "block geometry remains folded into complete and repeated-axis samplers"
)]
#[inline(always)]
fn sample_block_geometry(
    sampling: JpegBaselineSampling,
    component: usize,
    mcu_x: u32,
    mcu_y: u32,
    block_x: u8,
    block_y: u8,
) -> (u32, u32, u32, u32) {
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let comp_h = u32::from(sampling.h[component]);
    let comp_v = u32::from(sampling.v[component]);
    let x_scale = max_h / comp_h;
    let y_scale = max_v / comp_v;
    let mcu_origin_x = mcu_x * max_h * 8;
    let mcu_origin_y = mcu_y * max_v * 8;
    let block_origin_x = mcu_origin_x + u32::from(block_x) * 8 * x_scale;
    let block_origin_y = mcu_origin_y + u32::from(block_y) * 8 * y_scale;
    (block_origin_x, block_origin_y, x_scale, y_scale)
}

#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the outlined edge path keeps the common interior sampler compact"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the average of edge-replicated u8 samples remains in the u8 range"
)]
fn sample_block_edge(
    plane: &[u8],
    width: usize,
    height: usize,
    component: usize,
    origin_x: usize,
    origin_y: usize,
    x_scale: usize,
    y_scale: usize,
    out: &mut [u8; 64],
) {
    if height == 1 {
        sample_block_single_row(plane, width, component, origin_x, x_scale, y_scale, out);
        return;
    }
    if width == 1 {
        sample_block_single_column(plane, height, origin_y, y_scale, out);
        return;
    }
    let sample_count = u32::try_from(x_scale * y_scale).expect("sampling scale fits u32");
    for y in 0..8usize {
        for x in 0..8usize {
            let value = if component == 0 {
                let sx = (origin_x + x).min(width - 1);
                let sy = (origin_y + y).min(height - 1);
                plane[sy * width + sx]
            } else {
                let mut sum = 0u32;
                for dy in 0..y_scale {
                    for dx in 0..x_scale {
                        let sx = (origin_x + x * x_scale + dx).min(width - 1);
                        let sy = (origin_y + y * y_scale + dy).min(height - 1);
                        sum += u32::from(plane[sy * width + sx]);
                    }
                }
                (sum / sample_count) as u8
            };
            out[y * 8 + x] = value;
        }
    }
}

#[expect(
    clippy::chunks_exact_to_as_chunks,
    reason = "the established fixed-row loop preserves measured replication codegen"
)]
fn sample_block_single_row(
    plane: &[u8],
    width: usize,
    component: usize,
    origin_x: usize,
    x_scale: usize,
    y_scale: usize,
    out: &mut [u8; 64],
) {
    let first_row = sample_single_row_axis(plane, width, component, origin_x, x_scale, y_scale);
    out[..8].copy_from_slice(&first_row);
    for row in out[8..].chunks_exact_mut(8) {
        row.copy_from_slice(&first_row);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "bounded u8 averages and fixed-axis sampling remain folded into the thin-image loop"
)]
#[inline(always)]
fn sample_single_row_axis(
    plane: &[u8],
    width: usize,
    component: usize,
    origin_x: usize,
    x_scale: usize,
    y_scale: usize,
) -> [u8; 8] {
    let mut out = [0u8; 8];
    if origin_x + 8 * x_scale <= width {
        sample_horizontal_row(plane, width, origin_x, 0, 0, x_scale, y_scale, &mut out);
        return out;
    }
    let sample_count = u32::try_from(x_scale * y_scale).expect("sampling scale fits u32");
    for (x, sample) in out.iter_mut().enumerate() {
        *sample = if component == 0 {
            plane[(origin_x + x).min(width - 1)]
        } else {
            let mut sum = 0u32;
            for _ in 0..y_scale {
                for dx in 0..x_scale {
                    let sx = (origin_x + x * x_scale + dx).min(width - 1);
                    sum += u32::from(plane[sx]);
                }
            }
            (sum / sample_count) as u8
        };
    }
    out
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    clippy::manual_midpoint,
    clippy::too_many_arguments,
    reason = "explicit widened floor averages and scalar geometry preserve the measured horizontal sampler"
)]
#[inline(always)]
fn sample_horizontal_row(
    plane: &[u8],
    width: usize,
    origin_x: usize,
    top_y: usize,
    bottom_y: usize,
    x_scale: usize,
    y_scale: usize,
    out: &mut [u8],
) {
    match (x_scale, y_scale) {
        (1, 1) => {
            let start = top_y * width + origin_x;
            out.copy_from_slice(&plane[start..start + 8]);
        }
        (2, 1) => {
            let start = top_y * width + origin_x;
            for (sample, pair) in out
                .iter_mut()
                .zip(plane[start..start + 16].as_chunks::<2>().0)
            {
                *sample = ((u16::from(pair[0]) + u16::from(pair[1])) / 2) as u8;
            }
        }
        (2, 2) => {
            let top = top_y * width + origin_x;
            let bottom = bottom_y * width + origin_x;
            let top = &plane[top..top + 16];
            let bottom = &plane[bottom..bottom + 16];
            for ((sample, top), bottom) in out
                .iter_mut()
                .zip(top.as_chunks::<2>().0)
                .zip(bottom.as_chunks::<2>().0)
            {
                *sample = ((u16::from(top[0])
                    + u16::from(top[1])
                    + u16::from(bottom[0])
                    + u16::from(bottom[1]))
                    / 4) as u8;
            }
        }
        _ => unreachable!("baseline JPEG sampling scales are 1 or 2"),
    }
}

#[expect(
    clippy::chunks_exact_to_as_chunks,
    reason = "the established fixed-column loop preserves measured row-fill codegen"
)]
fn sample_block_single_column(
    plane: &[u8],
    height: usize,
    origin_y: usize,
    y_scale: usize,
    out: &mut [u8; 64],
) {
    let column = sample_single_column_axis(plane, height, origin_y, y_scale);
    for (row, value) in out.chunks_exact_mut(8).zip(column) {
        row.fill(value);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    clippy::manual_midpoint,
    reason = "the widened two-sample floor average fits u8 and remains folded into the thin-image loop"
)]
#[inline(always)]
fn sample_single_column_axis(
    plane: &[u8],
    height: usize,
    origin_y: usize,
    y_scale: usize,
) -> [u8; 8] {
    let mut out = [0u8; 8];
    let last_y = height - 1;
    for (y, sample) in out.iter_mut().enumerate() {
        let top_y = (origin_y + y * y_scale).min(last_y);
        let bottom_y = (top_y + usize::from(y_scale == 2)).min(last_y);
        *sample = match y_scale {
            1 => plane[top_y],
            2 => ((u16::from(plane[top_y]) + u16::from(plane[bottom_y])) / 2) as u8,
            _ => unreachable!("baseline JPEG sampling scales are 1 or 2"),
        };
    }
    out
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    clippy::manual_midpoint,
    reason = "explicit widened floor averages and forced inlining preserve the measured interior sampler"
)]
#[inline(always)]
fn sample_block_interior(
    plane: &[u8],
    width: usize,
    origin_x: usize,
    origin_y: usize,
    x_scale: usize,
    y_scale: usize,
    out: &mut [u8; 64],
) {
    match (x_scale, y_scale) {
        (1, 1) => {
            for y in 0..8 {
                let start = (origin_y + y) * width + origin_x;
                out[y * 8..y * 8 + 8].copy_from_slice(&plane[start..start + 8]);
            }
        }
        (2, 1) => {
            for y in 0..8 {
                let start = (origin_y + y) * width + origin_x;
                let source = &plane[start..start + 16];
                for (sample, pair) in out[y * 8..y * 8 + 8]
                    .iter_mut()
                    .zip(source.as_chunks::<2>().0)
                {
                    *sample = ((u16::from(pair[0]) + u16::from(pair[1])) / 2) as u8;
                }
            }
        }
        (2, 2) => {
            for y in 0..8 {
                let top = (origin_y + y * 2) * width + origin_x;
                let bottom = top + width;
                let top = &plane[top..top + 16];
                let bottom = &plane[bottom..bottom + 16];
                for ((sample, top), bottom) in out[y * 8..y * 8 + 8]
                    .iter_mut()
                    .zip(top.as_chunks::<2>().0)
                    .zip(bottom.as_chunks::<2>().0)
                {
                    *sample = ((u16::from(top[0])
                        + u16::from(top[1])
                        + u16::from(bottom[0])
                        + u16::from(bottom[1]))
                        / 4) as u8;
                }
            }
        }
        _ => unreachable!("baseline JPEG sampling scales are 1 or 2"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::baseline_encode_tables;
    use crate::encoder::{JpegBackend, JpegEncodeOptions};
    use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one edge matrix compares all dimensions, sampling modes, and block positions"
    )]
    fn axis_separated_edge_sampling_matches_per_sample_reference() {
        let dimensions: [(u32, u32); 47] = [
            (1, 1),
            (7, 1),
            (8, 1),
            (9, 1),
            (15, 1),
            (16, 1),
            (17, 1),
            (1_024, 1),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 1_024),
            (7, 9),
            (8, 8),
            (9, 7),
            (15, 17),
            (16, 16),
            (17, 15),
            (18, 1),
            (1, 18),
            (18, 17),
            (17, 18),
            (17, 32),
            (17, 33),
            (18, 18),
            // These missing residues, together with the existing 17, 25,
            // 29, 31, and 34 widths, exercise every nonzero sixteen-pixel
            // tail in both subsampled modes. Heights cover complete and
            // incomplete eight-row bands with odd and even edges.
            (19, 16),
            (20, 16),
            (21, 16),
            (22, 16),
            (23, 17),
            (24, 18),
            (25, 32),
            (26, 16),
            (27, 17),
            (28, 18),
            (29, 32),
            (30, 17),
            (31, 18),
            (32, 16),
            (32, 18),
            (32, 32),
            (33, 18),
            (34, 17),
            (34, 18),
        ];
        for subsampling in [
            JpegSubsampling::Ybr444,
            JpegSubsampling::Ybr422,
            JpegSubsampling::Ybr420,
        ] {
            let sampling = baseline_encode_tables(JpegEncodeOptions {
                quality: 90,
                subsampling,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            })
            .unwrap()
            .sampling;
            for (width, height) in dimensions {
                let pixels = width as usize * height as usize;
                let data = (0..pixels * 3)
                    .map(|index| index.to_le_bytes()[0].wrapping_mul(73).wrapping_add(19))
                    .collect::<Vec<_>>();
                let planes = component_planes(
                    JpegSamples::Rgb8 {
                        data: &data,
                        width,
                        height,
                    },
                    subsampling,
                    sampling,
                    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                )
                .expect("constructed component planes");
                assert_eq!(planes[0].len(), pixels);
                let expected_chroma_len = precomputed_chroma_layout(width, height, sampling, 1)
                    .and_then(PrecomputedChromaLayout::len)
                    .unwrap_or(pixels);
                assert_eq!(planes[1].len(), expected_chroma_len);
                assert_eq!(planes[2].len(), expected_chroma_len);
                assert!(
                    planes.iter().all(|plane| plane.len() <= pixels),
                    "physical plane exceeds legacy length for {subsampling:?} {width}x{height}"
                );
                if width == 1 || height == 1 {
                    assert!(
                        planes.iter().all(|plane| plane.len() == pixels),
                        "thin image must retain legacy planes for {subsampling:?} {width}x{height}"
                    );
                }
                let mut reference_y = Vec::with_capacity(pixels);
                let mut reference_cb = Vec::with_capacity(pixels);
                let mut reference_cr = Vec::with_capacity(pixels);
                fill_rgb_to_ycbcr_scalar(
                    &data,
                    &mut reference_y,
                    &mut reference_cb,
                    &mut reference_cr,
                );
                let reference_planes = vec![
                    Cow::Owned(reference_y),
                    Cow::Owned(reference_cb),
                    Cow::Owned(reference_cr),
                ];
                let mcu_columns = width.div_ceil(u32::from(sampling.max_h) * 8);
                let mcu_rows = height.div_ceil(u32::from(sampling.max_v) * 8);
                let block_layout_444 = uses_block_plane_layout(width, height, sampling);
                let block_layout_422 = uses_422_block_plane_layout(width, height, sampling);
                let block_layout_420 = uses_420_block_plane_layout(width, height, sampling);
                let compact_block_layout = uses_compact_block_plane_layout(width, height, sampling);
                for mcu_y in 0..mcu_rows {
                    for mcu_x in 0..mcu_columns {
                        for component in 0..usize::from(sampling.components) {
                            for block_y in 0..sampling.v[component] {
                                for block_x in 0..sampling.h[component] {
                                    let mut scratch = [0u8; 64];
                                    let actual = if block_layout_444
                                        || block_layout_422
                                        || block_layout_420
                                    {
                                        let mcu_index = mcu_y * mcu_columns + mcu_x;
                                        let block_index =
                                            match (component, block_layout_422, block_layout_420) {
                                                (0, true, _) => mcu_index * 2 + u32::from(block_x),
                                                (0, _, true) => {
                                                    mcu_index * 4
                                                        + u32::from(block_y) * 2
                                                        + u32::from(block_x)
                                                }
                                                _ => mcu_index,
                                            };
                                        sample_block_from_block_planes(
                                            &planes,
                                            component,
                                            block_index,
                                        )
                                    } else if compact_block_layout {
                                        let mcu_index = mcu_y * mcu_columns + mcu_x;
                                        if let Some(block) = sample_block_from_compact_planes(
                                            &planes, width, height, sampling, component, mcu_index,
                                            mcu_x, mcu_y, block_x, block_y,
                                        ) {
                                            block
                                        } else {
                                            sample_block_from_compact_edge(
                                                &planes,
                                                width,
                                                height,
                                                sampling,
                                                component,
                                                mcu_x,
                                                mcu_y,
                                                block_x,
                                                block_y,
                                                &mut scratch,
                                            );
                                            &scratch
                                        }
                                    } else {
                                        sample_block(
                                            &planes,
                                            width,
                                            height,
                                            sampling,
                                            component,
                                            mcu_x,
                                            mcu_y,
                                            block_x,
                                            block_y,
                                            &mut scratch,
                                        );
                                        &scratch
                                    };
                                    let expected = sample_block_reference(
                                        &reference_planes,
                                        width,
                                        height,
                                        sampling,
                                        component,
                                        mcu_x,
                                        mcu_y,
                                        block_x,
                                        block_y,
                                    );
                                    assert_eq!(
                                        *actual, expected,
                                        "{subsampling:?} {width}x{height} mcu=({mcu_x},{mcu_y}) component={component} block=({block_x},{block_y})"
                                    );
                                    #[cfg(all(
                                        target_arch = "aarch64",
                                        not(feature = "scalar-only")
                                    ))]
                                    if compact_block_layout
                                        && (component == 0 || sampling.max_h == 1)
                                    {
                                        let origin_x = (mcu_x as usize
                                            * usize::from(sampling.h[component])
                                            + usize::from(block_x))
                                            * 8;
                                        let origin_y = (mcu_y as usize
                                            * usize::from(sampling.v[component])
                                            + usize::from(block_y))
                                            * 8;
                                        if origin_y >= height as usize - 1 {
                                            let active = sample_compact_repeated_edge_axis::<1>(
                                                planes[component].as_ref(),
                                                width as usize,
                                                height as usize,
                                                origin_x,
                                                origin_y,
                                            );
                                            let expanded: [u8; 64] =
                                                core::array::from_fn(|index| active[index % 8]);
                                            assert_eq!(
                                                expanded, expected,
                                                "compact repeated rows {subsampling:?} {width}x{height} mcu=({mcu_x},{mcu_y}) component={component} block=({block_x},{block_y})"
                                            );
                                        }
                                        if origin_x >= width as usize - 1 {
                                            let active = sample_compact_repeated_edge_axis::<2>(
                                                planes[component].as_ref(),
                                                width as usize,
                                                height as usize,
                                                origin_x,
                                                origin_y,
                                            );
                                            let expanded: [u8; 64] =
                                                core::array::from_fn(|index| active[index / 8]);
                                            assert_eq!(
                                                expanded, expected,
                                                "compact repeated columns {subsampling:?} {width}x{height} mcu=({mcu_x},{mcu_y}) component={component} block=({block_x},{block_y})"
                                            );
                                        }
                                    }
                                    #[cfg(target_arch = "aarch64")]
                                    if height == 1 {
                                        let active = sample_repeated_axis::<1>(
                                            &planes, width, height, sampling, component, mcu_x,
                                            mcu_y, block_x, block_y,
                                        );
                                        let expanded: [u8; 64] =
                                            core::array::from_fn(|index| active[index % 8]);
                                        assert_eq!(
                                            expanded, *actual,
                                            "active rows {subsampling:?} {width}x{height} mcu=({mcu_x},{mcu_y}) component={component} block=({block_x},{block_y})"
                                        );
                                    }
                                    #[cfg(target_arch = "aarch64")]
                                    if width == 1 {
                                        let active = sample_repeated_axis::<2>(
                                            &planes, width, height, sampling, component, mcu_x,
                                            mcu_y, block_x, block_y,
                                        );
                                        let expanded: [u8; 64] =
                                            core::array::from_fn(|index| active[index / 8]);
                                        assert_eq!(
                                            expanded, *actual,
                                            "active columns {subsampling:?} {width}x{height} mcu=({mcu_x},{mcu_y}) component={component} block=({block_x},{block_y})"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the test oracle mirrors the complete block-sampling coordinate contract"
    )]
    fn sample_block_reference(
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
        let x_scale = max_h / u32::from(sampling.h[component]);
        let y_scale = max_v / u32::from(sampling.v[component]);
        let mcu_origin_x = mcu_x * max_h * 8;
        let mcu_origin_y = mcu_y * max_v * 8;
        for y in 0..8u32 {
            for x in 0..8u32 {
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
                out[(y * 8 + x) as usize] = u8::try_from(sum / (x_scale * y_scale))
                    .expect("average of eight-bit samples fits u8");
            }
        }
        out
    }
}
