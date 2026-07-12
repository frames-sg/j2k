// SPDX-License-Identifier: MIT OR Apache-2.0

use super::extended12::lossless_color_sampling;
use super::{
    DownscaleFactor, JpegError, LosslessColorSampling, OutputFormat, PreparedComponentPlan, Rect,
    SofKind, DEFAULT_MAX_DECODE_BYTES,
};
use crate::entropy::sequential::StripeLayout;
use crate::info::{ColorSpace, Info, SamplingFactors};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LosslessSampledPlaneLayout {
    pub(super) luma_len: usize,
    pub(super) chroma_len: usize,
    pub(super) chroma_dimensions: (usize, usize),
    pub(super) total_bytes: usize,
}

pub(super) fn compute_decode_scratch_bytes(
    (width, height): (u32, u32),
    sampling: SamplingFactors,
    cap: usize,
) -> Result<usize, JpegError> {
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let mcu_width = 8u32
        .checked_mul(max_h)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    let mcu_height = 8u32
        .checked_mul(max_v)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    let mcus_per_row = width.div_ceil(mcu_width);
    let _mcu_rows = height.div_ceil(mcu_height);

    let stripe_bytes = StripeLayout::for_sampling(sampling, mcus_per_row, 8)?.allocation_bytes()?;
    let stripe_buffers = checked_usize_product(&[stripe_bytes, 3], cap)?;
    let prev_dc = checked_usize_product(&[sampling.len(), core::mem::size_of::<i32>()], cap)?;
    // The reusable pool owns ten full-width upsample/component rows. Row-sink
    // paths temporarily detach two RGB rows (six more bytes per output pixel)
    // while the stripe and upsample storage remains live.
    let row_scratch = checked_usize_product(&[width as usize, 16], cap)?;
    let total = stripe_buffers
        .checked_add(prev_dc)
        .and_then(|bytes| bytes.checked_add(row_scratch))
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if total > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total,
            cap,
        });
    }

    Ok(total)
}

/// Checked size for a transient full-frame intermediate buffer, enforcing the
/// decode memory cap at the allocation site.
pub(super) fn checked_scratch_len(factors: &[usize]) -> Result<usize, JpegError> {
    let cap = DEFAULT_MAX_DECODE_BYTES;
    let len = checked_usize_product(factors, cap)?;
    if len > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: len,
            cap,
        });
    }
    Ok(len)
}

pub(super) fn additional_decode_scratch_bytes(
    sof_kind: SofKind,
    dimensions: (u32, u32),
    fmt: OutputFormat,
    roi: Rect,
    output_rect: Rect,
    downscale: DownscaleFactor,
) -> Result<usize, JpegError> {
    let rgb_intermediate = match fmt {
        OutputFormat::Rgba8 { .. } | OutputFormat::Rgba8Scaled { .. } => Some(3usize),
        OutputFormat::Rgba16 { .. } | OutputFormat::Rgba16Scaled { .. } => Some(6usize),
        _ => None,
    };
    let mut additional = if let Some(rgb_bytes_per_pixel) = rgb_intermediate {
        checked_scratch_len(&[
            output_rect.w as usize,
            output_rect.h as usize,
            rgb_bytes_per_pixel,
        ])?
    } else {
        0
    };

    if sof_kind != SofKind::Lossless
        || (roi == Rect::full(dimensions) && downscale == DownscaleFactor::Full)
    {
        return Ok(additional);
    }

    let full_frame_bytes_per_pixel = match fmt {
        OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => 1,
        OutputFormat::Gray16 | OutputFormat::Gray16Scaled { .. } => 2,
        OutputFormat::Rgb8
        | OutputFormat::Rgb8Scaled { .. }
        | OutputFormat::Rgba8 { .. }
        | OutputFormat::Rgba8Scaled { .. } => 3,
        OutputFormat::Rgb16
        | OutputFormat::Rgb16Scaled { .. }
        | OutputFormat::Rgba16 { .. }
        | OutputFormat::Rgba16Scaled { .. } => 6,
    };
    let full_frame = checked_scratch_len(&[
        dimensions.0 as usize,
        dimensions.1 as usize,
        full_frame_bytes_per_pixel,
    ])?;
    additional = additional
        .checked_add(full_frame)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_DECODE_BYTES,
        })?;
    if additional > DEFAULT_MAX_DECODE_BYTES {
        return Err(JpegError::MemoryCapExceeded {
            requested: additional,
            cap: DEFAULT_MAX_DECODE_BYTES,
        });
    }
    Ok(additional)
}

pub(super) fn compute_lossless_scratch_bytes(info: &Info, cap: usize) -> Result<usize, JpegError> {
    let sampled_planes =
        lossless_sampled_plane_layout(info, cap)?.map_or(0, |layout| layout.total_bytes);
    let row_scratch = compute_lossless_row_scratch_bytes(info, cap)?;
    Ok(sampled_planes.max(row_scratch))
}

pub(super) fn lossless_sampled_plane_layout(
    info: &Info,
    cap: usize,
) -> Result<Option<LosslessSampledPlaneLayout>, JpegError> {
    if !matches!(
        lossless_color_sampling(info),
        Some(LosslessColorSampling::S422 | LosslessColorSampling::S420)
    ) {
        return Ok(None);
    }
    let width = info.dimensions.0 as usize;
    let height = info.dimensions.1 as usize;
    let bytes_per_sample: usize = if info.bit_depth > 8 { 2 } else { 1 };
    let chroma_width = width.div_ceil(usize::from(info.sampling.max_h));
    let chroma_height = height.div_ceil(usize::from(info.sampling.max_v));
    let luma_len = checked_usize_product(&[width, height], cap)?;
    let chroma_len = checked_usize_product(&[chroma_width, chroma_height], cap)?;
    let luma_bytes = checked_usize_product(&[luma_len, bytes_per_sample], cap)?;
    let chroma_bytes = checked_usize_product(&[chroma_len, bytes_per_sample, 2], cap)?;
    let total_bytes = luma_bytes
        .checked_add(chroma_bytes)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if total_bytes > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total_bytes,
            cap,
        });
    }
    Ok(Some(LosslessSampledPlaneLayout {
        luma_len,
        chroma_len,
        chroma_dimensions: (chroma_width, chroma_height),
        total_bytes,
    }))
}

fn compute_lossless_row_scratch_bytes(info: &Info, cap: usize) -> Result<usize, JpegError> {
    let width = info.dimensions.0 as usize;
    let bytes_per_sample = if info.bit_depth > 8 { 2 } else { 1 };
    let channels = if info.color_space == ColorSpace::Grayscale {
        1
    } else {
        3
    };
    let predictor_row = checked_usize_product(&[width, channels, bytes_per_sample], cap)?;
    let predictor_rows = checked_usize_product(&[predictor_row, 2], cap)?;
    let conversion_rows = match (info.color_space, info.bit_depth) {
        (ColorSpace::Grayscale, 8) => checked_usize_product(&[width, 3, 2], cap)?,
        (ColorSpace::YCbCr, _) => checked_usize_product(&[predictor_row, 2], cap)?,
        _ => 0,
    };
    let total =
        predictor_rows
            .checked_add(conversion_rows)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    if total > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total,
            cap,
        });
    }
    Ok(total)
}

pub(super) fn compute_extended12_planes_scratch_bytes(
    components: &[PreparedComponentPlan],
    (width, height): (u32, u32),
    sampling: SamplingFactors,
    cap: usize,
) -> Result<usize, JpegError> {
    let mcu_cols = width.div_ceil(u32::from(sampling.max_h) * 8) as usize;
    let mcu_rows = height.div_ceil(u32::from(sampling.max_v) * 8) as usize;
    let mut total = 0usize;
    for component in components {
        let stride = checked_usize_product(&[mcu_cols, usize::from(component.h), 8], cap)?;
        let rows = checked_usize_product(&[mcu_rows, usize::from(component.v), 8], cap)?;
        let plane = checked_usize_product(&[stride, rows, core::mem::size_of::<u16>()], cap)?;
        total = total
            .checked_add(plane)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    }
    if total > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total,
            cap,
        });
    }
    Ok(total)
}

pub(super) fn checked_usize_product(factors: &[usize], cap: usize) -> Result<usize, JpegError> {
    let mut value = 1usize;
    for factor in factors {
        value = value
            .checked_mul(*factor)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_pool_formula_counts_outer_metadata_rows_and_payloads() {
        let sampling = SamplingFactors::from_components(&[(1, 1)]).unwrap();
        let stripe =
            core::mem::size_of::<alloc::vec::Vec<u8>>() + 2 * core::mem::size_of::<usize>() + 64;
        let expected = 3 * stripe + core::mem::size_of::<i32>() + 8 * 16;
        assert_eq!(
            compute_decode_scratch_bytes((8, 8), sampling, expected).unwrap(),
            expected
        );
        assert!(matches!(
            compute_decode_scratch_bytes((8, 8), sampling, expected - 1),
            Err(JpegError::MemoryCapExceeded { requested, cap })
                if requested == expected && cap == expected - 1
        ));
    }

    #[test]
    fn nested_lossless_rgba_intermediates_are_aggregated() {
        let roi = Rect {
            x: 1,
            y: 1,
            w: 2,
            h: 3,
        };
        assert_eq!(
            additional_decode_scratch_bytes(
                SofKind::Lossless,
                (10, 10),
                OutputFormat::Rgba16 { alpha: u16::MAX },
                roi,
                Rect::full((2, 3)),
                DownscaleFactor::Full,
            )
            .unwrap(),
            36 + 600
        );
        assert_eq!(
            additional_decode_scratch_bytes(
                SofKind::Extended12,
                (10, 10),
                OutputFormat::Rgba16 { alpha: u16::MAX },
                Rect::full((10, 10)),
                Rect::full((10, 10)),
                DownscaleFactor::Full,
            )
            .unwrap(),
            600
        );
    }
}
