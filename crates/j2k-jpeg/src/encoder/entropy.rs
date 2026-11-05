// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded CPU entropy orchestration and restart-chunk scheduling.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use crate::adapter::{
    checked_encode_host_live_bytes, JpegBaselineHuffmanTable, JpegBaselineSampling,
};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;

#[cfg(all(target_arch = "aarch64", not(feature = "scalar-only")))]
use super::sample_planes::sample_compact_repeated_edge_axis;
#[cfg(target_arch = "aarch64")]
use super::sample_planes::sample_repeated_axis;
use super::sample_planes::{
    sample_block, sample_block_from_block_planes, sample_block_from_compact_edge,
    sample_block_from_compact_planes, uses_420_block_plane_layout, uses_422_block_plane_layout,
    uses_block_plane_layout, uses_compact_block_plane_layout,
};
#[cfg(all(target_arch = "aarch64", not(feature = "scalar-only")))]
use super::transform::fdct_quantize_repeated_axis_into;
use super::transform::{fdct_quantize, QuantizationDivisors};
#[cfg(target_arch = "aarch64")]
use super::transform::{fdct_quantize_repeated_axis, prepare_entropy_ac, QuantizedBlock};
use super::JpegEncodeError;
#[cfg(target_arch = "aarch64")]
use crate::baseline_entropy::encode_token_block_zigzag;
use crate::baseline_entropy::{encode_block_zigzag, packed_standard_huffman_tables, BitWriter};

mod restart;
mod workspace;
use self::restart::encode_entropy_restart_chunk;
pub(super) use self::restart::encode_entropy_restart_segments;
#[cfg(test)]
pub(super) use self::restart::{parallel_entropy_chunk_count, MAX_PARALLEL_ENTROPY_CHUNKS};
pub(super) use self::workspace::entropy_host_workspace_bytes;

#[expect(
    clippy::too_many_arguments,
    reason = "private JPEG entropy hot path keeps scalar arguments for optimized codegen"
)]
pub(super) fn encode_entropy(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    restart_interval: Option<u16>,
    entropy_capacity: usize,
    external_live_bytes: usize,
) -> Result<Vec<u8>, JpegEncodeError> {
    if let Some(restart_interval) = restart_interval {
        return encode_entropy_restart_segments(
            planes,
            width,
            height,
            sampling,
            q_luma,
            q_chroma,
            dc_tables,
            ac_tables,
            cosine,
            restart_interval,
            entropy_capacity,
            external_live_bytes,
        );
    }
    encode_entropy_serial(
        planes,
        width,
        height,
        sampling,
        q_luma,
        q_chroma,
        dc_tables,
        ac_tables,
        cosine,
        None,
        entropy_capacity,
        external_live_bytes,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "private JPEG entropy hot path keeps scalar arguments for optimized codegen"
)]
pub(super) fn encode_entropy_serial(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    restart_interval: Option<u16>,
    entropy_capacity: usize,
    external_live_bytes: usize,
) -> Result<Vec<u8>, JpegEncodeError> {
    checked_jpeg_baseline_frame_capacity(entropy_capacity)?;
    let (mcus_per_row, total_mcus) = entropy_mcu_layout(width, height, sampling)?;
    if total_mcus == 0 {
        return Ok(Vec::new());
    }
    if let Some(restart_interval) = restart_interval {
        if restart_interval == 0 {
            return Err(JpegEncodeError::InvalidRestartInterval);
        }
        let restart_interval = u32::from(restart_interval);
        let segment_count = total_mcus.div_ceil(restart_interval);
        return encode_entropy_restart_chunk(
            planes,
            width,
            height,
            sampling,
            q_luma,
            q_chroma,
            dc_tables,
            ac_tables,
            cosine,
            mcus_per_row,
            total_mcus,
            restart_interval,
            0,
            segment_count,
            entropy_capacity,
            external_live_bytes,
        );
    }

    checked_encode_host_live_bytes([external_live_bytes, entropy_capacity])?;
    let mut writer = BitWriter::try_with_max_bytes(entropy_capacity)?;
    checked_encode_host_live_bytes([external_live_bytes, writer.capacity_bytes()])?;
    #[cfg(target_arch = "aarch64")]
    if !cfg!(feature = "scalar-only") && height == 1 && width > 1 {
        encode_entropy_thin_mcu_range_into::<1>(
            planes,
            width,
            height,
            sampling,
            q_luma,
            q_chroma,
            dc_tables,
            ac_tables,
            cosine,
            mcus_per_row,
            0,
            total_mcus,
            &mut writer,
        )?;
    } else if !cfg!(feature = "scalar-only") && width == 1 && height > 1 {
        encode_entropy_thin_mcu_range_into::<2>(
            planes,
            width,
            height,
            sampling,
            q_luma,
            q_chroma,
            dc_tables,
            ac_tables,
            cosine,
            mcus_per_row,
            0,
            total_mcus,
            &mut writer,
        )?;
    } else {
        encode_entropy_mcu_range_into(
            planes,
            width,
            height,
            sampling,
            q_luma,
            q_chroma,
            dc_tables,
            ac_tables,
            cosine,
            mcus_per_row,
            0,
            total_mcus,
            &mut writer,
        )?;
    }
    #[cfg(not(target_arch = "aarch64"))]
    encode_entropy_mcu_range_into(
        planes,
        width,
        height,
        sampling,
        q_luma,
        q_chroma,
        dc_tables,
        ac_tables,
        cosine,
        mcus_per_row,
        0,
        total_mcus,
        &mut writer,
    )?;
    writer.into_bytes()
}

const ENCODER_GRAY_SAMPLING: JpegBaselineSampling = JpegBaselineSampling {
    components: 1,
    h: [1, 0, 0],
    v: [1, 0, 0],
    max_h: 1,
    max_v: 1,
};
const ENCODER_444_SAMPLING: JpegBaselineSampling = JpegBaselineSampling {
    components: 3,
    h: [1, 1, 1],
    v: [1, 1, 1],
    max_h: 1,
    max_v: 1,
};
const ENCODER_422_SAMPLING: JpegBaselineSampling = JpegBaselineSampling {
    components: 3,
    h: [2, 1, 1],
    v: [1, 1, 1],
    max_h: 2,
    max_v: 1,
};
const ENCODER_420_SAMPLING: JpegBaselineSampling = JpegBaselineSampling {
    components: 3,
    h: [2, 1, 1],
    v: [2, 1, 1],
    max_h: 2,
    max_v: 2,
};

const SAMPLING_MODE_GRAY: u8 = 0;
const SAMPLING_MODE_444: u8 = 1;
const SAMPLING_MODE_422: u8 = 2;
const SAMPLING_MODE_420: u8 = 3;
const SAMPLING_MODE_DYNAMIC: u8 = 4;

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "private JPEG entropy dispatch keeps the measured sampling and layout specializations together"
)]
fn encode_entropy_mcu_range_into(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    mcus_per_row: u32,
    start_mcu: u32,
    end_mcu: u32,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    macro_rules! encode_mode {
        ($mode:expr) => {
            encode_mode!($mode, false, false, false)
        };
        ($mode:expr, $tokens:expr) => {
            encode_mode!($mode, $tokens, false, false)
        };
        ($mode:expr, $tokens:expr, $block_planes:expr) => {
            encode_mode!($mode, $tokens, $block_planes, false)
        };
        ($mode:expr, $tokens:expr, $block_planes:expr, $compact_block_planes:expr) => {
            encode_entropy_mcu_range_into_mode::<
                $mode,
                $tokens,
                $block_planes,
                $compact_block_planes,
            >(
                planes,
                width,
                height,
                sampling,
                q_luma,
                q_chroma,
                dc_tables,
                ac_tables,
                cosine,
                mcus_per_row,
                start_mcu,
                end_mcu,
                writer,
            )
        };
    }

    // Standard qualities 49 through 100 use no coarser DC quantizers
    // than the quality-50 luma/chroma values. Token preparation still
    // requires at least 24 nonzero AC coefficients in each block.
    match sampling {
        ENCODER_GRAY_SAMPLING => {
            #[cfg(target_arch = "aarch64")]
            if !cfg!(feature = "scalar-only") && q_luma[0] <= 16 {
                return encode_mode!(SAMPLING_MODE_GRAY, true);
            }
            encode_mode!(SAMPLING_MODE_GRAY)
        }
        ENCODER_444_SAMPLING => {
            let block_planes = uses_block_plane_layout(width, height, sampling);
            let compact_block_planes = uses_compact_block_plane_layout(width, height, sampling);
            #[cfg(target_arch = "aarch64")]
            if !cfg!(feature = "scalar-only") && q_luma[0] <= 16 && q_chroma[0] <= 17 {
                if block_planes {
                    return encode_mode!(SAMPLING_MODE_444, true, true);
                }
                if compact_block_planes {
                    return encode_mode!(SAMPLING_MODE_444, true, true, true);
                }
                return encode_mode!(SAMPLING_MODE_444, true);
            }
            if block_planes {
                return encode_mode!(SAMPLING_MODE_444, false, true);
            }
            if compact_block_planes {
                return encode_mode!(SAMPLING_MODE_444, false, true, true);
            }
            encode_mode!(SAMPLING_MODE_444)
        }
        ENCODER_422_SAMPLING => {
            let block_planes = uses_422_block_plane_layout(width, height, sampling);
            let compact_block_planes = uses_compact_block_plane_layout(width, height, sampling);
            #[cfg(target_arch = "aarch64")]
            if !cfg!(feature = "scalar-only") && q_luma[0] <= 16 && q_chroma[0] <= 17 {
                if block_planes {
                    return encode_mode!(SAMPLING_MODE_422, true, true);
                }
                if compact_block_planes {
                    return encode_mode!(SAMPLING_MODE_422, true, true, true);
                }
                return encode_mode!(SAMPLING_MODE_422, true);
            }
            if block_planes {
                return encode_mode!(SAMPLING_MODE_422, false, true);
            }
            if compact_block_planes {
                return encode_mode!(SAMPLING_MODE_422, false, true, true);
            }
            encode_mode!(SAMPLING_MODE_422)
        }
        ENCODER_420_SAMPLING => {
            let block_planes = uses_420_block_plane_layout(width, height, sampling);
            let compact_block_planes = uses_compact_block_plane_layout(width, height, sampling);
            #[cfg(target_arch = "aarch64")]
            if !cfg!(feature = "scalar-only") && q_luma[0] <= 16 && q_chroma[0] <= 17 {
                if block_planes {
                    return encode_mode!(SAMPLING_MODE_420, true, true);
                }
                if compact_block_planes {
                    return encode_mode!(SAMPLING_MODE_420, true, true, true);
                }
                return encode_mode!(SAMPLING_MODE_420, true);
            }
            if block_planes {
                return encode_mode!(SAMPLING_MODE_420, false, true);
            }
            if compact_block_planes {
                return encode_mode!(SAMPLING_MODE_420, false, true, true);
            }
            encode_mode!(SAMPLING_MODE_420)
        }
        _ => encode_mode!(SAMPLING_MODE_DYNAMIC),
    }
}

#[inline(never)]
#[expect(
    clippy::match_same_arms,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "outlined sampling modes preserve the measured per-MCU layout and mode-specific index structure"
)]
fn encode_entropy_mcu_range_into_mode<
    const MODE: u8,
    const AC_TOKENS: bool,
    const BLOCK_PLANES: bool,
    const COMPACT_BLOCK_PLANES: bool,
>(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    mcus_per_row: u32,
    start_mcu: u32,
    end_mcu: u32,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    let sampling = match MODE {
        SAMPLING_MODE_GRAY => ENCODER_GRAY_SAMPLING,
        SAMPLING_MODE_444 => ENCODER_444_SAMPLING,
        SAMPLING_MODE_422 => ENCODER_422_SAMPLING,
        SAMPLING_MODE_420 => ENCODER_420_SAMPLING,
        SAMPLING_MODE_DYNAMIC => sampling,
        _ => unreachable!("encoder sampling mode"),
    };
    let luma_divisors = QuantizationDivisors::new(q_luma);
    let chroma_divisors = QuantizationDivisors::new(q_chroma);
    let packed_huffman = packed_standard_huffman_tables(dc_tables, ac_tables);
    let mut prev_dc = [0i32; 3];
    for mcu_index in start_mcu..end_mcu {
        let mcu_y = mcu_index / mcus_per_row;
        let mcu_x = mcu_index % mcus_per_row;
        for_each_mode_block::<MODE, BLOCK_PLANES, _>(sampling, |component, block_x, block_y| {
            let quant = if component == 0 { q_luma } else { q_chroma };
            let table_index = usize::from(component != 0);
            let dc_table = packed_huffman.dc[table_index];
            let divisors = if component == 0 {
                &luma_divisors
            } else {
                &chroma_divisors
            };
            let ac_table = packed_huffman.ac[table_index];
            let mut block_storage;
            let block = if BLOCK_PLANES && !COMPACT_BLOCK_PLANES {
                debug_assert!(matches!(
                    MODE,
                    SAMPLING_MODE_444 | SAMPLING_MODE_422 | SAMPLING_MODE_420
                ));
                let block_index = match MODE {
                    SAMPLING_MODE_444 => mcu_index,
                    SAMPLING_MODE_422 if component == 0 => mcu_index * 2 + u32::from(block_x),
                    SAMPLING_MODE_420 if component == 0 => {
                        mcu_index * 4 + u32::from(block_y) * 2 + u32::from(block_x)
                    }
                    SAMPLING_MODE_422 | SAMPLING_MODE_420 => mcu_index,
                    _ => unreachable!("block planes use canonical color sampling"),
                };
                sample_block_from_block_planes(planes, component, block_index)
            } else if COMPACT_BLOCK_PLANES {
                debug_assert!(matches!(
                    MODE,
                    SAMPLING_MODE_444 | SAMPLING_MODE_422 | SAMPLING_MODE_420
                ));
                if let Some(block) = sample_block_from_compact_planes(
                    planes, width, height, sampling, component, mcu_index, mcu_x, mcu_y, block_x,
                    block_y,
                ) {
                    block
                } else {
                    #[cfg(all(target_arch = "aarch64", not(feature = "scalar-only")))]
                    {
                        debug_assert!(component == 0 || sampling.max_h == 1);
                        let block_column = mcu_x as usize * usize::from(sampling.h[component])
                            + usize::from(block_x);
                        let block_row = mcu_y as usize * usize::from(sampling.v[component])
                            + usize::from(block_y);
                        let origin_x = block_column * 8;
                        let origin_y = block_row * 8;
                        let pixel_width = width as usize;
                        let pixel_height = height as usize;
                        debug_assert!(origin_x + 8 > pixel_width || origin_y + 8 > pixel_height);
                        if origin_y >= pixel_height - 1 || origin_x >= pixel_width - 1 {
                            let mut edge_coeffs = QuantizedBlock::zeroed();
                            if origin_y >= pixel_height - 1 {
                                let samples = sample_compact_repeated_edge_axis::<1>(
                                    planes[component].as_ref(),
                                    pixel_width,
                                    pixel_height,
                                    origin_x,
                                    origin_y,
                                );
                                fdct_quantize_repeated_axis_into::<1>(
                                    &samples,
                                    quant,
                                    divisors,
                                    cosine,
                                    &mut edge_coeffs,
                                );
                            } else {
                                let samples = sample_compact_repeated_edge_axis::<2>(
                                    planes[component].as_ref(),
                                    pixel_width,
                                    pixel_height,
                                    origin_x,
                                    origin_y,
                                );
                                fdct_quantize_repeated_axis_into::<2>(
                                    &samples,
                                    quant,
                                    divisors,
                                    cosine,
                                    &mut edge_coeffs,
                                );
                            }
                            return encode_block_zigzag(
                                edge_coeffs.dc,
                                &edge_coeffs.ac_coefficients,
                                edge_coeffs.nonzero_ac,
                                &mut prev_dc[component],
                                dc_table,
                                ac_table,
                                writer,
                            );
                        }
                    }
                    block_storage = [0u8; 64];
                    sample_block_from_compact_edge(
                        planes,
                        width,
                        height,
                        sampling,
                        component,
                        mcu_x,
                        mcu_y,
                        block_x,
                        block_y,
                        &mut block_storage,
                    );
                    &block_storage
                }
            } else {
                block_storage = [0u8; 64];
                sample_block(
                    planes,
                    width,
                    height,
                    sampling,
                    component,
                    mcu_x,
                    mcu_y,
                    block_x,
                    block_y,
                    &mut block_storage,
                );
                &block_storage
            };
            #[cfg_attr(
                not(target_arch = "aarch64"),
                expect(
                    unused_mut,
                    reason = "AC token preparation mutates coefficients only on AArch64"
                )
            )]
            let mut coeffs = fdct_quantize(block, quant, divisors, cosine);
            #[cfg(target_arch = "aarch64")]
            if AC_TOKENS {
                let ac_state =
                    prepare_entropy_ac(&mut coeffs.ac_coefficients, coeffs.nonzero_ac, divisors);
                if ac_state & 1 != 0 {
                    return encode_token_block_zigzag(
                        coeffs.dc,
                        &coeffs.ac_coefficients,
                        ac_state,
                        &mut prev_dc[component],
                        dc_table,
                        ac_table,
                        writer,
                    );
                }
                return encode_block_zigzag(
                    coeffs.dc,
                    &coeffs.ac_coefficients,
                    ac_state,
                    &mut prev_dc[component],
                    dc_table,
                    ac_table,
                    writer,
                );
            }
            encode_block_zigzag(
                coeffs.dc,
                &coeffs.ac_coefficients,
                coeffs.nonzero_ac,
                &mut prev_dc[component],
                dc_table,
                ac_table,
                writer,
            )
        })?;
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
enum ThinLumaCacheAction {
    None,
    Store(usize),
    Reuse(usize),
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::inline_always,
    reason = "the thin-image cache decision is forced into the measured per-block loop"
)]
#[inline(always)]
fn thin_luma_cache_action<const AXIS: u8>(
    sampling: JpegBaselineSampling,
    component: usize,
    block_x: u8,
    block_y: u8,
) -> ThinLumaCacheAction {
    if component != 0 {
        return ThinLumaCacheAction::None;
    }
    match AXIS {
        1 if sampling.h[0] == 2 && sampling.v[0] == 2 => {
            let slot = usize::from(block_x);
            debug_assert!(slot < 2 && block_y < 2);
            if block_y == 0 {
                ThinLumaCacheAction::Store(slot)
            } else {
                ThinLumaCacheAction::Reuse(slot)
            }
        }
        2 if sampling.h[0] == 2 => {
            let slot = usize::from(block_y);
            debug_assert!(slot < 2 && block_x < 2);
            if block_x == 0 {
                ThinLumaCacheAction::Store(slot)
            } else {
                ThinLumaCacheAction::Reuse(slot)
            }
        }
        _ => ThinLumaCacheAction::None,
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "outlined thin-image JPEG entropy loop mirrors the ordinary private hot path"
)]
fn encode_entropy_thin_mcu_range_into<const AXIS: u8>(
    planes: &[Cow<'_, [u8]>],
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    q_luma: &[u8; 64],
    q_chroma: &[u8; 64],
    dc_tables: [&JpegBaselineHuffmanTable; 2],
    ac_tables: [&JpegBaselineHuffmanTable; 2],
    cosine: &[[f64; 8]; 8],
    mcus_per_row: u32,
    start_mcu: u32,
    end_mcu: u32,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    debug_assert!(AXIS == 1 || AXIS == 2);
    let luma_divisors = QuantizationDivisors::new(q_luma);
    let chroma_divisors = QuantizationDivisors::new(q_chroma);
    let packed_huffman = packed_standard_huffman_tables(dc_tables, ac_tables);
    let mut prev_dc = [0i32; 3];
    for mcu_index in start_mcu..end_mcu {
        let mcu_y = mcu_index / mcus_per_row;
        let mcu_x = mcu_index % mcus_per_row;
        let mut luma_cache: [Option<QuantizedBlock>; 2] = [None, None];
        for_each_mcu_block(sampling, |component, block_x, block_y| {
            let quant = if component == 0 { q_luma } else { q_chroma };
            let table_index = usize::from(component != 0);
            let dc_table = packed_huffman.dc[table_index];
            let divisors = if component == 0 {
                &luma_divisors
            } else {
                &chroma_divisors
            };
            let ac_table = packed_huffman.ac[table_index];
            let mut emit = |coeffs: &QuantizedBlock| {
                encode_block_zigzag(
                    coeffs.dc,
                    &coeffs.ac_coefficients,
                    coeffs.nonzero_ac,
                    &mut prev_dc[component],
                    dc_table,
                    ac_table,
                    writer,
                )
            };
            match thin_luma_cache_action::<AXIS>(sampling, component, block_x, block_y) {
                ThinLumaCacheAction::Reuse(slot) => emit(
                    luma_cache[slot]
                        .as_ref()
                        .expect("thin luma duplicate follows its cached source block"),
                ),
                action => {
                    let samples = sample_repeated_axis::<AXIS>(
                        planes, width, height, sampling, component, mcu_x, mcu_y, block_x, block_y,
                    );
                    let coeffs =
                        fdct_quantize_repeated_axis::<AXIS>(&samples, quant, divisors, cosine);
                    if let ThinLumaCacheAction::Store(slot) = action {
                        luma_cache[slot] = Some(coeffs);
                        emit(
                            luma_cache[slot]
                                .as_ref()
                                .expect("thin luma source block was just cached"),
                        )
                    } else {
                        emit(&coeffs)
                    }
                }
            }
        })?;
    }
    Ok(())
}

fn entropy_mcu_layout(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
) -> Result<(u32, u32), JpegEncodeError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = width.div_ceil(mcu_width);
    let mcu_rows = height.div_ceil(mcu_height);
    let total_mcus =
        mcus_per_row
            .checked_mul(mcu_rows)
            .ok_or(JpegEncodeError::InternalInvariant {
                reason: "JPEG MCU count overflow",
            })?;
    Ok((mcus_per_row, total_mcus))
}

fn for_each_mcu_block<F>(
    sampling: JpegBaselineSampling,
    mut visit: F,
) -> Result<(), JpegEncodeError>
where
    F: FnMut(usize, u8, u8) -> Result<(), JpegEncodeError>,
{
    for component in 0..sampling.components as usize {
        for block_y in 0..sampling.v[component] {
            for block_x in 0..sampling.h[component] {
                visit(component, block_x, block_y)?;
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "compile-time sampling modes inline block enumeration into the measured entropy loop"
)]
#[inline(always)]
fn for_each_mode_block<const MODE: u8, const BLOCK_PLANES: bool, F>(
    sampling: JpegBaselineSampling,
    mut visit: F,
) -> Result<(), JpegEncodeError>
where
    F: FnMut(usize, u8, u8) -> Result<(), JpegEncodeError>,
{
    if !BLOCK_PLANES {
        return for_each_mcu_block(sampling, visit);
    }

    debug_assert!(matches!(
        MODE,
        SAMPLING_MODE_444 | SAMPLING_MODE_422 | SAMPLING_MODE_420
    ));
    for component in 0usize..3 {
        let (horizontal, vertical): (u8, u8) = if component == 0 {
            match MODE {
                SAMPLING_MODE_444 => (1, 1),
                SAMPLING_MODE_422 => (2, 1),
                SAMPLING_MODE_420 => (2, 2),
                _ => unreachable!("block planes use canonical color sampling"),
            }
        } else {
            (1, 1)
        };
        for block_y in 0..vertical {
            for block_x in 0..horizontal {
                visit(component, block_x, block_y)?;
            }
        }
    }
    Ok(())
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::adapter::baseline_encode_tables;
    use crate::encoder::transform::cosine_table;
    use crate::encoder::{JpegBackend, JpegEncodeOptions, JpegSubsampling};

    fn assert_thin_entropy_matches_ordinary<const AXIS: u8>(
        width: u32,
        height: u32,
        subsampling: JpegSubsampling,
    ) {
        let tables = baseline_encode_tables(JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        })
        .expect("standard JPEG encode tables");
        let pixels = usize::try_from(width * height).expect("test geometry fits usize");
        let planes: Vec<Cow<'static, [u8]>> = vec![
            Cow::Owned(
                (0..pixels)
                    .map(|index| {
                        u8::try_from(index & 0xff)
                            .expect("masked test index fits u8")
                            .wrapping_mul(73)
                            .wrapping_add(19)
                    })
                    .collect(),
            ),
            Cow::Owned(vec![91; pixels]),
            Cow::Owned(vec![173; pixels]),
        ];
        let (mcus_per_row, total_mcus) =
            entropy_mcu_layout(width, height, tables.sampling).expect("test MCU layout");
        let mut ordinary = BitWriter::try_with_max_bytes(65_536).expect("ordinary writer");
        encode_entropy_mcu_range_into(
            &planes,
            width,
            height,
            tables.sampling,
            &tables.q_luma,
            &tables.q_chroma,
            [&tables.huff_dc_luma, &tables.huff_dc_chroma],
            [&tables.huff_ac_luma, &tables.huff_ac_chroma],
            cosine_table(),
            mcus_per_row,
            0,
            total_mcus,
            &mut ordinary,
        )
        .expect("ordinary entropy");
        let mut thin = BitWriter::try_with_max_bytes(65_536).expect("thin writer");
        encode_entropy_thin_mcu_range_into::<AXIS>(
            &planes,
            width,
            height,
            tables.sampling,
            &tables.q_luma,
            &tables.q_chroma,
            [&tables.huff_dc_luma, &tables.huff_dc_chroma],
            [&tables.huff_ac_luma, &tables.huff_ac_chroma],
            cosine_table(),
            mcus_per_row,
            0,
            total_mcus,
            &mut thin,
        )
        .expect("thin entropy");

        assert_eq!(
            thin.into_bytes().expect("thin bytes"),
            ordinary.into_bytes().expect("ordinary bytes"),
            "{width}x{height} {subsampling:?}"
        );
    }

    #[test]
    fn repeated_luma_cache_matches_ordinary_entropy_bytes() {
        assert_thin_entropy_matches_ordinary::<1>(33, 1, JpegSubsampling::Ybr420);
        assert_thin_entropy_matches_ordinary::<2>(1, 33, JpegSubsampling::Ybr422);
        assert_thin_entropy_matches_ordinary::<2>(1, 33, JpegSubsampling::Ybr420);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one matrix compares every entropy layout with the ordinary byte stream"
    )]
    fn token_dispatch_preserves_multiblock_entropy_bytes() {
        #[expect(
            clippy::too_many_lines,
            reason = "the local checker builds and compares one complete entropy configuration"
        )]
        fn check<const MODE: u8, const BLOCK_PLANES: bool, const COMPACT_BLOCK_PLANES: bool>(
            quality: u8,
            subsampling: JpegSubsampling,
            component_count: usize,
            width: u32,
            height: u32,
        ) {
            let tables = baseline_encode_tables(JpegEncodeOptions {
                quality,
                subsampling,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            })
            .expect("standard encode tables");
            assert_eq!(
                (BLOCK_PLANES, COMPACT_BLOCK_PLANES),
                (
                    uses_block_plane_layout(width, height, tables.sampling)
                        || uses_422_block_plane_layout(width, height, tables.sampling)
                        || uses_420_block_plane_layout(width, height, tables.sampling)
                        || uses_compact_block_plane_layout(width, height, tables.sampling),
                    uses_compact_block_plane_layout(width, height, tables.sampling),
                )
            );
            let mut seed = 17u32;
            let planes: Vec<Cow<'static, [u8]>> = (0..component_count)
                .map(|component| {
                    // Precomputed chroma planes use MCU-padded component
                    // dimensions for these fixtures.
                    let (plane_width, plane_height) = match (component > 0, subsampling) {
                        (true, JpegSubsampling::Ybr422) => (
                            width.div_ceil(16) as usize * 8,
                            height.div_ceil(8) as usize * 8,
                        ),
                        (true, JpegSubsampling::Ybr420) => (
                            width.div_ceil(16) as usize * 8,
                            height.div_ceil(16) as usize * 8,
                        ),
                        _ => (width as usize, height as usize),
                    };
                    Cow::Owned(
                        (0..plane_width * plane_height)
                            .map(|index| {
                                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                                // Direct component-plane fixtures store complete
                                // 8x8 blocks contiguously; row-major fixtures use
                                // their existing first-block coordinates.
                                let in_first_block = if BLOCK_PLANES || COMPACT_BLOCK_PLANES {
                                    index < 64
                                } else {
                                    index % plane_width < 8 && index / plane_width < 8
                                };
                                if in_first_block {
                                    127
                                } else if quality == 50 {
                                    u8::try_from(seed >> 24).expect("eight-bit test sample")
                                } else {
                                    96 + u8::try_from(seed >> 26).expect("six-bit test sample")
                                }
                            })
                            .collect(),
                    )
                })
                .collect();
            match quality {
                100 => {
                    assert!(tables.q_luma[1..].iter().all(|&value| value == 1));
                    if component_count > 1 {
                        assert!(tables.q_chroma[1..].iter().all(|&value| value == 1));
                    }
                }
                90 => {
                    assert_eq!(tables.q_luma[0], 3);
                    assert_eq!(tables.q_chroma[0], 3);
                }
                50 => {
                    assert_eq!(tables.q_luma[0], 16);
                    assert_eq!(tables.q_chroma[0], 17);
                }
                _ => panic!("token dispatch fixture quality"),
            }
            let (mcus_per_row, total_mcus) =
                entropy_mcu_layout(width, height, tables.sampling).expect("test MCU layout");
            let mut selected = BitWriter::try_with_max_bytes(65_536).expect("selected writer");
            let mut raw = BitWriter::try_with_max_bytes(65_536).expect("raw writer");
            encode_entropy_mcu_range_into(
                &planes,
                width,
                height,
                tables.sampling,
                &tables.q_luma,
                &tables.q_chroma,
                [&tables.huff_dc_luma, &tables.huff_dc_chroma],
                [&tables.huff_ac_luma, &tables.huff_ac_chroma],
                cosine_table(),
                mcus_per_row,
                0,
                total_mcus,
                &mut selected,
            )
            .expect("selected entropy");
            encode_entropy_mcu_range_into_mode::<MODE, false, BLOCK_PLANES, COMPACT_BLOCK_PLANES>(
                &planes,
                width,
                height,
                tables.sampling,
                &tables.q_luma,
                &tables.q_chroma,
                [&tables.huff_dc_luma, &tables.huff_dc_chroma],
                [&tables.huff_ac_luma, &tables.huff_ac_chroma],
                cosine_table(),
                mcus_per_row,
                0,
                total_mcus,
                &mut raw,
            )
            .expect("raw entropy");
            assert_eq!(
                selected.into_bytes().unwrap(),
                raw.into_bytes().unwrap(),
                "{subsampling:?}"
            );
        }

        check::<SAMPLING_MODE_GRAY, false, false>(100, JpegSubsampling::Gray, 1, 32, 24);
        check::<SAMPLING_MODE_444, true, false>(100, JpegSubsampling::Ybr444, 3, 32, 24);
        check::<SAMPLING_MODE_444, true, true>(100, JpegSubsampling::Ybr444, 3, 24, 17);
        check::<SAMPLING_MODE_422, true, false>(100, JpegSubsampling::Ybr422, 3, 32, 24);
        check::<SAMPLING_MODE_422, true, true>(100, JpegSubsampling::Ybr422, 3, 17, 32);
        check::<SAMPLING_MODE_420, false, false>(100, JpegSubsampling::Ybr420, 3, 8, 24);
        check::<SAMPLING_MODE_420, true, true>(100, JpegSubsampling::Ybr420, 3, 32, 24);
        check::<SAMPLING_MODE_420, true, false>(100, JpegSubsampling::Ybr420, 3, 32, 32);
        check::<SAMPLING_MODE_GRAY, false, false>(90, JpegSubsampling::Gray, 1, 32, 24);
        check::<SAMPLING_MODE_444, true, false>(90, JpegSubsampling::Ybr444, 3, 32, 24);
        check::<SAMPLING_MODE_422, true, false>(90, JpegSubsampling::Ybr422, 3, 32, 24);
        check::<SAMPLING_MODE_420, true, true>(90, JpegSubsampling::Ybr420, 3, 32, 24);
        check::<SAMPLING_MODE_420, true, false>(90, JpegSubsampling::Ybr420, 3, 32, 32);
        check::<SAMPLING_MODE_GRAY, false, false>(50, JpegSubsampling::Gray, 1, 32, 24);
        check::<SAMPLING_MODE_444, true, true>(50, JpegSubsampling::Ybr444, 3, 24, 17);
        check::<SAMPLING_MODE_422, true, false>(50, JpegSubsampling::Ybr422, 3, 32, 24);
        check::<SAMPLING_MODE_420, true, false>(50, JpegSubsampling::Ybr420, 3, 32, 32);
    }
}
