// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded CPU entropy orchestration and restart-chunk scheduling.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use crate::adapter::{
    checked_encode_host_live_bytes, JpegBaselineHuffmanTable, JpegBaselineSampling,
};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;

use super::{encode_block, fdct_quantize, sample_block, BitWriter, JpegEncodeError};

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

#[expect(
    clippy::too_many_arguments,
    reason = "private JPEG entropy hot path keeps scalar arguments for optimized codegen"
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
    let mut prev_dc = [0i32; 3];
    for mcu_index in start_mcu..end_mcu {
        let mcu_y = mcu_index / mcus_per_row;
        let mcu_x = mcu_index % mcus_per_row;
        for_each_mcu_block(sampling, |component, block_x, block_y| {
            let quant = if component == 0 { q_luma } else { q_chroma };
            let dc_table = if component == 0 {
                dc_tables[0]
            } else {
                dc_tables[1]
            };
            let ac_table = if component == 0 {
                ac_tables[0]
            } else {
                ac_tables[1]
            };
            let block = sample_block(
                planes, width, height, sampling, component, mcu_x, mcu_y, block_x, block_y,
            );
            let coeffs = fdct_quantize(&block, quant, cosine);
            encode_block(&coeffs, &mut prev_dc[component], dc_table, ac_table, writer)
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

#[cfg(test)]
mod tests {
    #[test]
    fn entropy_orchestration_module_stays_focused() {
        const SOURCE: &str = include_str!("entropy.rs");
        assert!(
            SOURCE.lines().count() <= 430,
            "entropy orchestration should be split before it exceeds 430 lines"
        );
    }
}
