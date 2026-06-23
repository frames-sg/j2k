// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use super::{
    MetalLosslessEncodeBatchStats, MetalLosslessEncodeConfig, MetalLosslessEncodeStageStats,
};

const GPU_ENCODE_DEFAULT_INFLIGHT_TILES: usize = 512;
const CLASSIC_GPU_ENCODE_SMALL_BATCH_INFLIGHT_TILES: usize = 16;
const CLASSIC_GPU_ENCODE_LARGE_BATCH_INFLIGHT_TILES: usize = 64;
const CLASSIC_GPU_ENCODE_VERY_LARGE_BATCH_MIN_TILES: usize = 64;
const CLASSIC_GPU_ENCODE_VERY_LARGE_BATCH_INFLIGHT_TILES: usize = 128;
const HTJ2K_GPU_ENCODE_MEDIUM_BATCH_TILES: usize = 64;
const HTJ2K_GPU_ENCODE_MEDIUM_BATCH_INFLIGHT_TILES: usize = 32;
const HTJ2K_GPU_ENCODE_LARGE_BATCH_MIN_TILES: usize = 64;
const HTJ2K_GPU_ENCODE_LARGE_BATCH_INFLIGHT_TILES: usize = 64;
const GPU_ENCODE_FALLBACK_HW_MEM_BYTES: usize = 8 * 1024 * 1024 * 1024;
const GPU_ENCODE_MAX_DEFAULT_MEMORY_BUDGET_BYTES: usize = 10 * 1024 * 1024 * 1024;
const GPU_ENCODE_MEMORY_BUDGET_PERCENT: usize = 40;
const RESIDENT_HT_DEFAULT_CHUNK_CODE_BLOCKS: usize = 131_072;

#[cfg(test)]
pub(super) fn default_gpu_encode_memory_budget_bytes_for_hw_mem(hw_memsize: usize) -> usize {
    default_gpu_encode_memory_budget_bytes_for_hw_mem_inner(hw_memsize)
}

fn default_gpu_encode_memory_budget_bytes_for_hw_mem_inner(hw_memsize: usize) -> usize {
    hw_memsize
        .saturating_mul(GPU_ENCODE_MEMORY_BUDGET_PERCENT)
        .checked_div(100)
        .unwrap_or(0)
        .clamp(1, GPU_ENCODE_MAX_DEFAULT_MEMORY_BUDGET_BYTES)
}

fn default_gpu_encode_memory_budget_bytes() -> usize {
    let hw_memsize = host_memory_bytes().unwrap_or(GPU_ENCODE_FALLBACK_HW_MEM_BYTES);
    default_gpu_encode_memory_budget_bytes_for_hw_mem_inner(hw_memsize)
}

pub(super) fn resident_lossless_encode_config_for_mode(
    config: MetalLosslessEncodeConfig,
    classic_resident_mode: bool,
    tile_count: usize,
) -> MetalLosslessEncodeConfig {
    if config.gpu_encode_inflight_tiles.is_some() {
        return config;
    }
    if classic_resident_mode {
        let classic_inflight_tiles = if tile_count <= CLASSIC_GPU_ENCODE_SMALL_BATCH_INFLIGHT_TILES
        {
            CLASSIC_GPU_ENCODE_SMALL_BATCH_INFLIGHT_TILES
        } else if tile_count <= CLASSIC_GPU_ENCODE_VERY_LARGE_BATCH_MIN_TILES {
            CLASSIC_GPU_ENCODE_LARGE_BATCH_INFLIGHT_TILES
        } else {
            CLASSIC_GPU_ENCODE_VERY_LARGE_BATCH_INFLIGHT_TILES
        };
        MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(classic_inflight_tiles),
            ..config
        }
    } else if tile_count == HTJ2K_GPU_ENCODE_MEDIUM_BATCH_TILES {
        MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(HTJ2K_GPU_ENCODE_MEDIUM_BATCH_INFLIGHT_TILES),
            ..config
        }
    } else if tile_count > HTJ2K_GPU_ENCODE_LARGE_BATCH_MIN_TILES {
        MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(HTJ2K_GPU_ENCODE_LARGE_BATCH_INFLIGHT_TILES),
            ..config
        }
    } else {
        config
    }
}

#[cfg(target_os = "macos")]
fn host_memory_bytes() -> Option<usize> {
    let mut value = 0u64;
    let mut len = core::mem::size_of::<u64>();
    let name = b"hw.memsize\0";
    // SAFETY: `sysctlbyname` writes a u64 into the provided buffer when the name is supported.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast(),
            (&raw mut value).cast(),
            &raw mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == core::mem::size_of::<u64>())
        .then(|| usize::try_from(value).ok())
        .flatten()
}

#[cfg(all(test, not(target_os = "macos")))]
fn host_memory_bytes() -> Option<usize> {
    None
}

pub(super) fn resolve_lossless_encode_config(
    tile_count: usize,
    estimated_peak_bytes_per_tile: usize,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessEncodeBatchStats, crate::Error> {
    if config.gpu_encode_inflight_tiles == Some(0) {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode in-flight tile cap must be greater than zero",
        });
    }
    if config.gpu_encode_memory_budget_bytes == Some(0) {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode memory budget must be greater than zero",
        });
    }

    let effective_memory_budget_bytes = config
        .gpu_encode_memory_budget_bytes
        .unwrap_or_else(default_gpu_encode_memory_budget_bytes)
        .max(1);
    let estimated_peak_bytes_per_tile = estimated_peak_bytes_per_tile.max(1);
    let memory_limited_tiles =
        (effective_memory_budget_bytes / estimated_peak_bytes_per_tile).max(1);
    let configured_or_default = config
        .gpu_encode_inflight_tiles
        .unwrap_or(GPU_ENCODE_DEFAULT_INFLIGHT_TILES);
    let effective_inflight_tiles = configured_or_default
        .min(memory_limited_tiles)
        .min(tile_count.max(1))
        .max(1);

    Ok(MetalLosslessEncodeBatchStats {
        configured_inflight_tiles: config.gpu_encode_inflight_tiles,
        effective_inflight_tiles,
        configured_memory_budget_bytes: config.gpu_encode_memory_budget_bytes,
        effective_memory_budget_bytes,
        estimated_peak_bytes_per_tile,
        max_observed_inflight_tiles: 0,
        encode_wall_duration: Duration::ZERO,
        stage_stats: MetalLosslessEncodeStageStats::default(),
    })
}

#[cfg(test)]
pub(super) fn resolve_lossless_encode_config_for_test(
    tile_count: usize,
    estimated_peak_bytes_per_tile: usize,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessEncodeBatchStats, crate::Error> {
    resolve_lossless_encode_config(tile_count, estimated_peak_bytes_per_tile, config)
}

pub(super) fn resident_lossless_code_block_chunk_cap(code_block_counts: &[usize]) -> usize {
    code_block_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(RESIDENT_HT_DEFAULT_CHUNK_CODE_BLOCKS)
}

pub(super) fn resident_lossless_chunk_ranges_from_code_blocks(
    code_block_counts: &[usize],
    max_tiles: usize,
    max_code_blocks: usize,
) -> Vec<std::ops::Range<usize>> {
    if code_block_counts.is_empty() {
        return Vec::new();
    }
    let max_tiles = max_tiles.max(1);
    let max_code_blocks = max_code_blocks.max(1);
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < code_block_counts.len() {
        let mut end = start;
        let mut chunk_code_blocks = 0usize;
        while end < code_block_counts.len() && end - start < max_tiles {
            let next_code_blocks = code_block_counts[end].max(1);
            let would_exceed_code_blocks =
                end > start && chunk_code_blocks.saturating_add(next_code_blocks) > max_code_blocks;
            if would_exceed_code_blocks {
                break;
            }
            chunk_code_blocks = chunk_code_blocks.saturating_add(next_code_blocks);
            end += 1;
        }
        if end == start {
            end += 1;
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}

#[cfg(test)]
pub(super) fn resident_lossless_chunk_ranges_for_test(
    code_block_counts: &[usize],
    max_tiles: usize,
    max_code_blocks: usize,
) -> Vec<std::ops::Range<usize>> {
    resident_lossless_chunk_ranges_from_code_blocks(code_block_counts, max_tiles, max_code_blocks)
}
