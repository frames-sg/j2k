// SPDX-License-Identifier: Apache-2.0

use j2k_native::EncodeProgressionOrder;

use super::{
    J2kLosslessCodestreamAssemblyJob, J2kLosslessCodestreamBlockCodingMode, HT_PACKET_CAPACITY_ENV,
    J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
    J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS, J2K_HT_ENCODE_BASE_OUTPUT_SIZE,
    J2K_HT_ENCODE_MAX_SAMPLES, J2K_HT_ENCODE_MEL_SIZE, J2K_HT_ENCODE_MS_BYTES_PER_SAMPLE_FLOOR,
    J2K_HT_ENCODE_MS_SIZE, J2K_HT_ENCODE_VLC_SIZE,
};
use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum J2kClassicEncodeOutputCapacityMode {
    Conservative,
    Tight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum J2kHtPacketOutputCapacityMode {
    Conservative,
    Tight,
}

pub(crate) fn ht_packet_output_capacity_mode_from_env() -> J2kHtPacketOutputCapacityMode {
    match std::env::var(HT_PACKET_CAPACITY_ENV) {
        Ok(value) if value.eq_ignore_ascii_case("conservative") => {
            J2kHtPacketOutputCapacityMode::Conservative
        }
        _ => J2kHtPacketOutputCapacityMode::Tight,
    }
}

pub(super) fn classic_encode_output_capacity_for_mode(
    width: u32,
    height: u32,
    total_bitplanes: u8,
    mode: J2kClassicEncodeOutputCapacityMode,
) -> Result<usize, Error> {
    let samples = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal encode block size overflow".to_string(),
        })?;
    let bitplane_bytes = samples
        .checked_mul(usize::from(total_bitplanes).max(1))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal encode output capacity overflow".to_string(),
        })?;
    let payload_bytes = match mode {
        J2kClassicEncodeOutputCapacityMode::Conservative => bitplane_bytes.checked_mul(8),
        J2kClassicEncodeOutputCapacityMode::Tight => Some(bitplane_bytes),
    }
    .ok_or_else(|| Error::MetalKernel {
        message: "classic J2K Metal encode output capacity overflow".to_string(),
    })?;
    payload_bytes
        .checked_add(4096)
        .map(|bytes| bytes.max(4096) + 1)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal encode output capacity overflow".to_string(),
        })
}

pub(super) fn classic_encode_output_capacity(
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> Result<usize, Error> {
    classic_encode_output_capacity_for_mode(
        width,
        height,
        total_bitplanes,
        J2kClassicEncodeOutputCapacityMode::Conservative,
    )
}

fn classic_encode_total_coding_passes(total_bitplanes: u8) -> usize {
    if total_bitplanes == 0 {
        0
    } else {
        1 + 3 * (usize::from(total_bitplanes) - 1)
    }
}

fn classic_bypass_segment_index(pass_idx: usize) -> usize {
    if pass_idx < 10 {
        0
    } else {
        1 + 2 * ((pass_idx - 10) / 3) + usize::from(((pass_idx - 10) % 3) == 2)
    }
}

pub(super) fn classic_encode_segment_capacity(style_flags: u32, total_bitplanes: u8) -> usize {
    let total_passes = classic_encode_total_coding_passes(total_bitplanes);
    if total_passes == 0 {
        return 1;
    }
    if (style_flags & J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS) != 0 {
        total_passes
    } else if (style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0 {
        classic_bypass_segment_index(total_passes - 1) + 1
    } else {
        1
    }
}

fn ht_scaled_scratch_size(max_size: usize, sample_count: usize) -> Result<usize, Error> {
    if sample_count > J2K_HT_ENCODE_MAX_SAMPLES {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal encode code-block exceeds maximum sample count".to_string(),
        });
    }

    max_size
        .checked_mul(sample_count)
        .map(|bytes| bytes.div_ceil(J2K_HT_ENCODE_MAX_SAMPLES).max(1))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal encode output capacity overflow".to_string(),
        })
}

pub(super) fn ht_encode_output_capacity(width: u32, height: u32) -> Result<usize, Error> {
    let sample_count = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal encode sample count overflow".to_string(),
        })?;
    let scaled_ms_size = ht_scaled_scratch_size(J2K_HT_ENCODE_MS_SIZE, sample_count)?;
    let ms_floor = sample_count
        .checked_mul(J2K_HT_ENCODE_MS_BYTES_PER_SAMPLE_FLOOR)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal encode output capacity overflow".to_string(),
        })?;
    let ms_size = scaled_ms_size.max(ms_floor).min(J2K_HT_ENCODE_MS_SIZE);
    let mel_size = J2K_HT_ENCODE_MEL_SIZE;
    let vlc_size = J2K_HT_ENCODE_VLC_SIZE;
    ms_size
        .checked_add(mel_size)
        .and_then(|bytes| bytes.checked_add(vlc_size))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal encode output capacity overflow".to_string(),
        })
}

pub(super) fn packet_tree_node_count(width: u32, height: u32) -> Result<usize, Error> {
    if width == 0 || height == 0 {
        return Ok(0);
    }
    let mut total = 0usize;
    let mut w = width;
    let mut h = height;
    loop {
        total = total
            .checked_add(
                usize::try_from(w)
                    .ok()
                    .and_then(|wu| usize::try_from(h).ok().and_then(|hu| wu.checked_mul(hu)))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet tag-tree size overflow".to_string(),
                    })?,
            )
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet tag-tree node count overflow".to_string(),
            })?;
        if w <= 1 && h <= 1 {
            break;
        }
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    Ok(total)
}

pub(super) fn lossless_codestream_payload_offset(
    job: J2kLosslessCodestreamAssemblyJob,
) -> Result<usize, Error> {
    let component_count = usize::from(job.num_components);
    let qcd_steps = 1usize
        .checked_add(
            usize::from(job.num_decomposition_levels)
                .checked_mul(3)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal codestream assembly QCD step count overflow".to_string(),
                })?,
        )
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream assembly QCD step count overflow".to_string(),
        })?;
    let siz_total = 40usize
        .checked_add(
            component_count
                .checked_mul(3)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal codestream assembly SIZ size overflow".to_string(),
                })?,
        )
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream assembly SIZ size overflow".to_string(),
        })?;
    let qcd_total = 5usize
        .checked_add(qcd_steps)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream assembly QCD size overflow".to_string(),
        })?;
    2usize
        .checked_add(siz_total)
        .and_then(|len| {
            len.checked_add(
                if job.block_coding_mode == J2kLosslessCodestreamBlockCodingMode::HighThroughput {
                    10
                } else {
                    0
                },
            )
        })
        .and_then(|len| len.checked_add(14))
        .and_then(|len| len.checked_add(qcd_total))
        .and_then(|len| len.checked_add(if job.write_tlm { 12 } else { 0 }))
        .and_then(|len| len.checked_add(12))
        .and_then(|len| len.checked_add(2))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream payload offset overflow".to_string(),
        })
}

pub(super) fn lossless_codestream_assembly_capacity(
    tile_capacity: usize,
    job: J2kLosslessCodestreamAssemblyJob,
) -> Result<usize, Error> {
    lossless_codestream_payload_offset(job)?
        .checked_add(tile_capacity)
        .and_then(|len| len.checked_add(2))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream assembly capacity overflow".to_string(),
        })
}

fn lossless_raw_sample_bytes(
    job: J2kLosslessCodestreamAssemblyJob,
    overflow_message: &'static str,
) -> Result<usize, Error> {
    let pixels = usize::try_from(job.width)
        .ok()
        .and_then(|width| {
            usize::try_from(job.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| Error::MetalKernel {
            message: overflow_message.to_string(),
        })?;
    let component_count = usize::from(job.num_components);
    let bytes_per_sample = usize::from(job.bit_depth).div_ceil(8).max(1);
    pixels
        .checked_mul(component_count)
        .and_then(|bytes| bytes.checked_mul(bytes_per_sample))
        .ok_or_else(|| Error::MetalKernel {
            message: overflow_message.to_string(),
        })
}

fn ht_lossless_raw_sample_bytes(job: J2kLosslessCodestreamAssemblyJob) -> Result<usize, Error> {
    lossless_raw_sample_bytes(job, "HTJ2K Metal batch raw sample byte count overflow")
}

pub(super) fn classic_packet_output_capacity(
    tier1_output_capacity: usize,
    header_capacity: usize,
    packet_descriptor_count: usize,
    codestream: J2kLosslessCodestreamAssemblyJob,
) -> Result<usize, Error> {
    let descriptor_count = packet_descriptor_count.max(1);
    let conservative_capacity = tier1_output_capacity
        .checked_add(header_capacity.saturating_mul(descriptor_count))
        .and_then(|bytes| bytes.checked_add(1024))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal batch packet output capacity overflow".to_string(),
        })?;
    let raw_bytes =
        lossless_raw_sample_bytes(codestream, "J2K Metal batch raw sample byte count overflow")?;
    let descriptor_header_slack =
        descriptor_count
            .checked_mul(256)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal batch packet descriptor slack overflow".to_string(),
            })?;
    let tight_capacity = raw_bytes
        .checked_add(header_capacity)
        .and_then(|bytes| bytes.checked_add(descriptor_header_slack))
        .and_then(|bytes| bytes.checked_add(64 * 1024))
        .map(|bytes| bytes.max(4096))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal batch packet output capacity overflow".to_string(),
        })?;

    Ok(tight_capacity.min(conservative_capacity))
}

pub(super) fn ht_packet_output_capacity_for_mode(
    code_block_count: usize,
    header_capacity: usize,
    packet_descriptor_count: usize,
    codestream: J2kLosslessCodestreamAssemblyJob,
    mode: J2kHtPacketOutputCapacityMode,
) -> Result<usize, Error> {
    let descriptor_count = packet_descriptor_count.max(1);
    match mode {
        J2kHtPacketOutputCapacityMode::Conservative => code_block_count
            .checked_mul(J2K_HT_ENCODE_BASE_OUTPUT_SIZE)
            .and_then(|bytes| bytes.checked_add(header_capacity.saturating_mul(descriptor_count)))
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal batch packet output capacity overflow".to_string(),
            }),
        J2kHtPacketOutputCapacityMode::Tight => {
            let raw_bytes = ht_lossless_raw_sample_bytes(codestream)?;
            let descriptor_header_slack =
                descriptor_count
                    .checked_mul(256)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch packet descriptor slack overflow".to_string(),
                    })?;
            raw_bytes
                .checked_add(header_capacity)
                .and_then(|bytes| bytes.checked_add(descriptor_header_slack))
                .and_then(|bytes| bytes.checked_add(64 * 1024))
                .map(|bytes| bytes.max(4096))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch packet output capacity overflow".to_string(),
                })
        }
    }
}

pub(super) fn codestream_progression_order_code(order: EncodeProgressionOrder) -> u32 {
    match order {
        EncodeProgressionOrder::Lrcp => 0x00,
        EncodeProgressionOrder::Rlcp => 0x01,
        EncodeProgressionOrder::Rpcl => 0x02,
        EncodeProgressionOrder::Pcrl => 0x03,
        EncodeProgressionOrder::Cprl => 0x04,
    }
}
