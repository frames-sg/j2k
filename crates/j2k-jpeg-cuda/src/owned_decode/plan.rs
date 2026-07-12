// SPDX-License-Identifier: MIT OR Apache-2.0

//! Neutral fast-packet projection into the CUDA JPEG decode ABI.

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
use j2k_cuda_runtime::{
    CudaJpegEntropyCheckpoint, CudaJpegHuffmanTable, CudaJpegRgb8DecodePlan, CudaJpegRgb8Sampling,
};
use j2k_jpeg::adapter::{
    JpegEntropyCheckpointV1, JpegFastPacket, JpegHuffmanTable, SharedJpegFastPacket,
};

use crate::{allocation::HostPhaseBudget, session::HostOwnerLease, CudaSession, Error};

#[derive(Clone, Copy)]
pub(super) struct FastRgb8PacketParts<'a> {
    pub(super) sampling: CudaJpegRgb8Sampling,
    pub(super) dimensions: (u32, u32),
    pub(super) mcus_per_row: u32,
    pub(super) mcu_rows: u32,
    pub(super) entropy_bytes: &'a [u8],
    pub(super) entropy_checkpoints: &'a [JpegEntropyCheckpointV1],
    pub(super) y_quant: &'a [u16; 64],
    pub(super) cb_quant: &'a [u16; 64],
    pub(super) cr_quant: &'a [u16; 64],
    pub(super) y_dc_table: &'a JpegHuffmanTable,
    pub(super) y_ac_table: &'a JpegHuffmanTable,
    pub(super) cb_dc_table: &'a JpegHuffmanTable,
    pub(super) cb_ac_table: &'a JpegHuffmanTable,
    pub(super) cr_dc_table: &'a JpegHuffmanTable,
    pub(super) cr_ac_table: &'a JpegHuffmanTable,
}

#[derive(Debug)]
pub(super) struct CudaRgb8PlanData<'a> {
    sampling: CudaJpegRgb8Sampling,
    dimensions: (u32, u32),
    mcus_per_row: u32,
    mcu_rows: u32,
    entropy_bytes: &'a [u8],
    entropy_checkpoints: Vec<CudaJpegEntropyCheckpoint>,
    y_quant: [u16; 64],
    cb_quant: [u16; 64],
    cr_quant: [u16; 64],
    y_dc_table: CudaJpegHuffmanTable,
    y_ac_table: CudaJpegHuffmanTable,
    cb_dc_table: CudaJpegHuffmanTable,
    cb_ac_table: CudaJpegHuffmanTable,
    cr_dc_table: CudaJpegHuffmanTable,
    cr_ac_table: CudaJpegHuffmanTable,
    _checkpoint_lease: HostOwnerLease,
}

impl CudaRgb8PlanData<'_> {
    pub(super) fn as_plan(&self) -> CudaJpegRgb8DecodePlan<'_> {
        CudaJpegRgb8DecodePlan {
            sampling: self.sampling,
            dimensions: self.dimensions,
            mcus_per_row: self.mcus_per_row,
            mcu_rows: self.mcu_rows,
            entropy_bytes: self.entropy_bytes,
            entropy_checkpoints: &self.entropy_checkpoints,
            y_quant: self.y_quant,
            cb_quant: self.cb_quant,
            cr_quant: self.cr_quant,
            y_dc_table: self.y_dc_table,
            y_ac_table: self.y_ac_table,
            cb_dc_table: self.cb_dc_table,
            cb_ac_table: self.cb_ac_table,
            cr_dc_table: self.cr_dc_table,
            cr_ac_table: self.cr_ac_table,
        }
    }
}

pub(super) fn build_cuda_rgb8_plan_data<'a>(
    packet: &FastRgb8PacketParts<'a>,
    dimensions: (u32, u32),
    session: &CudaSession,
) -> Result<CudaRgb8PlanData<'a>, Error> {
    if packet.dimensions != dimensions {
        return Err(Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG packet dimensions do not match decoder metadata",
        });
    }
    let (entropy_checkpoints, checkpoint_lease) =
        cuda_entropy_checkpoints(packet.entropy_checkpoints, session)?;
    Ok(CudaRgb8PlanData {
        sampling: packet.sampling,
        dimensions,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        entropy_bytes: packet.entropy_bytes,
        entropy_checkpoints,
        y_quant: *packet.y_quant,
        cb_quant: *packet.cb_quant,
        cr_quant: *packet.cr_quant,
        y_dc_table: cuda_huffman_table(packet.y_dc_table)?,
        y_ac_table: cuda_huffman_table(packet.y_ac_table)?,
        cb_dc_table: cuda_huffman_table(packet.cb_dc_table)?,
        cb_ac_table: cuda_huffman_table(packet.cb_ac_table)?,
        cr_dc_table: cuda_huffman_table(packet.cr_dc_table)?,
        cr_ac_table: cuda_huffman_table(packet.cr_ac_table)?,
        _checkpoint_lease: checkpoint_lease,
    })
}

macro_rules! fast_rgb8_packet_parts {
    ($sampling:expr, $packet:expr $(,)?) => {{
        let packet = $packet;
        FastRgb8PacketParts {
            sampling: $sampling,
            dimensions: packet.dimensions,
            mcus_per_row: packet.mcus_per_row,
            mcu_rows: packet.mcu_rows,
            entropy_bytes: &packet.entropy_bytes,
            entropy_checkpoints: &packet.entropy_checkpoints,
            y_quant: &packet.y_quant,
            cb_quant: &packet.cb_quant,
            cr_quant: &packet.cr_quant,
            y_dc_table: &packet.y_dc_table,
            y_ac_table: &packet.y_ac_table,
            cb_dc_table: &packet.cb_dc_table,
            cb_ac_table: &packet.cb_ac_table,
            cr_dc_table: &packet.cr_dc_table,
            cr_ac_table: &packet.cr_ac_table,
        }
    }};
}

pub(super) fn fast_rgb8_packet_parts(packet: &SharedJpegFastPacket) -> FastRgb8PacketParts<'_> {
    match packet.as_packet() {
        JpegFastPacket::Fast420(packet) => {
            fast_rgb8_packet_parts!(CudaJpegRgb8Sampling::Fast420, packet)
        }
        JpegFastPacket::Fast422(packet) => {
            fast_rgb8_packet_parts!(CudaJpegRgb8Sampling::Fast422, packet)
        }
        JpegFastPacket::Fast444(packet) => {
            fast_rgb8_packet_parts!(CudaJpegRgb8Sampling::Fast444, packet)
        }
    }
}

fn cuda_entropy_checkpoints(
    checkpoints: &[JpegEntropyCheckpointV1],
    session: &CudaSession,
) -> Result<(Vec<CudaJpegEntropyCheckpoint>, HostOwnerLease), Error> {
    session.allocate_owned_host_owner(|external_live_bytes| {
        let mut budget = HostPhaseBudget::with_cap(
            "CUDA JPEG entropy checkpoint conversion",
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        );
        budget.account_bytes(external_live_bytes)?;
        let mut converted = budget.try_vec_with_capacity(checkpoints.len())?;
        converted.extend(checkpoints.iter().copied().map(cuda_entropy_checkpoint));
        let retained_bytes = converted
            .capacity()
            .checked_mul(core::mem::size_of::<CudaJpegEntropyCheckpoint>())
            .ok_or(Error::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA JPEG entropy checkpoint conversion",
            })?;
        Ok((converted, retained_bytes))
    })
}

#[cfg(test)]
pub(super) fn cuda_entropy_checkpoints_with_cap(
    checkpoints: &[JpegEntropyCheckpointV1],
    cache_host_byte_limit: usize,
    total_active_packet_bytes: usize,
    existing_decoder_bytes: usize,
    allocation_cap: usize,
) -> Result<Vec<CudaJpegEntropyCheckpoint>, Error> {
    let mut budget =
        HostPhaseBudget::with_cap("CUDA JPEG entropy checkpoint conversion", allocation_cap);
    budget.account_bytes(cache_host_byte_limit)?;
    budget.account_bytes(total_active_packet_bytes)?;
    budget.account_bytes(existing_decoder_bytes)?;
    let mut converted = budget.try_vec_with_capacity(checkpoints.len())?;
    converted.extend(checkpoints.iter().copied().map(cuda_entropy_checkpoint));
    Ok(converted)
}

pub(super) fn cuda_huffman_table(table: &JpegHuffmanTable) -> Result<CudaJpegHuffmanTable, Error> {
    CudaJpegHuffmanTable::from_jpeg_bits_values(table.bits, table.values_len, table.values)
        .map_err(super::cuda_owned_decode_error)
}

fn cuda_entropy_checkpoint(value: JpegEntropyCheckpointV1) -> CudaJpegEntropyCheckpoint {
    CudaJpegEntropyCheckpoint {
        mcu_index: value.mcu_index,
        entropy_pos: value.entropy_pos,
        bit_acc: value.bit_acc,
        bit_count: value.bit_count,
        y_prev_dc: value.y_prev_dc,
        cb_prev_dc: value.cb_prev_dc,
        cr_prev_dc: value.cr_prev_dc,
        reserved: value.reserved,
        reserved_tail: 0,
    }
}
