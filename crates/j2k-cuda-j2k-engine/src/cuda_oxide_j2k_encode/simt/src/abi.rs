// SPDX-License-Identifier: MIT OR Apache-2.0

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtEncodeCompactJob {
    pub(crate) source_offset: u32,
    pub(crate) compact_offset: u32,
    pub(crate) data_len: u32,
    pub(crate) reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtPacketJob {
    pub(crate) block_start: u32,
    pub(crate) block_count: u32,
    pub(crate) subband_start: u32,
    pub(crate) subband_count: u32,
    pub(crate) output_offset: u32,
    pub(crate) output_capacity: u32,
    pub(crate) layer: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtPacketSubband {
    pub(crate) block_start: u32,
    pub(crate) block_count: u32,
    pub(crate) num_cbs_x: u32,
    pub(crate) num_cbs_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtPacketBlock {
    pub(crate) data_offset: u32,
    pub(crate) data_len: u32,
    pub(crate) cleanup_length: u32,
    pub(crate) refinement_length: u32,
    pub(crate) num_coding_passes: u32,
    pub(crate) num_zero_bitplanes: u32,
    pub(crate) l_block: u32,
    pub(crate) previously_included: u32,
    pub(crate) inclusion_layer: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtPacketSubbandTagState {
    pub(crate) inclusion_node_start: u32,
    pub(crate) zero_bitplane_node_start: u32,
    pub(crate) node_count: u32,
    pub(crate) reserved0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtPacketTagNodeState {
    pub(crate) current: u32,
    pub(crate) known: u32,
}

#[repr(C)]
pub(crate) struct J2kHtPacketStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) output_len: u32,
    pub(crate) reserved0: u32,
}
