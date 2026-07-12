use crate::{
    allocation::HostPhaseBudget,
    bytes::{
        htj2k_packetization_blocks_as_bytes, htj2k_packetization_packets_as_bytes,
        htj2k_packetization_statuses_as_bytes, htj2k_packetization_statuses_as_bytes_mut,
        htj2k_packetization_subband_tag_states_as_bytes, htj2k_packetization_subbands_as_bytes,
        htj2k_packetization_tag_nodes_as_bytes,
    },
    context::CudaContext,
    driver::CuFunction,
    error::CudaError,
    execution::{cuda_kernel_param, CudaExecutionStats},
    htj2k_decode::{HTJ2K_STATUS_OK, HTJ2K_STATUS_UNSUPPORTED},
    kernels::{htj2k_packetize_launch_geometry, CudaKernel},
    memory::CudaDeviceBuffer,
};

/// One HTJ2K packet prepared for CUDA Tier-2 packetization.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationPacket {
    /// First block metadata row for this packet.
    pub block_start: u32,
    /// Number of block metadata rows in this packet.
    pub block_count: u32,
    /// First subband metadata row for this packet.
    pub subband_start: u32,
    /// Number of subband metadata rows in this packet.
    pub subband_count: u32,
    /// Maximum bytes reserved for this packet's header and body.
    pub output_capacity: u32,
    /// Packet layer index used for first-inclusion tag-tree coding.
    pub layer: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaHtj2kPacketizationKernelPacket {
    pub(crate) block_start: u32,
    pub(crate) block_count: u32,
    pub(crate) subband_start: u32,
    pub(crate) subband_count: u32,
    pub(crate) output_offset: u32,
    pub(crate) output_capacity: u32,
    pub(crate) layer: u32,
}

/// One HTJ2K packet subband layout for CUDA packetization.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationSubband {
    /// First code-block metadata row for this subband.
    pub block_start: u32,
    /// Number of code-block metadata rows in this subband.
    pub block_count: u32,
    /// Number of code-blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub num_cbs_y: u32,
}

/// Initial tag-tree state for one HTJ2K packet subband.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationSubbandTagState {
    /// First inclusion tag-tree node state row for this packet subband.
    pub inclusion_node_start: u32,
    /// First zero-bitplane tag-tree node state row for this packet subband.
    pub zero_bitplane_node_start: u32,
    /// Number of node state rows in each tree.
    pub node_count: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
}

/// Current/known state for one HTJ2K packet tag-tree node.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationTagNodeState {
    /// Tag-tree current value before this packet is emitted.
    pub current: u32,
    /// Nonzero when this node value is already known before this packet.
    pub known: u32,
}

/// One HTJ2K code-block contribution for CUDA packetization.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationBlock {
    /// Byte offset into the contiguous encoded code-block payload.
    pub data_offset: u32,
    /// Encoded code-block payload length in bytes.
    pub data_len: u32,
    /// HTJ2K cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// HTJ2K refinement segment length in bytes.
    pub refinement_length: u32,
    /// Number of coding passes in this contribution.
    pub num_coding_passes: u32,
    /// Number of zero most-significant bitplanes before first inclusion.
    pub num_zero_bitplanes: u32,
    /// L-block value for segment-length coding.
    pub l_block: u32,
    /// Nonzero when this code block was included in an earlier packet for the same packet state.
    pub previously_included: u32,
    /// First packet layer where this code block is included, or tag-tree infinity when absent.
    pub inclusion_layer: u32,
}

/// Status written by the CUDA HTJ2K packetizer for one packet.
#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationStatus {
    /// Zero on success; nonzero values are kernel-defined failures.
    pub code: u32,
    /// Kernel-defined failure detail.
    pub detail: u32,
    /// Number of packet bytes written into this packet slot.
    pub output_len: u32,
    /// Reserved for ABI stability.
    pub reserved0: u32,
}

impl CudaHtj2kPacketizationStatus {
    /// Return true when the CUDA kernel reported success.
    pub fn is_ok(self) -> bool {
        self.code == HTJ2K_STATUS_OK
    }
}

/// CUDA event timings for HTJ2K Tier-2 packetization stages.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaHtj2kPacketizationStageTimings {
    /// Cleanup packetization dispatch time, in microseconds.
    pub packetize_us: u128,
}

/// Host-visible HTJ2K packet payload produced by the CUDA Tier-2 packetizer.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaHtj2kPacketizedTile {
    pub(crate) data: Vec<u8>,
    pub(crate) statuses: Vec<CudaHtj2kPacketizationStatus>,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) stage_timings: CudaHtj2kPacketizationStageTimings,
}

impl CudaHtj2kPacketizedTile {
    /// Concatenated tile packet payload bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consume the packetized tile and return its payload bytes.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Per-packet kernel status rows downloaded after dispatch.
    pub fn statuses(&self) -> &[CudaHtj2kPacketizationStatus] {
        &self.statuses
    }

    /// CUDA execution counters for the packetization dispatch.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// CUDA event timings for the packetization dispatch.
    pub fn stage_timings(&self) -> CudaHtj2kPacketizationStageTimings {
        self.stage_timings
    }
}

#[derive(Clone, Copy)]
struct Htj2kPacketizeCleanupLaunch<'a> {
    payload: &'a CudaDeviceBuffer,
    payload_len: usize,
    packets: &'a CudaDeviceBuffer,
    subbands: &'a CudaDeviceBuffer,
    blocks: &'a CudaDeviceBuffer,
    subband_tag_states: &'a CudaDeviceBuffer,
    tag_nodes: &'a CudaDeviceBuffer,
    subband_tag_state_count: usize,
    tag_node_count: usize,
    output: &'a CudaDeviceBuffer,
    statuses: &'a CudaDeviceBuffer,
    packet_count: usize,
}

impl CudaContext {
    /// Packetize HTJ2K code-block payloads with CUDA using caller-provided tag-tree state.
    #[doc(hidden)]
    pub fn packetize_htj2k_cleanup_packets_with_tag_state(
        &self,
        payload: &[u8],
        packets: &[CudaHtj2kPacketizationPacket],
        subbands: &[CudaHtj2kPacketizationSubband],
        blocks: &[CudaHtj2kPacketizationBlock],
        subband_tag_states: &[CudaHtj2kPacketizationSubbandTagState],
        tag_nodes: &[CudaHtj2kPacketizationTagNodeState],
    ) -> Result<CudaHtj2kPacketizedTile, CudaError> {
        self.packetize_htj2k_cleanup_packets_with_tag_state_and_live_host_bytes(
            payload,
            packets,
            subbands,
            blocks,
            subband_tag_states,
            tag_nodes,
            0,
        )
    }

    /// Packetize HTJ2K blocks while accounting caller-live host metadata.
    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the compatibility boundary mirrors the six existing packetization input slices plus the caller-live byte seed"
    )]
    pub fn packetize_htj2k_cleanup_packets_with_tag_state_and_live_host_bytes(
        &self,
        payload: &[u8],
        packets: &[CudaHtj2kPacketizationPacket],
        subbands: &[CudaHtj2kPacketizationSubband],
        blocks: &[CudaHtj2kPacketizationBlock],
        subband_tag_states: &[CudaHtj2kPacketizationSubbandTagState],
        tag_nodes: &[CudaHtj2kPacketizationTagNodeState],
        live_host_bytes: usize,
    ) -> Result<CudaHtj2kPacketizedTile, CudaError> {
        self.inner.set_current()?;
        let mut launch_budget = HostPhaseBudget::with_live_bytes(
            "CUDA HTJ2K packetization launch metadata",
            live_host_bytes,
        )?;
        let kernel_packets = htj2k_packetization_kernel_packets(
            packets,
            subbands,
            blocks,
            payload.len(),
            &mut launch_budget,
        )?;
        validate_htj2k_packetization_tag_state(subbands, subband_tag_states, tag_nodes)?;
        let total_output = kernel_packets.iter().try_fold(0usize, |acc, packet| {
            let end = usize::try_from(packet.output_offset)
                .ok()
                .and_then(|offset| offset.checked_add(packet.output_capacity as usize))
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            Ok::<usize, CudaError>(acc.max(end))
        })?;
        let output_buffer = self.allocate(total_output)?;
        if packets.is_empty() {
            return Ok(CudaHtj2kPacketizedTile {
                data: Vec::new(),
                statuses: Vec::new(),
                execution: CudaExecutionStats::default(),
                stage_timings: CudaHtj2kPacketizationStageTimings::default(),
            });
        }

        let payload_buffer = self.upload_pinned(payload)?;
        let packet_buffer = self.upload(htj2k_packetization_packets_as_bytes(&kernel_packets))?;
        let subband_buffer = self.upload(htj2k_packetization_subbands_as_bytes(subbands))?;
        let block_buffer = self.upload(htj2k_packetization_blocks_as_bytes(blocks))?;
        let subband_tag_state_buffer = self.upload(
            htj2k_packetization_subband_tag_states_as_bytes(subband_tag_states),
        )?;
        let tag_node_buffer = self.upload(htj2k_packetization_tag_nodes_as_bytes(tag_nodes))?;
        let initial_statuses = launch_budget.try_vec_filled(
            packets.len(),
            CudaHtj2kPacketizationStatus {
                code: HTJ2K_STATUS_UNSUPPORTED,
                ..CudaHtj2kPacketizationStatus::default()
            },
        )?;
        let status_buffer =
            self.upload(htj2k_packetization_statuses_as_bytes(&initial_statuses))?;
        drop(initial_statuses);

        let ((), packetize_us) =
            self.time_default_stream_named_us("j2k.htj2k.encode.packetize", || {
                self.launch_htj2k_packetize_cleanup(Htj2kPacketizeCleanupLaunch {
                    payload: &payload_buffer,
                    payload_len: payload.len(),
                    packets: &packet_buffer,
                    subbands: &subband_buffer,
                    blocks: &block_buffer,
                    subband_tag_states: &subband_tag_state_buffer,
                    tag_nodes: &tag_node_buffer,
                    subband_tag_state_count: subband_tag_states.len(),
                    tag_node_count: tag_nodes.len(),
                    output: &output_buffer,
                    statuses: &status_buffer,
                    packet_count: packets.len(),
                })
            })?;
        let stage_timings = CudaHtj2kPacketizationStageTimings { packetize_us };
        complete_htj2k_packetization(CompleteHtj2kPacketization {
            kernel_packets,
            output_buffer: &output_buffer,
            status_buffer: &status_buffer,
            packet_count: packets.len(),
            live_host_bytes,
            stage_timings,
        })
    }

    fn launch_htj2k_packetize_cleanup(
        &self,
        request: Htj2kPacketizeCleanupLaunch<'_>,
    ) -> Result<(), CudaError> {
        let function = self.htj2k_packetize_kernel_function(CudaKernel::Htj2kPacketizeCleanup)?;
        let mut payload_ptr = request.payload.device_ptr();
        let mut payload_len_u64 =
            u64::try_from(request.payload_len).map_err(|_| CudaError::LengthTooLarge {
                len: request.payload_len,
            })?;
        let mut packets_ptr = request.packets.device_ptr();
        let mut subbands_ptr = request.subbands.device_ptr();
        let mut blocks_ptr = request.blocks.device_ptr();
        let mut subband_tag_states_ptr = request.subband_tag_states.device_ptr();
        let mut tag_nodes_ptr = request.tag_nodes.device_ptr();
        let mut subband_tag_state_count_u64 = u64::try_from(request.subband_tag_state_count)
            .map_err(|_| CudaError::LengthTooLarge {
                len: request.subband_tag_state_count,
            })?;
        let mut tag_node_count_u64 =
            u64::try_from(request.tag_node_count).map_err(|_| CudaError::LengthTooLarge {
                len: request.tag_node_count,
            })?;
        let mut output_ptr = request.output.device_ptr();
        let mut statuses_ptr = request.statuses.device_ptr();
        let mut packet_count_u64 =
            u64::try_from(request.packet_count).map_err(|_| CudaError::LengthTooLarge {
                len: request.packet_count,
            })?;
        let mut params = cuda_kernel_params!(
            payload_ptr,
            payload_len_u64,
            packets_ptr,
            subbands_ptr,
            blocks_ptr,
            subband_tag_states_ptr,
            tag_nodes_ptr,
            subband_tag_state_count_u64,
            tag_node_count_u64,
            output_ptr,
            statuses_ptr,
            packet_count_u64
        );
        let geometry = htj2k_packetize_launch_geometry(request.packet_count).ok_or(
            CudaError::LengthTooLarge {
                len: request.packet_count,
            },
        )?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn htj2k_packetize_kernel_function(&self, kernel: CudaKernel) -> Result<CuFunction, CudaError> {
        self.inner.cuda_oxide_j2k_encode_kernel_function(kernel)
    }
}

struct CompleteHtj2kPacketization<'a> {
    kernel_packets: Vec<CudaHtj2kPacketizationKernelPacket>,
    output_buffer: &'a CudaDeviceBuffer,
    status_buffer: &'a CudaDeviceBuffer,
    packet_count: usize,
    live_host_bytes: usize,
    stage_timings: CudaHtj2kPacketizationStageTimings,
}

fn complete_htj2k_packetization(
    request: CompleteHtj2kPacketization<'_>,
) -> Result<CudaHtj2kPacketizedTile, CudaError> {
    let CompleteHtj2kPacketization {
        kernel_packets,
        output_buffer,
        status_buffer,
        packet_count,
        live_host_bytes,
        stage_timings,
    } = request;
    let mut completion_budget =
        HostPhaseBudget::with_live_bytes("CUDA HTJ2K packetization host outputs", live_host_bytes)?;
    completion_budget.account_vec(&kernel_packets)?;
    let mut statuses =
        completion_budget.try_vec_filled(packet_count, CudaHtj2kPacketizationStatus::default())?;
    status_buffer.copy_to_host(htj2k_packetization_statuses_as_bytes_mut(&mut statuses))?;
    if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
        return Err(CudaError::KernelStatus {
            kernel: "j2k_htj2k_packetize_cleanup",
            code: status.code,
            detail: status.detail,
        });
    }

    let data_len =
        kernel_packets
            .iter()
            .zip(&statuses)
            .try_fold(0usize, |data_len, (packet, status)| {
                validate_htj2k_packet_output(packet, *status, output_buffer.byte_len())?;
                data_len
                    .checked_add(status.output_len as usize)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
            })?;
    let mut data = completion_budget.try_vec_filled(data_len, 0u8)?;
    let mut data_offset = 0usize;
    for (packet, status) in kernel_packets.iter().zip(&statuses) {
        let output_len = status.output_len as usize;
        let next_data_offset = data_offset
            .checked_add(output_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let start = packet.output_offset as usize;
        output_buffer.copy_range_to_host(start, &mut data[data_offset..next_data_offset])?;
        data_offset = next_data_offset;
    }

    Ok(CudaHtj2kPacketizedTile {
        data,
        statuses,
        execution: CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        },
        stage_timings,
    })
}

fn validate_htj2k_packet_output(
    packet: &CudaHtj2kPacketizationKernelPacket,
    status: CudaHtj2kPacketizationStatus,
    output_buffer_len: usize,
) -> Result<(), CudaError> {
    if status.output_len > packet.output_capacity {
        return Err(CudaError::LengthTooLarge {
            len: status.output_len as usize,
        });
    }
    let start = packet.output_offset as usize;
    let end = start
        .checked_add(status.output_len as usize)
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    if end > output_buffer_len {
        return Err(CudaError::LengthTooLarge { len: end });
    }
    Ok(())
}

pub(crate) fn htj2k_packetization_kernel_packets(
    packets: &[CudaHtj2kPacketizationPacket],
    subbands: &[CudaHtj2kPacketizationSubband],
    blocks: &[CudaHtj2kPacketizationBlock],
    payload_len: usize,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<CudaHtj2kPacketizationKernelPacket>, CudaError> {
    let mut output_offset = 0usize;
    let mut kernel_packets = host_budget.try_vec_with_capacity(packets.len())?;
    for packet in packets {
        let block_start = packet.block_start as usize;
        let block_count = packet.block_count as usize;
        let block_end = block_start
            .checked_add(block_count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if block_end > blocks.len() {
            return Err(CudaError::LengthTooLarge { len: block_end });
        }
        let subband_start = packet.subband_start as usize;
        let subband_count = packet.subband_count as usize;
        let subband_end = subband_start
            .checked_add(subband_count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if subband_end > subbands.len() {
            return Err(CudaError::LengthTooLarge { len: subband_end });
        }
        for subband in &subbands[subband_start..subband_end] {
            if subband.num_cbs_x == 0 || subband.num_cbs_y == 0 {
                return Err(CudaError::LengthTooLarge { len: 0 });
            }
            let subband_block_start = subband.block_start as usize;
            let subband_block_count = subband.block_count as usize;
            let subband_block_end = subband_block_start
                .checked_add(subband_block_count)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if subband_block_start < block_start || subband_block_end > block_end {
                return Err(CudaError::LengthTooLarge {
                    len: subband_block_end,
                });
            }
            let expected_blocks = (subband.num_cbs_x as usize)
                .checked_mul(subband.num_cbs_y as usize)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if expected_blocks != subband_block_count {
                return Err(CudaError::LengthTooLarge {
                    len: expected_blocks,
                });
            }
        }
        for block in &blocks[block_start..block_end] {
            let data_end = (block.data_offset as usize)
                .checked_add(block.data_len as usize)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if data_end > payload_len {
                return Err(CudaError::LengthTooLarge { len: data_end });
            }
        }
        let output_capacity = packet.output_capacity as usize;
        let next_output = output_offset
            .checked_add(output_capacity)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if next_output > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge { len: next_output });
        }
        kernel_packets.push(CudaHtj2kPacketizationKernelPacket {
            block_start: packet.block_start,
            block_count: packet.block_count,
            subband_start: packet.subband_start,
            subband_count: packet.subband_count,
            output_offset: u32::try_from(output_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
            output_capacity: packet.output_capacity,
            layer: packet.layer,
        });
        output_offset = next_output;
    }
    Ok(kernel_packets)
}

pub(crate) fn validate_htj2k_packetization_tag_state(
    subbands: &[CudaHtj2kPacketizationSubband],
    subband_tag_states: &[CudaHtj2kPacketizationSubbandTagState],
    tag_nodes: &[CudaHtj2kPacketizationTagNodeState],
) -> Result<(), CudaError> {
    if subband_tag_states.is_empty() {
        if tag_nodes.is_empty() {
            return Ok(());
        }
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K packetization tag nodes require subband tag states".to_string(),
        });
    }
    if subband_tag_states.len() != subbands.len() {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K packetization subband tag-state count must match subband count"
                .to_string(),
        });
    }
    for (subband_index, (subband, state)) in subbands.iter().zip(subband_tag_states).enumerate() {
        let expected_node_count =
            htj2k_packetization_tag_tree_node_count(subband.num_cbs_x, subband.num_cbs_y)?;
        if state.node_count as usize != expected_node_count {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "HTJ2K packetization tag-state node count does not match subband {subband_index}"
                ),
            });
        }
        let node_count = state.node_count as usize;
        let inclusion_end = (state.inclusion_node_start as usize)
            .checked_add(node_count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let zero_bitplane_end = (state.zero_bitplane_node_start as usize)
            .checked_add(node_count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if inclusion_end > tag_nodes.len() || zero_bitplane_end > tag_nodes.len() {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "HTJ2K packetization tag-state offsets exceed tag node count at subband {subband_index}"
                ),
            });
        }
    }
    Ok(())
}

pub(crate) const HTJ2K_PACKET_MAX_TAG_NODES: usize = 2048;

pub(crate) const HTJ2K_PACKET_MAX_TAG_LEVELS: usize = 16;

pub(crate) fn htj2k_packetization_tag_tree_node_count(
    width: u32,
    height: u32,
) -> Result<usize, CudaError> {
    if width == 0 || height == 0 {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K packetization tag-tree dimensions must be nonzero".to_string(),
        });
    }
    let mut levels = 0usize;
    let mut total = 0usize;
    let mut w = width as usize;
    let mut h = height as usize;
    loop {
        if levels >= HTJ2K_PACKET_MAX_TAG_LEVELS {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K packetization tag-tree exceeds kernel level bounds".to_string(),
            });
        }
        let nodes = w
            .checked_mul(h)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        total = total
            .checked_add(nodes)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if total > HTJ2K_PACKET_MAX_TAG_NODES {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K packetization tag-tree exceeds kernel node bounds".to_string(),
            });
        }
        levels += 1;
        if w <= 1 && h <= 1 {
            return Ok(total);
        }
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
}
