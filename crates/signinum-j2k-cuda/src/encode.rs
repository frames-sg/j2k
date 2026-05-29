// SPDX-License-Identifier: Apache-2.0

use signinum_core::BackendKind;
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::{
    CudaContext, CudaDeviceBuffer, CudaDwt53LevelShape, CudaDwt53Output, CudaDwt97Output,
    CudaError, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeTables, CudaHtj2kPacketizationBlock,
    CudaHtj2kPacketizationPacket, CudaHtj2kPacketizationSubband,
    CudaHtj2kPacketizationSubbandTagState, CudaHtj2kPacketizationTagNodeState, CudaJ2kQuantizeJob,
    CudaJ2kQuantizeSubbandRegionJob,
};
use signinum_j2k_native::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kDeinterleaveToF32Job, J2kEncodeDispatchReport,
    J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Output, J2kForwardDwt97Job,
    J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob, J2kHtCodeBlockEncodeJob,
    J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
    J2kPacketizationResolution, J2kPacketizationSubband, J2kQuantizeSubbandJob,
    J2kTier1CodeBlockEncodeJob,
};
#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

use crate::profile;

/// Encode lossless JPEG 2000/HTJ2K samples through the CUDA encode-stage adapter.
///
/// This CUDA-named API is strict: every caller-provided backend preference is
/// treated as `EncodeBackendPreference::RequireDevice`, so unsupported stage
/// coverage returns an error instead of a CPU fallback codestream.
pub fn encode_j2k_lossless_with_cuda(
    samples: signinum_j2k::J2kLosslessSamples<'_>,
    options: &signinum_j2k::J2kLosslessEncodeOptions,
) -> Result<signinum_j2k::EncodedJ2k, crate::Error> {
    let strict_options = strict_cuda_encode_options(*options);
    let profile_enabled = profile::profile_stages_enabled();
    let mut accelerator = CudaEncodeStageAccelerator::with_profile_collection(profile_enabled);
    let total_start = profile::profile_now(profile_enabled);
    let encoded = signinum_j2k::encode_j2k_lossless_with_accelerator(
        samples,
        &strict_options,
        BackendKind::Cuda,
        &mut accelerator,
    )?;
    reject_non_cuda_encode_backend(&encoded)?;
    if profile_enabled {
        accelerator
            .encode_profile_report(
                &encoded,
                samples.data.len(),
                profile::elapsed_us(total_start),
            )
            .emit("encode");
    }
    Ok(encoded)
}

/// Encode lossless JPEG 2000/HTJ2K samples through CUDA and return stage timings.
pub fn encode_j2k_lossless_with_cuda_and_profile(
    samples: signinum_j2k::J2kLosslessSamples<'_>,
    options: &signinum_j2k::J2kLosslessEncodeOptions,
) -> Result<
    (
        signinum_j2k::EncodedJ2k,
        profile::CudaHtj2kEncodeProfileReport,
    ),
    crate::Error,
> {
    let input_bytes = samples.data.len();
    let strict_options = strict_cuda_encode_options(*options);
    let mut accelerator = CudaEncodeStageAccelerator::with_profile_collection(true);
    let total_start = profile::profile_now(true);
    let encoded = signinum_j2k::encode_j2k_lossless_with_accelerator(
        samples,
        &strict_options,
        BackendKind::Cuda,
        &mut accelerator,
    )?;
    reject_non_cuda_encode_backend(&encoded)?;
    let report =
        accelerator.encode_profile_report(&encoded, input_bytes, profile::elapsed_us(total_start));
    report.emit("encode");
    Ok((encoded, report))
}

fn strict_cuda_encode_options(
    options: signinum_j2k::J2kLosslessEncodeOptions,
) -> signinum_j2k::J2kLosslessEncodeOptions {
    options.with_backend(signinum_j2k::EncodeBackendPreference::RequireDevice)
}

fn reject_non_cuda_encode_backend(encoded: &signinum_j2k::EncodedJ2k) -> Result<(), crate::Error> {
    if encoded.backend == BackendKind::Cuda {
        Ok(())
    } else {
        Err(crate::Error::UnsupportedCudaRequest {
            reason: "strict CUDA HTJ2K encode did not dispatch all required stages",
        })
    }
}

/// CUDA implementation of selected JPEG 2000 encode stages.
#[derive(Debug, Default, Clone)]
pub struct CudaEncodeStageAccelerator {
    #[cfg(feature = "cuda-runtime")]
    context: Option<CudaContext>,
    #[cfg(feature = "cuda-runtime")]
    encode_resources: Option<Arc<CudaHtj2kEncodeResources>>,
    collect_profile: bool,
    deinterleave_attempts: usize,
    forward_rct_attempts: usize,
    forward_ict_attempts: usize,
    forward_dwt53_attempts: usize,
    forward_dwt97_attempts: usize,
    htj2k_tile_attempts: usize,
    quantize_subband_attempts: usize,
    ht_subband_attempts: usize,
    tier1_code_block_attempts: usize,
    ht_code_block_attempts: usize,
    packetization_attempts: usize,
    deinterleave_dispatches: usize,
    forward_rct_dispatches: usize,
    forward_ict_dispatches: usize,
    forward_dwt53_dispatches: usize,
    forward_dwt97_dispatches: usize,
    htj2k_tile_dispatches: usize,
    quantize_subband_dispatches: usize,
    ht_subband_dispatches: usize,
    tier1_code_block_dispatches: usize,
    ht_code_block_dispatches: usize,
    packetization_dispatches: usize,
    deinterleave_us: u128,
    mct_us: u128,
    dwt_us: u128,
    quantize_us: u128,
    ht_encode_us: u128,
    packetize_us: u128,
}

impl CudaEncodeStageAccelerator {
    fn with_profile_collection(collect_profile: bool) -> Self {
        Self {
            collect_profile,
            ..Self::default()
        }
    }

    #[cfg(feature = "cuda-runtime")]
    fn cuda_context(&mut self) -> core::result::Result<Option<CudaContext>, &'static str> {
        if self.context.is_none() {
            match CudaContext::system_default() {
                Ok(context) => self.context = Some(context),
                Err(_) if cuda_runtime_required() => return Err("CUDA encode stage unavailable"),
                Err(_) => return Ok(None),
            }
        }
        Ok(self.context.clone())
    }

    #[cfg(feature = "cuda-runtime")]
    fn cuda_encode_resources(
        &mut self,
        context: &CudaContext,
    ) -> core::result::Result<Arc<CudaHtj2kEncodeResources>, &'static str> {
        if self.encode_resources.is_none() {
            let resources = context
                .upload_htj2k_encode_resources(cuda_htj2k_encode_tables())
                .map_err(|_| "CUDA HTJ2K encode resource upload failed")?;
            self.encode_resources = Some(Arc::new(resources));
        }
        self.encode_resources
            .clone()
            .ok_or("CUDA HTJ2K encode resources unavailable")
    }

    fn encode_profile_report(
        &self,
        encoded: &signinum_j2k::EncodedJ2k,
        input_bytes: usize,
        total_us: u128,
    ) -> profile::CudaHtj2kEncodeProfileReport {
        profile::CudaHtj2kEncodeProfileReport {
            deinterleave_us: self.deinterleave_us,
            mct_us: self.mct_us,
            dwt_us: self.dwt_us,
            quantize_us: self.quantize_us,
            ht_encode_us: self.ht_encode_us,
            packetize_us: self.packetize_us,
            total_us,
            input_bytes,
            codestream_bytes: encoded.codestream.len(),
            block_count: self.ht_code_block_attempts,
            dispatch_count: self.dispatch_report().total(),
            backend: encoded.backend,
        }
    }

    /// Number of deinterleave attempts observed.
    pub fn deinterleave_attempts(&self) -> usize {
        self.deinterleave_attempts
    }

    /// Number of forward RCT attempts observed.
    pub fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    /// Number of forward ICT attempts observed.
    pub fn forward_ict_attempts(&self) -> usize {
        self.forward_ict_attempts
    }

    /// Number of forward 5/3 DWT attempts observed.
    pub fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
    }

    /// Number of forward 9/7 DWT attempts observed.
    pub fn forward_dwt97_attempts(&self) -> usize {
        self.forward_dwt97_attempts
    }

    /// Number of sub-band quantization attempts observed.
    pub fn quantize_subband_attempts(&self) -> usize {
        self.quantize_subband_attempts
    }

    /// Number of classic Tier-1 code-block attempts observed.
    pub fn tier1_code_block_attempts(&self) -> usize {
        self.tier1_code_block_attempts
    }

    /// Number of HT code-block attempts observed.
    pub fn ht_code_block_attempts(&self) -> usize {
        self.ht_code_block_attempts
    }

    /// Number of packetization attempts observed.
    pub fn packetization_attempts(&self) -> usize {
        self.packetization_attempts
    }

    /// Number of deinterleave CUDA dispatches.
    pub fn deinterleave_dispatches(&self) -> usize {
        self.deinterleave_dispatches
    }

    /// Number of forward RCT CUDA dispatches.
    pub fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    /// Number of forward ICT CUDA dispatches.
    pub fn forward_ict_dispatches(&self) -> usize {
        self.forward_ict_dispatches
    }

    /// Number of forward 5/3 DWT CUDA dispatches.
    pub fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
    }

    /// Number of forward 9/7 DWT CUDA dispatches.
    pub fn forward_dwt97_dispatches(&self) -> usize {
        self.forward_dwt97_dispatches
    }

    /// Number of sub-band quantization CUDA dispatches.
    pub fn quantize_subband_dispatches(&self) -> usize {
        self.quantize_subband_dispatches
    }

    /// Number of classic Tier-1 CUDA dispatches.
    pub fn tier1_code_block_dispatches(&self) -> usize {
        self.tier1_code_block_dispatches
    }

    /// Number of HT code-block CUDA dispatches.
    pub fn ht_code_block_dispatches(&self) -> usize {
        self.ht_code_block_dispatches
    }

    /// Number of packetization CUDA dispatches.
    pub fn packetization_dispatches(&self) -> usize {
        self.packetization_dispatches
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_runtime_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

#[cfg(feature = "cuda-runtime")]
fn time_cuda_stage<T>(
    name: &'static str,
    context: &CudaContext,
    collect_profile: bool,
    work: impl FnOnce() -> core::result::Result<T, CudaError>,
) -> core::result::Result<(T, u128), CudaError> {
    if collect_profile {
        context.time_default_stream_named_us(name, work)
    } else {
        context
            .with_nvtx_range(name, work)
            .map(|output| (output, 0))
    }
}

#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Default)]
struct CudaEncodeStageTimings {
    deinterleave_us: u128,
    mct_us: u128,
    dwt_us: u128,
    quantize_us: u128,
    ht_encode_us: u128,
    packetize_us: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaHtj2kPacketizationPlan {
    payload: Vec<u8>,
    packets: Vec<CudaHtj2kPacketizationPlanPacket>,
    subbands: Vec<CudaHtj2kPacketizationPlanSubband>,
    blocks: Vec<CudaHtj2kPacketizationPlanBlock>,
    tag_states: Vec<CudaHtj2kPacketizationPlanSubbandTagState>,
    tag_nodes: Vec<CudaHtj2kPacketizationPlanTagNodeState>,
}

struct CudaHtj2kPacketizationPlanSink<'a> {
    payload: &'a mut Vec<u8>,
    packets: &'a mut Vec<CudaHtj2kPacketizationPlanPacket>,
    subbands: &'a mut Vec<CudaHtj2kPacketizationPlanSubband>,
    blocks: &'a mut Vec<CudaHtj2kPacketizationPlanBlock>,
    tag_states: &'a mut Vec<CudaHtj2kPacketizationPlanSubbandTagState>,
    tag_nodes: &'a mut Vec<CudaHtj2kPacketizationPlanTagNodeState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaHtj2kPacketizationPlanPacket {
    block_start: u32,
    block_count: u32,
    subband_start: u32,
    subband_count: u32,
    output_capacity: u32,
    layer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaHtj2kPacketizationPlanSubband {
    block_start: u32,
    block_count: u32,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaHtj2kPacketizationPlanBlock {
    data_offset: u32,
    data_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    num_coding_passes: u32,
    num_zero_bitplanes: u32,
    l_block: u32,
    previously_included: u32,
    inclusion_layer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaHtj2kPacketizationPlanSubbandTagState {
    inclusion_node_start: u32,
    zero_bitplane_node_start: u32,
    node_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaHtj2kPacketizationPlanTagNodeState {
    current: u32,
    known: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaHtj2kPacketizationTagTreeState {
    values: Vec<u32>,
    current: Vec<u32>,
    known: Vec<u32>,
    widths: Vec<u32>,
    heights: Vec<u32>,
    offsets: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaHtj2kPacketizationBlockState {
    previously_included: bool,
    l_block: u32,
    inclusion_layer: u32,
    first_inclusion_zero_bitplanes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaHtj2kPacketizationSubbandState {
    num_cbs_x: u32,
    num_cbs_y: u32,
    inclusion_tree: CudaHtj2kPacketizationTagTreeState,
    zero_bitplane_tree: CudaHtj2kPacketizationTagTreeState,
    blocks: Vec<CudaHtj2kPacketizationBlockState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaHtj2kPacketizationState {
    subbands: Vec<CudaHtj2kPacketizationSubbandState>,
}

fn flatten_cuda_htj2k_packetization_job(
    job: J2kPacketizationEncodeJob<'_>,
) -> core::result::Result<CudaHtj2kPacketizationPlan, &'static str> {
    if job.resolution_count as usize != job.resolutions.len() {
        return Err("CUDA HTJ2K packetization resolution count mismatch");
    }

    let mut payload = Vec::new();
    let mut packets = Vec::new();
    let mut subbands = Vec::new();
    let mut blocks = Vec::new();
    let mut tag_states = Vec::new();
    let mut tag_nodes = Vec::new();

    {
        let mut sink = CudaHtj2kPacketizationPlanSink {
            payload: &mut payload,
            packets: &mut packets,
            subbands: &mut subbands,
            blocks: &mut blocks,
            tag_states: &mut tag_states,
            tag_nodes: &mut tag_nodes,
        };
        if job.packet_descriptors.is_empty() {
            if job.num_layers != 1 {
                return Err(
                    "CUDA HTJ2K packetization requires explicit descriptors for multiple layers",
                );
            }
            for packet_index in 0..job.resolutions.len() {
                flatten_cuda_htj2k_packet(
                    job.resolutions
                        .get(packet_index)
                        .ok_or("CUDA HTJ2K packet descriptor index out of range")?,
                    &mut sink,
                )?;
            }
        } else {
            let state_count = job
                .packet_descriptors
                .iter()
                .map(|descriptor| descriptor.state_index as usize)
                .max()
                .map_or(0usize, |max_state| max_state + 1);
            let mut states: Vec<Option<CudaHtj2kPacketizationState>> =
                core::iter::repeat_with(|| None).take(state_count).collect();
            for descriptor in job.packet_descriptors {
                if descriptor.layer >= job.num_layers {
                    return Err("CUDA HTJ2K packetization descriptor layer exceeds layer count");
                }
                let resolution = job
                    .resolutions
                    .get(descriptor.packet_index as usize)
                    .ok_or("CUDA HTJ2K packet descriptor index out of range")?;
                let state = states
                    .get_mut(descriptor.state_index as usize)
                    .ok_or("CUDA HTJ2K packet descriptor state index out of range")?;
                if let Some(existing) = state {
                    validate_cuda_htj2k_packetization_state_layout(existing, resolution)?;
                } else {
                    *state = Some(seed_cuda_htj2k_packetization_state(resolution)?);
                }
                let state = state
                    .as_mut()
                    .ok_or("CUDA HTJ2K packetization state initialization failed")?;
                record_cuda_htj2k_packetization_first_inclusion_layers(
                    state,
                    resolution,
                    descriptor.layer,
                )?;
            }
            for state in states.iter_mut().flatten() {
                finalize_cuda_htj2k_packetization_tag_trees(state);
            }
            for descriptor in job.packet_descriptors {
                if descriptor.layer >= job.num_layers {
                    return Err("CUDA HTJ2K packetization descriptor layer exceeds layer count");
                }
                let resolution = job
                    .resolutions
                    .get(descriptor.packet_index as usize)
                    .ok_or("CUDA HTJ2K packet descriptor index out of range")?;
                let state = states
                    .get_mut(descriptor.state_index as usize)
                    .ok_or("CUDA HTJ2K packet descriptor state index out of range")?;
                if let Some(existing) = state {
                    validate_cuda_htj2k_packetization_state_layout(existing, resolution)?;
                } else {
                    *state = Some(seed_cuda_htj2k_packetization_state(resolution)?);
                }
                let state = state
                    .as_mut()
                    .ok_or("CUDA HTJ2K packetization state initialization failed")?;
                flatten_cuda_htj2k_packet_with_state(
                    resolution,
                    descriptor.layer,
                    state,
                    &mut sink,
                )?;
            }
        }
    }

    if job.code_block_count as usize != blocks.len() {
        return Err("CUDA HTJ2K packetization code-block count mismatch");
    }

    Ok(CudaHtj2kPacketizationPlan {
        payload,
        packets,
        subbands,
        blocks,
        tag_states,
        tag_nodes,
    })
}

fn seed_cuda_htj2k_packetization_state(
    resolution: &signinum_j2k_native::J2kPacketizationResolution<'_>,
) -> core::result::Result<CudaHtj2kPacketizationState, &'static str> {
    let mut subbands = Vec::with_capacity(resolution.subbands.len());
    for subband in &resolution.subbands {
        let block_count = u32::try_from(subband.code_blocks.len())
            .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
        if subband.num_cbs_x == 0
            || subband.num_cbs_y == 0
            || subband.num_cbs_x.saturating_mul(subband.num_cbs_y) != block_count
        {
            return Err("CUDA HTJ2K packetization subband code-block layout mismatch");
        }
        let mut inclusion_tree =
            CudaHtj2kPacketizationTagTreeState::new(subband.num_cbs_x, subband.num_cbs_y)?;
        let zero_bitplane_tree =
            CudaHtj2kPacketizationTagTreeState::new(subband.num_cbs_x, subband.num_cbs_y)?;
        for idx in 0..subband.code_blocks.len() {
            let (x, y) = cuda_htj2k_packetization_block_xy(idx, subband.num_cbs_x)?;
            inclusion_tree.set_leaf_value(x, y, CUDA_HTJ2K_PACKET_TAG_INF);
        }
        subbands.push(CudaHtj2kPacketizationSubbandState {
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
            inclusion_tree,
            zero_bitplane_tree,
            blocks: subband
                .code_blocks
                .iter()
                .map(|block| CudaHtj2kPacketizationBlockState {
                    previously_included: block.previously_included,
                    l_block: block.l_block,
                    inclusion_layer: CUDA_HTJ2K_PACKET_TAG_INF,
                    first_inclusion_zero_bitplanes: 0,
                })
                .collect(),
        });
    }
    Ok(CudaHtj2kPacketizationState { subbands })
}

fn validate_cuda_htj2k_packetization_state_layout(
    state: &CudaHtj2kPacketizationState,
    resolution: &signinum_j2k_native::J2kPacketizationResolution<'_>,
) -> core::result::Result<(), &'static str> {
    if state.subbands.len() != resolution.subbands.len() {
        return Err("CUDA HTJ2K packetization state layout mismatch");
    }
    for (state_subband, packet_subband) in state.subbands.iter().zip(&resolution.subbands) {
        if state_subband.num_cbs_x != packet_subband.num_cbs_x
            || state_subband.num_cbs_y != packet_subband.num_cbs_y
            || state_subband.blocks.len() != packet_subband.code_blocks.len()
        {
            return Err("CUDA HTJ2K packetization state layout mismatch");
        }
    }
    Ok(())
}

const CUDA_HTJ2K_PACKET_TAG_INF: u32 = 0x7FFF_FFFF;
const CUDA_HTJ2K_PACKET_MAX_TAG_NODES: usize = 2048;
const CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS: usize = 16;

fn cuda_htj2k_packetization_block_xy(
    index: usize,
    num_cbs_x: u32,
) -> core::result::Result<(u32, u32), &'static str> {
    let index =
        u32::try_from(index).map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
    Ok((index % num_cbs_x, index / num_cbs_x))
}

impl CudaHtj2kPacketizationTagTreeState {
    fn new(width: u32, height: u32) -> core::result::Result<Self, &'static str> {
        if width == 0 || height == 0 {
            return Err("CUDA HTJ2K packetization subband code-block layout mismatch");
        }

        let mut widths = Vec::new();
        let mut heights = Vec::new();
        let mut offsets = Vec::new();
        let mut total_nodes = 0usize;
        let mut w = width;
        let mut h = height;
        loop {
            if widths.len() >= CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS {
                return Err("CUDA HTJ2K packetization tag-tree exceeds kernel bounds");
            }
            let nodes = (w as usize)
                .checked_mul(h as usize)
                .ok_or("CUDA HTJ2K packetization tag-tree exceeds kernel bounds")?;
            let next_total = total_nodes
                .checked_add(nodes)
                .ok_or("CUDA HTJ2K packetization tag-tree exceeds kernel bounds")?;
            if next_total > CUDA_HTJ2K_PACKET_MAX_TAG_NODES {
                return Err("CUDA HTJ2K packetization tag-tree exceeds kernel bounds");
            }
            offsets.push(total_nodes);
            widths.push(w);
            heights.push(h);
            total_nodes = next_total;
            if w <= 1 && h <= 1 {
                break;
            }
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        Ok(Self {
            values: vec![0; total_nodes],
            current: vec![0; total_nodes],
            known: vec![0; total_nodes],
            widths,
            heights,
            offsets,
        })
    }

    fn set_leaf_value(&mut self, x: u32, y: u32, value: u32) {
        let idx = self.offsets[0] + (y * self.widths[0] + x) as usize;
        self.values[idx] = value;
    }

    #[allow(clippy::similar_names)]
    fn propagate(&mut self) {
        for level in 1..self.widths.len() {
            let prev_w = self.widths[level - 1];
            let prev_h = self.heights[level - 1];
            let curr_w = self.widths[level];
            let curr_h = self.heights[level];
            for cy in 0..curr_h {
                for cx in 0..curr_w {
                    let child_x_start = cx * 2;
                    let child_y_start = cy * 2;
                    let child_x_end = ((cx + 1) * 2).min(prev_w);
                    let child_y_end = ((cy + 1) * 2).min(prev_h);
                    let mut min_value = u32::MAX;
                    for child_y in child_y_start..child_y_end {
                        for child_x in child_x_start..child_x_end {
                            let child_idx =
                                self.offsets[level - 1] + (child_y * prev_w + child_x) as usize;
                            min_value = min_value.min(self.values[child_idx]);
                        }
                    }
                    let parent_idx = self.offsets[level] + (cy * curr_w + cx) as usize;
                    self.values[parent_idx] = min_value;
                }
            }
        }
    }

    fn encode_state_only(&mut self, x: u32, y: u32, max_value: u32) {
        let mut path = Vec::with_capacity(self.widths.len());
        let mut cx = x;
        let mut cy = y;
        for level in 0..self.widths.len() {
            path.push(self.offsets[level] + (cy * self.widths[level] + cx) as usize);
            cx /= 2;
            cy /= 2;
        }

        for node_idx in path.into_iter().rev() {
            if self.known[node_idx] == 0 {
                let target = self.values[node_idx].min(max_value);
                if self.values[node_idx] < max_value {
                    self.known[node_idx] = 1;
                }
                self.current[node_idx] = target;
            }
        }
    }

    fn append_snapshot(
        &self,
        out: &mut Vec<CudaHtj2kPacketizationPlanTagNodeState>,
    ) -> core::result::Result<u32, &'static str> {
        let start = u32::try_from(out.len())
            .map_err(|_| "CUDA HTJ2K packetization tag-state exceeds u32")?;
        out.extend(
            self.current
                .iter()
                .copied()
                .zip(self.known.iter().copied())
                .map(|(current, known)| CudaHtj2kPacketizationPlanTagNodeState { current, known }),
        );
        Ok(start)
    }

    fn node_count(&self) -> u32 {
        u32::try_from(self.current.len()).expect("tag tree node count was bounded at construction")
    }
}

fn record_cuda_htj2k_packetization_first_inclusion_layers(
    state: &mut CudaHtj2kPacketizationState,
    resolution: &signinum_j2k_native::J2kPacketizationResolution<'_>,
    layer: u8,
) -> core::result::Result<(), &'static str> {
    validate_cuda_htj2k_packetization_state_layout(state, resolution)?;
    for (state_subband, packet_subband) in state.subbands.iter_mut().zip(&resolution.subbands) {
        for (idx, (state_block, packet_block)) in state_subband
            .blocks
            .iter_mut()
            .zip(&packet_subband.code_blocks)
            .enumerate()
        {
            if packet_block.num_coding_passes == 0 {
                continue;
            }
            let layer = u32::from(layer);
            if layer < state_block.inclusion_layer {
                state_block.inclusion_layer = layer;
                state_block.first_inclusion_zero_bitplanes =
                    u32::from(packet_block.num_zero_bitplanes);
                let (x, y) = cuda_htj2k_packetization_block_xy(idx, state_subband.num_cbs_x)?;
                state_subband.inclusion_tree.set_leaf_value(x, y, layer);
                state_subband.zero_bitplane_tree.set_leaf_value(
                    x,
                    y,
                    state_block.first_inclusion_zero_bitplanes,
                );
            }
        }
    }
    Ok(())
}

fn finalize_cuda_htj2k_packetization_tag_trees(state: &mut CudaHtj2kPacketizationState) {
    for subband in &mut state.subbands {
        subband.inclusion_tree.propagate();
        subband.zero_bitplane_tree.propagate();
    }
}

fn append_cuda_htj2k_packetization_tag_state(
    state_subband: Option<&CudaHtj2kPacketizationSubbandState>,
    num_cbs_x: u32,
    num_cbs_y: u32,
    tag_states: &mut Vec<CudaHtj2kPacketizationPlanSubbandTagState>,
    tag_nodes: &mut Vec<CudaHtj2kPacketizationPlanTagNodeState>,
) -> core::result::Result<(), &'static str> {
    let (inclusion_node_start, zero_bitplane_node_start, node_count) =
        if let Some(state_subband) = state_subband {
            let inclusion_start = state_subband.inclusion_tree.append_snapshot(tag_nodes)?;
            let zero_bitplane_start = state_subband
                .zero_bitplane_tree
                .append_snapshot(tag_nodes)?;
            (
                inclusion_start,
                zero_bitplane_start,
                state_subband.inclusion_tree.node_count(),
            )
        } else {
            let zero_tree = CudaHtj2kPacketizationTagTreeState::new(num_cbs_x, num_cbs_y)?;
            let inclusion_start = zero_tree.append_snapshot(tag_nodes)?;
            let zero_bitplane_start = zero_tree.append_snapshot(tag_nodes)?;
            (inclusion_start, zero_bitplane_start, zero_tree.node_count())
        };
    tag_states.push(CudaHtj2kPacketizationPlanSubbandTagState {
        inclusion_node_start,
        zero_bitplane_node_start,
        node_count,
    });
    Ok(())
}

fn update_cuda_htj2k_packetization_state_after_block(
    state: &mut CudaHtj2kPacketizationState,
    subband_index: usize,
    block_index: usize,
    layer: u8,
    code_block: &J2kPacketizationCodeBlock<'_>,
    l_block: u32,
) -> core::result::Result<(), &'static str> {
    let state_subband = state
        .subbands
        .get_mut(subband_index)
        .ok_or("CUDA HTJ2K packetization state layout mismatch")?;
    let (x, y) = cuda_htj2k_packetization_block_xy(block_index, state_subband.num_cbs_x)?;
    let previously_included = state_subband
        .blocks
        .get(block_index)
        .ok_or("CUDA HTJ2K packetization state layout mismatch")?
        .previously_included;

    if !previously_included {
        state_subband
            .inclusion_tree
            .encode_state_only(x, y, u32::from(layer) + 1);
        if code_block.num_coding_passes == 0 {
            return Ok(());
        }
        state_subband.zero_bitplane_tree.encode_state_only(
            x,
            y,
            u32::from(code_block.num_zero_bitplanes) + 1,
        );
    }

    if code_block.num_coding_passes > 0 {
        let state_block = state_subband
            .blocks
            .get_mut(block_index)
            .ok_or("CUDA HTJ2K packetization state layout mismatch")?;
        let (cleanup_length, refinement_length) = cuda_ht_segment_lengths(code_block)?;
        state_block.l_block = updated_ht_l_block(
            l_block,
            code_block.num_coding_passes,
            cleanup_length,
            refinement_length,
        )?;
        state_block.previously_included = true;
    }
    Ok(())
}

fn flatten_cuda_htj2k_packet(
    resolution: &signinum_j2k_native::J2kPacketizationResolution<'_>,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> core::result::Result<(), &'static str> {
    flatten_cuda_htj2k_packet_inner(resolution, 0, None, sink)
}

fn flatten_cuda_htj2k_packet_with_state(
    resolution: &signinum_j2k_native::J2kPacketizationResolution<'_>,
    layer: u8,
    state: &mut CudaHtj2kPacketizationState,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> core::result::Result<(), &'static str> {
    flatten_cuda_htj2k_packet_inner(resolution, layer, Some(state), sink)
}

fn flatten_cuda_htj2k_packet_inner(
    resolution: &signinum_j2k_native::J2kPacketizationResolution<'_>,
    layer: u8,
    mut state: Option<&mut CudaHtj2kPacketizationState>,
    sink: &mut CudaHtj2kPacketizationPlanSink<'_>,
) -> core::result::Result<(), &'static str> {
    let block_start = u32::try_from(sink.blocks.len())
        .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
    let subband_start = u32::try_from(sink.subbands.len())
        .map_err(|_| "CUDA HTJ2K packetization subband count exceeds u32")?;
    let mut body_len = 0usize;
    let mut block_count = 0usize;
    let packet_has_data = resolution.subbands.iter().any(|subband| {
        subband
            .code_blocks
            .iter()
            .any(|block| block.num_coding_passes > 0)
    });

    for (subband_index, subband) in resolution.subbands.iter().enumerate() {
        let subband_code_blocks = u32::try_from(subband.code_blocks.len())
            .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
        if subband.num_cbs_x == 0
            || subband.num_cbs_y == 0
            || subband.num_cbs_x.saturating_mul(subband.num_cbs_y) != subband_code_blocks
        {
            return Err("CUDA HTJ2K packetization subband code-block layout mismatch");
        }

        let subband_block_start = u32::try_from(sink.blocks.len())
            .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?;
        let state_subband = state
            .as_deref()
            .and_then(|state| state.subbands.get(subband_index));
        append_cuda_htj2k_packetization_tag_state(
            state_subband,
            subband.num_cbs_x,
            subband.num_cbs_y,
            sink.tag_states,
            sink.tag_nodes,
        )?;
        for (block_index, code_block) in subband.code_blocks.iter().enumerate() {
            if code_block.block_coding_mode != J2kPacketizationBlockCodingMode::HighThroughput {
                return Err("CUDA packetization only supports HTJ2K block-coded packets");
            }
            if code_block.num_coding_passes > 164 {
                return Err("CUDA HTJ2K packetization coding pass count exceeds JPEG 2000 bounds");
            }
            let (previously_included, l_block, inclusion_layer, zero_bitplanes) =
                if let Some(state) = state.as_deref() {
                    let state_block = state
                        .subbands
                        .get(subband_index)
                        .and_then(|state_subband| state_subband.blocks.get(block_index))
                        .ok_or("CUDA HTJ2K packetization state layout mismatch")?;
                    (
                        state_block.previously_included,
                        state_block.l_block,
                        state_block.inclusion_layer,
                        state_block.first_inclusion_zero_bitplanes,
                    )
                } else {
                    (
                        code_block.previously_included,
                        code_block.l_block,
                        if code_block.num_coding_passes > 0 {
                            0
                        } else {
                            CUDA_HTJ2K_PACKET_TAG_INF
                        },
                        u32::from(code_block.num_zero_bitplanes),
                    )
                };
            if code_block.num_coding_passes > 0
                && !previously_included
                && inclusion_layer != u32::from(layer)
            {
                return Err(
                    "CUDA HTJ2K packetization descriptor order does not match first inclusion layer",
                );
            }
            if state.is_none() && previously_included {
                return Err("CUDA HTJ2K packetization requires first-inclusion packets");
            }
            if code_block.num_coding_passes == 0 && !code_block.data.is_empty() {
                return Err("CUDA HTJ2K packetization empty contributions must not carry payload");
            }
            if zero_bitplanes > 31 || l_block > 31 {
                return Err("CUDA HTJ2K packetization header fields exceed kernel bounds");
            }

            let data_offset = u32::try_from(sink.payload.len())
                .map_err(|_| "CUDA HTJ2K packetization payload exceeds u32")?;
            let data_len = if code_block.num_coding_passes == 0 {
                0
            } else {
                u32::try_from(code_block.data.len())
                    .map_err(|_| "CUDA HTJ2K packetization code-block payload exceeds u32")?
            };
            let (cleanup_length, refinement_length) = cuda_ht_segment_lengths(code_block)?;
            if code_block.num_coding_passes > 0 {
                sink.payload.extend_from_slice(code_block.data);
                body_len = body_len
                    .checked_add(code_block.data.len())
                    .ok_or("CUDA HTJ2K packetization body length overflow")?;
            }
            sink.blocks.push(CudaHtj2kPacketizationPlanBlock {
                data_offset,
                data_len,
                cleanup_length,
                refinement_length,
                num_coding_passes: u32::from(code_block.num_coding_passes),
                num_zero_bitplanes: zero_bitplanes,
                l_block,
                previously_included: u32::from(previously_included),
                inclusion_layer,
            });
            if packet_has_data {
                if let Some(state) = state.as_deref_mut() {
                    update_cuda_htj2k_packetization_state_after_block(
                        state,
                        subband_index,
                        block_index,
                        layer,
                        code_block,
                        l_block,
                    )?;
                }
            }
            block_count = block_count
                .checked_add(1)
                .ok_or("CUDA HTJ2K packetization block count overflow")?;
        }
        sink.subbands.push(CudaHtj2kPacketizationPlanSubband {
            block_start: subband_block_start,
            block_count: subband_code_blocks,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
        });
    }

    let header_capacity = 256usize
        .checked_add(
            block_count
                .checked_mul(64)
                .ok_or("CUDA HTJ2K packetization capacity overflow")?,
        )
        .ok_or("CUDA HTJ2K packetization capacity overflow")?;
    let output_capacity = body_len
        .checked_add(header_capacity)
        .ok_or("CUDA HTJ2K packetization capacity overflow")?;
    sink.packets.push(CudaHtj2kPacketizationPlanPacket {
        block_start,
        block_count: u32::try_from(block_count)
            .map_err(|_| "CUDA HTJ2K packetization block count exceeds u32")?,
        subband_start,
        subband_count: u32::try_from(resolution.subbands.len())
            .map_err(|_| "CUDA HTJ2K packetization subband count exceeds u32")?,
        output_capacity: u32::try_from(output_capacity)
            .map_err(|_| "CUDA HTJ2K packetization packet capacity exceeds u32")?,
        layer: u32::from(layer),
    });
    Ok(())
}

fn updated_ht_l_block(
    mut l_block: u32,
    num_coding_passes: u8,
    cleanup_length: u32,
    refinement_length: u32,
) -> core::result::Result<u32, &'static str> {
    let mut num_bits = ht_cleanup_length_bits(l_block, num_coding_passes);
    let refinement_extra_bits = u32::from(num_coding_passes > 2);
    while !value_fits_in_bits(cleanup_length, num_bits)
        || (num_coding_passes > 1
            && !value_fits_in_bits(refinement_length, l_block + refinement_extra_bits))
    {
        l_block = l_block
            .checked_add(1)
            .ok_or("CUDA HTJ2K packetization L-block overflow")?;
        num_bits = num_bits
            .checked_add(1)
            .ok_or("CUDA HTJ2K packetization L-block overflow")?;
    }
    Ok(l_block)
}

fn cuda_ht_segment_lengths(
    code_block: &J2kPacketizationCodeBlock<'_>,
) -> core::result::Result<(u32, u32), &'static str> {
    if code_block.num_coding_passes == 0 {
        if code_block.data.is_empty()
            && code_block.ht_cleanup_length == 0
            && code_block.ht_refinement_length == 0
        {
            return Ok((0, 0));
        }
        return Err("CUDA HTJ2K packetization empty contributions must not carry payload");
    }

    let data_len = u32::try_from(code_block.data.len())
        .map_err(|_| "CUDA HTJ2K packetization code-block payload exceeds u32")?;
    if code_block.num_coding_passes == 1 {
        if code_block.ht_refinement_length != 0 {
            return Err("CUDA HTJ2K single-pass packet contribution has refinement bytes");
        }
        let cleanup_length = if code_block.ht_cleanup_length == 0 {
            data_len
        } else {
            code_block.ht_cleanup_length
        };
        if cleanup_length != data_len {
            return Err("CUDA HTJ2K single-pass packet contribution length mismatch");
        }
        return Ok((cleanup_length, 0));
    }

    if code_block.ht_cleanup_length == 0 || code_block.ht_refinement_length == 0 {
        return Err("CUDA HTJ2K multi-pass packet contribution requires segment lengths");
    }
    if code_block
        .ht_cleanup_length
        .checked_add(code_block.ht_refinement_length)
        .ok_or("CUDA HTJ2K multi-pass packet contribution length overflow")?
        != data_len
    {
        return Err("CUDA HTJ2K multi-pass packet contribution length mismatch");
    }
    if !(2..65535).contains(&code_block.ht_cleanup_length) {
        return Err("CUDA HTJ2K cleanup segment length is out of range");
    }
    if code_block.ht_refinement_length >= 2047 {
        return Err("CUDA HTJ2K refinement segment length is out of range");
    }

    Ok((
        code_block.ht_cleanup_length,
        code_block.ht_refinement_length,
    ))
}

fn ht_cleanup_length_bits(l_block: u32, num_coding_passes: u8) -> u32 {
    let placeholder_groups = u32::from(num_coding_passes.saturating_sub(1)) / 3;
    let placeholder_passes = placeholder_groups * 3;
    l_block + (placeholder_passes + 1).ilog2()
}

fn value_fits_in_bits(value: u32, bits: u32) -> bool {
    bits >= u32::BITS || value < (1u32 << bits)
}

impl J2kEncodeStageAccelerator for CudaEncodeStageAccelerator {
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        J2kEncodeDispatchReport {
            deinterleave: self.deinterleave_dispatches,
            forward_rct: self.forward_rct_dispatches,
            forward_ict: self.forward_ict_dispatches,
            forward_dwt53: self.forward_dwt53_dispatches,
            forward_dwt97: self.forward_dwt97_dispatches,
            quantize_subband: self.quantize_subband_dispatches,
            tier1_code_block: self.tier1_code_block_dispatches,
            ht_code_block: self.ht_code_block_dispatches,
            packetization: self.packetization_dispatches,
        }
    }

    fn encode_deinterleave(
        &mut self,
        job: J2kDeinterleaveToF32Job<'_>,
    ) -> core::result::Result<Option<Vec<Vec<f32>>>, &'static str> {
        self.deinterleave_attempts = self.deinterleave_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (output, elapsed_us) = time_cuda_stage(
                "signinum.j2k.cuda.encode.deinterleave",
                &context,
                self.collect_profile,
                || {
                    context.j2k_deinterleave_to_f32(
                        job.pixels,
                        job.num_pixels,
                        job.num_components,
                        job.bit_depth,
                        job.signed,
                    )
                },
            )
            .map_err(|_| "CUDA deinterleave encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.deinterleave_dispatches = self.deinterleave_dispatches.saturating_add(dispatches);
            self.deinterleave_us = self.deinterleave_us.saturating_add(elapsed_us);
            if profile::gpu_route_profile_enabled() {
                let pixels_s = job.num_pixels.to_string();
                let components_s = job.num_components.to_string();
                let dispatches_s = dispatches.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_deinterleave"),
                        ("decision", "cuda_dispatch"),
                        ("pixels", pixels_s.as_str()),
                        ("components", components_s.as_str()),
                        ("dispatches", dispatches_s.as_str()),
                    ],
                );
            }
            return Ok(Some(output.into_components()));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_deinterleave"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_forward_rct(
        &mut self,
        job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (execution, elapsed_us) = time_cuda_stage(
                "signinum.j2k.cuda.encode.rct",
                &context,
                self.collect_profile,
                || context.j2k_forward_rct(job.plane0, job.plane1, job.plane2),
            )
            .map_err(|_| "CUDA forward RCT encode kernel failed")?;
            self.forward_rct_dispatches = self
                .forward_rct_dispatches
                .saturating_add(execution.kernel_dispatches());
            self.mct_us = self.mct_us.saturating_add(elapsed_us);
            if profile::gpu_route_profile_enabled() {
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_forward_rct"),
                        ("decision", "cuda_dispatch"),
                        ("dispatches", "1"),
                    ],
                );
            }
            return Ok(true);
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_forward_rct"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(false)
    }

    fn encode_forward_ict(
        &mut self,
        job: J2kForwardIctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.forward_ict_attempts = self.forward_ict_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (execution, elapsed_us) = time_cuda_stage(
                "signinum.j2k.cuda.encode.ict",
                &context,
                self.collect_profile,
                || context.j2k_forward_ict(job.plane0, job.plane1, job.plane2),
            )
            .map_err(|_| "CUDA forward ICT encode kernel failed")?;
            self.forward_ict_dispatches = self
                .forward_ict_dispatches
                .saturating_add(execution.kernel_dispatches());
            self.mct_us = self.mct_us.saturating_add(elapsed_us);
            if profile::gpu_route_profile_enabled() {
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_forward_ict"),
                        ("decision", "cuda_dispatch"),
                        ("dispatches", "1"),
                    ],
                );
            }
            return Ok(true);
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_forward_ict"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(false)
    }

    fn encode_forward_dwt53(
        &mut self,
        job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        self.forward_dwt53_attempts = self.forward_dwt53_attempts.saturating_add(1);
        if job.num_levels == 0 {
            if profile::gpu_route_profile_enabled() {
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_forward_dwt53"),
                        ("decision", "cpu_fallback"),
                        ("reason", "zero_levels"),
                    ],
                );
            }
            return Ok(None);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (output, elapsed_us) = time_cuda_stage(
                "signinum.j2k.cuda.encode.dwt53",
                &context,
                self.collect_profile,
                || context.j2k_forward_dwt53(job.samples, job.width, job.height, job.num_levels),
            )
            .map_err(|_| "CUDA forward 5/3 DWT encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.forward_dwt53_dispatches =
                self.forward_dwt53_dispatches.saturating_add(dispatches);
            self.dwt_us = self.dwt_us.saturating_add(elapsed_us);
            if profile::gpu_route_profile_enabled() {
                let width_s = job.width.to_string();
                let height_s = job.height.to_string();
                let levels_s = job.num_levels.to_string();
                let dispatches_s = dispatches.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_forward_dwt53"),
                        ("decision", "cuda_dispatch"),
                        ("width", width_s.as_str()),
                        ("height", height_s.as_str()),
                        ("levels", levels_s.as_str()),
                        ("dispatches", dispatches_s.as_str()),
                    ],
                );
            }
            return Ok(Some(cuda_dwt53_output_to_j2k(&output)?));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_forward_dwt53"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_forward_dwt97(
        &mut self,
        job: J2kForwardDwt97Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt97Output>, &'static str> {
        self.forward_dwt97_attempts = self.forward_dwt97_attempts.saturating_add(1);
        if job.num_levels == 0 {
            if profile::gpu_route_profile_enabled() {
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_forward_dwt97"),
                        ("decision", "cpu_fallback"),
                        ("reason", "zero_levels"),
                    ],
                );
            }
            return Ok(None);
        }
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (output, elapsed_us) = time_cuda_stage(
                "signinum.j2k.cuda.encode.dwt97",
                &context,
                self.collect_profile,
                || context.j2k_forward_dwt97(job.samples, job.width, job.height, job.num_levels),
            )
            .map_err(|_| "CUDA forward 9/7 DWT encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.forward_dwt97_dispatches =
                self.forward_dwt97_dispatches.saturating_add(dispatches);
            self.dwt_us = self.dwt_us.saturating_add(elapsed_us);
            if profile::gpu_route_profile_enabled() {
                let width_s = job.width.to_string();
                let height_s = job.height.to_string();
                let levels_s = job.num_levels.to_string();
                let dispatches_s = dispatches.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_forward_dwt97"),
                        ("decision", "cuda_dispatch"),
                        ("width", width_s.as_str()),
                        ("height", height_s.as_str()),
                        ("levels", levels_s.as_str()),
                        ("dispatches", dispatches_s.as_str()),
                    ],
                );
            }
            return Ok(Some(cuda_dwt97_output_to_j2k(&output)?));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_forward_dwt97"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_quantize_subband(
        &mut self,
        job: J2kQuantizeSubbandJob<'_>,
    ) -> core::result::Result<Option<Vec<i32>>, &'static str> {
        self.quantize_subband_attempts = self.quantize_subband_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let (output, elapsed_us) = time_cuda_stage(
                "signinum.j2k.cuda.encode.quantize",
                &context,
                self.collect_profile,
                || {
                    context.j2k_quantize_subband(
                        job.coefficients,
                        CudaJ2kQuantizeJob {
                            step_exponent: job.step_exponent,
                            step_mantissa: job.step_mantissa,
                            range_bits: job.range_bits,
                            reversible: job.reversible,
                        },
                    )
                },
            )
            .map_err(|_| "CUDA quantize subband encode kernel failed")?;
            let dispatches = output.execution().kernel_dispatches();
            self.quantize_subband_dispatches =
                self.quantize_subband_dispatches.saturating_add(dispatches);
            self.quantize_us = self.quantize_us.saturating_add(elapsed_us);
            if profile::gpu_route_profile_enabled() {
                let samples_s = job.coefficients.len().to_string();
                let dispatches_s = dispatches.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_quantize_subband"),
                        ("decision", "cuda_dispatch"),
                        ("samples", samples_s.as_str()),
                        ("dispatches", dispatches_s.as_str()),
                    ],
                );
            }
            return Ok(Some(output.coefficients().to_vec()));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_quantize_subband"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_tier1_code_block(
        &mut self,
        _job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        self.tier1_code_block_attempts = self.tier1_code_block_attempts.saturating_add(1);
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_tier1_code_block"),
                    ("decision", "cpu_fallback"),
                    ("reason", "unsupported_stage"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_ht_code_block(
        &mut self,
        job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let resources = self.cuda_encode_resources(&context)?;
            let encoded = cuda_encode_ht_code_block(&context, resources.as_ref(), job)?;
            let dispatches = encoded.execution().kernel_dispatches();
            let ht_encode_us = encoded.stage_timings().ht_encode_us;
            let mut outputs = encoded_ht_code_blocks_from_cuda(&encoded);
            let output = outputs
                .pop()
                .ok_or("CUDA HTJ2K code-block encode returned no output")?;
            self.ht_code_block_dispatches =
                self.ht_code_block_dispatches.saturating_add(dispatches);
            if self.collect_profile {
                self.ht_encode_us = self.ht_encode_us.saturating_add(ht_encode_us);
            }
            if profile::gpu_route_profile_enabled() {
                let width_s = job.width.to_string();
                let height_s = job.height.to_string();
                let dispatches_s = dispatches.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_ht_code_block"),
                        ("decision", "cuda_dispatch"),
                        ("width", width_s.as_str()),
                        ("height", height_s.as_str()),
                        ("dispatches", dispatches_s.as_str()),
                    ],
                );
            }
            return Ok(Some(output));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_ht_code_block"),
                    ("decision", "cpu_fallback"),
                    ("reason", "unsupported_stage"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(jobs.len());
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let resources = self.cuda_encode_resources(&context)?;
            let encoded = cuda_encode_ht_code_blocks(&context, resources.as_ref(), jobs)?;
            let dispatches = encoded.execution().kernel_dispatches();
            let ht_encode_us = encoded.stage_timings().ht_encode_us;
            let outputs = encoded_ht_code_blocks_from_cuda(&encoded);
            self.ht_code_block_dispatches =
                self.ht_code_block_dispatches.saturating_add(dispatches);
            if self.collect_profile {
                self.ht_encode_us = self.ht_encode_us.saturating_add(ht_encode_us);
            }
            if profile::gpu_route_profile_enabled() {
                let jobs_s = jobs.len().to_string();
                let dispatches_s = dispatches.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_ht_code_blocks"),
                        ("decision", "cuda_dispatch"),
                        ("jobs", jobs_s.as_str()),
                        ("dispatches", dispatches_s.as_str()),
                    ],
                );
            }
            return Ok(Some(outputs));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = jobs;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_ht_code_blocks"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_htj2k_tile(
        &mut self,
        job: J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.htj2k_tile_attempts = self.htj2k_tile_attempts.saturating_add(1);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let resources = self.cuda_encode_resources(&context)?;
            let Some(encoded) = cuda_encode_htj2k_tile_body(
                &context,
                resources.as_ref(),
                job,
                self.collect_profile,
            )?
            else {
                return Ok(None);
            };
            self.htj2k_tile_dispatches = self.htj2k_tile_dispatches.saturating_add(1);
            self.deinterleave_attempts = self.deinterleave_attempts.saturating_add(1);
            self.deinterleave_dispatches = self
                .deinterleave_dispatches
                .saturating_add(encoded.deinterleave_dispatches);
            if job.use_mct {
                if job.reversible {
                    self.forward_rct_attempts = self.forward_rct_attempts.saturating_add(1);
                } else {
                    self.forward_ict_attempts = self.forward_ict_attempts.saturating_add(1);
                }
            }
            self.forward_rct_dispatches = self
                .forward_rct_dispatches
                .saturating_add(encoded.forward_rct_dispatches);
            self.forward_ict_dispatches = self
                .forward_ict_dispatches
                .saturating_add(encoded.forward_ict_dispatches);
            if job.num_decomposition_levels > 0 {
                if job.reversible {
                    self.forward_dwt53_attempts = self
                        .forward_dwt53_attempts
                        .saturating_add(usize::from(job.num_components));
                } else {
                    self.forward_dwt97_attempts = self
                        .forward_dwt97_attempts
                        .saturating_add(usize::from(job.num_components));
                }
            }
            self.forward_dwt53_dispatches = self
                .forward_dwt53_dispatches
                .saturating_add(encoded.forward_dwt53_dispatches);
            self.forward_dwt97_dispatches = self
                .forward_dwt97_dispatches
                .saturating_add(encoded.forward_dwt97_dispatches);
            self.quantize_subband_attempts = self
                .quantize_subband_attempts
                .saturating_add(encoded.quantize_jobs);
            self.quantize_subband_dispatches = self
                .quantize_subband_dispatches
                .saturating_add(encoded.quantize_dispatches);
            self.ht_code_block_attempts = self
                .ht_code_block_attempts
                .saturating_add(encoded.ht_code_block_jobs);
            self.ht_code_block_dispatches = self
                .ht_code_block_dispatches
                .saturating_add(encoded.ht_code_block_dispatches);
            self.packetization_attempts = self.packetization_attempts.saturating_add(1);
            self.packetization_dispatches = self
                .packetization_dispatches
                .saturating_add(encoded.packetization_dispatches);
            if self.collect_profile {
                self.deinterleave_us = self
                    .deinterleave_us
                    .saturating_add(encoded.timings.deinterleave_us);
                self.mct_us = self.mct_us.saturating_add(encoded.timings.mct_us);
                self.dwt_us = self.dwt_us.saturating_add(encoded.timings.dwt_us);
                self.quantize_us = self.quantize_us.saturating_add(encoded.timings.quantize_us);
                self.ht_encode_us = self
                    .ht_encode_us
                    .saturating_add(encoded.timings.ht_encode_us);
                self.packetize_us = self
                    .packetize_us
                    .saturating_add(encoded.timings.packetize_us);
            }
            if profile::gpu_route_profile_enabled() {
                let components_s = job.num_components.to_string();
                let blocks_s = encoded.ht_code_block_jobs.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_htj2k_tile"),
                        ("decision", "cuda_dispatch"),
                        ("components", components_s.as_str()),
                        ("blocks", blocks_s.as_str()),
                    ],
                );
            }
            return Ok(Some(encoded.tile_data));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_htj2k_tile"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_ht_subband(
        &mut self,
        job: J2kHtSubbandEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        let code_block_count = ht_subband_code_block_count(job)?;
        self.ht_subband_attempts = self.ht_subband_attempts.saturating_add(1);
        self.quantize_subband_attempts = self.quantize_subband_attempts.saturating_add(1);
        self.ht_code_block_attempts = self.ht_code_block_attempts.saturating_add(code_block_count);
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let resources = self.cuda_encode_resources(&context)?;
            let encoded =
                cuda_encode_ht_subband(&context, resources.as_ref(), job, self.collect_profile)?;
            let quantize_dispatches = encoded.quantize_dispatches;
            let encode_dispatches = encoded.encode.execution().kernel_dispatches();
            let outputs = encoded_ht_code_blocks_from_cuda(&encoded.encode);
            self.ht_subband_dispatches = self.ht_subband_dispatches.saturating_add(1);
            self.quantize_subband_dispatches = self
                .quantize_subband_dispatches
                .saturating_add(quantize_dispatches);
            self.ht_code_block_dispatches = self
                .ht_code_block_dispatches
                .saturating_add(encode_dispatches);
            if self.collect_profile {
                self.quantize_us = self.quantize_us.saturating_add(encoded.timings.quantize_us);
                self.ht_encode_us = self
                    .ht_encode_us
                    .saturating_add(encoded.timings.ht_encode_us);
            }
            if profile::gpu_route_profile_enabled() {
                let width_s = job.width.to_string();
                let height_s = job.height.to_string();
                let blocks_s = code_block_count.to_string();
                let quantize_dispatches_s = quantize_dispatches.to_string();
                let encode_dispatches_s = encode_dispatches.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_ht_subband"),
                        ("decision", "cuda_dispatch"),
                        ("width", width_s.as_str()),
                        ("height", height_s.as_str()),
                        ("blocks", blocks_s.as_str()),
                        ("quantize_dispatches", quantize_dispatches_s.as_str()),
                        ("encode_dispatches", encode_dispatches_s.as_str()),
                    ],
                );
            }
            return Ok(Some(outputs));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = job;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_ht_subband"),
                    ("decision", "cpu_fallback"),
                    ("reason", "cuda_unavailable"),
                ],
            );
        }
        Ok(None)
    }

    fn encode_packetization(
        &mut self,
        job: J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.packetization_attempts = self.packetization_attempts.saturating_add(1);
        let plan = match flatten_cuda_htj2k_packetization_job(job) {
            Ok(plan) => plan,
            Err(reason) => {
                if profile::gpu_route_profile_enabled() {
                    profile::emit_gpu_route_profile(
                        "j2k",
                        "gpu_route",
                        "cuda",
                        &[
                            ("op", "encode_packetization"),
                            ("decision", "cpu_fallback"),
                            ("reason", reason),
                        ],
                    );
                }
                return Ok(None);
            }
        };
        #[cfg(feature = "cuda-runtime")]
        if let Some(context) = self.cuda_context()? {
            let packets = cuda_packetization_packets(&plan);
            let subbands = cuda_packetization_subbands(&plan);
            let blocks = cuda_packetization_blocks(&plan);
            let tag_states = cuda_packetization_tag_states(&plan);
            let tag_nodes = cuda_packetization_tag_nodes(&plan);
            let packetized = context
                .packetize_htj2k_cleanup_packets_with_tag_state(
                    &plan.payload,
                    &packets,
                    &subbands,
                    &blocks,
                    &tag_states,
                    &tag_nodes,
                )
                .map_err(|_| "CUDA HTJ2K packetization kernel failed")?;
            let dispatches = packetized.execution().kernel_dispatches();
            let packetize_us = packetized.stage_timings().packetize_us;
            self.packetization_dispatches =
                self.packetization_dispatches.saturating_add(dispatches);
            if self.collect_profile {
                self.packetize_us = self.packetize_us.saturating_add(packetize_us);
            }
            if profile::gpu_route_profile_enabled() {
                let packets_s = packets.len().to_string();
                let dispatches_s = dispatches.to_string();
                profile::emit_gpu_route_profile(
                    "j2k",
                    "gpu_route",
                    "cuda",
                    &[
                        ("op", "encode_packetization"),
                        ("decision", "cuda_dispatch"),
                        ("packets", packets_s.as_str()),
                        ("dispatches", dispatches_s.as_str()),
                    ],
                );
            }
            return Ok(Some(packetized.data().to_vec()));
        }
        #[cfg(not(feature = "cuda-runtime"))]
        let _ = plan;
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "encode_packetization"),
                    ("decision", "cpu_fallback"),
                    ("reason", "unsupported_stage"),
                ],
            );
        }
        Ok(None)
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_packetization_packets(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationPacket> {
    plan.packets
        .iter()
        .map(|packet| CudaHtj2kPacketizationPacket {
            block_start: packet.block_start,
            block_count: packet.block_count,
            subband_start: packet.subband_start,
            subband_count: packet.subband_count,
            output_capacity: packet.output_capacity,
            layer: packet.layer,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn cuda_packetization_subbands(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationSubband> {
    plan.subbands
        .iter()
        .map(|subband| CudaHtj2kPacketizationSubband {
            block_start: subband.block_start,
            block_count: subband.block_count,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn cuda_packetization_blocks(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationBlock> {
    plan.blocks
        .iter()
        .map(|block| CudaHtj2kPacketizationBlock {
            data_offset: block.data_offset,
            data_len: block.data_len,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            num_coding_passes: block.num_coding_passes,
            num_zero_bitplanes: block.num_zero_bitplanes,
            l_block: block.l_block,
            previously_included: block.previously_included,
            inclusion_layer: block.inclusion_layer,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn cuda_packetization_tag_states(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationSubbandTagState> {
    plan.tag_states
        .iter()
        .map(|state| CudaHtj2kPacketizationSubbandTagState {
            inclusion_node_start: state.inclusion_node_start,
            zero_bitplane_node_start: state.zero_bitplane_node_start,
            node_count: state.node_count,
            reserved0: 0,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn cuda_packetization_tag_nodes(
    plan: &CudaHtj2kPacketizationPlan,
) -> Vec<CudaHtj2kPacketizationTagNodeState> {
    plan.tag_nodes
        .iter()
        .map(|node| CudaHtj2kPacketizationTagNodeState {
            current: node.current,
            known: node.known,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn cuda_encode_ht_code_block(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    job: J2kHtCodeBlockEncodeJob<'_>,
) -> core::result::Result<signinum_cuda_runtime::CudaHtj2kEncodedCodeBlocks, &'static str> {
    let coefficient_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or("CUDA HTJ2K code-block encode job is too large")?;
    if coefficient_len != job.coefficients.len() {
        return Err("CUDA HTJ2K code-block encode job has invalid coefficient length");
    }
    let cuda_jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: job.width,
        height: job.height,
        total_bitplanes: job.total_bitplanes,
    }];
    context
        .encode_htj2k_codeblocks_with_resources(job.coefficients, &cuda_jobs, resources)
        .map_err(|_| "CUDA HTJ2K code-block encode kernel failed")
}

#[cfg(feature = "cuda-runtime")]
fn cuda_encode_ht_code_blocks(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> core::result::Result<signinum_cuda_runtime::CudaHtj2kEncodedCodeBlocks, &'static str> {
    let total_coefficients = jobs.iter().try_fold(0usize, |acc, job| {
        let coefficient_len = (job.width as usize)
            .checked_mul(job.height as usize)
            .ok_or("CUDA HTJ2K code-block batch is too large")?;
        if coefficient_len != job.coefficients.len() {
            return Err("CUDA HTJ2K code-block encode job has invalid coefficient length");
        }
        acc.checked_add(coefficient_len)
            .ok_or("CUDA HTJ2K code-block batch is too large")
    })?;
    let mut coefficients = Vec::with_capacity(total_coefficients);
    let mut cuda_jobs = Vec::with_capacity(jobs.len());
    for job in jobs {
        let coefficient_offset = u32::try_from(coefficients.len())
            .map_err(|_| "CUDA HTJ2K code-block batch is too large")?;
        coefficients.extend_from_slice(job.coefficients);
        cuda_jobs.push(CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset,
            width: job.width,
            height: job.height,
            total_bitplanes: job.total_bitplanes,
        });
    }

    context
        .encode_htj2k_codeblocks_with_resources(&coefficients, &cuda_jobs, resources)
        .map_err(|_| "CUDA HTJ2K code-block batch encode kernel failed")
}

#[cfg(feature = "cuda-runtime")]
struct CudaEncodedHtj2kTile {
    tile_data: Vec<u8>,
    deinterleave_dispatches: usize,
    forward_rct_dispatches: usize,
    forward_ict_dispatches: usize,
    forward_dwt53_dispatches: usize,
    forward_dwt97_dispatches: usize,
    quantize_jobs: usize,
    quantize_dispatches: usize,
    ht_code_block_dispatches: usize,
    ht_code_block_jobs: usize,
    packetization_dispatches: usize,
    timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Default)]
struct CudaHtj2kTileEncodeStats {
    collect_profile: bool,
    deinterleave_dispatches: usize,
    forward_rct_dispatches: usize,
    forward_ict_dispatches: usize,
    forward_dwt53_dispatches: usize,
    forward_dwt97_dispatches: usize,
    quantize_jobs: usize,
    quantize_dispatches: usize,
    ht_code_block_dispatches: usize,
    ht_code_block_jobs: usize,
    timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
struct CudaEncodedHtj2kResolution {
    subbands: Vec<CudaEncodedHtj2kSubband>,
}

#[cfg(feature = "cuda-runtime")]
struct CudaEncodedHtj2kSubband {
    code_blocks: Vec<EncodedHtJ2kCodeBlock>,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy)]
struct CudaTileSubbandRegion {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    stride: u32,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy)]
enum CudaTileSubbandKind {
    LowLow,
    HighLow,
    LowHigh,
    HighHigh,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy)]
struct CudaHtj2kEncodeRuntime<'a> {
    context: &'a CudaContext,
    resources: &'a CudaHtj2kEncodeResources,
}

#[cfg(feature = "cuda-runtime")]
fn cuda_encode_htj2k_tile_body(
    context: &CudaContext,
    encode_resources: &CudaHtj2kEncodeResources,
    job: J2kHtj2kTileEncodeJob<'_>,
    collect_profile: bool,
) -> core::result::Result<Option<CudaEncodedHtj2kTile>, &'static str> {
    if job
        .component_sampling
        .iter()
        .any(|&sampling| sampling != (1, 1))
    {
        return Ok(None);
    }
    if job.use_mct && job.num_components != 3 {
        return Ok(None);
    }
    if job.code_block_width == 0 || job.code_block_height == 0 {
        return Err("CUDA HTJ2K tile encode job has invalid code-block dimensions");
    }
    let expected_quantization_steps = 1usize
        .checked_add(usize::from(job.num_decomposition_levels).saturating_mul(3))
        .ok_or("CUDA HTJ2K tile quantization step count overflow")?;
    if job.quantization_steps.len() != expected_quantization_steps {
        return Err("CUDA HTJ2K tile quantization step count mismatch");
    }

    let num_pixels = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or("CUDA HTJ2K tile dimensions are too large")?;
    let (mut components, deinterleave_us) = time_cuda_stage(
        "signinum.htj2k.encode.tile.deinterleave",
        context,
        collect_profile,
        || {
            context.j2k_deinterleave_to_f32_resident(
                job.pixels,
                num_pixels,
                job.num_components,
                job.bit_depth,
                job.signed,
            )
        },
    )
    .map_err(|_| "CUDA HTJ2K tile deinterleave failed")?;
    let mut stats = CudaHtj2kTileEncodeStats {
        collect_profile,
        deinterleave_dispatches: components.execution().kernel_dispatches(),
        timings: CudaEncodeStageTimings {
            deinterleave_us,
            ..CudaEncodeStageTimings::default()
        },
        ..CudaHtj2kTileEncodeStats::default()
    };
    let runtime = CudaHtj2kEncodeRuntime {
        context,
        resources: encode_resources,
    };

    if job.use_mct {
        let (execution, mct_us) = if job.reversible {
            time_cuda_stage(
                "signinum.htj2k.encode.tile.rct",
                context,
                collect_profile,
                || context.j2k_forward_rct_resident(&mut components),
            )
            .map_err(|_| "CUDA HTJ2K tile RCT failed")?
        } else {
            time_cuda_stage(
                "signinum.htj2k.encode.tile.ict",
                context,
                collect_profile,
                || context.j2k_forward_ict_resident(&mut components),
            )
            .map_err(|_| "CUDA HTJ2K tile ICT failed")?
        };
        stats.timings.mct_us = stats.timings.mct_us.saturating_add(mct_us);
        if job.reversible {
            stats.forward_rct_dispatches = execution.kernel_dispatches();
        } else {
            stats.forward_ict_dispatches = execution.kernel_dispatches();
        }
    }

    let mut component_resolution_packets = Vec::with_capacity(usize::from(job.num_components));
    if job.num_decomposition_levels == 0 {
        for component in 0..job.num_components {
            let y0 = u32::from(component)
                .checked_mul(job.height)
                .ok_or("CUDA HTJ2K tile component offset overflow")?;
            let subband = cuda_encode_tile_subband_region(
                runtime,
                components.buffer(),
                CudaTileSubbandRegion {
                    x0: 0,
                    y0,
                    width: job.width,
                    height: job.height,
                    stride: job.width,
                },
                job.quantization_steps[0],
                job,
                CudaTileSubbandKind::LowLow,
                &mut stats,
            )?;
            component_resolution_packets.push(vec![CudaEncodedHtj2kResolution {
                subbands: vec![subband],
            }]);
        }
    } else {
        for component in 0..job.num_components {
            let packets = if job.reversible {
                let (dwt, dwt_us) = time_cuda_stage(
                    "signinum.htj2k.encode.tile.dwt53",
                    context,
                    collect_profile,
                    || {
                        context.j2k_forward_dwt53_resident_component(
                            &components,
                            component,
                            job.width,
                            job.height,
                            job.num_decomposition_levels,
                        )
                    },
                )
                .map_err(|_| "CUDA HTJ2K tile DWT 5/3 failed")?;
                stats.forward_dwt53_dispatches = stats
                    .forward_dwt53_dispatches
                    .saturating_add(dwt.execution().kernel_dispatches());
                stats.timings.dwt_us = stats.timings.dwt_us.saturating_add(dwt_us);
                cuda_encode_dwt_component_packets(
                    runtime,
                    job,
                    dwt.buffer(),
                    dwt.levels(),
                    dwt.ll_dimensions(),
                    &mut stats,
                )?
            } else {
                let (dwt, dwt_us) = time_cuda_stage(
                    "signinum.htj2k.encode.tile.dwt97",
                    context,
                    collect_profile,
                    || {
                        context.j2k_forward_dwt97_resident_component(
                            &components,
                            component,
                            job.width,
                            job.height,
                            job.num_decomposition_levels,
                        )
                    },
                )
                .map_err(|_| "CUDA HTJ2K tile DWT 9/7 failed")?;
                stats.forward_dwt97_dispatches = stats
                    .forward_dwt97_dispatches
                    .saturating_add(dwt.execution().kernel_dispatches());
                stats.timings.dwt_us = stats.timings.dwt_us.saturating_add(dwt_us);
                cuda_encode_dwt_component_packets(
                    runtime,
                    job,
                    dwt.buffer(),
                    dwt.levels(),
                    dwt.ll_dimensions(),
                    &mut stats,
                )?
            };
            component_resolution_packets.push(packets);
        }
    }

    let resolution_packets =
        cuda_order_component_resolution_packets(component_resolution_packets, job.num_components)?;
    let (tile_data, packetization_dispatches, packetize_us) =
        cuda_packetize_tile_body(context, job, &resolution_packets, stats.ht_code_block_jobs)?;
    stats.timings.packetize_us = stats.timings.packetize_us.saturating_add(packetize_us);
    Ok(Some(CudaEncodedHtj2kTile {
        tile_data,
        deinterleave_dispatches: stats.deinterleave_dispatches,
        forward_rct_dispatches: stats.forward_rct_dispatches,
        forward_ict_dispatches: stats.forward_ict_dispatches,
        forward_dwt53_dispatches: stats.forward_dwt53_dispatches,
        forward_dwt97_dispatches: stats.forward_dwt97_dispatches,
        quantize_jobs: stats.quantize_jobs,
        quantize_dispatches: stats.quantize_dispatches,
        ht_code_block_dispatches: stats.ht_code_block_dispatches,
        ht_code_block_jobs: stats.ht_code_block_jobs,
        packetization_dispatches,
        timings: stats.timings,
    }))
}

#[cfg(feature = "cuda-runtime")]
fn cuda_encode_dwt_component_packets(
    runtime: CudaHtj2kEncodeRuntime<'_>,
    job: J2kHtj2kTileEncodeJob<'_>,
    transformed: &CudaDeviceBuffer,
    levels: &[CudaDwt53LevelShape],
    ll_dimensions: (u32, u32),
    stats: &mut CudaHtj2kTileEncodeStats,
) -> core::result::Result<Vec<CudaEncodedHtj2kResolution>, &'static str> {
    if levels.len() != usize::from(job.num_decomposition_levels) {
        return Err("CUDA HTJ2K tile DWT level count mismatch");
    }
    let (ll_width, ll_height) = ll_dimensions;
    let full_width = levels.first().map_or(ll_width, |level| level.width);
    let mut packets = Vec::with_capacity(levels.len().saturating_add(1));

    let ll_subband = cuda_encode_tile_subband_region(
        runtime,
        transformed,
        CudaTileSubbandRegion {
            x0: 0,
            y0: 0,
            width: ll_width,
            height: ll_height,
            stride: full_width,
        },
        job.quantization_steps[0],
        job,
        CudaTileSubbandKind::LowLow,
        stats,
    )?;
    packets.push(CudaEncodedHtj2kResolution {
        subbands: vec![ll_subband],
    });

    for (level_idx, level) in levels.iter().rev().enumerate() {
        let step_base = 1usize
            .checked_add(level_idx.saturating_mul(3))
            .ok_or("CUDA HTJ2K tile quantization step index overflow")?;
        let hl = cuda_encode_tile_subband_region(
            runtime,
            transformed,
            CudaTileSubbandRegion {
                x0: level.low_width,
                y0: 0,
                width: level.high_width,
                height: level.low_height,
                stride: full_width,
            },
            job.quantization_steps[step_base],
            job,
            CudaTileSubbandKind::HighLow,
            stats,
        )?;
        let lh = cuda_encode_tile_subband_region(
            runtime,
            transformed,
            CudaTileSubbandRegion {
                x0: 0,
                y0: level.low_height,
                width: level.low_width,
                height: level.high_height,
                stride: full_width,
            },
            job.quantization_steps[step_base + 1],
            job,
            CudaTileSubbandKind::LowHigh,
            stats,
        )?;
        let hh = cuda_encode_tile_subband_region(
            runtime,
            transformed,
            CudaTileSubbandRegion {
                x0: level.low_width,
                y0: level.low_height,
                width: level.high_width,
                height: level.high_height,
                stride: full_width,
            },
            job.quantization_steps[step_base + 2],
            job,
            CudaTileSubbandKind::HighHigh,
            stats,
        )?;
        packets.push(CudaEncodedHtj2kResolution {
            subbands: vec![hl, lh, hh],
        });
    }

    Ok(packets)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_encode_tile_subband_region(
    runtime: CudaHtj2kEncodeRuntime<'_>,
    source: &CudaDeviceBuffer,
    region: CudaTileSubbandRegion,
    quantization_step: (u16, u16),
    job: J2kHtj2kTileEncodeJob<'_>,
    subband_kind: CudaTileSubbandKind,
    stats: &mut CudaHtj2kTileEncodeStats,
) -> core::result::Result<CudaEncodedHtj2kSubband, &'static str> {
    if region.width == 0 || region.height == 0 {
        return Ok(CudaEncodedHtj2kSubband {
            code_blocks: Vec::new(),
            num_cbs_x: 0,
            num_cbs_y: 0,
        });
    }

    let (step_exponent, step_mantissa) = quantization_step;
    let step_exponent_u8 = u8::try_from(step_exponent)
        .map_err(|_| "CUDA HTJ2K tile quantization exponent exceeds u8")?;
    let total_bitplanes = job
        .guard_bits
        .saturating_add(step_exponent_u8)
        .saturating_sub(1);
    let (quantized, quantize_us) = time_cuda_stage(
        "signinum.htj2k.encode.tile.quantize",
        runtime.context,
        stats.collect_profile,
        || {
            runtime.context.j2k_quantize_subband_region_resident(
                source,
                CudaJ2kQuantizeSubbandRegionJob {
                    x0: region.x0,
                    y0: region.y0,
                    width: region.width,
                    height: region.height,
                    stride: region.stride,
                    quantization: CudaJ2kQuantizeJob {
                        step_exponent,
                        step_mantissa,
                        range_bits: cuda_tile_subband_range_bits(job.bit_depth, subband_kind),
                        reversible: job.reversible,
                    },
                },
            )
        },
    )
    .map_err(|_| "CUDA HTJ2K tile quantize failed")?;
    stats.quantize_jobs = stats.quantize_jobs.saturating_add(1);
    stats.quantize_dispatches = stats
        .quantize_dispatches
        .saturating_add(quantized.execution().kernel_dispatches());
    stats.timings.quantize_us = stats.timings.quantize_us.saturating_add(quantize_us);

    let region_jobs = cuda_ht_region_jobs(
        region.width,
        region.height,
        job.code_block_width,
        job.code_block_height,
        total_bitplanes,
    )?;
    stats.ht_code_block_jobs = stats.ht_code_block_jobs.saturating_add(region_jobs.len());
    let encoded = runtime
        .context
        .encode_htj2k_codeblock_regions_resident_with_resources(
            quantized.buffer(),
            quantized.coefficient_count(),
            &region_jobs,
            runtime.resources,
        )
        .map_err(|_| "CUDA HTJ2K tile code-block encode failed")?;
    stats.ht_code_block_dispatches = stats
        .ht_code_block_dispatches
        .saturating_add(encoded.execution().kernel_dispatches());
    stats.timings.ht_encode_us = stats
        .timings
        .ht_encode_us
        .saturating_add(encoded.stage_timings().ht_encode_us);

    Ok(CudaEncodedHtj2kSubband {
        code_blocks: encoded_ht_code_blocks_from_cuda(&encoded),
        num_cbs_x: region.width.div_ceil(job.code_block_width),
        num_cbs_y: region.height.div_ceil(job.code_block_height),
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_tile_subband_range_bits(bit_depth: u8, subband_kind: CudaTileSubbandKind) -> u8 {
    let log_gain = match subband_kind {
        CudaTileSubbandKind::LowLow => 0,
        CudaTileSubbandKind::HighLow | CudaTileSubbandKind::LowHigh => 1,
        CudaTileSubbandKind::HighHigh => 2,
    };
    bit_depth.saturating_add(log_gain)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_order_component_resolution_packets(
    component_resolution_packets: Vec<Vec<CudaEncodedHtj2kResolution>>,
    num_components: u8,
) -> core::result::Result<Vec<CudaEncodedHtj2kResolution>, &'static str> {
    if component_resolution_packets.len() != usize::from(num_components) {
        return Err("CUDA HTJ2K tile component packet count mismatch");
    }
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, Vec::len);
    let mut component_iters: Vec<_> = component_resolution_packets
        .into_iter()
        .map(Vec::into_iter)
        .collect();
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_iters.len()));

    for _resolution in 0..resolution_count {
        for component in &mut component_iters {
            resolution_packets.push(
                component
                    .next()
                    .ok_or("CUDA HTJ2K tile component resolution count mismatch")?,
            );
        }
    }
    if component_iters
        .iter_mut()
        .any(|component| component.next().is_some())
    {
        return Err("CUDA HTJ2K tile component resolution count mismatch");
    }

    Ok(resolution_packets)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_ht_region_jobs(
    width: u32,
    height: u32,
    code_block_width: u32,
    code_block_height: u32,
    total_bitplanes: u8,
) -> core::result::Result<Vec<CudaHtj2kEncodeCodeBlockRegionJob>, &'static str> {
    if code_block_width == 0 || code_block_height == 0 {
        return Err("CUDA HTJ2K encode job has invalid code-block dimensions");
    }
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }

    let num_cbs_x = width.div_ceil(code_block_width);
    let num_cbs_y = height.div_ceil(code_block_height);
    let count = (num_cbs_x as usize)
        .checked_mul(num_cbs_y as usize)
        .ok_or("CUDA HTJ2K code-block count overflow")?;
    let mut cuda_jobs = Vec::with_capacity(count);
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx
                .checked_mul(code_block_width)
                .ok_or("CUDA HTJ2K code-block x offset overflow")?;
            let y0 = cby
                .checked_mul(code_block_height)
                .ok_or("CUDA HTJ2K code-block y offset overflow")?;
            let block_width = (x0 + code_block_width).min(width) - x0;
            let block_height = (y0 + code_block_height).min(height) - y0;
            let offset = (y0 as usize)
                .checked_mul(width as usize)
                .and_then(|row| row.checked_add(x0 as usize))
                .ok_or("CUDA HTJ2K code-block offset overflow")?;
            cuda_jobs.push(CudaHtj2kEncodeCodeBlockRegionJob {
                coefficient_offset: u32::try_from(offset)
                    .map_err(|_| "CUDA HTJ2K code-block offset exceeds u32")?,
                coefficient_stride: width,
                width: block_width,
                height: block_height,
                total_bitplanes,
            });
        }
    }
    Ok(cuda_jobs)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_packetize_tile_body(
    context: &CudaContext,
    job: J2kHtj2kTileEncodeJob<'_>,
    resolution_packets: &[CudaEncodedHtj2kResolution],
    code_block_count: usize,
) -> core::result::Result<(Vec<u8>, usize, u128), &'static str> {
    let packet_descriptors =
        cuda_tile_packet_descriptors(resolution_packets.len(), 1, job.num_components)?;
    let resolutions: Vec<J2kPacketizationResolution<'_>> = resolution_packets
        .iter()
        .map(|resolution| J2kPacketizationResolution {
            subbands: resolution
                .subbands
                .iter()
                .map(|subband| {
                    let code_blocks = subband
                        .code_blocks
                        .iter()
                        .map(|block| J2kPacketizationCodeBlock {
                            data: block.data.as_slice(),
                            ht_cleanup_length: block.cleanup_length,
                            ht_refinement_length: block.refinement_length,
                            num_coding_passes: block.num_coding_passes,
                            num_zero_bitplanes: block.num_zero_bitplanes,
                            previously_included: false,
                            l_block: 3,
                            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                        })
                        .collect();
                    J2kPacketizationSubband {
                        code_blocks,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    }
                })
                .collect(),
        })
        .collect();

    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: u32::try_from(resolutions.len())
            .map_err(|_| "CUDA HTJ2K tile resolution count exceeds u32")?,
        num_layers: 1,
        num_components: job.num_components,
        code_block_count: u32::try_from(code_block_count)
            .map_err(|_| "CUDA HTJ2K tile code-block count exceeds u32")?,
        progression_order: job.progression_order,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };
    let plan = flatten_cuda_htj2k_packetization_job(packetization_job)?;
    let packets = cuda_packetization_packets(&plan);
    let subbands = cuda_packetization_subbands(&plan);
    let blocks = cuda_packetization_blocks(&plan);
    let tag_states = cuda_packetization_tag_states(&plan);
    let tag_nodes = cuda_packetization_tag_nodes(&plan);
    let packetized = context
        .packetize_htj2k_cleanup_packets_with_tag_state(
            &plan.payload,
            &packets,
            &subbands,
            &blocks,
            &tag_states,
            &tag_nodes,
        )
        .map_err(|_| "CUDA HTJ2K tile packetization failed")?;
    Ok((
        packetized.data().to_vec(),
        packetized.execution().kernel_dispatches(),
        packetized.stage_timings().packetize_us,
    ))
}

#[cfg(feature = "cuda-runtime")]
fn cuda_tile_packet_descriptors(
    packet_count: usize,
    num_layers: u8,
    num_components: u8,
) -> core::result::Result<Vec<J2kPacketizationPacketDescriptor>, &'static str> {
    if num_layers != 1 {
        return Err("CUDA HTJ2K tile encode currently prepares one packet layer");
    }
    let component_count = usize::from(num_components).max(1);
    (0..packet_count)
        .map(|packet_index| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index)
                    .map_err(|_| "CUDA HTJ2K tile packet index exceeds u32")?,
                state_index: u32::try_from(packet_index)
                    .map_err(|_| "CUDA HTJ2K tile packet state index exceeds u32")?,
                layer: 0,
                resolution: u32::try_from(packet_index / component_count)
                    .map_err(|_| "CUDA HTJ2K tile packet resolution exceeds u32")?,
                component: u8::try_from(packet_index % component_count)
                    .map_err(|_| "CUDA HTJ2K tile packet component exceeds u8")?,
                precinct: 0,
            })
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
struct CudaEncodedHtSubband {
    quantize_dispatches: usize,
    encode: signinum_cuda_runtime::CudaHtj2kEncodedCodeBlocks,
    timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
fn cuda_encode_ht_subband(
    context: &CudaContext,
    encode_resources: &CudaHtj2kEncodeResources,
    job: J2kHtSubbandEncodeJob<'_>,
    collect_profile: bool,
) -> core::result::Result<CudaEncodedHtSubband, &'static str> {
    let expected_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or("CUDA HTJ2K subband encode dimensions are too large")?;
    if expected_len != job.coefficients.len() {
        return Err("CUDA HTJ2K subband encode job has invalid coefficient length");
    }
    if job.code_block_width == 0 || job.code_block_height == 0 {
        return Err("CUDA HTJ2K subband encode job has invalid code-block dimensions");
    }

    let sample_buffer = context
        .upload_f32_pinned(job.coefficients)
        .map_err(|_| "CUDA HTJ2K subband upload failed")?;
    let (quantized, quantize_us) = time_cuda_stage(
        "signinum.htj2k.encode.subband.quantize",
        context,
        collect_profile,
        || {
            context.j2k_quantize_subband_resident(
                &sample_buffer,
                job.coefficients.len(),
                CudaJ2kQuantizeJob {
                    step_exponent: job.step_exponent,
                    step_mantissa: job.step_mantissa,
                    range_bits: job.range_bits,
                    reversible: job.reversible,
                },
            )
        },
    )
    .map_err(|_| "CUDA quantize subband encode kernel failed")?;
    let cuda_jobs = cuda_ht_subband_region_jobs(job)?;
    let encoded = context
        .encode_htj2k_codeblock_regions_resident_with_resources(
            quantized.buffer(),
            quantized.coefficient_count(),
            &cuda_jobs,
            encode_resources,
        )
        .map_err(|_| "CUDA HTJ2K resident subband encode kernel failed")?;

    Ok(CudaEncodedHtSubband {
        quantize_dispatches: quantized.execution().kernel_dispatches(),
        timings: CudaEncodeStageTimings {
            quantize_us,
            ht_encode_us: encoded.stage_timings().ht_encode_us,
            ..CudaEncodeStageTimings::default()
        },
        encode: encoded,
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_ht_subband_region_jobs(
    job: J2kHtSubbandEncodeJob<'_>,
) -> core::result::Result<Vec<CudaHtj2kEncodeCodeBlockRegionJob>, &'static str> {
    cuda_ht_region_jobs(
        job.width,
        job.height,
        job.code_block_width,
        job.code_block_height,
        job.total_bitplanes,
    )
}

fn ht_subband_code_block_count(
    job: J2kHtSubbandEncodeJob<'_>,
) -> core::result::Result<usize, &'static str> {
    if job.code_block_width == 0 || job.code_block_height == 0 {
        return Err("CUDA HTJ2K subband encode job has invalid code-block dimensions");
    }
    let num_cbs_x = job.width.div_ceil(job.code_block_width);
    let num_cbs_y = job.height.div_ceil(job.code_block_height);
    (num_cbs_x as usize)
        .checked_mul(num_cbs_y as usize)
        .ok_or("CUDA HTJ2K subband code-block count overflow")
}

#[cfg(feature = "cuda-runtime")]
fn encoded_ht_code_block_from_cuda(
    encoded: &signinum_cuda_runtime::CudaHtj2kEncodedCodeBlock,
) -> EncodedHtJ2kCodeBlock {
    EncodedHtJ2kCodeBlock {
        data: encoded.data().to_vec(),
        cleanup_length: encoded.cleanup_length(),
        refinement_length: encoded.refinement_length(),
        num_coding_passes: encoded.num_coding_passes(),
        num_zero_bitplanes: encoded.num_zero_bitplanes(),
    }
}

#[cfg(feature = "cuda-runtime")]
fn encoded_ht_code_blocks_from_cuda(
    encoded: &signinum_cuda_runtime::CudaHtj2kEncodedCodeBlocks,
) -> Vec<EncodedHtJ2kCodeBlock> {
    encoded
        .code_blocks()
        .iter()
        .map(encoded_ht_code_block_from_cuda)
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn cuda_htj2k_encode_tables() -> CudaHtj2kEncodeTables<'static> {
    CudaHtj2kEncodeTables {
        vlc_table0: signinum_j2k_native::ht_vlc_encode_table0(),
        vlc_table1: signinum_j2k_native::ht_vlc_encode_table1(),
        uvlc_table: ht_uvlc_encode_table_bytes(),
    }
}

#[cfg(feature = "cuda-runtime")]
fn ht_uvlc_encode_table_bytes() -> &'static [u8] {
    static TABLE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| {
            signinum_j2k_native::ht_uvlc_encode_table()
                .iter()
                .flat_map(|entry| {
                    [
                        entry.pre,
                        entry.pre_len,
                        entry.suf,
                        entry.suf_len,
                        entry.ext,
                        entry.ext_len,
                    ]
                })
                .collect()
        })
        .as_slice()
}

#[cfg(feature = "cuda-runtime")]
fn cuda_dwt53_output_to_j2k(
    output: &CudaDwt53Output,
) -> core::result::Result<J2kForwardDwt53Output, &'static str> {
    let (ll_width, ll_height) = output.ll_dimensions();
    let transformed = output.transformed();
    let full_width = output
        .levels()
        .first()
        .map_or(ll_width, |level| level.width) as usize;
    let mut ll = Vec::with_capacity((ll_width as usize) * (ll_height as usize));
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width)
            .ok_or("CUDA DWT LL row offset overflow")?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    let mut levels = Vec::with_capacity(output.levels().len());
    for shape in output.levels() {
        levels.push(signinum_j2k_native::J2kForwardDwt53Level {
            hl: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                0,
                shape.high_width,
                shape.low_height,
            )?,
            lh: extract_cuda_subband(
                transformed,
                full_width,
                0,
                shape.low_height,
                shape.low_width,
                shape.high_height,
            )?,
            hh: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                shape.low_height,
                shape.high_width,
                shape.high_height,
            )?,
            width: shape.width,
            height: shape.height,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }
    levels.reverse();

    Ok(J2kForwardDwt53Output {
        ll,
        ll_width,
        ll_height,
        levels,
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_dwt97_output_to_j2k(
    output: &CudaDwt97Output,
) -> core::result::Result<J2kForwardDwt97Output, &'static str> {
    let (ll_width, ll_height) = output.ll_dimensions();
    let transformed = output.transformed();
    let full_width = output
        .levels()
        .first()
        .map_or(ll_width, |level| level.width) as usize;
    let mut ll = Vec::with_capacity((ll_width as usize) * (ll_height as usize));
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width)
            .ok_or("CUDA DWT LL row offset overflow")?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    let mut levels = Vec::with_capacity(output.levels().len());
    for shape in output.levels() {
        levels.push(signinum_j2k_native::J2kForwardDwt97Level {
            hl: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                0,
                shape.high_width,
                shape.low_height,
            )?,
            lh: extract_cuda_subband(
                transformed,
                full_width,
                0,
                shape.low_height,
                shape.low_width,
                shape.high_height,
            )?,
            hh: extract_cuda_subband(
                transformed,
                full_width,
                shape.low_width,
                shape.low_height,
                shape.high_width,
                shape.high_height,
            )?,
            width: shape.width,
            height: shape.height,
            low_width: shape.low_width,
            low_height: shape.low_height,
            high_width: shape.high_width,
            high_height: shape.high_height,
        });
    }
    levels.reverse();

    Ok(J2kForwardDwt97Output {
        ll,
        ll_width,
        ll_height,
        levels,
    })
}

#[cfg(feature = "cuda-runtime")]
fn extract_cuda_subband(
    transformed: &[f32],
    full_width: usize,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
) -> core::result::Result<Vec<f32>, &'static str> {
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height as usize {
        let row_start = (y0 as usize)
            .checked_add(y)
            .and_then(|row| row.checked_mul(full_width))
            .and_then(|row| row.checked_add(x0 as usize))
            .ok_or("CUDA DWT subband offset overflow")?;
        out.extend_from_slice(&transformed[row_start..row_start + width as usize]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda-runtime")]
    use super::{cuda_htj2k_encode_tables, cuda_runtime_required};
    use super::{
        encode_j2k_lossless_with_cuda, encode_j2k_lossless_with_cuda_and_profile,
        flatten_cuda_htj2k_packetization_job, CudaEncodeStageAccelerator,
        CudaHtj2kPacketizationPlanTagNodeState,
    };
    use signinum_core::{BackendKind, CodecError};
    #[cfg(feature = "cuda-runtime")]
    use signinum_cuda_runtime::{
        CudaContext, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob,
        CudaJ2kQuantizeJob,
    };
    use signinum_j2k::{
        EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
        J2kLosslessSamples,
    };
    #[cfg(feature = "cuda-runtime")]
    use signinum_j2k_native::J2kEncodeStageAccelerator;
    use signinum_j2k_native::{
        encode_with_accelerator, DecodeSettings, EncodeOptions, Image,
        J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
        J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder,
        J2kPacketizationResolution, J2kPacketizationSubband,
    };

    #[test]
    fn cuda_lossless_encode_prefer_device_errors_for_unsupported_classic_tier1() {
        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 17 + 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::PreferDevice)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let err = encode_j2k_lossless_with_cuda(samples, &options)
            .expect_err("CUDA-named encode must not silently return CPU fallback");

        assert!(err.is_unsupported());
        let message = err.to_string();
        let expected_missing_stage = if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
        {
            "tier1_code_block"
        } else {
            "deinterleave"
        };
        assert!(
            message.contains(expected_missing_stage),
            "expected strict CUDA encode error to mention {expected_missing_stage}, got {message}"
        );
    }

    #[test]
    fn cuda_lossless_encode_profile_prefer_device_errors_for_unsupported_classic_tier1() {
        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 19 + 7) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::PreferDevice)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::External);

        let err = encode_j2k_lossless_with_cuda_and_profile(samples, &options)
            .expect_err("profiled CUDA encode must not silently return CPU fallback");

        assert!(err.is_unsupported());
        let message = err.to_string();
        let expected_missing_stage = if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
        {
            "tier1_code_block"
        } else {
            "deinterleave"
        };
        assert!(
            message.contains(expected_missing_stage),
            "expected profiled strict CUDA encode error to mention {expected_missing_stage}, got {message}"
        );
    }

    #[test]
    fn cuda_lossless_encode_require_device_errors_for_unsupported_classic_tier1() {
        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 29 + 11) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::External);

        let err = encode_j2k_lossless_with_cuda(samples, &options)
            .expect_err("strict CUDA encode must not silently fall back to CPU");

        assert!(err.is_unsupported());
        let message = err.to_string();
        let expected_missing_stage = if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
        {
            "tier1_code_block"
        } else {
            "deinterleave"
        };
        assert!(
            message.contains(expected_missing_stage),
            "expected strict CUDA encode error to mention {expected_missing_stage}, got {message}"
        );
    }

    #[test]
    fn cuda_packetization_flatten_accepts_cleanup_only_single_block_packet() {
        let payload = [0x12, 0x34, 0x56, 0x78];
        let code_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let subband = J2kPacketizationSubband {
            code_blocks: vec![code_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        };
        let resolution = J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let descriptor = J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        };
        let job = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 1,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        };

        let plan = flatten_cuda_htj2k_packetization_job(job).expect("supported CUDA packetization");

        assert_eq!(plan.payload, payload);
        assert_eq!(plan.packets.len(), 1);
        assert_eq!(plan.subbands.len(), 1);
        assert_eq!(plan.blocks.len(), 1);
        assert_eq!(plan.packets[0].block_start, 0);
        assert_eq!(plan.packets[0].block_count, 1);
        assert_eq!(plan.packets[0].subband_start, 0);
        assert_eq!(plan.packets[0].subband_count, 1);
        assert_eq!(plan.subbands[0].block_start, 0);
        assert_eq!(plan.subbands[0].block_count, 1);
        let payload_len = u32::try_from(payload.len()).expect("test payload length fits in u32");
        assert!(plan.packets[0].output_capacity >= payload_len + 256);
        assert_eq!(plan.blocks[0].data_offset, 0);
        assert_eq!(plan.blocks[0].data_len, payload_len);
        assert_eq!(plan.blocks[0].num_coding_passes, 1);
        assert_eq!(plan.blocks[0].num_zero_bitplanes, 2);
    }

    #[test]
    fn cuda_packetization_flatten_accepts_cleanup_only_multi_block_packet() {
        let payloads = vec![
            vec![0x10, 0x11, 0x12],
            vec![0x20, 0x21],
            vec![0x30, 0x31, 0x32, 0x33],
            vec![0x40],
        ];
        let code_blocks = payloads
            .iter()
            .enumerate()
            .map(|(idx, payload)| J2kPacketizationCodeBlock {
                data: payload.as_slice(),
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: u8::try_from(idx + 1).expect("test zbp fits in u8"),
                previously_included: false,
                l_block: 3,
                block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
            })
            .collect();
        let subband = J2kPacketizationSubband {
            code_blocks,
            num_cbs_x: 2,
            num_cbs_y: 2,
        };
        let resolution = J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let descriptor = J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        };
        let job = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 4,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        };

        let plan =
            flatten_cuda_htj2k_packetization_job(job).expect("multi-block CUDA packetization");

        assert_eq!(plan.packets.len(), 1);
        assert_eq!(plan.subbands.len(), 1);
        assert_eq!(plan.blocks.len(), 4);
        assert_eq!(plan.packets[0].block_start, 0);
        assert_eq!(plan.packets[0].block_count, 4);
        assert_eq!(plan.packets[0].subband_start, 0);
        assert_eq!(plan.packets[0].subband_count, 1);
        assert_eq!(plan.subbands[0].block_start, 0);
        assert_eq!(plan.subbands[0].block_count, 4);
        assert_eq!(plan.subbands[0].num_cbs_x, 2);
        assert_eq!(plan.subbands[0].num_cbs_y, 2);
        assert_eq!(
            plan.payload,
            payloads.into_iter().flatten().collect::<Vec<_>>()
        );
        assert_eq!(plan.blocks[2].num_zero_bitplanes, 3);
    }

    #[test]
    fn cuda_packetization_flatten_accepts_ht_refinement_pass_packet() {
        let payload = [0x12, 0x34, 0x56, 0x78, 0x9a];
        let code_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: 3,
            ht_refinement_length: 2,
            num_coding_passes: 3,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let subband = J2kPacketizationSubband {
            code_blocks: vec![code_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        };
        let resolution = J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let descriptor = J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        };
        let job = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 1,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        };

        let plan = flatten_cuda_htj2k_packetization_job(job).expect("HT refinement packetization");

        assert_eq!(plan.payload, payload);
        assert_eq!(plan.blocks.len(), 1);
        assert_eq!(plan.blocks[0].num_coding_passes, 3);
        assert_eq!(
            plan.blocks[0].data_len,
            u32::try_from(payload.len()).expect("test payload length fits in u32")
        );
    }

    #[test]
    fn cuda_packetization_rejects_overflowing_ht_refinement_lengths() {
        let payload = [0x12];
        let code_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: u32::MAX,
            ht_refinement_length: 1,
            num_coding_passes: 3,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };

        let err = super::cuda_ht_segment_lengths(&code_block)
            .expect_err("overflowing CUDA HT segment lengths rejected");

        assert_eq!(
            err,
            "CUDA HTJ2K multi-pass packet contribution length overflow"
        );
    }

    #[test]
    fn cuda_packetization_flatten_rejects_out_of_range_ht_pass_count() {
        let payload = [0u8; 1];
        let code_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 165,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let subband = J2kPacketizationSubband {
            code_blocks: vec![code_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        };
        let resolution = J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let descriptor = J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        };
        let job = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 1,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        };

        let err = flatten_cuda_htj2k_packetization_job(job)
            .expect_err("invalid HT pass count must be rejected before CUDA launch");

        assert_eq!(
            err,
            "CUDA HTJ2K packetization coding pass count exceeds JPEG 2000 bounds"
        );
    }

    #[test]
    fn cuda_packetization_flatten_accepts_previously_included_second_layer_packet() {
        let first_payload = [0x11u8; 20];
        let second_payload = [0x22u8; 5];
        let first_block = J2kPacketizationCodeBlock {
            data: &first_payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let second_block = J2kPacketizationCodeBlock {
            data: &second_payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let first_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![first_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let second_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![second_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let descriptors = [
            J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let resolutions = [first_resolution, second_resolution];
        let job = J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 2,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        };

        let plan =
            flatten_cuda_htj2k_packetization_job(job).expect("stateful CUDA packetization plan");

        assert_eq!(
            plan.payload,
            [first_payload.as_slice(), second_payload.as_slice()].concat()
        );
        assert_eq!(plan.packets.len(), 2);
        assert_eq!(plan.blocks.len(), 2);
        assert_eq!(plan.packets[0].layer, 0);
        assert_eq!(plan.packets[1].layer, 1);
        assert_eq!(plan.blocks[0].l_block, 3);
        assert_eq!(plan.blocks[0].previously_included, 0);
        assert_eq!(plan.blocks[1].previously_included, 1);
        assert_eq!(plan.blocks[0].inclusion_layer, 0);
        assert_eq!(plan.blocks[1].inclusion_layer, 0);
        assert_eq!(
            plan.blocks[1].l_block, 5,
            "first layer length must update L-block for later packet state"
        );
    }

    #[test]
    fn cuda_packetization_flatten_accepts_deferred_first_inclusion_second_layer_packet() {
        let payload = [0x44u8; 5];
        let first_block = J2kPacketizationCodeBlock {
            data: &[],
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 0,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let second_block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let first_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![first_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let second_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![second_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };
        let descriptors = [
            J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let resolutions = [first_resolution, second_resolution];
        let job = J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 2,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        };

        let plan =
            flatten_cuda_htj2k_packetization_job(job).expect("deferred first inclusion plan");

        assert_eq!(plan.payload, payload);
        assert_eq!(plan.packets.len(), 2);
        assert_eq!(plan.blocks.len(), 2);
        assert_eq!(plan.packets[0].layer, 0);
        assert_eq!(plan.packets[1].layer, 1);
        assert_eq!(plan.blocks[0].previously_included, 0);
        assert_eq!(plan.blocks[1].previously_included, 0);
        assert_eq!(plan.blocks[0].inclusion_layer, 1);
        assert_eq!(plan.blocks[1].inclusion_layer, 1);
    }

    #[test]
    fn cuda_packetization_flatten_accepts_deferred_first_inclusion_after_non_empty_packet() {
        let first_payload = [0x11u8; 3];
        let second_payload = [0x22u8; 5];
        let first_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![
                    J2kPacketizationCodeBlock {
                        data: &first_payload,
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                    J2kPacketizationCodeBlock {
                        data: &[],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 0,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                ],
                num_cbs_x: 2,
                num_cbs_y: 1,
            }],
        };
        let second_resolution = J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![
                    J2kPacketizationCodeBlock {
                        data: &[],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 0,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                    J2kPacketizationCodeBlock {
                        data: &second_payload,
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                ],
                num_cbs_x: 2,
                num_cbs_y: 1,
            }],
        };
        let descriptors = [
            J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
            J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 0,
                precinct: 0,
            },
        ];
        let resolutions = [first_resolution, second_resolution];
        let job = J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 4,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        };

        let plan = flatten_cuda_htj2k_packetization_job(job)
            .expect("persistent tag-tree state is flattened for CUDA packetization");

        assert_eq!(
            plan.payload,
            [first_payload.as_slice(), second_payload.as_slice()].concat()
        );
        assert_eq!(plan.packets.len(), 2);
        assert_eq!(plan.blocks.len(), 4);
        assert_eq!(plan.blocks[0].previously_included, 0);
        assert_eq!(plan.blocks[1].previously_included, 0);
        assert_eq!(plan.blocks[2].previously_included, 1);
        assert_eq!(plan.blocks[3].previously_included, 0);
        assert_eq!(plan.blocks[0].inclusion_layer, 0);
        assert_eq!(plan.blocks[1].inclusion_layer, 1);
        assert_eq!(plan.blocks[2].inclusion_layer, 0);
        assert_eq!(plan.blocks[3].inclusion_layer, 1);
        assert_eq!(plan.tag_states.len(), 2);
        assert_eq!(plan.tag_nodes.len(), 12);
        assert_eq!(plan.tag_states[1].inclusion_node_start, 6);
        assert_eq!(plan.tag_states[1].zero_bitplane_node_start, 9);
        assert_eq!(
            &plan.tag_nodes[6..9],
            &[
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 0,
                    known: 1,
                },
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 1,
                    known: 0,
                },
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 0,
                    known: 1,
                },
            ]
        );
        assert_eq!(
            &plan.tag_nodes[9..12],
            &[
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 2,
                    known: 1,
                },
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 0,
                    known: 0,
                },
                CudaHtj2kPacketizationPlanTagNodeState {
                    current: 2,
                    known: 1,
                },
            ]
        );
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_require_device_dispatches_cleanup_packetization_when_runtime_required()
    {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..8 * 8)
            .map(|value| u8::try_from((value * 31 + 7) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let encoded = encode_j2k_lossless_with_cuda(samples, &options)
            .expect("strict CUDA single-pass HT encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_deinterleave_stage_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels = [0u8, 128, 255, 64, 32, 16];
        let mut accelerator = CudaEncodeStageAccelerator::default();
        let components = accelerator
            .encode_deinterleave(signinum_j2k_native::J2kDeinterleaveToF32Job {
                pixels: &pixels,
                num_pixels: 2,
                num_components: 3,
                bit_depth: 8,
                signed: false,
            })
            .expect("CUDA deinterleave hook")
            .expect("CUDA deinterleave dispatch");

        assert_eq!(accelerator.deinterleave_dispatches(), 1);
        assert_eq!(
            components,
            vec![vec![-128.0, -64.0], vec![0.0, -96.0], vec![127.0, -112.0]]
        );
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_require_device_dispatches_multi_block_cleanup_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 19 + 23) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let encoded = encode_j2k_lossless_with_cuda(samples, &options)
            .expect("strict CUDA multi-block cleanup encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_require_device_dispatches_dwt53_cleanup_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 37 + 41) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let encoded = encode_j2k_lossless_with_cuda(samples, &options)
            .expect("strict CUDA DWT cleanup encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_profile_reports_resident_stage_timings_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u32..128 * 128)
            .map(|value| u8::try_from((value * 43 + 29) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).expect("valid gray8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let (encoded, report) = encode_j2k_lossless_with_cuda_and_profile(samples, &options)
            .expect("strict CUDA profiled DWT cleanup encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
        assert_eq!(report.backend, BackendKind::Cuda);
        assert_eq!(report.input_bytes, pixels.len());
        assert_eq!(report.codestream_bytes, encoded.codestream.len());
        assert!(report.dispatch_count > 0);
        assert!(report.block_count > 0);
        assert!(report.deinterleave_us > 0);
        assert_eq!(report.mct_us, 0);
        assert!(report.dwt_us > 0);
        assert!(report.quantize_us > 0);
        assert!(report.ht_encode_us > 0);
        assert!(report.packetize_us > 0);
        assert!(report.total_us > 0);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_lossless_encode_require_device_dispatches_rgb_rct_cleanup_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u32..128 * 128 * 3)
            .map(|value| u8::try_from((value * 13 + 71) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let samples =
            J2kLosslessSamples::new(&pixels, 128, 128, 3, 8, false).expect("valid rgb8 samples");
        let options = J2kLosslessEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip);

        let encoded = encode_j2k_lossless_with_cuda(samples, &options)
            .expect("strict CUDA RGB cleanup encode should dispatch all required stages");
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(encoded.backend, BackendKind::Cuda);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn cuda_encode_stage_accelerator_preserves_cpu_codestream_validity() {
        let pixels: Vec<u8> = (0u8..192).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 3, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA stage accelerator");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.num_components, 3);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 3);
        assert!(accelerator.tier1_code_block_attempts() > 0);
        assert_eq!(accelerator.packetization_attempts(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_rct_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..7 * 5 * 3)
            .map(|i| u8::try_from((i * 17) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 7, 5, 3, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA forward RCT");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_ict_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u32..32 * 32 * 3)
            .map(|i| u8::try_from((i * 23 + 19) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: false,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 32, 32, 3, 8, false, &options, &mut accelerator)
                .expect("encode irreversible RGB with CUDA forward ICT");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data.len(), pixels.len());
        assert_eq!(accelerator.forward_ict_attempts(), 1);
        assert_eq!(accelerator.forward_ict_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_dwt53_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..8 * 8)
            .map(|i| u8::try_from((i * 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 1, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA forward DWT 5/3");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 2);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_forward_dwt97_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 7 + 13) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: false,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 32, 32, 1, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA forward DWT 9/7");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data.len(), pixels.len());
        assert_eq!(accelerator.forward_dwt97_attempts(), 1);
        assert_eq!(accelerator.forward_dwt97_dispatches(), 3);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_quantize_subband_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 19 + 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: false,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 32, 32, 1, 8, false, &options, &mut accelerator)
                .expect("encode with CUDA quantization");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data.len(), pixels.len());
        assert_eq!(accelerator.quantize_subband_attempts(), 4);
        assert_eq!(accelerator.quantize_subband_dispatches(), 4);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_encode_uses_resident_tile_body_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 23 + 11) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 32, 32, 1, 8, false, &options, &mut accelerator)
                .expect("encode HTJ2K through CUDA tile-body hook");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.htj2k_tile_attempts, 1);
        assert_eq!(accelerator.htj2k_tile_dispatches, 1);
        assert_eq!(accelerator.ht_subband_attempts, 0);
        assert_eq!(accelerator.ht_subband_dispatches, 0);
        assert_eq!(accelerator.deinterleave_dispatches(), 1);
        assert_eq!(accelerator.quantize_subband_attempts(), 1);
        assert_eq!(accelerator.quantize_subband_dispatches(), 1);
        assert_eq!(accelerator.ht_code_block_attempts(), 4);
        assert_eq!(accelerator.ht_code_block_dispatches(), 1);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_encode_uses_resident_dwt_tile_body_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 29 + 5) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 32, 32, 1, 8, false, &options, &mut accelerator)
                .expect("encode HTJ2K DWT through CUDA tile-body hook");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.htj2k_tile_attempts, 1);
        assert_eq!(accelerator.htj2k_tile_dispatches, 1);
        assert_eq!(accelerator.ht_subband_attempts, 0);
        assert_eq!(accelerator.ht_subband_dispatches, 0);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert!(accelerator.forward_dwt53_dispatches() > 0);
        assert_eq!(accelerator.quantize_subband_attempts(), 4);
        assert_eq!(accelerator.quantize_subband_dispatches(), 4);
        assert_eq!(accelerator.ht_code_block_attempts(), 4);
        assert_eq!(accelerator.ht_code_block_dispatches(), 4);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_encode_uses_resident_mct_dwt_tile_body_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32 * 3)
            .map(|i| u8::try_from((i * 19 + 17) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_mct: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 32, 32, 3, 8, false, &options, &mut accelerator)
                .expect("encode HTJ2K RGB DWT through CUDA tile-body hook");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert_eq!(accelerator.htj2k_tile_attempts, 1);
        assert_eq!(accelerator.htj2k_tile_dispatches, 1);
        assert_eq!(accelerator.ht_subband_attempts, 0);
        assert_eq!(accelerator.forward_rct_attempts(), 1);
        assert_eq!(accelerator.forward_rct_dispatches(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 3);
        assert!(accelerator.forward_dwt53_dispatches() > 0);
        assert_eq!(accelerator.quantize_subband_attempts(), 12);
        assert_eq!(accelerator.quantize_subband_dispatches(), 12);
        assert_eq!(accelerator.ht_code_block_attempts(), 12);
        assert_eq!(accelerator.ht_code_block_dispatches(), 12);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_encode_uses_resident_dwt97_tile_body_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 31 + 7) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: false,
            use_ht_block_coding: true,
            num_decomposition_levels: 1,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 32, 32, 1, 8, false, &options, &mut accelerator)
                .expect("encode irreversible HTJ2K DWT through CUDA tile-body hook");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.width, 32);
        assert_eq!(decoded.height, 32);
        assert_eq!(decoded.num_components, 1);
        assert_eq!(accelerator.htj2k_tile_attempts, 1);
        assert_eq!(accelerator.htj2k_tile_dispatches, 1);
        assert_eq!(accelerator.ht_subband_attempts, 0);
        assert_eq!(accelerator.forward_dwt97_attempts(), 1);
        assert!(accelerator.forward_dwt97_dispatches() > 0);
        assert_eq!(accelerator.quantize_subband_attempts(), 4);
        assert_eq!(accelerator.quantize_subband_dispatches(), 4);
        assert_eq!(accelerator.ht_code_block_attempts(), 4);
        assert_eq!(accelerator.ht_code_block_dispatches(), 4);
        assert_eq!(accelerator.packetization_attempts(), 1);
        assert_eq!(accelerator.packetization_dispatches(), 1);
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_htj2k_codeblock_dispatches_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..8 * 8)
            .map(|i| u8::try_from((i * 11 + 3) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 1, 8, false, &options, &mut accelerator)
                .expect("encode HTJ2K with CUDA HT codeblock kernel");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert!(accelerator.ht_code_block_attempts() > 0);
        assert!(accelerator.ht_code_block_dispatches() > 0);
        assert!(accelerator.ht_code_block_dispatches() <= accelerator.ht_code_block_attempts());
        assert_eq!(
            accelerator.dispatch_report().ht_code_block,
            accelerator.ht_code_block_dispatches()
        );
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_htj2k_codeblock_batch_uses_single_dispatch_when_runtime_required() {
        if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_none() {
            return;
        }

        let pixels: Vec<u8> = (0u16..32 * 32)
            .map(|i| u8::try_from((i * 17 + 9) & 0xFF).expect("masked value fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CudaEncodeStageAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 32, 32, 1, 8, false, &options, &mut accelerator)
                .expect("encode HTJ2K with CUDA HT batch codeblock kernel");
        let decoded = Image::new(&codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(decoded.data, pixels);
        assert!(accelerator.ht_code_block_attempts() > 1);
        assert_eq!(accelerator.ht_code_block_dispatches(), 1);
        assert!(
            accelerator.ht_code_block_dispatches() < accelerator.ht_code_block_attempts(),
            "batch encode must not launch one kernel per codeblock"
        );
        assert_eq!(
            accelerator.dispatch_report().ht_code_block,
            accelerator.ht_code_block_dispatches()
        );
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_resident_quantized_subband_feeds_resident_ht_batch_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let samples = [-3.6f32, -2.5, -0.4, 0.0, 0.49, 1.5, 3.2, 9.9];
        let context = CudaContext::system_default().expect("CUDA context");
        let sample_buffer = context.upload_f32(&samples).expect("resident samples");
        let quantization = CudaJ2kQuantizeJob {
            step_exponent: 8,
            step_mantissa: 0,
            range_bits: 8,
            reversible: true,
        };
        let resident_quantized = context
            .j2k_quantize_subband_resident(&sample_buffer, samples.len(), quantization)
            .expect("resident quantization");
        let host_quantized = context
            .j2k_quantize_subband(&samples, quantization)
            .expect("host-staged quantization");
        let jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 4,
            height: 2,
            total_bitplanes: 5,
        }];

        let resident_encoded = context
            .encode_htj2k_codeblocks_resident(
                resident_quantized.buffer(),
                resident_quantized.coefficient_count(),
                &jobs,
                cuda_htj2k_encode_tables(),
            )
            .expect("resident HTJ2K encode");
        let staged_encoded = context
            .encode_htj2k_codeblocks(
                host_quantized.coefficients(),
                &jobs,
                cuda_htj2k_encode_tables(),
            )
            .expect("host-staged HTJ2K encode");

        assert_eq!(resident_quantized.coefficient_count(), samples.len());
        assert_eq!(resident_encoded.execution().kernel_dispatches(), 1);
        assert_eq!(
            resident_encoded.code_blocks().len(),
            staged_encoded.code_blocks().len()
        );
        for (resident, staged) in resident_encoded
            .code_blocks()
            .iter()
            .zip(staged_encoded.code_blocks())
        {
            assert_eq!(resident.data(), staged.data());
            assert_eq!(resident.cleanup_length(), staged.cleanup_length());
            assert_eq!(resident.refinement_length(), staged.refinement_length());
            assert_eq!(resident.num_coding_passes(), staged.num_coding_passes());
            assert_eq!(resident.num_zero_bitplanes(), staged.num_zero_bitplanes());
        }
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_resident_strided_codeblock_region_matches_host_gather_when_runtime_required() {
        if !cuda_runtime_required() {
            return;
        }

        let samples: Vec<f32> = (0u16..16).map(|value| f32::from(value) - 8.0).collect();
        let context = CudaContext::system_default().expect("CUDA context");
        let sample_buffer = context.upload_f32(&samples).expect("resident samples");
        let quantization = CudaJ2kQuantizeJob {
            step_exponent: 8,
            step_mantissa: 0,
            range_bits: 8,
            reversible: true,
        };
        let resident_quantized = context
            .j2k_quantize_subband_resident(&sample_buffer, samples.len(), quantization)
            .expect("resident quantization");
        let quantized = resident_quantized
            .download_coefficients()
            .expect("download quantized coefficients");
        let gathered_codeblock = vec![quantized[5], quantized[6], quantized[9], quantized[10]];
        let region_jobs = [CudaHtj2kEncodeCodeBlockRegionJob {
            coefficient_offset: 5,
            coefficient_stride: 4,
            width: 2,
            height: 2,
            total_bitplanes: 5,
        }];
        let contiguous_jobs = [CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: 0,
            width: 2,
            height: 2,
            total_bitplanes: 5,
        }];

        let resident_encoded = context
            .encode_htj2k_codeblock_regions_resident(
                resident_quantized.buffer(),
                resident_quantized.coefficient_count(),
                &region_jobs,
                cuda_htj2k_encode_tables(),
            )
            .expect("resident strided HTJ2K encode");
        let staged_encoded = context
            .encode_htj2k_codeblocks(
                &gathered_codeblock,
                &contiguous_jobs,
                cuda_htj2k_encode_tables(),
            )
            .expect("host-gathered HTJ2K encode");

        assert_eq!(resident_encoded.execution().kernel_dispatches(), 1);
        assert_eq!(resident_encoded.code_blocks().len(), 1);
        assert_eq!(
            resident_encoded.code_blocks()[0].data(),
            staged_encoded.code_blocks()[0].data()
        );
        assert_eq!(
            resident_encoded.code_blocks()[0].num_zero_bitplanes(),
            staged_encoded.code_blocks()[0].num_zero_bitplanes()
        );
    }
}
