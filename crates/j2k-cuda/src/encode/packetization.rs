// SPDX-License-Identifier: MIT OR Apache-2.0

mod error;
mod flatten;
#[cfg(feature = "cuda-runtime")]
mod runtime;
mod state;
mod tag_tree;
mod types;

#[cfg(test)]
pub(super) use self::flatten::flatten_cuda_htj2k_packetization_job;
pub(super) use self::flatten::flatten_cuda_htj2k_packetization_job_classified;
#[cfg(feature = "cuda-runtime")]
pub(super) use self::flatten::flatten_cuda_htj2k_packetization_job_classified_with_live_host_bytes;
#[cfg(feature = "cuda-runtime")]
pub(super) use self::runtime::{
    cuda_packetization_blocks, cuda_packetization_packets, cuda_packetization_subbands,
    cuda_packetization_tag_nodes, cuda_packetization_tag_states,
};
pub(super) use self::types::CudaHtj2kPacketizationPlanError;

#[cfg(test)]
pub(super) use self::state::cuda_ht_segment_lengths;
#[cfg(test)]
pub(super) use self::types::CudaHtj2kPacketizationPlanTagNodeState;
#[cfg(test)]
mod tests;
