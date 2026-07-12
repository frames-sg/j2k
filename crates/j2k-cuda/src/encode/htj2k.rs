// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::J2kEncodeStageError;

use super::stage_error::adapter_error;

fn htj2k_allocation_error(error: crate::Error) -> J2kEncodeStageError {
    adapter_error("allocate CUDA HTJ2K encode staging", error)
}

mod code_blocks;
mod host_budget;
mod ordering;
mod resident;
mod tile_packets;
mod types;
mod validation;

pub(crate) use self::code_blocks::cuda_htj2k_encode_tables;
pub(super) use self::code_blocks::{
    cuda_encode_ht_code_block, cuda_encode_ht_code_blocks, cuda_encode_ht_subband,
    encoded_ht_code_blocks_from_cuda,
};
pub(super) use self::resident::{cuda_encode_htj2k_device_tile_body, cuda_encode_htj2k_tile_body};
