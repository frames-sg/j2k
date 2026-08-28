// SPDX-License-Identifier: MIT OR Apache-2.0

//! Test-only observations for CUDA encode routing and dispatch behavior.

use super::CudaEncodeStageAccelerator;

impl CudaEncodeStageAccelerator {
    pub(crate) fn forward_rct_attempts(&self) -> usize {
        self.forward_rct_attempts
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn forward_ict_attempts(&self) -> usize {
        self.forward_ict_attempts
    }

    pub(crate) fn forward_dwt53_attempts(&self) -> usize {
        self.forward_dwt53_attempts
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn forward_dwt97_attempts(&self) -> usize {
        self.forward_dwt97_attempts
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn htj2k_tile_attempts(&self) -> usize {
        self.htj2k_tile_attempts
    }

    pub(crate) fn quantize_subband_attempts(&self) -> usize {
        self.quantize_subband_attempts
    }

    pub(crate) fn tier1_code_block_attempts(&self) -> usize {
        self.tier1_code_block_attempts
    }

    pub(crate) fn ht_code_block_attempts(&self) -> usize {
        self.ht_code_block_attempts
    }

    pub(crate) fn ht_subband_attempts(&self) -> usize {
        self.ht_subband_attempts
    }

    pub(crate) fn packetization_attempts(&self) -> usize {
        self.packetization_attempts
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn deinterleave_dispatches(&self) -> usize {
        self.deinterleave_dispatches
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn forward_rct_dispatches(&self) -> usize {
        self.forward_rct_dispatches
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn forward_ict_dispatches(&self) -> usize {
        self.forward_ict_dispatches
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn forward_dwt53_dispatches(&self) -> usize {
        self.forward_dwt53_dispatches
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn forward_dwt97_dispatches(&self) -> usize {
        self.forward_dwt97_dispatches
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn htj2k_tile_dispatches(&self) -> usize {
        self.htj2k_tile_dispatches
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn quantize_subband_dispatches(&self) -> usize {
        self.quantize_subband_dispatches
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn ht_code_block_dispatches(&self) -> usize {
        self.ht_code_block_dispatches
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn ht_subband_dispatches(&self) -> usize {
        self.ht_subband_dispatches
    }

    pub(crate) fn packetization_dispatches(&self) -> usize {
        self.packetization_dispatches
    }
}
