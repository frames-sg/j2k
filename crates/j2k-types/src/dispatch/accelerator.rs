// SPDX-License-Identifier: MIT OR Apache-2.0

//! Low-level encode-stage accelerator integration contract.

use alloc::vec::Vec;

use super::{J2kEncodeDispatchReport, J2kEncodeStageResult};
use crate::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kForwardDwt53Job, J2kForwardDwt53Output,
    J2kForwardDwt97Job, J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob,
    J2kHtCodeBlockEncodeJob, J2kHtSubbandEncodeJob, J2kPacketizationEncodeJob,
    J2kPacketizationProgressionOrder, J2kQuantizeSubbandJob, J2kResidentHtj2kTileEncodeJob,
    J2kTier1CodeBlockEncodeJob,
};

/// Pixel deinterleave and level-shift job supplied to an accelerator.
#[derive(Debug, Clone, Copy)]
pub struct J2kDeinterleaveToF32Job<'a> {
    /// Interleaved source pixel bytes.
    pub pixels: &'a [u8],
    /// Number of pixels to convert.
    pub num_pixels: usize,
    /// Number of interleaved components per pixel.
    pub num_components: u16,
    /// Source sample bit depth.
    pub bit_depth: u8,
    /// Whether source samples are signed.
    pub signed: bool,
}

/// Combined pixel deinterleave, level-shift, and forward MCT job supplied to an accelerator.
///
/// The native encoder only offers this job for three-component inputs with MCT enabled.
#[derive(Debug, Clone, Copy)]
pub struct J2kDeinterleaveMctToF32Job<'a> {
    /// Interleaved source pixel bytes.
    pub pixels: &'a [u8],
    /// Number of pixels to convert.
    pub num_pixels: usize,
    /// Source sample bit depth.
    pub bit_depth: u8,
    /// Whether source samples are signed.
    pub signed: bool,
    /// Whether to apply the reversible RCT (`true`) or irreversible ICT (`false`).
    pub reversible: bool,
}

/// Validated image and coding context supplied before encode-stage dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kEncodeContext {
    /// Number of pixels in the encoded image or tile.
    pub num_pixels: usize,
    /// Number of interleaved source components.
    pub num_components: u16,
    /// Source sample bit depth.
    pub bit_depth: u8,
    /// Whether source samples are signed.
    pub signed: bool,
    /// Whether the codestream uses reversible coding.
    pub reversible: bool,
}

/// HTJ2K tile-body encode job for a backend-resident full-tile path.
#[derive(Debug, Clone, Copy)]
pub struct J2kHtj2kTileEncodeJob<'a> {
    /// Interleaved source pixel bytes.
    pub pixels: &'a [u8],
    /// Tile/image width in samples.
    pub width: u32,
    /// Tile/image height in samples.
    pub height: u32,
    /// Number of interleaved image components.
    pub num_components: u16,
    /// Source component bit depth.
    pub bit_depth: u8,
    /// Whether source samples are signed.
    pub signed: bool,
    /// Number of DWT decomposition levels.
    pub num_decomposition_levels: u8,
    /// Whether the codestream uses reversible coding.
    pub reversible: bool,
    /// Whether a multi-component transform should be applied.
    pub use_mct: bool,
    /// JPEG 2000 guard bits used to derive total coded bitplanes.
    pub guard_bits: u8,
    /// Code-block width in samples.
    pub code_block_width: u32,
    /// Code-block height in samples.
    pub code_block_height: u32,
    /// Packet progression order to emit.
    pub progression_order: J2kPacketizationProgressionOrder,
    /// Per-component sampling factors, as `(x_rsiz, y_rsiz)`.
    pub component_sampling: &'a [(u8, u8)],
    /// Quantization step sizes, as `(exponent, mantissa)`, in codestream order.
    pub quantization_steps: &'a [(u16, u16)],
}

/// CPU-only encode accelerator that always falls back to native stages.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuOnlyJ2kEncodeStageAccelerator;

/// Low-level JPEG 2000 encode-stage accelerator integration contract.
pub trait J2kEncodeStageAccelerator {
    /// Supply validated context before any encode-stage hook is invoked.
    fn begin_encode(&mut self, _context: J2kEncodeContext) -> J2kEncodeStageResult<()> {
        Ok(())
    }

    /// Report cumulative backend dispatches completed by this accelerator.
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        J2kEncodeDispatchReport::default()
    }

    /// Report the exact maximum cleanup magnitude from the latest fused HT subband encode.
    fn ht_subband_maximum_cleanup_magnitude(&self) -> Option<u64> {
        None
    }

    /// Report the exact Part 15 magnitude bound from the latest complete HT tile encode.
    fn ht_tile_required_magnitude_bound(&self) -> Option<u8> {
        None
    }

    /// Optionally deinterleave interleaved pixel bytes into f32 component planes.
    fn encode_deinterleave(
        &mut self,
        _job: J2kDeinterleaveToF32Job<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
        Ok(None)
    }

    /// Optionally combine three-component deinterleave, level shift, and forward MCT.
    fn encode_deinterleave_mct(
        &mut self,
        _job: J2kDeinterleaveMctToF32Job<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
        Ok(None)
    }

    /// Optionally apply forward RCT in place.
    fn encode_forward_rct(&mut self, _job: J2kForwardRctJob<'_>) -> J2kEncodeStageResult<bool> {
        Ok(false)
    }

    /// Optionally apply forward ICT in place.
    fn encode_forward_ict(&mut self, _job: J2kForwardIctJob<'_>) -> J2kEncodeStageResult<bool> {
        Ok(false)
    }

    /// Optionally run a forward reversible 5/3 DWT.
    fn encode_forward_dwt53(
        &mut self,
        _job: J2kForwardDwt53Job<'_>,
    ) -> J2kEncodeStageResult<Option<J2kForwardDwt53Output>> {
        Ok(None)
    }

    /// Optionally run a forward irreversible 9/7 DWT.
    fn encode_forward_dwt97(
        &mut self,
        _job: J2kForwardDwt97Job<'_>,
    ) -> J2kEncodeStageResult<Option<J2kForwardDwt97Output>> {
        Ok(None)
    }

    /// Optionally quantize one subband.
    fn encode_quantize_subband(
        &mut self,
        _job: J2kQuantizeSubbandJob<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<i32>>> {
        Ok(None)
    }

    /// Optionally encode one classic Tier-1 code block.
    fn encode_tier1_code_block(
        &mut self,
        _job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<EncodedJ2kCodeBlock>> {
        Ok(None)
    }

    /// Optionally encode multiple classic Tier-1 code blocks in one backend dispatch.
    fn encode_tier1_code_blocks(
        &mut self,
        _jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    ) -> J2kEncodeStageResult<Option<Vec<EncodedJ2kCodeBlock>>> {
        Ok(None)
    }

    /// Optionally encode one HTJ2K code block.
    fn encode_ht_code_block(
        &mut self,
        _job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<EncodedHtJ2kCodeBlock>> {
        Ok(None)
    }

    /// Optionally encode multiple HTJ2K code blocks in one backend dispatch.
    fn encode_ht_code_blocks(
        &mut self,
        _jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> J2kEncodeStageResult<Option<Vec<EncodedHtJ2kCodeBlock>>> {
        Ok(None)
    }

    /// Optionally quantize and encode one HTJ2K cleanup/refinement subband.
    fn encode_ht_subband(
        &mut self,
        _job: J2kHtSubbandEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<EncodedHtJ2kCodeBlock>>> {
        Ok(None)
    }

    /// Optionally encode the complete HTJ2K tile packet body.
    fn encode_htj2k_tile(
        &mut self,
        _job: J2kHtj2kTileEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Optionally encode a complete HTJ2K tile whose pixels remain backend-resident.
    fn encode_resident_htj2k_tile(
        &mut self,
        _job: J2kResidentHtj2kTileEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Return whether CPU code-block fallback should use internal rayon parallelism.
    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        false
    }

    /// Return whether callers may parallelize whole-tile CPU-only batch encode.
    fn prefer_parallel_cpu_tile_encode(&self) -> bool {
        false
    }

    /// Optionally packetize prepared packet contributions.
    fn encode_packetization(
        &mut self,
        _job: J2kPacketizationEncodeJob<'_>,
    ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
        Ok(None)
    }
}

#[doc(hidden)]
impl J2kEncodeStageAccelerator for CpuOnlyJ2kEncodeStageAccelerator {
    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        true
    }

    fn prefer_parallel_cpu_tile_encode(&self) -> bool {
        true
    }
}
