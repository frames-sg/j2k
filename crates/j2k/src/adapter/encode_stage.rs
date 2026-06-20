// SPDX-License-Identifier: Apache-2.0

//! Public JPEG 2000 encode-stage adapter contracts.
//!
//! The encode-stage job, output, and report types are shared with
//! `j2k-native` through the neutral `j2k-types` contract
//! crate; this module re-exports them and keeps the adapter-side accelerator
//! trait plus the hidden native bridge.

use alloc::vec::Vec;

pub use j2k_types::{
    CpuOnlyJ2kEncodeStageAccelerator, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock,
    IrreversibleQuantizationStep, IrreversibleQuantizationSubbandScales, J2kCodeBlockSegment,
    J2kCodeBlockStyle, J2kDeinterleaveToF32Job, J2kEncodeDispatchReport, J2kForwardDwt53Job,
    J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Job, J2kForwardDwt97Level,
    J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob, J2kHtCodeBlockEncodeJob,
    J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
    J2kPacketizationProgressionOrder, J2kPacketizationResolution, J2kPacketizationSubband,
    J2kQuantizeSubbandJob, J2kSubBandType, J2kTier1CodeBlockEncodeJob, PrecomputedHtj2k53Component,
    PrecomputedHtj2k53Image, PrecomputedHtj2k97Component, PrecomputedHtj2k97Image,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};

/// Adapter JPEG 2000 encode-stage accelerator for backend experimentation.
pub trait J2kEncodeStageAccelerator {
    /// Report cumulative backend dispatches completed by this accelerator.
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        J2kEncodeDispatchReport::default()
    }

    /// Optionally deinterleave interleaved pixel bytes into f32 component planes.
    fn encode_deinterleave(
        &mut self,
        _job: J2kDeinterleaveToF32Job<'_>,
    ) -> core::result::Result<Option<Vec<Vec<f32>>>, &'static str> {
        Ok(None)
    }

    /// Optionally apply forward RCT in place.
    fn encode_forward_rct(
        &mut self,
        _job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        Ok(false)
    }

    /// Optionally apply forward ICT in place.
    fn encode_forward_ict(
        &mut self,
        _job: J2kForwardIctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        Ok(false)
    }

    /// Optionally run a forward reversible 5/3 DWT.
    fn encode_forward_dwt53(
        &mut self,
        _job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        Ok(None)
    }

    /// Optionally run a forward irreversible 9/7 DWT.
    fn encode_forward_dwt97(
        &mut self,
        _job: J2kForwardDwt97Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt97Output>, &'static str> {
        Ok(None)
    }

    /// Optionally quantize one sub-band.
    fn encode_quantize_subband(
        &mut self,
        _job: J2kQuantizeSubbandJob<'_>,
    ) -> core::result::Result<Option<Vec<i32>>, &'static str> {
        Ok(None)
    }

    /// Optionally encode one classic Tier-1 code-block.
    fn encode_tier1_code_block(
        &mut self,
        _job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        Ok(None)
    }

    /// Optionally encode multiple classic Tier-1 code-blocks in one backend dispatch.
    fn encode_tier1_code_blocks(
        &mut self,
        _jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedJ2kCodeBlock>>, &'static str> {
        Ok(None)
    }

    /// Optionally encode one HTJ2K code-block.
    fn encode_ht_code_block(
        &mut self,
        _job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        Ok(None)
    }

    /// Optionally encode multiple HTJ2K code-blocks in one backend dispatch.
    fn encode_ht_code_blocks(
        &mut self,
        _jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        Ok(None)
    }

    /// Optionally quantize and encode one HTJ2K cleanup-only sub-band.
    fn encode_ht_subband(
        &mut self,
        _job: J2kHtSubbandEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        Ok(None)
    }

    /// Optionally encode the complete HTJ2K tile packet body.
    fn encode_htj2k_tile(
        &mut self,
        _job: J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        Ok(None)
    }

    /// Return whether native CPU code-block fallback should use internal rayon parallelism.
    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        false
    }

    /// Return whether whole-tile CPU-only batch encode may be parallelized by callers.
    fn prefer_parallel_cpu_tile_encode(&self) -> bool {
        false
    }

    /// Optionally packetize prepared packet contributions.
    fn encode_packetization(
        &mut self,
        _job: J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        Ok(None)
    }
}

impl J2kEncodeStageAccelerator for CpuOnlyJ2kEncodeStageAccelerator {
    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        true
    }

    fn prefer_parallel_cpu_tile_encode(&self) -> bool {
        true
    }
}

/// Adapter that lets native encoder internals call a public encode-stage accelerator.
///
/// With the shared `j2k-types` contract this is a direct
/// passthrough: the job, output, and report types on both sides are the same
/// types.
#[doc(hidden)]
pub struct NativeEncodeStageAdapter<'a, A: J2kEncodeStageAccelerator + ?Sized> {
    inner: &'a mut A,
}

impl<'a, A: J2kEncodeStageAccelerator + ?Sized> NativeEncodeStageAdapter<'a, A> {
    /// Create an adapter around a public encode-stage accelerator.
    pub fn new(inner: &'a mut A) -> Self {
        Self { inner }
    }
}

#[doc(hidden)]
impl<A: J2kEncodeStageAccelerator + ?Sized> j2k_native::J2kEncodeStageAccelerator
    for NativeEncodeStageAdapter<'_, A>
{
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        self.inner.dispatch_report()
    }

    fn encode_deinterleave(
        &mut self,
        job: J2kDeinterleaveToF32Job<'_>,
    ) -> core::result::Result<Option<Vec<Vec<f32>>>, &'static str> {
        self.inner.encode_deinterleave(job)
    }

    fn encode_forward_rct(
        &mut self,
        job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.inner.encode_forward_rct(job)
    }

    fn encode_forward_ict(
        &mut self,
        job: J2kForwardIctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        self.inner.encode_forward_ict(job)
    }

    fn encode_forward_dwt53(
        &mut self,
        job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        self.inner.encode_forward_dwt53(job)
    }

    fn encode_forward_dwt97(
        &mut self,
        job: J2kForwardDwt97Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt97Output>, &'static str> {
        self.inner.encode_forward_dwt97(job)
    }

    fn encode_quantize_subband(
        &mut self,
        job: J2kQuantizeSubbandJob<'_>,
    ) -> core::result::Result<Option<Vec<i32>>, &'static str> {
        self.inner.encode_quantize_subband(job)
    }

    fn encode_tier1_code_block(
        &mut self,
        job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        self.inner.encode_tier1_code_block(job)
    }

    fn encode_tier1_code_blocks(
        &mut self,
        jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedJ2kCodeBlock>>, &'static str> {
        self.inner.encode_tier1_code_blocks(jobs)
    }

    fn encode_ht_code_block(
        &mut self,
        job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        self.inner.encode_ht_code_block(job)
    }

    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        self.inner.encode_ht_code_blocks(jobs)
    }

    fn encode_ht_subband(
        &mut self,
        job: J2kHtSubbandEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        self.inner.encode_ht_subband(job)
    }

    fn encode_htj2k_tile(
        &mut self,
        job: J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.inner.encode_htj2k_tile(job)
    }

    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        self.inner.prefer_parallel_cpu_code_block_fallback()
    }

    fn prefer_parallel_cpu_tile_encode(&self) -> bool {
        self.inner.prefer_parallel_cpu_tile_encode()
    }

    fn encode_packetization(
        &mut self,
        job: J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        self.inner.encode_packetization(job)
    }
}

#[cfg(test)]
mod tests {
    use super::{J2kEncodeDispatchReport, NativeEncodeStageAdapter};
    use crate::J2kEncodeStageAccelerator;

    #[derive(Default)]
    struct ReportingAccelerator {
        report: J2kEncodeDispatchReport,
    }

    impl J2kEncodeStageAccelerator for ReportingAccelerator {
        fn dispatch_report(&self) -> J2kEncodeDispatchReport {
            self.report
        }
    }

    #[test]
    fn native_adapter_forwards_dispatch_report() {
        let mut accelerator = ReportingAccelerator {
            report: J2kEncodeDispatchReport {
                deinterleave: 1,
                forward_rct: 2,
                forward_ict: 3,
                forward_dwt53: 4,
                forward_dwt97: 5,
                quantize_subband: 6,
                tier1_code_block: 7,
                ht_code_block: 8,
                packetization: 9,
            },
        };
        let adapter = NativeEncodeStageAdapter::new(&mut accelerator);

        let report = j2k_native::J2kEncodeStageAccelerator::dispatch_report(&adapter);

        assert_eq!(report.deinterleave, 1);
        assert_eq!(report.forward_rct, 2);
        assert_eq!(report.forward_ict, 3);
        assert_eq!(report.forward_dwt53, 4);
        assert_eq!(report.forward_dwt97, 5);
        assert_eq!(report.quantize_subband, 6);
        assert_eq!(report.tier1_code_block, 7);
        assert_eq!(report.ht_code_block, 8);
        assert_eq!(report.packetization, 9);
    }
}
