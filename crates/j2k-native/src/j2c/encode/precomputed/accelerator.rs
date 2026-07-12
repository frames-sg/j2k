// SPDX-License-Identifier: MIT OR Apache-2.0
// j2k-coverage: shared-accelerator-host

use super::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kEncodeStageAccelerator,
    J2kPacketizationEncodeJob, J2kQuantizeSubbandJob, J2kTier1CodeBlockEncodeJob, Vec,
};

pub(in crate::j2c::encode) struct PrecomputedStageAccelerator<'a, A: J2kEncodeStageAccelerator> {
    pub(super) encode_accelerator: &'a mut A,
}

// These adapters expose only the encode hooks supported by precomputed input.
// The direct 5/3 route skips sample/color/DWT stages entirely; the 9/7 wrapper
// replaces only forward DWT. Whole-subband/tile hooks keep their defaults.
macro_rules! forward_precomputed_encode_stage_hooks {
    () => {
        fn dispatch_report(&self) -> crate::J2kEncodeDispatchReport {
            self.encode_accelerator.dispatch_report()
        }

        fn encode_quantize_subband(
            &mut self,
            job: J2kQuantizeSubbandJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<Vec<i32>>> {
            self.encode_accelerator.encode_quantize_subband(job)
        }

        fn encode_tier1_code_block(
            &mut self,
            job: J2kTier1CodeBlockEncodeJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<EncodedJ2kCodeBlock>> {
            self.encode_accelerator.encode_tier1_code_block(job)
        }

        fn encode_tier1_code_blocks(
            &mut self,
            jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
        ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedJ2kCodeBlock>>> {
            self.encode_accelerator.encode_tier1_code_blocks(jobs)
        }

        fn encode_ht_code_block(
            &mut self,
            job: crate::J2kHtCodeBlockEncodeJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<EncodedHtJ2kCodeBlock>> {
            self.encode_accelerator.encode_ht_code_block(job)
        }

        fn encode_ht_code_blocks(
            &mut self,
            jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
        ) -> crate::J2kEncodeStageResult<Option<Vec<EncodedHtJ2kCodeBlock>>> {
            self.encode_accelerator.encode_ht_code_blocks(jobs)
        }

        fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
            self.encode_accelerator
                .prefer_parallel_cpu_code_block_fallback()
        }

        fn prefer_parallel_cpu_tile_encode(&self) -> bool {
            self.encode_accelerator.prefer_parallel_cpu_tile_encode()
        }

        fn encode_packetization(
            &mut self,
            job: J2kPacketizationEncodeJob<'_>,
        ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
            self.encode_accelerator.encode_packetization(job)
        }
    };
}

impl<A: J2kEncodeStageAccelerator> J2kEncodeStageAccelerator
    for PrecomputedStageAccelerator<'_, A>
{
    forward_precomputed_encode_stage_hooks!();
}
