// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kEncodeStageAccelerator, J2kForwardDwt53Job,
    J2kForwardDwt53Output, J2kForwardDwt97Job, J2kForwardDwt97Output, J2kPacketizationEncodeJob,
    J2kQuantizeSubbandJob, J2kTier1CodeBlockEncodeJob, Vec,
};

pub(in crate::j2c::encode) struct PrecomputedDwtAccelerator<'a, A: J2kEncodeStageAccelerator> {
    pub(super) outputs: Vec<J2kForwardDwt53Output>,
    pub(super) encode_accelerator: &'a mut A,
}

pub(in crate::j2c::encode) struct PrecomputedDwt97Accelerator<'a, A: J2kEncodeStageAccelerator> {
    pub(super) outputs: Vec<J2kForwardDwt97Output>,
    pub(super) encode_accelerator: &'a mut A,
}

// These wrappers only replace the forward-DWT stage with caller-supplied
// coefficients. Earlier sample/color hooks and whole-subband/tile HTJ2K hooks
// keep the trait defaults so precomputed-DWT APIs cannot intercept unrelated
// encode stages.
macro_rules! forward_precomputed_encode_stage_hooks {
    () => {
        fn dispatch_report(&self) -> crate::J2kEncodeDispatchReport {
            self.encode_accelerator.dispatch_report()
        }

        fn encode_quantize_subband(
            &mut self,
            job: J2kQuantizeSubbandJob<'_>,
        ) -> Result<Option<Vec<i32>>, &'static str> {
            self.encode_accelerator.encode_quantize_subband(job)
        }

        fn encode_tier1_code_block(
            &mut self,
            job: J2kTier1CodeBlockEncodeJob<'_>,
        ) -> Result<Option<EncodedJ2kCodeBlock>, &'static str> {
            self.encode_accelerator.encode_tier1_code_block(job)
        }

        fn encode_tier1_code_blocks(
            &mut self,
            jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
        ) -> Result<Option<Vec<EncodedJ2kCodeBlock>>, &'static str> {
            self.encode_accelerator.encode_tier1_code_blocks(jobs)
        }

        fn encode_ht_code_block(
            &mut self,
            job: crate::J2kHtCodeBlockEncodeJob<'_>,
        ) -> Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
            self.encode_accelerator.encode_ht_code_block(job)
        }

        fn encode_ht_code_blocks(
            &mut self,
            jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
        ) -> Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
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
        ) -> Result<Option<Vec<u8>>, &'static str> {
            self.encode_accelerator.encode_packetization(job)
        }
    };
}

impl<A: J2kEncodeStageAccelerator> J2kEncodeStageAccelerator for PrecomputedDwtAccelerator<'_, A> {
    fn encode_forward_dwt53(
        &mut self,
        _job: J2kForwardDwt53Job<'_>,
    ) -> Result<Option<J2kForwardDwt53Output>, &'static str> {
        if self.outputs.is_empty() {
            return Err("precomputed DWT output exhausted");
        }

        Ok(Some(self.outputs.remove(0)))
    }

    forward_precomputed_encode_stage_hooks!();
}

impl<A: J2kEncodeStageAccelerator> J2kEncodeStageAccelerator
    for PrecomputedDwt97Accelerator<'_, A>
{
    fn encode_forward_dwt97(
        &mut self,
        _job: J2kForwardDwt97Job<'_>,
    ) -> Result<Option<J2kForwardDwt97Output>, &'static str> {
        if self.outputs.is_empty() {
            return Err("precomputed DWT output exhausted");
        }

        Ok(Some(self.outputs.remove(0)))
    }

    forward_precomputed_encode_stage_hooks!();
}
