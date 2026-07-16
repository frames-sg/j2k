// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CudaClassicCodeBlock, CudaClassicSegment, CudaClassicSubband, CudaHtj2kCodeBlock,
    CudaHtj2kDecodePlan, CudaHtj2kIdwtStep, CudaHtj2kStoreStep, CudaHtj2kSubband,
    CudaHtj2kTransform, Error, HostPhaseBudget, PixelFormat, PLAN_PAYLOAD_TOO_LARGE,
};

impl CudaHtj2kDecodePlan {
    #[cfg(feature = "cuda-runtime")]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "output dimensions accessor supports CUDA plan tests"
        )
    )]
    pub(crate) fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    #[cfg(feature = "cuda-runtime")]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "output format accessor supports CUDA plan tests")
    )]
    pub(crate) fn output_format(&self) -> PixelFormat {
        self.output_format
    }

    #[cfg(feature = "cuda-runtime")]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "output origin accessor supports CUDA plan tests")
    )]
    pub(crate) fn output_origin(&self) -> (u32, u32) {
        self.output_origin
    }

    #[cfg(feature = "cuda-runtime")]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "transform accessor supports CUDA plan tests")
    )]
    pub(crate) fn transform(&self) -> CudaHtj2kTransform {
        self.transform
    }

    pub(crate) fn payload(&self) -> &[u8] {
        &self.payload
    }

    #[cfg(test)]
    pub(crate) fn append_payload_to_shared(
        &mut self,
        shared_payload: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let mut host_budget = HostPhaseBudget::new("CUDA direct-plan shared payload");
        host_budget.account_vec(shared_payload)?;
        host_budget.account_vec(&self.payload)?;
        self.append_payload_to_shared_with_budget(shared_payload, &mut host_budget)
    }

    #[cfg(any(feature = "cuda-runtime", test))]
    pub(crate) fn append_payload_to_shared_with_budget(
        &mut self,
        shared_payload: &mut Vec<u8>,
        host_budget: &mut HostPhaseBudget,
    ) -> Result<(), Error> {
        let base =
            u64::try_from(shared_payload.len()).map_err(|_| Error::UnsupportedCudaRequest {
                reason: PLAN_PAYLOAD_TOO_LARGE,
            })?;
        shared_payload.len().checked_add(self.payload.len()).ok_or(
            Error::UnsupportedCudaRequest {
                reason: PLAN_PAYLOAD_TOO_LARGE,
            },
        )?;
        if !shared_payload.is_empty() {
            host_budget.try_vec_reserve(shared_payload, self.payload.len())?;
        }
        for block in &mut self.code_blocks {
            block.payload_offset =
                block
                    .payload_offset
                    .checked_add(base)
                    .ok_or(Error::UnsupportedCudaRequest {
                        reason: PLAN_PAYLOAD_TOO_LARGE,
                    })?;
        }
        for block in &mut self.classic_code_blocks {
            block.payload_offset =
                block
                    .payload_offset
                    .checked_add(base)
                    .ok_or(Error::UnsupportedCudaRequest {
                        reason: PLAN_PAYLOAD_TOO_LARGE,
                    })?;
        }
        if shared_payload.is_empty() {
            *shared_payload = core::mem::take(&mut self.payload);
        } else {
            let mut payload = core::mem::take(&mut self.payload);
            shared_payload.append(&mut payload);
        }
        Ok(())
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn account_host_owners(&self, budget: &mut HostPhaseBudget) -> Result<(), Error> {
        budget.account_vec(&self.payload)?;
        budget.account_vec(&self.code_blocks)?;
        budget.account_vec(&self.classic_code_blocks)?;
        budget.account_vec(&self.classic_segments)?;
        budget.account_vec(&self.classic_subbands)?;
        budget.account_vec(&self.subbands)?;
        budget.account_vec(&self.idwt_steps)?;
        budget.account_vec(&self.store_steps)?;
        Ok(())
    }

    #[cfg_attr(
        all(not(feature = "cuda-runtime"), not(test)),
        expect(
            dead_code,
            reason = "payload rebasing is used only by CUDA batch decode"
        )
    )]
    pub(crate) fn rebase_payload_offsets(&mut self, base: u64) -> Result<(), Error> {
        for block in &mut self.code_blocks {
            block.payload_offset =
                block
                    .payload_offset
                    .checked_add(base)
                    .ok_or(Error::UnsupportedCudaRequest {
                        reason: PLAN_PAYLOAD_TOO_LARGE,
                    })?;
        }
        for block in &mut self.classic_code_blocks {
            block.payload_offset =
                block
                    .payload_offset
                    .checked_add(base)
                    .ok_or(Error::UnsupportedCudaRequest {
                        reason: PLAN_PAYLOAD_TOO_LARGE,
                    })?;
        }
        Ok(())
    }

    pub(crate) fn code_blocks(&self) -> &[CudaHtj2kCodeBlock] {
        &self.code_blocks
    }

    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "classic block metadata is consumed only by CUDA decode routes"
        )
    )]
    pub(crate) fn classic_code_blocks(&self) -> &[CudaClassicCodeBlock] {
        &self.classic_code_blocks
    }

    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "classic segment metadata is consumed only by CUDA decode routes"
        )
    )]
    pub(crate) fn classic_segments(&self) -> &[CudaClassicSegment] {
        &self.classic_segments
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn classic_subbands(&self) -> &[CudaClassicSubband] {
        &self.classic_subbands
    }

    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "combined block counts are consumed only by CUDA decode routes"
        )
    )]
    pub(crate) fn block_count(&self) -> usize {
        self.code_blocks.len() + self.classic_code_blocks.len()
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn subbands(&self) -> &[CudaHtj2kSubband] {
        &self.subbands
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn idwt_steps(&self) -> &[CudaHtj2kIdwtStep] {
        &self.idwt_steps
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn store_steps(&self) -> &[CudaHtj2kStoreStep] {
        &self.store_steps
    }

    #[cfg(feature = "cuda-runtime")]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "dispatch hint accessor supports CUDA plan tests")
    )]
    pub(crate) fn dispatch_count_hint(&self) -> usize {
        self.block_count()
    }
}
