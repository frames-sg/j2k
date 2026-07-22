// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned-batch lifecycle for reusable native worker state.

use j2k_core::ScratchPool;

use super::BatchWorker;
use crate::CpuDecodeParallelism;

impl BatchWorker {
    pub(crate) fn new_owned(batch_size: usize) -> Self {
        Self::new(batch_size, None)
    }

    /// Prepare retained facade context and row scratch for one independent
    /// owned batch job while keeping nested native parallelism disabled.
    pub(crate) fn prepare_owned_decode(&mut self) -> CpuDecodeParallelism {
        self.ctx
            .set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        self.pool.reset();
        self.ctx.cpu_decode_parallelism()
    }

    pub(crate) fn take_native_workspace(&mut self) -> j2k_native::DecoderWorkspace {
        core::mem::take(&mut self.native_workspace)
    }

    pub(crate) fn restore_native_workspace(&mut self, workspace: j2k_native::DecoderWorkspace) {
        self.native_workspace = workspace;
    }

    pub(crate) fn build_prepared_htj2k_plan(
        &mut self,
        image: &j2k_native::Image<'_>,
        output_region: (u32, u32, u32, u32),
    ) -> j2k_native::Result<j2k_native::J2kReferencedHtj2kPlan> {
        self.record_preparation_call();
        let workspace = self.take_native_workspace();
        let mut context = j2k_native::DecoderContext::from_workspace(workspace);
        context.set_cpu_decode_parallelism(j2k_native::CpuDecodeParallelism::Serial);
        let result =
            image.build_referenced_htj2k_plan_region_with_context(&mut context, output_region);
        self.restore_native_workspace(context.into_workspace());
        result
    }

    pub(crate) fn build_prepared_classic_plan(
        &mut self,
        image: &j2k_native::Image<'_>,
        output_region: (u32, u32, u32, u32),
    ) -> j2k_native::Result<j2k_native::J2kReferencedClassicPlan> {
        self.record_preparation_call();
        let workspace = self.take_native_workspace();
        let mut context = j2k_native::DecoderContext::from_workspace(workspace);
        context.set_cpu_decode_parallelism(j2k_native::CpuDecodeParallelism::Serial);
        let result =
            image.build_referenced_classic_plan_region_with_context(&mut context, output_region);
        self.restore_native_workspace(context.into_workspace());
        result
    }

    fn record_preparation_call(&mut self) {
        if self.preparation_calls != 0 {
            self.preparation_worker_reuses = self.preparation_worker_reuses.saturating_add(1);
        }
        self.preparation_calls = self.preparation_calls.saturating_add(1);
    }

    pub(crate) const fn preparation_calls(&self) -> u64 {
        self.preparation_calls
    }

    pub(crate) const fn preparation_worker_reuses(&self) -> u64 {
        self.preparation_worker_reuses
    }

    pub(crate) fn native_workspace_stats(&self) -> j2k_native::DecoderWorkspaceStats {
        self.native_workspace.stats()
    }

    pub(crate) fn execute_prepared_htj2k_plan<'scratch>(
        &'scratch mut self,
        plan: &j2k_native::J2kReferencedHtj2kPlan,
        encoded_input: &[u8],
        signed: bool,
    ) -> j2k_native::Result<j2k_native::J2kDirectDecodedComponents<'scratch>> {
        self.prepared_plan_decode_calls = self.prepared_plan_decode_calls.saturating_add(1);
        j2k_native::execute_referenced_htj2k_plan(
            plan,
            encoded_input,
            signed,
            &mut self.prepared_plan_scratch,
        )
    }

    pub(crate) fn execute_prepared_classic_plan<'scratch>(
        &'scratch mut self,
        plan: &j2k_native::J2kReferencedClassicPlan,
        encoded_input: &[u8],
        signed: bool,
    ) -> j2k_native::Result<j2k_native::J2kDirectDecodedComponents<'scratch>> {
        self.prepared_plan_decode_calls = self.prepared_plan_decode_calls.saturating_add(1);
        j2k_native::execute_referenced_classic_plan(
            plan,
            encoded_input,
            signed,
            &mut self.prepared_plan_scratch,
        )
    }

    pub(crate) const fn prepared_plan_decode_calls(&self) -> u64 {
        self.prepared_plan_decode_calls
    }

    pub(crate) fn prepare_staged_htj2k_plan(
        &mut self,
        plan: &j2k_native::J2kReferencedHtj2kPlan,
        image_scratch: &mut j2k_native::J2kDirectCpuScratch,
    ) -> j2k_native::Result<()> {
        self.prepared_plan_decode_calls = self.prepared_plan_decode_calls.saturating_add(1);
        j2k_native::prepare_referenced_htj2k_staged(
            plan,
            image_scratch,
            &mut self.prepared_entropy_workspace,
        )
    }

    pub(crate) fn prepare_staged_classic_plan(
        &mut self,
        plan: &j2k_native::J2kReferencedClassicPlan,
        image_scratch: &mut j2k_native::J2kDirectCpuScratch,
    ) -> j2k_native::Result<()> {
        self.prepared_plan_decode_calls = self.prepared_plan_decode_calls.saturating_add(1);
        j2k_native::prepare_referenced_classic_staged(
            plan,
            image_scratch,
            &mut self.prepared_entropy_workspace,
        )
    }

    pub(crate) fn execute_staged_htj2k_job(
        &mut self,
        plan: &j2k_native::J2kReferencedHtj2kPlan,
        index: j2k_native::J2kDirectCodeBlockIndex,
        payload_arena: &[u8],
        payload: j2k_native::HtCodeBlockPayloadRanges,
        image_scratch: &mut j2k_native::J2kDirectCpuScratch,
    ) -> j2k_native::Result<()> {
        j2k_native::execute_referenced_htj2k_entropy_job(
            plan,
            index,
            payload_arena,
            payload,
            image_scratch,
            &mut self.prepared_entropy_workspace,
        )
    }

    pub(crate) fn prepare_staged_htj2k_entropy_workspace(
        &mut self,
        plan: &j2k_native::J2kReferencedHtj2kPlan,
    ) -> j2k_native::Result<()> {
        j2k_native::prepare_referenced_htj2k_entropy_workspace(
            plan,
            &mut self.prepared_entropy_workspace,
        )
    }

    pub(crate) fn prepare_staged_classic_entropy_workspace(
        &mut self,
        plan: &j2k_native::J2kReferencedClassicPlan,
    ) -> j2k_native::Result<()> {
        j2k_native::prepare_referenced_classic_entropy_workspace(
            plan,
            &mut self.prepared_entropy_workspace,
        )
    }

    pub(crate) fn execute_staged_classic_job(
        &mut self,
        plan: &j2k_native::J2kReferencedClassicPlan,
        index: j2k_native::J2kDirectCodeBlockIndex,
        payload_arena: &[u8],
        payload: j2k_native::J2kCodestreamRange,
        image_scratch: &mut j2k_native::J2kDirectCpuScratch,
    ) -> j2k_native::Result<()> {
        j2k_native::execute_referenced_classic_entropy_job(
            plan,
            index,
            payload_arena,
            payload,
            image_scratch,
            &mut self.prepared_entropy_workspace,
        )
    }

    pub(crate) fn prepared_plan_ht_workspace_bytes(&self) -> usize {
        self.prepared_plan_scratch
            .retained_ht_workspace_bytes()
            .saturating_add(self.prepared_entropy_workspace.retained_ht_bytes())
    }

    pub(crate) fn prepared_plan_classic_workspace_bytes(&self) -> usize {
        self.prepared_plan_scratch
            .retained_classic_workspace_bytes()
            .saturating_add(self.prepared_entropy_workspace.retained_classic_bytes())
    }
}
