// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    encode_exact_native_color_batch_store_in_encoder,
    encode_prepared_direct_component_plane_in_encoder, encode_stacked_direct_component_plane_batch,
    Arc, BatchLayout, Buffer, DirectColorBatchCommandBuffers, DirectComponentPlaneRequest,
    DirectExecutionMetadata, DirectHybridStageTimings, DirectTier1Mode, Error,
    MetalImageDestination, MetalRuntime, PixelFormat, PreparedDirectColorPlan,
    StackedDirectComponentPlaneBatchRequest,
};

pub(super) fn color_component_plan_refs(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> Result<Vec<Vec<&super::PreparedDirectGrayscalePlan>>, Error> {
    let component_count = plans[0].component_plans.len();
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal stacked exact color plan references",
    );
    let mut grouped = budget.try_vec(
        component_count,
        "J2K Metal stacked exact color component reference groups",
    )?;
    for component_index in 0..component_count {
        let mut refs = budget.try_vec(
            plans.len(),
            "J2K Metal stacked exact color component references",
        )?;
        refs.extend(
            plans
                .iter()
                .map(|plan| &plan.component_plans[component_index]),
        );
        grouped.push(refs);
    }
    Ok(grouped)
}

#[cfg(target_os = "macos")]
pub(super) struct ColorGroupEncoder<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a metal::CommandBufferRef,
    pub(super) compute_encoder: &'a metal::ComputeCommandEncoderRef,
    pub(super) plans: &'a [Arc<PreparedDirectColorPlan>],
    pub(super) fmt: PixelFormat,
    pub(super) layout: BatchLayout,
    pub(super) destination: &'a MetalImageDestination,
    pub(super) source_indices: Option<&'a [usize]>,
    pub(super) metadata: &'a mut DirectExecutionMetadata,
    pub(super) stage_timings: DirectHybridStageTimings,
}

#[cfg(target_os = "macos")]
impl ColorGroupEncoder<'_> {
    pub(super) fn encode_coalesced(
        &mut self,
        component_plan_refs: &[Vec<&super::PreparedDirectGrayscalePlan>],
    ) -> Result<(), Error> {
        let broadcast = self
            .plans
            .first()
            .is_some_and(|first| self.plans.iter().all(|plan| Arc::ptr_eq(plan, first)));
        let status_start = self.metadata.status_checks.len();
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal stacked exact color output planes",
        );
        let mut planes = budget.try_vec(
            component_plan_refs.len(),
            "J2K Metal stacked exact color output planes",
        )?;
        if broadcast {
            self.encode_repeated_planes(&mut planes)?;
        } else {
            self.encode_distinct_planes(component_plan_refs, &mut planes)?;
        }
        encode_exact_native_color_batch_store_in_encoder(
            self.runtime,
            self.compute_encoder,
            &planes,
            &self.plans[0],
            self.fmt,
            self.layout,
            self.plans.len(),
            broadcast,
            0,
            self.destination,
        )?;
        self.remap_coalesced_status(status_start, broadcast)
    }

    fn encode_repeated_planes(&mut self, planes: &mut Vec<Buffer>) -> Result<(), Error> {
        // Identical prepared inputs share one immutable plan. Decode each
        // component once and broadcast those planes in the group final-store.
        for component in &self.plans[0].component_plans {
            planes.push(encode_prepared_direct_component_plane_in_encoder(
                DirectComponentPlaneRequest {
                    runtime: self.runtime,
                    command_buffer: self.command_buffer,
                    plan: component,
                    tier1_mode: component.tier1_prepare_mode,
                    stage_timings: &mut self.stage_timings,
                    retained_buffers: &mut self.metadata.retained_buffers,
                    status_checks: &mut self.metadata.status_checks,
                    scratch_buffers: &mut self.metadata.scratch_buffers,
                },
                self.compute_encoder,
            )?);
        }
        Ok(())
    }

    fn encode_distinct_planes(
        &mut self,
        component_plan_refs: &[Vec<&super::PreparedDirectGrayscalePlan>],
        planes: &mut Vec<Buffer>,
    ) -> Result<(), Error> {
        for (component_index, refs) in component_plan_refs.iter().enumerate() {
            let stacked = encode_stacked_direct_component_plane_batch(
                StackedDirectComponentPlaneBatchRequest {
                    runtime: self.runtime,
                    command_buffers: DirectColorBatchCommandBuffers::single(self.command_buffer),
                    compute_encoder: Some(self.compute_encoder),
                    plans: refs,
                    component_idx: component_index,
                    flattened_cpu_tier1_cache: None,
                    tier1_mode: DirectTier1Mode::Metal,
                    stage_timings: &mut self.stage_timings,
                    retained_buffers: &mut self.metadata.retained_buffers,
                    status_checks: &mut self.metadata.status_checks,
                    scratch_buffers: &mut self.metadata.scratch_buffers,
                },
            )?;
            if stacked.dimensions != self.plans[0].dimensions || stacked.count != self.plans.len() {
                return Err(Error::MetalStateInvariant {
                    state: "J2K Metal stacked exact color destination",
                    reason: "stacked component output does not match prepared group",
                });
            }
            planes.push(stacked.buffer);
        }
        Ok(())
    }

    fn remap_coalesced_status(&mut self, start: usize, broadcast: bool) -> Result<(), Error> {
        if broadcast {
            let source = self.source_indices.map_or(0, |indices| indices[0]);
            for status in &mut self.metadata.status_checks[start..] {
                status.remap_ht_source(source)?;
            }
        } else if let Some(sources) = self.source_indices {
            for status in &mut self.metadata.status_checks[start..] {
                status.remap_ht_sources(sources)?;
            }
        }
        Ok(())
    }

    pub(super) fn encode_individually(&mut self) -> Result<(), Error> {
        for (image_index, plan) in self.plans.iter().enumerate() {
            let status_start = self.metadata.status_checks.len();
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal exact color component plane handles",
            );
            let mut planes = budget.try_vec(
                plan.component_plans.len(),
                "J2K Metal exact color component plane handles",
            )?;
            for component in &plan.component_plans {
                planes.push(encode_prepared_direct_component_plane_in_encoder(
                    DirectComponentPlaneRequest {
                        runtime: self.runtime,
                        command_buffer: self.command_buffer,
                        plan: component,
                        tier1_mode: component.tier1_prepare_mode,
                        stage_timings: &mut self.stage_timings,
                        retained_buffers: &mut self.metadata.retained_buffers,
                        status_checks: &mut self.metadata.status_checks,
                        scratch_buffers: &mut self.metadata.scratch_buffers,
                    },
                    self.compute_encoder,
                )?);
            }
            encode_exact_native_color_batch_store_in_encoder(
                self.runtime,
                self.compute_encoder,
                &planes,
                plan,
                self.fmt,
                self.layout,
                1,
                false,
                image_index,
                self.destination,
            )?;
            let source = self
                .source_indices
                .map_or(image_index, |indices| indices[image_index]);
            for status in &mut self.metadata.status_checks[status_start..] {
                status.remap_ht_source(source)?;
            }
        }
        Ok(())
    }
}
