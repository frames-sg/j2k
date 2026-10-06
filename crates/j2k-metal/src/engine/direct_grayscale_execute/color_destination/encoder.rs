// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use super::{
    encode_exact_native_color_batch_store_in_encoder,
    encode_prepared_direct_component_plane_in_encoder, encode_stacked_direct_component_plane_batch,
    encode_stacked_direct_component_plane_batch_with_shared_tier1,
    supports_stacked_direct_component_plane_batch, Arc, BatchLayout, Buffer,
    DirectColorBatchCommandBuffers, DirectComponentPlaneRequest, DirectExecutionMetadata,
    DirectHybridStageTimings, DirectTier1Mode, Error, MetalImageDestination, MetalRuntime,
    PixelFormat, PreparedDirectColorPlan, StackedDirectComponentPlaneBatchRequest,
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

fn stackable_rgb_component_plans(
    plan: &PreparedDirectColorPlan,
    channel_count: usize,
) -> Option<[&super::PreparedDirectGrayscalePlan; 3]> {
    if channel_count != 3 || plan.component_plans.len() != 3 {
        return None;
    }
    let components = [
        &plan.component_plans[0],
        &plan.component_plans[1],
        &plan.component_plans[2],
    ];
    (components
        .iter()
        .all(|component| component.tier1_prepare_mode == DirectTier1Mode::Metal)
        && supports_stacked_direct_component_plane_batch(&components))
    .then_some(components)
}

#[cfg(target_os = "macos")]
pub(super) struct ColorGroupEncoder<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a crate::metal_types::CommandBufferRef,
    pub(super) compute_encoder: &'a crate::metal_types::ComputeCommandEncoderRef,
    pub(super) plans: &'a [Arc<PreparedDirectColorPlan>],
    pub(super) fmt: PixelFormat,
    pub(super) layout: BatchLayout,
    pub(super) destination: &'a MetalImageDestination,
    pub(super) source_indices: Option<&'a [usize]>,
    pub(super) metadata: &'a mut DirectExecutionMetadata,
    pub(super) stage_timings: DirectHybridStageTimings,
    pub(super) cpu_cache: Option<&'a super::super::super::FlattenedCpuTier1Cache>,
    pub(super) shared_tier1: Option<super::super::super::SharedClassicTier1Group>,
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
        let partially_repeated = !broadcast
            && self.plans.iter().enumerate().any(|(index, plan)| {
                self.plans[..index]
                    .iter()
                    .any(|seen| Arc::ptr_eq(seen, plan))
            });
        if partially_repeated {
            return self.encode_partially_repeated();
        }
        let status_start = self.metadata.status_checks.len();
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal stacked exact color output planes",
        );
        let mut planes = budget.try_vec(
            component_plan_refs.len(),
            "J2K Metal stacked exact color output planes",
        )?;
        let stacked_rgb = if broadcast {
            self.encode_repeated_planes(&mut planes)?
        } else {
            self.encode_distinct_planes(component_plan_refs, self.plans.len(), &mut planes)?
        };
        let source_plane_count = if broadcast { 1 } else { self.plans.len() };
        encode_exact_native_color_batch_store_in_encoder(
            self.runtime,
            self.compute_encoder,
            &planes,
            &self.plans[0],
            super::store::NativeColorStoreConfig {
                format: self.fmt,
                layout: self.layout,
                image_count: self.plans.len(),
                broadcast_planes: broadcast,
                destination_image_index: 0,
                source_image_index: 0,
                source_plane_base_indices: if stacked_rgb {
                    [0, source_plane_count, 2 * source_plane_count, 0]
                } else {
                    [0; 4]
                },
            },
            self.destination,
        )?;
        self.remap_coalesced_status(status_start, broadcast, stacked_rgb)?;
        let report_component_count = if stacked_rgb {
            1
        } else {
            self.plans[0].component_plans.len()
        };
        for component in &self.plans[0].component_plans[..report_component_count] {
            self.metadata.dispatch_report.idwt = self.metadata.dispatch_report.idwt.saturating_add(
                component
                    .steps
                    .iter()
                    .filter(|step| {
                        matches!(step, super::super::PreparedDirectGrayscaleStep::Idwt(_))
                    })
                    .count(),
            );
        }
        self.metadata.dispatch_report.mct = self
            .metadata
            .dispatch_report
            .mct
            .saturating_add(usize::from(self.plans[0].mct));
        self.metadata.dispatch_report.color_output =
            self.metadata.dispatch_report.color_output.saturating_add(1);
        Ok(())
    }

    fn encode_partially_repeated(&mut self) -> Result<(), Error> {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal partially repeated exact color batch",
        );
        let mut unique_plans = budget.try_vec(
            self.plans.len(),
            "J2K Metal partially repeated exact color unique plans",
        )?;
        let mut output_to_unique = budget.try_vec(
            self.plans.len(),
            "J2K Metal partially repeated exact color output indices",
        )?;
        let mut unique_sources = budget.try_vec(
            self.plans.len(),
            "J2K Metal partially repeated exact color source indices",
        )?;
        for (output_index, plan) in self.plans.iter().enumerate() {
            if let Some(unique_index) = unique_plans
                .iter()
                .position(|unique| Arc::ptr_eq(unique, plan))
            {
                output_to_unique.push(unique_index);
            } else {
                output_to_unique.push(unique_plans.len());
                unique_sources.push(
                    self.source_indices
                        .map_or(output_index, |indices| indices[output_index]),
                );
                unique_plans.push(plan.clone());
            }
        }

        let component_plan_refs = color_component_plan_refs(&unique_plans)?;
        let status_start = self.metadata.status_checks.len();
        let mut planes = budget.try_vec(
            component_plan_refs.len(),
            "J2K Metal partially repeated exact color output planes",
        )?;
        let stacked_rgb =
            self.encode_distinct_planes(&component_plan_refs, unique_plans.len(), &mut planes)?;
        for (destination_image_index, source_image_index) in
            output_to_unique.into_iter().enumerate()
        {
            encode_exact_native_color_batch_store_in_encoder(
                self.runtime,
                self.compute_encoder,
                &planes,
                &self.plans[destination_image_index],
                super::store::NativeColorStoreConfig {
                    format: self.fmt,
                    layout: self.layout,
                    image_count: 1,
                    broadcast_planes: false,
                    destination_image_index,
                    source_image_index,
                    source_plane_base_indices: if stacked_rgb {
                        [0, unique_plans.len(), 2 * unique_plans.len(), 0]
                    } else {
                        [0; 4]
                    },
                },
                self.destination,
            )?;
        }
        if stacked_rgb {
            let count = unique_sources
                .len()
                .checked_mul(3)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal stacked RGB source count overflow".to_string(),
                })?;
            let mut sources = budget.try_vec(count, "J2K Metal stacked RGB source indices")?;
            for _ in 0..3 {
                sources.extend_from_slice(&unique_sources);
            }
            unique_sources = sources;
        }
        for status in &mut self.metadata.status_checks[status_start..] {
            status.remap_sources(&unique_sources)?;
        }
        self.report_partially_repeated_dispatches(stacked_rgb);
        Ok(())
    }

    fn report_partially_repeated_dispatches(&mut self, stacked_rgb: bool) {
        let report_component_count = if stacked_rgb {
            1
        } else {
            self.plans[0].component_plans.len()
        };
        for component in &self.plans[0].component_plans[..report_component_count] {
            self.metadata.dispatch_report.idwt = self.metadata.dispatch_report.idwt.saturating_add(
                component
                    .steps
                    .iter()
                    .filter(|step| {
                        matches!(step, super::super::PreparedDirectGrayscaleStep::Idwt(_))
                    })
                    .count(),
            );
        }
        self.metadata.dispatch_report.mct = self.metadata.dispatch_report.mct.saturating_add(
            self.plans
                .len()
                .saturating_mul(usize::from(self.plans[0].mct)),
        );
        self.metadata.dispatch_report.color_output = self
            .metadata
            .dispatch_report
            .color_output
            .saturating_add(self.plans.len());
    }

    fn encode_repeated_planes(&mut self, planes: &mut Vec<Buffer>) -> Result<bool, Error> {
        // Identical prepared inputs share one immutable plan. Decode each
        // component once and broadcast those planes in the group final-store.
        let plan = self.plans[0].clone();
        if let Some(component_plans) = stackable_rgb_component_plans(&plan, self.fmt.channels()) {
            planes.extend(self.encode_stacked_rgb_planes(&plan, &component_plans)?);
            return Ok(true);
        }
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
        Ok(false)
    }

    fn encode_distinct_planes(
        &mut self,
        component_plan_refs: &[Vec<&super::PreparedDirectGrayscalePlan>],
        expected_count: usize,
        planes: &mut Vec<Buffer>,
    ) -> Result<bool, Error> {
        if self.cpu_cache.is_none() && self.fmt.channels() == 3 && component_plan_refs.len() == 3 {
            let count = expected_count
                .checked_mul(3)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal stacked RGB component count overflow".to_string(),
                })?;
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal stacked RGB component references",
            );
            let mut combined =
                budget.try_vec(count, "J2K Metal stacked RGB component references")?;
            combined.extend(component_plan_refs.iter().flatten().copied());
            if combined.len() == count
                && combined
                    .iter()
                    .all(|component| component.tier1_prepare_mode == DirectTier1Mode::Metal)
                && supports_stacked_direct_component_plane_batch(&combined)
            {
                let plan = self.plans[0].clone();
                planes.extend(self.encode_stacked_rgb_planes(&plan, &combined)?);
                return Ok(true);
            }
        }
        for (component_index, refs) in component_plan_refs.iter().enumerate() {
            let stacked = encode_stacked_direct_component_plane_batch(
                StackedDirectComponentPlaneBatchRequest {
                    runtime: self.runtime,
                    command_buffers: DirectColorBatchCommandBuffers::single(self.command_buffer),
                    compute_encoder: Some(self.compute_encoder),
                    plans: refs,
                    component_idx: component_index,
                    flattened_cpu_tier1_cache: self.cpu_cache,
                    tier1_mode: if self.cpu_cache.is_some() {
                        DirectTier1Mode::CpuUpload
                    } else {
                        DirectTier1Mode::Metal
                    },
                    stage_timings: &mut self.stage_timings,
                    retained_buffers: &mut self.metadata.retained_buffers,
                    status_checks: &mut self.metadata.status_checks,
                    scratch_buffers: &mut self.metadata.scratch_buffers,
                },
            )?;
            if stacked.dimensions != self.plans[0].dimensions || stacked.count != expected_count {
                return Err(Error::MetalStateInvariant {
                    state: "J2K Metal stacked exact color destination",
                    reason: "stacked component output does not match prepared group",
                });
            }
            planes.push(stacked.buffer);
        }
        Ok(false)
    }

    fn remap_coalesced_status(
        &mut self,
        start: usize,
        broadcast: bool,
        stacked_rgb: bool,
    ) -> Result<(), Error> {
        if broadcast {
            let source = self.source_indices.map_or(0, |indices| indices[0]);
            for status in &mut self.metadata.status_checks[start..] {
                status.remap_source(source)?;
            }
        } else if stacked_rgb {
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal stacked RGB source indices",
            );
            let count = self
                .plans
                .len()
                .checked_mul(3)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal stacked RGB source count overflow".to_string(),
                })?;
            let mut sources = budget.try_vec(count, "J2K Metal stacked RGB source indices")?;
            for _ in 0..3 {
                sources.extend(
                    (0..self.plans.len())
                        .map(|index| self.source_indices.map_or(index, |indices| indices[index])),
                );
            }
            for status in &mut self.metadata.status_checks[start..] {
                status.remap_sources(&sources)?;
            }
        } else if let Some(sources) = self.source_indices {
            for status in &mut self.metadata.status_checks[start..] {
                status.remap_sources(sources)?;
            }
        }
        Ok(())
    }

    pub(super) fn encode_individually(&mut self) -> Result<(), Error> {
        for image_index in 0..self.plans.len() {
            let plan = self.plans[image_index].clone();
            if let Some(component_plans) = stackable_rgb_component_plans(&plan, self.fmt.channels())
            {
                self.encode_single_stacked_rgb(image_index, &plan, &component_plans)?;
                continue;
            }
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
                &plan,
                super::store::NativeColorStoreConfig {
                    format: self.fmt,
                    layout: self.layout,
                    image_count: 1,
                    broadcast_planes: false,
                    destination_image_index: image_index,
                    source_image_index: 0,
                    source_plane_base_indices: [0; 4],
                },
                self.destination,
            )?;
            let source = self
                .source_indices
                .map_or(image_index, |indices| indices[image_index]);
            for status in &mut self.metadata.status_checks[status_start..] {
                status.remap_source(source)?;
            }
            for component in &plan.component_plans {
                self.metadata.dispatch_report.idwt =
                    self.metadata.dispatch_report.idwt.saturating_add(
                        component
                            .steps
                            .iter()
                            .filter(|step| {
                                matches!(step, super::super::PreparedDirectGrayscaleStep::Idwt(_))
                            })
                            .count(),
                    );
            }
            self.metadata.dispatch_report.mct = self
                .metadata
                .dispatch_report
                .mct
                .saturating_add(usize::from(plan.mct));
            self.metadata.dispatch_report.color_output =
                self.metadata.dispatch_report.color_output.saturating_add(1);
        }
        Ok(())
    }

    fn encode_single_stacked_rgb(
        &mut self,
        image_index: usize,
        plan: &PreparedDirectColorPlan,
        component_plans: &[&super::PreparedDirectGrayscalePlan; 3],
    ) -> Result<(), Error> {
        let status_start = self.metadata.status_checks.len();
        let planes = self.encode_stacked_rgb_planes(plan, component_plans)?;
        encode_exact_native_color_batch_store_in_encoder(
            self.runtime,
            self.compute_encoder,
            &planes,
            plan,
            super::store::NativeColorStoreConfig {
                format: self.fmt,
                layout: self.layout,
                image_count: 1,
                broadcast_planes: false,
                destination_image_index: image_index,
                source_image_index: 0,
                source_plane_base_indices: [0, 1, 2, 0],
            },
            self.destination,
        )?;
        let source = self
            .source_indices
            .map_or(image_index, |indices| indices[image_index]);
        for status in &mut self.metadata.status_checks[status_start..] {
            status.remap_source(source)?;
        }
        self.metadata.dispatch_report.idwt = self.metadata.dispatch_report.idwt.saturating_add(
            component_plans[0]
                .steps
                .iter()
                .filter(|step| matches!(step, super::super::PreparedDirectGrayscaleStep::Idwt(_)))
                .count(),
        );
        self.metadata.dispatch_report.mct = self
            .metadata
            .dispatch_report
            .mct
            .saturating_add(usize::from(plan.mct));
        self.metadata.dispatch_report.color_output =
            self.metadata.dispatch_report.color_output.saturating_add(1);
        Ok(())
    }

    fn encode_stacked_rgb_planes(
        &mut self,
        plan: &PreparedDirectColorPlan,
        component_plans: &[&super::PreparedDirectGrayscalePlan],
    ) -> Result<[Buffer; 3], Error> {
        let request = StackedDirectComponentPlaneBatchRequest {
            runtime: self.runtime,
            command_buffers: DirectColorBatchCommandBuffers::single(self.command_buffer),
            compute_encoder: Some(self.compute_encoder),
            plans: component_plans,
            component_idx: 0,
            flattened_cpu_tier1_cache: None,
            tier1_mode: DirectTier1Mode::Metal,
            stage_timings: &mut self.stage_timings,
            retained_buffers: &mut self.metadata.retained_buffers,
            status_checks: &mut self.metadata.status_checks,
            scratch_buffers: &mut self.metadata.scratch_buffers,
        };
        let stacked = match self.shared_tier1.as_mut() {
            Some(shared) => {
                encode_stacked_direct_component_plane_batch_with_shared_tier1(request, shared)?
            }
            None => encode_stacked_direct_component_plane_batch(request)?,
        };
        if stacked.dimensions != plan.dimensions || stacked.count != component_plans.len() {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal stacked exact RGB destination",
                reason: "stacked component output does not match prepared image",
            });
        }
        Ok([
            stacked.buffer.clone(),
            stacked.buffer.clone(),
            stacked.buffer,
        ])
    }
}
