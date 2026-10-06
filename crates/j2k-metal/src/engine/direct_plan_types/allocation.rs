// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actual retained host/device accounting for prepared direct-plan owners.

use core::mem::size_of;

use super::{
    ClassicTier1Inputs, PreparedClassicSubBand, PreparedClassicSubBandGroup,
    PreparedDirectColorPlan, PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep,
    PreparedHtExecutionOwner, PreparedHtPayloadSource, PreparedHtSubBand, PreparedHtSubBandGroup,
};
use crate::engine::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob};
use crate::engine::direct_tier1_input_buffer_bytes;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PreparedPlanRetainedBytes {
    pub(crate) host: usize,
    pub(crate) device: usize,
}

impl PreparedPlanRetainedBytes {
    fn include_host_capacity<T>(&mut self, capacity: usize) -> Result<(), &'static str> {
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .ok_or("prepared-plan host capacity overflow")?;
        self.host = self
            .host
            .checked_add(bytes)
            .ok_or("prepared-plan aggregate host byte overflow")?;
        Ok(())
    }

    fn include_host_bytes(&mut self, bytes: usize) -> Result<(), &'static str> {
        self.host = self
            .host
            .checked_add(bytes)
            .ok_or("prepared-plan aggregate host byte overflow")?;
        Ok(())
    }

    fn include_device_bytes(&mut self, bytes: usize) -> Result<(), &'static str> {
        self.device = self
            .device
            .checked_add(bytes)
            .ok_or("prepared-plan aggregate device byte overflow")?;
        Ok(())
    }

    /// Charges the Tier-1 input buffers whether or not a reader has built them
    /// yet. The cache weighs a plan once, at insertion, so this keeps buffers
    /// built later inside its device cap.
    fn include_classic_tier1_inputs(
        &mut self,
        inputs: &ClassicTier1Inputs,
        coded_data: &[u8],
        jobs: &[J2kClassicCleanupBatchJob],
        segments: &[J2kClassicSegment],
    ) -> Result<(), &'static str> {
        self.include_device_bytes(direct_tier1_input_buffer_bytes(coded_data, inputs.mode))?;
        self.include_device_bytes(direct_tier1_input_buffer_bytes(jobs, inputs.mode))?;
        self.include_device_bytes(direct_tier1_input_buffer_bytes(segments, inputs.mode))
    }
}

impl PreparedDirectGrayscalePlan {
    pub(crate) fn retained_cache_bytes(&self) -> Result<PreparedPlanRetainedBytes, &'static str> {
        let mut retained = PreparedPlanRetainedBytes::default();
        include_grayscale_plan(&mut retained, self)?;
        Ok(retained)
    }
}

impl PreparedDirectColorPlan {
    pub(crate) fn retained_cache_bytes(&self) -> Result<PreparedPlanRetainedBytes, &'static str> {
        let mut retained = PreparedPlanRetainedBytes::default();
        retained.include_host_capacity::<PreparedDirectGrayscalePlan>(
            self.component_plans.capacity(),
        )?;
        for component in &self.component_plans {
            include_grayscale_plan(&mut retained, component)?;
        }
        Ok(retained)
    }

    pub(crate) fn disable_dynamic_cpu_tier1_retention(&self) -> Result<(), crate::Error> {
        for component in &self.component_plans {
            component.disable_cpu_tier1_retention()?;
        }
        Ok(())
    }
}

fn include_grayscale_plan(
    retained: &mut PreparedPlanRetainedBytes,
    plan: &PreparedDirectGrayscalePlan,
) -> Result<(), &'static str> {
    retained.include_host_capacity::<PreparedDirectGrayscaleStep>(plan.steps.capacity())?;
    retained
        .include_host_capacity::<PreparedClassicSubBandGroup>(plan.classic_groups.capacity())?;
    retained.include_host_capacity::<PreparedHtSubBandGroup>(plan.ht_groups.capacity())?;
    retained.include_host_bytes(
        size_of::<crate::engine::CpuTier1CoefficientCache>() + 2 * size_of::<usize>(),
    )?;
    retained.include_host_bytes(plan.cpu_tier1_cache.retained_cache_bytes()?)?;

    for step in &plan.steps {
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                include_classic_sub_band(retained, sub_band)?;
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                include_ht_sub_band(retained, sub_band)?;
            }
            PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => {}
        }
    }
    for group in &plan.classic_groups {
        include_classic_group(retained, group)?;
    }
    for group in &plan.ht_groups {
        include_ht_group(retained, group)?;
    }
    Ok(())
}

fn include_classic_sub_band(
    retained: &mut PreparedPlanRetainedBytes,
    sub_band: &PreparedClassicSubBand,
) -> Result<(), &'static str> {
    retained.include_host_capacity::<u8>(sub_band.coded_data.capacity())?;
    retained.include_host_capacity::<J2kClassicCleanupBatchJob>(sub_band.jobs.capacity())?;
    retained.include_host_capacity::<J2kClassicSegment>(sub_band.segments.capacity())?;
    retained.include_classic_tier1_inputs(
        &sub_band.tier1_inputs,
        &sub_band.coded_data,
        &sub_band.jobs,
        &sub_band.segments,
    )
}

fn include_classic_group(
    retained: &mut PreparedPlanRetainedBytes,
    group: &PreparedClassicSubBandGroup,
) -> Result<(), &'static str> {
    retained.include_host_capacity::<u8>(group.coded_data.capacity())?;
    retained.include_host_capacity::<J2kClassicCleanupBatchJob>(group.jobs.capacity())?;
    retained.include_host_capacity::<J2kClassicSegment>(group.segments.capacity())?;
    retained.include_host_capacity::<super::PreparedClassicSubBandGroupMember>(
        group.members.capacity(),
    )?;
    retained.include_classic_tier1_inputs(
        &group.tier1_inputs,
        &group.coded_data,
        &group.jobs,
        &group.segments,
    )
}

fn include_ht_sub_band(
    retained: &mut PreparedPlanRetainedBytes,
    sub_band: &PreparedHtSubBand,
) -> Result<(), &'static str> {
    include_ht_payload_source(retained, &sub_band.payload_source)?;
    retained.include_host_capacity::<J2kHtCleanupBatchJob>(sub_band.jobs.capacity())?;
    retained.include_host_bytes(size_of::<PreparedHtExecutionOwner>() + 2 * size_of::<usize>())
}

fn include_ht_group(
    retained: &mut PreparedPlanRetainedBytes,
    group: &PreparedHtSubBandGroup,
) -> Result<(), &'static str> {
    include_ht_payload_source(retained, &group.payload_source)?;
    retained.include_host_capacity::<J2kHtCleanupBatchJob>(group.jobs.capacity())?;
    retained
        .include_host_capacity::<super::PreparedHtSubBandGroupMember>(group.members.capacity())?;
    retained.include_host_bytes(size_of::<PreparedHtExecutionOwner>() + 2 * size_of::<usize>())
}

fn include_ht_payload_source(
    retained: &mut PreparedPlanRetainedBytes,
    payload_source: &PreparedHtPayloadSource,
) -> Result<(), &'static str> {
    match payload_source {
        PreparedHtPayloadSource::Contiguous(data) => {
            retained.include_host_capacity::<u8>(data.capacity())
        }
        PreparedHtPayloadSource::Referenced { ranges, .. } => {
            retained
                .include_host_capacity::<j2k_native::HtCodeBlockPayloadRanges>(ranges.capacity())
        }
    }
}
