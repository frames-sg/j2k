// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate ownership for flattened multi-component CUDA decode plans.

use j2k_core::PixelFormat;
use j2k_native::J2kDirectColorPlan;

use crate::allocation::HostPhaseBudget;
use crate::{CudaHtj2kDecodePlan, Error};

use super::super::CudaHtj2kColorDecodePlans;

type OutputRegion = ((u32, u32), (u32, u32));

pub(super) fn flatten_cuda_color_components(
    native_plan: &J2kDirectColorPlan,
    format: PixelFormat,
    output_region: Option<OutputRegion>,
    what: &'static str,
) -> Result<(Vec<u8>, Vec<CudaHtj2kDecodePlan>), Error> {
    let mut initial_budget = HostPhaseBudget::new(what);
    let mut components = initial_budget.try_vec_with_capacity(native_plan.component_plans.len())?;
    let mut payload = Vec::new();

    for component_plan in &native_plan.component_plans {
        let mut component = match output_region {
            Some((origin, dimensions)) => CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
                component_plan,
                format,
                origin,
                dimensions,
            )?,
            None => {
                CudaHtj2kDecodePlan::from_grayscale_direct_plan(component_plan, format, (0, 0))?
            }
        };

        let mut append_budget =
            color_owner_graph_budget(&payload, &components, Some(&component), what)?;
        component.append_payload_to_shared_with_budget(&mut payload, &mut append_budget)?;
        components.push(component);
        color_owner_graph_budget(&payload, &components, None, what)?;
    }

    Ok((payload, components))
}

fn color_owner_graph_budget(
    payload: &Vec<u8>,
    components: &Vec<CudaHtj2kDecodePlan>,
    pending: Option<&CudaHtj2kDecodePlan>,
    what: &'static str,
) -> Result<HostPhaseBudget, Error> {
    let mut budget = HostPhaseBudget::new(what);
    budget.account_vec(payload)?;
    budget.account_vec(components)?;
    for component in components {
        component.account_host_owners(&mut budget)?;
    }
    if let Some(component) = pending {
        component.account_host_owners(&mut budget)?;
    }
    Ok(budget)
}

impl CudaHtj2kColorDecodePlans {
    pub(in crate::decoder) fn account_host_owners(
        &self,
        budget: &mut HostPhaseBudget,
    ) -> Result<(), Error> {
        budget.account_vec(&self.payload)?;
        budget.account_vec(&self.components)?;
        for component in &self.components {
            component.account_host_owners(budget)?;
        }
        Ok(())
    }
}
