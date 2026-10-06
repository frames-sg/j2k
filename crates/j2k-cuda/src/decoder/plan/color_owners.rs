// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate ownership for flattened multi-component CUDA decode plans.

mod referenced;

use j2k_core::PixelFormat;
use j2k_native::{J2kDirectColorPlan, J2kDirectGrayscaleStep};

use crate::allocation::HostPhaseBudget;
use crate::{CudaHtj2kDecodePlan, Error};

pub(super) use referenced::{
    flatten_direct_source_cuda_color_tile_components,
    flatten_referenced_classic_cuda_color_tile_components,
    flatten_referenced_cuda_color_tile_components,
};

use super::super::CudaHtj2kColorDecodePlans;

type OutputRegion = ((u32, u32), (u32, u32));

pub(super) fn flatten_cuda_color_components(
    native_plan: &J2kDirectColorPlan,
    format: PixelFormat,
    output_region: Option<OutputRegion>,
    what: &'static str,
) -> Result<(Vec<u8>, Vec<CudaHtj2kDecodePlan>), Error> {
    let mut initial_budget = HostPhaseBudget::new(what);
    flatten_cuda_color_components_with_budget(
        native_plan,
        format,
        output_region,
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        &mut initial_budget,
    )
}

/// Flattens every component into one shared payload. `host_budget` must
/// share the `host_cap` ceiling; it ends up charged for the shared payload
/// and each component's retained owners.
pub(super) fn flatten_cuda_color_components_with_budget(
    native_plan: &J2kDirectColorPlan,
    format: PixelFormat,
    output_region: Option<OutputRegion>,
    host_cap: usize,
    host_budget: &mut HostPhaseBudget,
) -> Result<(Vec<u8>, Vec<CudaHtj2kDecodePlan>), Error> {
    let mut components = host_budget.try_vec_with_capacity(native_plan.component_plans.len())?;
    let payload_capacity = native_plan
        .component_plans
        .iter()
        .flat_map(|plan| &plan.steps)
        .try_fold(0usize, |total, step| {
            let bytes = match step {
                J2kDirectGrayscaleStep::HtSubBand(subband) => subband
                    .jobs
                    .iter()
                    .try_fold(0usize, |bytes, job| bytes.checked_add(job.data.len())),
                J2kDirectGrayscaleStep::ClassicSubBand(subband) => subband
                    .jobs
                    .iter()
                    .try_fold(0usize, |bytes, job| bytes.checked_add(job.data.len())),
                J2kDirectGrayscaleStep::Idwt(_) | J2kDirectGrayscaleStep::Store(_) => Some(0),
            }?;
            total.checked_add(bytes)
        })
        .ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::resource_limit(
                "j2k CUDA color direct-plan payload size overflows",
            ),
        ))?;
    let mut payload = host_budget.try_vec_with_capacity(payload_capacity)?;

    for component_plan in &native_plan.component_plans {
        // A component's own payload lives only until it is moved into the
        // shared payload, so it is charged to a step budget on top of the
        // owners already live; the phase keeps only what the component retains.
        let mut step_budget =
            HostPhaseBudget::with_cap("j2k CUDA color component flattening", host_cap);
        step_budget.account_bytes(host_budget.live_bytes())?;
        let mut component = match output_region {
            Some((origin, dimensions)) => {
                CudaHtj2kDecodePlan::from_grayscale_direct_plan_region_with_budget(
                    component_plan,
                    format,
                    origin,
                    dimensions,
                    &mut step_budget,
                )?
            }
            None => CudaHtj2kDecodePlan::from_grayscale_direct_plan_with_budget(
                component_plan,
                format,
                (0, 0),
                &mut step_budget,
            )?,
        };

        component.append_payload_to_shared_with_budget(&mut payload, &mut step_budget)?;
        component.account_host_owners(host_budget)?;
        components.push(component);
    }

    Ok((payload, components))
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

#[cfg(test)]
mod tests {
    use j2k_core::PixelFormat;
    use j2k_native::{encode, DecodeSettings, DecoderContext, EncodeOptions, Image};

    use super::flatten_cuda_color_components_with_budget;
    use crate::allocation::HostPhaseBudget;

    #[test]
    fn flattening_charges_the_shared_payload_once() {
        let pixels = (0..64 * 48 * 3)
            .map(|index: usize| u8::try_from(index * 37 % 251).expect("byte"))
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            reversible: true,
            use_ht_block_coding: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let encoded = encode(&pixels, 64, 48, 3, 8, false, &options).expect("encode RGB fixture");
        let image = Image::new(&encoded, &DecodeSettings::default()).expect("parse fixture");
        let plan = image
            .build_direct_color_plan_with_context(&mut DecoderContext::default())
            .expect("native color plan");

        let mut budget = HostPhaseBudget::new("flattened color plan");
        let (payload, components) = flatten_cuda_color_components_with_budget(
            &plan,
            PixelFormat::Rgb8,
            None,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            &mut budget,
        )
        .expect("flatten color components");

        let mut retained = HostPhaseBudget::new("retained color owners");
        retained.account_vec(&components).expect("components");
        retained.account_vec(&payload).expect("shared payload");
        for component in &components {
            component
                .account_host_owners(&mut retained)
                .expect("component owners");
        }
        assert_ne!(payload, [] as [u8; 0]);
        assert_eq!(budget.live_bytes(), retained.live_bytes());
    }
}
