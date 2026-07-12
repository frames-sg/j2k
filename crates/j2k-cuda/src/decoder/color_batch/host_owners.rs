// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actual-capacity accounting for caller-live color decode owner graphs.

use super::{
    CudaComponentDecodeWork, CudaHtj2kColorDecodePlans, Error, CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
    CUDA_HTJ2K_KERNELS_NOT_READY,
};
use crate::allocation::HostPhaseBudget;

pub(super) fn account_colors(
    budget: &mut HostPhaseBudget,
    colors: &Vec<CudaHtj2kColorDecodePlans>,
) -> Result<(), Error> {
    budget.account_vec(colors)?;
    for color in colors {
        color.account_host_owners(budget)?;
    }
    Ok(())
}

pub(super) fn color_batch_budget(
    colors: &Vec<CudaHtj2kColorDecodePlans>,
    shared_payload: &Vec<u8>,
    pending: Option<&CudaHtj2kColorDecodePlans>,
    what: &'static str,
) -> Result<HostPhaseBudget, Error> {
    let mut budget = HostPhaseBudget::new(what);
    account_colors(&mut budget, colors)?;
    budget.account_vec(shared_payload)?;
    if let Some(color) = pending {
        color.account_host_owners(&mut budget)?;
    }
    Ok(budget)
}

pub(super) fn color_work_budget(
    color: &CudaHtj2kColorDecodePlans,
    work: &Vec<CudaComponentDecodeWork>,
    what: &'static str,
) -> Result<HostPhaseBudget, Error> {
    let mut budget = HostPhaseBudget::new(what);
    color.account_host_owners(&mut budget)?;
    account_component_work(&mut budget, work)?;
    Ok(budget)
}

pub(super) fn account_component_work(
    budget: &mut HostPhaseBudget,
    work: &Vec<CudaComponentDecodeWork>,
) -> Result<(), Error> {
    budget.account_vec(work)?;
    for component in work {
        budget.account_vec(&component.bands)?;
        budget.account_vec(&component.pending_dequant_bands)?;
        for pending in &component.pending_dequant_bands {
            budget.account_vec(&pending.jobs)?;
        }
    }
    Ok(())
}

pub(super) fn take_component_work(
    work_iter: &mut std::vec::IntoIter<CudaComponentDecodeWork>,
    component_count: usize,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<CudaComponentDecodeWork>, Error> {
    let mut component_work = host_budget.try_vec_with_capacity(component_count)?;
    for _ in 0..component_count {
        component_work.push(work_iter.next().ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?);
    }
    Ok(component_work)
}

pub(super) fn append_color_payload_to_shared(
    color: &mut CudaHtj2kColorDecodePlans,
    shared_payload: &mut Vec<u8>,
    host_budget: &mut HostPhaseBudget,
) -> Result<(), Error> {
    let base = u64::try_from(shared_payload.len()).map_err(|_| Error::UnsupportedCudaRequest {
        reason: CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
    })?;
    shared_payload
        .len()
        .checked_add(color.payload.len())
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
        })?;
    if !shared_payload.is_empty() {
        host_budget.try_vec_reserve(shared_payload, color.payload.len())?;
    }
    for component in &mut color.components {
        component.rebase_payload_offsets(base)?;
    }
    if shared_payload.is_empty() {
        *shared_payload = core::mem::take(&mut color.payload);
    } else {
        let mut payload = core::mem::take(&mut color.payload);
        shared_payload.append(&mut payload);
    }
    Ok(())
}
