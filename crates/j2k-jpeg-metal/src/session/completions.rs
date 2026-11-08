// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transactional accounting for completed host-resident surfaces.

use super::SessionState;
use crate::{Error, Surface};

impl SessionState {
    pub(crate) const fn completed_host_bytes(&self) -> usize {
        self.completed_host_bytes
    }

    pub(crate) fn store_completed_result(
        &mut self,
        slot: usize,
        result: Result<Surface, Error>,
        execution_live_bytes: usize,
        initial_completed_host_bytes: usize,
    ) -> Result<(), Error> {
        let destination =
            self.completed
                .get(slot)
                .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                    "JPEG Metal completion slot is out of bounds",
                ))?;
        if destination.is_some() {
            return Err(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal completion slot was already populated",
            )
            .into());
        }

        let result_host_bytes = result
            .as_ref()
            .map_or(0, Surface::retained_host_capacity_bytes);
        let additional_completed_host_bytes = self
            .completed_host_bytes
            .checked_sub(initial_completed_host_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal completed-host baseline underflow",
            ))?;
        let live_before_result = execution_live_bytes
            .checked_add(additional_completed_host_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal completion owner baseline overflow",
            ))?;
        crate::batch_allocation::BatchMetadataBudget::with_external_live(
            "JPEG Metal completed host surface retention",
            live_before_result,
        )
        .preflight(&[crate::batch_allocation::BatchMetadataRequest::of::<u8>(
            result_host_bytes,
        )])?;
        let completed_host_bytes = self
            .completed_host_bytes
            .checked_add(result_host_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal completed-host ledger overflow",
            ))?;
        let collective_result_bytes = live_before_result.checked_add(result_host_bytes).ok_or(
            j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal completion high-water overflow",
            ),
        )?;

        self.completed[slot] = Some(result);
        self.completed_host_bytes = completed_host_bytes;
        self.peak_collective_host_bytes =
            self.peak_collective_host_bytes.max(collective_result_bytes);
        Ok(())
    }

    pub(crate) fn take_completed_result(
        &mut self,
        slot: usize,
    ) -> Result<Result<Surface, Error>, Error> {
        let result = self
            .completed
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::MetalKernel {
                message: format!("missing queued Metal surface for slot {slot}"),
            })?;
        let result_host_bytes = result
            .as_ref()
            .map_or(0, Surface::retained_host_capacity_bytes);
        let completed_host_bytes = self
            .completed_host_bytes
            .checked_sub(result_host_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal completed-host ledger underflow",
            ))?;
        let result = self.completed.release(slot).flatten().ok_or(
            j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal completion disappeared during consumption",
            ),
        )?;
        self.completed_host_bytes = completed_host_bytes;
        Ok(result)
    }

    pub(crate) fn cancel_submission(&mut self, slot: usize) -> Result<(), Error> {
        let Some((request, result)) = self
            .completed
            .cancel(&mut self.queued, slot, |request| request.output_slot)
        else {
            return Ok(());
        };
        if let Some(request) = request {
            self.queued_plan_ledger.remove(&self.queued, &request)?;
        }
        if self.queued.is_empty() {
            self.retained_execution_metadata_bytes = 0;
        }
        let host_bytes = result
            .as_ref()
            .and_then(|result| result.as_ref().ok())
            .map_or(0, Surface::retained_host_capacity_bytes);
        self.completed_host_bytes = self.completed_host_bytes.checked_sub(host_bytes).ok_or(
            j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal cancelled completion ledger underflow",
            ),
        )?;
        Ok(())
    }
}
