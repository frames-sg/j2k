// SPDX-License-Identifier: MIT OR Apache-2.0

//! Session flush, fallback execution, and completion transfer.

use super::{add_execution_external_live_bytes, group_compatible_requests, QueuedRequest};
use crate::{Error, Surface};

pub(super) fn flush_if_needed(session: &mut crate::session::SessionState) -> Result<(), Error> {
    if session.queued.is_empty() {
        return Ok(());
    }

    let mut queued = match session.take_queued_requests() {
        Ok(queued) => queued,
        Err(error) => {
            session.complete_queued_with_error(&error);
            return Ok(());
        }
    };
    let flush_completed_host_bytes = session.completed_host_bytes();
    let batches = match group_compatible_requests(&mut queued) {
        Ok(batches) => batches,
        Err(error) => {
            for request in queued {
                if session.completed[request.output_slot].is_none() {
                    session.store_completed_result(
                        request.output_slot,
                        Err(error.clone()),
                        0,
                        session.completed_host_bytes(),
                    )?;
                }
            }
            return Ok(());
        }
    };
    drop(queued);
    for mut batch in batches {
        execute_batch(session, &mut batch, flush_completed_host_bytes)?;
    }
    Ok(())
}

fn execute_batch(
    session: &mut crate::session::SessionState,
    batch: &mut Vec<QueuedRequest>,
    flush_completed_host_bytes: usize,
) -> Result<(), Error> {
    let initial_completed_host_bytes = session.completed_host_bytes();
    let additional_completed_host_bytes = initial_completed_host_bytes
        .checked_sub(flush_completed_host_bytes)
        .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
            "JPEG Metal flush completion baseline underflow",
        ))?;
    add_execution_external_live_bytes(batch, additional_completed_host_bytes)?;
    let execution_live_bytes = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal completion execution owner graph",
        batch,
    )?
    .live_bytes();
    session.submissions = session.submissions.saturating_add(1);
    match crate::decode_compatible_batch_with_session(batch, session) {
        Ok(Some(results)) => {
            if let Err(error) = validate_result_count(batch.len(), results.len()) {
                for request in batch.drain(..) {
                    store_completion_or_error(
                        session,
                        request.output_slot,
                        Err(error.clone()),
                        execution_live_bytes,
                        initial_completed_host_bytes,
                    )?;
                }
                return Ok(());
            }
            for (request, result) in batch.drain(..).zip(results) {
                store_completion_or_error(
                    session,
                    request.output_slot,
                    result,
                    execution_live_bytes,
                    initial_completed_host_bytes,
                )?;
            }
        }
        Ok(None) => execute_cpu_fallbacks(
            session,
            batch,
            execution_live_bytes,
            initial_completed_host_bytes,
        )?,
        Err(error) => {
            for request in batch.drain(..) {
                store_completion_or_error(
                    session,
                    request.output_slot,
                    Err(error.clone()),
                    execution_live_bytes,
                    initial_completed_host_bytes,
                )?;
            }
        }
    }
    Ok(())
}

fn validate_result_count(request_count: usize, result_count: usize) -> Result<(), Error> {
    if request_count == result_count {
        return Ok(());
    }
    Err(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
        "JPEG Metal compatible batch returned a different result count",
    )
    .into())
}

fn execute_cpu_fallbacks(
    session: &mut crate::session::SessionState,
    batch: &mut Vec<QueuedRequest>,
    execution_live_bytes: usize,
    initial_completed_host_bytes: usize,
) -> Result<(), Error> {
    for request in batch.drain(..) {
        let additional_completed_host_bytes = session
            .completed_host_bytes()
            .checked_sub(initial_completed_host_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal CPU fallback completion baseline underflow",
            ))?;
        let fallback_live_bytes = execution_live_bytes
            .checked_add(additional_completed_host_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal CPU fallback owner baseline overflow",
            ))?;
        let decoder_baseline_bytes = fallback_live_bytes
            .checked_sub(request.retained_input_bytes()?)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal CPU fallback shared-input baseline underflow",
            ))?;
        let result = crate::decode_surface_from_shared_input(
            &request.input,
            request.fmt,
            request.backend,
            request.op,
            request.fast_packet.as_ref(),
            decoder_baseline_bytes,
            fallback_live_bytes,
        );
        store_completion_or_error(
            session,
            request.output_slot,
            result,
            execution_live_bytes,
            initial_completed_host_bytes,
        )?;
    }
    Ok(())
}

fn store_completion_or_error(
    session: &mut crate::session::SessionState,
    slot: usize,
    result: Result<Surface, Error>,
    execution_live_bytes: usize,
    initial_completed_host_bytes: usize,
) -> Result<(), Error> {
    match session.store_completed_result(
        slot,
        result,
        execution_live_bytes,
        initial_completed_host_bytes,
    ) {
        Ok(()) => Ok(()),
        Err(error) => session.store_completed_result(
            slot,
            Err(error),
            execution_live_bytes,
            initial_completed_host_bytes,
        ),
    }
}

pub(super) fn take_surface(
    session: &mut crate::session::SessionState,
    slot: usize,
) -> Result<Surface, Error> {
    let result = session.take_completed_result(slot)?;
    if session.queued.is_empty() && session.completed.iter().all(Option::is_none) {
        session.completed.clear();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_batch_result_count_rejects_short_and_extra_results() {
        validate_result_count(2, 2).expect("exact result count");
        for result_count in [1, 3] {
            assert!(matches!(
                validate_result_count(2, result_count),
                Err(Error::JpegPlanCache(
                    j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                        "JPEG Metal compatible batch returned a different result count"
                    )
                ))
            ));
        }
    }
}
