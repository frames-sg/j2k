// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared fallible ownership for ordered Metal adapter submissions.

use j2k_core::{
    try_batch_reserve_for_push, try_batch_reserve_to, BatchAllocationBudget,
    BatchAllocationRequest, BatchInfrastructureError, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

/// Ordered submission owner with deferred, fallible capacity reservation.
#[doc(hidden)]
pub struct FallibleSubmissionQueue<S> {
    submissions: Vec<S>,
    capacity_hint: usize,
}

impl<S> Default for FallibleSubmissionQueue<S> {
    fn default() -> Self {
        Self {
            submissions: Vec::new(),
            capacity_hint: 0,
        }
    }
}

impl<S> FallibleSubmissionQueue<S> {
    /// Adopts an already-reserved submission vector for aggregate finish accounting.
    #[must_use]
    pub fn from_retained(submissions: Vec<S>) -> Self {
        Self {
            submissions,
            capacity_hint: 0,
        }
    }

    /// Creates an empty queue whose capacity hint is applied at the first fallible push.
    #[must_use]
    pub fn with_capacity_hint(capacity_hint: usize) -> Self {
        Self {
            capacity_hint,
            ..Self::default()
        }
    }

    /// Number of retained submissions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.submissions.len()
    }

    /// Whether the queue retains no submissions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.submissions.is_empty()
    }

    /// Reserves one slot, constructs the submission, and commits it without further growth.
    ///
    /// # Errors
    ///
    /// Returns a converted [`BatchInfrastructureError`] when reservation fails, or the
    /// submission builder's error without mutating the retained submission count.
    pub fn try_push_with<E>(
        &mut self,
        what: &'static str,
        build: impl FnOnce(usize, usize) -> Result<S, E>,
    ) -> Result<usize, E>
    where
        E: From<BatchInfrastructureError>,
    {
        self.reserve_for_push(what).map_err(E::from)?;
        let slot = self.submissions.len();
        let submission = build(slot, self.submissions.capacity())?;
        self.submissions.push(submission);
        Ok(slot)
    }

    /// Finishes submissions in order under one submission-plus-output metadata budget.
    ///
    /// # Errors
    ///
    /// Returns a converted [`BatchInfrastructureError`] when the aggregate live set cannot
    /// be reserved, or the first error returned while finishing a submission.
    pub fn try_finish<O, E>(
        self,
        phase: &'static str,
        output_what: &'static str,
        finish: impl FnMut(S) -> Result<O, E>,
    ) -> Result<Vec<O>, E>
    where
        E: From<BatchInfrastructureError>,
    {
        self.try_finish_with_cap(
            phase,
            output_what,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            finish,
        )
    }

    fn reserve_for_push(&mut self, what: &'static str) -> Result<(), BatchInfrastructureError> {
        if self.capacity_hint != 0 {
            try_batch_reserve_to(&mut self.submissions, self.capacity_hint, what)?;
            self.capacity_hint = 0;
        }
        try_batch_reserve_for_push(&mut self.submissions, what)
    }

    fn try_finish_with_cap<O, E>(
        self,
        phase: &'static str,
        output_what: &'static str,
        cap: usize,
        mut finish: impl FnMut(S) -> Result<O, E>,
    ) -> Result<Vec<O>, E>
    where
        E: From<BatchInfrastructureError>,
    {
        let output_count = self.submissions.len();
        let mut budget = BatchAllocationBudget::with_cap(phase, cap);
        budget
            .account_capacity::<S>(self.submissions.capacity())
            .map_err(E::from)?;
        budget
            .preflight(&[BatchAllocationRequest::of::<O>(output_count)])
            .map_err(E::from)?;
        let mut outputs = budget.try_vec(output_count, output_what).map_err(E::from)?;
        for submission in self.submissions {
            outputs.push(finish(submission)?);
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, mem::size_of};

    use super::*;

    fn two_submissions() -> FallibleSubmissionQueue<u32> {
        let mut queue = FallibleSubmissionQueue::with_capacity_hint(2);
        for value in [3_u32, 5] {
            queue
                .try_push_with("test submissions", |_, _| {
                    Ok::<_, BatchInfrastructureError>(value)
                })
                .expect("bounded test submission");
        }
        queue
    }

    #[test]
    fn finish_budget_counts_live_submission_and_output_capacities() {
        let queue = two_submissions();
        let exact_cap =
            queue.submissions.capacity() * size_of::<u32>() + queue.len() * size_of::<u64>();
        let outputs = queue
            .try_finish_with_cap(
                "test submission finish",
                "test outputs",
                exact_cap,
                |value| Ok::<_, BatchInfrastructureError>(u64::from(value)),
            )
            .expect("exact submission plus output cap");
        assert_eq!(outputs, [3_u64, 5]);

        let queue = two_submissions();
        let finish_calls = Cell::new(0_usize);
        let error = queue
            .try_finish_with_cap(
                "test submission finish",
                "test outputs",
                exact_cap - 1,
                |value| {
                    finish_calls.set(finish_calls.get() + 1);
                    Ok::<_, BatchInfrastructureError>(u64::from(value))
                },
            )
            .expect_err("one byte below aggregate cap");
        assert!(matches!(
            error,
            BatchInfrastructureError::AllocationTooLarge {
                what: "test submission finish",
                requested,
                cap,
            } if requested == exact_cap && cap == exact_cap - 1
        ));
        assert_eq!(
            finish_calls.get(),
            0,
            "finish must not start after failed preflight"
        );
    }

    #[test]
    fn oversized_hint_fails_before_submission_construction() {
        let mut queue = FallibleSubmissionQueue::<u8>::with_capacity_hint(usize::MAX);
        let builder_called = Cell::new(false);
        let error = queue
            .try_push_with("test submissions", |_, _| {
                builder_called.set(true);
                Ok::<_, BatchInfrastructureError>(1)
            })
            .expect_err("oversized submission hint");
        assert!(matches!(
            error,
            BatchInfrastructureError::AllocationTooLarge {
                what: "test submissions",
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }
        ));
        assert!(!builder_called.get());
        assert!(queue.is_empty());
    }
}
