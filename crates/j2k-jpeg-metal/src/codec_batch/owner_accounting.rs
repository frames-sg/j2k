// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batch metadata, prepared-decoder, and plan-owner accounting.

use crate::{batch, plan_owner_ledger::PlanOwnerLedger, Decoder, Error};

pub(super) struct Rgb8BatchBuildContext {
    pub(super) budget: crate::batch_allocation::BatchMetadataBudget,
    pub(super) requests: Vec<batch::QueuedRequest>,
    pub(super) plan_owners: PlanOwnerLedger,
    pub(super) external_live_bytes: usize,
    pub(super) collective_what: &'static str,
}

impl Rgb8BatchBuildContext {
    pub(super) fn new(
        source_count: usize,
        phase: &'static str,
        request_what: &'static str,
        external_live_bytes: usize,
        collective_what: &'static str,
    ) -> Result<Self, Error> {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(phase);
        let requests = budget.try_vec(source_count, request_what)?;
        Ok(Self {
            budget,
            requests,
            plan_owners: PlanOwnerLedger::default(),
            external_live_bytes,
            collective_what,
        })
    }

    pub(super) fn resolver_external_live_bytes(&self) -> Result<usize, Error> {
        self.plan_owners.external_live_bytes(
            self.budget
                .live_bytes()
                .checked_add(self.external_live_bytes)
                .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                    "JPEG Metal batch resolver external baseline overflow",
                ))?,
        )
    }
}

pub(super) fn distinct_decoder_retained_bytes(decoders: &[&Decoder<'_>]) -> Result<usize, Error> {
    let mut retained_bytes = 0_usize;
    for (index, decoder) in decoders.iter().enumerate() {
        if decoders[..index]
            .iter()
            .any(|prior| std::ptr::eq::<Decoder<'_>>(*prior, *decoder))
        {
            continue;
        }
        retained_bytes = retained_bytes
            .checked_add(j2k_jpeg::adapter::decoder_retained_allocation_bytes(
                decoder.inner(),
            )?)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal decoder batch retained-byte count overflow",
            ))?;
    }
    Ok(retained_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_core::{
        BatchAllocationRequest, BatchInfrastructureError, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    };

    const BASELINE_420: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_420_16x16.jpg");

    #[test]
    fn decoder_owner_accounting_deduplicates_identity_not_equal_bytes() {
        let first = Decoder::new(BASELINE_420).expect("first decoder");
        let second = Decoder::new(BASELINE_420).expect("second decoder");
        let first_bytes = distinct_decoder_retained_bytes(&[&first]).expect("first bytes");
        assert_eq!(
            distinct_decoder_retained_bytes(&[&first, &first]).expect("repeated identity"),
            first_bytes
        );
        assert_eq!(
            distinct_decoder_retained_bytes(&[&first, &second]).expect("distinct identities"),
            first_bytes + distinct_decoder_retained_bytes(&[&second]).expect("second bytes")
        );
    }

    #[test]
    fn distinct_decoder_owners_compose_with_metadata_at_exact_and_one_over() {
        let first = Decoder::new(BASELINE_420).expect("first decoder");
        let second = Decoder::new(BASELINE_420).expect("second decoder");
        let decoder_bytes =
            distinct_decoder_retained_bytes(&[&first, &second]).expect("decoder owners");
        let exact_metadata = DEFAULT_MAX_HOST_ALLOCATION_BYTES
            .checked_sub(decoder_bytes)
            .expect("decoder owners fit below cap");

        j2k_core::BatchAllocationBudget::with_external_live(
            "JPEG Metal decoder owner composition",
            decoder_bytes,
        )
        .preflight(&[BatchAllocationRequest::of::<u8>(exact_metadata)])
        .expect("exact decoder and metadata cap");
        assert_eq!(
            j2k_core::BatchAllocationBudget::with_external_live(
                "JPEG Metal decoder owner composition",
                decoder_bytes,
            )
            .preflight(&[BatchAllocationRequest::of::<u8>(exact_metadata + 1)])
            .expect_err("one byte over decoder and metadata cap"),
            BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG Metal decoder owner composition",
                requested: DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }
        );
    }
}
