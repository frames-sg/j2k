// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{EncodedJ2k, J2kLosslessEncodeOptions};
use j2k_core::DeviceSubmission;
use rayon::prelude::*;
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use super::{
    encode_lossless_batch_with_report, encode_lossless_tile_with_report, submit_lossless_batch,
    submit_lossless_batch_to_metal, MetalEncodeInputStaging, MetalEncodeStageAccelerator,
    MetalLosslessBufferEncodeBatchOutcome, MetalLosslessBufferEncodeOutcome,
    MetalLosslessEncodeBatchRequest, MetalLosslessEncodeConfig, MetalLosslessEncodeOutcome,
    MetalLosslessEncodeTile, SubmittedJ2kLosslessMetalEncodeBatch,
};

pub(super) struct InflightLimitedOrderedItems<T> {
    pub(super) items: Vec<T>,
    pub(super) max_observed_inflight_items: usize,
}

pub(super) fn collect_inflight_limited_ordered<T, O, F>(
    items: Vec<T>,
    inflight_items: usize,
    f: F,
) -> Result<InflightLimitedOrderedItems<O>, crate::Error>
where
    T: Send,
    O: Send,
    F: Fn(usize, T) -> Result<O, crate::Error> + Sync,
{
    if items.is_empty() {
        return Ok(InflightLimitedOrderedItems {
            items: Vec::new(),
            max_observed_inflight_items: 0,
        });
    }

    let active = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(AtomicUsize::new(0));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(inflight_items.max(1))
        .build()
        .map_err(|err| crate::Error::MetalKernel {
            message: format!("J2K Metal encode worker pool initialization failed: {err}"),
        })?;

    let active_for_tasks = Arc::clone(&active);
    let observed_for_tasks = Arc::clone(&observed);
    let results = pool.install(|| {
        items
            .into_par_iter()
            .enumerate()
            .map(|(index, item)| {
                let _guard = ActiveTileGuard::new(&active_for_tasks, &observed_for_tasks);
                f(index, item)
            })
            .collect::<Vec<_>>()
    });

    let max_observed_inflight_items = observed.load(Ordering::Relaxed);
    let mut ordered = Vec::with_capacity(results.len());
    let mut first_error = None;
    for result in results {
        match result {
            Ok(item) if first_error.is_none() => ordered.push(item),
            Ok(_) => {}
            Err(err) => {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
    }

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(InflightLimitedOrderedItems {
        items: ordered,
        max_observed_inflight_items,
    })
}

struct ActiveTileGuard<'a> {
    active: &'a AtomicUsize,
}

impl<'a> ActiveTileGuard<'a> {
    fn new(active: &'a AtomicUsize, observed: &AtomicUsize) -> Self {
        let now = active.fetch_add(1, Ordering::AcqRel).saturating_add(1);
        let mut current = observed.load(Ordering::Relaxed);
        while now > current {
            match observed.compare_exchange(current, now, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
        Self { active }
    }
}

impl Drop for ActiveTileGuard<'_> {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

thread_local! {
    static TEST_RESIDENT_ENCODE_FAILURE_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(super) fn set_test_resident_encode_failure_index(index: Option<usize>) {
    TEST_RESIDENT_ENCODE_FAILURE_INDEX.set(index);
}

pub(super) fn test_resident_encode_failure_index() -> Option<usize> {
    TEST_RESIDENT_ENCODE_FAILURE_INDEX.get()
}

// Pre-8c permutation shims: the public API collapsed to the request-based
// entries, but the device tests keep their original entry-point coverage
// (and the single-tile report tests their exact staged routing) through
// these test-only equivalents of the removed wrappers.
pub(super) struct TestSubmittedSingleLosslessEncode {
    inner: SubmittedJ2kLosslessMetalEncodeBatch,
}

impl TestSubmittedSingleLosslessEncode {
    pub(super) fn wait(self) -> Result<EncodedJ2k, crate::Error> {
        let mut encoded = self.inner.wait()?;
        if encoded.len() != 1 {
            return Err(crate::Error::MetalKernel {
                message: "submitted J2K Metal single encode produced an unexpected batch length"
                    .to_string(),
            });
        }
        Ok(encoded.remove(0))
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn submit_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<TestSubmittedSingleLosslessEncode, crate::Error> {
    Ok(TestSubmittedSingleLosslessEncode {
        inner: submit_lossless_batch(
            MetalLosslessEncodeBatchRequest {
                tiles: &[tile],
                staging: MetalEncodeInputStaging::CopyAndPad,
                config: MetalLosslessEncodeConfig::default(),
            },
            options,
            session,
        )?,
    })
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn submit_lossless_from_padded_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<TestSubmittedSingleLosslessEncode, crate::Error> {
    Ok(TestSubmittedSingleLosslessEncode {
        inner: submit_lossless_batch(
            MetalLosslessEncodeBatchRequest {
                tiles: &[tile],
                staging: MetalEncodeInputStaging::AlreadyPaddedContiguous,
                config: MetalLosslessEncodeConfig::default(),
            },
            options,
            session,
        )?,
    })
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn encode_lossless_from_metal_buffer(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<EncodedJ2k, crate::Error> {
    submit_lossless_from_metal_buffer(tile, options, session)?.wait()
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn encode_lossless_from_padded_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    let mut accelerator = MetalEncodeStageAccelerator::for_host_output(*options);
    encode_lossless_tile_with_report(
        tile,
        *options,
        session,
        MetalEncodeInputStaging::AlreadyPaddedContiguous,
        &mut accelerator,
    )
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn encode_lossless_from_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    Ok(submit_lossless_batch_to_metal(
        MetalLosslessEncodeBatchRequest {
            tiles,
            staging: MetalEncodeInputStaging::CopyAndPad,
            config: MetalLosslessEncodeConfig::default(),
        },
        options,
        session,
    )?
    .wait()?
    .outcomes)
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn encode_lossless_from_padded_metal_buffers_to_metal_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    Ok(submit_lossless_batch_to_metal(
        MetalLosslessEncodeBatchRequest {
            tiles,
            staging: MetalEncodeInputStaging::AlreadyPaddedContiguous,
            config: MetalLosslessEncodeConfig::default(),
        },
        options,
        session,
    )?
    .wait()?
    .outcomes)
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn encode_lossless_from_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let mut outcomes =
        encode_lossless_from_metal_buffers_to_metal_with_report(&[tile], options, session)?;
    if outcomes.len() != 1 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal single buffer encode produced an unexpected batch length"
                .to_string(),
        });
    }
    Ok(outcomes.remove(0))
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn encode_lossless_from_padded_metal_buffer_to_metal_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    let mut outcomes =
        encode_lossless_from_padded_metal_buffers_to_metal_with_report(&[tile], options, session)?;
    if outcomes.len() != 1 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal single buffer encode produced an unexpected batch length"
                .to_string(),
        });
    }
    Ok(outcomes.remove(0))
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn encode_lossless_from_padded_metal_buffers_with_report(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    encode_lossless_batch_with_report(
        MetalLosslessEncodeBatchRequest {
            tiles,
            staging: MetalEncodeInputStaging::AlreadyPaddedContiguous,
            config: MetalLosslessEncodeConfig::default(),
        },
        options,
        session,
    )
}

#[allow(clippy::trivially_copy_pass_by_ref)] // shims keep the removed wrappers' exact signatures
pub(super) fn encode_lossless_from_padded_metal_buffers_to_metal_batch(
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    config: MetalLosslessEncodeConfig,
) -> Result<MetalLosslessBufferEncodeBatchOutcome, crate::Error> {
    submit_lossless_batch_to_metal(
        MetalLosslessEncodeBatchRequest {
            tiles,
            staging: MetalEncodeInputStaging::AlreadyPaddedContiguous,
            config,
        },
        options,
        session,
    )?
    .wait()
}
