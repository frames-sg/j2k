// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guarded chunked-entropy diagnostic transaction and retained report owner.

use crate::{session::HostOwnerLease, CudaSession, Error};

use super::plan::cuda_huffman_table;
use super::{cuda_chunked_entropy_diagnostic_error, resolve_owned_rgb8_packet};

const UNSUPPORTED_CHUNKED_ENTROPY_DIAGNOSTIC_INPUT: &str =
    "J2K CUDA JPEG chunked entropy diagnostic currently supports baseline 8-bit YCbCr 4:2:0 RGB8 inputs only";

#[derive(Debug)]
#[doc(hidden)]
/// Diagnostic report retaining an exact session host-owner lease.
pub struct CudaJpegChunkedEntropyReport {
    report: j2k_cuda_runtime::CudaJpegChunkedEntropyReport,
    _lease: HostOwnerLease,
}

impl core::ops::Deref for CudaJpegChunkedEntropyReport {
    type Target = j2k_cuda_runtime::CudaJpegChunkedEntropyReport;

    fn deref(&self) -> &Self::Target {
        &self.report
    }
}

pub(crate) fn diagnose_owned_cuda_420_entropy(
    bytes: &[u8],
    config: j2k_cuda_runtime::CudaJpegChunkedEntropyConfig,
    session: &mut CudaSession,
) -> Result<CudaJpegChunkedEntropyReport, Error> {
    config
        .validate()
        .map_err(cuda_chunked_entropy_diagnostic_error)?;
    let operation_gate = session.jpeg_host_operation_gate();
    let _operation = operation_gate
        .lock()
        .map_err(|_| Error::JpegHostOperationPoisoned)?;
    let leased_packet = resolve_owned_rgb8_packet(bytes, session)?;
    let fast420 = leased_packet
        .packet
        .fast420()
        .ok_or(Error::UnsupportedCudaRequest {
            reason: UNSUPPORTED_CHUNKED_ENTROPY_DIAGNOSTIC_INPUT,
        })?;
    let context = session.cuda_context()?;
    let pinned_upload = context
        .begin_pinned_upload_operation()
        .map_err(crate::runtime::cuda_error)?;
    let pinned_pool_retained_bytes = pinned_upload
        .diagnostics()
        .map_err(crate::runtime::cuda_error)?
        .retained_bytes;
    let pinned_pool_lease = session.reserve_existing_host_owner(pinned_pool_retained_bytes)?;
    let plan = j2k_cuda_runtime::CudaJpegChunkedEntropyPlan {
        config,
        entropy_bytes: &fast420.entropy_bytes,
        y_dc_table: cuda_huffman_table(&fast420.y_dc_table)?,
        y_ac_table: cuda_huffman_table(&fast420.y_ac_table)?,
        cb_dc_table: cuda_huffman_table(&fast420.cb_dc_table)?,
        cb_ac_table: cuda_huffman_table(&fast420.cb_ac_table)?,
        cr_dc_table: cuda_huffman_table(&fast420.cr_dc_table)?,
        cr_ac_table: cuda_huffman_table(&fast420.cr_ac_table)?,
    };
    drop(pinned_pool_lease);
    let runtime_external_live = session.owned_host_live_bytes()?;
    let report = context
        .diagnose_jpeg_420_entropy_self_sync_with_pinned_upload_operation(
            &plan,
            runtime_external_live,
            &pinned_upload,
        )
        .map_err(cuda_chunked_entropy_diagnostic_error)?;
    let _recycled_pinned_pool_lease = session.reserve_pinned_upload_retention(&pinned_upload)?;
    let retained_bytes = report
        .retained_host_bytes()
        .map_err(cuda_chunked_entropy_diagnostic_error)?;
    let lease = session.reserve_existing_host_owner(retained_bytes)?;
    Ok(CudaJpegChunkedEntropyReport {
        report,
        _lease: lease,
    })
}
