// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded-scratch decode of large exact-native colour groups.
//!
//! Each chunk is written into the next unused tail of one dense destination.
//! Chunks before the last complete before the next one starts, so their
//! scratch returns to the session cache; only the final chunk stays pending.

use j2k::BatchLayout;
use j2k_core::PixelFormat;
use j2k_cuda_runtime::{
    CudaDeviceBufferRange, CudaError, CudaExecutionStats, CudaExternalDeviceBufferViewMut,
};

use super::{
    cuda_error, profile, submit_native_color_group_into, ColorJobIndexBases,
    CudaHtj2kProfileReport, CudaSession, Error, NativeColorBatchInput, NativeColorOwnedBatch,
    SubmittedNativeColorExternalBatch, SubmittedNativeColorResidentBatch, COLOR_BATCH_CHUNK_IMAGES,
};
use crate::allocation::try_vec_with_capacity;
use crate::decoder::decode_profile::add_decode_report;

pub(super) fn submit_native_color_chunks_into(
    inputs: &[NativeColorBatchInput<'_>],
    session: &mut CudaSession,
    fmt: PixelFormat,
    layout: BatchLayout,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
) -> Result<SubmittedNativeColorExternalBatch, Error> {
    let mut ranges = try_vec_with_capacity(inputs.len(), "chunked CUDA colour output ranges")?;
    let mut execution = CudaExecutionStats::default();
    let mut aggregate: Option<CudaHtj2kProfileReport> = None;
    let mut completion = None;
    let mut final_bases = ColorJobIndexBases::default();
    let mut written = 0usize;
    let last_chunk = inputs
        .len()
        .div_ceil(COLOR_BATCH_CHUNK_IMAGES)
        .saturating_sub(1);
    for (chunk_index, chunk) in inputs.chunks(COLOR_BATCH_CHUNK_IMAGES).enumerate() {
        let bases = aggregate
            .as_ref()
            .map_or_else(ColorJobIndexBases::default, |report| ColorJobIndexBases {
                ht: report.ht_block_count,
                classic: report.classic_block_count,
            });
        let remaining = destination
            .byte_len()
            .checked_sub(written)
            .filter(|remaining| *remaining != 0)
            .ok_or_else(|| {
                cuda_error(CudaError::OutputTooSmall {
                    required: written.saturating_add(1),
                    have: destination.byte_len(),
                })
            })?;
        let mut tail = destination
            .sub_view_mut(written, remaining)
            .map_err(cuda_error)?;
        let SubmittedNativeColorExternalBatch {
            ranges: chunk_ranges,
            execution: chunk_execution,
            report,
            completion: chunk_completion,
            ..
        } = submit_native_color_group_into(chunk, session, fmt, layout, &mut tail)
            .map_err(|error| bases.rebase(error))?;
        if chunk_index == last_chunk {
            // A failure in the final chunk surfaces later, when it is waited.
            completion = chunk_completion;
            final_bases = bases;
        } else if let Some(mut pending) = chunk_completion {
            pending.complete().map_err(|error| bases.rebase(error))?;
        }

        let mut chunk_end = 0usize;
        for range in chunk_ranges {
            let offset = written
                .checked_add(range.offset)
                .ok_or_else(chunk_layout_overflow)?;
            chunk_end = chunk_end.max(
                range
                    .offset
                    .checked_add(range.len)
                    .ok_or_else(chunk_layout_overflow)?,
            );
            ranges.push(CudaDeviceBufferRange {
                offset,
                len: range.len,
            });
        }
        written = written
            .checked_add(chunk_end)
            .ok_or_else(chunk_layout_overflow)?;
        execution = add_execution(execution, chunk_execution);
        aggregate = Some(match aggregate {
            None => report,
            Some(mut total) => {
                add_decode_report(&mut total, &report);
                total.detail.wall_total_us = total
                    .detail
                    .wall_total_us
                    .saturating_add(report.detail.wall_total_us);
                total
            }
        });
    }
    let mut report = aggregate.ok_or(Error::capability_rejected(
        j2k_core::CapabilityRejection::contract_violation(
            "chunked CUDA colour batch received no inputs",
        ),
    ))?;
    profile::finalize_decode_total_us(&mut report);
    Ok(SubmittedNativeColorExternalBatch {
        ranges,
        execution,
        report,
        completion,
        job_index_bases: final_bases,
    })
}

pub(super) fn submit_native_color_resident_chunks(
    inputs: &[NativeColorBatchInput<'_>],
    session: &mut CudaSession,
    fmt: PixelFormat,
    layout: BatchLayout,
) -> Result<SubmittedNativeColorResidentBatch, Error> {
    let total_bytes = native_color_output_bytes(inputs, fmt)?;
    let mut buffer = session
        .cuda_context()?
        .allocate(total_bytes)
        .map_err(cuda_error)?;
    let submitted =
        buffer
            .destination_view_mut()
            .map_err(cuda_error)
            .and_then(|mut destination| {
                submit_native_color_chunks_into(inputs, session, fmt, layout, &mut destination)
            });
    let SubmittedNativeColorExternalBatch {
        ranges,
        execution,
        report,
        completion,
        job_index_bases,
    } = match submitted {
        Ok(submitted) => submitted,
        Err(error) => {
            if error.completion_is_uncertain() {
                // Work may still write this allocation; leak it rather than free it.
                core::mem::forget(buffer);
            }
            return Err(error);
        }
    };
    let filled = ranges
        .last()
        .and_then(|range| range.offset.checked_add(range.len))
        == Some(total_bytes);
    let submitted = SubmittedNativeColorResidentBatch {
        output: Some(NativeColorOwnedBatch {
            buffer,
            ranges,
            execution,
        }),
        report,
        completion,
        job_index_bases,
    };
    if !filled {
        // Dropping the submission retires the final chunk before releasing the buffer.
        drop(submitted);
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::contract_violation(
                "chunked CUDA colour output did not fill its dense allocation",
            ),
        ));
    }
    Ok(submitted)
}

fn native_color_output_bytes(
    inputs: &[NativeColorBatchInput<'_>],
    fmt: PixelFormat,
) -> Result<usize, Error> {
    inputs
        .iter()
        .try_fold(0usize, |total, input| {
            let (width, height) = input.device_plan.output_dims();
            (width as usize)
                .checked_mul(height as usize)?
                .checked_mul(fmt.bytes_per_pixel())?
                .checked_add(total)
        })
        .ok_or_else(chunk_layout_overflow)
}

fn add_execution(total: CudaExecutionStats, chunk: CudaExecutionStats) -> CudaExecutionStats {
    CudaExecutionStats::new(
        total
            .kernel_dispatches()
            .saturating_add(chunk.kernel_dispatches()),
        total
            .copy_kernel_dispatches()
            .saturating_add(chunk.copy_kernel_dispatches()),
        total
            .decode_kernel_dispatches()
            .saturating_add(chunk.decode_kernel_dispatches()),
        total.used_hardware_decode() || chunk.used_hardware_decode(),
    )
}

const fn chunk_layout_overflow() -> Error {
    Error::HostAllocationFailed {
        bytes: usize::MAX,
        what: "chunked CUDA colour output layout",
    }
}
