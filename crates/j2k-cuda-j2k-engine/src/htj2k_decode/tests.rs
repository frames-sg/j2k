// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::accelerator::GpuAbi;
use j2k_cuda_runtime::{CudaContext, CudaExecutionStats};

use crate::{bytes::htj2k_cleanup_multi_jobs_as_bytes, J2kCudaEngine};

use super::{
    planning::{
        htj2k_decode_multi_cleanup_dequant_kernel_for_jobs, htj2k_decode_multi_kernel_for_jobs,
    },
    types::CudaHtj2kCleanupMultiKernelJob,
    CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeTables,
    CudaHtj2kDequantizeTarget, CudaQueuedHtj2kCleanup,
};

fn cuda_runtime_gate() -> bool {
    j2k_test_support::cuda_runtime_gate(module_path!())
}

fn tables<'a>(
    initial_context_lookup: &'a [u16; 1024],
    later_context_lookup: &'a [u16; 1024],
    initial_prefix_lookup: &'a [u16; 320],
    later_prefix_lookup: &'a [u16; 256],
) -> CudaHtj2kDecodeTables<'a> {
    CudaHtj2kDecodeTables {
        vlc_table0: initial_context_lookup,
        vlc_table1: later_context_lookup,
        uvlc_table0: initial_prefix_lookup,
        uvlc_table1: later_prefix_lookup,
    }
}

fn cleanup_job() -> CudaHtj2kCleanupMultiKernelJob {
    CudaHtj2kCleanupMultiKernelJob {
        output_ptr: 0,
        coded_offset: 0,
        width: 64,
        height: 64,
        coded_len: 8,
        cleanup_length: 8,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 8,
        number_of_coding_passes: 1,
        output_stride: 64,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: 0,
        reconstruction: 0,
    }
}

#[test]
fn htj2k_decode_multi_kernel_routes_cleanup_only_jobs() {
    let cleanup = cleanup_job();
    let (_, cleanup_name) = htj2k_decode_multi_kernel_for_jobs(&[cleanup]);
    assert_eq!(
        cleanup_name,
        "j2k_htj2k_decode_codeblocks_multi_cleanup_only"
    );

    let mut refinement = cleanup;
    refinement.refinement_length = 4;
    refinement.number_of_coding_passes = 2;
    let (_, generic_name) = htj2k_decode_multi_kernel_for_jobs(&[refinement]);
    assert_eq!(generic_name, "j2k_htj2k_decode_codeblocks_multi");
}

#[test]
fn htj2k_decode_multi_cleanup_dequant_accepts_only_cleanup_jobs() {
    let cleanup = cleanup_job();
    let (_, kernel_name) = htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[cleanup])
        .expect("cleanup-only jobs use fused cleanup/dequant kernel");
    assert_eq!(
        kernel_name,
        "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize"
    );

    let mut refinement = cleanup;
    refinement.coded_len = 12;
    refinement.refinement_length = 4;
    refinement.number_of_coding_passes = 2;
    assert!(htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[refinement]).is_none());
    refinement.refinement_length = 0;
    assert!(htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[refinement]).is_none());
}

#[test]
fn empty_decode_zero_fills_coefficients_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let engine = J2kCudaEngine::new(&context);
    let initial_context_lookup = [0u16; 1024];
    let later_context_lookup = [0u16; 1024];
    let initial_prefix_lookup = [0u16; 320];
    let later_prefix_lookup = [0u16; 256];
    let output = engine
        .decode_htj2k_codeblocks(
            &[],
            &[],
            tables(
                &initial_context_lookup,
                &later_context_lookup,
                &initial_prefix_lookup,
                &later_prefix_lookup,
            ),
            8,
        )
        .expect("empty HTJ2K decode");
    let mut actual = vec![f32::NAN; 8];
    output
        .coefficients()
        .copy_to_host(<f32 as GpuAbi>::slice_as_bytes_mut(&mut actual))
        .expect("download coefficients");
    assert_eq!(actual, vec![0.0; 8]);
    assert_eq!(output.execution().kernel_dispatches(), 0);
}

#[test]
fn empty_resource_backed_decode_zero_fills_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let engine = J2kCudaEngine::new(&context);
    let initial_context_lookup = [0u16; 1024];
    let later_context_lookup = [0u16; 1024];
    let initial_prefix_lookup = [0u16; 320];
    let later_prefix_lookup = [0u16; 256];
    let table_resources = engine
        .upload_htj2k_decode_table_resources(tables(
            &initial_context_lookup,
            &later_context_lookup,
            &initial_prefix_lookup,
            &later_prefix_lookup,
        ))
        .expect("decode tables");
    let resources = engine
        .upload_htj2k_decode_resources_with_tables(&[], &table_resources)
        .expect("decode resources");
    let output = engine
        .decode_htj2k_codeblocks_with_resources(&resources, &[], 8)
        .expect("resource-backed empty decode");
    let mut actual = vec![f32::NAN; 8];
    output
        .coefficients()
        .copy_to_host(<f32 as GpuAbi>::slice_as_bytes_mut(&mut actual))
        .expect("download coefficients");
    assert_eq!(actual, vec![0.0; 8]);
    assert_eq!(output.execution().kernel_dispatches(), 0);
}

#[test]
fn decode_tables_feed_multiple_payload_uploads_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let engine = J2kCudaEngine::new(&context);
    let initial_context_lookup = [0u16; 1024];
    let later_context_lookup = [0u16; 1024];
    let initial_prefix_lookup = [0u16; 320];
    let later_prefix_lookup = [0u16; 256];
    let table_resources = engine
        .upload_htj2k_decode_table_resources(tables(
            &initial_context_lookup,
            &later_context_lookup,
            &initial_prefix_lookup,
            &later_prefix_lookup,
        ))
        .expect("decode tables");
    let first = engine
        .upload_htj2k_decode_resources_with_tables(&[0xAA, 0x55], &table_resources)
        .expect("first payload");
    let second = engine
        .upload_htj2k_decode_resources_with_tables(&[0x11, 0x22, 0x33], &table_resources)
        .expect("second payload");
    assert!(std::sync::Arc::ptr_eq(
        &first.tables.as_ref().expect("first tables").inner,
        &second.tables.as_ref().expect("second tables").inner,
    ));
    assert_eq!(first.payload_len, 2);
    assert_eq!(second.payload_len, 3);
}

#[test]
fn multi_dequantize_uses_one_dispatch_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let engine = J2kCudaEngine::new(&context);
    let first = context
        .upload(<i32 as GpuAbi>::slice_as_bytes(&[0, 0, 0, 0]))
        .expect("first coefficients");
    let second = context
        .upload(<i32 as GpuAbi>::slice_as_bytes(&[0, 0]))
        .expect("second coefficients");
    let first_jobs = [codeblock_job(2, 2)];
    let second_jobs = [codeblock_job(2, 1)];
    let execution = engine
        .j2k_dequantize_htj2k_codeblocks_multi_device_with_pool(
            &[
                CudaHtj2kDequantizeTarget {
                    coefficients: &first,
                    jobs: &first_jobs,
                    output_words: 4,
                },
                CudaHtj2kDequantizeTarget {
                    coefficients: &second,
                    jobs: &second_jobs,
                    output_words: 2,
                },
            ],
            &context.buffer_pool(),
        )
        .expect("multi-buffer dequantize");
    assert_eq!(execution.kernel_dispatches(), 1);

    let mut first_actual = vec![f32::NAN; 4];
    first
        .copy_to_host(<f32 as GpuAbi>::slice_as_bytes_mut(&mut first_actual))
        .expect("first result");
    let mut second_actual = vec![f32::NAN; 2];
    second
        .copy_to_host(<f32 as GpuAbi>::slice_as_bytes_mut(&mut second_actual))
        .expect("second result");
    assert_eq!(first_actual, vec![0.0; 4]);
    assert_eq!(second_actual, vec![0.0; 2]);
}

#[test]
fn queued_cleanup_metadata_dequantizes_without_second_upload_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let engine = J2kCudaEngine::new(&context);
    let pool = context.buffer_pool();
    let first = context
        .upload(<i32 as GpuAbi>::slice_as_bytes(&[1, i32::MIN + 2, 0, 3]))
        .expect("first coefficients");
    let second = context
        .upload(<i32 as GpuAbi>::slice_as_bytes(&[4, i32::MIN + 5]))
        .expect("second coefficients");
    let jobs = [
        queued_job(first.device_ptr(), 2, 2, 0.5),
        queued_job(second.device_ptr(), 2, 1, 0.25),
    ];
    let jobs_buffer = pool
        .upload(htj2k_cleanup_multi_jobs_as_bytes(&jobs))
        .expect("cleanup metadata");
    let queued = CudaQueuedHtj2kCleanup {
        context: context.clone(),
        resources: vec![jobs_buffer],
        status_buffer: None,
        status_count: jobs.len(),
        status_offset: 0,
        uses_external_status_group: false,
        kernel_name: "j2k_htj2k_decode_codeblocks_multi",
        execution: CudaExecutionStats::default(),
        pool_reuse_guard: None,
        finish_host_live_bytes: 0,
    };
    let execution = engine
        .j2k_dequantize_queued_htj2k_cleanup_with_pool(&queued)
        .expect("dequantize queued cleanup metadata");
    assert_eq!(execution.kernel_dispatches(), 1);

    let mut first_actual = vec![f32::NAN; 4];
    first
        .copy_to_host(<f32 as GpuAbi>::slice_as_bytes_mut(&mut first_actual))
        .expect("first result");
    let mut second_actual = vec![f32::NAN; 2];
    second
        .copy_to_host(<f32 as GpuAbi>::slice_as_bytes_mut(&mut second_actual))
        .expect("second result");
    assert_eq!(first_actual, vec![0.5, -1.0, 0.0, 1.5]);
    assert_eq!(second_actual, vec![1.0, -1.25]);
}

#[test]
fn empty_cleanup_paths_use_no_dispatch_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let engine = J2kCudaEngine::new(&context);
    let pool = context.buffer_pool();
    let initial_context_lookup = [0u16; 1024];
    let later_context_lookup = [0u16; 1024];
    let initial_prefix_lookup = [0u16; 320];
    let later_prefix_lookup = [0u16; 256];
    let table_resources = engine
        .upload_htj2k_decode_table_resources(tables(
            &initial_context_lookup,
            &later_context_lookup,
            &initial_prefix_lookup,
            &later_prefix_lookup,
        ))
        .expect("decode tables");
    let resources = engine
        .upload_htj2k_decode_resources_with_tables(&[], &table_resources)
        .expect("decode resources");

    let (execution, timings) = engine
        .decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(
            &resources,
            &[] as &[CudaHtj2kCleanupTarget<'_>],
            &pool,
            false,
        )
        .expect("empty cleanup batch");
    assert_eq!(execution.kernel_dispatches(), 0);
    assert_eq!(execution.decode_kernel_dispatches(), 0);
    assert_eq!(timings.status_d2h_us, 0);

    // SAFETY: no target borrows a device allocation.
    let queued = unsafe {
        engine.decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
            &resources,
            &[] as &[CudaHtj2kCleanupTarget<'_>],
            &pool,
        )
    }
    .expect("empty queued cleanup batch");
    assert_eq!(queued.execution().kernel_dispatches(), 0);
    assert_eq!(queued.execution().decode_kernel_dispatches(), 0);
    assert_eq!(queued.resource_count(), 0);
    let execution = queued.finish().expect("finish empty cleanup");
    assert_eq!(execution.kernel_dispatches(), 0);
    assert_eq!(execution.decode_kernel_dispatches(), 0);
}

fn codeblock_job(width: u32, height: u32) -> CudaHtj2kCodeBlockJob {
    CudaHtj2kCodeBlockJob {
        payload_offset: 0,
        width,
        height,
        payload_len: 0,
        cleanup_length: 0,
        refinement_length: 0,
        missing_bit_planes: 0,
        num_bitplanes: 1,
        roi_shift: 0,
        number_of_coding_passes: 1,
        output_stride: width,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: false,
        irreversible_midpoint: false,
    }
}

fn queued_job(
    output_ptr: u64,
    width: u32,
    height: u32,
    step: f32,
) -> CudaHtj2kCleanupMultiKernelJob {
    CudaHtj2kCleanupMultiKernelJob {
        output_ptr,
        coded_offset: 0,
        width,
        height,
        coded_len: 0,
        cleanup_length: 0,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 31,
        number_of_coding_passes: 1,
        output_stride: width,
        output_offset: 0,
        dequantization_step: step,
        stripe_causal: 0,
        reconstruction: 0,
    }
}
