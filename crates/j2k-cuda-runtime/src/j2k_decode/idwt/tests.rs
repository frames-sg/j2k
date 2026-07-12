// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    CudaContext, CudaDeviceBuffer, CudaError, CudaJ2kIdwtJob, CudaJ2kIdwtTarget, CudaJ2kRect,
};

fn zero_job() -> CudaJ2kIdwtJob {
    CudaJ2kIdwtJob {
        rect: CudaJ2kRect::default(),
        ll_rect: CudaJ2kRect::default(),
        hl_rect: CudaJ2kRect::default(),
        lh_rect: CudaJ2kRect::default(),
        hh_rect: CudaJ2kRect::default(),
        irreversible97: 0,
    }
}

fn two_by_two_job() -> CudaJ2kIdwtJob {
    let band_rect = CudaJ2kRect {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        ll_rect: band_rect,
        hl_rect: band_rect,
        lh_rect: band_rect,
        hh_rect: band_rect,
        irreversible97: 0,
    }
}

fn zero_target<'a>(
    input: &'a CudaDeviceBuffer,
    output: &'a CudaDeviceBuffer,
) -> CudaJ2kIdwtTarget<'a> {
    CudaJ2kIdwtTarget {
        ll: input,
        hl: input,
        lh: input,
        hh: input,
        output,
        job: zero_job(),
    }
}

fn assert_invalid_argument_contains<T>(result: Result<T, CudaError>, expected: &str) {
    match result {
        Err(CudaError::InvalidArgument { message }) => {
            assert!(message.contains(expected), "unexpected error: {message}");
        }
        Err(error) => panic!("expected invalid argument containing {expected:?}, got {error}"),
        Ok(_) => panic!("expected invalid argument containing {expected:?}"),
    }
}

#[test]
fn zero_work_batch_rejects_foreign_buffers_and_pool() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("launch CUDA context");
    let foreign_context = CudaContext::system_default().expect("foreign CUDA context");
    let local_pool = context.buffer_pool();
    let foreign_pool = foreign_context.buffer_pool();
    let foreign = foreign_context
        .allocate(std::mem::size_of::<f32>())
        .expect("foreign IDWT buffer");
    let foreign_target = [zero_target(&foreign, &foreign)];

    assert_invalid_argument_contains(
        context.j2k_inverse_dwt_batch_device_with_pool(&foreign_target, &local_pool),
        "must belong to the launch context",
    );
    assert_invalid_argument_contains(
        context.j2k_inverse_dwt_batch_device_with_pool(&[], &foreign_pool),
        "must belong to the launch context",
    );
}

#[test]
fn zero_work_batch_rejects_output_input_aliasing() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let buffer = context
        .allocate(std::mem::size_of::<f32>())
        .expect("IDWT buffer");
    let target = [zero_target(&buffer, &buffer)];

    assert_invalid_argument_contains(
        context.j2k_inverse_dwt_batch_device_with_pool(&target, &pool),
        "overlaps a concurrently read input",
    );
}

#[test]
fn zero_work_sequence_allows_aliasing_only_across_ordered_stages() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let first_input = context
        .allocate(std::mem::size_of::<f32>())
        .expect("first-stage input");
    let intermediate = context
        .allocate(std::mem::size_of::<f32>())
        .expect("intermediate buffer");
    let final_output = context
        .allocate(std::mem::size_of::<f32>())
        .expect("final output");
    let first_stage = [zero_target(&first_input, &intermediate)];
    let second_stage = [zero_target(&intermediate, &final_output)];

    // SAFETY: zero-sized jobs enqueue no CUDA work; all buffers also remain
    // live through the returned execution's completion below.
    let queued = unsafe {
        context
            .j2k_inverse_dwt_batch_sequence_enqueue_with_pool(&[&first_stage, &second_stage], &pool)
    }
    .expect("ordered cross-stage aliasing");
    assert_eq!(queued.execution().kernel_dispatches(), 0);
    queued.finish().expect("zero-work sequence completion");
}

#[test]
fn undersized_idwt_input_is_rejected_before_pool_allocation_or_launch() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let undersized_ll = context.allocate(0).expect("empty LL buffer");
    let band = context
        .allocate(std::mem::size_of::<f32>())
        .expect("IDWT band buffer");
    let batch_output = context
        .allocate(4 * std::mem::size_of::<f32>())
        .expect("IDWT batch output");
    let cached_before = pool.cached_count().expect("pool state before validation");

    assert_eq!(
        context
            .j2k_inverse_dwt_single_output_bytes(&band, &band, &band, &band, two_by_two_job())
            .expect("valid IDWT preflight"),
        4 * std::mem::size_of::<f32>()
    );
    assert_invalid_argument_contains(
        context.j2k_inverse_dwt_single_output_bytes(
            &undersized_ll,
            &band,
            &band,
            &band,
            two_by_two_job(),
        ),
        "LL buffer is too small",
    );
    assert_invalid_argument_contains(
        context.j2k_inverse_dwt_single_device_with_pool(
            &undersized_ll,
            &band,
            &band,
            &band,
            two_by_two_job(),
            &pool,
        ),
        "LL buffer is too small",
    );
    assert_eq!(
        pool.cached_count().expect("pool state after validation"),
        cached_before,
        "invalid input must be rejected before taking a pooled output"
    );

    let targets = [CudaJ2kIdwtTarget {
        ll: &undersized_ll,
        hl: &band,
        lh: &band,
        hh: &band,
        output: &batch_output,
        job: two_by_two_job(),
    }];
    assert_invalid_argument_contains(
        context.j2k_inverse_dwt_batch_device_with_pool(&targets, &pool),
        "LL buffer is too small",
    );
}
