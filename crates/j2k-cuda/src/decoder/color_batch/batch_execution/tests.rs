// SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::NonZeroUsize;

use super::execution::{
    should_pipeline_color_entropy_groups, COLOR_ENTROPY_PIPELINE_GROUP_IMAGES,
    COLOR_HT_PAYLOAD_CHUNK_BYTES,
};
use super::preparation::prepare_color_cuda_resident_batch;
use super::test_fixtures::{four_level_rgb8_fixture, invalidate_first_cleanup_scup};
use crate::decoder::color_batch::COLOR_BATCH_CHUNK_IMAGES;
use crate::{CudaSession, Error, J2kDecoder, Surface};
use j2k_core::{HtGpuJobChunkLimits, PixelFormat};

#[test]
fn color_entropy_pipeline_gate_requires_large_unprofiled_second_group() {
    let group = COLOR_ENTROPY_PIPELINE_GROUP_IMAGES;
    let max_payload = COLOR_HT_PAYLOAD_CHUNK_BYTES;
    // Two chunks' worth of bytes spread over one group: the second group
    // pushes the batch over the pipelining threshold.
    let group_sized = vec![0_u8; 2 * max_payload / group + 1];
    let over_group_large = vec![group_sized.as_slice(); group + 1];
    let group_large = vec![group_sized.as_slice(); group];
    let small = [0_u8; 1];
    let over_group_small = vec![small.as_slice(); group + 1];

    assert!(should_pipeline_color_entropy_groups(
        &over_group_large,
        false,
        max_payload,
    ));
    assert!(!should_pipeline_color_entropy_groups(
        &over_group_large,
        true,
        max_payload,
    ));
    assert!(!should_pipeline_color_entropy_groups(
        &group_large,
        false,
        max_payload,
    ));
    assert!(!should_pipeline_color_entropy_groups(
        &over_group_small,
        false,
        max_payload,
    ));
    assert!(should_pipeline_color_entropy_groups(
        &over_group_small,
        false,
        2
    ));
}

fn force_small_entropy_pipeline(session: &mut CudaSession, inputs: &[&[u8]]) {
    let total_bytes = inputs.iter().map(|input| input.len()).sum::<usize>();
    let largest_input = inputs.iter().map(|input| input.len()).max().unwrap_or(1);
    let max_payload_bytes = largest_input.max(total_bytes / 4).max(1);
    session.set_htj2k_decode_chunk_limits_for_test(HtGpuJobChunkLimits::new(
        NonZeroUsize::new(1_000_000).expect("nonzero job limit"),
        max_payload_bytes,
        usize::MAX,
    ));
    assert!(should_pipeline_color_entropy_groups(
        inputs,
        false,
        max_payload_bytes,
    ));
}

fn force_uploaded_entropy_pipeline(session: &mut CudaSession, inputs: &[&[u8]]) {
    let first_group_bytes = inputs[..4].iter().map(|input| input.len()).sum::<usize>();
    let total_bytes = inputs.iter().map(|input| input.len()).sum::<usize>();
    assert!(total_bytes > first_group_bytes.saturating_mul(2));
    session.set_htj2k_decode_chunk_limits_for_test(HtGpuJobChunkLimits::new(
        NonZeroUsize::new(1_000_000).expect("nonzero job limit"),
        first_group_bytes,
        usize::MAX,
    ));
    assert!(should_pipeline_color_entropy_groups(
        inputs,
        false,
        first_group_bytes,
    ));
}

#[test]
fn pipelined_color_entropy_preserves_order_across_four_plus_two_groups() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let (first, first_pixels) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(129, 131);
    let (second, second_pixels) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(257, 129);
    let inputs = [
        first.as_slice(),
        second.as_slice(),
        first.as_slice(),
        second.as_slice(),
        first.as_slice(),
        second.as_slice(),
    ];
    let expected = [
        first_pixels.as_slice(),
        second_pixels.as_slice(),
        first_pixels.as_slice(),
        second_pixels.as_slice(),
        first_pixels.as_slice(),
        second_pixels.as_slice(),
    ]
    .concat();
    let mut session = CudaSession::default();
    force_small_entropy_pipeline(&mut session, &inputs);

    let surfaces =
        J2kDecoder::decode_batch_to_device_with_session(&inputs, PixelFormat::Rgb8, &mut session)
            .expect("pipelined 4+2 color batch");
    assert_eq!(surfaces.len(), inputs.len());
    assert_eq!(
        Surface::download_batch_tight(&surfaces).expect("download pipelined batch"),
        expected
    );
    assert!(
        session.last_htj2k_decode_chunk_count_for_test() > 1,
        "the forced payload limit must exercise chunked entropy within the pipeline"
    );
    let batch = session
        .decode_pool_diagnostics()
        .expect("completed pipeline pools")
        .batch_decode
        .expect("batch pool");
    assert_eq!(batch.reuse_holds, 0);
    assert_eq!(batch.deferred_buffers, 0);
}

#[test]
fn large_surface_batch_preserves_order_and_reuses_bounded_scratch() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let (first, first_pixels) = four_level_rgb8_fixture(17);
    let (second, second_pixels) = four_level_rgb8_fixture(29);
    let mut inputs = Vec::with_capacity(65);
    let mut expected = Vec::with_capacity(first_pixels.len() * 65);
    for index in 0_usize..65 {
        let (encoded, pixels) = if index.is_multiple_of(2) {
            (&first, &first_pixels)
        } else {
            (&second, &second_pixels)
        };
        inputs.push(encoded.as_slice());
        expected.extend_from_slice(pixels);
    }

    let mut session = CudaSession::default();
    let surfaces =
        J2kDecoder::decode_batch_to_device_with_session(&inputs, PixelFormat::Rgb8, &mut session)
            .expect("first bounded surface batch");
    assert_eq!(surfaces.len(), inputs.len());
    assert_eq!(
        Surface::download_batch_tight(&surfaces).expect("first bounded batch readback"),
        expected
    );
    drop(surfaces);

    let allocations_before = session
        .diagnostics()
        .expect("warm bounded-batch diagnostics")
        .runtime
        .expect("CUDA runtime")
        .device_allocation_operations;
    let surfaces =
        J2kDecoder::decode_batch_to_device_with_session(&inputs, PixelFormat::Rgb8, &mut session)
            .expect("reused bounded surface batch");
    let runtime_before_readback = session
        .diagnostics()
        .expect("pre-readback diagnostics")
        .runtime
        .expect("CUDA runtime");
    assert_eq!(
        Surface::download_batch_tight(&surfaces).expect("reused bounded batch readback"),
        expected
    );
    let runtime_after_readback = session
        .diagnostics()
        .expect("post-readback diagnostics")
        .runtime
        .expect("CUDA runtime");
    assert_eq!(
        runtime_after_readback.device_to_host_operations
            - runtime_before_readback.device_to_host_operations,
        u64::try_from(inputs.len().div_ceil(COLOR_BATCH_CHUNK_IMAGES))
            .expect("chunk count fits in u64"),
        "readback must copy each chunk's contiguous outputs with one transfer"
    );
    drop(surfaces);
    let allocations_after = session
        .diagnostics()
        .expect("reused bounded-batch diagnostics")
        .runtime
        .expect("CUDA runtime")
        .device_allocation_operations;
    assert_eq!(
        allocations_after, allocations_before,
        "a warm large batch must reuse the bounded scratch and output working sets"
    );
}

#[test]
fn chunked_surface_batch_drops_completed_prefix_after_late_plan_error() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let (valid, valid_pixels) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(129, 131);
    let invalid = [0x00_u8, 0x01];
    let mut inputs = vec![valid.as_slice(); 33];
    inputs.push(invalid.as_slice());
    let expected = prepare_color_cuda_resident_batch(&[invalid.as_slice()], PixelFormat::Rgb8)
        .err()
        .expect("invalid color input");
    let mut session = CudaSession::default();

    let actual =
        J2kDecoder::decode_batch_to_device_with_session(&inputs, PixelFormat::Rgb8, &mut session)
            .expect_err("the second surface chunk must reject its invalid input");
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    let batch = session
        .decode_pool_diagnostics()
        .expect("retired chunk failure pools")
        .batch_decode
        .expect("batch pool");
    assert_eq!(batch.reuse_holds, 0);
    assert_eq!(batch.deferred_buffers, 0);

    let surfaces = J2kDecoder::decode_batch_to_device_with_session(
        &[valid.as_slice()],
        PixelFormat::Rgb8,
        &mut session,
    )
    .expect("session reuse after late chunk plan failure");
    assert_eq!(
        Surface::download_batch_tight(&surfaces).expect("download reused session output"),
        valid_pixels
    );
}

#[test]
fn pipelined_color_entropy_retires_prefix_before_returning_late_plan_error() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let (valid, valid_pixels) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(129, 131);
    let invalid = [0x00_u8, 0x01];
    let inputs = [
        valid.as_slice(),
        valid.as_slice(),
        valid.as_slice(),
        valid.as_slice(),
        invalid.as_slice(),
    ];
    let expected = prepare_color_cuda_resident_batch(&[invalid.as_slice()], PixelFormat::Rgb8)
        .err()
        .expect("invalid color input");
    let mut session = CudaSession::default();
    force_small_entropy_pipeline(&mut session, &inputs);
    let context = session
        .cuda_context()
        .expect("CUDA context for pipeline diagnostics");
    let launches_before = context
        .diagnostics()
        .expect("diagnostics before pipelined failure")
        .kernel_launches;

    let actual =
        J2kDecoder::decode_batch_to_device_with_session(&inputs, PixelFormat::Rgb8, &mut session)
            .expect_err("the second entropy group must reject its invalid input");
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    assert!(
        context
            .diagnostics()
            .expect("diagnostics after pipelined failure")
            .kernel_launches
            > launches_before,
        "the valid prefix must enqueue entropy before the later plan error"
    );
    let batch = session
        .decode_pool_diagnostics()
        .expect("retired failure pools")
        .batch_decode
        .expect("batch pool");
    assert_eq!(batch.reuse_holds, 0);
    assert_eq!(batch.deferred_buffers, 0);

    let surfaces = J2kDecoder::decode_batch_to_device_with_session(
        &[valid.as_slice()],
        PixelFormat::Rgb8,
        &mut session,
    )
    .expect("session reuse after pipelined plan failure");
    assert_eq!(
        Surface::download_batch_tight(&surfaces).expect("download reused session output"),
        valid_pixels
    );
}

#[test]
fn pipelined_color_entropy_maps_late_kernel_failure_to_global_source_and_job() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let (valid, _) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(129, 131);
    let mut corrupted = valid.clone();
    let (corrupted_job, jobs_per_image) = invalidate_first_cleanup_scup(&mut corrupted);
    assert_late_kernel_failure_mapping(
        &valid,
        &corrupted,
        corrupted_job,
        jobs_per_image,
        6,
        4,
        true,
    );
    assert_late_kernel_failure_mapping(
        &valid,
        &corrupted,
        corrupted_job,
        jobs_per_image,
        9,
        8,
        false,
    );
    assert_late_kernel_failure_mapping(
        &valid,
        &corrupted,
        corrupted_job,
        jobs_per_image,
        34,
        33,
        true,
    );
}

#[test]
fn unpipelined_color_batch_maps_kernel_failure_to_source_and_job() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let (valid, _) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(129, 131);
    let mut corrupted = valid.clone();
    let (corrupted_job, jobs_per_image) = invalidate_first_cleanup_scup(&mut corrupted);
    let inputs = [valid.as_slice(), corrupted.as_slice(), valid.as_slice()];
    assert!(inputs.len() <= COLOR_ENTROPY_PIPELINE_GROUP_IMAGES);
    let mut session = CudaSession::default();

    let error =
        J2kDecoder::decode_batch_to_device_with_session(&inputs, PixelFormat::Rgb8, &mut session)
            .expect_err("invalid cleanup SCUP must fail on the GPU");

    assert_eq!(session.last_htj2k_decode_chunk_count_for_test(), 0);
    assert!(
        matches!(
            error,
            Error::CudaTier1JobFailed {
                source_index: 1,
                original_job_index,
                ..
            } if original_job_index == jobs_per_image + corrupted_job
        ),
        "{error:?}"
    );
}

fn assert_late_kernel_failure_mapping(
    valid: &[u8],
    corrupted: &[u8],
    corrupted_job: usize,
    jobs_per_image: usize,
    input_count: usize,
    corrupted_source: usize,
    chunked: bool,
) {
    let valid_inputs = vec![valid; input_count];
    let mut corrupted_inputs = valid_inputs.clone();
    corrupted_inputs[corrupted_source] = corrupted;
    let mut session = CudaSession::default();
    if chunked {
        force_small_entropy_pipeline(&mut session, &valid_inputs);
    } else {
        force_uploaded_entropy_pipeline(&mut session, &valid_inputs);
    }
    let warm = J2kDecoder::decode_batch_to_device_with_session(
        &valid_inputs,
        PixelFormat::Rgb8,
        &mut session,
    )
    .expect("valid pipelined warm decode");
    drop(warm);
    // Larger surface batches run in groups, so the counter reflects only the last group.
    if input_count <= COLOR_BATCH_CHUNK_IMAGES {
        if chunked {
            assert!(session.last_htj2k_decode_chunk_count_for_test() > 0);
        } else {
            assert_eq!(session.last_htj2k_decode_chunk_count_for_test(), 0);
        }
    }

    let error = J2kDecoder::decode_batch_to_device_with_session(
        &corrupted_inputs,
        PixelFormat::Rgb8,
        &mut session,
    )
    .expect_err("invalid cleanup SCUP must fail on the GPU");
    assert!(matches!(
        error,
        Error::CudaTier1JobFailed {
            source_index,
            original_job_index,
            ..
        } if source_index == corrupted_source
            && original_job_index == jobs_per_image * corrupted_source + corrupted_job
    ));
    let batch = session
        .decode_pool_diagnostics()
        .expect("retired status failure pools")
        .batch_decode
        .expect("batch pool");
    assert_eq!(batch.reuse_holds, 0);
    assert_eq!(batch.deferred_buffers, 0);
}

#[test]
fn bounded_color_entropy_chunks_preserve_mixed_outputs_and_session_reuse() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let (first, first_pixels) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(129, 131);
    let (second, second_pixels) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(257, 129);
    let inputs = [first.as_slice(), second.as_slice(), first.as_slice()];
    let expected = [
        first_pixels.as_slice(),
        second_pixels.as_slice(),
        first_pixels.as_slice(),
    ]
    .concat();
    let mut session = CudaSession::default();
    let context = session
        .cuda_context()
        .expect("CUDA context for launch diagnostics");
    let launches_before_rejection = context
        .diagnostics()
        .expect("diagnostics before rejected sparse chunk plan")
        .kernel_launches;
    let max_jobs = NonZeroUsize::new(4).expect("nonzero chunk jobs");
    session.set_htj2k_decode_chunk_limits_for_test(HtGpuJobChunkLimits::new(
        max_jobs,
        8 * 1024 * 1024,
        j2k_cuda_j2k_engine::htj2k_cleanup_multi_descriptor_bytes() - 1,
    ));
    J2kDecoder::decode_batch_to_device_with_session(&inputs, PixelFormat::Rgb8, &mut session)
        .expect_err("a single cleanup descriptor must exceed the forced chunk cap");
    assert!(
        context
            .diagnostics()
            .expect("diagnostics after rejected sparse chunk plan")
            .kernel_launches
            > launches_before_rejection,
        "sparse coefficients must be cleared before the chunk planner rejects the batch"
    );

    session.set_htj2k_decode_chunk_limits_for_test(HtGpuJobChunkLimits::new(
        max_jobs,
        8 * 1024 * 1024,
        max_jobs.get() * j2k_cuda_j2k_engine::htj2k_cleanup_multi_descriptor_bytes(),
    ));
    for _ in 0..2 {
        let surfaces = J2kDecoder::decode_batch_to_device_with_session(
            &inputs,
            PixelFormat::Rgb8,
            &mut session,
        )
        .expect("bounded color batch");
        assert_eq!(
            Surface::download_batch_tight(&surfaces).expect("download batch"),
            expected
        );
        assert!(
            session.last_htj2k_decode_chunk_count_for_test() > 1,
            "the session's job limit must split this batch into multiple entropy chunks"
        );
        let diagnostics = session.decode_pool_diagnostics().expect("completed pools");
        let batch = diagnostics.batch_decode.expect("batch pool");
        assert_eq!(batch.reuse_holds, 0);
        assert_eq!(batch.deferred_buffers, 0);
    }
}
