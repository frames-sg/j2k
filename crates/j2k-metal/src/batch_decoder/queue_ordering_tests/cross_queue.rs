// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::EncodedImage;
use j2k_core::PixelFormat;

use super::super::external::validate_consumer_registry_ids;
use super::super::{
    Error, MetalBackendSession, MetalBatchDecoder, MetalImageDestination, MetalImageLayout,
};
use super::fixtures::{
    distinct_gray8_fixture, gray8_cpu_oracle, gray8_destination, prepared_gray_group,
};

#[test]
fn known_cross_queue_submissions_reuse_one_monotonic_event_timeline() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let device = metal::Device::system_default().expect("system Metal device");
    let producer_queue =
        j2k_metal_support::checked_command_queue(&device).expect("producer command queue");
    let consumer_queue =
        j2k_metal_support::checked_command_queue(&device).expect("consumer command queue");
    let backend = MetalBackendSession::with_command_queue(device.clone(), producer_queue)
        .expect("backend using producer queue");
    let mut decoder = MetalBatchDecoder::with_backend_session(backend);
    let prepared = prepared_gray_group(&decoder).expect("prepare Gray8 group");

    let first = decoder
        .submit_prepared_group_into_for_consumer_queue(
            &prepared.groups()[0],
            gray8_destination(&device).expect("first Gray8 destination"),
            &consumer_queue,
        )
        .expect("first cross-queue submission");
    let second = decoder
        .submit_prepared_group_into_for_consumer_queue(
            &prepared.groups()[0],
            gray8_destination(&device).expect("second Gray8 destination"),
            &consumer_queue,
        )
        .expect("second cross-queue submission");

    let (first_event, first_value) = first
        .submission
        .known_consumer_timeline_for_test()
        .expect("first cross-queue event dependency");
    let (second_event, second_value) = second
        .submission
        .known_consumer_timeline_for_test()
        .expect("second cross-queue event dependency");
    assert_eq!(
        first_event, second_event,
        "session must reuse one MTL event"
    );
    assert_eq!((first_value, second_value), (1, 2));
    assert_eq!(first.submission.ordering_diagnostics_for_test().2, 1);
    assert_eq!(second.submission.ordering_diagnostics_for_test().2, 1);

    first.wait().expect("first cross-queue completion");
    second.wait().expect("second cross-queue completion");
}

#[test]
fn known_cross_queue_wait_orders_subsequent_consumer_work() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let encoded = Arc::<[u8]>::from(j2k_test_support::htj2k_gray8_fixture(4, 4));
    let device = metal::Device::system_default().expect("system Metal device");
    let producer_queue =
        j2k_metal_support::checked_command_queue(&device).expect("producer command queue");
    let consumer_queue =
        j2k_metal_support::checked_command_queue(&device).expect("consumer command queue");
    let backend = MetalBackendSession::with_command_queue(device.clone(), producer_queue)
        .expect("backend using producer queue");
    let mut decoder = MetalBatchDecoder::with_backend_session(backend);
    let prepared = decoder
        .prepare(vec![EncodedImage::full(encoded.clone())])
        .expect("prepare Gray8 group");
    let destination_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(&device, 16)
        .expect("destination buffer");
    let copy_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(&device, 16)
        .expect("consumer copy buffer");
    let layout = MetalImageLayout::new_batch(0, (4, 4), 4, PixelFormat::Gray8, 1, 16)
        .expect("destination layout");
    // SAFETY: The fresh destination is retained exclusively by the pending
    // decode until producer completion.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(destination_buffer.clone(), layout)
            .expect("exclusive destination")
    };
    let pending = decoder
        .submit_prepared_group_into_for_consumer_queue(
            &prepared.groups()[0],
            destination,
            &consumer_queue,
        )
        .expect("cross-queue submission");

    let consumer_command =
        j2k_metal_support::checked_command_buffer(&consumer_queue).expect("consumer command");
    let blit =
        j2k_metal_support::checked_blit_command_encoder(&consumer_command).expect("consumer blit");
    blit.copy_from_buffer(&destination_buffer, 0, &copy_buffer, 0, 16);
    blit.end_encoding();
    consumer_command.commit();

    pending.wait().expect("producer completion");
    j2k_metal_support::wait_for_completion(&consumer_command).expect("consumer completion");
    // SAFETY: Both the producer and event-ordered consumer copy have completed.
    let copied = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&copy_buffer, 0, 16)
            .expect("copied pixels")
    };
    assert_eq!(copied, gray8_cpu_oracle(&encoded));
}

#[test]
fn consumer_queue_registry_mismatch_is_rejected() {
    let error = validate_consumer_registry_ids(17, 23)
        .expect_err("different Metal devices must be rejected");
    assert!(matches!(
        error,
        Error::MetalSupport {
            source: j2k_metal_support::MetalSupportError::MetalImageDeviceMismatch {
                image_registry_id: 17,
                requested_registry_id: 23,
            },
            ..
        }
    ));
}

#[test]
fn cloned_sessions_assign_cross_queue_values_in_commit_order() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let device = metal::Device::system_default().expect("system Metal device");
    let producer_queue =
        j2k_metal_support::checked_command_queue(&device).expect("producer command queue");
    let consumer_queue =
        j2k_metal_support::checked_command_queue(&device).expect("consumer command queue");
    let backend = MetalBackendSession::with_command_queue(device, producer_queue)
        .expect("backend using producer queue");
    let start = Arc::new(std::sync::Barrier::new(2));

    let observations = std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for seed in [0_u8, 31] {
            let backend = backend.clone();
            let consumer_queue = consumer_queue.clone();
            let start = start.clone();
            workers.push(scope.spawn(move || {
                let mut decoder = MetalBatchDecoder::with_backend_session(backend);
                let encoded = distinct_gray8_fixture(seed);
                let prepared = decoder
                    .prepare(vec![EncodedImage::full(encoded.clone())])
                    .expect("prepare worker group");
                let destination_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
                    decoder.backend_session().device(),
                    16,
                )
                .expect("worker destination buffer");
                let copy_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
                    decoder.backend_session().device(),
                    16,
                )
                .expect("worker consumer copy buffer");
                let layout = MetalImageLayout::new_batch(0, (4, 4), 4, PixelFormat::Gray8, 1, 16)
                    .expect("worker destination layout");
                // SAFETY: the fresh allocation is retained exclusively by
                // the pending decode until producer completion.
                let destination = unsafe {
                    MetalImageDestination::from_exclusive_buffer(destination_buffer.clone(), layout)
                        .expect("worker destination")
                };
                start.wait();
                let pending = decoder
                    .submit_prepared_group_into_for_consumer_queue(
                        &prepared.groups()[0],
                        destination,
                        &consumer_queue,
                    )
                    .expect("worker cross-queue submission");
                let timeline = pending
                    .submission
                    .known_consumer_timeline_for_test()
                    .expect("worker timeline");
                let consumer_command = j2k_metal_support::checked_command_buffer(&consumer_queue)
                    .expect("worker consumer command");
                let blit = j2k_metal_support::checked_blit_command_encoder(&consumer_command)
                    .expect("worker consumer blit");
                blit.copy_from_buffer(&destination_buffer, 0, &copy_buffer, 0, 16);
                blit.end_encoding();
                consumer_command.commit();
                pending.wait().expect("worker completion");
                j2k_metal_support::wait_for_completion(&consumer_command)
                    .expect("worker consumer completion");
                // SAFETY: both the producer and event-ordered consumer copy
                // completed before this read.
                let copied = unsafe {
                    j2k_metal_support::checked_buffer_read_vec::<u8>(&copy_buffer, 0, 16)
                        .expect("worker copied pixels")
                };
                (timeline, copied, gray8_cpu_oracle(&encoded))
            }));
        }
        workers
            .into_iter()
            .map(|worker| worker.join().expect("worker thread"))
            .collect::<Vec<_>>()
    });

    let first_timeline = observations[0].0;
    let second_timeline = observations[1].0;
    assert_eq!(first_timeline.0, second_timeline.0);
    let mut values = [first_timeline.1, second_timeline.1];
    values.sort_unstable();
    assert_eq!(values, [1, 2]);
    for (_, copied, expected) in &observations {
        assert_eq!(copied, expected);
    }
    assert_ne!(
        observations[0].1, observations[1].1,
        "concurrent consumer observations must come from distinct producer outputs"
    );
}
