// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::*;

fn prepared_gray_group(
    decoder: &MetalBatchDecoder,
) -> Result<PreparedBatch, Box<dyn std::error::Error>> {
    let bytes = Arc::<[u8]>::from(j2k_test_support::htj2k_gray8_fixture(4, 4));
    Ok(decoder.prepare(vec![EncodedImage::full(bytes)])?)
}

fn gray8_cpu_oracle(encoded: &[u8]) -> [u8; 16] {
    let mut expected = [0_u8; 16];
    j2k::J2kDecoder::new(encoded)
        .expect("CPU oracle decoder")
        .decode_into(&mut expected, 4, PixelFormat::Gray8)
        .expect("CPU oracle decode");
    expected
}

fn distinct_gray8_fixture(seed: u8) -> Arc<[u8]> {
    let pixels = (0_u8..16)
        .map(|value| value.wrapping_mul(13).wrapping_add(seed))
        .collect::<Vec<_>>();
    Arc::from(
        j2k_native::encode_htj2k(
            &pixels,
            4,
            4,
            1,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode distinct Gray8 fixture"),
    )
}

fn gray8_destination(
    device: &metal::DeviceRef,
) -> Result<MetalImageDestination, Box<dyn std::error::Error>> {
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(device, 16)?;
    let layout = MetalImageLayout::new_batch(0, (4, 4), 4, PixelFormat::Gray8, 1, 16)?;
    // SAFETY: The fresh allocation has one exclusive owner and the returned
    // submission retains that owner until completion or drop.
    Ok(unsafe { MetalImageDestination::from_exclusive_buffer(buffer, layout)? })
}

fn wrong_size_gray8_destination(
    device: &metal::DeviceRef,
) -> Result<MetalImageDestination, Box<dyn std::error::Error>> {
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(device, 4)?;
    let layout = MetalImageLayout::new_batch(0, (2, 2), 2, PixelFormat::Gray8, 1, 4)?;
    // SAFETY: The fresh allocation has one exclusive owner. Destination
    // preflight rejects its dimensions before any codec work can retain it.
    Ok(unsafe { MetalImageDestination::from_exclusive_buffer(buffer, layout)? })
}

#[test]
fn known_exact_consumer_queue_submits_without_an_event_bridge() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let device = metal::Device::system_default().expect("system Metal device");
    let queue =
        j2k_metal_support::checked_command_queue(&device).expect("exact consumer command queue");
    let backend = MetalBackendSession::with_command_queue(device.clone(), queue.clone())
        .expect("backend using exact consumer queue");
    let mut decoder = MetalBatchDecoder::with_backend_session(backend);
    let encoded = Arc::<[u8]>::from(j2k_test_support::htj2k_gray8_fixture(4, 4));
    let prepared = decoder
        .prepare(vec![EncodedImage::full(encoded.clone())])
        .expect("prepare Gray8 group");
    let destination_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(&device, 16)
        .expect("Gray8 destination buffer");
    let layout = MetalImageLayout::new_batch(0, (4, 4), 4, PixelFormat::Gray8, 1, 16)
        .expect("Gray8 destination layout");
    // SAFETY: the fresh allocation is retained exclusively by the pending
    // submission until completion.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(destination_buffer.clone(), layout)
            .expect("Gray8 destination")
    };

    let pending = decoder
        .submit_prepared_group_into_for_consumer_queue(&prepared.groups()[0], destination, &queue)
        .expect("submit on exact producer/consumer queue");

    assert_eq!(
        pending.submission.ordering_diagnostics_for_test(),
        (false, false, 0),
        "exact-queue submission must not allocate or signal an event or enqueue a wait command"
    );
    pending.wait().expect("exact-queue completion");
    // SAFETY: completion released the destination's exclusive write guard.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&destination_buffer, 0, 16)
            .expect("completed exact-queue pixels")
    };
    assert_eq!(actual, gray8_cpu_oracle(&encoded));
}

#[test]
fn synchronous_external_decode_has_no_consumer_event_bridge() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = prepared_gray_group(&decoder).expect("prepare Gray8 group");
    let destination = gray8_destination(decoder.backend_session().device())
        .expect("synchronous Gray8 destination");
    crate::compute::reset_direct_destination_event_bridge_for_test();

    decoder
        .decode_prepared_group_into(&prepared.groups()[0], &destination)
        .expect("synchronous external decode");

    assert_eq!(
        crate::compute::direct_destination_event_bridge_for_test(),
        (0, 0, 0),
        "host-completed external decode must not allocate, signal, or wait on an event"
    );
}

#[test]
fn codec_owned_resident_decode_has_no_consumer_event_bridge() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = prepared_gray_group(&decoder).expect("prepare Gray8 group");
    crate::compute::reset_direct_destination_event_bridge_for_test();

    let result = decoder
        .decode_prepared(&prepared)
        .expect("codec-owned resident decode");
    assert_eq!(result.groups().len(), 1);
    assert_eq!(
        crate::compute::direct_destination_event_bridge_for_test(),
        (0, 0, 0),
        "codec-owned output hidden behind host completion needs no consumer event bridge"
    );
}

#[test]
fn deferred_async_submission_preserves_the_compatibility_event_bridge() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = prepared_gray_group(&decoder).expect("prepare Gray8 group");
    let destination =
        gray8_destination(decoder.backend_session().device()).expect("async Gray8 destination");
    let consumer_queue =
        j2k_metal_support::checked_command_queue(decoder.backend_session().device())
            .expect("compatibility consumer command queue");
    crate::compute::reset_direct_destination_event_bridge_for_test();

    let mut pending = decoder
        .submit_prepared_group_into(&prepared.groups()[0], destination)
        .expect("deferred async submission");
    pending
        .enqueue_consumer_wait(&consumer_queue)
        .expect("compatibility consumer wait");

    assert_eq!(
        crate::compute::direct_destination_event_bridge_for_test(),
        (1, 1, 1),
        "deferred compatibility submission must retain its event bridge"
    );
    pending.wait().expect("deferred async completion");
}

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
fn overlapping_prepared_submissions_keep_distinct_status_and_scratch_owners() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let encoded = Arc::<[u8]>::from(j2k_test_support::openhtj2k_refinement_fixture());
    let device = metal::Device::system_default().expect("system Metal device");
    let queue = j2k_metal_support::checked_command_queue(&device).expect("overlap command queue");
    let backend = MetalBackendSession::with_command_queue(device.clone(), queue.clone())
        .expect("isolated exact-queue backend");
    let mut decoder = MetalBatchDecoder::with_backend_session(backend);
    let prepared = decoder
        .prepare(vec![EncodedImage::full(encoded.clone())])
        .expect("prepare overlapping HT group");
    let info = prepared.groups()[0].info();
    let pixel_format = info.native_pixel_format().expect("native HT pixel format");
    let row_bytes = info.dimensions.0 as usize * pixel_format.bytes_per_pixel();
    let image_bytes = row_bytes * info.dimensions.1 as usize;

    let allocate_destination = || {
        let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(&device, image_bytes)
            .expect("overlapping destination buffer");
        let layout = MetalImageLayout::new_batch(
            0,
            info.dimensions,
            row_bytes,
            pixel_format,
            1,
            image_bytes,
        )
        .expect("overlapping destination layout");
        // SAFETY: each fresh allocation has one exclusive owner retained by
        // its pending submission until completion.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
                .expect("overlapping destination")
        };
        (destination, buffer)
    };
    let (first_destination, first_buffer) = allocate_destination();
    let (second_destination, second_buffer) = allocate_destination();

    let first = decoder
        .submit_prepared_group_into_for_consumer_queue(
            &prepared.groups()[0],
            first_destination,
            &queue,
        )
        .expect("first overlapping submission");
    let second = decoder
        .submit_prepared_group_into_for_consumer_queue(
            &prepared.groups()[0],
            second_destination,
            &queue,
        )
        .expect("second overlapping submission");

    let (first_status, first_scratch) = first.submission.in_flight_owner_ptrs_for_test();
    let (second_status, second_scratch) = second.submission.in_flight_owner_ptrs_for_test();
    assert!(!first_status.is_empty() && !first_scratch.is_empty());
    assert!(!second_status.is_empty() && !second_scratch.is_empty());
    assert!(first_status
        .iter()
        .all(|owner| !second_status.contains(owner)));
    assert!(first_scratch
        .iter()
        .all(|owner| !second_scratch.contains(owner)));

    first.wait().expect("first overlapping completion");
    second.wait().expect("second overlapping completion");
    let mut expected = vec![0_u8; image_bytes];
    j2k::J2kDecoder::new(&encoded)
        .expect("overlapping CPU oracle decoder")
        .decode_into(&mut expected, row_bytes, pixel_format)
        .expect("overlapping CPU oracle decode");
    // SAFETY: both exclusive codec writes have completed and released their
    // destination guards.
    let first_pixels = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&first_buffer, 0, image_bytes)
            .expect("first overlapping pixels")
    };
    // SAFETY: as above for the independent second destination.
    let second_pixels = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&second_buffer, 0, image_bytes)
            .expect("second overlapping pixels")
    };
    assert_eq!(first_pixels, expected);
    assert_eq!(second_pixels, expected);
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
    let error = super::external::validate_consumer_registry_ids(17, 23)
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
fn dropping_exact_queue_work_leaves_the_session_reusable() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let device = metal::Device::system_default().expect("system Metal device");
    let queue =
        j2k_metal_support::checked_command_queue(&device).expect("exact consumer command queue");
    let backend = MetalBackendSession::with_command_queue(device.clone(), queue.clone())
        .expect("backend using exact consumer queue");
    let mut decoder = MetalBatchDecoder::with_backend_session(backend);
    let prepared = prepared_gray_group(&decoder).expect("prepare Gray8 group");

    let rejected = decoder.submit_prepared_group_into_for_consumer_queue(
        &prepared.groups()[0],
        wrong_size_gray8_destination(&device).expect("wrong-size Gray8 destination"),
        &queue,
    );
    assert!(
        rejected.is_err(),
        "wrong-size destination must fail preflight"
    );
    assert_eq!(
        decoder
            .submissions()
            .expect("submission count after rejection"),
        0,
        "rejected preflight must not increment the submission counter"
    );

    let pending = decoder
        .submit_prepared_group_into_for_consumer_queue(
            &prepared.groups()[0],
            gray8_destination(&device).expect("first Gray8 destination"),
            &queue,
        )
        .expect("first exact-queue submission");
    drop(pending);

    decoder
        .submit_prepared_group_into_for_consumer_queue(
            &prepared.groups()[0],
            gray8_destination(&device).expect("second Gray8 destination"),
            &queue,
        )
        .expect("second exact-queue submission")
        .wait()
        .expect("reused session completion");
    assert_eq!(decoder.submissions().expect("submission count"), 2);
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
