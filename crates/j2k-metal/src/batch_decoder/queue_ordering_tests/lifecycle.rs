// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::EncodedImage;

use super::super::{
    MetalBackendSession, MetalBatchDecoder, MetalImageDestination, MetalImageLayout,
};
use super::fixtures::{gray8_destination, prepared_gray_group, wrong_size_gray8_destination};

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
        decoder.submissions().expect("Metal batch submissions"),
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
    assert_eq!(decoder.submissions().expect("Metal batch submissions"), 2);
}
