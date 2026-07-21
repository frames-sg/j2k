// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::EncodedImage;
use j2k_core::PixelFormat;

use super::super::{
    MetalBackendSession, MetalBatchDecoder, MetalImageDestination, MetalImageLayout,
};
use super::fixtures::{gray8_cpu_oracle, gray8_destination, prepared_gray_group};

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
