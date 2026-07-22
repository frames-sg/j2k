// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    checked_blit_command_encoder, checked_buffer_read_vec, checked_command_buffer,
    checked_command_queue, checked_shared_buffer_for_len, checked_shared_buffer_with_slice,
    system_default_device, MetalImageDestination, MetalImageLayout, MetalSupportError,
    ResidentMetalImage, SubmittedMetalImages,
};
use j2k_core::{DeviceSubmission, PixelFormat};

#[test]
fn metal_image_layout_rejects_short_pitch_and_overflow() {
    assert!(matches!(
        MetalImageLayout::new(0, (0, 2), 0, PixelFormat::Gray8),
        Err(MetalSupportError::MetalImageLayout { .. })
    ));
    assert!(matches!(
        MetalImageLayout::new(0, (2, 0), 2, PixelFormat::Gray8),
        Err(MetalSupportError::MetalImageLayout { .. })
    ));
    assert!(matches!(
        MetalImageLayout::new(0, (4, 2), 11, PixelFormat::Rgb8),
        Err(MetalSupportError::MetalImageLayout { .. })
    ));
    assert!(matches!(
        MetalImageLayout::new(0, (1, 2), usize::MAX, PixelFormat::Gray8),
        Err(MetalSupportError::MetalImageLayout { .. })
    ));
}

#[test]
fn metal_image_batch_layout_allows_unaligned_gray8_item_offsets_from_aligned_base() {
    let layout = MetalImageLayout::new_batch(4, (3, 3), 3, PixelFormat::Gray8, 2, 9)
        .expect("valid odd Gray8 batch layout");

    assert_eq!(layout.byte_offset(), 4);
    assert_eq!(layout.image_count(), 2);
    assert_eq!(layout.image_stride_bytes(), 9);
    assert_eq!(layout.image_offset_bytes(0), Some(0));
    assert_eq!(layout.image_offset_bytes(1), Some(9));
    assert_eq!(layout.image_offset_bytes(2), None);
    assert_eq!(layout.byte_len(), 18);
    assert!(matches!(
        MetalImageLayout::new_batch(4, (3, 3), 3, PixelFormat::Gray8, 0, 9),
        Err(MetalSupportError::MetalImageLayout { .. })
    ));
    assert!(matches!(
        MetalImageLayout::new_batch(4, (3, 3), 3, PixelFormat::Gray8, 2, 8),
        Err(MetalSupportError::MetalImageLayout { .. })
    ));
}

#[test]
fn resident_metal_image_validates_bounds_and_device_identity() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let buffer = checked_shared_buffer_for_len::<u8>(&device, 9).expect("bounded buffer");
    let layout =
        MetalImageLayout::new(1, (4, 2), 4, PixelFormat::Gray8).expect("valid image layout");

    // SAFETY: The buffer has never been submitted or aliased for mutation.
    let image = unsafe { ResidentMetalImage::from_completed_buffer(buffer, layout) }
        .expect("bounded resident image");
    assert_eq!(image.device_registry_id(), device.registry_id());
    assert_eq!(image.byte_len(), 8);

    let out_of_bounds =
        MetalImageLayout::new(2, (4, 2), 4, PixelFormat::Gray8).expect("valid standalone layout");
    assert!(matches!(
        image.view(out_of_bounds),
        Err(MetalSupportError::BufferBounds { .. })
    ));
}

#[test]
fn resident_metal_image_views_stay_within_the_parent_image_range() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let buffer = checked_shared_buffer_for_len::<u8>(&device, 16).expect("bounded buffer");
    let parent =
        MetalImageLayout::new(4, (4, 2), 4, PixelFormat::Gray8).expect("parent image layout");
    // SAFETY: The buffer has never been submitted or aliased for mutation.
    let image = unsafe { ResidentMetalImage::from_completed_buffer(buffer, parent) }
        .expect("resident parent image");

    let child =
        MetalImageLayout::new(8, (4, 1), 4, PixelFormat::Gray8).expect("child image layout");
    assert!(image.view(child).is_ok());

    let sibling = MetalImageLayout::new(12, (4, 1), 4, PixelFormat::Gray8)
        .expect("allocation-bounded sibling layout");
    assert!(matches!(
        image.view(sibling),
        Err(MetalSupportError::MetalImageLayout { .. })
    ));
}

#[test]
fn submitted_metal_images_waits_and_returns_immutable_outputs() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let queue = checked_command_queue(&device).expect("Metal command queue");
    let command_buffer = checked_command_buffer(&queue).expect("Metal command buffer");
    let output = checked_shared_buffer_for_len::<u8>(&device, 4).expect("output buffer");
    let blit = checked_blit_command_encoder(&command_buffer).expect("Metal blit encoder");
    blit.fill_buffer(&output, metal::NSRange::new(0, 4), 7);
    blit.end_encoding();
    let layout = MetalImageLayout::new(0, (4, 1), 4, PixelFormat::Gray8).expect("output layout");

    // SAFETY: `output` is fresh, the command buffer is its only writer, and no
    // raw handle to the allocation survives this move into the submission.
    let submitted = unsafe {
        SubmittedMetalImages::from_uncommitted(
            &device,
            command_buffer,
            vec![(output, layout)],
            vec![],
        )
    }
    .expect("valid Metal image submission");
    let images = submitted.wait().expect("completed Metal image submission");

    assert_eq!(images.len(), 1);
    assert_eq!(images[0].dimensions(), (4, 1));
    assert_eq!(images[0].pixel_format(), PixelFormat::Gray8);
}

#[test]
fn submitted_metal_images_retain_inputs_until_completion() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let queue = checked_command_queue(&device).expect("Metal command queue");
    let command_buffer = checked_command_buffer(&queue).expect("Metal command buffer");
    let input_buffer =
        checked_shared_buffer_with_slice(&device, &[1u8, 2, 3, 4]).expect("resident input buffer");
    let layout =
        MetalImageLayout::new(0, (4, 1), 4, PixelFormat::Gray8).expect("resident input layout");
    // SAFETY: The synchronous upload is complete and no writable alias is
    // retained after the buffer is moved into the resident image.
    let input = unsafe { ResidentMetalImage::from_completed_buffer(input_buffer, layout) }
        .expect("resident input image");
    let output = checked_shared_buffer_for_len::<u8>(&device, 4).expect("output buffer");
    let blit = checked_blit_command_encoder(&command_buffer).expect("Metal blit encoder");
    // SAFETY: The raw input is bound only for a read retained by `inputs` until
    // the submission completes.
    blit.copy_from_buffer(unsafe { input.raw_buffer() }, 0, &output, 0, 4);
    blit.end_encoding();

    // SAFETY: `output` is fresh and written only by this command; the resident
    // input bound above is included in the submission keepalives.
    let submitted = unsafe {
        SubmittedMetalImages::from_uncommitted(
            &device,
            command_buffer,
            vec![(output, layout)],
            vec![input.clone()],
        )
    }
    .expect("valid retained-input submission");
    drop(input);
    let images = submitted.wait().expect("retained-input completion");
    // SAFETY: The submission has completed and the shared output is immutable.
    let bytes =
        unsafe { checked_buffer_read_vec::<u8>(images[0].raw_buffer(), 0, images[0].byte_len()) }
            .expect("resident output readback");

    assert_eq!(bytes, [1, 2, 3, 4]);
}

#[test]
fn dropping_submitted_metal_images_completes_the_command_buffer() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let queue = checked_command_queue(&device).expect("Metal command queue");
    let command_buffer = checked_command_buffer(&queue).expect("Metal command buffer");
    let observed_command_buffer = command_buffer.clone();
    let output = checked_shared_buffer_for_len::<u8>(&device, 4).expect("output buffer");
    let blit = checked_blit_command_encoder(&command_buffer).expect("Metal blit encoder");
    blit.fill_buffer(&output, metal::NSRange::new(0, 4), 9);
    blit.end_encoding();
    let layout = MetalImageLayout::new(0, (4, 1), 4, PixelFormat::Gray8).expect("output layout");

    // SAFETY: `output` is fresh, the command buffer is its only writer, and no
    // raw allocation handle survives the move into the submission.
    let submitted = unsafe {
        SubmittedMetalImages::from_uncommitted(
            &device,
            command_buffer,
            vec![(output, layout)],
            vec![],
        )
    }
    .expect("valid Metal image submission");
    drop(submitted);

    assert_eq!(
        observed_command_buffer.status(),
        metal::MTLCommandBufferStatus::Completed
    );
}

#[test]
fn metal_image_device_identity_reports_mismatched_registry_ids() {
    assert!(matches!(
        crate::resident::validate_registry_id(11, 29),
        Err(MetalSupportError::MetalImageDeviceMismatch {
            image_registry_id: 11,
            requested_registry_id: 29,
        })
    ));
}

#[test]
fn metal_image_destination_rejects_unaligned_and_out_of_bounds_subranges() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let buffer = checked_shared_buffer_for_len::<u8>(&device, 16).expect("bounded buffer");
    let unaligned =
        MetalImageLayout::new(1, (4, 1), 4, PixelFormat::Gray8).expect("standalone layout");

    // SAFETY: No CPU or GPU operation accesses this fresh allocation while
    // destination construction validates the proposed exclusive range.
    let error = unsafe { MetalImageDestination::from_exclusive_buffer(buffer.clone(), unaligned) }
        .expect_err("unaligned destination");
    assert_eq!(
        error,
        MetalSupportError::BufferAlignment {
            offset_bytes: 1,
            align: 4,
        }
    );

    let out_of_bounds =
        MetalImageLayout::new(12, (8, 1), 8, PixelFormat::Gray8).expect("standalone layout");
    // SAFETY: As above, construction is the only access to the fresh buffer.
    let error = unsafe { MetalImageDestination::from_exclusive_buffer(buffer, out_of_bounds) }
        .expect_err("out-of-bounds destination");
    assert_eq!(
        error,
        MetalSupportError::BufferBounds {
            offset_bytes: 12,
            byte_len: 8,
            buffer_len: 16,
        }
    );
}

#[test]
fn metal_image_destination_validates_device_and_preserves_subrange_layout() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let Ok(device) = system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let buffer = checked_shared_buffer_for_len::<u8>(&device, 32).expect("bounded buffer");
    let layout =
        MetalImageLayout::new(8, (4, 2), 4, PixelFormat::Gray8).expect("valid subrange layout");

    // SAFETY: The test keeps no concurrent CPU/GPU access to this destination.
    let destination = unsafe { MetalImageDestination::from_exclusive_buffer(buffer, layout) }
        .expect("valid destination");

    assert_eq!(destination.layout(), layout);
    assert_eq!(destination.device_registry_id(), device.registry_id());
    destination
        .validate_device(&device)
        .expect("matching destination device");
}
