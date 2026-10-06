// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use objc2::Message as _;
use objc2_metal::{MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _};

#[test]
fn distinct_classic_gray16_dense_tail_preserves_external_batch_order() {
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 512;
    const IMAGE_COUNT: usize = 17;

    if !should_run_metal_runtime() {
        return;
    }

    let base_samples = (0..WIDTH * HEIGHT)
        .map(|index| {
            let x = index % WIDTH;
            let y = index / WIDTH;
            u16::try_from(
                (x.wrapping_mul(193) ^ y.wrapping_mul(257) ^ (x * y).rotate_left(7)) & 0xffff,
            )
            .expect("masked fixture sample fits u16")
        })
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 5,
        ..EncodeOptions::default()
    };
    let mut expected = Vec::with_capacity(IMAGE_COUNT);
    let mut encoded = Vec::with_capacity(IMAGE_COUNT);
    for image_index in 0..IMAGE_COUNT {
        let mut samples = base_samples.clone();
        let marker_index = image_index * 15_419;
        samples[marker_index] = samples[marker_index]
            .wrapping_add(u16::try_from(image_index + 1).expect("image index fits u16") * 997);
        let pixels = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        encoded.push(Arc::<[u8]>::from(
            encode(&pixels, WIDTH, HEIGHT, 1, 16, false, &options)
                .expect("encode level-5 classic Gray16 tail fixture"),
        ));
        expected.push(pixels);
    }

    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(encoded.iter().cloned().map(EncodedImage::full).collect())
        .expect("prepare distinct classic Gray16 tail group");
    assert_eq!(prepared.errors(), []);
    assert_eq!(prepared.groups().len(), 1);
    let group = &prepared.groups()[0];
    assert_eq!(group.info().route, BatchCodecRoute::Classic);
    assert_eq!(group.images().len(), IMAGE_COUNT);
    assert!(group.source_indices().iter().copied().eq(0..IMAGE_COUNT));

    let image_bytes = WIDTH as usize * HEIGHT as usize * 2;
    let output_bytes = image_bytes * IMAGE_COUNT;
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        output_bytes + 8,
    )
    .expect("distinct classic Gray16 tail destination buffer");
    let layout = MetalImageLayout::new_batch(
        4,
        (WIDTH, HEIGHT),
        WIDTH as usize * 2,
        PixelFormat::Gray16,
        IMAGE_COUNT,
        image_bytes,
    )
    .expect("distinct classic Gray16 tail destination layout");
    // SAFETY: this fresh range remains exclusively retained by the submitted
    // codec work until its explicit completion wait.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("distinct classic Gray16 tail destination")
    };
    decoder
        .submit_prepared_group_into(group, destination)
        .expect("submit distinct classic Gray16 tail group")
        .wait()
        .expect("complete distinct classic Gray16 tail group");

    // SAFETY: codec completion released exclusive destination access.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, output_bytes)
            .expect("distinct classic Gray16 tail pixels")
    };
    for (image_index, (actual, expected)) in
        actual.chunks_exact(image_bytes).zip(&expected).enumerate()
    {
        assert_eq!(actual, expected, "image {image_index}");
    }
}

#[test]
fn repeated_and_partial_duplicate_ht_rgb_groups_preserve_external_batch_order() {
    if !should_run_metal_runtime() {
        return;
    }

    let rgb8 = [0, 17, 31].map(|offset| Arc::<[u8]>::from(fixture_ht_rgb_u8_sized(8, 8, offset)));
    let rgb16 =
        [0, 257, 509].map(|offset| Arc::<[u8]>::from(fixture_ht_rgb_u16_sized(8, 8, offset)));
    for (sources, format) in [(&rgb8, PixelFormat::Rgb8), (&rgb16, PixelFormat::Rgb16)] {
        for batch_layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
            for input_indices in [&[0_usize, 0, 0][..], &[0, 1, 0, 2, 1][..]] {
                let (actual, expected) =
                    decode_external_rgb_group(sources, input_indices, batch_layout, format);
                assert_eq!(actual, expected, "{format:?} {batch_layout:?}");
            }
        }
    }
}

/// Decodes `input_indices` of `sources` on the CPU and into an external Metal
/// destination that starts 4 bytes into its buffer. Returns the Metal bytes
/// and the CPU bytes.
fn decode_external_rgb_group(
    sources: &[Arc<[u8]>; 3],
    input_indices: &[usize],
    batch_layout: BatchLayout,
    format: PixelFormat,
) -> (Vec<u8>, Vec<u8>) {
    let options = BatchDecodeOptions {
        layout: batch_layout,
        ..BatchDecodeOptions::default()
    };
    let inputs = input_indices
        .iter()
        .map(|&index| EncodedImage::full(Arc::clone(&sources[index])))
        .collect::<Vec<_>>();
    let mut cpu = CpuBatchDecoder::new(options);
    let cpu_output = cpu
        .decode(inputs.clone())
        .expect("CPU repeated/partial-duplicate RGB oracle");
    let expected = match (format, cpu_output.groups()[0].samples()) {
        (PixelFormat::Rgb8, CpuBatchSamples::U8(samples)) => samples.clone(),
        (PixelFormat::Rgb16, CpuBatchSamples::U16(samples)) => samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect(),
        _ => panic!("six-bit RGB must use U8 batch storage and twelve-bit RGB U16"),
    };

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(inputs)
        .expect("prepare repeated/partial-duplicate RGB group");
    assert_eq!(prepared.errors(), []);
    assert_eq!(prepared.groups().len(), 1);
    let group = &prepared.groups()[0];
    assert_eq!(group.images().len(), input_indices.len());
    let (width, height) = group.info().dimensions;
    let row_bytes = width as usize * 3 * format.bytes_per_sample();
    let image_len = row_bytes * height as usize;
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        expected.len() + 8,
    )
    .expect("repeated/partial-duplicate RGB destination buffer");
    let destination_layout = MetalImageLayout::new_batch(
        4,
        (width, height),
        row_bytes,
        format,
        group.images().len(),
        image_len,
    )
    .expect("repeated/partial-duplicate RGB destination layout");
    // SAFETY: the fresh allocation stays exclusively retained until the
    // submitted group has completed.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), destination_layout)
            .expect("repeated/partial-duplicate RGB destination")
    };
    decoder
        .submit_prepared_group_into(group, destination)
        .expect("submit repeated/partial-duplicate RGB group")
        .wait()
        .expect("complete repeated/partial-duplicate RGB group");

    // SAFETY: completion released the exclusive destination owner.
    let actual =
        unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, expected.len()) }
            .expect("repeated/partial-duplicate RGB destination bytes");
    (actual, expected)
}

#[test]
fn shared_metal_batch_keeps_indexed_prepare_failures_and_decodes_other_groups() {
    if !should_run_metal_runtime() {
        return;
    }

    let valid = Arc::<[u8]>::from(fixture_ht_gray8());
    let malformed = Arc::<[u8]>::from(&b"not a codestream"[..]);
    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let result = decoder
        .decode_batch(vec![
            EncodedImage::full(malformed),
            EncodedImage::full(valid),
        ])
        .expect("one-shot shared Metal batch");

    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].index, 0);
    assert_eq!(result.groups().len(), 1);
    assert_eq!(result.groups()[0].source_indices(), &[1]);
}

#[test]
fn persistent_metal_batch_decoder_validates_external_destination_subranges() {
    if !should_run_metal_runtime() {
        return;
    }

    let decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let device = decoder.backend_session().device();
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(device, 32)
        .expect("destination buffer");
    let layout = j2k_metal_support::MetalImageLayout::new(8, (4, 2), 4, PixelFormat::Gray8)
        .expect("destination layout");
    // SAFETY: The test excludes every other CPU/GPU access to this fresh range.
    let destination = unsafe { MetalImageDestination::from_exclusive_buffer(buffer, layout) }
        .expect("validated external destination");

    decoder
        .validate_destination(&destination, (4, 2), PixelFormat::Gray8)
        .expect("matching decode destination");
    assert!(matches!(
        decoder.validate_destination(&destination, (8, 1), PixelFormat::Gray8),
        Err(Error::MetalSupport {
            source: j2k_metal_support::MetalSupportError::MetalImageLayout { .. },
            ..
        })
    ));
    assert!(matches!(
        decoder.validate_destination(&destination, (4, 2), PixelFormat::Gray16),
        Err(Error::MetalSupport {
            source: j2k_metal_support::MetalSupportError::MetalImageLayout { .. },
            ..
        })
    ));
}

#[test]
fn persistent_metal_batch_decoder_writes_distinct_ht_gray_group_into_external_subranges() {
    if !should_run_metal_runtime() {
        return;
    }

    let first_bytes = fixture_ht_gray8();
    let second_bytes = fixture_ht_gray8_reversed();
    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let device = decoder.backend_session().device();
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(device, 40)
        .expect("external batch destination");
    let layout =
        j2k_metal_support::MetalImageLayout::new_batch(4, (4, 4), 4, PixelFormat::Gray8, 2, 16)
            .expect("group layout");
    // SAFETY: No other CPU/GPU access occurs until the synchronous group
    // decode completes and the destination owner drops.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("group destination")
    };
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(Arc::from(first_bytes.as_slice())),
            EncodedImage::full(Arc::from(second_bytes.as_slice())),
        ])
        .expect("prepare external HT grayscale group");
    assert_eq!(prepared.errors(), []);
    assert_eq!(prepared.groups().len(), 1);

    decoder
        .decode_prepared_group_into(&prepared.groups()[0], &destination)
        .expect("direct external HT grayscale batch");
    drop(destination);

    // SAFETY: Decode completion is synchronous and all exclusive destination
    // owners have been dropped before this readback assertion.
    let bytes = unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 0, 40) }
        .expect("completed external batch bytes");
    let expected = [&first_bytes, &second_bytes]
        .into_iter()
        .map(|encoded| {
            let mut host = [0_u8; 16];
            J2kDecoder::new(encoded)
                .expect("host decoder")
                .decode_into(&mut host, 4, PixelFormat::Gray8)
                .expect("host decode");
            host
        })
        .collect::<Vec<_>>();

    assert_eq!(&bytes[4..20], expected[0]);
    assert_eq!(&bytes[20..36], expected[1]);
}

#[test]
fn shared_prepared_gray8_group_writes_one_destination_with_unaligned_item_offsets() {
    if !should_run_metal_runtime() {
        return;
    }

    let first_bytes = Arc::<[u8]>::from(fixture_ht_gray8_offset_sized(3, 3, 1));
    let second_bytes = Arc::<[u8]>::from(fixture_ht_gray8_offset_sized(3, 3, 101));
    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(first_bytes.clone()),
            EncodedImage::full(second_bytes.clone()),
        ])
        .expect("prepare odd Gray8 group");
    assert_eq!(prepared.errors(), []);
    assert_eq!(prepared.groups().len(), 1);

    let device = decoder.backend_session().device();
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(device, 26)
        .expect("group destination buffer");
    let layout =
        j2k_metal_support::MetalImageLayout::new_batch(4, (3, 3), 3, PixelFormat::Gray8, 2, 9)
            .expect("odd Gray8 group layout");
    // SAFETY: This fresh group allocation remains exclusively owned until the
    // synchronous direct final-store command has completed.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("group destination")
    };

    decoder
        .decode_prepared_group_into(&prepared.groups()[0], &destination)
        .expect("direct shared prepared group store");
    drop(destination);

    // SAFETY: The direct store completed and its exclusive destination owner
    // was dropped before host verification.
    let bytes = unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 0, 26) }
        .expect("completed group buffer");
    for (encoded, output) in [first_bytes, second_bytes]
        .iter()
        .zip([&bytes[4..13], &bytes[13..22]])
    {
        let mut expected = [0_u8; 9];
        J2kDecoder::new(encoded)
            .expect("CPU oracle decoder")
            .decode_into(&mut expected, 3, PixelFormat::Gray8)
            .expect("CPU oracle decode");
        assert_eq!(output, expected);
    }
}

#[test]
fn submitted_prepared_group_retains_destination_until_wait_and_decoder_reuses_after_drop() {
    if !should_run_metal_runtime() {
        return;
    }

    let first_bytes = Arc::<[u8]>::from(fixture_ht_gray8_offset_sized(3, 3, 1));
    let second_bytes = Arc::<[u8]>::from(fixture_ht_gray8_offset_sized(3, 3, 101));
    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(first_bytes.clone()),
            EncodedImage::full(second_bytes.clone()),
        ])
        .expect("prepare submitted Gray8 group");
    let layout = MetalImageLayout::new_batch(4, (3, 3), 3, PixelFormat::Gray8, 2, 9)
        .expect("submitted group layout");

    let dropped_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        26,
    )
    .expect("dropped pending destination buffer");
    // SAFETY: The fresh range is inaccessible until the pending owner is
    // dropped; its drop path must retire the committed command before release.
    let dropped_destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(dropped_buffer, layout)
            .expect("dropped pending destination")
    };
    let pending = decoder
        .submit_prepared_group_into(&prepared.groups()[0], dropped_destination)
        .expect("submit group before drop");
    drop(pending);

    let completed_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        26,
    )
    .expect("completed pending destination buffer");
    // SAFETY: The fresh range remains exclusively retained by the pending
    // submission through its explicit completion wait.
    let completed_destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(completed_buffer.clone(), layout)
            .expect("completed pending destination")
    };
    decoder
        .submit_prepared_group_into(&prepared.groups()[0], completed_destination)
        .expect("submit group after dropped pending")
        .wait()
        .expect("wait for submitted group");

    // SAFETY: `wait` completed GPU execution and released the exclusive
    // destination guard before this host verification.
    let bytes =
        unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&completed_buffer, 0, 26) }
            .expect("completed submitted group bytes");
    for (encoded, output) in [first_bytes, second_bytes]
        .iter()
        .zip([&bytes[4..13], &bytes[13..22]])
    {
        let mut expected = [0_u8; 9];
        J2kDecoder::new(encoded)
            .expect("CPU oracle decoder")
            .decode_into(&mut expected, 3, PixelFormat::Gray8)
            .expect("CPU oracle decode");
        assert_eq!(output, expected);
    }
}

#[test]
fn submitted_prepared_group_orders_a_same_device_consumer_queue() {
    if !should_run_metal_runtime() {
        return;
    }

    let first_bytes = Arc::<[u8]>::from(fixture_ht_gray8_offset_sized(3, 3, 7));
    let second_bytes = Arc::<[u8]>::from(fixture_ht_gray8_offset_sized(3, 3, 71));
    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(first_bytes.clone()),
            EncodedImage::full(second_bytes.clone()),
        ])
        .expect("prepare consumer-ordered Gray8 group");
    let device = decoder.backend_session().device().retain();
    let destination_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(&device, 26)
        .expect("consumer-ordered destination");
    let consumer_buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(&device, 18)
        .expect("consumer copy destination");
    let layout = MetalImageLayout::new_batch(4, (3, 3), 3, PixelFormat::Gray8, 2, 9)
        .expect("consumer-ordered group layout");
    // SAFETY: The consumer access below is submitted only after the codec
    // dependency is registered on that queue.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(destination_buffer.clone(), layout)
            .expect("consumer-ordered destination guard")
    };
    let mut pending = decoder
        .submit_prepared_group_into(&prepared.groups()[0], destination)
        .expect("submit producer group");
    let consumer_queue =
        j2k_metal_support::checked_command_queue(&device).expect("consumer command queue");
    pending
        .enqueue_consumer_wait(&consumer_queue)
        .expect("register producer dependency on consumer queue");
    let consumer_command = j2k_metal_support::checked_command_buffer(&consumer_queue)
        .expect("consumer command buffer");
    let blit = j2k_metal_support::checked_blit_command_encoder(&consumer_command)
        .expect("consumer blit encoder");
    // SAFETY: Both ranges are within their checked shared allocations. The
    // command buffer retains both buffers through completion, and the shared
    // event wait above orders this read after the producer's writes.
    unsafe {
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            &destination_buffer,
            4,
            &consumer_buffer,
            0,
            18,
        );
    };
    blit.endEncoding();
    consumer_command.commit();

    pending.wait().expect("producer group completion");
    j2k_metal_support::wait_for_completion(&consumer_command)
        .expect("consumer queue completion after producer signal");
    // SAFETY: The producer and the event-ordered consumer copy have completed.
    let copied =
        unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&consumer_buffer, 0, 18) }
            .expect("consumer-ordered copied bytes");
    for (encoded, output) in [first_bytes, second_bytes]
        .iter()
        .zip([&copied[..9], &copied[9..]])
    {
        let mut expected = [0_u8; 9];
        J2kDecoder::new(encoded)
            .expect("CPU oracle decoder")
            .decode_into(&mut expected, 3, PixelFormat::Gray8)
            .expect("CPU oracle decode");
        assert_eq!(output, expected);
    }
}

#[test]
fn submitted_prepared_signed_gray12_group_stores_native_i16_samples() {
    if !should_run_metal_runtime() {
        return;
    }

    let (first_encoded, first_expected) = fixture_ht_signed_gray12(0);
    let (second_encoded, second_expected) = fixture_ht_signed_gray12(31);
    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(Arc::<[u8]>::from(first_encoded)),
            EncodedImage::full(Arc::<[u8]>::from(second_encoded)),
        ])
        .expect("prepare signed Gray12 group");
    assert_eq!(prepared.errors(), []);
    assert_eq!(prepared.groups().len(), 1);
    assert_eq!(
        prepared.groups()[0].info().sample_type,
        NativeSampleType::I16
    );

    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        72,
    )
    .expect("signed group destination buffer");
    let layout = MetalImageLayout::new_batch(4, (4, 4), 8, PixelFormat::GrayI16, 2, 32)
        .expect("signed group layout");
    // SAFETY: The pending submission exclusively retains this fresh range
    // until its explicit completion wait.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("signed group destination")
    };
    decoder
        .submit_prepared_group_into(&prepared.groups()[0], destination)
        .expect("submit signed Gray12 group")
        .wait()
        .expect("complete signed Gray12 group");

    // SAFETY: Codec completion released exclusive destination access.
    let bytes = unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, 64) }
        .expect("signed group bytes");
    let actual = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>();
    assert_eq!(actual, [first_expected, second_expected].concat());
}
