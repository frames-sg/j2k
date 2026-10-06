// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic RGB groups submitted together decode their Tier-1 code blocks in
//! one shared dispatch without changing any output byte.

use std::sync::Arc;

use j2k::{BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, EncodedImage};
use j2k_core::PixelFormat;
use j2k_metal_support::{MetalImageDestination, MetalImageLayout};
use j2k_native::{encode, EncodeOptions};

use super::{MetalBatchDecoder, PreparedBatch};

/// Textured RGB so code blocks carry many coding passes. 32x32 code blocks
/// give 174 blocks per 256x192 image, so seven images reach the dense kernel.
fn classic_rgb(seed: u32, (width, height): (u32, u32)) -> Arc<[u8]> {
    let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let smooth = x * (3 + channel) + y * (5 + seed);
                pixels.push(((smooth + (state & 31)) & 255) as u8);
            }
        }
    }
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 5,
        code_block_width_exp: 3,
        code_block_height_exp: 3,
        use_mct: true,
        ..EncodeOptions::default()
    };
    Arc::from(encode(&pixels, width, height, 3, 8, false, &options).expect("encode classic RGB"))
}

/// Landscape and portrait images interleaved, so each prepared group holds
/// non-contiguous sources.
fn two_group_inputs() -> Vec<EncodedImage> {
    (0..7)
        .map(|index| {
            let dimensions = if index % 2 == 0 {
                (256, 192)
            } else {
                (192, 256)
            };
            EncodedImage::full(classic_rgb(index, dimensions))
        })
        .collect()
}

fn options() -> BatchDecodeOptions {
    BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    }
}

fn cpu_images_by_source(inputs: &[EncodedImage]) -> Vec<Vec<u8>> {
    let expected = CpuBatchDecoder::new(options())
        .decode(inputs.to_vec())
        .expect("CPU oracle");
    assert!(expected.errors().is_empty(), "{:?}", expected.errors());
    let mut images = vec![Vec::new(); inputs.len()];
    for group in expected.groups() {
        let CpuBatchSamples::U8(samples) = group.samples() else {
            panic!("RGB8 oracle");
        };
        let image_bytes = samples.len() / group.source_indices().len();
        for (&source, image) in group
            .source_indices()
            .iter()
            .zip(samples.chunks_exact(image_bytes))
        {
            images[source] = image.to_vec();
        }
    }
    images
}

/// Decodes into caller-owned buffers, either in one multi-group call or one
/// group at a time, and returns each image's bytes by source index.
fn decode_into(
    decoder: &mut MetalBatchDecoder,
    prepared: &PreparedBatch,
    together: bool,
) -> Vec<Vec<u8>> {
    let mut outputs = Vec::new();
    let mut requests = Vec::new();
    for group in prepared.groups() {
        let (width, height) = group.info().dimensions;
        let row_bytes = width as usize * 3;
        let image_bytes = row_bytes * height as usize;
        let count = group.images().len();
        let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
            decoder.backend_session().device(),
            image_bytes * count,
        )
        .expect("destination allocation");
        let layout = MetalImageLayout::new_batch(
            0,
            (width, height),
            row_bytes,
            PixelFormat::Rgb8,
            count,
            image_bytes,
        )
        .expect("destination layout");
        // SAFETY: the fresh allocation has one writer and is read only after
        // its submission completes.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
                .expect("destination")
        };
        outputs.push((buffer, image_bytes, group.source_indices().to_vec()));
        requests.push((group, destination));
    }
    if together {
        let submitted = decoder
            .submit_prepared_groups_into(requests)
            .expect("multi-group submission");
        for pending in submitted {
            pending
                .expect("group submission")
                .wait()
                .expect("group completion");
        }
    } else {
        for (group, destination) in requests {
            decoder
                .submit_prepared_group_into(group, destination)
                .expect("group submission")
                .wait()
                .expect("group completion");
        }
    }
    let mut images = vec![Vec::new(); prepared.groups().iter().map(|g| g.images().len()).sum()];
    for (buffer, image_bytes, sources) in outputs {
        // SAFETY: every submission completed and released its destination.
        let bytes = unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(
                &buffer,
                0,
                image_bytes * sources.len(),
            )
            .expect("completed destination bytes")
        };
        for (source, image) in sources.into_iter().zip(bytes.chunks_exact(image_bytes)) {
            images[source] = image.to_vec();
        }
    }
    images
}

#[test]
fn classic_rgb_groups_share_one_tier1_dispatch_bit_exactly() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let inputs = two_group_inputs();
    let expected = cpu_images_by_source(&inputs);
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options()).expect("Metal decoder");
    let prepared = decoder.prepare(inputs).expect("prepare two groups");
    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
    assert_eq!(prepared.groups().len(), 2);

    crate::engine::reset_shared_classic_tier1_passes_for_test();
    let resident = decoder.decode_prepared(&prepared).expect("resident decode");
    assert_eq!(crate::engine::shared_classic_tier1_passes_for_test(), 1);
    assert!(
        resident.group_errors().is_empty(),
        "{:?}",
        resident.group_errors()
    );
    assert_eq!(resident.groups().len(), 2);
    for group in resident.groups() {
        assert!(group.dispatch_report().classic_tier1 > 0);
        for (&source, surface) in group.source_indices().iter().zip(group.surfaces()) {
            assert_eq!(
                surface.as_bytes().expect("resident bytes").as_ref(),
                expected[source],
                "resident source {source}"
            );
        }
    }

    crate::engine::reset_shared_classic_tier1_passes_for_test();
    let together = decode_into(&mut decoder, &prepared, true);
    assert_eq!(crate::engine::shared_classic_tier1_passes_for_test(), 1);
    let separate = decode_into(&mut decoder, &prepared, false);
    assert_eq!(crate::engine::shared_classic_tier1_passes_for_test(), 1);
    for (source, expected) in expected.iter().enumerate() {
        assert_eq!(
            &together[source], expected,
            "shared dispatch source {source}"
        );
        assert_eq!(
            &separate[source], expected,
            "per-group dispatch source {source}"
        );
    }
}
