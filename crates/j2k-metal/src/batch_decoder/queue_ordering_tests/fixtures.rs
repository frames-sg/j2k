// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{EncodedImage, PreparedBatch};
use j2k_core::PixelFormat;

use super::super::{MetalBatchDecoder, MetalImageDestination, MetalImageLayout};

pub(super) fn prepared_gray_group(
    decoder: &MetalBatchDecoder,
) -> Result<PreparedBatch, Box<dyn std::error::Error>> {
    let bytes = Arc::<[u8]>::from(j2k_test_support::htj2k_gray8_fixture(4, 4));
    Ok(decoder.prepare(vec![EncodedImage::full(bytes)])?)
}

pub(super) fn gray8_cpu_oracle(encoded: &[u8]) -> [u8; 16] {
    let mut expected = [0_u8; 16];
    j2k::J2kDecoder::new(encoded)
        .expect("CPU oracle decoder")
        .decode_into(&mut expected, 4, PixelFormat::Gray8)
        .expect("CPU oracle decode");
    expected
}

pub(super) fn distinct_gray8_fixture(seed: u8) -> Arc<[u8]> {
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

pub(super) fn gray8_destination(
    device: &metal::DeviceRef,
) -> Result<MetalImageDestination, Box<dyn std::error::Error>> {
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(device, 16)?;
    let layout = MetalImageLayout::new_batch(0, (4, 4), 4, PixelFormat::Gray8, 1, 16)?;
    // SAFETY: The fresh allocation has one exclusive owner and the returned
    // submission retains that owner until completion or drop.
    Ok(unsafe { MetalImageDestination::from_exclusive_buffer(buffer, layout)? })
}

pub(super) fn wrong_size_gray8_destination(
    device: &metal::DeviceRef,
) -> Result<MetalImageDestination, Box<dyn std::error::Error>> {
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(device, 4)?;
    let layout = MetalImageLayout::new_batch(0, (2, 2), 2, PixelFormat::Gray8, 1, 4)?;
    // SAFETY: The fresh allocation has one exclusive owner. Destination
    // preflight rejects its dimensions before any codec work can retain it.
    Ok(unsafe { MetalImageDestination::from_exclusive_buffer(buffer, layout)? })
}
