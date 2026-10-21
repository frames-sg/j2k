// SPDX-License-Identifier: MIT OR Apache-2.0

use super::preparation::prepare_color_cuda_resident_batch;
use j2k_core::PixelFormat;

pub(super) fn four_level_rgb8_fixture(seed: u32) -> (Vec<u8>, Vec<u8>) {
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 512;
    let mut pixels = Vec::with_capacity(WIDTH as usize * HEIGHT as usize * 3);
    for index in 0..WIDTH * HEIGHT {
        pixels.push(((index * seed + index / 3) & 0xff) as u8);
        pixels.push(((index * (seed + 11) + 7) & 0xff) as u8);
        pixels.push(((index * (seed + 23) + 19) & 0xff) as u8);
    }
    let options = j2k_native::EncodeOptions {
        reversible: true,
        use_ht_block_coding: true,
        num_decomposition_levels: 4,
        ..j2k_native::EncodeOptions::default()
    };
    let codestream = j2k_native::encode(&pixels, WIDTH, HEIGHT, 3, 8, false, &options)
        .expect("encode four-level RGB8 fixture");
    (codestream, pixels)
}

pub(super) fn invalidate_first_cleanup_scup(encoded: &mut [u8]) -> (usize, usize) {
    let prepared = prepare_color_cuda_resident_batch(&[encoded], PixelFormat::Rgb8)
        .expect("prepare cleanup corruption fixture");
    let payload_parts = prepared
        .payload_parts_with_live_bytes(0)
        .expect("borrowed cleanup payload");
    assert_eq!(payload_parts[0].as_ptr(), encoded.as_ptr());
    let mut local_job_index = 0usize;
    let mut selected = None;
    for component in &prepared.colors[0].components {
        for block in component.code_blocks() {
            if selected.is_none() && block.cleanup_length >= 2 {
                let start = usize::try_from(block.payload_offset).expect("payload offset");
                let cleanup_len = usize::try_from(block.cleanup_length).expect("cleanup length");
                selected = Some((local_job_index, start + cleanup_len));
            }
            local_job_index += 1;
        }
    }
    let total_jobs = local_job_index;
    let (corrupted_job, cleanup_end) = selected.expect("nonempty HT cleanup job");
    assert!(cleanup_end <= encoded.len());
    drop(payload_parts);
    drop(prepared);
    encoded[cleanup_end - 2] = 0;
    encoded[cleanup_end - 1] = 0;
    (corrupted_job, total_jobs)
}
