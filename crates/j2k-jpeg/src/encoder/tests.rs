// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::borrow::Cow;
use alloc::vec::Vec;

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use crate::adapter::{baseline_encode_tables, jpeg_baseline_entropy_capacity_bytes};
use crate::baseline_entropy::magnitude;

use super::entropy::{
    encode_entropy_restart_segments, encode_entropy_serial, parallel_entropy_chunk_count,
    MAX_PARALLEL_ENTROPY_CHUNKS,
};
use super::planning::component_plane_capacity_bytes;
use super::sample_planes::component_planes;
use super::transform::cosine_table;
use super::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSamples,
    JpegSubsampling,
};

#[test]
fn encoder_rejects_geometry_above_host_cap_before_length_check() {
    let error = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &[],
            width: u32::from(u16::MAX),
            height: u32::from(u16::MAX),
        },
        JpegEncodeOptions {
            subsampling: JpegSubsampling::Ybr444,
            backend: JpegBackend::Cpu,
            ..JpegEncodeOptions::default()
        },
    )
    .expect_err("maximum baseline RGB geometry must exceed the host cap");

    assert!(matches!(
        error,
        JpegEncodeError::MemoryCapExceeded { requested, cap }
            if requested > cap && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}

#[test]
fn restart_one_rejects_cap_valid_geometry_before_sample_or_entropy_allocation() {
    let width = 8_225;
    let height = 65_273;
    assert!(
        usize::try_from(width).unwrap() * usize::try_from(height).unwrap()
            <= DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );

    let error = encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &[],
            width,
            height,
        },
        JpegEncodeOptions {
            quality: 100,
            subsampling: JpegSubsampling::Gray,
            restart_interval: Some(1),
            backend: JpegBackend::Cpu,
        },
    )
    .expect_err("conservative encoded output exceeds the shared host cap");

    assert!(matches!(
        error,
        JpegEncodeError::MemoryCapExceeded { requested, cap }
            if requested > cap && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}

#[test]
fn grayscale_rejects_entropy_and_frame_live_peak_before_sample_allocation() {
    let error = encode_jpeg_baseline(
        JpegSamples::Gray8 {
            data: &[],
            width: 4_096,
            height: 8_192,
        },
        JpegEncodeOptions {
            subsampling: JpegSubsampling::Gray,
            backend: JpegBackend::Cpu,
            ..JpegEncodeOptions::default()
        },
    )
    .expect_err("entropy plus frame capacity exceeds the shared live cap");

    assert!(matches!(
        error,
        JpegEncodeError::MemoryCapExceeded { requested, cap }
            if requested > cap && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}

#[test]
fn grayscale_component_plane_borrows_the_input() {
    let samples = [3u8, 7, 11, 19];
    let planes = component_planes(
        JpegSamples::Gray8 {
            data: &samples,
            width: 2,
            height: 2,
        },
        JpegSubsampling::Gray,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )
    .expect("grayscale planes");

    assert!(
        matches!(planes.as_slice(), [Cow::Borrowed(data)] if core::ptr::eq(*data, samples.as_slice()))
    );
}

#[test]
fn magnitude_represents_the_full_i32_domain() {
    assert_eq!(magnitude(0), (0, 0));
    assert_eq!(magnitude(5), (3, 5));
    assert_eq!(magnitude(-5), (3, 2));
    assert_eq!(magnitude(i32::MAX), (31, i32::MAX.unsigned_abs()));
    assert_eq!(magnitude(i32::MIN), (32, i32::MAX.unsigned_abs()));
}

fn patterned_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 3) & 0xFF) as u8);
            pixels.push(((x * 5 + y * 11 + 40) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 7 + 90) & 0xFF) as u8);
        }
    }
    pixels
}

fn assert_restart_entropy_matches_serial(restart_interval: u16) {
    let width = 160;
    let height = 80;
    let tables = baseline_encode_tables(JpegEncodeOptions {
        quality: 90,
        subsampling: JpegSubsampling::Ybr422,
        restart_interval: Some(restart_interval),
        backend: JpegBackend::Cpu,
    })
    .unwrap();
    let sampling = tables.sampling;
    let cosine = cosine_table();
    let pixels = patterned_rgb(width, height);
    let planes = component_planes(
        JpegSamples::Rgb8 {
            data: &pixels,
            width,
            height,
        },
        JpegSubsampling::Ybr422,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    )
    .unwrap();
    let plane_live_bytes = component_plane_capacity_bytes(planes.capacity(), &planes).unwrap();
    let entropy_capacity =
        jpeg_baseline_entropy_capacity_bytes(width, height, sampling, Some(restart_interval))
            .unwrap();

    let serial = encode_entropy_serial(
        &planes,
        width,
        height,
        sampling,
        &tables.q_luma,
        &tables.q_chroma,
        [&tables.huff_dc_luma, &tables.huff_dc_chroma],
        [&tables.huff_ac_luma, &tables.huff_ac_chroma],
        &cosine,
        Some(restart_interval),
        entropy_capacity,
        plane_live_bytes,
    )
    .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let segmented = pool
        .install(|| {
            encode_entropy_restart_segments(
                &planes,
                width,
                height,
                sampling,
                &tables.q_luma,
                &tables.q_chroma,
                [&tables.huff_dc_luma, &tables.huff_dc_chroma],
                [&tables.huff_ac_luma, &tables.huff_ac_chroma],
                &cosine,
                restart_interval,
                entropy_capacity,
                plane_live_bytes,
            )
        })
        .unwrap();

    assert_eq!(segmented, serial);
    assert!(segmented.windows(2).any(|window| window == [0xFF, 0xD0]));
}

#[test]
fn restart_entropy_segments_match_serial_entropy() {
    assert_restart_entropy_matches_serial(64);
}

#[test]
fn restart_one_entropy_chunks_match_serial_entropy() {
    assert_restart_entropy_matches_serial(1);
}

#[test]
fn restart_segment_fanout_is_bounded_by_chunk_policy() {
    let chunk_count = parallel_entropy_chunk_count(u32::MAX).unwrap();
    assert_eq!(chunk_count, MAX_PARALLEL_ENTROPY_CHUNKS);
    assert_eq!(parallel_entropy_chunk_count(1).unwrap(), 1);
}

#[test]
fn restart_segment_fanout_keeps_work_stealing_granularity() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    assert_eq!(
        pool.install(|| parallel_entropy_chunk_count(16)).unwrap(),
        16
    );
}
