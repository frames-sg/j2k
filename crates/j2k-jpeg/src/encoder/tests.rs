// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::borrow::Cow;
use alloc::vec::Vec;

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use crate::adapter::{baseline_encode_tables, jpeg_baseline_entropy_capacity_bytes};
use crate::baseline_entropy::magnitude;
use crate::encoded_output::JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY;

use super::entropy::{
    encode_entropy_restart_segments, encode_entropy_serial, parallel_entropy_chunk_count,
    MAX_PARALLEL_ENTROPY_CHUNKS,
};
use super::planning::{checked_cpu_encode_capacity_plan, component_plane_capacity_bytes};
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
fn bounded_grayscale_plan_reaches_sample_length_validation_before_allocation() {
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
    .expect_err("empty samples still fail after bounded capacity planning");

    assert!(matches!(
        error,
        JpegEncodeError::SampleLength {
            expected: 33_554_432,
            actual: 0
        }
    ));
}

#[test]
fn serial_capacity_uses_the_live_allowance_only_when_theoretical_capacity_needs_it() {
    let rgb_444 = baseline_encode_tables(JpegEncodeOptions {
        quality: 50,
        subsampling: JpegSubsampling::Ybr444,
        restart_interval: None,
        backend: JpegBackend::Cpu,
    })
    .unwrap();
    let rgb_4096_bytes = 4_096usize * 4_096 * 3;
    let rgb_4096 = JpegSamples::Rgb8 {
        data: &[],
        width: 4_096,
        height: 4_096,
    };
    let rgb_4096_plan =
        checked_cpu_encode_capacity_plan(rgb_4096, rgb_444.sampling, rgb_4096_bytes, None)
            .expect("4096 RGB 4:4:4 fits through the bounded serial allowance");
    let rgb_metadata = 3 * core::mem::size_of::<Cow<'static, [u8]>>();
    let rgb_4096_allowance = (DEFAULT_MAX_HOST_ALLOCATION_BYTES
        - rgb_4096_bytes
        - rgb_metadata
        - JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY)
        / 2;
    assert_eq!(rgb_4096_plan.entropy_capacity, rgb_4096_allowance);
    assert_eq!(rgb_4096_plan.entropy_workspace_bytes, rgb_4096_allowance);
    assert_eq!(
        rgb_4096_plan.plane_capacity_limit,
        rgb_4096_bytes + rgb_metadata
    );

    let gray = baseline_encode_tables(JpegEncodeOptions {
        quality: 50,
        subsampling: JpegSubsampling::Gray,
        restart_interval: None,
        backend: JpegBackend::Cpu,
    })
    .unwrap();
    let gray_samples = JpegSamples::Gray8 {
        data: &[],
        width: 4_096,
        height: 8_192,
    };
    let gray_plan =
        checked_cpu_encode_capacity_plan(gray_samples, gray.sampling, 4_096 * 8_192, None)
            .expect("large borrowed grayscale input fits through the serial allowance");
    let gray_metadata = core::mem::size_of::<Cow<'static, [u8]>>();
    let gray_allowance =
        (DEFAULT_MAX_HOST_ALLOCATION_BYTES - gray_metadata - JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY)
            / 2;
    assert_eq!(gray_plan.entropy_capacity, gray_allowance);
    assert_eq!(gray_plan.plane_capacity_limit, gray_metadata);

    let rgb_2048_bytes = 2_048usize * 2_048 * 3;
    let theoretical_2048 =
        jpeg_baseline_entropy_capacity_bytes(2_048, 2_048, rgb_444.sampling, None).unwrap();
    let rgb_2048_plan = checked_cpu_encode_capacity_plan(
        JpegSamples::Rgb8 {
            data: &[],
            width: 2_048,
            height: 2_048,
        },
        rgb_444.sampling,
        rgb_2048_bytes,
        None,
    )
    .unwrap();
    assert_eq!(rgb_2048_plan.entropy_capacity, theoretical_2048);

    let restart_options = JpegEncodeOptions {
        quality: 50,
        subsampling: JpegSubsampling::Ybr422,
        restart_interval: Some(64),
        backend: JpegBackend::Cpu,
    };
    let restart_tables = baseline_encode_tables(restart_options).unwrap();
    let theoretical_restart = jpeg_baseline_entropy_capacity_bytes(
        512,
        512,
        restart_tables.sampling,
        restart_options.restart_interval,
    )
    .unwrap();
    let restart_plan = checked_cpu_encode_capacity_plan(
        JpegSamples::Rgb8 {
            data: &[],
            width: 512,
            height: 512,
        },
        restart_tables.sampling,
        512 * 512 * 3,
        restart_options.restart_interval,
    )
    .unwrap();
    assert_eq!(restart_plan.entropy_capacity, theoretical_restart);
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
        baseline_encode_tables(JpegEncodeOptions {
            quality: 75,
            subsampling: JpegSubsampling::Gray,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        })
        .unwrap()
        .sampling,
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

fn assert_restart_entropy_matches_serial(
    restart_interval: u16,
    subsampling: JpegSubsampling,
    width: u32,
    height: u32,
) {
    let tables = baseline_encode_tables(JpegEncodeOptions {
        quality: 90,
        subsampling,
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
        subsampling,
        sampling,
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
        cosine,
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
                cosine,
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
    for subsampling in [
        JpegSubsampling::Ybr422,
        JpegSubsampling::Ybr444,
        JpegSubsampling::Ybr420,
    ] {
        let height = if subsampling == JpegSubsampling::Ybr420 {
            160
        } else {
            80
        };
        assert_restart_entropy_matches_serial(64, subsampling, 160, height);
    }
}

#[test]
fn compact_block_plane_restart_segments_match_serial_entropy() {
    for subsampling in [
        JpegSubsampling::Ybr422,
        JpegSubsampling::Ybr444,
        JpegSubsampling::Ybr420,
    ] {
        assert_restart_entropy_matches_serial(64, subsampling, 161, 81);
    }
}

#[test]
fn restart_one_entropy_chunks_match_serial_entropy() {
    for subsampling in [
        JpegSubsampling::Ybr422,
        JpegSubsampling::Ybr444,
        JpegSubsampling::Ybr420,
    ] {
        let height = if subsampling == JpegSubsampling::Ybr420 {
            160
        } else {
            80
        };
        assert_restart_entropy_matches_serial(1, subsampling, 160, height);
    }
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
