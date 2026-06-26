// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::Storage;
use std::sync::Arc;

const BASELINE_420: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_420_16x16.jpg");
const BASELINE_420_RESTART: &[u8] =
    include_bytes!("../../fixtures/jpeg/baseline_420_restart_32x16.jpg");
const BASELINE_422: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_422_16x8.jpg");
const BASELINE_444: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_444_8x8.jpg");

#[test]
fn mcu_range_for_rect_covers_only_touched_rows_and_columns() {
    let roi = j2k_jpeg::Rect {
        x: 19,
        y: 35,
        w: 11,
        h: 34,
    };

    assert_eq!(mcu_range_for_rect(roi, 8, 6, 16, 16), (17, 34));
}

#[test]
fn restart_work_for_mcu_range_slices_to_overlapping_restart_segments() {
    let restart_offsets = [0, 10, 20, 30, 40, 50];

    let (restart_start_mcu, offsets) = restart_work_for_mcu_range(&restart_offsets, 4, 24, 9, 15);

    assert_eq!(restart_start_mcu, 8);
    assert_eq!(offsets, &[20, 30]);
}

#[test]
fn runtime_initialization_error_classifies_device_unavailable() {
    assert!(matches!(
        runtime_initialization_error(&MetalSupportError::MetalUnavailable),
        Error::MetalUnavailable
    ));
    assert!(matches!(
        runtime_initialization_error(&MetalSupportError::ShaderLibrary {
            message: "failed to compile Metal library".to_string()
        }),
        Error::MetalRuntime { .. }
    ));
}

#[test]
fn fast420_params_rejects_output_stride_overflow_without_panic() {
    let packet = minimal_fast420_packet((u32::MAX, 1));

    let Err(err) = fast_subsampled_params(&packet, PixelFormat::Rgba8) else {
        panic!("expected stride overflow");
    };

    assert!(matches!(err, Error::MetalKernel { .. }));
    assert!(
        err.to_string().contains("fast420 output stride"),
        "unexpected error: {err}"
    );
}

#[test]
fn fast422_region_params_rejects_output_stride_overflow_without_panic() {
    let packet = minimal_fast422_packet((16, 8));
    let source_window = j2k_jpeg::Rect {
        x: 0,
        y: 0,
        w: u32::MAX,
        h: 1,
    };

    let Err(err) = fast_subsampled_region_params(&packet, PixelFormat::Rgba8, source_window) else {
        panic!("expected stride overflow");
    };

    assert!(matches!(err, Error::MetalKernel { .. }));
    assert!(
        err.to_string().contains("fast422 region output stride"),
        "unexpected error: {err}"
    );
}

#[test]
fn fast444_params_accepts_minimal_packet() {
    let packet = minimal_fast444_packet((8, 8));

    let params = fast444_params(&packet).expect("fast444 params");

    assert_eq!(params.width, 8);
    assert_eq!(params.height, 8);
    assert_eq!(params.restart_offset_count, 1);
    assert_eq!(params.entropy_len, 0);
}

#[test]
fn viewport_plane_cache_is_runtime_local() {
    let runtime_a = MetalRuntime::new().expect("Metal runtime");
    let runtime_b = MetalRuntime::new_with_device(runtime_a.device.clone()).expect("Metal runtime");

    let stage_a = cached_plane_stage(&runtime_a, JpegColorSpace::YCbCr, (8, 8)).expect("stage");
    let stage_b = cached_plane_stage(&runtime_b, JpegColorSpace::YCbCr, (8, 8)).expect("stage");
    drop((stage_a, stage_b));

    assert_ne!(
        runtime_a
            .viewport_plane_cache_id_for_test()
            .expect("cache")
            .expect("cache entry"),
        runtime_b
            .viewport_plane_cache_id_for_test()
            .expect("cache")
            .expect("cache entry")
    );
}

#[test]
fn shader_decode_block_clears_coefficients_with_vector_stores() {
    assert!(
        SHADER_SOURCE.contains("thread short4 *coeff_chunks"),
        "decode_block should clear coeffs with packed short4 stores"
    );
    assert!(
        SHADER_SOURCE.contains("coeff_chunks[i] = short4(0);"),
        "decode_block should zero each packed coefficient chunk"
    );
}

#[test]
fn shader_source_keeps_entropy_fast_paths() {
    assert!(SHADER_SOURCE.contains("inline bool refill_four_bytes("));
    assert!(SHADER_SOURCE.contains("return refill_four_bytes(br, bytes, len) || refill_one_byte"));
    assert!(SHADER_SOURCE.contains("ensure_bits_padded(br, bytes, len, 9)"));
    assert!(SHADER_SOURCE.contains("table.fast_len[fast_index]"));
    assert!(SHADER_SOURCE.contains("inline bool decode_block_skip("));
    assert!(SHADER_SOURCE.contains("skip_receive_extend(br, bytes, len, ssss, status)"));
    assert!(SHADER_SOURCE.contains("inline bool configure_batch_entropy_thread("));
}

#[test]
fn shader_kernels_use_incremental_mx_my() {
    assert!(
        SHADER_SOURCE.contains("inline void init_mcu_cursor("),
        "fast decode kernels should seed mx/my via init_mcu_cursor instead of dividing per MCU"
    );
    assert!(
        SHADER_SOURCE.contains("inline void advance_mcu_cursor("),
        "fast decode kernels should carry mx/my via advance_mcu_cursor instead of dividing per MCU"
    );
    assert!(
        !SHADER_SOURCE.contains("mcu_index / params.mcus_per_row"),
        "no fast kernel should still divide mcu_index by mcus_per_row inside the MCU loop"
    );
    assert!(
        !SHADER_SOURCE.contains("mcu_index % params.mcus_per_row"),
        "no fast kernel should still modulo mcu_index by mcus_per_row inside the MCU loop"
    );
}

#[test]
fn fast420_batch_timing_env_requires_explicit_one() {
    assert!(fast420_batch_timing_value_enabled(Some(
        std::ffi::OsStr::new("1")
    )));
    assert!(!fast420_batch_timing_value_enabled(Some(
        std::ffi::OsStr::new("true")
    )));
    assert!(!fast420_batch_timing_value_enabled(None));
}

#[test]
fn one_dimensional_dispatch_width_tracks_work_without_full_threadgroup_waste() {
    assert_eq!(choose_1d_threadgroup_width(32, 1024, 1), 32);
    assert_eq!(choose_1d_threadgroup_width(32, 1024, 33), 64);
    assert_eq!(choose_1d_threadgroup_width(32, 1024, 256), 256);
    assert_eq!(choose_1d_threadgroup_width(32, 1024, 257), 256);
}

#[test]
fn auto_batched_packets_skip_distinct_region_scaled_requests() {
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420_RESTART).expect("packet");
    let roi = Rect {
        x: 0,
        y: 0,
        w: 16,
        h: 16,
    };
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::<[u8]>::from(BASELINE_420_RESTART),
            PixelFormat::Rgb8,
            BackendRequest::Auto,
            batch::BatchOp::RegionScaled {
                roi,
                scale: j2k_core::Downscale::Quarter,
            },
            None,
            None,
            Some(packet.clone()),
        ),
        batch::QueuedRequest::new(
            Arc::<[u8]>::from(BASELINE_420_RESTART),
            PixelFormat::Rgb8,
            BackendRequest::Auto,
            batch::BatchOp::RegionScaled {
                roi,
                scale: j2k_core::Downscale::Quarter,
            },
            None,
            None,
            Some(packet),
        ),
    ];

    assert!(batched_fast_packets(&requests)
        .expect("packet lookup")
        .is_none());
}

#[test]
fn auto_batched_packets_keep_large_repeated_region_scaled_requests_off_metal() {
    let input = Arc::<[u8]>::from(BASELINE_420);
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
    let roi = Rect {
        x: 0,
        y: 0,
        w: 16,
        h: 16,
    };
    let requests = (0..=REGION_SCALED_BATCH_CHUNK)
        .map(|_| {
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Auto,
                batch::BatchOp::RegionScaled {
                    roi,
                    scale: j2k_core::Downscale::Quarter,
                },
                None,
                None,
                Some(packet.clone()),
            )
        })
        .collect::<Vec<_>>();

    assert!(batched_fast_packets(&requests)
        .expect("packet lookup")
        .is_none());
}

#[test]
fn auto_batched_packets_require_wsi_batch_threshold() {
    let input = Arc::<[u8]>::from(BASELINE_420_RESTART);
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420_RESTART).expect("packet");
    let requests = (0..7)
        .map(|_| {
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Auto,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            )
        })
        .collect::<Vec<_>>();

    assert!(batched_fast_packets(&requests)
        .expect("packet lookup")
        .is_none());
}

#[test]
fn auto_batched_packets_accept_restart_wsi_batch_at_threshold() {
    let input = Arc::<[u8]>::from(BASELINE_420_RESTART);
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420_RESTART).expect("packet");
    let requests = (0..8)
        .map(|_| {
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Auto,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            )
        })
        .collect::<Vec<_>>();

    assert!(batched_fast_packets(&requests)
        .expect("packet lookup")
        .is_some());
}

#[test]
fn auto_batched_packets_accept_large_nonrestart_wsi_batch_at_threshold() {
    let input = Arc::<[u8]>::from(generated_rgb_jpeg(512));
    let fast444_packet = j2k_jpeg::adapter::build_fast444_packet(input.as_ref()).ok();
    let fast422_packet = j2k_jpeg::adapter::build_fast422_packet(input.as_ref()).ok();
    let fast420_packet = j2k_jpeg::adapter::build_fast420_packet(input.as_ref()).ok();
    assert!(
        fast444_packet.is_some() || fast422_packet.is_some() || fast420_packet.is_some(),
        "generated JPEG must be packet-decodable"
    );
    let requests = (0..8)
        .map(|_| {
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Auto,
                batch::BatchOp::Full,
                fast444_packet.clone(),
                fast422_packet.clone(),
                fast420_packet.clone(),
            )
        })
        .collect::<Vec<_>>();

    assert!(batched_fast_packets(&requests)
        .expect("packet lookup")
        .is_some());
}

fn generated_rgb_jpeg(dim: u16) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(dim as usize * dim as usize * 3);
    for y in 0..dim {
        for x in 0..dim {
            let xf = u32::from(x);
            let yf = u32::from(y);
            rgb.push(((xf * 13 + yf * 3) & 0xff) as u8);
            rgb.push(((xf * 5 + yf * 11 + (xf ^ yf)) & 0xff) as u8);
            rgb.push(((xf * 7 + yf * 17 + (xf.wrapping_mul(yf) >> 5)) & 0xff) as u8);
        }
    }

    let mut jpeg = Vec::new();
    let mut encoder = jpeg_encoder::Encoder::new(&mut jpeg, 90);
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::F_2_2);
    encoder
        .encode(&rgb, dim, dim, jpeg_encoder::ColorType::Rgb)
        .expect("encode generated JPEG");
    jpeg
}

#[test]
fn fast420_packet_scaled_decode_matches_cpu_scaled_bytes() {
    let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
    for scale in [j2k_core::Downscale::Half, j2k_core::Downscale::Quarter] {
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast420_scaled_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                scale,
            )?
            .expect("fast420 scaled surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal scaled");

        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast420_packet_region_decode_matches_cpu_region_bytes() {
    let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
    let roi = j2k_jpeg::Rect {
        x: 3,
        y: 2,
        w: 9,
        h: 10,
    };
    let (expected, _) = decoder
        .decode_region(PixelFormat::Rgb8, roi)
        .expect("cpu region");

    let surface = with_runtime(|runtime| {
        let surface = try_decode_fast420_region_to_surface(
            runtime,
            &decoder,
            Some(&packet),
            PixelFormat::Rgb8,
            roi,
        )?
        .expect("fast420 region surface");
        Ok::<_, Error>(surface)
    })
    .expect("metal region");

    assert_eq!(surface.dimensions, (roi.w, roi.h));
    assert_eq!(surface.fmt, PixelFormat::Rgb8);
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[test]
fn fast420_region_batch_decode_matches_cpu_region_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_420);
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Region(roi),
            None,
            None,
            Some(packet.clone()),
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Region(roi),
            None,
            None,
            Some(packet),
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
    let (expected, _) = decoder
        .decode_region(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
        )
        .expect("cpu region");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("region batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (roi.w, roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast420_full_batch_decode_uses_shared_surface_offsets() {
    let input = Arc::<[u8]>::from(BASELINE_420);
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            None,
            Some(packet.clone()),
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            None,
            Some(packet),
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
    let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("fast420 full batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for (index, result) in results.iter().enumerate() {
        let surface = result.as_ref().expect("surface");
        assert_eq!(surface.dimensions, (16, 16));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
        let Storage::Metal { offset, .. } = &surface.storage else {
            panic!("expected Metal storage");
        };
        assert_eq!(*offset, index * expected.len());
    }
}

#[test]
fn fast420_split_full_batch_decode_matches_cpu_bytes() {
    let jpeg = generated_rgb_jpeg(32);
    let input = Arc::<[u8]>::from(jpeg.into_boxed_slice());
    let packet = j2k_jpeg::adapter::build_fast420_packet(input.as_ref()).expect("packet");
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            None,
            Some(packet.clone()),
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            None,
            Some(packet),
        ),
    ];
    let packets = batched_fast_packets(&requests)
        .expect("packet lookup")
        .expect("packets");
    let decoder = CpuDecoder::new(input.as_ref()).expect("decoder");
    let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

    let results = with_runtime(|runtime| {
        try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output::<
            JpegFast420PacketV1,
        >(
            runtime,
            &requests,
            &packets,
            FastBatchDecodeMode::SplitCoeffIdct,
            None,
        )
    })
    .expect("batch result")
    .expect("split fast420 full batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (32, 32));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast420_batch_clears_high_ac_before_dc_only_blocks() {
    let input = Arc::<[u8]>::from(fast420_high_ac_then_dc_only_jpeg(1));
    let packet = j2k_jpeg::adapter::build_fast420_packet(input.as_ref()).expect("packet");
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            None,
            Some(packet.clone()),
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            None,
            Some(packet),
        ),
    ];
    let decoder = CpuDecoder::new(input.as_ref()).expect("decoder");
    let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("fast420 full batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (16, 16));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast420_batch_matches_cpu_for_high_ac_overflow_coefficients() {
    let input = Arc::<[u8]>::from(fast420_high_ac_then_dc_only_jpeg(u8::MAX));
    let packet = j2k_jpeg::adapter::build_fast420_packet(input.as_ref()).expect("packet");
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            None,
            Some(packet.clone()),
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            None,
            Some(packet),
        ),
    ];
    let decoder = CpuDecoder::new(input.as_ref()).expect("decoder");
    let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("fast420 full batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (16, 16));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast420_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
    let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
    let roi = j2k_jpeg::Rect {
        x: 3,
        y: 2,
        w: 9,
        h: 10,
    };
    let scale = j2k_core::Downscale::Quarter;
    let (expected, _) = decoder
        .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
        .expect("cpu region scaled");
    let scaled_roi = j2k_jpeg::Rect {
        x: roi.x / 4,
        y: roi.y / 4,
        w: (roi.x + roi.w).div_ceil(4) - (roi.x / 4),
        h: (roi.y + roi.h).div_ceil(4) - (roi.y / 4),
    };

    let surface = with_runtime(|runtime| {
        let surface = try_decode_fast420_scaled_region_to_surface(
            runtime,
            &decoder,
            Some(&packet),
            PixelFormat::Rgb8,
            scaled_roi,
            scale,
        )?
        .expect("fast420 scaled region surface");
        Ok::<_, Error>(surface)
    })
    .expect("metal region scaled");

    assert_eq!(surface.dimensions, (scaled_roi.w, scaled_roi.h));
    assert_eq!(surface.fmt, PixelFormat::Rgb8);
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[test]
fn fast420_region_scaled_batch_decode_matches_cpu_region_scaled_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_420);
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
    let roi = Rect {
        x: 3,
        y: 2,
        w: 9,
        h: 10,
    };
    let scale = j2k_core::Downscale::Quarter;
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::RegionScaled { roi, scale },
            None,
            None,
            Some(packet.clone()),
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::RegionScaled { roi, scale },
            None,
            None,
            Some(packet),
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
    let (expected, _) = decoder
        .decode_region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("cpu region scaled");
    let scaled = roi.scaled_covering(scale);

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("region scaled batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (scaled.w, scaled.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast420_scaled_batch_decode_matches_cpu_scaled_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_420);
    let packet = j2k_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
    let scale = j2k_core::Downscale::Quarter;
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Scaled(scale),
            None,
            None,
            Some(packet.clone()),
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Scaled(scale),
            None,
            None,
            Some(packet),
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
    let (expected, _) = decoder
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("scaled batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (4, 4));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast422_packet_full_decode_matches_cpu_bytes() {
    let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
    let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

    let surface = with_runtime(|runtime| {
        let surface = try_decode_fast422_to_surface(runtime, Some(&packet), PixelFormat::Rgb8)?
            .expect("fast422 surface");
        Ok::<_, Error>(surface)
    })
    .expect("metal full decode");

    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[test]
fn fast422_full_batch_decode_matches_cpu_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_422);
    let packet = j2k_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            Some(packet.clone()),
            None,
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            None,
            Some(packet),
            None,
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
    let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("fast422 batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for (index, result) in results.iter().enumerate() {
        let surface = result.as_ref().expect("surface");
        assert_eq!(surface.dimensions, (16, 8));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
        let Storage::Metal { offset, .. } = &surface.storage else {
            panic!("expected Metal storage");
        };
        assert_eq!(*offset, index * expected.len());
    }
}

#[test]
fn fast422_packet_region_decode_matches_cpu_region_bytes() {
    let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
    let roi = j2k_jpeg::Rect {
        x: 3,
        y: 1,
        w: 9,
        h: 5,
    };
    let (expected, _) = decoder
        .decode_region(PixelFormat::Rgb8, roi)
        .expect("cpu region");

    let surface = with_runtime(|runtime| {
        let surface =
            try_decode_fast422_region_to_surface(runtime, Some(&packet), PixelFormat::Rgb8, roi)?
                .expect("fast422 region surface");
        Ok::<_, Error>(surface)
    })
    .expect("metal region");

    assert_eq!(surface.dimensions, (roi.w, roi.h));
    assert_eq!(surface.fmt, PixelFormat::Rgb8);
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[test]
fn fast422_region_batch_decode_matches_cpu_region_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_422);
    let packet = j2k_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
    let roi = Rect {
        x: 3,
        y: 1,
        w: 9,
        h: 5,
    };
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Region(roi),
            None,
            Some(packet.clone()),
            None,
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Region(roi),
            None,
            Some(packet),
            None,
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
    let (expected, _) = decoder
        .decode_region(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
        )
        .expect("cpu region");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("fast422 region batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (roi.w, roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast422_packet_scaled_decode_matches_cpu_scaled_bytes() {
    let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
    for (scale, dims) in [
        (j2k_core::Downscale::Half, (8, 4)),
        (j2k_core::Downscale::Quarter, (4, 2)),
    ] {
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast422_scaled_to_surface(
                runtime,
                Some(&packet),
                PixelFormat::Rgb8,
                scale,
            )?
            .expect("fast422 scaled surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal scaled");

        assert_eq!(surface.dimensions, dims);
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast422_scaled_batch_decode_matches_cpu_scaled_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_422);
    let packet = j2k_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
    let scale = j2k_core::Downscale::Quarter;
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Scaled(scale),
            None,
            Some(packet.clone()),
            None,
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Scaled(scale),
            None,
            Some(packet),
            None,
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
    let (expected, _) = decoder
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("fast422 scaled batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (4, 2));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast422_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
    let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
    let roi = j2k_jpeg::Rect {
        x: 3,
        y: 1,
        w: 9,
        h: 5,
    };
    let scale = j2k_core::Downscale::Half;
    let (expected, _) = decoder
        .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
        .expect("cpu region scaled");
    let scaled_roi = j2k_jpeg::Rect {
        x: roi.x / 2,
        y: roi.y / 2,
        w: (roi.x + roi.w).div_ceil(2) - (roi.x / 2),
        h: (roi.y + roi.h).div_ceil(2) - (roi.y / 2),
    };

    let surface = with_runtime(|runtime| {
        let surface = try_decode_fast422_scaled_region_to_surface(
            runtime,
            Some(&packet),
            PixelFormat::Rgb8,
            scaled_roi,
            scale,
        )?
        .expect("fast422 scaled region surface");
        Ok::<_, Error>(surface)
    })
    .expect("metal region scaled");

    assert_eq!(surface.dimensions, (scaled_roi.w, scaled_roi.h));
    assert_eq!(surface.fmt, PixelFormat::Rgb8);
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[test]
fn fast422_region_scaled_batch_decode_matches_cpu_region_scaled_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_422);
    let packet = j2k_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
    let roi = Rect {
        x: 3,
        y: 1,
        w: 9,
        h: 5,
    };
    let scale = j2k_core::Downscale::Half;
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::RegionScaled { roi, scale },
            None,
            Some(packet.clone()),
            None,
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::RegionScaled { roi, scale },
            None,
            Some(packet),
            None,
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
    let (expected, _) = decoder
        .decode_region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("cpu region scaled");
    let scaled = roi.scaled_covering(scale);

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("fast422 region scaled batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (scaled.w, scaled.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast444_packet_full_decode_matches_cpu_bytes() {
    let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
    let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

    let surface = with_runtime(|runtime| {
        let surface =
            try_decode_fast444_to_surface(runtime, &decoder, Some(&packet), PixelFormat::Rgb8)?
                .expect("fast444 surface");
        Ok::<_, Error>(surface)
    })
    .expect("metal full decode");

    assert_eq!(
        surface.residency(),
        crate::SurfaceResidency::MetalResidentDecode
    );
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[test]
fn fast444_packet_scaled_decode_matches_cpu_scaled_bytes() {
    let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
    for scale in [j2k_core::Downscale::Half, j2k_core::Downscale::Quarter] {
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast444_scaled_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                scale,
            )?
            .expect("fast444 scaled surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal scaled");

        assert_eq!(
            surface.residency(),
            crate::SurfaceResidency::MetalResidentDecode
        );
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast444_packet_region_decode_matches_cpu_region_bytes() {
    let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
    let roi = j2k_jpeg::Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let (expected, _) = decoder
        .decode_region(PixelFormat::Rgb8, roi)
        .expect("cpu region");

    let surface = with_runtime(|runtime| {
        let surface = try_decode_fast444_region_to_surface(
            runtime,
            &decoder,
            Some(&packet),
            PixelFormat::Rgb8,
            roi,
        )?
        .expect("fast444 region surface");
        Ok::<_, Error>(surface)
    })
    .expect("metal region");

    assert_eq!(surface.dimensions, (roi.w, roi.h));
    assert_eq!(surface.fmt, PixelFormat::Rgb8);
    assert_eq!(
        surface.residency(),
        crate::SurfaceResidency::MetalResidentDecode
    );
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[test]
fn fast444_region_batch_decode_matches_cpu_region_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_444);
    let packet = j2k_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Region(roi),
            Some(packet.clone()),
            None,
            None,
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Region(roi),
            Some(packet),
            None,
            None,
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
    let (expected, _) = decoder
        .decode_region(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
        )
        .expect("cpu region");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("region batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (roi.w, roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(
            surface.residency(),
            crate::SurfaceResidency::MetalResidentDecode
        );
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast444_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
    let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
    let packet = j2k_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
    let roi = j2k_jpeg::Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let scale = j2k_core::Downscale::Quarter;
    let (expected, _) = decoder
        .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
        .expect("cpu region scaled");
    let scaled_roi = j2k_jpeg::Rect {
        x: roi.x / 4,
        y: roi.y / 4,
        w: (roi.x + roi.w).div_ceil(4) - (roi.x / 4),
        h: (roi.y + roi.h).div_ceil(4) - (roi.y / 4),
    };

    let surface = with_runtime(|runtime| {
        let surface = try_decode_fast444_scaled_region_to_surface(
            runtime,
            &decoder,
            Some(&packet),
            PixelFormat::Rgb8,
            scaled_roi,
            scale,
        )?
        .expect("fast444 scaled region surface");
        Ok::<_, Error>(surface)
    })
    .expect("metal region scaled");

    assert_eq!(surface.dimensions, (scaled_roi.w, scaled_roi.h));
    assert_eq!(surface.fmt, PixelFormat::Rgb8);
    assert_eq!(
        surface.residency(),
        crate::SurfaceResidency::MetalResidentDecode
    );
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[test]
fn fast444_region_scaled_batch_decode_matches_cpu_region_scaled_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_444);
    let packet = j2k_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let scale = j2k_core::Downscale::Quarter;
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::RegionScaled { roi, scale },
            Some(packet.clone()),
            None,
            None,
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::RegionScaled { roi, scale },
            Some(packet),
            None,
            None,
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
    let (expected, _) = decoder
        .decode_region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        )
        .expect("cpu region scaled");
    let scaled = roi.scaled_covering(scale);

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("region scaled batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (scaled.w, scaled.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(
            surface.residency(),
            crate::SurfaceResidency::MetalResidentDecode
        );
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[test]
fn fast444_scaled_batch_decode_matches_cpu_scaled_bytes() {
    let input = Arc::<[u8]>::from(BASELINE_444);
    let packet = j2k_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
    let scale = j2k_core::Downscale::Quarter;
    let requests = vec![
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Scaled(scale),
            Some(packet.clone()),
            None,
            None,
        ),
        batch::QueuedRequest::new(
            Arc::clone(&input),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Scaled(scale),
            Some(packet),
            None,
            None,
        ),
    ];
    let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
    let (expected, _) = decoder
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled");

    let results = decode_full_batch_to_surfaces(&requests)
        .expect("batch result")
        .expect("scaled batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.dimensions, (2, 2));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(
            surface.residency(),
            crate::SurfaceResidency::MetalResidentDecode
        );
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

fn fast420_high_ac_then_dc_only_jpeg(ac_quant: u8) -> Vec<u8> {
    assert!(ac_quant > 0, "JPEG quant entries must be nonzero");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0xff, 0xd8]);

    let mut quant = [1u8; 64];
    quant[63] = ac_quant;
    let mut dqt = Vec::with_capacity(65);
    dqt.push(0x00);
    dqt.extend_from_slice(&quant);
    append_jpeg_segment(&mut bytes, 0xdb, &dqt);

    append_jpeg_segment(
        &mut bytes,
        0xc0,
        &[
            8,
            0,
            16,
            0,
            16,
            3,
            1,
            (2 << 4) | 2,
            0,
            2,
            (1 << 4) | 1,
            0,
            3,
            (1 << 4) | 1,
            0,
        ],
    );

    let mut dc_bits = [0u8; 16];
    dc_bits[0] = 1;
    let mut dht_dc = Vec::with_capacity(18);
    dht_dc.push(0x00);
    dht_dc.extend_from_slice(&dc_bits);
    dht_dc.push(0x00);
    append_jpeg_segment(&mut bytes, 0xc4, &dht_dc);

    let mut ac_bits = [0u8; 16];
    ac_bits[1] = 3;
    let mut dht_ac = Vec::with_capacity(20);
    dht_ac.push(0x10);
    dht_ac.extend_from_slice(&ac_bits);
    dht_ac.extend_from_slice(&[0x00, 0xf0, 0xea]);
    append_jpeg_segment(&mut bytes, 0xc4, &dht_ac);

    append_jpeg_segment(&mut bytes, 0xda, &[3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);

    bytes.extend_from_slice(&fast420_high_ac_entropy());
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

fn append_jpeg_segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    bytes.extend_from_slice(&[0xff, marker]);
    let len = u16::try_from(payload.len() + 2).expect("JPEG segment length fits in u16");
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(payload);
}

fn minimal_fast420_packet(dimensions: (u32, u32)) -> JpegFast420PacketV1 {
    let [y_dc_table, y_ac_table, cb_dc_table, cb_ac_table, cr_dc_table, cr_ac_table] =
        empty_packet_huffman_tables();
    JpegFast420PacketV1 {
        dimensions,
        mcus_per_row: 1,
        mcu_rows: 1,
        restart_interval_mcus: 0,
        restart_offsets: vec![0],
        entropy_checkpoints: vec![empty_entropy_checkpoint()],
        y_quant: [1; 64],
        cb_quant: [1; 64],
        cr_quant: [1; 64],
        y_dc_table,
        y_ac_table,
        cb_dc_table,
        cb_ac_table,
        cr_dc_table,
        cr_ac_table,
        entropy_bytes: Vec::new(),
    }
}

fn minimal_fast422_packet(dimensions: (u32, u32)) -> JpegFast422PacketV1 {
    let [y_dc_table, y_ac_table, cb_dc_table, cb_ac_table, cr_dc_table, cr_ac_table] =
        empty_packet_huffman_tables();
    JpegFast422PacketV1 {
        dimensions,
        mcus_per_row: 1,
        mcu_rows: 1,
        restart_interval_mcus: 0,
        restart_offsets: vec![0],
        entropy_checkpoints: vec![empty_entropy_checkpoint()],
        y_quant: [1; 64],
        cb_quant: [1; 64],
        cr_quant: [1; 64],
        y_dc_table,
        y_ac_table,
        cb_dc_table,
        cb_ac_table,
        cr_dc_table,
        cr_ac_table,
        entropy_bytes: Vec::new(),
    }
}

fn minimal_fast444_packet(dimensions: (u32, u32)) -> JpegFast444PacketV1 {
    let [y_dc_table, y_ac_table, cb_dc_table, cb_ac_table, cr_dc_table, cr_ac_table] =
        empty_packet_huffman_tables();
    JpegFast444PacketV1 {
        dimensions,
        mcus_per_row: 1,
        mcu_rows: 1,
        restart_interval_mcus: 0,
        restart_offsets: vec![0],
        entropy_checkpoints: vec![empty_entropy_checkpoint()],
        y_quant: [1; 64],
        cb_quant: [1; 64],
        cr_quant: [1; 64],
        y_dc_table,
        y_ac_table,
        cb_dc_table,
        cb_ac_table,
        cr_dc_table,
        cr_ac_table,
        entropy_bytes: Vec::new(),
    }
}

fn empty_packet_huffman_tables() -> [PacketHuffmanTable; 6] {
    std::array::from_fn(|_| PacketHuffmanTable {
        bits: [0; 16],
        values_len: 0,
        values: [0; 256],
    })
}

fn empty_entropy_checkpoint() -> JpegEntropyCheckpointV1 {
    JpegEntropyCheckpointV1 {
        mcu_index: 0,
        entropy_pos: 0,
        bit_acc: 0,
        bit_count: 0,
        y_prev_dc: 0,
        cb_prev_dc: 0,
        cr_prev_dc: 0,
        reserved: 0,
    }
}

fn fast420_high_ac_entropy() -> Vec<u8> {
    let mut writer = EntropyBitWriter::default();
    emit_high_ac_block(&mut writer);
    for _ in 0..5 {
        emit_dc_only_block(&mut writer);
    }
    writer.finish()
}

fn emit_high_ac_block(writer: &mut EntropyBitWriter) {
    writer.push_bits(0, 1);
    for _ in 0..3 {
        writer.push_bits(0b01, 2);
    }
    writer.push_bits(0b10, 2);
    writer.push_bits(0b11_1111_1111, 10);
}

fn emit_dc_only_block(writer: &mut EntropyBitWriter) {
    writer.push_bits(0, 1);
    writer.push_bits(0b00, 2);
}

#[derive(Default)]
struct EntropyBitWriter {
    bytes: Vec<u8>,
    current: u8,
    bit_count: u8,
}

impl EntropyBitWriter {
    fn push_bits(&mut self, bits: u16, len: u8) {
        for shift in (0..len).rev() {
            let bit = u8::from(((bits >> shift) & 1) != 0);
            self.current = (self.current << 1) | bit;
            self.bit_count += 1;
            if self.bit_count == 8 {
                self.push_current_byte();
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_count != 0 {
            let pad_bits = 8 - self.bit_count;
            self.current = (self.current << pad_bits) | ((1u8 << pad_bits) - 1);
            self.push_current_byte();
        }
        self.bytes
    }

    fn push_current_byte(&mut self) {
        self.bytes.push(self.current);
        if self.current == 0xff {
            self.bytes.push(0x00);
        }
        self.current = 0;
        self.bit_count = 0;
    }
}
