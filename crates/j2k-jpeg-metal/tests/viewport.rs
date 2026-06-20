// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use j2k_core::DeviceSurface;
use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};
use j2k_jpeg::{Decoder, ScratchPool};
#[cfg(target_os = "macos")]
use j2k_jpeg_metal::viewport::{
    choose_resizable_metal_viewport_strategy, compose_viewport_hybrid,
    compose_viewport_to_resizable_metal_buffer_with_session,
    compose_viewport_to_resizable_metal_textures_with_session, decode_viewport_region_hybrid,
    decode_viewport_region_to_resizable_metal_buffer_with_session,
    decode_viewport_region_to_resizable_metal_textures_with_session,
    decode_viewport_to_resizable_metal_buffer_with_decoder_session,
    decode_viewport_to_resizable_metal_buffer_with_session,
    decode_viewport_to_resizable_metal_textures_with_decoder_session,
    decode_viewport_to_resizable_metal_textures_with_session, ViewportResidentOutputStrategy,
};
use j2k_jpeg_metal::viewport::{
    choose_viewport_surface_strategy, compose_viewport_cpu, decode_viewport_region_cpu,
    decode_viewport_to_surface, is_contiguous_viewport_workload, suggest_viewport_workload,
    viewport_source_bounds, ViewportSurfaceStrategy, ViewportTile,
};
#[cfg(target_os = "macos")]
use j2k_jpeg_metal::{
    Decoder as MetalDecoder, MetalBackendSession, MetalBatchOutputBuffer, MetalBatchTextureOutput,
    SurfaceResidency,
};

const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const GRAYSCALE: &[u8] = include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg");

fn quadrant_tiles() -> [ViewportTile; 4] {
    [
        ViewportTile {
            source_roi: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
            dest: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
        },
        ViewportTile {
            source_roi: Rect {
                x: 8,
                y: 0,
                w: 8,
                h: 8,
            },
            dest: Rect {
                x: 8,
                y: 0,
                w: 8,
                h: 8,
            },
        },
        ViewportTile {
            source_roi: Rect {
                x: 0,
                y: 8,
                w: 8,
                h: 8,
            },
            dest: Rect {
                x: 0,
                y: 8,
                w: 8,
                h: 8,
            },
        },
        ViewportTile {
            source_roi: Rect {
                x: 8,
                y: 8,
                w: 8,
                h: 8,
            },
            dest: Rect {
                x: 8,
                y: 8,
                w: 8,
                h: 8,
            },
        },
    ]
}

#[cfg(target_os = "macos")]
fn rgb_to_rgba_opaque(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for pixel in rgb.chunks_exact(3) {
        rgba.extend_from_slice(pixel);
        rgba.push(u8::MAX);
    }
    rgba
}

#[cfg(target_os = "macos")]
fn download_rgba8_texture(
    session: &MetalBackendSession,
    texture: &metal::TextureRef,
    dimensions: (u32, u32),
) -> Vec<u8> {
    let row_bytes = dimensions.0 as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let byte_len = row_bytes * dimensions.1 as usize;
    let buffer = session.device().new_buffer(
        byte_len as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let queue = session.device().new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        metal::MTLOrigin { x: 0, y: 0, z: 0 },
        metal::MTLSize::new(u64::from(dimensions.0), u64::from(dimensions.1), 1),
        &buffer,
        0,
        row_bytes as u64,
        byte_len as u64,
        metal::MTLBlitOption::None,
    );
    blit.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // SAFETY: The test buffer size is computed from the validated viewport dimensions.
    unsafe { core::slice::from_raw_parts(buffer.contents().cast::<u8>(), byte_len).to_vec() }
}

#[test]
fn cpu_viewport_quadrants_match_full_decode() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut pool = ScratchPool::new();

    let actual = compose_viewport_cpu(
        &decoder,
        &mut pool,
        PixelFormat::Rgb8,
        Downscale::None,
        (16, 16),
        &quadrant_tiles(),
    )
    .expect("viewport");
    let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("full decode");

    assert_eq!(actual, expected);
}

#[test]
fn suggested_viewport_workload_is_fixed_for_macro_like_input() {
    let workload = suggest_viewport_workload((1_191, 408)).expect("workload");

    assert_eq!(workload.scale, Downscale::Half);
    assert_eq!(workload.viewport_dims, (576, 192));
    assert_eq!(workload.tiles.len(), 12);
    assert_eq!(
        workload.tiles.first(),
        Some(&ViewportTile {
            source_roi: Rect {
                x: 18,
                y: 12,
                w: 192,
                h: 192,
            },
            dest: Rect {
                x: 0,
                y: 0,
                w: 96,
                h: 96,
            },
        })
    );
    assert_eq!(
        workload.tiles.last(),
        Some(&ViewportTile {
            source_roi: Rect {
                x: 978,
                y: 204,
                w: 192,
                h: 192,
            },
            dest: Rect {
                x: 480,
                y: 96,
                w: 96,
                h: 96,
            },
        })
    );
    assert!(is_contiguous_viewport_workload(&workload));
}

#[test]
fn cpu_viewport_misaligned_scaled_tile_matches_direct_decode() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let roi = Rect {
        x: 1,
        y: 1,
        w: 10,
        h: 10,
    };
    let tiles = [ViewportTile {
        source_roi: roi,
        dest: Rect {
            x: 0,
            y: 0,
            w: 6,
            h: 6,
        },
    }];

    let viewport = compose_viewport_cpu(
        &decoder,
        &mut cpu_pool,
        PixelFormat::Rgb8,
        Downscale::Half,
        (6, 6),
        &tiles,
    )
    .expect("cpu viewport");
    let (expected, _outcome) = decoder
        .decode_region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            Downscale::Half,
        )
        .expect("direct decode");

    assert_eq!(expected.len(), 6 * 6 * 3);
    assert_eq!(viewport, expected);
}

#[test]
fn cpu_contiguous_viewport_region_matches_direct_decode() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut pool = ScratchPool::new();
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: quadrant_tiles().to_vec(),
    };

    let actual = decode_viewport_region_cpu(&decoder, &mut pool, PixelFormat::Rgb8, &workload)
        .expect("cpu viewport region");
    let (expected, _) = decoder
        .decode_region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: viewport_source_bounds(&workload).x,
                y: viewport_source_bounds(&workload).y,
                w: viewport_source_bounds(&workload).w,
                h: viewport_source_bounds(&workload).h,
            },
            workload.scale,
        )
        .expect("direct decode");

    assert_eq!(actual, expected);
}

#[test]
fn gapped_tiles_are_not_contiguous() {
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: vec![
            ViewportTile {
                source_roi: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
            },
            ViewportTile {
                source_roi: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
            },
        ],
    };

    assert!(!is_contiguous_viewport_workload(&workload));
    assert_eq!(
        choose_viewport_surface_strategy(&workload, BackendRequest::Cpu).expect("cpu strategy"),
        ViewportSurfaceStrategy::CpuComposite
    );
}

#[test]
fn cpu_auto_strategy_prefers_contiguous_when_available() {
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: quadrant_tiles().to_vec(),
    };

    assert!(is_contiguous_viewport_workload(&workload));
    assert_eq!(
        choose_viewport_surface_strategy(&workload, BackendRequest::Cpu).expect("cpu strategy"),
        ViewportSurfaceStrategy::CpuContiguous
    );
}

#[cfg(target_os = "macos")]
#[test]
fn reusable_metal_viewport_strategy_reports_direct_contiguous_workload() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: quadrant_tiles().to_vec(),
    };

    assert_eq!(
        choose_resizable_metal_viewport_strategy(&decoder, &workload)
            .expect("resident viewport strategy"),
        ViewportResidentOutputStrategy::DirectContiguous
    );
}

#[cfg(target_os = "macos")]
#[test]
fn reusable_metal_viewport_strategy_reports_sparse_composition_workload() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: vec![
            ViewportTile {
                source_roi: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
            },
            ViewportTile {
                source_roi: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
            },
        ],
    };

    assert_eq!(
        choose_resizable_metal_viewport_strategy(&decoder, &workload)
            .expect("resident viewport strategy"),
        ViewportResidentOutputStrategy::Composite
    );
}

#[cfg(target_os = "macos")]
#[test]
fn hybrid_viewport_quadrants_match_cpu_viewport() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut hybrid_pool = ScratchPool::new();

    let expected = compose_viewport_cpu(
        &decoder,
        &mut cpu_pool,
        PixelFormat::Rgb8,
        Downscale::None,
        (16, 16),
        &quadrant_tiles(),
    )
    .expect("cpu viewport");
    let actual = compose_viewport_hybrid(
        &decoder,
        &mut hybrid_pool,
        Downscale::None,
        (16, 16),
        &quadrant_tiles(),
    )
    .expect("hybrid viewport");

    assert_eq!(actual.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn hybrid_viewport_misaligned_scaled_tile_matches_cpu_viewport() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut hybrid_pool = ScratchPool::new();
    let tiles = [ViewportTile {
        source_roi: Rect {
            x: 1,
            y: 1,
            w: 10,
            h: 10,
        },
        dest: Rect {
            x: 0,
            y: 0,
            w: 6,
            h: 6,
        },
    }];

    let expected = compose_viewport_cpu(
        &decoder,
        &mut cpu_pool,
        PixelFormat::Rgb8,
        Downscale::Half,
        (6, 6),
        &tiles,
    )
    .expect("cpu viewport");
    let actual =
        compose_viewport_hybrid(&decoder, &mut hybrid_pool, Downscale::Half, (6, 6), &tiles)
            .expect("hybrid viewport");

    assert_eq!(actual.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn sparse_viewport_composition_resizes_reusable_metal_output_buffer() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut metal_pool = ScratchPool::new();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: vec![
            ViewportTile {
                source_roi: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
            },
            ViewportTile {
                source_roi: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
            },
        ],
    };
    assert!(!is_contiguous_viewport_workload(&workload));
    let expected = compose_viewport_cpu(
        &decoder,
        &mut cpu_pool,
        PixelFormat::Rgb8,
        workload.scale,
        workload.viewport_dims,
        &workload.tiles,
    )
    .expect("cpu viewport");

    let surface = compose_viewport_to_resizable_metal_buffer_with_session(
        &decoder,
        &mut metal_pool,
        &workload,
        &mut output,
        &session,
    )
    .expect("resident sparse viewport");

    assert_eq!(output.dimensions(), workload.viewport_dims);
    assert_eq!(output.tile_capacity(), 1);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert_eq!(surface.dimensions(), workload.viewport_dims);
    assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
    assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
    assert_eq!(offset, 0);
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn reusable_metal_viewport_buffer_helper_routes_sparse_workload() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut metal_pool = ScratchPool::new();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: vec![
            ViewportTile {
                source_roi: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
            },
            ViewportTile {
                source_roi: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
            },
        ],
    };
    assert!(!is_contiguous_viewport_workload(&workload));
    let expected = compose_viewport_cpu(
        &decoder,
        &mut cpu_pool,
        PixelFormat::Rgb8,
        workload.scale,
        workload.viewport_dims,
        &workload.tiles,
    )
    .expect("cpu viewport");

    let surface = decode_viewport_to_resizable_metal_buffer_with_session(
        &decoder,
        &mut metal_pool,
        &workload,
        &mut output,
        &session,
    )
    .expect("resident sparse viewport");

    assert_eq!(output.dimensions(), workload.viewport_dims);
    assert_eq!(output.tile_capacity(), 1);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert_eq!(surface.dimensions(), workload.viewport_dims);
    assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
    assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
    assert_eq!(offset, 0);
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn hybrid_contiguous_viewport_region_matches_cpu_region() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut hybrid_pool = ScratchPool::new();
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: quadrant_tiles().to_vec(),
    };

    let expected =
        decode_viewport_region_cpu(&decoder, &mut cpu_pool, PixelFormat::Rgb8, &workload)
            .expect("cpu viewport region");
    let actual = decode_viewport_region_hybrid(&decoder, &mut hybrid_pool, &workload)
        .expect("hybrid viewport region");

    assert_eq!(actual.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn contiguous_viewport_region_resizes_reusable_metal_output_buffer() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scaled = roi.scaled_covering(Downscale::Quarter);
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::Quarter,
        viewport_dims: (scaled.w, scaled.h),
        tiles: vec![ViewportTile {
            source_roi: roi,
            dest: Rect {
                x: 0,
                y: 0,
                w: scaled.w,
                h: scaled.h,
            },
        }],
    };
    let expected =
        decode_viewport_region_cpu(&decoder, &mut cpu_pool, PixelFormat::Rgb8, &workload)
            .expect("cpu viewport region");

    let surface = decode_viewport_region_to_resizable_metal_buffer_with_session(
        BASELINE_420,
        &workload,
        &mut output,
        &session,
    )
    .expect("resident viewport region");

    assert_eq!(output.dimensions(), workload.viewport_dims);
    assert_eq!(output.tile_capacity(), 1);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert_eq!(surface.dimensions(), workload.viewport_dims);
    assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
    assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
    assert_eq!(offset, 0);
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn contiguous_viewport_region_resizes_reusable_metal_textures() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scaled = roi.scaled_covering(Downscale::Quarter);
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::Quarter,
        viewport_dims: (scaled.w, scaled.h),
        tiles: vec![ViewportTile {
            source_roi: roi,
            dest: Rect {
                x: 0,
                y: 0,
                w: scaled.w,
                h: scaled.h,
            },
        }],
    };
    let expected_rgb =
        decode_viewport_region_cpu(&decoder, &mut cpu_pool, PixelFormat::Rgb8, &workload)
            .expect("cpu viewport region");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tile = decode_viewport_region_to_resizable_metal_textures_with_session(
        BASELINE_420,
        &workload,
        &mut output,
        &session,
    )
    .expect("resident viewport texture");

    assert_eq!(output.dimensions(), workload.viewport_dims);
    assert_eq!(output.tile_capacity(), 1);
    assert_eq!(tile.dimensions(), workload.viewport_dims);
    assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
    assert!(std::ptr::eq(
        tile.texture(),
        output.texture(0).expect("output texture")
    ));
    assert_eq!(
        download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
        expected_rgba
    );
}

#[cfg(target_os = "macos")]
#[test]
fn reusable_metal_viewport_decoder_helper_routes_contiguous_workload_to_buffer() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let metal_decoder = MetalDecoder::new(BASELINE_420).expect("metal decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut metal_pool = ScratchPool::new();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scaled = roi.scaled_covering(Downscale::Quarter);
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::Quarter,
        viewport_dims: (scaled.w, scaled.h),
        tiles: vec![ViewportTile {
            source_roi: roi,
            dest: Rect {
                x: 0,
                y: 0,
                w: scaled.w,
                h: scaled.h,
            },
        }],
    };
    assert!(is_contiguous_viewport_workload(&workload));
    let expected =
        decode_viewport_region_cpu(&decoder, &mut cpu_pool, PixelFormat::Rgb8, &workload)
            .expect("cpu viewport region");

    let surface = decode_viewport_to_resizable_metal_buffer_with_decoder_session(
        &metal_decoder,
        &mut metal_pool,
        &workload,
        &mut output,
        &session,
    )
    .expect("resident viewport buffer");

    assert_eq!(output.dimensions(), workload.viewport_dims);
    assert_eq!(output.tile_capacity(), 1);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert_eq!(surface.dimensions(), workload.viewport_dims);
    assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
    assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
    assert_eq!(offset, 0);
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn reusable_metal_viewport_decoder_helper_routes_contiguous_workload_to_textures() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let metal_decoder = MetalDecoder::new(BASELINE_420).expect("metal decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut metal_pool = ScratchPool::new();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scaled = roi.scaled_covering(Downscale::Quarter);
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::Quarter,
        viewport_dims: (scaled.w, scaled.h),
        tiles: vec![ViewportTile {
            source_roi: roi,
            dest: Rect {
                x: 0,
                y: 0,
                w: scaled.w,
                h: scaled.h,
            },
        }],
    };
    assert!(is_contiguous_viewport_workload(&workload));
    let expected_rgb =
        decode_viewport_region_cpu(&decoder, &mut cpu_pool, PixelFormat::Rgb8, &workload)
            .expect("cpu viewport region");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tile = decode_viewport_to_resizable_metal_textures_with_decoder_session(
        &metal_decoder,
        &mut metal_pool,
        &workload,
        &mut output,
        &session,
    )
    .expect("resident viewport texture");

    assert_eq!(output.dimensions(), workload.viewport_dims);
    assert_eq!(output.tile_capacity(), 1);
    assert_eq!(tile.dimensions(), workload.viewport_dims);
    assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
    assert!(std::ptr::eq(
        tile.texture(),
        output.texture(0).expect("output texture")
    ));
    assert_eq!(
        download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
        expected_rgba
    );
}

#[cfg(target_os = "macos")]
#[test]
fn reusable_metal_viewport_texture_helper_routes_contiguous_workload() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut metal_pool = ScratchPool::new();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scaled = roi.scaled_covering(Downscale::Quarter);
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::Quarter,
        viewport_dims: (scaled.w, scaled.h),
        tiles: vec![ViewportTile {
            source_roi: roi,
            dest: Rect {
                x: 0,
                y: 0,
                w: scaled.w,
                h: scaled.h,
            },
        }],
    };
    assert!(is_contiguous_viewport_workload(&workload));
    let expected_rgb =
        decode_viewport_region_cpu(&decoder, &mut cpu_pool, PixelFormat::Rgb8, &workload)
            .expect("cpu viewport region");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tile = decode_viewport_to_resizable_metal_textures_with_session(
        &decoder,
        &mut metal_pool,
        &workload,
        &mut output,
        &session,
    )
    .expect("resident viewport texture");

    assert_eq!(output.dimensions(), workload.viewport_dims);
    assert_eq!(output.tile_capacity(), 1);
    assert_eq!(tile.dimensions(), workload.viewport_dims);
    assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
    assert!(std::ptr::eq(
        tile.texture(),
        output.texture(0).expect("output texture")
    ));
    assert_eq!(
        download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
        expected_rgba
    );
}

#[cfg(target_os = "macos")]
#[test]
fn sparse_viewport_composition_resizes_reusable_metal_texture_output() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut cpu_pool = ScratchPool::new();
    let mut metal_pool = ScratchPool::new();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: vec![
            ViewportTile {
                source_roi: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 8,
                },
            },
            ViewportTile {
                source_roi: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
                dest: Rect {
                    x: 8,
                    y: 8,
                    w: 8,
                    h: 8,
                },
            },
        ],
    };
    assert!(!is_contiguous_viewport_workload(&workload));
    let expected_rgb = compose_viewport_cpu(
        &decoder,
        &mut cpu_pool,
        PixelFormat::Rgb8,
        workload.scale,
        workload.viewport_dims,
        &workload.tiles,
    )
    .expect("cpu viewport");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tile = compose_viewport_to_resizable_metal_textures_with_session(
        &decoder,
        &mut metal_pool,
        &workload,
        &mut output,
        &session,
    )
    .expect("resident sparse viewport texture");

    assert_eq!(output.dimensions(), workload.viewport_dims);
    assert_eq!(output.tile_capacity(), 1);
    assert_eq!(tile.dimensions(), workload.viewport_dims);
    assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
    assert!(std::ptr::eq(
        tile.texture(),
        output.texture(0).expect("output texture")
    ));
    assert_eq!(
        download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
        expected_rgba
    );
}

#[cfg(target_os = "macos")]
#[test]
fn auto_viewport_surface_path_prefers_cpu_for_small_contiguous_workloads() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut direct_pool = ScratchPool::new();
    let mut auto_pool = ScratchPool::new();
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: quadrant_tiles().to_vec(),
    };

    let expected = j2k_jpeg_metal::viewport::decode_viewport_region_cpu_to_surface(
        &decoder,
        &mut direct_pool,
        &workload,
    )
    .expect("cpu viewport surface");
    let actual =
        decode_viewport_to_surface(&decoder, &mut auto_pool, &workload, BackendRequest::Auto)
            .expect("auto viewport surface");

    assert_eq!(actual.as_bytes(), expected.as_bytes());
}

#[cfg(not(target_os = "macos"))]
#[test]
fn non_macos_auto_viewport_surface_returns_cpu_surface() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut pool = ScratchPool::new();
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: quadrant_tiles().to_vec(),
    };

    let surface = decode_viewport_to_surface(&decoder, &mut pool, &workload, BackendRequest::Auto)
        .expect("auto viewport surface");

    assert_eq!(
        j2k_core::DeviceSurface::backend_kind(&surface),
        j2k_core::BackendKind::Cpu
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn non_macos_explicit_metal_viewport_surface_is_unavailable() {
    let decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut pool = ScratchPool::new();
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (16, 16),
        tiles: quadrant_tiles().to_vec(),
    };

    let result = decode_viewport_to_surface(&decoder, &mut pool, &workload, BackendRequest::Metal);
    assert!(matches!(
        result,
        Err(j2k_jpeg_metal::Error::MetalUnavailable)
    ));
}

#[test]
fn explicit_metal_viewport_unsupported_shape_is_rejected() {
    let decoder = Decoder::new(GRAYSCALE).expect("decoder");
    let mut pool = ScratchPool::new();
    let workload = j2k_jpeg_metal::viewport::ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (8, 8),
        tiles: vec![ViewportTile {
            source_roi: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
            dest: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
        }],
    };

    let result = decode_viewport_to_surface(&decoder, &mut pool, &workload, BackendRequest::Metal);

    match result {
        Err(j2k_jpeg_metal::Error::UnsupportedMetalRequest { reason }) => {
            assert!(reason.contains("JPEG Metal"));
        }
        #[cfg(not(target_os = "macos"))]
        Err(j2k_jpeg_metal::Error::MetalUnavailable) => {
            panic!("unsupported shape should be rejected before host availability")
        }
        Err(other) => panic!("unexpected explicit Metal viewport error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal viewport must not fall back; got {:?}",
            j2k_core::DeviceSurface::backend_kind(&surface)
        ),
    }
}
