// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Metal baseline encoder produces the CPU encoder's bytes, and both
//! produce the cross-backend golden of `j2k_test_support`.

#![cfg(target_os = "macos")]

use j2k_core::PixelFormat;
use j2k_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use j2k_jpeg_metal::{
    encode_jpeg_baseline_batch_from_metal_buffers, encode_jpeg_baseline_from_metal_buffer,
    JpegBaselineMetalEncodeTile, MetalBackendSession,
};
use j2k_test_support::{
    jpeg_encode_matrix_cases, jpeg_encode_matrix_digest, jpeg_encode_padded_cases,
    jpeg_encode_transcode_fixtures, replicate_edges, JpegEncodeMatrixCase,
    JpegEncodeMatrixSampling, JPEG_ENCODE_MATRIX_SHA256, JPEG_ENCODE_TRANSCODE_FIXTURES_SHA256,
};

fn matrix_subsampling(sampling: JpegEncodeMatrixSampling) -> JpegSubsampling {
    match sampling {
        JpegEncodeMatrixSampling::Gray => JpegSubsampling::Gray,
        JpegEncodeMatrixSampling::Ybr444 => JpegSubsampling::Ybr444,
        JpegEncodeMatrixSampling::Ybr422 => JpegSubsampling::Ybr422,
        JpegEncodeMatrixSampling::Ybr420 => JpegSubsampling::Ybr420,
    }
}

fn options(case: JpegEncodeMatrixCase, backend: JpegBackend) -> JpegEncodeOptions {
    JpegEncodeOptions {
        quality: case.quality,
        subsampling: matrix_subsampling(case.sampling),
        restart_interval: case.restart_interval,
        backend,
    }
}

fn cpu_frame(case: JpegEncodeMatrixCase, data: &[u8]) -> Vec<u8> {
    let (width, height) = (case.width, case.height);
    let samples = if case.sampling == JpegEncodeMatrixSampling::Gray {
        JpegSamples::Gray8 {
            data,
            width,
            height,
        }
    } else {
        JpegSamples::Rgb8 {
            data,
            width,
            height,
        }
    };
    encode_jpeg_baseline(samples, options(case, JpegBackend::Cpu))
        .unwrap_or_else(|error| panic!("CPU {}: {error}", case.label()))
        .data
}

/// The Metal frame of `case`'s input `data`, coded as a frame of `output`
/// size.
fn metal_frame(
    case: JpegEncodeMatrixCase,
    data: &[u8],
    output: (u32, u32),
    session: &MetalBackendSession,
) -> Vec<u8> {
    let format = if case.sampling == JpegEncodeMatrixSampling::Gray {
        PixelFormat::Gray8
    } else {
        PixelFormat::Rgb8
    };
    let buffer = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), data)
        .expect("upload matrix samples");
    // SAFETY: the buffer was initialized before tile construction and no CPU
    // or GPU writer accesses it while the tile is alive.
    let tile = unsafe {
        JpegBaselineMetalEncodeTile::new(
            &buffer,
            0,
            (case.width, case.height),
            case.width as usize * case.channels(),
            output,
            format,
        )
    };
    encode_jpeg_baseline_from_metal_buffer(tile, options(case, JpegBackend::Metal), session)
        .unwrap_or_else(|error| panic!("Metal {}: {error}", case.label()))
        .data
}

#[test]
fn metal_encoder_matches_the_cpu_encoder_and_the_cross_backend_golden() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let frames = jpeg_encode_matrix_cases()
        .into_iter()
        .map(|case| {
            let data = case.samples();
            let metal = metal_frame(case, &data, (case.width, case.height), &session);
            assert!(
                metal == cpu_frame(case, &data),
                "{}: Metal and CPU bytes differ",
                case.label()
            );
            metal
        })
        .collect::<Vec<_>>();
    assert_eq!(
        jpeg_encode_matrix_digest(frames.iter().map(Vec::as_slice)),
        JPEG_ENCODE_MATRIX_SHA256,
        "Metal baseline JPEG bytes changed"
    );
}

/// The transcode tests' synthetic inputs, which the CPU once encoded
/// differently on aarch64 and x86-64, encode to the CPU's bytes.
#[test]
fn metal_encodes_the_transcode_fixtures_as_the_cpu_does() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let frames = jpeg_encode_transcode_fixtures()
        .into_iter()
        .map(|fixture| {
            let case = fixture.case;
            let metal = metal_frame(case, &fixture.samples, (case.width, case.height), &session);
            assert!(
                metal == cpu_frame(case, &fixture.samples),
                "{}: Metal and CPU bytes differ",
                fixture.name
            );
            metal
        })
        .collect::<Vec<_>>();
    assert_eq!(
        jpeg_encode_matrix_digest(frames.iter().map(Vec::as_slice)),
        JPEG_ENCODE_TRANSCODE_FIXTURES_SHA256,
        "Metal transcode fixture bytes changed"
    );
}

/// Three 2048x2048 4:4:4 tiles of one buffer have a worst case of 192 MiB
/// each, over the 512 MiB live host cap together. The batch encodes as a
/// chunk of two and then one, to the CPU's bytes.
#[test]
fn metal_batch_over_the_host_cap_encodes_in_chunks_as_the_cpu_does() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let (size, count) = (2048_u32, 3_usize);
    let pitch = size as usize * 3;
    let tile_bytes = pitch * size as usize;
    let rgb = j2k_test_support::patterned_rgb8_tiles(size, size, count);
    let options = |backend| JpegEncodeOptions {
        quality: 90,
        subsampling: JpegSubsampling::Ybr444,
        restart_interval: None,
        backend,
    };
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let buffer = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &rgb)
        .expect("upload batch samples");
    let tiles = (0..count)
        .map(|tile| {
            // SAFETY: the buffer was initialized before tile construction and
            // no CPU or GPU writer accesses it while the tiles are alive.
            unsafe {
                JpegBaselineMetalEncodeTile::new(
                    &buffer,
                    tile * tile_bytes,
                    (size, size),
                    pitch,
                    (size, size),
                    PixelFormat::Rgb8,
                )
            }
        })
        .collect::<Vec<_>>();

    let frames = encode_jpeg_baseline_batch_from_metal_buffers(
        &tiles,
        options(JpegBackend::Metal),
        &session,
    )
    .expect("Metal batch over the worst-case cap");

    assert_eq!(frames.len(), count);
    for (index, (frame, data)) in frames.iter().zip(rgb.chunks_exact(tile_bytes)).enumerate() {
        let samples = JpegSamples::Rgb8 {
            data,
            width: size,
            height: size,
        };
        let cpu = encode_jpeg_baseline(samples, options(JpegBackend::Cpu)).expect("CPU encode");
        assert!(
            frame.data == cpu.data,
            "tile {index}: Metal and CPU bytes differ"
        );
    }
}

/// A frame beyond its input codes the input's last column and row, as the
/// CPU encoder codes the samples beyond its image.
#[test]
fn metal_padded_frames_match_the_cpu_encoding_of_the_edge_replicated_image() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    for (case, frame) in jpeg_encode_padded_cases() {
        let data = case.samples();
        let metal = metal_frame(case, &data, frame, &session);
        let input = (case.width, case.height);
        let replicated = replicate_edges(&data, input, case.channels(), frame);
        let cpu = cpu_frame(
            JpegEncodeMatrixCase {
                width: frame.0,
                height: frame.1,
                ..case
            },
            &replicated,
        );
        assert!(
            metal == cpu,
            "{} in a {frame:?} frame: Metal and CPU bytes differ",
            case.label()
        );
    }
}
