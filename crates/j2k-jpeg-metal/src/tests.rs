use super::*;

// Shims over the collapsed batch API so every legacy entry shape
// (source x op x target) keeps device coverage.
#[cfg(target_os = "macos")]
fn decode_rgb8_batch_into_metal_buffer_with_session(
    inputs: &[&[u8]],
    output: &MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Bytes(inputs),
            op: Rgb8MetalBatchOp::Full,
        },
        MetalBufferBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_batch_into_metal_textures_with_session(
    inputs: &[&[u8]],
    output: &MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Bytes(inputs),
            op: Rgb8MetalBatchOp::Full,
        },
        MetalTextureBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_batch_into_metal_buffer_with_session(
    decoders: &[&Decoder<'_>],
    output: &MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::Full,
        },
        MetalBufferBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_batch_into_metal_textures_with_session(
    decoders: &[&Decoder<'_>],
    output: &MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::Full,
        },
        MetalTextureBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_batch_into_resizable_metal_buffer_with_session(
    decoders: &[&Decoder<'_>],
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::Full,
        },
        MetalBufferBatchTarget::Resizable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_scaled_batch_into_metal_buffer_with_session(
    inputs: &[&[u8]],
    scale: Downscale,
    output: &MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Bytes(inputs),
            op: Rgb8MetalBatchOp::Scaled(scale),
        },
        MetalBufferBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_scaled_batch_into_metal_textures_with_session(
    inputs: &[&[u8]],
    scale: Downscale,
    output: &MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Bytes(inputs),
            op: Rgb8MetalBatchOp::Scaled(scale),
        },
        MetalTextureBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_scaled_batch_into_resizable_metal_textures_with_session(
    inputs: &[&[u8]],
    scale: Downscale,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Bytes(inputs),
            op: Rgb8MetalBatchOp::Scaled(scale),
        },
        MetalTextureBatchTarget::Resizable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_scaled_batch_into_metal_buffer_with_session(
    decoders: &[&Decoder<'_>],
    scale: Downscale,
    output: &MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::Scaled(scale),
        },
        MetalBufferBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_scaled_batch_into_metal_textures_with_session(
    decoders: &[&Decoder<'_>],
    scale: Downscale,
    output: &MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::Scaled(scale),
        },
        MetalTextureBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_scaled_batch_into_resizable_metal_buffer_with_session(
    decoders: &[&Decoder<'_>],
    scale: Downscale,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::Scaled(scale),
        },
        MetalBufferBatchTarget::Resizable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_scaled_batch_into_resizable_metal_textures_with_session(
    decoders: &[&Decoder<'_>],
    scale: Downscale,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::Scaled(scale),
        },
        MetalTextureBatchTarget::Resizable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_region_scaled_batch_into_metal_buffer_with_session(
    inputs: &[&[u8]],
    roi: Rect,
    scale: Downscale,
    output: &MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Bytes(inputs),
            op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
        },
        MetalBufferBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
    inputs: &[&[u8]],
    roi: Rect,
    scale: Downscale,
    output: &MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Bytes(inputs),
            op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
        },
        MetalTextureBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_region_scaled_batch_into_metal_buffer_with_session(
    decoders: &[&Decoder<'_>],
    roi: Rect,
    scale: Downscale,
    output: &MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
        },
        MetalBufferBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_region_scaled_batch_into_metal_textures_with_session(
    decoders: &[&Decoder<'_>],
    roi: Rect,
    scale: Downscale,
    output: &MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
        },
        MetalTextureBatchTarget::Reusable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_region_scaled_batch_into_resizable_metal_buffer_with_session(
    decoders: &[&Decoder<'_>],
    roi: Rect,
    scale: Downscale,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    Codec::decode_rgb8_batch_into_buffer_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
        },
        MetalBufferBatchTarget::Resizable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn decode_rgb8_decoder_region_scaled_batch_into_resizable_metal_textures_with_session(
    decoders: &[&Decoder<'_>],
    roi: Rect,
    scale: Downscale,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
    Codec::decode_rgb8_batch_into_textures_with_session(
        Rgb8MetalBatchRequest {
            source: Rgb8MetalBatchSource::Decoders(decoders),
            op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
        },
        MetalTextureBatchTarget::Resizable(output),
        session,
    )
}

#[cfg(target_os = "macos")]
fn assert_reusable_rgba_texture_tiles(
    session: &MetalBackendSession,
    output: &MetalBatchTextureOutput,
    tiles: Vec<Result<MetalTextureTile, Error>>,
    dimensions: (u32, u32),
    expected_tiles: &[&[u8]],
) {
    assert_eq!(tiles.len(), expected_tiles.len());
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        let actual_rgba = download_rgba8_texture(session, tile.texture(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index]);
    }
}

#[cfg(target_os = "macos")]
use j2k_jpeg::adapter::build_fast422_packet;
use j2k_jpeg::adapter::{build_fast420_packet, build_fast444_packet};
#[cfg(target_os = "macos")]
use j2k_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};

const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
const BASELINE_420_RESTART: &[u8] =
    include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");
#[cfg(target_os = "macos")]
const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");
#[cfg(not(target_os = "macos"))]
const GRAYSCALE: &[u8] = include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg");

#[test]
fn metal_runtime_failures_are_not_unsupported_errors() {
    for err in [
        Error::MetalRuntime {
            message: "runtime".to_string(),
        },
        Error::MetalKernel {
            message: "kernel".to_string(),
        },
        Error::MetalStatePoisoned {
            state: "JPEG Metal session",
        },
    ] {
        assert!(!err.is_unsupported(), "{err:?}");
    }
}

#[test]
fn auto_route_prefers_cpu_host_for_nonrestart_packets() {
    let decoder_420 = CpuDecoder::new(BASELINE_420).expect("420 decoder");
    let packet_420 = build_fast420_packet(BASELINE_420).expect("420 packet");
    assert_eq!(
        choose_route(
            &decoder_420,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::Full,
            None,
            None,
            Some(&packet_420),
        ),
        routing::RouteDecision::CpuHost
    );

    let decoder_444 = CpuDecoder::new(BASELINE_444).expect("444 decoder");
    let packet_444 = build_fast444_packet(BASELINE_444).expect("444 packet");
    assert_eq!(
        choose_route(
            &decoder_444,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::Scaled(Downscale::Quarter),
            Some(&packet_444),
            None,
            None,
        ),
        routing::RouteDecision::CpuHost
    );
}

#[test]
fn auto_route_keeps_small_single_restart_packets_on_cpu_host() {
    let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
    let packet = build_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

    assert_eq!(
        choose_route(
            &decoder,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::Full,
            None,
            None,
            Some(&packet)
        ),
        routing::RouteDecision::CpuHost
    );
    assert_eq!(
        choose_route(
            &decoder,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::Region(Rect {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
            }),
            None,
            None,
            Some(&packet),
        ),
        routing::RouteDecision::CpuHost
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_backend_session_reuses_compiled_runtime() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    assert!(session.runtime.get().is_none());

    let mut first = Decoder::new(BASELINE_420).expect("first decoder");
    let first_surface = first
        .decode_to_device_with_session(PixelFormat::Rgb8, &session)
        .expect("first session decode");
    assert_eq!(
        first_surface.residency(),
        SurfaceResidency::MetalResidentDecode
    );
    let first_runtime = session
        .runtime
        .get()
        .and_then(|runtime| runtime.as_ref().ok())
        .map(std::ptr::from_ref::<compute::MetalRuntime>)
        .expect("session runtime after first decode");

    let mut second = Decoder::new(BASELINE_420).expect("second decoder");
    second
        .decode_to_device_with_session(PixelFormat::Rgb8, &session)
        .expect("second session decode");
    let second_runtime = session
        .runtime
        .get()
        .and_then(|runtime| runtime.as_ref().ok())
        .map(std::ptr::from_ref::<compute::MetalRuntime>)
        .expect("session runtime after second decode");

    assert_eq!(first_runtime, second_runtime);
}

#[cfg(target_os = "macos")]
#[test]
fn jpeg_rgb8_batch_decode_uses_backend_session_runtime() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    assert!(session.runtime.get().is_none());

    let inputs = [BASELINE_420, BASELINE_420];
    let results = decode_rgb8_batch_to_device_with_session(&inputs, &session)
        .expect("session batch decode")
        .expect("baseline JPEG batch should use Metal batch path");

    assert_eq!(results.len(), 2);
    assert!(session.runtime.get().is_some());
    for result in results {
        let surface = result.expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn queued_jpeg_batch_decode_uses_metal_session_runtime() {
    use j2k_core::DeviceSubmission as _;

    let backend_session = MetalBackendSession::system_default().expect("Metal backend session");
    assert!(backend_session.runtime.get().is_none());
    let mut session = MetalSession::with_backend_session(backend_session.clone());
    let mut ctx = j2k_core::DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = ScratchPool::new();

    let submissions = (0..2)
        .map(|_| {
            <Codec as j2k_core::TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("queued Metal tile submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("queued Metal surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
    }

    assert_eq!(session.submissions().expect("session submissions"), 1);
    assert!(
        backend_session.runtime.get().is_some(),
        "queued MetalSession batch decode should reuse its backend runtime"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn default_queued_jpeg_batch_decode_lazily_initializes_backend_session() {
    use j2k_core::DeviceSubmission as _;

    let mut session = MetalSession::default();
    assert!(session
        .shared
        .0
        .lock()
        .expect("metal session")
        .backend_session
        .is_none());
    let mut ctx = j2k_core::DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = ScratchPool::new();

    let submissions = (0..2)
        .map(|_| {
            <Codec as j2k_core::TileBatchDecodeSubmit>::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                BASELINE_420,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("queued Metal tile submit")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("queued Metal surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    }

    let runtime_initialized = session
        .shared
        .0
        .lock()
        .expect("metal session")
        .backend_session
        .as_ref()
        .and_then(|backend| backend.runtime.get())
        .is_some();
    assert!(runtime_initialized);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_batch_decode_can_write_into_reusable_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let surfaces = decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
        .expect("decode into reusable output");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(
        output.tile_stride_bytes(),
        16 * 16 * PixelFormat::Rgb8.bytes_per_pixel()
    );
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_resizes_reusable_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let surfaces = decode_rgb8_decoder_batch_into_resizable_metal_buffer_with_session(
        &decoders,
        &mut output,
        &session,
    )
    .expect("decode cached decoder batch into resizable reusable output");

    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_can_write_into_fixed_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let surfaces =
        decode_rgb8_decoder_batch_into_metal_buffer_with_session(&decoders, &output, &session)
            .expect("decode cached decoder batch into fixed reusable output");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (16, 16));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_rejects_mixed_output_dimensions_without_resizing_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_444).expect("second decoder");
    let decoders = [&first, &second];

    let Err(err) = decode_rgb8_decoder_batch_into_resizable_metal_buffer_with_session(
        &decoders,
        &mut output,
        &session,
    ) else {
        panic!("mixed output dimensions should be rejected");
    };

    assert!(matches!(err, Error::UnsupportedMetalRequest { .. }));
    assert_eq!(output.dimensions(), (1, 1));
    assert_eq!(output.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_rejects_mixed_sampling_without_resizing_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let rgb = j2k_test_support::patterned_rgb8(16, 16);
    let fast420 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast420 jpeg");
    let fast444 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast444 jpeg");
    let first = Decoder::new(&fast420.data).expect("first decoder");
    let second = Decoder::new(&fast444.data).expect("second decoder");
    let decoders = [&first, &second];

    let Err(err) = decode_rgb8_decoder_batch_into_resizable_metal_buffer_with_session(
        &decoders,
        &mut output,
        &session,
    ) else {
        panic!("mixed sampling should be rejected");
    };

    assert!(matches!(
        err,
        Error::UnsupportedMetalRequest { reason }
            if reason.contains("same fast-packet sampling family")
    ));
    assert_eq!(output.dimensions(), (1, 1));
    assert_eq!(output.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_metal_report_exposes_required_output_shape() {
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];

    let full =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);
    let scaled = Codec::inspect_rgb8_decoder_batch_metal_output(
        &decoders,
        j2k_jpeg::JpegDecodeOp::Scaled(Downscale::Quarter),
    );
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let region_scaled = Codec::inspect_rgb8_decoder_batch_metal_output(
        &decoders,
        j2k_jpeg::JpegDecodeOp::RegionScaled {
            roi: j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale: Downscale::Quarter,
        },
    );

    assert!(full.eligibility.eligible);
    assert_eq!(full.tile_count, 2);
    assert_eq!(full.output_dimensions, Some((16, 16)));
    assert_eq!(full.required_tile_capacity(), 2);

    assert!(scaled.eligibility.eligible);
    assert_eq!(scaled.output_dimensions, Some((4, 4)));

    assert!(region_scaled.eligibility.eligible);
    let expected = roi.scaled_covering(Downscale::Quarter);
    assert_eq!(
        region_scaled.output_dimensions,
        Some((expected.w, expected.h))
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_metal_report_rejects_incompatible_batches_without_launch() {
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_444).expect("second decoder");
    let decoders = [&first, &second];

    let mixed =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);
    let region = Codec::inspect_rgb8_decoder_batch_metal_output(
        &[&first],
        j2k_jpeg::JpegDecodeOp::Region(j2k_jpeg::Rect {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
        }),
    );

    assert!(!mixed.eligibility.eligible);
    assert_eq!(mixed.output_dimensions, None);
    assert!(mixed
        .eligibility
        .reason
        .expect("mixed rejection")
        .contains("matching output dimensions"));

    assert!(!region.eligibility.eligible);
    assert!(region
        .eligibility
        .reason
        .expect("region rejection")
        .contains("full, scaled, or region-scaled"));
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_metal_report_rejects_mixed_sampling_family() {
    let rgb = j2k_test_support::patterned_rgb8(16, 16);
    let fast420 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast420 jpeg");
    let fast444 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast444 jpeg");
    let first = Decoder::new(&fast420.data).expect("first decoder");
    let second = Decoder::new(&fast444.data).expect("second decoder");
    let decoders = [&first, &second];

    let report =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);

    assert!(!report.eligibility.eligible);
    assert_eq!(report.output_dimensions, None);
    assert!(report
        .eligibility
        .reason
        .expect("mixed sampling rejection")
        .contains("same fast-packet sampling family"));
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_metal_report_rejects_restart_fast422_full_tiles() {
    let rgb = j2k_test_support::patterned_rgb8(64, 32);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 64,
            height: 32,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart fast422 jpeg");
    let packet = build_fast422_packet(&jpeg.data).expect("restart fast422 packet");
    assert_ne!(packet.restart_interval_mcus, 0);
    let first = Decoder::new(&jpeg.data).expect("first decoder");
    let second = Decoder::new(&jpeg.data).expect("second decoder");
    let decoders = [&first, &second];

    let full =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);
    let scaled = Codec::inspect_rgb8_decoder_batch_metal_output(
        &decoders,
        j2k_jpeg::JpegDecodeOp::Scaled(Downscale::Half),
    );

    assert!(!full.eligibility.eligible);
    assert_eq!(full.output_dimensions, None);
    assert!(full
        .eligibility
        .reason
        .expect("restart fast422 full rejection")
        .contains("restart-coded full-tile 4:2:2 or 4:4:4"));

    assert!(scaled.eligibility.eligible);
    assert_eq!(scaled.output_dimensions, Some((32, 16)));
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast444_batch_decode_can_write_into_reusable_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (8, 8), 2).expect("output buffer");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");

    let surfaces = decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
        .expect("decode into reusable output");

    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (8, 8));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
fn assert_table_mixed_full_buffer_groups_resident(
    subsampling: JpegSubsampling,
    dimensions: (u32, u32),
    first_quality: u8,
    second_quality: u8,
) {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(43)
            .wrapping_add(17);
        pixel[0] ^= delta.rotate_left(1);
        pixel[1] = pixel[1].wrapping_sub(delta);
        pixel[2] = pixel[2].wrapping_add(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(47)
            .wrapping_add(23);
        pixel[0] = pixel[0].wrapping_add(delta.rotate_left(2));
        pixel[1] ^= delta.rotate_right(1);
        pixel[2] = pixel[2].wrapping_sub(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: first_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first table-mixed full buffer jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: second_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second table-mixed full buffer jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: first_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third table-mixed full buffer jpeg");

    match subsampling {
        JpegSubsampling::Ybr420 => {
            let packet_a = build_fast420_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast420_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast420_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Ybr422 => {
            let packet_a = build_fast422_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast422_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast422_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Ybr444 => {
            let packet_a = build_fast444_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast444_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast444_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Gray => panic!("table-mixed buffer helper expects YCbCr sampling"),
    }

    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, dimensions, 3).expect("output buffer");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let expected_tiles = inputs
        .iter()
        .map(|input| {
            CpuDecoder::new(input)
                .expect("cpu decoder")
                .decode(PixelFormat::Rgb8)
                .expect("cpu decode")
                .0
        })
        .collect::<Vec<_>>();
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let surfaces = decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
        .expect("decode table-mixed full tiles into reusable output buffer");

    assert_eq!(surfaces.len(), 3);
    for (index, surface) in surfaces.into_iter().enumerate() {
        let surface = surface.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), dimensions);
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected_tiles[index].as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast420_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_full_buffer_groups_resident(JpegSubsampling::Ybr420, (128, 96), 90, 72);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast422_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_full_buffer_groups_resident(JpegSubsampling::Ybr422, (128, 96), 91, 73);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast444_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_full_buffer_groups_resident(JpegSubsampling::Ybr444, (96, 96), 92, 74);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_scaled_batch_decode_can_write_into_reusable_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (4, 4), 2).expect("output buffer");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled decode");

    let surfaces =
        decode_rgb8_scaled_batch_into_metal_buffer_with_session(&inputs, scale, &output, &session)
            .expect("decode scaled into reusable output");

    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (4, 4));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_scaled_batch_resizes_reusable_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled decode");

    let surfaces = decode_rgb8_decoder_scaled_batch_into_resizable_metal_buffer_with_session(
        &decoders,
        scale,
        &mut output,
        &session,
    )
    .expect("decode cached decoder scaled batch into resizable reusable output");

    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (4, 4));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_scaled_batch_can_write_into_fixed_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (4, 4), 2).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled decode");

    let surfaces = decode_rgb8_decoder_scaled_batch_into_metal_buffer_with_session(
        &decoders, scale, &output, &session,
    )
    .expect("decode cached decoder scaled batch into fixed reusable output");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (4, 4));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_region_scaled_batch_decode_can_write_into_reusable_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("output buffer");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
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
        .expect("cpu region scaled decode");

    let surfaces = decode_rgb8_region_scaled_batch_into_metal_buffer_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode region scaled into reusable output");

    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_region_scaled_batch_decode_resizes_reusable_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
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
        .expect("cpu region scaled decode");

    let surfaces = Codec::decode_rgb8_region_scaled_batch_into_resizable_metal_buffer_with_session(
        &inputs,
        roi,
        scale,
        &mut output,
        &session,
    )
    .expect("decode region scaled into resizable reusable output");

    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_region_scaled_batch_resizes_reusable_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
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
        .expect("cpu region scaled decode");

    let surfaces =
        decode_rgb8_decoder_region_scaled_batch_into_resizable_metal_buffer_with_session(
            &decoders,
            roi,
            scale,
            &mut output,
            &session,
        )
        .expect("decode cached decoder batch into resizable reusable output");

    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(surfaces.len(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_region_scaled_batch_can_write_into_fixed_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("output buffer");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
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
        .expect("cpu region scaled decode");

    let surfaces = decode_rgb8_decoder_region_scaled_batch_into_metal_buffer_with_session(
        &decoders, roi, scale, &output, &session,
    )
    .expect("decode cached decoder region-scaled batch into fixed reusable output");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    for (index, result) in surfaces.into_iter().enumerate() {
        let surface = result.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast420_region_scaled_batch_decode_writes_reusable_metal_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (128, 128);
    let roi = Rect {
        x: 9,
        y: 11,
        w: 73,
        h: 67,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart-coded fast420 region-scaled jpeg");
    let packet = build_fast420_packet(&jpeg.data).expect("restart fast420 packet");
    assert_ne!(packet.restart_interval_mcus, 0);
    assert!(!packet.restart_offsets.is_empty());

    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("output buffer");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
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
        .expect("cpu region-scaled decode");

    let surfaces = decode_rgb8_region_scaled_batch_into_metal_buffer_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode restart-coded region-scaled tiles into reusable output buffer");

    assert_eq!(surfaces.len(), 2);
    for (index, surface) in surfaces.into_iter().enumerate() {
        let surface = surface.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
fn assert_restart_region_scaled_buffer_batch_writes_reusable_metal_output(
    subsampling: JpegSubsampling,
    dimensions: (u32, u32),
) {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval: Some(256),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart-coded region-scaled jpeg");
    match subsampling {
        JpegSubsampling::Ybr422 => {
            let packet = build_fast422_packet(&jpeg.data).expect("restart fast422 packet");
            assert_ne!(packet.restart_interval_mcus, 0);
            assert!(!packet.restart_offsets.is_empty());
        }
        JpegSubsampling::Ybr444 => {
            let packet = build_fast444_packet(&jpeg.data).expect("restart fast444 packet");
            assert_ne!(packet.restart_interval_mcus, 0);
            assert!(!packet.restart_offsets.is_empty());
        }
        _ => panic!("restart region-scaled buffer helper expects fast422 or fast444"),
    }

    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("output buffer");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
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
        .expect("cpu region-scaled decode");

    let surfaces = decode_rgb8_region_scaled_batch_into_metal_buffer_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode restart-coded region-scaled tiles into reusable output buffer");

    assert_eq!(surfaces.len(), 2);
    for (index, surface) in surfaces.into_iter().enumerate() {
        let surface = surface.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast422_region_scaled_batch_decode_writes_reusable_metal_output_buffer() {
    assert_restart_region_scaled_buffer_batch_writes_reusable_metal_output(
        JpegSubsampling::Ybr422,
        (128, 96),
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast444_region_scaled_batch_decode_writes_reusable_metal_output_buffer() {
    assert_restart_region_scaled_buffer_batch_writes_reusable_metal_output(
        JpegSubsampling::Ybr444,
        (96, 96),
    );
}

#[cfg(target_os = "macos")]
fn assert_table_mixed_region_scaled_buffer_groups_resident(
    subsampling: JpegSubsampling,
    dimensions: (u32, u32),
    first_quality: u8,
    second_quality: u8,
) {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(37)
            .wrapping_add(19);
        pixel[0] = pixel[0].wrapping_add(delta.rotate_left(1));
        pixel[1] ^= delta;
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(53)
            .wrapping_add(11);
        pixel[0] ^= delta.rotate_right(1);
        pixel[1] = pixel[1].wrapping_sub(delta.rotate_left(2));
        pixel[2] = pixel[2].wrapping_add(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: first_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first table-mixed region-scaled buffer jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: second_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second table-mixed region-scaled buffer jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: first_quality,
            subsampling,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third table-mixed region-scaled buffer jpeg");

    match subsampling {
        JpegSubsampling::Ybr420 => {
            let packet_a = build_fast420_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast420_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast420_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Ybr422 => {
            let packet_a = build_fast422_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast422_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast422_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Ybr444 => {
            let packet_a = build_fast444_packet(&jpeg_a.data).expect("first packet");
            let packet_b = build_fast444_packet(&jpeg_b.data).expect("second packet");
            let packet_c = build_fast444_packet(&jpeg_c.data).expect("third packet");
            assert_eq!(packet_a.y_quant, packet_c.y_quant);
            assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
            assert_eq!(
                packet_a.entropy_checkpoints.len(),
                packet_c.entropy_checkpoints.len()
            );
            assert_ne!(packet_a.y_quant, packet_b.y_quant);
        }
        JpegSubsampling::Gray => panic!("table-mixed buffer helper expects YCbCr sampling"),
    }

    let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 3)
        .expect("output buffer");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let expected_tiles = inputs
        .iter()
        .map(|input| {
            CpuDecoder::new(input)
                .expect("cpu decoder")
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
                .expect("cpu region-scaled decode")
                .0
        })
        .collect::<Vec<_>>();
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let surfaces = decode_rgb8_region_scaled_batch_into_metal_buffer_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode table-mixed region-scaled tiles into reusable output buffer");

    assert_eq!(surfaces.len(), 3);
    for (index, surface) in surfaces.into_iter().enumerate() {
        let surface = surface.expect("surface");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
        assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
        assert_eq!(offset, index * output.tile_stride_bytes());
        assert_eq!(surface.as_bytes(), expected_tiles[index].as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast420_region_scaled_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_region_scaled_buffer_groups_resident(
        JpegSubsampling::Ybr420,
        (128, 96),
        90,
        72,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast422_region_scaled_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_region_scaled_buffer_groups_resident(
        JpegSubsampling::Ybr422,
        (128, 96),
        91,
        73,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast444_region_scaled_buffer_batch_groups_resident_dispatches() {
    assert_table_mixed_region_scaled_buffer_groups_resident(
        JpegSubsampling::Ybr444,
        (96, 96),
        92,
        74,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast444_region_scaled_batch_decode_can_write_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
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
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode region scaled into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_output_buffer_ensure_reuses_matching_allocation_and_grows_capacity() {
    use metal::foreign_types::ForeignTypeRef;

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let original_buffer = output.buffer().as_ptr();

    output
        .ensure_rgb8_tiles(&session, (16, 16), 1)
        .expect("ensure smaller matching output");
    assert_eq!(output.buffer().as_ptr(), original_buffer);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);

    output
        .ensure_rgb8_tiles(&session, (16, 16), 3)
        .expect("ensure larger output");
    assert_ne!(output.buffer().as_ptr(), original_buffer);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 3);
    assert_eq!(
        output.byte_len(),
        16 * 16 * PixelFormat::Rgb8.bytes_per_pixel() * 3
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_texture_output_ensure_reuses_matching_textures_and_grows_capacity() {
    use metal::foreign_types::ForeignTypeRef;

    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2).expect("texture output");
    let original_texture = output.texture(0).expect("texture").as_ptr();

    output
        .ensure_rgba8_tiles(&session, (16, 16), 1)
        .expect("ensure smaller matching texture output");
    assert_eq!(
        output.texture(0).expect("texture").as_ptr(),
        original_texture
    );
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);

    output
        .ensure_rgba8_tiles(&session, (16, 16), 3)
        .expect("ensure larger texture output");
    assert_ne!(
        output.texture(0).expect("texture").as_ptr(),
        original_texture
    );
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 3);
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_output_buffer_ensure_region_scaled_tiles_uses_scaled_roi_shape() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scaled = roi.scaled_covering(Downscale::Quarter);
    let mut output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");

    output
        .ensure_rgb8_region_scaled_tiles(&session, roi, Downscale::Quarter, 2)
        .expect("ensure region-scaled output");

    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(
        output.tile_stride_bytes(),
        scaled.w as usize * scaled.h as usize * PixelFormat::Rgb8.bytes_per_pixel()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_texture_output_ensure_scaled_tiles_uses_scaled_full_shape() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");

    output
        .ensure_rgba8_scaled_tiles(&session, (16, 16), Downscale::Quarter, 2)
        .expect("ensure scaled texture output");

    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_outputs_can_ensure_from_resident_batch_report() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let report = Codec::inspect_rgb8_decoder_batch_metal_output(
        &decoders,
        j2k_jpeg::JpegDecodeOp::Scaled(Downscale::Quarter),
    );
    assert!(report.eligibility.eligible);

    let mut buffer =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let mut textures =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");

    buffer
        .ensure_rgb8_batch_report(&session, &report)
        .expect("ensure buffer from report");
    textures
        .ensure_rgba8_batch_report(&session, &report)
        .expect("ensure textures from report");

    assert_eq!(buffer.dimensions(), (4, 4));
    assert_eq!(buffer.tile_capacity(), 2);
    assert_eq!(
        buffer.tile_stride_bytes(),
        4 * 4 * PixelFormat::Rgb8.bytes_per_pixel()
    );
    assert_eq!(textures.dimensions(), (4, 4));
    assert_eq!(textures.tile_capacity(), 2);
    assert_eq!(textures.pixel_format(), PixelFormat::Rgba8);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_batch_outputs_reject_ineligible_report_without_resizing() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_444).expect("second decoder");
    let decoders = [&first, &second];
    let report =
        Codec::inspect_rgb8_decoder_batch_metal_output(&decoders, j2k_jpeg::JpegDecodeOp::Full);
    assert!(!report.eligibility.eligible);

    let mut buffer =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (1, 1), 1).expect("output buffer");
    let mut textures =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");

    let buffer_err = buffer
        .ensure_rgb8_batch_report(&session, &report)
        .expect_err("ineligible report should reject buffer ensure");
    let texture_err = textures
        .ensure_rgba8_batch_report(&session, &report)
        .expect_err("ineligible report should reject texture ensure");

    assert!(matches!(
        buffer_err,
        Error::UnsupportedMetalRequest { reason }
            if reason.contains("matching output dimensions")
    ));
    assert!(matches!(
        texture_err,
        Error::UnsupportedMetalRequest { reason }
            if reason.contains("matching output dimensions")
    ));
    assert_eq!(buffer.dimensions(), (1, 1));
    assert_eq!(buffer.tile_capacity(), 1);
    assert_eq!(textures.dimensions(), (1, 1));
    assert_eq!(textures.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn warm_session_reuses_private_intermediate_buffers_for_reusable_output_batches() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let inputs = [BASELINE_420, BASELINE_420];

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let first = decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
        .expect("first decode");
    for surface in first {
        assert_eq!(
            surface.expect("surface").residency(),
            SurfaceResidency::MetalResidentDecode
        );
    }
    let allocations_after_first = compute::jpeg_private_buffer_allocations_for_test();

    let second = decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
        .expect("second decode");
    for surface in second {
        assert_eq!(
            surface.expect("surface").residency(),
            SurfaceResidency::MetalResidentDecode
        );
    }

    assert!(
        allocations_after_first > 0,
        "first batch should allocate private intermediate buffers"
    );
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        allocations_after_first,
        "warm session batch should reuse private intermediate buffers"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn warm_session_reuses_shared_upload_buffers_for_reusable_output_batches() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
    let inputs = [BASELINE_420, BASELINE_420];

    compute::reset_jpeg_shared_buffer_allocations_for_test();
    decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
        .expect("first decode");
    let allocations_after_first = compute::jpeg_shared_buffer_allocations_for_test();

    decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
        .expect("second decode");

    assert!(
        allocations_after_first > 0,
        "first batch should allocate shared upload/status buffers"
    );
    assert_eq!(
        compute::jpeg_shared_buffer_allocations_for_test(),
        allocations_after_first,
        "warm session batch should reuse shared upload/status buffers"
    );
}

#[cfg(target_os = "macos")]
fn patterned_index_byte(index: usize) -> u8 {
    u8::try_from(index % 256).expect("modulo 256 fits in u8")
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
    j2k_metal_support::commit_and_wait(command_buffer).expect("texture readback blit");

    // SAFETY: Metal surface byte views are bounded by validated dimensions and formats.
    unsafe { core::slice::from_raw_parts(buffer.contents().cast::<u8>(), byte_len).to_vec() }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast444_batch_decode_can_write_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (8, 8), 2).expect("texture output");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (8, 8));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (8, 8));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_resizes_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        &decoders,
        &mut output,
        &session,
    )
    .expect("decode cached decoder batch into resizable reusable textures");

    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);
    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, (16, 16), &expected_tiles);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_can_write_into_fixed_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles =
        decode_rgb8_decoder_batch_into_metal_textures_with_session(&decoders, &output, &session)
            .expect("decode cached decoder batch into fixed reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.tile_capacity(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (16, 16));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_rejects_mixed_output_dimensions_without_resizing_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_444).expect("second decoder");
    let decoders = [&first, &second];

    let Err(err) = Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        &decoders,
        &mut output,
        &session,
    ) else {
        panic!("mixed output dimensions should be rejected");
    };

    assert!(matches!(err, Error::UnsupportedMetalRequest { .. }));
    assert_eq!(output.dimensions(), (1, 1));
    assert_eq!(output.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_batch_rejects_mixed_sampling_without_resizing_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let rgb = j2k_test_support::patterned_rgb8(16, 16);
    let fast420 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast420 jpeg");
    let fast444 = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: 16,
            height: 16,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode fast444 jpeg");
    let first = Decoder::new(&fast420.data).expect("first decoder");
    let second = Decoder::new(&fast444.data).expect("second decoder");
    let decoders = [&first, &second];

    let Err(err) = Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        &decoders,
        &mut output,
        &session,
    ) else {
        panic!("mixed sampling should be rejected");
    };

    assert!(matches!(
        err,
        Error::UnsupportedMetalRequest { reason }
            if reason.contains("same fast-packet sampling family")
    ));
    assert_eq!(output.dimensions(), (1, 1));
    assert_eq!(output.tile_capacity(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_scaled_batch_decode_can_write_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (4, 4), 2).expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_scaled_batch_into_metal_textures_with_session(
        &inputs, scale, &output, &session,
    )
    .expect("decode scaled into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (4, 4));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_scaled_batch_decode_resizes_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_scaled_batch_into_resizable_metal_textures_with_session(
        &inputs,
        scale,
        &mut output,
        &session,
    )
    .expect("decode scaled into resizable reusable textures");

    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (4, 4));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_scaled_batch_resizes_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_decoder_scaled_batch_into_resizable_metal_textures_with_session(
        &decoders,
        scale,
        &mut output,
        &session,
    )
    .expect("decode cached decoder scaled batch into resizable reusable textures");

    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (4, 4));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_scaled_batch_can_write_into_fixed_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let scale = Downscale::Quarter;
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (4, 4), 2).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode_scaled(PixelFormat::Rgb8, scale)
        .expect("cpu scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_decoder_scaled_batch_into_metal_textures_with_session(
        &decoders, scale, &output, &session,
    )
    .expect("decode cached decoder scaled batch into fixed reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.dimensions(), (4, 4));
    assert_eq!(output.tile_capacity(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (4, 4));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast422_region_scaled_batch_decode_can_write_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 9,
        h: 6,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [BASELINE_422, BASELINE_422];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_422)
        .expect("cpu decoder")
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
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode region scaled into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast422_region_scaled_texture_batch_groups_resident_dispatches() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (128, 96);
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(41)
            .wrapping_add(29);
        pixel[0] ^= delta.rotate_left(1);
        pixel[1] = pixel[1].wrapping_add(delta);
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index).wrapping_mul(59).wrapping_add(3);
        pixel[0] = pixel[0].wrapping_sub(delta.rotate_left(2));
        pixel[1] ^= delta.rotate_right(1);
        pixel[2] = pixel[2].wrapping_add(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast422 region-scaled table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 71,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast422 region-scaled table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast422 region-scaled table group jpeg");
    let packet_a = build_fast422_packet(&jpeg_a.data).expect("first fast422 packet");
    let packet_b = build_fast422_packet(&jpeg_b.data).expect("second fast422 packet");
    let packet_c = build_fast422_packet(&jpeg_c.data).expect("third fast422 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 3)
        .expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
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
        .expect("first cpu region scaled decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
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
        .expect("second cpu region scaled decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
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
        .expect("third cpu region scaled decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let tiles = decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode table-mixed fast422 region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        let actual_rgba = download_rgba8_texture(&session, tile.texture(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast444_region_scaled_texture_batch_groups_resident_dispatches() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (96, 96);
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(61)
            .wrapping_add(13);
        pixel[0] = pixel[0].wrapping_add(delta);
        pixel[1] ^= delta.rotate_left(1);
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(67)
            .wrapping_add(31);
        pixel[0] = pixel[0].wrapping_sub(delta.rotate_left(2));
        pixel[1] = pixel[1].wrapping_add(delta.rotate_right(1));
        pixel[2] ^= delta;
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 91,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast444 region-scaled table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 70,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast444 region-scaled table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 91,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast444 region-scaled table group jpeg");
    let packet_a = build_fast444_packet(&jpeg_a.data).expect("first fast444 packet");
    let packet_b = build_fast444_packet(&jpeg_b.data).expect("second fast444 packet");
    let packet_c = build_fast444_packet(&jpeg_c.data).expect("third fast444 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 3)
        .expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
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
        .expect("first cpu region scaled decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
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
        .expect("second cpu region scaled decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
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
        .expect("third cpu region scaled decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let tiles = decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode table-mixed fast444 region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        let actual_rgba = download_rgba8_texture(&session, tile.texture(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast420_region_scaled_batch_decode_can_write_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
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
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode region scaled into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_region_scaled_batch_resizes_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (1, 1), 1).expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
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
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_decoder_region_scaled_batch_into_resizable_metal_textures_with_session(
        &decoders,
        roi,
        scale,
        &mut output,
        &session,
    )
    .expect("decode cached decoder batch into resizable reusable textures");

    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_decoder_region_scaled_batch_can_write_into_fixed_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 1,
        y: 2,
        w: 10,
        h: 9,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let first = Decoder::new(BASELINE_420).expect("first decoder");
    let second = Decoder::new(BASELINE_420).expect("second decoder");
    let decoders = [&first, &second];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
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
        .expect("cpu region scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_decoder_region_scaled_batch_into_metal_textures_with_session(
        &decoders, roi, scale, &output, &session,
    )
    .expect("decode cached decoder region-scaled batch into fixed reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.dimensions(), (scaled.w, scaled.h));
    assert_eq!(output.tile_capacity(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast420_region_scaled_batch_decode_writes_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (128, 128);
    let roi = Rect {
        x: 9,
        y: 11,
        w: 73,
        h: 67,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart-coded fast420 region-scaled texture jpeg");
    let packet = build_fast420_packet(&jpeg.data).expect("restart fast420 packet");
    assert_ne!(packet.restart_interval_mcus, 0);
    assert!(!packet.restart_offsets.is_empty());

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
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
        .expect("cpu region-scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode restart-coded region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
fn assert_restart_region_scaled_texture_batch_writes_reusable_metal_output(
    subsampling: JpegSubsampling,
    dimensions: (u32, u32),
) {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let roi = Rect {
        x: 0,
        y: 0,
        w: dimensions.0,
        h: dimensions.1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval: Some(256),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart-coded region-scaled texture jpeg");
    match subsampling {
        JpegSubsampling::Ybr422 => {
            let packet = build_fast422_packet(&jpeg.data).expect("restart fast422 packet");
            assert_ne!(packet.restart_interval_mcus, 0);
            assert!(!packet.restart_offsets.is_empty());
        }
        JpegSubsampling::Ybr444 => {
            let packet = build_fast444_packet(&jpeg.data).expect("restart fast444 packet");
            assert_ne!(packet.restart_interval_mcus, 0);
            assert!(!packet.restart_offsets.is_empty());
        }
        _ => panic!("restart region-scaled texture helper expects fast422 or fast444"),
    }

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
        .expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
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
        .expect("cpu region-scaled decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode restart-coded region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast422_region_scaled_batch_decode_writes_reusable_metal_textures() {
    assert_restart_region_scaled_texture_batch_writes_reusable_metal_output(
        JpegSubsampling::Ybr422,
        (128, 96),
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast444_region_scaled_batch_decode_writes_reusable_metal_textures() {
    assert_restart_region_scaled_texture_batch_writes_reusable_metal_output(
        JpegSubsampling::Ybr444,
        (96, 96),
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast420_region_scaled_texture_batch_groups_resident_dispatches() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (128, 128);
    let roi = Rect {
        x: 9,
        y: 11,
        w: 77,
        h: 65,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(43)
            .wrapping_add(19);
        pixel[0] = pixel[0].wrapping_add(delta.rotate_left(1));
        pixel[1] = pixel[1].wrapping_sub(delta);
        pixel[2] ^= delta.rotate_right(2);
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(47)
            .wrapping_add(23);
        pixel[0] ^= delta.rotate_left(2);
        pixel[1] = pixel[1].wrapping_add(delta.rotate_right(1));
        pixel[2] = pixel[2].wrapping_sub(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast420 region-scaled table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 72,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast420 region-scaled table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast420 region-scaled table group jpeg");
    let packet_a = build_fast420_packet(&jpeg_a.data).expect("first fast420 packet");
    let packet_b = build_fast420_packet(&jpeg_b.data).expect("second fast420 packet");
    let packet_c = build_fast420_packet(&jpeg_c.data).expect("third fast420 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 3)
        .expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
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
        .expect("first cpu region scaled decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
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
        .expect("second cpu region scaled decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
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
        .expect("third cpu region scaled decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    let tiles = decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
        &inputs, roi, scale, &output, &session,
    )
    .expect("decode table-mixed fast420 region-scaled tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        let actual_rgba = download_rgba8_texture(&session, tile.texture(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast420_batch_decode_can_write_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2).expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (16, 16));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (16, 16));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast422_batch_decode_can_write_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 8), 2).expect("texture output");
    let inputs = [BASELINE_422, BASELINE_422];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_422)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    assert_eq!(tiles.len(), 2);
    assert_eq!(output.tile_capacity(), 2);
    assert_eq!(output.dimensions(), (16, 8));
    assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (16, 8));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_texture_batch_decode_avoids_private_rgba_staging_buffers() {
    let cases = [
        (BASELINE_420, (16, 16), 0),
        (BASELINE_422, (16, 8), 0),
        (BASELINE_444, (8, 8), 0),
    ];

    for (input, dimensions, expected_private_allocations) in cases {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2)
            .expect("texture output");
        let inputs = [input, input];

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
            .expect("decode into reusable textures");
        assert_eq!(tiles.len(), 2);
        for tile in tiles {
            assert_eq!(
                tile.expect("texture tile").pixel_format(),
                PixelFormat::Rgba8
            );
        }

        assert_eq!(
                compute::jpeg_private_buffer_allocations_for_test(),
                expected_private_allocations,
                "texture batch decode should not allocate a private RGBA staging buffer for {dimensions:?}"
            );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast444_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (8, 8), 2).expect("texture output");
    let inputs = [BASELINE_444, BASELINE_444];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_444)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, (8, 8), &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        0,
        "fused 4:4:4 texture batch decode should not allocate private Y/Cb/Cr staging planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast444_texture_batch_groups_resident_dispatches() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (64, 64);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index).wrapping_mul(31).wrapping_add(5);
        pixel[0] = pixel[0].wrapping_sub(delta);
        pixel[1] = pixel[1].wrapping_add(delta.rotate_left(1));
        pixel[2] ^= delta.rotate_right(2);
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(37)
            .wrapping_add(17);
        pixel[0] ^= delta.rotate_left(3);
        pixel[1] = pixel[1].wrapping_sub(delta.rotate_right(1));
        pixel[2] = pixel[2].wrapping_add(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast444 table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 71,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast444 table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast444 table group jpeg");
    let packet_a = build_fast444_packet(&jpeg_a.data).expect("first fast444 packet");
    let packet_b = build_fast444_packet(&jpeg_b.data).expect("second fast444 packet");
    let packet_c = build_fast444_packet(&jpeg_c.data).expect("third fast444 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 3).expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("first cpu decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("second cpu decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("third cpu decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode table-mixed fast444 tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        let actual_rgba = download_rgba8_texture(&session, tile.texture(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
    assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "table-mixed resident 4:4:4 texture dispatches should not allocate private Y/Cb/Cr staging planes"
        );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast422_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 8), 2).expect("texture output");
    let inputs = [BASELINE_422, BASELINE_422];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_422)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, (16, 8), &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        0,
        "fused 4:2:2 texture batch decode should not allocate private Y/Cb/Cr staging planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_wide_fast422_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (48, 16);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode 4:2:2 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        0,
        "wide fused 4:2:2 texture batch decode should not allocate private Y/Cb/Cr staging planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_fast422_texture_batch_groups_resident_dispatches() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (96, 48);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(23)
            .wrapping_add(11);
        pixel[0] = pixel[0].wrapping_add(delta.rotate_left(1));
        pixel[1] ^= delta;
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(19)
            .wrapping_add(53);
        pixel[0] ^= delta.rotate_left(2);
        pixel[1] = pixel[1].wrapping_sub(delta);
        pixel[2] = pixel[2].wrapping_add(delta.rotate_right(1));
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 91,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first fast422 table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 73,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second fast422 table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 91,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third fast422 table group jpeg");
    let packet_a = build_fast422_packet(&jpeg_a.data).expect("first fast422 packet");
    let packet_b = build_fast422_packet(&jpeg_b.data).expect("second fast422 packet");
    let packet_c = build_fast422_packet(&jpeg_c.data).expect("third fast422 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 3).expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("first cpu decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("second cpu decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("third cpu decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode table-mixed fast422 tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        let actual_rgba = download_rgba8_texture(&session, tile.texture(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
    assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "table-mixed resident 4:2:2 texture dispatches should not allocate private Y/Cb/Cr staging planes"
        );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_fast420_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2).expect("texture output");
    let inputs = [BASELINE_420, BASELINE_420];
    let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), (16, 16));
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        assert_eq!(
            download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
            expected_rgba
        );
    }
    assert_eq!(
        compute::jpeg_private_buffer_allocations_for_test(),
        0,
        "fused 4:2:0 texture batch decode should not allocate private Y/Cb/Cr staging planes"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_wide_row_fast420_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (32, 16);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode 4:2:0 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "wide-row fused 4:2:0 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_multi_row_fast420_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (16, 32);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 92,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode 4:2:0 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "multi-row fused 4:2:0 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_multi_axis_fast420_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    for dimensions in [(32, 32), (48, 48)] {
        let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
        let jpeg = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: dimensions.0,
                height: dimensions.1,
            },
            JpegEncodeOptions {
                quality: 92,
                subsampling: JpegSubsampling::Ybr420,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            },
        )
        .expect("encode 4:2:0 source jpeg");
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2)
            .expect("texture output");
        let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
        let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
            .expect("decode into reusable textures");

        let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
        assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
        assert_eq!(
                compute::jpeg_private_buffer_allocations_for_test(),
                0,
                "multi-axis fused 4:2:0 texture batch decode should not allocate private Y/Cb/Cr staging planes for {dimensions:?}"
            );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_chunked_multi_axis_fast420_texture_batch_decode_fuses_directly_into_reusable_metal_textures(
) {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (736, 720);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode chunked 4:2:0 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "chunked multi-axis fused 4:2:0 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_restart_fast420_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (48, 48);
    let rgb = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let jpeg = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(2),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode restart 4:2:0 source jpeg");
    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
    let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
        .expect("cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("cpu decode");
    let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode into reusable textures");

    let expected_tiles = [expected_rgba.as_slice(), expected_rgba.as_slice()];
    assert_reusable_rgba_texture_tiles(&session, &output, tiles, dimensions, &expected_tiles);
    assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "restart fused 4:2:0 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_distinct_restart_fast420_texture_batch_decode_fuses_directly_into_reusable_metal_textures()
{
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (128, 128);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(17)
            .wrapping_add(31);
        pixel[0] = pixel[0].wrapping_add(delta);
        pixel[1] = pixel[1].wrapping_sub(delta.rotate_left(1));
        pixel[2] ^= delta.rotate_right(1);
    }
    assert_ne!(rgb_a, rgb_b);

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first restart 4:2:0 source jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second restart 4:2:0 source jpeg");
    assert_ne!(jpeg_a.data, jpeg_b.data);

    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2).expect("texture output");
    let inputs = [jpeg_a.data.as_slice(), jpeg_b.data.as_slice()];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("first cpu decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("second cpu decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode distinct restart tiles into reusable textures");

    assert_eq!(tiles.len(), 2);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        let actual_rgba = download_rgba8_texture(&session, tile.texture(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
    assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "distinct restart fused 4:2:0 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
}

#[cfg(target_os = "macos")]
#[test]
fn rgb8_table_mixed_restart_fast420_texture_batch_groups_resident_dispatches() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let dimensions = (128, 128);
    let rgb_a = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_b = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    let mut rgb_c = j2k_test_support::patterned_rgb8(dimensions.0, dimensions.1);
    for (index, pixel) in rgb_b.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index).wrapping_mul(29).wrapping_add(7);
        pixel[0] ^= delta;
        pixel[1] = pixel[1].wrapping_add(delta.rotate_left(2));
        pixel[2] = pixel[2].wrapping_sub(delta.rotate_right(2));
    }
    for (index, pixel) in rgb_c.chunks_exact_mut(3).enumerate() {
        let delta = patterned_index_byte(index)
            .wrapping_mul(13)
            .wrapping_add(41);
        pixel[0] = pixel[0].wrapping_sub(delta.rotate_left(1));
        pixel[1] ^= delta.rotate_right(3);
        pixel[2] = pixel[2].wrapping_add(delta);
    }

    let jpeg_a = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_a,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode first table group jpeg");
    let jpeg_b = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_b,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 74,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode second table group jpeg");
    let jpeg_c = encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb_c,
            width: dimensions.0,
            height: dimensions.1,
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(4),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode third table group jpeg");
    let packet_a = build_fast420_packet(&jpeg_a.data).expect("first fast420 packet");
    let packet_b = build_fast420_packet(&jpeg_b.data).expect("second fast420 packet");
    let packet_c = build_fast420_packet(&jpeg_c.data).expect("third fast420 packet");
    assert_eq!(packet_a.y_quant, packet_c.y_quant);
    assert_eq!(packet_a.cb_quant, packet_c.cb_quant);
    assert_eq!(packet_a.cr_quant, packet_c.cr_quant);
    assert_eq!(packet_a.y_dc_table, packet_c.y_dc_table);
    assert_eq!(packet_a.y_ac_table, packet_c.y_ac_table);
    assert_eq!(
        packet_a.entropy_checkpoints.len(),
        packet_c.entropy_checkpoints.len()
    );
    assert_ne!(packet_a.y_quant, packet_b.y_quant);

    let output =
        MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 3).expect("texture output");
    let inputs = [
        jpeg_a.data.as_slice(),
        jpeg_b.data.as_slice(),
        jpeg_c.data.as_slice(),
    ];
    let (expected_rgb_a, _) = CpuDecoder::new(&jpeg_a.data)
        .expect("first cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("first cpu decode");
    let (expected_rgb_b, _) = CpuDecoder::new(&jpeg_b.data)
        .expect("second cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("second cpu decode");
    let (expected_rgb_c, _) = CpuDecoder::new(&jpeg_c.data)
        .expect("third cpu decoder")
        .decode(PixelFormat::Rgb8)
        .expect("third cpu decode");
    let expected_tiles = [
        rgb_to_rgba_opaque(&expected_rgb_a),
        rgb_to_rgba_opaque(&expected_rgb_b),
        rgb_to_rgba_opaque(&expected_rgb_c),
    ];
    assert_ne!(expected_tiles[0], expected_tiles[1]);
    assert_ne!(expected_tiles[0], expected_tiles[2]);
    assert_ne!(expected_tiles[1], expected_tiles[2]);

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let tiles = decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
        .expect("decode table-mixed restart tiles into reusable textures");

    assert_eq!(tiles.len(), 3);
    for (index, tile) in tiles.into_iter().enumerate() {
        let tile = tile.expect("texture tile");
        assert_eq!(tile.dimensions(), dimensions);
        assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
        assert!(std::ptr::eq(
            tile.texture(),
            output.texture(index).expect("output texture")
        ));
        let actual_rgba = download_rgba8_texture(&session, tile.texture(), tile.dimensions());
        assert_eq!(actual_rgba.as_slice(), expected_tiles[index].as_slice());
    }
    assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "table-mixed resident 4:2:0 texture dispatches should not allocate private Y/Cb/Cr staging planes"
        );
}

#[cfg(target_os = "macos")]
#[test]
fn jpeg_device_decode_uses_private_internal_planes() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let surface = decoder
        .decode_to_device_with_session(PixelFormat::Rgb8, &session)
        .expect("resident JPEG Metal decode");
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert!(
        compute::jpeg_private_buffer_allocations_for_test() > 0,
        "resident JPEG Metal decode should use Private internal planes"
    );
    let _ = surface.as_bytes();
}

#[cfg(target_os = "macos")]
#[test]
fn jpeg_private_rgb8_tile_uses_private_output_buffer() {
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let tile = decoder
        .decode_private_rgb8_tile_with_session(&session)
        .expect("resident private JPEG Metal decode");

    assert_eq!(tile.dimensions, (16, 16));
    assert_eq!(tile.pixel_format, PixelFormat::Rgb8);
    assert_eq!(tile.pitch_bytes, 16 * PixelFormat::Rgb8.bytes_per_pixel());
    assert_eq!(tile.byte_offset, 0);
    assert_eq!(tile.buffer.storage_mode(), metal::MTLStorageMode::Private);
    assert!(tile.status_buffer.length() > 0);
}

#[cfg(target_os = "macos")]
#[test]
fn jpeg_gray_region_decode_uses_private_internal_planes() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let mut expected_decoder = Decoder::new(BASELINE_420).expect("expected decoder");
    let mut expected = vec![0; roi.w as usize * roi.h as usize];
    expected_decoder
        .decode_region_into(
            &mut CpuScratchPool::new(),
            &mut expected,
            roi.w as usize,
            PixelFormat::Gray8,
            roi,
        )
        .expect("expected CPU region decode");

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    compute::reset_jpeg_private_buffer_allocations_for_test();
    let surface = decoder
        .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Metal)
        .expect("resident JPEG Metal region decode");
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert!(
        compute::jpeg_private_buffer_allocations_for_test() >= 3,
        "resident Gray8 region decode should keep decoded Y/Cb/Cr planes Private"
    );
    assert_eq!(surface.as_bytes(), expected.as_slice());
}

#[cfg(target_os = "macos")]
#[test]
fn uploaded_metal_surface_is_marked_cpu_staged() {
    let surface = upload_surface(
        vec![1, 2, 3],
        (1, 1),
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("CPU staged Metal upload");

    assert_eq!(surface.residency(), SurfaceResidency::CpuStagedMetalUpload);
}

#[test]
fn auto_route_prefers_cpu_host_for_region_scaled_even_with_restart_packets() {
    let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
    let packet = build_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

    assert_eq!(
        choose_route(
            &decoder,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::RegionScaled {
                roi: Rect {
                    x: 0,
                    y: 0,
                    w: 16,
                    h: 16,
                },
                scale: Downscale::Quarter,
            },
            None,
            None,
            Some(&packet),
        ),
        routing::RouteDecision::CpuHost
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn session_decode_rejects_unsupported_shape_before_host_unavailability() {
    let mut decoder = Decoder::new(GRAYSCALE).expect("decoder");
    let session = MetalBackendSession::default();

    assert!(matches!(
        decoder.decode_to_device_with_session(PixelFormat::Gray8, &session),
        Err(Error::UnsupportedMetalRequest { .. })
    ));
}
