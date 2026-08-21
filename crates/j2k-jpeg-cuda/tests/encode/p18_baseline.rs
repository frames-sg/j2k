// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact, independently decoded matrix for promoted staged CUDA baseline encode.

use std::io::Cursor;

use j2k_core::PixelFormat;
use j2k_jpeg::{DecodeRequest, Decoder, JpegBackend, JpegEncodeOptions, JpegSubsampling};
use j2k_jpeg_cuda::{
    encode_jpeg_baseline_batch_from_cuda_buffers, CudaSession, JpegBaselineCudaEncodeTile,
};
use sha2::{Digest, Sha256};

const P18_RESTART_INPUT_HASH_DOMAIN: &[u8] = b"P18-CUDA-JPEG-RESTART16-INPUTS\0";
const P18_RESTART_OUTPUT_HASH_DOMAIN: &[u8] = b"P18-CUDA-JPEG-RESTART16-OUTPUTS\0";
const P18_EXACT_MATRIX_INPUT_HASH_DOMAIN: &[u8] = b"P18-CUDA-STAGED-JPEG-EXACT-INPUTS\0";
const P18_EXACT_MATRIX_HASH_DOMAIN: &[u8] = b"P18-CUDA-STAGED-JPEG-EXACT-MATRIX\0";
const P18_EXACT_MATRIX_INPUT_SHA256: &str =
    "5fbd44a6890bfe562d66709eda023f0b5b8f942f0e113824399cfd39f06fe570";
const P18_EXACT_MATRIX_OUTPUT_SHA256: &str =
    "99b76d5a103ed958e4a4cdef80fb8e48cd8f2c6e28ababbf4ff787fea67ab314";

#[test]
fn rtx_cuda_promoted_staged_baseline_encode_is_exact_across_matrix_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = j2k_cuda_runtime::CudaContext::system_default().expect("CUDA context");
    let mut input_hasher = Sha256::new();
    input_hasher.update(P18_EXACT_MATRIX_INPUT_HASH_DOMAIN);
    let mut frames = assert_rgb_matrix(&context, &mut input_hasher);
    frames.extend(assert_gray_matrix(&context, &mut input_hasher));
    assert_eq!(frames.len(), 16, "P18 staged matrix remains complete");
    let input_sha256 = format!("{:x}", input_hasher.finalize());
    let output_sha256 = framed_sha256(
        P18_EXACT_MATRIX_HASH_DOMAIN,
        frames.iter().map(Vec::as_slice),
    );
    eprintln!(
        "p18_cuda_staged_exact_matrix frames=16 input_sha256={input_sha256} output_sha256={output_sha256} deterministic=true repository_decode=true independent_decode=true"
    );
    assert_eq!(
        input_sha256, P18_EXACT_MATRIX_INPUT_SHA256,
        "P18 promoted staged exact-matrix input digest changed"
    );
    assert_eq!(
        output_sha256, P18_EXACT_MATRIX_OUTPUT_SHA256,
        "P18 promoted staged exact-matrix digest changed"
    );
}

#[test]
fn cuda_staged_rgb422_restart16_is_complete_for_batch1_and_batch8_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = j2k_cuda_runtime::CudaContext::system_default().expect("CUDA context");
    for batch_size in [1usize, 8] {
        assert_large_restart16_batch(&context, batch_size);
    }
}

fn assert_large_restart16_batch(context: &j2k_cuda_runtime::CudaContext, batch_size: usize) {
    const DIMENSION: u32 = 512;
    const RESTART_INTERVAL: u16 = 16;
    const EXPECTED_RESTART_MARKERS: usize = 127;
    const PLANNED_ENTROPY_CAPACITY: usize = 4_194_574;

    let tile_bytes = DIMENSION as usize * DIMENSION as usize * 3;
    let pixels = j2k_test_support::patterned_rgb8_tiles(DIMENSION, DIMENSION, batch_size);
    let input_sha256 = framed_sha256(
        P18_RESTART_INPUT_HASH_DOMAIN,
        pixels.chunks_exact(tile_bytes),
    );
    let buffer = context
        .upload(&pixels)
        .expect("upload P18 large restart input");
    let tiles = (0..batch_size)
        .map(|index| JpegBaselineCudaEncodeTile {
            buffer: &buffer,
            byte_offset: index * tile_bytes,
            width: DIMENSION,
            height: DIMENSION,
            pitch_bytes: DIMENSION as usize * 3,
            output_width: DIMENSION,
            output_height: DIMENSION,
            format: PixelFormat::Rgb8,
        })
        .collect::<Vec<_>>();
    let options = JpegEncodeOptions {
        quality: 90,
        subsampling: JpegSubsampling::Ybr422,
        restart_interval: Some(RESTART_INTERVAL),
        backend: JpegBackend::Cuda,
    };
    let mut session = CudaSession::default();
    let frames = encode_jpeg_baseline_batch_from_cuda_buffers(&tiles, options, &mut session)
        .expect("P18 large restart serial encode");
    let repeat = encode_jpeg_baseline_batch_from_cuda_buffers(&tiles, options, &mut session)
        .expect("repeat P18 large restart serial encode");
    assert_eq!(
        frames, repeat,
        "P18 large restart serial output must be exact for batch {batch_size}"
    );
    let output_sha256 = framed_sha256(
        P18_RESTART_OUTPUT_HASH_DOMAIN,
        frames.iter().map(|frame| frame.data.as_slice()),
    );
    eprintln!(
        "p18_cuda_restart16_probe dimensions=512x512 sampling=4:2:2 quality=90 restart_interval=16 batch={batch_size} input_sha256={input_sha256} output_sha256={output_sha256} deterministic=true expected_restart_markers={EXPECTED_RESTART_MARKERS} planned_entropy_capacity={PLANNED_ENTROPY_CAPACITY}"
    );

    for (index, frame) in frames.iter().enumerate() {
        let entropy_len = assert_exact_restart_scan(&frame.data, EXPECTED_RESTART_MARKERS);
        assert_independent_decoder_accepts_rgb8(&frame.data, DIMENSION, DIMENSION);
        assert_repository_decoder_accepts_rgb8(&frame.data, DIMENSION, DIMENSION);
        eprintln!(
            "p18_cuda_restart16_frame batch={batch_size} index={index} entropy_len={entropy_len} frame_len={} frame_capacity={}",
            frame.data.len(),
            frame.data.capacity(),
        );
    }
}

fn assert_exact_restart_scan(encoded: &[u8], expected_restart_markers: usize) -> usize {
    assert!(encoded.ends_with(&[0xff, 0xd9]), "JPEG must end with EOI");
    let sos = encoded
        .windows(2)
        .position(|bytes| bytes == [0xff, 0xda])
        .expect("JPEG contains SOS");
    let header_len = usize::from(u16::from_be_bytes([encoded[sos + 2], encoded[sos + 3]]));
    let entropy_start = sos + 2 + header_len;
    let entropy_end = encoded.len() - 2;
    let entropy = &encoded[entropy_start..entropy_end];
    let mut restart_count = 0usize;
    let mut index = 0usize;
    while index < entropy.len() {
        if entropy[index] != 0xff {
            index += 1;
            continue;
        }
        let marker = *entropy
            .get(index + 1)
            .expect("entropy cannot end with a bare 0xff byte");
        match marker {
            0x00 => {}
            0xd0..=0xd7 => {
                let expected = 0xd0 | u8::try_from(restart_count & 0x07).expect("RST fits u8");
                assert_eq!(
                    marker, expected,
                    "restart marker {restart_count} has the wrong sequence value"
                );
                restart_count += 1;
            }
            _ => panic!(
                "entropy 0xff byte must be stuffed or introduce RST, got 0x{marker:02x} at entropy offset {index}"
            ),
        }
        index += 2;
    }
    assert_eq!(
        restart_count, expected_restart_markers,
        "restart-coded entropy has the wrong marker count"
    );
    entropy.len()
}

fn assert_independent_decoder_accepts_rgb8(encoded: &[u8], width: u32, height: u32) {
    let mut independent = jpeg_decoder::Decoder::new(Cursor::new(encoded));
    let pixels = independent
        .decode()
        .expect("independent jpeg-decoder accepts restart-coded CUDA JPEG");
    let info = independent.info().expect("independent decoder frame info");
    assert_eq!(
        (u32::from(info.width), u32::from(info.height)),
        (width, height)
    );
    assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::RGB24);
    assert_eq!(pixels.len(), width as usize * height as usize * 3);
}

fn assert_repository_decoder_accepts_rgb8(encoded: &[u8], width: u32, height: u32) {
    let decoder = Decoder::new(encoded).expect("repository parser accepts restart-coded CUDA JPEG");
    let (pixels, outcome) = decoder
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("repository decoder accepts restart-coded CUDA JPEG");
    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    assert_eq!(pixels.len(), width as usize * height as usize * 3);
}

fn framed_sha256<'a>(domain: &[u8], frames: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    update_framed_sha256(&mut hasher, frames);
    format!("{:x}", hasher.finalize())
}

fn update_framed_sha256<'a>(hasher: &mut Sha256, frames: impl IntoIterator<Item = &'a [u8]>) {
    for frame in frames {
        hasher.update((frame.len() as u64).to_le_bytes());
        hasher.update(frame);
    }
}

fn assert_rgb_matrix(
    context: &j2k_cuda_runtime::CudaContext,
    input_hasher: &mut Sha256,
) -> Vec<Vec<u8>> {
    let (width, height) = (23u32, 19u32);
    let batch_size = 2usize;
    let tile_bytes = width as usize * height as usize * 3;
    let pixels = j2k_test_support::patterned_rgb8_tiles(width, height, batch_size);
    let buffer = context.upload(&pixels).expect("upload P18 RGB matrix");
    let tiles = (0..batch_size)
        .map(|index| JpegBaselineCudaEncodeTile {
            buffer: &buffer,
            byte_offset: index * tile_bytes,
            width,
            height,
            pitch_bytes: width as usize * 3,
            output_width: width,
            output_height: height,
            format: PixelFormat::Rgb8,
        })
        .collect::<Vec<_>>();

    let mut exact_frames = Vec::new();
    for (subsampling, restart_interval, quality) in [
        (JpegSubsampling::Ybr444, None, 1u8),
        (JpegSubsampling::Ybr444, Some(5), 90),
        (JpegSubsampling::Ybr422, None, 90),
        (JpegSubsampling::Ybr422, Some(5), 100),
        (JpegSubsampling::Ybr420, None, 100),
        (JpegSubsampling::Ybr420, Some(5), 1),
    ] {
        update_framed_sha256(input_hasher, pixels.chunks_exact(tile_bytes));
        let options = JpegEncodeOptions {
            quality,
            subsampling,
            restart_interval,
            backend: JpegBackend::Cuda,
        };
        let mut session = CudaSession::default();
        let frames = encode_jpeg_baseline_batch_from_cuda_buffers(&tiles, options, &mut session)
            .expect("P18 RGB serial matrix encode");
        let repeat = encode_jpeg_baseline_batch_from_cuda_buffers(&tiles, options, &mut session)
            .expect("repeat P18 RGB serial matrix encode");
        assert_eq!(
            frames, repeat,
            "serial RGB route must be exact for {subsampling:?}, restart={restart_interval:?}, quality={quality}"
        );
        for frame in frames {
            assert_entropy_byte_stuffing_and_restart_markers(&frame.data);
            assert_decoders_accept(
                &frame.data,
                width,
                height,
                PixelFormat::Rgb8,
                jpeg_decoder::PixelFormat::RGB24,
            );
            exact_frames.push(frame.data);
        }
    }
    exact_frames
}

fn assert_gray_matrix(
    context: &j2k_cuda_runtime::CudaContext,
    input_hasher: &mut Sha256,
) -> Vec<Vec<u8>> {
    let (width, height) = (7u32, 5u32);
    let (output_width, output_height) = (13u32, 11u32);
    let batch_size = 2usize;
    let tile_bytes = width as usize * height as usize;
    let base = j2k_test_support::patterned_gray8(width, height);
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(tile_bytes * batch_size)
        .expect("allocate P18 grayscale matrix");
    for salt in 0..batch_size {
        let salt = u8::try_from(salt).expect("grayscale salt fits u8");
        pixels.extend(base.iter().map(|sample| sample.wrapping_add(salt)));
    }
    let buffer = context.upload(&pixels).expect("upload P18 gray matrix");
    let tiles = (0..batch_size)
        .map(|index| JpegBaselineCudaEncodeTile {
            buffer: &buffer,
            byte_offset: index * tile_bytes,
            width,
            height,
            pitch_bytes: width as usize,
            output_width,
            output_height,
            format: PixelFormat::Gray8,
        })
        .collect::<Vec<_>>();

    let mut exact_frames = Vec::new();
    for (quality, restart_interval) in [(1u8, None), (100, Some(3))] {
        update_framed_sha256(input_hasher, pixels.chunks_exact(tile_bytes));
        let options = JpegEncodeOptions {
            quality,
            subsampling: JpegSubsampling::Gray,
            restart_interval,
            backend: JpegBackend::Cuda,
        };
        let mut session = CudaSession::default();
        let frames = encode_jpeg_baseline_batch_from_cuda_buffers(&tiles, options, &mut session)
            .expect("P18 grayscale serial matrix encode");
        let repeat = encode_jpeg_baseline_batch_from_cuda_buffers(&tiles, options, &mut session)
            .expect("repeat P18 grayscale serial matrix encode");
        assert_eq!(
            frames, repeat,
            "serial grayscale route must be exact for restart={restart_interval:?}, quality={quality}"
        );
        for frame in frames {
            assert_entropy_byte_stuffing_and_restart_markers(&frame.data);
            assert_decoders_accept(
                &frame.data,
                output_width,
                output_height,
                PixelFormat::Gray8,
                jpeg_decoder::PixelFormat::L8,
            );
            exact_frames.push(frame.data);
        }
    }
    exact_frames
}

fn assert_decoders_accept(
    encoded: &[u8],
    width: u32,
    height: u32,
    output_format: PixelFormat,
    independent_format: jpeg_decoder::PixelFormat,
) {
    let decoder = Decoder::new(encoded).expect("repository parser accepts CUDA JPEG");
    let (pixels, outcome) = decoder
        .decode_request(DecodeRequest::full(output_format))
        .expect("repository decoder accepts CUDA JPEG");
    assert_eq!((outcome.decoded.w, outcome.decoded.h), (width, height));
    let components = if output_format == PixelFormat::Gray8 {
        1
    } else {
        3
    };
    assert_eq!(pixels.len(), width as usize * height as usize * components);

    let mut independent = jpeg_decoder::Decoder::new(Cursor::new(encoded));
    let pixels = independent
        .decode()
        .expect("independent jpeg-decoder accepts CUDA JPEG");
    let info = independent.info().expect("independent decoder frame info");
    assert_eq!(
        (u32::from(info.width), u32::from(info.height)),
        (width, height)
    );
    assert_eq!(info.pixel_format, independent_format);
    assert_eq!(pixels.len(), width as usize * height as usize * components);
}

fn assert_entropy_byte_stuffing_and_restart_markers(encoded: &[u8]) {
    let sos = encoded
        .windows(2)
        .position(|bytes| bytes == [0xff, 0xda])
        .expect("JPEG contains SOS");
    let header_len = usize::from(u16::from_be_bytes([encoded[sos + 2], encoded[sos + 3]]));
    let mut index = sos + 2 + header_len;
    while index + 1 < encoded.len() {
        if encoded[index] != 0xff {
            index += 1;
            continue;
        }
        let marker = encoded[index + 1];
        assert!(
            marker == 0x00 || (0xd0..=0xd7).contains(&marker) || marker == 0xd9,
            "entropy 0xff byte must be stuffed or introduce restart/EOI marker, got 0x{marker:02x}"
        );
        if marker == 0xd9 {
            return;
        }
        index += 2;
    }
    panic!("JPEG entropy scan did not reach EOI");
}
