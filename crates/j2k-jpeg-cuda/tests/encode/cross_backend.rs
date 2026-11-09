// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CUDA baseline encoder produces the CPU encoder's bytes, and both
//! produce the cross-backend golden of `j2k_test_support`.

use j2k_core::PixelFormat;
use j2k_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use j2k_jpeg_cuda::{
    encode_jpeg_baseline_batch_from_cuda_buffers, CudaSession, JpegBaselineCudaEncodeTile,
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

/// The CUDA frame of `case`'s input `data`, coded as a frame of `output`
/// size.
fn cuda_frame(
    case: JpegEncodeMatrixCase,
    data: &[u8],
    output: (u32, u32),
    context: &j2k_cuda_runtime::CudaContext,
    session: &mut CudaSession,
) -> Vec<u8> {
    let format = if case.sampling == JpegEncodeMatrixSampling::Gray {
        PixelFormat::Gray8
    } else {
        PixelFormat::Rgb8
    };
    let buffer = context.upload(data).expect("upload matrix samples");
    let tile = JpegBaselineCudaEncodeTile {
        buffer: &buffer,
        byte_offset: 0,
        width: case.width,
        height: case.height,
        pitch_bytes: case.width as usize * case.channels(),
        output_width: output.0,
        output_height: output.1,
        format,
    };
    let mut frames = encode_jpeg_baseline_batch_from_cuda_buffers(
        &[tile],
        options(case, JpegBackend::Cuda),
        session,
    )
    .unwrap_or_else(|error| panic!("CUDA {}: {error}", case.label()));
    assert_eq!(frames.len(), 1, "{}: one frame per tile", case.label());
    frames.remove(0).data
}

#[test]
fn cuda_encoder_matches_the_cpu_encoder_and_the_cross_backend_golden() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = j2k_cuda_runtime::CudaContext::system_default().expect("CUDA context");
    let mut session = CudaSession::default();
    let frames = jpeg_encode_matrix_cases()
        .into_iter()
        .map(|case| {
            let data = case.samples();
            let cuda = cuda_frame(
                case,
                &data,
                (case.width, case.height),
                &context,
                &mut session,
            );
            assert!(
                cuda == cpu_frame(case, &data),
                "{}: CUDA and CPU bytes differ",
                case.label()
            );
            cuda
        })
        .collect::<Vec<_>>();
    assert_eq!(
        jpeg_encode_matrix_digest(frames.iter().map(Vec::as_slice)),
        JPEG_ENCODE_MATRIX_SHA256,
        "CUDA baseline JPEG bytes changed"
    );
}

/// The transcode tests' synthetic inputs, which the CPU once encoded
/// differently on aarch64 and x86-64, encode to the CPU's bytes.
#[test]
fn cuda_encodes_the_transcode_fixtures_as_the_cpu_does() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = j2k_cuda_runtime::CudaContext::system_default().expect("CUDA context");
    let mut session = CudaSession::default();
    let frames = jpeg_encode_transcode_fixtures()
        .into_iter()
        .map(|fixture| {
            let case = fixture.case;
            let size = (case.width, case.height);
            let cuda = cuda_frame(case, &fixture.samples, size, &context, &mut session);
            assert!(
                cuda == cpu_frame(case, &fixture.samples),
                "{}: CUDA and CPU bytes differ",
                fixture.name
            );
            cuda
        })
        .collect::<Vec<_>>();
    assert_eq!(
        jpeg_encode_matrix_digest(frames.iter().map(Vec::as_slice)),
        JPEG_ENCODE_TRANSCODE_FIXTURES_SHA256,
        "CUDA transcode fixture bytes changed"
    );
}

/// A frame beyond its input codes the input's last column and row, as the
/// CPU encoder codes the samples beyond its image.
#[test]
fn cuda_padded_frames_match_the_cpu_encoding_of_the_edge_replicated_image() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = j2k_cuda_runtime::CudaContext::system_default().expect("CUDA context");
    let mut session = CudaSession::default();
    for (case, frame) in jpeg_encode_padded_cases() {
        let data = case.samples();
        let cuda = cuda_frame(case, &data, frame, &context, &mut session);
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
            cuda == cpu,
            "{} in a {frame:?} frame: CUDA and CPU bytes differ",
            case.label()
        );
    }
}
