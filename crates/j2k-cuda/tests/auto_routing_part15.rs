// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cuda-runtime")]

use j2k_core::{
    BackendKind, BackendRequest, CompressedPayloadKind, CompressedTransferSyntax, DeviceSurface,
    Downscale, ImageDecodeDevice, PixelFormat, Rect, TileBatchDecodeManyDevice,
};
use j2k_cuda::{Codec, J2kContext, J2kDecoder, J2kScratchPool, Surface, SurfaceResidency};
use j2k_test_support::{
    cuda_runtime_and_strict_oxide_gate, htj2k_rgb8_97_fixture, htj2k_rgb8_pattern_fixture,
};

fn assert_cuda_matches_cpu(cuda: &Surface, cpu: &Surface) {
    assert_eq!(cuda.backend_kind(), BackendKind::Cuda);
    assert_eq!(cuda.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(cuda.dimensions(), cpu.dimensions());
    assert_eq!(cuda.pixel_format(), cpu.pixel_format());

    let stride = cuda.dimensions().0 as usize * cuda.pixel_format().bytes_per_pixel();
    let mut actual = vec![0; stride * cuda.dimensions().1 as usize];
    let mut expected = vec![0; stride * cpu.dimensions().1 as usize];
    cuda.download_into(&mut actual, stride)
        .expect("download CUDA Auto surface");
    cpu.download_into(&mut expected, stride)
        .expect("copy CPU Auto oracle surface");
    assert_eq!(actual, expected);
}

#[test]
fn auto_routes_benchmarked_raw_ht_lossy_operations_to_cuda() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = htj2k_rgb8_97_fixture(640, 480);
    let support = j2k::J2kDecoder::inspect_support(&bytes).expect("inspect raw HT fixture");
    assert_eq!(support.info.dimensions, (640, 480));
    assert_eq!(
        support.transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossy
    );
    assert_eq!(
        support.payload_kind,
        CompressedPayloadKind::Jpeg2000Codestream
    );

    let mut auto = J2kDecoder::new(&bytes).expect("Auto full decoder");
    let cuda = auto
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("Auto full decode");
    let mut oracle = J2kDecoder::new(&bytes).expect("CPU full decoder");
    let cpu = oracle
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cpu)
        .expect("CPU full decode");
    assert_cuda_matches_cpu(&cuda, &cpu);

    let roi = Rect {
        x: 160,
        y: 120,
        w: 320,
        h: 240,
    };
    let mut auto = J2kDecoder::new(&bytes).expect("Auto ROI decoder");
    let cuda = auto
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Auto)
        .expect("Auto ROI decode");
    let mut oracle = J2kDecoder::new(&bytes).expect("CPU ROI decoder");
    let cpu = oracle
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Cpu)
        .expect("CPU ROI decode");
    assert_cuda_matches_cpu(&cuda, &cpu);

    let mut auto = J2kDecoder::new(&bytes).expect("Auto scaled decoder");
    let cuda = auto
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Auto)
        .expect("Auto scaled decode");
    let mut oracle = J2kDecoder::new(&bytes).expect("CPU scaled decoder");
    let cpu = oracle
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Cpu)
        .expect("CPU scaled decode");
    assert_cuda_matches_cpu(&cuda, &cpu);
}

#[test]
fn auto_routes_benchmarked_jph_lossless_operations_to_cuda() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let codestream = htj2k_rgb8_pattern_fixture(768, 512, 17);
    let bytes = j2k::wrap_j2k_codestream(&codestream, j2k::J2kFileWrapOptions::jph())
        .expect("wrap lossless HT fixture as JPH");
    let support = j2k::J2kDecoder::inspect_support(&bytes).expect("inspect JPH fixture");
    assert_eq!(support.info.dimensions, (768, 512));
    assert_eq!(
        support.transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossless
    );
    assert_eq!(support.payload_kind, CompressedPayloadKind::JphFile);

    let mut auto = J2kDecoder::new(&bytes).expect("Auto full decoder");
    let cuda = auto
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("Auto full decode");
    let mut oracle = J2kDecoder::new(&bytes).expect("CPU full decoder");
    let cpu = oracle
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cpu)
        .expect("CPU full decode");
    assert_cuda_matches_cpu(&cuda, &cpu);

    let roi = Rect {
        x: 192,
        y: 128,
        w: 384,
        h: 256,
    };
    let mut auto = J2kDecoder::new(&bytes).expect("Auto ROI decoder");
    let cuda = auto
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Auto)
        .expect("Auto ROI decode");
    let mut oracle = J2kDecoder::new(&bytes).expect("CPU ROI decoder");
    let cpu = oracle
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Cpu)
        .expect("CPU ROI decode");
    assert_cuda_matches_cpu(&cuda, &cpu);

    let mut auto = J2kDecoder::new(&bytes).expect("Auto scaled decoder");
    let cuda = auto
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Auto)
        .expect("Auto scaled decode");
    let mut oracle = J2kDecoder::new(&bytes).expect("CPU scaled decoder");
    let cpu = oracle
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Cpu)
        .expect("CPU scaled decode");
    assert_cuda_matches_cpu(&cuda, &cpu);
}

#[test]
fn auto_routes_benchmarked_part15_repeated_batches_to_cuda() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let raw = htj2k_rgb8_97_fixture(640, 480);
    let jph_codestream = htj2k_rgb8_pattern_fixture(768, 512, 29);
    let jph = j2k::wrap_j2k_codestream(&jph_codestream, j2k::J2kFileWrapOptions::jph())
        .expect("wrap repeated lossless HT fixture as JPH");

    for (label, bytes) in [("raw HT lossy", raw), ("JPH HT lossless", jph)] {
        let inputs = vec![bytes.as_slice(); 16];
        let mut context = J2kContext::default();
        let mut pool = J2kScratchPool::new();
        let surfaces = Codec::decode_tiles_to_device(
            &mut context,
            &mut pool,
            &inputs,
            PixelFormat::Rgb8,
            BackendRequest::Auto,
        )
        .unwrap_or_else(|error| panic!("Auto {label} batch decode: {error}"));

        let mut oracle = J2kDecoder::new(&bytes).expect("CPU batch oracle decoder");
        let cpu = oracle
            .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cpu)
            .expect("CPU batch oracle decode");
        assert_eq!(surfaces.len(), inputs.len());
        for surface in &surfaces {
            assert_cuda_matches_cpu(surface, &cpu);
        }
    }
}
