use j2k_core::{
    BackendRequest, CodecError, DecoderContext, DeviceSubmission, DeviceSurface, Downscale,
    ImageDecode, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice,
};
use j2k_cuda::{Codec, CudaSession, Error, J2kDecoder, SurfaceResidency};
use j2k_native::{encode, EncodeOptions};
use j2k_test_support::{
    cuda_device_unavailable_is_skip, cuda_runtime_and_strict_oxide_gate, cuda_runtime_gate,
    htj2k_gray8_97_fixture, htj2k_gray8_fixture, htj2k_rgb8_97_fixture,
    htj2k_rgb8_fixture_with_pixels, htj2k_rgb8_pattern_fixture, openhtj2k_refinement_odd_fixture,
    rgb16ne_to_opaque_rgba16ne,
};

fn fixture() -> Vec<u8> {
    let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
}

fn fixture_ht_gray8() -> Vec<u8> {
    htj2k_gray8_fixture(4, 4)
}

fn fixture_ht_gray8_irreversible_97() -> Vec<u8> {
    htj2k_gray8_97_fixture(4, 4)
}

fn fixture_ht_rgb8() -> (Vec<u8>, Vec<u8>) {
    htj2k_rgb8_fixture_with_pixels(4, 4)
}

fn fixture_ht_rgb8_pattern(width: u32, height: u32, seed: u32) -> Vec<u8> {
    htj2k_rgb8_pattern_fixture(width, height, seed)
}

fn fixture_ht_rgb8_irreversible_97() -> Vec<u8> {
    htj2k_rgb8_97_fixture(4, 4)
}

fn fixture_openhtj2k_refinement_odd() -> &'static [u8] {
    openhtj2k_refinement_odd_fixture()
}

fn fixture_openhtj2k_refinement_odd_pixels() -> &'static [u8] {
    include_bytes!("fixtures/htj2k/openhtj2k_ds0_ht_09_b11.gray")
}

#[derive(Clone, Copy, Debug)]
enum StrictDecodeCase {
    Full,
    Region(Rect),
    Scaled(Downscale),
    RegionScaled(Rect, Downscale),
}

impl StrictDecodeCase {
    fn output_dims(self, full_dims: (u32, u32)) -> (u32, u32) {
        match self {
            Self::Full => full_dims,
            Self::Region(roi) => (roi.w, roi.h),
            Self::Scaled(scale) => {
                let scaled = Rect::full(full_dims).scaled_covering(scale);
                (scaled.w, scaled.h)
            }
            Self::RegionScaled(roi, scale) => {
                let scaled = roi.scaled_covering(scale);
                (scaled.w, scaled.h)
            }
        }
    }
}

fn decode_strict_cuda_case(
    bytes: &[u8],
    format: PixelFormat,
    case: StrictDecodeCase,
) -> Result<j2k_cuda::Surface, Error> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    match case {
        StrictDecodeCase::Full => decoder.decode_to_device(format, BackendRequest::Cuda),
        StrictDecodeCase::Region(roi) => {
            decoder.decode_region_to_device(format, roi, BackendRequest::Cuda)
        }
        StrictDecodeCase::Scaled(scale) => {
            decoder.decode_scaled_to_device(format, scale, BackendRequest::Cuda)
        }
        StrictDecodeCase::RegionScaled(roi, scale) => {
            decoder.decode_region_scaled_to_device(format, roi, scale, BackendRequest::Cuda)
        }
    }
}

fn expected_host_decode_case(
    bytes: &[u8],
    format: PixelFormat,
    case: StrictDecodeCase,
    full_dims: (u32, u32),
) -> Vec<u8> {
    if format == PixelFormat::Rgba16 {
        let rgb16 = expected_host_decode_case(bytes, PixelFormat::Rgb16, case, full_dims);
        return rgb16ne_to_opaque_rgba16ne(&rgb16);
    }

    let dims = case.output_dims(full_dims);
    let stride = dims.0 as usize * format.bytes_per_pixel();
    let mut expected = vec![0u8; stride * dims.1 as usize];
    let mut decoder = J2kDecoder::new(bytes).expect("host decoder");
    match case {
        StrictDecodeCase::Full => decoder
            .decode_into(&mut expected, stride, format)
            .expect("host full decode"),
        StrictDecodeCase::Region(roi) => decoder
            .decode_region_into(
                &mut j2k_cuda::J2kScratchPool::new(),
                &mut expected,
                stride,
                format,
                roi,
            )
            .expect("host ROI decode"),
        StrictDecodeCase::Scaled(scale) => decoder
            .decode_scaled_into(
                &mut j2k_cuda::J2kScratchPool::new(),
                &mut expected,
                stride,
                format,
                scale,
            )
            .expect("host scaled decode"),
        StrictDecodeCase::RegionScaled(roi, scale) => decoder
            .decode_region_scaled_into(
                &mut j2k_cuda::J2kScratchPool::new(),
                &mut expected,
                stride,
                format,
                roi,
                scale,
            )
            .expect("host region+scaled decode"),
    };
    expected
}

fn assert_bytes_within(actual: &[u8], expected: &[u8], tolerance: u8, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let mut max_delta = 0u8;
    for (&lhs, &rhs) in actual.iter().zip(expected) {
        max_delta = max_delta.max(lhs.abs_diff(rhs));
    }
    assert!(
        max_delta <= tolerance,
        "{label} max byte delta {max_delta} exceeded tolerance {tolerance}"
    );
}

#[test]
fn auto_falls_back_to_cpu_surface() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert_eq!(surface.residency(), SurfaceResidency::Host);
    assert!(surface.as_host_bytes().is_some());
}

#[test]
fn explicit_cuda_classic_j2k_request_rejects_cpu_staged_upload() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    let error = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect_err("classic J2K must not be CPU-decoded and uploaded");

    assert!(error.is_unsupported());
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_request_validates_decode_before_upload() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    let error = decoder
        .decode_to_device(PixelFormat::Rgba16, BackendRequest::Cuda)
        .expect_err("unsupported decode");
    assert!(error.is_unsupported());
    assert!(!matches!(error, Error::CudaUnavailable));
}

#[test]
fn explicit_cuda_request_returns_cuda_surface_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (4, 4));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = [0u8; 16];
    host_decoder
        .decode_into(&mut expected, 4, PixelFormat::Gray8)
        .expect("host decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_profile_reports_gpu_stage_timings_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut session = CudaSession::default();
    let (surface, report) = decoder
        .decode_to_device_with_session_and_profile(PixelFormat::Gray8, &mut session)
        .expect("strict CUDA HTJ2K profiled surface");

    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_resident_cuda_surface(&surface);
    assert!(report.dispatch_count > 0);
    assert!(report.ht_cleanup_us > 0);
    assert_eq!(report.dequant_us, 0);
    assert_eq!(report.detail.dequant_dispatch_count, 0);
    assert!(report.idwt_us > 0);
    assert!(report.store_us > 0);
    assert_eq!(report.residency, SurfaceResidency::CudaResidentDecode);
}

#[test]
fn explicit_cuda_region_surface_matches_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K ROI surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (roi.w, roi.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda ROI surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; roi.w as usize * roi.h as usize];
    host_decoder
        .decode_region_into(
            &mut j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            roi.w as usize,
            PixelFormat::Gray8,
            roi,
        )
        .expect("host ROI decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_scaled_surface_matches_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = fixture_ht_gray8();
    let scale = Downscale::Half;
    let scaled = Rect::full((4, 4)).scaled_covering(scale);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_scaled_to_device(PixelFormat::Gray8, scale, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K scaled surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda scaled surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_scaled_into(
            &mut j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            scaled.w as usize,
            PixelFormat::Gray8,
            scale,
        )
        .expect("host scaled decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_rgb8_request_returns_resident_surface_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let (bytes, pixels) = fixture_ht_rgb8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K RGB surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (4, 4));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; pixels.len()];
    host_decoder
        .decode_into(&mut expected, 4 * 3, PixelFormat::Rgb8)
        .expect("host decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_rgb8_profile_keeps_fused_mct_store_accounting_when_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let (bytes, pixels) = fixture_ht_rgb8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut session = CudaSession::default();
    let (surface, report) = decoder
        .decode_to_device_with_session_and_profile(PixelFormat::Rgb8, &mut session)
        .expect("profiled strict CUDA HTJ2K RGB surface");

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download profiled CUDA RGB surface");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; pixels.len()];
    host_decoder
        .decode_into(&mut expected, 4 * 3, PixelFormat::Rgb8)
        .expect("host RGB decode");

    assert_eq!(downloaded, expected);
    assert_eq!(report.mct_us, 0);
    assert_eq!(report.detail.mct_dispatch_count, 0);
    assert_eq!(report.detail.store_dispatch_count, 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_rgb8_region_request_reaches_runtime_boundary() {
    let (bytes, _) = fixture_ht_rgb8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    match decoder.decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Cuda) {
        Ok(surface) => {
            assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
            assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
            assert_eq!(surface.dimensions(), (roi.w, roi.h));
        }
        Err(Error::UnsupportedCudaRequest { reason }) => {
            panic!("RGB ROI must not stop at strict unsupported before CUDA runtime: {reason}");
        }
        Err(Error::CudaUnavailable | Error::CudaRuntime { .. }) => {}
        Err(error) => panic!("unexpected RGB ROI strict CUDA error: {error}"),
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_rgba_requests_reach_runtime_boundary() {
    let (bytes, _) = fixture_ht_rgb8();

    for format in [PixelFormat::Rgba8, PixelFormat::Rgba16] {
        let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
        match decoder.decode_to_device(format, BackendRequest::Cuda) {
            Ok(surface) => {
                assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
                assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                assert_eq!(surface.dimensions(), (4, 4));
            }
            Err(Error::UnsupportedCudaRequest { reason }) => {
                panic!(
                    "{format:?} must not stop at strict unsupported before CUDA runtime: {reason}"
                );
            }
            Err(Error::CudaUnavailable | Error::CudaRuntime { .. }) => {}
            Err(error) => panic!("unexpected {format:?} strict CUDA error: {error}"),
        }
    }
}

#[test]
fn explicit_cuda_rgb8_region_surface_matches_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let (bytes, _) = fixture_ht_rgb8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K RGB ROI surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (roi.w, roi.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda RGB ROI surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; roi.w as usize * roi.h as usize * 3];
    host_decoder
        .decode_region_into(
            &mut j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            roi.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("host RGB ROI decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_scaled_rgb8_surface_matches_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let (bytes, _) = fixture_ht_rgb8();
    let scale = Downscale::Half;
    let scaled = Rect::full((4, 4)).scaled_covering(scale);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_scaled_to_device(PixelFormat::Rgb8, scale, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K scaled RGB surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda scaled RGB surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; scaled.w as usize * scaled.h as usize * 3];
    host_decoder
        .decode_scaled_into(
            &mut j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            scaled.w as usize * 3,
            PixelFormat::Rgb8,
            scale,
        )
        .expect("host scaled RGB decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_rgb8_region_scaled_surface_matches_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let (bytes, _) = fixture_ht_rgb8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K RGB region+scaled surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda RGB region+scaled surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; scaled.w as usize * scaled.h as usize * 3];
    host_decoder
        .decode_region_scaled_into(
            &mut j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            scaled.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("host RGB region+scaled decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_gray16_and_rgb16_requests_return_resident_surfaces_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let gray_bytes = fixture_ht_gray8();
    let mut gray_decoder = J2kDecoder::new(&gray_bytes).expect("gray decoder");
    let gray_surface = gray_decoder
        .decode_to_device(PixelFormat::Gray16, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K Gray16 surface");
    assert_eq!(gray_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(
        gray_surface.residency(),
        SurfaceResidency::CudaResidentDecode
    );
    assert_resident_cuda_surface(&gray_surface);

    let mut gray_downloaded = vec![0u8; gray_surface.byte_len()];
    gray_surface
        .download_into(&mut gray_downloaded, gray_surface.pitch_bytes())
        .expect("download Gray16 cuda surface");
    let mut host_gray_decoder = J2kDecoder::new(&gray_bytes).expect("host gray decoder");
    let mut expected_gray = vec![0u8; 4 * 4 * 2];
    host_gray_decoder
        .decode_into(&mut expected_gray, 4 * 2, PixelFormat::Gray16)
        .expect("host Gray16 decode");
    assert_eq!(gray_downloaded, expected_gray);

    let (rgb_bytes, _) = fixture_ht_rgb8();
    let mut rgb_decoder = J2kDecoder::new(&rgb_bytes).expect("rgb decoder");
    let rgb_surface = rgb_decoder
        .decode_to_device(PixelFormat::Rgb16, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K Rgb16 surface");
    assert_eq!(rgb_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(
        rgb_surface.residency(),
        SurfaceResidency::CudaResidentDecode
    );
    assert_resident_cuda_surface(&rgb_surface);

    let mut rgb_downloaded = vec![0u8; rgb_surface.byte_len()];
    rgb_surface
        .download_into(&mut rgb_downloaded, rgb_surface.pitch_bytes())
        .expect("download Rgb16 cuda surface");
    let mut host_rgb_decoder = J2kDecoder::new(&rgb_bytes).expect("host rgb decoder");
    let mut expected_rgb = vec![0u8; 4 * 4 * 3 * 2];
    host_rgb_decoder
        .decode_into(&mut expected_rgb, 4 * 3 * 2, PixelFormat::Rgb16)
        .expect("host Rgb16 decode");
    assert_eq!(rgb_downloaded, expected_rgb);
}

#[test]
fn explicit_cuda_rgba8_and_rgba16_requests_return_resident_surfaces_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let (bytes, _) = fixture_ht_rgb8();

    let mut rgba8_decoder = J2kDecoder::new(&bytes).expect("rgba8 decoder");
    let rgba8_surface = rgba8_decoder
        .decode_to_device(PixelFormat::Rgba8, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K Rgba8 surface");
    assert_eq!(rgba8_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(
        rgba8_surface.residency(),
        SurfaceResidency::CudaResidentDecode
    );
    assert_eq!(rgba8_surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&rgba8_surface);
    assert_eq!(rgba8_surface.dimensions(), (4, 4));

    let mut rgba8_downloaded = vec![0u8; rgba8_surface.byte_len()];
    rgba8_surface
        .download_into(&mut rgba8_downloaded, rgba8_surface.pitch_bytes())
        .expect("download Rgba8 cuda surface");
    let mut host_rgba8_decoder = J2kDecoder::new(&bytes).expect("host rgba8 decoder");
    let mut expected_rgba8 = vec![0u8; 4 * 4 * 4];
    host_rgba8_decoder
        .decode_into(&mut expected_rgba8, 4 * 4, PixelFormat::Rgba8)
        .expect("host Rgba8 decode");
    assert_eq!(rgba8_downloaded, expected_rgba8);

    let mut rgba16_decoder = J2kDecoder::new(&bytes).expect("rgba16 decoder");
    let rgba16_surface = rgba16_decoder
        .decode_to_device(PixelFormat::Rgba16, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K Rgba16 surface");
    assert_eq!(rgba16_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(
        rgba16_surface.residency(),
        SurfaceResidency::CudaResidentDecode
    );
    assert_eq!(rgba16_surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&rgba16_surface);
    assert_eq!(rgba16_surface.dimensions(), (4, 4));

    let mut rgba16_downloaded = vec![0u8; rgba16_surface.byte_len()];
    rgba16_surface
        .download_into(&mut rgba16_downloaded, rgba16_surface.pitch_bytes())
        .expect("download Rgba16 cuda surface");
    let expected_rgba16 =
        expected_host_decode_case(&bytes, PixelFormat::Rgba16, StrictDecodeCase::Full, (4, 4));
    assert_eq!(rgba16_downloaded, expected_rgba16);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_16bit_and_rgba_region_scaled_requests_reach_runtime_boundary() {
    let gray_bytes = fixture_ht_gray8();
    let (rgb_bytes, _) = fixture_ht_rgb8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let cases = [
        StrictDecodeCase::Region(roi),
        StrictDecodeCase::Scaled(Downscale::Half),
        StrictDecodeCase::RegionScaled(roi, Downscale::Half),
    ];

    for case in cases {
        match decode_strict_cuda_case(&gray_bytes, PixelFormat::Gray16, case) {
            Ok(surface) => {
                assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
                assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                assert_eq!(surface.dimensions(), case.output_dims((4, 4)));
            }
            Err(Error::UnsupportedCudaRequest { reason }) => {
                panic!("Gray16 {case:?} must reach CUDA runtime boundary: {reason}");
            }
            Err(Error::CudaUnavailable | Error::CudaRuntime { .. }) => {}
            Err(error) => panic!("unexpected Gray16 {case:?} strict CUDA error: {error}"),
        }
    }

    for format in [PixelFormat::Rgb16, PixelFormat::Rgba8, PixelFormat::Rgba16] {
        for case in cases {
            match decode_strict_cuda_case(&rgb_bytes, format, case) {
                Ok(surface) => {
                    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
                    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                    assert_eq!(surface.dimensions(), case.output_dims((4, 4)));
                }
                Err(Error::UnsupportedCudaRequest { reason }) => {
                    panic!("{format:?} {case:?} must reach CUDA runtime boundary: {reason}");
                }
                Err(Error::CudaUnavailable | Error::CudaRuntime { .. }) => {}
                Err(error) => panic!("unexpected {format:?} {case:?} strict CUDA error: {error}"),
            }
        }
    }
}

#[test]
fn explicit_cuda_16bit_and_rgba_region_scaled_surfaces_match_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let gray_bytes = fixture_ht_gray8();
    let (rgb_bytes, _) = fixture_ht_rgb8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let cases = [
        StrictDecodeCase::Region(roi),
        StrictDecodeCase::Scaled(Downscale::Half),
        StrictDecodeCase::RegionScaled(roi, Downscale::Half),
    ];

    for case in cases {
        let surface = decode_strict_cuda_case(&gray_bytes, PixelFormat::Gray16, case)
            .expect("strict CUDA HTJ2K Gray16 surface");
        assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
        assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
        assert_eq!(surface.as_host_bytes(), None);
        assert_resident_cuda_surface(&surface);
        assert_eq!(surface.dimensions(), case.output_dims((4, 4)));

        let mut downloaded = vec![0u8; surface.byte_len()];
        surface
            .download_into(&mut downloaded, surface.pitch_bytes())
            .expect("download Gray16 cuda surface");
        let expected = expected_host_decode_case(&gray_bytes, PixelFormat::Gray16, case, (4, 4));
        assert_eq!(downloaded, expected, "Gray16 {case:?}");
    }

    for format in [PixelFormat::Rgb16, PixelFormat::Rgba8, PixelFormat::Rgba16] {
        for case in cases {
            let surface = decode_strict_cuda_case(&rgb_bytes, format, case)
                .expect("strict CUDA HTJ2K color surface");
            assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
            assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
            assert_eq!(surface.as_host_bytes(), None);
            assert_resident_cuda_surface(&surface);
            assert_eq!(surface.dimensions(), case.output_dims((4, 4)));

            let mut downloaded = vec![0u8; surface.byte_len()];
            surface
                .download_into(&mut downloaded, surface.pitch_bytes())
                .expect("download color cuda surface");
            let expected = expected_host_decode_case(&rgb_bytes, format, case, (4, 4));
            assert_eq!(downloaded, expected, "{format:?} {case:?}");
        }
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_irreversible_97_requests_reach_runtime_boundary() {
    let gray_bytes = fixture_ht_gray8_irreversible_97();
    let rgb_bytes = fixture_ht_rgb8_irreversible_97();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let cases = [
        StrictDecodeCase::Full,
        StrictDecodeCase::Region(roi),
        StrictDecodeCase::Scaled(Downscale::Half),
        StrictDecodeCase::RegionScaled(roi, Downscale::Half),
    ];

    for case in cases {
        match decode_strict_cuda_case(&gray_bytes, PixelFormat::Gray8, case) {
            Ok(surface) => {
                assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
                assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                assert_eq!(surface.dimensions(), case.output_dims((4, 4)));
            }
            Err(Error::UnsupportedCudaRequest { reason }) => {
                panic!("irreversible Gray8 {case:?} must reach CUDA runtime boundary: {reason}");
            }
            Err(Error::CudaUnavailable | Error::CudaRuntime { .. }) => {}
            Err(error) => panic!("unexpected irreversible Gray8 {case:?} CUDA error: {error}"),
        }
    }

    for case in cases {
        match decode_strict_cuda_case(&rgb_bytes, PixelFormat::Rgb8, case) {
            Ok(surface) => {
                assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
                assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                assert_eq!(surface.dimensions(), case.output_dims((4, 4)));
            }
            Err(Error::UnsupportedCudaRequest { reason }) => {
                panic!("irreversible Rgb8 {case:?} must reach CUDA runtime boundary: {reason}");
            }
            Err(Error::CudaUnavailable | Error::CudaRuntime { .. }) => {}
            Err(error) => panic!("unexpected irreversible Rgb8 {case:?} CUDA error: {error}"),
        }
    }
}

#[test]
fn explicit_cuda_irreversible_97_surfaces_match_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let gray_bytes = fixture_ht_gray8_irreversible_97();
    let rgb_bytes = fixture_ht_rgb8_irreversible_97();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let cases = [
        StrictDecodeCase::Full,
        StrictDecodeCase::Region(roi),
        StrictDecodeCase::Scaled(Downscale::Half),
        StrictDecodeCase::RegionScaled(roi, Downscale::Half),
    ];

    for case in cases {
        let surface = decode_strict_cuda_case(&gray_bytes, PixelFormat::Gray8, case)
            .expect("strict CUDA HTJ2K irreversible Gray8 surface");
        assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
        assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
        assert_eq!(surface.as_host_bytes(), None);
        assert_resident_cuda_surface(&surface);
        assert_eq!(surface.dimensions(), case.output_dims((4, 4)));

        let mut downloaded = vec![0u8; surface.byte_len()];
        surface
            .download_into(&mut downloaded, surface.pitch_bytes())
            .expect("download irreversible Gray8 cuda surface");
        let expected = expected_host_decode_case(&gray_bytes, PixelFormat::Gray8, case, (4, 4));
        assert_bytes_within(
            &downloaded,
            &expected,
            2,
            &format!("irreversible Gray8 {case:?}"),
        );
    }

    for case in cases {
        let surface = decode_strict_cuda_case(&rgb_bytes, PixelFormat::Rgb8, case)
            .expect("strict CUDA HTJ2K irreversible Rgb8 surface");
        assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
        assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
        assert_eq!(surface.as_host_bytes(), None);
        assert_resident_cuda_surface(&surface);
        assert_eq!(surface.dimensions(), case.output_dims((4, 4)));

        let mut downloaded = vec![0u8; surface.byte_len()];
        surface
            .download_into(&mut downloaded, surface.pitch_bytes())
            .expect("download irreversible Rgb8 cuda surface");
        let expected = expected_host_decode_case(&rgb_bytes, PixelFormat::Rgb8, case, (4, 4));
        assert_bytes_within(
            &downloaded,
            &expected,
            2,
            &format!("irreversible Rgb8 {case:?}"),
        );
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_refinement_fixture_request_reaches_runtime_boundary() {
    let mut decoder = J2kDecoder::new(fixture_openhtj2k_refinement_odd()).expect("decoder");

    match decoder.decode_to_device(PixelFormat::Gray8, BackendRequest::Cuda) {
        Ok(surface) => {
            assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
            assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
            assert_eq!(surface.dimensions(), (17, 37));
        }
        Err(Error::UnsupportedCudaRequest { reason }) => {
            panic!("refinement fixture must reach CUDA runtime boundary: {reason}");
        }
        Err(Error::CudaUnavailable | Error::CudaRuntime { .. }) => {}
        Err(error) => panic!("unexpected refinement fixture strict CUDA error: {error}"),
    }
}

#[test]
fn explicit_cuda_refinement_fixture_surface_matches_oracle_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let mut decoder = J2kDecoder::new(fixture_openhtj2k_refinement_odd()).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K refinement surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (17, 37));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download refinement cuda surface");
    assert_eq!(downloaded, fixture_openhtj2k_refinement_odd_pixels());
}

#[test]
fn explicit_cuda_refinement_fixture_profile_reports_refine_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let mut decoder = J2kDecoder::new(fixture_openhtj2k_refinement_odd()).expect("decoder");
    let mut session = CudaSession::default();
    let (surface, report) = decoder
        .decode_to_device_with_session_and_profile(PixelFormat::Gray8, &mut session)
        .expect("strict CUDA HTJ2K profiled refinement surface");

    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_resident_cuda_surface(&surface);
    assert_eq!(report.block_count, 14);
    assert!(report.ht_cleanup_us > 0);
    assert!(report.ht_refine_us > 0);
    assert!(report.dequant_us > 0);
    assert!(report.idwt_us > 0);
    assert!(report.store_us > 0);
}

#[test]
fn explicit_cpu_staged_cuda_api_marks_cpu_upload_residency_when_cuda_runtime_required() {
    if !cuda_runtime_gate(module_path!()) {
        return;
    }

    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut session = CudaSession::default();
    let surface = decoder
        .decode_to_cpu_staged_cuda_surface_with_session(PixelFormat::Rgb8, &mut session)
        .expect("CPU-staged CUDA surface");

    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CpuStagedCudaUpload);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cpu_staged_cuda_surface(&surface);

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download CPU-staged CUDA surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = [0u8; 12];
    host_decoder
        .decode_into(&mut expected, 6, PixelFormat::Rgb8)
        .expect("host decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_region_scaled_surface_matches_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Gray8, roi, scale, BackendRequest::Cuda)
        .expect("strict CUDA HTJ2K region+scaled surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda region+scaled surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host region+scaled decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn explicit_cuda_download_respects_padded_stride_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_resident_cuda_surface(&surface);
    let row_bytes = surface.pitch_bytes();
    let stride = row_bytes + 5;
    let mut downloaded = vec![0xCD; stride * surface.dimensions().1 as usize];
    surface
        .download_into(&mut downloaded, stride)
        .expect("download cuda surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = [0u8; 16];
    host_decoder
        .decode_into(&mut expected, row_bytes, PixelFormat::Gray8)
        .expect("host decode");
    for (row, expected_row) in expected.chunks(row_bytes).enumerate() {
        let start = row * stride;
        assert_eq!(&downloaded[start..start + row_bytes], expected_row);
        assert_eq!(&downloaded[start + row_bytes..start + stride], &[0xCD; 5]);
    }
}

fn assert_cpu_staged_cuda_surface(surface: &j2k_cuda::Surface) {
    let cuda = surface.cuda_surface().expect("cuda surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert!(cuda.stats().kernel_dispatches() > 0);
    assert!(cuda.stats().copy_kernel_dispatches() > 0);
    assert_eq!(cuda.stats().decode_kernel_dispatches(), 0);
}

fn assert_resident_cuda_surface(surface: &j2k_cuda::Surface) {
    let cuda = surface.cuda_surface().expect("cuda surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert!(cuda.stats().kernel_dispatches() > 0);
    assert_eq!(cuda.stats().copy_kernel_dispatches(), 0);
    assert!(cuda.stats().decode_kernel_dispatches() > 0);
}

fn assert_cuda_batch_surface(surface: &j2k_cuda::Surface) {
    let cuda = surface.cuda_surface().expect("cuda batch surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert_eq!(cuda.stats().copy_kernel_dispatches(), 0);
}

#[test]
fn submit_to_device_auto_falls_back_to_cpu_surface() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("submission")
    .wait()
    .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
    assert!(session.submissions() >= 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn submit_to_device_auto_does_not_initialize_cuda_runtime() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("submission")
    .wait()
    .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert_eq!(session.submissions(), 1);
    assert!(!session.is_runtime_initialized());
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_submissions_reuse_session_runtime_when_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = fixture_ht_gray8();
    let mut session = CudaSession::default();
    assert!(!session.is_runtime_initialized());

    let mut first = J2kDecoder::new(&bytes).expect("decoder");
    let first_surface = <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut first,
        &mut session,
        PixelFormat::Gray8,
        BackendRequest::Cuda,
    )
    .expect("first submission")
    .wait()
    .expect("first surface");
    assert_eq!(first_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_resident_cuda_surface(&first_surface);
    assert!(session.is_runtime_initialized());

    let mut second = J2kDecoder::new(&bytes).expect("decoder");
    let second_surface = <J2kDecoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut second,
        &mut session,
        PixelFormat::Gray8,
        BackendRequest::Cuda,
    )
    .expect("second submission")
    .wait()
    .expect("second surface");
    assert_eq!(second_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_resident_cuda_surface(&second_surface);
    assert_eq!(session.submissions(), 2);
    assert!(session.is_runtime_initialized());
}

#[test]
fn auto_classic_full_frame_surface_matches_host_decode() {
    let bytes = fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 12];
    host_decoder
        .decode_into(&mut host, 6, PixelFormat::Rgb8)
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn auto_htj2k_full_frame_surface_matches_host_decode() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn auto_region_scaled_surface_matches_host_decode() {
    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Gray8, roi, scale, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut j2k_cuda::J2kScratchPool::new(),
            &mut host,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn tile_batch_region_cuda_surface_matches_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let mut ctx = DecoderContext::<j2k_cuda::J2kContext>::new();
    let mut pool = j2k_cuda::J2kScratchPool::new();
    let surface = Codec::decode_tile_region_to_device(
        &mut ctx,
        &mut pool,
        &bytes,
        PixelFormat::Gray8,
        roi,
        BackendRequest::Cuda,
    )
    .expect("cuda tile batch ROI surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (roi.w, roi.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; roi.w as usize * roi.h as usize];
    host_decoder
        .decode_region_into(
            &mut j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            roi.w as usize,
            PixelFormat::Gray8,
            roi,
        )
        .expect("host decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn tile_batch_region_scaled_cuda_surface_matches_host_when_cuda_runtime_required() {
    if !cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }

    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = DecoderContext::<j2k_cuda::J2kContext>::new();
    let mut pool = j2k_cuda::J2kScratchPool::new();
    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        &bytes,
        PixelFormat::Gray8,
        roi,
        scale,
        BackendRequest::Cuda,
    )
    .expect("cuda tile batch scaled ROI surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
    assert_eq!(surface.as_host_bytes(), None);
    assert_resident_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda scaled ROI surface");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut expected = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut j2k_cuda::J2kScratchPool::new(),
            &mut expected,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host scaled ROI decode");
    assert_eq!(downloaded, expected);
}

#[test]
fn decode_tiles_to_device_auto_preserves_order_and_matches_host_bytes() {
    let bytes = fixture_ht_gray8();
    let mut ctx = DecoderContext::<j2k_cuda::J2kContext>::new();
    let mut pool = j2k_cuda::J2kScratchPool::new();
    let inputs = [bytes.as_slice(), bytes.as_slice()];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Gray8,
        BackendRequest::Auto,
    )
    .expect("batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    let mut expected = [0u8; 16];
    J2kDecoder::new(&bytes)
        .expect("host decoder")
        .decode_into(&mut expected, 4, PixelFormat::Gray8)
        .expect("host decode");
    for surface in surfaces {
        assert_eq!(surface.dimensions(), (4, 4));
        assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
        assert_eq!(surface.residency(), SurfaceResidency::Host);
        assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
    }
}

#[test]
fn decode_tiles_to_device_cpu_preserves_host_residency() {
    let bytes = fixture_ht_gray8();
    let mut ctx = DecoderContext::<j2k_cuda::J2kContext>::new();
    let mut pool = j2k_cuda::J2kScratchPool::new();
    let inputs = [bytes.as_slice(), bytes.as_slice()];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Gray8,
        BackendRequest::Cpu,
    )
    .expect("CPU batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    for surface in surfaces {
        assert_eq!(surface.dimensions(), (4, 4));
        assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
        assert_eq!(surface.residency(), SurfaceResidency::Host);
        assert!(surface.as_host_bytes().is_some());
    }
}

#[test]
fn decode_tiles_to_device_explicit_cuda_rgb8_batch_matches_host_bytes() {
    let first = fixture_ht_rgb8_pattern(32, 32, 17);
    let second = fixture_ht_rgb8_pattern(32, 32, 29);
    let inputs = [first.as_slice(), second.as_slice()];
    let mut ctx = DecoderContext::<j2k_cuda::J2kContext>::new();
    let mut pool = j2k_cuda::J2kScratchPool::new();

    let surfaces = match Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    ) {
        Ok(surfaces) => surfaces,
        Err(Error::CudaUnavailable) if cuda_device_unavailable_is_skip(module_path!()) => return,
        #[cfg(feature = "cuda-runtime")]
        Err(Error::CudaRuntime { .. }) if cuda_device_unavailable_is_skip(module_path!()) => return,
        Err(error) => panic!("strict CUDA RGB8 batch decode failed: {error}"),
    };

    assert_eq!(surfaces.len(), inputs.len());
    for surface in &surfaces {
        assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
        assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
        assert_eq!(surface.as_host_bytes(), None);
        assert_cuda_batch_surface(surface);
    }

    let downloaded =
        j2k_cuda::Surface::download_batch_tight(&surfaces).expect("download tight batch");
    let expected = inputs
        .iter()
        .flat_map(|input| {
            let mut out = vec![0u8; 32 * 32 * 3];
            J2kDecoder::new(input)
                .expect("host decoder")
                .decode_into(&mut out, 32 * 3, PixelFormat::Rgb8)
                .expect("host decode");
            out
        })
        .collect::<Vec<_>>();
    assert_eq!(downloaded, expected);
}

#[test]
fn decode_tiles_to_device_explicit_cuda_rgba8_batch_matches_host_bytes() {
    let first = fixture_ht_rgb8_pattern(32, 32, 31);
    let second = fixture_ht_rgb8_pattern(32, 32, 47);
    let inputs = [first.as_slice(), second.as_slice()];
    let mut ctx = DecoderContext::<j2k_cuda::J2kContext>::new();
    let mut pool = j2k_cuda::J2kScratchPool::new();

    let surfaces = match Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgba8,
        BackendRequest::Cuda,
    ) {
        Ok(surfaces) => surfaces,
        Err(Error::CudaUnavailable) if cuda_device_unavailable_is_skip(module_path!()) => return,
        #[cfg(feature = "cuda-runtime")]
        Err(Error::CudaRuntime { .. }) if cuda_device_unavailable_is_skip(module_path!()) => return,
        Err(error) => panic!("strict CUDA Rgba8 batch decode failed: {error}"),
    };

    assert_eq!(surfaces.len(), inputs.len());
    for surface in &surfaces {
        assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
        assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
        assert_eq!(surface.as_host_bytes(), None);
        assert_cuda_batch_surface(surface);
    }

    let downloaded =
        j2k_cuda::Surface::download_batch_tight(&surfaces).expect("download tight batch");
    let expected = inputs
        .iter()
        .flat_map(|input| {
            let mut out = vec![0u8; 32 * 32 * 4];
            J2kDecoder::new(input)
                .expect("host decoder")
                .decode_into(&mut out, 32 * 4, PixelFormat::Rgba8)
                .expect("host decode");
            out
        })
        .collect::<Vec<_>>();
    assert_eq!(downloaded, expected);
}

#[test]
fn decode_tiles_to_device_explicit_cuda_returns_cuda_surfaces_or_clear_unavailable_error() {
    let bytes = fixture_ht_gray8();
    let mut ctx = DecoderContext::<j2k_cuda::J2kContext>::new();
    let mut pool = j2k_cuda::J2kScratchPool::new();
    let inputs = [bytes.as_slice(), bytes.as_slice()];

    match Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Gray8,
        BackendRequest::Cuda,
    ) {
        Ok(surfaces) => {
            assert_eq!(surfaces.len(), inputs.len());
            for surface in surfaces {
                assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
                assert_eq!(surface.residency(), SurfaceResidency::CudaResidentDecode);
                assert_eq!(surface.as_host_bytes(), None);
                assert_resident_cuda_surface(&surface);
            }
        }
        Err(error) => assert!(error.is_unsupported()),
    }
}
