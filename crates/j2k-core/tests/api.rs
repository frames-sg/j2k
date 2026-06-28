use j2k_core::{
    copy_tight_pixels_to_strided_output, strided_output_len, submit_ready_device,
    validate_cuda_surface_backend_request, validate_strided_output_buffer, BackendCapabilities,
    BackendKind, BackendRequest, BufferError, CodecContext, CodecError, CpuFeatures,
    DecoderContext, DeviceSubmission, DeviceSubmitSession, DeviceSurface, Downscale, ImageCodec,
    ImageDecode, ImageDecodeDevice, ImageDecodeSubmit, PassthroughCandidate, PassthroughDecision,
    PassthroughRejectReason, PassthroughRequirements, PixelFormat, PixelLayout, ReadySubmission,
    Rect, SampleType, ScratchPool, TileBatchDecodeDevice, TileBatchDecodeManyDevice,
    TileBatchDecodeSubmit, TileBatchOptions,
};
use j2k_core::{
    CodedUnitLayout, Colorspace, CompressedPayloadKind, CompressedTransferSyntax, Info, TileLayout,
};
use std::num::NonZeroUsize;

#[test]
fn pixel_format_reports_layout_and_sample_type() {
    assert_eq!(PixelFormat::Rgb8.layout(), PixelLayout::Rgb);
    assert_eq!(PixelFormat::Rgb8.sample(), SampleType::U8);
    assert_eq!(PixelFormat::Rgb16.sample(), SampleType::U16);
}

#[derive(Default)]
struct SubmitCounter {
    submissions: u64,
}

impl DeviceSubmitSession for SubmitCounter {
    fn record_submit(&mut self) {
        self.submissions += 1;
    }
}

#[test]
fn submit_ready_device_records_and_returns_success() {
    let mut session = SubmitCounter::default();

    let surface = submit_ready_device(&mut session, |session| {
        assert_eq!(session.submissions, 1);
        Ok::<_, DummyError>(DummySurface {
            backend: BackendKind::Cuda,
            dims: (2, 3),
            fmt: PixelFormat::Rgb8,
            len: 18,
        })
    })
    .wait()
    .expect("ready submission succeeds");

    assert_eq!(session.submissions, 1);
    assert_eq!(surface.backend_kind(), BackendKind::Cuda);
    assert_eq!(surface.dimensions(), (2, 3));
}

#[test]
fn submit_ready_device_records_and_returns_error() {
    let mut session = SubmitCounter::default();

    let error = submit_ready_device(&mut session, |_session| Err::<DummySurface, _>(DummyError))
        .wait()
        .expect_err("ready submission surfaces the error");

    assert_eq!(error, DummyError);
    assert_eq!(session.submissions, 1);
}

#[test]
fn cuda_surface_backend_request_validation_rejects_metal_only() {
    assert_eq!(
        validate_cuda_surface_backend_request(BackendRequest::Cpu),
        Ok(())
    );
    assert_eq!(
        validate_cuda_surface_backend_request(BackendRequest::Auto),
        Ok(())
    );
    assert_eq!(
        validate_cuda_surface_backend_request(BackendRequest::Cuda),
        Ok(())
    );
    assert_eq!(
        validate_cuda_surface_backend_request(BackendRequest::Metal),
        Err(BackendRequest::Metal)
    );
}

#[test]
fn downscale_reports_expected_denominators() {
    assert_eq!(Downscale::None.denominator(), 1);
    assert_eq!(Downscale::Half.denominator(), 2);
    assert_eq!(Downscale::Quarter.denominator(), 4);
    assert_eq!(Downscale::Eighth.denominator(), 8);
}

#[test]
fn rect_scaled_covering_uses_floor_start_and_ceil_end() {
    let roi = Rect {
        x: 3,
        y: 5,
        w: 10,
        h: 11,
    };

    assert_eq!(
        roi.scaled_covering(Downscale::Quarter),
        Rect {
            x: 0,
            y: 1,
            w: 4,
            h: 3,
        }
    );
    assert_eq!(roi.scaled_covering(Downscale::None), roi);
}

#[test]
fn rect_full_and_is_within_match_existing_jpeg_behavior() {
    let full = Rect::full((640, 480));
    assert_eq!(
        full,
        Rect {
            x: 0,
            y: 0,
            w: 640,
            h: 480,
        }
    );
    assert!(Rect {
        x: 10,
        y: 10,
        w: 100,
        h: 100,
    }
    .is_within((640, 480)));
}

#[test]
fn copy_tight_pixels_to_strided_output_copies_exact_rows() {
    let src = [1, 2, 3, 4, 5, 6];
    let mut out = [0; 6];

    copy_tight_pixels_to_strided_output(&src, (2, 1), PixelFormat::Rgb8, &mut out, 6)
        .expect("copy exact rows");

    assert_eq!(out, src);
}

#[test]
fn copy_tight_pixels_to_strided_output_preserves_row_padding() {
    let src = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut out = [0xee; 12];

    copy_tight_pixels_to_strided_output(&src, (2, 2), PixelFormat::Gray16, &mut out, 6)
        .expect("copy padded rows");

    assert_eq!(out, [1, 2, 3, 4, 0xee, 0xee, 5, 6, 7, 8, 0xee, 0xee]);
}

#[test]
fn copy_tight_pixels_to_strided_output_accepts_empty_height() {
    let mut out = [0xee; 2];

    copy_tight_pixels_to_strided_output(&[], (3, 0), PixelFormat::Rgba8, &mut out, 0)
        .expect("copy zero rows");

    assert_eq!(out, [0xee, 0xee]);
}

#[test]
fn copy_tight_pixels_to_strided_output_accepts_empty_width() {
    let mut out = [0xee; 2];

    copy_tight_pixels_to_strided_output(&[], (0, 3), PixelFormat::Rgba8, &mut out, 0)
        .expect("copy zero-width rows");

    assert_eq!(out, [0xee, 0xee]);
}

#[test]
fn copy_tight_pixels_to_strided_output_rejects_short_source() {
    let mut out = [0; 6];

    let err = copy_tight_pixels_to_strided_output(
        &[1, 2, 3, 4, 5],
        (2, 1),
        PixelFormat::Rgb8,
        &mut out,
        6,
    )
    .expect_err("source too small");

    assert_eq!(
        err,
        BufferError::InputTooSmall {
            required: 6,
            have: 5,
        }
    );
}

#[test]
fn copy_tight_pixels_to_strided_output_rejects_small_stride() {
    let mut out = [0; 6];

    let err = copy_tight_pixels_to_strided_output(
        &[1, 2, 3, 4, 5, 6],
        (2, 1),
        PixelFormat::Rgb8,
        &mut out,
        5,
    )
    .expect_err("stride too small");

    assert_eq!(
        err,
        BufferError::StrideTooSmall {
            row_bytes: 6,
            stride: 5,
        }
    );
}

#[test]
fn copy_tight_pixels_to_strided_output_rejects_small_output() {
    let src = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut out = [0; 9];

    let err = copy_tight_pixels_to_strided_output(&src, (2, 2), PixelFormat::Gray16, &mut out, 6)
        .expect_err("output too small");

    assert_eq!(
        err,
        BufferError::OutputTooSmall {
            required: 10,
            have: 9,
        }
    );
}

#[test]
fn copy_tight_pixels_to_strided_output_rejects_strided_output_overflow() {
    let mut out = [];

    let err = copy_tight_pixels_to_strided_output(
        &[1, 2],
        (1, 2),
        PixelFormat::Gray8,
        &mut out,
        usize::MAX,
    )
    .expect_err("strided output overflows");

    assert_eq!(
        err,
        BufferError::SizeOverflow {
            what: "strided output size",
        }
    );
}

#[test]
fn strided_output_len_counts_last_row_without_trailing_padding() {
    assert_eq!(
        strided_output_len((3, 4), 16, PixelFormat::Rgb8).expect("output length"),
        57
    );
    assert_eq!(
        strided_output_len((3, 0), 16, PixelFormat::Rgb8).expect("zero height"),
        0
    );
}

#[test]
fn validate_strided_output_buffer_reports_stride_and_output_errors() {
    assert_eq!(
        validate_strided_output_buffer((3, 4), 57, 8, PixelFormat::Rgb8)
            .expect_err("stride too small"),
        BufferError::StrideTooSmall {
            row_bytes: 9,
            stride: 8,
        }
    );

    assert_eq!(
        validate_strided_output_buffer((3, 4), 56, 16, PixelFormat::Rgb8)
            .expect_err("output too small"),
        BufferError::OutputTooSmall {
            required: 57,
            have: 56,
        }
    );
}

#[test]
fn tile_batch_worker_count_uses_available_workers_when_unspecified() {
    assert_eq!(
        j2k_core::tile_batch_worker_count(8, TileBatchOptions::default(), 4),
        4
    );
}

#[test]
fn tile_batch_worker_count_clamps_to_batch_size_and_at_least_one_worker() {
    let options = TileBatchOptions {
        workers: Some(NonZeroUsize::new(16).expect("nonzero")),
    };

    assert_eq!(j2k_core::tile_batch_worker_count(3, options, 8), 3);
    assert_eq!(
        j2k_core::tile_batch_worker_count(8, TileBatchOptions::default(), 0),
        1
    );
    assert_eq!(
        j2k_core::tile_batch_worker_count(0, TileBatchOptions::default(), 8),
        1
    );
}

#[test]
fn collect_indexed_batch_results_restores_input_order() {
    let results = vec![(2, Ok("two")), (0, Ok("zero")), (1, Ok("one"))];

    let outcomes =
        j2k_core::collect_indexed_batch_results(3, results, |index, source: &str| (index, source))
            .expect("ordered outcomes");

    assert_eq!(outcomes, ["zero", "one", "two"]);
}

#[test]
fn collect_indexed_batch_results_returns_first_error_by_input_index() {
    let results = vec![
        (2, Err("later")),
        (0, Ok("zero")),
        (1, Err("first")),
        (3, Ok("three")),
    ];

    let err = j2k_core::collect_indexed_batch_results(4, results, |index, source| (index, source))
        .expect_err("first failing input index");

    assert_eq!(err, (1, "first"));
}

#[test]
#[should_panic(expected = "indexed batch result index 3 outside job count 3")]
fn collect_indexed_batch_results_rejects_out_of_bounds_error_index() {
    let results = vec![(3, Err::<&str, _>("outside"))];

    let _ = j2k_core::collect_indexed_batch_results(3, results, |index, source| (index, source));
}

#[test]
fn backend_capabilities_resolve_auto_and_explicit_requests() {
    let caps = BackendCapabilities {
        cpu: CpuFeatures::default(),
        metal: true,
        cuda: false,
    };
    assert_eq!(caps.resolve(BackendRequest::Auto), Some(BackendKind::Cpu));
    assert_eq!(caps.resolve(BackendRequest::Cpu), Some(BackendKind::Cpu));
    assert_eq!(
        caps.resolve(BackendRequest::Metal),
        Some(BackendKind::Metal)
    );
    assert_eq!(caps.resolve(BackendRequest::Cuda), None);
    assert_eq!(caps.first_available_accelerator(), Some(BackendKind::Metal));
    assert!(caps.supports(BackendRequest::Metal));
    assert!(!caps.supports(BackendRequest::Cuda));
}

#[test]
fn backend_request_exposes_adaptive_cpu_and_strict_aliases() {
    assert_eq!(BackendRequest::ACCELERATED, BackendRequest::Auto);
    assert_eq!(BackendRequest::CPU_ONLY, BackendRequest::Cpu);
    assert_eq!(BackendRequest::STRICT_METAL, BackendRequest::Metal);
    assert_eq!(BackendRequest::STRICT_CUDA, BackendRequest::Cuda);
}

#[test]
fn decode_request_default_and_scaled_rects_are_stable() {
    let full = j2k_core::DecodeRequest::default();
    assert!(full.is_full_resolution_full_image());
    assert_eq!(full.decoded_rect((17, 9)), Rect::full((17, 9)));

    let roi = Rect {
        x: 3,
        y: 5,
        w: 9,
        h: 7,
    };
    let request = j2k_core::DecodeRequest::region_scaled(roi, Downscale::Quarter);
    assert_eq!(
        request.decoded_rect((99, 99)),
        Rect {
            x: 0,
            y: 1,
            w: 3,
            h: 2
        }
    );
}

#[test]
fn execution_stats_and_gpu_abi_helpers_are_stable() {
    let combined = j2k_core::ExecutionStats {
        submissions: 1,
        kernel_dispatches: 2,
        upload_bytes: 3,
        readback_bytes: 4,
        device_us: 5,
    }
    .saturating_add(j2k_core::ExecutionStats {
        submissions: u64::MAX,
        kernel_dispatches: 1,
        upload_bytes: 1,
        readback_bytes: 1,
        device_us: 1,
    });

    assert_eq!(combined.submissions, u64::MAX);
    assert_eq!(combined.kernel_dispatches, 3);

    let values = [1_u32, 2_u32];
    let bytes = <u32 as j2k_core::GpuAbi>::slice_as_bytes(&values);
    assert_eq!(bytes.len(), 8);
    assert_eq!(
        j2k_core::DeviceMemoryRange::new(j2k_core::BackendKind::Cuda, 7, 8, 9).len,
        9
    );
}

#[derive(Debug, Clone, Copy)]
struct DummySurface {
    backend: BackendKind,
    dims: (u32, u32),
    fmt: PixelFormat,
    len: usize,
}

impl DeviceSurface for DummySurface {
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dims
    }

    fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    fn byte_len(&self) -> usize {
        self.len
    }
}

#[test]
fn device_surface_contract_reports_metadata() {
    let surface = DummySurface {
        backend: BackendKind::Metal,
        dims: (32, 16),
        fmt: PixelFormat::Rgb8,
        len: 32 * 16 * 3,
    };
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (32, 16));
    assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(surface.byte_len(), 32 * 16 * 3);
}

#[test]
fn ready_submission_waits_immediate_success() {
    let submission = ReadySubmission::<u32, &'static str>::from_result(Ok(7));
    assert_eq!(submission.wait().expect("success"), 7);
}

#[test]
fn ready_submission_waits_immediate_error() {
    let submission = ReadySubmission::<u32, &'static str>::from_result(Err("nope"));
    assert_eq!(submission.wait().expect_err("error"), "nope");
}

#[derive(Default)]
struct DummyPool;

impl ScratchPool for DummyPool {
    fn bytes_allocated(&self) -> usize {
        0
    }

    fn reset(&mut self) {}
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("dummy decode error")]
struct DummyError;

impl CodecError for DummyError {
    fn is_truncated(&self) -> bool {
        false
    }

    fn is_not_implemented(&self) -> bool {
        false
    }

    fn is_unsupported(&self) -> bool {
        false
    }

    fn is_buffer_error(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy)]
struct DummyCodec;

#[derive(Default)]
struct DummyContext;

impl CodecContext for DummyContext {
    fn clear(&mut self) {}
}

impl ImageCodec for DummyCodec {
    type Error = DummyError;
    type Warning = core::convert::Infallible;
    type Pool = DummyPool;
}

struct DummyImageDecoder {
    submissions: usize,
}

impl ImageCodec for DummyImageDecoder {
    type Error = DummyError;
    type Warning = core::convert::Infallible;
    type Pool = DummyPool;
}

impl<'a> ImageDecode<'a> for DummyImageDecoder {
    type View = ();

    fn inspect(_input: &'a [u8]) -> Result<Info, Self::Error> {
        Err(DummyError)
    }

    fn parse(_input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(())
    }

    fn from_view(_view: Self::View) -> Result<Self, Self::Error> {
        Ok(Self { submissions: 0 })
    }

    fn decode_into(
        &mut self,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        Err(DummyError)
    }

    fn decode_into_with_scratch(
        &mut self,
        _pool: &mut Self::Pool,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        Err(DummyError)
    }

    fn decode_region_into(
        &mut self,
        _pool: &mut Self::Pool,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
        _roi: Rect,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        Err(DummyError)
    }

    fn decode_scaled_into(
        &mut self,
        _pool: &mut Self::Pool,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
        _scale: Downscale,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        Err(DummyError)
    }

    fn decode_region_scaled_into(
        &mut self,
        _pool: &mut Self::Pool,
        _out: &mut [u8],
        _stride: usize,
        _fmt: PixelFormat,
        _roi: Rect,
        _scale: Downscale,
    ) -> Result<j2k_core::DecodeOutcome<Self::Warning>, Self::Error> {
        Err(DummyError)
    }
}

impl ImageDecodeSubmit<'_> for DummyImageDecoder {
    type Session = usize;
    type DeviceSurface = DummySurface;
    type SubmittedSurface = ReadySubmission<DummySurface, DummyError>;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        self.submissions += 1;
        *session += 1;
        Ok(ReadySubmission::from_result(Ok(DummySurface {
            backend: backend_kind_for_request(backend),
            dims: (1, 1),
            fmt,
            len: fmt.bytes_per_pixel(),
        })))
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        self.submissions += 1;
        *session += 1;
        Ok(ReadySubmission::from_result(Ok(DummySurface {
            backend: backend_kind_for_request(backend),
            dims: (roi.w, roi.h),
            fmt,
            len: roi.w as usize * roi.h as usize * fmt.bytes_per_pixel(),
        })))
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        self.submissions += 1;
        *session += 1;
        let denom = scale.denominator();
        Ok(ReadySubmission::from_result(Ok(DummySurface {
            backend: backend_kind_for_request(backend),
            dims: (8 / denom, 8 / denom),
            fmt,
            len: (64 / (denom * denom)) as usize * fmt.bytes_per_pixel(),
        })))
    }

    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        self.submissions += 1;
        *session += 1;
        let scaled = roi.scaled_covering(scale);
        Ok(ReadySubmission::from_result(Ok(DummySurface {
            backend: backend_kind_for_request(backend),
            dims: (scaled.w, scaled.h),
            fmt,
            len: scaled.w as usize * scaled.h as usize * fmt.bytes_per_pixel(),
        })))
    }
}

impl ImageDecodeDevice<'_> for DummyImageDecoder {
    type DeviceSurface = DummySurface;
}

impl TileBatchDecodeSubmit for DummyCodec {
    type Context = DummyContext;
    type Session = usize;
    type DeviceSurface = DummySurface;
    type SubmittedSurface = ReadySubmission<DummySurface, DummyError>;

    fn submit_tile_to_device(
        _ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        _pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        *session += 1;
        let input_width = u32::try_from(input.len()).expect("dummy test input fits in u32");
        Ok(ReadySubmission::from_result(Ok(DummySurface {
            backend: backend_kind_for_request(backend),
            dims: (input_width, 1),
            fmt,
            len: input.len() * fmt.bytes_per_pixel(),
        })))
    }

    fn submit_tile_region_to_device(
        _ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        _pool: &mut Self::Pool,
        _input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        *session += 1;
        Ok(ReadySubmission::from_result(Ok(DummySurface {
            backend: backend_kind_for_request(backend),
            dims: (roi.w, roi.h),
            fmt,
            len: roi.w as usize * roi.h as usize * fmt.bytes_per_pixel(),
        })))
    }

    fn submit_tile_scaled_to_device(
        _ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        _pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        *session += 1;
        let denom = scale.denominator();
        let input_width = u32::try_from(input.len()).expect("dummy test input fits in u32");
        Ok(ReadySubmission::from_result(Ok(DummySurface {
            backend: backend_kind_for_request(backend),
            dims: (input_width.div_ceil(denom), 1),
            fmt,
            len: input.len() * fmt.bytes_per_pixel(),
        })))
    }

    fn submit_tile_region_scaled_to_device(
        _ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        _pool: &mut Self::Pool,
        _input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        *session += 1;
        let scaled = roi.scaled_covering(scale);
        Ok(ReadySubmission::from_result(Ok(DummySurface {
            backend: backend_kind_for_request(backend),
            dims: (scaled.w, scaled.h),
            fmt,
            len: scaled.w as usize * scaled.h as usize * fmt.bytes_per_pixel(),
        })))
    }
}

impl TileBatchDecodeDevice for DummyCodec {
    type Context = DummyContext;
    type DeviceSurface = DummySurface;
}

fn backend_kind_for_request(request: BackendRequest) -> BackendKind {
    match request {
        BackendRequest::Cuda => BackendKind::Cuda,
        BackendRequest::Metal => BackendKind::Metal,
        BackendRequest::Auto | BackendRequest::Cpu => BackendKind::Cpu,
    }
}

#[test]
fn image_decode_device_defaults_wait_on_submit_calls() {
    let mut decoder = DummyImageDecoder { submissions: 0 };

    let surface = decoder
        .decode_region_scaled_to_device(
            PixelFormat::Gray8,
            Rect {
                x: 2,
                y: 2,
                w: 5,
                h: 5,
            },
            Downscale::Half,
            BackendRequest::Metal,
        )
        .expect("default decode delegates through submit");

    assert_eq!(decoder.submissions, 1);
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (3, 3));
}

#[test]
fn tile_batch_decode_device_defaults_wait_on_submit_calls() {
    let mut ctx = DecoderContext::<DummyContext>::new();
    let mut pool = DummyPool;

    let surface = DummyCodec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        b"abc",
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("default tile decode delegates through submit");

    assert_eq!(surface.backend_kind(), BackendKind::Cuda);
    assert_eq!(surface.dimensions(), (3, 1));
    assert_eq!(surface.byte_len(), 9);
}

impl TileBatchDecodeManyDevice for DummyCodec {
    type Context = DummyContext;
    type DeviceSurface = DummySurface;

    fn decode_tiles_to_device(
        _ctx: &mut DecoderContext<Self::Context>,
        _pool: &mut Self::Pool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Vec<Self::DeviceSurface>, Self::Error> {
        Ok(inputs
            .iter()
            .map(|input| DummySurface {
                backend: match backend {
                    BackendRequest::Cuda => BackendKind::Cuda,
                    BackendRequest::Metal => BackendKind::Metal,
                    BackendRequest::Auto | BackendRequest::Cpu => BackendKind::Cpu,
                },
                dims: (
                    u32::try_from(input.len()).expect("dummy input length fits in u32"),
                    1,
                ),
                fmt,
                len: input.len() * fmt.bytes_per_pixel(),
            })
            .collect())
    }
}

#[test]
fn tile_batch_decode_many_device_returns_ordered_surfaces() {
    let mut ctx = DecoderContext::<DummyContext>::new();
    let mut pool = DummyPool;
    let inputs: [&[u8]; 2] = [b"abc".as_slice(), b"abcdef".as_slice()];

    let surfaces = DummyCodec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("batch surfaces");

    assert_eq!(surfaces.len(), 2);
    assert_eq!(surfaces[0].backend_kind(), BackendKind::Cuda);
    assert_eq!(surfaces[0].dimensions(), (3, 1));
    assert_eq!(surfaces[1].dimensions(), (6, 1));
    assert_eq!(surfaces[1].byte_len(), 18);
}

fn passthrough_info() -> Info {
    Info {
        dimensions: (512, 512),
        components: 3,
        colorspace: Colorspace::SRgb,
        bit_depth: 8,
        tile_layout: Some(TileLayout {
            tile_width: 512,
            tile_height: 512,
            tiles_x: 1,
            tiles_y: 1,
        }),
        coded_unit_layout: Some(CodedUnitLayout {
            unit_width: 512,
            unit_height: 512,
            units_x: 1,
            units_y: 1,
        }),
        restart_interval: None,
        resolution_levels: 1,
    }
}

#[test]
fn passthrough_candidate_copies_when_syntax_payload_and_metadata_match() {
    let bytes = [0xff, 0x4f, 0xff, 0xd9];
    let candidate = PassthroughCandidate::new(
        &bytes,
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
        passthrough_info(),
    );
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    )
    .with_dimensions((512, 512))
    .with_components(3)
    .with_bit_depth(8)
    .with_colorspace(Colorspace::SRgb);

    assert_eq!(
        candidate.evaluate(&requirements),
        PassthroughDecision::Copy { bytes: &bytes }
    );
    assert!(core::ptr::eq(
        candidate
            .copy_bytes_if_eligible(&requirements)
            .expect("eligible bytes")
            .as_ptr(),
        bytes.as_ptr()
    ));
}

#[test]
fn passthrough_candidate_rejects_transfer_syntax_mismatch_before_metadata() {
    let bytes = [0xff, 0x4f, 0xff, 0xd9];
    let candidate = PassthroughCandidate::new(
        &bytes,
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
        passthrough_info(),
    );
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::HtJpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    )
    .with_dimensions((256, 256));

    assert_eq!(
        candidate.evaluate(&requirements),
        PassthroughDecision::Transcode {
            reason: PassthroughRejectReason::TransferSyntaxMismatch {
                source: CompressedTransferSyntax::Jpeg2000Lossless,
                destination: CompressedTransferSyntax::HtJpeg2000Lossless,
            }
        }
    );
}

#[test]
fn passthrough_candidate_rejects_jp2_container_for_dicom_codestream_payload() {
    let bytes = [0, 0, 0, 12, b'j', b'P', b' ', b' '];
    let candidate = PassthroughCandidate::new(
        &bytes,
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jp2File,
        passthrough_info(),
    );
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    );

    assert_eq!(
        candidate.copy_bytes_if_eligible(&requirements),
        Err(PassthroughRejectReason::PayloadKindMismatch {
            source: CompressedPayloadKind::Jp2File,
            destination: CompressedPayloadKind::Jpeg2000Codestream,
        })
    );
}

#[test]
fn passthrough_requirements_can_match_large_jpeg2000_component_counts() {
    let bytes = [0xff, 0x4f, 0xff, 0xd9];
    let mut info = passthrough_info();
    info.components = 256;
    let candidate = PassthroughCandidate::new(
        &bytes,
        CompressedTransferSyntax::HtJpeg2000Lossless,
        CompressedPayloadKind::JphFile,
        info,
    );
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::HtJpeg2000Lossless,
        CompressedPayloadKind::JphFile,
    )
    .with_component_count(256);

    assert_eq!(
        candidate.copy_bytes_if_eligible(&requirements),
        Ok(bytes.as_slice())
    );
}
