// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use j2k_core::BackendKind;

use super::super::native::map_native_resident_encode_error;
use super::encode_j2k_lossless_resident_with_accelerator;
use crate::{
    encode_j2k_lossless_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kEncodeValidation, J2kHtj2kTileEncodeJob,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kMarkerSegment, J2kResidentEncodeInput,
    J2kResidentEncodeInputError, J2kResidentHtj2kTileEncodeJob, ReversibleTransform,
};

#[derive(Debug, Clone, Copy)]
enum MockResponse {
    Success(&'static [u8]),
    Decline,
    Error(&'static str),
}

#[test]
fn resident_resource_errors_preserve_native_encode_source() {
    let error = map_native_resident_encode_error(j2k_native::ResidentHtj2kEncodeError::Resource(
        j2k_native::EncodeError::AllocationTooLarge {
            what: "resident output fixture",
            requested: 9,
            cap: 8,
        },
    ));

    assert!(matches!(
        error,
        crate::J2kError::NativeEncode {
            context: "native JPEG 2000 resident lossless encode failed",
            source,
        }
        if source == crate::NativeBackendError::encode(
            j2k_native::EncodeError::AllocationTooLarge {
                what: "resident output fixture",
                requested: 9,
                cap: 8,
            },
        )
    ));
}

#[test]
fn resident_invalid_input_preserves_native_encode_category() {
    let error = map_native_resident_encode_error(
        j2k_native::ResidentHtj2kEncodeError::InvalidInput("invalid resident fixture"),
    );

    assert!(matches!(
        error,
        crate::J2kError::NativeEncode {
            context: "native JPEG 2000 resident lossless encode failed",
            source,
        }
        if source == crate::NativeBackendError::encode(
            j2k_native::EncodeError::InvalidInput {
                what: "invalid resident fixture",
            },
        )
    ));
}

#[derive(Debug)]
struct MockResidentAccelerator {
    response: MockResponse,
    dispatch: J2kEncodeDispatchReport,
    attempts: usize,
    observed_input: Option<J2kResidentEncodeInput>,
    report_dispatches: bool,
}

impl MockResidentAccelerator {
    fn new(response: MockResponse) -> Self {
        Self {
            response,
            dispatch: J2kEncodeDispatchReport::default(),
            attempts: 0,
            observed_input: None,
            report_dispatches: true,
        }
    }

    fn without_dispatch_report(mut self) -> Self {
        self.report_dispatches = false;
        self
    }

    fn complete_dispatch(&mut self, levels: u8, use_mct: bool) {
        if !self.report_dispatches {
            return;
        }
        self.dispatch.deinterleave += 1;
        self.dispatch.quantize_subband += 1;
        self.dispatch.ht_code_block += 1;
        self.dispatch.packetization += 1;
        if levels > 0 {
            self.dispatch.forward_dwt53 += 1;
        }
        if use_mct {
            self.dispatch.forward_rct += 1;
        }
    }

    fn respond(
        &mut self,
        input: J2kResidentEncodeInput,
        levels: u8,
        use_mct: bool,
    ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
        self.attempts += 1;
        self.observed_input = Some(input);
        match self.response {
            MockResponse::Success(tile_body) => {
                self.complete_dispatch(levels, use_mct);
                Ok(Some(tile_body.to_vec()))
            }
            MockResponse::Decline => Ok(None),
            MockResponse::Error(error) => {
                Err(crate::J2kEncodeStageError::internal_invariant(error))
            }
        }
    }
}

impl J2kEncodeStageAccelerator for MockResidentAccelerator {
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        self.dispatch
    }

    fn encode_htj2k_tile(
        &mut self,
        job: J2kHtj2kTileEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
        let input = J2kResidentEncodeInput::new(
            job.width,
            job.height,
            job.num_components,
            job.bit_depth,
            job.signed,
        )
        .map_err(|error| crate::J2kEncodeStageError::invalid_request(error.reason()))?;
        self.respond(input, job.num_decomposition_levels, job.use_mct)
    }

    fn encode_resident_htj2k_tile(
        &mut self,
        job: J2kResidentHtj2kTileEncodeJob<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<u8>>> {
        self.respond(job.input, job.num_decomposition_levels, job.use_mct)
    }
}

fn resident_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::new(
        EncodeBackendPreference::RequireDevice,
        J2kBlockCodingMode::HighThroughput,
        crate::J2kProgressionOrder::Lrcp,
        Some(0),
        ReversibleTransform::None53,
        J2kEncodeValidation::External,
    )
}

#[test]
fn resident_encode_matches_host_hook_codestream_and_reports_dispatch() {
    const TILE_BODY: &[u8] = &[0x80, 0x00, 0x7f];
    let input = J2kResidentEncodeInput::new(8, 8, 1, 8, false).expect("valid resident input");
    let options = resident_options();
    let mut resident_accelerator = MockResidentAccelerator::new(MockResponse::Success(TILE_BODY));
    let resident = encode_j2k_lossless_resident_with_accelerator(
        input,
        &options,
        BackendKind::Cuda,
        &mut resident_accelerator,
    )
    .expect("resident encode");

    let pixels = vec![0_u8; 8 * 8];
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).expect("host samples");
    let mut host_accelerator = MockResidentAccelerator::new(MockResponse::Success(TILE_BODY));
    let host = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Cuda,
        &mut host_accelerator,
    )
    .expect("host whole-tile encode");

    assert_eq!(resident.codestream, host.codestream);
    assert_eq!(resident.backend, BackendKind::Cuda);
    assert_eq!(resident.dispatch_report.deinterleave, 1);
    assert_eq!(resident.dispatch_report.quantize_subband, 1);
    assert_eq!(resident.dispatch_report.ht_code_block, 1);
    assert_eq!(resident.dispatch_report.packetization, 1);
    assert_eq!(resident_accelerator.attempts, 1);
    assert_eq!(resident_accelerator.observed_input, Some(input));
}

#[test]
fn resident_encode_decline_and_accelerator_error_are_explicit() {
    let input = J2kResidentEncodeInput::new(8, 8, 1, 8, false).expect("valid resident input");
    let options = resident_options();
    let mut declining = MockResidentAccelerator::new(MockResponse::Decline);
    let decline = encode_j2k_lossless_resident_with_accelerator(
        input,
        &options,
        BackendKind::Cuda,
        &mut declining,
    )
    .expect_err("resident decline must fail closed");
    assert!(
        decline
            .to_string()
            .contains("resident HTJ2K tile accelerator declined encode"),
        "unexpected decline error: {decline}"
    );
    assert_eq!(declining.attempts, 1);

    let mut failing = MockResidentAccelerator::new(MockResponse::Error("mock resident failure"));
    let failure = encode_j2k_lossless_resident_with_accelerator(
        input,
        &options,
        BackendKind::Cuda,
        &mut failing,
    )
    .expect_err("resident accelerator error must surface");
    assert!(
        failure.to_string().contains("mock resident failure"),
        "unexpected accelerator error: {failure}"
    );
    assert_eq!(failing.attempts, 1);
}

#[test]
fn resident_encode_rejects_success_without_required_dispatch_accounting() {
    let input = J2kResidentEncodeInput::new(8, 8, 1, 8, false).expect("valid resident input");
    let mut accelerator =
        MockResidentAccelerator::new(MockResponse::Success(&[])).without_dispatch_report();
    let options = resident_options().with_accelerated_backend();
    let error = encode_j2k_lossless_resident_with_accelerator(
        input,
        &options,
        BackendKind::Cuda,
        &mut accelerator,
    )
    .expect_err("unreported resident work must not be labeled CUDA");
    assert!(
        error.to_string().contains("did not dispatch deinterleave"),
        "unexpected dispatch error: {error}"
    );
}

fn assert_contract_rejection(
    input: J2kResidentEncodeInput,
    options: J2kLosslessEncodeOptions,
    expected: &str,
) {
    let mut accelerator = MockResidentAccelerator::new(MockResponse::Success(&[]));
    let error = encode_j2k_lossless_resident_with_accelerator(
        input,
        &options,
        BackendKind::Cuda,
        &mut accelerator,
    )
    .expect_err("incompatible resident option must fail before dispatch");
    assert!(
        error.to_string().contains(expected),
        "unexpected resident option error: {error}"
    );
    assert_eq!(accelerator.attempts, 0);
}

#[test]
fn resident_encode_preserves_strict_option_contract_without_host_fallback() {
    let input = J2kResidentEncodeInput::new(64, 64, 1, 8, false).expect("valid resident input");
    assert_contract_rejection(
        input,
        resident_options().with_cpu_only_backend(),
        "no CPU input or fallback",
    );
    assert_contract_rejection(
        input,
        resident_options().with_block_coding_mode(J2kBlockCodingMode::Classic),
        "requires HTJ2K block coding",
    );
    assert_contract_rejection(
        input,
        resident_options().with_validation(J2kEncodeValidation::CpuRoundTrip),
        "requires external validation",
    );
    assert_contract_rejection(
        input,
        resident_options().with_tile_size(Some((32, 32))),
        "requires a single whole-image tile",
    );
    assert_contract_rejection(
        input,
        resident_options().with_marker_segments(&[J2kMarkerSegment::Ppm]),
        "options require the staged host pipeline",
    );
    assert_contract_rejection(
        input,
        resident_options().with_quality_layers(2),
        "options require the staged host pipeline",
    );
}

#[test]
fn resident_encode_rejects_cpu_backend_kind_before_dispatch() {
    let input = J2kResidentEncodeInput::new(8, 8, 1, 8, false).expect("valid resident input");
    let mut accelerator = MockResidentAccelerator::new(MockResponse::Success(&[]));
    let error = encode_j2k_lossless_resident_with_accelerator(
        input,
        &resident_options(),
        BackendKind::Cpu,
        &mut accelerator,
    )
    .expect_err("resident input cannot be attributed to the CPU backend");
    assert!(
        error.to_string().contains("requires a device backend kind"),
        "unexpected backend-kind error: {error}"
    );
    assert_eq!(accelerator.attempts, 0);
}

#[test]
fn resident_input_rejects_invalid_geometry_without_calling_accelerator() {
    assert_eq!(
        J2kResidentEncodeInput::new(0, 8, 1, 8, false),
        Err(J2kResidentEncodeInputError::EmptyGeometry {
            width: 0,
            height: 8,
        })
    );
    assert_eq!(
        J2kResidentEncodeInput::new(8, 0, 1, 8, false),
        Err(J2kResidentEncodeInputError::EmptyGeometry {
            width: 8,
            height: 0,
        })
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn resident_encode_handles_huge_logical_geometry_without_host_image_allocation() {
    let input = J2kResidentEncodeInput::new(100_000, 100_000, 1, 8, false)
        .expect("huge geometry is addressable without allocating it");
    let mut accelerator = MockResidentAccelerator::new(MockResponse::Success(&[]));
    let encoded = encode_j2k_lossless_resident_with_accelerator(
        input,
        &resident_options(),
        BackendKind::Cuda,
        &mut accelerator,
    )
    .expect("resident geometry must not materialize host pixels");
    assert_eq!((encoded.width, encoded.height), (100_000, 100_000));
    assert!(encoded.codestream.len() < 1_024);
    assert_eq!(accelerator.attempts, 1);
}
