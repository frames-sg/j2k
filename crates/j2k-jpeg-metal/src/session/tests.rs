// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[path = "tests/ledger.rs"]
mod ledger;

#[cfg(target_os = "macos")]
const BASELINE_420: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_420_16x16.jpg");
#[cfg(target_os = "macos")]
const BASELINE_422: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_422_16x8.jpg");
#[cfg(target_os = "macos")]
const BASELINE_444: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_444_8x8.jpg");

#[cfg(target_os = "macos")]
#[test]
fn repeated_plan_hits_share_input_packet_and_eager_shape() {
    let mut session = SessionState::default();
    let first = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("first plan");
    let second = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("second plan");

    assert_eq!(first.shape, second.shape);
    assert_eq!(first.shape.sampling_family, batch::SamplingFamily::Fast420);
    assert!(SharedJpegInput::ptr_eq(&first.input, &second.input));
    assert!(SharedJpegFastPacket::ptr_eq(
        first.fast_packet.as_ref().expect("first packet"),
        second.fast_packet.as_ref().expect("second packet"),
    ));
    let diagnostics = session.jpeg_plan_cache_diagnostics();
    assert_eq!(diagnostics.entries, 1);
    assert_eq!(diagnostics.misses, 1);
    assert_eq!(diagnostics.hits, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn inspected_plan_tracks_fast422_sampling_family() {
    let mut session = SessionState::default();
    let plan = session
        .resolve_jpeg_plan(BASELINE_422, BackendRequest::Metal)
        .expect("fast422 plan");

    assert_eq!(plan.shape.sampling_family, batch::SamplingFamily::Fast422);
    assert!(plan
        .fast_packet
        .as_ref()
        .is_some_and(|packet| packet.fast422().is_some()));
}

#[cfg(target_os = "macos")]
#[test]
fn reused_source_pointer_with_new_valid_bytes_never_cross_hits() {
    let mut session = SessionState::default();
    let mut buffer = BASELINE_420.to_vec();
    let source_pointer = buffer.as_ptr();
    let first = session
        .resolve_jpeg_plan(&buffer, BackendRequest::Metal)
        .expect("first plan");

    buffer[..BASELINE_444.len()].copy_from_slice(BASELINE_444);
    buffer[BASELINE_444.len()..].fill(0);
    assert_eq!(buffer.as_ptr(), source_pointer);
    let second = session
        .resolve_jpeg_plan(&buffer, BackendRequest::Metal)
        .expect("overwritten plan");

    assert_eq!(first.shape.sampling_family, batch::SamplingFamily::Fast420);
    assert_eq!(second.shape.sampling_family, batch::SamplingFamily::Fast444);
    assert!(!SharedJpegInput::ptr_eq(&first.input, &second.input));
    let diagnostics = session.jpeg_plan_cache_diagnostics();
    assert_eq!(diagnostics.entries, 2);
    assert_eq!(diagnostics.misses, 2);
    assert_eq!(diagnostics.hits, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn oversized_plan_is_returned_without_retention_and_diagnostics_are_stable() {
    let mut session = SessionState {
        jpeg_plans: JpegPlanCache::with_limits(8, 1),
        ..SessionState::default()
    };

    let first = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("oversized plan remains usable");
    let first_diagnostics = session.jpeg_plan_cache_diagnostics();
    assert_eq!(first_diagnostics.entries, 0);
    assert_eq!(first_diagnostics.retained_bytes, 0);
    assert_eq!(first_diagnostics.metadata_capacity_bytes, 0);
    assert_eq!(first_diagnostics.oversized_rejections, 1);
    assert_eq!(first_diagnostics.misses, 1);

    let second = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("second oversized plan remains usable");
    assert!(!SharedJpegInput::ptr_eq(&first.input, &second.input));
    let second_diagnostics = session.jpeg_plan_cache_diagnostics();
    assert_eq!(second_diagnostics.entries, 0);
    assert_eq!(second_diagnostics.oversized_rejections, 2);
    assert_eq!(second_diagnostics.misses, 2);
    assert_eq!(second_diagnostics.peak_entries, 0);
    assert_eq!(second_diagnostics.peak_bytes, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn malformed_plan_errors_are_typed_and_never_cached() {
    let mut session = SessionState::default();
    let decode = session
        .resolve_jpeg_plan(&[0xff, 0xd8], BackendRequest::Metal)
        .expect_err("truncated JPEG must fail");
    assert!(matches!(decode, Error::Decode(_)));

    let invalid_table = rewrite_first_sof_quant_table_selector(BASELINE_420.to_vec(), u8::MAX);
    let packet = session
        .resolve_jpeg_plan(&invalid_table, BackendRequest::Metal)
        .expect_err("invalid packet table must fail");
    assert!(matches!(packet, Error::FastPacket { .. }));
    assert_eq!(session.jpeg_plan_cache_diagnostics().entries, 0);
    assert_eq!(session.jpeg_plan_cache_diagnostics().misses, 2);
}

#[test]
fn cpu_and_cuda_requests_copy_without_inspection_or_cache_admission() {
    let mut session = SessionState::default();
    for backend in [BackendRequest::Cpu, BackendRequest::Cuda] {
        let plan = session
            .resolve_jpeg_plan(b"not a jpeg", backend)
            .expect("non-Metal request remains uninspected");
        assert_eq!(plan.input.as_bytes(), b"not a jpeg");
        assert!(plan.fast_packet.is_none());
        assert_eq!(plan.shape, batch::BatchShape::unknown());
    }

    let diagnostics = session.jpeg_plan_cache_diagnostics();
    assert_eq!(diagnostics.entries, 0);
    assert_eq!(diagnostics.hits, 0);
    assert_eq!(diagnostics.misses, 0);
}

#[test]
fn public_session_diagnostics_report_default_empty_limits() {
    let session = MetalSession::default();
    let diagnostics = session
        .jpeg_plan_cache_diagnostics()
        .expect("session diagnostics");

    assert_eq!(diagnostics.entries, 0);
    assert_eq!(
        diagnostics.entry_limit,
        j2k_jpeg::adapter::DEFAULT_JPEG_PLAN_CACHE_ENTRIES
    );
    assert_eq!(
        diagnostics.host_byte_limit,
        j2k_jpeg::adapter::DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn non_macos_auto_and_metal_plan_resolution_stays_unparsed() {
    let mut session = SessionState::default();
    for backend in [BackendRequest::Auto, BackendRequest::Metal] {
        let plan = session
            .resolve_jpeg_plan(b"not a jpeg", backend)
            .expect("non-macOS plan stays unparsed");
        assert_eq!(plan.shape, batch::BatchShape::unknown());
        assert!(plan.fast_packet.is_none());
    }
    assert_eq!(session.jpeg_plan_cache_diagnostics().entries, 0);
    assert_eq!(session.jpeg_plan_cache_diagnostics().misses, 0);
}

#[cfg(target_os = "macos")]
fn rewrite_first_sof_quant_table_selector(mut bytes: Vec<u8>, selector: u8) -> Vec<u8> {
    let mut position = 2_usize;
    while position + 4 <= bytes.len() {
        assert_eq!(bytes[position], 0xff, "fixture marker alignment");
        let marker = bytes[position + 1];
        position += 2;
        let length = usize::from(u16::from_be_bytes([bytes[position], bytes[position + 1]]));
        let payload_start = position + 2;
        if matches!(marker, 0xc0..=0xc3) {
            bytes[payload_start + 8] = selector;
            return bytes;
        }
        position += length;
    }
    panic!("fixture must contain a supported SOF marker");
}
