// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_core::DeviceSubmission;

fn ht_fixture(components: u16) -> (Vec<u8>, Vec<u8>) {
    let pixels = j2k_test_support::gradient_u8(640, 480, usize::from(components));
    let bytes = j2k_native::encode_htj2k(
        &pixels,
        640,
        480,
        components,
        8,
        false,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode session-routing fixture");
    (bytes, pixels)
}

fn session_with_queue() -> MetalBackendSession {
    let device = j2k_metal_support::system_default_device().expect("Metal device");
    let queue = j2k_metal_support::checked_command_queue(&device).expect("caller command queue");
    let session = MetalBackendSession::with_command_queue(device, queue.clone())
        .expect("session with caller queue");
    assert!(session.uses_command_queue(&queue).expect("session queue"));
    session
}

fn assert_session_used(session: &MetalBackendSession, bytes: &[u8], fmt: PixelFormat) {
    use crate::session::direct_plan_cache::{
        cached_session_direct_color_plan, cached_session_direct_gray_plan,
        direct_gray_plan_cache_key, direct_plan_cache_key,
    };

    let has_plan = match fmt {
        PixelFormat::Gray8 => {
            cached_session_direct_gray_plan(session, direct_gray_plan_cache_key(bytes, fmt))
                .expect("session grayscale cache")
                .is_some()
        }
        PixelFormat::Rgb8 => {
            cached_session_direct_color_plan(session, direct_plan_cache_key(bytes, fmt))
                .expect("session color cache")
                .is_some()
        }
        _ => unreachable!("fixture formats"),
    };
    assert!(has_plan, "{fmt:?} plan must belong to the supplied session");
    let pools = session
        .buffer_pool_diagnostics()
        .expect("session scratch pools");
    assert!(
        pools.private.peak_cached_buffers + pools.shared.peak_cached_buffers > 0,
        "{fmt:?} execution must use the runtime attached to the caller queue"
    );
}

#[test]
fn single_auto_decode_preserves_supplied_session() {
    if !should_run_metal_runtime() {
        return;
    }
    for (components, fmt) in [(1, PixelFormat::Gray8), (3, PixelFormat::Rgb8)] {
        let (bytes, pixels) = ht_fixture(components);
        let session = session_with_queue();
        let surface = J2kDecoder::new(&bytes)
            .expect("decoder")
            .decode_request_to_device_with_session(
                MetalDecodeRequest::full(fmt, BackendRequest::Auto),
                &session,
            )
            .expect("Auto decode with supplied session");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.as_bytes().expect("decoded samples"), pixels);
        assert_session_used(&session, &bytes, fmt);
    }
}

#[test]
fn repeated_auto_decode_preserves_supplied_session() {
    if !should_run_metal_runtime() {
        return;
    }
    for (components, fmt) in [(1, PixelFormat::Gray8), (3, PixelFormat::Rgb8)] {
        let (bytes, pixels) = ht_fixture(components);
        let input = Arc::<[u8]>::from(bytes);
        let backend = session_with_queue();
        let mut session = crate::MetalSession::with_backend_session(backend.clone());
        let submissions = (0..16)
            .map(|_| {
                batch::queue_tile_request_shared_with_retained(
                    &mut session,
                    input.clone(),
                    fmt,
                    BackendRequest::Auto,
                    batch::BatchOp::Full,
                    0,
                )
                .expect("queue Auto decode")
            })
            .collect::<Vec<_>>();
        for (index, submission) in submissions.into_iter().enumerate() {
            let surface = submission.wait().expect("repeated Auto decode");
            assert_eq!(surface.backend_kind(), BackendKind::Metal);
            if index == 0 {
                assert_eq!(surface.as_bytes().expect("decoded samples"), pixels);
            }
        }
        assert_eq!(session.submissions().expect("submission count"), 1);
        assert_session_used(&backend, &input, fmt);
    }
}
