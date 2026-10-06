// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaSession;
use crate::Error;

fn cuda_required() -> bool {
    std::env::var_os("J2K_REQUIRE_CUDA_RUNTIME").is_some()
}

#[test]
fn uninitialized_decode_pool_diagnostics_are_empty() {
    let diagnostics = CudaSession::default()
        .decode_pool_diagnostics()
        .expect("empty session diagnostics");
    assert!(diagnostics.decode.is_none());
    assert!(diagnostics.batch_decode.is_none());
    assert_eq!(diagnostics.retained_bytes(), 0);
}

#[test]
fn htj2k_decode_tables_are_uploaded_once_per_session() {
    crate::session::reset_htj2k_decode_table_uploads_for_test();
    let mut session = CudaSession::default();

    let first = session.htj2k_decode_table_resources();
    if matches!(
        first,
        Err(Error::CudaUnavailable | Error::CudaRuntime { .. })
    ) && !cuda_required()
    {
        return;
    }
    first.expect("first HTJ2K decode table upload");
    session
        .htj2k_decode_table_resources()
        .expect("cached HTJ2K decode tables");

    assert_eq!(crate::session::htj2k_decode_table_uploads_for_test(), 1);
}

#[test]
fn classic_decode_tables_are_uploaded_once_per_session() {
    crate::session::reset_classic_decode_table_uploads_for_test();
    let mut session = CudaSession::default();

    let first = session.classic_decode_table_resources();
    if matches!(
        first,
        Err(Error::CudaUnavailable | Error::CudaRuntime { .. })
    ) && !cuda_required()
    {
        return;
    }
    first.expect("first classic decode table upload");
    session
        .classic_decode_table_resources()
        .expect("cached classic decode tables");

    assert_eq!(crate::session::classic_decode_table_uploads_for_test(), 1);
}

#[test]
fn cuda_session_reuses_one_decode_buffer_pool_when_required() {
    let mut session = CudaSession::default();

    let first = session.decode_buffer_pool();
    if matches!(
        first,
        Err(Error::CudaUnavailable | Error::CudaRuntime { .. })
    ) && !cuda_required()
    {
        return;
    }
    let first = first.expect("first decode buffer pool");
    let second = session
        .decode_buffer_pool()
        .expect("cached decode buffer pool");
    {
        let buffer = first.take(16).expect("pooled decode buffer");
        assert_eq!(buffer.byte_len(), 16);
    }

    assert!(second.cached_count().expect("shared pool cached count") >= 1);
}

#[test]
fn warm_dense_decode_batch_does_not_allocate_per_image_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let (fixture, pixels) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(128, 128);
    let mut allocation_counts = Vec::new();
    for batch_size in [1, 16] {
        let inputs = j2k_core::try_host_vec_filled(batch_size, fixture.as_slice())
            .expect("batch fixture inputs");
        let mut session = CudaSession::default();
        for _ in 0..3 {
            let surfaces = crate::J2kDecoder::decode_batch_to_device_with_session(
                &inputs,
                j2k_core::PixelFormat::Rgb8,
                &mut session,
            )
            .expect("warm dense batch");
            let actual = crate::Surface::download_batch_tight(&surfaces).expect("batch readback");
            assert_eq!(actual, pixels.repeat(inputs.len()));
        }
        let before = session.diagnostics().expect("warm diagnostics");
        drop(
            crate::J2kDecoder::decode_batch_to_device_with_session(
                &inputs,
                j2k_core::PixelFormat::Rgb8,
                &mut session,
            )
            .expect("reuse dense batch"),
        );
        let after = session.diagnostics().expect("reused diagnostics");
        allocation_counts.push(
            after
                .runtime
                .expect("CUDA runtime")
                .device_allocation_operations
                - before
                    .runtime
                    .expect("CUDA runtime")
                    .device_allocation_operations,
        );
        let pool = after.pools.batch_decode.expect("batch pool");
        assert_eq!(pool.deferred_buffers, 0);
        assert_eq!(pool.reuse_holds, 0);
        assert!(pool.cached_bytes <= super::DECODE_BATCH_POOL_MAX_CACHED_BYTES);
    }
    // This API returns a fresh owned output allocation and descriptor buffer.
    // Its coefficient/IDWT scratch must be reused even as the batch grows.
    assert_eq!(
        allocation_counts[1], allocation_counts[0],
        "warm scratch allocations within the byte budget must not scale with batch size"
    );
}
