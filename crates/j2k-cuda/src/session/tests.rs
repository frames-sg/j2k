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
