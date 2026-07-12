// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    validate_transcode_pool_context, validate_transcode_pool_context_match,
    TRANSCODE_POOL_CONTEXT_MISMATCH,
};
use crate::{
    CudaBufferPool, CudaContext, CudaDwt97BatchGeometry, CudaDwt97BatchWithPoolRequest, CudaError,
    CudaHtj2k97CodeblockBatchWithPoolRequest, CudaHtj2k97I16CodeblockBatchWithPoolRequest,
    CudaHtj2k97QuantizeParams,
};

fn assert_transcode_pool_context_mismatch<T>(result: Result<T, CudaError>) {
    match result {
        Err(CudaError::InvalidArgument { message }) => {
            assert_eq!(message, TRANSCODE_POOL_CONTEXT_MISMATCH);
        }
        Err(error) => panic!("expected a CUDA transcode pool context mismatch, got {error}"),
        Ok(_) => panic!("expected a CUDA transcode pool context mismatch"),
    }
}

fn empty_geometry() -> CudaDwt97BatchGeometry {
    CudaDwt97BatchGeometry {
        item_count: 0,
        block_cols: 0,
        block_rows: 0,
        width: 0,
        height: 0,
    }
}

fn quantize_params() -> CudaHtj2k97QuantizeParams {
    CudaHtj2k97QuantizeParams {
        inv_delta_ll: 1.0,
        inv_delta_hl: 1.0,
        inv_delta_lh: 1.0,
        inv_delta_hh: 1.0,
        cb_width: 1,
        cb_height: 1,
    }
}

fn f32_request(pool: &CudaBufferPool) -> CudaHtj2k97CodeblockBatchWithPoolRequest<'_> {
    CudaHtj2k97CodeblockBatchWithPoolRequest {
        blocks: &[],
        geometry: empty_geometry(),
        params: quantize_params(),
        pool,
    }
}

#[test]
fn transcode_pool_context_validation_accepts_match_and_rejects_mismatch() {
    assert!(validate_transcode_pool_context_match(true).is_ok());
    assert_transcode_pool_context_mismatch(validate_transcode_pool_context_match(false));
}

#[test]
fn every_public_transcode_pool_api_rejects_a_foreign_pool_first() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("launch CUDA context");
    let foreign_context = CudaContext::system_default().expect("foreign CUDA context");
    let local_pool = context.buffer_pool();
    let foreign_pool = foreign_context.buffer_pool();
    assert!(validate_transcode_pool_context(&context, &local_pool).is_ok());

    assert_transcode_pool_context_mismatch(context.j2k_transcode_dwt97_batch_with_pool(
        CudaDwt97BatchWithPoolRequest {
            blocks: &[],
            geometry: empty_geometry(),
            pool: &foreign_pool,
        },
    ));
    assert_transcode_pool_context_mismatch(
        context.j2k_transcode_htj2k97_codeblock_batch_with_pool(f32_request(&foreign_pool)),
    );
    assert_transcode_pool_context_mismatch(
        context
            .j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(f32_request(&foreign_pool)),
    );
    assert_transcode_pool_context_mismatch(
        context.j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
            CudaHtj2k97I16CodeblockBatchWithPoolRequest {
                blocks: &[],
                geometry: empty_geometry(),
                params: quantize_params(),
                pool: &foreign_pool,
            },
        ),
    );
}
