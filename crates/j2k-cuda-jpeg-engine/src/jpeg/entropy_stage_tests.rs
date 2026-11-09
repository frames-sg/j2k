// SPDX-License-Identifier: MIT OR Apache-2.0

//! The entropy stage alone: Huffman coding, DC prediction, restart intervals
//! and byte stuffing of fixed coefficients produce the CPU encoder's entropy
//! bytes. See `j2k_test_support::jpeg_entropy_stage_coefficients`.
//!
//! The engine is published before j2k-jpeg, so it cannot dev-depend on the
//! CPU encoder. The test takes the standard tables from j2k-test-support and
//! checks the shared golden of the CPU's entropy segments, which j2k-jpeg's
//! tests tie to the CPU encoder.

use j2k_test_support::{
    jpeg_entropy_stage_cases, jpeg_entropy_stage_coefficients, jpeg_entropy_stage_digest,
    jpeg_standard_tables, JpegHuffmanBySymbol, JPEG_ENTROPY_STAGE_SHA256,
    JPEG_ENTROPY_STAGE_SIZE as SIZE,
};

use super::encode_launch::CudaJpegBaselineHuffmanLaunch;
use super::encode_staging_launch::CudaJpegBaselineEntropyFromCoeffsLaunch;
use super::CudaJpegBaselineEncodeStatus;
use crate::bytes::{
    cuda_jpeg_baseline_encode_huffman_table_as_bytes, cuda_jpeg_baseline_encode_params_as_bytes,
    cuda_jpeg_baseline_encode_statuses_as_bytes, cuda_jpeg_baseline_encode_statuses_as_bytes_mut,
};
use crate::kernels::CudaLaunchGeometry;
use crate::{
    CudaContext, CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams, JpegCudaEngine,
};

fn huffman(&(codes, lens): &JpegHuffmanBySymbol) -> CudaJpegBaselineEncodeHuffmanTable {
    CudaJpegBaselineEncodeHuffmanTable { codes, lens }
}

/// Unsubsampled single-tile parameters: one block per component per MCU.
fn params(components: u32, restart_interval: Option<u16>) -> CudaJpegBaselineEncodeParams {
    CudaJpegBaselineEncodeParams {
        input_width: SIZE.0,
        input_height: SIZE.1,
        output_width: SIZE.0,
        output_height: SIZE.1,
        pitch_bytes: SIZE.0 * components,
        mcus_per_row: SIZE.0.div_ceil(8),
        mcu_rows: SIZE.1.div_ceil(8),
        restart_interval_mcus: restart_interval.map_or(0, u32::from),
        format: components / 3,
        components,
        max_h: 1,
        max_v: 1,
        h0: 1,
        v0: 1,
        h1: 1,
        v1: 1,
        h2: 1,
        v2: 1,
        entropy_capacity: 1 << 20,
        ..CudaJpegBaselineEncodeParams::default()
    }
}

fn cuda_entropy(
    engine: JpegCudaEngine<'_>,
    params: CudaJpegBaselineEncodeParams,
    blocks: &[[i32; 64]],
    tables: &[CudaJpegBaselineEncodeHuffmanTable; 4],
) -> Vec<u8> {
    let coefficient_bytes = blocks
        .iter()
        .flatten()
        .flat_map(|coefficient| coefficient.to_ne_bytes())
        .collect::<Vec<_>>();
    let coefficients = engine.upload(&coefficient_bytes).expect("coefficients");
    let entropy = engine
        .allocate(params.entropy_capacity as usize)
        .expect("entropy");
    let mut statuses = [CudaJpegBaselineEncodeStatus::default()];
    let status = engine
        .upload(cuda_jpeg_baseline_encode_statuses_as_bytes(&statuses))
        .expect("status");
    let params_buffer = engine
        .upload(cuda_jpeg_baseline_encode_params_as_bytes(&[params]))
        .expect("params");
    let [dc_luma, ac_luma, dc_chroma, ac_chroma] = tables.map(|table| {
        engine
            .upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(&table))
            .expect("Huffman table")
    });
    engine
        .launch_jpeg_encode_baseline_entropy_from_coeffs_batch(
            &CudaJpegBaselineEntropyFromCoeffsLaunch {
                coefficients: &coefficients,
                entropy: &entropy,
                status: &status,
                params: &params_buffer,
                huffman: CudaJpegBaselineHuffmanLaunch {
                    dc_luma: &dc_luma,
                    ac_luma: &ac_luma,
                    dc_chroma: &dc_chroma,
                    ac_chroma: &ac_chroma,
                },
                tile_count: 1,
                geometry: CudaLaunchGeometry::new((1, 1, 1), (1, 1, 1)).expect("geometry"),
            },
        )
        .expect("entropy stage launch");
    status
        .copy_to_host(cuda_jpeg_baseline_encode_statuses_as_bytes_mut(
            &mut statuses,
        ))
        .expect("read status");
    assert_eq!(statuses[0].code, 0, "entropy stage status");
    let mut out = vec![0u8; statuses[0].entropy_len as usize];
    entropy
        .copy_range_to_host(0, &mut out)
        .expect("read entropy");
    out
}

#[test]
fn entropy_stage_codes_fixed_coefficients_as_the_cpu_does() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let engine = JpegCudaEngine::new(&context);
    let segments = jpeg_entropy_stage_cases()
        .into_iter()
        .map(|case| {
            let tables = jpeg_standard_tables(case.quality);
            let channels = case.channels();
            let blocks = jpeg_entropy_stage_coefficients(
                &case.samples(),
                SIZE,
                channels,
                &tables.q_luma,
                &tables.q_chroma,
            );
            let components = u32::try_from(channels).expect("channels fit u32");
            cuda_entropy(
                engine,
                params(components, case.restart_interval),
                &blocks,
                &tables.huffman.each_ref().map(huffman),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        jpeg_entropy_stage_digest(segments.iter().map(Vec::as_slice)),
        JPEG_ENTROPY_STAGE_SHA256,
        "the CUDA entropy stage differs from the CPU encoder's segments"
    );
}
