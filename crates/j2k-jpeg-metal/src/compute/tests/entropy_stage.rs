// SPDX-License-Identifier: MIT OR Apache-2.0

//! The entropy stage alone: Huffman coding, DC prediction, restart intervals
//! and byte stuffing of fixed coefficients produce the CPU encoder's entropy
//! bytes. See `j2k_test_support::jpeg_entropy_stage_coefficients`.

use crate::abi::{
    JpegBaselineEncodeHuffmanTable, JpegBaselineEncodeParams, JpegBaselineEncodeStatus,
    JPEG_BASELINE_ENCODE_STATUS_OK,
};
use crate::compute::encode::{record_baseline_entropy_passes, JPEG_BASELINE_ENCODE_SEGMENT_BYTES};
use crate::compute::{commit_and_wait_jpeg, new_command_buffer, with_runtime_for_session};
use j2k_jpeg::adapter::{baseline_encode_tables, JpegBaselineHuffmanTable};
use j2k_jpeg::{JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling};
use j2k_metal_support::{
    checked_buffer_read_vec, checked_shared_buffer_for_len, checked_shared_buffer_with_slice,
};
use j2k_test_support::{
    jpeg_entropy_segment, jpeg_entropy_stage_cases, jpeg_entropy_stage_coefficients,
    jpeg_entropy_stage_digest, JpegEncodeMatrixSampling, JPEG_ENTROPY_STAGE_SHA256,
    JPEG_ENTROPY_STAGE_SIZE as SIZE,
};

fn huffman(table: &JpegBaselineHuffmanTable) -> JpegBaselineEncodeHuffmanTable {
    JpegBaselineEncodeHuffmanTable {
        codes: table.codes,
        lens: table.lens,
    }
}

/// Unsubsampled single-tile parameters: one block per component per MCU.
fn params(components: u32, restart_interval: Option<u16>) -> JpegBaselineEncodeParams {
    JpegBaselineEncodeParams {
        input_offset_bytes: 0,
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
        entropy_offset_bytes: 0,
        entropy_capacity: 1 << 20,
    }
}

fn metal_entropy(
    params: JpegBaselineEncodeParams,
    blocks: &[[i32; 64]],
    tables: &[JpegBaselineEncodeHuffmanTable; 4],
) -> Vec<u8> {
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let mcu_count = params.mcus_per_row * params.mcu_rows;
    with_runtime_for_session(&session, |runtime| {
        let device = &runtime.device;
        let segment_bytes = (mcu_count * JPEG_BASELINE_ENCODE_SEGMENT_BYTES) as usize;
        let coefficients = checked_shared_buffer_with_slice(device, blocks).expect("coefficients");
        let segments =
            checked_shared_buffer_for_len::<u8>(device, segment_bytes).expect("segments");
        let segment_statuses =
            checked_shared_buffer_for_len::<JpegBaselineEncodeStatus>(device, mcu_count as usize)
                .expect("segment statuses");
        let entropy = checked_shared_buffer_for_len::<u8>(device, params.entropy_capacity as usize)
            .expect("entropy");
        let status =
            checked_shared_buffer_with_slice(device, &[JpegBaselineEncodeStatus::default()])
                .expect("status");
        let params_buffer = checked_shared_buffer_with_slice(device, &[params]).expect("params");
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let [dc_luma, ac_luma, dc_chroma, ac_chroma] = tables;
        record_baseline_entropy_passes(
            runtime,
            &command_buffer,
            [
                &coefficients,
                &segments,
                &segment_statuses,
                &entropy,
                &status,
                &params_buffer,
            ],
            [dc_luma, ac_luma, dc_chroma, ac_chroma],
            1,
            mcu_count,
        )?;
        commit_and_wait_jpeg(&command_buffer)?;
        // SAFETY: the command buffer completed, so the GPU no longer writes
        // the status buffer, and the read covers its one status.
        let status = unsafe { checked_buffer_read_vec::<JpegBaselineEncodeStatus>(&status, 0, 1) }
            .expect("status")[0];
        assert_eq!(status.code, JPEG_BASELINE_ENCODE_STATUS_OK);
        let len = status.entropy_len as usize;
        // SAFETY: as above for the entropy buffer; the stage reported `len`
        // bytes within its capacity.
        Ok(unsafe { checked_buffer_read_vec::<u8>(&entropy, 0, len) }.expect("entropy"))
    })
    .expect("Metal entropy stage")
}

#[test]
fn entropy_stage_codes_fixed_coefficients_as_the_cpu_does() {
    if !super::should_run_metal_runtime() {
        return;
    }
    let mut segments = Vec::new();
    for case in jpeg_entropy_stage_cases() {
        let data = case.samples();
        let gray = case.sampling == JpegEncodeMatrixSampling::Gray;
        let options = JpegEncodeOptions {
            quality: case.quality,
            subsampling: if gray {
                JpegSubsampling::Gray
            } else {
                JpegSubsampling::Ybr444
            },
            restart_interval: case.restart_interval,
            backend: JpegBackend::Cpu,
        };
        let tables = baseline_encode_tables(options).expect("standard tables");
        let channels = case.channels();
        let blocks = jpeg_entropy_stage_coefficients(
            &data,
            SIZE,
            channels,
            &tables.q_luma,
            &tables.q_chroma,
        );
        let components = u32::try_from(channels).expect("channels fit u32");
        let metal = metal_entropy(
            params(components, case.restart_interval),
            &blocks,
            &[
                huffman(&tables.huff_dc_luma),
                huffman(&tables.huff_ac_luma),
                huffman(&tables.huff_dc_chroma),
                huffman(&tables.huff_ac_chroma),
            ],
        );
        let (width, height) = SIZE;
        let input = if gray {
            JpegSamples::Gray8 {
                data: &data,
                width,
                height,
            }
        } else {
            JpegSamples::Rgb8 {
                data: &data,
                width,
                height,
            }
        };
        let cpu = j2k_jpeg::encode_jpeg_baseline(input, options).expect("CPU encode");
        assert!(
            metal == jpeg_entropy_segment(&cpu.data),
            "{}: the Metal entropy stage differs from the CPU",
            case.label()
        );
        segments.push(metal);
    }
    assert_eq!(
        jpeg_entropy_stage_digest(segments.iter().map(Vec::as_slice)),
        JPEG_ENTROPY_STAGE_SHA256
    );
}
