// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use j2k_cuda_j2k_engine::{CudaHtj2kCodeBlockJob, CudaHtj2kDecodeTables, J2kCudaEngine};
use j2k_cuda_runtime::CudaContext;
#[cfg(feature = "cuda-runtime")]
use j2k_native::{
    decode_ht_code_block_scalar, decode_ht_code_block_scalar_with_workspace_midpoint,
    encode_ht_code_block_scalar, ht_uvlc_table0, ht_uvlc_table1, ht_vlc_table0, ht_vlc_table1,
    HtCodeBlockDecodeJob, HtCodeBlockDecodeWorkspace,
};

#[cfg(feature = "cuda-runtime")]
fn decode_cuda(payload: &[u8], job: CudaHtj2kCodeBlockJob) -> Vec<f32> {
    let output_words = job.width as usize * job.height as usize;
    let context = CudaContext::system_default().expect("CUDA context");
    let output = J2kCudaEngine::new(&context)
        .decode_htj2k_codeblocks(
            payload,
            &[job],
            CudaHtj2kDecodeTables {
                vlc_table0: ht_vlc_table0(),
                vlc_table1: ht_vlc_table1(),
                uvlc_table0: ht_uvlc_table0(),
                uvlc_table1: ht_uvlc_table1(),
            },
            output_words,
        )
        .expect("CUDA HTJ2K decode");
    assert_eq!(output.execution().decode_kernel_dispatches(), 2);
    assert!(output.statuses().iter().all(|status| status.is_ok()));

    let mut bytes = vec![0_u8; output_words * core::mem::size_of::<f32>()];
    output
        .coefficients()
        .copy_to_host(&mut bytes)
        .expect("download CUDA coefficients");
    bytes
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|word| f32::from_ne_bytes(word.try_into().expect("one f32 word")))
        .collect()
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_roi_reconstruction_matches_native_exactly_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let source = [0_i32, 2, -3, 1, 4, 0, -1, 2, 3, -2, 0, 1, 0, 0, 5, -4];
    let shifted = source.map(|sample| sample << 7);
    let encoded =
        encode_ht_code_block_scalar(&shifted, 4, 4, 12).expect("encode ROI-shifted HT block");
    let native_job = HtCodeBlockDecodeJob {
        data: &encoded.data,
        cleanup_length: encoded.cleanup_length,
        refinement_length: encoded.refinement_length,
        width: 4,
        height: 4,
        output_stride: 4,
        missing_bit_planes: encoded.num_zero_bitplanes,
        number_of_coding_passes: encoded.num_coding_passes,
        num_bitplanes: 5,
        roi_shift: 7,
        stripe_causal: false,
        strict: true,
        dequantization_step: 1.0,
    };
    let mut expected = vec![0.0_f32; source.len()];
    decode_ht_code_block_scalar(native_job, &mut expected).expect("native ROI HT decode");
    let expected_bits = expected
        .iter()
        .copied()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let source_bits = source
        .map(|sample| f32::from(i16::try_from(sample).expect("fixture sample fits i16")).to_bits());
    assert_eq!(
        expected_bits.as_slice(),
        source_bits.as_slice(),
        "fixture must exercise inverse ROI maxshift"
    );

    let actual = decode_cuda(
        &encoded.data,
        CudaHtj2kCodeBlockJob {
            payload_offset: 0,
            width: 4,
            height: 4,
            payload_len: u32::try_from(encoded.data.len()).expect("payload length"),
            cleanup_length: encoded.cleanup_length,
            refinement_length: encoded.refinement_length,
            missing_bit_planes: encoded.num_zero_bitplanes,
            num_bitplanes: 5,
            roi_shift: 7,
            number_of_coding_passes: encoded.num_coding_passes,
            output_stride: 4,
            output_offset: 0,
            dequantization_step: 1.0,
            stripe_causal: false,
            irreversible_midpoint: false,
        },
    );
    assert_eq!(
        actual.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
        expected_bits
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_irreversible_midpoint_matches_native_bits_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let coefficients = [0_i32, 3, -5, 7, 1, -2, 4, -6, 2, -1, 5, -7, 6, -4, 3, 0];
    let encoded = encode_ht_code_block_scalar(&coefficients, 4, 4, 4).expect("encode HT block");
    let native_job = HtCodeBlockDecodeJob {
        data: &encoded.data,
        cleanup_length: encoded.cleanup_length,
        refinement_length: encoded.refinement_length,
        width: 4,
        height: 4,
        output_stride: 4,
        missing_bit_planes: encoded.num_zero_bitplanes,
        number_of_coding_passes: encoded.num_coding_passes,
        num_bitplanes: 4,
        roi_shift: 0,
        stripe_causal: false,
        strict: true,
        dequantization_step: 0.5,
    };
    let mut expected = vec![0.0_f32; coefficients.len()];
    let mut workspace = HtCodeBlockDecodeWorkspace::default();
    decode_ht_code_block_scalar_with_workspace_midpoint(native_job, &mut expected, &mut workspace)
        .expect("native midpoint HT decode");
    let mut integer_reconstruction = vec![0.0_f32; coefficients.len()];
    decode_ht_code_block_scalar(native_job, &mut integer_reconstruction)
        .expect("native integer HT decode");
    assert_ne!(
        expected
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        integer_reconstruction
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        "fixture must distinguish midpoint from integer reconstruction"
    );

    let actual = decode_cuda(
        &encoded.data,
        CudaHtj2kCodeBlockJob {
            payload_offset: 0,
            width: 4,
            height: 4,
            payload_len: u32::try_from(encoded.data.len()).expect("payload length"),
            cleanup_length: encoded.cleanup_length,
            refinement_length: encoded.refinement_length,
            missing_bit_planes: encoded.num_zero_bitplanes,
            num_bitplanes: 4,
            roi_shift: 0,
            number_of_coding_passes: encoded.num_coding_passes,
            output_stride: 4,
            output_offset: 0,
            dequantization_step: 0.5,
            stripe_causal: false,
            irreversible_midpoint: true,
        },
    );
    assert_eq!(
        actual
            .iter()
            .map(|sample| sample.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|sample| sample.to_bits())
            .collect::<Vec<_>>()
    );
}
