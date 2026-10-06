// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use j2k_cuda_j2k_engine::{
    CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeTables, J2kCudaEngine,
};
use j2k_cuda_runtime::CudaContext;
#[cfg(feature = "cuda-runtime")]
use j2k_native::{
    decode_ht_code_block_scalar, decode_ht_code_block_scalar_with_workspace_midpoint,
    encode_ht_code_block_scalar, ht_uvlc_table0, ht_uvlc_table1, ht_vlc_table0, ht_vlc_table1,
    HtCodeBlockDecodeJob, HtCodeBlockDecodeWorkspace,
};

#[cfg(feature = "cuda-runtime")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "shape matrix keeps the scalar reference and guarded device result together"
)]
fn cuda_cleanup_matches_scalar_with_row_guards_for_tall_wide_and_odd_blocks_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = CudaContext::system_default().expect("CUDA context");
    let engine = J2kCudaEngine::new(&context);
    let pool = context.buffer_pool();
    let tables = engine
        .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
            vlc_table0: ht_vlc_table0(),
            vlc_table1: ht_vlc_table1(),
            uvlc_table0: ht_uvlc_table0(),
            uvlc_table1: ht_uvlc_table1(),
        })
        .expect("cleanup tables");
    for (width, height) in [
        (1, 1),
        (1, 127),
        (7, 9),
        (32, 32),
        (63, 65),
        (64, 64),
        (128, 32),
        (256, 16),
        (16, 256),
    ] {
        let source = (0..width * height)
            .map(|i| {
                if i % 11 == 10 {
                    0
                } else {
                    i32::try_from((i * 37 + i / width * 13) % 1023).unwrap() - 511
                }
            })
            .collect::<Vec<_>>();
        let encoded = encode_ht_code_block_scalar(&source, width, height, 12)
            .expect("encode shaped cleanup block");
        assert_eq!(encoded.num_coding_passes, 1);
        let mut expected =
            j2k_core::try_host_vec_filled(source.len(), 0.0_f32).expect("scalar reference storage");
        decode_ht_code_block_scalar(
            HtCodeBlockDecodeJob {
                data: &encoded.data,
                cleanup_length: encoded.cleanup_length,
                refinement_length: 0,
                width,
                height,
                output_stride: width as usize,
                missing_bit_planes: encoded.num_zero_bitplanes,
                number_of_coding_passes: 1,
                num_bitplanes: 12,
                roi_shift: 0,
                stripe_causal: false,
                strict: true,
                dequantization_step: 0.25,
            },
            &mut expected,
        )
        .expect("scalar cleanup reference");
        let job = CudaHtj2kCodeBlockJob {
            payload_offset: 0,
            payload_len: u32::try_from(encoded.data.len()).unwrap(),
            cleanup_length: encoded.cleanup_length,
            refinement_length: 0,
            width,
            height,
            output_stride: width + 3,
            output_offset: 3,
            missing_bit_planes: encoded.num_zero_bitplanes,
            number_of_coding_passes: 1,
            num_bitplanes: 12,
            roi_shift: 0,
            stripe_causal: false,
            irreversible_midpoint: false,
            dequantization_step: 0.25,
        };
        let resources = engine
            .upload_htj2k_decode_resources_with_tables_and_pool(&[&encoded.data], &tables, &pool)
            .expect("cleanup payload");
        let sentinel = -12345.0_f32;
        let output_words = ((width + 3) * height + 4) as usize;
        let mut guarded_expected = j2k_core::try_host_vec_filled(output_words, sentinel)
            .expect("guarded reference storage");
        for row in 0..height as usize {
            let start = 3 + row * (width as usize + 3);
            guarded_expected[start..start + width as usize]
                .copy_from_slice(&expected[row * width as usize..(row + 1) * width as usize]);
        }
        let initial = j2k_core::try_host_vec_filled(output_words, sentinel)
            .expect("guarded device initialization")
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        let output = context.upload(&initial).expect("guarded cleanup output");
        engine
            .decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed(
                &resources,
                &[CudaHtj2kCleanupTarget {
                    coefficients: &output,
                    jobs: &[job],
                    output_words,
                }],
                &pool,
                false,
            )
            .expect("fused cleanup decode");
        let mut bytes = j2k_core::try_host_vec_filled(initial.len(), 0).expect("readback storage");
        output
            .copy_to_host(&mut bytes)
            .expect("read cleanup output");
        let actual = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_ne_bytes(*v))
            .collect::<Vec<_>>();
        assert_eq!(
            actual, guarded_expected,
            "cleanup {width}x{height}, including row guards"
        );
    }
}

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
        .as_chunks::<{ core::mem::size_of::<f32>() }>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
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
