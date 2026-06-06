#[cfg(feature = "cuda-runtime")]
use signinum_core::PixelFormat;
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::{
    CudaContext, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeTables, CudaHtj2kEncodeCodeBlockJob,
    CudaHtj2kEncodeTables, CudaHtj2kPacketizationBlock, CudaHtj2kPacketizationPacket,
    CudaHtj2kPacketizationSubband, CudaHtj2kPacketizationSubbandTagState,
    CudaHtj2kPacketizationTagNodeState, CudaJ2kInverseMctJob, CudaJ2kStoreGray16Job,
    CudaJ2kStoreRgb16Job, CudaJ2kStoreRgb8Job,
};
#[cfg(feature = "cuda-runtime")]
use signinum_j2k_cuda::J2kDecoder;
#[cfg(feature = "cuda-runtime")]
use signinum_j2k_native::{
    decode_ht_code_block_scalar, encode_ht_code_block_scalar, encode_htj2k, ht_uvlc_encode_table,
    ht_uvlc_table0, ht_uvlc_table1, ht_vlc_encode_table0, ht_vlc_encode_table1, ht_vlc_table0,
    ht_vlc_table1, EncodeOptions, HtCodeBlockDecodeJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
    J2kPacketizationProgressionOrder, J2kPacketizationResolution, J2kPacketizationSubband,
};

#[cfg(feature = "cuda-runtime")]
fn runtime_required() -> bool {
    std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
}

#[cfg(feature = "cuda-runtime")]
fn ht_gray8_fixture() -> Vec<u8> {
    let pixels: Vec<u8> = (0..64).collect();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8")
}

#[cfg(feature = "cuda-runtime")]
fn openhtj2k_refinement_fixture() -> &'static [u8] {
    include_bytes!("fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k")
}

#[cfg(feature = "cuda-runtime")]
fn uvlc_encode_table_bytes() -> Vec<u8> {
    ht_uvlc_encode_table()
        .iter()
        .flat_map(|entry| {
            [
                entry.pre,
                entry.pre_len,
                entry.suf,
                entry.suf_len,
                entry.ext,
                entry.ext_len,
            ]
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn rounded_u8(sample: f32) -> u8 {
    sample.round().clamp(0.0, 255.0) as u8
}

#[cfg(feature = "cuda-runtime")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn rounded_u16(sample: f32, bit_depth: u32) -> u16 {
    let rounded = sample.round();
    if bit_depth >= 16 {
        return rounded.clamp(0.0, f32::from(u16::MAX)) as u16;
    }
    let shift = u16::try_from(bit_depth.min(15)).expect("bounded bit depth fits in u16");
    let max_value = f32::from((1u16 << shift).saturating_sub(1).max(1));
    ((rounded.clamp(0.0, max_value) / max_value) * f32::from(u16::MAX)).round() as u16
}

#[cfg(feature = "cuda-runtime")]
fn push_u16_ne(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_ne_bytes());
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_entropy_kernel_matches_native_scalar_codeblock_when_required() {
    if !runtime_required() {
        return;
    }

    let bytes = ht_gray8_fixture();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let (cuda_plan, _) = decoder
        .build_cuda_htj2k_grayscale_plan_with_profile(PixelFormat::Gray8)
        .expect("CUDA flat plan");
    let block = cuda_plan
        .code_blocks()
        .first()
        .copied()
        .expect("at least one HT block");
    let payload_start = usize::try_from(block.payload_offset).expect("payload offset");
    let payload_end = payload_start + block.payload_len as usize;
    let block_payload = &cuda_plan.payload()[payload_start..payload_end];

    let mut expected = vec![0.0f32; block.width as usize * block.height as usize];
    decode_ht_code_block_scalar(
        HtCodeBlockDecodeJob {
            data: block_payload,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            width: block.width,
            height: block.height,
            output_stride: block.width as usize,
            missing_bit_planes: block.missing_bit_planes,
            number_of_coding_passes: block.number_of_coding_passes,
            num_bitplanes: block.num_bitplanes,
            roi_shift: 0,
            stripe_causal: block.stripe_causal != 0,
            strict: true,
            dequantization_step: block.dequantization_step,
        },
        &mut expected,
    )
    .expect("native scalar HT decode");

    let context = CudaContext::system_default().expect("CUDA context");
    let output = context
        .decode_htj2k_codeblocks(
            block_payload,
            &[CudaHtj2kCodeBlockJob {
                payload_offset: 0,
                width: block.width,
                height: block.height,
                payload_len: block.payload_len,
                cleanup_length: block.cleanup_length,
                refinement_length: block.refinement_length,
                missing_bit_planes: block.missing_bit_planes,
                num_bitplanes: block.num_bitplanes,
                number_of_coding_passes: block.number_of_coding_passes,
                output_stride: block.width,
                output_offset: 0,
                dequantization_step: block.dequantization_step,
                stripe_causal: block.stripe_causal != 0,
            }],
            CudaHtj2kDecodeTables {
                vlc_table0: ht_vlc_table0(),
                vlc_table1: ht_vlc_table1(),
                uvlc_table0: ht_uvlc_table0(),
                uvlc_table1: ht_uvlc_table1(),
            },
            expected.len(),
        )
        .expect("CUDA HT decode");

    assert_eq!(output.execution().decode_kernel_dispatches(), 2);
    assert!(output.stage_timings().ht_cleanup_us > 0);
    assert!(output.stage_timings().dequant_us > 0);
    assert!(output.statuses().iter().all(|status| status.is_ok()));

    let mut actual_bytes = vec![0u8; expected.len() * std::mem::size_of::<f32>()];
    output
        .coefficients()
        .copy_to_host(&mut actual_bytes)
        .expect("download coefficients");
    let actual = actual_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("f32 bytes")))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_refinement_kernel_matches_native_scalar_codeblock_when_required() {
    if !runtime_required() {
        return;
    }

    let mut decoder = J2kDecoder::new(openhtj2k_refinement_fixture()).expect("decoder");
    let (cuda_plan, _) = decoder
        .build_cuda_htj2k_grayscale_plan_with_profile(PixelFormat::Gray8)
        .expect("CUDA flat plan");
    let block = cuda_plan
        .code_blocks()
        .iter()
        .copied()
        .find(|block| block.refinement_length > 0)
        .expect("fixture must contain a refinement block");
    let payload_start = usize::try_from(block.payload_offset).expect("payload offset");
    let payload_end = payload_start + block.payload_len as usize;
    let block_payload = &cuda_plan.payload()[payload_start..payload_end];

    let mut expected = vec![0.0f32; block.width as usize * block.height as usize];
    decode_ht_code_block_scalar(
        HtCodeBlockDecodeJob {
            data: block_payload,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            width: block.width,
            height: block.height,
            output_stride: block.width as usize,
            missing_bit_planes: block.missing_bit_planes,
            number_of_coding_passes: block.number_of_coding_passes,
            num_bitplanes: block.num_bitplanes,
            roi_shift: 0,
            stripe_causal: block.stripe_causal != 0,
            strict: true,
            dequantization_step: block.dequantization_step,
        },
        &mut expected,
    )
    .expect("native scalar HT refinement decode");

    let context = CudaContext::system_default().expect("CUDA context");
    let output = context
        .decode_htj2k_codeblocks(
            block_payload,
            &[CudaHtj2kCodeBlockJob {
                payload_offset: 0,
                width: block.width,
                height: block.height,
                payload_len: block.payload_len,
                cleanup_length: block.cleanup_length,
                refinement_length: block.refinement_length,
                missing_bit_planes: block.missing_bit_planes,
                num_bitplanes: block.num_bitplanes,
                number_of_coding_passes: block.number_of_coding_passes,
                output_stride: block.width,
                output_offset: 0,
                dequantization_step: block.dequantization_step,
                stripe_causal: block.stripe_causal != 0,
            }],
            CudaHtj2kDecodeTables {
                vlc_table0: ht_vlc_table0(),
                vlc_table1: ht_vlc_table1(),
                uvlc_table0: ht_uvlc_table0(),
                uvlc_table1: ht_uvlc_table1(),
            },
            expected.len(),
        )
        .expect("CUDA HT refinement decode");

    assert_eq!(output.execution().decode_kernel_dispatches(), 2);
    assert!(output.stage_timings().ht_cleanup_us > 0);
    assert!(output.stage_timings().ht_refine_us > 0);
    assert!(output.stage_timings().dequant_us > 0);
    assert!(output.statuses().iter().all(|status| status.is_ok()));

    let mut actual_bytes = vec![0u8; expected.len() * std::mem::size_of::<f32>()];
    output
        .coefficients()
        .copy_to_host(&mut actual_bytes)
        .expect("download coefficients");
    let actual = actual_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("f32 bytes")))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_encode_kernel_matches_native_scalar_codeblock_when_required() {
    if !runtime_required() {
        return;
    }

    let coefficients = [
        0, 2, -3, 1, 4, 0, -1, 2, 3, -2, 0, 1, 0, 0, 5, -4, 1, 0, -2, 3, 0, 1, 0, -1, 2, -3, 4, 0,
        -5, 1, 2, 0, 0, -1, 3, 2, -2, 0, 1, -3, 4, 0, 0, 2, -1, 5, 0, -4, 3, 0, 1, -2, 2, 0, -1, 4,
        0, 3, -3, 1, 0, 2, -4, 5,
    ];
    let expected =
        encode_ht_code_block_scalar(&coefficients, 8, 8, 8).expect("native scalar HT encode");

    let context = CudaContext::system_default().expect("CUDA context");
    let uvlc_table = uvlc_encode_table_bytes();
    let encoded = context
        .encode_htj2k_codeblock(
            &coefficients,
            8,
            8,
            8,
            CudaHtj2kEncodeTables {
                vlc_table0: ht_vlc_encode_table0(),
                vlc_table1: ht_vlc_encode_table1(),
                uvlc_table: &uvlc_table,
            },
        )
        .expect("CUDA HT encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert_eq!(encoded.data(), expected.data);
    assert_eq!(encoded.cleanup_length(), expected.cleanup_length);
    assert_eq!(encoded.refinement_length(), expected.refinement_length);
    assert_eq!(encoded.num_coding_passes(), expected.num_coding_passes);
    assert_eq!(encoded.num_zero_bitplanes(), expected.num_zero_bitplanes);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_encode_target_two_passes_round_trips_with_sigprop_segment_when_required() {
    if !runtime_required() {
        return;
    }

    let coefficients = [0, 3, -5, 3, 5, 0, -3, 3, 7, -3, 0, 3, 0, 0, 5, -5];

    let context = CudaContext::system_default().expect("CUDA context");
    let uvlc_table = uvlc_encode_table_bytes();
    let encoded = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &[CudaHtj2kEncodeCodeBlockJob {
                coefficient_offset: 0,
                width: 4,
                height: 4,
                total_bitplanes: 4,
                target_coding_passes: 2,
            }],
            CudaHtj2kEncodeTables {
                vlc_table0: ht_vlc_encode_table0(),
                vlc_table1: ht_vlc_encode_table1(),
                uvlc_table: &uvlc_table,
            },
        )
        .expect("CUDA two-pass HT encode");
    let block = encoded
        .code_blocks()
        .first()
        .expect("one encoded code block");

    assert_eq!(block.num_coding_passes(), 2);
    assert_eq!(block.num_zero_bitplanes(), 2);
    assert_eq!(block.refinement_length(), 1);
    assert_eq!(
        block.cleanup_length() + block.refinement_length(),
        u32::try_from(block.data().len()).expect("test payload length fits u32")
    );

    let mut decoded = vec![0.0f32; coefficients.len()];
    decode_ht_code_block_scalar(
        HtCodeBlockDecodeJob {
            data: block.data(),
            cleanup_length: block.cleanup_length(),
            refinement_length: block.refinement_length(),
            width: 4,
            height: 4,
            output_stride: 4,
            missing_bit_planes: block.num_zero_bitplanes(),
            number_of_coding_passes: block.num_coding_passes(),
            num_bitplanes: 4,
            roi_shift: 0,
            stripe_causal: false,
            strict: true,
            dequantization_step: 1.0,
        },
        &mut decoded,
    )
    .expect("two-pass HT block decodes");

    assert_eq!(
        decoded,
        coefficients
            .iter()
            .map(|value| f32::from(i16::try_from(*value).expect("test coefficient fits i16")))
            .collect::<Vec<_>>()
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_encode_target_three_passes_round_trips_with_magref_segment_when_required() {
    if !runtime_required() {
        return;
    }

    let coefficients = [5, -7, 9, -11];

    let context = CudaContext::system_default().expect("CUDA context");
    let uvlc_table = uvlc_encode_table_bytes();
    let encoded = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &[CudaHtj2kEncodeCodeBlockJob {
                coefficient_offset: 0,
                width: 2,
                height: 2,
                total_bitplanes: 4,
                target_coding_passes: 3,
            }],
            CudaHtj2kEncodeTables {
                vlc_table0: ht_vlc_encode_table0(),
                vlc_table1: ht_vlc_encode_table1(),
                uvlc_table: &uvlc_table,
            },
        )
        .expect("CUDA three-pass HT encode");
    let block = encoded
        .code_blocks()
        .first()
        .expect("one encoded code block");

    assert_eq!(block.num_coding_passes(), 3);
    assert_eq!(block.num_zero_bitplanes(), 1);
    assert_eq!(block.refinement_length(), 2);
    assert_eq!(
        block.cleanup_length() + block.refinement_length(),
        u32::try_from(block.data().len()).expect("test payload length fits u32")
    );

    let mut decoded = vec![0.0f32; coefficients.len()];
    decode_ht_code_block_scalar(
        HtCodeBlockDecodeJob {
            data: block.data(),
            cleanup_length: block.cleanup_length(),
            refinement_length: block.refinement_length(),
            width: 2,
            height: 2,
            output_stride: 2,
            missing_bit_planes: block.num_zero_bitplanes(),
            number_of_coding_passes: block.num_coding_passes(),
            num_bitplanes: 4,
            roi_shift: 0,
            stripe_causal: false,
            strict: true,
            dequantization_step: 1.0,
        },
        &mut decoded,
    )
    .expect("three-pass HT block decodes");

    assert_eq!(
        decoded,
        coefficients
            .iter()
            .map(|value| f32::from(i16::try_from(*value).expect("test coefficient fits i16")))
            .collect::<Vec<_>>()
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_encode_target_three_passes_stuffs_magref_all_one_bytes_when_required() {
    if !runtime_required() {
        return;
    }

    let coefficients = [7; 8];

    let context = CudaContext::system_default().expect("CUDA context");
    let uvlc_table = uvlc_encode_table_bytes();
    let encoded = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &[CudaHtj2kEncodeCodeBlockJob {
                coefficient_offset: 0,
                width: 4,
                height: 2,
                total_bitplanes: 4,
                target_coding_passes: 3,
            }],
            CudaHtj2kEncodeTables {
                vlc_table0: ht_vlc_encode_table0(),
                vlc_table1: ht_vlc_encode_table1(),
                uvlc_table: &uvlc_table,
            },
        )
        .expect("CUDA three-pass HT encode with stuffed MagRef bytes");
    let block = encoded
        .code_blocks()
        .first()
        .expect("one encoded code block");

    assert_eq!(block.num_coding_passes(), 3);
    assert_eq!(block.num_zero_bitplanes(), 1);
    assert_eq!(block.refinement_length(), 3);
    assert_eq!(
        block.cleanup_length() + block.refinement_length(),
        u32::try_from(block.data().len()).expect("test payload length fits u32")
    );

    let mut decoded = vec![0.0f32; coefficients.len()];
    decode_ht_code_block_scalar(
        HtCodeBlockDecodeJob {
            data: block.data(),
            cleanup_length: block.cleanup_length(),
            refinement_length: block.refinement_length(),
            width: 4,
            height: 2,
            output_stride: 4,
            missing_bit_planes: block.num_zero_bitplanes(),
            number_of_coding_passes: block.num_coding_passes(),
            num_bitplanes: 4,
            roi_shift: 0,
            stripe_causal: false,
            strict: true,
            dequantization_step: 1.0,
        },
        &mut decoded,
    )
    .expect("stuffed MagRef HT block decodes");

    assert_eq!(
        decoded,
        coefficients
            .iter()
            .map(|value| f32::from(i16::try_from(*value).expect("test coefficient fits i16")))
            .collect::<Vec<_>>()
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_encode_target_three_passes_round_trips_nonzero_sigprop_when_required() {
    if !runtime_required() {
        return;
    }

    let coefficients = [0, 3, -5, 7];

    let context = CudaContext::system_default().expect("CUDA context");
    let uvlc_table = uvlc_encode_table_bytes();
    let encoded = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &[CudaHtj2kEncodeCodeBlockJob {
                coefficient_offset: 0,
                width: 2,
                height: 2,
                total_bitplanes: 4,
                target_coding_passes: 3,
            }],
            CudaHtj2kEncodeTables {
                vlc_table0: ht_vlc_encode_table0(),
                vlc_table1: ht_vlc_encode_table1(),
                uvlc_table: &uvlc_table,
            },
        )
        .expect("CUDA three-pass HT encode with nonzero SigProp");
    let block = encoded
        .code_blocks()
        .first()
        .expect("one encoded code block");

    assert_eq!(block.num_coding_passes(), 3);
    assert_eq!(block.num_zero_bitplanes(), 1);
    assert_eq!(block.refinement_length(), 2);
    assert_eq!(
        block.cleanup_length() + block.refinement_length(),
        u32::try_from(block.data().len()).expect("test payload length fits u32")
    );

    let mut decoded = vec![0.0f32; coefficients.len()];
    decode_ht_code_block_scalar(
        HtCodeBlockDecodeJob {
            data: block.data(),
            cleanup_length: block.cleanup_length(),
            refinement_length: block.refinement_length(),
            width: 2,
            height: 2,
            output_stride: 2,
            missing_bit_planes: block.num_zero_bitplanes(),
            number_of_coding_passes: block.num_coding_passes(),
            num_bitplanes: 4,
            roi_shift: 0,
            stripe_causal: false,
            strict: true,
            dequantization_step: 1.0,
        },
        &mut decoded,
    )
    .expect("nonzero SigProp HT block decodes");

    assert_eq!(
        decoded,
        coefficients
            .iter()
            .map(|value| f32::from(i16::try_from(*value).expect("test coefficient fits i16")))
            .collect::<Vec<_>>()
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_batch_encode_kernel_matches_native_scalar_codeblocks_when_required() {
    if !runtime_required() {
        return;
    }

    let block0 = [
        0, 2, -3, 1, 4, 0, -1, 2, 3, -2, 0, 1, 0, 0, 5, -4, 1, 0, -2, 3, 0, 1, 0, -1, 2, -3, 4, 0,
        -5, 1, 2, 0, 0, -1, 3, 2, -2, 0, 1, -3, 4, 0, 0, 2, -1, 5, 0, -4, 3, 0, 1, -2, 2, 0, -1, 4,
        0, 3, -3, 1, 0, 2, -4, 5,
    ];
    let block1 = [1, 0, -1, 2, -2, 3, 0, -3, 4, 0, 2, -1, 0, 5, -4, 1];
    let expected0 =
        encode_ht_code_block_scalar(&block0, 8, 8, 8).expect("native scalar HT encode 0");
    let expected1 =
        encode_ht_code_block_scalar(&block1, 4, 4, 6).expect("native scalar HT encode 1");

    let mut coefficients = Vec::with_capacity(block0.len() + block1.len());
    coefficients.extend_from_slice(&block0);
    let block1_offset = u32::try_from(coefficients.len()).expect("test offset fits in u32");
    coefficients.extend_from_slice(&block1);

    let context = CudaContext::system_default().expect("CUDA context");
    let uvlc_table = uvlc_encode_table_bytes();
    let encoded = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &[
                CudaHtj2kEncodeCodeBlockJob {
                    coefficient_offset: 0,
                    width: 8,
                    height: 8,
                    total_bitplanes: 8,
                    target_coding_passes: 1,
                },
                CudaHtj2kEncodeCodeBlockJob {
                    coefficient_offset: block1_offset,
                    width: 4,
                    height: 4,
                    total_bitplanes: 6,
                    target_coding_passes: 1,
                },
            ],
            CudaHtj2kEncodeTables {
                vlc_table0: ht_vlc_encode_table0(),
                vlc_table1: ht_vlc_encode_table1(),
                uvlc_table: &uvlc_table,
            },
        )
        .expect("CUDA batch HT encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert!(encoded.stage_timings().ht_encode_us > 0);
    assert_eq!(encoded.code_blocks().len(), 2);
    assert_eq!(encoded.code_blocks()[0].data(), expected0.data);
    assert_eq!(
        encoded.code_blocks()[0].cleanup_length(),
        expected0.cleanup_length
    );
    assert_eq!(
        encoded.code_blocks()[0].refinement_length(),
        expected0.refinement_length
    );
    assert_eq!(
        encoded.code_blocks()[0].num_coding_passes(),
        expected0.num_coding_passes
    );
    assert_eq!(
        encoded.code_blocks()[0].num_zero_bitplanes(),
        expected0.num_zero_bitplanes
    );
    assert_eq!(encoded.code_blocks()[1].data(), expected1.data);
    assert_eq!(
        encoded.code_blocks()[1].cleanup_length(),
        expected1.cleanup_length
    );
    assert_eq!(
        encoded.code_blocks()[1].refinement_length(),
        expected1.refinement_length
    );
    assert_eq!(
        encoded.code_blocks()[1].num_coding_passes(),
        expected1.num_coding_passes
    );
    assert_eq!(
        encoded.code_blocks()[1].num_zero_bitplanes(),
        expected1.num_zero_bitplanes
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_forward_ict_kernel_matches_cpu_transform_when_required() {
    if !runtime_required() {
        return;
    }

    let mut red = [12.0f32, 25.0, 40.0, 60.0];
    let mut green = [5.0f32, 15.0, 45.0, 100.0];
    let mut blue = [90.0f32, 70.0, 30.0, 10.0];
    let mut expected = Vec::with_capacity(red.len() * 3);
    for ((r, g), b) in red.iter().copied().zip(green).zip(blue) {
        expected.push(0.299 * r + 0.587 * g + 0.114 * b);
        expected.push(-0.16875 * r - 0.33126 * g + 0.5 * b);
        expected.push(0.5 * r - 0.41869 * g - 0.08131 * b);
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let stats = context
        .j2k_forward_ict(&mut red, &mut green, &mut blue)
        .expect("CUDA forward ICT");

    assert_eq!(stats.kernel_dispatches(), 1);
    for (((actual_y, actual_cb), actual_cr), expected) in red
        .into_iter()
        .zip(green)
        .zip(blue)
        .zip(expected.chunks_exact(3))
    {
        assert!((actual_y - expected[0]).abs() < 0.0001);
        assert!((actual_cb - expected[1]).abs() < 0.0001);
        assert!((actual_cr - expected[2]).abs() < 0.0001);
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_packetization_kernel_matches_native_scalar_cleanup_packet_when_required() {
    if !runtime_required() {
        return;
    }

    let payload = [0x12, 0x34, 0x56, 0x78];
    let code_block = J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let subband = J2kPacketizationSubband {
        code_blocks: vec![code_block],
        num_cbs_x: 1,
        num_cbs_y: 1,
    };
    let resolution = J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let descriptor = J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    };
    let expected =
        signinum_j2k_native::encode_j2k_packetization_scalar(J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 1,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        })
        .expect("native scalar packetization");

    let context = CudaContext::system_default().expect("CUDA context");
    let payload_len = u32::try_from(payload.len()).expect("test payload length fits in u32");
    let packetized = context
        .packetize_htj2k_cleanup_packets(
            &payload,
            &[CudaHtj2kPacketizationPacket {
                block_start: 0,
                block_count: 1,
                subband_start: 0,
                subband_count: 1,
                output_capacity: 512,
                layer: 0,
            }],
            &[CudaHtj2kPacketizationSubband {
                block_start: 0,
                block_count: 1,
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
            &[CudaHtj2kPacketizationBlock {
                data_offset: 0,
                data_len: payload_len,
                cleanup_length: 0,
                refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 2,
                l_block: 3,
                previously_included: 0,
                inclusion_layer: 0,
            }],
        )
        .expect("CUDA packetization");

    assert_eq!(packetized.execution().kernel_dispatches(), 1);
    assert!(packetized.stage_timings().packetize_us > 0);
    assert!(packetized.statuses().iter().all(|status| status.is_ok()));
    assert_eq!(packetized.data(), expected);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_packetization_kernel_matches_native_scalar_multi_block_packet_when_required() {
    if !runtime_required() {
        return;
    }

    let payloads = vec![
        vec![0x10, 0x11, 0x12],
        vec![0x20, 0x21],
        vec![0x30, 0x31, 0x32, 0x33],
        vec![0x40],
    ];
    let code_blocks = payloads
        .iter()
        .enumerate()
        .map(|(idx, payload)| J2kPacketizationCodeBlock {
            data: payload.as_slice(),
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: u8::try_from(idx + 1).expect("test zbp fits in u8"),
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        })
        .collect();
    let subband = J2kPacketizationSubband {
        code_blocks,
        num_cbs_x: 2,
        num_cbs_y: 2,
    };
    let resolution = J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let descriptor = J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    };
    let expected =
        signinum_j2k_native::encode_j2k_packetization_scalar(J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 4,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        })
        .expect("native scalar packetization");

    let payload = payloads.into_iter().flatten().collect::<Vec<_>>();
    let blocks = [
        CudaHtj2kPacketizationBlock {
            data_offset: 0,
            data_len: 3,
            cleanup_length: 0,
            refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 1,
            l_block: 3,
            previously_included: 0,
            inclusion_layer: 0,
        },
        CudaHtj2kPacketizationBlock {
            data_offset: 3,
            data_len: 2,
            cleanup_length: 0,
            refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            l_block: 3,
            previously_included: 0,
            inclusion_layer: 0,
        },
        CudaHtj2kPacketizationBlock {
            data_offset: 5,
            data_len: 4,
            cleanup_length: 0,
            refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 3,
            l_block: 3,
            previously_included: 0,
            inclusion_layer: 0,
        },
        CudaHtj2kPacketizationBlock {
            data_offset: 9,
            data_len: 1,
            cleanup_length: 0,
            refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 4,
            l_block: 3,
            previously_included: 0,
            inclusion_layer: 0,
        },
    ];
    let context = CudaContext::system_default().expect("CUDA context");
    let packetized = context
        .packetize_htj2k_cleanup_packets(
            &payload,
            &[CudaHtj2kPacketizationPacket {
                block_start: 0,
                block_count: 4,
                subband_start: 0,
                subband_count: 1,
                output_capacity: 512,
                layer: 0,
            }],
            &[CudaHtj2kPacketizationSubband {
                block_start: 0,
                block_count: 4,
                num_cbs_x: 2,
                num_cbs_y: 2,
            }],
            &blocks,
        )
        .expect("CUDA multi-block packetization");

    assert_eq!(packetized.execution().kernel_dispatches(), 1);
    assert!(packetized.stage_timings().packetize_us > 0);
    assert!(packetized.statuses().iter().all(|status| status.is_ok()));
    assert_eq!(packetized.data(), expected);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_packetization_kernel_matches_native_scalar_refinement_pass_packet_when_required() {
    if !runtime_required() {
        return;
    }

    let payload = [0x12, 0x34, 0x56, 0x78, 0x9a];
    let code_block = J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: 3,
        ht_refinement_length: 2,
        num_coding_passes: 3,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let subband = J2kPacketizationSubband {
        code_blocks: vec![code_block],
        num_cbs_x: 1,
        num_cbs_y: 1,
    };
    let resolution = J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let descriptor = J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    };
    let expected =
        signinum_j2k_native::encode_j2k_packetization_scalar(J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 1,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[descriptor],
            resolutions: &[resolution],
        })
        .expect("native scalar refinement packetization");

    let context = CudaContext::system_default().expect("CUDA context");
    let packetized = context
        .packetize_htj2k_cleanup_packets(
            &payload,
            &[CudaHtj2kPacketizationPacket {
                block_start: 0,
                block_count: 1,
                subband_start: 0,
                subband_count: 1,
                output_capacity: 512,
                layer: 0,
            }],
            &[CudaHtj2kPacketizationSubband {
                block_start: 0,
                block_count: 1,
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
            &[CudaHtj2kPacketizationBlock {
                data_offset: 0,
                data_len: u32::try_from(payload.len()).expect("test payload length fits in u32"),
                cleanup_length: 3,
                refinement_length: 2,
                num_coding_passes: 3,
                num_zero_bitplanes: 2,
                l_block: 3,
                previously_included: 0,
                inclusion_layer: 0,
            }],
        )
        .expect("CUDA refinement packetization");

    assert_eq!(packetized.execution().kernel_dispatches(), 1);
    assert!(packetized.stage_timings().packetize_us > 0);
    assert!(packetized.statuses().iter().all(|status| status.is_ok()));
    assert_eq!(packetized.data(), expected);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_packetization_kernel_matches_native_scalar_previously_included_layer_when_required() {
    if !runtime_required() {
        return;
    }

    let first_payload = [0x11u8; 20];
    let second_payload = [0x22u8; 5];
    let first_block = J2kPacketizationCodeBlock {
        data: &first_payload,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let second_block = J2kPacketizationCodeBlock {
        data: &second_payload,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let resolutions = [
        J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![first_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        },
        J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![second_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        },
    ];
    let descriptors = [
        J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
        J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 0,
            layer: 1,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let expected =
        signinum_j2k_native::encode_j2k_packetization_scalar(J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 2,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        })
        .expect("native scalar stateful packetization");

    let payload = [first_payload.as_slice(), second_payload.as_slice()].concat();
    let context = CudaContext::system_default().expect("CUDA context");
    let packetized = context
        .packetize_htj2k_cleanup_packets(
            &payload,
            &[
                CudaHtj2kPacketizationPacket {
                    block_start: 0,
                    block_count: 1,
                    subband_start: 0,
                    subband_count: 1,
                    output_capacity: 512,
                    layer: 0,
                },
                CudaHtj2kPacketizationPacket {
                    block_start: 1,
                    block_count: 1,
                    subband_start: 1,
                    subband_count: 1,
                    output_capacity: 512,
                    layer: 1,
                },
            ],
            &[
                CudaHtj2kPacketizationSubband {
                    block_start: 0,
                    block_count: 1,
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
                CudaHtj2kPacketizationSubband {
                    block_start: 1,
                    block_count: 1,
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
            ],
            &[
                CudaHtj2kPacketizationBlock {
                    data_offset: 0,
                    data_len: u32::try_from(first_payload.len())
                        .expect("test payload length fits in u32"),
                    cleanup_length: 0,
                    refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 2,
                    l_block: 3,
                    previously_included: 0,
                    inclusion_layer: 0,
                },
                CudaHtj2kPacketizationBlock {
                    data_offset: u32::try_from(first_payload.len())
                        .expect("test payload length fits in u32"),
                    data_len: u32::try_from(second_payload.len())
                        .expect("test payload length fits in u32"),
                    cleanup_length: 0,
                    refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 2,
                    l_block: 5,
                    previously_included: 1,
                    inclusion_layer: 0,
                },
            ],
        )
        .expect("CUDA stateful packetization");

    assert_eq!(packetized.execution().kernel_dispatches(), 1);
    assert!(packetized.stage_timings().packetize_us > 0);
    assert!(packetized.statuses().iter().all(|status| status.is_ok()));
    assert_eq!(packetized.data(), expected);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_packetization_kernel_matches_native_scalar_deferred_first_inclusion_when_required() {
    if !runtime_required() {
        return;
    }

    let payload = [0x44u8; 5];
    let first_block = J2kPacketizationCodeBlock {
        data: &[],
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 0,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let second_block = J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let resolutions = [
        J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![first_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        },
        J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![second_block],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        },
    ];
    let descriptors = [
        J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
        J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 0,
            layer: 1,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let expected =
        signinum_j2k_native::encode_j2k_packetization_scalar(J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 2,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        })
        .expect("native scalar deferred inclusion packetization");

    let context = CudaContext::system_default().expect("CUDA context");
    let packetized = context
        .packetize_htj2k_cleanup_packets(
            &payload,
            &[
                CudaHtj2kPacketizationPacket {
                    block_start: 0,
                    block_count: 1,
                    subband_start: 0,
                    subband_count: 1,
                    output_capacity: 512,
                    layer: 0,
                },
                CudaHtj2kPacketizationPacket {
                    block_start: 1,
                    block_count: 1,
                    subband_start: 1,
                    subband_count: 1,
                    output_capacity: 512,
                    layer: 1,
                },
            ],
            &[
                CudaHtj2kPacketizationSubband {
                    block_start: 0,
                    block_count: 1,
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
                CudaHtj2kPacketizationSubband {
                    block_start: 1,
                    block_count: 1,
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
            ],
            &[
                CudaHtj2kPacketizationBlock {
                    data_offset: 0,
                    data_len: 0,
                    cleanup_length: 0,
                    refinement_length: 0,
                    num_coding_passes: 0,
                    num_zero_bitplanes: 2,
                    l_block: 3,
                    previously_included: 0,
                    inclusion_layer: 1,
                },
                CudaHtj2kPacketizationBlock {
                    data_offset: 0,
                    data_len: u32::try_from(payload.len())
                        .expect("test payload length fits in u32"),
                    cleanup_length: 0,
                    refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 2,
                    l_block: 3,
                    previously_included: 0,
                    inclusion_layer: 1,
                },
            ],
        )
        .expect("CUDA deferred inclusion packetization");

    assert_eq!(packetized.execution().kernel_dispatches(), 1);
    assert!(packetized.stage_timings().packetize_us > 0);
    assert!(packetized.statuses().iter().all(|status| status.is_ok()));
    assert_eq!(packetized.data(), expected);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_htj2k_packetization_kernel_matches_native_scalar_deferred_first_inclusion_after_non_empty_packet_when_required(
) {
    if !runtime_required() {
        return;
    }

    let first_payload = [0x11u8; 3];
    let second_payload = [0x22u8; 5];
    let resolutions = [
        J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![
                    J2kPacketizationCodeBlock {
                        data: &first_payload,
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                    J2kPacketizationCodeBlock {
                        data: &[],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 0,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                ],
                num_cbs_x: 2,
                num_cbs_y: 1,
            }],
        },
        J2kPacketizationResolution {
            subbands: vec![J2kPacketizationSubband {
                code_blocks: vec![
                    J2kPacketizationCodeBlock {
                        data: &[],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 0,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                    J2kPacketizationCodeBlock {
                        data: &second_payload,
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        num_zero_bitplanes: 2,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                ],
                num_cbs_x: 2,
                num_cbs_y: 1,
            }],
        },
    ];
    let descriptors = [
        J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
        J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 0,
            layer: 1,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let expected =
        signinum_j2k_native::encode_j2k_packetization_scalar(J2kPacketizationEncodeJob {
            resolution_count: 2,
            num_layers: 2,
            num_components: 1,
            code_block_count: 4,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &descriptors,
            resolutions: &resolutions,
        })
        .expect("native scalar deferred first inclusion after non-empty packetization");

    let payload = [first_payload.as_slice(), second_payload.as_slice()].concat();
    let context = CudaContext::system_default().expect("CUDA context");
    let packetized = context
        .packetize_htj2k_cleanup_packets_with_tag_state(
            &payload,
            &[
                CudaHtj2kPacketizationPacket {
                    block_start: 0,
                    block_count: 2,
                    subband_start: 0,
                    subband_count: 1,
                    output_capacity: 512,
                    layer: 0,
                },
                CudaHtj2kPacketizationPacket {
                    block_start: 2,
                    block_count: 2,
                    subband_start: 1,
                    subband_count: 1,
                    output_capacity: 512,
                    layer: 1,
                },
            ],
            &[
                CudaHtj2kPacketizationSubband {
                    block_start: 0,
                    block_count: 2,
                    num_cbs_x: 2,
                    num_cbs_y: 1,
                },
                CudaHtj2kPacketizationSubband {
                    block_start: 2,
                    block_count: 2,
                    num_cbs_x: 2,
                    num_cbs_y: 1,
                },
            ],
            &[
                CudaHtj2kPacketizationBlock {
                    data_offset: 0,
                    data_len: u32::try_from(first_payload.len())
                        .expect("test payload length fits in u32"),
                    cleanup_length: 0,
                    refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 2,
                    l_block: 3,
                    previously_included: 0,
                    inclusion_layer: 0,
                },
                CudaHtj2kPacketizationBlock {
                    data_offset: 0,
                    data_len: 0,
                    cleanup_length: 0,
                    refinement_length: 0,
                    num_coding_passes: 0,
                    num_zero_bitplanes: 2,
                    l_block: 3,
                    previously_included: 0,
                    inclusion_layer: 1,
                },
                CudaHtj2kPacketizationBlock {
                    data_offset: 0,
                    data_len: 0,
                    cleanup_length: 0,
                    refinement_length: 0,
                    num_coding_passes: 0,
                    num_zero_bitplanes: 2,
                    l_block: 3,
                    previously_included: 1,
                    inclusion_layer: 0,
                },
                CudaHtj2kPacketizationBlock {
                    data_offset: u32::try_from(first_payload.len())
                        .expect("test payload length fits in u32"),
                    data_len: u32::try_from(second_payload.len())
                        .expect("test payload length fits in u32"),
                    cleanup_length: 0,
                    refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 2,
                    l_block: 3,
                    previously_included: 0,
                    inclusion_layer: 1,
                },
            ],
            &[
                CudaHtj2kPacketizationSubbandTagState {
                    inclusion_node_start: 0,
                    zero_bitplane_node_start: 3,
                    node_count: 3,
                    reserved0: 0,
                },
                CudaHtj2kPacketizationSubbandTagState {
                    inclusion_node_start: 6,
                    zero_bitplane_node_start: 9,
                    node_count: 3,
                    reserved0: 0,
                },
            ],
            &[
                CudaHtj2kPacketizationTagNodeState {
                    current: 0,
                    known: 0,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 0,
                    known: 0,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 0,
                    known: 0,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 0,
                    known: 0,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 0,
                    known: 0,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 0,
                    known: 0,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 0,
                    known: 1,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 1,
                    known: 0,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 0,
                    known: 1,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 2,
                    known: 1,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 0,
                    known: 0,
                },
                CudaHtj2kPacketizationTagNodeState {
                    current: 2,
                    known: 1,
                },
            ],
        )
        .expect("CUDA deferred first inclusion after non-empty packetization");

    assert_eq!(packetized.execution().kernel_dispatches(), 1);
    assert!(packetized.stage_timings().packetize_us > 0);
    assert!(packetized.statuses().iter().all(|status| status.is_ok()));
    assert_eq!(packetized.data(), expected);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_mct_and_rgb_store_kernels_match_reversible_cpu_transform_when_required() {
    if !runtime_required() {
        return;
    }

    let mut plane0 = [12.0f32, 25.0, 40.0, 60.0];
    let mut plane1 = [-3.0f32, 6.0, -10.0, 12.0];
    let mut plane2 = [5.0f32, -7.0, 11.0, -13.0];
    let mut expected = Vec::with_capacity(plane0.len() * 3);
    for ((y0, y1), y2) in plane0
        .iter()
        .copied()
        .zip(plane1.iter().copied())
        .zip(plane2.iter().copied())
    {
        let green = y0 - ((y2 + y1) * 0.25).floor();
        expected.push(rounded_u8(y2 + green + 128.0));
        expected.push(rounded_u8(green + 128.0));
        expected.push(rounded_u8(y1 + green + 128.0));
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0_buffer = context.upload_f32(&plane0).expect("upload plane 0");
    let plane1_buffer = context.upload_f32(&plane1).expect("upload plane 1");
    let plane2_buffer = context.upload_f32(&plane2).expect("upload plane 2");
    let stats = context
        .j2k_inverse_mct_device(
            &plane0_buffer,
            &plane1_buffer,
            &plane2_buffer,
            CudaJ2kInverseMctJob {
                len: u32::try_from(plane0.len()).expect("test plane length fits in u32"),
                irreversible97: 0,
                addend0: 128.0,
                addend1: 128.0,
                addend2: 128.0,
            },
        )
        .expect("CUDA inverse RCT");
    assert_eq!(stats.kernel_dispatches(), 1);

    let stored = context
        .j2k_store_rgb8_device(
            &plane0_buffer,
            &plane1_buffer,
            &plane2_buffer,
            CudaJ2kStoreRgb8Job {
                input_width0: 2,
                input_width1: 2,
                input_width2: 2,
                source_x0: 0,
                source_y0: 0,
                source_x1: 0,
                source_y1: 0,
                source_x2: 0,
                source_y2: 0,
                copy_width: 2,
                copy_height: 2,
                output_width: 2,
                output_height: 2,
                output_x: 0,
                output_y: 0,
                addend0: 0.0,
                addend1: 0.0,
                addend2: 0.0,
                bit_depth0: 8,
                bit_depth1: 8,
                bit_depth2: 8,
                rgba: 0,
            },
        )
        .expect("CUDA RGB store");
    assert_eq!(stored.execution().kernel_dispatches(), 1);

    let mut actual = vec![0u8; expected.len()];
    stored
        .buffer()
        .copy_to_host(&mut actual)
        .expect("download RGB pixels");
    assert_eq!(actual, expected);

    plane0[0] = 0.0;
    plane1[0] = 0.0;
    plane2[0] = 0.0;
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_gray16_and_rgb16_store_kernels_match_cpu_scaling_when_required() {
    if !runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let gray = [0.0f32, 128.0, 255.0, 300.0];
    let gray_buffer = context.upload_f32(&gray).expect("upload gray plane");
    let gray_output = context
        .j2k_store_gray16_device(
            &gray_buffer,
            CudaJ2kStoreGray16Job {
                input_width: 2,
                source_x: 0,
                source_y: 0,
                copy_width: 2,
                copy_height: 2,
                output_width: 2,
                output_height: 2,
                output_x: 0,
                output_y: 0,
                addend: 0.0,
                bit_depth: 8,
            },
        )
        .expect("CUDA Gray16 store");
    assert_eq!(gray_output.execution().kernel_dispatches(), 1);

    let mut expected_gray = Vec::with_capacity(gray.len() * 2);
    for sample in gray {
        push_u16_ne(&mut expected_gray, rounded_u16(sample, 8));
    }
    let mut actual_gray = vec![0u8; expected_gray.len()];
    gray_output
        .buffer()
        .copy_to_host(&mut actual_gray)
        .expect("download Gray16 pixels");
    assert_eq!(actual_gray, expected_gray);

    let red = [0.0f32, 64.0, 128.0, 255.0];
    let green = [255.0f32, 128.0, 64.0, 0.0];
    let blue = [12.0f32, 34.0, 56.0, 78.0];
    let red_buffer = context.upload_f32(&red).expect("upload red plane");
    let green_buffer = context.upload_f32(&green).expect("upload green plane");
    let blue_buffer = context.upload_f32(&blue).expect("upload blue plane");
    let rgb_output = context
        .j2k_store_rgb16_device(
            &red_buffer,
            &green_buffer,
            &blue_buffer,
            CudaJ2kStoreRgb16Job {
                input_width0: 2,
                input_width1: 2,
                input_width2: 2,
                source_x0: 0,
                source_y0: 0,
                source_x1: 0,
                source_y1: 0,
                source_x2: 0,
                source_y2: 0,
                copy_width: 2,
                copy_height: 2,
                output_width: 2,
                output_height: 2,
                output_x: 0,
                output_y: 0,
                addend0: 0.0,
                addend1: 0.0,
                addend2: 0.0,
                bit_depth0: 8,
                bit_depth1: 8,
                bit_depth2: 8,
                rgba: 1,
            },
        )
        .expect("CUDA RGBA16 store");
    assert_eq!(rgb_output.execution().kernel_dispatches(), 1);

    let mut expected_rgb = Vec::with_capacity(red.len() * 4 * 2);
    for ((r, g), b) in red.into_iter().zip(green).zip(blue) {
        push_u16_ne(&mut expected_rgb, rounded_u16(r, 8));
        push_u16_ne(&mut expected_rgb, rounded_u16(g, 8));
        push_u16_ne(&mut expected_rgb, rounded_u16(b, 8));
        push_u16_ne(&mut expected_rgb, u16::MAX);
    }
    let mut actual_rgb = vec![0u8; expected_rgb.len()];
    rgb_output
        .buffer()
        .copy_to_host(&mut actual_rgb)
        .expect("download RGBA16 pixels");
    assert_eq!(actual_rgb, expected_rgb);
}
