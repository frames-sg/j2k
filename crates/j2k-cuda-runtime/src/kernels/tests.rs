// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn kernel_inventory_forbids_test_only_orphan_entrypoints() {
    let kernel_source = include_str!("../kernels.rs");
    let production_kernel_source = kernel_source
        .split("\n#[cfg(test)]\nmod tests")
        .next()
        .expect("production kernel source");
    assert!(
        !production_kernel_source.contains("#[cfg_attr(not(test), allow(dead_code))]"),
        "production CUDA kernels must not use test-only dead-code exemptions"
    );

    let context_source = include_str!("../context.rs");
    for variant in [
        "J2kIdwtHorizontal",
        "J2kIdwtVertical",
        "Htj2kEncodeCodeblock",
        "J2kInverseDwtSingle",
        "J2kStoreRgb8Mct",
    ] {
        assert!(
            !production_kernel_source.contains(&format!("{variant},")),
            "orphan CUDA kernel variant returned: {variant}"
        );
        assert!(
            !context_source.contains(&format!("{variant},")),
            "test kernel inventory must not retain orphan variant: {variant}"
        );
    }

    for (source, entrypoint) in [
        (
            include_str!("../cuda_oxide_j2k_idwt/simt/src/main.rs"),
            "j2k_idwt_horizontal",
        ),
        (
            include_str!("../cuda_oxide_j2k_idwt/simt/src/main.rs"),
            "j2k_idwt_vertical",
        ),
        (
            include_str!("../cuda_oxide_j2k_idwt/simt/src/main.rs"),
            "j2k_inverse_dwt_single",
        ),
        (
            include_str!("../cuda_oxide_htj2k_encode/simt/src/main.rs"),
            "j2k_htj2k_encode_codeblock",
        ),
        (
            include_str!("../cuda_oxide_j2k_decode_store/simt/src/main.rs"),
            "j2k_store_rgb8_mct",
        ),
    ] {
        assert!(
            !source.contains(&format!("fn {entrypoint}(")),
            "orphan CUDA device entrypoint returned: {entrypoint}"
        );
    }
}

#[cfg(all(feature = "cuda-oxide-copy-u8", j2k_cuda_oxide_copy_u8_built))]
#[test]
fn cuda_oxide_copy_u8_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_copy_u8_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    assert!(source.contains(".visible .entry j2k_copy_u8("));
    assert_eq!(CudaKernel::CopyU8.entrypoint(), b"j2k_copy_u8\0");
}

#[test]
fn jpeg_decode_entrypoints_are_stable() {
    assert_eq!(CudaKernel::CopyU8.entrypoint(), b"j2k_copy_u8\0");
    assert_eq!(
        CudaKernel::JpegDecodeFast420Rgb8.entrypoint(),
        b"j2k_jpeg_decode_fast420_rgb8\0"
    );
    assert_eq!(
        CudaKernel::JpegDecodeFast422Rgb8.entrypoint(),
        b"j2k_jpeg_decode_fast422_rgb8\0"
    );
    assert_eq!(
        CudaKernel::JpegDecodeFast444Rgb8.entrypoint(),
        b"j2k_jpeg_decode_fast444_rgb8\0"
    );
    assert_eq!(
        CudaKernel::JpegSubsampledPlanesToRgb8.entrypoint(),
        b"j2k_jpeg_subsampled_planes_to_rgb8\0"
    );
    assert_eq!(
        CudaKernel::JpegEntropySync420.entrypoint(),
        b"j2k_jpeg_entropy_sync420\0"
    );
    assert_eq!(
        CudaKernel::JpegEntropyOverflow420.entrypoint(),
        b"j2k_jpeg_entropy_overflow420\0"
    );
}

#[cfg(all(feature = "cuda-oxide-jpeg-decode", j2k_cuda_oxide_jpeg_decode_built))]
#[test]
fn cuda_oxide_jpeg_decode_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_jpeg_decode_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    let kernels = [
        CudaKernel::JpegDecodeFast420Rgb8,
        CudaKernel::JpegDecodeFast422Rgb8,
        CudaKernel::JpegDecodeFast444Rgb8,
        CudaKernel::JpegSubsampledPlanesToRgb8,
        CudaKernel::JpegEntropySync420,
        CudaKernel::JpegEntropyOverflow420,
    ];
    for kernel in kernels {
        assert!(kernel.is_cuda_oxide_jpeg_decode_stage());
        let entrypoint = std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
            .expect("entrypoint utf8");
        assert!(
            source.contains(&format!(".visible .entry {entrypoint}(")),
            "missing cuda-oxide JPEG decode entrypoint {entrypoint}"
        );
    }
}

#[cfg(all(feature = "cuda-oxide-jpeg-encode", j2k_cuda_oxide_jpeg_encode_built))]
#[test]
fn cuda_oxide_jpeg_encode_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_jpeg_encode_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    let kernels = [
        CudaKernel::JpegEncodeBaselineEntropy,
        CudaKernel::JpegEncodeBaselineEntropyBatch,
    ];
    for kernel in kernels {
        assert!(kernel.is_cuda_oxide_jpeg_encode_stage());
        let entrypoint = std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
            .expect("entrypoint utf8");
        assert!(
            source.contains(&format!(".visible .entry {entrypoint}(")),
            "missing cuda-oxide JPEG encode entrypoint {entrypoint}"
        );
    }
}

#[test]
fn htj2k_sample_geometry_uses_threads_with_one_block_per_codeblock() {
    let geometry = htj2k_codeblock_sample_launch_geometry(3).expect("geometry");
    assert_eq!(geometry.grid(), (3, 1, 1));
    assert_eq!(geometry.block(), (COPY_U8_THREADS_CUDA, 1, 1));
}

#[test]
fn htj2k_cleanup_decode_geometry_packs_large_batches_into_warps() {
    let small_geometry = htj2k_codeblock_launch_geometry(1_200).expect("small geometry");
    assert_eq!(small_geometry.grid(), (1_200, 1, 1));
    assert_eq!(small_geometry.block(), (1, 1, 1));

    let large_geometry = htj2k_codeblock_launch_geometry(2_048).expect("large geometry");
    assert_eq!(large_geometry.grid(), (64, 1, 1));
    assert_eq!(large_geometry.block(), (32, 1, 1));
}

#[test]
fn htj2k_sigprop_forward_reader_discards_a_set_stuffed_overlap_bit() {
    let data = [0xFF_u8, 0x80, 0x00, 0x00, 0x00];
    let mut tmp = 0_u64;
    let mut bits = 0_u32;
    let mut unstuff = false;
    for byte in data {
        let valid_bits = 8 - u32::from(unstuff);
        let next_unstuff = byte == 0xFF;
        let byte = if unstuff { byte & 0x7F } else { byte };
        tmp |= u64::from(byte) << bits;
        bits += valid_bits;
        unstuff = next_unstuff;
    }
    assert_eq!(
        u32::try_from(tmp).expect("low CUDA reservoir word"),
        0x0000_00FF
    );

    let device = include_str!("../cuda_oxide_htj2k_decode/simt/src/main.rs");
    let fill = device
        .split("fn forward_reader_fill")
        .nth(1)
        .expect("CUDA HT forward-reader fill")
        .split("fn forward_reader_fetch")
        .next()
        .expect("CUDA HT forward-reader fill body");
    assert!(
        fill.contains("let byte = if reader.unstuff { byte & 0x7f } else { byte };")
            || fill.contains("let byte = if reader.unstuff { byte & 0x7F } else { byte };")
    );
    assert!(fill.contains("reader.unstuff = next_unstuff;"));
}

#[test]
fn htj2k_sigprop_quad_preserves_the_next_above_stripe_context() {
    let mut previous_row = [0x0000_u16, 0xA5A5];
    let new_sig = 0x0000_0088_u32;
    let cleanup_sig_pair = 0xF00F_0011_u32;
    let combined_sig = new_sig | (cleanup_sig_pair & 0xFFFF);
    previous_row[0] = u16::try_from(combined_sig).expect("low significance half");
    assert_eq!(previous_row, [0x0099, 0xA5A5]);

    let device = include_str!("../cuda_oxide_htj2k_decode/simt/src/main.rs");
    let sigprop = device
        .split("fn apply_significance_propagation")
        .nth(1)
        .expect("CUDA HT SigProp phase")
        .split("fn apply_magnitude_refinement")
        .next()
        .expect("CUDA HT SigProp phase body");
    assert!(sigprop.contains("let combined_sig = new_sig | (cs & 0xffff);"));
    assert!(!sigprop.contains("prev_row_sig[idx as usize + 1] ="));
}

#[test]
fn htj2k_magref_reverse_reader_discards_a_set_stuffed_overlap_bit() {
    let data = [0x00_u8, 0x00, 0x00, 0x00, 0xFF];
    let mut tmp = 0_u64;
    let mut bits = 0_u32;
    let mut unstuff = true;
    for raw in data.into_iter().rev() {
        let stuffed = unstuff && (raw & 0x7F) == 0x7F;
        let valid_bits = 8 - u32::from(stuffed);
        let next_unstuff = raw > 0x8F;
        let byte = if stuffed { raw & 0x7F } else { raw };
        tmp |= u64::from(byte) << bits;
        bits += valid_bits;
        unstuff = next_unstuff;
    }
    assert_eq!(
        u32::try_from(tmp).expect("low CUDA reservoir word"),
        0x0000_007F
    );

    let device = include_str!("../cuda_oxide_htj2k_decode/simt/src/main.rs");
    let fill = device
        .split("fn reverse_reader_fill")
        .nth(1)
        .expect("CUDA HT reverse-reader fill")
        .split("fn reverse_reader_fetch")
        .next()
        .expect("CUDA HT reverse-reader fill body");
    assert!(fill.contains("let stuffed = reader.unstuff && (byte & 0x7f) == 0x7f;"));
    assert!(fill.contains("let byte = if stuffed { byte & 0x7f } else { byte };"));
    assert!(fill.contains("reader.unstuff = next_unstuff;"));
}

#[test]
fn classic_decode_geometry_and_device_stride_share_a_classic_owned_constant() {
    let geometry = j2k_classic_codeblock_launch_geometry(3).expect("classic geometry");
    assert_eq!(geometry.grid(), (3, 1, 1));
    assert_eq!(geometry.block(), (32, 1, 1));

    let host = include_str!("j2k.rs");
    let geometry_source = host
        .split("pub(crate) fn j2k_classic_codeblock_launch_geometry")
        .nth(1)
        .expect("classic geometry source")
        .split('}')
        .next()
        .expect("classic geometry body");
    assert!(geometry_source.contains("CLASSIC_DECODE_CODEBLOCK_THREADS_CUDA"));
    assert!(!geometry_source.contains("HTJ2K_DECODE_CODEBLOCK_THREADS_CUDA"));

    let device = include_str!("../cuda_oxide_j2k_classic_decode/simt/src/main.rs");
    let entrypoint = device
        .split("pub unsafe fn j2k_decode_classic_codeblocks_multi")
        .nth(1)
        .expect("classic device entrypoint");
    assert!(device.contains("const CLASSIC_DECODE_THREADS: u32 = 32;"));
    assert!(entrypoint.contains("index += CLASSIC_DECODE_THREADS"));
    assert!(entrypoint.contains("sample += CLASSIC_DECODE_THREADS"));
    assert!(!entrypoint.contains("index += 32"));
    assert!(!entrypoint.contains("sample += 32"));
}

#[test]
fn classic_device_validates_job_before_using_job_dimensions() {
    let device = include_str!("../cuda_oxide_j2k_classic_decode/simt/src/main.rs");
    let entrypoint = device
        .split("pub unsafe fn j2k_decode_classic_codeblocks_multi")
        .nth(1)
        .expect("classic device entrypoint");
    let validation = entrypoint
        .find("validate_job_header(")
        .expect("early device job validation");
    let coefficient_count = entrypoint
        .find("let coefficient_count")
        .expect("coefficient count");
    assert!(validation < coefficient_count);
    assert!(device.contains("segment.end_coding_pass > job.number_of_coding_passes"));
    assert!(entrypoint.contains("job.output_offset as usize"));
    assert!(entrypoint.contains("y as usize * job.output_stride as usize"));
}

#[test]
fn htj2k_encode_geometry_uses_cooperative_threads_per_codeblock() {
    let geometry = htj2k_encode_codeblock_launch_geometry(327).expect("geometry");
    assert_eq!(geometry.grid(), (327, 1, 1));
    assert_eq!(geometry.block(), (128, 1, 1));
}

#[test]
fn htj2k_packetize_geometry_uses_cooperative_threads_per_packet() {
    let geometry = htj2k_packetize_launch_geometry(5).expect("geometry");
    assert_eq!(geometry.grid(), (5, 1, 1));
    assert_eq!(geometry.block(), (COPY_U8_THREADS_CUDA, 1, 1));
}

#[test]
fn j2k_encode_entrypoints_are_stable() {
    assert_eq!(
        CudaKernel::J2kDeinterleaveToF32.entrypoint(),
        b"j2k_deinterleave_to_f32\0"
    );
    assert_eq!(CudaKernel::J2kForwardRct.entrypoint(), b"j2k_forward_rct\0");
    assert_eq!(CudaKernel::J2kForwardIct.entrypoint(), b"j2k_forward_ict\0");
    assert_eq!(
        CudaKernel::J2kForwardDwt53Horizontal.entrypoint(),
        b"j2k_forward_dwt53_horizontal\0"
    );
    assert_eq!(
        CudaKernel::J2kForwardDwt53Vertical.entrypoint(),
        b"j2k_forward_dwt53_vertical\0"
    );
    assert_eq!(
        CudaKernel::J2kForwardDwt97Horizontal.entrypoint(),
        b"j2k_forward_dwt97_horizontal\0"
    );
    assert_eq!(
        CudaKernel::J2kForwardDwt97Vertical.entrypoint(),
        b"j2k_forward_dwt97_vertical\0"
    );
    assert_eq!(
        CudaKernel::J2kQuantizeSubband.entrypoint(),
        b"j2k_quantize_subband\0"
    );
    assert_eq!(
        CudaKernel::J2kQuantizeSubbandStrided.entrypoint(),
        b"j2k_quantize_subband_strided\0"
    );
}

#[cfg(all(feature = "cuda-oxide-j2k-encode", j2k_cuda_oxide_j2k_encode_built))]
#[test]
fn cuda_oxide_j2k_encode_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_j2k_encode_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    let kernels = [
        CudaKernel::J2kDeinterleaveToF32,
        CudaKernel::J2kDeinterleaveStridedToF32,
        CudaKernel::J2kForwardRct,
        CudaKernel::J2kForwardIct,
        CudaKernel::J2kForwardDwt53Horizontal,
        CudaKernel::J2kForwardDwt53Vertical,
        CudaKernel::J2kForwardDwt97Horizontal,
        CudaKernel::J2kForwardDwt97Vertical,
        CudaKernel::J2kQuantizeSubband,
        CudaKernel::J2kQuantizeSubbandStrided,
        CudaKernel::Htj2kCompactCodeblocks,
        CudaKernel::Htj2kPacketizeCleanup,
    ];
    for kernel in kernels {
        assert!(kernel.is_cuda_oxide_j2k_encode_stage());
        let entrypoint = std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
            .expect("entrypoint utf8");
        assert!(
            source.contains(&format!(".visible .entry {entrypoint}(")),
            "missing cuda-oxide J2K encode entrypoint {entrypoint}"
        );
    }
}

#[cfg(all(
    feature = "cuda-oxide-j2k-decode-store",
    j2k_cuda_oxide_j2k_decode_store_built
))]
#[test]
fn cuda_oxide_j2k_decode_store_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_j2k_decode_store_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    let kernels = [
        CudaKernel::J2kInverseMct,
        CudaKernel::J2kStoreGray8,
        CudaKernel::J2kStoreGray16,
        CudaKernel::J2kStoreRgb8,
        CudaKernel::J2kStoreRgb8MctBatch,
        CudaKernel::J2kStoreRgb8NativeBatch,
        CudaKernel::J2kStoreRgb16NativeBatch,
        CudaKernel::J2kStoreRgbI16NativeBatch,
        CudaKernel::J2kStoreRgba8NativeBatch,
        CudaKernel::J2kStoreRgba16NativeBatch,
        CudaKernel::J2kStoreRgbaI16NativeBatch,
        CudaKernel::J2kStoreRgb16,
        CudaKernel::J2kStoreRgb16Mct,
    ];
    for kernel in kernels {
        assert!(kernel.is_j2k_decode_store_stage());
        let entrypoint = std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
            .expect("entrypoint utf8");
        assert!(
            source.contains(&format!(".visible .entry {entrypoint}(")),
            "missing cuda-oxide J2K decode-store entrypoint {entrypoint}"
        );
    }
}

#[cfg(all(
    feature = "cuda-oxide-j2k-dequantize",
    j2k_cuda_oxide_j2k_dequantize_built
))]
#[test]
fn cuda_oxide_j2k_dequantize_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_j2k_dequantize_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    let kernels = [
        CudaKernel::J2kDequantizeHtj2kCodeblocks,
        CudaKernel::J2kDequantizeHtj2kCodeblocksMulti,
        CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti,
    ];
    for kernel in kernels {
        assert!(kernel.is_j2k_dequantize_stage());
        let entrypoint = std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
            .expect("entrypoint utf8");
        assert!(
            source.contains(&format!(".visible .entry {entrypoint}(")),
            "missing cuda-oxide J2K dequantize entrypoint {entrypoint}"
        );
    }
}

#[cfg(all(feature = "cuda-oxide-j2k-idwt", j2k_cuda_oxide_j2k_idwt_built))]
#[test]
fn cuda_oxide_j2k_idwt_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_j2k_idwt_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    let kernels = [
        CudaKernel::J2kIdwtInterleave,
        CudaKernel::J2kIdwtInterleaveHorizontalMulti,
        CudaKernel::J2kIdwtInterleaveHorizontal53Multi,
        CudaKernel::J2kIdwtInterleaveHorizontal97Multi,
        CudaKernel::J2kIdwtHorizontal53,
        CudaKernel::J2kIdwtHorizontal97,
        CudaKernel::J2kIdwtVerticalMulti,
        CudaKernel::J2kIdwtVertical53Multi,
        CudaKernel::J2kIdwtVertical97Multi,
        CudaKernel::J2kIdwtVertical97MultiCols4,
        CudaKernel::J2kIdwtVertical53,
        CudaKernel::J2kIdwtVertical97,
    ];
    for kernel in kernels {
        assert!(kernel.is_j2k_idwt_stage());
        let entrypoint = std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
            .expect("entrypoint utf8");
        assert!(
            source.contains(&format!(".visible .entry {entrypoint}(")),
            "missing cuda-oxide J2K IDWT entrypoint {entrypoint}"
        );
    }
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
#[test]
fn cuda_oxide_transcode_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_transcode_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    let kernels = [
        CudaKernel::TranscodeReversible53Idct,
        CudaKernel::TranscodeReversible53VerticalLow,
        CudaKernel::TranscodeReversible53VerticalHigh,
        CudaKernel::TranscodeReversible53HorizontalLow,
        CudaKernel::TranscodeReversible53HorizontalHigh,
        CudaKernel::TranscodeDwt97Idct,
        CudaKernel::TranscodeDwt97RowLift,
        CudaKernel::TranscodeDwt97ColumnLift,
        CudaKernel::TranscodeDwt97IdctBatch,
        CudaKernel::TranscodeDwt97IdctI16Batch,
        CudaKernel::TranscodeDwt97RowLiftBatch,
        CudaKernel::TranscodeDwt97RowLiftBatchCoop,
        CudaKernel::TranscodeDwt97ColumnLiftBatch,
        CudaKernel::TranscodeDwt97QuantizeCodeblocks,
        CudaKernel::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch,
    ];
    for kernel in kernels {
        assert!(kernel.is_cuda_oxide_transcode_stage());
        let entrypoint = std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
            .expect("entrypoint utf8");
        assert!(
            source.contains(&format!(".visible .entry {entrypoint}(")),
            "missing cuda-oxide transcode entrypoint {entrypoint}"
        );
    }
}

#[cfg(all(feature = "cuda-oxide-htj2k-decode", j2k_cuda_oxide_htj2k_decode_built))]
#[test]
fn cuda_oxide_htj2k_decode_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_htj2k_decode_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    let kernels = [
        CudaKernel::Htj2kDecodeCodeblocks,
        CudaKernel::Htj2kDecodeCodeblocksMulti,
        CudaKernel::Htj2kDecodeCodeblocksMultiCleanupOnly,
        CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize,
    ];
    for kernel in kernels {
        assert!(kernel.is_htj2k_decode_stage());
        let entrypoint = std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
            .expect("entrypoint utf8");
        assert!(
            source.contains(&format!(".visible .entry {entrypoint}(")),
            "missing cuda-oxide HTJ2K decode entrypoint {entrypoint}"
        );
    }
}

#[cfg(all(feature = "cuda-oxide-htj2k-encode", j2k_cuda_oxide_htj2k_encode_built))]
#[test]
fn cuda_oxide_htj2k_encode_kernel_metadata_matches_generated_ptx() {
    let ptx = cuda_oxide_htj2k_encode_ptx();
    assert_eq!(ptx.last(), Some(&0));
    let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
    let kernels = [
        CudaKernel::Htj2kEncodeCodeblocks,
        CudaKernel::Htj2kEncodeCodeblocksMultiInput,
        CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup,
        CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64,
    ];
    for kernel in kernels {
        assert!(kernel.is_htj2k_encode_codeblock_stage());
        let entrypoint = std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
            .expect("entrypoint utf8");
        assert!(
            source.contains(&format!(".visible .entry {entrypoint}(")),
            "missing cuda-oxide HTJ2K encode entrypoint {entrypoint}"
        );
    }
}

#[test]
fn transcode_kernel_entrypoints_match_names() {
    assert_eq!(
        CudaKernel::TranscodeDwt97Idct.entrypoint(),
        b"transcode_dwt97_idct\0"
    );
    assert_eq!(
        CudaKernel::TranscodeDwt97RowLift.entrypoint(),
        b"transcode_dwt97_row_lift\0"
    );
    assert_eq!(
        CudaKernel::TranscodeDwt97ColumnLift.entrypoint(),
        b"transcode_dwt97_column_lift\0"
    );
    assert_eq!(
        CudaKernel::TranscodeDwt97IdctBatch.entrypoint(),
        b"transcode_dwt97_idct_batch\0"
    );
    assert_eq!(
        CudaKernel::TranscodeDwt97IdctI16Batch.entrypoint(),
        b"transcode_dwt97_idct_i16_batch\0"
    );
    assert_eq!(
        CudaKernel::TranscodeDwt97RowLiftBatch.entrypoint(),
        b"transcode_dwt97_row_lift_batch\0"
    );
    assert_eq!(
        CudaKernel::TranscodeDwt97RowLiftBatchCoop.entrypoint(),
        b"transcode_dwt97_row_lift_batch_coop\0"
    );
    assert_eq!(
        CudaKernel::TranscodeDwt97ColumnLiftBatch.entrypoint(),
        b"transcode_dwt97_column_lift_batch\0"
    );
    assert_eq!(
        CudaKernel::TranscodeDwt97QuantizeCodeblocks.entrypoint(),
        b"transcode_dwt97_quantize_codeblocks\0"
    );
    assert_eq!(
        CudaKernel::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch.entrypoint(),
        b"transcode_dwt97_column_lift_quantize_codeblocks_batch\0"
    );
}

#[test]
fn copy_u8_launch_geometry_rounds_up_to_256_thread_blocks() {
    assert_eq!(copy_u8_launch_geometry(0), None);
    assert_eq!(copy_u8_launch_geometry(1).unwrap().grid(), (1, 1, 1));
    assert_eq!(copy_u8_launch_geometry(256).unwrap().grid(), (1, 1, 1));
    assert_eq!(copy_u8_launch_geometry(257).unwrap().grid(), (2, 1, 1));
}

#[test]
fn x_blocks_launch_geometry_rounds_work_items_and_preserves_y_grid() {
    let geometry = x_blocks_launch_geometry(513, 7, COPY_U8_THREADS).unwrap();

    assert_eq!(geometry.grid(), (3, 7, 1));
    assert_eq!(geometry.block(), (COPY_U8_THREADS_CUDA, 1, 1));
}

#[test]
fn x_blocks_launch_geometry_rejects_zero_threads() {
    assert_eq!(x_blocks_launch_geometry(513, 7, 0), None);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn x_blocks_launch_geometry_enforces_static_grid_boundaries() {
    let max_work_items = CUDA_MAX_GRID_DIM_X as usize * COPY_U8_THREADS;
    assert!(copy_u8_launch_geometry(max_work_items).is_some());
    assert_eq!(copy_u8_launch_geometry(max_work_items + 1), None);
    assert!(x_blocks_launch_geometry(1, CUDA_MAX_GRID_DIM_Y_Z as usize, 1).is_some());
    assert_eq!(
        x_blocks_launch_geometry(1, CUDA_MAX_GRID_DIM_Y_Z as usize + 1, 1),
        None
    );
}

#[test]
fn with_grid_y_preserves_block_and_other_grid_axes() {
    let base = CudaLaunchGeometry::new((2, 3, 4), (16, 8, 1)).unwrap();

    let geometry = with_grid_y(base, 9).unwrap();

    assert_eq!(geometry.grid(), (2, 9, 4));
    assert_eq!(geometry.block(), base.block());
}

#[test]
fn with_grid_z_preserves_block_and_other_grid_axes() {
    let base = CudaLaunchGeometry::new((2, 3, 4), (16, 8, 1)).unwrap();

    let geometry = with_grid_z(base, 11).unwrap();

    assert_eq!(geometry.grid(), (2, 3, 11));
    assert_eq!(geometry.block(), base.block());
}

#[test]
fn j2k_dwt53_launch_geometry_uses_16_by_16_thread_blocks() {
    let geometry = j2k_dwt53_launch_geometry(17, 33).unwrap();
    assert_eq!(geometry.grid(), (2, 3, 1));
    assert_eq!(geometry.block(), (16, 16, 1));
}

#[test]
fn j2k_dwt53_launch_geometry_enforces_exact_grid_y_boundary() {
    let max_height = CUDA_MAX_GRID_DIM_Y_Z * J2K_ENCODE_THREADS_Y;
    assert!(j2k_dwt53_launch_geometry(1, max_height).is_some());
    assert_eq!(j2k_dwt53_launch_geometry(1, max_height + 1), None);
}

#[test]
fn cuda_launch_geometry_policy_is_centralized_and_defensively_enforced() {
    let geometry = include_str!("geometry.rs");
    let geometry_tests = include_str!("geometry/tests.rs");
    let execution = include_str!("../execution.rs");
    assert!(geometry.lines().count() < 100);
    assert!(geometry_tests.lines().count() < 100);
    for required in [
        "CUDA_MAX_GRID_DIM_X",
        "CUDA_MAX_GRID_DIM_Y_Z",
        "CUDA_MAX_BLOCK_DIM_X_Y",
        "CUDA_MAX_BLOCK_DIM_Z",
        "CUDA_MAX_THREADS_PER_BLOCK",
        "pub(crate) const fn is_valid",
        "pub(crate) const fn grid",
        "pub(crate) const fn block",
    ] {
        assert!(geometry.contains(required));
    }
    assert!(!geometry.contains("pub(crate) grid:"));
    assert!(!geometry.contains("pub(crate) block:"));
    let launch = execution
        .split("pub(crate) fn launch_kernel_async")
        .nth(1)
        .expect("launch_kernel_async source");
    let validation = launch
        .find("if !geometry.is_valid()")
        .expect("defensive geometry validation");
    let driver_scope = launch
        .find("with_current_resource_operation")
        .expect("CUDA driver operation scope");
    assert!(validation < driver_scope);
    for source in [
        include_str!("../jpeg/decode.rs"),
        include_str!("../jpeg/diagnostics.rs"),
        include_str!("../jpeg/encode_launch.rs"),
        include_str!("../transcode/launch.rs"),
    ] {
        assert!(!source.contains("CudaLaunchGeometry {"));
    }
}
