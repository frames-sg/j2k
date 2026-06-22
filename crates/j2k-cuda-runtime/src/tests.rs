use super::{
    checked_f32_words_byte_len, f32_slice_as_bytes_mut, format_idwt_batch_trace_row,
    idwt_batch_kernel_mode, idwt_batch_trace_row, idwt_batch_uses_cooperative_53,
    jpeg_entropy_overflow_count, pool_fit_buffer_index_by_len, validate_dct_block_grid,
    CudaContext, CudaError, CudaExecutionStats, CudaHtj2kCleanupMultiKernelJob,
    CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeTables,
    CudaHtj2kDequantizeTarget, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob,
    CudaHtj2kEncodeResidentTarget, CudaHtj2kEncodeTables, CudaJ2kIdwtBatchKernelMode,
    CudaJ2kIdwtJob, CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtTarget, CudaJ2kQuantizeJob,
    CudaJ2kQuantizeSubbandRegionJob, CudaJ2kRect, CudaJpegChunkedEntropyConfig,
    CudaJpegChunkedEntropyPlan, CudaJpegChunkedEntropyReport, CudaJpegEntropyOverflowState,
    CudaJpegEntropySyncState, CudaJpegHuffmanTable, CudaKernelName, CudaQueuedHtj2kCleanup,
};

fn cuda_runtime_required() -> bool {
    std::env::var_os("J2K_REQUIRE_CUDA_RUNTIME").is_some()
}

#[test]
fn jpeg_chunked_entropy_config_counts_bit_subsequences() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: 4,
        sequence_len: 8,
        max_overflow_subsequences: 2,
    };

    assert_eq!(config.subsequence_bits(), 128);
    assert_eq!(config.subsequence_count_for_entropy_bytes(0).unwrap(), 0);
    assert_eq!(config.subsequence_count_for_entropy_bytes(1).unwrap(), 1);
    assert_eq!(config.subsequence_count_for_entropy_bytes(16).unwrap(), 1);
    assert_eq!(config.subsequence_count_for_entropy_bytes(17).unwrap(), 2);
}

#[test]
fn checked_f32_words_byte_len_rejects_multiplication_overflow() {
    assert_eq!(checked_f32_words_byte_len(2).expect("byte len"), 8);
    assert!(matches!(
        checked_f32_words_byte_len(usize::MAX),
        Err(CudaError::LengthTooLarge { len }) if len == usize::MAX
    ));
}

#[test]
fn validate_dct_block_grid_checks_shape_and_coefficient_count() {
    let grid = validate_dct_block_grid(2, 1, 15, 8, 3, 384, "invalid").expect("valid grid");

    assert_eq!(grid.block_count, 2);
    assert_eq!(grid.expected_coeffs, 384);
    assert_eq!((grid.low_width, grid.high_width), (8, 7));
    assert_eq!((grid.low_height, grid.high_height), (4, 4));
    assert!(matches!(
        validate_dct_block_grid(2, 1, 15, 8, 3, 383, "invalid"),
        Err(CudaError::InvalidArgument { .. })
    ));
    assert!(matches!(
        validate_dct_block_grid(2, 1, 15, 8, 0, 0, "invalid"),
        Err(CudaError::InvalidArgument { .. })
    ));
    assert!(matches!(
        validate_dct_block_grid(usize::MAX, 2, 1, 1, 1, 64, "invalid"),
        Err(CudaError::LengthTooLarge { .. })
    ));
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
#[test]
fn cuda_oxide_reversible53_transcode_matches_scalar_fixture_when_required() {
    if !cuda_runtime_required()
        || std::env::var_os("J2K_CUDA_USE_OXIDE_TRANSCODE").is_none()
        || !super::transcode_kernels_built()
    {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let mut blocks = [0i16; 64];
    for (index, value) in [
        (0, 80),
        (1, -24),
        (2, 13),
        (3, 5),
        (5, -3),
        (8, 31),
        (9, -11),
        (10, 7),
        (16, -9),
        (17, 4),
        (18, 3),
        (27, -5),
        (36, 6),
        (45, -4),
        (54, 2),
        (63, -1),
    ] {
        blocks[index] = value;
    }

    let bands = context
        .j2k_transcode_reversible_dwt53(&blocks, 1, 1, 8, 8)
        .expect("cuda-oxide reversible 5/3 transcode");

    assert_eq!((bands.low_width, bands.low_height), (4, 4));
    assert_eq!((bands.high_width, bands.high_height), (4, 4));
    assert_eq!(
        bands.ll.as_slice(),
        &[14, 8, 12, 22, 13, 7, 14, 22, 8, 7, 12, 15, 6, 3, 5, 7]
    );
    assert_eq!(
        bands.hl.as_slice(),
        &[2, -1, -1, 5, 1, -4, 2, 0, -1, 1, 0, 3, 3, -3, 2, 0]
    );
    assert_eq!(
        bands.lh.as_slice(),
        &[2, 1, -1, 2, 2, -1, 3, 0, -1, 3, -1, 1, 1, -4, -1, -2]
    );
    assert_eq!(
        bands.hh.as_slice(),
        &[1, 2, -1, -4, 1, -1, 0, 1, -1, -1, 1, -2, -5, 2, -1, -1]
    );
}

#[test]
fn jpeg_chunked_entropy_report_has_one_less_overflow_than_subsequence_count() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: 1,
        sequence_len: 8,
        max_overflow_subsequences: 2,
    };
    let subsequences = config.subsequence_count_for_entropy_bytes(16).unwrap();

    assert_eq!(subsequences, 4);
    assert_eq!(jpeg_entropy_overflow_count(subsequences), 3);
    assert_eq!(jpeg_entropy_overflow_count(0), 0);
}

#[test]
fn jpeg_chunked_entropy_config_rejects_zero_subsequence_or_sequence() {
    let zero_words = CudaJpegChunkedEntropyConfig {
        subsequence_words: 0,
        ..CudaJpegChunkedEntropyConfig::default()
    };
    let zero_sequence = CudaJpegChunkedEntropyConfig {
        sequence_len: 0,
        ..CudaJpegChunkedEntropyConfig::default()
    };

    assert!(zero_words.validate().is_err());
    assert!(zero_sequence.validate().is_err());
}

#[test]
fn jpeg_chunked_entropy_config_rejects_subsequence_bit_overflow() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: (u32::MAX / 32) + 1,
        ..CudaJpegChunkedEntropyConfig::default()
    };

    assert!(config.validate().is_err());
    assert!(config.subsequence_count_for_entropy_bytes(1).is_err());
}

#[test]
fn jpeg_chunked_entropy_report_summarizes_sync_quality() {
    let report = CudaJpegChunkedEntropyReport {
        config: CudaJpegChunkedEntropyConfig {
            subsequence_words: 4,
            sequence_len: 8,
            max_overflow_subsequences: 2,
        },
        entropy_bytes: 4096,
        states: vec![
            CudaJpegEntropySyncState {
                code: 0,
                start_bit: 0,
                end_bit: 128,
                bit_pos: 128,
                symbol_count: 10,
                block_phase: 0,
                zigzag_index: 0,
                reserved: 0,
            },
            CudaJpegEntropySyncState {
                code: 0,
                start_bit: 128,
                end_bit: 256,
                bit_pos: 256,
                symbol_count: 9,
                block_phase: 3,
                zigzag_index: 12,
                reserved: 0,
            },
        ],
        overflows: vec![CudaJpegEntropyOverflowState {
            code: 0,
            from_subsequence: 0,
            to_subsequence: 1,
            overflow_bits: 96,
            synchronized: 1,
            reserved: [0; 3],
        }],
        execution: CudaExecutionStats {
            kernel_dispatches: 2,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        },
    };

    assert_eq!(report.subsequence_count(), 2);
    assert_eq!(report.synchronized_overflow_count(), 1);
    assert_eq!(report.max_overflow_bits(), Some(96));
    assert_eq!(report.failed_state_count(), 0);
}

#[test]
fn jpeg_entropy_self_sync_returns_empty_report_for_empty_entropy_when_runtime_required() {
    if std::env::var_os("J2K_REQUIRE_CUDA_RUNTIME").is_none() {
        return;
    }

    let context = CudaContext::system_default().expect("cuda context");
    let plan = CudaJpegChunkedEntropyPlan {
        config: CudaJpegChunkedEntropyConfig::default(),
        entropy_bytes: &[],
        y_dc_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        y_ac_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cb_dc_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cb_ac_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cr_dc_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cr_ac_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
    };

    let report = context
        .diagnose_jpeg_420_entropy_self_sync(&plan)
        .expect("empty diagnostic report");
    assert_eq!(report.subsequence_count(), 0);
    assert_eq!(report.overflows.len(), 0);
}

#[test]
#[allow(clippy::too_many_lines)]
fn runtime_raii_primitives_smoke_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let mut pinned = context.pinned_host_buffer(16).expect("pinned host buffer");
    pinned.as_mut_slice().copy_from_slice(&[7u8; 16]);
    assert_eq!(pinned.as_slice(), &[7u8; 16]);
    let pinned_upload = context
        .upload_pinned(&[1u8, 2, 3, 4])
        .expect("pinned upload");
    let mut uploaded = [0u8; 4];
    pinned_upload
        .copy_to_host(&mut uploaded)
        .expect("download pinned upload");
    assert_eq!(uploaded, [1, 2, 3, 4]);
    let pinned_float_upload = context
        .upload_f32_pinned(&[1.25, -2.5])
        .expect("pinned f32 upload");
    let mut downloaded_float_values = [0.0f32; 2];
    pinned_float_upload
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut downloaded_float_values))
        .expect("download pinned f32 upload");
    assert!((downloaded_float_values[0] - 1.25).abs() < f32::EPSILON);
    assert!((downloaded_float_values[1] + 2.5).abs() < f32::EPSILON);
    let pinned_integer_upload = context
        .upload_i32_pinned(&[7, -11])
        .expect("pinned i32 upload");
    let mut downloaded_integer_values = [0i32; 2];
    pinned_integer_upload
        .copy_to_host(super::i32_slice_as_bytes_mut(
            &mut downloaded_integer_values,
        ))
        .expect("download pinned i32 upload");
    assert_eq!(downloaded_integer_values, [7, -11]);
    let ranged_upload = context
        .upload(&[9u8, 8, 7, 6, 5, 4])
        .expect("range-copy upload");
    let mut range = [0u8; 3];
    ranged_upload
        .copy_range_to_host(2, &mut range)
        .expect("copy device range");
    assert_eq!(range, [7, 6, 5]);
    let mut uninit_range = Vec::with_capacity(3);
    ranged_upload
        .copy_range_to_host_uninit(1, uninit_range.spare_capacity_mut())
        .expect("copy device range into spare capacity");
    // SAFETY: copy_range_to_host_uninit returned success after writing
    // exactly three bytes into the Vec spare capacity.
    unsafe {
        uninit_range.set_len(3);
    }
    assert_eq!(uninit_range, [8, 7, 6]);
    let pool = context.buffer_pool();
    let pooled_upload = pool.upload(&[3u8, 1, 4, 1]).expect("pooled upload");
    let pooled_output = super::copy_pooled_bytes_to_vec_uninit(&pooled_upload, 4)
        .expect("copy pooled bytes into spare capacity");
    assert_eq!(pooled_output, [3, 1, 4, 1]);

    let module = context
        .preload_kernel_module(CudaKernelName::CopyU8)
        .expect("preload copy kernel");
    assert_eq!(module.entrypoint(), "j2k_copy_u8");

    let stream = context.create_stream().expect("CUDA stream");
    let start = context.create_event().expect("start event");
    let end = context.create_event().expect("end event");
    start.record(&stream).expect("record start");
    end.record(&stream).expect("record end");
    end.synchronize().expect("synchronize event");
    let elapsed = super::CudaEvent::elapsed_time_us(&start, &end).expect("elapsed time");
    assert!(elapsed >= 0.0);

    let pool = context.buffer_pool();
    {
        let buffer = pool.take(32).expect("pooled buffer");
        assert!(buffer.device_ptr() != 0);
        assert_eq!(buffer.byte_len(), 32);
        assert!(buffer.allocation_byte_len() >= 32);
    }
    let cached_count = pool.cached_count().expect("cached count");
    assert_eq!(cached_count, 1);
    {
        let buffer = pool.take(16).expect("reused pooled buffer");
        assert_eq!(buffer.byte_len(), 16);
        assert!(buffer.allocation_byte_len() >= 32);
    }

    let samples = [1.25f32, -2.5, 3.75, 4.5];
    {
        let buffer = pool.upload_f32(&samples).expect("pooled f32 upload");
        assert_eq!(
            buffer.byte_len(),
            samples.len() * std::mem::size_of::<f32>()
        );
        let mut downloaded = vec![0.0f32; samples.len()];
        buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut downloaded))
            .expect("download pooled f32 upload");
        assert_eq!(downloaded, samples);
    }
    let i16_samples = [-12i16, 7, 19, -4];
    {
        let buffer = pool
            .upload_i16_pinned(&i16_samples)
            .expect("pooled pinned i16 upload");
        assert_eq!(
            buffer.byte_len(),
            i16_samples.len() * std::mem::size_of::<i16>()
        );
        let mut downloaded_bytes = vec![0u8; std::mem::size_of_val(&i16_samples)];
        buffer
            .copy_to_host(&mut downloaded_bytes)
            .expect("download pooled pinned i16 upload");
        let downloaded = downloaded_bytes
            .chunks_exact(std::mem::size_of::<i16>())
            .map(|chunk| i16::from_ne_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        assert_eq!(downloaded, i16_samples);
    }
    let cached_after_upload = pool.cached_count().expect("cached after upload");
    assert!(cached_after_upload >= cached_count);
}

#[test]
fn pooled_i16_pinned_upload_is_size_gated() {
    assert!(super::should_use_pinned_pooled_i16_upload(4 * 1024 * 1024));
    assert!(!super::should_use_pinned_pooled_i16_upload(
        4 * 1024 * 1024 + 1
    ));
}

#[test]
fn pooled_buffer_selection_uses_smallest_sufficient_fit() {
    let buffers = [(1usize, 32usize), (0, 64)];

    assert_eq!(
        pool_fit_buffer_index_by_len(buffers.iter().copied(), 16),
        Some(1)
    );
    let mut large_pool = (0..1024).map(|index| (index, 8usize)).collect::<Vec<_>>();
    large_pool[1022] = (1022, 32);
    large_pool[1023] = (1023, 64);

    assert_eq!(
        pool_fit_buffer_index_by_len(large_pool.iter().copied(), 16),
        Some(1022)
    );
    let mut recent_fit_pool = (0..4096).map(|index| (index, 8usize)).collect::<Vec<_>>();
    recent_fit_pool[4094] = (4094, 32);
    recent_fit_pool[4095] = (4095, 64);

    assert_eq!(
        pool_fit_buffer_index_by_len(recent_fit_pool.iter().copied(), 16),
        Some(4094)
    );
    let fallback_pool = (0..4096)
        .map(|index| match index.cmp(&3000) {
            std::cmp::Ordering::Less => (index, 8usize),
            std::cmp::Ordering::Equal => (index, 32),
            std::cmp::Ordering::Greater => (index, 64),
        })
        .collect::<Vec<_>>();

    assert_eq!(
        pool_fit_buffer_index_by_len(fallback_pool.iter().copied(), 16),
        Some(3000)
    );
}

#[test]
fn pooled_take_with_trace_reports_allocation_and_reuse_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let (fresh, fresh_trace) = pool.take_with_trace(32).expect("fresh traced take");

    assert_eq!(fresh.byte_len(), 32);
    assert_eq!(fresh_trace.requested_len, 32);
    assert_eq!(fresh_trace.free_count_before, 0);
    assert_eq!(fresh_trace.scanned_count, 0);
    assert!(!fresh_trace.reused);
    assert!(fresh_trace.allocation_byte_len >= 32);
    drop(fresh);

    let (reused, reuse_trace) = pool.take_with_trace(16).expect("reused traced take");

    assert_eq!(reused.byte_len(), 16);
    assert_eq!(reuse_trace.requested_len, 16);
    assert_eq!(reuse_trace.free_count_before, 1);
    assert_eq!(reuse_trace.scanned_count, 1);
    assert!(reuse_trace.reused);
    assert!(reuse_trace.allocation_byte_len >= 32);
}

#[test]
fn pooled_buffer_can_detach_and_recycle_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let raw = pool
        .take(32)
        .expect("pooled buffer")
        .into_device_buffer()
        .expect("detach pooled buffer");
    assert_eq!(pool.cached_count().expect("cached after detach"), 0);

    pool.recycle(raw).expect("explicit recycle");
    assert_eq!(pool.cached_count().expect("cached after recycle"), 1);

    let (_reused, trace) = pool.take_with_trace(16).expect("reused traced take");
    assert!(trace.reused);
    assert!(trace.allocation_byte_len >= 32);
}

#[test]
fn htj2k_encoded_codeblock_reports_segment_lengths_from_status() {
    let encoded = super::CudaHtj2kEncodedCodeBlock {
        data: vec![0u8; 10],
        status: super::CudaHtj2kEncodeStatus {
            code: super::HTJ2K_STATUS_OK,
            detail: 0,
            data_len: 10,
            number_of_coding_passes: 3,
            missing_bit_planes: 4,
            reserved0: 7,
            reserved1: 3,
            reserved2: 0,
        },
        execution: super::CudaExecutionStats::default(),
        stage_timings: super::CudaHtj2kEncodeStageTimings::default(),
    };

    assert_eq!(encoded.cleanup_length(), 7);
    assert_eq!(encoded.refinement_length(), 3);
}

#[test]
fn htj2k_encode_compact_jobs_pack_actual_payloads() {
    let capacity = u32::try_from(super::HTJ2K_ENCODE_OUTPUT_CAPACITY)
        .expect("HTJ2K encode output capacity fits u32");
    let double_capacity = capacity
        .checked_mul(2)
        .expect("test output capacity fits u32");
    let kernel_jobs = [
        super::CudaHtj2kEncodeKernelJob {
            coefficient_offset: 0,
            coefficient_stride: 64,
            width: 64,
            height: 64,
            total_bitplanes: 8,
            output_offset: 0,
            output_capacity: capacity,
            target_coding_passes: 1,
        },
        super::CudaHtj2kEncodeKernelJob {
            coefficient_offset: 4096,
            coefficient_stride: 64,
            width: 64,
            height: 64,
            total_bitplanes: 8,
            output_offset: capacity,
            output_capacity: capacity,
            target_coding_passes: 1,
        },
        super::CudaHtj2kEncodeKernelJob {
            coefficient_offset: 8192,
            coefficient_stride: 64,
            width: 64,
            height: 64,
            total_bitplanes: 8,
            output_offset: double_capacity,
            output_capacity: capacity,
            target_coding_passes: 1,
        },
    ];
    let statuses = [
        super::CudaHtj2kEncodeStatus {
            code: super::HTJ2K_STATUS_OK,
            data_len: 12,
            reserved2: 0x8001_8002,
            ..super::CudaHtj2kEncodeStatus::default()
        },
        super::CudaHtj2kEncodeStatus {
            code: super::HTJ2K_STATUS_OK,
            data_len: 0,
            ..super::CudaHtj2kEncodeStatus::default()
        },
        super::CudaHtj2kEncodeStatus {
            code: super::HTJ2K_STATUS_OK,
            data_len: 7,
            ..super::CudaHtj2kEncodeStatus::default()
        },
    ];

    let (compact_jobs, compact_len) =
        super::htj2k_encode_compact_jobs(&statuses, &kernel_jobs).expect("valid compact jobs");

    assert_eq!(compact_len, 19);
    assert_eq!(
        compact_jobs,
        vec![
            super::CudaHtj2kEncodeCompactJob {
                source_offset: 0,
                compact_offset: 0,
                data_len: 12,
                reserved: 0x8001_8002,
            },
            super::CudaHtj2kEncodeCompactJob {
                source_offset: capacity,
                compact_offset: 12,
                data_len: 0,
                reserved: 0,
            },
            super::CudaHtj2kEncodeCompactJob {
                source_offset: double_capacity,
                compact_offset: 12,
                data_len: 7,
                reserved: 0,
            },
        ]
    );
}

#[test]
fn htj2k_encode_resources_feed_resident_region_encode_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = context
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let coefficients = context
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("resident coefficients");
    let jobs = [CudaHtj2kEncodeCodeBlockRegionJob {
        coefficient_offset: 0,
        coefficient_stride: 2,
        width: 2,
        height: 2,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];

    let encoded = context
        .encode_htj2k_codeblock_regions_resident_with_resources(&coefficients, 4, &jobs, &resources)
        .expect("resource-backed resident HTJ2K encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert_eq!(encoded.code_blocks().len(), 1);
}

#[test]
fn htj2k_encode_resident_region_reuses_pool_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = context
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let coefficients = context
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("resident coefficients");
    let jobs = [CudaHtj2kEncodeCodeBlockRegionJob {
        coefficient_offset: 0,
        coefficient_stride: 2,
        width: 2,
        height: 2,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];

    let encoded = context
        .encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
            &coefficients,
            4,
            &jobs,
            &resources,
            &pool,
        )
        .expect("pooled resource-backed resident HTJ2K encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert_eq!(encoded.code_blocks().len(), 1);
    assert!(pool.cached_count().expect("cached pooled encode buffers") >= 3);
}

#[test]
fn htj2k_encode_codeblocks_resident_reuses_pool_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = context
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let coefficients = context
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("resident coefficients");
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];

    let encoded = context
        .encode_htj2k_codeblocks_resident_with_resources_and_pool(
            &coefficients,
            4,
            &jobs,
            &resources,
            &pool,
        )
        .expect("pooled resource-backed resident HTJ2K codeblock encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert_eq!(encoded.code_blocks().len(), 1);
    assert!(pool.cached_count().expect("cached pooled encode buffers") >= 3);
}

#[test]
fn htj2k_encode_multi_resident_inputs_match_separate_batches_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = context
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let first = context
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("first resident coefficients");
    let second = context
        .upload_i32_pinned(&[0, 0])
        .expect("second resident coefficients");
    let first_jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];
    let second_jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 1,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];

    let first_separate = context
        .encode_htj2k_codeblocks_resident_with_resources_and_pool(
            &first,
            4,
            &first_jobs,
            &resources,
            &pool,
        )
        .expect("first separate resident encode");
    let second_separate = context
        .encode_htj2k_codeblocks_resident_with_resources_and_pool(
            &second,
            2,
            &second_jobs,
            &resources,
            &pool,
        )
        .expect("second separate resident encode");

    let combined = context
        .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(
            &[
                CudaHtj2kEncodeResidentTarget {
                    coefficients: &first,
                    coefficient_count: 4,
                    jobs: &first_jobs,
                },
                CudaHtj2kEncodeResidentTarget {
                    coefficients: &second,
                    coefficient_count: 2,
                    jobs: &second_jobs,
                },
            ],
            &resources,
            &pool,
        )
        .expect("combined resident encode");

    assert_eq!(combined.execution().kernel_dispatches(), 1);
    assert_eq!(combined.code_blocks().len(), 2);
    assert_eq!(
        combined.code_blocks()[0].data(),
        first_separate.code_blocks()[0].data()
    );
    assert_eq!(
        combined.code_blocks()[1].data(),
        second_separate.code_blocks()[0].data()
    );
    let timings = combined.stage_timings();
    assert_eq!(
        timings.ht_encode_us,
        timings
            .ht_kernel_us
            .saturating_add(timings.ht_status_readback_us)
            .saturating_add(timings.ht_compact_us)
            .saturating_add(timings.ht_output_readback_us)
    );
    assert!(timings.ht_kernel_us > 0);
    assert!(timings.ht_status_readback_us > 0);
}

#[test]
fn htj2k97_resident_batch_returns_pooled_quantized_bands_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let blocks = vec![0.0f32; 64];
    let params = super::CudaHtj2k97QuantizeParams {
        inv_delta_ll: 1.0,
        inv_delta_hl: 1.0,
        inv_delta_lh: 1.0,
        inv_delta_hh: 1.0,
        cb_width: 64,
        cb_height: 64,
    };

    let (bands, _) = context
        .j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(
            &blocks, 1, 1, 1, 8, 8, params, &pool,
        )
        .expect("resident HTJ2K 9/7 codeblock batch");

    assert!(bands.ll.as_device_buffer().is_some());
    assert!(bands.hl.as_device_buffer().is_some());
    assert!(bands.lh.as_device_buffer().is_some());
    assert!(bands.hh.as_device_buffer().is_some());
    let cached_while_bands_live = pool.cached_count().expect("cached buffers while live");

    drop(bands);

    assert!(pool.cached_count().expect("cached buffers after drop") >= cached_while_bands_live + 4);
}

#[test]
fn htj2k_encode_rejects_unsupported_refinement_pass_count_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let coefficients = [0, 2, -3, 1];
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 3,
        target_coding_passes: 4,
    }];

    let error = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &[0u16; 2048],
                vlc_table1: &[0u16; 2048],
                uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
            },
        )
        .expect_err("unsupported HTJ2K encode pass count is explicit");

    match error {
        CudaError::KernelStatus {
            kernel,
            code,
            detail,
        } => {
            assert_eq!(kernel, "j2k_htj2k_encode_codeblocks");
            assert_eq!(code, super::HTJ2K_STATUS_UNSUPPORTED);
            assert_eq!(detail, 5);
        }
        other => panic!("unexpected CUDA encode error: {other:?}"),
    }
}

#[test]
fn htj2k_encode_rejects_lossy_zero_sigprop_request_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let coefficients = [0, 2, -3, 4];
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 3,
        target_coding_passes: 2,
    }];

    let error = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &[0u16; 2048],
                vlc_table1: &[0u16; 2048],
                uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
            },
        )
        .expect_err("target-2 zero SigProp cannot silently drop low coefficient bits");

    match error {
        CudaError::KernelStatus {
            kernel,
            code,
            detail,
        } => {
            assert_eq!(kernel, "j2k_htj2k_encode_codeblocks");
            assert_eq!(code, super::HTJ2K_STATUS_UNSUPPORTED);
            assert_eq!(detail, 6);
        }
        other => panic!("unexpected CUDA encode error: {other:?}"),
    }
}

#[test]
fn htj2k_encode_rejects_unreachable_target_three_sigprop_coefficients_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let coefficients = [3, 0, 0, 0];
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 4,
        target_coding_passes: 3,
    }];

    let error = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &[0u16; 2048],
                vlc_table1: &[0u16; 2048],
                uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
            },
        )
        .expect_err("isolated target-3 SigProp coefficient is explicitly unsupported");

    match error {
        CudaError::KernelStatus {
            kernel,
            code,
            detail,
        } => {
            assert_eq!(kernel, "j2k_htj2k_encode_codeblocks");
            assert_eq!(code, super::HTJ2K_STATUS_UNSUPPORTED);
            assert_eq!(detail, 6);
        }
        other => panic!("unexpected CUDA encode error: {other:?}"),
    }
}

#[test]
fn htj2k_encode_resources_feed_single_codeblock_encode_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = context
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");

    let encoded = context
        .encode_htj2k_codeblock_with_resources(&[0, 0, 0, 0], 2, 2, 1, &resources)
        .expect("resource-backed single HTJ2K encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    // An all-zero codeblock has no significant bitplanes, so the encoder emits zero
    // coding passes (matching native ht_block_encode::encode_code_block).
    assert_eq!(encoded.num_coding_passes(), 0);
    assert_eq!(encoded.cleanup_length(), 0);
    assert_eq!(encoded.data().len(), 0);
    assert_eq!(encoded.refinement_length(), 0);
}

#[test]
fn default_stream_timer_reports_elapsed_time_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = vec![17u8; 4096];
    let (output, elapsed_us) = context
        .time_default_stream_us(|| context.copy_with_kernel(&input))
        .expect("timed CUDA copy kernel");

    assert_eq!(output.execution().kernel_dispatches(), 1);
    assert!(elapsed_us > 0);
}

#[cfg(all(feature = "cuda-oxide-copy-u8", j2k_cuda_oxide_copy_u8_built))]
#[test]
fn cuda_oxide_copy_u8_matches_builtin_copy_and_cpu_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = (0..4099)
        .map(|index| ((index * 31 + 17) % 251) as u8)
        .collect::<Vec<_>>();

    let builtin = context
        .copy_with_kernel(&input)
        .expect("builtin CUDA copy kernel");
    let cuda_oxide = context
        .copy_with_cuda_oxide_kernel(&input)
        .expect("cuda-oxide CUDA copy kernel");

    let mut builtin_bytes = vec![0u8; input.len()];
    builtin
        .buffer()
        .copy_to_host(&mut builtin_bytes)
        .expect("download builtin CUDA copy");
    let mut cuda_oxide_bytes = vec![0u8; input.len()];
    cuda_oxide
        .buffer()
        .copy_to_host(&mut cuda_oxide_bytes)
        .expect("download cuda-oxide CUDA copy");

    assert_eq!(builtin.execution().kernel_dispatches(), 1);
    assert_eq!(cuda_oxide.execution().kernel_dispatches(), 1);
    assert_eq!(builtin_bytes, input);
    assert_eq!(cuda_oxide_bytes, input);
    assert_eq!(cuda_oxide_bytes, builtin_bytes);
}

#[test]
fn named_default_stream_timer_is_available_for_profiling_ranges_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = vec![23u8; 4096];
    let (output, elapsed_us) = context
        .time_default_stream_named_us("j2k.test.copy", || context.copy_with_kernel(&input))
        .expect("named timed CUDA copy kernel");

    assert_eq!(output.execution().kernel_dispatches(), 1);
    assert!(elapsed_us > 0);
}

#[test]
fn typed_device_view_reports_element_count_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let mut aligned = context.allocate(16).expect("aligned buffer");
    let view = aligned.typed_view::<u32>().expect("typed immutable view");
    assert_eq!(view.len(), 4);
    let mut_view = aligned.typed_view_mut::<u64>().expect("typed mutable view");
    assert_eq!(mut_view.len(), 2);

    let unaligned = context.allocate(3).expect("unaligned buffer");
    let error = unaligned
        .typed_view::<u16>()
        .expect_err("unaligned typed view");
    assert!(matches!(
        error,
        CudaError::LengthNotElementAligned {
            bytes: 3,
            element_size: 2
        }
    ));
}

#[test]
#[allow(clippy::too_many_lines)]
fn kernel_module_names_cover_htj2k_decode_and_encode_stages() {
    let cases = [
        (
            CudaKernelName::Htj2kDecodeCodeblocks,
            "j2k_htj2k_decode_codeblocks",
        ),
        (
            CudaKernelName::Htj2kDecodeCodeblocksMultiCleanupDequantize,
            "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize",
        ),
        (
            CudaKernelName::J2kDequantizeHtj2kCodeblocks,
            "j2k_dequantize_htj2k_codeblocks",
        ),
        (
            CudaKernelName::J2kDequantizeHtj2kCodeblocksMulti,
            "j2k_dequantize_htj2k_codeblocks_multi",
        ),
        (
            CudaKernelName::J2kDequantizeHtj2kCleanupJobsMulti,
            "j2k_dequantize_htj2k_cleanup_jobs_multi",
        ),
        (CudaKernelName::J2kIdwtInterleave, "j2k_idwt_interleave"),
        (
            CudaKernelName::J2kIdwtInterleaveHorizontal53Multi,
            "j2k_idwt_interleave_horizontal_53_multi",
        ),
        (
            CudaKernelName::J2kIdwtInterleaveHorizontal97Multi,
            "j2k_idwt_interleave_horizontal_97_multi",
        ),
        (CudaKernelName::J2kIdwtHorizontal, "j2k_idwt_horizontal"),
        (
            CudaKernelName::J2kIdwtHorizontal53,
            "j2k_idwt_horizontal_53",
        ),
        (
            CudaKernelName::J2kIdwtHorizontal97,
            "j2k_idwt_horizontal_97",
        ),
        (CudaKernelName::J2kIdwtVertical, "j2k_idwt_vertical"),
        (
            CudaKernelName::J2kIdwtVertical53Multi,
            "j2k_idwt_vertical_53_multi",
        ),
        (
            CudaKernelName::J2kIdwtVertical97Multi,
            "j2k_idwt_vertical_97_multi",
        ),
        (
            CudaKernelName::J2kIdwtVertical97MultiCols4,
            "j2k_idwt_vertical_97_multi_cols4",
        ),
        (CudaKernelName::J2kIdwtVertical53, "j2k_idwt_vertical_53"),
        (CudaKernelName::J2kIdwtVertical97, "j2k_idwt_vertical_97"),
        (
            CudaKernelName::J2kInverseDwtSingle,
            "j2k_inverse_dwt_single",
        ),
        (CudaKernelName::J2kInverseMct, "j2k_inverse_mct"),
        (CudaKernelName::J2kStoreGray8, "j2k_store_gray8"),
        (CudaKernelName::J2kStoreGray16, "j2k_store_gray16"),
        (CudaKernelName::J2kStoreRgb8, "j2k_store_rgb8"),
        (CudaKernelName::J2kStoreRgb8Mct, "j2k_store_rgb8_mct"),
        (
            CudaKernelName::J2kStoreRgb8MctBatch,
            "j2k_store_rgb8_mct_batch",
        ),
        (CudaKernelName::J2kStoreRgb16, "j2k_store_rgb16"),
        (CudaKernelName::J2kStoreRgb16Mct, "j2k_store_rgb16_mct"),
        (
            CudaKernelName::Htj2kEncodeCodeblock,
            "j2k_htj2k_encode_codeblock",
        ),
        (
            CudaKernelName::Htj2kEncodeCodeblocks,
            "j2k_htj2k_encode_codeblocks",
        ),
        (
            CudaKernelName::Htj2kEncodeCodeblocksMultiInput,
            "j2k_htj2k_encode_codeblocks_multi_input",
        ),
        (
            CudaKernelName::Htj2kEncodeCodeblocksMultiInputCleanup,
            "j2k_htj2k_encode_codeblocks_multi_input_cleanup",
        ),
        (
            CudaKernelName::Htj2kEncodeCodeblocksMultiInputCleanup64,
            "j2k_htj2k_encode_codeblocks_multi_input_cleanup_64",
        ),
        (
            CudaKernelName::Htj2kCompactCodeblocks,
            "j2k_htj2k_compact_codeblocks",
        ),
        (
            CudaKernelName::Htj2kPacketizeCleanup,
            "j2k_htj2k_packetize_cleanup",
        ),
    ];

    for (kernel, entrypoint) in cases {
        assert_eq!(kernel.entrypoint(), entrypoint);
        let raw_entrypoint = kernel.kernel().entrypoint();
        assert_eq!(
            &raw_entrypoint[..raw_entrypoint.len() - 1],
            entrypoint.as_bytes()
        );
        assert_eq!(raw_entrypoint.last(), Some(&0));
    }
}

#[test]
#[allow(clippy::similar_names)]
fn htj2k_empty_codeblock_decode_zero_fills_coefficients_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let output = context
        .decode_htj2k_codeblocks(
            &[],
            &[],
            CudaHtj2kDecodeTables {
                vlc_table0: &first_vlc,
                vlc_table1: &later_vlc,
                uvlc_table0: &first_uvlc,
                uvlc_table1: &later_uvlc,
            },
            8,
        )
        .expect("empty HTJ2K decode");
    let mut actual = vec![f32::NAN; 8];
    output
        .coefficients()
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
        .expect("download coefficients");

    assert_eq!(actual, vec![0.0; 8]);
    assert_eq!(output.execution().kernel_dispatches(), 0);
}

#[test]
#[allow(clippy::similar_names)]
fn htj2k_empty_codeblock_decode_reuses_pool_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let tables = context
        .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
            vlc_table0: &first_vlc,
            vlc_table1: &later_vlc,
            uvlc_table0: &first_uvlc,
            uvlc_table1: &later_uvlc,
        })
        .expect("decode tables");
    let resources = context
        .upload_htj2k_decode_resources_with_tables(&[], &tables)
        .expect("decode resources");

    let output = context
        .decode_htj2k_codeblocks_with_resources_and_pool(&resources, &[], 8, &pool)
        .expect("pooled empty HTJ2K decode");
    let mut actual = vec![f32::NAN; 8];
    output
        .coefficients()
        .expect("pooled coefficients")
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
        .expect("download coefficients");

    assert_eq!(actual, vec![0.0; 8]);
    assert_eq!(output.execution().kernel_dispatches(), 0);
    let cached_while_live = pool.cached_count().expect("cached while live");

    drop(output);

    assert!(pool.cached_count().expect("cached after drop") > cached_while_live);
}

#[test]
#[allow(clippy::similar_names)]
fn htj2k_decode_table_resources_feed_multiple_payload_uploads_when_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let tables = context
        .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
            vlc_table0: &first_vlc,
            vlc_table1: &later_vlc,
            uvlc_table0: &first_uvlc,
            uvlc_table1: &later_uvlc,
        })
        .expect("decode table resources");

    let first_resources = context
        .upload_htj2k_decode_resources_with_tables(&[0xAA, 0x55], &tables)
        .expect("first payload resources");
    let second_resources = context
        .upload_htj2k_decode_resources_with_tables(&[0x11, 0x22, 0x33], &tables)
        .expect("second payload resources");

    assert!(std::sync::Arc::ptr_eq(
        &first_resources.tables.inner,
        &second_resources.tables.inner
    ));
    assert_eq!(first_resources.payload_len, 2);
    assert_eq!(second_resources.payload_len, 3);
}

#[test]
fn j2k_inverse_dwt_single_dispatches_parallel_stages_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let ll = context
        .upload(super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");

    let output = context
        .j2k_inverse_dwt_single_device(
            &ll,
            &hl,
            &lh,
            &hh,
            CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 2,
                    y1: 2,
                },
                ll_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                hl_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                lh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                hh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                irreversible97: 0,
            },
        )
        .expect("CUDA inverse DWT");

    assert_eq!(output.execution().kernel_dispatches(), 3);
    let mut actual = vec![0.0f32; 4];
    output
        .buffer()
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
        .expect("download inverse DWT");
    assert_eq!(actual, vec![7.0, 9.0, 10.0, 13.0]);
}

#[test]
fn j2k_inverse_dwt_single_reuses_pool_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");

    let output = context
        .j2k_inverse_dwt_single_device_with_pool(
            &ll,
            &hl,
            &lh,
            &hh,
            CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 2,
                    y1: 2,
                },
                ll_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                hl_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                lh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                hh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                irreversible97: 0,
            },
            &pool,
        )
        .expect("pooled CUDA inverse DWT");

    assert_eq!(output.execution().kernel_dispatches(), 3);
    let cached_while_live = pool.cached_count().expect("cached while live");

    drop(output);

    assert!(pool.cached_count().expect("cached after drop") > cached_while_live);
}

#[test]
fn idwt_cooperative_53_selection_requires_large_reversible_batches() {
    let mut kernel_job = CudaJ2kIdwtMultiKernelJob {
        ll_ptr: 0,
        hl_ptr: 0,
        lh_ptr: 0,
        hh_ptr: 0,
        output_ptr: 0,
        job: CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            irreversible97: 0,
        },
    };

    assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 127, 128));
    assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 128, 127));
    assert!(idwt_batch_uses_cooperative_53(&[kernel_job], 128, 128));
    assert!(idwt_batch_uses_cooperative_53(&[kernel_job], 512, 512));
    assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 513, 128));
    kernel_job.job.irreversible97 = 1;
    assert!(!idwt_batch_uses_cooperative_53(&[kernel_job], 128, 128));
}

#[test]
fn idwt_cooperative_97_selection_requires_large_irreversible_batches() {
    let mut kernel_job = CudaJ2kIdwtMultiKernelJob {
        ll_ptr: 0,
        hl_ptr: 0,
        lh_ptr: 0,
        hh_ptr: 0,
        output_ptr: 0,
        job: CudaJ2kIdwtJob {
            rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            ll_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            hl_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            lh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            hh_rect: CudaJ2kRect {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            },
            irreversible97: 1,
        },
    };

    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 128, 128),
        CudaJ2kIdwtBatchKernelMode::Cooperative97
    );
    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 64, 64),
        CudaJ2kIdwtBatchKernelMode::Cooperative97
    );
    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 512, 512),
        CudaJ2kIdwtBatchKernelMode::Cooperative97
    );
    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 63, 64),
        CudaJ2kIdwtBatchKernelMode::Generic
    );
    assert_eq!(
        idwt_batch_kernel_mode(&[kernel_job], 513, 128),
        CudaJ2kIdwtBatchKernelMode::Generic
    );
    kernel_job.job.irreversible97 = 0;
    assert_ne!(
        idwt_batch_kernel_mode(&[kernel_job], 128, 128),
        CudaJ2kIdwtBatchKernelMode::Cooperative97
    );
}

#[test]
fn idwt_batch_trace_row_reports_stage_shape_and_mode() {
    let kernel_jobs = [
        CudaJ2kIdwtMultiKernelJob {
            ll_ptr: 0,
            hl_ptr: 0,
            lh_ptr: 0,
            hh_ptr: 0,
            output_ptr: 0,
            job: CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 128,
                    y1: 96,
                },
                ll_rect: CudaJ2kRect::default(),
                hl_rect: CudaJ2kRect::default(),
                lh_rect: CudaJ2kRect::default(),
                hh_rect: CudaJ2kRect::default(),
                irreversible97: 1,
            },
        },
        CudaJ2kIdwtMultiKernelJob {
            ll_ptr: 0,
            hl_ptr: 0,
            lh_ptr: 0,
            hh_ptr: 0,
            output_ptr: 0,
            job: CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 64,
                    y1: 48,
                },
                ll_rect: CudaJ2kRect::default(),
                hl_rect: CudaJ2kRect::default(),
                lh_rect: CudaJ2kRect::default(),
                hh_rect: CudaJ2kRect::default(),
                irreversible97: 1,
            },
        },
    ];

    let row = idwt_batch_trace_row(
        3,
        &kernel_jobs,
        128,
        96,
        CudaJ2kIdwtBatchKernelMode::Cooperative97,
        42,
    );

    assert_eq!(
            format_idwt_batch_trace_row(row),
            "j2k_profile codec=j2k op=cuda_idwt_batch path=decode stage_index=3 mode=Cooperative97 job_count=2 max_width=128 max_height=96 min_width=64 min_height=48 total_pixels=15360 irreversible_jobs=2 elapsed_us=42"
        );
}

#[test]
fn j2k_inverse_dwt_batch_empty_uses_no_dispatch_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(&[] as &[CudaJ2kIdwtTarget<'_>], &pool)
        .expect("empty batched CUDA inverse DWT");

    assert_eq!(execution.kernel_dispatches(), 0);
    assert_eq!(execution.decode_kernel_dispatches(), 0);
}

#[test]
fn j2k_inverse_dwt_batch_matches_expected_outputs_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");
    let first_output = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("first batched IDWT output");
    let second_output = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("second batched IDWT output");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        irreversible97: 0,
    };

    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[
                CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: first_output
                        .as_device_buffer()
                        .expect("first output device buffer"),
                    job,
                },
                CudaJ2kIdwtTarget {
                    ll: &ll,
                    hl: &hl,
                    lh: &lh,
                    hh: &hh,
                    output: second_output
                        .as_device_buffer()
                        .expect("second output device buffer"),
                    job,
                },
            ],
            &pool,
        )
        .expect("batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut first_actual = vec![0.0f32; 4];
    first_output
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut first_actual))
        .expect("download first batched IDWT");
    assert_eq!(first_actual, vec![7.0, 9.0, 10.0, 13.0]);
    let mut second_actual = vec![0.0f32; 4];
    second_output
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut second_actual))
        .expect("download second batched IDWT");
    assert_eq!(second_actual, vec![7.0, 9.0, 10.0, 13.0]);
}

#[test]
fn j2k_inverse_dwt_batch_odd_origin_matches_single_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::f32_slice_as_bytes(&[10.0]))
        .expect("upload odd LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&[2.0, 5.0]))
        .expect("upload odd HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&[4.0, 7.0]))
        .expect("upload odd LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&[1.0, 3.0, 6.0, 8.0]))
        .expect("upload odd HH");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 1,
            y0: 1,
            x1: 4,
            y1: 4,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 1,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 2,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        irreversible97: 0,
    };

    let single = context
        .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
        .expect("single CUDA inverse DWT");
    assert_eq!(single.execution().kernel_dispatches(), 3);
    let batch_output = pool
        .take(9 * std::mem::size_of::<f32>())
        .expect("odd batched IDWT output");
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: batch_output
                    .as_device_buffer()
                    .expect("odd batch output device buffer"),
                job,
            }],
            &pool,
        )
        .expect("odd-origin batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut single_actual = vec![0.0f32; 9];
    single
        .buffer()
        .expect("single odd output device buffer")
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut single_actual))
        .expect("download single odd IDWT");
    let mut batch_actual = vec![0.0f32; 9];
    batch_output
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut batch_actual))
        .expect("download batch odd IDWT");
    assert_eq!(batch_actual, single_actual);
}

#[test]
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn j2k_inverse_dwt_batch_large_reversible_matches_single_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let band_len = 64 * 64;
    let ll_values: Vec<f32> = (0..band_len).map(|idx| (idx % 19) as f32).collect();
    let hl_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 3) % 23) as f32).collect();
    let lh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 5) % 29) as f32).collect();
    let hh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 7) % 31) as f32).collect();
    let ll = context
        .upload(super::f32_slice_as_bytes(&ll_values))
        .expect("upload large LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&hl_values))
        .expect("upload large HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&lh_values))
        .expect("upload large LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&hh_values))
        .expect("upload large HH");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 64,
            y1: 64,
        },
        irreversible97: 0,
    };

    let single = context
        .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
        .expect("large single CUDA inverse DWT");
    let batch_output = pool
        .take(128 * 128 * std::mem::size_of::<f32>())
        .expect("large batched IDWT output");
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: batch_output
                    .as_device_buffer()
                    .expect("large batch output device buffer"),
                job,
            }],
            &pool,
        )
        .expect("large batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut single_actual = vec![0.0f32; 128 * 128];
    single
        .buffer()
        .expect("large single output device buffer")
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut single_actual))
        .expect("download large single IDWT");
    let mut batch_actual = vec![0.0f32; 128 * 128];
    batch_output
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut batch_actual))
        .expect("download large batch IDWT");
    assert_eq!(batch_actual, single_actual);
}

#[test]
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn j2k_inverse_dwt_batch_large_irreversible_matches_single_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let band_len = 128 * 128;
    let ll_values: Vec<f32> = (0..band_len)
        .map(|idx| ((idx % 43) as f32) * 0.25)
        .collect();
    let hl_values: Vec<f32> = (0..band_len)
        .map(|idx| (((idx * 3) % 47) as f32) * 0.125)
        .collect();
    let lh_values: Vec<f32> = (0..band_len)
        .map(|idx| (((idx * 5) % 53) as f32) * 0.0625)
        .collect();
    let hh_values: Vec<f32> = (0..band_len)
        .map(|idx| (((idx * 7) % 59) as f32) * 0.03125)
        .collect();
    let ll = context
        .upload(super::f32_slice_as_bytes(&ll_values))
        .expect("upload large irreversible LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&hl_values))
        .expect("upload large irreversible HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&lh_values))
        .expect("upload large irreversible LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&hh_values))
        .expect("upload large irreversible HH");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 128,
            y1: 128,
        },
        irreversible97: 1,
    };

    let single = context
        .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
        .expect("large irreversible single CUDA inverse DWT");
    let batch_output = pool
        .take(256 * 256 * std::mem::size_of::<f32>())
        .expect("large irreversible batched IDWT output");
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: batch_output
                    .as_device_buffer()
                    .expect("large irreversible batch output device buffer"),
                job,
            }],
            &pool,
        )
        .expect("large irreversible batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut single_actual = vec![0.0f32; 256 * 256];
    single
        .buffer()
        .expect("large irreversible single output device buffer")
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut single_actual))
        .expect("download large irreversible single IDWT");
    let mut batch_actual = vec![0.0f32; 256 * 256];
    batch_output
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut batch_actual))
        .expect("download large irreversible batch IDWT");
    assert_eq!(batch_actual, single_actual);
}

#[test]
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn j2k_inverse_dwt_batch_512_reversible_matches_single_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let band_len = 256 * 256;
    let ll_values: Vec<f32> = (0..band_len).map(|idx| (idx % 43) as f32).collect();
    let hl_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 3) % 47) as f32).collect();
    let lh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 5) % 53) as f32).collect();
    let hh_values: Vec<f32> = (0..band_len).map(|idx| ((idx * 7) % 59) as f32).collect();
    let ll = context
        .upload(super::f32_slice_as_bytes(&ll_values))
        .expect("upload 512 LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&hl_values))
        .expect("upload 512 HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&lh_values))
        .expect("upload 512 LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&hh_values))
        .expect("upload 512 HH");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 512,
            y1: 512,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 256,
            y1: 256,
        },
        irreversible97: 0,
    };

    let single = context
        .j2k_inverse_dwt_single_device_with_pool(&ll, &hl, &lh, &hh, job, &pool)
        .expect("512 single CUDA inverse DWT");
    let batch_output = pool
        .take(512 * 512 * std::mem::size_of::<f32>())
        .expect("512 batched IDWT output");
    let execution = context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: batch_output
                    .as_device_buffer()
                    .expect("512 batch output device buffer"),
                job,
            }],
            &pool,
        )
        .expect("512 batched CUDA inverse DWT");
    assert_eq!(execution.kernel_dispatches(), 2);

    let mut single_actual = vec![0.0f32; 512 * 512];
    single
        .buffer()
        .expect("512 single output device buffer")
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut single_actual))
        .expect("download 512 single IDWT");
    let mut batch_actual = vec![0.0f32; 512 * 512];
    batch_output
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut batch_actual))
        .expect("download 512 batch IDWT");
    assert_eq!(batch_actual, single_actual);
}

#[test]
fn j2k_inverse_dwt_batch_enqueue_matches_expected_outputs_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");
    let output = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("batched IDWT output");
    let job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        irreversible97: 0,
    };

    let queued = context
        .j2k_inverse_dwt_batch_device_enqueue_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: output.as_device_buffer().expect("output device buffer"),
                job,
            }],
            &pool,
        )
        .expect("enqueue batched CUDA inverse DWT");
    assert_eq!(queued.execution().kernel_dispatches(), 2);
    context.synchronize().expect("queued IDWT completion");
    drop(queued);

    let mut actual = vec![0.0f32; 4];
    output
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
        .expect("download queued batched IDWT");
    assert_eq!(actual, vec![7.0, 9.0, 10.0, 13.0]);
}

#[test]
#[allow(clippy::similar_names, clippy::too_many_lines)]
fn j2k_inverse_dwt_batch_sequence_enqueue_matches_two_stage_path_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");
    let stage2_hl = context
        .upload(super::f32_slice_as_bytes(&[0.0, 1.0, 2.0, 3.0]))
        .expect("upload stage2 HL");
    let stage2_lh = context
        .upload(super::f32_slice_as_bytes(&[4.0, 5.0, 6.0, 7.0]))
        .expect("upload stage2 LH");
    let stage2_hh = context
        .upload(super::f32_slice_as_bytes(&[8.0, 9.0, 10.0, 11.0]))
        .expect("upload stage2 HH");
    let stage1_job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
        },
        irreversible97: 0,
    };
    let stage2_job = CudaJ2kIdwtJob {
        rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 4,
        },
        ll_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        hl_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        lh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        hh_rect: CudaJ2kRect {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 2,
        },
        irreversible97: 0,
    };
    let legacy_stage1 = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("legacy stage1 output");
    let legacy_stage2 = pool
        .take(16 * std::mem::size_of::<f32>())
        .expect("legacy stage2 output");
    let sequence_stage1 = pool
        .take(4 * std::mem::size_of::<f32>())
        .expect("sequence stage1 output");
    let sequence_stage2 = pool
        .take(16 * std::mem::size_of::<f32>())
        .expect("sequence stage2 output");

    context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: &ll,
                hl: &hl,
                lh: &lh,
                hh: &hh,
                output: legacy_stage1
                    .as_device_buffer()
                    .expect("legacy stage1 device buffer"),
                job: stage1_job,
            }],
            &pool,
        )
        .expect("legacy stage1 IDWT");
    context
        .j2k_inverse_dwt_batch_device_with_pool(
            &[CudaJ2kIdwtTarget {
                ll: legacy_stage1
                    .as_device_buffer()
                    .expect("legacy stage1 device buffer"),
                hl: &stage2_hl,
                lh: &stage2_lh,
                hh: &stage2_hh,
                output: legacy_stage2
                    .as_device_buffer()
                    .expect("legacy stage2 device buffer"),
                job: stage2_job,
            }],
            &pool,
        )
        .expect("legacy stage2 IDWT");

    let sequence_stage1_targets = [CudaJ2kIdwtTarget {
        ll: &ll,
        hl: &hl,
        lh: &lh,
        hh: &hh,
        output: sequence_stage1
            .as_device_buffer()
            .expect("sequence stage1 device buffer"),
        job: stage1_job,
    }];
    let sequence_stage2_targets = [CudaJ2kIdwtTarget {
        ll: sequence_stage1
            .as_device_buffer()
            .expect("sequence stage1 device buffer"),
        hl: &stage2_hl,
        lh: &stage2_lh,
        hh: &stage2_hh,
        output: sequence_stage2
            .as_device_buffer()
            .expect("sequence stage2 device buffer"),
        job: stage2_job,
    }];
    let queued = context
        .j2k_inverse_dwt_batch_sequence_enqueue_with_pool(
            &[&sequence_stage1_targets, &sequence_stage2_targets],
            &pool,
        )
        .expect("queued IDWT sequence");
    assert_eq!(queued.execution().kernel_dispatches(), 4);
    assert_eq!(queued.resource_count(), 1);
    context
        .synchronize()
        .expect("queued IDWT sequence completion");
    drop(queued);

    let mut legacy_actual = vec![0.0f32; 16];
    legacy_stage2
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut legacy_actual))
        .expect("download legacy stage2 IDWT");
    let mut sequence_actual = vec![0.0f32; 16];
    sequence_stage2
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut sequence_actual))
        .expect("download sequence stage2 IDWT");
    assert_eq!(sequence_actual, legacy_actual);
}

#[test]
fn j2k_store_rgb8_mct_matches_inverse_mct_plus_store_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = [16.0f32, 18.0, 21.0, 24.0];
    let plane1 = [-3.0f32, 4.0, 5.0, -6.0];
    let plane2 = [2.0f32, -1.0, 7.0, 3.0];
    let legacy0 = context
        .upload(super::f32_slice_as_bytes(&plane0))
        .expect("upload legacy MCT plane 0");
    let legacy1 = context
        .upload(super::f32_slice_as_bytes(&plane1))
        .expect("upload legacy MCT plane 1");
    let legacy2 = context
        .upload(super::f32_slice_as_bytes(&plane2))
        .expect("upload legacy MCT plane 2");
    let fused0 = context
        .upload(super::f32_slice_as_bytes(&plane0))
        .expect("upload fused MCT plane 0");
    let fused1 = context
        .upload(super::f32_slice_as_bytes(&plane1))
        .expect("upload fused MCT plane 1");
    let fused2 = context
        .upload(super::f32_slice_as_bytes(&plane2))
        .expect("upload fused MCT plane 2");
    let addend = 128.0;

    let mct_stats = context
        .j2k_inverse_mct_device(
            &legacy0,
            &legacy1,
            &legacy2,
            super::CudaJ2kInverseMctJob {
                len: 4,
                irreversible97: 0,
                addend0: addend,
                addend1: addend,
                addend2: addend,
            },
        )
        .expect("legacy inverse MCT");
    assert_eq!(mct_stats.kernel_dispatches(), 1);
    let store_job = super::CudaJ2kStoreRgb8Job {
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
    };
    let legacy_output = context
        .j2k_store_rgb8_device(&legacy0, &legacy1, &legacy2, store_job)
        .expect("legacy RGB8 store");
    let fused_output = context
        .j2k_store_rgb8_mct_device(
            &fused0,
            &fused1,
            &fused2,
            super::CudaJ2kStoreRgb8MctJob {
                store: super::CudaJ2kStoreRgb8Job {
                    addend0: addend,
                    addend1: addend,
                    addend2: addend,
                    ..store_job
                },
                irreversible97: 0,
            },
        )
        .expect("fused RGB8 MCT store");

    assert_eq!(legacy_output.execution().kernel_dispatches(), 1);
    assert_eq!(fused_output.execution().kernel_dispatches(), 1);
    let mut legacy_bytes = vec![0u8; 16];
    legacy_output
        .buffer()
        .copy_to_host(&mut legacy_bytes)
        .expect("download legacy RGB8");
    let mut fused_bytes = vec![0u8; 16];
    fused_output
        .buffer()
        .copy_to_host(&mut fused_bytes)
        .expect("download fused RGB8");
    assert_eq!(fused_bytes, legacy_bytes);
}

#[test]
#[allow(clippy::similar_names, clippy::too_many_lines)]
fn j2k_store_rgb8_mct_batch_matches_separate_stores_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0_a = [16.0f32, 18.0, 21.0, 24.0];
    let plane1_a = [-3.0f32, 4.0, 5.0, -6.0];
    let plane2_a = [2.0f32, -1.0, 7.0, 3.0];
    let plane0_b = [3.0f32, 7.0, 11.0, 13.0];
    let plane1_b = [5.0f32, -2.0, 9.0, 1.0];
    let plane2_b = [-4.0f32, 6.0, 0.0, 8.0];

    let plane0_a = context
        .upload(super::f32_slice_as_bytes(&plane0_a))
        .expect("upload plane 0 A");
    let plane1_a = context
        .upload(super::f32_slice_as_bytes(&plane1_a))
        .expect("upload plane 1 A");
    let plane2_a = context
        .upload(super::f32_slice_as_bytes(&plane2_a))
        .expect("upload plane 2 A");
    let plane0_b = context
        .upload(super::f32_slice_as_bytes(&plane0_b))
        .expect("upload plane 0 B");
    let plane1_b = context
        .upload(super::f32_slice_as_bytes(&plane1_b))
        .expect("upload plane 1 B");
    let plane2_b = context
        .upload(super::f32_slice_as_bytes(&plane2_b))
        .expect("upload plane 2 B");

    let store = super::CudaJ2kStoreRgb8Job {
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
        addend0: 128.0,
        addend1: 128.0,
        addend2: 128.0,
        bit_depth0: 8,
        bit_depth1: 8,
        bit_depth2: 8,
        rgba: 1,
    };
    let separate_a = context
        .j2k_store_rgb8_mct_device(
            &plane0_a,
            &plane1_a,
            &plane2_a,
            super::CudaJ2kStoreRgb8MctJob {
                store,
                irreversible97: 0,
            },
        )
        .expect("separate fused store A");
    let separate_b = context
        .j2k_store_rgb8_mct_device(
            &plane0_b,
            &plane1_b,
            &plane2_b,
            super::CudaJ2kStoreRgb8MctJob {
                store,
                irreversible97: 0,
            },
        )
        .expect("separate fused store B");

    let batched = context
        .j2k_store_rgb8_mct_batch_device(&[
            super::CudaJ2kStoreRgb8MctTarget {
                plane0: &plane0_a,
                plane1: &plane1_a,
                plane2: &plane2_a,
                job: super::CudaJ2kStoreRgb8MctJob {
                    store,
                    irreversible97: 0,
                },
            },
            super::CudaJ2kStoreRgb8MctTarget {
                plane0: &plane0_b,
                plane1: &plane1_b,
                plane2: &plane2_b,
                job: super::CudaJ2kStoreRgb8MctJob {
                    store,
                    irreversible97: 0,
                },
            },
        ])
        .expect("batched fused store");

    assert_eq!(batched.execution().kernel_dispatches(), 1);
    assert_eq!(batched.outputs().len(), 2);
    let mut separate_a_bytes = vec![0u8; 16];
    separate_a
        .buffer()
        .copy_to_host(&mut separate_a_bytes)
        .expect("download separate A");
    let mut separate_b_bytes = vec![0u8; 16];
    separate_b
        .buffer()
        .copy_to_host(&mut separate_b_bytes)
        .expect("download separate B");
    let mut batch_a_bytes = vec![0u8; 16];
    batched.outputs()[0]
        .copy_to_host(&mut batch_a_bytes)
        .expect("download batch A");
    let mut batch_b_bytes = vec![0u8; 16];
    batched.outputs()[1]
        .copy_to_host(&mut batch_b_bytes)
        .expect("download batch B");
    assert_eq!(batch_a_bytes, separate_a_bytes);
    assert_eq!(batch_b_bytes, separate_b_bytes);
}

#[test]
fn j2k_store_rgb8_mct_single_matches_one_item_batch_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = [16.0f32, 18.0, 21.0, 24.0];
    let plane1 = [-3.0f32, 4.0, 5.0, -6.0];
    let plane2 = [2.0f32, -1.0, 7.0, 3.0];
    let single0 = context
        .upload(super::f32_slice_as_bytes(&plane0))
        .expect("upload single plane 0");
    let single1 = context
        .upload(super::f32_slice_as_bytes(&plane1))
        .expect("upload single plane 1");
    let single2 = context
        .upload(super::f32_slice_as_bytes(&plane2))
        .expect("upload single plane 2");
    let batch0 = context
        .upload(super::f32_slice_as_bytes(&plane0))
        .expect("upload batch plane 0");
    let batch1 = context
        .upload(super::f32_slice_as_bytes(&plane1))
        .expect("upload batch plane 1");
    let batch2 = context
        .upload(super::f32_slice_as_bytes(&plane2))
        .expect("upload batch plane 2");

    let store = super::CudaJ2kStoreRgb8Job {
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
        addend0: 128.0,
        addend1: 128.0,
        addend2: 128.0,
        bit_depth0: 8,
        bit_depth1: 8,
        bit_depth2: 8,
        rgba: 1,
    };
    let job = super::CudaJ2kStoreRgb8MctJob {
        store,
        irreversible97: 0,
    };
    let single = context
        .j2k_store_rgb8_mct_device(&single0, &single1, &single2, job)
        .expect("single RGB8 MCT store");
    let batch = context
        .j2k_store_rgb8_mct_batch_device(&[super::CudaJ2kStoreRgb8MctTarget {
            plane0: &batch0,
            plane1: &batch1,
            plane2: &batch2,
            job,
        }])
        .expect("one-item batch RGB8 MCT store");

    assert_eq!(single.execution().kernel_dispatches(), 1);
    assert_eq!(batch.execution().kernel_dispatches(), 1);
    let mut single_bytes = vec![0u8; 16];
    single
        .buffer()
        .copy_to_host(&mut single_bytes)
        .expect("download single RGB8 MCT store");
    let mut batch_bytes = vec![0u8; 16];
    batch.outputs()[0]
        .copy_to_host(&mut batch_bytes)
        .expect("download one-item batch RGB8 MCT store");
    assert_eq!(single_bytes, batch_bytes);
}

#[test]
fn j2k_store_rgb16_mct_matches_inverse_mct_plus_store_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let plane0 = [40.0f32, 44.0, 52.0, 55.0];
    let plane1 = [-3.5f32, 1.25, 2.75, -4.0];
    let plane2 = [5.0f32, -2.0, 1.5, 6.0];
    let legacy0 = context
        .upload(super::f32_slice_as_bytes(&plane0))
        .expect("upload legacy ICT plane 0");
    let legacy1 = context
        .upload(super::f32_slice_as_bytes(&plane1))
        .expect("upload legacy ICT plane 1");
    let legacy2 = context
        .upload(super::f32_slice_as_bytes(&plane2))
        .expect("upload legacy ICT plane 2");
    let fused0 = context
        .upload(super::f32_slice_as_bytes(&plane0))
        .expect("upload fused ICT plane 0");
    let fused1 = context
        .upload(super::f32_slice_as_bytes(&plane1))
        .expect("upload fused ICT plane 1");
    let fused2 = context
        .upload(super::f32_slice_as_bytes(&plane2))
        .expect("upload fused ICT plane 2");
    let addend = 32768.0;

    context
        .j2k_inverse_mct_device(
            &legacy0,
            &legacy1,
            &legacy2,
            super::CudaJ2kInverseMctJob {
                len: 4,
                irreversible97: 1,
                addend0: addend,
                addend1: addend,
                addend2: addend,
            },
        )
        .expect("legacy inverse ICT");
    let store_job = super::CudaJ2kStoreRgb16Job {
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
        bit_depth0: 16,
        bit_depth1: 16,
        bit_depth2: 16,
        rgba: 0,
    };
    let legacy_output = context
        .j2k_store_rgb16_device(&legacy0, &legacy1, &legacy2, store_job)
        .expect("legacy RGB16 store");
    let fused_output = context
        .j2k_store_rgb16_mct_device(
            &fused0,
            &fused1,
            &fused2,
            super::CudaJ2kStoreRgb16MctJob {
                store: super::CudaJ2kStoreRgb16Job {
                    addend0: addend,
                    addend1: addend,
                    addend2: addend,
                    ..store_job
                },
                irreversible97: 1,
            },
        )
        .expect("fused RGB16 MCT store");

    assert_eq!(legacy_output.execution().kernel_dispatches(), 1);
    assert_eq!(fused_output.execution().kernel_dispatches(), 1);
    let mut legacy_bytes = vec![0u8; 24];
    legacy_output
        .buffer()
        .copy_to_host(&mut legacy_bytes)
        .expect("download legacy RGB16");
    let mut fused_bytes = vec![0u8; 24];
    fused_output
        .buffer()
        .copy_to_host(&mut fused_bytes)
        .expect("download fused RGB16");
    assert_eq!(fused_bytes, legacy_bytes);
}

#[test]
fn j2k_dequantize_htj2k_codeblocks_multi_uses_one_dispatch_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let first = context
        .upload(super::i32_slice_as_bytes(&[0, 0, 0, 0]))
        .expect("upload first coefficients");
    let second = context
        .upload(super::i32_slice_as_bytes(&[0, 0]))
        .expect("upload second coefficients");
    let first_jobs = [CudaHtj2kCodeBlockJob {
        payload_offset: 0,
        width: 2,
        height: 2,
        payload_len: 0,
        cleanup_length: 0,
        refinement_length: 0,
        missing_bit_planes: 0,
        num_bitplanes: 1,
        number_of_coding_passes: 1,
        output_stride: 2,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: false,
    }];
    let second_jobs = [CudaHtj2kCodeBlockJob {
        payload_offset: 0,
        width: 2,
        height: 1,
        payload_len: 0,
        cleanup_length: 0,
        refinement_length: 0,
        missing_bit_planes: 0,
        num_bitplanes: 1,
        number_of_coding_passes: 1,
        output_stride: 2,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: false,
    }];

    let execution = context
        .j2k_dequantize_htj2k_codeblocks_multi_device(&[
            CudaHtj2kDequantizeTarget {
                coefficients: &first,
                jobs: &first_jobs,
                output_words: 4,
            },
            CudaHtj2kDequantizeTarget {
                coefficients: &second,
                jobs: &second_jobs,
                output_words: 2,
            },
        ])
        .expect("multi-buffer HTJ2K dequant");
    assert_eq!(execution.kernel_dispatches(), 1);

    let mut first_actual = vec![f32::NAN; 4];
    first
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut first_actual))
        .expect("download first coefficients");
    assert_eq!(first_actual, vec![0.0; 4]);
    let mut second_actual = vec![f32::NAN; 2];
    second
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut second_actual))
        .expect("download second coefficients");
    assert_eq!(second_actual, vec![0.0; 2]);
}

#[test]
fn queued_cleanup_metadata_dequantizes_without_second_job_upload_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let first = context
        .upload(super::i32_slice_as_bytes(&[1, i32::MIN + 2, 0, 3]))
        .expect("upload first coefficients");
    let second = context
        .upload(super::i32_slice_as_bytes(&[4, i32::MIN + 5]))
        .expect("upload second coefficients");
    let jobs = [
        CudaHtj2kCleanupMultiKernelJob {
            output_ptr: first.device_ptr(),
            coded_offset: 0,
            width: 2,
            height: 2,
            coded_len: 0,
            cleanup_length: 0,
            refinement_length: 0,
            missing_msbs: 0,
            num_bitplanes: 31,
            number_of_coding_passes: 1,
            output_stride: 2,
            output_offset: 0,
            dequantization_step: 0.5,
            stripe_causal: 0,
        },
        CudaHtj2kCleanupMultiKernelJob {
            output_ptr: second.device_ptr(),
            coded_offset: 0,
            width: 2,
            height: 1,
            coded_len: 0,
            cleanup_length: 0,
            refinement_length: 0,
            missing_msbs: 0,
            num_bitplanes: 31,
            number_of_coding_passes: 1,
            output_stride: 2,
            output_offset: 0,
            dequantization_step: 0.25,
            stripe_causal: 0,
        },
    ];
    let jobs_buffer = pool
        .upload(super::htj2k_cleanup_multi_jobs_as_bytes(&jobs))
        .expect("upload cleanup metadata");
    let queued = CudaQueuedHtj2kCleanup {
        resources: vec![jobs_buffer],
        status_buffer: None,
        status_count: jobs.len(),
        kernel_name: "j2k_htj2k_decode_codeblocks_multi",
        execution: CudaExecutionStats::default(),
    };

    let execution = context
        .j2k_dequantize_queued_htj2k_cleanup_with_pool(&queued)
        .expect("dequant from queued cleanup metadata");
    assert_eq!(execution.kernel_dispatches(), 1);

    let mut first_actual = vec![f32::NAN; 4];
    first
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut first_actual))
        .expect("download first coefficients");
    assert_eq!(first_actual, vec![0.5, -1.0, 0.0, 1.5]);
    let mut second_actual = vec![f32::NAN; 2];
    second
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut second_actual))
        .expect("download second coefficients");
    assert_eq!(second_actual, vec![1.0, -1.25]);
}

#[test]
fn htj2k_decode_multi_kernel_routes_cleanup_only_jobs() {
    let cleanup_job = CudaHtj2kCleanupMultiKernelJob {
        output_ptr: 0,
        coded_offset: 0,
        width: 64,
        height: 64,
        coded_len: 8,
        cleanup_length: 8,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 8,
        number_of_coding_passes: 1,
        output_stride: 64,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: 0,
    };
    let (_, cleanup_kernel_name) = super::htj2k_decode_multi_kernel_for_jobs(&[cleanup_job]);
    assert_eq!(
        cleanup_kernel_name,
        "j2k_htj2k_decode_codeblocks_multi_cleanup_only"
    );

    let mut refinement_job = cleanup_job;
    refinement_job.refinement_length = 4;
    refinement_job.number_of_coding_passes = 2;
    let (_, generic_kernel_name) = super::htj2k_decode_multi_kernel_for_jobs(&[refinement_job]);
    assert_eq!(generic_kernel_name, "j2k_htj2k_decode_codeblocks_multi");
}

#[test]
fn htj2k_decode_multi_cleanup_dequant_kernel_accepts_cleanup_only_jobs() {
    let cleanup_job = CudaHtj2kCleanupMultiKernelJob {
        output_ptr: 0,
        coded_offset: 0,
        width: 64,
        height: 64,
        coded_len: 8,
        cleanup_length: 8,
        refinement_length: 0,
        missing_msbs: 0,
        num_bitplanes: 8,
        number_of_coding_passes: 1,
        output_stride: 64,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: 0,
    };
    let (_, cleanup_dequant_kernel_name) =
        super::htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[cleanup_job])
            .expect("cleanup-only jobs use fused cleanup/dequant kernel");
    assert_eq!(
        cleanup_dequant_kernel_name,
        "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize"
    );
}

#[test]
fn htj2k_decode_multi_cleanup_dequant_kernel_rejects_refinement_jobs() {
    let mut refinement_job = CudaHtj2kCleanupMultiKernelJob {
        output_ptr: 0,
        coded_offset: 0,
        width: 64,
        height: 64,
        coded_len: 12,
        cleanup_length: 8,
        refinement_length: 4,
        missing_msbs: 0,
        num_bitplanes: 8,
        number_of_coding_passes: 2,
        output_stride: 64,
        output_offset: 0,
        dequantization_step: 1.0,
        stripe_causal: 0,
    };
    assert!(super::htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[refinement_job]).is_none());

    refinement_job.refinement_length = 0;
    assert!(super::htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&[refinement_job]).is_none());
}

#[test]
#[allow(clippy::similar_names)]
fn htj2k_cleanup_multi_empty_targets_use_no_dispatch_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let tables = context
        .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
            vlc_table0: &first_vlc,
            vlc_table1: &later_vlc,
            uvlc_table0: &first_uvlc,
            uvlc_table1: &later_uvlc,
        })
        .expect("decode tables");
    let resources = context
        .upload_htj2k_decode_resources_with_tables(&[], &tables)
        .expect("decode resources");

    let execution = context
        .decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool(
            &resources,
            &[] as &[CudaHtj2kCleanupTarget<'_>],
            &pool,
        )
        .expect("empty cleanup batch");

    assert_eq!(execution.kernel_dispatches(), 0);
    assert_eq!(execution.decode_kernel_dispatches(), 0);
}

#[test]
#[allow(clippy::similar_names)]
fn htj2k_cleanup_multi_enqueue_empty_targets_finish_with_no_dispatch_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let tables = context
        .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
            vlc_table0: &first_vlc,
            vlc_table1: &later_vlc,
            uvlc_table0: &first_uvlc,
            uvlc_table1: &later_uvlc,
        })
        .expect("decode tables");
    let resources = context
        .upload_htj2k_decode_resources_with_tables(&[], &tables)
        .expect("decode resources");

    let queued = context
        .decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
            &resources,
            &[] as &[CudaHtj2kCleanupTarget<'_>],
            &pool,
        )
        .expect("empty queued cleanup batch");
    assert_eq!(queued.execution().kernel_dispatches(), 0);
    assert_eq!(queued.execution().decode_kernel_dispatches(), 0);
    assert_eq!(queued.resource_count(), 0);

    let execution = queued.finish().expect("finish empty queued cleanup");
    assert_eq!(execution.kernel_dispatches(), 0);
    assert_eq!(execution.decode_kernel_dispatches(), 0);
}

#[test]
fn j2k_forward_rct_matches_cpu_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let mut plane0 = vec![10.0, 1.0, 0.0, 255.0, 128.0];
    let mut plane1 = vec![20.0, 2.0, 255.0, 0.0, 64.0];
    let mut plane2 = vec![30.0, 3.0, 128.0, 127.0, 32.0];
    let mut expected0 = plane0.clone();
    let mut expected1 = plane1.clone();
    let mut expected2 = plane2.clone();
    for ((r, g), b) in expected0
        .iter_mut()
        .zip(expected1.iter_mut())
        .zip(expected2.iter_mut())
    {
        let r0 = *r;
        let g0 = *g;
        let b0 = *b;
        *r = ((r0 + 2.0_f32 * g0 + b0) * 0.25_f32).floor();
        *g = b0 - g0;
        *b = r0 - g0;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let execution = context
        .j2k_forward_rct(&mut plane0, &mut plane1, &mut plane2)
        .expect("CUDA forward RCT");

    assert_eq!(execution.kernel_dispatches(), 1);
    assert_eq!(plane0, expected0);
    assert_eq!(plane1, expected1);
    assert_eq!(plane2, expected2);
}

#[test]
fn j2k_deinterleave_to_f32_matches_cpu_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let pixels = [0u8, 128, 255, 64, 32, 16];
    let context = CudaContext::system_default().expect("CUDA context");
    let output = context
        .j2k_deinterleave_to_f32(&pixels, 2, 3, 8, false)
        .expect("CUDA deinterleave");

    assert_eq!(output.execution().kernel_dispatches(), 1);
    assert_eq!(
        output.components(),
        &[vec![-128.0, -64.0], vec![0.0, -96.0], vec![127.0, -112.0],]
    );
}

#[test]
fn j2k_deinterleave_then_rct_can_stay_resident_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let pixels = [10u8, 20, 30, 40, 50, 60];
    let context = CudaContext::system_default().expect("CUDA context");
    let mut components = context
        .j2k_deinterleave_to_f32_resident(&pixels, 2, 3, 8, false)
        .expect("resident CUDA deinterleave");

    assert_eq!(components.num_components(), 3);
    assert_eq!(components.num_pixels(), 2);
    assert_eq!(components.execution().kernel_dispatches(), 1);

    let rct_execution = context
        .j2k_forward_rct_resident(&mut components)
        .expect("resident CUDA forward RCT");

    assert_eq!(rct_execution.kernel_dispatches(), 1);
    assert_eq!(
        components
            .download_components()
            .expect("download resident components"),
        vec![vec![-108.0, -78.0], vec![10.0, 10.0], vec![-10.0, -10.0]]
    );
}

#[test]
fn j2k_deinterleave_then_ict_can_stay_resident_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let pixels = [10u8, 20, 30, 40, 50, 60];
    let context = CudaContext::system_default().expect("CUDA context");
    let mut components = context
        .j2k_deinterleave_to_f32_resident(&pixels, 2, 3, 8, false)
        .expect("resident CUDA deinterleave");

    let ict_execution = context
        .j2k_forward_ict_resident(&mut components)
        .expect("resident CUDA forward ICT");

    assert_eq!(ict_execution.kernel_dispatches(), 1);
    let actual = components
        .download_components()
        .expect("download resident components");
    let expected = [[-118.0f32, -88.0], [-108.0, -78.0], [-98.0, -68.0]];
    for idx in 0..2 {
        let r = expected[0][idx];
        let g = expected[1][idx];
        let b = expected[2][idx];
        let expected_y = 0.299 * r + 0.587 * g + 0.114 * b;
        let blue_chroma = -0.16875 * r - 0.33126 * g + 0.5 * b;
        let red_chroma = 0.5 * r - 0.41869 * g - 0.08131 * b;
        assert!((actual[0][idx] - expected_y).abs() < 0.000_1);
        assert!((actual[1][idx] - blue_chroma).abs() < 0.000_1);
        assert!((actual[2][idx] - red_chroma).abs() < 0.000_1);
    }
}

#[test]
fn j2k_resident_deinterleave_can_feed_resident_dwt53_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let pixels = [0u8, 64, 128, 255];
    let context = CudaContext::system_default().expect("CUDA context");
    let components = context
        .j2k_deinterleave_to_f32_resident(&pixels, 4, 1, 8, false)
        .expect("resident CUDA deinterleave");
    let host_component = components
        .download_components()
        .expect("download source component")[0]
        .clone();
    let expected = context
        .j2k_forward_dwt53(&host_component, 2, 2, 1)
        .expect("host-staged CUDA DWT");

    let resident = context
        .j2k_forward_dwt53_resident_component(&components, 0, 2, 2, 1)
        .expect("resident CUDA DWT");

    assert_eq!(resident.levels(), expected.levels());
    assert_eq!(resident.ll_dimensions(), expected.ll_dimensions());
    assert_eq!(resident.execution().copy_kernel_dispatches, 1);
    assert_eq!(
        resident
            .download_transformed()
            .expect("download resident DWT"),
        expected.transformed()
    );
}

#[test]
fn j2k_resident_deinterleave_can_feed_resident_dwt97_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let pixels = [0u8, 64, 128, 255];
    let context = CudaContext::system_default().expect("CUDA context");
    let components = context
        .j2k_deinterleave_to_f32_resident(&pixels, 4, 1, 8, false)
        .expect("resident CUDA deinterleave");
    let host_component = components
        .download_components()
        .expect("download source component")[0]
        .clone();
    let expected = context
        .j2k_forward_dwt97(&host_component, 2, 2, 1)
        .expect("host-staged CUDA DWT");

    let resident = context
        .j2k_forward_dwt97_resident_component(&components, 0, 2, 2, 1)
        .expect("resident CUDA DWT");

    assert_eq!(resident.levels(), expected.levels());
    assert_eq!(resident.ll_dimensions(), expected.ll_dimensions());
    assert_eq!(resident.execution().copy_kernel_dispatches, 1);
    assert_eq!(
        resident
            .download_transformed()
            .expect("download resident DWT"),
        expected.transformed()
    );
}

#[test]
fn j2k_forward_ict_matches_cpu_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let mut plane0 = vec![10.0, 1.0, 0.0, 255.0, 128.0];
    let mut plane1 = vec![20.0, 2.0, 255.0, 0.0, 64.0];
    let mut plane2 = vec![30.0, 3.0, 128.0, 127.0, 32.0];
    let mut expected0 = plane0.clone();
    let mut expected1 = plane1.clone();
    let mut expected2 = plane2.clone();
    for ((r, g), b) in expected0
        .iter_mut()
        .zip(expected1.iter_mut())
        .zip(expected2.iter_mut())
    {
        let r0 = *r;
        let g0 = *g;
        let b0 = *b;
        *r = 0.299 * r0 + 0.587 * g0 + 0.114 * b0;
        *g = -0.16875 * r0 - 0.33126 * g0 + 0.5 * b0;
        *b = 0.5 * r0 - 0.41869 * g0 - 0.08131 * b0;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let execution = context
        .j2k_forward_ict(&mut plane0, &mut plane1, &mut plane2)
        .expect("CUDA forward ICT");

    assert_eq!(execution.kernel_dispatches(), 1);
    for (actual, expected) in plane0.iter().zip(expected0) {
        assert!((*actual - expected).abs() < 0.0001);
    }
    for (actual, expected) in plane1.iter().zip(expected1) {
        assert!((*actual - expected).abs() < 0.0001);
    }
    for (actual, expected) in plane2.iter().zip(expected2) {
        assert!((*actual - expected).abs() < 0.0001);
    }
}

#[test]
fn j2k_forward_dwt53_matches_cpu_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let width = 5usize;
    let height = 3usize;
    let samples: Vec<f32> = (0..width * height)
        .map(|value| {
            let sample = u16::try_from((value * 7 + 3) % 19).expect("sample fits in u16");
            f32::from(sample)
        })
        .collect();
    let expected = cpu_forward_dwt53_buffer(&samples, width, height, 1);

    let context = CudaContext::system_default().expect("CUDA context");
    let output = context
        .j2k_forward_dwt53(
            &samples,
            u32::try_from(width).expect("width fits in u32"),
            u32::try_from(height).expect("height fits in u32"),
            1,
        )
        .expect("CUDA forward 5/3 DWT");

    assert_eq!(output.execution().kernel_dispatches(), 2);
    assert_eq!(output.transformed(), expected.as_slice());
    assert_eq!(output.ll_dimensions(), (3, 2));
}

#[test]
fn j2k_forward_dwt97_matches_cpu_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let width = 5usize;
    let height = 3usize;
    let samples: Vec<f32> = (0..width * height)
        .map(|value| {
            let sample = u16::try_from((value * 11 + 5) % 31).expect("sample fits in u16");
            f32::from(sample) - 12.0
        })
        .collect();
    let expected = cpu_forward_dwt97_buffer(&samples, width, height, 1);

    let context = CudaContext::system_default().expect("CUDA context");
    let output = context
        .j2k_forward_dwt97(
            &samples,
            u32::try_from(width).expect("width fits in u32"),
            u32::try_from(height).expect("height fits in u32"),
            1,
        )
        .expect("CUDA forward 9/7 DWT");

    assert_eq!(output.execution().kernel_dispatches(), 2);
    for (actual, expected) in output.transformed().iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.001);
    }
    assert_eq!(output.ll_dimensions(), (3, 2));
}

#[test]
fn j2k_quantize_subband_matches_cpu_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let samples = [-3.6f32, -2.5, -0.4, 0.0, 0.49, 1.5, 3.2, 9.9];
    let context = CudaContext::system_default().expect("CUDA context");
    let reversible = context
        .j2k_quantize_subband(
            &samples,
            CudaJ2kQuantizeJob {
                step_exponent: 8,
                step_mantissa: 0,
                range_bits: 8,
                reversible: true,
            },
        )
        .expect("CUDA reversible quantize");
    assert_eq!(reversible.execution().kernel_dispatches(), 1);
    assert_eq!(reversible.coefficients(), &[-4, -3, 0, 0, 0, 2, 3, 10]);

    let irreversible = context
        .j2k_quantize_subband(
            &samples,
            CudaJ2kQuantizeJob {
                step_exponent: 9,
                step_mantissa: 0,
                range_bits: 8,
                reversible: false,
            },
        )
        .expect("CUDA irreversible quantize");
    assert_eq!(irreversible.execution().kernel_dispatches(), 1);
    // delta = 2^(range_bits - step_exponent) = 2^(8 - 9) = 0.5, so q = sign*floor(|s|/0.5).
    // Matches native QuantStepSize::delta and JPEG 2000 T.800 Annex E.
    assert_eq!(irreversible.coefficients(), &[-7, -5, 0, 0, 0, 3, 6, 19]);
}

#[test]
fn j2k_quantize_strided_resident_subband_matches_contiguous_when_runtime_required() {
    if !cuda_runtime_required() {
        return;
    }

    let samples: Vec<f32> = (0u16..12).map(|value| f32::from(value) - 6.0).collect();
    let context = CudaContext::system_default().expect("CUDA context");
    let sample_buffer = context.upload_f32(&samples).expect("resident samples");
    let quantization = CudaJ2kQuantizeJob {
        step_exponent: 8,
        step_mantissa: 0,
        range_bits: 8,
        reversible: true,
    };
    let resident = context
        .j2k_quantize_subband_region_resident(
            &sample_buffer,
            CudaJ2kQuantizeSubbandRegionJob {
                x0: 1,
                y0: 1,
                width: 2,
                height: 2,
                stride: 4,
                quantization,
            },
        )
        .expect("resident strided quantize");
    let contiguous = [samples[5], samples[6], samples[9], samples[10]];
    let expected = context
        .j2k_quantize_subband(&contiguous, quantization)
        .expect("contiguous quantize");

    assert_eq!(resident.coefficient_count(), 4);
    assert_eq!(resident.execution().kernel_dispatches(), 1);
    assert_eq!(
        resident
            .download_coefficients()
            .expect("download resident quantized coefficients"),
        expected.coefficients()
    );
}

fn cpu_forward_dwt53_buffer(samples: &[f32], width: usize, height: usize, levels: u8) -> Vec<f32> {
    let mut buffer = samples.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    for _ in 0..levels {
        if current_width < 2 && current_height < 2 {
            break;
        }
        if current_height >= 2 {
            let low_height = current_height.div_ceil(2);
            let mut col = vec![0.0; current_height];
            for x in 0..current_width {
                for y in 0..current_height {
                    col[y] = buffer[y * width + x];
                }
                forward_lift_53(&mut col);
                for y in 0..low_height {
                    buffer[y * width + x] = col[y * 2];
                }
                for y in 0..current_height / 2 {
                    buffer[(low_height + y) * width + x] = col[y * 2 + 1];
                }
            }
        }
        if current_width >= 2 {
            let mut row = vec![0.0; current_width];
            for y in 0..current_height {
                let row_start = y * width;
                row.copy_from_slice(&buffer[row_start..row_start + current_width]);
                forward_lift_53(&mut row);
                let low_width = current_width.div_ceil(2);
                for x in 0..low_width {
                    buffer[row_start + x] = row[x * 2];
                }
                for x in 0..current_width / 2 {
                    buffer[row_start + low_width + x] = row[x * 2 + 1];
                }
            }
        }
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
    }

    buffer
}

fn cpu_forward_dwt97_buffer(samples: &[f32], width: usize, height: usize, levels: u8) -> Vec<f32> {
    let mut buffer = samples.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    for _ in 0..levels {
        if current_width < 2 && current_height < 2 {
            break;
        }
        if current_height >= 2 {
            let low_height = current_height.div_ceil(2);
            let mut col = vec![0.0; current_height];
            for x in 0..current_width {
                for y in 0..current_height {
                    col[y] = buffer[y * width + x];
                }
                forward_lift_97(&mut col);
                for y in 0..low_height {
                    buffer[y * width + x] = col[y * 2];
                }
                for y in 0..current_height / 2 {
                    buffer[(low_height + y) * width + x] = col[y * 2 + 1];
                }
            }
        }
        if current_width >= 2 {
            let mut row = vec![0.0; current_width];
            for y in 0..current_height {
                let row_start = y * width;
                row.copy_from_slice(&buffer[row_start..row_start + current_width]);
                forward_lift_97(&mut row);
                let low_width = current_width.div_ceil(2);
                for x in 0..low_width {
                    buffer[row_start + x] = row[x * 2];
                }
                for x in 0..current_width / 2 {
                    buffer[row_start + low_width + x] = row[x * 2 + 1];
                }
            }
        }
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
    }

    buffer
}

fn forward_lift_53(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] -= ((left + right) * 0.5).floor();
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += ((left + right) * 0.25 + 0.5).floor();
    }
}

fn forward_lift_97(data: &mut [f32]) {
    const ALPHA: f32 = -1.586_134_3;
    const BETA: f32 = -0.052_980_117;
    const GAMMA: f32 = 0.882_911_1;
    const DELTA: f32 = 0.443_506_87;
    const KAPPA: f32 = 1.230_174_1;
    const INV_KAPPA: f32 = 1.0 / KAPPA;

    let n = data.len();
    if n < 2 {
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += ALPHA * (left + right);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += BETA * (left + right);
    }
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += GAMMA * (left + right);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += DELTA * (left + right);
    }
    for i in (0..n).step_by(2) {
        data[i] *= INV_KAPPA;
    }
    for i in (1..n).step_by(2) {
        data[i] *= KAPPA;
    }
}
