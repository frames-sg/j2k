use super::*;

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
            reserved1: 2,
            reserved2: 0,
        },
        execution: super::CudaExecutionStats::default(),
        stage_timings: super::CudaHtj2kEncodeStageTimings::default(),
    };

    assert_eq!(encoded.cleanup_length(), 7);
    assert_eq!(encoded.refinement_length(), 3);
    assert_eq!(encoded.sigprop_length(), 2);
    assert_eq!(encoded.magref_length(), 1);
}

fn htj2k_multi_input_compact_job(
    job: super::CudaHtj2kEncodeKernelJob,
) -> super::CudaHtj2kEncodeMultiInputKernelJob {
    super::CudaHtj2kEncodeMultiInputKernelJob {
        coefficient_ptr: 0x1000,
        coefficient_offset: job.coefficient_offset,
        coefficient_stride: job.coefficient_stride,
        width: job.width,
        height: job.height,
        total_bitplanes: job.total_bitplanes,
        output_offset: job.output_offset,
        output_capacity: job.output_capacity,
        target_coding_passes: job.target_coding_passes,
    }
}

fn assert_compact_jobs_match_for_single_and_multi_input(
    statuses: &[super::CudaHtj2kEncodeStatus],
    kernel_jobs: &[super::CudaHtj2kEncodeKernelJob],
) -> Result<(Vec<super::CudaHtj2kEncodeCompactJob>, usize), CudaError> {
    let multi_input_jobs = kernel_jobs
        .iter()
        .copied()
        .map(htj2k_multi_input_compact_job)
        .collect::<Vec<_>>();
    let mut single_budget = crate::allocation::HostPhaseBudget::new("test compact jobs");
    let mut multi_budget = crate::allocation::HostPhaseBudget::new("test compact jobs");
    let single = super::htj2k_encode_compact_jobs(statuses, kernel_jobs, &mut single_budget);
    let multi = super::htj2k_encode_compact_jobs_multi_input(
        statuses,
        &multi_input_jobs,
        &mut multi_budget,
    );
    match (single, multi) {
        (Ok(single), Ok(multi)) => {
            assert_eq!(single, multi);
            Ok(single)
        }
        (Err(single), Err(multi)) => {
            assert_eq!(format!("{single:?}"), format!("{multi:?}"));
            Err(single)
        }
        (single, multi) => panic!(
            "single and multi-input compact planners diverged: single={single:?} multi={multi:?}"
        ),
    }
}

#[test]
fn htj2k_encode_compact_jobs_accept_empty_batches() {
    let (compact_jobs, compact_len) =
        assert_compact_jobs_match_for_single_and_multi_input(&[], &[]).expect("empty compact plan");

    assert!(compact_jobs.is_empty());
    assert_eq!(compact_len, 0);
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
        assert_compact_jobs_match_for_single_and_multi_input(&statuses, &kernel_jobs)
            .expect("valid compact jobs");

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
fn htj2k_encode_compact_jobs_accept_exact_capacity_payloads() {
    let kernel_jobs = [super::CudaHtj2kEncodeKernelJob {
        coefficient_offset: 0,
        coefficient_stride: 64,
        width: 64,
        height: 64,
        total_bitplanes: 8,
        output_offset: 11,
        output_capacity: 5,
        target_coding_passes: 1,
    }];
    let statuses = [super::CudaHtj2kEncodeStatus {
        code: super::HTJ2K_STATUS_OK,
        data_len: 5,
        reserved2: 9,
        ..super::CudaHtj2kEncodeStatus::default()
    }];

    let (compact_jobs, compact_len) =
        assert_compact_jobs_match_for_single_and_multi_input(&statuses, &kernel_jobs)
            .expect("exact-capacity compact job");

    assert_eq!(compact_len, 5);
    assert_eq!(
        compact_jobs,
        vec![super::CudaHtj2kEncodeCompactJob {
            source_offset: 11,
            compact_offset: 0,
            data_len: 5,
            reserved: 9,
        }]
    );
}

#[test]
fn htj2k_encode_compact_jobs_reject_payloads_larger_than_capacity() {
    let kernel_jobs = [super::CudaHtj2kEncodeKernelJob {
        coefficient_offset: 0,
        coefficient_stride: 64,
        width: 64,
        height: 64,
        total_bitplanes: 8,
        output_offset: 0,
        output_capacity: 5,
        target_coding_passes: 1,
    }];
    let statuses = [super::CudaHtj2kEncodeStatus {
        code: super::HTJ2K_STATUS_OK,
        data_len: 6,
        ..super::CudaHtj2kEncodeStatus::default()
    }];

    assert!(matches!(
        assert_compact_jobs_match_for_single_and_multi_input(&statuses, &kernel_jobs),
        Err(CudaError::LengthTooLarge { len }) if len == 6
    ));
}

#[cfg(all(feature = "cuda-oxide-j2k-encode", j2k_cuda_oxide_j2k_encode_built))]
#[test]
fn cuda_oxide_htj2k_compact_codeblocks_assembles_payload_when_required() {
    const J2K_HT_MEL_SIZE: usize = 192;
    const J2K_HT_VLC_SIZE: usize = 3072 - J2K_HT_MEL_SIZE;
    const J2K_HT_MS_SIZE: usize = (16384usize * 16).div_ceil(15);
    const J2K_HT_MEL_OFFSET: usize = J2K_HT_MS_SIZE;
    const J2K_HT_VLC_OFFSET: usize = J2K_HT_MS_SIZE + J2K_HT_MEL_SIZE;
    const J2K_HT_COMPACT_ASSEMBLE_FLAG: u32 = 0x8000_0000;

    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let source_offset = 3usize;
    let plain_source_offset = source_offset + J2K_HT_VLC_OFFSET + J2K_HT_VLC_SIZE + 8;
    let mut scratch = vec![0u8; plain_source_offset + 4];
    scratch[source_offset..source_offset + 3].copy_from_slice(&[10, 11, 12]);
    scratch[source_offset + J2K_HT_MEL_OFFSET..source_offset + J2K_HT_MEL_OFFSET + 2]
        .copy_from_slice(&[20, 21]);
    let vlc_start = source_offset + J2K_HT_VLC_OFFSET + J2K_HT_VLC_SIZE - 3;
    scratch[vlc_start..vlc_start + 3].copy_from_slice(&[30, 31, 32]);
    scratch[plain_source_offset..plain_source_offset + 4].copy_from_slice(&[40, 41, 42, 43]);
    let jobs = [
        super::CudaHtj2kEncodeCompactJob {
            source_offset: u32::try_from(source_offset).expect("source offset fits"),
            compact_offset: 0,
            data_len: 8,
            reserved: J2K_HT_COMPACT_ASSEMBLE_FLAG | 2 | (3 << 15),
        },
        super::CudaHtj2kEncodeCompactJob {
            source_offset: u32::try_from(plain_source_offset).expect("plain offset fits"),
            compact_offset: 8,
            data_len: 4,
            reserved: 0,
        },
    ];
    let expected = [10, 11, 12, 20, 21, 30, 0x15, 0, 40, 41, 42, 43];

    let scratch_buffer = context.upload(&scratch).expect("scratch upload");
    let compact_buffer = context.allocate(expected.len()).expect("compact output");
    let jobs_buffer = context
        .upload(crate::bytes::htj2k_encode_compact_jobs_as_bytes(&jobs))
        .expect("compact job upload");

    crate::J2kCudaEngine::new(&context)
        .launch_htj2k_compact_codeblocks(&scratch_buffer, &compact_buffer, &jobs_buffer, jobs.len())
        .expect("cuda-oxide compact codeblocks");
    let mut actual = vec![0u8; expected.len()];
    compact_buffer
        .copy_to_host(&mut actual)
        .expect("download compact output");

    assert_eq!(actual, expected);
}

#[test]
fn htj2k_encode_tables_feed_resident_region_encode_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let coefficients = crate::J2kCudaEngine::new(&context)
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

    let encoded = crate::J2kCudaEngine::new(&context)
        .encode_htj2k_codeblock_regions_resident(
            &coefficients,
            4,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &vlc_table0,
                vlc_table1: &vlc_table1,
                uvlc_table: &uvlc_table,
            },
        )
        .expect("resource-backed resident HTJ2K encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert_eq!(encoded.code_blocks().len(), 1);
}

#[test]
fn htj2k_encode_resident_region_reuses_pool_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = crate::J2kCudaEngine::new(&context)
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let coefficients = crate::J2kCudaEngine::new(&context)
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

    let encoded = crate::J2kCudaEngine::new(&context)
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
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = crate::J2kCudaEngine::new(&context)
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let coefficients = crate::J2kCudaEngine::new(&context)
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("resident coefficients");
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];

    let encoded = crate::J2kCudaEngine::new(&context)
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
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = crate::J2kCudaEngine::new(&context)
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let first = crate::J2kCudaEngine::new(&context)
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("first resident coefficients");
    let second = crate::J2kCudaEngine::new(&context)
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

    let first_separate = crate::J2kCudaEngine::new(&context)
        .encode_htj2k_codeblocks_resident_with_resources_and_pool(
            &first,
            4,
            &first_jobs,
            &resources,
            &pool,
        )
        .expect("first separate resident encode");
    let second_separate = crate::J2kCudaEngine::new(&context)
        .encode_htj2k_codeblocks_resident_with_resources_and_pool(
            &second,
            2,
            &second_jobs,
            &resources,
            &pool,
        )
        .expect("second separate resident encode");

    let combined = crate::J2kCudaEngine::new(&context)
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
fn htj2k_encode_rejects_unsupported_refinement_pass_count_when_required() {
    if !cuda_runtime_gate() {
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

    let error = crate::J2kCudaEngine::new(&context)
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
fn htj2k_encode_accepts_general_sigprop_coefficients_when_required() {
    if !cuda_runtime_gate() {
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

    let encoded = crate::J2kCudaEngine::new(&context)
        .encode_htj2k_codeblocks(
            &coefficients,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &[0u16; 2048],
                vlc_table1: &[0u16; 2048],
                uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
            },
        )
        .expect("general target-2 SigProp coefficients encode");

    assert_eq!(encoded.code_blocks()[0].num_coding_passes(), 2);
    assert!(encoded.code_blocks()[0].sigprop_length() > 0);
}

#[test]
fn htj2k_encode_accepts_isolated_target_three_coefficients_when_required() {
    if !cuda_runtime_gate() {
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

    let encoded = crate::J2kCudaEngine::new(&context)
        .encode_htj2k_codeblocks(
            &coefficients,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &[0u16; 2048],
                vlc_table1: &[0u16; 2048],
                uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
            },
        )
        .expect("isolated target-3 coefficient encode");

    assert_eq!(encoded.code_blocks()[0].num_coding_passes(), 3);
    assert!(encoded.code_blocks()[0].sigprop_length() > 0);
}

#[test]
fn htj2k_encode_resources_feed_one_job_batch_encode_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = crate::J2kCudaEngine::new(&context)
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");

    let encoded = crate::J2kCudaEngine::new(&context)
        .encode_htj2k_codeblocks_with_resources(
            &[0, 0, 0, 0],
            &[CudaHtj2kEncodeCodeBlockJob {
                coefficient_offset: 0,
                width: 2,
                height: 2,
                total_bitplanes: 1,
                target_coding_passes: 1,
            }],
            &resources,
        )
        .expect("resource-backed one-job HTJ2K encode");
    let block = encoded
        .code_blocks()
        .first()
        .expect("one encoded code block");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    // An all-zero codeblock has no significant bitplanes, so the encoder emits zero
    // coding passes (matching native ht_block_encode::encode_code_block).
    assert_eq!(block.num_coding_passes(), 0);
    assert_eq!(block.cleanup_length(), 0);
    assert_eq!(block.data().len(), 0);
    assert_eq!(block.refinement_length(), 0);
}
