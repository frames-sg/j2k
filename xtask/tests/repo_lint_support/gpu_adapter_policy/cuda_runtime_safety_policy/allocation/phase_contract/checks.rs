// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::super::super::{assert_pattern_checks, PatternCheck};
use super::sources::PhaseSources;

pub(super) fn assert_decode_contracts(sources: &PhaseSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA HTJ2K decode phase planning", &sources.decode_planning).required(
            &[
                "htj2k_cleanup_multi_kernel_jobs_with_live_host_bytes(",
                "htj2k_dequantize_kernel_jobs_with_live_host_bytes(",
                "HostPhaseBudget::with_live_bytes(",
                "validate_htj2k_output_layout_with_live_bytes(",
                "host_budget.live_bytes()",
            ],
        ),
        PatternCheck::new(
            "CUDA HTJ2K decode completion ownership",
            &sources.decode_completion,
        )
        .required(&[
            "finish_host_live_bytes: finish_budget.live_bytes()",
            "host_budget.account_vec(&kernel_jobs)?",
            "host_budget.try_vec_filled(",
            "j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_and_live_host_bytes(",
        ]),
        PatternCheck::new(
            "CUDA queued cleanup completion ownership",
            &sources.decode_queued,
        )
        .required(&[
            "finish_host_live_bytes: usize",
            "HostPhaseBudget::with_live_bytes(",
            "self.finish_host_live_bytes",
            ".try_vec_filled(self.status_count, CudaHtj2kStatus::default())",
        ]),
        PatternCheck::new("CUDA IDWT sequence phase ownership", &sources.idwt_sequence).required(
            &[
                "j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes(",
                "HostPhaseBudget::with_live_bytes(",
                "host_budget.try_vec_with_capacity(total_target_count)?",
                "host_budget.try_vec_with_capacity(target_batches.len())?",
                "host_budget.try_vec_with_capacity(1)?",
            ],
        ),
    ]);
}

pub(super) fn assert_encode_and_transfer_contracts(sources: &PhaseSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA HTJ2K encode phase planning", &sources.encode_planning).required(
            &[
                "htj2k_encode_kernel_jobs_with_live_host_bytes(",
                "htj2k_encode_multi_input_kernel_jobs_with_live_host_bytes(",
                "htj2k_encode_region_kernel_jobs_with_live_host_bytes(",
                "host_budget: &mut HostPhaseBudget",
            ],
        ),
        PatternCheck::new(
            "CUDA HTJ2K encode live-byte API handoff",
            &sources.encode_api,
        )
        .required(&[
            "into_owned_code_blocks_with_live_host_bytes(live_host_bytes)",
            "htj2k_encode_multi_input_kernel_jobs_with_live_host_bytes(",
            "htj2k_encode_region_kernel_jobs_with_live_host_bytes(",
        ]),
        PatternCheck::new(
            "CUDA HTJ2K encode completion ownership",
            &sources.encode_completion,
        )
        .required(&[
            "HostPhaseBudget::with_live_bytes(",
            "host_budget.account_capacity::<CudaHtj2kEncodeKernelJob>(",
            "host_budget.account_capacity::<CudaHtj2kEncodeMultiInputKernelJob>(",
            "copy_pooled_bytes_to_vec_uninit_with_budget(",
        ]),
        PatternCheck::new(
            "CUDA compact encode expansion ownership",
            &sources.compact_expansion,
        )
        .required(&[
            "into_owned_code_blocks_with_live_host_bytes(",
            "HostPhaseBudget::with_live_bytes(",
            "host_budget.account_vec(&payload)?",
            "host_budget.account_vec(&code_blocks)?",
            "host_budget.try_vec_from_slice(&payload[payload_range])?",
        ]),
        PatternCheck::new("CUDA packetization phase ownership", &sources.packetize).required(&[
            "packetize_htj2k_cleanup_packets_with_tag_state_and_live_host_bytes(",
            "HostPhaseBudget::with_live_bytes(",
            "drop(initial_statuses);",
            "completion_budget.account_vec(&kernel_packets)?",
            "completion_budget.try_vec_filled(data_len, 0u8)?",
            "host_budget: &mut HostPhaseBudget",
        ]),
        PatternCheck::new("CUDA color-store phase ownership", &sources.store_batch).required(&[
            "validate_rgb8_mct_targets(",
            "j2k_store_rgb8_mct_batch_contiguous_device_with_live_host_bytes(",
            "HostPhaseBudget::with_live_bytes(",
            "host_budget.account_vec(&plan.targets)?",
        ]),
        PatternCheck::new(
            "CUDA transcode aggregate readback ownership",
            &sources.transcode,
        )
        .required(&[
            "j2k_transcode_reversible_dwt53_and_live_host_bytes(",
            "j2k_transcode_dwt97_and_live_host_bytes(",
            "j2k_transcode_dwt97_batch_with_pool_and_live_host_bytes(",
            "j2k_transcode_htj2k97_codeblock_batch_with_pool_and_live_host_bytes(",
            "HostPhaseBudget::with_live_bytes(",
            "download_pooled_f32_band(",
        ]),
        PatternCheck::new("CUDA budgeted band transfer", &sources.band_transfer).required(&[
            "host_budget: &mut HostPhaseBudget",
            "host_budget.try_vec_filled(",
        ]),
        PatternCheck::new("CUDA budgeted pooled readback", &sources.pool_readback).required(&[
            "copy_pooled_bytes_to_vec_uninit_with_budget(",
            "host_budget.try_vec_with_capacity(byte_len)?",
        ]),
        PatternCheck::new("CUDA forward-DWT aggregate readback", &sources.dwt).required(&[
            "download_transformed_with_budget(",
            "host_budget.account_vec(&resident.levels)?",
        ]),
    ]);
}
