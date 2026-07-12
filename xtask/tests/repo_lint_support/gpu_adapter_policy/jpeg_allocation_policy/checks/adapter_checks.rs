// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::JpegAllocationSources;
use super::{assert_ordered, calls};

pub(super) fn assert_policy(sources: &JpegAllocationSources) {
    assert!(
        sources
            .device_plan
            .contains("#[derive(Debug, PartialEq, Eq)]\n#[doc(hidden)]\n/// Validated decode plan")
            && sources
                .device_plan
                .contains("The plan is move-only because an owned scan payload")
            && !sources.device_plan.contains(
                "#[derive(Debug, Clone, PartialEq, Eq)]\n#[doc(hidden)]\n/// Validated decode plan"
            ),
        "DeviceDecodePlan must remain move-only while small device metadata may stay Clone"
    );
    calls(
        "JPEG device-plan builder",
        &sources.device_plan,
        "build_device_plan",
    )
    .assert_contains(
        "JPEG device-plan aggregate ownership",
        &[
            "validate_scan_bytes",
            "retained_decoder_allocation_bytes",
            "device_plan_output_allocation_bytes",
            "terminated_copy_len",
            "ensure_allocation_bytes",
            "try_reserve_for_len_with_live_budget",
            "build_checkpoint_plan_from_validated_with_live_budget",
        ],
    );
    assert_ordered(
        "JPEG CPU checkpoint baseline",
        &sources.decoder_sequential,
        &[
            "fn checkpoint_for_mcu(",
            "retained_allocation_bytes_excluding_cpu_checkpoint_cache()?",
            "self.cpu_entropy_checkpoints",
            "checkpoint_before_mcu(",
            "retained_decoder_baseline_bytes",
        ],
    );
    assert_ordered(
        "JPEG CUDA checkpoint conversion",
        &sources.owned_decode_plan,
        &[
            "fn cuda_entropy_checkpoints_with_cap(",
            "HostPhaseBudget::with_cap(",
            "\"CUDA JPEG entropy checkpoint conversion\"",
            ".try_vec_with_capacity(checkpoints.len())",
            "converted.extend(",
        ],
    );
}
