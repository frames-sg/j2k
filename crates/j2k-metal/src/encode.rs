// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
mod batch;
#[cfg(any(test, target_os = "macos"))]
mod config;
#[cfg(target_os = "macos")]
mod device_resident;
mod encoded;
#[cfg(target_os = "macos")]
mod host_fallback;
#[cfg(target_os = "macos")]
mod packet_plan;
#[cfg(target_os = "macos")]
mod plan;
#[cfg(target_os = "macos")]
mod resident_estimate;
#[cfg(target_os = "macos")]
mod resident_hybrid;
#[cfg(target_os = "macos")]
mod resident_plan;
#[cfg(target_os = "macos")]
mod resident_prepare;
#[cfg(target_os = "macos")]
mod resident_submit;
#[cfg(target_os = "macos")]
mod resident_types;
#[cfg(target_os = "macos")]
mod resident_validation;
#[cfg(target_os = "macos")]
mod resident_wait;
mod roundtrip_validation;
#[cfg(target_os = "macos")]
mod routing;
mod stage_accelerator;
mod stats;
mod submitted;
#[cfg(all(test, target_os = "macos"))]
mod test_helpers;
mod types;
#[cfg(not(target_os = "macos"))]
mod unavailable;
#[cfg(target_os = "macos")]
mod validation;

#[cfg(target_os = "macos")]
use crate::compute;
use j2k::J2kLosslessEncodeOptions;
#[cfg(target_os = "macos")]
use j2k::J2kLosslessSamples;
#[cfg(target_os = "macos")]
use j2k::{EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, ReversibleTransform};
#[cfg(target_os = "macos")]
use j2k_core::{BackendKind, DeviceSurface, PixelFormat};
#[cfg(target_os = "macos")]
use j2k_native::J2kPacketizationEncodeJob;
#[cfg(target_os = "macos")]
use metal::Buffer;
#[cfg(target_os = "macos")]
use std::time::Duration;
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(test)]
use self::config::{
    default_gpu_encode_memory_budget_bytes_for_hw_mem, resident_lossless_chunk_ranges_for_test,
    resolve_lossless_encode_config_for_test,
};
#[cfg(target_os = "macos")]
use self::config::{
    resident_lossless_chunk_ranges_from_code_blocks, resolve_lossless_encode_config,
};
#[cfg(any(test, target_os = "macos"))]
use self::config::{
    resident_lossless_code_block_chunk_cap, resident_lossless_encode_config_for_mode,
};
pub use self::encoded::MetalEncodedJ2k;
#[cfg(target_os = "macos")]
use self::packet_plan::{
    cpu_packetization_resolutions_from_lossless_device_plan,
    lossless_options_for_resident_htj2k_tile_job, packet_descriptors_for_lossless_device_order,
    packetization_progression_order, resident_packetization_resolutions_from_lossless_device_plan,
    should_use_resident_htj2k_host_shape_for_auto, should_use_resident_htj2k_host_tile_for_auto,
};
#[cfg(target_os = "macos")]
use self::plan::{
    lossless_device_encode_plan, LosslessDeviceEncodePlan, RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
};
#[cfg(all(test, target_os = "macos"))]
use self::resident_estimate::estimated_tier1_output_bytes;
#[cfg(target_os = "macos")]
use self::resident_estimate::{
    estimate_resident_lossless_encode_peak_bytes,
    resident_classic_batch_encode_should_retry_conservative,
    resident_codestream_assembly_job_for_metadata,
    resident_ht_batch_encode_should_retry_conservative,
};
#[cfg(target_os = "macos")]
use self::resident_types::{
    FinishedResidentLosslessBufferEncode, PlannedResidentLosslessBufferEncode,
    PreparedResidentLosslessBufferEncode, ResidentLosslessBufferEncodeMetadata,
    SubmittedResidentLosslessMetalBufferEncodeBatch,
    SubmittedResidentLosslessMetalBufferEncodeBatchKind,
    SubmittedResidentLosslessMetalBufferEncodeChunk,
};
pub use self::roundtrip_validation::{
    validate_lossless_roundtrip_on_metal, validate_lossless_roundtrip_on_metal_with_session,
};
#[cfg(all(test, target_os = "macos"))]
use self::stage_accelerator::metal_dispatch_option;
pub use self::stage_accelerator::MetalEncodeStageAccelerator;
#[cfg(test)]
use self::stats::add_resident_prep_duration;
#[cfg(any(test, target_os = "macos"))]
use self::stats::add_resident_prep_wall_duration;
pub use self::stats::{
    MetalLosslessBufferEncodeBatchOutcome, MetalLosslessEncodeBatchStats,
    MetalLosslessEncodeStageStats,
};
#[cfg(target_os = "macos")]
use self::submitted::{
    OwnedMetalLosslessEncodeTile, SubmittedJ2kLosslessMetalBufferEncodeBatchState,
    SubmittedJ2kLosslessMetalEncodeBatchState,
};
pub use self::submitted::{
    SubmittedJ2kLosslessMetalBufferEncodeBatch, SubmittedJ2kLosslessMetalEncodeBatch,
};
#[cfg(all(test, target_os = "macos"))]
use self::test_helpers::{
    collect_inflight_limited_ordered, encode_lossless_from_metal_buffer,
    encode_lossless_from_metal_buffer_to_metal_with_report,
    encode_lossless_from_metal_buffers_to_metal_with_report,
    encode_lossless_from_padded_metal_buffer_to_metal_with_report,
    encode_lossless_from_padded_metal_buffer_with_report,
    encode_lossless_from_padded_metal_buffers_to_metal_batch,
    encode_lossless_from_padded_metal_buffers_to_metal_with_report,
    encode_lossless_from_padded_metal_buffers_with_report, set_test_resident_encode_failure_index,
    submit_lossless_from_metal_buffer, submit_lossless_from_padded_metal_buffer,
    test_resident_encode_failure_index,
};
pub use self::types::{
    MetalEncodeInputStaging, MetalLosslessBufferEncodeOutcome, MetalLosslessEncodeBatchRequest,
    MetalLosslessEncodeConfig, MetalLosslessEncodeOutcome, MetalLosslessEncodeResidency,
    MetalLosslessEncodeTile,
};
#[cfg(target_os = "macos")]
use self::validation::{
    lossless_sample_shape, validate_metal_encode_tile, validate_padded_contiguous_metal_encode_tile,
};

#[cfg(target_os = "macos")]
pub use self::batch::{
    encode_lossless_batch_with_report, submit_lossless_batch, submit_lossless_batch_to_metal,
};
#[cfg(target_os = "macos")]
use self::batch::{
    encode_lossless_owned_tiles_with_report,
    encode_owned_lossless_tiles_to_metal_buffer_fallback_batch, host_outcome_from_buffer_outcome,
};
#[cfg(target_os = "macos")]
use self::device_resident::{
    encode_lossless_tile_to_metal_buffer_with_report,
    try_encode_lossless_tile_device_resident_with_report,
};
#[cfg(target_os = "macos")]
use self::host_fallback::encode_lossless_tile_with_report;
#[cfg(target_os = "macos")]
use self::resident_hybrid::{
    encode_resident_ht_tile_body_with_cpu_packetization, lossless_device_coefficient_count,
};
#[cfg(target_os = "macos")]
use self::resident_plan::plan_resident_lossless_buffer_encode;
#[cfg(target_os = "macos")]
use self::resident_submit::{duration_share, submit_planned_resident_lossless_tiles};
#[cfg(target_os = "macos")]
use self::resident_validation::{
    validate_lossless_roundtrip_on_metal_region_with_session,
    validate_lossless_roundtrip_on_metal_tile_with_session,
};
#[cfg(target_os = "macos")]
use self::resident_wait::wait_submitted_resident_lossless_buffer_encode_batch;
#[cfg(all(test, target_os = "macos"))]
use self::routing::should_try_auto_resident_lossless_host_format;
#[cfg(target_os = "macos")]
use self::routing::{
    copy_padded_metal_buffer_from_bytes, host_output_encode_options,
    should_try_auto_resident_lossless_host_encode, should_try_resident_lossless_host_encode,
    should_try_resident_lossless_host_encode_for_tiles,
};
#[cfg(not(target_os = "macos"))]
pub use self::unavailable::{
    encode_lossless_batch_with_report, submit_lossless_batch, submit_lossless_batch_to_metal,
};

#[cfg(test)]
mod structure_tests;
#[cfg(test)]
mod tests;
