// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    codestream_write, count_compact_code_blocks, encode_precomputed_53_single_tile,
    encode_precomputed_97_single_tile, encode_prepared_resolution_packets_for_session,
    internal_sub_band_type, ordered_prepared_resolution_packets_for_session,
    packet_descriptors_for_order_for_session, packet_encode, packetization_requires_scalar,
    packetize_resolution_packets_with_options_for_session,
    prepare_precomputed_htj2k97_image_for_batch, quantize,
    split_component_resolution_packets_by_precinct_for_session, validate_band_len,
    validate_component_sample_info, validate_irreversible_quantization_profile,
    validate_precinct_exponents_for_options, write_single_tile_packetized_codestream_for_session,
    BlockCodingMode, CpuOnlyJ2kEncodeStageAccelerator, EncodeComponentSampleInfo, EncodeOptions,
    EncodeParams, EncodeProgressionOrder, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock,
    J2kEncodeStageAccelerator, J2kForwardDwt53Output, J2kForwardDwt97Output,
    J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor, J2kQuantizeSubbandJob,
    J2kSubBandType, J2kTier1CodeBlockEncodeJob, NativeEncodePhase, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeRetainedInput, NativeEncodeSession,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PreparedCompactCodeBlock, PreparedCompactResolutionPacket,
    PreparedCompactSubband, PreparedEncodeCodeBlock, PreparedEncodeSubband,
    PreparedResolutionPacket, PrequantizedHtj2k97Component, PrequantizedHtj2k97Image,
    PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband, QuantStepSize, Range,
    ResolutionPacket, Vec, MAX_J2K_SPEC_COMPONENTS, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};

mod accelerator;
use self::accelerator::PrecomputedStageAccelerator;
pub(super) mod allocation;
mod api53;
mod options;
pub(super) mod orchestrator;
pub(super) use self::api53::encode_precomputed_53_with_component_sample_info_for_session;
pub(in crate::j2c) use self::api53::encode_precomputed_htj2k_53_with_mct_and_retained_owner;
pub use self::api53::{
    encode_precomputed_htj2k_53, encode_precomputed_htj2k_53_with_accelerator,
    encode_precomputed_htj2k_53_with_mct, encode_precomputed_htj2k_53_with_mct_and_accelerator,
    encode_precomputed_j2k_53, encode_precomputed_j2k_53_with_accelerator,
    encode_precomputed_j2k_53_with_mct, encode_precomputed_j2k_53_with_mct_and_accelerator,
};
use self::options::{try_precomputed_options, PrecomputedOptionMode};
mod api97;
pub use self::api97::{
    encode_precomputed_htj2k_97, encode_precomputed_htj2k_97_with_accelerator,
    encode_preencoded_htj2k_97, encode_preencoded_htj2k_97_compact_owned_with_accelerator,
    encode_preencoded_htj2k_97_owned_with_accelerator, encode_preencoded_htj2k_97_with_accelerator,
    encode_prequantized_htj2k_97, encode_prequantized_htj2k_97_with_accelerator,
};
mod batch97;
pub use self::batch97::{
    encode_precomputed_htj2k_97_batch_owned_with_accelerator,
    encode_precomputed_htj2k_97_batch_with_accelerator,
};
mod compact97;
mod limits;
pub use self::limits::{
    encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_owned_with_accelerator_and_max_host_bytes,
    encode_prequantized_htj2k_97_with_accelerator_and_max_host_bytes,
};
mod packets;
#[cfg(test)]
pub(in crate::j2c::encode) use self::packets::prepared_subband_from_preencoded_owned_for_test as prepared_subband_from_preencoded_owned_for_tests;
use self::packets::{
    compact_payload_slice, move_preencoded_payloads_into_skeleton, try_preencoded_owned_skeleton,
    try_prepared_packets_from_preencoded_component,
    try_prepared_packets_from_prequantized_component,
};
mod validation;
pub(super) use self::validation::{
    precomputed_97_level_count, validate_precomputed_dwt97_geometry,
    validate_precomputed_dwt_geometry,
};
use self::validation::{
    precomputed_level_count, preencoded_97_level_count, preencoded_compact_97_level_count,
    prequantized_97_level_count, validate_preencoded_compact_htj2k97_image,
    validate_preencoded_htj2k97_image, validate_prequantized_htj2k97_image,
};
