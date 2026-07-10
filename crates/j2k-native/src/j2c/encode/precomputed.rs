// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    codestream_write, count_compact_code_blocks, encode_prepared_resolution_packets,
    encode_with_accelerator, encode_with_accelerator_and_component_sample_info,
    internal_sub_band_type, ordered_prepared_compact_resolution_packets,
    ordered_prepared_resolution_packets, packet_descriptors_for_compact_order,
    packet_descriptors_for_order, packet_encode, packetize_resolution_packets_with_options,
    precinct_exponents_for_options, prepare_precomputed_htj2k97_image_for_batch,
    public_packetization_progression_order, public_packetization_resolutions_from_compact,
    quantize, raw_pixel_bytes_per_sample, scalar_packet_descriptors, validate_band_len,
    validate_component_sample_info, validate_irreversible_quantization_profile, vec,
    write_single_tile_packetized_codestream, BlockCodingMode, CpuOnlyJ2kEncodeStageAccelerator,
    EncodeComponentSampleInfo, EncodeOptions, EncodeParams, EncodedHtJ2kCodeBlock,
    EncodedJ2kCodeBlock, J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Output,
    J2kForwardDwt97Job, J2kForwardDwt97Output, J2kPacketizationEncodeJob, J2kQuantizeSubbandJob,
    J2kSubBandType, J2kTier1CodeBlockEncodeJob, PrecomputedHtj2k53Component,
    PrecomputedHtj2k53Image, PrecomputedHtj2k97Component, PrecomputedHtj2k97Image,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PreparedCompactCodeBlock, PreparedCompactResolutionPacket,
    PreparedCompactSubband, PreparedEncodeCodeBlock, PreparedEncodeSubband,
    PreparedPrecomputedHtj2k97Image, PreparedResolutionPacket, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    QuantStepSize, Range, Vec, MAX_J2K_SPEC_COMPONENTS, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};

mod accelerator;
use self::accelerator::{PrecomputedDwt97Accelerator, PrecomputedDwtAccelerator};
mod api53;
pub(super) use self::api53::encode_precomputed_53_with_component_sample_info_and_accelerator;
pub use self::api53::{
    encode_precomputed_htj2k_53, encode_precomputed_htj2k_53_with_accelerator,
    encode_precomputed_htj2k_53_with_mct, encode_precomputed_htj2k_53_with_mct_and_accelerator,
    encode_precomputed_j2k_53, encode_precomputed_j2k_53_with_accelerator,
    encode_precomputed_j2k_53_with_mct, encode_precomputed_j2k_53_with_mct_and_accelerator,
};
mod api97;
pub use self::api97::{
    encode_precomputed_htj2k_97, encode_precomputed_htj2k_97_with_accelerator,
    encode_preencoded_htj2k_97, encode_preencoded_htj2k_97_compact_owned_with_accelerator,
    encode_preencoded_htj2k_97_owned_with_accelerator, encode_preencoded_htj2k_97_with_accelerator,
    encode_prequantized_htj2k_97, encode_prequantized_htj2k_97_with_accelerator,
};
mod batch97;
pub use self::batch97::encode_precomputed_htj2k_97_batch_with_accelerator;
mod packets;
#[cfg(test)]
pub(in crate::j2c::encode) use self::packets::prepared_subband_from_preencoded_owned as prepared_subband_from_preencoded_owned_for_tests;
use self::packets::{
    prepared_resolution_packets_from_preencoded_compact_component,
    prepared_resolution_packets_from_preencoded_component,
    prepared_resolution_packets_from_preencoded_component_owned,
    prepared_resolution_packets_from_prequantized_component, zero_pixel_buffer,
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
