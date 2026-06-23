// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kValidateBytesParams {
    pub(crate) byte_len: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kValidateBytesStatus {
    pub(crate) code: u32,
    pub(crate) index: u32,
    pub(crate) expected: u32,
    pub(crate) actual: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kCopyInterleavedParams {
    pub(crate) src_width: u32,
    pub(crate) src_height: u32,
    pub(crate) src_stride: u32,
    pub(crate) dst_width: u32,
    pub(crate) dst_height: u32,
    pub(crate) dst_stride: u32,
    pub(crate) bytes_per_pixel: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kLosslessDeinterleaveParams {
    pub(crate) src_width: u32,
    pub(crate) src_height: u32,
    pub(crate) src_stride: u32,
    pub(crate) dst_width: u32,
    pub(crate) dst_height: u32,
    pub(crate) components: u32,
    pub(crate) bytes_per_sample: u32,
    pub(crate) sample_offset: u32,
    pub(crate) signed_samples: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kLosslessCoefficientJob {
    pub(crate) coefficient_offset: u32,
    pub(crate) component: u32,
    pub(crate) subband_x: u32,
    pub(crate) subband_y: u32,
    pub(crate) block_x: u32,
    pub(crate) block_y: u32,
    pub(crate) block_width: u32,
    pub(crate) block_height: u32,
    pub(crate) full_width: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kPackParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) out_stride: u32,
    pub(crate) output_channels: u32,
    pub(crate) opaque_alpha: u32,
    pub(crate) max_values: [f32; 4],
    pub(crate) u8_scales: [f32; 4],
    pub(crate) u16_scales: [f32; 4],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kMctRgb8PackParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) out_stride: u32,
    pub(crate) transform: u32,
    pub(crate) addends: [f32; 3],
    pub(crate) max_values: [f32; 3],
    pub(crate) u8_scales: [f32; 3],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kBatchedMctRgb8PackParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) out_stride: u32,
    pub(crate) transform: u32,
    pub(crate) batch_count: u32,
    pub(crate) plane_stride: u32,
    pub(crate) output_stride: u32,
    pub(crate) addends: [f32; 3],
    pub(crate) max_values: [f32; 3],
    pub(crate) u8_scales: [f32; 3],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kRepeatedGrayPackParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) out_stride: u32,
    pub(crate) batch_count: u32,
    pub(crate) max_value: f32,
    pub(crate) u8_scale: f32,
    pub(crate) u16_scale: f32,
}
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_STATUS_UNSUPPORTED: u32 = 2;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES: u32 = 1 << 0;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS: u32 = 1 << 1;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT: u32 = 1 << 2;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS: u32 = 1 << 3;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS: u32 = 1 << 4;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_MAX_WIDTH: u32 = 64;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_MAX_HEIGHT: u32 = 64;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_MAX_COEFF_COUNT: usize =
    (J2K_CLASSIC_MAX_WIDTH as usize + 2) * (J2K_CLASSIC_MAX_HEIGHT as usize + 2);
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_ENCODE_32_MAX_WIDTH: u32 = 32;
#[cfg(target_os = "macos")]
pub(crate) const J2K_CLASSIC_ENCODE_32_MAX_HEIGHT: u32 = 32;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kClassicCleanupBatchJob {
    pub(crate) coded_offset: u32,
    pub(crate) coded_len: u32,
    pub(crate) segment_offset: u32,
    pub(crate) segment_count: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) missing_msbs: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) roi_shift: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) sub_band_type: u32,
    pub(crate) style_flags: u32,
    pub(crate) strict: u32,
    pub(crate) dequantization_step: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kClassicSegment {
    pub(crate) data_offset: u32,
    pub(crate) data_length: u32,
    pub(crate) start_coding_pass: u32,
    pub(crate) end_coding_pass: u32,
    pub(crate) use_arithmetic: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kClassicStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
}

#[cfg(target_os = "macos")]
pub(crate) const J2K_IDWT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const J2K_IDWT_STATUS_FAIL: u32 = 1;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kIdwtSingleDecompositionParams {
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) ll_x: u32,
    pub(crate) ll_y: u32,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) hl_x: u32,
    pub(crate) hl_y: u32,
    pub(crate) hl_width: u32,
    pub(crate) hl_height: u32,
    pub(crate) lh_x: u32,
    pub(crate) lh_y: u32,
    pub(crate) lh_width: u32,
    pub(crate) lh_height: u32,
    pub(crate) hh_x: u32,
    pub(crate) hh_y: u32,
    pub(crate) hh_width: u32,
    pub(crate) hh_height: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kRepeatedIdwtSingleDecompositionParams {
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) ll_x: u32,
    pub(crate) ll_y: u32,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) hl_x: u32,
    pub(crate) hl_y: u32,
    pub(crate) hl_width: u32,
    pub(crate) hl_height: u32,
    pub(crate) lh_x: u32,
    pub(crate) lh_y: u32,
    pub(crate) lh_width: u32,
    pub(crate) lh_height: u32,
    pub(crate) hh_x: u32,
    pub(crate) hh_y: u32,
    pub(crate) hh_width: u32,
    pub(crate) hh_height: u32,
    pub(crate) ll_instance_stride: u32,
    pub(crate) hl_instance_stride: u32,
    pub(crate) lh_instance_stride: u32,
    pub(crate) hh_instance_stride: u32,
    pub(crate) batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kIdwtStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
}

#[cfg(target_os = "macos")]
pub(crate) const J2K_MCT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const J2K_MCT_STATUS_FAIL: u32 = 1;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kInverseMctParams {
    pub(crate) _len: u32,
    pub(crate) _transform: u32,
    pub(crate) _addend0: f32,
    pub(crate) _addend1: f32,
    pub(crate) _addend2: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kForwardRctParams {
    pub(crate) _len: u32,
    pub(crate) _reserved0: u32,
    pub(crate) _reserved1: u32,
    pub(crate) _reserved2: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kForwardIctParams {
    pub(crate) _len: u32,
    pub(crate) _reserved0: u32,
    pub(crate) _reserved1: u32,
    pub(crate) _reserved2: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kQuantizeSubbandParams {
    pub(crate) _len: u32,
    pub(crate) _step_exponent: u32,
    pub(crate) _step_mantissa: u32,
    pub(crate) _range_bits: u32,
    pub(crate) _reversible: u32,
    pub(crate) _reserved0: u32,
    pub(crate) _reserved1: u32,
    pub(crate) _reserved2: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kForwardDwt53Params {
    pub(crate) full_width: u32,
    pub(crate) current_width: u32,
    pub(crate) current_height: u32,
    pub(crate) low_width: u32,
    pub(crate) low_height: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kForwardDwt53BatchedParams {
    pub(crate) full_width: u32,
    pub(crate) current_width: u32,
    pub(crate) current_height: u32,
    pub(crate) low_width: u32,
    pub(crate) low_height: u32,
    pub(crate) component_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kMctStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) _reserved0: u32,
    pub(crate) _reserved1: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kStoreParams {
    pub(crate) input_width: u32,
    pub(crate) source_x: u32,
    pub(crate) source_y: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kRepeatedStoreParams {
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) input_instance_stride: u32,
    pub(crate) source_x: u32,
    pub(crate) source_y: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend: f32,
    pub(crate) batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kRepeatedGrayStoreParams {
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) source_x: u32,
    pub(crate) source_y: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend: f32,
    pub(crate) batch_count: u32,
    pub(crate) max_value: f32,
    pub(crate) u8_scale: f32,
    pub(crate) u16_scale: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kGrayStoreParams {
    pub(crate) input_width: u32,
    pub(crate) source_x: u32,
    pub(crate) source_y: u32,
    pub(crate) copy_width: u32,
    pub(crate) copy_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) addend: f32,
    pub(crate) max_value: f32,
    pub(crate) u8_scale: f32,
    pub(crate) u16_scale: f32,
}

pub(crate) const J2K_HT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const J2K_HT_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
pub(crate) const J2K_HT_STATUS_UNSUPPORTED: u32 = 2;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtCleanupParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coded_len: u32,
    pub(crate) cleanup_length: u32,
    pub(crate) refinement_length: u32,
    pub(crate) missing_msbs: u32,
    pub(crate) num_bitplanes: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) dequantization_step: f32,
    pub(crate) stripe_causal: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtCleanupBatchJob {
    pub(crate) coded_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coded_len: u32,
    pub(crate) cleanup_length: u32,
    pub(crate) refinement_length: u32,
    pub(crate) missing_msbs: u32,
    pub(crate) num_bitplanes: u32,
    pub(crate) roi_shift: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) output_stride: u32,
    pub(crate) output_offset: u32,
    pub(crate) dequantization_step: f32,
    pub(crate) stripe_causal: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtRepeatedBatchParams {
    pub(crate) job_count: u32,
    pub(crate) output_plane_len: u32,
    pub(crate) batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kClassicRepeatedBatchParams {
    pub(crate) job_count: u32,
    pub(crate) output_plane_len: u32,
    pub(crate) batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kHtStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
}

#[cfg(target_os = "macos")]
pub(crate) const J2K_ENCODE_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const J2K_ENCODE_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
pub(crate) const J2K_ENCODE_STATUS_UNSUPPORTED: u32 = 2;
#[cfg(target_os = "macos")]
pub(crate) const J2K_HT_ENCODE_MEL_SIZE: usize = 192;
#[cfg(target_os = "macos")]
pub(crate) const J2K_HT_ENCODE_VLC_SIZE: usize = 3072 - J2K_HT_ENCODE_MEL_SIZE;
#[cfg(target_os = "macos")]
pub(crate) const J2K_HT_ENCODE_MS_SIZE: usize = (16_384usize * 16).div_ceil(15);
#[cfg(target_os = "macos")]
pub(crate) const J2K_HT_ENCODE_BASE_OUTPUT_SIZE: usize =
    J2K_HT_ENCODE_MS_SIZE + J2K_HT_ENCODE_MEL_SIZE + J2K_HT_ENCODE_VLC_SIZE;
#[cfg(target_os = "macos")]
pub(crate) const J2K_HT_ENCODE_MAX_SAMPLES: usize = 16_384;
#[cfg(target_os = "macos")]
pub(crate) const J2K_HT_ENCODE_MS_BYTES_PER_SAMPLE_FLOOR: usize = 5;
#[cfg(target_os = "macos")]
pub(crate) const PACKET_PAYLOAD_COPY_BYTES_PER_STRIPE: u32 = 256;
#[cfg(target_os = "macos")]
pub(crate) const PACKET_PAYLOAD_COPY_STRIPES_PER_JOB: u32 = 4;

#[cfg(target_os = "macos")]
pub(crate) const HT_PACKET_CAPACITY_ENV: &str = "J2K_METAL_HT_PACKET_CAPACITY";
#[cfg(target_os = "macos")]
pub(crate) const CLASSIC_TIER1_TOKEN_ARENA_BYTES: usize = 4096;
#[cfg(target_os = "macos")]
pub(crate) const CLASSIC_TIER1_MQ_BYTE_TOKEN_ARENA_BYTES: usize = 8192;
#[cfg(target_os = "macos")]
pub(crate) const CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY: usize = 48;
#[cfg(target_os = "macos")]
pub(crate) const CLASSIC_TIER1_PASS_PLAN_CAPACITY: usize = 48;
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kClassicEncodeParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) sub_band_type: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) style_flags: u32,
    pub(crate) output_capacity: u32,
    pub(crate) segment_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kClassicEncodeBatchJob {
    pub(crate) coefficient_offset: u32,
    pub(crate) output_offset: u32,
    pub(crate) segment_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) sub_band_type: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) style_flags: u32,
    pub(crate) output_capacity: u32,
    pub(crate) segment_capacity: u32,
}
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kClassicEncodeStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) data_len: u32,
    pub(crate) number_of_coding_passes: u32,
    pub(crate) missing_bit_planes: u32,
    pub(crate) segment_count: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kClassicTier1DensityCounters {
    pub(crate) sigprop_active_candidates: u32,
    pub(crate) sigprop_new_significant: u32,
    pub(crate) magref_active_candidates: u32,
    pub(crate) cleanup_active_candidates: u32,
    pub(crate) cleanup_new_significant: u32,
    pub(crate) cleanup_rlc_stripes: u32,
    pub(crate) cleanup_rlc_zero_stripes: u32,
    pub(crate) arithmetic_sigprop_active_candidates: u32,
    pub(crate) arithmetic_sigprop_new_significant: u32,
    pub(crate) raw_sigprop_active_candidates: u32,
    pub(crate) raw_sigprop_new_significant: u32,
    pub(crate) arithmetic_magref_active_candidates: u32,
    pub(crate) raw_magref_active_candidates: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kClassicTier1SymbolPlanCounters {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) coding_passes: u32,
    pub(crate) missing_bit_planes: u32,
    pub(crate) segment_count: u32,
    pub(crate) mq_symbol_count: u32,
    pub(crate) raw_bit_count: u32,
    pub(crate) cleanup_mq_symbol_count: u32,
    pub(crate) sigprop_mq_symbol_count: u32,
    pub(crate) magref_mq_symbol_count: u32,
    pub(crate) raw_sigprop_bit_count: u32,
    pub(crate) raw_magref_bit_count: u32,
    pub(crate) cleanup_sign_symbol_count: u32,
    pub(crate) sigprop_sign_symbol_count: u32,
    pub(crate) mq_symbol_hash: u32,
    pub(crate) raw_bit_hash: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kClassicTier1PassPlanCounters {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) coding_passes: u32,
    pub(crate) missing_bit_planes: u32,
    pub(crate) segment_count: u32,
    pub(crate) mq_symbol_count: u32,
    pub(crate) raw_bit_count: u32,
    pub(crate) nonempty_mq_passes: u32,
    pub(crate) nonempty_raw_passes: u32,
    pub(crate) max_mq_symbols_per_pass: u32,
    pub(crate) max_raw_bits_per_pass: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
    pub(crate) reserved2: u32,
    pub(crate) reserved3: u32,
    pub(crate) reserved4: u32,
    pub(crate) mq_symbols_by_pass: [u32; CLASSIC_TIER1_PASS_PLAN_CAPACITY],
    pub(crate) raw_bits_by_pass: [u32; CLASSIC_TIER1_PASS_PLAN_CAPACITY],
}

#[cfg(target_os = "macos")]
impl Default for J2kClassicTier1PassPlanCounters {
    fn default() -> Self {
        Self {
            code: 0,
            detail: 0,
            coding_passes: 0,
            missing_bit_planes: 0,
            segment_count: 0,
            mq_symbol_count: 0,
            raw_bit_count: 0,
            nonempty_mq_passes: 0,
            nonempty_raw_passes: 0,
            max_mq_symbols_per_pass: 0,
            max_raw_bits_per_pass: 0,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            reserved4: 0,
            mq_symbols_by_pass: [0; CLASSIC_TIER1_PASS_PLAN_CAPACITY],
            raw_bits_by_pass: [0; CLASSIC_TIER1_PASS_PLAN_CAPACITY],
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kClassicTier1TokenSegment {
    pub(crate) token_bit_offset: u32,
    pub(crate) token_bit_count: u32,
    pub(crate) pass_range: u32,
    pub(crate) flags: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kHtEncodeParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kHtEncodeBatchJob {
    pub(crate) coefficient_offset: u32,
    pub(crate) output_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) total_bitplanes: u32,
    pub(crate) output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kHtEncodeStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) data_len: u32,
    pub(crate) num_coding_passes: u32,
    pub(crate) num_zero_bitplanes: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
    pub(crate) reserved2: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kPacketEncodeParams {
    pub(crate) resolution_count: u32,
    pub(crate) num_layers: u32,
    pub(crate) num_components: u32,
    pub(crate) code_block_count: u32,
    pub(crate) subband_count: u32,
    pub(crate) descriptor_count: u32,
    pub(crate) output_capacity: u32,
    pub(crate) header_capacity: u32,
    pub(crate) scratch_node_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kBatchedPacketEncodeJob {
    pub(crate) resolution_offset: u32,
    pub(crate) subband_offset: u32,
    pub(crate) block_offset: u32,
    pub(crate) descriptor_offset: u32,
    pub(crate) state_block_offset: u32,
    pub(crate) output_offset: u32,
    pub(crate) header_offset: u32,
    pub(crate) scratch_offset: u32,
    pub(crate) payload_copy_offset: u32,
    pub(crate) payload_copy_capacity: u32,
    pub(crate) resolution_count: u32,
    pub(crate) num_layers: u32,
    pub(crate) num_components: u32,
    pub(crate) code_block_count: u32,
    pub(crate) subband_count: u32,
    pub(crate) descriptor_count: u32,
    pub(crate) output_capacity: u32,
    pub(crate) header_capacity: u32,
    pub(crate) scratch_node_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kPacketPayloadCopyJob {
    pub(crate) src_offset: u32,
    pub(crate) dst_offset: u32,
    pub(crate) byte_len: u32,
    pub(crate) reserved0: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kPacketPayloadCopyParams {
    pub(crate) bytes_per_thread: u32,
    pub(crate) stripes_per_job: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kPacketDescriptor {
    pub(crate) packet_index: u32,
    pub(crate) state_index: u32,
    pub(crate) layer: u32,
    pub(crate) resolution: u32,
    pub(crate) component: u32,
    pub(crate) precinct_lo: u32,
    pub(crate) precinct_hi: u32,
    pub(crate) state_block_offset: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kPacketResolution {
    pub(crate) subband_offset: u32,
    pub(crate) subband_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kPacketSubband {
    pub(crate) block_offset: u32,
    pub(crate) block_count: u32,
    pub(crate) num_cbs_x: u32,
    pub(crate) num_cbs_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kPacketBlock {
    pub(crate) data_offset: u32,
    pub(crate) data_len: u32,
    pub(crate) num_coding_passes: u32,
    pub(crate) num_zero_bitplanes: u32,
    pub(crate) previously_included: u32,
    pub(crate) l_block: u32,
    pub(crate) block_coding_mode: u32,
    pub(crate) reserved0: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kResidentPacketBlock {
    pub(crate) tier1_job_index: u32,
    pub(crate) previously_included: u32,
    pub(crate) l_block: u32,
    pub(crate) block_coding_mode: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kResidentPacketBlockParams {
    pub(crate) block_count: u32,
    pub(crate) tier1_job_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kPacketStateBlock {
    pub(crate) previously_included: u32,
    pub(crate) l_block: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kPacketEncodeStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) data_len: u32,
    pub(crate) reserved0: u32,
    pub(crate) payload_copy_bytes: u32,
    pub(crate) payload_copy_small_jobs: u32,
    pub(crate) payload_copy_medium_jobs: u32,
    pub(crate) payload_copy_large_jobs: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct J2kLosslessCodestreamAssemblyParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) num_components: u32,
    pub(crate) bit_depth: u32,
    pub(crate) signed_samples: u32,
    pub(crate) num_decomposition_levels: u32,
    pub(crate) use_mct: u32,
    pub(crate) guard_bits: u32,
    pub(crate) progression_order: u32,
    pub(crate) write_tlm: u32,
    pub(crate) high_throughput: u32,
    pub(crate) code_block_style: u32,
    pub(crate) code_block_width_exp: u32,
    pub(crate) code_block_height_exp: u32,
    pub(crate) output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kBatchedCodestreamAssemblyJob {
    pub(crate) tile_data_offset: u32,
    pub(crate) codestream_offset: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) num_components: u32,
    pub(crate) bit_depth: u32,
    pub(crate) signed_samples: u32,
    pub(crate) num_decomposition_levels: u32,
    pub(crate) use_mct: u32,
    pub(crate) guard_bits: u32,
    pub(crate) progression_order: u32,
    pub(crate) write_tlm: u32,
    pub(crate) high_throughput: u32,
    pub(crate) code_block_style: u32,
    pub(crate) code_block_width_exp: u32,
    pub(crate) code_block_height_exp: u32,
    pub(crate) output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct J2kCodestreamAssemblyStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) data_len: u32,
    pub(crate) reserved0: u32,
}
