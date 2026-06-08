// SPDX-License-Identifier: Apache-2.0

#[cfg(all(target_os = "macos", test))]
use std::cell::Cell;
#[cfg(all(target_os = "macos", test))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(target_os = "macos")]
use std::{
    cell::RefCell,
    collections::HashMap,
    mem::{size_of, size_of_val},
    sync::{
        atomic::{AtomicU64, Ordering as AtomicOrdering},
        Arc, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};

#[cfg(target_os = "macos")]
use metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    objc::{runtime::Sel, Message},
    Buffer, CommandBuffer, CommandBufferRef, CommandQueue, CompileOptions,
    ComputeCommandEncoderRef, ComputePipelineState, Device, MTLCommandQueue, MTLResourceOptions,
    MTLSize,
};
#[cfg(target_os = "macos")]
use rayon::prelude::*;
use signinum_core::{PixelFormat, Rect};
#[cfg(target_os = "macos")]
use signinum_j2k_native::HtCodeBlockDecoder;
use signinum_j2k_native::{
    decode_ht_code_block_scalar_with_workspace,
    decode_ht_code_block_scalar_with_workspace_profiled,
    decode_j2k_code_block_scalar_with_workspace,
    decode_j2k_code_block_scalar_with_workspace_profiled, ht_uvlc_encode_table, ht_uvlc_table0,
    ht_uvlc_table1, ht_vlc_encode_table0, ht_vlc_encode_table1, ht_vlc_table0, ht_vlc_table1,
    pack_j2k_code_block_scalar_from_tier1_tokens, ColorSpace as NativeColorSpace,
    DecodedComponents as NativeDecodedComponents, EncodeProgressionOrder, EncodedHtJ2kCodeBlock,
    EncodedJ2kCodeBlock, HtCodeBlockDecodeJob, HtCodeBlockDecodeProfile,
    HtCodeBlockDecodeWorkspace, HtSubBandDecodeJob, J2kCodeBlockDecodeJob,
    J2kCodeBlockDecodeProfile, J2kCodeBlockDecodeWorkspace, J2kCodeBlockSegment, J2kCodeBlockStyle,
    J2kDirectBandId, J2kDirectColorPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kDirectIdwtStep, J2kDirectStoreStep, J2kForwardDwt53Level, J2kForwardDwt53Output,
    J2kHtCodeBlockEncodeJob, J2kInverseMctJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor, J2kSingleDecompositionIdwtJob,
    J2kStoreComponentJob, J2kSubBandDecodeJob, J2kSubBandType, J2kTier1CodeBlockEncodeJob,
    J2kTier1TokenSegment, J2kWaveletTransform,
};
#[cfg(target_os = "macos")]
use signinum_j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};

#[cfg(target_os = "macos")]
use crate::{
    classic::MetalClassicBlockDecoder, ht::MetalHtBlockDecoder, idwt::MetalIdwtDecoder,
    mct::MetalMctDecoder, store::MetalStoreDecoder,
};
use crate::{Error, Surface};

#[cfg(all(target_os = "macos", test))]
static HT_BATCH_COEFFICIENT_COPY_BLITS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(target_os = "macos", test))]
static HYBRID_STACKED_COMPONENT_BATCHES: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(target_os = "macos", test))]
static HYBRID_REPEATED_OUTPUT_BLITS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(target_os = "macos", test))]
static HYBRID_CPU_DECODE_WORKER_INITS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(target_os = "macos", test))]
static HYBRID_CPU_DECODE_INPUTS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(target_os = "macos", test))]
static FLATTENED_HYBRID_CPU_DECODE_BATCHES: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(target_os = "macos", test))]
std::thread_local! {
    static RESIDENT_GPU_TIMESTAMP_QUERIES: Cell<usize> = const { Cell::new(0) };
    static RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS: Cell<usize> = const { Cell::new(0) };
    static DIRECT_TIER1_INPUT_BUFFER_PREPARES: Cell<usize> = const { Cell::new(0) };
    static PRIVATE_BUFFER_POOL_MISSES: Cell<usize> = const { Cell::new(0) };
    static SHARED_BUFFER_POOL_MISSES: Cell<usize> = const { Cell::new(0) };
    static LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES: Cell<usize> = const { Cell::new(0) };
    static CLASSIC_GPU_TOKEN_PACK_DISPATCHES: Cell<usize> = const { Cell::new(0) };
    static CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES: Cell<usize> = const { Cell::new(0) };
    static CLASSIC_GPU_TOKEN_PACK_ROUTE_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static METAL_PROFILE_STAGES_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_ht_batch_coefficient_copy_blits_for_test() {
    HT_BATCH_COEFFICIENT_COPY_BLITS.store(0, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn ht_batch_coefficient_copy_blits_for_test() -> usize {
    HT_BATCH_COEFFICIENT_COPY_BLITS.load(Ordering::Relaxed)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_resident_gpu_timestamp_queries_for_test() {
    RESIDENT_GPU_TIMESTAMP_QUERIES.with(|queries| queries.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn resident_gpu_timestamp_queries_for_test() -> usize {
    RESIDENT_GPU_TIMESTAMP_QUERIES.with(Cell::get)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_resident_codestream_command_buffer_waits_for_test() {
    RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS.with(|waits| waits.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn resident_codestream_command_buffer_waits_for_test() -> usize {
    RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS.with(Cell::get)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_direct_tier1_input_buffer_prepares_for_test() {
    DIRECT_TIER1_INPUT_BUFFER_PREPARES.with(|counter| counter.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn direct_tier1_input_buffer_prepares_for_test() -> usize {
    DIRECT_TIER1_INPUT_BUFFER_PREPARES.with(Cell::get)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_hybrid_stacked_component_batches_for_test() {
    HYBRID_STACKED_COMPONENT_BATCHES.store(0, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn hybrid_stacked_component_batches_for_test() -> usize {
    HYBRID_STACKED_COMPONENT_BATCHES.load(Ordering::Relaxed)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_hybrid_repeated_output_blits_for_test() {
    HYBRID_REPEATED_OUTPUT_BLITS.store(0, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn hybrid_repeated_output_blits_for_test() -> usize {
    HYBRID_REPEATED_OUTPUT_BLITS.load(Ordering::Relaxed)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_hybrid_cpu_decode_worker_inits_for_test() {
    HYBRID_CPU_DECODE_WORKER_INITS.store(0, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn hybrid_cpu_decode_worker_inits_for_test() -> usize {
    HYBRID_CPU_DECODE_WORKER_INITS.load(Ordering::Relaxed)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_hybrid_cpu_decode_inputs_for_test() {
    HYBRID_CPU_DECODE_INPUTS.store(0, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn hybrid_cpu_decode_inputs_for_test() -> usize {
    HYBRID_CPU_DECODE_INPUTS.load(Ordering::Relaxed)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_flattened_hybrid_cpu_decode_batches_for_test() {
    FLATTENED_HYBRID_CPU_DECODE_BATCHES.store(0, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn flattened_hybrid_cpu_decode_batches_for_test() -> usize {
    FLATTENED_HYBRID_CPU_DECODE_BATCHES.load(Ordering::Relaxed)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_private_buffer_pool_misses_for_test() {
    PRIVATE_BUFFER_POOL_MISSES.with(|misses| misses.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn private_buffer_pool_misses_for_test() -> usize {
    PRIVATE_BUFFER_POOL_MISSES.with(Cell::get)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_shared_buffer_pool_misses_for_test() {
    SHARED_BUFFER_POOL_MISSES.with(|misses| misses.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn shared_buffer_pool_misses_for_test() -> usize {
    SHARED_BUFFER_POOL_MISSES.with(Cell::get)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_lossless_deinterleave_rct_fused_dispatches_for_test() {
    LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES.with(|dispatches| dispatches.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn lossless_deinterleave_rct_fused_dispatches_for_test() -> usize {
    LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES.with(Cell::get)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_classic_gpu_token_pack_dispatches_for_test() {
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES.with(|dispatches| dispatches.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn classic_gpu_token_pack_dispatches_for_test() -> usize {
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES.with(Cell::get)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_classic_split_mq_byte_gpu_token_pack_dispatches_for_test() {
    CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES.with(|dispatches| dispatches.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn classic_split_mq_byte_gpu_token_pack_dispatches_for_test() -> usize {
    CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES.with(Cell::get)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) struct ClassicGpuTokenPackRouteOverrideGuard {
    previous: Option<bool>,
}

#[cfg(all(target_os = "macos", test))]
impl Drop for ClassicGpuTokenPackRouteOverrideGuard {
    fn drop(&mut self) {
        CLASSIC_GPU_TOKEN_PACK_ROUTE_OVERRIDE.with(|route| route.set(self.previous));
    }
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn force_classic_gpu_token_pack_route_for_test(
    enabled: bool,
) -> ClassicGpuTokenPackRouteOverrideGuard {
    let previous = CLASSIC_GPU_TOKEN_PACK_ROUTE_OVERRIDE.with(|route| route.replace(Some(enabled)));
    ClassicGpuTokenPackRouteOverrideGuard { previous }
}

#[cfg(all(target_os = "macos", test))]
pub(crate) struct MetalProfileStagesOverrideGuard {
    previous: Option<bool>,
}

#[cfg(all(target_os = "macos", test))]
impl Drop for MetalProfileStagesOverrideGuard {
    fn drop(&mut self) {
        METAL_PROFILE_STAGES_OVERRIDE.with(|profile| profile.set(self.previous));
    }
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn force_metal_profile_stages_for_test(
    enabled: bool,
) -> MetalProfileStagesOverrideGuard {
    let previous = METAL_PROFILE_STAGES_OVERRIDE.with(|profile| profile.replace(Some(enabled)));
    MetalProfileStagesOverrideGuard { previous }
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct MetalCodeBlockDecoder {
    classic: MetalClassicBlockDecoder,
    ht: MetalHtBlockDecoder,
    idwt: MetalIdwtDecoder,
    mct: MetalMctDecoder,
    store: MetalStoreDecoder,
}

#[cfg(target_os = "macos")]
impl HtCodeBlockDecoder for MetalCodeBlockDecoder {
    fn decode_j2k_sub_band(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<bool> {
        self.classic.decode_j2k_sub_band(job, output)
    }

    fn decode_j2k_code_block(
        &mut self,
        job: signinum_j2k_native::J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<bool> {
        self.classic.decode_j2k_code_block(job, output)
    }

    fn decode_sub_band(
        &mut self,
        job: HtSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<bool> {
        self.ht.decode_sub_band(job, output)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<()> {
        self.ht.decode_code_block(job, output)
    }

    fn decode_single_decomposition_idwt(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        output: &mut [f32],
    ) -> signinum_j2k_native::Result<bool> {
        self.idwt.decode_single_decomposition_idwt(job, output)
    }

    fn decode_inverse_mct(
        &mut self,
        job: J2kInverseMctJob<'_>,
    ) -> signinum_j2k_native::Result<bool> {
        self.mct.decode_inverse_mct(job)
    }

    fn decode_store_component(
        &mut self,
        job: J2kStoreComponentJob<'_>,
    ) -> signinum_j2k_native::Result<bool> {
        self.store.decode_store_component(job)
    }
}

#[cfg(target_os = "macos")]
const SHADER_SOURCE: &str = concat!(
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void j2k_zero_u32_buffer(
    device uint *buffer [[buffer(0)]],
    constant uint &word_count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= word_count) {
        return;
    }

    buffer[gid] = 0u;
}

struct J2kValidateBytesParams {
    uint byte_len;
};

struct J2kValidateBytesStatus {
    uint code;
    uint index;
    uint expected;
    uint actual;
};

kernel void j2k_validate_bytes_equal(
    device const uchar *actual [[buffer(0)]],
    device const uchar *expected [[buffer(1)]],
    device J2kValidateBytesStatus *status [[buffer(2)]],
    constant J2kValidateBytesParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    status[0].code = 0u;
    status[0].index = 0u;
    status[0].expected = 0u;
    status[0].actual = 0u;

    for (uint i = 0u; i < params.byte_len; ++i) {
        const uchar actual_byte = actual[i];
        const uchar expected_byte = expected[i];
        if (actual_byte != expected_byte) {
            status[0].code = 1u;
            status[0].index = i;
            status[0].expected = uint(expected_byte);
            status[0].actual = uint(actual_byte);
            return;
        }
    }
}

struct J2kCopyInterleavedParams {
    uint src_width;
    uint src_height;
    uint src_stride;
    uint dst_width;
    uint dst_height;
    uint dst_stride;
    uint bytes_per_pixel;
};

kernel void j2k_copy_interleaved_padded(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant J2kCopyInterleavedParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    const uint dst_idx = gid.y * params.dst_stride + gid.x * params.bytes_per_pixel;
    const bool inside_src = gid.x < params.src_width && gid.y < params.src_height;
    const uint src_idx = gid.y * params.src_stride + gid.x * params.bytes_per_pixel;
    for (uint byte_idx = 0u; byte_idx < params.bytes_per_pixel; ++byte_idx) {
        dst[dst_idx + byte_idx] = inside_src ? src[src_idx + byte_idx] : uchar(0);
    }
}

struct J2kLosslessDeinterleaveParams {
    uint src_width;
    uint src_height;
    uint src_stride;
    uint dst_width;
    uint dst_height;
    uint components;
    uint bytes_per_sample;
    uint sample_offset;
};

inline float j2k_lossless_load_sample(
    device const uchar *src,
    uint base,
    uint component,
    uint components,
    uint bytes_per_sample,
    uint sample_offset,
    bool inside_src
) {
    if (!inside_src) {
        return -float(int(sample_offset));
    }
    if (bytes_per_sample == 1u) {
        return float(int(src[base + component]) - int(sample_offset));
    }
    const uint byte_offset = base + component * 2u;
    const uint raw = uint(src[byte_offset]) | (uint(src[byte_offset + 1u]) << 8u);
    return float(int(raw) - int(sample_offset));
}

kernel void j2k_lossless_deinterleave_to_planes(
    device const uchar *src [[buffer(0)]],
    device float *plane0 [[buffer(1)]],
    device float *plane1 [[buffer(2)]],
    device float *plane2 [[buffer(3)]],
    constant J2kLosslessDeinterleaveParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    const bool inside_src = gid.x < params.src_width && gid.y < params.src_height;
    const uint src_base = gid.y * params.src_stride
        + gid.x * params.components * params.bytes_per_sample;
    const uint dst_idx = gid.y * params.dst_width + gid.x;
    plane0[dst_idx] = j2k_lossless_load_sample(
        src,
        src_base,
        0u,
        params.components,
        params.bytes_per_sample,
        params.sample_offset,
        inside_src
    );
    if (params.components >= 3u) {
        plane1[dst_idx] = j2k_lossless_load_sample(
            src,
            src_base,
            1u,
            params.components,
            params.bytes_per_sample,
            params.sample_offset,
            inside_src
        );
        plane2[dst_idx] = j2k_lossless_load_sample(
            src,
            src_base,
            2u,
            params.components,
            params.bytes_per_sample,
            params.sample_offset,
            inside_src
        );
    }
}

struct J2kLosslessCoefficientJob {
    uint coefficient_offset;
    uint component;
    uint subband_x;
    uint subband_y;
    uint block_x;
    uint block_y;
    uint block_width;
    uint block_height;
    uint full_width;
};

kernel void j2k_lossless_extract_coefficients(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device int *coefficients [[buffer(3)]],
    constant J2kLosslessCoefficientJob *jobs [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.z >= job_count) {
        return;
    }
    constant J2kLosslessCoefficientJob &job = jobs[gid.z];
    if (gid.x >= job.block_width || gid.y >= job.block_height) {
        return;
    }

    device const float *plane = plane0;
    if (job.component == 1u) {
        plane = plane1;
    } else if (job.component == 2u) {
        plane = plane2;
    }
    const uint src_x = job.subband_x + job.block_x + gid.x;
    const uint src_y = job.subband_y + job.block_y + gid.y;
    const uint src_idx = src_y * job.full_width + src_x;
    const uint dst_idx = job.coefficient_offset + gid.y * job.block_width + gid.x;
    coefficients[dst_idx] = int(round(plane[src_idx]));
}

struct J2kPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint output_channels;
    uint opaque_alpha;
    float max_values[4];
    float u8_scales[4];
    float u16_scales[4];
};

struct J2kMctRgb8PackParams {
    uint width;
    uint height;
    uint out_stride;
    uint transform;
    float addends[3];
    float max_values[3];
    float u8_scales[3];
};

struct J2kBatchedMctRgb8PackParams {
    uint width;
    uint height;
    uint out_stride;
    uint transform;
    uint batch_count;
    uint plane_stride;
    uint output_stride;
    float addends[3];
    float max_values[3];
    float u8_scales[3];
};

inline uchar scale_to_u8(float sample, float max_value, float scale) {
    const float clamped = clamp(sample, 0.0f, max_value);
    return uchar(min(floor(clamped * scale + 0.5f), 255.0f));
}

inline ushort pack_to_u16(float sample, float max_value, float scale) {
    const float clamped = clamp(sample, 0.0f, max_value);
    return ushort(min(floor(clamped * scale + 0.5f), 65535.0f));
}

kernel void j2k_pack_gray8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = gid.y * params.out_stride + gid.x;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
}

kernel void j2k_pack_rgb8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);
}

kernel void j2k_pack_mct_rgb8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant J2kMctRgb8PackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const float y0 = plane0[idx];
    const float y1 = plane1[idx];
    const float y2 = plane2[idx];
    float rgb0;
    float rgb1;
    float rgb2;

    if (params.transform == 0u) {
        const float i1 = y0 - floor((y2 + y1) * 0.25f);
        rgb0 = y2 + i1 + params.addends[0];
        rgb1 = i1 + params.addends[1];
        rgb2 = y1 + i1 + params.addends[2];
    } else {
        rgb0 = y2 * 1.402f + y0 + params.addends[0];
        rgb1 = y2 * -0.71414f + y1 * -0.34413f + y0 + params.addends[1];
        rgb2 = y1 * 1.772f + y0 + params.addends[2];
    }

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = scale_to_u8(rgb0, params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(rgb1, params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(rgb2, params.max_values[2], params.u8_scales[2]);
}

kernel void j2k_pack_mct_rgb8_batched(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant J2kBatchedMctRgb8PackParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint plane_base = gid.z * params.plane_stride;
    const uint idx = plane_base + gid.y * params.width + gid.x;
    const float y0 = plane0[idx];
    const float y1 = plane1[idx];
    const float y2 = plane2[idx];
    float rgb0;
    float rgb1;
    float rgb2;

    if (params.transform == 0u) {
        const float i1 = y0 - floor((y2 + y1) * 0.25f);
        rgb0 = y2 + i1 + params.addends[0];
        rgb1 = i1 + params.addends[1];
        rgb2 = y1 + i1 + params.addends[2];
    } else {
        rgb0 = y2 * 1.402f + y0 + params.addends[0];
        rgb1 = y2 * -0.71414f + y1 * -0.34413f + y0 + params.addends[1];
        rgb2 = y1 * 1.772f + y0 + params.addends[2];
    }

    const uint out_idx = gid.z * params.output_stride + gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = scale_to_u8(rgb0, params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(rgb1, params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(rgb2, params.max_values[2], params.u8_scales[2]);
}

kernel void j2k_pack_rgb_opaque_rgba8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);
    out[out_idx + 3] = uchar(255);
}

kernel void j2k_pack_rgba8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);
    out[out_idx + 3] = scale_to_u8(plane3[idx], params.max_values[3], params.u8_scales[3]);
}

kernel void j2k_pack_gray16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device ushort *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = (gid.y * params.out_stride) / 2u + gid.x;
    out[out_idx] = pack_to_u16(plane0[idx], params.max_values[0], params.u16_scales[0]);
}

kernel void j2k_pack_rgb16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device ushort *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = (gid.y * params.out_stride) / 2u + gid.x * 3u;
    out[out_idx] = pack_to_u16(plane0[idx], params.max_values[0], params.u16_scales[0]);
    out[out_idx + 1] = pack_to_u16(plane1[idx], params.max_values[1], params.u16_scales[1]);
    out[out_idx + 2] = pack_to_u16(plane2[idx], params.max_values[2], params.u16_scales[2]);
}

struct J2kRepeatedGrayPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint batch_count;
    float max_value;
    float u8_scale;
    float u16_scale;
};

kernel void j2k_pack_u8_repeated_gray(
    device const float *plane0 [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant J2kRepeatedGrayPackParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint plane_base = gid.z * params.width * params.height;
    const uint out_base = gid.z * params.out_stride * params.height;
    const uint plane_idx = plane_base + gid.y * params.width + gid.x;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x;
    out[out_idx] = scale_to_u8(plane0[plane_idx], params.max_value, params.u8_scale);
}

kernel void j2k_pack_u16_repeated_gray(
    device const float *plane0 [[buffer(0)]],
    device ushort *out [[buffer(1)]],
    constant J2kRepeatedGrayPackParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint plane_base = gid.z * params.width * params.height;
    const uint out_base = (gid.z * params.out_stride * params.height) / 2u;
    const uint plane_idx = plane_base + gid.y * params.width + gid.x;
    const uint out_idx = out_base + gid.y * (params.out_stride / 2u) + gid.x;
    out[out_idx] = pack_to_u16(plane0[plane_idx], params.max_value, params.u16_scale);
}
"#,
    "\n",
    include_str!("classic.metal"),
    "\n",
    include_str!("encode_bitstream.metal"),
    "\n",
    include_str!("idwt.metal"),
    "\n",
    include_str!("fdwt.metal"),
    "\n",
    include_str!("mct.metal"),
    "\n",
    include_str!("store.metal"),
    "\n",
    include_str!("ht_cleanup.metal"),
);

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kValidateBytesParams {
    byte_len: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kValidateBytesStatus {
    code: u32,
    index: u32,
    expected: u32,
    actual: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kCopyInterleavedParams {
    src_width: u32,
    src_height: u32,
    src_stride: u32,
    dst_width: u32,
    dst_height: u32,
    dst_stride: u32,
    bytes_per_pixel: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kLosslessDeinterleaveParams {
    src_width: u32,
    src_height: u32,
    src_stride: u32,
    dst_width: u32,
    dst_height: u32,
    components: u32,
    bytes_per_sample: u32,
    sample_offset: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kLosslessCoefficientJob {
    coefficient_offset: u32,
    component: u32,
    subband_x: u32,
    subband_y: u32,
    block_x: u32,
    block_y: u32,
    block_width: u32,
    block_height: u32,
    full_width: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    output_channels: u32,
    opaque_alpha: u32,
    max_values: [f32; 4],
    u8_scales: [f32; 4],
    u16_scales: [f32; 4],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kMctRgb8PackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    transform: u32,
    addends: [f32; 3],
    max_values: [f32; 3],
    u8_scales: [f32; 3],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kBatchedMctRgb8PackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    transform: u32,
    batch_count: u32,
    plane_stride: u32,
    output_stride: u32,
    addends: [f32; 3],
    max_values: [f32; 3],
    u8_scales: [f32; 3],
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kRepeatedGrayPackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    batch_count: u32,
    max_value: f32,
    u8_scale: f32,
    u16_scale: f32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct J2kScalarPackParams {
    max_value: f32,
    u8_scale: f32,
    u16_scale: f32,
}

#[cfg(target_os = "macos")]
fn j2k_scalar_pack_params(bit_depth: u32) -> J2kScalarPackParams {
    let clamped = bit_depth.min(16);
    let max_value_u16 = u16::try_from(((1u32 << clamped) - 1).max(1))
        .expect("clamped J2K bit depth max fits in u16");
    let max_value = f32::from(max_value_u16);
    let u8_scale = 255.0 / max_value;
    let u16_scale = if bit_depth <= 8 {
        65_535.0 / max_value
    } else {
        1.0
    };
    J2kScalarPackParams {
        max_value,
        u8_scale,
        u16_scale,
    }
}

#[cfg(target_os = "macos")]
fn j2k_pack_scale_arrays(bit_depths: [u32; 4]) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let mut max_values = [1.0f32; 4];
    let mut u8_scales = [255.0f32; 4];
    let mut u16_scales = [65_535.0f32; 4];
    for (index, bit_depth) in bit_depths.into_iter().enumerate() {
        let params = j2k_scalar_pack_params(bit_depth);
        max_values[index] = params.max_value;
        u8_scales[index] = params.u8_scale;
        u16_scales[index] = params.u16_scale;
    }
    (max_values, u8_scales, u16_scales)
}

#[cfg(target_os = "macos")]
const J2K_CLASSIC_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STATUS_UNSUPPORTED: u32 = 2;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES: u32 = 1 << 0;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS: u32 = 1 << 1;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT: u32 = 1 << 2;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS: u32 = 1 << 3;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS: u32 = 1 << 4;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_MAX_WIDTH: u32 = 64;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_MAX_HEIGHT: u32 = 64;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_MAX_COEFF_COUNT: usize =
    (J2K_CLASSIC_MAX_WIDTH as usize + 2) * (J2K_CLASSIC_MAX_HEIGHT as usize + 2);
#[cfg(target_os = "macos")]
const J2K_CLASSIC_ENCODE_32_MAX_WIDTH: u32 = 32;
#[cfg(target_os = "macos")]
const J2K_CLASSIC_ENCODE_32_MAX_HEIGHT: u32 = 32;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicCleanupBatchJob {
    coded_offset: u32,
    coded_len: u32,
    segment_offset: u32,
    segment_count: u32,
    width: u32,
    height: u32,
    output_stride: u32,
    output_offset: u32,
    missing_msbs: u32,
    total_bitplanes: u32,
    roi_shift: u32,
    number_of_coding_passes: u32,
    sub_band_type: u32,
    style_flags: u32,
    strict: u32,
    dequantization_step: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicSegment {
    data_offset: u32,
    data_length: u32,
    start_coding_pass: u32,
    end_coding_pass: u32,
    use_arithmetic: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kClassicStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
const J2K_IDWT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_IDWT_STATUS_FAIL: u32 = 1;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kIdwtSingleDecompositionParams {
    x0: u32,
    y0: u32,
    output_x: u32,
    output_y: u32,
    width: u32,
    height: u32,
    ll_x: u32,
    ll_y: u32,
    ll_width: u32,
    ll_height: u32,
    hl_x: u32,
    hl_y: u32,
    hl_width: u32,
    hl_height: u32,
    lh_x: u32,
    lh_y: u32,
    lh_width: u32,
    lh_height: u32,
    hh_x: u32,
    hh_y: u32,
    hh_width: u32,
    hh_height: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kRepeatedIdwtSingleDecompositionParams {
    x0: u32,
    y0: u32,
    output_x: u32,
    output_y: u32,
    width: u32,
    height: u32,
    ll_x: u32,
    ll_y: u32,
    ll_width: u32,
    ll_height: u32,
    hl_x: u32,
    hl_y: u32,
    hl_width: u32,
    hl_height: u32,
    lh_x: u32,
    lh_y: u32,
    lh_width: u32,
    lh_height: u32,
    hh_x: u32,
    hh_y: u32,
    hh_width: u32,
    hh_height: u32,
    ll_instance_stride: u32,
    hl_instance_stride: u32,
    lh_instance_stride: u32,
    hh_instance_stride: u32,
    batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kIdwtStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
const J2K_MCT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_MCT_STATUS_FAIL: u32 = 1;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kInverseMctParams {
    _len: u32,
    _transform: u32,
    _addend0: f32,
    _addend1: f32,
    _addend2: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kForwardRctParams {
    _len: u32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kForwardDwt53Params {
    full_width: u32,
    current_width: u32,
    current_height: u32,
    low_width: u32,
    low_height: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kForwardDwt53BatchedParams {
    full_width: u32,
    current_width: u32,
    current_height: u32,
    low_width: u32,
    low_height: u32,
    component_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kMctStatus {
    code: u32,
    detail: u32,
    _reserved0: u32,
    _reserved1: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kStoreParams {
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kRepeatedStoreParams {
    input_width: u32,
    input_height: u32,
    input_instance_stride: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
    batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kRepeatedGrayStoreParams {
    input_width: u32,
    input_height: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
    batch_count: u32,
    max_value: f32,
    u8_scale: f32,
    u16_scale: f32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kGrayStoreParams {
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
    output_width: u32,
    output_x: u32,
    output_y: u32,
    addend: f32,
    max_value: f32,
    u8_scale: f32,
    u16_scale: f32,
}

const J2K_HT_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_HT_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
const J2K_HT_STATUS_UNSUPPORTED: u32 = 2;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupParams {
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtCleanupBatchJob {
    coded_offset: u32,
    width: u32,
    height: u32,
    coded_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    missing_msbs: u32,
    num_bitplanes: u32,
    roi_shift: u32,
    number_of_coding_passes: u32,
    output_stride: u32,
    output_offset: u32,
    dequantization_step: f32,
    stripe_causal: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtRepeatedBatchParams {
    job_count: u32,
    output_plane_len: u32,
    batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicRepeatedBatchParams {
    job_count: u32,
    output_plane_len: u32,
    batch_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kHtStatus {
    code: u32,
    detail: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
const J2K_ENCODE_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const J2K_ENCODE_STATUS_FAIL: u32 = 1;
#[cfg(target_os = "macos")]
const J2K_ENCODE_STATUS_UNSUPPORTED: u32 = 2;
#[cfg(target_os = "macos")]
const J2K_HT_ENCODE_MEL_SIZE: usize = 192;
#[cfg(target_os = "macos")]
const J2K_HT_ENCODE_VLC_SIZE: usize = 3072 - J2K_HT_ENCODE_MEL_SIZE;
#[cfg(target_os = "macos")]
const J2K_HT_ENCODE_MS_SIZE: usize = (16_384usize * 16).div_ceil(15);
#[cfg(target_os = "macos")]
const J2K_HT_ENCODE_BASE_OUTPUT_SIZE: usize =
    J2K_HT_ENCODE_MS_SIZE + J2K_HT_ENCODE_MEL_SIZE + J2K_HT_ENCODE_VLC_SIZE;
#[cfg(target_os = "macos")]
const J2K_HT_ENCODE_MAX_SAMPLES: usize = 16_384;
#[cfg(target_os = "macos")]
const J2K_HT_ENCODE_MS_BYTES_PER_SAMPLE_FLOOR: usize = 5;
#[cfg(target_os = "macos")]
const PACKET_PAYLOAD_COPY_BYTES_PER_STRIPE: u32 = 256;
#[cfg(target_os = "macos")]
const PACKET_PAYLOAD_COPY_STRIPES_PER_JOB: u32 = 4;

#[cfg(target_os = "macos")]
const HT_PACKET_CAPACITY_ENV: &str = "SIGNINUM_J2K_METAL_HT_PACKET_CAPACITY";
#[cfg(target_os = "macos")]
const CLASSIC_SELECTIVE_BYPASS_ENV: &str = "SIGNINUM_J2K_METAL_CLASSIC_SELECTIVE_BYPASS";
const METAL_PROFILE_STAGES_ENV: &str = "SIGNINUM_J2K_METAL_PROFILE_STAGES";
#[cfg(target_os = "macos")]
const METAL_PROFILE_SIGNPOSTS_ENV: &str = "SIGNINUM_J2K_METAL_PROFILE_SIGNPOSTS";
#[cfg(target_os = "macos")]
const METAL_PROFILE_DECODE_LABEL_ENV: &str = "SIGNINUM_J2K_METAL_PROFILE_DECODE_LABEL";
#[cfg(target_os = "macos")]
const METAL_PROFILE_DECODE_SPLIT_COMMANDS_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_DECODE_SPLIT_COMMANDS";
#[cfg(target_os = "macos")]
const METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS";
#[cfg(target_os = "macos")]
const METAL_PROFILE_CLASSIC_TIER1_DENSITY_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_DENSITY";
#[cfg(target_os = "macos")]
const METAL_PROFILE_CLASSIC_TIER1_RAW_PACK_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_RAW_PACK";
#[cfg(target_os = "macos")]
const METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK";
#[cfg(target_os = "macos")]
const METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN";
#[cfg(target_os = "macos")]
const METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN";
#[cfg(target_os = "macos")]
const METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT";
#[cfg(target_os = "macos")]
const METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT";
#[cfg(target_os = "macos")]
const METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK_ENV: &str =
    "SIGNINUM_J2K_METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK";
#[cfg(target_os = "macos")]
const CLASSIC_TIER1_GPU_TOKEN_PACK_ENV: &str = "SIGNINUM_J2K_METAL_CLASSIC_TIER1_GPU_TOKEN_PACK";
#[cfg(target_os = "macos")]
const CLASSIC_TIER1_SPLIT_GPU_TOKEN_PACK_ENV: &str =
    "SIGNINUM_J2K_METAL_CLASSIC_TIER1_SPLIT_GPU_TOKEN_PACK";
#[cfg(target_os = "macos")]
const CLASSIC_TIER1_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_ENV: &str =
    "SIGNINUM_J2K_METAL_CLASSIC_TIER1_SPLIT_MQ_BYTE_GPU_TOKEN_PACK";
#[cfg(target_os = "macos")]
const CLASSIC_TIER1_TOKEN_ARENA_BYTES: usize = 4096;
#[cfg(target_os = "macos")]
const CLASSIC_TIER1_MQ_BYTE_TOKEN_ARENA_BYTES: usize = 8192;
#[cfg(target_os = "macos")]
const CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY: usize = 48;
#[cfg(target_os = "macos")]
const CLASSIC_TIER1_PASS_PLAN_CAPACITY: usize = 48;
#[cfg(target_os = "macos")]
type HybridSignpostName = u32;
#[cfg(target_os = "macos")]
const SIGNPOST_DECODE_HYBRID_CPU_TIER1: HybridSignpostName = 1;
#[cfg(target_os = "macos")]
const SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD: HybridSignpostName = 2;
#[cfg(target_os = "macos")]
const SIGNPOST_DECODE_HYBRID_COMMAND_WAIT: HybridSignpostName = 3;
#[cfg(target_os = "macos")]
const SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE: HybridSignpostName = 4;
#[cfg(target_os = "macos")]
const SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE: HybridSignpostName = 5;
#[cfg(target_os = "macos")]
const SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE: HybridSignpostName = 6;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT: HybridSignpostName = 7;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST: HybridSignpostName = 8;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP: HybridSignpostName = 9;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE: HybridSignpostName = 10;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN: HybridSignpostName = 11;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP: HybridSignpostName = 12;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE: HybridSignpostName = 13;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE: HybridSignpostName = 14;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE: HybridSignpostName = 15;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP: HybridSignpostName = 16;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE: HybridSignpostName = 17;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN: HybridSignpostName = 18;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP: HybridSignpostName = 19;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE: HybridSignpostName = 20;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE: HybridSignpostName = 21;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE: HybridSignpostName = 22;
#[cfg(target_os = "macos")]
const SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE: HybridSignpostName = 23;
#[cfg(target_os = "macos")]
fn env_flag_enabled(name: &str) -> bool {
    matches!(std::env::var(name), Ok(value) if value == "1")
}

#[cfg(target_os = "macos")]
fn classic_resident_style_flags_from_env() -> u32 {
    if matches!(std::env::var(CLASSIC_SELECTIVE_BYPASS_ENV), Ok(value) if value == "0") {
        0
    } else {
        J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS
    }
}

fn classic_cod_block_style_from_flags(flags: u32) -> u32 {
    let mut style = 0u32;
    if (flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0 {
        style |= 0x01;
    }
    if (flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0 {
        style |= 0x02;
    }
    if (flags & J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS) != 0 {
        style |= 0x04;
    }
    if (flags & J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT) != 0 {
        style |= 0x08;
    }
    if (flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0 {
        style |= 0x20;
    }
    style
}

#[cfg(target_os = "macos")]
fn classic_tier1_gpu_token_pack_requested() -> bool {
    #[cfg(test)]
    if let Some(enabled) = CLASSIC_GPU_TOKEN_PACK_ROUTE_OVERRIDE.with(Cell::get) {
        return enabled;
    }
    env_flag_enabled(CLASSIC_TIER1_GPU_TOKEN_PACK_ENV)
}

#[cfg(target_os = "macos")]
fn classic_tier1_split_gpu_token_pack_requested() -> bool {
    env_flag_enabled(CLASSIC_TIER1_SPLIT_GPU_TOKEN_PACK_ENV)
}

#[cfg(target_os = "macos")]
fn classic_tier1_split_mq_byte_gpu_token_pack_setting() -> Option<bool> {
    match std::env::var(CLASSIC_TIER1_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_ENV) {
        Ok(value) if value == "1" => Some(true),
        Ok(value) if value == "0" => Some(false),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn classic_tier1_split_mq_byte_gpu_token_pack_requested() -> bool {
    classic_tier1_split_mq_byte_gpu_token_pack_setting() == Some(true)
}

#[cfg(target_os = "macos")]
fn classic_tier1_split_mq_byte_gpu_token_pack_disabled() -> bool {
    classic_tier1_split_mq_byte_gpu_token_pack_setting() == Some(false)
}

fn classic_tier1_gpu_token_pack_supported(jobs: &[J2kClassicEncodeBatchJob]) -> bool {
    !jobs.is_empty()
        && classic_encode_code_blocks_pipeline_kind(jobs)
            == J2kClassicEncodePipelineKind::BypassU16_32
}

#[cfg(target_os = "macos")]
pub(crate) fn metal_profile_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    #[cfg(test)]
    if let Some(enabled) = METAL_PROFILE_STAGES_OVERRIDE.with(Cell::get) {
        return enabled;
    }
    *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_STAGES_ENV))
}

#[cfg(target_os = "macos")]
fn metal_profile_signposts_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_SIGNPOSTS_ENV))
}

#[cfg(target_os = "macos")]
fn metal_profile_decode_split_commands_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_DECODE_SPLIT_COMMANDS_ENV))
}

#[cfg(target_os = "macos")]
fn metal_profile_coefficient_prep_split_commands_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED
            .get_or_init(|| env_flag_enabled(METAL_PROFILE_COEFFICIENT_PREP_SPLIT_COMMANDS_ENV))
}

#[cfg(target_os = "macos")]
fn metal_profile_classic_tier1_density_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_DENSITY_ENV))
}

#[cfg(target_os = "macos")]
fn metal_profile_classic_tier1_raw_pack_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_RAW_PACK_ENV))
}

#[cfg(target_os = "macos")]
fn metal_profile_classic_tier1_arithmetic_pack_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED
            .get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_ARITHMETIC_PACK_ENV))
}

#[cfg(target_os = "macos")]
fn metal_profile_classic_tier1_symbol_plan_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| {
            env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_SYMBOL_PLAN_ENV)
                || env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN_ENV)
        })
}

#[cfg(target_os = "macos")]
fn metal_profile_classic_tier1_pass_plan_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_PASS_PLAN_ENV))
}

#[cfg(target_os = "macos")]
fn metal_profile_classic_tier1_token_emit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| {
            env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_TOKEN_EMIT_ENV)
                || env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK_ENV)
        })
}

#[cfg(target_os = "macos")]
fn metal_profile_classic_tier1_split_token_emit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED
            .get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_SPLIT_TOKEN_EMIT_ENV))
}

#[cfg(target_os = "macos")]
fn metal_profile_classic_tier1_token_pack_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    metal_profile_stages_enabled()
        && *ENABLED.get_or_init(|| env_flag_enabled(METAL_PROFILE_CLASSIC_TIER1_TOKEN_PACK_ENV))
}

#[cfg(target_os = "macos")]
fn decode_profile_label() -> String {
    std::env::var(METAL_PROFILE_DECODE_LABEL_ENV)
        .ok()
        .filter(|label| !label.is_empty())
        .map_or_else(
            || "unlabeled".to_string(),
            |label| sanitize_profile_label(&label),
        )
}

#[cfg(target_os = "macos")]
fn sanitize_profile_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn label_command_buffer(command_buffer: &CommandBufferRef, label: &str) {
    if metal_profile_stages_enabled() {
        command_buffer.set_label(label);
    }
}

#[cfg(target_os = "macos")]
type OsSignpostId = u64;

#[cfg(target_os = "macos")]
const OS_SIGNPOST_ID_NULL: OsSignpostId = 0;
#[cfg(target_os = "macos")]
const OS_SIGNPOST_ID_INVALID: OsSignpostId = OsSignpostId::MAX;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn signinum_j2k_metal_signpost_begin(name: HybridSignpostName) -> OsSignpostId;
    fn signinum_j2k_metal_signpost_end(name: HybridSignpostName, id: OsSignpostId);
}

#[cfg(target_os = "macos")]
struct HybridStageSignpost {
    id: OsSignpostId,
    name: HybridSignpostName,
}

#[cfg(target_os = "macos")]
impl Drop for HybridStageSignpost {
    fn drop(&mut self) {
        unsafe {
            signinum_j2k_metal_signpost_end(self.name, self.id);
        }
    }
}

#[cfg(target_os = "macos")]
fn hybrid_stage_signpost(name: HybridSignpostName) -> Option<HybridStageSignpost> {
    if !metal_profile_signposts_enabled() {
        return None;
    }
    let id = unsafe { signinum_j2k_metal_signpost_begin(name) };
    if id == OS_SIGNPOST_ID_NULL || id == OS_SIGNPOST_ID_INVALID {
        return None;
    }
    Some(HybridStageSignpost { id, name })
}

#[cfg(target_os = "macos")]
fn label_compute_encoder(encoder: &ComputeCommandEncoderRef, label: &str) {
    if metal_profile_stages_enabled() {
        encoder.set_label(label);
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicEncodeParams {
    width: u32,
    height: u32,
    sub_band_type: u32,
    total_bitplanes: u32,
    style_flags: u32,
    output_capacity: u32,
    segment_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kClassicEncodeBatchJob {
    coefficient_offset: u32,
    output_offset: u32,
    segment_offset: u32,
    width: u32,
    height: u32,
    sub_band_type: u32,
    total_bitplanes: u32,
    style_flags: u32,
    output_capacity: u32,
    segment_capacity: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum J2kClassicEncodePipelineKind {
    Generic,
    Generic32,
    Bypass32,
    BypassU16_32,
    Style0,
    Style0_32,
}

#[cfg(target_os = "macos")]
fn classic_encode_code_blocks_pipeline_kind(
    jobs: &[J2kClassicEncodeBatchJob],
) -> J2kClassicEncodePipelineKind {
    let all_32 = jobs.iter().all(|job| {
        job.width <= J2K_CLASSIC_ENCODE_32_MAX_WIDTH
            && job.height <= J2K_CLASSIC_ENCODE_32_MAX_HEIGHT
    });
    let all_style0 = jobs.iter().all(|job| job.style_flags == 0);
    let all_bypass = jobs
        .iter()
        .all(|job| job.style_flags == J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS);
    let all_u16_bitplanes = jobs.iter().all(|job| job.total_bitplanes <= 16);
    match (all_style0, all_bypass, all_32, all_u16_bitplanes) {
        (true, _, true, _) => J2kClassicEncodePipelineKind::Style0_32,
        (true, _, false, _) => J2kClassicEncodePipelineKind::Style0,
        (false, true, true, true) => J2kClassicEncodePipelineKind::BypassU16_32,
        (false, true, true, false) => J2kClassicEncodePipelineKind::Bypass32,
        (false, _, true, _) => J2kClassicEncodePipelineKind::Generic32,
        (false, _, false, _) => J2kClassicEncodePipelineKind::Generic,
    }
}

#[cfg(target_os = "macos")]
fn classic_encode_code_blocks_pipeline<'a>(
    runtime: &'a MetalRuntime,
    jobs: &[J2kClassicEncodeBatchJob],
) -> &'a ComputePipelineState {
    match classic_encode_code_blocks_pipeline_kind(jobs) {
        J2kClassicEncodePipelineKind::Generic => &runtime.classic_encode_code_blocks,
        J2kClassicEncodePipelineKind::Generic32 => &runtime.classic_encode_code_blocks_32,
        J2kClassicEncodePipelineKind::Bypass32 => &runtime.classic_encode_code_blocks_bypass_32,
        J2kClassicEncodePipelineKind::BypassU16_32 => {
            &runtime.classic_encode_code_blocks_bypass_u16_32
        }
        J2kClassicEncodePipelineKind::Style0 => &runtime.classic_encode_code_blocks_style0,
        J2kClassicEncodePipelineKind::Style0_32 => &runtime.classic_encode_code_blocks_style0_32,
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kClassicEncodeStatus {
    code: u32,
    detail: u32,
    data_len: u32,
    number_of_coding_passes: u32,
    missing_bit_planes: u32,
    segment_count: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kClassicTier1DensityCounters {
    sigprop_active_candidates: u32,
    sigprop_new_significant: u32,
    magref_active_candidates: u32,
    cleanup_active_candidates: u32,
    cleanup_new_significant: u32,
    cleanup_rlc_stripes: u32,
    cleanup_rlc_zero_stripes: u32,
    arithmetic_sigprop_active_candidates: u32,
    arithmetic_sigprop_new_significant: u32,
    raw_sigprop_active_candidates: u32,
    raw_sigprop_new_significant: u32,
    arithmetic_magref_active_candidates: u32,
    raw_magref_active_candidates: u32,
    reserved0: u32,
    reserved1: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kClassicTier1SymbolPlanCounters {
    code: u32,
    detail: u32,
    coding_passes: u32,
    missing_bit_planes: u32,
    segment_count: u32,
    mq_symbol_count: u32,
    raw_bit_count: u32,
    cleanup_mq_symbol_count: u32,
    sigprop_mq_symbol_count: u32,
    magref_mq_symbol_count: u32,
    raw_sigprop_bit_count: u32,
    raw_magref_bit_count: u32,
    cleanup_sign_symbol_count: u32,
    sigprop_sign_symbol_count: u32,
    mq_symbol_hash: u32,
    raw_bit_hash: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kClassicTier1PassPlanCounters {
    code: u32,
    detail: u32,
    coding_passes: u32,
    missing_bit_planes: u32,
    segment_count: u32,
    mq_symbol_count: u32,
    raw_bit_count: u32,
    nonempty_mq_passes: u32,
    nonempty_raw_passes: u32,
    max_mq_symbols_per_pass: u32,
    max_raw_bits_per_pass: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
    reserved3: u32,
    reserved4: u32,
    mq_symbols_by_pass: [u32; CLASSIC_TIER1_PASS_PLAN_CAPACITY],
    raw_bits_by_pass: [u32; CLASSIC_TIER1_PASS_PLAN_CAPACITY],
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
struct J2kClassicTier1TokenSegment {
    token_bit_offset: u32,
    token_bit_count: u32,
    pass_range: u32,
    flags: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtEncodeParams {
    width: u32,
    height: u32,
    total_bitplanes: u32,
    output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kHtEncodeBatchJob {
    coefficient_offset: u32,
    output_offset: u32,
    width: u32,
    height: u32,
    total_bitplanes: u32,
    output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kHtEncodeStatus {
    code: u32,
    detail: u32,
    data_len: u32,
    num_coding_passes: u32,
    num_zero_bitplanes: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketEncodeParams {
    resolution_count: u32,
    num_layers: u32,
    num_components: u32,
    code_block_count: u32,
    subband_count: u32,
    descriptor_count: u32,
    output_capacity: u32,
    header_capacity: u32,
    scratch_node_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kBatchedPacketEncodeJob {
    resolution_offset: u32,
    subband_offset: u32,
    block_offset: u32,
    descriptor_offset: u32,
    state_block_offset: u32,
    output_offset: u32,
    header_offset: u32,
    scratch_offset: u32,
    payload_copy_offset: u32,
    payload_copy_capacity: u32,
    resolution_count: u32,
    num_layers: u32,
    num_components: u32,
    code_block_count: u32,
    subband_count: u32,
    descriptor_count: u32,
    output_capacity: u32,
    header_capacity: u32,
    scratch_node_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketPayloadCopyJob {
    src_offset: u32,
    dst_offset: u32,
    byte_len: u32,
    reserved0: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kPacketPayloadCopyParams {
    bytes_per_thread: u32,
    stripes_per_job: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketDescriptor {
    packet_index: u32,
    state_index: u32,
    layer: u32,
    resolution: u32,
    component: u32,
    precinct_lo: u32,
    precinct_hi: u32,
    state_block_offset: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketResolution {
    subband_offset: u32,
    subband_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketSubband {
    block_offset: u32,
    block_count: u32,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kPacketBlock {
    data_offset: u32,
    data_len: u32,
    num_coding_passes: u32,
    num_zero_bitplanes: u32,
    previously_included: u32,
    l_block: u32,
    block_coding_mode: u32,
    reserved0: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kResidentPacketBlock {
    tier1_job_index: u32,
    previously_included: u32,
    l_block: u32,
    block_coding_mode: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kResidentPacketBlockParams {
    block_count: u32,
    tier1_job_count: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kPacketStateBlock {
    previously_included: u32,
    l_block: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kPacketEncodeStatus {
    code: u32,
    detail: u32,
    data_len: u32,
    reserved0: u32,
    payload_copy_bytes: u32,
    payload_copy_small_jobs: u32,
    payload_copy_medium_jobs: u32,
    payload_copy_large_jobs: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct J2kLosslessCodestreamAssemblyParams {
    width: u32,
    height: u32,
    num_components: u32,
    bit_depth: u32,
    signed_samples: u32,
    num_decomposition_levels: u32,
    use_mct: u32,
    guard_bits: u32,
    progression_order: u32,
    write_tlm: u32,
    high_throughput: u32,
    code_block_style: u32,
    code_block_width_exp: u32,
    code_block_height_exp: u32,
    output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kBatchedCodestreamAssemblyJob {
    tile_data_offset: u32,
    codestream_offset: u32,
    width: u32,
    height: u32,
    num_components: u32,
    bit_depth: u32,
    signed_samples: u32,
    num_decomposition_levels: u32,
    use_mct: u32,
    guard_bits: u32,
    progression_order: u32,
    write_tlm: u32,
    high_throughput: u32,
    code_block_style: u32,
    code_block_width_exp: u32,
    code_block_height_exp: u32,
    output_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct J2kCodestreamAssemblyStatus {
    code: u32,
    detail: u32,
    data_len: u32,
    reserved0: u32,
}

#[cfg(target_os = "macos")]
static METAL_RUNTIME: OnceLock<Result<Arc<MetalRuntime>, String>> = OnceLock::new();

#[cfg(target_os = "macos")]
type MetalRuntimeCache = Mutex<HashMap<usize, Result<Arc<MetalRuntime>, String>>>;

#[cfg(target_os = "macos")]
static METAL_DEVICE_RUNTIMES: OnceLock<MetalRuntimeCache> = OnceLock::new();

#[cfg(target_os = "macos")]
thread_local! {
    static METAL_RUNTIME_OVERRIDE: RefCell<Option<Arc<MetalRuntime>>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    zero_u32_buffer: ComputePipelineState,
    validate_bytes_equal: ComputePipelineState,
    copy_interleaved_padded: ComputePipelineState,
    lossless_deinterleave_to_planes: ComputePipelineState,
    lossless_deinterleave_rct_rgb8_to_planes: ComputePipelineState,
    lossless_extract_coefficients: ComputePipelineState,
    pack_gray8: ComputePipelineState,
    pack_rgb8: ComputePipelineState,
    pack_mct_rgb8: ComputePipelineState,
    pack_mct_rgb8_batched: ComputePipelineState,
    pack_rgb_opaque_rgba8: ComputePipelineState,
    pack_rgba8: ComputePipelineState,
    pack_gray16: ComputePipelineState,
    pack_rgb16: ComputePipelineState,
    pack_u8_repeated_gray: ComputePipelineState,
    pack_u16_repeated_gray: ComputePipelineState,
    classic_cleanup_plain_batched: ComputePipelineState,
    classic_cleanup_batched: ComputePipelineState,
    classic_cleanup_plain_repeated_batched: ComputePipelineState,
    classic_cleanup_plain_dev_repeated_batched: ComputePipelineState,
    classic_cleanup_repeated_batched: ComputePipelineState,
    classic_store_repeated_batched: ComputePipelineState,
    idwt_interleave: ComputePipelineState,
    idwt_reversible53_horizontal: ComputePipelineState,
    idwt_reversible53_vertical: ComputePipelineState,
    idwt_interleave_batched: ComputePipelineState,
    idwt_reversible53_horizontal_batched: ComputePipelineState,
    idwt_reversible53_vertical_batched: ComputePipelineState,
    idwt_irreversible97_single_decomposition: ComputePipelineState,
    fdwt53_horizontal: ComputePipelineState,
    fdwt53_vertical: ComputePipelineState,
    fdwt53_horizontal_batched: ComputePipelineState,
    fdwt53_vertical_batched: ComputePipelineState,
    inverse_mct: ComputePipelineState,
    forward_rct: ComputePipelineState,
    store_component: ComputePipelineState,
    store_component_repeated: ComputePipelineState,
    store_component_repeated_gray_u8: ComputePipelineState,
    store_component_repeated_gray_u16: ComputePipelineState,
    store_component_repeated_gray_u8_contiguous: ComputePipelineState,
    store_component_repeated_gray_u16_contiguous: ComputePipelineState,
    store_component_gray_u8: ComputePipelineState,
    store_component_gray_u16: ComputePipelineState,
    ht_cleanup: ComputePipelineState,
    ht_cleanup_batched: ComputePipelineState,
    ht_cleanup_repeated_batched: ComputePipelineState,
    classic_encode_code_block: ComputePipelineState,
    classic_encode_code_blocks: ComputePipelineState,
    classic_encode_code_blocks_32: ComputePipelineState,
    classic_encode_code_blocks_bypass_32: ComputePipelineState,
    classic_encode_code_blocks_bypass_u16_32: ComputePipelineState,
    classic_tier1_density_bypass_u16_32: ComputePipelineState,
    classic_tier1_raw_pack_bypass_u16_32: ComputePipelineState,
    classic_tier1_arithmetic_pack_bypass_u16_32: ComputePipelineState,
    classic_tier1_symbol_plan_bypass_u16_32: ComputePipelineState,
    classic_tier1_pass_plan_bypass_u16_32: ComputePipelineState,
    classic_tier1_token_emit_bypass_u16_32: ComputePipelineState,
    classic_tier1_split_token_emit_bypass_u16_32: ComputePipelineState,
    classic_tier1_split_mq_byte_token_emit_bypass_u16_32: ComputePipelineState,
    classic_tier1_token_pack_bypass_u16_32: ComputePipelineState,
    classic_tier1_split_token_pack_bypass_u16_32: ComputePipelineState,
    classic_encode_code_blocks_style0: ComputePipelineState,
    classic_encode_code_blocks_style0_32: ComputePipelineState,
    ht_encode_code_block: ComputePipelineState,
    ht_encode_code_blocks: ComputePipelineState,
    packet_block_prepare_resident_classic: ComputePipelineState,
    packet_block_prepare_resident_ht: ComputePipelineState,
    packet_encode: ComputePipelineState,
    packet_encode_batched: ComputePipelineState,
    packet_encode_resident_classic_batched: ComputePipelineState,
    packet_payload_copy_batched: ComputePipelineState,
    lossless_codestream_assemble: ComputePipelineState,
    lossless_codestream_assemble_batched: ComputePipelineState,
    ht_vlc_table0: Buffer,
    ht_vlc_table1: Buffer,
    ht_uvlc_table0: Buffer,
    ht_uvlc_table1: Buffer,
    ht_vlc_encode_table0: Buffer,
    ht_vlc_encode_table1: Buffer,
    ht_uvlc_encode_table: Buffer,
    tier1_dummy_buffer: Buffer,
    private_buffer_pool: Mutex<HashMap<usize, Vec<Buffer>>>,
    shared_buffer_pool: Mutex<HashMap<usize, Vec<Buffer>>>,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "Metal is unavailable on this host".to_string())?;
        Self::new_with_device(&device)
    }

    fn new_with_device(device: &Device) -> Result<Self, String> {
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_SOURCE, &options)?;
        let pipeline = |name: &str| {
            let function = library.get_function(name, None)?;
            device.new_compute_pipeline_state_with_function(&function)
        };
        let classic_cleanup_plain_batched_fn =
            library.get_function("j2k_decode_classic_cleanup_plain_batched", None)?;
        let classic_cleanup_batched_fn =
            library.get_function("j2k_decode_classic_cleanup_batched", None)?;
        let classic_cleanup_plain_repeated_batched_fn =
            library.get_function("j2k_decode_classic_cleanup_plain_repeated_batched", None)?;
        let classic_cleanup_plain_dev_repeated_batched_fn = library.get_function(
            "j2k_decode_classic_cleanup_plain_dev_repeated_batched",
            None,
        )?;
        let classic_cleanup_repeated_batched_fn =
            library.get_function("j2k_decode_classic_cleanup_repeated_batched", None)?;
        let classic_store_repeated_batched_fn =
            library.get_function("j2k_store_classic_repeated_batched", None)?;
        let idwt_interleave_fn = library.get_function("j2k_idwt_interleave", None)?;
        let idwt_interleave_batched_fn =
            library.get_function("j2k_idwt_interleave_batched", None)?;
        let idwt_reversible53_horizontal_fn =
            library.get_function("j2k_idwt_reversible53_horizontal_pass", None)?;
        let idwt_reversible53_horizontal_batched_fn =
            library.get_function("j2k_idwt_reversible53_horizontal_pass_batched", None)?;
        let idwt_reversible53_vertical_fn =
            library.get_function("j2k_idwt_reversible53_vertical_pass", None)?;
        let idwt_reversible53_vertical_batched_fn =
            library.get_function("j2k_idwt_reversible53_vertical_pass_batched", None)?;
        let idwt_irreversible97_single_decomposition_fn =
            library.get_function("j2k_idwt_irreversible97_single_decomposition", None)?;
        let fdwt53_horizontal_fn = library.get_function("j2k_forward_dwt53_horizontal", None)?;
        let fdwt53_vertical_fn = library.get_function("j2k_forward_dwt53_vertical", None)?;
        let fdwt53_horizontal_batched_fn =
            library.get_function("j2k_forward_dwt53_horizontal_batched", None)?;
        let fdwt53_vertical_batched_fn =
            library.get_function("j2k_forward_dwt53_vertical_batched", None)?;
        let inverse_mct_fn = library.get_function("j2k_inverse_mct", None)?;
        let forward_rct_fn = library.get_function("j2k_forward_rct", None)?;
        let store_component_fn = library.get_function("j2k_store_component", None)?;
        let store_component_repeated_fn =
            library.get_function("j2k_store_component_repeated", None)?;
        let store_component_repeated_gray_u8_fn =
            library.get_function("j2k_store_component_repeated_gray_u8", None)?;
        let store_component_repeated_gray_u16_fn =
            library.get_function("j2k_store_component_repeated_gray_u16", None)?;
        let store_component_repeated_gray_u8_contiguous_fn =
            library.get_function("j2k_store_component_repeated_gray_u8_contiguous", None)?;
        let store_component_repeated_gray_u16_contiguous_fn =
            library.get_function("j2k_store_component_repeated_gray_u16_contiguous", None)?;
        let store_component_gray_u8_fn =
            library.get_function("j2k_store_component_gray_u8", None)?;
        let store_component_gray_u16_fn =
            library.get_function("j2k_store_component_gray_u16", None)?;
        let ht_cleanup_fn = library.get_function("j2k_decode_ht_cleanup", None)?;
        let ht_cleanup_batched_fn = library.get_function("j2k_decode_ht_cleanup_batched", None)?;
        let ht_cleanup_repeated_batched_fn =
            library.get_function("j2k_decode_ht_cleanup_repeated_batched", None)?;
        let classic_cleanup_plain_batched =
            device.new_compute_pipeline_state_with_function(&classic_cleanup_plain_batched_fn)?;
        let classic_cleanup_batched =
            device.new_compute_pipeline_state_with_function(&classic_cleanup_batched_fn)?;
        let classic_cleanup_plain_repeated_batched = device
            .new_compute_pipeline_state_with_function(&classic_cleanup_plain_repeated_batched_fn)?;
        let classic_cleanup_plain_dev_repeated_batched = device
            .new_compute_pipeline_state_with_function(
                &classic_cleanup_plain_dev_repeated_batched_fn,
            )?;
        let classic_cleanup_repeated_batched = device
            .new_compute_pipeline_state_with_function(&classic_cleanup_repeated_batched_fn)?;
        let classic_store_repeated_batched =
            device.new_compute_pipeline_state_with_function(&classic_store_repeated_batched_fn)?;
        let idwt_interleave =
            device.new_compute_pipeline_state_with_function(&idwt_interleave_fn)?;
        let idwt_interleave_batched =
            device.new_compute_pipeline_state_with_function(&idwt_interleave_batched_fn)?;
        let idwt_reversible53_horizontal =
            device.new_compute_pipeline_state_with_function(&idwt_reversible53_horizontal_fn)?;
        let idwt_reversible53_horizontal_batched = device
            .new_compute_pipeline_state_with_function(&idwt_reversible53_horizontal_batched_fn)?;
        let idwt_reversible53_vertical =
            device.new_compute_pipeline_state_with_function(&idwt_reversible53_vertical_fn)?;
        let idwt_reversible53_vertical_batched = device
            .new_compute_pipeline_state_with_function(&idwt_reversible53_vertical_batched_fn)?;
        let idwt_irreversible97_single_decomposition = device
            .new_compute_pipeline_state_with_function(
                &idwt_irreversible97_single_decomposition_fn,
            )?;
        let fdwt53_horizontal =
            device.new_compute_pipeline_state_with_function(&fdwt53_horizontal_fn)?;
        let fdwt53_vertical =
            device.new_compute_pipeline_state_with_function(&fdwt53_vertical_fn)?;
        let fdwt53_horizontal_batched =
            device.new_compute_pipeline_state_with_function(&fdwt53_horizontal_batched_fn)?;
        let fdwt53_vertical_batched =
            device.new_compute_pipeline_state_with_function(&fdwt53_vertical_batched_fn)?;
        let inverse_mct = device.new_compute_pipeline_state_with_function(&inverse_mct_fn)?;
        let forward_rct = device.new_compute_pipeline_state_with_function(&forward_rct_fn)?;
        let store_component =
            device.new_compute_pipeline_state_with_function(&store_component_fn)?;
        let store_component_repeated =
            device.new_compute_pipeline_state_with_function(&store_component_repeated_fn)?;
        let store_component_repeated_gray_u8 = device
            .new_compute_pipeline_state_with_function(&store_component_repeated_gray_u8_fn)?;
        let store_component_repeated_gray_u16 = device
            .new_compute_pipeline_state_with_function(&store_component_repeated_gray_u16_fn)?;
        let store_component_repeated_gray_u8_contiguous = device
            .new_compute_pipeline_state_with_function(
                &store_component_repeated_gray_u8_contiguous_fn,
            )?;
        let store_component_repeated_gray_u16_contiguous = device
            .new_compute_pipeline_state_with_function(
                &store_component_repeated_gray_u16_contiguous_fn,
            )?;
        let store_component_gray_u8 =
            device.new_compute_pipeline_state_with_function(&store_component_gray_u8_fn)?;
        let store_component_gray_u16 =
            device.new_compute_pipeline_state_with_function(&store_component_gray_u16_fn)?;
        let ht_cleanup = device.new_compute_pipeline_state_with_function(&ht_cleanup_fn)?;
        let ht_cleanup_batched =
            device.new_compute_pipeline_state_with_function(&ht_cleanup_batched_fn)?;
        let ht_cleanup_repeated_batched =
            device.new_compute_pipeline_state_with_function(&ht_cleanup_repeated_batched_fn)?;
        let queue = new_command_queue(device)?;
        Ok(Self {
            device: device.clone(),
            queue,
            zero_u32_buffer: pipeline("j2k_zero_u32_buffer")?,
            validate_bytes_equal: pipeline("j2k_validate_bytes_equal")?,
            copy_interleaved_padded: pipeline("j2k_copy_interleaved_padded")?,
            lossless_deinterleave_to_planes: pipeline("j2k_lossless_deinterleave_to_planes")?,
            lossless_deinterleave_rct_rgb8_to_planes: pipeline(
                "j2k_lossless_deinterleave_rct_rgb8_to_planes",
            )?,
            lossless_extract_coefficients: pipeline("j2k_lossless_extract_coefficients")?,
            pack_gray8: pipeline("j2k_pack_gray8")?,
            pack_rgb8: pipeline("j2k_pack_rgb8")?,
            pack_mct_rgb8: pipeline("j2k_pack_mct_rgb8")?,
            pack_mct_rgb8_batched: pipeline("j2k_pack_mct_rgb8_batched")?,
            pack_rgb_opaque_rgba8: pipeline("j2k_pack_rgb_opaque_rgba8")?,
            pack_rgba8: pipeline("j2k_pack_rgba8")?,
            pack_gray16: pipeline("j2k_pack_gray16")?,
            pack_rgb16: pipeline("j2k_pack_rgb16")?,
            pack_u8_repeated_gray: pipeline("j2k_pack_u8_repeated_gray")?,
            pack_u16_repeated_gray: pipeline("j2k_pack_u16_repeated_gray")?,
            classic_cleanup_plain_batched,
            classic_cleanup_batched,
            classic_cleanup_plain_repeated_batched,
            classic_cleanup_plain_dev_repeated_batched,
            classic_cleanup_repeated_batched,
            classic_store_repeated_batched,
            idwt_interleave,
            idwt_reversible53_horizontal,
            idwt_reversible53_vertical,
            idwt_interleave_batched,
            idwt_reversible53_horizontal_batched,
            idwt_reversible53_vertical_batched,
            idwt_irreversible97_single_decomposition,
            fdwt53_horizontal,
            fdwt53_vertical,
            fdwt53_horizontal_batched,
            fdwt53_vertical_batched,
            inverse_mct,
            forward_rct,
            store_component,
            store_component_repeated,
            store_component_repeated_gray_u8,
            store_component_repeated_gray_u16,
            store_component_repeated_gray_u8_contiguous,
            store_component_repeated_gray_u16_contiguous,
            store_component_gray_u8,
            store_component_gray_u16,
            ht_cleanup,
            ht_cleanup_batched,
            ht_cleanup_repeated_batched,
            classic_encode_code_block: pipeline("j2k_encode_classic_code_block")?,
            classic_encode_code_blocks: pipeline("j2k_encode_classic_code_blocks")?,
            classic_encode_code_blocks_32: pipeline("j2k_encode_classic_code_blocks_32")?,
            classic_encode_code_blocks_bypass_32: pipeline(
                "j2k_encode_classic_code_blocks_bypass_32",
            )?,
            classic_encode_code_blocks_bypass_u16_32: pipeline(
                "j2k_encode_classic_code_blocks_bypass_u16_32",
            )?,
            classic_tier1_density_bypass_u16_32: pipeline(
                "j2k_profile_classic_tier1_density_bypass_u16_32",
            )?,
            classic_tier1_raw_pack_bypass_u16_32: pipeline(
                "j2k_profile_classic_tier1_raw_pack_bypass_u16_32",
            )?,
            classic_tier1_arithmetic_pack_bypass_u16_32: pipeline(
                "j2k_profile_classic_tier1_arithmetic_pack_bypass_u16_32",
            )?,
            classic_tier1_symbol_plan_bypass_u16_32: pipeline(
                "j2k_plan_classic_tier1_symbols_bypass_u16_32",
            )?,
            classic_tier1_pass_plan_bypass_u16_32: pipeline(
                "j2k_plan_classic_tier1_passes_bypass_u16_32",
            )?,
            classic_tier1_token_emit_bypass_u16_32: pipeline(
                "j2k_emit_classic_tier1_tokens_bypass_u16_32",
            )?,
            classic_tier1_split_token_emit_bypass_u16_32: pipeline(
                "j2k_emit_classic_tier1_split_tokens_bypass_u16_32",
            )?,
            classic_tier1_split_mq_byte_token_emit_bypass_u16_32: pipeline(
                "j2k_emit_classic_tier1_split_mq_byte_raw_tokens_bypass_u16_32",
            )?,
            classic_tier1_token_pack_bypass_u16_32: pipeline(
                "j2k_pack_classic_tier1_tokens_bypass_u16_32",
            )?,
            classic_tier1_split_token_pack_bypass_u16_32: pipeline(
                "j2k_pack_classic_tier1_split_tokens_bypass_u16_32",
            )?,
            classic_encode_code_blocks_style0: pipeline("j2k_encode_classic_code_blocks_style0")?,
            classic_encode_code_blocks_style0_32: pipeline(
                "j2k_encode_classic_code_blocks_style0_32",
            )?,
            ht_encode_code_block: pipeline("j2k_encode_ht_code_block")?,
            ht_encode_code_blocks: pipeline("j2k_encode_ht_code_blocks")?,
            packet_block_prepare_resident_classic: pipeline(
                "j2k_prepare_packet_blocks_from_classic_status",
            )?,
            packet_block_prepare_resident_ht: pipeline("j2k_prepare_packet_blocks_from_ht_status")?,
            packet_encode: pipeline("j2k_encode_packetization")?,
            packet_encode_batched: pipeline("j2k_encode_packetization_batched")?,
            packet_encode_resident_classic_batched: pipeline(
                "j2k_encode_packetization_resident_classic_batched",
            )?,
            packet_payload_copy_batched: pipeline("j2k_copy_packet_payload_batched")?,
            lossless_codestream_assemble: pipeline("j2k_assemble_lossless_classic_codestream")?,
            lossless_codestream_assemble_batched: pipeline(
                "j2k_assemble_lossless_codestream_batched",
            )?,
            ht_vlc_table0: device.new_buffer_with_data(
                ht_vlc_table0().as_ptr().cast(),
                size_of_val(ht_vlc_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_table1: device.new_buffer_with_data(
                ht_vlc_table1().as_ptr().cast(),
                size_of_val(ht_vlc_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_table0: device.new_buffer_with_data(
                ht_uvlc_table0().as_ptr().cast(),
                size_of_val(ht_uvlc_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_table1: device.new_buffer_with_data(
                ht_uvlc_table1().as_ptr().cast(),
                size_of_val(ht_uvlc_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_encode_table0: device.new_buffer_with_data(
                ht_vlc_encode_table0().as_ptr().cast(),
                size_of_val(ht_vlc_encode_table0()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_vlc_encode_table1: device.new_buffer_with_data(
                ht_vlc_encode_table1().as_ptr().cast(),
                size_of_val(ht_vlc_encode_table1()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            ht_uvlc_encode_table: device.new_buffer_with_data(
                ht_uvlc_encode_table().as_ptr().cast(),
                size_of_val(ht_uvlc_encode_table()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            tier1_dummy_buffer: device.new_buffer(1, MTLResourceOptions::StorageModeShared),
            private_buffer_pool: Mutex::new(HashMap::new()),
            shared_buffer_pool: Mutex::new(HashMap::new()),
        })
    }

    fn take_private_buffer(&self, bytes: usize) -> Buffer {
        let bytes = bytes.max(1);
        let mut pool = self
            .private_buffer_pool
            .lock()
            .expect("private buffer pool lock not poisoned");
        if let Some(buffer) = pool.get_mut(&bytes).and_then(Vec::pop) {
            buffer
        } else {
            #[cfg(test)]
            PRIVATE_BUFFER_POOL_MISSES.with(|misses| misses.set(misses.get() + 1));
            self.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModePrivate)
        }
    }

    fn recycle_private_buffer(&self, bytes: usize, buffer: Buffer) {
        let bytes = bytes.max(1);
        self.private_buffer_pool
            .lock()
            .expect("private buffer pool lock not poisoned")
            .entry(bytes)
            .or_default()
            .push(buffer);
    }

    fn take_shared_buffer(&self, bytes: usize) -> Buffer {
        let bytes = bytes.max(1);
        let mut pool = self
            .shared_buffer_pool
            .lock()
            .expect("shared buffer pool lock not poisoned");
        if let Some(buffer) = pool.get_mut(&bytes).and_then(Vec::pop) {
            buffer
        } else {
            #[cfg(test)]
            SHARED_BUFFER_POOL_MISSES.with(|misses| misses.set(misses.get() + 1));
            self.device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
        }
    }

    fn recycle_shared_buffer(&self, bytes: usize, buffer: Buffer) {
        let bytes = bytes.max(1);
        self.shared_buffer_pool
            .lock()
            .expect("shared buffer pool lock not poisoned")
            .entry(bytes)
            .or_default()
            .push(buffer);
    }
}

#[cfg(target_os = "macos")]
fn new_command_queue(device: &Device) -> Result<CommandQueue, String> {
    let queue: *mut MTLCommandQueue = unsafe {
        device
            .as_ref()
            .send_message(Sel::register("newCommandQueue"), ())
            .map_err(|error| format!("Metal command queue creation failed: {error}"))?
    };
    if queue.is_null() {
        return Err("Metal command queue is unavailable on this host".to_string());
    }
    Ok(unsafe { CommandQueue::from_ptr(queue) })
}

#[cfg(target_os = "macos")]
fn with_runtime<R>(f: impl FnOnce(&MetalRuntime) -> Result<R, Error>) -> Result<R, Error> {
    let override_runtime = METAL_RUNTIME_OVERRIDE.with(|slot| slot.borrow().clone());
    if let Some(runtime) = override_runtime {
        return f(&runtime);
    }

    match METAL_RUNTIME.get_or_init(|| MetalRuntime::new().map(Arc::new)) {
        Ok(runtime) => f(runtime),
        Err(message) => Err(runtime_initialization_error(message)),
    }
}

#[cfg(target_os = "macos")]
fn runtime_initialization_error(message: &str) -> Error {
    match message {
        "Metal is unavailable on this host" | "Metal command queue is unavailable on this host" => {
            Error::MetalUnavailable
        }
        _ => Error::MetalKernel {
            message: message.to_string(),
        },
    }
}

#[cfg(target_os = "macos")]
struct RuntimeOverrideGuard {
    previous: Option<Arc<MetalRuntime>>,
}

#[cfg(target_os = "macos")]
impl Drop for RuntimeOverrideGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        METAL_RUNTIME_OVERRIDE.with(|slot| {
            slot.replace(previous);
        });
    }
}

#[cfg(target_os = "macos")]
fn with_runtime_for_device<R>(
    device: &Device,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let override_runtime = METAL_RUNTIME_OVERRIDE.with(|slot| slot.borrow().clone());
    if let Some(runtime) = override_runtime {
        if runtime.device.as_ptr() == device.as_ptr() {
            return f(&runtime);
        }
    }

    let cache = METAL_DEVICE_RUNTIMES.get_or_init(|| Mutex::new(HashMap::new()));
    let key = device.as_ptr() as usize;
    let runtime = {
        let mut cache = cache
            .lock()
            .expect("J2K Metal runtime cache lock not poisoned");
        cache
            .entry(key)
            .or_insert_with(|| MetalRuntime::new_with_device(device).map(Arc::new))
            .clone()
    }
    .map_err(|message| runtime_initialization_error(&message))?;
    let previous = METAL_RUNTIME_OVERRIDE.with(|slot| slot.replace(Some(runtime.clone())));
    let _guard = RuntimeOverrideGuard { previous };
    f(&runtime)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn with_isolated_runtime_for_device_for_test<R>(
    device: &Device,
    f: impl FnOnce() -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = Arc::new(
        MetalRuntime::new_with_device(device)
            .map_err(|message| runtime_initialization_error(&message))?,
    );
    let previous = METAL_RUNTIME_OVERRIDE.with(|slot| slot.replace(Some(runtime)));
    let _guard = RuntimeOverrideGuard { previous };
    f()
}

#[cfg(target_os = "macos")]
pub(crate) fn validate_metal_buffer_matches_bytes(
    expected: &[u8],
    actual_buffer: &Buffer,
    actual_byte_offset: usize,
    session: &crate::MetalBackendSession,
) -> Result<(), Error> {
    if expected.is_empty() {
        return Ok(());
    }
    let byte_len = u32::try_from(expected.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation buffer exceeds u32 byte length".to_string(),
    })?;
    let actual_offset = u64::try_from(actual_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation buffer offset exceeds u64".to_string(),
    })?;

    with_runtime_for_device(&session.device, |runtime| {
        let expected_buffer = runtime.device.new_buffer_with_data(
            expected.as_ptr().cast(),
            expected.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status = J2kValidateBytesStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kValidateBytesStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = J2kValidateBytesParams { byte_len };

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "signinum-j2k lossless coefficient prep");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.validate_bytes_equal);
        encoder.set_buffer(0, Some(actual_buffer), actual_offset);
        encoder.set_buffer(1, Some(&expected_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kValidateBytesParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kValidateBytesStatus>()
                .read()
        };
        if status.code == 0 {
            return Ok(());
        }

        Err(Error::MetalKernel {
            message: format!(
                "J2K Metal validation mismatch at byte {}: expected {}, got {}",
                status.index, status.expected, status.actual
            ),
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn validate_metal_buffers_match(
    expected_buffer: &Buffer,
    expected_byte_offset: usize,
    actual_buffer: &Buffer,
    actual_byte_offset: usize,
    byte_len: usize,
    session: &crate::MetalBackendSession,
) -> Result<(), Error> {
    if byte_len == 0 {
        return Ok(());
    }
    let byte_len_u32 = u32::try_from(byte_len).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation buffer exceeds u32 byte length".to_string(),
    })?;
    let expected_offset = u64::try_from(expected_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation expected buffer offset exceeds u64".to_string(),
    })?;
    let actual_offset = u64::try_from(actual_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation actual buffer offset exceeds u64".to_string(),
    })?;

    with_runtime_for_device(&session.device, |runtime| {
        let status = J2kValidateBytesStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kValidateBytesStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = J2kValidateBytesParams {
            byte_len: byte_len_u32,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(
            command_buffer,
            "signinum-j2k lossless coefficient prep batch",
        );
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.validate_bytes_equal);
        encoder.set_buffer(0, Some(actual_buffer), actual_offset);
        encoder.set_buffer(1, Some(expected_buffer), expected_offset);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kValidateBytesParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kValidateBytesStatus>()
                .read()
        };
        if status.code == 0 {
            return Ok(());
        }

        Err(Error::MetalKernel {
            message: format!(
                "J2K Metal validation mismatch at byte {}: expected {}, got {}",
                status.index, status.expected, status.actual
            ),
        })
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_interleaved_padded_to_shared_buffer(
    src_buffer: &Buffer,
    src_byte_offset: usize,
    src_width: u32,
    src_height: u32,
    src_pitch_bytes: usize,
    dst_width: u32,
    dst_height: u32,
    bytes_per_pixel: usize,
    session: &crate::MetalBackendSession,
) -> Result<Buffer, Error> {
    if src_width > dst_width || src_height > dst_height {
        return Err(Error::MetalKernel {
            message: "J2K Metal input tile cannot be larger than encoded tile".to_string(),
        });
    }
    let src_stride = u32::try_from(src_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal input tile pitch exceeds u32".to_string(),
    })?;
    let dst_stride_usize = (dst_width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal padded tile stride overflow".to_string(),
        })?;
    let dst_stride = u32::try_from(dst_stride_usize).map_err(|_| Error::MetalKernel {
        message: "J2K Metal padded tile stride exceeds u32".to_string(),
    })?;
    let dst_len = dst_stride_usize
        .checked_mul(dst_height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal padded tile byte length overflow".to_string(),
        })?;
    let bytes_per_pixel = u32::try_from(bytes_per_pixel).map_err(|_| Error::MetalKernel {
        message: "J2K Metal bytes-per-pixel exceeds u32".to_string(),
    })?;
    let src_offset = u64::try_from(src_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal input tile offset exceeds u64".to_string(),
    })?;

    with_runtime_for_device(&session.device, |runtime| {
        let dst_buffer = runtime
            .device
            .new_buffer(dst_len as u64, MTLResourceOptions::StorageModeShared);
        let params = J2kCopyInterleavedParams {
            src_width,
            src_height,
            src_stride,
            dst_width,
            dst_height,
            dst_stride,
            bytes_per_pixel,
        };
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.copy_interleaved_padded);
        encoder.set_buffer(0, Some(src_buffer), src_offset);
        encoder.set_buffer(1, Some(&dst_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kCopyInterleavedParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(
            encoder,
            &runtime.copy_interleaved_padded,
            (dst_width, dst_height),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(dst_buffer)
    })
}

#[cfg(target_os = "macos")]
enum DirectStatusCheck {
    Classic { buffer: Buffer, len: usize },
    Ht { buffer: Buffer, len: usize },
    Idwt(Buffer),
    Mct(Buffer),
}

#[cfg(target_os = "macos")]
struct DirectScratchBuffer {
    bytes: usize,
    buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum DirectTier1Mode {
    Metal,
    CpuUpload,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CpuTier1CoefficientCacheKey {
    step_idx: usize,
    output_len: usize,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct CpuTier1CoefficientCache {
    entries: Mutex<HashMap<CpuTier1CoefficientCacheKey, Arc<[f32]>>>,
}

#[cfg(all(target_os = "macos", test))]
fn record_direct_tier1_input_buffer_prepare() {
    DIRECT_TIER1_INPUT_BUFFER_PREPARES.with(|counter| counter.set(counter.get() + 1));
}

#[cfg(all(target_os = "macos", not(test)))]
fn record_direct_tier1_input_buffer_prepare() {}

#[cfg(target_os = "macos")]
fn prepare_direct_tier1_input_buffer<T>(
    runtime: &MetalRuntime,
    data: &[T],
    mode: DirectTier1Mode,
) -> Buffer {
    match mode {
        DirectTier1Mode::Metal => {
            record_direct_tier1_input_buffer_prepare();
            borrow_slice_buffer(&runtime.device, data)
        }
        DirectTier1Mode::CpuUpload => runtime.tier1_dummy_buffer.clone(),
    }
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct DirectHybridStageTimings {
    cpu_tier1: u128,
    cpu_tier1_flattened_batches: u128,
    cpu_tier1_classic_segment_prep: u128,
    cpu_tier1_classic_block_decode: u128,
    cpu_tier1_classic_sigprop: u128,
    cpu_tier1_classic_magref: u128,
    cpu_tier1_classic_cleanup: u128,
    cpu_tier1_classic_bypass: u128,
    cpu_tier1_classic_output_convert: u128,
    cpu_tier1_ht_block_decode: u128,
    cpu_tier1_ht_cleanup: u128,
    cpu_tier1_ht_mag_sgn: u128,
    cpu_tier1_ht_sigma: u128,
    cpu_tier1_ht_sigprop: u128,
    cpu_tier1_ht_magref: u128,
    coefficient_upload: u128,
    metal_idwt_encode: u128,
    metal_store_encode: u128,
    metal_mct_pack_encode: u128,
    command_wait: u128,
    gpu_command: u128,
    metal_idwt_gpu: u128,
    metal_idwt_interleave_gpu: u128,
    metal_idwt_horizontal_gpu: u128,
    metal_idwt_vertical_gpu: u128,
    metal_store_gpu: u128,
    metal_mct_pack_gpu: u128,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct CpuTier1DecodeSubstageCounters {
    classic_segment_prep: AtomicU64,
    classic_block_decode: AtomicU64,
    classic_sigprop: AtomicU64,
    classic_magref: AtomicU64,
    classic_cleanup: AtomicU64,
    classic_bypass: AtomicU64,
    classic_output_convert: AtomicU64,
    ht_block_decode: AtomicU64,
    ht_cleanup: AtomicU64,
    ht_mag_sgn: AtomicU64,
    ht_sigma: AtomicU64,
    ht_sigprop: AtomicU64,
    ht_magref: AtomicU64,
}

#[cfg(target_os = "macos")]
impl CpuTier1DecodeSubstageCounters {
    fn add_counter(counter: &AtomicU64, elapsed_us: u128) {
        counter.fetch_add(
            elapsed_us.min(u128::from(u64::MAX)) as u64,
            AtomicOrdering::Relaxed,
        );
    }

    fn record_classic_segment_prep(&self, started: Instant) {
        self.classic_segment_prep
            .fetch_add(elapsed_us_u64(started), AtomicOrdering::Relaxed);
    }

    fn record_classic_block_decode(&self, started: Instant, profile: &J2kCodeBlockDecodeProfile) {
        self.classic_block_decode
            .fetch_add(elapsed_us_u64(started), AtomicOrdering::Relaxed);
        Self::add_counter(&self.classic_sigprop, profile.sigprop_us);
        Self::add_counter(&self.classic_magref, profile.magref_us);
        Self::add_counter(&self.classic_cleanup, profile.cleanup_us);
        Self::add_counter(&self.classic_bypass, profile.bypass_us);
        Self::add_counter(&self.classic_output_convert, profile.output_convert_us);
    }

    fn record_ht_block_decode(&self, started: Instant, profile: &HtCodeBlockDecodeProfile) {
        self.ht_block_decode
            .fetch_add(elapsed_us_u64(started), AtomicOrdering::Relaxed);
        Self::add_counter(&self.ht_cleanup, profile.cleanup_us);
        Self::add_counter(&self.ht_mag_sgn, profile.mag_sgn_us);
        Self::add_counter(&self.ht_sigma, profile.sigma_us);
        Self::add_counter(&self.ht_sigprop, profile.sigprop_us);
        Self::add_counter(&self.ht_magref, profile.magref_us);
    }

    fn load_counter(counter: &AtomicU64) -> u128 {
        u128::from(counter.load(AtomicOrdering::Relaxed))
    }

    fn add_to_stage_timings(&self, timings: &mut DirectHybridStageTimings) {
        timings.cpu_tier1_classic_segment_prep = timings
            .cpu_tier1_classic_segment_prep
            .saturating_add(Self::load_counter(&self.classic_segment_prep));
        timings.cpu_tier1_classic_block_decode = timings
            .cpu_tier1_classic_block_decode
            .saturating_add(Self::load_counter(&self.classic_block_decode));
        timings.cpu_tier1_classic_sigprop = timings
            .cpu_tier1_classic_sigprop
            .saturating_add(Self::load_counter(&self.classic_sigprop));
        timings.cpu_tier1_classic_magref = timings
            .cpu_tier1_classic_magref
            .saturating_add(Self::load_counter(&self.classic_magref));
        timings.cpu_tier1_classic_cleanup = timings
            .cpu_tier1_classic_cleanup
            .saturating_add(Self::load_counter(&self.classic_cleanup));
        timings.cpu_tier1_classic_bypass = timings
            .cpu_tier1_classic_bypass
            .saturating_add(Self::load_counter(&self.classic_bypass));
        timings.cpu_tier1_classic_output_convert = timings
            .cpu_tier1_classic_output_convert
            .saturating_add(Self::load_counter(&self.classic_output_convert));
        timings.cpu_tier1_ht_block_decode = timings
            .cpu_tier1_ht_block_decode
            .saturating_add(Self::load_counter(&self.ht_block_decode));
        timings.cpu_tier1_ht_cleanup = timings
            .cpu_tier1_ht_cleanup
            .saturating_add(Self::load_counter(&self.ht_cleanup));
        timings.cpu_tier1_ht_mag_sgn = timings
            .cpu_tier1_ht_mag_sgn
            .saturating_add(Self::load_counter(&self.ht_mag_sgn));
        timings.cpu_tier1_ht_sigma = timings
            .cpu_tier1_ht_sigma
            .saturating_add(Self::load_counter(&self.ht_sigma));
        timings.cpu_tier1_ht_sigprop = timings
            .cpu_tier1_ht_sigprop
            .saturating_add(Self::load_counter(&self.ht_sigprop));
        timings.cpu_tier1_ht_magref = timings
            .cpu_tier1_ht_magref
            .saturating_add(Self::load_counter(&self.ht_magref));
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct DirectIdwtCommandBuffers<'a> {
    interleave: &'a CommandBufferRef,
    horizontal: &'a CommandBufferRef,
    vertical: &'a CommandBufferRef,
}

#[cfg(target_os = "macos")]
impl<'a> DirectIdwtCommandBuffers<'a> {
    fn single(command_buffer: &'a CommandBufferRef) -> Self {
        Self {
            interleave: command_buffer,
            horizontal: command_buffer,
            vertical: command_buffer,
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct DirectColorBatchCommandBuffers<'a> {
    default: &'a CommandBufferRef,
    idwt: DirectIdwtCommandBuffers<'a>,
    store: &'a CommandBufferRef,
    mct_pack: &'a CommandBufferRef,
}

#[cfg(target_os = "macos")]
impl<'a> DirectColorBatchCommandBuffers<'a> {
    fn single(command_buffer: &'a CommandBufferRef) -> Self {
        Self {
            default: command_buffer,
            idwt: DirectIdwtCommandBuffers::single(command_buffer),
            store: command_buffer,
            mct_pack: command_buffer,
        }
    }
}

#[cfg(target_os = "macos")]
struct DecodeHybridSplitCommandBuffers {
    idwt_interleave: CommandBuffer,
    idwt_horizontal: CommandBuffer,
    idwt_vertical: CommandBuffer,
    store: CommandBuffer,
    mct_pack: CommandBuffer,
}

#[cfg(target_os = "macos")]
impl DecodeHybridSplitCommandBuffers {
    fn new(runtime: &MetalRuntime) -> Self {
        let idwt_interleave = runtime.queue.new_command_buffer().to_owned();
        label_command_buffer(
            &idwt_interleave,
            "signinum-j2k decode hybrid IDWT interleave stage",
        );
        let idwt_horizontal = runtime.queue.new_command_buffer().to_owned();
        label_command_buffer(
            &idwt_horizontal,
            "signinum-j2k decode hybrid IDWT horizontal stage",
        );
        let idwt_vertical = runtime.queue.new_command_buffer().to_owned();
        label_command_buffer(
            &idwt_vertical,
            "signinum-j2k decode hybrid IDWT vertical stage",
        );
        let store = runtime.queue.new_command_buffer().to_owned();
        label_command_buffer(&store, "signinum-j2k decode hybrid store stage");
        let mct_pack = runtime.queue.new_command_buffer().to_owned();
        label_command_buffer(&mct_pack, "signinum-j2k decode hybrid MCT pack stage");
        Self {
            idwt_interleave,
            idwt_horizontal,
            idwt_vertical,
            store,
            mct_pack,
        }
    }

    fn refs(&self) -> DirectColorBatchCommandBuffers<'_> {
        DirectColorBatchCommandBuffers {
            default: &self.idwt_interleave,
            idwt: DirectIdwtCommandBuffers {
                interleave: &self.idwt_interleave,
                horizontal: &self.idwt_horizontal,
                vertical: &self.idwt_vertical,
            },
            store: &self.store,
            mct_pack: &self.mct_pack,
        }
    }

    fn commit_in_order(&self) {
        self.idwt_interleave.commit();
        self.idwt_horizontal.commit();
        self.idwt_vertical.commit();
        self.store.commit();
        self.mct_pack.commit();
    }
}

#[cfg(target_os = "macos")]
const HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK: usize = 1;
#[cfg(target_os = "macos")]
const HYBRID_FLAT_CPU_TIER1_MIN_DIM: u32 = 1024;
#[cfg(target_os = "macos")]
const HYBRID_FLAT_CPU_TIER1_MIN_COUNT: usize = 16;
#[cfg(target_os = "macos")]
const HYBRID_FLAT_CPU_TIER1_ENV: &str = "SIGNINUM_J2K_HYBRID_FLAT_CPU_TIER1";

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct PreparedDirectGrayscalePlan {
    dimensions: (u32, u32),
    bit_depth: u8,
    tier1_prepare_mode: DirectTier1Mode,
    steps: Vec<PreparedDirectGrayscaleStep>,
    classic_groups: Vec<PreparedClassicSubBandGroup>,
    ht_groups: Vec<PreparedHtSubBandGroup>,
    cpu_tier1_cache: Arc<CpuTier1CoefficientCache>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct PreparedDirectColorPlan {
    dimensions: (u32, u32),
    bit_depths: [u8; 3],
    mct: bool,
    transform: J2kWaveletTransform,
    component_plans: Vec<PreparedDirectGrayscalePlan>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
enum PreparedDirectGrayscaleStep {
    ClassicSubBand(PreparedClassicSubBand),
    HtSubBand(PreparedHtSubBand),
    Idwt(PreparedDirectIdwt),
    Store(J2kDirectStoreStep),
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedDirectIdwt {
    step: J2kDirectIdwtStep,
    output_window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBand {
    band_id: J2kDirectBandId,
    width: u32,
    height: u32,
    zero_fill: bool,
    coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    jobs_buffer: Buffer,
    segments: Vec<J2kClassicSegment>,
    segments_buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBandGroup {
    start_step: usize,
    end_step: usize,
    total_coefficients: usize,
    zero_fill: bool,
    coded_data: Vec<u8>,
    coded_buffer: Buffer,
    jobs: Vec<J2kClassicCleanupBatchJob>,
    jobs_buffer: Buffer,
    segments: Vec<J2kClassicSegment>,
    segments_buffer: Buffer,
    members: Vec<PreparedClassicSubBandGroupMember>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedClassicSubBandGroupMember {
    band_id: J2kDirectBandId,
    offset_elements: usize,
    window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBand {
    band_id: J2kDirectBandId,
    width: u32,
    height: u32,
    coded_data: Vec<u8>,
    coded_buffer: Option<Buffer>,
    jobs: Vec<J2kHtCleanupBatchJob>,
    jobs_buffer: Option<Buffer>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct HtCodedArena {
    data: Vec<u8>,
    buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBandGroup {
    start_step: usize,
    end_step: usize,
    total_coefficients: usize,
    coded_arena: HtCodedArena,
    jobs: Vec<J2kHtCleanupBatchJob>,
    jobs_buffer: Buffer,
    members: Vec<PreparedHtSubBandGroupMember>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedHtSubBandGroupMember {
    band_id: J2kDirectBandId,
    offset_elements: usize,
    window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
struct PlaneStage {
    dims: (u32, u32),
    plane_count: usize,
    color_space: NativeColorSpace,
    has_alpha: bool,
    bit_depths: [u32; 4],
    planes: [Option<Buffer>; 4],
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    fn from_planes(
        device: &Device,
        decoded: &NativeDecodedComponents<'_>,
        roi: Option<Rect>,
    ) -> Result<Self, Error> {
        let full_dims = decoded.dimensions();
        let roi = roi.unwrap_or(Rect {
            x: 0,
            y: 0,
            w: full_dims.0,
            h: full_dims.1,
        });
        let dims = (roi.w, roi.h);
        let plane_count = decoded.planes().len();
        if plane_count == 0 || plane_count > 4 {
            return Err(Error::MetalKernel {
                message: format!("unsupported J2K plane count {plane_count}"),
            });
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, plane) in decoded.planes().iter().enumerate() {
            bit_depths[index] = u32::from(plane.bit_depth());
            let len = dims.0 as usize * dims.1 as usize;
            let buffer = device.new_buffer(
                (len * size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            copy_plane_samples(&buffer, plane.samples(), full_dims.0 as usize, roi);
            planes[index] = Some(buffer);
        }

        Ok(Self {
            dims,
            plane_count,
            color_space: decoded.color_space().clone(),
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    fn from_captured_planes(
        decoded: &NativeDecodedComponents<'_>,
        captured_planes: Vec<Buffer>,
    ) -> Option<Self> {
        let plane_count = decoded.planes().len();
        let supported_shape = matches!(
            (decoded.color_space(), decoded.has_alpha(), plane_count),
            (NativeColorSpace::Gray, false, 1) | (NativeColorSpace::RGB, false, 3)
        );
        if !supported_shape {
            return None;
        }
        if captured_planes.len() != plane_count || plane_count == 0 || plane_count > 4 {
            return None;
        }

        let mut bit_depths = [0u32; 4];
        let mut planes: [Option<Buffer>; 4] = [None, None, None, None];
        for (index, (plane, buffer)) in decoded.planes().iter().zip(captured_planes).enumerate() {
            bit_depths[index] = u32::from(plane.bit_depth());
            planes[index] = Some(buffer);
        }

        Some(Self {
            dims: decoded.dimensions(),
            plane_count,
            color_space: decoded.color_space().clone(),
            has_alpha: decoded.has_alpha(),
            bit_depths,
            planes,
        })
    }

    fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        let command_buffer = runtime.queue.new_command_buffer();
        let surface =
            encode_plane_stage_to_surface_in_command_buffer(runtime, command_buffer, &self, fmt)?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(surface)
    }
}

#[cfg(target_os = "macos")]
fn two_d_threads_per_group(simd_width: u64, max_threads: u64) -> MTLSize {
    let width = simd_width.max(1);
    let max_threads = max_threads.max(width);
    MTLSize {
        width,
        height: (max_threads / width).max(1),
        depth: 1,
    }
}

#[cfg(target_os = "macos")]
fn one_d_threads_per_group(simd_width: u64) -> MTLSize {
    MTLSize {
        width: simd_width.max(1),
        height: 1,
        depth: 1,
    }
}

#[cfg(target_os = "macos")]
fn dispatch_1d_pipeline(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    width: u64,
) {
    encoder.dispatch_threads(
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
        one_d_threads_per_group(pipeline.thread_execution_width()),
    );
}

#[cfg(target_os = "macos")]
fn dispatch_single_thread(encoder: &ComputeCommandEncoderRef) {
    encoder.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
fn dispatch_2d_pipeline(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32),
) {
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: 1,
        },
        two_d_threads_per_group(
            pipeline.thread_execution_width(),
            pipeline.max_total_threads_per_threadgroup(),
        ),
    );
}

#[cfg(target_os = "macos")]
fn dispatch_3d_pipeline(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32, u32),
) {
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: u64::from(dims.2),
        },
        two_d_threads_per_group(
            pipeline.thread_execution_width(),
            pipeline.max_total_threads_per_threadgroup(),
        ),
    );
}

#[cfg(target_os = "macos")]
fn encode_plane_stage_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    stage: &PlaneStage,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = stage.dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * stage.dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) = output_shape_for(
        &stage.color_space,
        stage.has_alpha,
        stage.plane_count,
        fmt,
        runtime,
    )?;
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays(stage.bit_depths);

    let params = J2kPackParams {
        width: stage.dims.0,
        height: stage.dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        output_channels,
        opaque_alpha,
        max_values,
        u8_scales,
        u16_scales,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid plane pack");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(
        0,
        stage.planes[0].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        1,
        stage.planes[1].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        2,
        stage.planes[2].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(
        3,
        stage.planes[3].as_ref().map(std::convert::AsRef::as_ref),
        0,
    );
    encoder.set_buffer(4, Some(&out_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, stage.dims);
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, stage.dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_mct_rgb8_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Surface {
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        transform: mct_transform_code(transform),
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid MCT RGB8 pack");
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, &runtime.pack_mct_rgb8, dims);
    encoder.end_encoding();
    drop(signpost);

    Surface::from_metal_buffer(out_buffer, dims, PixelFormat::Rgb8)
}

#[cfg(target_os = "macos")]
fn encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    count: usize,
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Vec<Surface>, Error> {
    let count_u32 = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "J2K MetalDirect color batch count exceeds u32".to_string(),
    })?;
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect color batch output size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect color batch output size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let plane_stride = dims
        .0
        .checked_mul(dims.1)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect color batch plane stride overflow".to_string(),
        })?;
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kBatchedMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        transform: mct_transform_code(transform),
        batch_count: count_u32,
        plane_stride,
        output_stride: u32::try_from(surface_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect color batch surface stride exceeds u32".to_string(),
        })?,
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid batched MCT RGB8 pack");
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8_batched);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kBatchedMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        &runtime.pack_mct_rgb8_batched,
        (dims.0, dims.1, count_u32),
    );
    encoder.end_encoding();
    drop(signpost);

    Ok((0..count)
        .map(|index| {
            Surface::from_metal_buffer_with_offset(
                out_buffer.clone(),
                dims,
                PixelFormat::Rgb8,
                index * surface_bytes,
            )
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn encode_repeated_mct_rgb8_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    dims: (u32, u32),
    count: usize,
    bit_depths: [u8; 3],
    transform: J2kWaveletTransform,
) -> Result<Vec<Surface>, Error> {
    let pitch_bytes = dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect repeated color batch output size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch output size overflow".to_string(),
        })?;
    let output_len = u64::try_from(total_bytes.max(1)).map_err(|_| Error::MetalKernel {
        message: "J2K MetalDirect repeated output buffer exceeds u64".to_string(),
    })?;
    let out_buffer = runtime
        .device
        .new_buffer(output_len, MTLResourceOptions::StorageModeShared);
    let plane_stride = dims
        .0
        .checked_mul(dims.1)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch plane stride overflow".to_string(),
        })?;
    let (max_values, u8_scales, _) = j2k_pack_scale_arrays([
        u32::from(bit_depths[0]),
        u32::from(bit_depths[1]),
        u32::from(bit_depths[2]),
        0,
    ]);
    let params = J2kBatchedMctRgb8PackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        transform: mct_transform_code(transform),
        batch_count: 1,
        plane_stride,
        output_stride: u32::try_from(surface_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect repeated color batch surface stride exceeds u32".to_string(),
        })?,
        addends: [
            signed_sample_bias(bit_depths[0]),
            signed_sample_bias(bit_depths[1]),
            signed_sample_bias(bit_depths[2]),
        ],
        max_values: [max_values[0], max_values[1], max_values[2]],
        u8_scales: [u8_scales[0], u8_scales[1], u8_scales[2]],
    };

    let signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated MCT RGB8 pack");
    encoder.set_compute_pipeline_state(&runtime.pack_mct_rgb8_batched);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kBatchedMctRgb8PackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, &runtime.pack_mct_rgb8_batched, dims);
    encoder.end_encoding();
    drop(signpost);

    if surface_bytes > 0 && count > 1 {
        let blit = command_buffer.new_blit_command_encoder();
        if metal_profile_stages_enabled() {
            blit.set_label("J2K decode hybrid repeated output blit");
        }
        let mut copied = 1usize;
        while copied < count {
            let copy_count = copied.min(count - copied);
            let dst_offset =
                copied
                    .checked_mul(surface_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K MetalDirect repeated output destination offset overflow"
                            .to_string(),
                    })?;
            let copy_bytes =
                copy_count
                    .checked_mul(surface_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K MetalDirect repeated output copy size overflow".to_string(),
                    })?;
            blit.copy_from_buffer(
                &out_buffer,
                0,
                &out_buffer,
                u64::try_from(dst_offset).map_err(|_| Error::MetalKernel {
                    message: "J2K MetalDirect repeated output destination offset exceeds u64"
                        .to_string(),
                })?,
                u64::try_from(copy_bytes).map_err(|_| Error::MetalKernel {
                    message: "J2K MetalDirect repeated output copy size exceeds u64".to_string(),
                })?,
            );
            record_hybrid_repeated_output_blit();
            copied += copy_count;
        }
        blit.end_encoding();
    }

    Ok((0..count)
        .map(|index| {
            Surface::from_metal_buffer_with_offset(
                out_buffer.clone(),
                dims,
                PixelFormat::Rgb8,
                index * surface_bytes,
            )
        })
        .collect())
}

#[cfg(target_os = "macos")]
fn repeated_shared_direct_color_plan_count(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> Option<usize> {
    let first = plans.first()?;
    (plans.len() > 1 && plans.iter().all(|plan| Arc::ptr_eq(plan, first))).then_some(plans.len())
}

#[cfg(target_os = "macos")]
fn mct_transform_code(transform: J2kWaveletTransform) -> u32 {
    match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    }
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band(
    job: &signinum_j2k_native::J2kOwnedSubBandPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBand, Error> {
    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    let mut segments = Vec::new();

    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect segment table exceeds u32".to_string(),
        })?;
        for segment in &block.segments {
            let data_offset = coded_offset
                .checked_add(segment.data_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect segment offset overflow".to_string(),
                })?;
            segments.push(J2kClassicSegment {
                data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            });
        }
        jobs.push(J2kClassicCleanupBatchJob {
            coded_offset,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            segment_offset,
            segment_count: u32::try_from(block.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K MetalDirect segment count exceeds u32".to_string(),
            })?,
            width: block.width,
            height: block.height,
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect output offset overflow".to_string(),
                })?,
            missing_msbs: u32::from(block.missing_bit_planes),
            total_bitplanes: u32::from(block.total_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            sub_band_type: match block.sub_band_type {
                signinum_j2k_native::J2kSubBandType::LowLow => 0,
                signinum_j2k_native::J2kSubBandType::HighLow => 1,
                signinum_j2k_native::J2kSubBandType::LowHigh => 2,
                signinum_j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(block.style),
            strict: u32::from(block.strict),
            dequantization_step: block.dequantization_step,
        });
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode);
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode);
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &segments, tier1_prepare_mode);
        Ok(PreparedClassicSubBand {
            band_id: job.band_id,
            width: job.width,
            height: job.height,
            zero_fill: false,
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedClassicSubBandGroup>, Error> {
    let mut groups = Vec::new();
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        let mut sub_bands = Vec::new();
        while let Some(PreparedDirectGrayscaleStep::ClassicSubBand(sub_band)) = steps.get(step_idx)
        {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_classic_sub_band_group(
                start_step,
                step_idx,
                &sub_bands,
                tier1_prepare_mode,
            )?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
fn prepare_classic_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedClassicSubBand],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedClassicSubBandGroup, Error> {
    let mut members = Vec::with_capacity(sub_bands.len());
    let mut jobs = Vec::new();
    let mut segments = Vec::new();
    let mut coded_data = Vec::new();
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedClassicSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let segment_base = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped segment table exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;

        for segment in &sub_band.segments {
            let mut grouped_segment = *segment;
            grouped_segment.data_offset =
                coded_base
                    .checked_add(segment.data_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped segment offset overflow"
                            .to_string(),
                    })?;
            segments.push(grouped_segment);
        }

        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job coded offset overflow"
                            .to_string(),
                    })?;
            grouped_job.segment_offset =
                segment_base
                    .checked_add(job.segment_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped job segment offset overflow"
                            .to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect grouped output offset overflow"
                            .to_string(),
                    })?;
            jobs.push(grouped_job);
        }

        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode);
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode);
        let segments_buffer =
            prepare_direct_tier1_input_buffer(runtime, &segments, tier1_prepare_mode);
        Ok(PreparedClassicSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            zero_fill: sub_bands.iter().any(|sub_band| sub_band.zero_fill),
            coded_data,
            coded_buffer,
            jobs,
            jobs_buffer,
            segments,
            segments_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band(
    job: &signinum_j2k_native::HtOwnedSubBandPlan,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBand, Error> {
    let mut jobs = Vec::with_capacity(job.jobs.len());
    let mut coded_data = Vec::new();
    for block in &job.jobs {
        let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(&block.data);
        jobs.push(J2kHtCleanupBatchJob {
            coded_offset,
            width: block.width,
            height: block.height,
            coded_len: u32::try_from(block.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: block.cleanup_length,
            refinement_length: block.refinement_length,
            missing_msbs: u32::from(block.missing_bit_planes),
            num_bitplanes: u32::from(block.num_bitplanes),
            roi_shift: u32::from(block.roi_shift),
            number_of_coding_passes: u32::from(block.number_of_coding_passes),
            output_stride: job.width,
            output_offset: block
                .output_y
                .checked_mul(job.width)
                .and_then(|row| row.checked_add(block.output_x))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect output offset overflow".to_string(),
                })?,
            dequantization_step: block.dequantization_step,
            stripe_causal: u32::from(block.stripe_causal),
        });
    }

    Ok(PreparedHtSubBand {
        band_id: job.band_id,
        width: job.width,
        height: job.height,
        coded_data,
        coded_buffer: None,
        jobs,
        jobs_buffer: None,
    })
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band_groups(
    steps: &[PreparedDirectGrayscaleStep],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<Vec<PreparedHtSubBandGroup>, Error> {
    let mut groups = Vec::new();
    let mut step_idx = 0;
    while step_idx < steps.len() {
        let start_step = step_idx;
        let mut sub_bands = Vec::new();
        while let Some(PreparedDirectGrayscaleStep::HtSubBand(sub_band)) = steps.get(step_idx) {
            sub_bands.push(sub_band);
            step_idx += 1;
        }
        if sub_bands.len() > 1 {
            groups.push(prepare_ht_sub_band_group(
                start_step,
                step_idx,
                &sub_bands,
                tier1_prepare_mode,
            )?);
        }
        if step_idx == start_step {
            step_idx += 1;
        }
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
fn prepare_ht_sub_band_group(
    start_step: usize,
    end_step: usize,
    sub_bands: &[&PreparedHtSubBand],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedHtSubBandGroup, Error> {
    let mut members = Vec::with_capacity(sub_bands.len());
    let mut jobs = Vec::new();
    let mut coded_data = Vec::new();
    let mut output_base = 0usize;

    for sub_band in sub_bands {
        members.push(PreparedHtSubBandGroupMember {
            band_id: sub_band.band_id,
            offset_elements: output_base,
            window: BandRequiredRegion::full(sub_band.width, sub_band.height),
        });

        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coded payload exceeds u32".to_string(),
        })?;
        let output_base_u32 = u32::try_from(output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect grouped coefficient arena exceeds u32".to_string(),
        })?;
        for job in &sub_band.jobs {
            let mut grouped_job = *job;
            grouped_job.coded_offset =
                coded_base
                    .checked_add(job.coded_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped coded offset overflow".to_string(),
                    })?;
            grouped_job.output_offset =
                output_base_u32
                    .checked_add(job.output_offset)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect grouped output offset overflow".to_string(),
                    })?;
            jobs.push(grouped_job);
        }
        coded_data.extend_from_slice(&sub_band.coded_data);
        let sub_band_len =
            sub_band
                .width
                .checked_mul(sub_band.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect grouped sub-band size overflow".to_string(),
                })? as usize;
        output_base = output_base
            .checked_add(sub_band_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect grouped coefficient arena overflow".to_string(),
            })?;
    }

    with_runtime(|runtime| {
        let coded_buffer =
            prepare_direct_tier1_input_buffer(runtime, &coded_data, tier1_prepare_mode);
        let jobs_buffer = prepare_direct_tier1_input_buffer(runtime, &jobs, tier1_prepare_mode);
        Ok(PreparedHtSubBandGroup {
            start_step,
            end_step,
            total_coefficients: output_base,
            coded_arena: HtCodedArena {
                data: coded_data,
                buffer: coded_buffer,
            },
            jobs,
            jobs_buffer,
            members,
        })
    })
}

#[cfg(target_os = "macos")]
fn prepare_ungrouped_ht_sub_band_buffers(
    steps: &mut [PreparedDirectGrayscaleStep],
    groups: &[PreparedHtSubBandGroup],
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<(), Error> {
    if tier1_prepare_mode != DirectTier1Mode::Metal {
        return Ok(());
    }

    for (step_idx, step) in steps.iter_mut().enumerate() {
        let PreparedDirectGrayscaleStep::HtSubBand(sub_band) = step else {
            continue;
        };
        if groups
            .iter()
            .any(|group| group.start_step <= step_idx && step_idx < group.end_step)
        {
            sub_band.coded_buffer = None;
            sub_band.jobs_buffer = None;
            continue;
        }
        with_runtime(|runtime| {
            sub_band.coded_buffer = Some(prepare_direct_tier1_input_buffer(
                runtime,
                &sub_band.coded_data,
                tier1_prepare_mode,
            ));
            sub_band.jobs_buffer = Some(prepare_direct_tier1_input_buffer(
                runtime,
                &sub_band.jobs,
                tier1_prepare_mode,
            ));
            Ok(())
        })?;
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn prepared_ht_buffer<'a>(buffer: Option<&'a Buffer>, label: &str) -> Result<&'a Buffer, Error> {
    buffer.ok_or_else(|| Error::MetalKernel {
        message: format!("HTJ2K MetalDirect ungrouped sub-band is missing prepared {label} buffer"),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_grayscale_plan(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
fn prepare_direct_grayscale_plan_for_cpu_upload(
    plan: &J2kDirectGrayscalePlan,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    prepare_direct_grayscale_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn prepare_direct_grayscale_plan_with_tier1_mode(
    plan: &J2kDirectGrayscalePlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectGrayscalePlan, Error> {
    let mut steps = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::ClassicSubBand(
                    prepare_classic_sub_band(sub_band, tier1_prepare_mode)?,
                ));
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                steps.push(PreparedDirectGrayscaleStep::HtSubBand(prepare_ht_sub_band(
                    sub_band,
                    tier1_prepare_mode,
                )?));
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                steps.push(PreparedDirectGrayscaleStep::Idwt(PreparedDirectIdwt {
                    step: *idwt,
                    output_window: BandRequiredRegion::full(idwt.rect.width(), idwt.rect.height()),
                }));
            }
            J2kDirectGrayscaleStep::Store(store) => {
                steps.push(PreparedDirectGrayscaleStep::Store(*store));
            }
        }
    }
    let classic_groups = prepare_classic_sub_band_groups(&steps, tier1_prepare_mode)?;
    let ht_groups = prepare_ht_sub_band_groups(&steps, tier1_prepare_mode)?;
    prepare_ungrouped_ht_sub_band_buffers(&mut steps, &ht_groups, tier1_prepare_mode)?;
    Ok(PreparedDirectGrayscalePlan {
        dimensions: plan.dimensions,
        bit_depth: plan.bit_depth,
        tier1_prepare_mode,
        steps,
        classic_groups,
        ht_groups,
        cpu_tier1_cache: Arc::new(CpuTier1CoefficientCache::default()),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn crop_prepared_direct_grayscale_plan_to_output_region(
    plan: &mut PreparedDirectGrayscalePlan,
    region: Rect,
) -> Result<(), Error> {
    if region.w == 0 || region.h == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect region-scaled grayscale plan has an empty output region"
                .to_string(),
        });
    }
    if region.x == 0
        && region.y == 0
        && region.w == plan.dimensions.0
        && region.h == plan.dimensions.1
    {
        return Ok(());
    }

    plan.clear_cpu_tier1_cache()?;
    let mut store_count = 0;
    for step in &mut plan.steps {
        if let PreparedDirectGrayscaleStep::Store(store) = step {
            crop_direct_store_step_to_output_region(store, region)?;
            store_count += 1;
        }
    }

    if store_count == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect grayscale plan has no store step to crop".to_string(),
        });
    }

    prune_prepared_direct_grayscale_plan_to_store_windows(plan)?;
    plan.dimensions = (region.w, region.h);
    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
struct BandRequiredRegion {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

#[cfg(target_os = "macos")]
impl BandRequiredRegion {
    fn full(width: u32, height: u32) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        }
    }

    fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Option<Self> {
        (x0 < x1 && y0 < y1).then_some(Self { x0, y0, x1, y1 })
    }

    fn width(self) -> u32 {
        self.x1 - self.x0
    }

    fn height(self) -> u32 {
        self.y1 - self.y0
    }

    fn expanded(self, margin: u32, width: u32, height: u32) -> Self {
        Self {
            x0: self.x0.saturating_sub(margin),
            y0: self.y0.saturating_sub(margin),
            x1: self.x1.saturating_add(margin).min(width),
            y1: self.y1.saturating_add(margin).min(height),
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    fn intersects(self, x0: u32, y0: u32, width: u32, height: u32) -> bool {
        let x1 = x0.saturating_add(width);
        let y1 = y0.saturating_add(height);
        self.x0 < x1 && x0 < self.x1 && self.y0 < y1 && y0 < self.y1
    }
}

#[cfg(target_os = "macos")]
fn prune_prepared_direct_grayscale_plan_to_store_windows(
    plan: &mut PreparedDirectGrayscalePlan,
) -> Result<(), Error> {
    let mut required = HashMap::<J2kDirectBandId, BandRequiredRegion>::new();
    for step in &plan.steps {
        if let PreparedDirectGrayscaleStep::Store(store) = step {
            let source_right = store
                .source_x
                .checked_add(store.copy_width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect ROI source width overflows u32".to_string(),
                })?;
            let source_bottom = store
                .source_y
                .checked_add(store.copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect ROI source height overflows u32".to_string(),
                })?;
            if let Some(region) =
                BandRequiredRegion::new(store.source_x, store.source_y, source_right, source_bottom)
            {
                add_required_region(&mut required, store.input_band_id, region);
            }
        }
    }

    let mut idwt_output_windows = HashMap::<J2kDirectBandId, BandRequiredRegion>::new();
    for step in plan.steps.iter().rev() {
        if let PreparedDirectGrayscaleStep::Idwt(idwt) = step {
            let Some(output_region) = required.get(&idwt.step.output_band_id).copied() else {
                continue;
            };
            let expanded = output_region.expanded(
                idwt_required_output_margin(idwt.step.transform),
                idwt.step.rect.width(),
                idwt.step.rect.height(),
            );
            idwt_output_windows.insert(idwt.step.output_band_id, expanded);
            add_idwt_input_required_regions(&mut required, &idwt.step, expanded);
        }
    }

    for step in &mut plan.steps {
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let before = sub_band.jobs.len();
                retain_classic_jobs_for_required_region(
                    &mut sub_band.jobs,
                    required.get(&sub_band.band_id).copied(),
                );
                if sub_band.jobs.len() != before {
                    sub_band.zero_fill = true;
                    if plan.tier1_prepare_mode == DirectTier1Mode::Metal {
                        with_runtime(|runtime| {
                            sub_band.jobs_buffer =
                                borrow_slice_buffer(&runtime.device, &sub_band.jobs);
                            Ok(())
                        })?;
                    }
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let before = sub_band.jobs.len();
                retain_ht_jobs_for_required_region(
                    &mut sub_band.jobs,
                    required.get(&sub_band.band_id).copied(),
                );
                if sub_band.jobs.len() != before {
                    compact_ht_sub_band_coded_data(sub_band, plan.tier1_prepare_mode)?;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => {}
        }
    }

    apply_prepared_direct_idwt_output_windows(plan, &idwt_output_windows)?;
    plan.classic_groups = prepare_classic_sub_band_groups(&plan.steps, plan.tier1_prepare_mode)?;
    plan.ht_groups = prepare_ht_sub_band_groups(&plan.steps, plan.tier1_prepare_mode)?;
    prepare_ungrouped_ht_sub_band_buffers(
        &mut plan.steps,
        &plan.ht_groups,
        plan.tier1_prepare_mode,
    )?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn apply_prepared_direct_idwt_output_windows(
    plan: &mut PreparedDirectGrayscalePlan,
    windows: &HashMap<J2kDirectBandId, BandRequiredRegion>,
) -> Result<(), Error> {
    for step in &mut plan.steps {
        if let PreparedDirectGrayscaleStep::Idwt(idwt) = step {
            idwt.output_window = windows
                .get(&idwt.step.output_band_id)
                .copied()
                .unwrap_or_else(|| {
                    BandRequiredRegion::full(idwt.step.rect.width(), idwt.step.rect.height())
                });
        }
    }

    for step in &mut plan.steps {
        let PreparedDirectGrayscaleStep::Store(store) = step else {
            continue;
        };
        let Some(window) = windows.get(&store.input_band_id).copied() else {
            continue;
        };

        store.source_x =
            store
                .source_x
                .checked_sub(window.x0)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect cropped IDWT store source x underflow".to_string(),
                })?;
        store.source_y =
            store
                .source_y
                .checked_sub(window.y0)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect cropped IDWT store source y underflow".to_string(),
                })?;
        store.input_rect = signinum_j2k_native::J2kRect {
            x0: store.input_rect.x0.saturating_add(window.x0),
            y0: store.input_rect.y0.saturating_add(window.y0),
            x1: store.input_rect.x0.saturating_add(window.x1),
            y1: store.input_rect.y0.saturating_add(window.y1),
        };
    }

    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct PreparedIdwtInputWindows {
    ll: BandRequiredRegion,
    hl: BandRequiredRegion,
    lh: BandRequiredRegion,
    hh: BandRequiredRegion,
}

fn idwt_input_windows_from_slices(
    ll: &DirectBandSlice,
    hl: &DirectBandSlice,
    lh: &DirectBandSlice,
    hh: &DirectBandSlice,
) -> PreparedIdwtInputWindows {
    PreparedIdwtInputWindows {
        ll: BandRequiredRegion::full(ll.window.width(), ll.window.height()),
        hl: BandRequiredRegion::full(hl.window.width(), hl.window.height()),
        lh: BandRequiredRegion::full(lh.window.width(), lh.window.height()),
        hh: BandRequiredRegion::full(hh.window.width(), hh.window.height()),
    }
}

#[cfg(target_os = "macos")]
fn prepared_idwt_params(
    idwt: &PreparedDirectIdwt,
    inputs: PreparedIdwtInputWindows,
) -> J2kIdwtSingleDecompositionParams {
    J2kIdwtSingleDecompositionParams {
        x0: idwt.step.rect.x0,
        y0: idwt.step.rect.y0,
        output_x: idwt.output_window.x0,
        output_y: idwt.output_window.y0,
        width: idwt.output_window.width(),
        height: idwt.output_window.height(),
        ll_x: inputs.ll.x0,
        ll_y: inputs.ll.y0,
        ll_width: inputs.ll.width(),
        ll_height: inputs.ll.height(),
        hl_x: inputs.hl.x0,
        hl_y: inputs.hl.y0,
        hl_width: inputs.hl.width(),
        hl_height: inputs.hl.height(),
        lh_x: inputs.lh.x0,
        lh_y: inputs.lh.y0,
        lh_width: inputs.lh.width(),
        lh_height: inputs.lh.height(),
        hh_x: inputs.hh.x0,
        hh_y: inputs.hh.y0,
        hh_width: inputs.hh.width(),
        hh_height: inputs.hh.height(),
    }
}

#[cfg(target_os = "macos")]
fn repeated_idwt_params(
    idwt: &PreparedDirectIdwt,
    inputs: PreparedIdwtInputWindows,
    strides: PreparedIdwtInputStrides,
    batch_count: usize,
    context: &'static str,
) -> Result<J2kRepeatedIdwtSingleDecompositionParams, Error> {
    Ok(J2kRepeatedIdwtSingleDecompositionParams {
        x0: idwt.step.rect.x0,
        y0: idwt.step.rect.y0,
        output_x: idwt.output_window.x0,
        output_y: idwt.output_window.y0,
        width: idwt.output_window.width(),
        height: idwt.output_window.height(),
        ll_x: inputs.ll.x0,
        ll_y: inputs.ll.y0,
        ll_width: inputs.ll.width(),
        ll_height: inputs.ll.height(),
        hl_x: inputs.hl.x0,
        hl_y: inputs.hl.y0,
        hl_width: inputs.hl.width(),
        hl_height: inputs.hl.height(),
        lh_x: inputs.lh.x0,
        lh_y: inputs.lh.y0,
        lh_width: inputs.lh.width(),
        lh_height: inputs.lh.height(),
        hh_x: inputs.hh.x0,
        hh_y: inputs.hh.y0,
        hh_width: inputs.hh.width(),
        hh_height: inputs.hh.height(),
        ll_instance_stride: strides.ll,
        hl_instance_stride: strides.hl,
        lh_instance_stride: strides.lh,
        hh_instance_stride: strides.hh,
        batch_count: u32::try_from(batch_count).map_err(|_| Error::MetalKernel {
            message: format!("J2K MetalDirect {context} IDWT batch count exceeds u32"),
        })?,
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct PreparedIdwtInputStrides {
    ll: u32,
    hl: u32,
    lh: u32,
    hh: u32,
}

#[cfg(target_os = "macos")]
fn prepared_idwt_output_len(idwt: &PreparedDirectIdwt) -> usize {
    idwt.output_window.width() as usize * idwt.output_window.height() as usize
}

#[cfg(target_os = "macos")]
fn add_required_region(
    required: &mut HashMap<J2kDirectBandId, BandRequiredRegion>,
    band_id: J2kDirectBandId,
    region: BandRequiredRegion,
) {
    required
        .entry(band_id)
        .and_modify(|existing| *existing = existing.union(region))
        .or_insert(region);
}

#[cfg(target_os = "macos")]
fn idwt_required_output_margin(transform: J2kWaveletTransform) -> u32 {
    match transform {
        J2kWaveletTransform::Reversible53 => 16,
        J2kWaveletTransform::Irreversible97 => 40,
    }
}

#[cfg(target_os = "macos")]
fn add_idwt_input_required_regions(
    required: &mut HashMap<J2kDirectBandId, BandRequiredRegion>,
    idwt: &J2kDirectIdwtStep,
    output_region: BandRequiredRegion,
) {
    add_required_region(
        required,
        idwt.ll_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            true,
            true,
            idwt.ll.width(),
            idwt.ll.height(),
        ),
    );
    add_required_region(
        required,
        idwt.hl_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            false,
            true,
            idwt.hl.width(),
            idwt.hl.height(),
        ),
    );
    add_required_region(
        required,
        idwt.lh_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            true,
            false,
            idwt.lh.width(),
            idwt.lh.height(),
        ),
    );
    add_required_region(
        required,
        idwt.hh_band_id,
        idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            false,
            false,
            idwt.hh.width(),
            idwt.hh.height(),
        ),
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::fn_params_excessive_bools)]
fn idwt_input_required_region(
    output_region: BandRequiredRegion,
    output_origin_x: u32,
    output_origin_y: u32,
    low_x: bool,
    low_y: bool,
    band_width: u32,
    band_height: u32,
) -> BandRequiredRegion {
    let x0 = signinum_j2k_native::idwt_band_index(output_origin_x, output_region.x0, low_x);
    let x1 = signinum_j2k_native::idwt_band_index(output_origin_x, output_region.x1 - 1, low_x)
        .saturating_add(1);
    let y0 = signinum_j2k_native::idwt_band_index(output_origin_y, output_region.y0, low_y);
    let y1 = signinum_j2k_native::idwt_band_index(output_origin_y, output_region.y1 - 1, low_y)
        .saturating_add(1);
    BandRequiredRegion {
        x0: x0.min(band_width),
        y0: y0.min(band_height),
        x1: x1.min(band_width),
        y1: y1.min(band_height),
    }
}

#[cfg(target_os = "macos")]
fn retain_classic_jobs_for_required_region(
    jobs: &mut Vec<J2kClassicCleanupBatchJob>,
    required: Option<BandRequiredRegion>,
) {
    let Some(required) = required else {
        jobs.clear();
        return;
    };
    jobs.retain(|job| {
        let output_x = job.output_offset % job.output_stride;
        let output_y = job.output_offset / job.output_stride;
        required.intersects(output_x, output_y, job.width, job.height)
    });
}

#[cfg(target_os = "macos")]
fn retain_ht_jobs_for_required_region(
    jobs: &mut Vec<J2kHtCleanupBatchJob>,
    required: Option<BandRequiredRegion>,
) {
    let Some(required) = required else {
        jobs.clear();
        return;
    };
    jobs.retain(|job| {
        let output_x = job.output_offset % job.output_stride;
        let output_y = job.output_offset / job.output_stride;
        required.intersects(output_x, output_y, job.width, job.height)
    });
}

#[cfg(target_os = "macos")]
fn compact_ht_sub_band_coded_data(
    sub_band: &mut PreparedHtSubBand,
    _tier1_prepare_mode: DirectTier1Mode,
) -> Result<(), Error> {
    let previous = std::mem::take(&mut sub_band.coded_data);
    let mut compacted = Vec::new();

    for job in &mut sub_band.jobs {
        let start = job.coded_offset as usize;
        let len = job.coded_len as usize;
        let end = start.checked_add(len).ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect cropped coded payload range overflow".to_string(),
        })?;
        if end > previous.len() {
            return Err(Error::MetalKernel {
                message: "HTJ2K MetalDirect cropped coded payload range out of bounds".to_string(),
            });
        }
        job.coded_offset = u32::try_from(compacted.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect cropped coded payload exceeds u32".to_string(),
        })?;
        compacted.extend_from_slice(&previous[start..end]);
    }

    sub_band.coded_data = compacted;
    sub_band.coded_buffer = None;
    sub_band.jobs_buffer = None;
    Ok(())
}

#[cfg(target_os = "macos")]
fn checked_rect_end(origin: u32, length: u32, label: &str) -> Result<u32, Error> {
    origin
        .checked_add(length)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("J2K MetalDirect region-scaled {label} overflows u32"),
        })
}

#[cfg(target_os = "macos")]
fn crop_direct_store_step_to_output_region(
    store: &mut J2kDirectStoreStep,
    region: Rect,
) -> Result<(), Error> {
    let store_bounds = (
        store.output_x,
        store.output_y,
        checked_rect_end(store.output_x, store.copy_width, "store width")?,
        checked_rect_end(store.output_y, store.copy_height, "store height")?,
    );
    let region_bounds = (
        region.x,
        region.y,
        checked_rect_end(region.x, region.w, "ROI width")?,
        checked_rect_end(region.y, region.h, "ROI height")?,
    );
    let intersection = (
        store_bounds.0.max(region_bounds.0),
        store_bounds.1.max(region_bounds.1),
        store_bounds.2.min(region_bounds.2),
        store_bounds.3.min(region_bounds.3),
    );
    if intersection.0 >= intersection.2 || intersection.1 >= intersection.3 {
        return Err(Error::MetalKernel {
            message:
                "J2K MetalDirect region-scaled ROI does not intersect the decoded store window"
                    .to_string(),
        });
    }

    let source_delta = (
        intersection.0 - store_bounds.0,
        intersection.1 - store_bounds.1,
    );
    store.source_x =
        store
            .source_x
            .checked_add(source_delta.0)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect region-scaled source x overflows u32".to_string(),
            })?;
    store.source_y =
        store
            .source_y
            .checked_add(source_delta.1)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect region-scaled source y overflows u32".to_string(),
            })?;
    store.copy_width = intersection.2 - intersection.0;
    store.copy_height = intersection.3 - intersection.1;
    store.output_width = region.w;
    store.output_height = region.h;
    store.output_x = intersection.0 - region_bounds.0;
    store.output_y = intersection.1 - region_bounds.1;
    Ok(())
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_color_plan(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_direct_color_plan_for_cpu_upload(
    plan: &J2kDirectColorPlan,
) -> Result<PreparedDirectColorPlan, Error> {
    prepare_direct_color_plan_with_tier1_mode(plan, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn prepare_direct_color_plan_with_tier1_mode(
    plan: &J2kDirectColorPlan,
    tier1_prepare_mode: DirectTier1Mode,
) -> Result<PreparedDirectColorPlan, Error> {
    let component_plans = plan
        .component_plans
        .iter()
        .map(|component| match tier1_prepare_mode {
            DirectTier1Mode::Metal => prepare_direct_grayscale_plan(component),
            DirectTier1Mode::CpuUpload => prepare_direct_grayscale_plan_for_cpu_upload(component),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color plan expected 3 component plans, got {}",
                component_plans.len()
            ),
        });
    }
    Ok(PreparedDirectColorPlan {
        dimensions: plan.dimensions,
        bit_depths: plan.bit_depths,
        mct: plan.mct,
        transform: plan.transform,
        component_plans,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn crop_prepared_direct_color_plan_to_output_region(
    plan: &mut PreparedDirectColorPlan,
    region: Rect,
) -> Result<(), Error> {
    if region.w == 0 || region.h == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect region-scaled color plan has an empty output region"
                .to_string(),
        });
    }

    for component_plan in &mut plan.component_plans {
        crop_prepared_direct_grayscale_plan_to_output_region(component_plan, region)?;
        if component_plan.dimensions != (region.w, region.h) {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K MetalDirect color component crop produced {:?}, expected {:?}",
                    component_plan.dimensions,
                    (region.w, region.h)
                ),
            });
        }
    }

    plan.dimensions = (region.w, region.h);
    Ok(())
}

#[cfg(target_os = "macos")]
impl PreparedDirectGrayscalePlan {
    fn classic_group_starting_at(&self, step_idx: usize) -> Option<&PreparedClassicSubBandGroup> {
        self.classic_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }

    fn ht_group_starting_at(&self, step_idx: usize) -> Option<&PreparedHtSubBandGroup> {
        self.ht_groups
            .iter()
            .find(|group| group.start_step == step_idx)
    }

    fn cached_cpu_tier1_coefficients(
        &self,
        step_idx: usize,
        output_len: usize,
    ) -> Result<Option<Vec<f32>>, Error> {
        let key = CpuTier1CoefficientCacheKey {
            step_idx,
            output_len,
        };
        let entries = self
            .cpu_tier1_cache
            .entries
            .lock()
            .map_err(|_| Error::MetalKernel {
                message: "J2K MetalDirect hybrid CPU Tier-1 cache lock is poisoned".to_string(),
            })?;
        Ok(entries.get(&key).map(|coefficients| coefficients.to_vec()))
    }

    fn store_cpu_tier1_coefficients(
        &self,
        step_idx: usize,
        output_len: usize,
        coefficients: Vec<f32>,
    ) -> Result<Vec<f32>, Error> {
        let key = CpuTier1CoefficientCacheKey {
            step_idx,
            output_len,
        };
        let cached = Arc::<[f32]>::from(coefficients.clone());
        let mut entries = self
            .cpu_tier1_cache
            .entries
            .lock()
            .map_err(|_| Error::MetalKernel {
                message: "J2K MetalDirect hybrid CPU Tier-1 cache lock is poisoned".to_string(),
            })?;
        entries.insert(key, cached);
        Ok(coefficients)
    }

    fn clear_cpu_tier1_cache(&self) -> Result<(), Error> {
        let mut entries = self
            .cpu_tier1_cache
            .entries
            .lock()
            .map_err(|_| Error::MetalKernel {
                message: "J2K MetalDirect hybrid CPU Tier-1 cache lock is poisoned".to_string(),
            })?;
        entries.clear();
        Ok(())
    }
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_direct_grayscale_plan_compute_encoder_count(
    plan: &PreparedDirectGrayscalePlan,
    _fmt: PixelFormat,
) -> usize {
    usize::from(!plan.steps.is_empty())
}

#[cfg(all(test, target_os = "macos"))]
fn prepared_repeated_direct_ht_cleanup_dispatch_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    let mut dispatches = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            dispatches += 1;
            step_idx = group.end_step;
            continue;
        }
        if matches!(
            plan.steps[step_idx],
            PreparedDirectGrayscaleStep::HtSubBand(_)
        ) {
            dispatches += 1;
        }
        step_idx += 1;
    }
    dispatches
}

#[cfg(target_os = "macos")]
fn encode_prepared_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_surface = None;
        let mut step_idx = 0;

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients);
                let (buffers, status_check) =
                    encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let output = take_f32_scratch_buffer(runtime, group.total_coefficients);
                let (buffers, status_check) =
                    encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        group,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                scratch_buffers.push(output);
                step_idx = group.end_step;
                continue;
            }

            let step = &plan.steps[step_idx];
            match step {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    );
                    let (buffers, status_check) =
                        encode_prepared_classic_sub_band_to_buffer_in_encoder(
                            runtime,
                            encoder,
                            sub_band,
                            &output.buffer,
                            scratch_buffers,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let output = take_f32_scratch_buffer(
                        runtime,
                        sub_band.width as usize * sub_band.height as usize,
                    );
                    let (buffers, status_check) = encode_prepared_ht_sub_band_to_buffer_in_encoder(
                        runtime,
                        encoder,
                        sub_band,
                        &output.buffer,
                    )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let ll =
                        lookup_direct_band_slice_entry(&bands, idwt.step.ll_band_id, idwt.step.ll)?;
                    let hl =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hl_band_id, idwt.step.hl)?;
                    let lh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.lh_band_id, idwt.step.lh)?;
                    let hh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hh_band_id, idwt.step.hh)?;
                    let params = prepared_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    );
                    let output = take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt));
                    match idwt.step.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                runtime,
                                encoder,
                                &ll.buffer,
                                ll.offset_bytes,
                                &hl.buffer,
                                hl.offset_bytes,
                                &lh.buffer,
                                lh.offset_bytes,
                                &hh.buffer,
                                hh.offset_bytes,
                                params,
                                &output.buffer,
                                0,
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            let status_check =
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    runtime,
                                    encoder,
                                    &ll.buffer,
                                    ll.offset_bytes,
                                    &hl.buffer,
                                    hl.offset_bytes,
                                    &lh.buffer,
                                    lh.offset_bytes,
                                    &hh.buffer,
                                    hh.offset_bytes,
                                    params,
                                    &output.buffer,
                                    0,
                                );
                            status_checks.push(status_check);
                        }
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: idwt.output_window,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        final_surface = Some(encode_gray_store_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            J2kGrayStoreParams {
                                input_width: store.input_rect.width(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                        )?);
                    } else {
                        let output = take_f32_scratch_buffer(
                            runtime,
                            store.output_width as usize * store.output_height as usize,
                        );
                        let params = J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        };
                        dispatch_store_component_buffer_in_encoder_with_offsets(
                            runtime,
                            encoder,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        final_surface = Some(encode_gray_plane_to_surface_in_encoder(
                            runtime,
                            encoder,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
            step_idx += 1;
        }

        final_surface.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect prepared grayscale plan did not produce a final stored plane"
                .to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(all(target_os = "macos", test))]
fn decode_prepared_classic_sub_band_on_cpu(
    sub_band: &PreparedClassicSubBand,
) -> Result<Vec<f32>, Error> {
    decode_prepared_classic_sub_band_on_cpu_profile(sub_band, None)
}

#[cfg(target_os = "macos")]
fn decode_prepared_classic_sub_band_on_cpu_profile(
    sub_band: &PreparedClassicSubBand,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let len = checked_coefficient_len(
        sub_band.width,
        sub_band.height,
        "classic J2K MetalDirect hybrid sub-band size overflow",
    )?;
    let mut output = vec![0.0_f32; len];
    if let Some(counters) = profile_counters {
        let mut scratch = ClassicCpuDecodeScratch::default();
        decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
            &sub_band.coded_data,
            &sub_band.segments,
            &sub_band.jobs,
            &mut output,
            &mut scratch,
            counters,
        )?;
    } else {
        decode_prepared_classic_jobs_on_cpu(
            &sub_band.coded_data,
            &sub_band.segments,
            &sub_band.jobs,
            &mut output,
        )?;
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn decode_prepared_classic_sub_band_group_on_cpu_profile(
    group: &PreparedClassicSubBandGroup,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let mut output = vec![0.0_f32; group.total_coefficients];
    if let Some(counters) = profile_counters {
        let mut scratch = ClassicCpuDecodeScratch::default();
        decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
            &group.coded_data,
            &group.segments,
            &group.jobs,
            &mut output,
            &mut scratch,
            counters,
        )?;
    } else {
        decode_prepared_classic_jobs_on_cpu(
            &group.coded_data,
            &group.segments,
            &group.jobs,
            &mut output,
        )?;
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct ClassicCpuDecodeScratch {
    segments: Vec<J2kCodeBlockSegment>,
    decode: J2kCodeBlockDecodeWorkspace,
}

#[cfg(target_os = "macos")]
fn decode_prepared_classic_jobs_on_cpu(
    coded_data: &[u8],
    segments: &[J2kClassicSegment],
    jobs: &[J2kClassicCleanupBatchJob],
    output: &mut [f32],
) -> Result<(), Error> {
    let mut scratch = ClassicCpuDecodeScratch::default();
    decode_prepared_classic_jobs_on_cpu_with_scratch(
        coded_data,
        segments,
        jobs,
        output,
        &mut scratch,
    )
}

#[cfg(target_os = "macos")]
fn decode_prepared_classic_jobs_on_cpu_with_scratch(
    coded_data: &[u8],
    segments: &[J2kClassicSegment],
    jobs: &[J2kClassicCleanupBatchJob],
    output: &mut [f32],
    scratch: &mut ClassicCpuDecodeScratch,
) -> Result<(), Error> {
    decode_prepared_classic_jobs_on_cpu_with_scratch_impl::<false>(
        coded_data, segments, jobs, output, scratch, None,
    )
}

#[cfg(target_os = "macos")]
fn decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
    coded_data: &[u8],
    segments: &[J2kClassicSegment],
    jobs: &[J2kClassicCleanupBatchJob],
    output: &mut [f32],
    scratch: &mut ClassicCpuDecodeScratch,
    profile_counters: &CpuTier1DecodeSubstageCounters,
) -> Result<(), Error> {
    decode_prepared_classic_jobs_on_cpu_with_scratch_impl::<true>(
        coded_data,
        segments,
        jobs,
        output,
        scratch,
        Some(profile_counters),
    )
}

#[cfg(target_os = "macos")]
fn decode_prepared_classic_jobs_on_cpu_with_scratch_impl<const PROFILE: bool>(
    coded_data: &[u8],
    segments: &[J2kClassicSegment],
    jobs: &[J2kClassicCleanupBatchJob],
    output: &mut [f32],
    scratch: &mut ClassicCpuDecodeScratch,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<(), Error> {
    for job in jobs {
        let prep_started = PROFILE.then(Instant::now);
        let start = job.output_offset as usize;
        let segment_window = prepared_classic_segment_window(segments, job)?;
        scratch.segments.clear();
        scratch.segments.reserve(segment_window.len());
        for segment in segment_window {
            scratch.segments.push(prepared_classic_segment(segment)?);
        }
        let decode_job = prepared_classic_decode_job(coded_data, &scratch.segments, job)?;
        let required_len = required_classic_output_len(decode_job)?;
        let end = start
            .checked_add(required_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect hybrid output offset overflow".to_string(),
            })?;
        let Some(output_window) = output.get_mut(start..end) else {
            return Err(Error::MetalKernel {
                message: "classic J2K MetalDirect hybrid output slice is too small".to_string(),
            });
        };
        if let Some(started) = prep_started {
            profile_counters
                .expect("profile counters required for profiled classic decode")
                .record_classic_segment_prep(started);
        }
        if PROFILE {
            let decode_started = Instant::now();
            let mut profile = J2kCodeBlockDecodeProfile::default();
            decode_j2k_code_block_scalar_with_workspace_profiled(
                decode_job,
                output_window,
                &mut scratch.decode,
                &mut profile,
            )
            .map_err(native_decode_error)?;
            profile_counters
                .expect("profile counters required for profiled classic decode")
                .record_classic_block_decode(decode_started, &profile);
        } else {
            decode_j2k_code_block_scalar_with_workspace(
                decode_job,
                output_window,
                &mut scratch.decode,
            )
            .map_err(native_decode_error)?;
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn prepared_classic_segment_window<'a>(
    segments: &'a [J2kClassicSegment],
    job: &J2kClassicCleanupBatchJob,
) -> Result<&'a [J2kClassicSegment], Error> {
    let segment_start = job.segment_offset as usize;
    let segment_end = segment_start
        .checked_add(job.segment_count as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect hybrid segment span overflow".to_string(),
        })?;
    segments
        .get(segment_start..segment_end)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect hybrid segment span is invalid".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn prepared_classic_decode_job<'a>(
    coded_data: &'a [u8],
    segments: &'a [J2kCodeBlockSegment],
    job: &J2kClassicCleanupBatchJob,
) -> Result<J2kCodeBlockDecodeJob<'a>, Error> {
    Ok(J2kCodeBlockDecodeJob {
        data: coded_data,
        segments,
        width: job.width,
        height: job.height,
        output_stride: job.output_stride as usize,
        missing_bit_planes: checked_u8(job.missing_msbs, "classic missing bit planes")?,
        number_of_coding_passes: checked_u8(job.number_of_coding_passes, "classic coding passes")?,
        total_bitplanes: checked_u8(job.total_bitplanes, "classic total bitplanes")?,
        roi_shift: checked_u8(job.roi_shift, "classic ROI shift")?,
        sub_band_type: prepared_classic_sub_band_type(job.sub_band_type)?,
        style: prepared_classic_style(job.style_flags),
        strict: job.strict != 0,
        dequantization_step: job.dequantization_step,
    })
}

#[cfg(target_os = "macos")]
fn prepared_classic_segment(segment: &J2kClassicSegment) -> Result<J2kCodeBlockSegment, Error> {
    Ok(J2kCodeBlockSegment {
        data_offset: segment.data_offset,
        data_length: segment.data_length,
        start_coding_pass: checked_u8(segment.start_coding_pass, "classic segment start pass")?,
        end_coding_pass: checked_u8(segment.end_coding_pass, "classic segment end pass")?,
        use_arithmetic: segment.use_arithmetic != 0,
    })
}

#[cfg(target_os = "macos")]
fn prepared_classic_sub_band_type(value: u32) -> Result<J2kSubBandType, Error> {
    match value {
        0 => Ok(J2kSubBandType::LowLow),
        1 => Ok(J2kSubBandType::HighLow),
        2 => Ok(J2kSubBandType::LowHigh),
        3 => Ok(J2kSubBandType::HighHigh),
        _ => Err(Error::MetalKernel {
            message: format!("classic J2K MetalDirect hybrid sub-band type {value} is invalid"),
        }),
    }
}

#[cfg(target_os = "macos")]
fn prepared_classic_style(flags: u32) -> J2kCodeBlockStyle {
    J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: (flags
            & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS)
            != 0,
        reset_context_probabilities: (flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0,
        termination_on_each_pass: (flags & J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS) != 0,
        vertically_causal_context: (flags & J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT) != 0,
        segmentation_symbols: (flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0,
    }
}

#[cfg(target_os = "macos")]
fn decode_prepared_ht_sub_band_on_cpu_profile(
    sub_band: &PreparedHtSubBand,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let len = checked_coefficient_len(
        sub_band.width,
        sub_band.height,
        "HTJ2K MetalDirect hybrid sub-band size overflow",
    )?;
    let mut output = vec![0.0_f32; len];
    if let Some(counters) = profile_counters {
        let mut workspace = HtCodeBlockDecodeWorkspace::default();
        decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
            &sub_band.coded_data,
            &sub_band.jobs,
            &mut output,
            &mut workspace,
            counters,
        )?;
    } else {
        decode_prepared_ht_jobs_on_cpu(&sub_band.coded_data, &sub_band.jobs, &mut output)?;
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn decode_prepared_ht_sub_band_group_on_cpu_profile(
    group: &PreparedHtSubBandGroup,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let mut output = vec![0.0_f32; group.total_coefficients];
    if let Some(counters) = profile_counters {
        let mut workspace = HtCodeBlockDecodeWorkspace::default();
        decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
            &group.coded_arena.data,
            &group.jobs,
            &mut output,
            &mut workspace,
            counters,
        )?;
    } else {
        decode_prepared_ht_jobs_on_cpu(&group.coded_arena.data, &group.jobs, &mut output)?;
    }
    Ok(output)
}

#[cfg(target_os = "macos")]
fn decode_prepared_ht_jobs_on_cpu(
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    output: &mut [f32],
) -> Result<(), Error> {
    let mut workspace = HtCodeBlockDecodeWorkspace::default();
    decode_prepared_ht_jobs_on_cpu_with_workspace(coded_data, jobs, output, &mut workspace)
}

#[cfg(target_os = "macos")]
fn decode_prepared_ht_jobs_on_cpu_with_workspace(
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<(), Error> {
    decode_prepared_ht_jobs_on_cpu_with_workspace_impl::<false>(
        coded_data, jobs, output, workspace, None,
    )
}

#[cfg(target_os = "macos")]
fn decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile_counters: &CpuTier1DecodeSubstageCounters,
) -> Result<(), Error> {
    decode_prepared_ht_jobs_on_cpu_with_workspace_impl::<true>(
        coded_data,
        jobs,
        output,
        workspace,
        Some(profile_counters),
    )
}

#[cfg(target_os = "macos")]
fn decode_prepared_ht_jobs_on_cpu_with_workspace_impl<const PROFILE: bool>(
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<(), Error> {
    for job in jobs {
        let start = job.output_offset as usize;
        let decode_job = prepared_ht_decode_job(coded_data, job)?;
        let required_len = required_ht_output_len(decode_job)?;
        let end = start
            .checked_add(required_len)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect hybrid output offset overflow".to_string(),
            })?;
        let Some(output_window) = output.get_mut(start..end) else {
            return Err(Error::MetalKernel {
                message: "HTJ2K MetalDirect hybrid output slice is too small".to_string(),
            });
        };
        if PROFILE {
            let decode_started = Instant::now();
            let mut profile = HtCodeBlockDecodeProfile::default();
            decode_ht_code_block_scalar_with_workspace_profiled(
                decode_job,
                output_window,
                workspace,
                &mut profile,
            )
            .map_err(native_decode_error)?;
            profile_counters
                .expect("profile counters required for profiled HT decode")
                .record_ht_block_decode(decode_started, &profile);
        } else {
            decode_ht_code_block_scalar_with_workspace(decode_job, output_window, workspace)
                .map_err(native_decode_error)?;
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
struct ClassicCpuDecodeInput<'a> {
    coded_data: &'a [u8],
    segments: &'a [J2kClassicSegment],
    jobs: &'a [J2kClassicCleanupBatchJob],
    output_len: usize,
}

#[cfg(target_os = "macos")]
struct HtCpuDecodeInput<'a> {
    coded_data: &'a [u8],
    jobs: &'a [J2kHtCleanupBatchJob],
    output_len: usize,
}

#[cfg(target_os = "macos")]
fn decode_classic_inputs_on_cpu_parallel(
    inputs: &[ClassicCpuDecodeInput<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    record_hybrid_cpu_decode_inputs(inputs.len());
    let Some(output_len) = packed_cpu_decode_output_len(
        inputs.iter().map(|input| input.output_len),
        "classic J2K MetalDirect hybrid batch",
    )?
    else {
        return Ok(Vec::new());
    };
    let mut coefficients = packed_cpu_decode_coefficients(inputs.len(), output_len)?;
    coefficients
        .par_chunks_mut(output_len)
        .zip(inputs.par_iter())
        .with_min_len(HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK)
        .try_for_each_init(
            || {
                record_hybrid_cpu_decode_worker_init();
                ClassicCpuDecodeScratch::default()
            },
            |scratch, (output, input)| {
                if let Some(counters) = profile_counters {
                    decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
                        input.coded_data,
                        input.segments,
                        input.jobs,
                        output,
                        scratch,
                        counters,
                    )
                } else {
                    decode_prepared_classic_jobs_on_cpu_with_scratch(
                        input.coded_data,
                        input.segments,
                        input.jobs,
                        output,
                        scratch,
                    )
                }
            },
        )?;
    Ok(coefficients)
}

#[cfg(target_os = "macos")]
fn decode_ht_inputs_on_cpu_parallel(
    inputs: &[HtCpuDecodeInput<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    record_hybrid_cpu_decode_inputs(inputs.len());
    let Some(output_len) = packed_cpu_decode_output_len(
        inputs.iter().map(|input| input.output_len),
        "HTJ2K MetalDirect hybrid batch",
    )?
    else {
        return Ok(Vec::new());
    };
    let mut coefficients = packed_cpu_decode_coefficients(inputs.len(), output_len)?;
    coefficients
        .par_chunks_mut(output_len)
        .zip(inputs.par_iter())
        .with_min_len(HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK)
        .try_for_each_init(
            || {
                record_hybrid_cpu_decode_worker_init();
                HtCodeBlockDecodeWorkspace::default()
            },
            |workspace, (output, input)| {
                if let Some(counters) = profile_counters {
                    decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
                        input.coded_data,
                        input.jobs,
                        output,
                        workspace,
                        counters,
                    )
                } else {
                    decode_prepared_ht_jobs_on_cpu_with_workspace(
                        input.coded_data,
                        input.jobs,
                        output,
                        workspace,
                    )
                }
            },
        )?;
    Ok(coefficients)
}

#[cfg(target_os = "macos")]
fn decode_classic_inputs_on_cpu_with_plan_cache(
    plan: &PreparedDirectGrayscalePlan,
    step_idx: usize,
    inputs: &[ClassicCpuDecodeInput<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    if inputs.len() != 1 {
        return decode_classic_inputs_on_cpu_parallel(inputs, profile_counters);
    }

    let output_len = inputs[0].output_len;
    if let Some(coefficients) = plan.cached_cpu_tier1_coefficients(step_idx, output_len)? {
        return Ok(coefficients);
    }

    let coefficients = decode_classic_inputs_on_cpu_parallel(inputs, profile_counters)?;
    plan.store_cpu_tier1_coefficients(step_idx, output_len, coefficients)
}

#[cfg(target_os = "macos")]
fn decode_ht_inputs_on_cpu_with_plan_cache(
    plan: &PreparedDirectGrayscalePlan,
    step_idx: usize,
    inputs: &[HtCpuDecodeInput<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<f32>, Error> {
    if inputs.len() != 1 {
        return decode_ht_inputs_on_cpu_parallel(inputs, profile_counters);
    }

    let output_len = inputs[0].output_len;
    if let Some(coefficients) = plan.cached_cpu_tier1_coefficients(step_idx, output_len)? {
        return Ok(coefficients);
    }

    let coefficients = decode_ht_inputs_on_cpu_parallel(inputs, profile_counters)?;
    plan.store_cpu_tier1_coefficients(step_idx, output_len, coefficients)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FlattenedCpuTier1Key {
    component_idx: usize,
    step_idx: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
enum FlattenedCpuTier1Source<'a> {
    Classic {
        coded_data: &'a [u8],
        segments: &'a [J2kClassicSegment],
        jobs: &'a [J2kClassicCleanupBatchJob],
    },
    Ht {
        coded_data: &'a [u8],
        jobs: &'a [J2kHtCleanupBatchJob],
    },
}

#[cfg(target_os = "macos")]
struct FlattenedCpuTier1BucketSpec<'a> {
    key: FlattenedCpuTier1Key,
    output_len: usize,
    inputs: Vec<FlattenedCpuTier1Source<'a>>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct FlattenedCpuTier1Bucket {
    buffer: Buffer,
    output_len: usize,
    input_count: usize,
}

#[cfg(target_os = "macos")]
struct FlattenedCpuTier1Cache {
    buckets: HashMap<FlattenedCpuTier1Key, FlattenedCpuTier1Bucket>,
}

#[cfg(target_os = "macos")]
impl FlattenedCpuTier1Cache {
    fn buffer_for(
        &self,
        component_idx: usize,
        step_idx: usize,
        output_len: usize,
        input_count: usize,
    ) -> Result<Buffer, Error> {
        let key = FlattenedCpuTier1Key {
            component_idx,
            step_idx,
        };
        let bucket = self.buckets.get(&key).ok_or_else(|| Error::MetalKernel {
            message: format!(
                "J2K MetalDirect flattened hybrid cache is missing component {component_idx} step {step_idx}"
            ),
        })?;
        if bucket.output_len != output_len || bucket.input_count != input_count {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K MetalDirect flattened hybrid cache shape mismatch at component {component_idx} step {step_idx}"
                ),
            });
        }
        Ok(bucket.buffer.clone())
    }
}

#[cfg(target_os = "macos")]
struct FlattenedCpuTier1WorkItem<'a> {
    output_len: usize,
    output: FlattenedCpuTier1Output,
    source: FlattenedCpuTier1Source<'a>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct FlattenedCpuTier1Output(*mut f32);

// SAFETY: Work items are constructed from non-overlapping ranges in preallocated
// packed coefficient slabs. Each pointer is written exactly once before the
// owning Vec is moved or exposed again.
#[cfg(target_os = "macos")]
unsafe impl Send for FlattenedCpuTier1Output {}

#[cfg(target_os = "macos")]
unsafe impl Sync for FlattenedCpuTier1Output {}

#[cfg(target_os = "macos")]
impl FlattenedCpuTier1Output {
    unsafe fn as_slice_mut<'a>(self, len: usize) -> &'a mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0, len) }
    }
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct FlattenedCpuTier1DecodeScratch {
    classic: ClassicCpuDecodeScratch,
    ht: HtCodeBlockDecodeWorkspace,
}

#[cfg(target_os = "macos")]
fn build_flattened_cpu_tier1_cache(
    runtime: &MetalRuntime,
    plans: &[Arc<PreparedDirectColorPlan>],
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
) -> Result<FlattenedCpuTier1Cache, Error> {
    let specs = collect_flattened_cpu_tier1_bucket_specs(plans)?;
    stage_timings.cpu_tier1_flattened_batches =
        stage_timings.cpu_tier1_flattened_batches.saturating_add(1);
    let decode_started = metal_profile_stages_enabled().then(Instant::now);
    let cpu_tier1_counters =
        metal_profile_stages_enabled().then(CpuTier1DecodeSubstageCounters::default);
    let decoded_buckets = decode_flattened_cpu_tier1_buckets(&specs, cpu_tier1_counters.as_ref())?;
    if let Some(started) = decode_started {
        stage_timings.cpu_tier1 += elapsed_us(started);
    }
    if let Some(counters) = &cpu_tier1_counters {
        counters.add_to_stage_timings(stage_timings);
    }

    let upload_started = metal_profile_stages_enabled().then(Instant::now);
    let mut buckets = HashMap::with_capacity(specs.len());
    for (spec, coefficients) in specs.iter().zip(decoded_buckets) {
        let input_count = spec.inputs.len();
        let buffer = upload_cpu_decoded_coefficients(
            runtime,
            coefficients,
            retained_buffers,
            retained_cpu_coefficients,
        );
        buckets.insert(
            spec.key,
            FlattenedCpuTier1Bucket {
                buffer,
                output_len: spec.output_len,
                input_count,
            },
        );
    }
    if let Some(started) = upload_started {
        stage_timings.coefficient_upload += elapsed_us(started);
    }

    Ok(FlattenedCpuTier1Cache { buckets })
}

#[cfg(target_os = "macos")]
fn collect_flattened_cpu_tier1_bucket_specs(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> Result<Vec<FlattenedCpuTier1BucketSpec<'_>>, Error> {
    let Some(first) = plans.first() else {
        return Ok(Vec::new());
    };
    let mut specs = Vec::new();

    for component_idx in 0..3 {
        let component_plans = plans
            .iter()
            .map(|plan| &plan.component_plans[component_idx])
            .collect::<Vec<_>>();
        let Some(first_component) = component_plans.first().copied() else {
            continue;
        };
        let broadcast_tier1_inputs = component_plans
            .iter()
            .all(|plan| std::ptr::eq(*plan, first_component));
        let mut step_idx = 0;
        while step_idx < first.component_plans[component_idx].steps.len() {
            if let Some(group) = first_component.classic_group_starting_at(step_idx) {
                let inputs = component_plans
                    .iter()
                    .take(if broadcast_tier1_inputs {
                        1
                    } else {
                        component_plans.len()
                    })
                    .map(|plan| {
                        let group = plan.classic_group_starting_at(step_idx).ok_or_else(|| {
                            Error::MetalKernel {
                                message: "J2K MetalDirect flattened hybrid missing classic group"
                                    .to_string(),
                            }
                        })?;
                        Ok(FlattenedCpuTier1Source::Classic {
                            coded_data: &group.coded_data,
                            segments: &group.segments,
                            jobs: &group.jobs,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                specs.push(FlattenedCpuTier1BucketSpec {
                    key: FlattenedCpuTier1Key {
                        component_idx,
                        step_idx,
                    },
                    output_len: group.total_coefficients,
                    inputs,
                });
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = first_component.ht_group_starting_at(step_idx) {
                let inputs = component_plans
                    .iter()
                    .take(if broadcast_tier1_inputs {
                        1
                    } else {
                        component_plans.len()
                    })
                    .map(|plan| {
                        let group = plan.ht_group_starting_at(step_idx).ok_or_else(|| {
                            Error::MetalKernel {
                                message: "J2K MetalDirect flattened hybrid missing HT group"
                                    .to_string(),
                            }
                        })?;
                        Ok(FlattenedCpuTier1Source::Ht {
                            coded_data: &group.coded_arena.data,
                            jobs: &group.jobs,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                specs.push(FlattenedCpuTier1BucketSpec {
                    key: FlattenedCpuTier1Key {
                        component_idx,
                        step_idx,
                    },
                    output_len: group.total_coefficients,
                    inputs,
                });
                step_idx = group.end_step;
                continue;
            }

            match &first_component.steps[step_idx] {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let output_len = checked_coefficient_len(
                        sub_band.width,
                        sub_band.height,
                        "classic J2K MetalDirect flattened hybrid sub-band size overflow",
                    )?;
                    let inputs = component_plans
                        .iter()
                        .take(if broadcast_tier1_inputs {
                            1
                        } else {
                            component_plans.len()
                        })
                        .map(|plan| match &plan.steps[step_idx] {
                            PreparedDirectGrayscaleStep::ClassicSubBand(other) => {
                                Ok(FlattenedCpuTier1Source::Classic {
                                    coded_data: &other.coded_data,
                                    segments: &other.segments,
                                    jobs: &other.jobs,
                                })
                            }
                            _ => Err(Error::MetalKernel {
                                message:
                                    "J2K MetalDirect flattened hybrid missing classic sub-band"
                                        .to_string(),
                            }),
                        })
                        .collect::<Result<Vec<_>, Error>>()?;
                    specs.push(FlattenedCpuTier1BucketSpec {
                        key: FlattenedCpuTier1Key {
                            component_idx,
                            step_idx,
                        },
                        output_len,
                        inputs,
                    });
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let output_len = checked_coefficient_len(
                        sub_band.width,
                        sub_band.height,
                        "HTJ2K MetalDirect flattened hybrid sub-band size overflow",
                    )?;
                    let inputs = component_plans
                        .iter()
                        .take(if broadcast_tier1_inputs {
                            1
                        } else {
                            component_plans.len()
                        })
                        .map(|plan| match &plan.steps[step_idx] {
                            PreparedDirectGrayscaleStep::HtSubBand(other) => {
                                Ok(FlattenedCpuTier1Source::Ht {
                                    coded_data: &other.coded_data,
                                    jobs: &other.jobs,
                                })
                            }
                            _ => Err(Error::MetalKernel {
                                message: "J2K MetalDirect flattened hybrid missing HT sub-band"
                                    .to_string(),
                            }),
                        })
                        .collect::<Result<Vec<_>, Error>>()?;
                    specs.push(FlattenedCpuTier1BucketSpec {
                        key: FlattenedCpuTier1Key {
                            component_idx,
                            step_idx,
                        },
                        output_len,
                        inputs,
                    });
                }
                PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => {}
            }
            step_idx += 1;
        }
    }

    Ok(specs)
}

#[cfg(target_os = "macos")]
fn decode_flattened_cpu_tier1_buckets(
    specs: &[FlattenedCpuTier1BucketSpec<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<Vec<Vec<f32>>, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_CPU_TIER1);
    let mut buckets = specs
        .iter()
        .map(|spec| packed_cpu_decode_coefficients(spec.inputs.len(), spec.output_len))
        .collect::<Result<Vec<_>, Error>>()?;
    let mut work_items = Vec::new();
    for (bucket_idx, spec) in specs.iter().enumerate() {
        for (input_idx, source) in spec.inputs.iter().copied().enumerate() {
            let start =
                input_idx
                    .checked_mul(spec.output_len)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K MetalDirect flattened hybrid bucket offset overflow"
                            .to_string(),
                    })?;
            let end = start
                .checked_add(spec.output_len)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K MetalDirect flattened hybrid bucket end overflow".to_string(),
                })?;
            if end > buckets[bucket_idx].len() {
                return Err(Error::MetalKernel {
                    message: "J2K MetalDirect flattened hybrid bucket slice is too small"
                        .to_string(),
                });
            }
            let output =
                FlattenedCpuTier1Output(unsafe { buckets[bucket_idx].as_mut_ptr().add(start) });
            work_items.push(FlattenedCpuTier1WorkItem {
                output_len: spec.output_len,
                output,
                source,
            });
        }
    }

    record_flattened_hybrid_cpu_decode_batch();
    record_hybrid_cpu_decode_inputs(work_items.len());

    decode_flattened_cpu_tier1_work_items_chunked(&work_items, profile_counters)?;

    Ok(buckets)
}

#[cfg(target_os = "macos")]
fn decode_flattened_cpu_tier1_work_items_chunked(
    work_items: &[FlattenedCpuTier1WorkItem<'_>],
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<(), Error> {
    if work_items.is_empty() {
        return Ok(());
    }

    let worker_count = hybrid_cpu_decode_worker_count(work_items.len());
    let chunk_size = work_items.len().div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for chunk in work_items.chunks(chunk_size) {
            handles.push(scope.spawn(move || {
                record_hybrid_cpu_decode_worker_init();
                let mut scratch = FlattenedCpuTier1DecodeScratch::default();
                for item in chunk {
                    decode_flattened_cpu_tier1_work_item(item, &mut scratch, profile_counters)?;
                }
                Ok(())
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(error),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(())
    })
}

#[cfg(target_os = "macos")]
fn hybrid_cpu_decode_worker_count(item_count: usize) -> usize {
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let useful = item_count
        .div_ceil(HYBRID_CPU_DECODE_MIN_INPUTS_PER_TASK.max(1))
        .max(1);
    available.min(useful).max(1)
}

#[cfg(target_os = "macos")]
fn decode_flattened_cpu_tier1_work_item(
    item: &FlattenedCpuTier1WorkItem<'_>,
    scratch: &mut FlattenedCpuTier1DecodeScratch,
    profile_counters: Option<&CpuTier1DecodeSubstageCounters>,
) -> Result<(), Error> {
    let output = unsafe { item.output.as_slice_mut(item.output_len) };
    match item.source {
        FlattenedCpuTier1Source::Classic {
            coded_data,
            segments,
            jobs,
        } => {
            if let Some(counters) = profile_counters {
                decode_prepared_classic_jobs_on_cpu_with_scratch_profiled(
                    coded_data,
                    segments,
                    jobs,
                    output,
                    &mut scratch.classic,
                    counters,
                )
            } else {
                decode_prepared_classic_jobs_on_cpu_with_scratch(
                    coded_data,
                    segments,
                    jobs,
                    output,
                    &mut scratch.classic,
                )
            }
        }
        FlattenedCpuTier1Source::Ht { coded_data, jobs } => {
            if let Some(counters) = profile_counters {
                decode_prepared_ht_jobs_on_cpu_with_workspace_profiled(
                    coded_data,
                    jobs,
                    output,
                    &mut scratch.ht,
                    counters,
                )
            } else {
                decode_prepared_ht_jobs_on_cpu_with_workspace(
                    coded_data,
                    jobs,
                    output,
                    &mut scratch.ht,
                )
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn packed_cpu_decode_output_len(
    output_lens: impl IntoIterator<Item = usize>,
    label: &str,
) -> Result<Option<usize>, Error> {
    let mut output_lens = output_lens.into_iter();
    let Some(output_len) = output_lens.next() else {
        return Ok(None);
    };
    if output_len == 0 {
        return Ok(None);
    }
    if output_lens.any(|other| other != output_len) {
        return Err(Error::MetalKernel {
            message: format!("{label} has mixed coefficient lengths"),
        });
    }
    Ok(Some(output_len))
}

#[cfg(target_os = "macos")]
fn packed_cpu_decode_coefficients(count: usize, output_len: usize) -> Result<Vec<f32>, Error> {
    let total_len = count
        .checked_mul(output_len)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect hybrid packed coefficient length overflows usize".to_string(),
        })?;
    Ok(vec![0.0_f32; total_len])
}

#[cfg(target_os = "macos")]
fn flattened_hybrid_cpu_tier1_enabled() -> bool {
    std::env::var_os(HYBRID_FLAT_CPU_TIER1_ENV).is_some_and(|value| {
        let value = value.to_string_lossy();
        !value.is_empty() && value != "0" && value != "false"
    })
}

#[cfg(target_os = "macos")]
fn should_flatten_hybrid_cpu_tier1_color_batch(plans: &[Arc<PreparedDirectColorPlan>]) -> bool {
    let Some(first) = plans.first() else {
        return false;
    };
    plans.len() >= HYBRID_FLAT_CPU_TIER1_MIN_COUNT
        && first.dimensions.0.max(first.dimensions.1) >= HYBRID_FLAT_CPU_TIER1_MIN_DIM
        && !plans.iter().all(|plan| Arc::ptr_eq(plan, first))
}

#[cfg(target_os = "macos")]
fn prepared_ht_decode_job<'a>(
    coded_data: &'a [u8],
    job: &J2kHtCleanupBatchJob,
) -> Result<HtCodeBlockDecodeJob<'a>, Error> {
    let start = job.coded_offset as usize;
    let len = job.coded_len as usize;
    let end = start.checked_add(len).ok_or_else(|| Error::MetalKernel {
        message: "HTJ2K MetalDirect hybrid coded span overflow".to_string(),
    })?;
    let Some(data) = coded_data.get(start..end) else {
        return Err(Error::MetalKernel {
            message: "HTJ2K MetalDirect hybrid coded span is invalid".to_string(),
        });
    };

    Ok(HtCodeBlockDecodeJob {
        data,
        cleanup_length: job.cleanup_length,
        refinement_length: job.refinement_length,
        width: job.width,
        height: job.height,
        output_stride: job.output_stride as usize,
        missing_bit_planes: checked_u8(job.missing_msbs, "HTJ2K missing bit planes")?,
        number_of_coding_passes: checked_u8(job.number_of_coding_passes, "HTJ2K coding passes")?,
        num_bitplanes: checked_u8(job.num_bitplanes, "HTJ2K total bitplanes")?,
        roi_shift: checked_u8(job.roi_shift, "HTJ2K ROI shift")?,
        stripe_causal: job.stripe_causal != 0,
        strict: true,
        dequantization_step: job.dequantization_step,
    })
}

#[cfg(target_os = "macos")]
fn checked_coefficient_len(width: u32, height: u32, message: &str) -> Result<usize, Error> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: message.to_string(),
        })
}

#[cfg(target_os = "macos")]
fn checked_u8(value: u32, label: &str) -> Result<u8, Error> {
    u8::try_from(value).map_err(|_| Error::MetalKernel {
        message: format!("J2K MetalDirect hybrid {label} exceeds u8"),
    })
}

#[cfg(target_os = "macos")]
fn native_decode_error(error: signinum_j2k_native::DecodeError) -> Error {
    Error::Decode(signinum_j2k::J2kError::Backend(error.to_string()))
}

#[cfg(target_os = "macos")]
fn upload_cpu_decoded_coefficients(
    runtime: &MetalRuntime,
    mut coefficients: Vec<f32>,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
) -> Buffer {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COEFFICIENT_UPLOAD);
    let buffer = borrow_mut_slice_buffer(&runtime.device, &mut coefficients);
    retained_buffers.push(buffer.clone());
    retained_cpu_coefficients.push(coefficients);
    buffer
}

#[cfg(target_os = "macos")]
fn elapsed_us(started: Instant) -> u128 {
    started.elapsed().as_micros()
}

#[cfg(target_os = "macos")]
fn elapsed_us_u64(started: Instant) -> u64 {
    elapsed_us(started).min(u128::from(u64::MAX)) as u64
}

#[cfg(target_os = "macos")]
fn emit_direct_hybrid_stage_timings(
    timings: &DirectHybridStageTimings,
    fmt: PixelFormat,
    batch_count: usize,
) {
    if !metal_profile_stages_enabled() {
        return;
    }

    let fmt_s = format!("{fmt:?}");
    let batch_count_s = batch_count.to_string();
    let label = decode_profile_label();
    for (stage, elapsed_us) in [
        ("cpu_tier1", timings.cpu_tier1),
        (
            "cpu_tier1_flattened_batches",
            timings.cpu_tier1_flattened_batches,
        ),
        (
            "cpu_tier1_classic_segment_prep",
            timings.cpu_tier1_classic_segment_prep,
        ),
        (
            "cpu_tier1_classic_block_decode",
            timings.cpu_tier1_classic_block_decode,
        ),
        (
            "cpu_tier1_classic_sigprop",
            timings.cpu_tier1_classic_sigprop,
        ),
        ("cpu_tier1_classic_magref", timings.cpu_tier1_classic_magref),
        (
            "cpu_tier1_classic_cleanup",
            timings.cpu_tier1_classic_cleanup,
        ),
        ("cpu_tier1_classic_bypass", timings.cpu_tier1_classic_bypass),
        (
            "cpu_tier1_classic_output_convert",
            timings.cpu_tier1_classic_output_convert,
        ),
        (
            "cpu_tier1_ht_block_decode",
            timings.cpu_tier1_ht_block_decode,
        ),
        ("cpu_tier1_ht_cleanup", timings.cpu_tier1_ht_cleanup),
        ("cpu_tier1_ht_mag_sgn", timings.cpu_tier1_ht_mag_sgn),
        ("cpu_tier1_ht_sigma", timings.cpu_tier1_ht_sigma),
        ("cpu_tier1_ht_sigprop", timings.cpu_tier1_ht_sigprop),
        ("cpu_tier1_ht_magref", timings.cpu_tier1_ht_magref),
        ("coefficient_upload", timings.coefficient_upload),
        ("metal_idwt_encode", timings.metal_idwt_encode),
        ("metal_store_encode", timings.metal_store_encode),
        ("metal_mct_pack_encode", timings.metal_mct_pack_encode),
        ("command_wait", timings.command_wait),
        ("gpu_command", timings.gpu_command),
        ("metal_idwt_gpu", timings.metal_idwt_gpu),
        (
            "metal_idwt_interleave_gpu",
            timings.metal_idwt_interleave_gpu,
        ),
        (
            "metal_idwt_horizontal_gpu",
            timings.metal_idwt_horizontal_gpu,
        ),
        ("metal_idwt_vertical_gpu", timings.metal_idwt_vertical_gpu),
        ("metal_store_gpu", timings.metal_store_gpu),
        ("metal_mct_pack_gpu", timings.metal_mct_pack_gpu),
    ] {
        let elapsed_us_s = elapsed_us.to_string();
        let processor = stage_processor(stage);
        let metric = stage_metric(stage);
        let metric_kind = stage_metric_kind(stage);
        let aggregation = stage_aggregation(stage);
        eprintln!(
            "signinum_profile codec=j2k op=decode path=metal_cpu_hybrid pipeline=decode_hybrid label={label} stage={stage} processor={processor} metric={metric} metric_kind={metric_kind} aggregation={aggregation} fmt={fmt_s} batch_count={batch_count_s} elapsed_us={elapsed_us_s}"
        );
    }
}

#[cfg(target_os = "macos")]
fn stage_processor(stage: &str) -> &'static str {
    match stage {
        "cpu_tier1_flattened_batches" => "scheduler",
        "cpu_tier1"
        | "cpu_tier1_classic_segment_prep"
        | "cpu_tier1_classic_block_decode"
        | "cpu_tier1_classic_sigprop"
        | "cpu_tier1_classic_magref"
        | "cpu_tier1_classic_cleanup"
        | "cpu_tier1_classic_bypass"
        | "cpu_tier1_classic_output_convert"
        | "cpu_tier1_ht_block_decode"
        | "cpu_tier1_ht_cleanup"
        | "cpu_tier1_ht_mag_sgn"
        | "cpu_tier1_ht_sigma"
        | "cpu_tier1_ht_sigprop"
        | "cpu_tier1_ht_magref" => "cpu",
        "coefficient_upload" => "transfer",
        "metal_idwt_encode"
        | "metal_store_encode"
        | "metal_mct_pack_encode"
        | "gpu_command"
        | "metal_idwt_gpu"
        | "metal_idwt_interleave_gpu"
        | "metal_idwt_horizontal_gpu"
        | "metal_idwt_vertical_gpu"
        | "metal_store_gpu"
        | "metal_mct_pack_gpu" => "metal",
        "command_wait" => "wait",
        _ => "hybrid",
    }
}

#[cfg(target_os = "macos")]
fn stage_metric(stage: &str) -> &'static str {
    match stage {
        "cpu_tier1_flattened_batches" => "count",
        "cpu_tier1_classic_segment_prep"
        | "cpu_tier1_classic_block_decode"
        | "cpu_tier1_classic_sigprop"
        | "cpu_tier1_classic_magref"
        | "cpu_tier1_classic_cleanup"
        | "cpu_tier1_classic_bypass"
        | "cpu_tier1_classic_output_convert"
        | "cpu_tier1_ht_block_decode"
        | "cpu_tier1_ht_cleanup"
        | "cpu_tier1_ht_mag_sgn"
        | "cpu_tier1_ht_sigma"
        | "cpu_tier1_ht_sigprop"
        | "cpu_tier1_ht_magref" => "cpu_worker_us",
        "gpu_command"
        | "metal_idwt_gpu"
        | "metal_idwt_interleave_gpu"
        | "metal_idwt_horizontal_gpu"
        | "metal_idwt_vertical_gpu"
        | "metal_store_gpu"
        | "metal_mct_pack_gpu" => "gpu_elapsed_us",
        _ => "wall_us",
    }
}

#[cfg(target_os = "macos")]
fn stage_metric_kind(stage: &str) -> &'static str {
    match stage_metric(stage) {
        "count" => "counter",
        "cpu_worker_us" => "cpu_worker_sum",
        "gpu_elapsed_us" => "gpu_busy_sum",
        _ => "wall_elapsed",
    }
}

#[cfg(target_os = "macos")]
fn stage_aggregation(stage: &str) -> &'static str {
    match stage_metric(stage) {
        "count" | "cpu_worker_us" | "gpu_elapsed_us" => "sum",
        _ => "exclusive",
    }
}

#[cfg(all(target_os = "macos", test))]
fn record_hybrid_stacked_component_batch(tier1_mode: DirectTier1Mode) {
    if tier1_mode == DirectTier1Mode::CpuUpload {
        HYBRID_STACKED_COMPONENT_BATCHES.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(all(target_os = "macos", not(test)))]
fn record_hybrid_stacked_component_batch(_tier1_mode: DirectTier1Mode) {}

#[cfg(all(target_os = "macos", test))]
fn record_hybrid_repeated_output_blit() {
    HYBRID_REPEATED_OUTPUT_BLITS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", not(test)))]
fn record_hybrid_repeated_output_blit() {}

#[cfg(all(target_os = "macos", test))]
fn record_hybrid_cpu_decode_worker_init() {
    HYBRID_CPU_DECODE_WORKER_INITS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", not(test)))]
fn record_hybrid_cpu_decode_worker_init() {}

#[cfg(all(target_os = "macos", test))]
fn record_hybrid_cpu_decode_inputs(count: usize) {
    HYBRID_CPU_DECODE_INPUTS.fetch_add(count, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", not(test)))]
fn record_hybrid_cpu_decode_inputs(_count: usize) {}

#[cfg(all(target_os = "macos", test))]
fn record_flattened_hybrid_cpu_decode_batch() {
    FLATTENED_HYBRID_CPU_DECODE_BATCHES.fetch_add(1, Ordering::Relaxed);
}

#[cfg(all(target_os = "macos", not(test)))]
fn record_flattened_hybrid_cpu_decode_batch() {}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_prepared_direct_component_plane_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Buffer, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = (|| {
        let mut bands = Vec::<DirectBandSlice>::new();
        let mut final_plane = None;
        let mut step_idx = 0;
        let profile_stages = metal_profile_stages_enabled();

        while step_idx < plan.steps.len() {
            if let Some(group) = plan.classic_group_starting_at(step_idx) {
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, group.total_coefficients);
                        let (buffers, status_check) =
                            encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
                                runtime,
                                encoder,
                                group,
                                &output.buffer,
                                scratch_buffers,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_prepared_classic_sub_band_group_on_cpu_profile(
                            group,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                };
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                step_idx = group.end_step;
                continue;
            }

            if let Some(group) = plan.ht_group_starting_at(step_idx) {
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, group.total_coefficients);
                        let (buffers, status_check) =
                            encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
                                runtime,
                                encoder,
                                group,
                                &output.buffer,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_prepared_ht_sub_band_group_on_cpu_profile(
                            group,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                };
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
                step_idx = group.end_step;
                continue;
            }

            match &plan.steps[step_idx] {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                    let buffer = match tier1_mode {
                        DirectTier1Mode::Metal => {
                            let output = take_f32_scratch_buffer(
                                runtime,
                                sub_band.width as usize * sub_band.height as usize,
                            );
                            let (buffers, status_check) =
                                encode_prepared_classic_sub_band_to_buffer_in_encoder(
                                    runtime,
                                    encoder,
                                    sub_band,
                                    &output.buffer,
                                    scratch_buffers,
                                )?;
                            retained_buffers.extend(buffers);
                            status_checks.push(status_check);
                            let buffer = output.buffer.clone();
                            scratch_buffers.push(output);
                            buffer
                        }
                        DirectTier1Mode::CpuUpload => {
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_prepared_classic_sub_band_on_cpu_profile(
                                sub_band,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    };
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer,
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                    let buffer = match tier1_mode {
                        DirectTier1Mode::Metal => {
                            let output = take_f32_scratch_buffer(
                                runtime,
                                sub_band.width as usize * sub_band.height as usize,
                            );
                            let (buffers, status_check) =
                                encode_prepared_ht_sub_band_to_buffer_in_encoder(
                                    runtime,
                                    encoder,
                                    sub_band,
                                    &output.buffer,
                                )?;
                            retained_buffers.extend(buffers);
                            status_checks.push(status_check);
                            let buffer = output.buffer.clone();
                            scratch_buffers.push(output);
                            buffer
                        }
                        DirectTier1Mode::CpuUpload => {
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_prepared_ht_sub_band_on_cpu_profile(
                                sub_band,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    };
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer,
                        offset_bytes: 0,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                PreparedDirectGrayscaleStep::Idwt(idwt) => {
                    let ll =
                        lookup_direct_band_slice_entry(&bands, idwt.step.ll_band_id, idwt.step.ll)?;
                    let hl =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hl_band_id, idwt.step.hl)?;
                    let lh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.lh_band_id, idwt.step.lh)?;
                    let hh =
                        lookup_direct_band_slice_entry(&bands, idwt.step.hh_band_id, idwt.step.hh)?;
                    let params = prepared_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                    );
                    let output = take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt));
                    let encode_started = profile_stages.then(Instant::now);
                    match idwt.step.transform {
                        J2kWaveletTransform::Reversible53 => {
                            dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
                                runtime,
                                encoder,
                                &ll.buffer,
                                ll.offset_bytes,
                                &hl.buffer,
                                hl.offset_bytes,
                                &lh.buffer,
                                lh.offset_bytes,
                                &hh.buffer,
                                hh.offset_bytes,
                                params,
                                &output.buffer,
                                0,
                            );
                        }
                        J2kWaveletTransform::Irreversible97 => {
                            status_checks.push(
                                dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
                                    runtime,
                                    encoder,
                                    &ll.buffer,
                                    ll.offset_bytes,
                                    &hl.buffer,
                                    hl.offset_bytes,
                                    &lh.buffer,
                                    lh.offset_bytes,
                                    &hh.buffer,
                                    hh.offset_bytes,
                                    params,
                                    &output.buffer,
                                    0,
                                ),
                            );
                        }
                    }
                    if let Some(started) = encode_started {
                        stage_timings.metal_idwt_encode += elapsed_us(started);
                    }
                    bands.push(DirectBandSlice {
                        band_id: idwt.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: 0,
                        window: idwt.output_window,
                    });
                    scratch_buffers.push(output);
                }
                PreparedDirectGrayscaleStep::Store(store) => {
                    let (input, input_offset) =
                        lookup_direct_band_slice(&bands, store.input_band_id, store.input_rect)?;
                    let output = take_f32_scratch_buffer(
                        runtime,
                        store.output_width as usize * store.output_height as usize,
                    );
                    let encode_started = profile_stages.then(Instant::now);
                    dispatch_store_component_buffer_in_encoder_with_offsets(
                        runtime,
                        encoder,
                        &input,
                        input_offset,
                        &output.buffer,
                        0,
                        J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        },
                    );
                    if let Some(started) = encode_started {
                        stage_timings.metal_store_encode += elapsed_us(started);
                    }
                    final_plane = Some(output.buffer.clone());
                    scratch_buffers.push(output);
                }
            }
            step_idx += 1;
        }

        final_plane.ok_or_else(|| Error::MetalKernel {
            message: "J2K MetalDirect component plan did not produce a stored plane".to_string(),
        })
    })();
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_repeated_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surfaces = encode_repeated_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            count,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers);
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let surface = encode_prepared_direct_grayscale_plan_in_command_buffer(
            runtime,
            command_buffer,
            plan,
            fmt,
            &mut retained_buffers,
            &mut status_checks,
            &mut scratch_buffers,
        )?;
        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        recycle_scratch_buffers(runtime, scratch_buffers);
        Ok(surface)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_with_device(
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_prepared_direct_grayscale_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_grayscale_plan_batch(
    plans: &[Arc<PreparedDirectGrayscalePlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }

    with_runtime(|runtime| {
        let command_buffer = runtime.queue.new_command_buffer();
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let mut surfaces = Vec::with_capacity(plans.len());

        let component_plan_refs = plans.iter().map(Arc::as_ref).collect::<Vec<_>>();
        if plans.len() > 1 && supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            let stacked_plane = encode_stacked_direct_component_plane_batch(
                runtime,
                DirectColorBatchCommandBuffers::single(command_buffer),
                &component_plan_refs,
                0,
                None,
                DirectTier1Mode::Metal,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )?;
            let first = plans.first().expect("plans is not empty");
            if stacked_plane.dimensions == first.dimensions && stacked_plane.count == plans.len() {
                surfaces = encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                    runtime,
                    command_buffer,
                    &stacked_plane.buffer,
                    first.dimensions,
                    first.bit_depth,
                    fmt,
                    plans.len(),
                )?;
            }
        }

        for plan in plans {
            if !surfaces.is_empty() {
                break;
            }
            surfaces.push(encode_prepared_direct_grayscale_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                &mut retained_buffers,
                &mut status_checks,
                &mut scratch_buffers,
            )?);
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers);
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_prepared_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| execute_prepared_direct_color_plan(plan, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_prepared_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::Metal)
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let plans = [Arc::new(plan.clone())];
    let mut surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)?;
    surfaces.pop().ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect hybrid color plan produced no surface".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_with_device(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        execute_hybrid_cpu_tier1_direct_color_plan(plan, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn execute_hybrid_cpu_tier1_direct_color_plan_batch(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1(plans, fmt, DirectTier1Mode::CpuUpload)
}

#[cfg(target_os = "macos")]
fn execute_direct_color_plan_batch_with_tier1(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, tier1_mode, false)
}

#[cfg(all(target_os = "macos", test))]
fn execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    execute_direct_color_plan_batch_with_tier1_options(plans, fmt, DirectTier1Mode::CpuUpload, true)
}

#[cfg(target_os = "macos")]
fn execute_direct_color_plan_batch_with_tier1_options(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
) -> Result<Vec<Surface>, Error> {
    if plans.is_empty() {
        return Ok(Vec::new());
    }
    if tier1_mode == DirectTier1Mode::Metal
        && plans
            .iter()
            .any(|plan| !prepared_direct_color_plan_supports_runtime(plan, fmt))
    {
        return Err(Error::MetalKernel {
            message: "unsupported classic kernel input in direct component plan".to_string(),
        });
    }

    with_runtime(|runtime| {
        let mut retained_buffers = Vec::new();
        let mut retained_cpu_coefficients = Vec::<Vec<f32>>::new();
        let mut status_checks = Vec::new();
        let mut scratch_buffers = Vec::new();
        let mut stage_timings = DirectHybridStageTimings::default();
        let profile_hybrid_stages =
            tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();

        if fmt == PixelFormat::Rgb8
            && profile_hybrid_stages
            && metal_profile_decode_split_commands_enabled()
        {
            let split_command_buffers = DecodeHybridSplitCommandBuffers::new(runtime);
            if let Some(surfaces) = try_encode_stacked_mct_rgb8_direct_color_batch(
                runtime,
                split_command_buffers.refs(),
                plans,
                tier1_mode,
                force_flattened_cpu_tier1,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )? {
                split_command_buffers.commit_in_order();
                let wait_started = Instant::now();
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                split_command_buffers.mct_pack.wait_until_completed();
                stage_timings.command_wait += elapsed_us(wait_started);
                record_completed_decode_split_gpu_stages(
                    &mut stage_timings,
                    &split_command_buffers,
                );
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers);
                return Ok(surfaces);
            }

            drop(split_command_buffers);
            retained_buffers.clear();
            retained_cpu_coefficients.clear();
            status_checks.clear();
            scratch_buffers.clear();
            stage_timings = DirectHybridStageTimings::default();
        }

        let command_buffer = runtime.queue.new_command_buffer();
        if profile_hybrid_stages {
            label_command_buffer(
                command_buffer,
                "signinum-j2k decode hybrid direct color batch",
            );
        }

        if fmt == PixelFormat::Rgb8 {
            if let Some(surfaces) = try_encode_stacked_mct_rgb8_direct_color_batch(
                runtime,
                DirectColorBatchCommandBuffers::single(command_buffer),
                plans,
                tier1_mode,
                force_flattened_cpu_tier1,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )? {
                command_buffer.commit();
                let wait_started = profile_hybrid_stages.then(Instant::now);
                let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
                command_buffer.wait_until_completed();
                if let Some(started) = wait_started {
                    stage_timings.command_wait += elapsed_us(started);
                }
                if profile_hybrid_stages {
                    if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                        stage_timings.gpu_command += duration.as_micros();
                    }
                }
                for status_check in status_checks {
                    validate_direct_status(status_check)?;
                }
                if tier1_mode == DirectTier1Mode::CpuUpload {
                    emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
                }
                drop(retained_buffers);
                drop(retained_cpu_coefficients);
                recycle_scratch_buffers(runtime, scratch_buffers);
                return Ok(surfaces);
            }
        }

        let mut surfaces = Vec::with_capacity(plans.len());

        for plan in plans {
            let surface = encode_prepared_direct_color_plan_in_command_buffer(
                runtime,
                command_buffer,
                plan,
                fmt,
                tier1_mode,
                &mut stage_timings,
                &mut retained_buffers,
                &mut retained_cpu_coefficients,
                &mut status_checks,
                &mut scratch_buffers,
            )?;
            surfaces.push(surface);
        }

        command_buffer.commit();
        let wait_started = profile_hybrid_stages.then(Instant::now);
        let _wait_signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_COMMAND_WAIT);
        command_buffer.wait_until_completed();
        if let Some(started) = wait_started {
            stage_timings.command_wait += elapsed_us(started);
        }
        if profile_hybrid_stages {
            if let Some(duration) = completed_command_buffer_gpu_duration(command_buffer) {
                stage_timings.gpu_command += duration.as_micros();
            }
        }
        for status_check in status_checks {
            validate_direct_status(status_check)?;
        }
        if tier1_mode == DirectTier1Mode::CpuUpload {
            emit_direct_hybrid_stage_timings(&stage_timings, fmt, plans.len());
        }
        drop(retained_buffers);
        drop(retained_cpu_coefficients);
        recycle_scratch_buffers(runtime, scratch_buffers);
        Ok(surfaces)
    })
}

#[cfg(target_os = "macos")]
fn signed_sample_bias(bit_depth: u8) -> f32 {
    2.0_f32.powi(i32::from(bit_depth) - 1)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_prepared_direct_color_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Surface, Error> {
    if plan.component_plans.len() != 3 {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect color execution expected 3 component plans, got {}",
                plan.component_plans.len()
            ),
        });
    }

    let mut planes = Vec::with_capacity(3);
    for component_plan in &plan.component_plans {
        planes.push(encode_prepared_direct_component_plane_in_command_buffer(
            runtime,
            command_buffer,
            component_plan,
            tier1_mode,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
            status_checks,
            scratch_buffers,
        )?);
    }

    if plan.mct && fmt == PixelFormat::Rgb8 {
        let encode_started = metal_profile_stages_enabled().then(Instant::now);
        let surface = encode_mct_rgb8_to_surface_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            plan.dimensions,
            plan.bit_depths,
            plan.transform,
        );
        if let Some(started) = encode_started {
            stage_timings.metal_mct_pack_encode += elapsed_us(started);
        }
        return Ok(surface);
    }

    if plan.mct {
        let len = plan.dimensions.0 as usize * plan.dimensions.1 as usize;
        let encode_started = metal_profile_stages_enabled().then(Instant::now);
        status_checks.push(dispatch_inverse_mct_buffers_in_command_buffer(
            runtime,
            command_buffer,
            [&planes[0], &planes[1], &planes[2]],
            len,
            plan.transform,
            [
                signed_sample_bias(plan.bit_depths[0]),
                signed_sample_bias(plan.bit_depths[1]),
                signed_sample_bias(plan.bit_depths[2]),
            ],
        )?);
        if let Some(started) = encode_started {
            stage_timings.metal_mct_pack_encode += elapsed_us(started);
        }
    }

    let stage = PlaneStage {
        dims: plan.dimensions,
        plane_count: 3,
        color_space: NativeColorSpace::RGB,
        has_alpha: false,
        bit_depths: [
            u32::from(plan.bit_depths[0]),
            u32::from(plan.bit_depths[1]),
            u32::from(plan.bit_depths[2]),
            0,
        ],
        planes: [
            Some(planes[0].clone()),
            Some(planes[1].clone()),
            Some(planes[2].clone()),
            None,
        ],
    };
    let encode_started = metal_profile_stages_enabled().then(Instant::now);
    let surface =
        encode_plane_stage_to_surface_in_command_buffer(runtime, command_buffer, &stage, fmt);
    if let Some(started) = encode_started {
        stage_timings.metal_mct_pack_encode += elapsed_us(started);
    }
    surface
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectBandSlice {
    band_id: J2kDirectBandId,
    buffer: Buffer,
    offset_bytes: usize,
    window: BandRequiredRegion,
}

#[cfg(target_os = "macos")]
fn lookup_direct_band_slice_entry(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: signinum_j2k_native::J2kRect,
) -> Result<DirectBandSlice, Error> {
    bands
        .iter()
        .find(|existing| existing.band_id == band_id)
        .cloned()
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "missing J2K MetalDirect device band {} for rect ({}, {}, {}, {})",
                band_id, rect.x0, rect.y0, rect.x1, rect.y1
            ),
        })
}

#[cfg(target_os = "macos")]
fn lookup_direct_band_slice(
    bands: &[DirectBandSlice],
    band_id: J2kDirectBandId,
    rect: signinum_j2k_native::J2kRect,
) -> Result<(Buffer, usize), Error> {
    let entry = lookup_direct_band_slice_entry(bands, band_id, rect)?;
    Ok((entry.buffer, entry.offset_bytes))
}

#[cfg(target_os = "macos")]
fn lookup_repeated_direct_band_layout_entry(
    band_sets: &[Vec<DirectBandSlice>],
    band_id: J2kDirectBandId,
    rect: signinum_j2k_native::J2kRect,
) -> Result<(DirectBandSlice, u32), Error> {
    let first_bands = band_sets.first().ok_or_else(|| Error::MetalKernel {
        message: "missing J2K MetalDirect repeated band set".to_string(),
    })?;
    let entry = lookup_direct_band_slice_entry(first_bands, band_id, rect)?;
    let stride_bytes = if let Some(second_bands) = band_sets.get(1) {
        let next = lookup_direct_band_slice_entry(second_bands, band_id, rect)?;
        next.offset_bytes
            .checked_sub(entry.offset_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K MetalDirect repeated band offsets are not monotonic".to_string(),
            })?
    } else {
        entry.window.width() as usize * entry.window.height() as usize * size_of::<f32>()
    };
    if stride_bytes % size_of::<f32>() != 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride is not f32-aligned".to_string(),
        });
    }
    let stride_elements =
        u32::try_from(stride_bytes / size_of::<f32>()).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect repeated band stride exceeds u32".to_string(),
        })?;
    Ok((entry, stride_elements))
}

#[cfg(target_os = "macos")]
struct StackedDirectComponentPlane {
    buffer: Buffer,
    dimensions: (u32, u32),
    count: usize,
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn try_encode_stacked_mct_rgb8_direct_color_batch(
    runtime: &MetalRuntime,
    command_buffers: DirectColorBatchCommandBuffers<'_>,
    plans: &[Arc<PreparedDirectColorPlan>],
    tier1_mode: DirectTier1Mode,
    force_flattened_cpu_tier1: bool,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Option<Vec<Surface>>, Error> {
    let Some(first) = plans.first() else {
        return Ok(Some(Vec::new()));
    };
    let repeated_count = repeated_shared_direct_color_plan_count(plans);
    if plans.len() <= 1
        || !first.mct
        || first.component_plans.len() != 3
        || !plans.iter().all(|plan| {
            plan.mct
                && plan.dimensions == first.dimensions
                && plan.bit_depths == first.bit_depths
                && plan.transform == first.transform
                && plan.component_plans.len() == 3
        })
    {
        return Ok(None);
    }
    let execution_plans = if repeated_count.is_some() {
        &plans[..1]
    } else {
        plans
    };

    let flattened_cpu_tier1_cache = if tier1_mode == DirectTier1Mode::CpuUpload
        && (force_flattened_cpu_tier1
            || flattened_hybrid_cpu_tier1_enabled()
            || should_flatten_hybrid_cpu_tier1_color_batch(execution_plans))
    {
        Some(build_flattened_cpu_tier1_cache(
            runtime,
            execution_plans,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
        )?)
    } else {
        None
    };

    let mut stacked_planes = Vec::with_capacity(3);
    for component_idx in 0..3 {
        let component_plan_refs = execution_plans
            .iter()
            .map(|plan| &plan.component_plans[component_idx])
            .collect::<Vec<_>>();
        if !supports_stacked_direct_component_plane_batch(&component_plan_refs) {
            return Ok(None);
        }
        stacked_planes.push(encode_stacked_direct_component_plane_batch(
            runtime,
            command_buffers,
            &component_plan_refs,
            component_idx,
            flattened_cpu_tier1_cache.as_ref(),
            tier1_mode,
            stage_timings,
            retained_buffers,
            retained_cpu_coefficients,
            status_checks,
            scratch_buffers,
        )?);
    }

    if !stacked_planes
        .iter()
        .all(|plane| plane.dimensions == first.dimensions && plane.count == execution_plans.len())
    {
        return Ok(None);
    }

    let encode_started = metal_profile_stages_enabled().then(Instant::now);
    let mct_plane_buffers = [
        &stacked_planes[0].buffer,
        &stacked_planes[1].buffer,
        &stacked_planes[2].buffer,
    ];
    let surfaces = if let Some(count) = repeated_count {
        encode_repeated_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            first.dimensions,
            count,
            first.bit_depths,
            first.transform,
        )?
    } else {
        encode_batched_mct_rgb8_to_surfaces_in_command_buffer(
            runtime,
            command_buffers.mct_pack,
            mct_plane_buffers,
            first.dimensions,
            execution_plans.len(),
            first.bit_depths,
            first.transform,
        )?
    };
    if let Some(started) = encode_started {
        stage_timings.metal_mct_pack_encode += elapsed_us(started);
    }
    Ok(Some(surfaces))
}

#[cfg(target_os = "macos")]
fn supports_stacked_direct_component_plane_batch(plans: &[&PreparedDirectGrayscalePlan]) -> bool {
    let Some(first) = plans.first() else {
        return false;
    };
    if plans.iter().any(|plan| {
        plan.dimensions != first.dimensions
            || plan.bit_depth != first.bit_depth
            || plan.steps.len() != first.steps.len()
    }) {
        return false;
    }

    let mut step_idx = 0;
    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            if group.end_step <= step_idx
                || !plans.iter().all(|plan| {
                    plan.classic_group_starting_at(step_idx)
                        .is_some_and(|other| classic_group_shapes_match(group, other))
                })
            {
                return false;
            }
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = first.ht_group_starting_at(step_idx) {
            if group.end_step <= step_idx
                || !plans.iter().all(|plan| {
                    plan.ht_group_starting_at(step_idx)
                        .is_some_and(|other| ht_group_shapes_match(group, other))
                })
            {
                return false;
            }
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::ClassicSubBand(other)
                            if classic_sub_band_shapes_match(sub_band, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::HtSubBand(other)
                            if ht_sub_band_shapes_match(sub_band, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Idwt(other)
                            if idwt_shapes_match(idwt, other)
                    )
                }) {
                    return false;
                }
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                if !plans.iter().all(|plan| {
                    matches!(
                        &plan.steps[step_idx],
                        PreparedDirectGrayscaleStep::Store(other)
                            if store_shapes_match(store, other)
                    )
                }) {
                    return false;
                }
            }
        }
        step_idx += 1;
    }

    true
}

#[cfg(all(target_os = "macos", test))]
fn prepared_direct_color_tier1_input_count(plan: &PreparedDirectColorPlan) -> usize {
    plan.component_plans
        .iter()
        .map(prepared_direct_component_tier1_input_count)
        .sum()
}

#[cfg(all(target_os = "macos", test))]
fn prepared_direct_component_tier1_input_count(plan: &PreparedDirectGrayscalePlan) -> usize {
    let mut count = 0;
    let mut step_idx = 0;
    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            count += 1;
            step_idx = group.end_step;
            continue;
        }
        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            count += 1;
            step_idx = group.end_step;
            continue;
        }
        if matches!(
            &plan.steps[step_idx],
            PreparedDirectGrayscaleStep::ClassicSubBand(_)
                | PreparedDirectGrayscaleStep::HtSubBand(_)
        ) {
            count += 1;
        }
        step_idx += 1;
    }
    count
}

#[cfg(target_os = "macos")]
fn prepared_direct_color_plan_supports_runtime(
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
) -> bool {
    matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) && plan.component_plans.len() == 3
        && plan
            .component_plans
            .iter()
            .all(prepared_direct_component_plan_supports_runtime)
}

#[cfg(target_os = "macos")]
fn prepared_direct_component_plan_supports_runtime(plan: &PreparedDirectGrayscalePlan) -> bool {
    plan.tier1_prepare_mode == DirectTier1Mode::Metal
        && plan.steps.iter().all(|step| match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => sub_band
                .jobs
                .iter()
                .all(|job| classic_prepared_job_supports_runtime(job, &sub_band.segments)),
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                sub_band.jobs.iter().all(ht_prepared_job_supports_runtime)
            }
            PreparedDirectGrayscaleStep::Idwt(_) | PreparedDirectGrayscaleStep::Store(_) => true,
        })
        && plan.classic_groups.iter().all(|group| {
            group
                .jobs
                .iter()
                .all(|job| classic_prepared_job_supports_runtime(job, &group.segments))
        })
        && plan
            .ht_groups
            .iter()
            .all(|group| group.jobs.iter().all(ht_prepared_job_supports_runtime))
}

#[cfg(target_os = "macos")]
fn classic_prepared_job_supports_runtime(
    job: &J2kClassicCleanupBatchJob,
    segments: &[J2kClassicSegment],
) -> bool {
    if job.width == 0 || job.height == 0 {
        return true;
    }
    if job.width > J2K_CLASSIC_MAX_WIDTH || job.height > J2K_CLASSIC_MAX_HEIGHT {
        return false;
    }
    if job.output_stride < job.width {
        return false;
    }
    if job.roi_shift != 0 {
        return false;
    }
    if job.total_bitplanes == 0 || job.total_bitplanes > 31 || job.missing_msbs >= 31 {
        return false;
    }
    let bitplanes = job.total_bitplanes.saturating_sub(job.missing_msbs);
    if bitplanes == 0 {
        return false;
    }
    let max_coding_passes = 1 + 3 * (bitplanes - 1);
    if job.number_of_coding_passes == 0 || job.number_of_coding_passes > max_coding_passes {
        return false;
    }

    let start = job.segment_offset as usize;
    let count = job.segment_count as usize;
    let Some(end) = start.checked_add(count) else {
        return false;
    };
    if end > segments.len() || count == 0 {
        return false;
    }

    let uses_bypass = (job.style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0;
    let mut expected_start = 0u32;
    let mut expected_offset = job.coded_offset;
    for segment in &segments[start..end] {
        if segment.start_coding_pass != expected_start
            || segment.start_coding_pass > segment.end_coding_pass
        {
            return false;
        }
        if uses_bypass {
            let expected_arithmetic =
                segment.start_coding_pass <= 9 || segment.start_coding_pass % 3 == 0;
            if (segment.use_arithmetic != 0) != expected_arithmetic {
                return false;
            }
            if segment.use_arithmetic == 0 {
                if segment.start_coding_pass % 3 != 1 {
                    return false;
                }
                if segment
                    .end_coding_pass
                    .saturating_sub(segment.start_coding_pass)
                    > 2
                {
                    return false;
                }
                if (segment.start_coding_pass..segment.end_coding_pass).any(|pass| pass % 3 == 0) {
                    return false;
                }
            }
        } else if segment.use_arithmetic == 0 {
            return false;
        }

        let Some(data_end) = segment.data_offset.checked_add(segment.data_length) else {
            return false;
        };
        if segment.data_offset != expected_offset
            || segment.data_offset < job.coded_offset
            || data_end > job.coded_offset.saturating_add(job.coded_len)
        {
            return false;
        }
        expected_offset = data_end;
        expected_start = segment.end_coding_pass;
    }

    expected_start == job.number_of_coding_passes
        && expected_offset == job.coded_offset.saturating_add(job.coded_len)
}

#[cfg(target_os = "macos")]
fn classic_group_shapes_match(
    first: &PreparedClassicSubBandGroup,
    other: &PreparedClassicSubBandGroup,
) -> bool {
    first.end_step == other.end_step
        && first.total_coefficients == other.total_coefficients
        && first.members.len() == other.members.len()
        && first
            .members
            .iter()
            .zip(&other.members)
            .all(|(left, right)| left.offset_elements == right.offset_elements)
}

#[cfg(target_os = "macos")]
fn ht_group_shapes_match(first: &PreparedHtSubBandGroup, other: &PreparedHtSubBandGroup) -> bool {
    first.end_step == other.end_step
        && first.total_coefficients == other.total_coefficients
        && first.members.len() == other.members.len()
        && first
            .members
            .iter()
            .zip(&other.members)
            .all(|(left, right)| left.offset_elements == right.offset_elements)
}

#[cfg(target_os = "macos")]
fn classic_sub_band_shapes_match(
    first: &PreparedClassicSubBand,
    other: &PreparedClassicSubBand,
) -> bool {
    first.width == other.width && first.height == other.height
}

#[cfg(target_os = "macos")]
fn ht_sub_band_shapes_match(first: &PreparedHtSubBand, other: &PreparedHtSubBand) -> bool {
    first.width == other.width && first.height == other.height
}

#[cfg(target_os = "macos")]
fn rect_shapes_match(
    first: signinum_j2k_native::J2kRect,
    other: signinum_j2k_native::J2kRect,
) -> bool {
    first.x0 == other.x0 && first.y0 == other.y0 && first.x1 == other.x1 && first.y1 == other.y1
}

#[cfg(target_os = "macos")]
fn idwt_shapes_match(first: &PreparedDirectIdwt, other: &PreparedDirectIdwt) -> bool {
    first.step.transform == other.step.transform
        && rect_shapes_match(first.step.rect, other.step.rect)
        && first.output_window.x0 == other.output_window.x0
        && first.output_window.y0 == other.output_window.y0
        && first.output_window.x1 == other.output_window.x1
        && first.output_window.y1 == other.output_window.y1
        && rect_shapes_match(first.step.ll, other.step.ll)
        && rect_shapes_match(first.step.hl, other.step.hl)
        && rect_shapes_match(first.step.lh, other.step.lh)
        && rect_shapes_match(first.step.hh, other.step.hh)
}

#[cfg(target_os = "macos")]
fn store_shapes_match(first: &J2kDirectStoreStep, other: &J2kDirectStoreStep) -> bool {
    rect_shapes_match(first.input_rect, other.input_rect)
        && first.source_x == other.source_x
        && first.source_y == other.source_y
        && first.copy_width == other.copy_width
        && first.copy_height == other.copy_height
        && first.output_width == other.output_width
        && first.output_height == other.output_height
        && first.output_x == other.output_x
        && first.output_y == other.output_y
        && first.addend.to_bits() == other.addend.to_bits()
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_stacked_direct_component_plane_batch(
    runtime: &MetalRuntime,
    command_buffers: DirectColorBatchCommandBuffers<'_>,
    plans: &[&PreparedDirectGrayscalePlan],
    component_idx: usize,
    flattened_cpu_tier1_cache: Option<&FlattenedCpuTier1Cache>,
    tier1_mode: DirectTier1Mode,
    stage_timings: &mut DirectHybridStageTimings,
    retained_buffers: &mut Vec<Buffer>,
    retained_cpu_coefficients: &mut Vec<Vec<f32>>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<StackedDirectComponentPlane, Error> {
    let Some(first) = plans.first() else {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color batch has no component plans".to_string(),
        });
    };

    let count = plans.len();
    let broadcast_tier1_inputs = tier1_mode == DirectTier1Mode::CpuUpload
        && plans.iter().all(|plan| std::ptr::eq(*plan, *first));
    let mut band_sets = vec![Vec::<DirectBandSlice>::new(); count];
    let mut final_plane = None;
    let mut step_idx = 0;
    let profile_stages = tier1_mode == DirectTier1Mode::CpuUpload && metal_profile_stages_enabled();

    while step_idx < first.steps.len() {
        if let Some(group) = first.classic_group_starting_at(step_idx) {
            let groups = plans
                .iter()
                .map(|plan| {
                    plan.classic_group_starting_at(step_idx)
                        .expect("preflight validated classic group")
                })
                .collect::<Vec<_>>();
            let buffer = match tier1_mode {
                DirectTier1Mode::Metal => {
                    let output = take_f32_scratch_buffer(runtime, group.total_coefficients * count);
                    let (buffers, status_check) =
                        encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
                            runtime,
                            command_buffers.default,
                            &groups,
                            &output.buffer,
                            scratch_buffers,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    let buffer = output.buffer.clone();
                    scratch_buffers.push(output);
                    buffer
                }
                DirectTier1Mode::CpuUpload => {
                    let input_groups = if broadcast_tier1_inputs {
                        &groups[..1]
                    } else {
                        &groups
                    };
                    if let Some(cache) = flattened_cpu_tier1_cache {
                        cache.buffer_for(
                            component_idx,
                            step_idx,
                            group.total_coefficients,
                            input_groups.len(),
                        )?
                    } else {
                        let inputs = input_groups
                            .iter()
                            .map(|group| ClassicCpuDecodeInput {
                                coded_data: &group.coded_data,
                                segments: &group.segments,
                                jobs: &group.jobs,
                                output_len: group.total_coefficients,
                            })
                            .collect::<Vec<_>>();
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_classic_inputs_on_cpu_with_plan_cache(
                            first,
                            step_idx,
                            &inputs,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                }
            };
            let stride_bytes = group.total_coefficients * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                let source_group = if broadcast_tier1_inputs {
                    groups[0]
                } else {
                    groups[instance_idx]
                };
                let instance_offset = if broadcast_tier1_inputs {
                    0
                } else {
                    instance_idx * stride_bytes
                };
                for member in &source_group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            step_idx = group.end_step;
            continue;
        }

        if let Some(group) = first.ht_group_starting_at(step_idx) {
            let groups = plans
                .iter()
                .map(|plan| {
                    plan.ht_group_starting_at(step_idx)
                        .expect("preflight validated HT group")
                })
                .collect::<Vec<_>>();
            let buffer = match tier1_mode {
                DirectTier1Mode::Metal => {
                    let output = take_f32_scratch_buffer(runtime, group.total_coefficients * count);
                    let (buffers, status_check) =
                        encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
                            runtime,
                            command_buffers.default,
                            &groups,
                            &output.buffer,
                        )?;
                    retained_buffers.extend(buffers);
                    status_checks.push(status_check);
                    let buffer = output.buffer.clone();
                    scratch_buffers.push(output);
                    buffer
                }
                DirectTier1Mode::CpuUpload => {
                    let input_groups = if broadcast_tier1_inputs {
                        &groups[..1]
                    } else {
                        &groups
                    };
                    if let Some(cache) = flattened_cpu_tier1_cache {
                        cache.buffer_for(
                            component_idx,
                            step_idx,
                            group.total_coefficients,
                            input_groups.len(),
                        )?
                    } else {
                        let inputs = input_groups
                            .iter()
                            .map(|group| HtCpuDecodeInput {
                                coded_data: &group.coded_arena.data,
                                jobs: &group.jobs,
                                output_len: group.total_coefficients,
                            })
                            .collect::<Vec<_>>();
                        let decode_started = profile_stages.then(Instant::now);
                        let cpu_tier1_counters =
                            profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                        let coefficients = decode_ht_inputs_on_cpu_with_plan_cache(
                            first,
                            step_idx,
                            &inputs,
                            cpu_tier1_counters.as_ref(),
                        )?;
                        if let Some(started) = decode_started {
                            stage_timings.cpu_tier1 += elapsed_us(started);
                        }
                        if let Some(counters) = &cpu_tier1_counters {
                            counters.add_to_stage_timings(stage_timings);
                        }
                        let upload_started = profile_stages.then(Instant::now);
                        let buffer = upload_cpu_decoded_coefficients(
                            runtime,
                            coefficients,
                            retained_buffers,
                            retained_cpu_coefficients,
                        );
                        if let Some(started) = upload_started {
                            stage_timings.coefficient_upload += elapsed_us(started);
                        }
                        buffer
                    }
                }
            };
            let stride_bytes = group.total_coefficients * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                let source_group = if broadcast_tier1_inputs {
                    groups[0]
                } else {
                    groups[instance_idx]
                };
                let instance_offset = if broadcast_tier1_inputs {
                    0
                } else {
                    instance_idx * stride_bytes
                };
                for member in &source_group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            step_idx = group.end_step;
            continue;
        }

        match &first.steps[step_idx] {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let sub_bands = plans
                    .iter()
                    .map(|plan| match &plan.steps[step_idx] {
                        PreparedDirectGrayscaleStep::ClassicSubBand(other) => other,
                        _ => unreachable!("preflight validated classic sub-band"),
                    })
                    .collect::<Vec<_>>();
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                        let (buffers, status_check) =
                            encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
                                runtime,
                                command_buffers.default,
                                &sub_bands,
                                &output.buffer,
                                scratch_buffers,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let input_sub_bands = if broadcast_tier1_inputs {
                            &sub_bands[..1]
                        } else {
                            &sub_bands
                        };
                        if let Some(cache) = flattened_cpu_tier1_cache {
                            cache.buffer_for(
                                component_idx,
                                step_idx,
                                per_instance_len,
                                input_sub_bands.len(),
                            )?
                        } else {
                            let inputs = input_sub_bands
                                .iter()
                                .map(|sub_band| ClassicCpuDecodeInput {
                                    coded_data: &sub_band.coded_data,
                                    segments: &sub_band.segments,
                                    jobs: &sub_band.jobs,
                                    output_len: per_instance_len,
                                })
                                .collect::<Vec<_>>();
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_classic_inputs_on_cpu_with_plan_cache(
                                first,
                                step_idx,
                                &inputs,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    }
                };
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let source_sub_band = if broadcast_tier1_inputs {
                        sub_bands[0]
                    } else {
                        sub_bands[instance_idx]
                    };
                    let instance_offset = if broadcast_tier1_inputs {
                        0
                    } else {
                        instance_idx * stride_bytes
                    };
                    bands.push(DirectBandSlice {
                        band_id: source_sub_band.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset,
                        window: BandRequiredRegion::full(
                            source_sub_band.width,
                            source_sub_band.height,
                        ),
                    });
                }
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let sub_bands = plans
                    .iter()
                    .map(|plan| match &plan.steps[step_idx] {
                        PreparedDirectGrayscaleStep::HtSubBand(other) => other,
                        _ => unreachable!("preflight validated HT sub-band"),
                    })
                    .collect::<Vec<_>>();
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let buffer = match tier1_mode {
                    DirectTier1Mode::Metal => {
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                        let (buffers, status_check) =
                            encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
                                runtime,
                                command_buffers.default,
                                &sub_bands,
                                &output.buffer,
                            )?;
                        retained_buffers.extend(buffers);
                        status_checks.push(status_check);
                        let buffer = output.buffer.clone();
                        scratch_buffers.push(output);
                        buffer
                    }
                    DirectTier1Mode::CpuUpload => {
                        let input_sub_bands = if broadcast_tier1_inputs {
                            &sub_bands[..1]
                        } else {
                            &sub_bands
                        };
                        if let Some(cache) = flattened_cpu_tier1_cache {
                            cache.buffer_for(
                                component_idx,
                                step_idx,
                                per_instance_len,
                                input_sub_bands.len(),
                            )?
                        } else {
                            let inputs = input_sub_bands
                                .iter()
                                .map(|sub_band| HtCpuDecodeInput {
                                    coded_data: &sub_band.coded_data,
                                    jobs: &sub_band.jobs,
                                    output_len: per_instance_len,
                                })
                                .collect::<Vec<_>>();
                            let decode_started = profile_stages.then(Instant::now);
                            let cpu_tier1_counters =
                                profile_stages.then(CpuTier1DecodeSubstageCounters::default);
                            let coefficients = decode_ht_inputs_on_cpu_with_plan_cache(
                                first,
                                step_idx,
                                &inputs,
                                cpu_tier1_counters.as_ref(),
                            )?;
                            if let Some(started) = decode_started {
                                stage_timings.cpu_tier1 += elapsed_us(started);
                            }
                            if let Some(counters) = &cpu_tier1_counters {
                                counters.add_to_stage_timings(stage_timings);
                            }
                            let upload_started = profile_stages.then(Instant::now);
                            let buffer = upload_cpu_decoded_coefficients(
                                runtime,
                                coefficients,
                                retained_buffers,
                                retained_cpu_coefficients,
                            );
                            if let Some(started) = upload_started {
                                stage_timings.coefficient_upload += elapsed_us(started);
                            }
                            buffer
                        }
                    }
                };
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let source_sub_band = if broadcast_tier1_inputs {
                        sub_bands[0]
                    } else {
                        sub_bands[instance_idx]
                    };
                    let instance_offset = if broadcast_tier1_inputs {
                        0
                    } else {
                        instance_idx * stride_bytes
                    };
                    bands.push(DirectBandSlice {
                        band_id: source_sub_band.band_id,
                        buffer: buffer.clone(),
                        offset_bytes: instance_offset,
                        window: BandRequiredRegion::full(
                            source_sub_band.width,
                            source_sub_band.height,
                        ),
                    });
                }
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => {
                let per_instance_len = prepared_idwt_output_len(idwt);
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                let encode_started = profile_stages.then(Instant::now);
                match idwt.step.transform {
                    J2kWaveletTransform::Reversible53 => {
                        let (ll, low_low_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.ll_band_id,
                            idwt.step.ll,
                        )?;
                        let (hl, high_low_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.hl_band_id,
                            idwt.step.hl,
                        )?;
                        let (lh, low_high_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.lh_band_id,
                            idwt.step.lh,
                        )?;
                        let (hh, high_high_stride) = lookup_repeated_direct_band_layout_entry(
                            &band_sets,
                            idwt.step.hh_band_id,
                            idwt.step.hh,
                        )?;
                        let params = repeated_idwt_params(
                            idwt,
                            idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                            PreparedIdwtInputStrides {
                                ll: low_low_stride,
                                hl: high_low_stride,
                                lh: low_high_stride,
                                hh: high_high_stride,
                            },
                            count,
                            "color",
                        )?;
                        dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                            runtime,
                            command_buffers.idwt,
                            &ll.buffer,
                            ll.offset_bytes,
                            &hl.buffer,
                            hl.offset_bytes,
                            &lh.buffer,
                            lh.offset_bytes,
                            &hh.buffer,
                            hh.offset_bytes,
                            params,
                            &output.buffer,
                        );
                    }
                    J2kWaveletTransform::Irreversible97 => {
                        for (instance_idx, bands) in band_sets.iter().enumerate() {
                            let PreparedDirectGrayscaleStep::Idwt(step) =
                                &plans[instance_idx].steps[step_idx]
                            else {
                                unreachable!("preflight validated IDWT")
                            };
                            let ll = lookup_direct_band_slice_entry(
                                bands,
                                step.step.ll_band_id,
                                step.step.ll,
                            )?;
                            let hl = lookup_direct_band_slice_entry(
                                bands,
                                step.step.hl_band_id,
                                step.step.hl,
                            )?;
                            let lh = lookup_direct_band_slice_entry(
                                bands,
                                step.step.lh_band_id,
                                step.step.lh,
                            )?;
                            let hh = lookup_direct_band_slice_entry(
                                bands,
                                step.step.hh_band_id,
                                step.step.hh,
                            )?;
                            let params = prepared_idwt_params(
                                step,
                                idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                            );
                            status_checks.push(
                                dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                    runtime,
                                    command_buffers.idwt.interleave,
                                    &ll.buffer,
                                    ll.offset_bytes,
                                    &hl.buffer,
                                    hl.offset_bytes,
                                    &lh.buffer,
                                    lh.offset_bytes,
                                    &hh.buffer,
                                    hh.offset_bytes,
                                    params,
                                    &output.buffer,
                                    instance_idx * per_instance_len * size_of::<f32>(),
                                ),
                            );
                        }
                    }
                }
                if let Some(started) = encode_started {
                    stage_timings.metal_idwt_encode += elapsed_us(started);
                }
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    let PreparedDirectGrayscaleStep::Idwt(step) =
                        &plans[instance_idx].steps[step_idx]
                    else {
                        unreachable!("preflight validated IDWT")
                    };
                    bands.push(DirectBandSlice {
                        band_id: step.step.output_band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: step.output_window,
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::Store(store) => {
                let (input, input_instance_stride) = lookup_repeated_direct_band_layout_entry(
                    &band_sets,
                    store.input_band_id,
                    store.input_rect,
                )?;
                let per_instance_len = store.output_width as usize * store.output_height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                let encode_started = profile_stages.then(Instant::now);
                dispatch_store_component_repeated_in_command_buffer(
                    runtime,
                    command_buffers.store,
                    &input.buffer,
                    input.offset_bytes,
                    &output.buffer,
                    J2kRepeatedStoreParams {
                        input_width: store.input_rect.width(),
                        input_height: store.input_rect.height(),
                        input_instance_stride,
                        source_x: store.source_x,
                        source_y: store.source_y,
                        copy_width: store.copy_width,
                        copy_height: store.copy_height,
                        output_width: store.output_width,
                        output_height: store.output_height,
                        output_x: store.output_x,
                        output_y: store.output_y,
                        addend: store.addend,
                        batch_count: u32::try_from(count).map_err(|_| Error::MetalKernel {
                            message: "J2K MetalDirect color store batch count exceeds u32"
                                .to_string(),
                        })?,
                    },
                );
                if let Some(started) = encode_started {
                    stage_timings.metal_store_encode += elapsed_us(started);
                }
                final_plane = Some(output.buffer.clone());
                scratch_buffers.push(output);
            }
        }
        step_idx += 1;
    }

    let buffer = final_plane.ok_or_else(|| Error::MetalKernel {
        message: "J2K MetalDirect color component batch did not produce a final plane".to_string(),
    })?;
    record_hybrid_stacked_component_batch(tier1_mode);
    Ok(StackedDirectComponentPlane {
        buffer,
        dimensions: first.dimensions,
        count,
    })
}

#[cfg(target_os = "macos")]
fn ht_prepared_job_supports_runtime(job: &J2kHtCleanupBatchJob) -> bool {
    if job.width == 0 || job.height == 0 {
        return true;
    }
    job.roi_shift == 0
        && job.output_stride >= job.width
        && crate::ht::supports_metal_ht_geometry(job.width, job.height)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_repeated_direct_grayscale_plan_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plan: &PreparedDirectGrayscalePlan,
    fmt: PixelFormat,
    count: usize,
    retained_buffers: &mut Vec<Buffer>,
    status_checks: &mut Vec<DirectStatusCheck>,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<Vec<Surface>, Error> {
    let mut band_sets = vec![Vec::<DirectBandSlice>::new(); count];
    let mut surfaces = Vec::with_capacity(count);
    let mut stacked_outputs = true;
    let mut step_idx = 0;

    while step_idx < plan.steps.len() {
        if let Some(group) = plan.classic_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
            let (buffers, status_check) =
                encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                    scratch_buffers,
                )?;
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        if let Some(group) = plan.ht_group_starting_at(step_idx) {
            let per_instance_len = group.total_coefficients;
            let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
            let (buffers, status_check) =
                encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
                    runtime,
                    command_buffer,
                    group,
                    count,
                    &output.buffer,
                )?;
            retained_buffers.extend(buffers);
            status_checks.push(status_check);
            let stride_bytes = per_instance_len * size_of::<f32>();
            for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                for member in &group.members {
                    bands.push(DirectBandSlice {
                        band_id: member.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes
                            + member.offset_elements * size_of::<f32>(),
                        window: member.window,
                    });
                }
            }
            scratch_buffers.push(output);
            step_idx = group.end_step;
            continue;
        }

        let step = &plan.steps[step_idx];
        match step {
            PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                let (buffers, status_check) =
                    encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                        scratch_buffers,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::HtSubBand(sub_band) => {
                let per_instance_len = sub_band.width as usize * sub_band.height as usize;
                let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                let (buffers, status_check) =
                    encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
                        runtime,
                        command_buffer,
                        sub_band,
                        count,
                        &output.buffer,
                    )?;
                retained_buffers.extend(buffers);
                status_checks.push(status_check);
                let stride_bytes = per_instance_len * size_of::<f32>();
                for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                    bands.push(DirectBandSlice {
                        band_id: sub_band.band_id,
                        buffer: output.buffer.clone(),
                        offset_bytes: instance_idx * stride_bytes,
                        window: BandRequiredRegion::full(sub_band.width, sub_band.height),
                    });
                }
                scratch_buffers.push(output);
            }
            PreparedDirectGrayscaleStep::Idwt(idwt) => match idwt.step.transform {
                J2kWaveletTransform::Reversible53 if stacked_outputs => {
                    let (ll, low_low_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.ll_band_id,
                        idwt.step.ll,
                    )?;
                    let (hl, high_low_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.hl_band_id,
                        idwt.step.hl,
                    )?;
                    let (lh, low_high_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.lh_band_id,
                        idwt.step.lh,
                    )?;
                    let (hh, high_high_stride) = lookup_repeated_direct_band_layout_entry(
                        &band_sets,
                        idwt.step.hh_band_id,
                        idwt.step.hh,
                    )?;
                    let params = repeated_idwt_params(
                        idwt,
                        idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                        PreparedIdwtInputStrides {
                            ll: low_low_stride,
                            hl: high_low_stride,
                            lh: low_high_stride,
                            hh: high_high_stride,
                        },
                        count,
                        "repeated",
                    )?;
                    let per_instance_len = prepared_idwt_output_len(idwt);
                    let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                    dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
                        runtime,
                        DirectIdwtCommandBuffers::single(command_buffer),
                        &ll.buffer,
                        ll.offset_bytes,
                        &hl.buffer,
                        hl.offset_bytes,
                        &lh.buffer,
                        lh.offset_bytes,
                        &hh.buffer,
                        hh.offset_bytes,
                        params,
                        &output.buffer,
                    );
                    let stride_bytes = per_instance_len * size_of::<f32>();
                    for (instance_idx, bands) in band_sets.iter_mut().enumerate() {
                        bands.push(DirectBandSlice {
                            band_id: idwt.step.output_band_id,
                            buffer: output.buffer.clone(),
                            offset_bytes: instance_idx * stride_bytes,
                            window: idwt.output_window,
                        });
                    }
                    scratch_buffers.push(output);
                }
                _ => {
                    stacked_outputs = false;
                    for bands in &mut band_sets {
                        let ll = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.ll_band_id,
                            idwt.step.ll,
                        )?;
                        let hl = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.hl_band_id,
                            idwt.step.hl,
                        )?;
                        let lh = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.lh_band_id,
                            idwt.step.lh,
                        )?;
                        let hh = lookup_direct_band_slice_entry(
                            bands,
                            idwt.step.hh_band_id,
                            idwt.step.hh,
                        )?;
                        let params = prepared_idwt_params(
                            idwt,
                            idwt_input_windows_from_slices(&ll, &hl, &lh, &hh),
                        );
                        let output =
                            take_f32_scratch_buffer(runtime, prepared_idwt_output_len(idwt));
                        match idwt.step.transform {
                                J2kWaveletTransform::Reversible53 => {
                                    dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        runtime,
                                        command_buffer,
                                        &ll.buffer,
                                        ll.offset_bytes,
                                        &hl.buffer,
                                        hl.offset_bytes,
                                        &lh.buffer,
                                        lh.offset_bytes,
                                        &hh.buffer,
                                        hh.offset_bytes,
                                        params,
                                        &output.buffer,
                                        0,
                                    );
                                }
                                J2kWaveletTransform::Irreversible97 => status_checks.push(
                                    dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
                                        runtime,
                                        command_buffer,
                                        &ll.buffer,
                                        ll.offset_bytes,
                                        &hl.buffer,
                                        hl.offset_bytes,
                                        &lh.buffer,
                                        lh.offset_bytes,
                                        &hh.buffer,
                                        hh.offset_bytes,
                                        params,
                                        &output.buffer,
                                        0,
                                    ),
                                ),
                            }
                        bands.push(DirectBandSlice {
                            band_id: idwt.step.output_band_id,
                            buffer: output.buffer.clone(),
                            offset_bytes: 0,
                            window: idwt.output_window,
                        });
                        scratch_buffers.push(output);
                    }
                }
            },
            PreparedDirectGrayscaleStep::Store(store) => {
                if stacked_outputs {
                    let (input, _) = lookup_direct_band_slice(
                        &band_sets[0],
                        store.input_band_id,
                        store.input_rect,
                    )?;
                    let batch_count = u32::try_from(count).map_err(|_| Error::MetalKernel {
                        message: "J2K MetalDirect repeated store batch count exceeds u32"
                            .to_string(),
                    })?;
                    if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
                        let scale = j2k_scalar_pack_params(u32::from(plan.bit_depth));
                        surfaces.extend(encode_repeated_gray_store_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            J2kRepeatedGrayStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                                max_value: scale.max_value,
                                u8_scale: scale.u8_scale,
                                u16_scale: scale.u16_scale,
                            },
                            plan.dimensions,
                            fmt,
                            count,
                        )?);
                    } else {
                        let per_instance_len =
                            store.output_width as usize * store.output_height as usize;
                        let output = take_f32_scratch_buffer(runtime, per_instance_len * count);
                        dispatch_store_component_repeated_in_command_buffer(
                            runtime,
                            command_buffer,
                            &input,
                            0,
                            &output.buffer,
                            J2kRepeatedStoreParams {
                                input_width: store.input_rect.width(),
                                input_height: store.input_rect.height(),
                                input_instance_stride: store
                                    .input_rect
                                    .width()
                                    .checked_mul(store.input_rect.height())
                                    .ok_or_else(|| Error::MetalKernel {
                                        message: "J2K MetalDirect repeated store input stride overflows u32"
                                            .to_string(),
                                    })?,
                                source_x: store.source_x,
                                source_y: store.source_y,
                                copy_width: store.copy_width,
                                copy_height: store.copy_height,
                                output_width: store.output_width,
                                output_height: store.output_height,
                                output_x: store.output_x,
                                output_y: store.output_y,
                                addend: store.addend,
                                batch_count,
                            },
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.extend(encode_repeated_gray_plane_to_surfaces_in_command_buffer(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                            count,
                        )?);
                        scratch_buffers.push(output);
                    }
                } else {
                    for bands in &band_sets {
                        let (input, input_offset) =
                            lookup_direct_band_slice(bands, store.input_band_id, store.input_rect)?;
                        let output = take_f32_scratch_buffer(
                            runtime,
                            store.output_width as usize * store.output_height as usize,
                        );
                        let params = J2kStoreParams {
                            input_width: store.input_rect.width(),
                            source_x: store.source_x,
                            source_y: store.source_y,
                            copy_width: store.copy_width,
                            copy_height: store.copy_height,
                            output_width: store.output_width,
                            output_x: store.output_x,
                            output_y: store.output_y,
                            addend: store.addend,
                        };
                        dispatch_store_component_buffer_in_command_buffer_with_offsets(
                            runtime,
                            command_buffer,
                            &input,
                            input_offset,
                            &output.buffer,
                            0,
                            params,
                        );
                        retained_buffers.push(output.buffer.clone());
                        surfaces.push(encode_gray_plane_to_surface_in_command_buffer_with_offset(
                            runtime,
                            command_buffer,
                            &output.buffer,
                            0,
                            plan.dimensions,
                            plan.bit_depth,
                            fmt,
                        )?);
                        scratch_buffers.push(output);
                    }
                }
            }
        }
        step_idx += 1;
    }

    if surfaces.len() != count {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect repeated grayscale plan produced {} surfaces for count {}",
                surfaces.len(),
                count
            ),
        });
    }

    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn copy_plane_samples(buffer: &Buffer, samples: &[f32], image_width: usize, roi: Rect) {
    let row_width = roi.w as usize;
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<f32>(), row_width * roi.h as usize)
    };

    for row in 0..roi.h as usize {
        let src_start = (roi.y as usize + row) * image_width + roi.x as usize;
        let src_end = src_start + row_width;
        let dst_start = row * row_width;
        dst[dst_start..dst_start + row_width].copy_from_slice(&samples[src_start..src_end]);
    }
}

#[cfg(target_os = "macos")]
fn take_f32_scratch_buffer(runtime: &MetalRuntime, len: usize) -> DirectScratchBuffer {
    let bytes = len.max(1).saturating_mul(size_of::<f32>());
    DirectScratchBuffer {
        bytes,
        buffer: runtime.take_private_buffer(bytes),
    }
}

#[cfg(target_os = "macos")]
fn recycle_scratch_buffers(runtime: &MetalRuntime, scratch_buffers: Vec<DirectScratchBuffer>) {
    for scratch in scratch_buffers {
        runtime.recycle_private_buffer(scratch.bytes, scratch.buffer);
    }
}

#[cfg(target_os = "macos")]
fn take_recyclable_private_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_private_buffers: &mut Vec<(usize, Buffer)>,
) -> Buffer {
    let bytes = bytes.max(1);
    let buffer = runtime.take_private_buffer(bytes);
    recyclable_private_buffers.push((bytes, buffer.clone()));
    buffer
}

#[cfg(target_os = "macos")]
fn recycle_private_buffers(
    runtime: &MetalRuntime,
    recyclable_private_buffers: Vec<(usize, Buffer)>,
) {
    for (bytes, buffer) in recyclable_private_buffers {
        runtime.recycle_private_buffer(bytes, buffer);
    }
}

#[cfg(target_os = "macos")]
fn take_recyclable_shared_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_shared_buffers: &mut Vec<(usize, Buffer)>,
) -> Buffer {
    let bytes = bytes.max(1);
    let buffer = runtime.take_shared_buffer(bytes);
    recyclable_shared_buffers.push((bytes, buffer.clone()));
    buffer
}

#[cfg(target_os = "macos")]
fn recycle_shared_buffers(runtime: &MetalRuntime, recyclable_shared_buffers: Vec<(usize, Buffer)>) {
    for (bytes, buffer) in recyclable_shared_buffers {
        runtime.recycle_shared_buffer(bytes, buffer);
    }
}

#[cfg(target_os = "macos")]
fn validate_direct_status(status_check: DirectStatusCheck) -> Result<(), Error> {
    match status_check {
        DirectStatusCheck::Classic { buffer, len } => {
            let statuses = unsafe {
                core::slice::from_raw_parts(buffer.contents().cast::<J2kClassicStatus>(), len)
            };
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_CLASSIC_STATUS_OK)
            {
                return Err(decode_classic_status_error(status));
            }
        }
        DirectStatusCheck::Ht { buffer, len } => {
            let statuses = unsafe {
                core::slice::from_raw_parts(buffer.contents().cast::<J2kHtStatus>(), len)
            };
            if let Some(status) = statuses
                .iter()
                .copied()
                .find(|status| status.code != J2K_HT_STATUS_OK)
            {
                return Err(decode_ht_status_error(status));
            }
        }
        DirectStatusCheck::Idwt(buffer) => {
            let status = unsafe { buffer.contents().cast::<J2kIdwtStatus>().read() };
            if status.code != J2K_IDWT_STATUS_OK {
                return Err(decode_idwt_status_error(status));
            }
        }
        DirectStatusCheck::Mct(buffer) => {
            let status = unsafe { buffer.contents().cast::<J2kMctStatus>().read() };
            if status.code != J2K_MCT_STATUS_OK {
                return Err(decode_mct_status_error(status));
            }
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime, encoder, plane, 0, dims, bit_depth, fmt,
    )
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_command_buffer_with_offset(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let encoder = command_buffer.new_compute_command_encoder();
    let result = encode_gray_plane_to_surface_in_encoder_with_offset(
        runtime,
        encoder,
        plane,
        plane_offset_bytes,
        dims,
        bit_depth,
        fmt,
    );
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
fn encode_gray_plane_to_surface_in_encoder_with_offset(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    plane: &Buffer,
    plane_offset_bytes: usize,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (output_channels, opaque_alpha, pipeline) =
        output_shape_for(&NativeColorSpace::Gray, false, 1, fmt, runtime)?;
    let mut bit_depths = [0u32; 4];
    bit_depths[0] = u32::from(bit_depth);
    let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays(bit_depths);
    let params = J2kPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        output_channels,
        opaque_alpha,
        max_values,
        u8_scales,
        u16_scales,
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), plane_offset_bytes as u64);
    encoder.set_buffer(1, None, 0);
    encoder.set_buffer(2, None, 0);
    encoder.set_buffer(3, None, 0);
    encoder.set_buffer(4, Some(&out_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, dims);

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_repeated_gray_plane_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane: &Buffer,
    dims: (u32, u32),
    bit_depth: u8,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    let count_u32 = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal repeated grayscale surface count exceeds u32".to_string(),
    })?;
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal repeated grayscale surface size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal repeated grayscale output size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let scale = j2k_scalar_pack_params(u32::from(bit_depth));
    let params = J2kRepeatedGrayPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("J2K Metal output stride fits in u32"),
        batch_count: count_u32,
        max_value: scale.max_value,
        u8_scale: scale.u8_scale,
        u16_scale: scale.u16_scale,
    };
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.pack_u8_repeated_gray,
        PixelFormat::Gray16 => &runtime.pack_u16_repeated_gray,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("J2K Metal repeated grayscale pack does not support {fmt:?}"),
            })
        }
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(plane), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(encoder, pipeline, (dims.0, dims.1, count_u32));
    encoder.end_encoding();

    let mut surfaces = Vec::with_capacity(count);
    for instance_idx in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            fmt,
            instance_idx * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn owned_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let size = size_of_val(data).max(1);
    let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
    if !data.is_empty() {
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                size_of_val(data),
            );
        }
    }
    buffer
}

#[cfg(target_os = "macos")]
fn j2k_pack_kernel_name_for(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
) -> Option<&'static str> {
    match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8) => Some("j2k_pack_gray8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => Some("j2k_pack_rgb8"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => Some("j2k_pack_rgb_opaque_rgba8"),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => Some("j2k_pack_rgba8"),
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray16) => Some("j2k_pack_gray16"),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16) => Some("j2k_pack_rgb16"),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn j2k_pack_pipeline_for<'a>(
    runtime: &'a MetalRuntime,
    kernel_name: &str,
) -> &'a ComputePipelineState {
    match kernel_name {
        "j2k_pack_gray8" => &runtime.pack_gray8,
        "j2k_pack_rgb8" => &runtime.pack_rgb8,
        "j2k_pack_rgb_opaque_rgba8" => &runtime.pack_rgb_opaque_rgba8,
        "j2k_pack_rgba8" => &runtime.pack_rgba8,
        "j2k_pack_gray16" => &runtime.pack_gray16,
        "j2k_pack_rgb16" => &runtime.pack_rgb16,
        _ => unreachable!("validated J2K pack kernel name"),
    }
}

#[cfg(target_os = "macos")]
fn output_shape_for<'a>(
    color_space: &NativeColorSpace,
    has_alpha: bool,
    plane_count: usize,
    fmt: PixelFormat,
    runtime: &'a MetalRuntime,
) -> Result<(u32, u32, &'a ComputePipelineState), Error> {
    let Some(kernel_name) = j2k_pack_kernel_name_for(color_space, has_alpha, plane_count, fmt)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "unsupported J2K Metal mapping for {color_space:?}, alpha={has_alpha}, planes={plane_count}, fmt={fmt:?}"
            ),
        });
    };
    let (output_channels, opaque_alpha) = match (color_space, has_alpha, plane_count, fmt) {
        (NativeColorSpace::Gray, false, 1, PixelFormat::Gray8 | PixelFormat::Gray16) => (1, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8 | PixelFormat::Rgb16)
        | (NativeColorSpace::RGB, true, 4, PixelFormat::Rgb8) => (3, 0),
        (NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8) => (4, 1),
        (NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8) => (4, 0),
        _ => unreachable!("validated J2K pack shape"),
    };
    Ok((
        output_channels,
        opaque_alpha,
        j2k_pack_pipeline_for(runtime, kernel_name),
    ))
}

#[cfg(target_os = "macos")]
fn required_classic_output_len(job: J2kCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn classic_style_flags(style: signinum_j2k_native::J2kCodeBlockStyle) -> u32 {
    let mut flags = 0u32;
    if style.reset_context_probabilities {
        flags |= J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES;
    }
    if style.termination_on_each_pass {
        flags |= J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS;
    }
    if style.vertically_causal_context {
        flags |= J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT;
    }
    if style.segmentation_symbols {
        flags |= J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
    }
    if style.selective_arithmetic_coding_bypass {
        flags |= J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
    }
    flags
}

#[cfg(target_os = "macos")]
fn decode_classic_status_error(status: J2kClassicStatus) -> Error {
    let kind = match status.code {
        J2K_CLASSIC_STATUS_FAIL => "decode failure",
        J2K_CLASSIC_STATUS_UNSUPPORTED => "unsupported classic kernel input",
        _ => "unexpected classic kernel status",
    };
    Error::MetalKernel {
        message: format!("classic J2K Metal kernel {kind} (detail={})", status.detail),
    }
}

#[cfg(target_os = "macos")]
fn decode_idwt_status_error(status: J2kIdwtStatus) -> Error {
    let kind = match status.code {
        J2K_IDWT_STATUS_FAIL => "decode failure",
        _ => "unexpected IDWT kernel status",
    };
    Error::MetalKernel {
        message: format!("J2K Metal IDWT kernel {kind} (detail={})", status.detail),
    }
}

#[cfg(target_os = "macos")]
fn decode_mct_status_error(status: J2kMctStatus) -> Error {
    let kind = match status.code {
        J2K_MCT_STATUS_FAIL => "decode failure",
        _ => "unexpected inverse MCT kernel status",
    };
    Error::MetalKernel {
        message: format!(
            "J2K Metal inverse MCT kernel {kind} (detail={})",
            status.detail
        ),
    }
}

fn wrap_f32_output_buffer(device: &Device, output: &mut [f32]) -> Buffer {
    if output.is_empty() {
        device.new_buffer(
            size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    } else {
        device.new_buffer_with_bytes_no_copy(
            output.as_mut_ptr().cast(),
            size_of_val(output) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

#[cfg(target_os = "macos")]
fn borrow_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_bytes_no_copy(
            data.as_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

#[cfg(target_os = "macos")]
fn borrow_mut_slice_buffer<T>(device: &Device, data: &mut [T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_bytes_no_copy(
            data.as_mut_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

#[cfg(target_os = "macos")]
fn copied_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_data(
            data.as_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

#[cfg(target_os = "macos")]
fn copied_recyclable_shared_slice_buffer<T>(
    runtime: &MetalRuntime,
    data: &[T],
    recyclable_shared_buffers: &mut Vec<(usize, Buffer)>,
) -> Buffer {
    let size = size_of_val(data).max(1);
    let buffer = take_recyclable_shared_buffer(runtime, size, recyclable_shared_buffers);
    if !data.is_empty() {
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                size_of_val(data),
            );
        }
    }
    buffer
}

#[cfg(target_os = "macos")]
fn zeroed_recyclable_shared_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_shared_buffers: &mut Vec<(usize, Buffer)>,
) -> Buffer {
    let bytes = bytes.max(1);
    let buffer = take_recyclable_shared_buffer(runtime, bytes, recyclable_shared_buffers);
    unsafe {
        core::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes);
    }
    buffer
}

#[cfg(target_os = "macos")]
fn classic_coefficients_scratch_bytes(job_count: usize) -> Result<usize, Error> {
    job_count
        .max(1)
        .checked_mul(J2K_CLASSIC_MAX_COEFF_COUNT)
        .and_then(|count| count.checked_mul(size_of::<u32>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K coefficient scratch size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn take_classic_coefficients_scratch_buffer(
    runtime: &MetalRuntime,
    job_count: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = classic_coefficients_scratch_bytes(job_count)?;
    Ok(DirectScratchBuffer {
        bytes,
        buffer: runtime.take_private_buffer(bytes),
    })
}

#[cfg(target_os = "macos")]
fn classic_states_scratch_bytes(job_count: usize) -> Result<usize, Error> {
    job_count
        .max(1)
        .checked_mul(J2K_CLASSIC_MAX_COEFF_COUNT)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect states scratch overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn take_classic_states_scratch_buffer(
    runtime: &MetalRuntime,
    job_count: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = classic_states_scratch_bytes(job_count)?;
    Ok(DirectScratchBuffer {
        bytes,
        buffer: runtime.take_private_buffer(bytes),
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_dwt53(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> Result<J2kForwardDwt53Output, Error> {
    if width == 0 || height == 0 {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions must be non-zero".to_string(),
        });
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions overflow".to_string(),
        })?;
    if samples.len() != expected_len {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT sample length mismatch".to_string(),
        });
    }

    with_runtime(|runtime| {
        let bytes = size_of_val(samples);
        let buffer_a = copied_slice_buffer(&runtime.device, samples);
        let buffer_b = runtime
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        let command_buffer = runtime.queue.new_command_buffer();

        let mut current_width = width;
        let mut current_height = height;
        let mut shapes = Vec::new();
        let mut levels_run = 0u8;
        let mut active_is_a = true;

        while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
            let low_width = current_width.div_ceil(2);
            let low_height = current_height.div_ceil(2);
            let params = J2kForwardDwt53Params {
                full_width: width,
                current_width,
                current_height,
                low_width,
                low_height,
            };

            if current_height >= 2 {
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt53_pass(
                    &runtime.fdwt53_vertical,
                    command_buffer,
                    input,
                    output,
                    params,
                    "J2K forward DWT 5/3 vertical",
                );
                active_is_a = !active_is_a;
            }
            if current_width >= 2 {
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt53_pass(
                    &runtime.fdwt53_horizontal,
                    command_buffer,
                    input,
                    output,
                    params,
                    "J2K forward DWT 5/3 horizontal",
                );
                active_is_a = !active_is_a;
            }

            shapes.push(J2kForwardDwt53Level {
                hl: Vec::new(),
                lh: Vec::new(),
                hh: Vec::new(),
                width: current_width,
                height: current_height,
                low_width,
                low_height,
                high_width: current_width / 2,
                high_height: current_height / 2,
            });
            current_width = low_width;
            current_height = low_height;
            levels_run = levels_run.saturating_add(1);
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let active_buffer = if active_is_a { &buffer_a } else { &buffer_b };
        let transformed = unsafe {
            core::slice::from_raw_parts(active_buffer.contents().cast::<f32>(), samples.len())
        };
        let output = extract_forward_dwt53_output(
            transformed,
            width,
            current_width,
            current_height,
            shapes,
        )?;
        Ok(output)
    })
}

#[cfg(target_os = "macos")]
fn active_forward_dwt53_buffers<'a>(
    buffer_a: &'a Buffer,
    buffer_b: &'a Buffer,
    active_is_a: bool,
) -> (&'a Buffer, &'a Buffer) {
    if active_is_a {
        (buffer_a, buffer_b)
    } else {
        (buffer_b, buffer_a)
    }
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_pass(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: J2kForwardDwt53Params,
    label: &str,
) {
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, label);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kForwardDwt53Params>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        pipeline,
        (params.current_width, params.current_height),
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_batched_pass(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    inputs: &[Buffer],
    outputs: &[Buffer],
    params: J2kForwardDwt53BatchedParams,
    label: &str,
) {
    debug_assert!(!inputs.is_empty());
    debug_assert!(!outputs.is_empty());
    debug_assert!(params.component_count >= 1 && params.component_count <= 3);
    let first_input_buffer = &inputs[0];
    let second_input_buffer = inputs.get(1).unwrap_or(first_input_buffer);
    let third_input_buffer = inputs.get(2).unwrap_or(first_input_buffer);
    let first_output_buffer = &outputs[0];
    let second_output_buffer = outputs.get(1).unwrap_or(first_output_buffer);
    let third_output_buffer = outputs.get(2).unwrap_or(first_output_buffer);

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, label);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(first_input_buffer), 0);
    encoder.set_buffer(1, Some(second_input_buffer), 0);
    encoder.set_buffer(2, Some(third_input_buffer), 0);
    encoder.set_buffer(3, Some(first_output_buffer), 0);
    encoder.set_buffer(4, Some(second_output_buffer), 0);
    encoder.set_buffer(5, Some(third_output_buffer), 0);
    encoder.set_bytes(
        6,
        size_of::<J2kForwardDwt53BatchedParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        pipeline,
        (
            params.current_width,
            params.current_height,
            params.component_count,
        ),
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn extract_forward_dwt53_output(
    transformed: &[f32],
    full_width: u32,
    ll_width: u32,
    ll_height: u32,
    mut shapes: Vec<J2kForwardDwt53Level>,
) -> Result<J2kForwardDwt53Output, Error> {
    let full_width_usize = full_width as usize;
    let mut ll = Vec::with_capacity((ll_width as usize) * (ll_height as usize));
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width_usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row offset overflow".to_string(),
            })?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    for shape in &mut shapes {
        shape.hl = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            0,
            shape.high_width,
            shape.low_height,
        )?;
        shape.lh = extract_subband(
            transformed,
            full_width_usize,
            0,
            shape.low_height,
            shape.low_width,
            shape.high_height,
        )?;
        shape.hh = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            shape.low_height,
            shape.high_width,
            shape.high_height,
        )?;
    }
    shapes.reverse();

    Ok(J2kForwardDwt53Output {
        ll,
        ll_width,
        ll_height,
        levels: shapes,
    })
}

#[cfg(target_os = "macos")]
fn extract_subband(
    transformed: &[f32],
    full_width: usize,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, Error> {
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height as usize {
        let row_start = (y0 as usize)
            .checked_add(y)
            .and_then(|row| row.checked_mul(full_width))
            .and_then(|row| row.checked_add(x0 as usize))
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT subband offset overflow".to_string(),
            })?;
        out.extend_from_slice(&transformed[row_start..row_start + width as usize]);
    }
    Ok(out)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct J2kLosslessDeviceCodeBlock {
    pub(crate) coefficient_offset: u32,
    pub(crate) component: u32,
    pub(crate) subband_x: u32,
    pub(crate) subband_y: u32,
    pub(crate) block_x: u32,
    pub(crate) block_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) sub_band_type: signinum_j2k_native::J2kSubBandType,
    pub(crate) total_bitplanes: u8,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct J2kLosslessDevicePrepareJob<'a> {
    pub(crate) input: &'a Buffer,
    pub(crate) input_byte_offset: usize,
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) input_pitch_bytes: usize,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) components: u8,
    pub(crate) bytes_per_sample: u8,
    pub(crate) bit_depth: u8,
    pub(crate) num_decomposition_levels: u8,
    pub(crate) coefficient_count: usize,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kLosslessDeviceBatchPrepareItem<'a> {
    pub(crate) tile_index: usize,
    pub(crate) job: J2kLosslessDevicePrepareJob<'a>,
    pub(crate) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kPreparedLosslessDeviceCodeBlocks {
    coefficient_buffer: Buffer,
    coefficient_byte_offset: usize,
    coefficient_byte_len: usize,
    coefficient_buffer_is_batch_shared: bool,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    recyclable_private_buffers: Vec<(usize, Buffer)>,
    _prepare_command_buffer: CommandBuffer,
    _prepare_deinterleave_rct_command_buffer: Option<CommandBuffer>,
    _prepare_dwt53_command_buffer: Option<CommandBuffer>,
    _prepare_dwt53_vertical_command_buffers: Vec<CommandBuffer>,
    _prepare_dwt53_horizontal_command_buffers: Vec<CommandBuffer>,
    _prepare_coefficient_extract_command_buffer: Option<CommandBuffer>,
    _deinterleave_status_buffer: Buffer,
    _plane_buffers: Vec<Buffer>,
    _scratch_buffers: Vec<Buffer>,
    _coefficient_job_buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct J2kResidentPacketizationSubband {
    pub(crate) code_block_start: u32,
    pub(crate) code_block_count: u32,
    pub(crate) num_cbs_x: u32,
    pub(crate) num_cbs_y: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub(crate) struct J2kResidentPacketizationResolution {
    pub(crate) subbands: Vec<J2kResidentPacketizationSubband>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(crate) struct J2kResidentPacketizationEncodeJob<'a> {
    pub(crate) resolution_count: u32,
    pub(crate) num_layers: u8,
    pub(crate) num_components: u8,
    pub(crate) code_block_count: u32,
    pub(crate) packet_descriptors: &'a [J2kPacketizationPacketDescriptor],
    pub(crate) resolutions: &'a [J2kResidentPacketizationResolution],
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum J2kLosslessCodestreamBlockCodingMode {
    Classic,
    HighThroughput,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct J2kLosslessCodestreamAssemblyJob {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) num_components: u8,
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
    pub(crate) num_decomposition_levels: u8,
    pub(crate) use_mct: bool,
    pub(crate) guard_bits: u8,
    pub(crate) code_block_width_exp: u8,
    pub(crate) code_block_height_exp: u8,
    pub(crate) progression_order: EncodeProgressionOrder,
    pub(crate) write_tlm: bool,
    pub(crate) block_coding_mode: J2kLosslessCodestreamBlockCodingMode,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentLosslessTier1CodeBlocks {
    output_buffer: Buffer,
    status_buffer: Buffer,
    job_buffer: Buffer,
    batch_jobs: Vec<J2kClassicEncodeBatchJob>,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    output_capacity_total: usize,
    _segment_buffer: Buffer,
    tier1_command_buffer: CommandBuffer,
    _coefficient_buffer: Buffer,
    prepare_command_buffer: CommandBuffer,
    _deinterleave_status_buffer: Buffer,
    _plane_buffers: Vec<Buffer>,
    _scratch_buffers: Vec<Buffer>,
    _coefficient_job_buffer: Buffer,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentLosslessHtCodeBlocks {
    output_buffer: Buffer,
    status_buffer: Buffer,
    job_buffer: Buffer,
    batch_jobs: Vec<J2kHtEncodeBatchJob>,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    output_capacity_total: usize,
    tier1_command_buffer: CommandBuffer,
    _coefficient_buffer: Buffer,
    prepare_command_buffer: CommandBuffer,
    _deinterleave_status_buffer: Buffer,
    _plane_buffers: Vec<Buffer>,
    _scratch_buffers: Vec<Buffer>,
    _coefficient_job_buffer: Buffer,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentLosslessCodestream {
    pub(crate) buffer: Buffer,
    pub(crate) byte_offset: usize,
    pub(crate) byte_len: usize,
    pub(crate) capacity: usize,
    pub(crate) gpu_duration: Option<Duration>,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kPendingResidentLosslessCodestream {
    buffer: Buffer,
    capacity: usize,
    status_buffer: Buffer,
    command_buffer: CommandBuffer,
    retained_command_buffers: Vec<CommandBuffer>,
    _retained_buffers: Vec<Buffer>,
    status_stage: &'static str,
    length_error: &'static str,
    capacity_error: &'static str,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentHtBatchEncodeItem {
    pub(crate) prepared: J2kPreparedLosslessDeviceCodeBlocks,
    pub(crate) resolution_count: u32,
    pub(crate) num_layers: u8,
    pub(crate) num_components: u8,
    pub(crate) code_block_count: u32,
    pub(crate) packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    pub(crate) resolutions: Vec<J2kResidentPacketizationResolution>,
    pub(crate) codestream: J2kLosslessCodestreamAssemblyJob,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentClassicBatchEncodeItem {
    pub(crate) prepared: J2kPreparedLosslessDeviceCodeBlocks,
    pub(crate) resolution_count: u32,
    pub(crate) num_layers: u8,
    pub(crate) num_components: u8,
    pub(crate) code_block_count: u32,
    pub(crate) packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    pub(crate) resolutions: Vec<J2kResidentPacketizationResolution>,
    pub(crate) codestream: J2kLosslessCodestreamAssemblyJob,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct J2kResidentEncodeStageStats {
    /// Host-side wall time spent preparing resident encode coefficients.
    pub(crate) coefficient_prep_duration: Duration,
    /// Reserved for future finer-grained profiling within coefficient prep.
    pub(crate) deinterleave_rct_duration: Duration,
    /// Reserved for future finer-grained profiling within coefficient prep.
    pub(crate) dwt53_duration: Duration,
    /// Reserved for future finer-grained profiling within coefficient prep.
    pub(crate) coefficient_extract_duration: Duration,
    pub(crate) ht_table_build_duration: Duration,
    pub(crate) ht_buffer_allocation_duration: Duration,
    pub(crate) ht_command_encode_duration: Duration,
    pub(crate) ht_block_encode_duration: Duration,
    pub(crate) classic_tier1_setup_duration: Duration,
    pub(crate) classic_block_encode_duration: Duration,
    pub(crate) classic_tier1_token_pack_duration: Duration,
    pub(crate) classic_packet_plan_duration: Duration,
    pub(crate) classic_packet_buffer_setup_duration: Duration,
    pub(crate) classic_command_buffer_commit_duration: Duration,
    pub(crate) result_harvest_duration: Duration,
    pub(crate) result_status_copy_duration: Duration,
    pub(crate) result_private_recycle_duration: Duration,
    pub(crate) result_shared_recycle_duration: Duration,
    pub(crate) result_codestream_collect_duration: Duration,
    pub(crate) packet_block_prep_duration: Duration,
    pub(crate) packetization_duration: Duration,
    pub(crate) codestream_assembly_duration: Duration,
    pub(crate) coefficient_prep_gpu_duration: Duration,
    pub(crate) coefficient_deinterleave_rct_gpu_duration: Duration,
    pub(crate) coefficient_dwt53_gpu_duration: Duration,
    pub(crate) coefficient_dwt53_vertical_gpu_duration: Duration,
    pub(crate) coefficient_dwt53_horizontal_gpu_duration: Duration,
    pub(crate) coefficient_extract_gpu_duration: Duration,
    pub(crate) coefficient_copy_gpu_duration: Duration,
    pub(crate) gpu_elapsed_wall_duration: Duration,
    pub(crate) classic_block_gpu_duration: Duration,
    pub(crate) classic_tier1_density_gpu_duration: Duration,
    pub(crate) classic_tier1_raw_pack_gpu_duration: Duration,
    pub(crate) classic_tier1_arithmetic_pack_gpu_duration: Duration,
    pub(crate) classic_tier1_symbol_plan_gpu_duration: Duration,
    pub(crate) classic_tier1_pass_plan_gpu_duration: Duration,
    pub(crate) classic_tier1_token_emit_gpu_duration: Duration,
    pub(crate) classic_tier1_split_token_emit_gpu_duration: Duration,
    pub(crate) classic_tier1_token_pack_gpu_duration: Duration,
    pub(crate) ht_block_gpu_duration: Duration,
    pub(crate) packet_block_prep_gpu_duration: Duration,
    pub(crate) packetization_gpu_duration: Duration,
    pub(crate) packet_payload_copy_gpu_duration: Duration,
    pub(crate) codestream_assembly_gpu_duration: Duration,
    pub(crate) codestream_payload_copy_gpu_duration: Duration,
    pub(crate) tier1_output_capacity_total: usize,
    pub(crate) max_tier1_output_capacity: usize,
    pub(crate) tier1_output_used_bytes_total: usize,
    pub(crate) max_tier1_output_used_bytes: usize,
    pub(crate) tier1_segment_capacity_total: usize,
    pub(crate) max_tier1_segment_capacity_per_block: usize,
    pub(crate) tier1_coding_pass_count_total: usize,
    pub(crate) max_tier1_coding_passes_per_block: usize,
    pub(crate) tier1_arithmetic_pass_count_total: usize,
    pub(crate) tier1_raw_pass_count_total: usize,
    pub(crate) tier1_cleanup_pass_count_total: usize,
    pub(crate) tier1_sigprop_pass_count_total: usize,
    pub(crate) tier1_magref_pass_count_total: usize,
    pub(crate) tier1_arithmetic_cleanup_pass_count_total: usize,
    pub(crate) tier1_arithmetic_sigprop_pass_count_total: usize,
    pub(crate) tier1_arithmetic_magref_pass_count_total: usize,
    pub(crate) tier1_raw_sigprop_pass_count_total: usize,
    pub(crate) tier1_raw_magref_pass_count_total: usize,
    pub(crate) tier1_full_scan_coeff_visit_count_total: usize,
    pub(crate) tier1_arithmetic_scan_coeff_visit_count_total: usize,
    pub(crate) tier1_raw_scan_coeff_visit_count_total: usize,
    pub(crate) tier1_cleanup_scan_coeff_visit_count_total: usize,
    pub(crate) tier1_sigprop_scan_coeff_visit_count_total: usize,
    pub(crate) tier1_magref_scan_coeff_visit_count_total: usize,
    pub(crate) max_tier1_full_scan_coeff_visits_per_block: usize,
    pub(crate) tier1_sigprop_active_candidate_count_total: usize,
    pub(crate) tier1_sigprop_new_significant_count_total: usize,
    pub(crate) tier1_magref_active_candidate_count_total: usize,
    pub(crate) tier1_arithmetic_sigprop_active_candidate_count_total: usize,
    pub(crate) tier1_arithmetic_sigprop_new_significant_count_total: usize,
    pub(crate) tier1_raw_sigprop_active_candidate_count_total: usize,
    pub(crate) tier1_raw_sigprop_new_significant_count_total: usize,
    pub(crate) tier1_arithmetic_magref_active_candidate_count_total: usize,
    pub(crate) tier1_raw_magref_active_candidate_count_total: usize,
    pub(crate) tier1_cleanup_active_candidate_count_total: usize,
    pub(crate) tier1_cleanup_new_significant_count_total: usize,
    pub(crate) tier1_cleanup_rlc_stripe_count_total: usize,
    pub(crate) tier1_cleanup_rlc_zero_stripe_count_total: usize,
    pub(crate) tier1_symbol_plan_mq_symbol_count_total: usize,
    pub(crate) tier1_symbol_plan_raw_bit_count_total: usize,
    pub(crate) max_tier1_symbol_plan_mq_symbols_per_block: usize,
    pub(crate) max_tier1_symbol_plan_raw_bits_per_block: usize,
    pub(crate) tier1_symbol_plan_packed_token_bytes_total: usize,
    pub(crate) max_tier1_symbol_plan_packed_token_bytes_per_block: usize,
    pub(crate) tier1_symbol_plan_cleanup_mq_symbol_count_total: usize,
    pub(crate) tier1_symbol_plan_sigprop_mq_symbol_count_total: usize,
    pub(crate) tier1_symbol_plan_magref_mq_symbol_count_total: usize,
    pub(crate) tier1_symbol_plan_raw_sigprop_bit_count_total: usize,
    pub(crate) tier1_symbol_plan_raw_magref_bit_count_total: usize,
    pub(crate) tier1_symbol_plan_cleanup_sign_symbol_count_total: usize,
    pub(crate) tier1_symbol_plan_sigprop_sign_symbol_count_total: usize,
    pub(crate) tier1_symbol_plan_mq_symbol_hash_xor: usize,
    pub(crate) tier1_symbol_plan_raw_bit_hash_xor: usize,
    pub(crate) tier1_pass_plan_mq_symbol_count_total: usize,
    pub(crate) tier1_pass_plan_raw_bit_count_total: usize,
    pub(crate) tier1_pass_plan_nonempty_mq_pass_count_total: usize,
    pub(crate) tier1_pass_plan_nonempty_raw_pass_count_total: usize,
    pub(crate) max_tier1_pass_plan_mq_symbols_per_pass: usize,
    pub(crate) max_tier1_pass_plan_raw_bits_per_pass: usize,
    pub(crate) tier1_token_emit_mq_symbol_count_total: usize,
    pub(crate) tier1_token_emit_raw_bit_count_total: usize,
    pub(crate) tier1_token_emit_token_bytes_total: usize,
    pub(crate) max_tier1_token_emit_token_bytes_per_block: usize,
    pub(crate) tier1_token_emit_segment_count_total: usize,
    pub(crate) max_tier1_token_emit_segments_per_block: usize,
    pub(crate) tier1_token_emit_mq_symbol_hash_xor: usize,
    pub(crate) tier1_token_emit_raw_bit_hash_xor: usize,
    pub(crate) tier1_token_pack_output_bytes_total: usize,
    pub(crate) max_tier1_token_pack_output_bytes_per_block: usize,
    pub(crate) tier1_nonzero_block_count_total: usize,
    pub(crate) tier1_zero_block_count_total: usize,
    pub(crate) tier1_missing_bitplane_count_total: usize,
    pub(crate) max_tier1_missing_bitplanes_per_block: usize,
    pub(crate) tier1_segment_count_total: usize,
    pub(crate) max_tier1_segments_per_block: usize,
    pub(crate) packet_payload_copy_job_capacity_total: usize,
    pub(crate) max_packet_payload_copy_jobs_per_tile: usize,
    pub(crate) packet_payload_copy_job_count_total: usize,
    pub(crate) max_packet_payload_copy_jobs_used_per_tile: usize,
    pub(crate) packet_payload_copy_bytes_total: usize,
    pub(crate) max_packet_payload_copy_bytes_per_tile: usize,
    pub(crate) packet_payload_copy_small_job_count_total: usize,
    pub(crate) packet_payload_copy_medium_job_count_total: usize,
    pub(crate) packet_payload_copy_large_job_count_total: usize,
    pub(crate) packet_payload_copy_launched_stripe_count_total: usize,
    pub(crate) packet_payload_copy_active_stripe_count_total: usize,
    pub(crate) packet_output_capacity_total: usize,
    pub(crate) max_packet_output_capacity: usize,
    pub(crate) packet_output_used_bytes_total: usize,
    pub(crate) max_packet_output_used_bytes: usize,
    pub(crate) codestream_payload_copy_bytes_total: usize,
    pub(crate) codestream_payload_copy_launched_thread_count_total: usize,
    pub(crate) codestream_payload_copy_active_thread_count_total: usize,
    pub(crate) code_block_count: usize,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kResidentLosslessCodestreamBatchResult {
    pub(crate) codestreams: Vec<J2kResidentLosslessCodestream>,
    pub(crate) stage_stats: J2kResidentEncodeStageStats,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
enum J2kResidentTier1StatusKind {
    Classic,
    HighThroughput,
}

#[cfg(target_os = "macos")]
struct J2kResidentTier1StatusReadback {
    buffer: Buffer,
    kind: J2kResidentTier1StatusKind,
    classic_style_flags: u32,
    classic_jobs: Option<Vec<J2kClassicEncodeBatchJob>>,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1DensityReadback {
    buffer: Buffer,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1SymbolPlanReadback {
    buffer: Buffer,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1PassPlanReadback {
    buffer: Buffer,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1TokenEmitReadback {
    counter_buffer: Buffer,
    token_buffer: Option<Buffer>,
    segment_buffer: Option<Buffer>,
    token_stride_bytes: usize,
    token_segment_stride: usize,
    count: usize,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1GpuTokenBuffers {
    counter_buffer: Buffer,
    token_buffer: Buffer,
    segment_buffer: Buffer,
    job_count: u32,
    token_stride_bytes: u32,
    token_segment_stride: u32,
}

#[cfg(target_os = "macos")]
struct J2kResidentClassicTier1SplitTokenBuffers {
    counter_buffer: Buffer,
    mq_token_buffer: Buffer,
    raw_token_buffer: Buffer,
    segment_buffer: Buffer,
    job_count: u32,
    mq_token_stride_bytes: u32,
    raw_token_stride_bytes: u32,
    token_segment_stride: u32,
}

#[cfg(target_os = "macos")]
pub(crate) struct J2kPendingResidentLosslessCodestreamBatch {
    device: Device,
    buffer: Buffer,
    byte_offsets: Vec<usize>,
    capacities: Vec<usize>,
    status_buffer: Buffer,
    packet_status_buffer: Buffer,
    tier1_status_readback: Option<J2kResidentTier1StatusReadback>,
    classic_tier1_density_readback: Option<J2kResidentClassicTier1DensityReadback>,
    classic_tier1_symbol_plan_readback: Option<J2kResidentClassicTier1SymbolPlanReadback>,
    classic_tier1_pass_plan_readback: Option<J2kResidentClassicTier1PassPlanReadback>,
    classic_tier1_token_emit_readback: Option<J2kResidentClassicTier1TokenEmitReadback>,
    classic_tier1_split_token_emit_readback: Option<J2kResidentClassicTier1SplitTokenBuffers>,
    classic_gpu_token_pack_used: bool,
    command_buffer: CommandBuffer,
    retained_command_buffers: Vec<CommandBuffer>,
    _retained_buffers: Vec<Buffer>,
    recyclable_private_buffers: Vec<(usize, Buffer)>,
    recyclable_shared_buffers: Vec<(usize, Buffer)>,
    gpu_stage_command_buffers: Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    stage_stats: J2kResidentEncodeStageStats,
    codestream_payload_copy_dispatched: bool,
    status_stage: &'static str,
    length_error: &'static str,
    capacity_error: &'static str,
}

#[cfg(target_os = "macos")]
struct J2kResidentEncodeGpuStageCommandBuffer {
    stage: J2kResidentEncodeGpuStage,
    command_buffer: CommandBuffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct J2kBatchedPacketPayloadCopyDispatch<'a> {
    payload_buffer: &'a Buffer,
    packet_output_buffer: &'a Buffer,
    packet_job_buffer: &'a Buffer,
    packet_status_buffer: &'a Buffer,
    packet_payload_copy_job_buffer: &'a Buffer,
    tile_count: u64,
    max_payload_copy_jobs_per_tile: u64,
    label: &'a str,
    signpost_name: HybridSignpostName,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum J2kResidentEncodeGpuStage {
    CoefficientPrep,
    CoefficientDeinterleaveRct,
    CoefficientDwt53,
    CoefficientDwt53Vertical,
    CoefficientDwt53Horizontal,
    CoefficientExtract,
    CoefficientCopy,
    ClassicBlock,
    ClassicTier1Density,
    ClassicTier1RawPack,
    ClassicTier1ArithmeticPack,
    ClassicTier1SymbolPlan,
    ClassicTier1PassPlan,
    ClassicTier1TokenEmit,
    ClassicTier1SplitTokenEmit,
    ClassicTier1TokenPack,
    HtBlock,
    PacketBlockPrep,
    Packetization,
    PacketPayloadCopy,
    CodestreamAssembly,
    CodestreamPayloadCopy,
}

#[cfg(target_os = "macos")]
struct PreparedLosslessBatchTile {
    coefficient_buffer: Buffer,
    coefficient_byte_offset: usize,
    coefficient_byte_len: usize,
    coefficient_buffer_is_batch_shared: bool,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    recyclable_private_buffers: Vec<(usize, Buffer)>,
    prepare_command_buffer: CommandBuffer,
    prepare_deinterleave_rct_command_buffer: Option<CommandBuffer>,
    prepare_dwt53_command_buffer: Option<CommandBuffer>,
    prepare_dwt53_vertical_command_buffers: Vec<CommandBuffer>,
    prepare_dwt53_horizontal_command_buffers: Vec<CommandBuffer>,
    prepare_coefficient_extract_command_buffer: Option<CommandBuffer>,
    deinterleave_status_buffer: Buffer,
    plane_buffers: Vec<Buffer>,
    scratch_buffers: Vec<Buffer>,
    coefficient_job_buffer: Buffer,
    resolution_count: u32,
    num_layers: u8,
    num_components: u8,
    code_block_count: u32,
    packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    resolutions: Vec<J2kResidentPacketizationResolution>,
    codestream: J2kLosslessCodestreamAssemblyJob,
}

#[cfg(target_os = "macos")]
pub(crate) fn wait_resident_lossless_codestream(
    pending: J2kPendingResidentLosslessCodestream,
) -> Result<J2kResidentLosslessCodestream, Error> {
    wait_resident_codestream_command_buffer(&pending.command_buffer);
    let gpu_duration = completed_command_buffers_gpu_duration(
        &pending.retained_command_buffers,
        &pending.command_buffer,
    );
    let status = unsafe {
        pending
            .status_buffer
            .contents()
            .cast::<J2kCodestreamAssemblyStatus>()
            .read()
    };
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            pending.status_stage,
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: pending.length_error.to_string(),
    })?;
    if data_len > pending.capacity {
        return Err(Error::MetalKernel {
            message: pending.capacity_error.to_string(),
        });
    }
    Ok(J2kResidentLosslessCodestream {
        buffer: pending.buffer,
        byte_offset: 0,
        byte_len: data_len,
        capacity: pending.capacity,
        gpu_duration,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn wait_resident_lossless_codestream_batch(
    pending: J2kPendingResidentLosslessCodestreamBatch,
) -> Result<J2kResidentLosslessCodestreamBatchResult, Error> {
    wait_resident_codestream_command_buffer(&pending.command_buffer);
    finish_completed_resident_lossless_codestream_batch(pending)
}

#[cfg(target_os = "macos")]
pub(crate) fn wait_resident_lossless_codestream_batches(
    pending_batches: Vec<J2kPendingResidentLosslessCodestreamBatch>,
) -> Result<Vec<J2kResidentLosslessCodestreamBatchResult>, Error> {
    if let Some(last) = pending_batches.last() {
        // These command buffers are submitted on the same Metal queue before
        // harvest, so completing the final one implies earlier chunks are done.
        wait_resident_codestream_command_buffer(&last.command_buffer);
    }
    pending_batches
        .into_iter()
        .map(finish_completed_resident_lossless_codestream_batch)
        .collect()
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn schedule_resident_tier1_status_readback(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    status_buffer: &Buffer,
    kind: J2kResidentTier1StatusKind,
    classic_style_flags: u32,
    classic_jobs: Option<&[J2kClassicEncodeBatchJob]>,
    count: usize,
    status_size: usize,
    profile_stages: bool,
) -> Result<Option<J2kResidentTier1StatusReadback>, Error> {
    if !profile_stages || count == 0 {
        return Ok(None);
    }
    let byte_len = count
        .checked_mul(status_size)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident Tier-1 status readback size overflow".to_string(),
        })?;
    let readback = runtime.device.new_buffer(
        byte_len.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(status_buffer, 0, &readback, 0, byte_len as u64);
    blit.end_encoding();
    Ok(Some(J2kResidentTier1StatusReadback {
        buffer: readback,
        kind,
        classic_style_flags,
        classic_jobs: classic_jobs.map(<[J2kClassicEncodeBatchJob]>::to_vec),
        count,
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_density_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1DensityReadback>, Error> {
    if !metal_profile_classic_tier1_density_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 density profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1DensityCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 density job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 density profile");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_density_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_density_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(J2kResidentClassicTier1DensityReadback {
        buffer: counter_buffer,
        count: tier1_jobs.len(),
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_raw_pack_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    tier1_output_capacity_total: usize,
) -> Result<Option<Buffer>, Error> {
    if !metal_profile_classic_tier1_raw_pack_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 raw-pack profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let raw_output_buffer = runtime.device.new_buffer(
        tier1_output_capacity_total.max(1) as u64,
        MTLResourceOptions::StorageModePrivate,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 raw-pack job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 raw-pack profile");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_raw_pack_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&raw_output_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_raw_pack_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(raw_output_buffer))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_arithmetic_pack_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    tier1_output_capacity_total: usize,
) -> Result<Option<Buffer>, Error> {
    if !metal_profile_classic_tier1_arithmetic_pack_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 arithmetic-pack profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let arithmetic_output_buffer = runtime.device.new_buffer(
        tier1_output_capacity_total.max(1) as u64,
        MTLResourceOptions::StorageModePrivate,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 arithmetic-pack job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 arithmetic-pack profile");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_arithmetic_pack_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&arithmetic_output_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_arithmetic_pack_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(arithmetic_output_buffer))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_symbol_plan_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1SymbolPlanReadback>, Error> {
    if !metal_profile_classic_tier1_symbol_plan_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 symbol-plan profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1SymbolPlanCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 symbol-plan job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 symbol plan");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_symbol_plan_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_symbol_plan_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(J2kResidentClassicTier1SymbolPlanReadback {
        buffer: counter_buffer,
        count: tier1_jobs.len(),
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_pass_plan_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1PassPlanReadback>, Error> {
    if !metal_profile_classic_tier1_pass_plan_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 pass-plan profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1PassPlanCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 pass-plan job count exceeds u32".to_string(),
    })?;
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 pass plan");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_pass_plan_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_pass_plan_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(J2kResidentClassicTier1PassPlanReadback {
        buffer: counter_buffer,
        count: tier1_jobs.len(),
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_token_emit_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1TokenEmitReadback>, Error> {
    if !metal_profile_classic_tier1_token_emit_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    if classic_encode_code_blocks_pipeline_kind(tier1_jobs)
        != J2kClassicEncodePipelineKind::BypassU16_32
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token-emitter profiling currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1SymbolPlanCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token buffer size overflow".to_string(),
        })?;
    let token_buffer = runtime.device.new_buffer(
        token_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer = runtime.device.new_buffer(
        segment_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 token-emitter job count exceeds u32".to_string(),
    })?;
    let token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token arena stride exceeds u32".to_string(),
        })?;
    let token_segment_stride =
        u32::try_from(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token segment stride exceeds u32".to_string(),
        })?;

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 token emit");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_token_emit_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_buffer(3, Some(&token_buffer), 0);
    encoder.set_buffer(4, Some(&segment_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        (&raw const token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        (&raw const token_segment_stride).cast(),
    );
    encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_token_emit_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(Some(J2kResidentClassicTier1TokenEmitReadback {
        counter_buffer,
        token_buffer: Some(token_buffer),
        segment_buffer: Some(segment_buffer),
        token_stride_bytes: CLASSIC_TIER1_TOKEN_ARENA_BYTES,
        token_segment_stride: CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY,
        count: tier1_jobs.len(),
    }))
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_split_token_emit_for_cpu_pack(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<J2kResidentClassicTier1SplitTokenBuffers, Error> {
    if !classic_tier1_gpu_token_pack_supported(tier1_jobs) {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic split-token route currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = runtime.device.new_buffer(
        (tier1_jobs.len().max(1) * size_of::<J2kClassicTier1SymbolPlanCounters>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mq_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token MQ buffer size overflow".to_string(),
        })?;
    let mq_token_buffer = runtime.device.new_buffer(
        mq_token_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let raw_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token raw buffer size overflow".to_string(),
        })?;
    let raw_token_buffer = runtime.device.new_buffer(
        raw_token_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split-token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer = runtime.device.new_buffer(
        segment_buffer_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic split-token job count exceeds u32".to_string(),
    })?;
    let mq_token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split-token MQ arena stride exceeds u32".to_string(),
        })?;
    let raw_token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split-token raw arena stride exceeds u32".to_string(),
        })?;
    let token_segment_stride =
        u32::try_from(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split-token segment stride exceeds u32".to_string(),
        })?;

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 split token emit");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_split_token_emit_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_buffer(3, Some(&mq_token_buffer), 0);
    encoder.set_buffer(4, Some(&raw_token_buffer), 0);
    encoder.set_buffer(5, Some(&segment_buffer), 0);
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        (&raw const mq_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        7,
        size_of::<u32>() as u64,
        (&raw const raw_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        8,
        size_of::<u32>() as u64,
        (&raw const token_segment_stride).cast(),
    );
    encoder.set_bytes(9, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_split_token_emit_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(J2kResidentClassicTier1SplitTokenBuffers {
        counter_buffer,
        mq_token_buffer,
        raw_token_buffer,
        segment_buffer,
        job_count,
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_split_token_emit_profile(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<Option<J2kResidentClassicTier1SplitTokenBuffers>, Error> {
    if !metal_profile_classic_tier1_split_token_emit_enabled() || tier1_jobs.is_empty() {
        return Ok(None);
    }
    dispatch_classic_tier1_split_token_emit_for_cpu_pack(
        runtime,
        command_buffer,
        coefficient_buffer,
        tier1_job_buffer,
        tier1_jobs,
    )
    .map(Some)
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_split_token_emit_for_gpu_pack(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    recyclable_private_buffers: &mut Vec<(usize, Buffer)>,
    use_mq_byte_emit: bool,
) -> Result<J2kResidentClassicTier1SplitTokenBuffers, Error> {
    if !classic_tier1_gpu_token_pack_supported(tier1_jobs) {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic split GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }
    #[cfg(test)]
    if use_mq_byte_emit {
        CLASSIC_SPLIT_MQ_BYTE_GPU_TOKEN_PACK_DISPATCHES.with(|dispatches| {
            dispatches.set(dispatches.get().saturating_add(1));
        });
    }

    let counter_buffer = take_recyclable_private_buffer(
        runtime,
        tier1_jobs
            .len()
            .max(1)
            .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal classic split GPU token counter buffer size overflow"
                    .to_string(),
            })?,
        recyclable_private_buffers,
    );
    let mq_token_arena_bytes = if use_mq_byte_emit {
        CLASSIC_TIER1_MQ_BYTE_TOKEN_ARENA_BYTES
    } else {
        CLASSIC_TIER1_TOKEN_ARENA_BYTES
    };
    let mq_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(mq_token_arena_bytes)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split GPU token MQ buffer size overflow".to_string(),
        })?;
    let mq_token_buffer =
        take_recyclable_private_buffer(runtime, mq_token_buffer_len, recyclable_private_buffers);
    let raw_token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split GPU token raw buffer size overflow".to_string(),
        })?;
    let raw_token_buffer =
        take_recyclable_private_buffer(runtime, raw_token_buffer_len, recyclable_private_buffers);
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic split GPU token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer =
        take_recyclable_private_buffer(runtime, segment_buffer_len, recyclable_private_buffers);
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic split GPU token job count exceeds u32".to_string(),
    })?;
    let mq_token_stride_bytes =
        u32::try_from(mq_token_arena_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split GPU token MQ arena stride exceeds u32".to_string(),
        })?;
    let raw_token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split GPU token raw arena stride exceeds u32".to_string(),
        })?;
    let token_segment_stride =
        u32::try_from(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic split GPU token segment stride exceeds u32".to_string(),
        })?;

    let emit_pipeline = if use_mq_byte_emit {
        &runtime.classic_tier1_split_mq_byte_token_emit_bypass_u16_32
    } else {
        &runtime.classic_tier1_split_token_emit_bypass_u16_32
    };

    let encoder = command_buffer.new_compute_command_encoder();
    if use_mq_byte_emit {
        label_compute_encoder(encoder, "J2K classic Tier-1 split MQ-byte token emit");
    } else {
        label_compute_encoder(encoder, "J2K classic Tier-1 split token emit");
    }
    encoder.set_compute_pipeline_state(emit_pipeline);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_buffer(3, Some(&mq_token_buffer), 0);
    encoder.set_buffer(4, Some(&raw_token_buffer), 0);
    encoder.set_buffer(5, Some(&segment_buffer), 0);
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        (&raw const mq_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        7,
        size_of::<u32>() as u64,
        (&raw const raw_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        8,
        size_of::<u32>() as u64,
        (&raw const token_segment_stride).cast(),
    );
    encoder.set_bytes(9, size_of::<u32>() as u64, (&raw const job_count).cast());
    dispatch_1d_pipeline(encoder, emit_pipeline, u64::from(job_count));
    encoder.end_encoding();

    Ok(J2kResidentClassicTier1SplitTokenBuffers {
        counter_buffer,
        mq_token_buffer,
        raw_token_buffer,
        segment_buffer,
        job_count,
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_token_emit_for_gpu_pack(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coefficient_buffer: &Buffer,
    tier1_job_buffer: &Buffer,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    recyclable_private_buffers: &mut Vec<(usize, Buffer)>,
) -> Result<J2kResidentClassicTier1GpuTokenBuffers, Error> {
    if !classic_tier1_gpu_token_pack_supported(tier1_jobs) {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
        });
    }

    let counter_buffer = take_recyclable_private_buffer(
        runtime,
        tier1_jobs
            .len()
            .max(1)
            .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token counter buffer size overflow".to_string(),
            })?,
        recyclable_private_buffers,
    );
    let token_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_ARENA_BYTES)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token buffer size overflow".to_string(),
        })?;
    let token_buffer =
        take_recyclable_private_buffer(runtime, token_buffer_len, recyclable_private_buffers);
    let segment_buffer_len = tier1_jobs
        .len()
        .max(1)
        .checked_mul(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY)
        .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token segment buffer size overflow".to_string(),
        })?;
    let segment_buffer =
        take_recyclable_private_buffer(runtime, segment_buffer_len, recyclable_private_buffers);
    let job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 token-emitter job count exceeds u32".to_string(),
    })?;
    let token_stride_bytes =
        u32::try_from(CLASSIC_TIER1_TOKEN_ARENA_BYTES).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token arena stride exceeds u32".to_string(),
        })?;
    let token_segment_stride =
        u32::try_from(CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token segment stride exceeds u32".to_string(),
        })?;

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 token emit");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_token_emit_bypass_u16_32);
    encoder.set_buffer(0, Some(coefficient_buffer), 0);
    encoder.set_buffer(1, Some(tier1_job_buffer), 0);
    encoder.set_buffer(2, Some(&counter_buffer), 0);
    encoder.set_buffer(3, Some(&token_buffer), 0);
    encoder.set_buffer(4, Some(&segment_buffer), 0);
    encoder.set_bytes(
        5,
        size_of::<u32>() as u64,
        (&raw const token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        6,
        size_of::<u32>() as u64,
        (&raw const token_segment_stride).cast(),
    );
    encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_token_emit_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(J2kResidentClassicTier1GpuTokenBuffers {
        counter_buffer,
        token_buffer,
        segment_buffer,
        job_count,
        token_stride_bytes,
        token_segment_stride,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_token_pack_from_gpu_tokens(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    tier1_job_buffer: &Buffer,
    token_buffers: &J2kResidentClassicTier1GpuTokenBuffers,
    tier1_output_buffer: &Buffer,
    tier1_status_buffer: &Buffer,
    tier1_segment_buffer: &Buffer,
) {
    #[cfg(test)]
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES.with(|dispatches| {
        dispatches.set(dispatches.get().saturating_add(1));
    });

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 token pack");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_token_pack_bypass_u16_32);
    encoder.set_buffer(0, Some(tier1_job_buffer), 0);
    encoder.set_buffer(1, Some(&token_buffers.counter_buffer), 0);
    encoder.set_buffer(2, Some(&token_buffers.token_buffer), 0);
    encoder.set_buffer(3, Some(&token_buffers.segment_buffer), 0);
    encoder.set_buffer(4, Some(tier1_output_buffer), 0);
    encoder.set_buffer(5, Some(tier1_status_buffer), 0);
    encoder.set_buffer(6, Some(tier1_segment_buffer), 0);
    encoder.set_bytes(
        7,
        size_of::<u32>() as u64,
        (&raw const token_buffers.token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        8,
        size_of::<u32>() as u64,
        (&raw const token_buffers.token_segment_stride).cast(),
    );
    encoder.set_bytes(
        9,
        size_of::<u32>() as u64,
        (&raw const token_buffers.job_count).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(token_buffers.job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_token_pack_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    tier1_job_buffer: &Buffer,
    token_buffers: &J2kResidentClassicTier1SplitTokenBuffers,
    tier1_output_buffer: &Buffer,
    tier1_status_buffer: &Buffer,
    tier1_segment_buffer: &Buffer,
) {
    #[cfg(test)]
    CLASSIC_GPU_TOKEN_PACK_DISPATCHES.with(|dispatches| {
        dispatches.set(dispatches.get().saturating_add(1));
    });

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K classic Tier-1 split token pack");
    encoder.set_compute_pipeline_state(&runtime.classic_tier1_split_token_pack_bypass_u16_32);
    encoder.set_buffer(0, Some(tier1_job_buffer), 0);
    encoder.set_buffer(1, Some(&token_buffers.counter_buffer), 0);
    encoder.set_buffer(2, Some(&token_buffers.mq_token_buffer), 0);
    encoder.set_buffer(3, Some(&token_buffers.raw_token_buffer), 0);
    encoder.set_buffer(4, Some(&token_buffers.segment_buffer), 0);
    encoder.set_buffer(5, Some(tier1_output_buffer), 0);
    encoder.set_buffer(6, Some(tier1_status_buffer), 0);
    encoder.set_buffer(7, Some(tier1_segment_buffer), 0);
    encoder.set_bytes(
        8,
        size_of::<u32>() as u64,
        (&raw const token_buffers.mq_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        9,
        size_of::<u32>() as u64,
        (&raw const token_buffers.raw_token_stride_bytes).cast(),
    );
    encoder.set_bytes(
        10,
        size_of::<u32>() as u64,
        (&raw const token_buffers.token_segment_stride).cast(),
    );
    encoder.set_bytes(
        11,
        size_of::<u32>() as u64,
        (&raw const token_buffers.job_count).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(token_buffers.job_count),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .classic_tier1_split_token_pack_bypass_u16_32
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn schedule_classic_tier1_gpu_token_pack_readback(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    token_buffers: &J2kResidentClassicTier1GpuTokenBuffers,
    profile_stages: bool,
) -> Result<Option<J2kResidentClassicTier1TokenEmitReadback>, Error> {
    if !profile_stages || token_buffers.job_count == 0 {
        return Ok(None);
    }

    let count = usize::try_from(token_buffers.job_count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic GPU token-pack readback job count exceeds usize".to_string(),
    })?;
    let token_stride_bytes =
        usize::try_from(token_buffers.token_stride_bytes).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic GPU token-pack token stride exceeds usize".to_string(),
        })?;
    let token_segment_stride =
        usize::try_from(token_buffers.token_segment_stride).map_err(|_| Error::MetalKernel {
            message: "J2K Metal classic GPU token-pack segment stride exceeds usize".to_string(),
        })?;
    let counter_byte_len = count
        .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal classic GPU token-pack counter readback size overflow".to_string(),
        })?;
    let counter_readback = runtime.device.new_buffer(
        counter_byte_len.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let copy_token_payloads = metal_profile_classic_tier1_token_pack_enabled();
    let (token_readback, token_byte_len) = if copy_token_payloads {
        let byte_len = count
            .checked_mul(token_stride_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal classic GPU token-pack token readback size overflow"
                    .to_string(),
            })?;
        (
            Some(runtime.device.new_buffer(
                byte_len.max(1) as u64,
                MTLResourceOptions::StorageModeShared,
            )),
            byte_len,
        )
    } else {
        (None, 0)
    };
    let (segment_readback, segment_byte_len) = if copy_token_payloads {
        let byte_len = count
            .checked_mul(token_segment_stride)
            .and_then(|segment_count| {
                segment_count.checked_mul(size_of::<J2kClassicTier1TokenSegment>())
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal classic GPU token-pack segment readback size overflow"
                    .to_string(),
            })?;
        (
            Some(runtime.device.new_buffer(
                byte_len.max(1) as u64,
                MTLResourceOptions::StorageModeShared,
            )),
            byte_len,
        )
    } else {
        (None, 0)
    };

    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(
        &token_buffers.counter_buffer,
        0,
        &counter_readback,
        0,
        counter_byte_len as u64,
    );
    if let Some(token_readback) = token_readback.as_ref() {
        blit.copy_from_buffer(
            &token_buffers.token_buffer,
            0,
            token_readback,
            0,
            token_byte_len as u64,
        );
    }
    if let Some(segment_readback) = segment_readback.as_ref() {
        blit.copy_from_buffer(
            &token_buffers.segment_buffer,
            0,
            segment_readback,
            0,
            segment_byte_len as u64,
        );
    }
    blit.end_encoding();

    Ok(Some(J2kResidentClassicTier1TokenEmitReadback {
        counter_buffer: counter_readback,
        token_buffer: token_readback,
        segment_buffer: segment_readback,
        token_stride_bytes,
        token_segment_stride,
        count,
    }))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct J2kClassicTier1PassClassCounts {
    arithmetic: usize,
    raw: usize,
    cleanup: usize,
    sigprop: usize,
    magref: usize,
    arithmetic_cleanup: usize,
    arithmetic_sigprop: usize,
    arithmetic_magref: usize,
    raw_sigprop: usize,
    raw_magref: usize,
}

#[cfg(target_os = "macos")]
fn classic_tier1_pass_class_counts(
    coding_passes: usize,
    style_flags: u32,
) -> J2kClassicTier1PassClassCounts {
    let selective_bypass =
        (style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0;
    let mut counts = J2kClassicTier1PassClassCounts::default();
    for coding_pass in 0..coding_passes {
        let pass_type = coding_pass % 3;
        let arithmetic = !selective_bypass || coding_pass <= 9 || pass_type == 0;
        match pass_type {
            0 => {
                counts.cleanup = counts.cleanup.saturating_add(1);
                counts.arithmetic_cleanup = counts.arithmetic_cleanup.saturating_add(1);
            }
            1 => {
                counts.sigprop = counts.sigprop.saturating_add(1);
                if arithmetic {
                    counts.arithmetic_sigprop = counts.arithmetic_sigprop.saturating_add(1);
                } else {
                    counts.raw_sigprop = counts.raw_sigprop.saturating_add(1);
                }
            }
            _ => {
                counts.magref = counts.magref.saturating_add(1);
                if arithmetic {
                    counts.arithmetic_magref = counts.arithmetic_magref.saturating_add(1);
                } else {
                    counts.raw_magref = counts.raw_magref.saturating_add(1);
                }
            }
        }
        if arithmetic {
            counts.arithmetic = counts.arithmetic.saturating_add(1);
        } else {
            counts.raw = counts.raw.saturating_add(1);
        }
    }
    counts
}

#[cfg(target_os = "macos")]
fn accumulate_classic_tier1_scan_estimates(
    stage_stats: &mut J2kResidentEncodeStageStats,
    pass_counts: J2kClassicTier1PassClassCounts,
    coeff_count: usize,
) {
    let full_scan_visits = pass_counts
        .cleanup
        .saturating_add(pass_counts.sigprop)
        .saturating_add(pass_counts.magref)
        .saturating_mul(coeff_count);
    stage_stats.tier1_full_scan_coeff_visit_count_total = stage_stats
        .tier1_full_scan_coeff_visit_count_total
        .saturating_add(full_scan_visits);
    stage_stats.max_tier1_full_scan_coeff_visits_per_block = stage_stats
        .max_tier1_full_scan_coeff_visits_per_block
        .max(full_scan_visits);
    stage_stats.tier1_arithmetic_scan_coeff_visit_count_total = stage_stats
        .tier1_arithmetic_scan_coeff_visit_count_total
        .saturating_add(pass_counts.arithmetic.saturating_mul(coeff_count));
    stage_stats.tier1_raw_scan_coeff_visit_count_total = stage_stats
        .tier1_raw_scan_coeff_visit_count_total
        .saturating_add(pass_counts.raw.saturating_mul(coeff_count));
    stage_stats.tier1_cleanup_scan_coeff_visit_count_total = stage_stats
        .tier1_cleanup_scan_coeff_visit_count_total
        .saturating_add(pass_counts.cleanup.saturating_mul(coeff_count));
    stage_stats.tier1_sigprop_scan_coeff_visit_count_total = stage_stats
        .tier1_sigprop_scan_coeff_visit_count_total
        .saturating_add(pass_counts.sigprop.saturating_mul(coeff_count));
    stage_stats.tier1_magref_scan_coeff_visit_count_total = stage_stats
        .tier1_magref_scan_coeff_visit_count_total
        .saturating_add(pass_counts.magref.saturating_mul(coeff_count));
}

#[cfg(target_os = "macos")]
fn record_classic_tier1_density_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1DensityReadback,
) -> Result<(), Error> {
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .buffer
                .contents()
                .cast::<J2kClassicTier1DensityCounters>(),
            readback.count,
        )
    };
    for counter in counters {
        stage_stats.tier1_sigprop_active_candidate_count_total = stage_stats
            .tier1_sigprop_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 sigprop candidate count exceeds usize"
                            .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_sigprop_new_significant_count_total = stage_stats
            .tier1_sigprop_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 sigprop significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_magref_active_candidate_count_total = stage_stats
            .tier1_magref_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.magref_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 magref candidate count exceeds usize"
                            .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_arithmetic_sigprop_active_candidate_count_total = stage_stats
            .tier1_arithmetic_sigprop_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.arithmetic_sigprop_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 arithmetic sigprop candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_arithmetic_sigprop_new_significant_count_total = stage_stats
            .tier1_arithmetic_sigprop_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.arithmetic_sigprop_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 arithmetic sigprop significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_raw_sigprop_active_candidate_count_total = stage_stats
            .tier1_raw_sigprop_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.raw_sigprop_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 raw sigprop candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_raw_sigprop_new_significant_count_total = stage_stats
            .tier1_raw_sigprop_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.raw_sigprop_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 raw sigprop significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_arithmetic_magref_active_candidate_count_total = stage_stats
            .tier1_arithmetic_magref_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.arithmetic_magref_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 arithmetic magref candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_raw_magref_active_candidate_count_total = stage_stats
            .tier1_raw_magref_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.raw_magref_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 raw magref candidate count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_cleanup_active_candidate_count_total = stage_stats
            .tier1_cleanup_active_candidate_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_active_candidates).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 cleanup candidate count exceeds usize"
                            .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_cleanup_new_significant_count_total = stage_stats
            .tier1_cleanup_new_significant_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_new_significant).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 cleanup significance count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_cleanup_rlc_stripe_count_total = stage_stats
            .tier1_cleanup_rlc_stripe_count_total
            .saturating_add(usize::try_from(counter.cleanup_rlc_stripes).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 cleanup RLC stripe count exceeds usize"
                        .to_string(),
                }
            })?);
        stage_stats.tier1_cleanup_rlc_zero_stripe_count_total = stage_stats
            .tier1_cleanup_rlc_zero_stripe_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_rlc_zero_stripes).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 cleanup zero-RLC stripe count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn record_classic_tier1_symbol_plan_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1SymbolPlanReadback,
) -> Result<(), Error> {
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            readback.count,
        )
    };
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 symbol plan",
                counter.code,
                counter.detail,
            ));
        }
        stage_stats.tier1_symbol_plan_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_mq_symbol_count_total
            .saturating_add(usize::try_from(counter.mq_symbol_count).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 symbol-plan MQ count exceeds usize"
                        .to_string(),
                }
            })?);
        stage_stats.tier1_symbol_plan_raw_bit_count_total = stage_stats
            .tier1_symbol_plan_raw_bit_count_total
            .saturating_add(usize::try_from(counter.raw_bit_count).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 symbol-plan raw bit count exceeds usize"
                        .to_string(),
                }
            })?);
        let mq_symbol_count =
            usize::try_from(counter.mq_symbol_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan MQ count exceeds usize".to_string(),
            })?;
        let raw_bit_count =
            usize::try_from(counter.raw_bit_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan raw bit count exceeds usize"
                    .to_string(),
            })?;
        stage_stats.max_tier1_symbol_plan_mq_symbols_per_block = stage_stats
            .max_tier1_symbol_plan_mq_symbols_per_block
            .max(mq_symbol_count);
        stage_stats.max_tier1_symbol_plan_raw_bits_per_block = stage_stats
            .max_tier1_symbol_plan_raw_bits_per_block
            .max(raw_bit_count);
        let mq_packed_bytes = mq_symbol_count
            .saturating_mul(6)
            .saturating_add(7)
            .checked_div(8)
            .unwrap_or(usize::MAX);
        let raw_packed_bytes = raw_bit_count
            .saturating_add(7)
            .checked_div(8)
            .unwrap_or(usize::MAX);
        let packed_token_bytes = mq_packed_bytes.saturating_add(raw_packed_bytes);
        stage_stats.tier1_symbol_plan_packed_token_bytes_total = stage_stats
            .tier1_symbol_plan_packed_token_bytes_total
            .saturating_add(packed_token_bytes);
        stage_stats.max_tier1_symbol_plan_packed_token_bytes_per_block = stage_stats
            .max_tier1_symbol_plan_packed_token_bytes_per_block
            .max(packed_token_bytes);
        stage_stats.tier1_symbol_plan_cleanup_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_cleanup_mq_symbol_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_mq_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan cleanup MQ count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_sigprop_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_sigprop_mq_symbol_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_mq_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan sigprop MQ count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_magref_mq_symbol_count_total = stage_stats
            .tier1_symbol_plan_magref_mq_symbol_count_total
            .saturating_add(
                usize::try_from(counter.magref_mq_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan magref MQ count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_raw_sigprop_bit_count_total = stage_stats
            .tier1_symbol_plan_raw_sigprop_bit_count_total
            .saturating_add(usize::try_from(counter.raw_sigprop_bit_count).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 symbol-plan raw sigprop bit count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.tier1_symbol_plan_raw_magref_bit_count_total = stage_stats
            .tier1_symbol_plan_raw_magref_bit_count_total
            .saturating_add(usize::try_from(counter.raw_magref_bit_count).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 symbol-plan raw magref bit count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.tier1_symbol_plan_cleanup_sign_symbol_count_total = stage_stats
            .tier1_symbol_plan_cleanup_sign_symbol_count_total
            .saturating_add(
                usize::try_from(counter.cleanup_sign_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan cleanup sign count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_sigprop_sign_symbol_count_total = stage_stats
            .tier1_symbol_plan_sigprop_sign_symbol_count_total
            .saturating_add(
                usize::try_from(counter.sigprop_sign_symbol_count).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 symbol-plan sigprop sign count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.tier1_symbol_plan_mq_symbol_hash_xor ^= usize::try_from(counter.mq_symbol_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan MQ hash exceeds usize".to_string(),
            })?;
        stage_stats.tier1_symbol_plan_raw_bit_hash_xor ^= usize::try_from(counter.raw_bit_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 symbol-plan raw hash exceeds usize".to_string(),
            })?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn record_classic_tier1_pass_plan_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1PassPlanReadback,
) -> Result<(), Error> {
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .buffer
                .contents()
                .cast::<J2kClassicTier1PassPlanCounters>(),
            readback.count,
        )
    };
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 pass plan",
                counter.code,
                counter.detail,
            ));
        }
        let mq_symbol_count =
            usize::try_from(counter.mq_symbol_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 pass-plan MQ count exceeds usize".to_string(),
            })?;
        let raw_bit_count =
            usize::try_from(counter.raw_bit_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 pass-plan raw bit count exceeds usize"
                    .to_string(),
            })?;
        stage_stats.tier1_pass_plan_mq_symbol_count_total = stage_stats
            .tier1_pass_plan_mq_symbol_count_total
            .saturating_add(mq_symbol_count);
        stage_stats.tier1_pass_plan_raw_bit_count_total = stage_stats
            .tier1_pass_plan_raw_bit_count_total
            .saturating_add(raw_bit_count);
        stage_stats.tier1_pass_plan_nonempty_mq_pass_count_total = stage_stats
            .tier1_pass_plan_nonempty_mq_pass_count_total
            .saturating_add(usize::try_from(counter.nonempty_mq_passes).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 pass-plan nonempty MQ pass count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.tier1_pass_plan_nonempty_raw_pass_count_total = stage_stats
            .tier1_pass_plan_nonempty_raw_pass_count_total
            .saturating_add(usize::try_from(counter.nonempty_raw_passes).map_err(|_| {
                Error::MetalKernel {
                    message:
                        "J2K Metal classic Tier-1 pass-plan nonempty raw pass count exceeds usize"
                            .to_string(),
                }
            })?);
        stage_stats.max_tier1_pass_plan_mq_symbols_per_pass =
            stage_stats.max_tier1_pass_plan_mq_symbols_per_pass.max(
                usize::try_from(counter.max_mq_symbols_per_pass).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 pass-plan max MQ pass count exceeds usize"
                                .to_string(),
                    }
                })?,
            );
        stage_stats.max_tier1_pass_plan_raw_bits_per_pass =
            stage_stats.max_tier1_pass_plan_raw_bits_per_pass.max(
                usize::try_from(counter.max_raw_bits_per_pass).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 pass-plan max raw pass count exceeds usize"
                        .to_string(),
                })?,
            );

        let pass_mq_total = counter.mq_symbols_by_pass.iter().try_fold(
            0usize,
            |acc, &value| -> Result<usize, Error> {
                Ok(acc.saturating_add(
                    usize::try_from(value).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 pass-plan MQ pass count exceeds usize"
                            .to_string(),
                    })?,
                ))
            },
        )?;
        let pass_raw_total = counter.raw_bits_by_pass.iter().try_fold(
            0usize,
            |acc, &value| -> Result<usize, Error> {
                Ok(acc.saturating_add(usize::try_from(value).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 pass-plan raw pass count exceeds usize"
                            .to_string(),
                    }
                })?))
            },
        )?;
        if pass_mq_total != mq_symbol_count || pass_raw_total != raw_bit_count {
            return Err(Error::MetalKernel {
                message: "J2K Metal classic Tier-1 pass-plan per-pass totals are inconsistent"
                    .to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn compare_classic_tier1_symbol_plan_and_pass_plan_counters(
    symbol_plan: &J2kResidentClassicTier1SymbolPlanReadback,
    pass_plan: &J2kResidentClassicTier1PassPlanReadback,
) -> Result<(), Error> {
    if symbol_plan.count != pass_plan.count {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 pass-plan comparison count mismatch".to_string(),
        });
    }
    let symbol_plan_counters = unsafe {
        core::slice::from_raw_parts(
            symbol_plan
                .buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            symbol_plan.count,
        )
    };
    let pass_plan_counters = unsafe {
        core::slice::from_raw_parts(
            pass_plan
                .buffer
                .contents()
                .cast::<J2kClassicTier1PassPlanCounters>(),
            pass_plan.count,
        )
    };
    for (idx, (plan, pass)) in symbol_plan_counters
        .iter()
        .zip(pass_plan_counters)
        .enumerate()
    {
        let plan_values = [
            plan.code,
            plan.detail,
            plan.coding_passes,
            plan.missing_bit_planes,
            plan.segment_count,
            plan.mq_symbol_count,
            plan.raw_bit_count,
        ];
        let pass_values = [
            pass.code,
            pass.detail,
            pass.coding_passes,
            pass.missing_bit_planes,
            pass.segment_count,
            pass.mq_symbol_count,
            pass.raw_bit_count,
        ];
        if plan_values != pass_values {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal classic Tier-1 pass-plan diverged from symbol plan at block {idx}"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn record_classic_tier1_token_emit_counters(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1TokenEmitReadback,
) -> Result<(), Error> {
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            readback.count,
        )
    };
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 token emit",
                counter.code,
                counter.detail,
            ));
        }
        let mq_symbol_count =
            usize::try_from(counter.mq_symbol_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter MQ count exceeds usize"
                    .to_string(),
            })?;
        let raw_bit_count =
            usize::try_from(counter.raw_bit_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter raw bit count exceeds usize"
                    .to_string(),
            })?;
        let segment_count =
            usize::try_from(counter.segment_count).map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter segment count exceeds usize"
                    .to_string(),
            })?;
        let token_bytes = mq_symbol_count
            .saturating_mul(6)
            .saturating_add(raw_bit_count)
            .saturating_add(7)
            .checked_div(8)
            .unwrap_or(usize::MAX);
        stage_stats.tier1_token_emit_mq_symbol_count_total = stage_stats
            .tier1_token_emit_mq_symbol_count_total
            .saturating_add(mq_symbol_count);
        stage_stats.tier1_token_emit_raw_bit_count_total = stage_stats
            .tier1_token_emit_raw_bit_count_total
            .saturating_add(raw_bit_count);
        stage_stats.tier1_token_emit_token_bytes_total = stage_stats
            .tier1_token_emit_token_bytes_total
            .saturating_add(token_bytes);
        stage_stats.max_tier1_token_emit_token_bytes_per_block = stage_stats
            .max_tier1_token_emit_token_bytes_per_block
            .max(token_bytes);
        stage_stats.tier1_token_emit_segment_count_total = stage_stats
            .tier1_token_emit_segment_count_total
            .saturating_add(segment_count);
        stage_stats.max_tier1_token_emit_segments_per_block = stage_stats
            .max_tier1_token_emit_segments_per_block
            .max(segment_count);
        stage_stats.tier1_token_emit_mq_symbol_hash_xor ^= usize::try_from(counter.mq_symbol_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter MQ hash exceeds usize".to_string(),
            })?;
        stage_stats.tier1_token_emit_raw_bit_hash_xor ^= usize::try_from(counter.raw_bit_hash)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal classic Tier-1 token-emitter raw hash exceeds usize"
                    .to_string(),
            })?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn compare_classic_tier1_symbol_plan_and_token_emit_counters(
    symbol_plan: &J2kResidentClassicTier1SymbolPlanReadback,
    token_emit: &J2kResidentClassicTier1TokenEmitReadback,
) -> Result<(), Error> {
    if symbol_plan.count != token_emit.count {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 token-emitter comparison count mismatch".to_string(),
        });
    }
    let symbol_plan_counters = unsafe {
        core::slice::from_raw_parts(
            symbol_plan
                .buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            symbol_plan.count,
        )
    };
    let token_emit_counters = unsafe {
        core::slice::from_raw_parts(
            token_emit
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            token_emit.count,
        )
    };
    for (idx, (plan, emit)) in symbol_plan_counters
        .iter()
        .zip(token_emit_counters)
        .enumerate()
    {
        let plan_values = [
            plan.code,
            plan.detail,
            plan.coding_passes,
            plan.missing_bit_planes,
            plan.segment_count,
            plan.mq_symbol_count,
            plan.raw_bit_count,
            plan.cleanup_mq_symbol_count,
            plan.sigprop_mq_symbol_count,
            plan.magref_mq_symbol_count,
            plan.raw_sigprop_bit_count,
            plan.raw_magref_bit_count,
            plan.cleanup_sign_symbol_count,
            plan.sigprop_sign_symbol_count,
            plan.mq_symbol_hash,
            plan.raw_bit_hash,
        ];
        let emit_values = [
            emit.code,
            emit.detail,
            emit.coding_passes,
            emit.missing_bit_planes,
            emit.segment_count,
            emit.mq_symbol_count,
            emit.raw_bit_count,
            emit.cleanup_mq_symbol_count,
            emit.sigprop_mq_symbol_count,
            emit.magref_mq_symbol_count,
            emit.raw_sigprop_bit_count,
            emit.raw_magref_bit_count,
            emit.cleanup_sign_symbol_count,
            emit.sigprop_sign_symbol_count,
            emit.mq_symbol_hash,
            emit.raw_bit_hash,
        ];
        if plan_values != emit_values {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal classic Tier-1 token-emitter diverged from symbol plan at block {idx}"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_classic_tier1_split_token_emit_counters(
    readback: &J2kResidentClassicTier1SplitTokenBuffers,
) -> Result<(), Error> {
    if readback.mq_token_stride_bytes == 0
        || readback.raw_token_stride_bytes == 0
        || readback.token_segment_stride == 0
    {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 split-token readback has empty stride".to_string(),
        });
    }
    let count = usize::try_from(readback.job_count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 split-token counter count exceeds usize".to_string(),
    })?;
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            count,
        )
    };
    for counter in counters {
        if counter.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1 split-token emit",
                counter.code,
                counter.detail,
            ));
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn compare_classic_tier1_symbol_plan_and_split_token_emit_counters(
    symbol_plan: &J2kResidentClassicTier1SymbolPlanReadback,
    split_emit: &J2kResidentClassicTier1SplitTokenBuffers,
) -> Result<(), Error> {
    let split_count = usize::try_from(split_emit.job_count).map_err(|_| Error::MetalKernel {
        message: "J2K Metal classic Tier-1 split-token comparison count exceeds usize".to_string(),
    })?;
    if symbol_plan.count != split_count {
        return Err(Error::MetalKernel {
            message: "J2K Metal classic Tier-1 split-token comparison count mismatch".to_string(),
        });
    }
    let symbol_plan_counters = unsafe {
        core::slice::from_raw_parts(
            symbol_plan
                .buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            symbol_plan.count,
        )
    };
    let split_emit_counters = unsafe {
        core::slice::from_raw_parts(
            split_emit
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            split_count,
        )
    };
    for (idx, (plan, emit)) in symbol_plan_counters
        .iter()
        .zip(split_emit_counters)
        .enumerate()
    {
        let plan_values = [
            plan.code,
            plan.detail,
            plan.coding_passes,
            plan.missing_bit_planes,
            plan.segment_count,
            plan.mq_symbol_count,
            plan.raw_bit_count,
            plan.cleanup_mq_symbol_count,
            plan.sigprop_mq_symbol_count,
            plan.magref_mq_symbol_count,
            plan.raw_sigprop_bit_count,
            plan.raw_magref_bit_count,
            plan.cleanup_sign_symbol_count,
            plan.sigprop_sign_symbol_count,
            plan.mq_symbol_hash,
            plan.raw_bit_hash,
        ];
        let emit_values = [
            emit.code,
            emit.detail,
            emit.coding_passes,
            emit.missing_bit_planes,
            emit.segment_count,
            emit.mq_symbol_count,
            emit.raw_bit_count,
            emit.cleanup_mq_symbol_count,
            emit.sigprop_mq_symbol_count,
            emit.magref_mq_symbol_count,
            emit.raw_sigprop_bit_count,
            emit.raw_magref_bit_count,
            emit.cleanup_sign_symbol_count,
            emit.sigprop_sign_symbol_count,
            emit.mq_symbol_hash,
            emit.raw_bit_hash,
        ];
        if plan_values != emit_values {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal classic Tier-1 split-token emitter diverged from symbol plan at block {idx}"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn profile_classic_tier1_token_pack(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentClassicTier1TokenEmitReadback,
) -> Result<(), Error> {
    if !metal_profile_classic_tier1_token_pack_enabled() {
        return Ok(());
    }
    let counters = unsafe {
        core::slice::from_raw_parts(
            readback
                .counter_buffer
                .contents()
                .cast::<J2kClassicTier1SymbolPlanCounters>(),
            readback.count,
        )
    };
    let token_buffer = readback
        .token_buffer
        .as_ref()
        .ok_or_else(|| Error::MetalKernel {
            message:
                "J2K Metal classic Tier-1 token-pack profiling requires token payload readback"
                    .to_string(),
        })?;
    let segment_buffer = readback
        .segment_buffer
        .as_ref()
        .ok_or_else(|| Error::MetalKernel {
            message:
                "J2K Metal classic Tier-1 token-pack profiling requires token segment readback"
                    .to_string(),
        })?;
    let token_bytes = unsafe {
        core::slice::from_raw_parts(
            token_buffer.contents().cast::<u8>(),
            readback.count.saturating_mul(readback.token_stride_bytes),
        )
    };
    let token_segments = unsafe {
        core::slice::from_raw_parts(
            segment_buffer
                .contents()
                .cast::<J2kClassicTier1TokenSegment>(),
            readback.count.saturating_mul(readback.token_segment_stride),
        )
    };
    let token_stride_bytes = readback.token_stride_bytes;
    let token_segment_stride = readback.token_segment_stride;

    let started = Instant::now();
    let packed_lengths = (0..readback.count)
        .into_par_iter()
        .map(|block_idx| -> Result<usize, String> {
            let counter = &counters[block_idx];
            if counter.code != J2K_ENCODE_STATUS_OK {
                return Err(format!(
                "classic Tier-1 token pack input failed at block {block_idx}: code={} detail={}",
                counter.code, counter.detail
            ));
            }
            let segment_count = usize::try_from(counter.segment_count)
                .map_err(|_| "J2K Metal classic Tier-1 token-pack segment count exceeds usize")?;
            if segment_count > CLASSIC_TIER1_TOKEN_SEGMENT_CAPACITY {
                return Err(
                    "J2K Metal classic Tier-1 token-pack segment count exceeds capacity"
                        .to_string(),
                );
            }
            let token_start = block_idx
                .checked_mul(token_stride_bytes)
                .ok_or("J2K Metal classic Tier-1 token-pack byte offset overflow")?;
            let segment_start = block_idx
                .checked_mul(token_segment_stride)
                .ok_or("J2K Metal classic Tier-1 token-pack segment offset overflow")?;
            let mut native_segments = Vec::with_capacity(segment_count);
            for segment in &token_segments[segment_start..segment_start + segment_count] {
                let start_coding_pass = u8::try_from(segment.pass_range & 0xFFFF)
                    .map_err(|_| "J2K Metal classic Tier-1 token-pack start pass exceeds u8")?;
                let end_coding_pass = u8::try_from(segment.pass_range >> 16)
                    .map_err(|_| "J2K Metal classic Tier-1 token-pack end pass exceeds u8")?;
                native_segments.push(J2kTier1TokenSegment {
                    token_bit_offset: segment.token_bit_offset,
                    token_bit_count: segment.token_bit_count,
                    start_coding_pass,
                    end_coding_pass,
                    use_arithmetic: (segment.flags & 1) != 0,
                });
            }
            let packed = pack_j2k_code_block_scalar_from_tier1_tokens(
                &token_bytes[token_start..token_start + token_stride_bytes],
                &native_segments,
                u8::try_from(counter.coding_passes).map_err(|_| {
                    "J2K Metal classic Tier-1 token-pack coding-pass count exceeds u8"
                })?,
                u8::try_from(counter.missing_bit_planes).map_err(|_| {
                    "J2K Metal classic Tier-1 token-pack missing bitplanes exceed u8"
                })?,
            )
            .map_err(|message| format!("J2K Metal classic Tier-1 token-pack failed: {message}"))?;
            Ok(packed.data.len())
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|message| Error::MetalKernel { message })?;
    for output_len in packed_lengths {
        stage_stats.tier1_token_pack_output_bytes_total = stage_stats
            .tier1_token_pack_output_bytes_total
            .saturating_add(output_len);
        stage_stats.max_tier1_token_pack_output_bytes_per_block = stage_stats
            .max_tier1_token_pack_output_bytes_per_block
            .max(output_len);
    }
    stage_stats.classic_tier1_token_pack_duration = started.elapsed();
    Ok(())
}

#[cfg(target_os = "macos")]
fn record_resident_tier1_output_usage(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentTier1StatusReadback,
    classic_gpu_token_pack_used: bool,
) -> Result<(), Error> {
    match readback.kind {
        J2kResidentTier1StatusKind::Classic => {
            let classic_jobs =
                readback
                    .classic_jobs
                    .as_ref()
                    .ok_or_else(|| Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 profile readback is missing job metadata"
                                .to_string(),
                    })?;
            let statuses = unsafe {
                core::slice::from_raw_parts(
                    readback.buffer.contents().cast::<J2kClassicEncodeStatus>(),
                    readback.count,
                )
            };
            if classic_jobs.len() != statuses.len() {
                return Err(Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 profile readback job/status count mismatch"
                        .to_string(),
                });
            }
            for (status, job) in statuses.iter().zip(classic_jobs) {
                if status.code != J2K_ENCODE_STATUS_OK {
                    return Err(encode_status_error(
                        "classic Tier-1",
                        status.code,
                        status.detail,
                    ));
                }
                let data_len =
                    usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 output length exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_output_used_bytes_total = stage_stats
                    .tier1_output_used_bytes_total
                    .saturating_add(data_len);
                stage_stats.max_tier1_output_used_bytes =
                    stage_stats.max_tier1_output_used_bytes.max(data_len);
                if classic_gpu_token_pack_used {
                    stage_stats.tier1_token_pack_output_bytes_total = stage_stats
                        .tier1_token_pack_output_bytes_total
                        .saturating_add(data_len);
                    stage_stats.max_tier1_token_pack_output_bytes_per_block = stage_stats
                        .max_tier1_token_pack_output_bytes_per_block
                        .max(data_len);
                }
                let coding_passes =
                    usize::try_from(status.number_of_coding_passes).map_err(|_| {
                        Error::MetalKernel {
                            message: "J2K Metal classic Tier-1 coding-pass count exceeds usize"
                                .to_string(),
                        }
                    })?;
                stage_stats.tier1_coding_pass_count_total = stage_stats
                    .tier1_coding_pass_count_total
                    .saturating_add(coding_passes);
                stage_stats.max_tier1_coding_passes_per_block = stage_stats
                    .max_tier1_coding_passes_per_block
                    .max(coding_passes);
                let pass_counts =
                    classic_tier1_pass_class_counts(coding_passes, readback.classic_style_flags);
                let coeff_count = usize::try_from(job.width)
                    .and_then(|width| {
                        usize::try_from(job.height).map(|height| width.saturating_mul(height))
                    })
                    .map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 code-block dimensions exceed usize"
                            .to_string(),
                    })?;
                accumulate_classic_tier1_scan_estimates(stage_stats, pass_counts, coeff_count);
                stage_stats.tier1_arithmetic_pass_count_total = stage_stats
                    .tier1_arithmetic_pass_count_total
                    .saturating_add(pass_counts.arithmetic);
                stage_stats.tier1_raw_pass_count_total = stage_stats
                    .tier1_raw_pass_count_total
                    .saturating_add(pass_counts.raw);
                stage_stats.tier1_cleanup_pass_count_total = stage_stats
                    .tier1_cleanup_pass_count_total
                    .saturating_add(pass_counts.cleanup);
                stage_stats.tier1_sigprop_pass_count_total = stage_stats
                    .tier1_sigprop_pass_count_total
                    .saturating_add(pass_counts.sigprop);
                stage_stats.tier1_magref_pass_count_total = stage_stats
                    .tier1_magref_pass_count_total
                    .saturating_add(pass_counts.magref);
                stage_stats.tier1_arithmetic_cleanup_pass_count_total = stage_stats
                    .tier1_arithmetic_cleanup_pass_count_total
                    .saturating_add(pass_counts.arithmetic_cleanup);
                stage_stats.tier1_arithmetic_sigprop_pass_count_total = stage_stats
                    .tier1_arithmetic_sigprop_pass_count_total
                    .saturating_add(pass_counts.arithmetic_sigprop);
                stage_stats.tier1_arithmetic_magref_pass_count_total = stage_stats
                    .tier1_arithmetic_magref_pass_count_total
                    .saturating_add(pass_counts.arithmetic_magref);
                stage_stats.tier1_raw_sigprop_pass_count_total = stage_stats
                    .tier1_raw_sigprop_pass_count_total
                    .saturating_add(pass_counts.raw_sigprop);
                stage_stats.tier1_raw_magref_pass_count_total = stage_stats
                    .tier1_raw_magref_pass_count_total
                    .saturating_add(pass_counts.raw_magref);
                if coding_passes == 0 {
                    stage_stats.tier1_zero_block_count_total =
                        stage_stats.tier1_zero_block_count_total.saturating_add(1);
                } else {
                    stage_stats.tier1_nonzero_block_count_total = stage_stats
                        .tier1_nonzero_block_count_total
                        .saturating_add(1);
                }
                let missing_bitplanes =
                    usize::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 missing-bitplane count exceeds usize"
                            .to_string(),
                    })?;
                stage_stats.tier1_missing_bitplane_count_total = stage_stats
                    .tier1_missing_bitplane_count_total
                    .saturating_add(missing_bitplanes);
                stage_stats.max_tier1_missing_bitplanes_per_block = stage_stats
                    .max_tier1_missing_bitplanes_per_block
                    .max(missing_bitplanes);
                let segment_count =
                    usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 segment count exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_segment_count_total = stage_stats
                    .tier1_segment_count_total
                    .saturating_add(segment_count);
                stage_stats.max_tier1_segments_per_block =
                    stage_stats.max_tier1_segments_per_block.max(segment_count);
            }
        }
        J2kResidentTier1StatusKind::HighThroughput => {
            let statuses = unsafe {
                core::slice::from_raw_parts(
                    readback.buffer.contents().cast::<J2kHtEncodeStatus>(),
                    readback.count,
                )
            };
            for status in statuses {
                if status.code != J2K_ENCODE_STATUS_OK {
                    return Err(encode_status_error(
                        "HTJ2K Tier-1",
                        status.code,
                        status.detail,
                    ));
                }
                let data_len =
                    usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal Tier-1 output length exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_output_used_bytes_total = stage_stats
                    .tier1_output_used_bytes_total
                    .saturating_add(data_len);
                stage_stats.max_tier1_output_used_bytes =
                    stage_stats.max_tier1_output_used_bytes.max(data_len);
                let coding_passes =
                    usize::try_from(status.num_coding_passes).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal Tier-1 coding-pass count exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_coding_pass_count_total = stage_stats
                    .tier1_coding_pass_count_total
                    .saturating_add(coding_passes);
                stage_stats.max_tier1_coding_passes_per_block = stage_stats
                    .max_tier1_coding_passes_per_block
                    .max(coding_passes);
                if coding_passes == 0 {
                    stage_stats.tier1_zero_block_count_total =
                        stage_stats.tier1_zero_block_count_total.saturating_add(1);
                } else {
                    stage_stats.tier1_nonzero_block_count_total = stage_stats
                        .tier1_nonzero_block_count_total
                        .saturating_add(1);
                }
                let missing_bitplanes =
                    usize::try_from(status.num_zero_bitplanes).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal Tier-1 missing-bitplane count exceeds usize"
                            .to_string(),
                    })?;
                stage_stats.tier1_missing_bitplane_count_total = stage_stats
                    .tier1_missing_bitplane_count_total
                    .saturating_add(missing_bitplanes);
                stage_stats.max_tier1_missing_bitplanes_per_block = stage_stats
                    .max_tier1_missing_bitplanes_per_block
                    .max(missing_bitplanes);
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn wait_resident_codestream_command_buffer(command_buffer: &CommandBufferRef) {
    #[cfg(test)]
    RESIDENT_CODESTREAM_COMMAND_BUFFER_WAITS.with(|waits| waits.set(waits.get() + 1));
    let _signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT);
    command_buffer.wait_until_completed();
}

#[cfg(target_os = "macos")]
fn finish_completed_resident_lossless_codestream_batch(
    pending: J2kPendingResidentLosslessCodestreamBatch,
) -> Result<J2kResidentLosslessCodestreamBatchResult, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST);
    let profile_stages = metal_profile_stages_enabled();
    let result_harvest_started = profile_stages.then(Instant::now);
    let gpu_timings = completed_command_buffers_gpu_duration_and_elapsed_window(
        &pending.retained_command_buffers,
        &pending.command_buffer,
    );
    let gpu_duration = gpu_timings.map(|timings| timings.0);
    let gpu_elapsed_wall_duration = gpu_timings.map(|timings| timings.1);
    let mut stage_stats = pending.stage_stats;
    if let Some(duration) = gpu_elapsed_wall_duration {
        stage_stats.gpu_elapsed_wall_duration = duration;
    }
    if profile_stages {
        record_completed_resident_encode_gpu_stages(
            &mut stage_stats,
            &pending.gpu_stage_command_buffers,
        );
    }
    if let Some(readback) = pending.tier1_status_readback.as_ref() {
        record_resident_tier1_output_usage(
            &mut stage_stats,
            readback,
            pending.classic_gpu_token_pack_used,
        )?;
    }
    if let Some(readback) = pending.classic_tier1_density_readback.as_ref() {
        record_classic_tier1_density_counters(&mut stage_stats, readback)?;
    }
    if let Some(readback) = pending.classic_tier1_symbol_plan_readback.as_ref() {
        record_classic_tier1_symbol_plan_counters(&mut stage_stats, readback)?;
    }
    if let (Some(symbol_plan), Some(pass_plan)) = (
        pending.classic_tier1_symbol_plan_readback.as_ref(),
        pending.classic_tier1_pass_plan_readback.as_ref(),
    ) {
        compare_classic_tier1_symbol_plan_and_pass_plan_counters(symbol_plan, pass_plan)?;
    }
    if let Some(readback) = pending.classic_tier1_pass_plan_readback.as_ref() {
        record_classic_tier1_pass_plan_counters(&mut stage_stats, readback)?;
    }
    if let (Some(symbol_plan), Some(token_emit)) = (
        pending.classic_tier1_symbol_plan_readback.as_ref(),
        pending.classic_tier1_token_emit_readback.as_ref(),
    ) {
        compare_classic_tier1_symbol_plan_and_token_emit_counters(symbol_plan, token_emit)?;
    }
    if let Some(readback) = pending.classic_tier1_token_emit_readback.as_ref() {
        record_classic_tier1_token_emit_counters(&mut stage_stats, readback)?;
        profile_classic_tier1_token_pack(&mut stage_stats, readback)?;
    }
    if let Some(readback) = pending.classic_tier1_split_token_emit_readback.as_ref() {
        validate_classic_tier1_split_token_emit_counters(readback)?;
    }
    if let (Some(symbol_plan), Some(split_emit)) = (
        pending.classic_tier1_symbol_plan_readback.as_ref(),
        pending.classic_tier1_split_token_emit_readback.as_ref(),
    ) {
        compare_classic_tier1_symbol_plan_and_split_token_emit_counters(symbol_plan, split_emit)?;
    }
    let recyclable_private_buffers = pending.recyclable_private_buffers;
    let private_recycle_started = profile_stages.then(Instant::now);
    with_runtime_for_device(&pending.device, |runtime| {
        recycle_private_buffers(runtime, recyclable_private_buffers);
        Ok(())
    })?;
    if let Some(started) = private_recycle_started {
        stage_stats.result_private_recycle_duration = started.elapsed();
    }
    let gpu_duration_share =
        gpu_duration.map(|duration| duration_share(duration, pending.capacities.len()));
    let status_copy_started = profile_stages.then(Instant::now);
    let statuses = unsafe {
        core::slice::from_raw_parts(
            pending
                .status_buffer
                .contents()
                .cast::<J2kCodestreamAssemblyStatus>(),
            pending.capacities.len(),
        )
    }
    .to_vec();
    let packet_statuses = unsafe {
        core::slice::from_raw_parts(
            pending
                .packet_status_buffer
                .contents()
                .cast::<J2kPacketEncodeStatus>(),
            pending.capacities.len(),
        )
    }
    .to_vec();
    if let Some(started) = status_copy_started {
        stage_stats.result_status_copy_duration = started.elapsed();
    }
    let recyclable_shared_buffers = pending.recyclable_shared_buffers;
    let shared_recycle_started = profile_stages.then(Instant::now);
    with_runtime_for_device(&pending.device, |runtime| {
        recycle_shared_buffers(runtime, recyclable_shared_buffers);
        Ok(())
    })?;
    if let Some(started) = shared_recycle_started {
        stage_stats.result_shared_recycle_duration = started.elapsed();
    }
    let codestream_collect_started = profile_stages.then(Instant::now);
    let mut codestreams = Vec::with_capacity(pending.capacities.len());
    for (index, status) in statuses.into_iter().enumerate() {
        let packet_status = packet_statuses
            .get(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal packetization status missing for resident batch tile"
                    .to_string(),
            })?;
        if packet_status.code != J2K_ENCODE_STATUS_OK {
            return Err(packet_encode_status_error(*packet_status));
        }
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                pending.status_stage,
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: pending.length_error.to_string(),
        })?;
        let capacity = pending.capacities[index];
        if data_len > capacity {
            return Err(Error::MetalKernel {
                message: pending.capacity_error.to_string(),
            });
        }
        let packet_output_used =
            usize::try_from(packet_status.data_len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packet output length exceeds usize".to_string(),
            })?;
        let packet_payload_copy_jobs =
            usize::try_from(packet_status.detail).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packet payload-copy count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_bytes =
            usize::try_from(packet_status.payload_copy_bytes).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packet payload-copy byte count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_small_jobs = usize::try_from(packet_status.payload_copy_small_jobs)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal small packet payload-copy count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_medium_jobs =
            usize::try_from(packet_status.payload_copy_medium_jobs).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal medium packet payload-copy count exceeds usize".to_string(),
                }
            })?;
        let packet_payload_copy_large_jobs = usize::try_from(packet_status.payload_copy_large_jobs)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal large packet payload-copy count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_active_stripes =
            packet_payload_copy_jobs.saturating_mul(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB as usize);
        stage_stats.packet_output_used_bytes_total = stage_stats
            .packet_output_used_bytes_total
            .saturating_add(packet_output_used);
        stage_stats.max_packet_output_used_bytes = stage_stats
            .max_packet_output_used_bytes
            .max(packet_output_used);
        stage_stats.packet_payload_copy_job_count_total = stage_stats
            .packet_payload_copy_job_count_total
            .saturating_add(packet_payload_copy_jobs);
        stage_stats.max_packet_payload_copy_jobs_used_per_tile = stage_stats
            .max_packet_payload_copy_jobs_used_per_tile
            .max(packet_payload_copy_jobs);
        stage_stats.packet_payload_copy_bytes_total = stage_stats
            .packet_payload_copy_bytes_total
            .saturating_add(packet_payload_copy_bytes);
        stage_stats.max_packet_payload_copy_bytes_per_tile = stage_stats
            .max_packet_payload_copy_bytes_per_tile
            .max(packet_payload_copy_bytes);
        stage_stats.packet_payload_copy_small_job_count_total = stage_stats
            .packet_payload_copy_small_job_count_total
            .saturating_add(packet_payload_copy_small_jobs);
        stage_stats.packet_payload_copy_medium_job_count_total = stage_stats
            .packet_payload_copy_medium_job_count_total
            .saturating_add(packet_payload_copy_medium_jobs);
        stage_stats.packet_payload_copy_large_job_count_total = stage_stats
            .packet_payload_copy_large_job_count_total
            .saturating_add(packet_payload_copy_large_jobs);
        stage_stats.packet_payload_copy_active_stripe_count_total = stage_stats
            .packet_payload_copy_active_stripe_count_total
            .saturating_add(packet_payload_copy_active_stripes);
        if pending.codestream_payload_copy_dispatched {
            stage_stats.codestream_payload_copy_bytes_total = stage_stats
                .codestream_payload_copy_bytes_total
                .saturating_add(packet_output_used);
        }
        codestreams.push(J2kResidentLosslessCodestream {
            buffer: pending.buffer.clone(),
            byte_offset: pending.byte_offsets[index],
            byte_len: data_len,
            capacity,
            gpu_duration: gpu_duration_share,
        });
    }
    if let Some(started) = codestream_collect_started {
        stage_stats.result_codestream_collect_duration = started.elapsed();
    }
    if let Some(started) = result_harvest_started {
        stage_stats.result_harvest_duration = started.elapsed();
    }
    Ok(J2kResidentLosslessCodestreamBatchResult {
        codestreams,
        stage_stats,
    })
}

#[cfg(target_os = "macos")]
fn duration_share(duration: Duration, count: usize) -> Duration {
    if count == 0 {
        return Duration::ZERO;
    }
    let nanos = duration.as_nanos() / count as u128;
    Duration::from_nanos(nanos.min(u128::from(u64::MAX)) as u64)
}

#[cfg(target_os = "macos")]
fn record_completed_resident_encode_gpu_stages(
    stats: &mut J2kResidentEncodeStageStats,
    command_buffers: &[J2kResidentEncodeGpuStageCommandBuffer],
) {
    for stage_command_buffer in command_buffers {
        let Some(duration) =
            completed_command_buffer_gpu_duration(&stage_command_buffer.command_buffer)
        else {
            continue;
        };
        match stage_command_buffer.stage {
            J2kResidentEncodeGpuStage::CoefficientPrep => {
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientDeinterleaveRct => {
                stats.coefficient_deinterleave_rct_gpu_duration = stats
                    .coefficient_deinterleave_rct_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientDwt53 => {
                stats.coefficient_dwt53_gpu_duration = stats
                    .coefficient_dwt53_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientDwt53Vertical => {
                stats.coefficient_dwt53_vertical_gpu_duration = stats
                    .coefficient_dwt53_vertical_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_dwt53_gpu_duration = stats
                    .coefficient_dwt53_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientDwt53Horizontal => {
                stats.coefficient_dwt53_horizontal_gpu_duration = stats
                    .coefficient_dwt53_horizontal_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_dwt53_gpu_duration = stats
                    .coefficient_dwt53_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientExtract => {
                stats.coefficient_extract_gpu_duration = stats
                    .coefficient_extract_gpu_duration
                    .saturating_add(duration);
                stats.coefficient_prep_gpu_duration =
                    stats.coefficient_prep_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CoefficientCopy => {
                stats.coefficient_copy_gpu_duration =
                    stats.coefficient_copy_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicBlock => {
                stats.classic_block_gpu_duration =
                    stats.classic_block_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1Density => {
                stats.classic_tier1_density_gpu_duration = stats
                    .classic_tier1_density_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1RawPack => {
                stats.classic_tier1_raw_pack_gpu_duration = stats
                    .classic_tier1_raw_pack_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1ArithmeticPack => {
                stats.classic_tier1_arithmetic_pack_gpu_duration = stats
                    .classic_tier1_arithmetic_pack_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1SymbolPlan => {
                stats.classic_tier1_symbol_plan_gpu_duration = stats
                    .classic_tier1_symbol_plan_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1PassPlan => {
                stats.classic_tier1_pass_plan_gpu_duration = stats
                    .classic_tier1_pass_plan_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1TokenEmit => {
                stats.classic_tier1_token_emit_gpu_duration = stats
                    .classic_tier1_token_emit_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1SplitTokenEmit => {
                stats.classic_tier1_split_token_emit_gpu_duration = stats
                    .classic_tier1_split_token_emit_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::ClassicTier1TokenPack => {
                stats.classic_tier1_token_pack_gpu_duration = stats
                    .classic_tier1_token_pack_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::HtBlock => {
                stats.ht_block_gpu_duration = stats.ht_block_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::PacketBlockPrep => {
                stats.packet_block_prep_gpu_duration = stats
                    .packet_block_prep_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::Packetization => {
                stats.packetization_gpu_duration =
                    stats.packetization_gpu_duration.saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::PacketPayloadCopy => {
                stats.packet_payload_copy_gpu_duration = stats
                    .packet_payload_copy_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CodestreamAssembly => {
                stats.codestream_assembly_gpu_duration = stats
                    .codestream_assembly_gpu_duration
                    .saturating_add(duration);
            }
            J2kResidentEncodeGpuStage::CodestreamPayloadCopy => {
                stats.codestream_payload_copy_gpu_duration = stats
                    .codestream_payload_copy_gpu_duration
                    .saturating_add(duration);
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn new_resident_encode_command_buffer(runtime: &MetalRuntime, label: &str) -> CommandBuffer {
    let command_buffer = runtime.queue.new_command_buffer().to_owned();
    label_command_buffer(&command_buffer, label);
    command_buffer
}

#[cfg(target_os = "macos")]
fn finish_resident_encode_split_command_buffer(
    command_buffer: CommandBuffer,
    runtime: &MetalRuntime,
    stage: J2kResidentEncodeGpuStage,
    next_label: &str,
    command_buffers: &mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
) -> CommandBuffer {
    command_buffer.commit();
    command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
        stage,
        command_buffer,
    });
    new_resident_encode_command_buffer(runtime, next_label)
}

#[cfg(target_os = "macos")]
fn finish_resident_encode_split_command_buffer_timed(
    command_buffer: CommandBuffer,
    runtime: &MetalRuntime,
    stage: J2kResidentEncodeGpuStage,
    next_label: &str,
    command_buffers: &mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    profile_stages: bool,
    accumulated: &mut Duration,
) -> CommandBuffer {
    let started = profile_stages.then(Instant::now);
    let next = finish_resident_encode_split_command_buffer(
        command_buffer,
        runtime,
        stage,
        next_label,
        command_buffers,
    );
    if let Some(started) = started {
        *accumulated = accumulated.saturating_add(started.elapsed());
    }
    next
}

#[cfg(target_os = "macos")]
fn dispatch_batched_packet_payload_copy(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    dispatch: J2kBatchedPacketPayloadCopyDispatch<'_>,
) -> bool {
    if dispatch.tile_count == 0 || dispatch.max_payload_copy_jobs_per_tile == 0 {
        return false;
    }

    let signpost = hybrid_stage_signpost(dispatch.signpost_name);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, dispatch.label);
    encoder.set_compute_pipeline_state(&runtime.packet_payload_copy_batched);
    encoder.set_buffer(0, Some(dispatch.payload_buffer), 0);
    encoder.set_buffer(1, Some(dispatch.packet_output_buffer), 0);
    encoder.set_buffer(2, Some(dispatch.packet_job_buffer), 0);
    encoder.set_buffer(3, Some(dispatch.packet_status_buffer), 0);
    encoder.set_buffer(4, Some(dispatch.packet_payload_copy_job_buffer), 0);
    let params = J2kPacketPayloadCopyParams {
        bytes_per_thread: PACKET_PAYLOAD_COPY_BYTES_PER_STRIPE,
        stripes_per_job: PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
    };
    encoder.set_bytes(
        5,
        size_of::<J2kPacketPayloadCopyParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: dispatch.max_payload_copy_jobs_per_tile,
            height: dispatch.tile_count,
            depth: u64::from(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB),
        },
        MTLSize {
            width: runtime
                .packet_payload_copy_batched
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    drop(signpost);
    true
}

#[cfg(target_os = "macos")]
fn record_completed_decode_split_gpu_stages(
    timings: &mut DirectHybridStageTimings,
    command_buffers: &DecodeHybridSplitCommandBuffers,
) {
    let mut gpu_command = Duration::ZERO;
    let mut idwt_gpu = Duration::ZERO;
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.idwt_interleave)
    {
        timings.metal_idwt_interleave_gpu = timings
            .metal_idwt_interleave_gpu
            .saturating_add(duration.as_micros());
        idwt_gpu = idwt_gpu.saturating_add(duration);
        gpu_command = gpu_command.saturating_add(duration);
    }
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.idwt_horizontal)
    {
        timings.metal_idwt_horizontal_gpu = timings
            .metal_idwt_horizontal_gpu
            .saturating_add(duration.as_micros());
        idwt_gpu = idwt_gpu.saturating_add(duration);
        gpu_command = gpu_command.saturating_add(duration);
    }
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.idwt_vertical) {
        timings.metal_idwt_vertical_gpu = timings
            .metal_idwt_vertical_gpu
            .saturating_add(duration.as_micros());
        idwt_gpu = idwt_gpu.saturating_add(duration);
        gpu_command = gpu_command.saturating_add(duration);
    }
    timings.metal_idwt_gpu = timings.metal_idwt_gpu.saturating_add(idwt_gpu.as_micros());
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.store) {
        timings.metal_store_gpu = timings.metal_store_gpu.saturating_add(duration.as_micros());
        gpu_command = gpu_command.saturating_add(duration);
    }
    if let Some(duration) = completed_command_buffer_gpu_duration(&command_buffers.mct_pack) {
        timings.metal_mct_pack_gpu = timings
            .metal_mct_pack_gpu
            .saturating_add(duration.as_micros());
        gpu_command = gpu_command.saturating_add(duration);
    }
    timings.gpu_command = timings.gpu_command.saturating_add(gpu_command.as_micros());
}

#[cfg(target_os = "macos")]
fn completed_command_buffers_gpu_duration(
    retained: &[CommandBuffer],
    final_buffer: &CommandBufferRef,
) -> Option<Duration> {
    completed_command_buffers_gpu_duration_and_elapsed_window(retained, final_buffer)
        .map(|(duration, _window)| duration)
}

#[cfg(target_os = "macos")]
fn completed_command_buffers_gpu_duration_and_elapsed_window(
    retained: &[CommandBuffer],
    final_buffer: &CommandBufferRef,
) -> Option<(Duration, Duration)> {
    let mut total = Duration::ZERO;
    let mut min_start = f64::INFINITY;
    let mut max_end = f64::NEG_INFINITY;
    let mut seen = Vec::with_capacity(retained.len().saturating_add(1));
    for command_buffer in retained {
        let ptr = command_buffer.as_ptr();
        if seen.contains(&ptr) {
            continue;
        }
        seen.push(ptr);
        let (start, end) = completed_command_buffer_gpu_times(command_buffer)?;
        total = total.saturating_add(Duration::from_secs_f64(end - start));
        min_start = min_start.min(start);
        max_end = max_end.max(end);
    }
    let final_ptr = final_buffer.as_ptr();
    if !seen.contains(&final_ptr) {
        let (start, end) = completed_command_buffer_gpu_times(final_buffer)?;
        total = total.saturating_add(Duration::from_secs_f64(end - start));
        min_start = min_start.min(start);
        max_end = max_end.max(end);
    }
    if min_start.is_finite() && max_end.is_finite() && max_end > min_start {
        Some((total, Duration::from_secs_f64(max_end - min_start)))
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
fn completed_command_buffer_gpu_duration(command_buffer: &CommandBufferRef) -> Option<Duration> {
    let (start, end) = completed_command_buffer_gpu_times(command_buffer)?;
    Some(Duration::from_secs_f64(end - start))
}

#[cfg(target_os = "macos")]
fn completed_command_buffer_gpu_times(command_buffer: &CommandBufferRef) -> Option<(f64, f64)> {
    #[cfg(test)]
    RESIDENT_GPU_TIMESTAMP_QUERIES.with(|queries| queries.set(queries.get() + 1));

    let start: f64 = unsafe {
        command_buffer
            .send_message::<(), f64>(Sel::register("GPUStartTime"), ())
            .ok()?
    };
    let end: f64 = unsafe {
        command_buffer
            .send_message::<(), f64>(Sel::register("GPUEndTime"), ())
            .ok()?
    };
    if start.is_finite() && end.is_finite() && end > start {
        Some((start, end))
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
fn dispatch_lossless_deinterleave(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: J2kLosslessDevicePrepareJob<'_>,
    plane0: &Buffer,
    plane1: &Buffer,
    plane2: &Buffer,
) -> Result<(), Error> {
    let input_byte_offset =
        u64::try_from(job.input_byte_offset).map_err(|_| Error::MetalKernel {
            message: "J2K Metal resident encode input offset exceeds u64".to_string(),
        })?;
    let src_stride = u32::try_from(job.input_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal resident encode input pitch exceeds u32".to_string(),
    })?;
    let sample_offset = if job.bit_depth == 0 || job.bit_depth > 16 {
        return Err(Error::MetalKernel {
            message: "J2K Metal resident encode bit depth must be 1-16".to_string(),
        });
    } else {
        1u32 << (u32::from(job.bit_depth) - 1)
    };
    let params = J2kLosslessDeinterleaveParams {
        src_width: job.input_width,
        src_height: job.input_height,
        src_stride,
        dst_width: job.output_width,
        dst_height: job.output_height,
        components: u32::from(job.components),
        bytes_per_sample: u32::from(job.bytes_per_sample),
        sample_offset,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep deinterleave");
    encoder.set_compute_pipeline_state(&runtime.lossless_deinterleave_to_planes);
    encoder.set_buffer(0, Some(job.input), input_byte_offset);
    encoder.set_buffer(1, Some(plane0), 0);
    encoder.set_buffer(2, Some(plane1), 0);
    encoder.set_buffer(3, Some(plane2), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kLosslessDeinterleaveParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.lossless_deinterleave_to_planes,
        (job.output_width, job.output_height),
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_lossless_deinterleave_rct_rgb8(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: J2kLosslessDevicePrepareJob<'_>,
    plane0: &Buffer,
    plane1: &Buffer,
    plane2: &Buffer,
    status_buffer: &Buffer,
) -> Result<(), Error> {
    let input_byte_offset =
        u64::try_from(job.input_byte_offset).map_err(|_| Error::MetalKernel {
            message: "J2K Metal resident encode input offset exceeds u64".to_string(),
        })?;
    let src_stride = u32::try_from(job.input_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal resident encode input pitch exceeds u32".to_string(),
    })?;
    let sample_offset = if job.bit_depth == 0 || job.bit_depth > 16 {
        return Err(Error::MetalKernel {
            message: "J2K Metal resident encode bit depth must be 1-16".to_string(),
        });
    } else {
        1u32 << (u32::from(job.bit_depth) - 1)
    };
    let params = J2kLosslessDeinterleaveParams {
        src_width: job.input_width,
        src_height: job.input_height,
        src_stride,
        dst_width: job.output_width,
        dst_height: job.output_height,
        components: u32::from(job.components),
        bytes_per_sample: u32::from(job.bytes_per_sample),
        sample_offset,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep deinterleave + RCT");
    encoder.set_compute_pipeline_state(&runtime.lossless_deinterleave_rct_rgb8_to_planes);
    encoder.set_buffer(0, Some(job.input), input_byte_offset);
    encoder.set_buffer(1, Some(plane0), 0);
    encoder.set_buffer(2, Some(plane1), 0);
    encoder.set_buffer(3, Some(plane2), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kLosslessDeinterleaveParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(5, Some(status_buffer), 0);
    dispatch_2d_pipeline(
        encoder,
        &runtime.lossless_deinterleave_rct_rgb8_to_planes,
        (job.output_width, job.output_height),
    );
    encoder.end_encoding();
    #[cfg(test)]
    LOSSLESS_DEINTERLEAVE_RCT_FUSED_DISPATCHES.with(|dispatches| {
        dispatches.set(dispatches.get().saturating_add(1));
    });
    Ok(())
}

#[cfg(target_os = "macos")]
fn lossless_deinterleave_rct_rgb8_supported(job: J2kLosslessDevicePrepareJob<'_>) -> bool {
    job.components == 3 && job.bytes_per_sample == 1
}

#[cfg(target_os = "macos")]
fn dispatch_forward_rct_on_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane0: &Buffer,
    plane1: &Buffer,
    plane2: &Buffer,
    len: usize,
    status_buffer: &Buffer,
) -> Result<(), Error> {
    if len == 0 {
        return Ok(());
    }
    let params = J2kForwardRctParams {
        _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
            message: "J2K Metal resident encode RCT length exceeds u32".to_string(),
        })?,
        _reserved0: 0,
        _reserved1: 0,
        _reserved2: 0,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep RCT");
    encoder.set_compute_pipeline_state(&runtime.forward_rct);
    encoder.set_buffer(0, Some(plane0), 0);
    encoder.set_buffer(1, Some(plane1), 0);
    encoder.set_buffer(2, Some(plane2), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kForwardRctParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(4, Some(status_buffer), 0);
    let width = runtime
        .forward_rct
        .thread_execution_width()
        .max(1)
        .min(len as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_on_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    scratch: &Buffer,
    width: u32,
    height: u32,
    num_levels: u8,
) -> Buffer {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53Params {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
        };

        if current_height >= 2 {
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_vertical,
                command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_horizontal,
                command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    if active_is_input {
        input.to_owned()
    } else {
        scratch.to_owned()
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_forward_dwt53_components_on_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane_buffers: &[Buffer],
    scratch_buffers: &[Buffer],
    width: u32,
    height: u32,
    num_levels: u8,
    component_count: usize,
) -> Vec<Buffer> {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let component_count_u32 = component_count as u32;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53BatchedParams {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
            component_count: component_count_u32,
        };

        if current_height >= 2 {
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_vertical_batched,
                command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_horizontal_batched,
                command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    let active_buffers = if active_is_input {
        plane_buffers
    } else {
        scratch_buffers
    };
    active_buffers[..component_count].to_vec()
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_on_buffers_split_profile(
    runtime: &MetalRuntime,
    input: &Buffer,
    scratch: &Buffer,
    width: u32,
    height: u32,
    num_levels: u8,
) -> (Buffer, Vec<CommandBuffer>, Vec<CommandBuffer>) {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let mut vertical_command_buffers = Vec::new();
    let mut horizontal_command_buffers = Vec::new();

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53Params {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
        };

        if current_height >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "signinum-j2k coefficient prep DWT 5/3 vertical",
            );
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_vertical,
                &command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            command_buffer.commit();
            vertical_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "signinum-j2k coefficient prep DWT 5/3 horizontal",
            );
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_horizontal,
                &command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            command_buffer.commit();
            horizontal_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    let active = if active_is_input {
        input.to_owned()
    } else {
        scratch.to_owned()
    };
    (active, vertical_command_buffers, horizontal_command_buffers)
}

#[cfg(target_os = "macos")]
fn dispatch_forward_dwt53_components_split_profile(
    runtime: &MetalRuntime,
    plane_buffers: &[Buffer],
    scratch_buffers: &[Buffer],
    width: u32,
    height: u32,
    num_levels: u8,
    component_count: usize,
) -> (Vec<Buffer>, Vec<CommandBuffer>, Vec<CommandBuffer>) {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let mut vertical_command_buffers = Vec::new();
    let mut horizontal_command_buffers = Vec::new();
    let component_count_u32 = component_count as u32;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53BatchedParams {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
            component_count: component_count_u32,
        };

        if current_height >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "signinum-j2k coefficient prep DWT 5/3 vertical",
            );
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_vertical_batched,
                &command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            command_buffer.commit();
            vertical_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "signinum-j2k coefficient prep DWT 5/3 horizontal",
            );
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_horizontal_batched,
                &command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            command_buffer.commit();
            horizontal_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    let active_buffers = if active_is_input {
        plane_buffers
    } else {
        scratch_buffers
    };
    (
        active_buffers[..component_count].to_vec(),
        vertical_command_buffers,
        horizontal_command_buffers,
    )
}

#[cfg(target_os = "macos")]
fn dispatch_lossless_extract_coefficients(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: &[Buffer],
    coefficient_buffer: &Buffer,
    coefficient_jobs: &[J2kLosslessCoefficientJob],
    output_width: u32,
) -> Result<Buffer, Error> {
    let coefficient_job_buffer = copied_slice_buffer(&runtime.device, coefficient_jobs);
    let job_count = u32::try_from(coefficient_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal resident encode coefficient job count exceeds u32".to_string(),
    })?;
    let max_block_width = coefficient_jobs
        .iter()
        .map(|job| job.block_width)
        .max()
        .unwrap_or(1);
    let max_block_height = coefficient_jobs
        .iter()
        .map(|job| job.block_height)
        .max()
        .unwrap_or(1);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep extract");
    encoder.set_compute_pipeline_state(&runtime.lossless_extract_coefficients);
    encoder.set_buffer(0, planes.first().map(|buffer| &**buffer), 0);
    encoder.set_buffer(
        1,
        planes
            .get(1)
            .or_else(|| planes.first())
            .map(|buffer| &**buffer),
        0,
    );
    encoder.set_buffer(
        2,
        planes
            .get(2)
            .or_else(|| planes.first())
            .map(|buffer| &**buffer),
        0,
    );
    encoder.set_buffer(3, Some(coefficient_buffer), 0);
    encoder.set_buffer(4, Some(&coefficient_job_buffer), 0);
    encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
    dispatch_3d_pipeline(
        encoder,
        &runtime.lossless_extract_coefficients,
        (max_block_width, max_block_height, job_count),
    );
    encoder.end_encoding();
    let _ = output_width;
    Ok(coefficient_job_buffer)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct J2kLosslessPrepareSizes {
    plane_len: usize,
    plane_bytes: usize,
    coefficient_bytes: usize,
}

#[cfg(target_os = "macos")]
fn lossless_prepare_sizes(
    job: J2kLosslessDevicePrepareJob<'_>,
) -> Result<J2kLosslessPrepareSizes, Error> {
    if job.components != 1 && job.components != 3 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode supports grayscale or RGB input",
        });
    }
    if job.bytes_per_sample != 1 && job.bytes_per_sample != 2 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode supports 8-bit or 16-bit samples",
        });
    }
    let plane_len = (job.output_width as usize)
        .checked_mul(job.output_height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident encode plane size overflow".to_string(),
        })?;
    let plane_bytes =
        plane_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal resident encode plane byte size overflow".to_string(),
            })?;
    let coefficient_bytes = job
        .coefficient_count
        .max(1)
        .checked_mul(size_of::<i32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident encode coefficient size overflow".to_string(),
        })?;
    Ok(J2kLosslessPrepareSizes {
        plane_len,
        plane_bytes,
        coefficient_bytes,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_lossless_device_code_blocks(
    session: &crate::MetalBackendSession,
    job: J2kLosslessDevicePrepareJob<'_>,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
) -> Result<J2kPreparedLosslessDeviceCodeBlocks, Error> {
    let sizes = lossless_prepare_sizes(job)?;

    with_runtime_for_device(&session.device, |runtime| {
        let mut plane_buffers = Vec::with_capacity(3);
        let mut scratch_buffers = Vec::with_capacity(usize::from(job.components));
        for _ in 0..3 {
            plane_buffers.push(runtime.device.new_buffer(
                sizes.plane_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ));
        }
        for _ in 0..job.components {
            scratch_buffers.push(runtime.device.new_buffer(
                sizes.plane_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ));
        }
        let coefficient_buffer = runtime.device.new_buffer(
            sizes.coefficient_bytes as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let deinterleave_status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const deinterleave_status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = runtime.queue.new_command_buffer();

        if lossless_deinterleave_rct_rgb8_supported(job) {
            dispatch_lossless_deinterleave_rct_rgb8(
                runtime,
                command_buffer,
                job,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
                &status_buffer,
            )?;
        } else {
            dispatch_lossless_deinterleave(
                runtime,
                command_buffer,
                job,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
            )?;
        }
        if job.components == 3 && !lossless_deinterleave_rct_rgb8_supported(job) {
            dispatch_forward_rct_on_buffers(
                runtime,
                command_buffer,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
                sizes.plane_len,
                &status_buffer,
            )?;
        }

        let mut active_planes = Vec::with_capacity(usize::from(job.components));
        for component in 0..usize::from(job.components) {
            if job.num_decomposition_levels == 0 {
                active_planes.push(plane_buffers[component].clone());
            } else {
                active_planes.push(dispatch_forward_dwt53_on_buffers(
                    runtime,
                    command_buffer,
                    &plane_buffers[component],
                    &scratch_buffers[component],
                    job.output_width,
                    job.output_height,
                    job.num_decomposition_levels,
                ));
            }
        }
        while active_planes.len() < 3 {
            active_planes.push(active_planes[0].clone());
        }

        let coefficient_jobs = code_blocks
            .iter()
            .map(|block| J2kLosslessCoefficientJob {
                coefficient_offset: block.coefficient_offset,
                component: block.component,
                subband_x: block.subband_x,
                subband_y: block.subband_y,
                block_x: block.block_x,
                block_y: block.block_y,
                block_width: block.width,
                block_height: block.height,
                full_width: job.output_width,
            })
            .collect::<Vec<_>>();
        let coefficient_job_buffer = dispatch_lossless_extract_coefficients(
            runtime,
            command_buffer,
            &active_planes,
            &coefficient_buffer,
            &coefficient_jobs,
            job.output_width,
        )?;

        command_buffer.commit();
        Ok(J2kPreparedLosslessDeviceCodeBlocks {
            coefficient_buffer,
            coefficient_byte_offset: 0,
            coefficient_byte_len: sizes.coefficient_bytes,
            coefficient_buffer_is_batch_shared: false,
            code_blocks,
            recyclable_private_buffers: Vec::new(),
            _prepare_command_buffer: command_buffer.to_owned(),
            _prepare_deinterleave_rct_command_buffer: None,
            _prepare_dwt53_command_buffer: None,
            _prepare_dwt53_vertical_command_buffers: Vec::new(),
            _prepare_dwt53_horizontal_command_buffers: Vec::new(),
            _prepare_coefficient_extract_command_buffer: None,
            _deinterleave_status_buffer: status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_lossless_device_code_blocks_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kLosslessDeviceBatchPrepareItem<'_>>,
) -> Result<Vec<J2kPreparedLosslessDeviceCodeBlocks>, Error> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let mut sizes = Vec::with_capacity(items.len());
    let mut coefficient_byte_offsets = Vec::with_capacity(items.len());
    let mut total_coefficient_bytes = 0usize;
    for item in &items {
        let item_sizes = lossless_prepare_sizes(item.job).map_err(|err| Error::MetalKernel {
            message: format!(
                "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                item.tile_index
            ),
        })?;
        coefficient_byte_offsets.push(total_coefficient_bytes);
        total_coefficient_bytes = total_coefficient_bytes
            .checked_add(item_sizes.coefficient_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal resident batch coefficient size overflow".to_string(),
            })?;
        sizes.push(item_sizes);
    }

    with_runtime_for_device(&session.device, |runtime| {
        let mut shared_recyclable_private_buffers = Vec::new();
        let coefficient_buffer = take_recyclable_private_buffer(
            runtime,
            total_coefficient_bytes.max(1),
            &mut shared_recyclable_private_buffers,
        );
        let split_prepare_command_buffers = metal_profile_coefficient_prep_split_commands_enabled();
        let shared_command_buffer = if split_prepare_command_buffers {
            None
        } else {
            Some(runtime.queue.new_command_buffer().to_owned())
        };
        let mut prepared = Vec::with_capacity(items.len());

        for ((item, item_sizes), coefficient_byte_offset) in
            items.into_iter().zip(sizes).zip(coefficient_byte_offsets)
        {
            let job = item.job;
            let mut recyclable_private_buffers = Vec::new();
            if !shared_recyclable_private_buffers.is_empty() {
                recyclable_private_buffers.append(&mut shared_recyclable_private_buffers);
            }
            let mut plane_buffers = Vec::with_capacity(3);
            let mut scratch_buffers = Vec::with_capacity(usize::from(job.components));
            for _ in 0..3 {
                plane_buffers.push(take_recyclable_private_buffer(
                    runtime,
                    item_sizes.plane_bytes,
                    &mut recyclable_private_buffers,
                ));
            }
            for _ in 0..job.components {
                scratch_buffers.push(take_recyclable_private_buffer(
                    runtime,
                    item_sizes.plane_bytes,
                    &mut recyclable_private_buffers,
                ));
            }

            let deinterleave_status = J2kMctStatus::default();
            let status_buffer = runtime.device.new_buffer_with_data(
                (&raw const deinterleave_status).cast(),
                size_of::<J2kMctStatus>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let mut prepare_deinterleave_rct_command_buffer = None;
            let prepare_dwt53_command_buffer = None;
            let mut prepare_dwt53_vertical_command_buffers = Vec::new();
            let mut prepare_dwt53_horizontal_command_buffers = Vec::new();
            let mut prepare_coefficient_extract_command_buffer = None;
            let deinterleave_command_buffer = if split_prepare_command_buffers {
                new_resident_encode_command_buffer(
                    runtime,
                    "signinum-j2k coefficient prep deinterleave rct",
                )
            } else {
                shared_command_buffer
                    .as_ref()
                    .expect("shared coefficient prep command buffer exists")
                    .clone()
            };
            if lossless_deinterleave_rct_rgb8_supported(job) {
                dispatch_lossless_deinterleave_rct_rgb8(
                    runtime,
                    &deinterleave_command_buffer,
                    job,
                    &plane_buffers[0],
                    &plane_buffers[1],
                    &plane_buffers[2],
                    &status_buffer,
                )
            } else {
                dispatch_lossless_deinterleave(
                    runtime,
                    &deinterleave_command_buffer,
                    job,
                    &plane_buffers[0],
                    &plane_buffers[1],
                    &plane_buffers[2],
                )
            }
            .map_err(|err| Error::MetalKernel {
                message: format!(
                    "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                    item.tile_index
                ),
            })?;
            if job.components == 3 && !lossless_deinterleave_rct_rgb8_supported(job) {
                dispatch_forward_rct_on_buffers(
                    runtime,
                    &deinterleave_command_buffer,
                    &plane_buffers[0],
                    &plane_buffers[1],
                    &plane_buffers[2],
                    item_sizes.plane_len,
                    &status_buffer,
                )
                .map_err(|err| Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                        item.tile_index
                    ),
                })?;
            }
            if split_prepare_command_buffers {
                deinterleave_command_buffer.commit();
                prepare_deinterleave_rct_command_buffer = Some(deinterleave_command_buffer);
            }

            let mut active_planes = Vec::with_capacity(usize::from(job.components));
            if job.num_decomposition_levels == 0 {
                active_planes.extend(
                    plane_buffers
                        .iter()
                        .take(usize::from(job.components))
                        .cloned(),
                );
            } else if split_prepare_command_buffers {
                let component_count = usize::from(job.components);
                if component_count > 1 {
                    let (
                        mut component_active_planes,
                        mut vertical_command_buffers,
                        mut horizontal_command_buffers,
                    ) = dispatch_forward_dwt53_components_split_profile(
                        runtime,
                        &plane_buffers,
                        &scratch_buffers,
                        job.output_width,
                        job.output_height,
                        job.num_decomposition_levels,
                        component_count,
                    );
                    active_planes.append(&mut component_active_planes);
                    prepare_dwt53_vertical_command_buffers.append(&mut vertical_command_buffers);
                    prepare_dwt53_horizontal_command_buffers
                        .append(&mut horizontal_command_buffers);
                } else {
                    for component in 0..component_count {
                        let (
                            active_plane,
                            mut vertical_command_buffers,
                            mut horizontal_command_buffers,
                        ) = dispatch_forward_dwt53_on_buffers_split_profile(
                            runtime,
                            &plane_buffers[component],
                            &scratch_buffers[component],
                            job.output_width,
                            job.output_height,
                            job.num_decomposition_levels,
                        );
                        active_planes.push(active_plane);
                        prepare_dwt53_vertical_command_buffers
                            .append(&mut vertical_command_buffers);
                        prepare_dwt53_horizontal_command_buffers
                            .append(&mut horizontal_command_buffers);
                    }
                }
            } else {
                let dwt_command_buffer_ref = shared_command_buffer
                    .as_ref()
                    .expect("shared coefficient prep command buffer exists");
                let component_count = usize::from(job.components);
                if component_count > 1 {
                    active_planes = dispatch_forward_dwt53_components_on_buffers(
                        runtime,
                        dwt_command_buffer_ref,
                        &plane_buffers,
                        &scratch_buffers,
                        job.output_width,
                        job.output_height,
                        job.num_decomposition_levels,
                        component_count,
                    );
                } else {
                    for component in 0..component_count {
                        active_planes.push(dispatch_forward_dwt53_on_buffers(
                            runtime,
                            dwt_command_buffer_ref,
                            &plane_buffers[component],
                            &scratch_buffers[component],
                            job.output_width,
                            job.output_height,
                            job.num_decomposition_levels,
                        ));
                    }
                }
            }
            while active_planes.len() < 3 {
                active_planes.push(active_planes[0].clone());
            }

            let coefficient_word_offset = coefficient_byte_offset
                .checked_div(size_of::<i32>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal resident batch coefficient offset division failed"
                        .to_string(),
                })?;
            let coefficient_word_offset_u32 =
                u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident batch coefficient offset exceeds u32 at tile {}",
                        item.tile_index
                    ),
                })?;
            let coefficient_jobs = item
                .code_blocks
                .iter()
                .map(|block| {
                    let coefficient_offset = block
                        .coefficient_offset
                        .checked_add(coefficient_word_offset_u32)
                        .ok_or_else(|| Error::MetalKernel {
                            message: format!(
                                "J2K Metal resident batch coefficient offset overflow at tile {}",
                                item.tile_index
                            ),
                        })?;
                    Ok(J2kLosslessCoefficientJob {
                        coefficient_offset,
                        component: block.component,
                        subband_x: block.subband_x,
                        subband_y: block.subband_y,
                        block_x: block.block_x,
                        block_y: block.block_y,
                        block_width: block.width,
                        block_height: block.height,
                        full_width: job.output_width,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let extract_command_buffer = if split_prepare_command_buffers {
                new_resident_encode_command_buffer(runtime, "signinum-j2k coefficient prep extract")
            } else {
                shared_command_buffer
                    .as_ref()
                    .expect("shared coefficient prep command buffer exists")
                    .clone()
            };
            let coefficient_job_buffer = dispatch_lossless_extract_coefficients(
                runtime,
                &extract_command_buffer,
                &active_planes,
                &coefficient_buffer,
                &coefficient_jobs,
                job.output_width,
            )
            .map_err(|err| Error::MetalKernel {
                message: format!(
                    "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                    item.tile_index
                ),
            })?;
            let prepare_command_buffer = extract_command_buffer.clone();
            if split_prepare_command_buffers {
                extract_command_buffer.commit();
                prepare_coefficient_extract_command_buffer = Some(extract_command_buffer);
            }

            prepared.push(J2kPreparedLosslessDeviceCodeBlocks {
                coefficient_buffer: coefficient_buffer.clone(),
                coefficient_byte_offset,
                coefficient_byte_len: item_sizes.coefficient_bytes,
                coefficient_buffer_is_batch_shared: true,
                code_blocks: item.code_blocks,
                recyclable_private_buffers,
                _prepare_command_buffer: prepare_command_buffer,
                _prepare_deinterleave_rct_command_buffer: prepare_deinterleave_rct_command_buffer,
                _prepare_dwt53_command_buffer: prepare_dwt53_command_buffer,
                _prepare_dwt53_vertical_command_buffers: prepare_dwt53_vertical_command_buffers,
                _prepare_dwt53_horizontal_command_buffers: prepare_dwt53_horizontal_command_buffers,
                _prepare_coefficient_extract_command_buffer:
                    prepare_coefficient_extract_command_buffer,
                _deinterleave_status_buffer: status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }

        if let Some(command_buffer) = shared_command_buffer {
            command_buffer.commit();
        }
        Ok(prepared)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_rct(
    plane0: &mut [f32],
    plane1: &mut [f32],
    plane2: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal forward RCT plane lengths must match".to_string(),
            });
        }

        let params = J2kForwardRctParams {
            _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal forward RCT plane length exceeds u32".to_string(),
            })?,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let plane0_buffer = borrow_mut_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = borrow_mut_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = borrow_mut_slice_buffer(&runtime.device, plane2);
        let status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.forward_rct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kForwardRctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .forward_rct
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kMctStatus>().read() };
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_inverse_mct(job: J2kInverseMctJob<'_>) -> Result<Vec<Buffer>, Error> {
    let J2kInverseMctJob {
        transform,
        plane0,
        plane1,
        plane2,
        addend0,
        addend1,
        addend2,
    } = job;
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(Vec::new());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal inverse MCT plane lengths must match".to_string(),
            });
        }

        let transform = match transform {
            J2kWaveletTransform::Reversible53 => 0,
            J2kWaveletTransform::Irreversible97 => 1,
        };
        let params = J2kInverseMctParams {
            _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal inverse MCT plane length exceeds u32".to_string(),
            })?,
            _transform: transform,
            _addend0: addend0,
            _addend1: addend1,
            _addend2: addend2,
        };
        let plane0_buffer = copied_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = copied_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = copied_slice_buffer(&runtime.device, plane2);
        let status = J2kMctStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kMctStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.inverse_mct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kInverseMctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .inverse_mct
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kMctStatus>().read() };
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        let plane0_host =
            unsafe { core::slice::from_raw_parts(plane0_buffer.contents().cast::<f32>(), len) };
        let plane1_host =
            unsafe { core::slice::from_raw_parts(plane1_buffer.contents().cast::<f32>(), len) };
        let plane2_host =
            unsafe { core::slice::from_raw_parts(plane2_buffer.contents().cast::<f32>(), len) };
        for (dst, sample) in plane0.iter_mut().zip(plane0_host.iter().copied()) {
            *dst = sample - addend0;
        }
        for (dst, sample) in plane1.iter_mut().zip(plane1_host.iter().copied()) {
            *dst = sample - addend1;
        }
        for (dst, sample) in plane2.iter_mut().zip(plane2_host.iter().copied()) {
            *dst = sample - addend2;
        }
        Ok(vec![plane0_buffer, plane1_buffer, plane2_buffer])
    })
}

#[cfg(target_os = "macos")]
fn dispatch_inverse_mct_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    len: usize,
    transform: J2kWaveletTransform,
    addends: [f32; 3],
) -> Result<DirectStatusCheck, Error> {
    if len == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color MCT cannot run on an empty plane".to_string(),
        });
    }

    let transform = match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    };
    let params = J2kInverseMctParams {
        _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect color MCT plane length exceeds u32".to_string(),
        })?,
        _transform: transform,
        _addend0: addends[0],
        _addend1: addends[1],
        _addend2: addends[2],
    };
    let status = J2kMctStatus::default();
    let status_buffer = runtime.device.new_buffer_with_data(
        (&raw const status).cast(),
        size_of::<J2kMctStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.inverse_mct);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kInverseMctParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(4, Some(&status_buffer), 0);
    let width = runtime
        .inverse_mct
        .thread_execution_width()
        .max(1)
        .min(len as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Mct(status_buffer))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_store_component_and_capture(
    job: J2kStoreComponentJob<'_>,
) -> Result<Buffer, Error> {
    let J2kStoreComponentJob {
        input,
        input_width,
        source_x,
        source_y,
        copy_width,
        copy_height,
        output,
        output_width,
        output_x,
        output_y,
        addend,
    } = job;
    with_runtime(|runtime| {
        if copy_width == 0 || copy_height == 0 {
            return Ok(wrap_f32_output_buffer(&runtime.device, output));
        }

        let required_input_height =
            source_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store source height overflow".to_string(),
                })?;
        let required_output_height =
            output_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store destination height overflow".to_string(),
                })?;
        if source_x
            .checked_add(copy_width)
            .is_none_or(|end| end > input_width)
            || output_x
                .checked_add(copy_width)
                .is_none_or(|end| end > output_width)
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store copy rectangle exceeds row bounds".to_string(),
            });
        }
        if input.len()
            < input_width as usize
                * usize::try_from(required_input_height).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal store source height exceeds usize".to_string(),
                })?
            || output.len()
                < output_width as usize
                    * usize::try_from(required_output_height).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal store destination height exceeds usize".to_string(),
                    })?
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store buffers are smaller than required".to_string(),
            });
        }

        let params = J2kStoreParams {
            input_width,
            source_x,
            source_y,
            copy_width,
            copy_height,
            output_width,
            output_x,
            output_y,
            addend,
        };
        let input_buffer = borrow_slice_buffer(&runtime.device, input);
        let output_buffer = wrap_f32_output_buffer(&runtime.device, output);
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.store_component);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kStoreParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(encoder, &runtime.store_component, (copy_width, copy_height));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(output_buffer)
    })
}

#[cfg(target_os = "macos")]
fn dispatch_store_component_buffer_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    output_offset_bytes: usize,
    params: J2kStoreParams,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid component store");
    dispatch_store_component_buffer_in_encoder_with_offsets(
        runtime,
        encoder,
        input,
        input_offset_bytes,
        output,
        output_offset_bytes,
        params,
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn dispatch_store_component_buffer_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    output_offset_bytes: usize,
    params: J2kStoreParams,
) {
    encoder.set_compute_pipeline_state(&runtime.store_component);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(output), output_offset_bytes as u64);
    encoder.set_bytes(
        2,
        size_of::<J2kStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.store_component,
        (params.copy_width, params.copy_height),
    );
}

fn dispatch_store_component_repeated_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    params: J2kRepeatedStoreParams,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated component store");
    encoder.set_compute_pipeline_state(&runtime.store_component_repeated);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        &runtime.store_component_repeated,
        (params.copy_width, params.copy_height, params.batch_count),
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn repeated_gray_store_is_contiguous_full_surface(params: J2kRepeatedGrayStoreParams) -> bool {
    params.source_x == 0
        && params.source_y == 0
        && params.output_x == 0
        && params.output_y == 0
        && params.copy_width == params.input_width
        && params.copy_height == params.input_height
        && params.copy_width == params.output_width
        && params.copy_height == params.output_height
}

#[cfg(target_os = "macos")]
fn encode_repeated_gray_store_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    params: J2kRepeatedGrayStoreParams,
    dims: (u32, u32),
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    let bytes_per_pixel = fmt.bytes_per_pixel();
    let pitch_bytes = dims.0 as usize * bytes_per_pixel;
    let surface_bytes =
        pitch_bytes
            .checked_mul(dims.1 as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal repeated grayscale fused store size overflow".to_string(),
            })?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal repeated grayscale fused store total size overflow".to_string(),
        })?;
    let out_buffer = runtime
        .device
        .new_buffer(total_bytes as u64, MTLResourceOptions::StorageModeShared);
    let contiguous_full_surface = repeated_gray_store_is_contiguous_full_surface(params);
    let pipeline = match (fmt, contiguous_full_surface) {
        (PixelFormat::Gray8, true) => &runtime.store_component_repeated_gray_u8_contiguous,
        (PixelFormat::Gray8, false) => &runtime.store_component_repeated_gray_u8,
        (PixelFormat::Gray16, true) => &runtime.store_component_repeated_gray_u16_contiguous,
        (PixelFormat::Gray16, false) => &runtime.store_component_repeated_gray_u16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal repeated grayscale fused store does not support {fmt:?}"
                ),
            })
        }
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    if contiguous_full_surface {
        let total_samples = u64::from(params.input_width)
            * u64::from(params.input_height)
            * u64::from(params.batch_count);
        encoder.dispatch_threads(
            MTLSize {
                width: total_samples,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: max_threads,
                height: 1,
                depth: 1,
            },
        );
    } else {
        dispatch_3d_pipeline(
            encoder,
            pipeline,
            (params.copy_width, params.copy_height, params.batch_count),
        );
    }
    encoder.end_encoding();

    let mut surfaces = Vec::with_capacity(count);
    for instance_idx in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            fmt,
            instance_idx * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
fn encode_gray_store_to_surface_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    input: &Buffer,
    input_offset_bytes: usize,
    params: J2kGrayStoreParams,
    dims: (u32, u32),
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.store_component_gray_u8,
        PixelFormat::Gray16 => &runtime.store_component_gray_u16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("J2K Metal grayscale fused store does not support {fmt:?}"),
            })
        }
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, (params.copy_width, params.copy_height));

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_reversible53_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            output_x: 0,
            output_y: 0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_x: 0,
            ll_y: 0,
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_x: 0,
            hl_y: 0,
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_x: 0,
            lh_y: 0,
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_x: 0,
            hh_y: 0,
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let command_buffer = runtime.queue.new_command_buffer();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(
            encoder,
            &runtime.idwt_interleave,
            (params.width, params.height),
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let horizontal_width = runtime
            .idwt_reversible53_horizontal
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.height),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: horizontal_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let vertical_width = runtime
            .idwt_reversible53_vertical
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: vertical_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid reversible53 IDWT");
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
        runtime,
        encoder,
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
        params,
        decoded,
        decoded_offset,
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) {
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.idwt_interleave,
        (params.width, params.height),
    );

    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffers: DirectIdwtCommandBuffers<'_>,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kRepeatedIdwtSingleDecompositionParams,
    decoded: &Buffer,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = command_buffers.interleave.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated IDWT interleave");
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave_batched);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        &runtime.idwt_interleave_batched,
        (params.width, params.height, params.batch_count),
    );
    encoder.end_encoding();

    let encoder = command_buffers.horizontal.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated IDWT horizontal");
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: u64::from(params.batch_count),
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffers.vertical.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated IDWT vertical");
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: u64::from(params.batch_count),
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_irreversible97_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            output_x: 0,
            output_y: 0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_x: 0,
            ll_y: 0,
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_x: 0,
            hl_y: 0,
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_x: 0,
            lh_y: 0,
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_x: 0,
            hh_y: 0,
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kIdwtStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(6, Some(&status_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kIdwtStatus>().read() };
        if status.code != J2K_IDWT_STATUS_OK {
            return Err(decode_idwt_status_error(status));
        }
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) -> DirectStatusCheck {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kIdwtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid irreversible97 IDWT");
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        runtime,
        encoder,
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
        params,
        decoded,
        decoded_offset,
        &status_buffer,
    );
    encoder.end_encoding();

    DirectStatusCheck::Idwt(status_buffer)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
) -> DirectStatusCheck {
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kIdwtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        runtime,
        encoder,
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
        params,
        decoded,
        decoded_offset,
        &status_buffer,
    );

    DirectStatusCheck::Idwt(status_buffer)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    ll: &Buffer,
    ll_offset: usize,
    hl: &Buffer,
    hl_offset: usize,
    lh: &Buffer,
    lh_offset: usize,
    hh: &Buffer,
    hh_offset: usize,
    params: J2kIdwtSingleDecompositionParams,
    decoded: &Buffer,
    decoded_offset: usize,
    status_buffer: &Buffer,
) {
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(6, Some(status_buffer), 0);
    dispatch_single_thread(encoder);
}

#[cfg(target_os = "macos")]
fn classic_batch_uses_plain_fast_path(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        if job.style_flags != 0
            || job.width > J2K_CLASSIC_MAX_WIDTH
            || job.height > J2K_CLASSIC_MAX_HEIGHT
        {
            return false;
        }
        let start = job.segment_offset as usize;
        let Some(end) = start.checked_add(job.segment_count as usize) else {
            return false;
        };
        segments.get(start..end).is_some_and(|job_segments| {
            job_segments
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
        })
    })
}

#[cfg(target_os = "macos")]
fn classic_repeated_uses_plain_fast_path(
    count: usize,
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    let _ = (count, jobs, segments);
    // Batch-16 WSI benches are faster with device-state cleanup plus the separate parallel store.
    false
}

#[cfg(target_os = "macos")]
fn classic_batch_is_plain_arithmetic(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        job.style_flags == 0
            && segments[job.segment_offset as usize
                ..job.segment_offset as usize + job.segment_count as usize]
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
    })
}

#[cfg(target_os = "macos")]
fn dispatch_classic_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, jobs);
    let segments_buffer = borrow_slice_buffer(&runtime.device, segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(jobs, segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    let status_buffer = runtime.device.new_buffer(
        (jobs.len().max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&segments_buffer), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(&coefficients_scratch.buffer), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: jobs.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(jobs.len() as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: jobs.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let statuses = unsafe {
        core::slice::from_raw_parts(
            status_buffer.contents().cast::<J2kClassicStatus>(),
            jobs.len(),
        )
    };
    let status = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_CLASSIC_STATUS_OK);
    runtime.recycle_private_buffer(coefficients_scratch.bytes, coefficients_scratch.buffer);
    if let Some(status) = status {
        return Err(decode_classic_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> (DirectStatusCheck, Option<Buffer>) {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_classic_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
        &status_buffer,
    );
    encoder.end_encoding();

    (
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: job_count,
        },
        None,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_batched_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> (DirectStatusCheck, Option<Buffer>) {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    dispatch_classic_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
        &status_buffer,
    );

    (
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: job_count,
        },
        None,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_batched_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
    status_buffer: &Buffer,
) {
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(job_count as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: job_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    use_plain_fast_path: bool,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) -> DirectStatusCheck {
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_repeated_batched
    } else {
        &runtime.classic_cleanup_repeated_batched
    };
    let status_buffer = runtime.device.new_buffer(
        (total_job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let repeated = J2kClassicRepeatedBatchParams {
        job_count: u32::try_from(job_count).expect("classic repeated base job count fits in u32"),
        output_plane_len: u32::try_from(output_plane_len)
            .expect("classic repeated output plane len fits in u32"),
        batch_count: u32::try_from(total_job_count / job_count.max(1))
            .expect("classic repeated batch count fits in u32"),
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        6,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(job_count as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();

    DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    segments: &Buffer,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
    states_scratch: &Buffer,
) -> DirectStatusCheck {
    let status_buffer = runtime.device.new_buffer(
        (total_job_count.max(1) * size_of::<J2kClassicStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let repeated = J2kClassicRepeatedBatchParams {
        job_count: u32::try_from(job_count).expect("classic repeated base job count fits in u32"),
        output_plane_len: u32::try_from(output_plane_len)
            .expect("classic repeated output plane len fits in u32"),
        batch_count: u32::try_from(total_job_count / job_count.max(1))
            .expect("classic repeated batch count fits in u32"),
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_cleanup_plain_dev_repeated_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_buffer(6, Some(states_scratch), 0);
    encoder.set_bytes(
        7,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    let width = runtime
        .classic_cleanup_plain_dev_repeated_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_classic_store_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    jobs: &Buffer,
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    decoded: &Buffer,
    coefficients_scratch: &Buffer,
) {
    let repeated = J2kClassicRepeatedBatchParams {
        job_count: u32::try_from(job_count).expect("classic repeated base job count fits in u32"),
        output_plane_len: u32::try_from(output_plane_len)
            .expect("classic repeated output plane len fits in u32"),
        batch_count: u32::try_from(total_job_count / job_count.max(1))
            .expect("classic repeated batch count fits in u32"),
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_store_repeated_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_buffer(1, Some(jobs), 0);
    encoder.set_buffer(2, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedClassicSubBand],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        sub_bands.iter().map(|sub_band| DistinctClassicBatch {
            coded_data: &sub_band.coded_data,
            jobs: &sub_band.jobs,
            segments: &sub_band.segments,
            output_base: sub_bands
                .iter()
                .position(|candidate| core::ptr::eq(*candidate, *sub_band))
                .expect("sub-band exists")
                * per_instance_len,
        }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
fn encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedClassicSubBandGroup],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctClassicBatch {
                coded_data: &group.coded_data,
                jobs: &group.jobs,
                segments: &group.segments,
                output_base: index * per_instance_len,
            }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
struct DistinctClassicBatch<'a> {
    coded_data: &'a [u8],
    jobs: &'a [J2kClassicCleanupBatchJob],
    segments: &'a [J2kClassicSegment],
    output_base: usize,
}

#[cfg(target_os = "macos")]
fn encode_distinct_classic_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl IntoIterator<Item = DistinctClassicBatch<'a>>,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let mut coded_data = Vec::new();
    let mut jobs = Vec::new();
    let mut segments = Vec::new();

    for batch in batches {
        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color coded payload exceeds u32".to_string(),
        })?;
        let segment_base = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color segment table exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(batch.coded_data);
        for segment in batch.segments {
            let mut adjusted = *segment;
            adjusted.data_offset =
                adjusted
                    .data_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect distinct color segment offset overflow"
                            .to_string(),
                    })?;
            segments.push(adjusted);
        }
        let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color output offset exceeds u32".to_string(),
        })?;
        for job in batch.jobs {
            let mut adjusted = *job;
            adjusted.coded_offset =
                adjusted
                    .coded_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect distinct color job coded offset overflow"
                            .to_string(),
                    })?;
            adjusted.segment_offset = adjusted
                .segment_offset
                .checked_add(segment_base)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect distinct color job segment offset overflow"
                        .to_string(),
                })?;
            adjusted.output_offset =
                adjusted
                    .output_offset
                    .checked_add(output_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message:
                            "classic J2K MetalDirect distinct color job output offset overflow"
                                .to_string(),
                    })?;
            jobs.push(adjusted);
        }
    }

    if jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = owned_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = owned_slice_buffer(&runtime.device, &jobs);
    let segments_buffer = owned_slice_buffer(&runtime.device, &segments);
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&jobs, &segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedHtSubBand],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_ht_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        sub_bands
            .iter()
            .enumerate()
            .map(|(index, sub_band)| DistinctHtBatch {
                coded_data: &sub_band.coded_data,
                jobs: &sub_band.jobs,
                output_base: index * per_instance_len,
            }),
        output,
    )
}

#[cfg(target_os = "macos")]
fn encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedHtSubBandGroup],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_ht_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctHtBatch {
                coded_data: &group.coded_arena.data,
                jobs: &group.jobs,
                output_base: index * per_instance_len,
            }),
        output,
    )
}

#[cfg(target_os = "macos")]
struct DistinctHtBatch<'a> {
    coded_data: &'a [u8],
    jobs: &'a [J2kHtCleanupBatchJob],
    output_base: usize,
}

#[cfg(target_os = "macos")]
fn encode_distinct_ht_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl IntoIterator<Item = DistinctHtBatch<'a>>,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let mut coded_data = Vec::new();
    let mut jobs = Vec::new();

    for batch in batches {
        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect distinct grayscale coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(batch.coded_data);
        let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect distinct grayscale output offset exceeds u32".to_string(),
        })?;
        for job in batch.jobs {
            let mut adjusted = *job;
            adjusted.coded_offset =
                adjusted
                    .coded_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect distinct grayscale job coded offset overflow"
                            .to_string(),
                    })?;
            adjusted.output_offset =
                adjusted
                    .output_offset
                    .checked_add(output_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect distinct grayscale job output offset overflow"
                            .to_string(),
                    })?;
            jobs.push(adjusted);
        }
    }

    if jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = owned_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = owned_slice_buffer(&runtime.device, &jobs);
    let status_check = dispatch_ht_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        output,
        ht_batch_output_word_count(&jobs)?,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedClassicSubBand,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = job
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect repeated job count overflow".to_string(),
        })?;
    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let segments_buffer = job.segments_buffer.clone();
    let use_plain_fast_path =
        classic_repeated_uses_plain_fast_path(count, &job.jobs, &job.segments)
            && runtime
                .classic_cleanup_plain_repeated_batched
                .max_total_threads_per_threadgroup()
                >= 32;
    let use_plain_dev_path = !use_plain_fast_path
        && count <= 16
        && classic_batch_is_plain_arithmetic(&job.jobs, &job.segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, total_jobs)?;
    let states_scratch = if use_plain_dev_path {
        Some(take_classic_states_scratch_buffer(runtime, total_jobs)?)
    } else {
        None
    };
    let status_check = if use_plain_fast_path {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            true,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )
    } else if let Some(states_scratch) = states_scratch.as_ref() {
        dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
            &states_scratch.buffer,
        )
    } else {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            use_plain_fast_path,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )
    };
    if !use_plain_fast_path {
        dispatch_classic_store_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &jobs_buffer,
            job.jobs.len(),
            total_jobs,
            job.width as usize * job.height as usize,
            output,
            &coefficients_scratch.buffer,
        );
    }
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        scratch_buffers.push(states_scratch);
    }
    let retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedClassicSubBandGroup,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = group
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect repeated grouped job count overflow".to_string(),
        })?;
    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let segments_buffer = group.segments_buffer.clone();
    let use_plain_fast_path =
        classic_repeated_uses_plain_fast_path(count, &group.jobs, &group.segments)
            && runtime
                .classic_cleanup_plain_repeated_batched
                .max_total_threads_per_threadgroup()
                >= 32;
    let use_plain_dev_path = !use_plain_fast_path
        && count <= 16
        && classic_batch_is_plain_arithmetic(&group.jobs, &group.segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, total_jobs)?;
    let states_scratch = if use_plain_dev_path {
        Some(take_classic_states_scratch_buffer(runtime, total_jobs)?)
    } else {
        None
    };
    let status_check = if use_plain_fast_path {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            true,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )
    } else if let Some(states_scratch) = states_scratch.as_ref() {
        dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
            &states_scratch.buffer,
        )
    } else {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &coded_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            use_plain_fast_path,
            &segments_buffer,
            output,
            &coefficients_scratch.buffer,
        )
    };
    if !use_plain_fast_path {
        dispatch_classic_store_repeated_batched_in_command_buffer(
            runtime,
            command_buffer,
            &jobs_buffer,
            group.jobs.len(),
            total_jobs,
            group.total_coefficients,
            output,
            &coefficients_scratch.buffer,
        );
    }
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        scratch_buffers.push(states_scratch);
    }
    let retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_prepared_classic_sub_band_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    job: &PreparedClassicSubBand,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(
            runtime,
            encoder,
            output,
            job.width as usize * job.height as usize,
        )?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let segments_buffer = job.segments_buffer.clone();
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&job.jobs, &job.segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, job.jobs.len())?;
    if job.zero_fill {
        dispatch_zero_u32_buffer_in_encoder(
            runtime,
            encoder,
            output,
            job.width as usize * job.height as usize,
        )?;
    }
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedClassicSubBandGroup,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if group.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let segments_buffer = group.segments_buffer.clone();
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&group.jobs, &group.segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, group.jobs.len())?;
    if group.zero_fill {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
    }
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        use_plain_fast_path,
        &segments_buffer,
        output,
        &coefficients_scratch.buffer,
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
fn required_ht_output_len(job: HtCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedHtSubBand,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = job
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated job count overflow".to_string(),
        })?;
    let coded_buffer = prepared_ht_buffer(job.coded_buffer.as_ref(), "coded")?.clone();
    let jobs_buffer = prepared_ht_buffer(job.jobs_buffer.as_ref(), "jobs")?.clone();
    let status_check = dispatch_ht_cleanup_repeated_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        total_jobs,
        job.width as usize * job.height as usize,
        output,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedHtSubBandGroup,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = group
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated grouped job count overflow".to_string(),
        })?;
    let coded_buffer = group.coded_arena.buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let status_check = dispatch_ht_cleanup_repeated_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        total_jobs,
        group.total_coefficients,
        output,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn encode_prepared_ht_sub_band_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    job: &PreparedHtSubBand,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(
            runtime,
            encoder,
            output,
            job.width as usize * job.height as usize,
        )?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = prepared_ht_buffer(job.coded_buffer.as_ref(), "coded")?.clone();
    let jobs_buffer = prepared_ht_buffer(job.jobs_buffer.as_ref(), "jobs")?.clone();
    let status_check = dispatch_ht_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        output,
        job.width as usize * job.height as usize,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedHtSubBandGroup,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if group.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = group.coded_arena.buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let status_check = dispatch_ht_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        output,
        group.total_coefficients,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
fn decode_ht_status_error(status: J2kHtStatus) -> Error {
    let kind = match status.code {
        J2K_HT_STATUS_FAIL => "decode failure",
        J2K_HT_STATUS_UNSUPPORTED => "unsupported HT kernel input",
        _ => "unexpected HT kernel status",
    };
    Error::MetalKernel {
        message: format!("HTJ2K Metal kernel {kind} (detail={})", status.detail),
    }
}

#[cfg(target_os = "macos")]
fn ht_output_word_count(
    output_offset: u32,
    output_stride: u32,
    width: u32,
    height: u32,
) -> Result<usize, Error> {
    let end = if width == 0 || height == 0 {
        u64::from(output_offset)
    } else {
        u64::from(output_offset)
            .checked_add(u64::from(height - 1) * u64::from(output_stride))
            .and_then(|offset| offset.checked_add(u64::from(width)))
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal output span overflow".to_string(),
            })?
    };
    usize::try_from(end).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal output span exceeds usize".to_string(),
    })
}

#[cfg(target_os = "macos")]
fn ht_batch_output_word_count(jobs: &[J2kHtCleanupBatchJob]) -> Result<usize, Error> {
    let mut word_count = 0usize;
    for job in jobs {
        let job_word_count =
            ht_output_word_count(job.output_offset, job.output_stride, job.width, job.height)?;
        word_count = word_count.max(job_word_count);
    }
    Ok(word_count)
}

#[cfg(target_os = "macos")]
fn dispatch_zero_u32_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    buffer: &Buffer,
    word_count: usize,
) -> Result<(), Error> {
    let word_count = u32::try_from(word_count).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal zero-fill word count exceeds u32".to_string(),
    })?;
    if word_count == 0 {
        return Ok(());
    }

    encoder.set_compute_pipeline_state(&runtime.zero_u32_buffer);
    encoder.set_buffer(0, Some(buffer), 0);
    encoder.set_bytes(1, size_of::<u32>() as u64, (&raw const word_count).cast());
    dispatch_1d_pipeline(encoder, &runtime.zero_u32_buffer, u64::from(word_count));
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_status_error(stage: &str, code: u32, detail: u32) -> Error {
    let kind = match code {
        J2K_ENCODE_STATUS_FAIL => "failure",
        J2K_ENCODE_STATUS_UNSUPPORTED => "unsupported input",
        _ => "unexpected status",
    };
    Error::MetalKernel {
        message: format!("{stage} Metal encode kernel {kind} (detail={detail})"),
    }
}

#[cfg(target_os = "macos")]
fn packet_encode_status_error(status: J2kPacketEncodeStatus) -> Error {
    if status.code == J2K_ENCODE_STATUS_FAIL && status.detail == 7 {
        return Error::MetalKernel {
            message: format!(
                "packetization Metal encode kernel failure (detail=7, tier1_detail={})",
                status.data_len
            ),
        };
    }
    encode_status_error("packetization", status.code, status.detail)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum J2kClassicEncodeOutputCapacityMode {
    Conservative,
    Tight,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum J2kHtPacketOutputCapacityMode {
    Conservative,
    Tight,
}

#[cfg(target_os = "macos")]
pub(crate) fn ht_packet_output_capacity_mode_from_env() -> J2kHtPacketOutputCapacityMode {
    match std::env::var(HT_PACKET_CAPACITY_ENV) {
        Ok(value) if value.eq_ignore_ascii_case("conservative") => {
            J2kHtPacketOutputCapacityMode::Conservative
        }
        _ => J2kHtPacketOutputCapacityMode::Tight,
    }
}

#[cfg(target_os = "macos")]
fn classic_encode_output_capacity_for_mode(
    width: u32,
    height: u32,
    total_bitplanes: u8,
    mode: J2kClassicEncodeOutputCapacityMode,
) -> Result<usize, Error> {
    let samples = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal encode block size overflow".to_string(),
        })?;
    let bitplane_bytes = samples
        .checked_mul(usize::from(total_bitplanes).max(1))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal encode output capacity overflow".to_string(),
        })?;
    let payload_bytes = match mode {
        J2kClassicEncodeOutputCapacityMode::Conservative => bitplane_bytes.checked_mul(8),
        J2kClassicEncodeOutputCapacityMode::Tight => Some(bitplane_bytes),
    }
    .ok_or_else(|| Error::MetalKernel {
        message: "classic J2K Metal encode output capacity overflow".to_string(),
    })?;
    payload_bytes
        .checked_add(4096)
        .map(|bytes| bytes.max(4096) + 1)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal encode output capacity overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn classic_encode_output_capacity(
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> Result<usize, Error> {
    classic_encode_output_capacity_for_mode(
        width,
        height,
        total_bitplanes,
        J2kClassicEncodeOutputCapacityMode::Conservative,
    )
}

#[cfg(target_os = "macos")]
fn classic_encode_total_coding_passes(total_bitplanes: u8) -> usize {
    if total_bitplanes == 0 {
        0
    } else {
        1 + 3 * (usize::from(total_bitplanes) - 1)
    }
}

#[cfg(target_os = "macos")]
fn classic_bypass_segment_index(pass_idx: usize) -> usize {
    if pass_idx < 10 {
        0
    } else {
        1 + 2 * ((pass_idx - 10) / 3) + usize::from(((pass_idx - 10) % 3) == 2)
    }
}

#[cfg(target_os = "macos")]
fn classic_encode_segment_capacity(style_flags: u32, total_bitplanes: u8) -> usize {
    let total_passes = classic_encode_total_coding_passes(total_bitplanes);
    if total_passes == 0 {
        return 1;
    }
    if (style_flags & J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS) != 0 {
        total_passes
    } else if (style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0 {
        classic_bypass_segment_index(total_passes - 1) + 1
    } else {
        1
    }
}

#[cfg(target_os = "macos")]
fn ht_scaled_scratch_size(max_size: usize, sample_count: usize) -> Result<usize, Error> {
    if sample_count > J2K_HT_ENCODE_MAX_SAMPLES {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal encode code-block exceeds maximum sample count".to_string(),
        });
    }

    max_size
        .checked_mul(sample_count)
        .map(|bytes| bytes.div_ceil(J2K_HT_ENCODE_MAX_SAMPLES).max(1))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal encode output capacity overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn ht_encode_output_capacity(width: u32, height: u32) -> Result<usize, Error> {
    let sample_count = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal encode sample count overflow".to_string(),
        })?;
    let scaled_ms_size = ht_scaled_scratch_size(J2K_HT_ENCODE_MS_SIZE, sample_count)?;
    let ms_floor = sample_count
        .checked_mul(J2K_HT_ENCODE_MS_BYTES_PER_SAMPLE_FLOOR)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal encode output capacity overflow".to_string(),
        })?;
    let ms_size = scaled_ms_size.max(ms_floor).min(J2K_HT_ENCODE_MS_SIZE);
    let mel_size = J2K_HT_ENCODE_MEL_SIZE;
    let vlc_size = J2K_HT_ENCODE_VLC_SIZE;
    ms_size
        .checked_add(mel_size)
        .and_then(|bytes| bytes.checked_add(vlc_size))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal encode output capacity overflow".to_string(),
        })
}

fn classic_encode_sub_band_code(sub_band_type: signinum_j2k_native::J2kSubBandType) -> u32 {
    match sub_band_type {
        signinum_j2k_native::J2kSubBandType::LowLow => 0,
        signinum_j2k_native::J2kSubBandType::HighLow => 1,
        signinum_j2k_native::J2kSubBandType::LowHigh => 2,
        signinum_j2k_native::J2kSubBandType::HighHigh => 3,
    }
}

#[cfg(target_os = "macos")]
fn read_classic_encoded_code_block(
    status: J2kClassicEncodeStatus,
    output: &Buffer,
    output_offset: usize,
    output_capacity: usize,
    segment_buffer: &Buffer,
    segment_offset: usize,
    segment_capacity: usize,
) -> Result<EncodedJ2kCodeBlock, Error> {
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "classic Tier-1",
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode length exceeds usize".to_string(),
    })?;
    let payload_skip = usize::try_from(status.reserved0).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode payload skip exceeds usize".to_string(),
    })?;
    let number_of_coding_passes =
        u8::try_from(status.number_of_coding_passes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode pass count exceeds u8".to_string(),
        })?;
    let missing_bit_planes =
        u8::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode missing bitplanes exceeds u8".to_string(),
        })?;
    let segment_count = usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
        message: "classic J2K Metal encode segment count exceeds usize".to_string(),
    })?;
    if segment_count > segment_capacity {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal encode segment count exceeds buffer".to_string(),
        });
    }
    let raw_segments = if segment_count == 0 {
        &[][..]
    } else {
        unsafe {
            core::slice::from_raw_parts(
                segment_buffer
                    .contents()
                    .cast::<J2kClassicSegment>()
                    .add(segment_offset),
                segment_count,
            )
        }
    };
    let data = if data_len == 0 {
        Vec::new()
    } else {
        let payload_span =
            data_len
                .checked_add(payload_skip)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode payload span overflow".to_string(),
                })?;
        if payload_span > output_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let payload_offset =
            output_offset
                .checked_add(payload_skip)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode payload offset overflow".to_string(),
                })?;
        unsafe {
            core::slice::from_raw_parts(
                output.contents().cast::<u8>().add(payload_offset),
                data_len,
            )
        }
        .to_vec()
    };
    let segments = raw_segments
        .iter()
        .map(|segment| {
            Ok(J2kCodeBlockSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: u8::try_from(segment.start_coding_pass).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment start pass exceeds u8"
                            .to_string(),
                    }
                })?,
                end_coding_pass: u8::try_from(segment.end_coding_pass).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment end pass exceeds u8".to_string(),
                    }
                })?,
                use_arithmetic: segment.use_arithmetic != 0,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(EncodedJ2kCodeBlock {
        data,
        segments,
        number_of_coding_passes,
        missing_bit_planes,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_classic_tier1_code_blocks(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient slice is too small".to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode coefficient table exceeds u32".to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment table exceeds u32".to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode output capacity exceeds u32".to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal encode segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode segment buffer overflow".to_string(),
                })?;
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            (jobs.len() * size_of::<J2kClassicEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let classic_encode_pipeline = classic_encode_code_blocks_pipeline(runtime, &batch_jobs);
        encoder.set_compute_pipeline_state(classic_encode_pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(encoder, classic_encode_pipeline, u64::from(job_count));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kClassicEncodeStatus>(),
                jobs.len(),
            )
        };
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode output capacity exceeds usize".to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal token-pack coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal token-pack coefficient slice is too small"
                        .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment table exceeds u32".to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal token-pack output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal token-pack segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment buffer overflow".to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message:
                    "classic J2K Metal token-pack parity helper supports only bypass_u16_32 jobs"
                        .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            (jobs.len() * size_of::<J2kClassicEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal token-pack job count exceeds u32".to_string(),
        })?;
        let command_buffer = runtime.queue.new_command_buffer();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
        )?;
        debug_assert_eq!(token_buffers.job_count, job_count);
        dispatch_classic_tier1_token_pack_from_gpu_tokens(
            runtime,
            command_buffer,
            &job_buffer,
            &token_buffers,
            &output,
            &status_buffer,
            &segment_buffer,
        );
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kClassicEncodeStatus>(),
                jobs.len(),
            )
        };
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output offset exceeds usize".to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack output capacity exceeds usize"
                        .to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment offset exceeds usize"
                        .to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal token-pack segment capacity exceeds usize"
                        .to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test_with_emit_route(
        jobs, false,
    )
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_split_mq_byte_raw_tokens_gpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test_with_emit_route(
        jobs, true,
    )
}

#[cfg(all(test, target_os = "macos"))]
fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test_with_emit_route(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    use_mq_byte_emit: bool,
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack coefficient count overflow"
                        .to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message:
                        "classic J2K Metal split GPU token-pack coefficient slice is too small"
                            .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output table exceeds u32"
                        .to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack segment table exceeds u32"
                        .to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "classic J2K Metal split GPU token-pack output capacity exceeds u32"
                                .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "classic J2K Metal split GPU token-pack segment capacity exceeds u32"
                                .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output buffer overflow"
                        .to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack segment buffer overflow"
                        .to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message:
                    "classic J2K Metal split GPU token-pack helper supports only bypass_u16_32 jobs"
                        .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            (jobs.len() * size_of::<J2kClassicEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = runtime.queue.new_command_buffer();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let split_buffers = dispatch_classic_tier1_split_token_emit_for_gpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
            use_mq_byte_emit,
        )?;
        dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
            runtime,
            command_buffer,
            &job_buffer,
            &split_buffers,
            &output,
            &status_buffer,
            &segment_buffer,
        );
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kClassicEncodeStatus>(),
                jobs.len(),
            )
        };
        let mut results = Vec::with_capacity(jobs.len());
        for (idx, status) in statuses.iter().copied().enumerate() {
            let batch_job = batch_jobs[idx];
            results.push(read_classic_encoded_code_block(
                status,
                &output,
                usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output offset exceeds usize"
                        .to_string(),
                })?,
                usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack output capacity exceeds usize"
                        .to_string(),
                })?,
                &segment_buffer,
                usize::try_from(batch_job.segment_offset).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split GPU token-pack segment offset exceeds usize"
                        .to_string(),
                })?,
                usize::try_from(batch_job.segment_capacity).map_err(|_| Error::MetalKernel {
                    message:
                        "classic J2K Metal split GPU token-pack segment capacity exceeds usize"
                            .to_string(),
                })?,
            )?);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coefficient count overflow"
                        .to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coefficient slice is too small"
                        .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment table exceeds u32"
                        .to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal ordered-token output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal ordered-token segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment buffer overflow".to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal ordered-token helper supports only bypass_u16_32 jobs"
                    .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let command_buffer = runtime.queue.new_command_buffer();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
            &mut recyclable_private_buffers,
        )?;
        let job_count =
            usize::try_from(token_buffers.job_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal ordered-token job count exceeds usize".to_string(),
            })?;
        let token_stride_bytes =
            usize::try_from(token_buffers.token_stride_bytes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal ordered-token byte stride exceeds usize".to_string(),
            })?;
        let token_segment_stride =
            usize::try_from(token_buffers.token_segment_stride).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment stride exceeds usize"
                        .to_string(),
                }
            })?;
        let counter_byte_len = job_count
            .checked_mul(size_of::<J2kClassicTier1SymbolPlanCounters>())
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal ordered-token counter readback overflow".to_string(),
            })?;
        let token_byte_len =
            job_count
                .checked_mul(token_stride_bytes)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token byte readback overflow".to_string(),
                })?;
        let token_segment_byte_len = job_count
            .checked_mul(token_segment_stride)
            .and_then(|count| count.checked_mul(size_of::<J2kClassicTier1TokenSegment>()))
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal ordered-token segment readback overflow".to_string(),
            })?;
        let counter_readback = runtime.device.new_buffer(
            counter_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let token_readback = runtime.device.new_buffer(
            token_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let token_segment_readback = runtime.device.new_buffer(
            token_segment_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &token_buffers.counter_buffer,
            0,
            &counter_readback,
            0,
            counter_byte_len as u64,
        );
        blit.copy_from_buffer(
            &token_buffers.token_buffer,
            0,
            &token_readback,
            0,
            token_byte_len as u64,
        );
        blit.copy_from_buffer(
            &token_buffers.segment_buffer,
            0,
            &token_segment_readback,
            0,
            token_segment_byte_len as u64,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let counters = unsafe {
            core::slice::from_raw_parts(
                counter_readback
                    .contents()
                    .cast::<J2kClassicTier1SymbolPlanCounters>(),
                job_count,
            )
        };
        let token_bytes = unsafe {
            core::slice::from_raw_parts(token_readback.contents().cast::<u8>(), token_byte_len)
        };
        let token_segments = unsafe {
            core::slice::from_raw_parts(
                token_segment_readback
                    .contents()
                    .cast::<J2kClassicTier1TokenSegment>(),
                job_count.saturating_mul(token_segment_stride),
            )
        };

        let mut results = Vec::with_capacity(job_count);
        for (block_idx, counter) in counters.iter().enumerate() {
            if counter.code != J2K_ENCODE_STATUS_OK {
                return Err(encode_status_error(
                    "classic Tier-1 ordered-token emit",
                    counter.code,
                    counter.detail,
                ));
            }
            let segment_count =
                usize::try_from(counter.segment_count).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment count exceeds usize"
                        .to_string(),
                })?;
            if segment_count > token_segment_stride {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal ordered-token segment count exceeds capacity"
                        .to_string(),
                });
            }
            let token_start =
                block_idx
                    .checked_mul(token_stride_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token byte offset overflow".to_string(),
                    })?;
            let segment_start =
                block_idx
                    .checked_mul(token_segment_stride)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token segment offset overflow"
                            .to_string(),
                    })?;
            let mut native_segments = Vec::with_capacity(segment_count);
            for segment in &token_segments[segment_start..segment_start + segment_count] {
                let start_coding_pass =
                    u8::try_from(segment.pass_range & 0xFFFF).map_err(|_| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token start pass exceeds u8"
                            .to_string(),
                    })?;
                let end_coding_pass =
                    u8::try_from(segment.pass_range >> 16).map_err(|_| Error::MetalKernel {
                        message: "classic J2K Metal ordered-token end pass exceeds u8".to_string(),
                    })?;
                native_segments.push(J2kTier1TokenSegment {
                    token_bit_offset: segment.token_bit_offset,
                    token_bit_count: segment.token_bit_count,
                    start_coding_pass,
                    end_coding_pass,
                    use_arithmetic: (segment.flags & 1) != 0,
                });
            }
            let packed = pack_j2k_code_block_scalar_from_tier1_tokens(
                &token_bytes[token_start..token_start + token_stride_bytes],
                &native_segments,
                u8::try_from(counter.coding_passes).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token coding-pass count exceeds u8"
                        .to_string(),
                })?,
                u8::try_from(counter.missing_bit_planes).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal ordered-token missing bitplanes exceed u8"
                        .to_string(),
                })?,
            )
            .map_err(|message| Error::MetalKernel {
                message: format!("classic J2K Metal ordered-token CPU pack failed: {message}"),
            })?;
            results.push(packed);
        }

        Ok(results)
    })
}

#[cfg(all(test, target_os = "macos"))]
#[derive(Default)]
struct ClassicTier1MsbBitWriter {
    bytes: Vec<u8>,
    current_byte: u8,
    bits_in_current: u8,
    bit_count: usize,
}

#[cfg(all(test, target_os = "macos"))]
impl ClassicTier1MsbBitWriter {
    fn write_bit(&mut self, bit: u8) {
        self.current_byte = (self.current_byte << 1) | (bit & 1);
        self.bits_in_current += 1;
        self.bit_count += 1;
        if self.bits_in_current == 8 {
            self.bytes.push(self.current_byte);
            self.current_byte = 0;
            self.bits_in_current = 0;
        }
    }

    fn bit_count_u32(&self) -> Result<u32, Error> {
        u32::try_from(self.bit_count).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal split-token combined bit offset exceeds u32".to_string(),
        })
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_in_current != 0 {
            self.bytes
                .push(self.current_byte << (8 - self.bits_in_current));
        }
        self.bytes
    }
}

#[cfg(all(test, target_os = "macos"))]
fn classic_tier1_split_token_bit(source: &[u8], bit_offset: usize) -> Result<u8, Error> {
    if bit_offset >= source.len().saturating_mul(8) {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal split-token bit offset exceeds stream".to_string(),
        });
    }
    let byte = source[bit_offset / 8];
    let shift = 7 - (bit_offset % 8);
    Ok((byte >> shift) & 1)
}

#[cfg(all(test, target_os = "macos"))]
fn classic_tier1_append_split_token_bits(
    writer: &mut ClassicTier1MsbBitWriter,
    source: &[u8],
    bit_offset: usize,
    bit_count: usize,
) -> Result<(), Error> {
    let end = bit_offset
        .checked_add(bit_count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal split-token bit range overflow".to_string(),
        })?;
    if end > source.len().saturating_mul(8) {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal split-token bit range exceeds stream".to_string(),
        });
    }
    for bit_idx in 0..bit_count {
        writer.write_bit(classic_tier1_split_token_bit(source, bit_offset + bit_idx)?);
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
fn pack_classic_split_mq_raw_tokens_for_test(
    mq_token_bytes: &[u8],
    raw_token_bytes: &[u8],
    split_segments: &[J2kClassicTier1TokenSegment],
    counter: J2kClassicTier1SymbolPlanCounters,
) -> Result<EncodedJ2kCodeBlock, Error> {
    if counter.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "classic Tier-1 split-token emit",
            counter.code,
            counter.detail,
        ));
    }

    let mut combined = ClassicTier1MsbBitWriter::default();
    let mut native_segments = Vec::with_capacity(split_segments.len());
    for segment in split_segments {
        if (segment.flags & !1) != 0 {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal split-token segment has unsupported flags".to_string(),
            });
        }
        let use_arithmetic = (segment.flags & 1) != 0;
        let source = if use_arithmetic {
            mq_token_bytes
        } else {
            raw_token_bytes
        };
        let source_bit_offset =
            usize::try_from(segment.token_bit_offset).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token bit offset exceeds usize".to_string(),
            })?;
        let source_bit_count =
            usize::try_from(segment.token_bit_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token bit count exceeds usize".to_string(),
            })?;
        let combined_bit_offset = combined.bit_count_u32()?;
        classic_tier1_append_split_token_bits(
            &mut combined,
            source,
            source_bit_offset,
            source_bit_count,
        )?;
        let start_coding_pass =
            u8::try_from(segment.pass_range & 0xFFFF).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token start pass exceeds u8".to_string(),
            })?;
        let end_coding_pass =
            u8::try_from(segment.pass_range >> 16).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token end pass exceeds u8".to_string(),
            })?;
        native_segments.push(J2kTier1TokenSegment {
            token_bit_offset: combined_bit_offset,
            token_bit_count: segment.token_bit_count,
            start_coding_pass,
            end_coding_pass,
            use_arithmetic,
        });
    }

    pack_j2k_code_block_scalar_from_tier1_tokens(
        &combined.finish(),
        &native_segments,
        u8::try_from(counter.coding_passes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal split-token coding-pass count exceeds u8".to_string(),
        })?,
        u8::try_from(counter.missing_bit_planes).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal split-token missing bitplanes exceed u8".to_string(),
        })?,
    )
    .map_err(|message| Error::MetalKernel {
        message: format!("classic J2K Metal split-token CPU pack failed: {message}"),
    })
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) fn encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedJ2kCodeBlock>, Error> {
    with_runtime(|runtime| {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let mut coefficients = Vec::<i32>::new();
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(jobs.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for job in jobs {
            let expected_coefficients = usize::try_from(job.width)
                .ok()
                .and_then(|w| {
                    usize::try_from(job.height)
                        .ok()
                        .and_then(|h| w.checked_mul(h))
                })
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token coefficient count overflow".to_string(),
                })?;
            if job.coefficients.len() < expected_coefficients {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal split-token coefficient slice is too small"
                        .to_string(),
                });
            }
            let coefficient_offset =
                u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token coefficient table exceeds u32"
                        .to_string(),
                })?;
            coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
            let output_capacity =
                classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token output table exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token segment table exceeds u32".to_string(),
                })?;
            let style_flags = classic_style_flags(job.style);
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, job.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: job.width,
                height: job.height,
                sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
                total_bitplanes: u32::from(job.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal split-token output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal split-token segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token segment buffer overflow".to_string(),
                })?;
        }

        if !classic_tier1_gpu_token_pack_supported(&batch_jobs) {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal split-token helper supports only bypass_u16_32 jobs"
                    .to_string(),
            });
        }

        let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let command_buffer = runtime.queue.new_command_buffer();
        let split_buffers = dispatch_classic_tier1_split_token_emit_for_cpu_pack(
            runtime,
            command_buffer,
            &coefficient_buffer,
            &job_buffer,
            &batch_jobs,
        )?;
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let job_count =
            usize::try_from(split_buffers.job_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token job count exceeds usize".to_string(),
            })?;
        let mq_token_stride_bytes =
            usize::try_from(split_buffers.mq_token_stride_bytes).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal split-token MQ byte stride exceeds usize"
                        .to_string(),
                }
            })?;
        let raw_token_stride_bytes = usize::try_from(split_buffers.raw_token_stride_bytes)
            .map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal split-token raw byte stride exceeds usize".to_string(),
            })?;
        let token_segment_stride =
            usize::try_from(split_buffers.token_segment_stride).map_err(|_| {
                Error::MetalKernel {
                    message: "classic J2K Metal split-token segment stride exceeds usize"
                        .to_string(),
                }
            })?;
        let counters = unsafe {
            core::slice::from_raw_parts(
                split_buffers
                    .counter_buffer
                    .contents()
                    .cast::<J2kClassicTier1SymbolPlanCounters>(),
                job_count,
            )
        };
        let mq_token_bytes = unsafe {
            core::slice::from_raw_parts(
                split_buffers.mq_token_buffer.contents().cast::<u8>(),
                job_count.saturating_mul(mq_token_stride_bytes),
            )
        };
        let raw_token_bytes = unsafe {
            core::slice::from_raw_parts(
                split_buffers.raw_token_buffer.contents().cast::<u8>(),
                job_count.saturating_mul(raw_token_stride_bytes),
            )
        };
        let token_segments = unsafe {
            core::slice::from_raw_parts(
                split_buffers
                    .segment_buffer
                    .contents()
                    .cast::<J2kClassicTier1TokenSegment>(),
                job_count.saturating_mul(token_segment_stride),
            )
        };

        let mut results = Vec::with_capacity(job_count);
        for (block_idx, counter) in counters.iter().copied().enumerate() {
            let segment_count =
                usize::try_from(counter.segment_count).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal split-token segment count exceeds usize"
                        .to_string(),
                })?;
            if segment_count > token_segment_stride {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal split-token segment count exceeds capacity"
                        .to_string(),
                });
            }
            let mq_token_start = block_idx
                .checked_mul(mq_token_stride_bytes)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal split-token MQ byte offset overflow".to_string(),
                })?;
            let raw_token_start =
                block_idx
                    .checked_mul(raw_token_stride_bytes)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal split-token raw byte offset overflow"
                            .to_string(),
                    })?;
            let segment_start =
                block_idx
                    .checked_mul(token_segment_stride)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal split-token segment offset overflow"
                            .to_string(),
                    })?;
            results.push(pack_classic_split_mq_raw_tokens_for_test(
                &mq_token_bytes[mq_token_start..mq_token_start + mq_token_stride_bytes],
                &raw_token_bytes[raw_token_start..raw_token_start + raw_token_stride_bytes],
                &token_segments[segment_start..segment_start + segment_count],
                counter,
            )?);
        }

        Ok(results)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_classic_tier1_prepared_device_code_blocks_resident(
    session: &crate::MetalBackendSession,
    prepared: J2kPreparedLosslessDeviceCodeBlocks,
) -> Result<J2kResidentLosslessTier1CodeBlocks, Error> {
    let J2kPreparedLosslessDeviceCodeBlocks {
        coefficient_buffer,
        coefficient_byte_offset: _,
        coefficient_byte_len: _,
        coefficient_buffer_is_batch_shared: _,
        code_blocks,
        recyclable_private_buffers: _,
        _prepare_command_buffer: prepare_command_buffer,
        _prepare_deinterleave_rct_command_buffer: _,
        _prepare_dwt53_command_buffer: _,
        _prepare_dwt53_vertical_command_buffers: _,
        _prepare_dwt53_horizontal_command_buffers: _,
        _prepare_coefficient_extract_command_buffer: _,
        _deinterleave_status_buffer: deinterleave_status_buffer,
        _plane_buffers: plane_buffers,
        _scratch_buffers: scratch_buffers,
        _coefficient_job_buffer: coefficient_job_buffer,
    } = prepared;
    with_runtime_for_device(&session.device, |runtime| {
        if code_blocks.is_empty() {
            let output = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let status_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let segment_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let job_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModeShared);
            let command_buffer = runtime.queue.new_command_buffer();
            command_buffer.commit();
            return Ok(J2kResidentLosslessTier1CodeBlocks {
                output_buffer: output,
                status_buffer,
                job_buffer,
                batch_jobs: Vec::new(),
                code_blocks,
                output_capacity_total: 0,
                _segment_buffer: segment_buffer,
                tier1_command_buffer: command_buffer.to_owned(),
                _coefficient_buffer: coefficient_buffer,
                prepare_command_buffer,
                _deinterleave_status_buffer: deinterleave_status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }
        let mut batch_jobs = Vec::<J2kClassicEncodeBatchJob>::with_capacity(code_blocks.len());
        let mut output_capacity_total = 0usize;
        let mut segment_capacity_total = 0usize;

        for block in &code_blocks {
            let output_capacity =
                classic_encode_output_capacity(block.width, block.height, block.total_bitplanes)?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal resident encode output table exceeds u32"
                        .to_string(),
                })?;
            let segment_offset =
                u32::try_from(segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "classic J2K Metal resident encode segment table exceeds u32"
                        .to_string(),
                })?;
            let style_flags = 0;
            let segment_capacity =
                classic_encode_segment_capacity(style_flags, block.total_bitplanes);
            batch_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset: block.coefficient_offset,
                output_offset,
                segment_offset,
                width: block.width,
                height: block.height,
                sub_band_type: classic_encode_sub_band_code(block.sub_band_type),
                total_bitplanes: u32::from(block.total_bitplanes),
                style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal resident encode output capacity exceeds u32"
                            .to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal resident encode segment capacity exceeds u32"
                            .to_string(),
                    }
                })?,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal resident encode output buffer overflow".to_string(),
                })?;
            segment_capacity_total = segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal resident encode segment buffer overflow"
                        .to_string(),
                })?;
        }

        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = runtime.device.new_buffer(
            (batch_jobs.len() * size_of::<J2kClassicEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let segment_buffer = runtime.device.new_buffer(
            (segment_capacity_total * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal resident encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let classic_encode_pipeline = classic_encode_code_blocks_pipeline(runtime, &batch_jobs);
        encoder.set_compute_pipeline_state(classic_encode_pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(encoder, classic_encode_pipeline, u64::from(job_count));
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kResidentLosslessTier1CodeBlocks {
            output_buffer: output,
            status_buffer,
            job_buffer,
            batch_jobs,
            code_blocks,
            output_capacity_total,
            _segment_buffer: segment_buffer,
            tier1_command_buffer: command_buffer.to_owned(),
            _coefficient_buffer: coefficient_buffer,
            prepare_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_prepared_device_code_blocks_resident(
    session: &crate::MetalBackendSession,
    prepared: J2kPreparedLosslessDeviceCodeBlocks,
) -> Result<J2kResidentLosslessHtCodeBlocks, Error> {
    let J2kPreparedLosslessDeviceCodeBlocks {
        coefficient_buffer,
        coefficient_byte_offset: _,
        coefficient_byte_len: _,
        coefficient_buffer_is_batch_shared: _,
        code_blocks,
        recyclable_private_buffers: _,
        _prepare_command_buffer: prepare_command_buffer,
        _prepare_deinterleave_rct_command_buffer: _,
        _prepare_dwt53_command_buffer: _,
        _prepare_dwt53_vertical_command_buffers: _,
        _prepare_dwt53_horizontal_command_buffers: _,
        _prepare_coefficient_extract_command_buffer: _,
        _deinterleave_status_buffer: deinterleave_status_buffer,
        _plane_buffers: plane_buffers,
        _scratch_buffers: scratch_buffers,
        _coefficient_job_buffer: coefficient_job_buffer,
    } = prepared;
    with_runtime_for_device(&session.device, |runtime| {
        if code_blocks.is_empty() {
            let output = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let status_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModePrivate);
            let job_buffer = runtime
                .device
                .new_buffer(1, MTLResourceOptions::StorageModeShared);
            let command_buffer = runtime.queue.new_command_buffer();
            command_buffer.commit();
            return Ok(J2kResidentLosslessHtCodeBlocks {
                output_buffer: output,
                status_buffer,
                job_buffer,
                batch_jobs: Vec::new(),
                code_blocks,
                output_capacity_total: 0,
                tier1_command_buffer: command_buffer.to_owned(),
                _coefficient_buffer: coefficient_buffer,
                prepare_command_buffer,
                _deinterleave_status_buffer: deinterleave_status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }

        let mut batch_jobs = Vec::<J2kHtEncodeBatchJob>::with_capacity(code_blocks.len());
        let mut output_capacity_total = 0usize;

        for block in &code_blocks {
            let output_capacity = ht_encode_output_capacity(block.width, block.height)?;
            let output_capacity_u32 =
                u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal resident encode output capacity exceeds u32".to_string(),
                })?;
            let output_offset =
                u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal resident encode output table exceeds u32".to_string(),
                })?;
            batch_jobs.push(J2kHtEncodeBatchJob {
                coefficient_offset: block.coefficient_offset,
                output_offset,
                width: block.width,
                height: block.height,
                total_bitplanes: u32::from(block.total_bitplanes),
                output_capacity: output_capacity_u32,
            });
            output_capacity_total = output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal resident encode output buffer overflow".to_string(),
                })?;
        }

        let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
        let output = runtime.device.new_buffer(
            output_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = runtime.device.new_buffer(
            (batch_jobs.len() * size_of::<J2kHtEncodeStatus>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal resident encode job count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "signinum-j2k htj2k resident tier1");
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "HTJ2K Tier-1 encode");
        let pipeline = &runtime.ht_encode_code_blocks;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_buffer(2, Some(&job_buffer), 0);
        encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
        encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
        encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
        encoder.set_buffer(6, Some(&status_buffer), 0);
        encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
        dispatch_1d_pipeline(encoder, pipeline, u64::from(job_count));
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kResidentLosslessHtCodeBlocks {
            output_buffer: output,
            status_buffer,
            job_buffer,
            batch_jobs,
            code_blocks,
            output_capacity_total,
            tier1_command_buffer: command_buffer.to_owned(),
            _coefficient_buffer: coefficient_buffer,
            prepare_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_classic_tier1_code_block(
    job: J2kTier1CodeBlockEncodeJob<'_>,
) -> Result<EncodedJ2kCodeBlock, Error> {
    with_runtime(|runtime| {
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode coefficient slice is too small".to_string(),
            });
        }

        let output_capacity =
            classic_encode_output_capacity(job.width, job.height, job.total_bitplanes)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let style_flags = classic_style_flags(job.style);
        let segment_capacity = classic_encode_segment_capacity(style_flags, job.total_bitplanes);
        let params = J2kClassicEncodeParams {
            width: job.width,
            height: job.height,
            sub_band_type: classic_encode_sub_band_code(job.sub_band_type),
            total_bitplanes: u32::from(job.total_bitplanes),
            style_flags,
            output_capacity: output_capacity_u32,
            segment_capacity: u32::try_from(segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds u32".to_string(),
            })?,
        };
        let coefficients =
            borrow_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients]);
        let output = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kClassicEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let segment_buffer = runtime.device.new_buffer(
            (usize::try_from(params.segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
            })? * size_of::<J2kClassicSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.classic_encode_code_block);
        encoder.set_buffer(0, Some(&coefficients), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kClassicEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(3, Some(&status_buffer), 0);
        encoder.set_buffer(4, Some(&segment_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kClassicEncodeStatus>()
                .read()
        };
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "classic Tier-1",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode length exceeds usize".to_string(),
        })?;
        let payload_skip = usize::try_from(status.reserved0).map_err(|_| Error::MetalKernel {
            message: "classic J2K Metal encode payload skip exceeds usize".to_string(),
        })?;
        let payload_span =
            data_len
                .checked_add(payload_skip)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal encode payload span overflow".to_string(),
                })?;
        if payload_span > output_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let payload_offset = payload_skip;
        let data = if data_len == 0 {
            Vec::new()
        } else {
            unsafe {
                core::slice::from_raw_parts(
                    output.contents().cast::<u8>().add(payload_offset),
                    data_len,
                )
            }
            .to_vec()
        };
        let number_of_coding_passes =
            u8::try_from(status.number_of_coding_passes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode pass count exceeds u8".to_string(),
            })?;
        let missing_bit_planes =
            u8::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode missing bitplanes exceeds u8".to_string(),
            })?;
        let segment_count =
            usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment count exceeds usize".to_string(),
            })?;
        let segment_capacity =
            usize::try_from(params.segment_capacity).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal encode segment capacity exceeds usize".to_string(),
            })?;
        if segment_count > segment_capacity {
            return Err(Error::MetalKernel {
                message: "classic J2K Metal encode segment count exceeds buffer".to_string(),
            });
        }
        let raw_segments = if segment_count == 0 {
            &[][..]
        } else {
            unsafe {
                core::slice::from_raw_parts(
                    segment_buffer.contents().cast::<J2kClassicSegment>(),
                    segment_count,
                )
            }
        };
        let segments = raw_segments
            .iter()
            .map(|segment| {
                Ok(J2kCodeBlockSegment {
                    data_offset: segment.data_offset,
                    data_length: segment.data_length,
                    start_coding_pass: u8::try_from(segment.start_coding_pass).map_err(|_| {
                        Error::MetalKernel {
                            message: "classic J2K Metal encode segment start pass exceeds u8"
                                .to_string(),
                        }
                    })?,
                    end_coding_pass: u8::try_from(segment.end_coding_pass).map_err(|_| {
                        Error::MetalKernel {
                            message: "classic J2K Metal encode segment end pass exceeds u8"
                                .to_string(),
                        }
                    })?,
                    use_arithmetic: segment.use_arithmetic != 0,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(EncodedJ2kCodeBlock {
            data,
            segments,
            number_of_coding_passes,
            missing_bit_planes,
        })
    })
}

#[cfg(target_os = "macos")]
fn read_ht_encoded_code_block(
    status: J2kHtEncodeStatus,
    output: &Buffer,
    output_offset: usize,
    output_capacity: usize,
) -> Result<EncodedHtJ2kCodeBlock, Error> {
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            "HTJ2K cleanup",
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal encode length exceeds usize".to_string(),
    })?;
    if data_len > output_capacity {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal encode length exceeds output buffer".to_string(),
        });
    }
    let data = if data_len == 0 {
        Vec::new()
    } else {
        unsafe {
            core::slice::from_raw_parts(output.contents().cast::<u8>().add(output_offset), data_len)
        }
        .to_vec()
    };
    Ok(EncodedHtJ2kCodeBlock {
        data,
        cleanup_length: status.data_len,
        refinement_length: 0,
        num_coding_passes: u8::try_from(status.num_coding_passes).map_err(|_| {
            Error::MetalKernel {
                message: "HTJ2K Metal encode pass count exceeds u8".to_string(),
            }
        })?,
        num_zero_bitplanes: u8::try_from(status.num_zero_bitplanes).map_err(|_| {
            Error::MetalKernel {
                message: "HTJ2K Metal encode zero bitplanes exceeds u8".to_string(),
            }
        })?,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn read_resident_ht_tier1_code_blocks_for_cpu_packetization(
    session: &crate::MetalBackendSession,
    tier1: &J2kResidentLosslessHtCodeBlocks,
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    with_runtime_for_device(&session.device, |runtime| {
        if tier1.batch_jobs.is_empty() {
            return Ok(Vec::new());
        }
        let output_bytes = tier1.output_capacity_total.max(1);
        let status_bytes = tier1
            .batch_jobs
            .len()
            .checked_mul(size_of::<J2kHtEncodeStatus>())
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal resident status readback size overflow".to_string(),
            })?;
        let output = runtime
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let status_buffer = runtime
            .device
            .new_buffer(status_bytes as u64, MTLResourceOptions::StorageModeShared);

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(
            command_buffer,
            "signinum-j2k htj2k resident tier1 cpu readback",
        );
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &tier1.output_buffer,
            0,
            &output,
            0,
            tier1.output_capacity_total as u64,
        );
        blit.copy_from_buffer(
            &tier1.status_buffer,
            0,
            &status_buffer,
            0,
            status_bytes as u64,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let statuses = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<J2kHtEncodeStatus>(),
                tier1.batch_jobs.len(),
            )
        };
        tier1
            .batch_jobs
            .iter()
            .zip(statuses.iter().copied())
            .map(|(batch_job, status)| {
                read_ht_encoded_code_block(
                    status,
                    &output,
                    usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal resident output offset exceeds usize".to_string(),
                    })?,
                    usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal resident output capacity exceeds usize".to_string(),
                    })?,
                )
            })
            .collect()
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_cleanup_code_blocks(
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    with_runtime(|runtime| encode_ht_cleanup_code_blocks_with_runtime(runtime, jobs))
}

#[cfg(target_os = "macos")]
fn encode_ht_cleanup_code_blocks_with_runtime(
    runtime: &MetalRuntime,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<EncodedHtJ2kCodeBlock>, Error> {
    encode_ht_cleanup_code_blocks_with_runtime_and_statuses(runtime, jobs).map(|blocks| {
        blocks
            .into_iter()
            .map(|(encoded, _status)| encoded)
            .collect()
    })
}

#[cfg(target_os = "macos")]
fn encode_ht_cleanup_code_blocks_with_runtime_and_statuses(
    runtime: &MetalRuntime,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<(EncodedHtJ2kCodeBlock, J2kHtEncodeStatus)>, Error> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    if jobs.iter().any(|job| job.target_coding_passes != 1) {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal cleanup encode supports one coding pass".to_string(),
        });
    }

    let mut coefficients = Vec::<i32>::new();
    let mut batch_jobs = Vec::<J2kHtEncodeBatchJob>::with_capacity(jobs.len());
    let mut output_capacity_total = 0usize;

    for job in jobs {
        let output_capacity = ht_encode_output_capacity(job.width, job.height)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient slice is too small".to_string(),
            });
        }
        let coefficient_offset =
            u32::try_from(coefficients.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient table exceeds u32".to_string(),
            })?;
        coefficients.extend_from_slice(&job.coefficients[..expected_coefficients]);
        let output_offset =
            u32::try_from(output_capacity_total).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output table exceeds u32".to_string(),
            })?;
        batch_jobs.push(J2kHtEncodeBatchJob {
            coefficient_offset,
            output_offset,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_capacity: output_capacity_u32,
        });
        output_capacity_total = output_capacity_total
            .checked_add(output_capacity)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode output buffer overflow".to_string(),
            })?;
    }

    let coefficient_buffer = owned_slice_buffer(&runtime.device, &coefficients);
    let job_buffer = owned_slice_buffer(&runtime.device, &batch_jobs);
    let output = runtime.device.new_buffer(
        output_capacity_total.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let status_buffer = runtime.device.new_buffer(
        (jobs.len() * size_of::<J2kHtEncodeStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let job_count = u32::try_from(batch_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal encode job count exceeds u32".to_string(),
    })?;

    let command_buffer = runtime.queue.new_command_buffer();
    label_command_buffer(command_buffer, "signinum-j2k htj2k tier1 batch");
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "HTJ2K Tier-1 encode");
    let pipeline = &runtime.ht_encode_code_blocks;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&coefficient_buffer), 0);
    encoder.set_buffer(1, Some(&output), 0);
    encoder.set_buffer(2, Some(&job_buffer), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
    encoder.set_buffer(6, Some(&status_buffer), 0);
    encoder.set_bytes(7, size_of::<u32>() as u64, (&raw const job_count).cast());
    dispatch_1d_pipeline(encoder, pipeline, u64::from(job_count));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let statuses = unsafe {
        core::slice::from_raw_parts(
            status_buffer.contents().cast::<J2kHtEncodeStatus>(),
            jobs.len(),
        )
    };
    let mut results = Vec::with_capacity(jobs.len());
    for (idx, status) in statuses.iter().copied().enumerate() {
        let batch_job = batch_jobs[idx];
        let encoded_block = read_ht_encoded_code_block(
            status,
            &output,
            usize::try_from(batch_job.output_offset).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output offset exceeds usize".to_string(),
            })?,
            usize::try_from(batch_job.output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds usize".to_string(),
            })?,
        )?;
        results.push((encoded_block, status));
    }

    Ok(results)
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_ht_cleanup_code_block(
    job: J2kHtCodeBlockEncodeJob<'_>,
) -> Result<EncodedHtJ2kCodeBlock, Error> {
    with_runtime(|runtime| {
        if job.target_coding_passes != 1 {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal cleanup encode supports one coding pass".to_string(),
            });
        }
        let expected_coefficients = usize::try_from(job.width)
            .ok()
            .and_then(|w| {
                usize::try_from(job.height)
                    .ok()
                    .and_then(|h| w.checked_mul(h))
            })
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient count overflow".to_string(),
            })?;
        if job.coefficients.len() < expected_coefficients {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode coefficient slice is too small".to_string(),
            });
        }
        let output_capacity = ht_encode_output_capacity(job.width, job.height)?;
        let output_capacity_u32 =
            u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal encode output capacity exceeds u32".to_string(),
            })?;
        let params = J2kHtEncodeParams {
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_capacity: output_capacity_u32,
        };
        let coefficients =
            borrow_slice_buffer(&runtime.device, &job.coefficients[..expected_coefficients]);
        let output = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kHtEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.ht_encode_code_block);
        encoder.set_buffer(0, Some(&coefficients), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kHtEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
        encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
        encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
        encoder.set_buffer(6, Some(&status_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe { status_buffer.contents().cast::<J2kHtEncodeStatus>().read() };
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "HTJ2K cleanup",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal encode length exceeds usize".to_string(),
        })?;
        if data_len > output_capacity {
            return Err(Error::MetalKernel {
                message: "HTJ2K Metal encode length exceeds output buffer".to_string(),
            });
        }
        let data = if data_len == 0 {
            Vec::new()
        } else {
            unsafe { core::slice::from_raw_parts(output.contents().cast::<u8>(), data_len) }
                .to_vec()
        };
        Ok(EncodedHtJ2kCodeBlock {
            data,
            cleanup_length: status.data_len,
            refinement_length: 0,
            num_coding_passes: u8::try_from(status.num_coding_passes).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Metal encode pass count exceeds u8".to_string(),
                }
            })?,
            num_zero_bitplanes: u8::try_from(status.num_zero_bitplanes).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Metal encode zero bitplanes exceeds u8".to_string(),
                }
            })?,
        })
    })
}

#[cfg(target_os = "macos")]
fn packet_tree_node_count(width: u32, height: u32) -> Result<usize, Error> {
    if width == 0 || height == 0 {
        return Ok(0);
    }
    let mut total = 0usize;
    let mut w = width;
    let mut h = height;
    loop {
        total = total
            .checked_add(
                usize::try_from(w)
                    .ok()
                    .and_then(|wu| usize::try_from(h).ok().and_then(|hu| wu.checked_mul(hu)))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet tag-tree size overflow".to_string(),
                    })?,
            )
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet tag-tree node count overflow".to_string(),
            })?;
        if w <= 1 && h <= 1 {
            break;
        }
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    Ok(total)
}

#[cfg(target_os = "macos")]
fn lossless_codestream_payload_offset(
    job: J2kLosslessCodestreamAssemblyJob,
) -> Result<usize, Error> {
    let component_count = usize::from(job.num_components);
    let qcd_steps = 1usize
        .checked_add(
            usize::from(job.num_decomposition_levels)
                .checked_mul(3)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal codestream assembly QCD step count overflow".to_string(),
                })?,
        )
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream assembly QCD step count overflow".to_string(),
        })?;
    let siz_total = 40usize
        .checked_add(
            component_count
                .checked_mul(3)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal codestream assembly SIZ size overflow".to_string(),
                })?,
        )
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream assembly SIZ size overflow".to_string(),
        })?;
    let qcd_total = 5usize
        .checked_add(qcd_steps)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream assembly QCD size overflow".to_string(),
        })?;
    2usize
        .checked_add(siz_total)
        .and_then(|len| {
            len.checked_add(
                if job.block_coding_mode == J2kLosslessCodestreamBlockCodingMode::HighThroughput {
                    10
                } else {
                    0
                },
            )
        })
        .and_then(|len| len.checked_add(14))
        .and_then(|len| len.checked_add(qcd_total))
        .and_then(|len| len.checked_add(if job.write_tlm { 12 } else { 0 }))
        .and_then(|len| len.checked_add(12))
        .and_then(|len| len.checked_add(2))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream payload offset overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn lossless_codestream_assembly_capacity(
    tile_capacity: usize,
    job: J2kLosslessCodestreamAssemblyJob,
) -> Result<usize, Error> {
    lossless_codestream_payload_offset(job)?
        .checked_add(tile_capacity)
        .and_then(|len| len.checked_add(2))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal codestream assembly capacity overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn lossless_raw_sample_bytes(
    job: J2kLosslessCodestreamAssemblyJob,
    overflow_message: &'static str,
) -> Result<usize, Error> {
    let pixels = usize::try_from(job.width)
        .ok()
        .and_then(|width| {
            usize::try_from(job.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| Error::MetalKernel {
            message: overflow_message.to_string(),
        })?;
    let component_count = usize::from(job.num_components);
    let bytes_per_sample = usize::from(job.bit_depth).div_ceil(8).max(1);
    pixels
        .checked_mul(component_count)
        .and_then(|bytes| bytes.checked_mul(bytes_per_sample))
        .ok_or_else(|| Error::MetalKernel {
            message: overflow_message.to_string(),
        })
}

#[cfg(target_os = "macos")]
fn ht_lossless_raw_sample_bytes(job: J2kLosslessCodestreamAssemblyJob) -> Result<usize, Error> {
    lossless_raw_sample_bytes(job, "HTJ2K Metal batch raw sample byte count overflow")
}

#[cfg(target_os = "macos")]
fn classic_packet_output_capacity(
    tier1_output_capacity: usize,
    header_capacity: usize,
    packet_descriptor_count: usize,
    codestream: J2kLosslessCodestreamAssemblyJob,
) -> Result<usize, Error> {
    let descriptor_count = packet_descriptor_count.max(1);
    let conservative_capacity = tier1_output_capacity
        .checked_add(header_capacity.saturating_mul(descriptor_count))
        .and_then(|bytes| bytes.checked_add(1024))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal batch packet output capacity overflow".to_string(),
        })?;
    let raw_bytes =
        lossless_raw_sample_bytes(codestream, "J2K Metal batch raw sample byte count overflow")?;
    let descriptor_header_slack =
        descriptor_count
            .checked_mul(256)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal batch packet descriptor slack overflow".to_string(),
            })?;
    let tight_capacity = raw_bytes
        .checked_add(header_capacity)
        .and_then(|bytes| bytes.checked_add(descriptor_header_slack))
        .and_then(|bytes| bytes.checked_add(64 * 1024))
        .map(|bytes| bytes.max(4096))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal batch packet output capacity overflow".to_string(),
        })?;

    Ok(tight_capacity.min(conservative_capacity))
}

#[cfg(target_os = "macos")]
fn ht_packet_output_capacity_for_mode(
    code_block_count: usize,
    header_capacity: usize,
    packet_descriptor_count: usize,
    codestream: J2kLosslessCodestreamAssemblyJob,
    mode: J2kHtPacketOutputCapacityMode,
) -> Result<usize, Error> {
    let descriptor_count = packet_descriptor_count.max(1);
    match mode {
        J2kHtPacketOutputCapacityMode::Conservative => code_block_count
            .checked_mul(J2K_HT_ENCODE_BASE_OUTPUT_SIZE)
            .and_then(|bytes| bytes.checked_add(header_capacity.saturating_mul(descriptor_count)))
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal batch packet output capacity overflow".to_string(),
            }),
        J2kHtPacketOutputCapacityMode::Tight => {
            let raw_bytes = ht_lossless_raw_sample_bytes(codestream)?;
            let descriptor_header_slack =
                descriptor_count
                    .checked_mul(256)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch packet descriptor slack overflow".to_string(),
                    })?;
            raw_bytes
                .checked_add(header_capacity)
                .and_then(|bytes| bytes.checked_add(descriptor_header_slack))
                .and_then(|bytes| bytes.checked_add(64 * 1024))
                .map(|bytes| bytes.max(4096))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch packet output capacity overflow".to_string(),
                })
        }
    }
}

#[cfg(target_os = "macos")]
fn codestream_progression_order_code(order: EncodeProgressionOrder) -> u32 {
    match order {
        EncodeProgressionOrder::Lrcp => 0x00,
        EncodeProgressionOrder::Rlcp => 0x01,
        EncodeProgressionOrder::Rpcl => 0x02,
        EncodeProgressionOrder::Pcrl => 0x03,
        EncodeProgressionOrder::Cprl => 0x04,
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_tier2_packetization(
    job: J2kPacketizationEncodeJob<'_>,
) -> Result<Vec<u8>, Error> {
    with_runtime(|runtime| {
        let mut resolutions = Vec::<J2kPacketResolution>::new();
        let mut subbands = Vec::<J2kPacketSubband>::new();
        let mut blocks = Vec::<J2kPacketBlock>::new();
        let mut payload = Vec::<u8>::new();
        let mut max_tree_nodes = 1usize;

        for resolution in job.resolutions {
            let subband_offset = u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet subband table exceeds u32".to_string(),
            })?;
            for subband in &resolution.subbands {
                let block_offset = u32::try_from(blocks.len()).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet block table exceeds u32".to_string(),
                })?;
                max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                    subband.num_cbs_x,
                    subband.num_cbs_y,
                )?);
                for code_block in &subband.code_blocks {
                    let data_offset =
                        u32::try_from(payload.len()).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet payload exceeds u32".to_string(),
                        })?;
                    let data_len =
                        u32::try_from(code_block.data.len()).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet code-block payload exceeds u32"
                                .to_string(),
                        })?;
                    payload.extend_from_slice(code_block.data);
                    blocks.push(J2kPacketBlock {
                        data_offset,
                        data_len,
                        num_coding_passes: u32::from(code_block.num_coding_passes),
                        num_zero_bitplanes: u32::from(code_block.num_zero_bitplanes),
                        previously_included: u32::from(code_block.previously_included),
                        l_block: code_block.l_block,
                        block_coding_mode: match code_block.block_coding_mode {
                            J2kPacketizationBlockCodingMode::Classic => 0,
                            J2kPacketizationBlockCodingMode::HighThroughput => 1,
                        },
                        reserved0: 0,
                    });
                }
                subbands.push(J2kPacketSubband {
                    block_offset,
                    block_count: u32::try_from(subband.code_blocks.len()).map_err(|_| {
                        Error::MetalKernel {
                            message: "Tier-2 Metal packet subband block count exceeds u32"
                                .to_string(),
                        }
                    })?,
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                });
            }
            resolutions.push(J2kPacketResolution {
                subband_offset,
                subband_count: u32::try_from(resolution.subbands.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "Tier-2 Metal packet resolution subband count exceeds u32"
                            .to_string(),
                    }
                })?,
            });
        }

        let mut state_block_offsets = HashMap::<u32, (u32, usize)>::new();
        let mut state_blocks = Vec::<J2kPacketStateBlock>::new();
        let mut descriptors =
            Vec::<J2kPacketDescriptor>::with_capacity(job.packet_descriptors.len());
        for descriptor in job.packet_descriptors {
            let packet_index =
                usize::try_from(descriptor.packet_index).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor packet index exceeds usize"
                        .to_string(),
                })?;
            let resolution = resolutions
                .get(packet_index)
                .ok_or_else(|| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor packet index out of range".to_string(),
                })?;
            let subband_start =
                usize::try_from(resolution.subband_offset).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband offset exceeds usize"
                        .to_string(),
                })?;
            let subband_count =
                usize::try_from(resolution.subband_count).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband count exceeds usize"
                        .to_string(),
                })?;
            let subband_end =
                subband_start
                    .checked_add(subband_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet descriptor subband range overflow"
                            .to_string(),
                    })?;
            if subband_end > subbands.len() {
                return Err(Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor subband range out of bounds"
                        .to_string(),
                });
            }
            let mut packet_block_count = 0usize;
            for subband in &subbands[subband_start..subband_end] {
                packet_block_count = packet_block_count
                    .checked_add(usize::try_from(subband.block_count).map_err(|_| {
                        Error::MetalKernel {
                            message: "Tier-2 Metal packet descriptor block count exceeds usize"
                                .to_string(),
                        }
                    })?)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet descriptor block count overflow".to_string(),
                    })?;
            }

            let (state_block_offset, existing_count) = if let Some(&(offset, count)) =
                state_block_offsets.get(&descriptor.state_index)
            {
                (offset, count)
            } else {
                let offset = u32::try_from(state_blocks.len()).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal packet state block offset exceeds u32".to_string(),
                })?;
                for subband in &subbands[subband_start..subband_end] {
                    let block_start =
                        usize::try_from(subband.block_offset).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet state block offset exceeds usize"
                                .to_string(),
                        })?;
                    let block_count =
                        usize::try_from(subband.block_count).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal packet state block count exceeds usize"
                                .to_string(),
                        })?;
                    let block_end =
                        block_start
                            .checked_add(block_count)
                            .ok_or_else(|| Error::MetalKernel {
                                message: "Tier-2 Metal packet state block range overflow"
                                    .to_string(),
                            })?;
                    if block_end > blocks.len() {
                        return Err(Error::MetalKernel {
                            message: "Tier-2 Metal packet state block range out of bounds"
                                .to_string(),
                        });
                    }
                    for block in &blocks[block_start..block_end] {
                        state_blocks.push(J2kPacketStateBlock {
                            previously_included: block.previously_included,
                            l_block: block.l_block,
                        });
                    }
                }
                state_block_offsets.insert(descriptor.state_index, (offset, packet_block_count));
                (offset, packet_block_count)
            };
            if existing_count != packet_block_count {
                return Err(Error::MetalKernel {
                    message: "Tier-2 Metal packet descriptor state layout mismatch".to_string(),
                });
            }

            descriptors.push(J2kPacketDescriptor {
                packet_index: descriptor.packet_index,
                state_index: descriptor.state_index,
                layer: u32::from(descriptor.layer),
                resolution: descriptor.resolution,
                component: u32::from(descriptor.component),
                precinct_lo: descriptor.precinct as u32,
                precinct_hi: (descriptor.precinct >> 32) as u32,
                state_block_offset,
            });
        }

        let header_capacity = blocks
            .len()
            .checked_mul(256)
            .and_then(|bytes| bytes.checked_add(4096))
            .map(|bytes| bytes.max(4096))
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet header capacity overflow".to_string(),
            })?;
        let output_capacity = payload
            .len()
            .checked_add(
                header_capacity
                    .checked_mul(descriptors.len().max(resolutions.len()).max(1))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal packet output capacity overflow".to_string(),
                    })?,
            )
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet output capacity overflow".to_string(),
            })?;

        let params = J2kPacketEncodeParams {
            resolution_count: u32::try_from(resolutions.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet resolution count exceeds u32".to_string(),
            })?,
            num_layers: u32::from(job.num_layers),
            num_components: u32::from(job.num_components),
            code_block_count: u32::try_from(blocks.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet code-block count exceeds u32".to_string(),
            })?,
            subband_count: u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet subband count exceeds u32".to_string(),
            })?,
            descriptor_count: u32::try_from(descriptors.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet descriptor count exceeds u32".to_string(),
            })?,
            output_capacity: u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet output capacity exceeds u32".to_string(),
            })?,
            header_capacity: u32::try_from(header_capacity).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal packet header capacity exceeds u32".to_string(),
            })?,
            scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| {
                Error::MetalKernel {
                    message: "Tier-2 Metal packet scratch node capacity exceeds u32".to_string(),
                }
            })?,
        };

        let resolution_buffer = copied_slice_buffer(&runtime.device, &resolutions);
        let subband_buffer = copied_slice_buffer(&runtime.device, &subbands);
        let block_buffer = copied_slice_buffer(&runtime.device, &blocks);
        let payload_buffer = copied_slice_buffer(&runtime.device, &payload);
        let descriptor_buffer = copied_slice_buffer(&runtime.device, &descriptors);
        let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks);
        let output_buffer = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let header_buffer = runtime.device.new_buffer(
            header_capacity as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let scratch_words = max_tree_nodes
            .checked_mul(6)
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal packet scratch size overflow".to_string(),
            })?;
        let scratch_buffer = runtime.device.new_buffer(
            (scratch_words * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kPacketEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.packet_encode);
        encoder.set_buffer(0, Some(&resolution_buffer), 0);
        encoder.set_buffer(1, Some(&subband_buffer), 0);
        encoder.set_buffer(2, Some(&block_buffer), 0);
        encoder.set_buffer(3, Some(&payload_buffer), 0);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_buffer(5, Some(&header_buffer), 0);
        encoder.set_buffer(6, Some(&scratch_buffer), 0);
        encoder.set_bytes(
            7,
            size_of::<J2kPacketEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(8, Some(&status_buffer), 0);
        encoder.set_buffer(9, Some(&descriptor_buffer), 0);
        encoder.set_buffer(10, Some(&state_block_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kPacketEncodeStatus>()
                .read()
        };
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                "Tier-2 packetization",
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: "Tier-2 Metal packet output length exceeds usize".to_string(),
        })?;
        if data_len > output_capacity {
            return Err(Error::MetalKernel {
                message: "Tier-2 Metal packet output length exceeds buffer".to_string(),
            });
        }
        Ok(if data_len == 0 {
            Vec::new()
        } else {
            unsafe { core::slice::from_raw_parts(output_buffer.contents().cast::<u8>(), data_len) }
                .to_vec()
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_lossless_codestream_buffer_from_resident_classic_tier1(
    session: &crate::MetalBackendSession,
    tier1: &J2kResidentLosslessTier1CodeBlocks,
    job: J2kResidentPacketizationEncodeJob<'_>,
    codestream_job: J2kLosslessCodestreamAssemblyJob,
) -> Result<J2kResidentLosslessCodestream, Error> {
    wait_resident_lossless_codestream(
        submit_lossless_codestream_buffer_from_resident_classic_tier1(
            session,
            tier1,
            job,
            codestream_job,
        )?,
    )
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_lossless_codestream_buffer_from_resident_classic_tier1(
    session: &crate::MetalBackendSession,
    tier1: &J2kResidentLosslessTier1CodeBlocks,
    job: J2kResidentPacketizationEncodeJob<'_>,
    codestream_job: J2kLosslessCodestreamAssemblyJob,
) -> Result<J2kPendingResidentLosslessCodestream, Error> {
    with_runtime_for_device(&session.device, |runtime| {
        if tier1.batch_jobs.len() != tier1.code_blocks.len() {
            return Err(Error::MetalKernel {
                message: "Tier-2 Metal resident packetization Tier-1 table mismatch".to_string(),
            });
        }

        let mut resolutions = Vec::<J2kPacketResolution>::new();
        let mut subbands = Vec::<J2kPacketSubband>::new();
        let mut resident_blocks = Vec::<J2kResidentPacketBlock>::new();
        let mut max_tree_nodes = 1usize;

        for resolution in job.resolutions {
            let subband_offset = u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal resident packet subband table exceeds u32".to_string(),
            })?;
            for subband in &resolution.subbands {
                let block_offset =
                    u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
                        message: "Tier-2 Metal resident packet block table exceeds u32".to_string(),
                    })?;
                max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                    subband.num_cbs_x,
                    subband.num_cbs_y,
                )?);
                let code_block_start =
                    usize::try_from(subband.code_block_start).map_err(|_| Error::MetalKernel {
                        message: "Tier-2 Metal resident packet code-block offset exceeds usize"
                            .to_string(),
                    })?;
                let code_block_count =
                    usize::try_from(subband.code_block_count).map_err(|_| Error::MetalKernel {
                        message: "Tier-2 Metal resident packet code-block count exceeds usize"
                            .to_string(),
                    })?;
                let code_block_end =
                    code_block_start
                        .checked_add(code_block_count)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "Tier-2 Metal resident packet code-block range overflow"
                                .to_string(),
                        })?;
                if code_block_end > tier1.batch_jobs.len() {
                    return Err(Error::MetalKernel {
                        message: "Tier-2 Metal resident packet code-block range out of bounds"
                            .to_string(),
                    });
                }
                for tier1_job_index in code_block_start..code_block_end {
                    resident_blocks.push(J2kResidentPacketBlock {
                        tier1_job_index: u32::try_from(tier1_job_index).map_err(|_| {
                            Error::MetalKernel {
                                message: "Tier-2 Metal resident packet Tier-1 index exceeds u32"
                                    .to_string(),
                            }
                        })?,
                        previously_included: 0,
                        l_block: 3,
                        block_coding_mode: 0,
                    });
                }
                subbands.push(J2kPacketSubband {
                    block_offset,
                    block_count: subband.code_block_count,
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                });
            }
            resolutions.push(J2kPacketResolution {
                subband_offset,
                subband_count: u32::try_from(resolution.subbands.len()).map_err(|_| {
                    Error::MetalKernel {
                        message:
                            "Tier-2 Metal resident packet resolution subband count exceeds u32"
                                .to_string(),
                    }
                })?,
            });
        }

        if resolutions.len()
            != usize::try_from(job.resolution_count).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal resident packet resolution count exceeds usize".to_string(),
            })?
        {
            return Err(Error::MetalKernel {
                message: "Tier-2 Metal resident packet resolution count mismatch".to_string(),
            });
        }
        if resident_blocks.len()
            != usize::try_from(job.code_block_count).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal resident packet code-block count exceeds usize".to_string(),
            })?
        {
            return Err(Error::MetalKernel {
                message: "Tier-2 Metal resident packet code-block count mismatch".to_string(),
            });
        }

        let mut state_block_offsets = HashMap::<u32, (u32, usize)>::new();
        let mut state_blocks = Vec::<J2kPacketStateBlock>::new();
        let mut descriptors =
            Vec::<J2kPacketDescriptor>::with_capacity(job.packet_descriptors.len());
        for descriptor in job.packet_descriptors {
            let packet_index =
                usize::try_from(descriptor.packet_index).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal resident packet descriptor packet index exceeds usize"
                        .to_string(),
                })?;
            let resolution = resolutions
                .get(packet_index)
                .ok_or_else(|| Error::MetalKernel {
                    message: "Tier-2 Metal resident packet descriptor packet index out of range"
                        .to_string(),
                })?;
            let subband_start =
                usize::try_from(resolution.subband_offset).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal resident packet descriptor subband offset exceeds usize"
                        .to_string(),
                })?;
            let subband_count =
                usize::try_from(resolution.subband_count).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal resident packet descriptor subband count exceeds usize"
                        .to_string(),
                })?;
            let subband_end =
                subband_start
                    .checked_add(subband_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal resident packet descriptor subband range overflow"
                            .to_string(),
                    })?;
            if subband_end > subbands.len() {
                return Err(Error::MetalKernel {
                    message: "Tier-2 Metal resident packet descriptor subband range out of bounds"
                        .to_string(),
                });
            }
            let mut packet_block_count = 0usize;
            for subband in &subbands[subband_start..subband_end] {
                packet_block_count = packet_block_count
                    .checked_add(usize::try_from(subband.block_count).map_err(|_| {
                        Error::MetalKernel {
                            message:
                                "Tier-2 Metal resident packet descriptor block count exceeds usize"
                                    .to_string(),
                        }
                    })?)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal resident packet descriptor block count overflow"
                            .to_string(),
                    })?;
            }

            let (state_block_offset, existing_count) = if let Some(&(offset, count)) =
                state_block_offsets.get(&descriptor.state_index)
            {
                (offset, count)
            } else {
                let offset = u32::try_from(state_blocks.len()).map_err(|_| Error::MetalKernel {
                    message: "Tier-2 Metal resident packet state block offset exceeds u32"
                        .to_string(),
                })?;
                for subband in &subbands[subband_start..subband_end] {
                    let block_start =
                        usize::try_from(subband.block_offset).map_err(|_| Error::MetalKernel {
                            message:
                                "Tier-2 Metal resident packet state block offset exceeds usize"
                                    .to_string(),
                        })?;
                    let block_count =
                        usize::try_from(subband.block_count).map_err(|_| Error::MetalKernel {
                            message: "Tier-2 Metal resident packet state block count exceeds usize"
                                .to_string(),
                        })?;
                    let block_end =
                        block_start
                            .checked_add(block_count)
                            .ok_or_else(|| Error::MetalKernel {
                                message: "Tier-2 Metal resident packet state block range overflow"
                                    .to_string(),
                            })?;
                    if block_end > resident_blocks.len() {
                        return Err(Error::MetalKernel {
                            message: "Tier-2 Metal resident packet state block range out of bounds"
                                .to_string(),
                        });
                    }
                    for block in &resident_blocks[block_start..block_end] {
                        state_blocks.push(J2kPacketStateBlock {
                            previously_included: block.previously_included,
                            l_block: block.l_block,
                        });
                    }
                }
                state_block_offsets.insert(descriptor.state_index, (offset, packet_block_count));
                (offset, packet_block_count)
            };
            if existing_count != packet_block_count {
                return Err(Error::MetalKernel {
                    message: "Tier-2 Metal resident packet descriptor state layout mismatch"
                        .to_string(),
                });
            }

            descriptors.push(J2kPacketDescriptor {
                packet_index: descriptor.packet_index,
                state_index: descriptor.state_index,
                layer: u32::from(descriptor.layer),
                resolution: descriptor.resolution,
                component: u32::from(descriptor.component),
                precinct_lo: descriptor.precinct as u32,
                precinct_hi: (descriptor.precinct >> 32) as u32,
                state_block_offset,
            });
        }

        let header_capacity = resident_blocks
            .len()
            .checked_mul(256)
            .and_then(|bytes| bytes.checked_add(4096))
            .map(|bytes| bytes.max(4096))
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal resident packet header capacity overflow".to_string(),
            })?;
        let output_capacity = tier1
            .output_capacity_total
            .checked_add(
                header_capacity
                    .checked_mul(descriptors.len().max(resolutions.len()).max(1))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "Tier-2 Metal resident packet output capacity overflow"
                            .to_string(),
                    })?,
            )
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal resident packet output capacity overflow".to_string(),
            })?;
        let codestream_capacity =
            lossless_codestream_assembly_capacity(output_capacity, codestream_job)?;

        let params = J2kPacketEncodeParams {
            resolution_count: u32::try_from(resolutions.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal resident packet resolution count exceeds u32".to_string(),
            })?,
            num_layers: u32::from(job.num_layers),
            num_components: u32::from(job.num_components),
            code_block_count: u32::try_from(resident_blocks.len()).map_err(|_| {
                Error::MetalKernel {
                    message: "Tier-2 Metal resident packet code-block count exceeds u32"
                        .to_string(),
                }
            })?,
            subband_count: u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal resident packet subband count exceeds u32".to_string(),
            })?,
            descriptor_count: u32::try_from(descriptors.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal resident packet descriptor count exceeds u32".to_string(),
            })?,
            output_capacity: u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal resident packet output capacity exceeds u32".to_string(),
            })?,
            header_capacity: u32::try_from(header_capacity).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal resident packet header capacity exceeds u32".to_string(),
            })?,
            scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| {
                Error::MetalKernel {
                    message: "Tier-2 Metal resident packet scratch node capacity exceeds u32"
                        .to_string(),
                }
            })?,
        };
        let codestream_params = J2kLosslessCodestreamAssemblyParams {
            width: codestream_job.width,
            height: codestream_job.height,
            num_components: u32::from(codestream_job.num_components),
            bit_depth: u32::from(codestream_job.bit_depth),
            signed_samples: u32::from(codestream_job.signed),
            num_decomposition_levels: u32::from(codestream_job.num_decomposition_levels),
            use_mct: u32::from(codestream_job.use_mct),
            guard_bits: u32::from(codestream_job.guard_bits),
            progression_order: codestream_progression_order_code(codestream_job.progression_order),
            write_tlm: u32::from(codestream_job.write_tlm),
            high_throughput: u32::from(
                codestream_job.block_coding_mode
                    == J2kLosslessCodestreamBlockCodingMode::HighThroughput,
            ),
            code_block_style: match codestream_job.block_coding_mode {
                J2kLosslessCodestreamBlockCodingMode::Classic => 0,
                J2kLosslessCodestreamBlockCodingMode::HighThroughput => 0x40,
            },
            code_block_width_exp: u32::from(codestream_job.code_block_width_exp),
            code_block_height_exp: u32::from(codestream_job.code_block_height_exp),
            output_capacity: u32::try_from(codestream_capacity).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal codestream assembly capacity exceeds u32".to_string(),
                }
            })?,
        };

        let resident_block_params = J2kResidentPacketBlockParams {
            block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
                message: "Tier-2 Metal resident packet block count exceeds u32".to_string(),
            })?,
            tier1_job_count: u32::try_from(tier1.batch_jobs.len()).map_err(|_| {
                Error::MetalKernel {
                    message: "Tier-2 Metal resident packet Tier-1 job count exceeds u32"
                        .to_string(),
                }
            })?,
        };

        let resolution_buffer = copied_slice_buffer(&runtime.device, &resolutions);
        let subband_buffer = copied_slice_buffer(&runtime.device, &subbands);
        let resident_block_buffer = copied_slice_buffer(&runtime.device, &resident_blocks);
        let packet_block_buffer = runtime.device.new_buffer(
            (resident_blocks.len().max(1) * size_of::<J2kPacketBlock>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let descriptor_buffer = copied_slice_buffer(&runtime.device, &descriptors);
        let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks);
        let output_buffer = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let codestream_buffer = runtime.device.new_buffer(
            codestream_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let header_buffer = runtime.device.new_buffer(
            header_capacity as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let scratch_words = max_tree_nodes
            .checked_mul(6)
            .ok_or_else(|| Error::MetalKernel {
                message: "Tier-2 Metal resident packet scratch size overflow".to_string(),
            })?;
        let scratch_buffer = runtime.device.new_buffer(
            (scratch_words * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kPacketEncodeStatus>() as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let codestream_status_buffer = runtime.device.new_buffer(
            size_of::<J2kCodestreamAssemblyStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        if !resident_blocks.is_empty() {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&runtime.packet_block_prepare_resident_classic);
            encoder.set_buffer(0, Some(&resident_block_buffer), 0);
            encoder.set_buffer(1, Some(&tier1.job_buffer), 0);
            encoder.set_buffer(2, Some(&tier1.status_buffer), 0);
            encoder.set_buffer(3, Some(&packet_block_buffer), 0);
            encoder.set_bytes(
                4,
                size_of::<J2kResidentPacketBlockParams>() as u64,
                (&raw const resident_block_params).cast(),
            );
            encoder.dispatch_threads(
                MTLSize {
                    width: resident_blocks.len() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: runtime
                        .packet_block_prepare_resident_classic
                        .thread_execution_width()
                        .max(1),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.packet_encode);
        encoder.set_buffer(0, Some(&resolution_buffer), 0);
        encoder.set_buffer(1, Some(&subband_buffer), 0);
        encoder.set_buffer(2, Some(&packet_block_buffer), 0);
        encoder.set_buffer(3, Some(&tier1.output_buffer), 0);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_buffer(5, Some(&header_buffer), 0);
        encoder.set_buffer(6, Some(&scratch_buffer), 0);
        encoder.set_bytes(
            7,
            size_of::<J2kPacketEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(8, Some(&status_buffer), 0);
        encoder.set_buffer(9, Some(&descriptor_buffer), 0);
        encoder.set_buffer(10, Some(&state_block_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.lossless_codestream_assemble);
        encoder.set_buffer(0, Some(&output_buffer), 0);
        encoder.set_buffer(1, Some(&status_buffer), 0);
        encoder.set_buffer(2, Some(&codestream_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kLosslessCodestreamAssemblyParams>() as u64,
            (&raw const codestream_params).cast(),
        );
        encoder.set_buffer(4, Some(&codestream_status_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kPendingResidentLosslessCodestream {
            buffer: codestream_buffer,
            capacity: codestream_capacity,
            status_buffer: codestream_status_buffer,
            command_buffer: command_buffer.to_owned(),
            retained_command_buffers: vec![
                tier1.prepare_command_buffer.clone(),
                tier1.tier1_command_buffer.clone(),
            ],
            _retained_buffers: vec![
                resolution_buffer,
                subband_buffer,
                resident_block_buffer,
                packet_block_buffer,
                descriptor_buffer,
                state_block_buffer,
                output_buffer,
                header_buffer,
                scratch_buffer,
                status_buffer,
                tier1.output_buffer.clone(),
                tier1.status_buffer.clone(),
                tier1.job_buffer.clone(),
            ],
            status_stage: "J2K codestream assembly",
            length_error: "J2K Metal codestream output length exceeds usize",
            capacity_error: "J2K Metal codestream output length exceeds buffer",
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_lossless_codestream_buffer_from_resident_ht_tier1(
    session: &crate::MetalBackendSession,
    tier1: &J2kResidentLosslessHtCodeBlocks,
    job: J2kResidentPacketizationEncodeJob<'_>,
    codestream_job: J2kLosslessCodestreamAssemblyJob,
) -> Result<J2kResidentLosslessCodestream, Error> {
    wait_resident_lossless_codestream(submit_lossless_codestream_buffer_from_resident_ht_tier1(
        session,
        tier1,
        job,
        codestream_job,
    )?)
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_lossless_codestream_buffer_from_resident_ht_tier1(
    session: &crate::MetalBackendSession,
    tier1: &J2kResidentLosslessHtCodeBlocks,
    job: J2kResidentPacketizationEncodeJob<'_>,
    codestream_job: J2kLosslessCodestreamAssemblyJob,
) -> Result<J2kPendingResidentLosslessCodestream, Error> {
    with_runtime_for_device(&session.device, |runtime| {
        if tier1.batch_jobs.len() != tier1.code_blocks.len() {
            return Err(Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packetization Tier-1 table mismatch"
                    .to_string(),
            });
        }

        let mut resolutions = Vec::<J2kPacketResolution>::new();
        let mut subbands = Vec::<J2kPacketSubband>::new();
        let mut resident_blocks = Vec::<J2kResidentPacketBlock>::new();
        let mut max_tree_nodes = 1usize;

        for resolution in job.resolutions {
            let subband_offset = u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet subband table exceeds u32".to_string(),
            })?;
            for subband in &resolution.subbands {
                let block_offset =
                    u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Tier-2 Metal resident packet block table exceeds u32"
                            .to_string(),
                    })?;
                max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                    subband.num_cbs_x,
                    subband.num_cbs_y,
                )?);
                let code_block_start =
                    usize::try_from(subband.code_block_start).map_err(|_| Error::MetalKernel {
                        message:
                            "HTJ2K Tier-2 Metal resident packet code-block offset exceeds usize"
                                .to_string(),
                    })?;
                let code_block_count =
                    usize::try_from(subband.code_block_count).map_err(|_| Error::MetalKernel {
                        message:
                            "HTJ2K Tier-2 Metal resident packet code-block count exceeds usize"
                                .to_string(),
                    })?;
                let code_block_end =
                    code_block_start
                        .checked_add(code_block_count)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "HTJ2K Tier-2 Metal resident packet code-block range overflow"
                                .to_string(),
                        })?;
                if code_block_end > tier1.batch_jobs.len() {
                    return Err(Error::MetalKernel {
                        message:
                            "HTJ2K Tier-2 Metal resident packet code-block range out of bounds"
                                .to_string(),
                    });
                }
                for tier1_job_index in code_block_start..code_block_end {
                    resident_blocks.push(J2kResidentPacketBlock {
                        tier1_job_index: u32::try_from(tier1_job_index).map_err(|_| {
                            Error::MetalKernel {
                                message:
                                    "HTJ2K Tier-2 Metal resident packet Tier-1 index exceeds u32"
                                        .to_string(),
                            }
                        })?,
                        previously_included: 0,
                        l_block: 3,
                        block_coding_mode: 1,
                    });
                }
                subbands.push(J2kPacketSubband {
                    block_offset,
                    block_count: subband.code_block_count,
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                });
            }
            resolutions.push(J2kPacketResolution {
                subband_offset,
                subband_count: u32::try_from(resolution.subbands.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Tier-2 Metal resident packet resolution subband count exceeds u32"
                            .to_string(),
                    }
                })?,
            });
        }

        if resolutions.len()
            != usize::try_from(job.resolution_count).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet resolution count exceeds usize"
                    .to_string(),
            })?
        {
            return Err(Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet resolution count mismatch".to_string(),
            });
        }
        if resident_blocks.len()
            != usize::try_from(job.code_block_count).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet code-block count exceeds usize"
                    .to_string(),
            })?
        {
            return Err(Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet code-block count mismatch".to_string(),
            });
        }

        let mut state_block_offsets = HashMap::<u32, (u32, usize)>::new();
        let mut state_blocks = Vec::<J2kPacketStateBlock>::new();
        let mut descriptors =
            Vec::<J2kPacketDescriptor>::with_capacity(job.packet_descriptors.len());
        for descriptor in job.packet_descriptors {
            let packet_index =
                usize::try_from(descriptor.packet_index).map_err(|_| Error::MetalKernel {
                    message:
                        "HTJ2K Tier-2 Metal resident packet descriptor packet index exceeds usize"
                            .to_string(),
                })?;
            let resolution = resolutions
                .get(packet_index)
                .ok_or_else(|| Error::MetalKernel {
                    message:
                        "HTJ2K Tier-2 Metal resident packet descriptor packet index out of range"
                            .to_string(),
                })?;
            let subband_start =
                usize::try_from(resolution.subband_offset).map_err(|_| Error::MetalKernel {
                    message:
                        "HTJ2K Tier-2 Metal resident packet descriptor subband offset exceeds usize"
                            .to_string(),
                })?;
            let subband_count =
                usize::try_from(resolution.subband_count).map_err(|_| Error::MetalKernel {
                    message:
                        "HTJ2K Tier-2 Metal resident packet descriptor subband count exceeds usize"
                            .to_string(),
                })?;
            let subband_end =
                subband_start
                    .checked_add(subband_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message:
                            "HTJ2K Tier-2 Metal resident packet descriptor subband range overflow"
                                .to_string(),
                    })?;
            if subband_end > subbands.len() {
                return Err(Error::MetalKernel {
                    message:
                        "HTJ2K Tier-2 Metal resident packet descriptor subband range out of bounds"
                            .to_string(),
                });
            }
            let mut packet_block_count = 0usize;
            for subband in &subbands[subband_start..subband_end] {
                packet_block_count = packet_block_count
                    .checked_add(usize::try_from(subband.block_count).map_err(|_| {
                        Error::MetalKernel {
                            message: "HTJ2K Tier-2 Metal resident packet descriptor block count exceeds usize"
                                .to_string(),
                        }
                    })?)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Tier-2 Metal resident packet descriptor block count overflow"
                            .to_string(),
                    })?;
            }

            let (state_block_offset, existing_count) = if let Some(&(offset, count)) =
                state_block_offsets.get(&descriptor.state_index)
            {
                (offset, count)
            } else {
                let offset = u32::try_from(state_blocks.len()).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Tier-2 Metal resident packet state block offset exceeds u32"
                        .to_string(),
                })?;
                for subband in &subbands[subband_start..subband_end] {
                    let block_start =
                        usize::try_from(subband.block_offset).map_err(|_| Error::MetalKernel {
                            message: "HTJ2K Tier-2 Metal resident packet state block offset exceeds usize"
                                .to_string(),
                        })?;
                    let block_count =
                        usize::try_from(subband.block_count).map_err(|_| Error::MetalKernel {
                            message:
                                "HTJ2K Tier-2 Metal resident packet state block count exceeds usize"
                                    .to_string(),
                        })?;
                    let block_end =
                        block_start
                            .checked_add(block_count)
                            .ok_or_else(|| Error::MetalKernel {
                                message:
                                    "HTJ2K Tier-2 Metal resident packet state block range overflow"
                                        .to_string(),
                            })?;
                    if block_end > resident_blocks.len() {
                        return Err(Error::MetalKernel {
                            message:
                                "HTJ2K Tier-2 Metal resident packet state block range out of bounds"
                                    .to_string(),
                        });
                    }
                    for block in &resident_blocks[block_start..block_end] {
                        state_blocks.push(J2kPacketStateBlock {
                            previously_included: block.previously_included,
                            l_block: block.l_block,
                        });
                    }
                }
                state_block_offsets.insert(descriptor.state_index, (offset, packet_block_count));
                (offset, packet_block_count)
            };
            if existing_count != packet_block_count {
                return Err(Error::MetalKernel {
                    message: "HTJ2K Tier-2 Metal resident packet descriptor state layout mismatch"
                        .to_string(),
                });
            }

            descriptors.push(J2kPacketDescriptor {
                packet_index: descriptor.packet_index,
                state_index: descriptor.state_index,
                layer: u32::from(descriptor.layer),
                resolution: descriptor.resolution,
                component: u32::from(descriptor.component),
                precinct_lo: descriptor.precinct as u32,
                precinct_hi: (descriptor.precinct >> 32) as u32,
                state_block_offset,
            });
        }

        let header_capacity = resident_blocks
            .len()
            .checked_mul(256)
            .and_then(|bytes| bytes.checked_add(4096))
            .map(|bytes| bytes.max(4096))
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet header capacity overflow".to_string(),
            })?;
        let output_capacity = tier1
            .output_capacity_total
            .checked_add(
                header_capacity
                    .checked_mul(descriptors.len().max(resolutions.len()).max(1))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Tier-2 Metal resident packet output capacity overflow"
                            .to_string(),
                    })?,
            )
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet output capacity overflow".to_string(),
            })?;
        let codestream_capacity =
            lossless_codestream_assembly_capacity(output_capacity, codestream_job)?;

        let params = J2kPacketEncodeParams {
            resolution_count: u32::try_from(resolutions.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet resolution count exceeds u32"
                    .to_string(),
            })?,
            num_layers: u32::from(job.num_layers),
            num_components: u32::from(job.num_components),
            code_block_count: u32::try_from(resident_blocks.len()).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Tier-2 Metal resident packet code-block count exceeds u32"
                        .to_string(),
                }
            })?,
            subband_count: u32::try_from(subbands.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet subband count exceeds u32".to_string(),
            })?,
            descriptor_count: u32::try_from(descriptors.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet descriptor count exceeds u32"
                    .to_string(),
            })?,
            output_capacity: u32::try_from(output_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet output capacity exceeds u32"
                    .to_string(),
            })?,
            header_capacity: u32::try_from(header_capacity).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet header capacity exceeds u32"
                    .to_string(),
            })?,
            scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Tier-2 Metal resident packet scratch node capacity exceeds u32"
                        .to_string(),
                }
            })?,
        };
        let codestream_params = J2kLosslessCodestreamAssemblyParams {
            width: codestream_job.width,
            height: codestream_job.height,
            num_components: u32::from(codestream_job.num_components),
            bit_depth: u32::from(codestream_job.bit_depth),
            signed_samples: u32::from(codestream_job.signed),
            num_decomposition_levels: u32::from(codestream_job.num_decomposition_levels),
            use_mct: u32::from(codestream_job.use_mct),
            guard_bits: u32::from(codestream_job.guard_bits),
            progression_order: codestream_progression_order_code(codestream_job.progression_order),
            write_tlm: u32::from(codestream_job.write_tlm),
            high_throughput: u32::from(
                codestream_job.block_coding_mode
                    == J2kLosslessCodestreamBlockCodingMode::HighThroughput,
            ),
            code_block_style: match codestream_job.block_coding_mode {
                J2kLosslessCodestreamBlockCodingMode::Classic => 0,
                J2kLosslessCodestreamBlockCodingMode::HighThroughput => 0x40,
            },
            code_block_width_exp: u32::from(codestream_job.code_block_width_exp),
            code_block_height_exp: u32::from(codestream_job.code_block_height_exp),
            output_capacity: u32::try_from(codestream_capacity).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Metal codestream assembly capacity exceeds u32".to_string(),
                }
            })?,
        };

        let resident_block_params = J2kResidentPacketBlockParams {
            block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet block count exceeds u32".to_string(),
            })?,
            tier1_job_count: u32::try_from(tier1.batch_jobs.len()).map_err(|_| {
                Error::MetalKernel {
                    message: "HTJ2K Tier-2 Metal resident packet Tier-1 job count exceeds u32"
                        .to_string(),
                }
            })?,
        };

        let resolution_buffer = copied_slice_buffer(&runtime.device, &resolutions);
        let subband_buffer = copied_slice_buffer(&runtime.device, &subbands);
        let resident_block_buffer = copied_slice_buffer(&runtime.device, &resident_blocks);
        let packet_block_buffer = runtime.device.new_buffer(
            (resident_blocks.len().max(1) * size_of::<J2kPacketBlock>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let descriptor_buffer = copied_slice_buffer(&runtime.device, &descriptors);
        let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks);
        let output_buffer = runtime.device.new_buffer(
            output_capacity as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let codestream_buffer = runtime.device.new_buffer(
            codestream_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let header_buffer = runtime.device.new_buffer(
            header_capacity as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let scratch_words = max_tree_nodes
            .checked_mul(6)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Tier-2 Metal resident packet scratch size overflow".to_string(),
            })?;
        let scratch_buffer = runtime.device.new_buffer(
            (scratch_words * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = runtime.device.new_buffer(
            size_of::<J2kPacketEncodeStatus>() as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let codestream_status_buffer = runtime.device.new_buffer(
            size_of::<J2kCodestreamAssemblyStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        if !resident_blocks.is_empty() {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&runtime.packet_block_prepare_resident_ht);
            encoder.set_buffer(0, Some(&resident_block_buffer), 0);
            encoder.set_buffer(1, Some(&tier1.job_buffer), 0);
            encoder.set_buffer(2, Some(&tier1.status_buffer), 0);
            encoder.set_buffer(3, Some(&packet_block_buffer), 0);
            encoder.set_bytes(
                4,
                size_of::<J2kResidentPacketBlockParams>() as u64,
                (&raw const resident_block_params).cast(),
            );
            encoder.dispatch_threads(
                MTLSize {
                    width: resident_blocks.len() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: runtime
                        .packet_block_prepare_resident_ht
                        .thread_execution_width()
                        .max(1),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
        }

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.packet_encode);
        encoder.set_buffer(0, Some(&resolution_buffer), 0);
        encoder.set_buffer(1, Some(&subband_buffer), 0);
        encoder.set_buffer(2, Some(&packet_block_buffer), 0);
        encoder.set_buffer(3, Some(&tier1.output_buffer), 0);
        encoder.set_buffer(4, Some(&output_buffer), 0);
        encoder.set_buffer(5, Some(&header_buffer), 0);
        encoder.set_buffer(6, Some(&scratch_buffer), 0);
        encoder.set_bytes(
            7,
            size_of::<J2kPacketEncodeParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(8, Some(&status_buffer), 0);
        encoder.set_buffer(9, Some(&descriptor_buffer), 0);
        encoder.set_buffer(10, Some(&state_block_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.lossless_codestream_assemble);
        encoder.set_buffer(0, Some(&output_buffer), 0);
        encoder.set_buffer(1, Some(&status_buffer), 0);
        encoder.set_buffer(2, Some(&codestream_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kLosslessCodestreamAssemblyParams>() as u64,
            (&raw const codestream_params).cast(),
        );
        encoder.set_buffer(4, Some(&codestream_status_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();

        Ok(J2kPendingResidentLosslessCodestream {
            buffer: codestream_buffer,
            capacity: codestream_capacity,
            status_buffer: codestream_status_buffer,
            command_buffer: command_buffer.to_owned(),
            retained_command_buffers: vec![
                tier1.prepare_command_buffer.clone(),
                tier1.tier1_command_buffer.clone(),
            ],
            _retained_buffers: vec![
                resolution_buffer,
                subband_buffer,
                resident_block_buffer,
                packet_block_buffer,
                descriptor_buffer,
                state_block_buffer,
                output_buffer,
                header_buffer,
                scratch_buffer,
                status_buffer,
                tier1.output_buffer.clone(),
                tier1.status_buffer.clone(),
                tier1.job_buffer.clone(),
            ],
            status_stage: "HTJ2K codestream assembly",
            length_error: "HTJ2K Metal codestream output length exceeds usize",
            capacity_error: "HTJ2K Metal codestream output length exceeds buffer",
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_lossless_codestream_buffers_from_prepared_ht_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kResidentHtBatchEncodeItem>,
    packet_capacity_mode: J2kHtPacketOutputCapacityMode,
) -> Result<J2kPendingResidentLosslessCodestreamBatch, Error> {
    if items.is_empty() {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal resident batch encode requires at least one tile".to_string(),
        });
    }

    let mut prepared_tiles = Vec::with_capacity(items.len());
    for item in items {
        let J2kPreparedLosslessDeviceCodeBlocks {
            coefficient_buffer,
            coefficient_byte_offset,
            coefficient_byte_len,
            coefficient_buffer_is_batch_shared,
            code_blocks,
            recyclable_private_buffers,
            _prepare_command_buffer: prepare_command_buffer,
            _prepare_deinterleave_rct_command_buffer: prepare_deinterleave_rct_command_buffer,
            _prepare_dwt53_command_buffer: prepare_dwt53_command_buffer,
            _prepare_dwt53_vertical_command_buffers: prepare_dwt53_vertical_command_buffers,
            _prepare_dwt53_horizontal_command_buffers: prepare_dwt53_horizontal_command_buffers,
            _prepare_coefficient_extract_command_buffer: prepare_coefficient_extract_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        } = item.prepared;
        prepared_tiles.push(PreparedLosslessBatchTile {
            coefficient_buffer,
            coefficient_byte_offset,
            coefficient_byte_len,
            coefficient_buffer_is_batch_shared,
            code_blocks,
            recyclable_private_buffers,
            prepare_command_buffer,
            prepare_deinterleave_rct_command_buffer,
            prepare_dwt53_command_buffer,
            prepare_dwt53_vertical_command_buffers,
            prepare_dwt53_horizontal_command_buffers,
            prepare_coefficient_extract_command_buffer,
            deinterleave_status_buffer,
            plane_buffers,
            scratch_buffers,
            coefficient_job_buffer,
            resolution_count: item.resolution_count,
            num_layers: item.num_layers,
            num_components: item.num_components,
            code_block_count: item.code_block_count,
            packet_descriptors: item.packet_descriptors,
            resolutions: item.resolutions,
            codestream: item.codestream,
        });
    }

    with_runtime_for_device(&session.device, |runtime| {
        let profile_stages = metal_profile_stages_enabled();
        let mut stage_stats = J2kResidentEncodeStageStats::default();
        let mut ht_table_build_duration = Duration::ZERO;
        let mut ht_block_encode_duration = Duration::ZERO;
        let mut packet_block_prep_duration = Duration::ZERO;
        let mut packetization_duration = Duration::ZERO;
        let mut codestream_assembly_duration = Duration::ZERO;
        let mut ht_table_build_started = profile_stages.then(Instant::now);
        let ht_tier1_setup_signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP);
        let split_profile_commands = true;
        let mut retained_command_buffers = Vec::with_capacity(prepared_tiles.len());
        let mut gpu_stage_command_buffers = Vec::new();
        let mut retained_buffers = Vec::<Buffer>::new();
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let mut recyclable_shared_buffers = Vec::<(usize, Buffer)>::new();
        let shared_coefficient_buffer = prepared_tiles.first().and_then(|first| {
            let ptr = first.coefficient_buffer.as_ptr();
            prepared_tiles
                .iter()
                .all(|tile| {
                    tile.coefficient_buffer_is_batch_shared
                        && tile.coefficient_buffer.as_ptr() == ptr
                })
                .then(|| first.coefficient_buffer.clone())
        });
        let needs_coefficient_copy = shared_coefficient_buffer.is_none();
        let initial_command_buffer_label = if split_profile_commands && needs_coefficient_copy {
            "signinum-j2k htj2k resident coefficient copy"
        } else if split_profile_commands {
            "signinum-j2k htj2k resident tier1 encode"
        } else {
            "signinum-j2k htj2k resident encode batch"
        };
        let mut command_buffer =
            new_resident_encode_command_buffer(runtime, initial_command_buffer_label);
        let (coefficient_buffer, coefficient_offsets) = if let Some(coefficient_buffer) =
            shared_coefficient_buffer
        {
            (
                coefficient_buffer,
                prepared_tiles
                    .iter()
                    .map(|tile| tile.coefficient_byte_offset)
                    .collect::<Vec<_>>(),
            )
        } else {
            let mut coefficient_offsets = Vec::<usize>::with_capacity(prepared_tiles.len());
            let mut total_coefficient_bytes = 0usize;
            for tile in &prepared_tiles {
                coefficient_offsets.push(total_coefficient_bytes);
                total_coefficient_bytes = total_coefficient_bytes
                    .checked_add(tile.coefficient_byte_len)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch coefficient buffer size overflow".to_string(),
                    })?;
            }
            let coefficient_buffer = take_recyclable_private_buffer(
                runtime,
                total_coefficient_bytes.max(1),
                &mut recyclable_private_buffers,
            );
            let blit = command_buffer.new_blit_command_encoder();
            if metal_profile_stages_enabled() {
                blit.set_label("HTJ2K coefficient prep");
            }
            for (tile, &dst_offset) in prepared_tiles.iter().zip(coefficient_offsets.iter()) {
                if tile.coefficient_byte_len > 0 {
                    #[cfg(test)]
                    HT_BATCH_COEFFICIENT_COPY_BLITS.fetch_add(1, Ordering::Relaxed);
                    blit.copy_from_buffer(
                        &tile.coefficient_buffer,
                        tile.coefficient_byte_offset as u64,
                        &coefficient_buffer,
                        dst_offset as u64,
                        tile.coefficient_byte_len as u64,
                    );
                }
            }
            blit.end_encoding();
            if split_profile_commands {
                command_buffer = finish_resident_encode_split_command_buffer(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::CoefficientCopy,
                    "signinum-j2k htj2k resident tier1 encode",
                    &mut gpu_stage_command_buffers,
                );
            }
            (coefficient_buffer, coefficient_offsets)
        };

        let mut tier1_jobs = Vec::<J2kHtEncodeBatchJob>::new();
        let mut tier1_output_capacity_total = 0usize;
        let mut max_tier1_output_capacity = 0usize;
        let mut tile_tier1_job_bases = Vec::<usize>::with_capacity(prepared_tiles.len());
        for (tile, &coefficient_byte_offset) in
            prepared_tiles.iter().zip(coefficient_offsets.iter())
        {
            tile_tier1_job_bases.push(tier1_jobs.len());
            let coefficient_word_offset = coefficient_byte_offset
                .checked_div(size_of::<i32>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch coefficient offset division failed".to_string(),
                })?;
            let coefficient_word_offset_u32 =
                u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal batch coefficient offset exceeds u32".to_string(),
                })?;
            for block in &tile.code_blocks {
                let output_capacity_per_job = ht_encode_output_capacity(block.width, block.height)?;
                max_tier1_output_capacity = max_tier1_output_capacity.max(output_capacity_per_job);
                let output_capacity_per_job_u32 =
                    u32::try_from(output_capacity_per_job).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal batch output capacity exceeds u32".to_string(),
                    })?;
                let output_offset =
                    u32::try_from(tier1_output_capacity_total).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal batch Tier-1 output offset exceeds u32".to_string(),
                    })?;
                let coefficient_offset = block
                    .coefficient_offset
                    .checked_add(coefficient_word_offset_u32)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch coefficient offset overflow".to_string(),
                    })?;
                tier1_jobs.push(J2kHtEncodeBatchJob {
                    coefficient_offset,
                    output_offset,
                    width: block.width,
                    height: block.height,
                    total_bitplanes: u32::from(block.total_bitplanes),
                    output_capacity: output_capacity_per_job_u32,
                });
                tier1_output_capacity_total = tier1_output_capacity_total
                    .checked_add(output_capacity_per_job)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch Tier-1 output buffer overflow".to_string(),
                    })?;
            }
        }

        let tier1_job_buffer = owned_slice_buffer(&runtime.device, &tier1_jobs);
        let tier1_output_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_output_capacity_total.max(1),
            &mut recyclable_private_buffers,
        );
        let tier1_status_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_jobs.len().max(1) * size_of::<J2kHtEncodeStatus>(),
            &mut recyclable_private_buffers,
        );
        let tier1_job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal batch Tier-1 job count exceeds u32".to_string(),
        })?;
        drop(ht_tier1_setup_signpost);
        if let Some(started) = ht_table_build_started.take() {
            ht_table_build_duration = ht_table_build_duration.saturating_add(started.elapsed());
        }
        if tier1_job_count > 0 {
            let command_encode_started = profile_stages.then(Instant::now);
            let signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE);
            let encoder = command_buffer.new_compute_command_encoder();
            label_compute_encoder(encoder, "HTJ2K Tier-1 encode");
            let pipeline = &runtime.ht_encode_code_blocks;
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&coefficient_buffer), 0);
            encoder.set_buffer(1, Some(&tier1_output_buffer), 0);
            encoder.set_buffer(2, Some(&tier1_job_buffer), 0);
            encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
            encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
            encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
            encoder.set_buffer(6, Some(&tier1_status_buffer), 0);
            encoder.set_bytes(
                7,
                size_of::<u32>() as u64,
                (&raw const tier1_job_count).cast(),
            );
            dispatch_1d_pipeline(encoder, pipeline, u64::from(tier1_job_count));
            encoder.end_encoding();
            drop(signpost);
            if let Some(started) = command_encode_started {
                ht_block_encode_duration =
                    ht_block_encode_duration.saturating_add(started.elapsed());
            }
            if split_profile_commands {
                command_buffer = finish_resident_encode_split_command_buffer(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::HtBlock,
                    "signinum-j2k htj2k resident packetization",
                    &mut gpu_stage_command_buffers,
                );
            }
        } else if split_profile_commands {
            label_command_buffer(&command_buffer, "signinum-j2k htj2k resident packetization");
        }

        ht_table_build_started = profile_stages.then(Instant::now);
        let ht_packet_plan_signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN);
        let mut packet_resolutions = Vec::<J2kPacketResolution>::new();
        let mut packet_subbands = Vec::<J2kPacketSubband>::new();
        let mut resident_blocks = Vec::<J2kResidentPacketBlock>::new();
        let mut packet_descriptors = Vec::<J2kPacketDescriptor>::new();
        let mut state_blocks = Vec::<J2kPacketStateBlock>::new();
        let mut packet_jobs = Vec::<J2kBatchedPacketEncodeJob>::with_capacity(prepared_tiles.len());
        let mut assembly_jobs =
            Vec::<J2kBatchedCodestreamAssemblyJob>::with_capacity(prepared_tiles.len());
        let mut packet_output_capacity_total = 0usize;
        let mut packet_payload_copy_job_capacity_total = 0usize;
        let mut max_payload_copy_jobs_per_tile = 0usize;
        let mut header_capacity_total = 0usize;
        let mut scratch_words_total = 0usize;
        let mut codestream_capacity_total = 0usize;
        let mut codestream_offsets = Vec::<usize>::with_capacity(prepared_tiles.len());
        let mut codestream_capacities = Vec::<usize>::with_capacity(prepared_tiles.len());

        for (tile_index, tile) in prepared_tiles.iter().enumerate() {
            let local_resolution_offset = packet_resolutions.len();
            let local_subband_offset = packet_subbands.len();
            let local_block_offset = resident_blocks.len();
            let local_descriptor_offset = packet_descriptors.len();
            let local_state_block_offset = state_blocks.len();
            let tier1_job_base = tile_tier1_job_bases[tile_index];
            let mut max_tree_nodes = 1usize;
            let mut local_subband_count = 0usize;
            let mut local_resident_block_count = 0usize;
            let mut local_payload_copy_job_capacity = 0usize;

            for resolution in &tile.resolutions {
                let subband_offset =
                    u32::try_from(local_subband_count).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal batch packet subband offset exceeds u32".to_string(),
                    })?;
                for subband in &resolution.subbands {
                    let block_offset = u32::try_from(local_resident_block_count).map_err(|_| {
                        Error::MetalKernel {
                            message: "HTJ2K Metal batch packet block offset exceeds u32"
                                .to_string(),
                        }
                    })?;
                    max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                    )?);
                    let code_block_start =
                        usize::try_from(subband.code_block_start).map_err(|_| {
                            Error::MetalKernel {
                                message: "HTJ2K Metal batch packet code-block offset exceeds usize"
                                    .to_string(),
                            }
                        })?;
                    let code_block_count =
                        usize::try_from(subband.code_block_count).map_err(|_| {
                            Error::MetalKernel {
                                message: "HTJ2K Metal batch packet code-block count exceeds usize"
                                    .to_string(),
                            }
                        })?;
                    let code_block_end = code_block_start
                        .checked_add(code_block_count)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "HTJ2K Metal batch packet code-block range overflow"
                                .to_string(),
                        })?;
                    if code_block_end > tile.code_blocks.len() {
                        return Err(Error::MetalKernel {
                            message: "HTJ2K Metal batch packet code-block range out of bounds"
                                .to_string(),
                        });
                    }
                    for tier1_job_index in code_block_start..code_block_end {
                        resident_blocks.push(J2kResidentPacketBlock {
                            tier1_job_index: u32::try_from(
                                tier1_job_base.checked_add(tier1_job_index).ok_or_else(|| {
                                    Error::MetalKernel {
                                        message: "HTJ2K Metal batch Tier-1 index overflow"
                                            .to_string(),
                                    }
                                })?,
                            )
                            .map_err(|_| Error::MetalKernel {
                                message: "HTJ2K Metal batch Tier-1 index exceeds u32".to_string(),
                            })?,
                            previously_included: 0,
                            l_block: 3,
                            block_coding_mode: 1,
                        });
                    }
                    packet_subbands.push(J2kPacketSubband {
                        block_offset,
                        block_count: subband.code_block_count,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    });
                    local_subband_count =
                        local_subband_count
                            .checked_add(1)
                            .ok_or_else(|| Error::MetalKernel {
                                message: "HTJ2K Metal batch subband count overflow".to_string(),
                            })?;
                    local_resident_block_count = local_resident_block_count
                        .checked_add(code_block_count)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "HTJ2K Metal batch resident block count overflow".to_string(),
                        })?;
                }
                packet_resolutions.push(J2kPacketResolution {
                    subband_offset,
                    subband_count: u32::try_from(resolution.subbands.len()).map_err(|_| {
                        Error::MetalKernel {
                            message: "HTJ2K Metal batch resolution subband count exceeds u32"
                                .to_string(),
                        }
                    })?,
                });
            }

            if tile.resolutions.len()
                != usize::try_from(tile.resolution_count).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal batch resolution count exceeds usize".to_string(),
                })?
            {
                return Err(Error::MetalKernel {
                    message: "HTJ2K Metal batch resolution count mismatch".to_string(),
                });
            }
            if local_resident_block_count
                != usize::try_from(tile.code_block_count).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal batch code-block count exceeds usize".to_string(),
                })?
            {
                return Err(Error::MetalKernel {
                    message: "HTJ2K Metal batch code-block count mismatch".to_string(),
                });
            }

            let mut state_block_offsets = HashMap::<u32, (u32, usize)>::new();
            for descriptor in &tile.packet_descriptors {
                let packet_index =
                    usize::try_from(descriptor.packet_index).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal batch descriptor packet index exceeds usize"
                            .to_string(),
                    })?;
                let resolution = packet_resolutions
                    .get(local_resolution_offset + packet_index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch descriptor packet index out of range"
                            .to_string(),
                    })?;
                let subband_start =
                    usize::try_from(resolution.subband_offset).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal batch descriptor subband offset exceeds usize"
                            .to_string(),
                    })?;
                let subband_count =
                    usize::try_from(resolution.subband_count).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal batch descriptor subband count exceeds usize"
                            .to_string(),
                    })?;
                let mut packet_block_count = 0usize;
                for subband in &packet_subbands[local_subband_offset + subband_start
                    ..local_subband_offset + subband_start + subband_count]
                {
                    packet_block_count = packet_block_count
                        .checked_add(usize::try_from(subband.block_count).map_err(|_| {
                            Error::MetalKernel {
                                message: "HTJ2K Metal batch descriptor block count exceeds usize"
                                    .to_string(),
                            }
                        })?)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "HTJ2K Metal batch descriptor block count overflow"
                                .to_string(),
                        })?;
                }
                let (state_block_offset, existing_count) = if let Some(&(offset, count)) =
                    state_block_offsets.get(&descriptor.state_index)
                {
                    (offset, count)
                } else {
                    let offset = u32::try_from(state_blocks.len() - local_state_block_offset)
                        .map_err(|_| Error::MetalKernel {
                            message: "HTJ2K Metal batch state block offset exceeds u32".to_string(),
                        })?;
                    for subband in &packet_subbands[local_subband_offset + subband_start
                        ..local_subband_offset + subband_start + subband_count]
                    {
                        for idx in 0..subband.block_count {
                            let _ = idx;
                            state_blocks.push(J2kPacketStateBlock {
                                previously_included: 0,
                                l_block: 3,
                            });
                        }
                    }
                    state_block_offsets
                        .insert(descriptor.state_index, (offset, packet_block_count));
                    (offset, packet_block_count)
                };
                if existing_count != packet_block_count {
                    return Err(Error::MetalKernel {
                        message: "HTJ2K Metal batch descriptor state layout mismatch".to_string(),
                    });
                }
                local_payload_copy_job_capacity = local_payload_copy_job_capacity
                    .checked_add(packet_block_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch packet payload-copy job count overflow"
                            .to_string(),
                    })?;
                packet_descriptors.push(J2kPacketDescriptor {
                    packet_index: descriptor.packet_index,
                    state_index: descriptor.state_index,
                    layer: u32::from(descriptor.layer),
                    resolution: descriptor.resolution,
                    component: u32::from(descriptor.component),
                    precinct_lo: descriptor.precinct as u32,
                    precinct_hi: (descriptor.precinct >> 32) as u32,
                    state_block_offset,
                });
            }

            let header_capacity = local_resident_block_count
                .checked_mul(256)
                .and_then(|bytes| bytes.checked_add(4096))
                .map(|bytes| bytes.max(4096))
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch packet header capacity overflow".to_string(),
                })?;
            let packet_output_capacity = ht_packet_output_capacity_for_mode(
                tile.code_blocks.len(),
                header_capacity,
                tile.packet_descriptors.len().max(tile.resolutions.len()),
                tile.codestream,
                packet_capacity_mode,
            )?;
            let codestream_capacity =
                lossless_codestream_assembly_capacity(packet_output_capacity, tile.codestream)?;
            let codestream_payload_offset = lossless_codestream_payload_offset(tile.codestream)?;
            let scratch_words =
                max_tree_nodes
                    .checked_mul(6)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch scratch size overflow".to_string(),
                    })?;

            let header_offset = header_capacity_total;
            let scratch_offset = scratch_words_total;
            if tile.packet_descriptors.is_empty() {
                local_payload_copy_job_capacity = local_resident_block_count;
            }
            let payload_copy_offset = packet_payload_copy_job_capacity_total;
            let codestream_offset = codestream_capacity_total;
            let packet_output_offset = codestream_offset
                .checked_add(codestream_payload_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch direct packet output offset overflow".to_string(),
                })?;
            packet_jobs.push(J2kBatchedPacketEncodeJob {
                resolution_offset: u32::try_from(local_resolution_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch resolution offset exceeds u32".to_string(),
                    }
                })?,
                subband_offset: u32::try_from(local_subband_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch subband offset exceeds u32".to_string(),
                    }
                })?,
                block_offset: u32::try_from(local_block_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch block offset exceeds u32".to_string(),
                    }
                })?,
                descriptor_offset: u32::try_from(local_descriptor_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch descriptor offset exceeds u32".to_string(),
                    }
                })?,
                state_block_offset: u32::try_from(local_state_block_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch state block offset exceeds u32".to_string(),
                    }
                })?,
                output_offset: u32::try_from(packet_output_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch packet output offset exceeds u32".to_string(),
                    }
                })?,
                header_offset: u32::try_from(header_offset).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal batch header offset exceeds u32".to_string(),
                })?,
                scratch_offset: u32::try_from(scratch_offset).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal batch scratch offset exceeds u32".to_string(),
                })?,
                payload_copy_offset: u32::try_from(payload_copy_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch packet payload-copy offset exceeds u32"
                            .to_string(),
                    }
                })?,
                payload_copy_capacity: u32::try_from(local_payload_copy_job_capacity).map_err(
                    |_| Error::MetalKernel {
                        message: "HTJ2K Metal batch packet payload-copy capacity exceeds u32"
                            .to_string(),
                    },
                )?,
                resolution_count: tile.resolution_count,
                num_layers: u32::from(tile.num_layers),
                num_components: u32::from(tile.num_components),
                code_block_count: tile.code_block_count,
                subband_count: u32::try_from(local_subband_count).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch local subband count exceeds u32".to_string(),
                    }
                })?,
                descriptor_count: u32::try_from(tile.packet_descriptors.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch descriptor count exceeds u32".to_string(),
                    }
                })?,
                output_capacity: u32::try_from(packet_output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch packet output capacity exceeds u32".to_string(),
                    }
                })?,
                header_capacity: u32::try_from(header_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch header capacity exceeds u32".to_string(),
                    }
                })?,
                scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch scratch node capacity exceeds u32".to_string(),
                    }
                })?,
            });
            assembly_jobs.push(J2kBatchedCodestreamAssemblyJob {
                tile_data_offset: u32::try_from(packet_output_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch assembly packet offset exceeds u32".to_string(),
                    }
                })?,
                codestream_offset: u32::try_from(codestream_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch codestream offset exceeds u32".to_string(),
                    }
                })?,
                width: tile.codestream.width,
                height: tile.codestream.height,
                num_components: u32::from(tile.codestream.num_components),
                bit_depth: u32::from(tile.codestream.bit_depth),
                signed_samples: u32::from(tile.codestream.signed),
                num_decomposition_levels: u32::from(tile.codestream.num_decomposition_levels),
                use_mct: u32::from(tile.codestream.use_mct),
                guard_bits: u32::from(tile.codestream.guard_bits),
                progression_order: codestream_progression_order_code(
                    tile.codestream.progression_order,
                ),
                write_tlm: u32::from(tile.codestream.write_tlm),
                high_throughput: 1,
                code_block_style: 0x40,
                code_block_width_exp: u32::from(tile.codestream.code_block_width_exp),
                code_block_height_exp: u32::from(tile.codestream.code_block_height_exp),
                output_capacity: u32::try_from(codestream_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal batch codestream capacity exceeds u32".to_string(),
                    }
                })?,
            });
            codestream_offsets.push(codestream_offset);
            codestream_capacities.push(codestream_capacity);
            packet_output_capacity_total = packet_output_capacity_total
                .checked_add(packet_output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch packet output total overflow".to_string(),
                })?;
            packet_payload_copy_job_capacity_total = packet_payload_copy_job_capacity_total
                .checked_add(local_payload_copy_job_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch packet payload-copy job total overflow".to_string(),
                })?;
            max_payload_copy_jobs_per_tile =
                max_payload_copy_jobs_per_tile.max(local_payload_copy_job_capacity);
            header_capacity_total = header_capacity_total
                .checked_add(header_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch header total overflow".to_string(),
                })?;
            scratch_words_total =
                scratch_words_total
                    .checked_add(scratch_words)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch scratch total overflow".to_string(),
                    })?;
            codestream_capacity_total = codestream_capacity_total
                .checked_add(codestream_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch codestream total overflow".to_string(),
                })?;
        }

        drop(ht_packet_plan_signpost);
        if let Some(started) = ht_table_build_started.take() {
            ht_table_build_duration = ht_table_build_duration.saturating_add(started.elapsed());
        }
        let ht_buffer_allocation_started = profile_stages.then(Instant::now);
        let ht_packet_buffer_setup_signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP);
        let packet_resolution_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &packet_resolutions,
            &mut recyclable_shared_buffers,
        );
        let packet_subband_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &packet_subbands,
            &mut recyclable_shared_buffers,
        );
        let resident_block_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &resident_blocks,
            &mut recyclable_shared_buffers,
        );
        let packet_block_buffer = take_recyclable_private_buffer(
            runtime,
            resident_blocks.len().max(1) * size_of::<J2kPacketBlock>(),
            &mut recyclable_private_buffers,
        );
        let packet_descriptor_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &packet_descriptors,
            &mut recyclable_shared_buffers,
        );
        let state_block_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &state_blocks,
            &mut recyclable_shared_buffers,
        );
        let packet_payload_copy_job_buffer = take_recyclable_private_buffer(
            runtime,
            packet_payload_copy_job_capacity_total
                .max(1)
                .checked_mul(size_of::<J2kPacketPayloadCopyJob>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch packet payload-copy buffer size overflow"
                        .to_string(),
                })?,
            &mut recyclable_private_buffers,
        );
        let header_buffer = take_recyclable_private_buffer(
            runtime,
            header_capacity_total.max(1),
            &mut recyclable_private_buffers,
        );
        let scratch_buffer = take_recyclable_private_buffer(
            runtime,
            scratch_words_total.max(1) * size_of::<u32>(),
            &mut recyclable_private_buffers,
        );
        let packet_job_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &packet_jobs,
            &mut recyclable_shared_buffers,
        );
        let packet_status_buffer = zeroed_recyclable_shared_buffer(
            runtime,
            packet_jobs.len().max(1) * size_of::<J2kPacketEncodeStatus>(),
            &mut recyclable_shared_buffers,
        );
        let codestream_job_buffer = copied_recyclable_shared_slice_buffer(
            runtime,
            &assembly_jobs,
            &mut recyclable_shared_buffers,
        );
        let codestream_buffer = runtime.device.new_buffer(
            codestream_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let codestream_status_buffer = zeroed_recyclable_shared_buffer(
            runtime,
            assembly_jobs.len() * size_of::<J2kCodestreamAssemblyStatus>(),
            &mut recyclable_shared_buffers,
        );
        drop(ht_packet_buffer_setup_signpost);
        if let Some(started) = ht_buffer_allocation_started {
            stage_stats.ht_buffer_allocation_duration = started.elapsed();
        }

        let resident_block_params = J2kResidentPacketBlockParams {
            block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batch resident block count exceeds u32".to_string(),
            })?,
            tier1_job_count,
        };

        let tile_count = u64::try_from(packet_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal batch tile count exceeds u64".to_string(),
        })?;
        if !resident_blocks.is_empty() {
            let command_encode_started = profile_stages.then(Instant::now);
            let signpost =
                hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE);
            let encoder = command_buffer.new_compute_command_encoder();
            label_compute_encoder(encoder, "HTJ2K packet block prep");
            encoder.set_compute_pipeline_state(&runtime.packet_block_prepare_resident_ht);
            encoder.set_buffer(0, Some(&resident_block_buffer), 0);
            encoder.set_buffer(1, Some(&tier1_job_buffer), 0);
            encoder.set_buffer(2, Some(&tier1_status_buffer), 0);
            encoder.set_buffer(3, Some(&packet_block_buffer), 0);
            encoder.set_bytes(
                4,
                size_of::<J2kResidentPacketBlockParams>() as u64,
                (&raw const resident_block_params).cast(),
            );
            encoder.dispatch_threads(
                MTLSize {
                    width: resident_blocks.len() as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: runtime
                        .packet_block_prepare_resident_ht
                        .thread_execution_width()
                        .max(1),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            drop(signpost);
            if let Some(started) = command_encode_started {
                packet_block_prep_duration =
                    packet_block_prep_duration.saturating_add(started.elapsed());
            }
            if split_profile_commands {
                command_buffer = finish_resident_encode_split_command_buffer(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::PacketBlockPrep,
                    "signinum-j2k htj2k resident packetization",
                    &mut gpu_stage_command_buffers,
                );
            }
        } else if split_profile_commands {
            label_command_buffer(&command_buffer, "signinum-j2k htj2k resident packetization");
        }
        let command_encode_started = profile_stages.then(Instant::now);
        let signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE);
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "HTJ2K packetization");
        encoder.set_compute_pipeline_state(&runtime.packet_encode_batched);
        encoder.set_buffer(0, Some(&packet_resolution_buffer), 0);
        encoder.set_buffer(1, Some(&packet_subband_buffer), 0);
        encoder.set_buffer(2, Some(&packet_block_buffer), 0);
        encoder.set_buffer(3, Some(&tier1_output_buffer), 0);
        encoder.set_buffer(4, Some(&codestream_buffer), 0);
        encoder.set_buffer(5, Some(&header_buffer), 0);
        encoder.set_buffer(6, Some(&scratch_buffer), 0);
        encoder.set_buffer(7, Some(&packet_job_buffer), 0);
        encoder.set_buffer(8, Some(&packet_status_buffer), 0);
        encoder.set_buffer(9, Some(&packet_descriptor_buffer), 0);
        encoder.set_buffer(10, Some(&state_block_buffer), 0);
        encoder.set_buffer(11, Some(&packet_payload_copy_job_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: tile_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .packet_encode_batched
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        drop(signpost);
        if let Some(started) = command_encode_started {
            packetization_duration = packetization_duration.saturating_add(started.elapsed());
        }
        if split_profile_commands {
            command_buffer = finish_resident_encode_split_command_buffer(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::Packetization,
                "signinum-j2k htj2k resident packet payload copy",
                &mut gpu_stage_command_buffers,
            );
        }
        let packet_payload_copy_dispatched = dispatch_batched_packet_payload_copy(
            runtime,
            &command_buffer,
            J2kBatchedPacketPayloadCopyDispatch {
                payload_buffer: &tier1_output_buffer,
                packet_output_buffer: &codestream_buffer,
                packet_job_buffer: &packet_job_buffer,
                packet_status_buffer: &packet_status_buffer,
                packet_payload_copy_job_buffer: &packet_payload_copy_job_buffer,
                tile_count,
                max_payload_copy_jobs_per_tile: max_payload_copy_jobs_per_tile as u64,
                label: "HTJ2K packetization payload copy",
                signpost_name: SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
            },
        );
        if split_profile_commands {
            if packet_payload_copy_dispatched {
                command_buffer = finish_resident_encode_split_command_buffer(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::PacketPayloadCopy,
                    "signinum-j2k htj2k resident codestream assembly",
                    &mut gpu_stage_command_buffers,
                );
            } else {
                label_command_buffer(
                    &command_buffer,
                    "signinum-j2k htj2k resident codestream assembly",
                );
            }
        }

        let command_encode_started = profile_stages.then(Instant::now);
        let signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE);
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "HTJ2K codestream assembly");
        encoder.set_compute_pipeline_state(&runtime.lossless_codestream_assemble_batched);
        encoder.set_buffer(0, Some(&codestream_buffer), 0);
        encoder.set_buffer(1, Some(&packet_status_buffer), 0);
        encoder.set_buffer(2, Some(&codestream_buffer), 0);
        encoder.set_buffer(3, Some(&codestream_job_buffer), 0);
        encoder.set_buffer(4, Some(&codestream_status_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: tile_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .lossless_codestream_assemble_batched
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        drop(signpost);
        let max_packet_output_capacity = packet_jobs
            .iter()
            .map(|job| job.output_capacity)
            .max()
            .unwrap_or(0);
        let max_packet_output_capacity_usize = usize::try_from(max_packet_output_capacity)
            .map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batch max packet output capacity exceeds usize".to_string(),
            })?;
        if split_profile_commands {
            command_buffer = finish_resident_encode_split_command_buffer(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::CodestreamAssembly,
                "signinum-j2k htj2k resident result readback",
                &mut gpu_stage_command_buffers,
            );
        }
        let codestream_payload_copy_dispatched = false;
        if let Some(started) = command_encode_started {
            codestream_assembly_duration =
                codestream_assembly_duration.saturating_add(started.elapsed());
        }
        let tier1_status_readback = schedule_resident_tier1_status_readback(
            runtime,
            &command_buffer,
            &tier1_status_buffer,
            J2kResidentTier1StatusKind::HighThroughput,
            0,
            None,
            tier1_jobs.len(),
            size_of::<J2kHtEncodeStatus>(),
            profile_stages,
        )?;
        command_buffer.commit();
        if split_profile_commands && codestream_payload_copy_dispatched {
            gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                stage: J2kResidentEncodeGpuStage::CodestreamPayloadCopy,
                command_buffer: command_buffer.clone(),
            });
        }

        if profile_stages {
            let mut prepare_command_buffer_ptrs = Vec::new();
            for tile in &prepared_tiles {
                let mut pushed_split_prepare = false;
                for (stage, command_buffer) in [
                    (
                        J2kResidentEncodeGpuStage::CoefficientDeinterleaveRct,
                        tile.prepare_deinterleave_rct_command_buffer.as_ref(),
                    ),
                    (
                        J2kResidentEncodeGpuStage::CoefficientDwt53,
                        tile.prepare_dwt53_command_buffer.as_ref(),
                    ),
                    (
                        J2kResidentEncodeGpuStage::CoefficientExtract,
                        tile.prepare_coefficient_extract_command_buffer.as_ref(),
                    ),
                ] {
                    if let Some(command_buffer) = command_buffer {
                        let ptr = command_buffer.as_ptr();
                        if prepare_command_buffer_ptrs.contains(&ptr) {
                            continue;
                        }
                        prepare_command_buffer_ptrs.push(ptr);
                        pushed_split_prepare = true;
                        gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                            stage,
                            command_buffer: command_buffer.clone(),
                        });
                    }
                }
                for command_buffer in &tile.prepare_dwt53_vertical_command_buffers {
                    let ptr = command_buffer.as_ptr();
                    if prepare_command_buffer_ptrs.contains(&ptr) {
                        continue;
                    }
                    prepare_command_buffer_ptrs.push(ptr);
                    pushed_split_prepare = true;
                    gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                        stage: J2kResidentEncodeGpuStage::CoefficientDwt53Vertical,
                        command_buffer: command_buffer.clone(),
                    });
                }
                for command_buffer in &tile.prepare_dwt53_horizontal_command_buffers {
                    let ptr = command_buffer.as_ptr();
                    if prepare_command_buffer_ptrs.contains(&ptr) {
                        continue;
                    }
                    prepare_command_buffer_ptrs.push(ptr);
                    pushed_split_prepare = true;
                    gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                        stage: J2kResidentEncodeGpuStage::CoefficientDwt53Horizontal,
                        command_buffer: command_buffer.clone(),
                    });
                }
                if pushed_split_prepare {
                    continue;
                }
                let ptr = tile.prepare_command_buffer.as_ptr();
                if prepare_command_buffer_ptrs.contains(&ptr) {
                    continue;
                }
                prepare_command_buffer_ptrs.push(ptr);
                gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                    stage: J2kResidentEncodeGpuStage::CoefficientPrep,
                    command_buffer: tile.prepare_command_buffer.clone(),
                });
            }
        }

        retained_command_buffers.extend(
            gpu_stage_command_buffers
                .iter()
                .map(|stage_command_buffer| stage_command_buffer.command_buffer.clone()),
        );
        for tile in prepared_tiles {
            if let Some(command_buffer) = tile.prepare_deinterleave_rct_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            if let Some(command_buffer) = tile.prepare_dwt53_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            retained_command_buffers.extend(tile.prepare_dwt53_vertical_command_buffers);
            retained_command_buffers.extend(tile.prepare_dwt53_horizontal_command_buffers);
            if let Some(command_buffer) = tile.prepare_coefficient_extract_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            retained_command_buffers.push(tile.prepare_command_buffer);
            retained_buffers.push(tile.coefficient_buffer);
            retained_buffers.push(tile.deinterleave_status_buffer);
            retained_buffers.extend(tile.plane_buffers);
            retained_buffers.extend(tile.scratch_buffers);
            retained_buffers.push(tile.coefficient_job_buffer);
            recyclable_private_buffers.extend(tile.recyclable_private_buffers);
        }
        retained_buffers.push(coefficient_buffer);
        retained_buffers.push(tier1_job_buffer);
        retained_buffers.push(tier1_output_buffer);
        retained_buffers.push(tier1_status_buffer);
        retained_buffers.push(packet_resolution_buffer);
        retained_buffers.push(packet_subband_buffer);
        retained_buffers.push(resident_block_buffer);
        retained_buffers.push(packet_block_buffer);
        retained_buffers.push(packet_descriptor_buffer);
        retained_buffers.push(state_block_buffer);
        retained_buffers.push(packet_payload_copy_job_buffer);
        retained_buffers.push(header_buffer);
        retained_buffers.push(scratch_buffer);
        retained_buffers.push(packet_job_buffer);
        retained_buffers.push(packet_status_buffer.clone());
        retained_buffers.push(codestream_job_buffer);

        stage_stats.ht_table_build_duration = ht_table_build_duration;
        stage_stats.ht_block_encode_duration = ht_block_encode_duration;
        stage_stats.packet_block_prep_duration = packet_block_prep_duration;
        stage_stats.packetization_duration = packetization_duration;
        stage_stats.codestream_assembly_duration = codestream_assembly_duration;
        stage_stats.ht_command_encode_duration = ht_block_encode_duration
            .saturating_add(packet_block_prep_duration)
            .saturating_add(packetization_duration)
            .saturating_add(codestream_assembly_duration);
        stage_stats.packet_payload_copy_job_capacity_total = packet_payload_copy_job_capacity_total;
        stage_stats.max_packet_payload_copy_jobs_per_tile = max_payload_copy_jobs_per_tile;
        stage_stats.packet_payload_copy_launched_stripe_count_total = packet_jobs
            .len()
            .saturating_mul(max_payload_copy_jobs_per_tile)
            .saturating_mul(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB as usize);
        stage_stats.tier1_output_capacity_total = tier1_output_capacity_total;
        stage_stats.max_tier1_output_capacity = max_tier1_output_capacity;
        stage_stats.packet_output_capacity_total = packet_output_capacity_total;
        stage_stats.max_packet_output_capacity = max_packet_output_capacity_usize;
        stage_stats.codestream_payload_copy_launched_thread_count_total = 0;
        stage_stats.code_block_count = tier1_jobs.len();

        Ok(J2kPendingResidentLosslessCodestreamBatch {
            device: runtime.device.clone(),
            buffer: codestream_buffer,
            byte_offsets: codestream_offsets,
            capacities: codestream_capacities,
            status_buffer: codestream_status_buffer,
            packet_status_buffer,
            tier1_status_readback,
            classic_tier1_density_readback: None,
            classic_tier1_symbol_plan_readback: None,
            classic_tier1_pass_plan_readback: None,
            classic_tier1_token_emit_readback: None,
            classic_tier1_split_token_emit_readback: None,
            classic_gpu_token_pack_used: false,
            command_buffer,
            retained_command_buffers,
            _retained_buffers: retained_buffers,
            recyclable_private_buffers,
            recyclable_shared_buffers,
            gpu_stage_command_buffers,
            stage_stats,
            codestream_payload_copy_dispatched,
            status_stage: "HTJ2K batched codestream assembly",
            length_error: "HTJ2K Metal batched codestream output length exceeds usize",
            capacity_error: "HTJ2K Metal batched codestream output length exceeds buffer",
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_lossless_codestream_buffers_from_prepared_classic_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kResidentClassicBatchEncodeItem>,
    output_capacity_mode: J2kClassicEncodeOutputCapacityMode,
) -> Result<J2kPendingResidentLosslessCodestreamBatch, Error> {
    if items.is_empty() {
        return Err(Error::MetalKernel {
            message: "J2K Metal resident batch encode requires at least one tile".to_string(),
        });
    }

    let mut prepared_tiles = Vec::with_capacity(items.len());
    for item in items {
        let J2kPreparedLosslessDeviceCodeBlocks {
            coefficient_buffer,
            coefficient_byte_offset,
            coefficient_byte_len,
            coefficient_buffer_is_batch_shared,
            code_blocks,
            recyclable_private_buffers,
            _prepare_command_buffer: prepare_command_buffer,
            _prepare_deinterleave_rct_command_buffer: prepare_deinterleave_rct_command_buffer,
            _prepare_dwt53_command_buffer: prepare_dwt53_command_buffer,
            _prepare_dwt53_vertical_command_buffers: prepare_dwt53_vertical_command_buffers,
            _prepare_dwt53_horizontal_command_buffers: prepare_dwt53_horizontal_command_buffers,
            _prepare_coefficient_extract_command_buffer: prepare_coefficient_extract_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        } = item.prepared;
        prepared_tiles.push(PreparedLosslessBatchTile {
            coefficient_buffer,
            coefficient_byte_offset,
            coefficient_byte_len,
            coefficient_buffer_is_batch_shared,
            code_blocks,
            recyclable_private_buffers,
            prepare_command_buffer,
            prepare_deinterleave_rct_command_buffer,
            prepare_dwt53_command_buffer,
            prepare_dwt53_vertical_command_buffers,
            prepare_dwt53_horizontal_command_buffers,
            prepare_coefficient_extract_command_buffer,
            deinterleave_status_buffer,
            plane_buffers,
            scratch_buffers,
            coefficient_job_buffer,
            resolution_count: item.resolution_count,
            num_layers: item.num_layers,
            num_components: item.num_components,
            code_block_count: item.code_block_count,
            packet_descriptors: item.packet_descriptors,
            resolutions: item.resolutions,
            codestream: item.codestream,
        });
    }

    with_runtime_for_device(&session.device, |runtime| {
        let profile_stages = metal_profile_stages_enabled();
        // Commit classic stages independently so the long Tier-1 kernel can run
        // while CPU packet metadata for the following stages is built.
        let split_command_buffers = true;
        let mut stage_stats = J2kResidentEncodeStageStats::default();
        let mut classic_tier1_setup_duration = Duration::ZERO;
        let mut classic_block_encode_duration = Duration::ZERO;
        let mut classic_packet_plan_duration = Duration::ZERO;
        let mut classic_packet_buffer_setup_duration = Duration::ZERO;
        let mut classic_command_buffer_commit_duration = Duration::ZERO;
        let packet_block_prep_duration = Duration::ZERO;
        let mut packetization_duration = Duration::ZERO;
        let mut codestream_assembly_duration = Duration::ZERO;
        let mut retained_command_buffers = Vec::with_capacity(prepared_tiles.len());
        let mut gpu_stage_command_buffers = Vec::new();
        let mut retained_buffers = Vec::<Buffer>::new();
        let profile_classic_tier1_density = metal_profile_classic_tier1_density_enabled();
        let profile_classic_tier1_raw_pack = metal_profile_classic_tier1_raw_pack_enabled();
        let profile_classic_tier1_arithmetic_pack =
            metal_profile_classic_tier1_arithmetic_pack_enabled();
        let profile_classic_tier1_pass_plan = metal_profile_classic_tier1_pass_plan_enabled();
        let profile_classic_tier1_symbol_plan = metal_profile_classic_tier1_symbol_plan_enabled();
        let profile_classic_tier1_token_emit = metal_profile_classic_tier1_token_emit_enabled();
        let profile_classic_tier1_split_token_emit =
            metal_profile_classic_tier1_split_token_emit_enabled();
        let shared_coefficient_buffer = prepared_tiles.first().and_then(|first| {
            let ptr = first.coefficient_buffer.as_ptr();
            prepared_tiles
                .iter()
                .all(|tile| {
                    tile.coefficient_buffer_is_batch_shared
                        && tile.coefficient_buffer.as_ptr() == ptr
                })
                .then(|| first.coefficient_buffer.clone())
        });
        let needs_coefficient_copy = shared_coefficient_buffer.is_none();
        let initial_command_buffer_label = if split_command_buffers && needs_coefficient_copy {
            "signinum-j2k classic resident coefficient copy"
        } else if split_command_buffers {
            "signinum-j2k classic resident Tier-1 encode"
        } else {
            "signinum-j2k classic resident encode batch"
        };
        let mut command_buffer =
            new_resident_encode_command_buffer(runtime, initial_command_buffer_label);
        let (coefficient_buffer, coefficient_offsets) =
            if let Some(coefficient_buffer) = shared_coefficient_buffer {
                (
                    coefficient_buffer,
                    prepared_tiles
                        .iter()
                        .map(|tile| tile.coefficient_byte_offset)
                        .collect::<Vec<_>>(),
                )
            } else {
                let mut coefficient_offsets = Vec::<usize>::with_capacity(prepared_tiles.len());
                let mut total_coefficient_bytes = 0usize;
                for tile in &prepared_tiles {
                    coefficient_offsets.push(total_coefficient_bytes);
                    total_coefficient_bytes = total_coefficient_bytes
                        .checked_add(tile.coefficient_byte_len)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "J2K Metal batch coefficient buffer size overflow".to_string(),
                        })?;
                }
                let coefficient_buffer = runtime.device.new_buffer(
                    total_coefficient_bytes.max(1) as u64,
                    MTLResourceOptions::StorageModePrivate,
                );
                let blit = command_buffer.new_blit_command_encoder();
                if profile_stages {
                    blit.set_label("J2K coefficient prep");
                }
                for (tile, &dst_offset) in prepared_tiles.iter().zip(coefficient_offsets.iter()) {
                    if tile.coefficient_byte_len > 0 {
                        blit.copy_from_buffer(
                            &tile.coefficient_buffer,
                            tile.coefficient_byte_offset as u64,
                            &coefficient_buffer,
                            dst_offset as u64,
                            tile.coefficient_byte_len as u64,
                        );
                    }
                }
                blit.end_encoding();
                if split_command_buffers {
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::CoefficientCopy,
                        "signinum-j2k classic resident Tier-1 encode",
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
                (coefficient_buffer, coefficient_offsets)
            };

        let classic_tier1_setup_started = profile_stages.then(Instant::now);
        let classic_tier1_setup_signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP);
        let classic_resident_style_flags = classic_resident_style_flags_from_env();
        let mut tier1_jobs = Vec::<J2kClassicEncodeBatchJob>::new();
        let mut tier1_output_capacity_total = 0usize;
        let mut max_tier1_output_capacity = 0usize;
        let mut tier1_segment_capacity_total = 0usize;
        let mut tile_tier1_job_bases = Vec::<usize>::with_capacity(prepared_tiles.len());
        let mut tile_tier1_output_capacities = Vec::<usize>::with_capacity(prepared_tiles.len());
        for (tile, &coefficient_byte_offset) in
            prepared_tiles.iter().zip(coefficient_offsets.iter())
        {
            tile_tier1_job_bases.push(tier1_jobs.len());
            let tile_output_start = tier1_output_capacity_total;
            let coefficient_word_offset = coefficient_byte_offset
                .checked_div(size_of::<i32>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch coefficient offset division failed".to_string(),
                })?;
            let coefficient_word_offset_u32 =
                u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal batch coefficient offset exceeds u32".to_string(),
                })?;
            for block in &tile.code_blocks {
                let output_capacity = classic_encode_output_capacity_for_mode(
                    block.width,
                    block.height,
                    block.total_bitplanes,
                    output_capacity_mode,
                )?;
                max_tier1_output_capacity = max_tier1_output_capacity.max(output_capacity);
                let output_offset =
                    u32::try_from(tier1_output_capacity_total).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 output offset exceeds u32".to_string(),
                    })?;
                let segment_offset = u32::try_from(tier1_segment_capacity_total).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 segment offset exceeds u32".to_string(),
                    }
                })?;
                let coefficient_offset = block
                    .coefficient_offset
                    .checked_add(coefficient_word_offset_u32)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch coefficient offset overflow".to_string(),
                    })?;
                let segment_capacity = classic_encode_segment_capacity(
                    classic_resident_style_flags,
                    block.total_bitplanes,
                );
                tier1_jobs.push(J2kClassicEncodeBatchJob {
                    coefficient_offset,
                    output_offset,
                    segment_offset,
                    width: block.width,
                    height: block.height,
                    sub_band_type: classic_encode_sub_band_code(block.sub_band_type),
                    total_bitplanes: u32::from(block.total_bitplanes),
                    style_flags: classic_resident_style_flags,
                    output_capacity: u32::try_from(output_capacity).map_err(|_| {
                        Error::MetalKernel {
                            message: "J2K Metal batch Tier-1 output capacity exceeds u32"
                                .to_string(),
                        }
                    })?,
                    segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                        Error::MetalKernel {
                            message: "J2K Metal batch Tier-1 segment capacity exceeds u32"
                                .to_string(),
                        }
                    })?,
                });
                tier1_output_capacity_total = tier1_output_capacity_total
                    .checked_add(output_capacity)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 output buffer overflow".to_string(),
                    })?;
                tier1_segment_capacity_total = tier1_segment_capacity_total
                    .checked_add(segment_capacity)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 segment buffer overflow".to_string(),
                    })?;
            }
            tile_tier1_output_capacities.push(
                tier1_output_capacity_total
                    .checked_sub(tile_output_start)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch tile Tier-1 capacity underflow".to_string(),
                    })?,
            );
        }

        let tier1_job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "J2K Metal batch Tier-1 job count exceeds u32".to_string(),
        })?;
        let tier1_job_buffer = owned_slice_buffer(&runtime.device, &tier1_jobs);
        let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
        let mut recyclable_shared_buffers = Vec::<(usize, Buffer)>::new();
        let tier1_output_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_output_capacity_total.max(1),
            &mut recyclable_private_buffers,
        );
        let tier1_status_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_jobs.len().max(1) * size_of::<J2kClassicEncodeStatus>(),
            &mut recyclable_private_buffers,
        );
        let tier1_segment_buffer = take_recyclable_private_buffer(
            runtime,
            tier1_segment_capacity_total.max(1) * size_of::<J2kClassicSegment>(),
            &mut recyclable_private_buffers,
        );
        drop(classic_tier1_setup_signpost);
        if let Some(started) = classic_tier1_setup_started {
            classic_tier1_setup_duration = started.elapsed();
        }
        let classic_split_mq_byte_gpu_token_pack_requested =
            classic_tier1_split_mq_byte_gpu_token_pack_requested();
        let classic_split_mq_byte_gpu_token_pack_disabled =
            classic_tier1_split_mq_byte_gpu_token_pack_disabled();
        let classic_split_gpu_token_pack_requested = classic_tier1_split_gpu_token_pack_requested();
        let classic_gpu_token_pack_requested = classic_tier1_gpu_token_pack_requested();
        let use_classic_split_mq_byte_gpu_token_pack = if tier1_job_count > 0 {
            if classic_split_mq_byte_gpu_token_pack_requested {
                if !classic_tier1_gpu_token_pack_supported(&tier1_jobs) {
                    return Err(Error::MetalKernel {
                        message: "J2K Metal classic split MQ-byte GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
                    });
                }
                true
            } else {
                !classic_split_mq_byte_gpu_token_pack_disabled
                    && !classic_split_gpu_token_pack_requested
                    && !classic_gpu_token_pack_requested
                    && classic_tier1_gpu_token_pack_supported(&tier1_jobs)
            }
        } else {
            false
        };
        let use_classic_split_gpu_token_pack = if classic_split_gpu_token_pack_requested
            && !use_classic_split_mq_byte_gpu_token_pack
            && tier1_job_count > 0
        {
            if !classic_tier1_gpu_token_pack_supported(&tier1_jobs) {
                return Err(Error::MetalKernel {
                    message: "J2K Metal classic split GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
                });
            }
            true
        } else {
            false
        };
        let use_classic_gpu_token_pack = if !use_classic_split_mq_byte_gpu_token_pack
            && !use_classic_split_gpu_token_pack
            && classic_gpu_token_pack_requested
            && tier1_job_count > 0
        {
            if !classic_tier1_gpu_token_pack_supported(&tier1_jobs) {
                return Err(Error::MetalKernel {
                    message: "J2K Metal classic GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
                });
            }
            true
        } else {
            false
        };
        let mut classic_gpu_token_pack_readback = None;
        if tier1_job_count > 0 {
            let command_encode_started = profile_stages.then(Instant::now);
            let signpost =
                hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE);
            if use_classic_split_mq_byte_gpu_token_pack || use_classic_split_gpu_token_pack {
                let token_buffers = dispatch_classic_tier1_split_token_emit_for_gpu_pack(
                    runtime,
                    &command_buffer,
                    &coefficient_buffer,
                    &tier1_job_buffer,
                    &tier1_jobs,
                    &mut recyclable_private_buffers,
                    use_classic_split_mq_byte_gpu_token_pack,
                )?;
                if split_command_buffers {
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicTier1SplitTokenEmit,
                        "signinum-j2k classic resident Tier-1 split token pack",
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
                dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
                    runtime,
                    &command_buffer,
                    &tier1_job_buffer,
                    &token_buffers,
                    &tier1_output_buffer,
                    &tier1_status_buffer,
                    &tier1_segment_buffer,
                );
                drop(signpost);
                if let Some(started) = command_encode_started {
                    classic_block_encode_duration =
                        classic_block_encode_duration.saturating_add(started.elapsed());
                }
                if split_command_buffers {
                    let next_label = if profile_classic_tier1_density {
                        "signinum-j2k classic resident Tier-1 density profile"
                    } else if profile_classic_tier1_raw_pack {
                        "signinum-j2k classic resident Tier-1 raw-pack profile"
                    } else if profile_classic_tier1_arithmetic_pack {
                        "signinum-j2k classic resident Tier-1 arithmetic-pack profile"
                    } else if profile_classic_tier1_symbol_plan {
                        "signinum-j2k classic resident Tier-1 symbol plan"
                    } else if profile_classic_tier1_token_emit {
                        "signinum-j2k classic resident Tier-1 token emit"
                    } else if profile_classic_tier1_split_token_emit {
                        "signinum-j2k classic resident Tier-1 split token emit"
                    } else {
                        "signinum-j2k classic resident packetization"
                    };
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicTier1TokenPack,
                        next_label,
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
            } else if use_classic_gpu_token_pack {
                let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
                    runtime,
                    &command_buffer,
                    &coefficient_buffer,
                    &tier1_job_buffer,
                    &tier1_jobs,
                    &mut recyclable_private_buffers,
                )?;
                if split_command_buffers {
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicTier1TokenEmit,
                        "signinum-j2k classic resident Tier-1 token pack",
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
                dispatch_classic_tier1_token_pack_from_gpu_tokens(
                    runtime,
                    &command_buffer,
                    &tier1_job_buffer,
                    &token_buffers,
                    &tier1_output_buffer,
                    &tier1_status_buffer,
                    &tier1_segment_buffer,
                );
                classic_gpu_token_pack_readback = schedule_classic_tier1_gpu_token_pack_readback(
                    runtime,
                    &command_buffer,
                    &token_buffers,
                    profile_stages,
                )?;
                drop(signpost);
                if let Some(started) = command_encode_started {
                    classic_block_encode_duration =
                        classic_block_encode_duration.saturating_add(started.elapsed());
                }
                if split_command_buffers {
                    let next_label = if profile_classic_tier1_density {
                        "signinum-j2k classic resident Tier-1 density profile"
                    } else if profile_classic_tier1_raw_pack {
                        "signinum-j2k classic resident Tier-1 raw-pack profile"
                    } else if profile_classic_tier1_arithmetic_pack {
                        "signinum-j2k classic resident Tier-1 arithmetic-pack profile"
                    } else if profile_classic_tier1_symbol_plan {
                        "signinum-j2k classic resident Tier-1 symbol plan"
                    } else if profile_classic_tier1_token_emit {
                        "signinum-j2k classic resident Tier-1 token emit"
                    } else if profile_classic_tier1_split_token_emit {
                        "signinum-j2k classic resident Tier-1 split token emit"
                    } else {
                        "signinum-j2k classic resident packetization"
                    };
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicTier1TokenPack,
                        next_label,
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
            } else {
                let encoder = command_buffer.new_compute_command_encoder();
                label_compute_encoder(encoder, "J2K Tier-1 encode");
                let classic_encode_pipeline =
                    classic_encode_code_blocks_pipeline(runtime, &tier1_jobs);
                encoder.set_compute_pipeline_state(classic_encode_pipeline);
                encoder.set_buffer(0, Some(&coefficient_buffer), 0);
                encoder.set_buffer(1, Some(&tier1_output_buffer), 0);
                encoder.set_buffer(2, Some(&tier1_job_buffer), 0);
                encoder.set_buffer(3, Some(&tier1_status_buffer), 0);
                encoder.set_buffer(4, Some(&tier1_segment_buffer), 0);
                encoder.set_bytes(
                    5,
                    size_of::<u32>() as u64,
                    (&raw const tier1_job_count).cast(),
                );
                dispatch_1d_pipeline(encoder, classic_encode_pipeline, u64::from(tier1_job_count));
                encoder.end_encoding();
                drop(signpost);
                if let Some(started) = command_encode_started {
                    classic_block_encode_duration =
                        classic_block_encode_duration.saturating_add(started.elapsed());
                }
                if split_command_buffers {
                    let next_label = if profile_classic_tier1_density {
                        "signinum-j2k classic resident Tier-1 density profile"
                    } else if profile_classic_tier1_raw_pack {
                        "signinum-j2k classic resident Tier-1 raw-pack profile"
                    } else if profile_classic_tier1_arithmetic_pack {
                        "signinum-j2k classic resident Tier-1 arithmetic-pack profile"
                    } else if profile_classic_tier1_symbol_plan {
                        "signinum-j2k classic resident Tier-1 symbol plan"
                    } else if profile_classic_tier1_token_emit {
                        "signinum-j2k classic resident Tier-1 token emit"
                    } else if profile_classic_tier1_split_token_emit {
                        "signinum-j2k classic resident Tier-1 split token emit"
                    } else {
                        "signinum-j2k classic resident packetization"
                    };
                    command_buffer = finish_resident_encode_split_command_buffer_timed(
                        command_buffer,
                        runtime,
                        J2kResidentEncodeGpuStage::ClassicBlock,
                        next_label,
                        &mut gpu_stage_command_buffers,
                        profile_stages,
                        &mut classic_command_buffer_commit_duration,
                    );
                }
            }
        } else if split_command_buffers {
            label_command_buffer(
                &command_buffer,
                "signinum-j2k classic resident packetization",
            );
        }
        let classic_tier1_density_readback = if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_density_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                let next_label = if profile_classic_tier1_raw_pack {
                    "signinum-j2k classic resident Tier-1 raw-pack profile"
                } else if profile_classic_tier1_arithmetic_pack {
                    "signinum-j2k classic resident Tier-1 arithmetic-pack profile"
                } else if profile_classic_tier1_symbol_plan {
                    "signinum-j2k classic resident Tier-1 symbol plan"
                } else if profile_classic_tier1_token_emit {
                    "signinum-j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "signinum-j2k classic resident Tier-1 split token emit"
                } else {
                    "signinum-j2k classic resident packetization"
                };
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1Density,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };
        let classic_tier1_raw_pack_buffer = if tier1_job_count > 0 {
            let buffer = dispatch_classic_tier1_raw_pack_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
                tier1_output_capacity_total,
            )?;
            if buffer.is_some() && split_command_buffers {
                let next_label = if profile_classic_tier1_arithmetic_pack {
                    "signinum-j2k classic resident Tier-1 arithmetic-pack profile"
                } else if profile_classic_tier1_symbol_plan {
                    "signinum-j2k classic resident Tier-1 symbol plan"
                } else if profile_classic_tier1_token_emit {
                    "signinum-j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "signinum-j2k classic resident Tier-1 split token emit"
                } else {
                    "signinum-j2k classic resident packetization"
                };
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1RawPack,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            buffer
        } else {
            None
        };
        let classic_tier1_arithmetic_pack_buffer = if tier1_job_count > 0 {
            let buffer = dispatch_classic_tier1_arithmetic_pack_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
                tier1_output_capacity_total,
            )?;
            if buffer.is_some() && split_command_buffers {
                let next_label = if profile_classic_tier1_symbol_plan {
                    "signinum-j2k classic resident Tier-1 symbol plan"
                } else if profile_classic_tier1_token_emit {
                    "signinum-j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "signinum-j2k classic resident Tier-1 split token emit"
                } else {
                    "signinum-j2k classic resident packetization"
                };
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1ArithmeticPack,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            buffer
        } else {
            None
        };
        let classic_tier1_symbol_plan_readback = if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_symbol_plan_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                let next_label = if profile_classic_tier1_pass_plan {
                    "signinum-j2k classic resident Tier-1 pass plan"
                } else if profile_classic_tier1_token_emit {
                    "signinum-j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "signinum-j2k classic resident Tier-1 split token emit"
                } else {
                    "signinum-j2k classic resident packetization"
                };
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1SymbolPlan,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };
        let classic_tier1_pass_plan_readback = if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_pass_plan_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                let next_label = if profile_classic_tier1_token_emit {
                    "signinum-j2k classic resident Tier-1 token emit"
                } else if profile_classic_tier1_split_token_emit {
                    "signinum-j2k classic resident Tier-1 split token emit"
                } else {
                    "signinum-j2k classic resident packetization"
                };
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1PassPlan,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };
        let classic_tier1_token_emit_readback = if classic_gpu_token_pack_readback.is_some() {
            classic_gpu_token_pack_readback
        } else if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_token_emit_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                let next_label = if profile_classic_tier1_split_token_emit {
                    "signinum-j2k classic resident Tier-1 split token emit"
                } else {
                    "signinum-j2k classic resident packetization"
                };
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1TokenEmit,
                    next_label,
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };
        let classic_tier1_split_token_emit_readback = if tier1_job_count > 0 {
            let readback = dispatch_classic_tier1_split_token_emit_profile(
                runtime,
                &command_buffer,
                &coefficient_buffer,
                &tier1_job_buffer,
                &tier1_jobs,
            )?;
            if readback.is_some() && split_command_buffers {
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1SplitTokenEmit,
                    "signinum-j2k classic resident packetization",
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            }
            readback
        } else {
            None
        };

        let classic_packet_plan_started = profile_stages.then(Instant::now);
        let classic_packet_plan_signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN);
        let mut packet_resolutions = Vec::<J2kPacketResolution>::new();
        let mut packet_subbands = Vec::<J2kPacketSubband>::new();
        let mut resident_blocks = Vec::<J2kResidentPacketBlock>::new();
        let mut packet_descriptors = Vec::<J2kPacketDescriptor>::new();
        let mut state_blocks = Vec::<J2kPacketStateBlock>::new();
        let mut packet_jobs = Vec::<J2kBatchedPacketEncodeJob>::with_capacity(prepared_tiles.len());
        let mut assembly_jobs =
            Vec::<J2kBatchedCodestreamAssemblyJob>::with_capacity(prepared_tiles.len());
        let mut packet_output_capacity_total = 0usize;
        let mut packet_payload_copy_job_capacity_total = 0usize;
        let mut max_payload_copy_jobs_per_tile = 0usize;
        let mut header_capacity_total = 0usize;
        let mut scratch_words_total = 0usize;
        let mut codestream_capacity_total = 0usize;
        let mut codestream_offsets = Vec::<usize>::with_capacity(prepared_tiles.len());
        let mut codestream_capacities = Vec::<usize>::with_capacity(prepared_tiles.len());

        for (tile_index, tile) in prepared_tiles.iter().enumerate() {
            let local_resolution_offset = packet_resolutions.len();
            let local_subband_offset = packet_subbands.len();
            let local_block_offset = resident_blocks.len();
            let local_descriptor_offset = packet_descriptors.len();
            let local_state_block_offset = state_blocks.len();
            let tier1_job_base = tile_tier1_job_bases[tile_index];
            let mut max_tree_nodes = 1usize;
            let mut local_subband_count = 0usize;
            let mut local_resident_block_count = 0usize;
            let mut local_payload_copy_job_capacity = 0usize;

            for resolution in &tile.resolutions {
                let subband_offset =
                    u32::try_from(local_subband_count).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal batch packet subband offset exceeds u32".to_string(),
                    })?;
                for subband in &resolution.subbands {
                    let block_offset = u32::try_from(local_resident_block_count).map_err(|_| {
                        Error::MetalKernel {
                            message: "J2K Metal batch packet block offset exceeds u32".to_string(),
                        }
                    })?;
                    max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                    )?);
                    let code_block_start =
                        usize::try_from(subband.code_block_start).map_err(|_| {
                            Error::MetalKernel {
                                message: "J2K Metal batch packet code-block offset exceeds usize"
                                    .to_string(),
                            }
                        })?;
                    let code_block_count =
                        usize::try_from(subband.code_block_count).map_err(|_| {
                            Error::MetalKernel {
                                message: "J2K Metal batch packet code-block count exceeds usize"
                                    .to_string(),
                            }
                        })?;
                    let code_block_end = code_block_start
                        .checked_add(code_block_count)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "J2K Metal batch packet code-block range overflow".to_string(),
                        })?;
                    if code_block_end > tile.code_blocks.len() {
                        return Err(Error::MetalKernel {
                            message: "J2K Metal batch packet code-block range out of bounds"
                                .to_string(),
                        });
                    }
                    for tier1_job_index in code_block_start..code_block_end {
                        resident_blocks.push(J2kResidentPacketBlock {
                            tier1_job_index: u32::try_from(
                                tier1_job_base.checked_add(tier1_job_index).ok_or_else(|| {
                                    Error::MetalKernel {
                                        message: "J2K Metal batch Tier-1 index overflow"
                                            .to_string(),
                                    }
                                })?,
                            )
                            .map_err(|_| Error::MetalKernel {
                                message: "J2K Metal batch Tier-1 index exceeds u32".to_string(),
                            })?,
                            previously_included: 0,
                            l_block: 3,
                            block_coding_mode: 0,
                        });
                    }
                    packet_subbands.push(J2kPacketSubband {
                        block_offset,
                        block_count: subband.code_block_count,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    });
                    local_subband_count =
                        local_subband_count
                            .checked_add(1)
                            .ok_or_else(|| Error::MetalKernel {
                                message: "J2K Metal batch subband count overflow".to_string(),
                            })?;
                    local_resident_block_count = local_resident_block_count
                        .checked_add(code_block_count)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "J2K Metal batch resident block count overflow".to_string(),
                        })?;
                }
                packet_resolutions.push(J2kPacketResolution {
                    subband_offset,
                    subband_count: u32::try_from(resolution.subbands.len()).map_err(|_| {
                        Error::MetalKernel {
                            message: "J2K Metal batch resolution subband count exceeds u32"
                                .to_string(),
                        }
                    })?,
                });
            }

            if tile.resolutions.len()
                != usize::try_from(tile.resolution_count).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal batch resolution count exceeds usize".to_string(),
                })?
            {
                return Err(Error::MetalKernel {
                    message: "J2K Metal batch resolution count mismatch".to_string(),
                });
            }
            if local_resident_block_count
                != usize::try_from(tile.code_block_count).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal batch code-block count exceeds usize".to_string(),
                })?
            {
                return Err(Error::MetalKernel {
                    message: "J2K Metal batch code-block count mismatch".to_string(),
                });
            }

            let mut state_block_offsets = HashMap::<u32, (u32, usize)>::new();
            for descriptor in &tile.packet_descriptors {
                let packet_index =
                    usize::try_from(descriptor.packet_index).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal batch descriptor packet index exceeds usize"
                            .to_string(),
                    })?;
                let resolution = packet_resolutions
                    .get(local_resolution_offset + packet_index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch descriptor packet index out of range".to_string(),
                    })?;
                let subband_start =
                    usize::try_from(resolution.subband_offset).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal batch descriptor subband offset exceeds usize"
                            .to_string(),
                    })?;
                let subband_count =
                    usize::try_from(resolution.subband_count).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal batch descriptor subband count exceeds usize"
                            .to_string(),
                    })?;
                let mut packet_block_count = 0usize;
                for subband in &packet_subbands[local_subband_offset + subband_start
                    ..local_subband_offset + subband_start + subband_count]
                {
                    let subband_block_count =
                        usize::try_from(subband.block_count).map_err(|_| Error::MetalKernel {
                            message: "J2K Metal batch descriptor block count exceeds usize"
                                .to_string(),
                        })?;
                    packet_block_count = packet_block_count
                        .checked_add(subband_block_count)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "J2K Metal batch descriptor block count overflow".to_string(),
                        })?;
                }
                let (state_block_offset, existing_count) = if let Some(&(offset, count)) =
                    state_block_offsets.get(&descriptor.state_index)
                {
                    (offset, count)
                } else {
                    let offset = u32::try_from(state_blocks.len() - local_state_block_offset)
                        .map_err(|_| Error::MetalKernel {
                            message: "J2K Metal batch state block offset exceeds u32".to_string(),
                        })?;
                    for subband in &packet_subbands[local_subband_offset + subband_start
                        ..local_subband_offset + subband_start + subband_count]
                    {
                        for _ in 0..subband.block_count {
                            state_blocks.push(J2kPacketStateBlock {
                                previously_included: 0,
                                l_block: 3,
                            });
                        }
                    }
                    state_block_offsets
                        .insert(descriptor.state_index, (offset, packet_block_count));
                    (offset, packet_block_count)
                };
                if existing_count != packet_block_count {
                    return Err(Error::MetalKernel {
                        message: "J2K Metal batch descriptor state layout mismatch".to_string(),
                    });
                }
                local_payload_copy_job_capacity = local_payload_copy_job_capacity
                    .checked_add(packet_block_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch packet payload-copy job count overflow"
                            .to_string(),
                    })?;
                packet_descriptors.push(J2kPacketDescriptor {
                    packet_index: descriptor.packet_index,
                    state_index: descriptor.state_index,
                    layer: u32::from(descriptor.layer),
                    resolution: descriptor.resolution,
                    component: u32::from(descriptor.component),
                    precinct_lo: descriptor.precinct as u32,
                    precinct_hi: (descriptor.precinct >> 32) as u32,
                    state_block_offset,
                });
            }

            let header_capacity = local_resident_block_count
                .checked_mul(256)
                .and_then(|bytes| bytes.checked_add(4096))
                .map(|bytes| bytes.max(4096))
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch packet header capacity overflow".to_string(),
                })?;
            let packet_output_capacity = classic_packet_output_capacity(
                tile_tier1_output_capacities[tile_index],
                header_capacity,
                tile.packet_descriptors.len().max(tile.resolutions.len()),
                tile.codestream,
            )?;
            let codestream_capacity =
                lossless_codestream_assembly_capacity(packet_output_capacity, tile.codestream)?;
            let codestream_payload_offset = lossless_codestream_payload_offset(tile.codestream)?;
            let scratch_words =
                max_tree_nodes
                    .checked_mul(6)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch scratch size overflow".to_string(),
                    })?;

            let header_offset = header_capacity_total;
            let scratch_offset = scratch_words_total;
            if tile.packet_descriptors.is_empty() {
                local_payload_copy_job_capacity = local_resident_block_count;
            }
            let payload_copy_offset = packet_payload_copy_job_capacity_total;
            let codestream_offset = codestream_capacity_total;
            let packet_output_offset = codestream_offset
                .checked_add(codestream_payload_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch direct packet output offset overflow".to_string(),
                })?;
            packet_jobs.push(J2kBatchedPacketEncodeJob {
                resolution_offset: u32::try_from(local_resolution_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch resolution offset exceeds u32".to_string(),
                    }
                })?,
                subband_offset: u32::try_from(local_subband_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch subband offset exceeds u32".to_string(),
                    }
                })?,
                block_offset: u32::try_from(local_block_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch block offset exceeds u32".to_string(),
                    }
                })?,
                descriptor_offset: u32::try_from(local_descriptor_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch descriptor offset exceeds u32".to_string(),
                    }
                })?,
                state_block_offset: u32::try_from(local_state_block_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch state block offset exceeds u32".to_string(),
                    }
                })?,
                output_offset: u32::try_from(packet_output_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch packet output offset exceeds u32".to_string(),
                    }
                })?,
                header_offset: u32::try_from(header_offset).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal batch header offset exceeds u32".to_string(),
                })?,
                scratch_offset: u32::try_from(scratch_offset).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal batch scratch offset exceeds u32".to_string(),
                })?,
                payload_copy_offset: u32::try_from(payload_copy_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch packet payload-copy offset exceeds u32"
                            .to_string(),
                    }
                })?,
                payload_copy_capacity: u32::try_from(local_payload_copy_job_capacity).map_err(
                    |_| Error::MetalKernel {
                        message: "J2K Metal batch packet payload-copy capacity exceeds u32"
                            .to_string(),
                    },
                )?,
                resolution_count: tile.resolution_count,
                num_layers: u32::from(tile.num_layers),
                num_components: u32::from(tile.num_components),
                code_block_count: tile.code_block_count,
                subband_count: u32::try_from(local_subband_count).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch local subband count exceeds u32".to_string(),
                    }
                })?,
                descriptor_count: u32::try_from(tile.packet_descriptors.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch descriptor count exceeds u32".to_string(),
                    }
                })?,
                output_capacity: u32::try_from(packet_output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch packet output capacity exceeds u32".to_string(),
                    }
                })?,
                header_capacity: u32::try_from(header_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch header capacity exceeds u32".to_string(),
                    }
                })?,
                scratch_node_capacity: u32::try_from(max_tree_nodes).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch scratch node capacity exceeds u32".to_string(),
                    }
                })?,
            });
            assembly_jobs.push(J2kBatchedCodestreamAssemblyJob {
                tile_data_offset: u32::try_from(packet_output_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch assembly packet offset exceeds u32".to_string(),
                    }
                })?,
                codestream_offset: u32::try_from(codestream_offset).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch codestream offset exceeds u32".to_string(),
                    }
                })?,
                width: tile.codestream.width,
                height: tile.codestream.height,
                num_components: u32::from(tile.codestream.num_components),
                bit_depth: u32::from(tile.codestream.bit_depth),
                signed_samples: u32::from(tile.codestream.signed),
                num_decomposition_levels: u32::from(tile.codestream.num_decomposition_levels),
                use_mct: u32::from(tile.codestream.use_mct),
                guard_bits: u32::from(tile.codestream.guard_bits),
                progression_order: codestream_progression_order_code(
                    tile.codestream.progression_order,
                ),
                write_tlm: u32::from(tile.codestream.write_tlm),
                high_throughput: 0,
                code_block_style: classic_cod_block_style_from_flags(classic_resident_style_flags),
                code_block_width_exp: u32::from(tile.codestream.code_block_width_exp),
                code_block_height_exp: u32::from(tile.codestream.code_block_height_exp),
                output_capacity: u32::try_from(codestream_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch codestream capacity exceeds u32".to_string(),
                    }
                })?,
            });
            codestream_offsets.push(codestream_offset);
            codestream_capacities.push(codestream_capacity);
            packet_output_capacity_total = packet_output_capacity_total
                .checked_add(packet_output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch packet output total overflow".to_string(),
                })?;
            packet_payload_copy_job_capacity_total = packet_payload_copy_job_capacity_total
                .checked_add(local_payload_copy_job_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch packet payload-copy job total overflow".to_string(),
                })?;
            max_payload_copy_jobs_per_tile =
                max_payload_copy_jobs_per_tile.max(local_payload_copy_job_capacity);
            header_capacity_total = header_capacity_total
                .checked_add(header_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch header total overflow".to_string(),
                })?;
            scratch_words_total =
                scratch_words_total
                    .checked_add(scratch_words)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch scratch total overflow".to_string(),
                    })?;
            codestream_capacity_total = codestream_capacity_total
                .checked_add(codestream_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch codestream total overflow".to_string(),
                })?;
        }
        drop(classic_packet_plan_signpost);
        if let Some(started) = classic_packet_plan_started {
            classic_packet_plan_duration = started.elapsed();
        }

        let classic_packet_buffer_setup_started = profile_stages.then(Instant::now);
        let classic_packet_buffer_setup_signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP);
        let packet_resolution_buffer = copied_slice_buffer(&runtime.device, &packet_resolutions);
        let packet_subband_buffer = copied_slice_buffer(&runtime.device, &packet_subbands);
        let resident_block_buffer = copied_slice_buffer(&runtime.device, &resident_blocks);
        let packet_descriptor_buffer = copied_slice_buffer(&runtime.device, &packet_descriptors);
        let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks);
        let packet_payload_copy_job_buffer = take_recyclable_private_buffer(
            runtime,
            packet_payload_copy_job_capacity_total
                .max(1)
                .checked_mul(size_of::<J2kPacketPayloadCopyJob>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch packet payload-copy buffer size overflow".to_string(),
                })?,
            &mut recyclable_private_buffers,
        );
        let header_buffer = take_recyclable_private_buffer(
            runtime,
            header_capacity_total.max(1),
            &mut recyclable_private_buffers,
        );
        let scratch_buffer = take_recyclable_private_buffer(
            runtime,
            scratch_words_total.max(1) * size_of::<u32>(),
            &mut recyclable_private_buffers,
        );
        let packet_job_buffer = copied_slice_buffer(&runtime.device, &packet_jobs);
        let packet_status_buffer = zeroed_recyclable_shared_buffer(
            runtime,
            packet_jobs.len().max(1) * size_of::<J2kPacketEncodeStatus>(),
            &mut recyclable_shared_buffers,
        );
        let codestream_job_buffer = copied_slice_buffer(&runtime.device, &assembly_jobs);
        let codestream_buffer = runtime.device.new_buffer(
            codestream_capacity_total.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let codestream_status_buffer = runtime.device.new_buffer(
            (assembly_jobs.len() * size_of::<J2kCodestreamAssemblyStatus>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        drop(classic_packet_buffer_setup_signpost);
        if let Some(started) = classic_packet_buffer_setup_started {
            classic_packet_buffer_setup_duration = started.elapsed();
        }

        let resident_block_params = J2kResidentPacketBlockParams {
            block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
                message: "J2K Metal batch resident block count exceeds u32".to_string(),
            })?,
            tier1_job_count,
        };

        let tile_count = u64::try_from(packet_jobs.len()).map_err(|_| Error::MetalKernel {
            message: "J2K Metal batch tile count exceeds u64".to_string(),
        })?;
        let command_encode_started = profile_stages.then(Instant::now);
        let signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE);
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "J2K packetization");
        encoder.set_compute_pipeline_state(&runtime.packet_encode_resident_classic_batched);
        encoder.set_buffer(0, Some(&packet_resolution_buffer), 0);
        encoder.set_buffer(1, Some(&packet_subband_buffer), 0);
        encoder.set_buffer(2, Some(&resident_block_buffer), 0);
        encoder.set_buffer(3, Some(&tier1_output_buffer), 0);
        encoder.set_buffer(4, Some(&codestream_buffer), 0);
        encoder.set_buffer(5, Some(&header_buffer), 0);
        encoder.set_buffer(6, Some(&scratch_buffer), 0);
        encoder.set_buffer(7, Some(&packet_job_buffer), 0);
        encoder.set_buffer(8, Some(&packet_status_buffer), 0);
        encoder.set_buffer(9, Some(&packet_descriptor_buffer), 0);
        encoder.set_buffer(10, Some(&state_block_buffer), 0);
        encoder.set_buffer(11, Some(&packet_payload_copy_job_buffer), 0);
        encoder.set_buffer(12, Some(&tier1_job_buffer), 0);
        encoder.set_buffer(13, Some(&tier1_status_buffer), 0);
        encoder.set_buffer(14, Some(&tier1_segment_buffer), 0);
        encoder.set_bytes(
            15,
            size_of::<J2kResidentPacketBlockParams>() as u64,
            (&raw const resident_block_params).cast(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: tile_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .packet_encode_resident_classic_batched
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        drop(signpost);
        if let Some(started) = command_encode_started {
            packetization_duration = packetization_duration.saturating_add(started.elapsed());
        }
        if split_command_buffers {
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::Packetization,
                "signinum-j2k classic resident packet payload copy",
                &mut gpu_stage_command_buffers,
                profile_stages,
                &mut classic_command_buffer_commit_duration,
            );
        }
        let packet_payload_copy_dispatched = dispatch_batched_packet_payload_copy(
            runtime,
            &command_buffer,
            J2kBatchedPacketPayloadCopyDispatch {
                payload_buffer: &tier1_output_buffer,
                packet_output_buffer: &codestream_buffer,
                packet_job_buffer: &packet_job_buffer,
                packet_status_buffer: &packet_status_buffer,
                packet_payload_copy_job_buffer: &packet_payload_copy_job_buffer,
                tile_count,
                max_payload_copy_jobs_per_tile: max_payload_copy_jobs_per_tile as u64,
                label: "J2K packetization payload copy",
                signpost_name: SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
            },
        );
        if split_command_buffers {
            if packet_payload_copy_dispatched {
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::PacketPayloadCopy,
                    "signinum-j2k classic resident codestream assembly",
                    &mut gpu_stage_command_buffers,
                    profile_stages,
                    &mut classic_command_buffer_commit_duration,
                );
            } else {
                label_command_buffer(
                    &command_buffer,
                    "signinum-j2k classic resident codestream assembly",
                );
            }
        }

        let max_packet_output_capacity = packet_jobs
            .iter()
            .map(|job| job.output_capacity)
            .max()
            .unwrap_or(0);
        let max_packet_output_capacity_usize = usize::try_from(max_packet_output_capacity)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal batch max packet output capacity exceeds usize".to_string(),
            })?;
        let command_encode_started = profile_stages.then(Instant::now);
        let signpost = hybrid_stage_signpost(
            SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
        );
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "J2K codestream assembly");
        encoder.set_compute_pipeline_state(&runtime.lossless_codestream_assemble_batched);
        encoder.set_buffer(0, Some(&codestream_buffer), 0);
        encoder.set_buffer(1, Some(&packet_status_buffer), 0);
        encoder.set_buffer(2, Some(&codestream_buffer), 0);
        encoder.set_buffer(3, Some(&codestream_job_buffer), 0);
        encoder.set_buffer(4, Some(&codestream_status_buffer), 0);
        encoder.dispatch_threads(
            MTLSize {
                width: tile_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .lossless_codestream_assemble_batched
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        drop(signpost);
        if split_command_buffers {
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::CodestreamAssembly,
                "signinum-j2k classic resident result readback",
                &mut gpu_stage_command_buffers,
                profile_stages,
                &mut classic_command_buffer_commit_duration,
            );
        }
        let codestream_payload_copy_dispatched = false;
        if let Some(started) = command_encode_started {
            codestream_assembly_duration =
                codestream_assembly_duration.saturating_add(started.elapsed());
        }
        let tier1_status_readback = schedule_resident_tier1_status_readback(
            runtime,
            &command_buffer,
            &tier1_status_buffer,
            J2kResidentTier1StatusKind::Classic,
            classic_resident_style_flags,
            Some(&tier1_jobs),
            tier1_jobs.len(),
            size_of::<J2kClassicEncodeStatus>(),
            profile_stages,
        )?;
        let final_commit_started = profile_stages.then(Instant::now);
        command_buffer.commit();
        if let Some(started) = final_commit_started {
            classic_command_buffer_commit_duration =
                classic_command_buffer_commit_duration.saturating_add(started.elapsed());
        }
        if split_command_buffers && codestream_payload_copy_dispatched {
            gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                stage: J2kResidentEncodeGpuStage::CodestreamPayloadCopy,
                command_buffer: command_buffer.clone(),
            });
        }

        if profile_stages {
            let mut prepare_command_buffer_ptrs = Vec::new();
            for tile in &prepared_tiles {
                let mut pushed_split_prepare = false;
                for (stage, command_buffer) in [
                    (
                        J2kResidentEncodeGpuStage::CoefficientDeinterleaveRct,
                        tile.prepare_deinterleave_rct_command_buffer.as_ref(),
                    ),
                    (
                        J2kResidentEncodeGpuStage::CoefficientDwt53,
                        tile.prepare_dwt53_command_buffer.as_ref(),
                    ),
                    (
                        J2kResidentEncodeGpuStage::CoefficientExtract,
                        tile.prepare_coefficient_extract_command_buffer.as_ref(),
                    ),
                ] {
                    if let Some(command_buffer) = command_buffer {
                        let ptr = command_buffer.as_ptr();
                        if prepare_command_buffer_ptrs.contains(&ptr) {
                            continue;
                        }
                        prepare_command_buffer_ptrs.push(ptr);
                        pushed_split_prepare = true;
                        gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                            stage,
                            command_buffer: command_buffer.clone(),
                        });
                    }
                }
                for command_buffer in &tile.prepare_dwt53_vertical_command_buffers {
                    let ptr = command_buffer.as_ptr();
                    if prepare_command_buffer_ptrs.contains(&ptr) {
                        continue;
                    }
                    prepare_command_buffer_ptrs.push(ptr);
                    pushed_split_prepare = true;
                    gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                        stage: J2kResidentEncodeGpuStage::CoefficientDwt53Vertical,
                        command_buffer: command_buffer.clone(),
                    });
                }
                for command_buffer in &tile.prepare_dwt53_horizontal_command_buffers {
                    let ptr = command_buffer.as_ptr();
                    if prepare_command_buffer_ptrs.contains(&ptr) {
                        continue;
                    }
                    prepare_command_buffer_ptrs.push(ptr);
                    pushed_split_prepare = true;
                    gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                        stage: J2kResidentEncodeGpuStage::CoefficientDwt53Horizontal,
                        command_buffer: command_buffer.clone(),
                    });
                }
                if pushed_split_prepare {
                    continue;
                }
                let ptr = tile.prepare_command_buffer.as_ptr();
                if prepare_command_buffer_ptrs.contains(&ptr) {
                    continue;
                }
                prepare_command_buffer_ptrs.push(ptr);
                gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                    stage: J2kResidentEncodeGpuStage::CoefficientPrep,
                    command_buffer: tile.prepare_command_buffer.clone(),
                });
            }
        }

        retained_command_buffers.extend(
            gpu_stage_command_buffers
                .iter()
                .map(|stage_command_buffer| stage_command_buffer.command_buffer.clone()),
        );
        for tile in prepared_tiles {
            if let Some(command_buffer) = tile.prepare_deinterleave_rct_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            if let Some(command_buffer) = tile.prepare_dwt53_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            retained_command_buffers.extend(tile.prepare_dwt53_vertical_command_buffers);
            retained_command_buffers.extend(tile.prepare_dwt53_horizontal_command_buffers);
            if let Some(command_buffer) = tile.prepare_coefficient_extract_command_buffer {
                retained_command_buffers.push(command_buffer);
            }
            retained_command_buffers.push(tile.prepare_command_buffer);
            retained_buffers.push(tile.coefficient_buffer);
            retained_buffers.push(tile.deinterleave_status_buffer);
            retained_buffers.extend(tile.plane_buffers);
            retained_buffers.extend(tile.scratch_buffers);
            retained_buffers.push(tile.coefficient_job_buffer);
            recyclable_private_buffers.extend(tile.recyclable_private_buffers);
        }
        retained_buffers.push(coefficient_buffer);
        retained_buffers.push(tier1_job_buffer);
        retained_buffers.push(tier1_output_buffer);
        retained_buffers.push(tier1_status_buffer);
        retained_buffers.push(tier1_segment_buffer);
        if let Some(buffer) = classic_tier1_raw_pack_buffer {
            retained_buffers.push(buffer);
        }
        if let Some(buffer) = classic_tier1_arithmetic_pack_buffer {
            retained_buffers.push(buffer);
        }
        retained_buffers.push(packet_resolution_buffer);
        retained_buffers.push(packet_subband_buffer);
        retained_buffers.push(resident_block_buffer);
        retained_buffers.push(packet_descriptor_buffer);
        retained_buffers.push(state_block_buffer);
        retained_buffers.push(packet_payload_copy_job_buffer);
        retained_buffers.push(header_buffer);
        retained_buffers.push(scratch_buffer);
        retained_buffers.push(packet_job_buffer);
        retained_buffers.push(packet_status_buffer.clone());
        retained_buffers.push(codestream_job_buffer);

        stage_stats.classic_tier1_setup_duration = classic_tier1_setup_duration;
        stage_stats.classic_block_encode_duration = classic_block_encode_duration;
        stage_stats.classic_packet_plan_duration = classic_packet_plan_duration;
        stage_stats.classic_packet_buffer_setup_duration = classic_packet_buffer_setup_duration;
        stage_stats.classic_command_buffer_commit_duration = classic_command_buffer_commit_duration;
        stage_stats.packet_block_prep_duration = packet_block_prep_duration;
        stage_stats.packetization_duration = packetization_duration;
        stage_stats.codestream_assembly_duration = codestream_assembly_duration;
        stage_stats.packet_payload_copy_job_capacity_total = packet_payload_copy_job_capacity_total;
        stage_stats.max_packet_payload_copy_jobs_per_tile = max_payload_copy_jobs_per_tile;
        stage_stats.packet_payload_copy_launched_stripe_count_total = packet_jobs
            .len()
            .saturating_mul(max_payload_copy_jobs_per_tile)
            .saturating_mul(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB as usize);
        stage_stats.tier1_output_capacity_total = tier1_output_capacity_total;
        stage_stats.max_tier1_output_capacity = max_tier1_output_capacity;
        stage_stats.tier1_segment_capacity_total = tier1_segment_capacity_total;
        stage_stats.max_tier1_segment_capacity_per_block = tier1_jobs
            .iter()
            .map(|job| job.segment_capacity as usize)
            .max()
            .unwrap_or(0);
        stage_stats.packet_output_capacity_total = packet_output_capacity_total;
        stage_stats.max_packet_output_capacity = max_packet_output_capacity_usize;
        stage_stats.codestream_payload_copy_launched_thread_count_total = 0;
        stage_stats.code_block_count = tier1_jobs.len();

        Ok(J2kPendingResidentLosslessCodestreamBatch {
            device: runtime.device.clone(),
            buffer: codestream_buffer,
            byte_offsets: codestream_offsets,
            capacities: codestream_capacities,
            status_buffer: codestream_status_buffer,
            packet_status_buffer,
            tier1_status_readback,
            classic_tier1_density_readback,
            classic_tier1_symbol_plan_readback,
            classic_tier1_pass_plan_readback,
            classic_tier1_token_emit_readback,
            classic_tier1_split_token_emit_readback,
            classic_gpu_token_pack_used: use_classic_gpu_token_pack
                || use_classic_split_gpu_token_pack
                || use_classic_split_mq_byte_gpu_token_pack,
            command_buffer: command_buffer.clone(),
            retained_command_buffers,
            _retained_buffers: retained_buffers,
            recyclable_private_buffers,
            recyclable_shared_buffers,
            gpu_stage_command_buffers,
            stage_stats,
            codestream_payload_copy_dispatched,
            status_stage: "J2K batched codestream assembly",
            length_error: "J2K Metal batched codestream output length exceeds usize",
            capacity_error: "J2K Metal batched codestream output length exceeds buffer",
        })
    })
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    params: J2kHtCleanupParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let status_buffer = runtime.device.new_buffer(
        size_of::<J2kHtStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        encoder,
        decoded,
        ht_output_word_count(
            params.output_offset,
            params.output_stride,
            params.width,
            params.height,
        )?,
    )?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kHtCleanupParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    dispatch_single_thread(encoder);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let status = unsafe { status_buffer.contents().cast::<J2kHtStatus>().read() };
    if status.code != J2K_HT_STATUS_OK {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, jobs);
    let status_buffer = runtime.device.new_buffer(
        (jobs.len().max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        encoder,
        decoded,
        ht_batch_output_word_count(jobs)?,
    )?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(jobs.len() as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: jobs.len() as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let statuses = unsafe {
        core::slice::from_raw_parts(status_buffer.contents().cast::<J2kHtStatus>(), jobs.len())
    };
    if let Some(status) = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_HT_STATUS_OK)
    {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    decoded_word_count: usize,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    dispatch_ht_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        decoded,
        &status_buffer,
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: job_count,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    decoded_word_count: usize,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    dispatch_ht_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        decoded,
        &status_buffer,
    );

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: job_count,
    })
}

#[cfg(target_os = "macos")]
fn dispatch_ht_cleanup_batched_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    status_buffer: &Buffer,
) {
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
        .thread_execution_width()
        .max(1)
        .min(job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn dispatch_ht_cleanup_repeated_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    base_job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
    decoded: &Buffer,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer = runtime.device.new_buffer(
        (total_job_count.max(1) * size_of::<J2kHtStatus>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let batch_count =
        total_job_count
            .checked_div(base_job_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect repeated base job count is zero".to_string(),
            })?;
    let decoded_word_count =
        output_plane_len
            .checked_mul(batch_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect repeated output span overflow".to_string(),
            })?;
    let repeated = J2kHtRepeatedBatchParams {
        job_count: u32::try_from(base_job_count).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated base job count exceeds u32".to_string(),
        })?,
        output_plane_len: u32::try_from(output_plane_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated output plane length exceeds u32".to_string(),
        })?,
        batch_count: u32::try_from(batch_count).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated batch count exceeds u32".to_string(),
        })?,
    };

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_repeated_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kHtRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(5, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(7, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(8, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_repeated_batched
        .thread_execution_width()
        .max(1)
        .min(base_job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: base_job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: total_job_count,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_code_block(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_classic_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        let batch_job = J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal coded payload exceeds u32".to_string(),
            })?,
            segment_offset: 0,
            segment_count: u32::try_from(job.segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment count exceeds u32".to_string(),
            })?,
            width: job.width,
            height: job.height,
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            missing_msbs: u32::from(job.missing_bit_planes),
            total_bitplanes: u32::from(job.total_bitplanes),
            roi_shift: u32::from(job.roi_shift),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            sub_band_type: match job.sub_band_type {
                signinum_j2k_native::J2kSubBandType::LowLow => 0,
                signinum_j2k_native::J2kSubBandType::HighLow => 1,
                signinum_j2k_native::J2kSubBandType::LowHigh => 2,
                signinum_j2k_native::J2kSubBandType::HighHigh => 3,
            },
            style_flags: classic_style_flags(job.style),
            strict: u32::from(job.strict),
            dequantization_step: job.dequantization_step,
        };
        let segments: Vec<_> = job
            .segments
            .iter()
            .map(|segment| J2kClassicSegment {
                data_offset: segment.data_offset,
                data_length: segment.data_length,
                start_coding_pass: u32::from(segment.start_coding_pass),
                end_coding_pass: u32::from(segment.end_coding_pass),
                use_arithmetic: u32::from(segment.use_arithmetic),
            })
            .collect();
        dispatch_classic_cleanup_batched(runtime, job.data, &[batch_job], &segments, &decoded)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_classic_cleanup_sub_band(
    job: J2kSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K Metal sub-band size overflow".to_string(),
        })?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "classic J2K Metal sub-band output slice is too small".to_string(),
        });
    }
    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let mut jobs = Vec::with_capacity(job.jobs.len());
        let mut coded_data = Vec::new();
        let mut segments = Vec::new();

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);
            let segment_offset = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
                message: "classic J2K Metal segment table exceeds u32".to_string(),
            })?;
            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "classic J2K Metal batched block lies outside sub-band bounds"
                        .to_string(),
                });
            }
            for segment in block.code_block.segments {
                let data_offset =
                    coded_offset
                        .checked_add(segment.data_offset)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "classic J2K Metal segment offset overflow".to_string(),
                        })?;
                segments.push(J2kClassicSegment {
                    data_offset,
                    data_length: segment.data_length,
                    start_coding_pass: u32::from(segment.start_coding_pass),
                    end_coding_pass: u32::from(segment.end_coding_pass),
                    use_arithmetic: u32::from(segment.use_arithmetic),
                });
            }
            jobs.push(J2kClassicCleanupBatchJob {
                coded_offset,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                segment_offset,
                segment_count: u32::try_from(block.code_block.segments.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "classic J2K Metal segment count exceeds u32".to_string(),
                    }
                })?,
                width: block.code_block.width,
                height: block.code_block.height,
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K Metal output offset overflow".to_string(),
                    })?,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                total_bitplanes: u32::from(block.code_block.total_bitplanes),
                roi_shift: u32::from(block.code_block.roi_shift),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                sub_band_type: match block.code_block.sub_band_type {
                    signinum_j2k_native::J2kSubBandType::LowLow => 0,
                    signinum_j2k_native::J2kSubBandType::HighLow => 1,
                    signinum_j2k_native::J2kSubBandType::LowHigh => 2,
                    signinum_j2k_native::J2kSubBandType::HighHigh => 3,
                },
                style_flags: classic_style_flags(block.code_block.style),
                strict: u32::from(block.code_block.strict),
                dequantization_step: block.code_block.dequantization_step,
            });
        }

        dispatch_classic_cleanup_batched(runtime, &coded_data, &jobs, &segments, &decoded)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_code_block(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = required_ht_output_len(job)?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal output slice is too small".to_string(),
        });
    }

    if job.width == 0 || job.height == 0 {
        return Ok(());
    }

    with_runtime(|runtime| {
        let params = J2kHtCleanupParams {
            width: job.width,
            height: job.height,
            coded_len: u32::try_from(job.data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal coded payload exceeds u32".to_string(),
            })?,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            missing_msbs: u32::from(job.missing_bit_planes),
            num_bitplanes: u32::from(job.num_bitplanes),
            number_of_coding_passes: u32::from(job.number_of_coding_passes),
            output_stride: u32::try_from(job.output_stride).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal output stride exceeds u32".to_string(),
            })?,
            output_offset: 0,
            dequantization_step: job.dequantization_step,
            stripe_causal: u32::from(job.stripe_causal),
        };
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        dispatch_ht_cleanup(runtime, job.data, params, &decoded)?;

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_ht_cleanup_sub_band(
    job: HtSubBandDecodeJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    let required_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal sub-band size overflow".to_string(),
        })?;
    if output.len() < required_len {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal sub-band output slice is too small".to_string(),
        });
    }

    if job.jobs.is_empty() {
        return Ok(());
    }

    with_runtime(|runtime| {
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let mut jobs = Vec::with_capacity(job.jobs.len());
        let mut coded_data = Vec::new();

        for block in job.jobs {
            let coded_offset = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batched coded payload exceeds u32".to_string(),
            })?;
            coded_data.extend_from_slice(block.code_block.data);

            jobs.push(J2kHtCleanupBatchJob {
                coded_offset,
                width: block.code_block.width,
                height: block.code_block.height,
                coded_len: u32::try_from(block.code_block.data.len()).map_err(|_| {
                    Error::MetalKernel {
                        message: "HTJ2K Metal coded payload exceeds u32".to_string(),
                    }
                })?,
                cleanup_length: block.code_block.cleanup_length,
                refinement_length: block.code_block.refinement_length,
                missing_msbs: u32::from(block.code_block.missing_bit_planes),
                num_bitplanes: u32::from(block.code_block.num_bitplanes),
                roi_shift: u32::from(block.code_block.roi_shift),
                number_of_coding_passes: u32::from(block.code_block.number_of_coding_passes),
                output_stride: job.width,
                output_offset: block
                    .output_y
                    .checked_mul(job.width)
                    .and_then(|row| row.checked_add(block.output_x))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal output offset overflow".to_string(),
                    })?,
                dequantization_step: block.code_block.dequantization_step,
                stripe_causal: u32::from(block.code_block.stripe_causal),
            });

            let end_x = block
                .output_x
                .checked_add(block.code_block.width)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block width overflow".to_string(),
                })?;
            let end_y = block
                .output_y
                .checked_add(block.code_block.height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batched block height overflow".to_string(),
                })?;
            if end_x > job.width || end_y > job.height {
                return Err(Error::MetalKernel {
                    message: "HTJ2K Metal batched block lies outside sub-band bounds".to_string(),
                });
            }
        }

        dispatch_ht_cleanup_batched(runtime, &coded_data, &jobs, &decoded)?;
        Ok(())
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{
        classic_batch_uses_plain_fast_path, classic_repeated_uses_plain_fast_path,
        crop_prepared_direct_grayscale_plan_to_output_region,
        decode_prepared_classic_sub_band_on_cpu, decode_scaled_to_surface,
        direct_tier1_input_buffer_prepares_for_test,
        execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test,
        execute_hybrid_cpu_tier1_direct_color_plan_batch,
        flattened_hybrid_cpu_decode_batches_for_test, hybrid_cpu_decode_inputs_for_test,
        hybrid_cpu_decode_worker_count, hybrid_cpu_decode_worker_inits_for_test,
        hybrid_repeated_output_blits_for_test, hybrid_stacked_component_batches_for_test,
        j2k_pack_kernel_name_for, j2k_pack_scale_arrays, one_d_threads_per_group, output_shape_for,
        prepare_direct_color_plan, prepare_direct_color_plan_for_cpu_upload,
        prepare_direct_grayscale_plan, prepared_direct_color_tier1_input_count,
        prepared_direct_grayscale_plan_compute_encoder_count, prepared_idwt_output_len,
        prepared_repeated_direct_ht_cleanup_dispatch_count,
        repeated_gray_store_is_contiguous_full_surface,
        reset_direct_tier1_input_buffer_prepares_for_test,
        reset_flattened_hybrid_cpu_decode_batches_for_test,
        reset_hybrid_cpu_decode_inputs_for_test, reset_hybrid_cpu_decode_worker_inits_for_test,
        reset_hybrid_repeated_output_blits_for_test,
        reset_hybrid_stacked_component_batches_for_test, reset_shared_buffer_pool_misses_for_test,
        runtime_initialization_error, shared_buffer_pool_misses_for_test,
        should_flatten_hybrid_cpu_tier1_color_batch, supports_stacked_direct_component_plane_batch,
        two_d_threads_per_group, with_runtime_for_device, J2kClassicCleanupBatchJob,
        J2kClassicSegment, J2kRepeatedGrayStoreParams, MetalRuntime, PreparedClassicSubBand,
        PreparedDirectColorPlan, PreparedDirectGrayscaleStep,
    };
    use metal::Device;
    use signinum_core::PixelFormat;
    use signinum_j2k_native::{
        decode_j2k_sub_band_scalar, encode, encode_htj2k, ColorSpace as NativeColorSpace,
        DecodeSettings, DecoderContext, EncodeOptions, Image, J2kCodeBlockBatchJob,
        J2kCodeBlockDecodeJob, J2kDirectGrayscaleStep as NativeDirectGrayscaleStep,
        J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan, J2kSubBandDecodeJob, J2kWaveletTransform,
    };
    use std::sync::{Arc, Mutex};

    static HYBRID_COUNTER_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn rgb16_with_alpha_is_rejected() {
        let runtime = MetalRuntime::new().expect("Metal runtime");
        let result = output_shape_for(
            &NativeColorSpace::RGB,
            true,
            4,
            PixelFormat::Rgb16,
            &runtime,
        );
        assert!(result.is_err(), "RGBA input must not silently map to Rgb16");
    }

    #[test]
    fn runtime_initialization_error_classifies_null_queue_as_unavailable() {
        assert!(matches!(
            runtime_initialization_error("Metal command queue is unavailable on this host"),
            crate::Error::MetalUnavailable
        ));
    }

    #[test]
    fn classic_encode_output_capacity_keeps_conservative_default() {
        let capacity =
            super::classic_encode_output_capacity(64, 64, 11).expect("classic output capacity");

        assert_eq!(capacity, 64 * 64 * 11 * 8 + 4097);
    }

    #[test]
    fn classic_encode_segment_capacity_uses_coding_style_bound() {
        assert_eq!(super::classic_encode_segment_capacity(0, 16), 1);
        assert_eq!(
            super::classic_encode_segment_capacity(
                super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
                9,
            ),
            11
        );
        assert_eq!(
            super::classic_encode_segment_capacity(
                super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
                16,
            ),
            25
        );
        assert_eq!(
            super::classic_encode_segment_capacity(
                super::J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS,
                16,
            ),
            46
        );
    }

    #[test]
    fn two_d_threads_per_group_clamps_empty_pipeline_limits() {
        let threads = two_d_threads_per_group(0, 0);

        assert_eq!((threads.width, threads.height, threads.depth), (1, 1, 1));
    }

    #[test]
    fn one_d_threads_per_group_clamps_empty_pipeline_width() {
        let threads = one_d_threads_per_group(0);

        assert_eq!((threads.width, threads.height, threads.depth), (1, 1, 1));
    }

    #[test]
    fn two_d_threads_per_group_preserves_simd_width_and_derives_height() {
        let threads = two_d_threads_per_group(32, 1024);

        assert_eq!((threads.width, threads.height, threads.depth), (32, 32, 1));
    }

    #[test]
    fn classic_tier1_pass_class_counts_split_bypass_pass_types() {
        let counts = super::classic_tier1_pass_class_counts(
            23,
            super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
        );

        assert_eq!(counts.arithmetic, 14);
        assert_eq!(counts.raw, 9);
        assert_eq!(counts.cleanup, 8);
        assert_eq!(counts.sigprop, 8);
        assert_eq!(counts.magref, 7);
        assert_eq!(counts.arithmetic_cleanup, 8);
        assert_eq!(counts.arithmetic_sigprop, 3);
        assert_eq!(counts.arithmetic_magref, 3);
        assert_eq!(counts.raw_sigprop, 5);
        assert_eq!(counts.raw_magref, 4);
    }

    #[test]
    fn classic_tier1_pass_class_counts_style0_stays_arithmetic() {
        let counts = super::classic_tier1_pass_class_counts(5, 0);

        assert_eq!(counts.arithmetic, 5);
        assert_eq!(counts.raw, 0);
        assert_eq!(counts.cleanup, 2);
        assert_eq!(counts.sigprop, 2);
        assert_eq!(counts.magref, 1);
        assert_eq!(counts.arithmetic_cleanup, 2);
        assert_eq!(counts.arithmetic_sigprop, 2);
        assert_eq!(counts.arithmetic_magref, 1);
        assert_eq!(counts.raw_sigprop, 0);
        assert_eq!(counts.raw_magref, 0);
    }

    #[test]
    fn classic_tier1_scan_estimates_multiply_passes_by_block_area() {
        let pass_counts = super::classic_tier1_pass_class_counts(
            23,
            super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
        );
        let mut stats = super::J2kResidentEncodeStageStats::default();

        super::accumulate_classic_tier1_scan_estimates(&mut stats, pass_counts, 32 * 32);

        assert_eq!(stats.tier1_full_scan_coeff_visit_count_total, 23 * 1024);
        assert_eq!(
            stats.tier1_arithmetic_scan_coeff_visit_count_total,
            14 * 1024
        );
        assert_eq!(stats.tier1_raw_scan_coeff_visit_count_total, 9 * 1024);
        assert_eq!(stats.tier1_cleanup_scan_coeff_visit_count_total, 8 * 1024);
        assert_eq!(stats.tier1_sigprop_scan_coeff_visit_count_total, 8 * 1024);
        assert_eq!(stats.tier1_magref_scan_coeff_visit_count_total, 7 * 1024);
        assert_eq!(stats.max_tier1_full_scan_coeff_visits_per_block, 23 * 1024);
    }

    #[test]
    fn classic_packet_output_capacity_uses_raw_sample_bound_when_smaller() {
        let codestream = super::J2kLosslessCodestreamAssemblyJob {
            width: 512,
            height: 512,
            num_components: 3,
            bit_depth: 8,
            signed: false,
            num_decomposition_levels: 3,
            use_mct: true,
            guard_bits: 2,
            code_block_width_exp: 4,
            code_block_height_exp: 4,
            progression_order: signinum_j2k_native::EncodeProgressionOrder::Lrcp,
            write_tlm: false,
            block_coding_mode: super::J2kLosslessCodestreamBlockCodingMode::Classic,
        };
        let header_capacity = 1024 * 256 + 4096;
        let conservative_capacity = 12 * 1024 * 1024;
        let packet_descriptor_count = 3;

        let capacity = super::classic_packet_output_capacity(
            conservative_capacity,
            header_capacity,
            packet_descriptor_count,
            codestream,
        )
        .expect("classic packet capacity");

        let raw_bytes = 512 * 512 * 3;
        let descriptor_slack = packet_descriptor_count * 256;
        assert_eq!(
            capacity,
            raw_bytes + header_capacity + descriptor_slack + 64 * 1024
        );

        let tiny_tier1_capacity = 4096;
        let clamped = super::classic_packet_output_capacity(
            tiny_tier1_capacity,
            header_capacity,
            packet_descriptor_count,
            codestream,
        )
        .expect("classic packet capacity");
        let conservative_packet_capacity =
            tiny_tier1_capacity + header_capacity * packet_descriptor_count + 1024;
        assert_eq!(clamped, conservative_packet_capacity);
    }

    #[test]
    fn ht_encode_output_capacity_scales_with_code_block_area() {
        let max_block = super::ht_encode_output_capacity(128, 128).expect("max HT output capacity");
        assert_eq!(max_block, super::J2K_HT_ENCODE_BASE_OUTPUT_SIZE);

        let smaller_block =
            super::ht_encode_output_capacity(32, 32).expect("scaled HT output capacity");
        assert!(smaller_block < max_block / 2);
        assert!(smaller_block >= 8192);
    }

    #[test]
    fn classic_encode_pipeline_kind_prefers_style0_32_for_resident_jobs() {
        let jobs = [super::J2kClassicEncodeBatchJob {
            width: 32,
            height: 32,
            style_flags: 0,
            ..super::J2kClassicEncodeBatchJob::default()
        }];

        assert_eq!(
            super::classic_encode_code_blocks_pipeline_kind(&jobs),
            super::J2kClassicEncodePipelineKind::Style0_32
        );
    }

    #[test]
    fn classic_encode_pipeline_kind_prefers_bypass_32_for_resident_jobs() {
        let jobs = [super::J2kClassicEncodeBatchJob {
            width: 32,
            height: 32,
            style_flags: super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
            total_bitplanes: 31,
            ..super::J2kClassicEncodeBatchJob::default()
        }];

        assert_eq!(
            super::classic_encode_code_blocks_pipeline_kind(&jobs),
            super::J2kClassicEncodePipelineKind::Bypass32
        );
    }

    #[test]
    fn classic_encode_pipeline_kind_prefers_bypass_u16_32_for_low_bitplane_resident_jobs() {
        let jobs = [super::J2kClassicEncodeBatchJob {
            width: 32,
            height: 32,
            style_flags: super::J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
            total_bitplanes: 16,
            ..super::J2kClassicEncodeBatchJob::default()
        }];

        assert_eq!(
            super::classic_encode_code_blocks_pipeline_kind(&jobs),
            super::J2kClassicEncodePipelineKind::BypassU16_32
        );
    }

    #[test]
    fn with_runtime_for_device_reuses_cached_runtime_for_device() {
        let Some(device) = Device::system_default() else {
            return;
        };

        let first = with_runtime_for_device(&device, |runtime| Ok(std::ptr::from_ref(runtime)))
            .expect("first Metal runtime");
        let second = with_runtime_for_device(&device, |runtime| Ok(std::ptr::from_ref(runtime)))
            .expect("second Metal runtime");

        assert_eq!(first, second);
    }

    #[test]
    fn runtime_reuses_recycled_shared_buffers() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let runtime = MetalRuntime::new_with_device(&device).expect("Metal runtime");

        reset_shared_buffer_pool_misses_for_test();
        let first = runtime.take_shared_buffer(64);
        runtime.recycle_shared_buffer(64, first);
        let _second = runtime.take_shared_buffer(64);

        assert_eq!(
            shared_buffer_pool_misses_for_test(),
            1,
            "recycled shared metadata buffers should be reused instead of allocating again"
        );
    }

    #[test]
    fn j2k_pack_selects_specialized_kernels_for_wsi_formats() {
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::Gray, false, 1, PixelFormat::Gray8),
            Some("j2k_pack_gray8")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, false, 3, PixelFormat::Rgb8),
            Some("j2k_pack_rgb8")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, false, 3, PixelFormat::Rgba8),
            Some("j2k_pack_rgb_opaque_rgba8")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, true, 4, PixelFormat::Rgba8),
            Some("j2k_pack_rgba8")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::Gray, false, 1, PixelFormat::Gray16),
            Some("j2k_pack_gray16")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, false, 3, PixelFormat::Rgb16),
            Some("j2k_pack_rgb16")
        );
        assert_eq!(
            j2k_pack_kernel_name_for(&NativeColorSpace::RGB, true, 4, PixelFormat::Rgb16),
            None,
            "RGBA input must not silently drop alpha when packing RGB16"
        );
    }

    #[test]
    fn j2k_pack_precomputes_scale_factors_on_cpu() {
        let (max_values, u8_scales, u16_scales) = j2k_pack_scale_arrays([8, 12, 16, 0]);

        assert_f32_near(max_values[0], 255.0);
        assert_f32_near(max_values[1], 4095.0);
        assert_f32_near(max_values[2], 65_535.0);
        assert_f32_near(max_values[3], 1.0);
        assert_f32_near(u8_scales[0], 1.0);
        assert_f32_near(u8_scales[1], 255.0 / 4095.0);
        assert_f32_near(u16_scales[0], 257.0);
        assert_f32_near(u16_scales[1], 1.0);
        assert_f32_near(u16_scales[2], 1.0);
        assert_f32_near(u16_scales[3], 65_535.0);
    }

    fn assert_f32_near(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= f32::EPSILON,
            "expected {actual} to be within f32 epsilon of {expected}"
        );
    }

    #[test]
    fn scaled_htj2k_decode_runs_through_metal_compute_path() {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8");

        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((2, 2)),
                ..DecodeSettings::default()
            },
        )
        .expect("image");
        let host = image.decode().expect("host scaled decode");

        let surface = decode_scaled_to_surface(
            &bytes,
            (4, 4),
            PixelFormat::Gray8,
            signinum_core::Downscale::Half,
        )
        .expect("metal scaled decode");
        assert_eq!(surface.as_bytes(), host.as_slice());
    }

    #[test]
    fn prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let ht_subband_steps = plan
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    step,
                    signinum_j2k_native::J2kDirectGrayscaleStep::HtSubBand(_)
                )
            })
            .count();
        assert!(
            ht_subband_steps > 1,
            "fixture must exercise multiple HT sub-band cleanup steps"
        );

        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        assert_eq!(
            prepared.ht_groups.len(),
            1,
            "single-tile HTJ2K direct decode should group adjacent HT sub-bands into one cleanup dispatch"
        );
        assert_eq!(prepared.ht_groups[0].members.len(), ht_subband_steps);
        assert!(matches!(
            prepared.steps[prepared.ht_groups[0].start_step],
            PreparedDirectGrayscaleStep::HtSubBand(_)
        ));
    }

    #[test]
    fn grouped_ht_direct_plan_uses_one_group_coded_arena() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");

        reset_direct_tier1_input_buffer_prepares_for_test();
        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        assert_eq!(
            prepared.ht_groups.len(),
            1,
            "fixture must exercise one grouped HT dispatch"
        );
        let group = &prepared.ht_groups[0];
        assert!(!group.coded_arena.data.is_empty());
        assert_eq!(
            direct_tier1_input_buffer_prepares_for_test(),
            2,
            "grouped HT dispatch should prepare one coded arena buffer and one job buffer"
        );

        for step in &prepared.steps[group.start_step..group.end_step] {
            let PreparedDirectGrayscaleStep::HtSubBand(sub_band) = step else {
                panic!("HT group should only span HT sub-band steps");
            };
            assert!(sub_band.coded_buffer.is_none());
            assert!(sub_band.jobs_buffer.is_none());
        }
    }

    #[test]
    fn prepared_classic_sub_band_decodes_on_cpu_for_hybrid_upload() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode classic gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        let native_sub_band = first_native_classic_sub_band(&plan);
        let prepared_sub_band = first_prepared_classic_sub_band(&prepared);

        let expected = decode_native_classic_sub_band(native_sub_band);
        let actual = decode_prepared_classic_sub_band_on_cpu(prepared_sub_band)
            .expect("prepared CPU decode");

        assert_eq!(actual, expected);
    }

    #[test]
    fn cpu_upload_color_prepare_skips_tier1_metal_input_buffers() {
        if Device::system_default().is_none() {
            eprintln!("skipping CPUUpload prepare test: no Metal device");
            return;
        }

        let pixels = signinum_test_support::gradient_u8(32, 32, 3);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_color_plan_with_context(&mut context)
            .expect("direct color plan");

        reset_direct_tier1_input_buffer_prepares_for_test();
        let metal_prepared = prepare_direct_color_plan(&plan).expect("Metal prepared color plan");
        assert_eq!(metal_prepared.component_plans.len(), 3);
        assert!(
            direct_tier1_input_buffer_prepares_for_test() > 0,
            "normal Metal preparation should build Tier-1 input buffers"
        );

        reset_direct_tier1_input_buffer_prepares_for_test();
        let cpu_upload_prepared =
            prepare_direct_color_plan_for_cpu_upload(&plan).expect("CPUUpload prepared color plan");
        assert_eq!(cpu_upload_prepared.component_plans.len(), 3);
        assert_eq!(
            direct_tier1_input_buffer_prepares_for_test(),
            0,
            "CPUUpload preparation should keep coded Tier-1 payloads on CPU and skip Metal input buffers"
        );
    }

    fn first_native_classic_sub_band(
        plan: &signinum_j2k_native::J2kDirectGrayscalePlan,
    ) -> &J2kOwnedSubBandPlan {
        plan.steps
            .iter()
            .find_map(|step| match step {
                NativeDirectGrayscaleStep::ClassicSubBand(sub_band) => Some(sub_band),
                _ => None,
            })
            .expect("classic sub-band step")
    }

    fn first_prepared_classic_sub_band(
        plan: &super::PreparedDirectGrayscalePlan,
    ) -> &PreparedClassicSubBand {
        plan.steps
            .iter()
            .find_map(|step| match step {
                PreparedDirectGrayscaleStep::ClassicSubBand(sub_band) => Some(sub_band),
                _ => None,
            })
            .expect("prepared classic sub-band step")
    }

    fn decode_native_classic_sub_band(plan: &J2kOwnedSubBandPlan) -> Vec<f32> {
        let mut output = vec![0.0_f32; plan.width as usize * plan.height as usize];
        let jobs = plan
            .jobs
            .iter()
            .map(|job| J2kCodeBlockBatchJob {
                output_x: job.output_x,
                output_y: job.output_y,
                code_block: native_classic_job(job),
            })
            .collect::<Vec<_>>();
        decode_j2k_sub_band_scalar(
            J2kSubBandDecodeJob {
                width: plan.width,
                height: plan.height,
                jobs: &jobs,
            },
            &mut output,
        )
        .expect("native scalar classic sub-band decode");
        output
    }

    fn native_classic_job(job: &J2kOwnedCodeBlockBatchJob) -> J2kCodeBlockDecodeJob<'_> {
        J2kCodeBlockDecodeJob {
            data: &job.data,
            segments: &job.segments,
            width: job.width,
            height: job.height,
            output_stride: job.output_stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            total_bitplanes: job.total_bitplanes,
            roi_shift: job.roi_shift,
            sub_band_type: job.sub_band_type,
            style: job.style,
            strict: job.strict,
            dequantization_step: job.dequantization_step,
        }
    }

    #[test]
    fn prepared_ht_direct_plan_encodes_full_decode_in_one_compute_encoder() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");

        assert_eq!(
            prepared_direct_grayscale_plan_compute_encoder_count(&prepared, PixelFormat::Gray8),
            1,
            "prepared single-tile direct decode should keep cleanup, IDWT, and grayscale store in one compute encoder"
        );
    }

    #[test]
    fn repeated_prepared_ht_direct_plan_groups_cleanup_subbands_before_idwt() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let ht_subband_steps = plan
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    step,
                    signinum_j2k_native::J2kDirectGrayscaleStep::HtSubBand(_)
                )
            })
            .count();
        assert!(
            ht_subband_steps > 1,
            "fixture must exercise multiple HT sub-band cleanup steps"
        );

        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        assert_eq!(
            prepared_repeated_direct_ht_cleanup_dispatch_count(&prepared),
            1,
            "repeated HTJ2K WSI tile batches should group adjacent sub-band cleanups like the single-tile path"
        );
    }

    #[test]
    fn distinct_prepared_ht_direct_plans_support_stacked_component_batch() {
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes_a = encode_htj2k(&(0..64).collect::<Vec<u8>>(), 8, 8, 1, 8, false, &options)
            .expect("encode first ht gray8");
        let bytes_b = encode_htj2k(
            &(0..64).rev().collect::<Vec<u8>>(),
            8,
            8,
            1,
            8,
            false,
            &options,
        )
        .expect("encode second ht gray8");
        let image_a = Image::new(&bytes_a, &DecodeSettings::default()).expect("first image");
        let image_b = Image::new(&bytes_b, &DecodeSettings::default()).expect("second image");
        let mut context_a = DecoderContext::default();
        let mut context_b = DecoderContext::default();
        let plan_a = image_a
            .build_direct_grayscale_plan_with_context(&mut context_a)
            .expect("first direct plan");
        let plan_b = image_b
            .build_direct_grayscale_plan_with_context(&mut context_b)
            .expect("second direct plan");
        let prepared_a = prepare_direct_grayscale_plan(&plan_a).expect("first prepared plan");
        let prepared_b = prepare_direct_grayscale_plan(&plan_b).expect("second prepared plan");

        assert!(
            supports_stacked_direct_component_plane_batch(&[&prepared_a, &prepared_b]),
            "distinct same-shape HTJ2K grayscale plans should be eligible for one stacked batch graph"
        );
    }

    #[test]
    fn hybrid_rgb8_batch_uses_stacked_component_graph() {
        let pixels = signinum_test_support::gradient_u8(32, 32, 3);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_color_plan_with_context(&mut context)
            .expect("direct color plan");
        let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
        let _guard = HYBRID_COUNTER_TEST_LOCK
            .lock()
            .expect("hybrid counter lock");
        reset_hybrid_stacked_component_batches_for_test();
        reset_hybrid_cpu_decode_worker_inits_for_test();

        let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
            &[prepared.clone(), prepared],
            PixelFormat::Rgb8,
        )
        .expect("hybrid RGB8 batch");

        assert_eq!(surfaces.len(), 2);
        assert!(
            hybrid_stacked_component_batches_for_test() >= 3,
            "hybrid RGB batch should stack each component plane instead of encoding each tile/component serially"
        );
        assert!(
            hybrid_cpu_decode_worker_inits_for_test() > 0,
            "hybrid RGB batch should use worker-local CPU decode scratch instead of per-input decode/flatten"
        );
    }

    #[test]
    fn hybrid_rgb8_repeated_batch_decodes_shared_tier1_inputs_once() {
        let pixels = signinum_test_support::gradient_u8(32, 32, 3);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_color_plan_with_context(&mut context)
            .expect("direct color plan");
        let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
        let unique_tier1_inputs = prepared_direct_color_tier1_input_count(&prepared);
        assert!(
            unique_tier1_inputs > 0,
            "fixture should have Tier-1 inputs to decode"
        );
        let _guard = HYBRID_COUNTER_TEST_LOCK
            .lock()
            .expect("hybrid counter lock");
        reset_hybrid_cpu_decode_inputs_for_test();

        let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
            &[prepared.clone(), prepared.clone(), prepared],
            PixelFormat::Rgb8,
        )
        .expect("hybrid repeated RGB8 batch");

        assert_eq!(surfaces.len(), 3);
        assert_eq!(
            hybrid_cpu_decode_inputs_for_test(),
            unique_tier1_inputs,
            "repeated RGB hybrid batches should decode each shared coefficient input once, not once per output surface"
        );
    }

    #[test]
    fn hybrid_rgb8_reused_plan_caches_cpu_tier1_inputs_across_calls() {
        let pixels = signinum_test_support::gradient_u8(32, 32, 3);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_color_plan_with_context(&mut context)
            .expect("direct color plan");
        let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
        let unique_tier1_inputs = prepared_direct_color_tier1_input_count(&prepared);
        assert!(
            unique_tier1_inputs > 0,
            "fixture should have Tier-1 inputs to decode"
        );
        let _guard = HYBRID_COUNTER_TEST_LOCK
            .lock()
            .expect("hybrid counter lock");
        reset_hybrid_cpu_decode_inputs_for_test();

        for _ in 0..2 {
            let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
                &[prepared.clone(), prepared.clone()],
                PixelFormat::Rgb8,
            )
            .expect("hybrid repeated RGB8 batch");
            assert_eq!(surfaces.len(), 2);
        }

        assert_eq!(
            hybrid_cpu_decode_inputs_for_test(),
            unique_tier1_inputs,
            "reusing the same RGB hybrid plan across calls should reuse decoded CPU Tier-1 coefficients"
        );
    }

    #[test]
    fn hybrid_rgb8_repeated_batch_decodes_once_and_blits_distinct_outputs() {
        let pixels = signinum_test_support::gradient_u8(32, 32, 3);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_color_plan_with_context(&mut context)
            .expect("direct color plan");
        let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
        let _guard = HYBRID_COUNTER_TEST_LOCK
            .lock()
            .expect("hybrid counter lock");
        reset_hybrid_repeated_output_blits_for_test();

        let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
            &[
                prepared.clone(),
                prepared.clone(),
                prepared.clone(),
                prepared,
            ],
            PixelFormat::Rgb8,
        )
        .expect("hybrid repeated RGB8 batch");

        assert_eq!(surfaces.len(), 4);
        let surface_bytes = surfaces[0].as_bytes().len();
        let offsets = surfaces
            .iter()
            .map(|surface| surface.metal_buffer().expect("resident Metal surface").1)
            .collect::<Vec<_>>();
        assert_eq!(
            offsets,
            (0..surfaces.len())
                .map(|index| index * surface_bytes)
                .collect::<Vec<_>>(),
            "repeated outputs must retain distinct Metal buffer offsets"
        );
        for surface in &surfaces[1..] {
            assert_eq!(
                surface.as_bytes(),
                surfaces[0].as_bytes(),
                "repeated outputs should remain byte-identical"
            );
        }
        assert_eq!(
            hybrid_repeated_output_blits_for_test(),
            2,
            "repeated RGB hybrid batches should duplicate packed output surfaces with logarithmic Metal blit ranges"
        );
    }

    #[test]
    fn hybrid_rgb8_distinct_batch_keeps_tier1_inputs_separate() {
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes_a = encode(
            &signinum_test_support::gradient_variant_u8(32, 32, 3, 0),
            32,
            32,
            3,
            8,
            false,
            &options,
        )
        .expect("encode first rgb8");
        let bytes_b = encode(
            &signinum_test_support::gradient_variant_u8(32, 32, 3, 7),
            32,
            32,
            3,
            8,
            false,
            &options,
        )
        .expect("encode second rgb8");
        let image_a = Image::new(&bytes_a, &DecodeSettings::default()).expect("first image");
        let image_b = Image::new(&bytes_b, &DecodeSettings::default()).expect("second image");
        let mut context_a = DecoderContext::default();
        let mut context_b = DecoderContext::default();
        let plan_a = image_a
            .build_direct_color_plan_with_context(&mut context_a)
            .expect("first direct color plan");
        let plan_b = image_b
            .build_direct_color_plan_with_context(&mut context_b)
            .expect("second direct color plan");
        let prepared_a = Arc::new(prepare_direct_color_plan(&plan_a).expect("first prepared"));
        let prepared_b = Arc::new(prepare_direct_color_plan(&plan_b).expect("second prepared"));
        let expected_inputs = prepared_direct_color_tier1_input_count(&prepared_a)
            + prepared_direct_color_tier1_input_count(&prepared_b);
        let _guard = HYBRID_COUNTER_TEST_LOCK
            .lock()
            .expect("hybrid counter lock");
        reset_hybrid_cpu_decode_inputs_for_test();

        let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
            &[prepared_a, prepared_b],
            PixelFormat::Rgb8,
        )
        .expect("hybrid distinct RGB8 batch");

        assert_eq!(surfaces.len(), 2);
        assert_ne!(
            surfaces[0].as_bytes(),
            surfaces[1].as_bytes(),
            "distinct RGB inputs must not reuse the first tile's decoded coefficients"
        );
        assert_eq!(
            hybrid_cpu_decode_inputs_for_test(),
            expected_inputs,
            "distinct RGB hybrid batches should decode each tile's own Tier-1 inputs"
        );
    }

    #[test]
    fn hybrid_rgb8_flattened_cpu_tier1_batch_uses_one_decode_queue() {
        let pixels_a = signinum_test_support::gradient_variant_u8(32, 32, 3, 0);
        let pixels_b = signinum_test_support::gradient_variant_u8(32, 32, 3, 11);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes_a = encode(&pixels_a, 32, 32, 3, 8, false, &options).expect("encode first rgb8");
        let bytes_b = encode(&pixels_b, 32, 32, 3, 8, false, &options).expect("encode second rgb8");
        let image_a = Image::new(&bytes_a, &DecodeSettings::default()).expect("first image");
        let image_b = Image::new(&bytes_b, &DecodeSettings::default()).expect("second image");
        let mut context_a = DecoderContext::default();
        let mut context_b = DecoderContext::default();
        let plan_a = image_a
            .build_direct_color_plan_with_context(&mut context_a)
            .expect("first direct color plan");
        let plan_b = image_b
            .build_direct_color_plan_with_context(&mut context_b)
            .expect("second direct color plan");
        let prepared_a = Arc::new(prepare_direct_color_plan(&plan_a).expect("first prepared"));
        let prepared_b = Arc::new(prepare_direct_color_plan(&plan_b).expect("second prepared"));
        let expected_inputs = prepared_direct_color_tier1_input_count(&prepared_a)
            + prepared_direct_color_tier1_input_count(&prepared_b);
        let _guard = HYBRID_COUNTER_TEST_LOCK
            .lock()
            .expect("hybrid counter lock");
        reset_hybrid_cpu_decode_inputs_for_test();
        reset_flattened_hybrid_cpu_decode_batches_for_test();

        let surfaces = execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test(
            &[prepared_a, prepared_b],
            PixelFormat::Rgb8,
        )
        .expect("flattened hybrid distinct RGB8 batch");

        assert_eq!(surfaces.len(), 2);
        assert_ne!(
            surfaces[0].as_bytes(),
            surfaces[1].as_bytes(),
            "flattened distinct RGB hybrid batches must keep each tile's coefficients separate"
        );
        assert_eq!(
            hybrid_cpu_decode_inputs_for_test(),
            expected_inputs,
            "flattened RGB hybrid batches should still decode every distinct Tier-1 input"
        );
        assert_eq!(
            flattened_hybrid_cpu_decode_batches_for_test(),
            1,
            "flattened RGB hybrid should collect Tier-1 work into one CPU decode queue"
        );
    }

    #[test]
    fn flattened_cpu_tier1_default_gate_targets_large_distinct_batches_only() {
        fn color_plan(width: u32, height: u32) -> Arc<PreparedDirectColorPlan> {
            Arc::new(PreparedDirectColorPlan {
                dimensions: (width, height),
                bit_depths: [8, 8, 8],
                mct: true,
                transform: J2kWaveletTransform::Reversible53,
                component_plans: Vec::new(),
            })
        }

        let repeated = vec![color_plan(1024, 1024); 16];
        assert!(
            !should_flatten_hybrid_cpu_tier1_color_batch(&repeated),
            "repeated RGB batches already win through shared Tier-1 decode and should not use the flattened distinct scheduler"
        );

        let small_distinct = (0..16).map(|_| color_plan(256, 256)).collect::<Vec<_>>();
        assert!(
            !should_flatten_hybrid_cpu_tier1_color_batch(&small_distinct),
            "small RGB batches measured slower with flattened Tier-1 and should stay on the grouped path"
        );

        let large_distinct = (0..16).map(|_| color_plan(1024, 1024)).collect::<Vec<_>>();
        assert!(
            should_flatten_hybrid_cpu_tier1_color_batch(&large_distinct),
            "large distinct RGB explicit hybrid batches measured faster with flattened Tier-1"
        );
    }

    #[test]
    fn hybrid_cpu_decode_worker_count_allows_two_way_small_batch_parallelism() {
        let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        if available < 2 {
            return;
        }

        assert_eq!(
            hybrid_cpu_decode_worker_count(2),
            2,
            "two independent hybrid CPU Tier-1 inputs should be able to use two workers"
        );
    }

    #[test]
    fn cropped_region_scaled_ht_direct_plan_prunes_codeblocks_outside_output_roi() {
        let mut pixels = Vec::with_capacity(256 * 256);
        for y in 0..256u32 {
            for x in 0..256u32 {
                pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            }
        }
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 3,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        let bytes =
            encode_htj2k(&pixels, 256, 256, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((64, 64)),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let mut prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        let full_jobs = prepared_direct_grayscale_ht_job_count(&prepared);
        assert!(
            full_jobs > 8,
            "fixture should have multiple HT code-block jobs"
        );

        crop_prepared_direct_grayscale_plan_to_output_region(
            &mut prepared,
            signinum_core::Rect {
                x: 24,
                y: 24,
                w: 8,
                h: 8,
            },
        )
        .expect("crop direct plan");
        let cropped_jobs = prepared_direct_grayscale_ht_job_count(&prepared);

        assert!(
            cropped_jobs > 0 && cropped_jobs < full_jobs,
            "cropped ROI should prune HT code-block jobs; full={full_jobs}, cropped={cropped_jobs}"
        );
    }

    #[test]
    fn cropped_region_scaled_ht_direct_plan_compacts_coded_payloads() {
        let mut pixels = Vec::with_capacity(256 * 256);
        for y in 0..256u32 {
            for x in 0..256u32 {
                pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            }
        }
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 3,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        let bytes =
            encode_htj2k(&pixels, 256, 256, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((64, 64)),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let mut prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        let full_bytes = prepared_direct_grayscale_ht_coded_byte_count(&prepared);
        assert!(full_bytes > 0, "fixture should carry HT coded payloads");

        crop_prepared_direct_grayscale_plan_to_output_region(
            &mut prepared,
            signinum_core::Rect {
                x: 24,
                y: 24,
                w: 8,
                h: 8,
            },
        )
        .expect("crop direct plan");
        let cropped_bytes = prepared_direct_grayscale_ht_coded_byte_count(&prepared);

        assert!(
            cropped_bytes > 0 && cropped_bytes < full_bytes,
            "cropped ROI should compact HT coded bytes; full={full_bytes}, cropped={cropped_bytes}"
        );
    }

    #[test]
    fn cropped_region_scaled_ht_direct_plan_reduces_idwt_output_work() {
        let mut pixels = Vec::with_capacity(128 * 128);
        for y in 0..128u32 {
            for x in 0..128u32 {
                pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            }
        }
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 3,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        let bytes =
            encode_htj2k(&pixels, 128, 128, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((32, 32)),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let mut prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        let full_samples = prepared_direct_grayscale_idwt_output_sample_count(&prepared);

        crop_prepared_direct_grayscale_plan_to_output_region(
            &mut prepared,
            signinum_core::Rect {
                x: 10,
                y: 10,
                w: 4,
                h: 4,
            },
        )
        .expect("crop direct plan");
        let cropped_samples = prepared_direct_grayscale_idwt_output_sample_count(&prepared);

        assert!(
            cropped_samples > 0 && cropped_samples < full_samples,
            "cropped ROI should reduce IDWT output work; full={full_samples}, cropped={cropped_samples}"
        );
    }

    #[test]
    fn cropped_region_ht_direct_plan_keeps_idwt_windows_bounded() {
        let mut pixels = Vec::with_capacity(256 * 256);
        for y in 0..256u32 {
            for x in 0..256u32 {
                pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            }
        }
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 3,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        let bytes =
            encode_htj2k(&pixels, 256, 256, 1, 8, false, &options).expect("encode ht gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let mut prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        let idwt_levels = prepared_direct_grayscale_idwt_full_and_prepared_lens(&prepared);
        assert!(
            idwt_levels.len() >= 3,
            "fixture should exercise a multi-level IDWT plan"
        );

        crop_prepared_direct_grayscale_plan_to_output_region(
            &mut prepared,
            signinum_core::Rect {
                x: 112,
                y: 112,
                w: 32,
                h: 32,
            },
        )
        .expect("crop direct plan");
        let cropped_idwt_levels = prepared_direct_grayscale_idwt_full_and_prepared_lens(&prepared);

        assert_eq!(cropped_idwt_levels.len(), idwt_levels.len());
        for (level_idx, (full_len, cropped_len)) in cropped_idwt_levels.iter().copied().enumerate()
        {
            assert!(
                cropped_len > 0 && cropped_len <= full_len,
                "cropped ROI should keep IDWT level {level_idx} bounded; full={full_len}, cropped={cropped_len}"
            );
        }
        assert!(
            cropped_idwt_levels
                .iter()
                .any(|(full_len, cropped_len)| cropped_len < full_len),
            "cropped ROI should reduce at least one IDWT level"
        );
    }

    fn prepared_direct_grayscale_ht_job_count(plan: &super::PreparedDirectGrayscalePlan) -> usize {
        plan.steps
            .iter()
            .map(|step| match step {
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => sub_band.jobs.len(),
                _ => 0,
            })
            .sum()
    }

    fn prepared_direct_grayscale_ht_coded_byte_count(
        plan: &super::PreparedDirectGrayscalePlan,
    ) -> usize {
        plan.steps
            .iter()
            .map(|step| match step {
                PreparedDirectGrayscaleStep::HtSubBand(sub_band) => sub_band.coded_data.len(),
                _ => 0,
            })
            .sum()
    }

    fn prepared_direct_grayscale_idwt_output_sample_count(
        plan: &super::PreparedDirectGrayscalePlan,
    ) -> usize {
        plan.steps
            .iter()
            .map(|step| match step {
                PreparedDirectGrayscaleStep::Idwt(idwt) => prepared_idwt_output_len(idwt),
                _ => 0,
            })
            .sum()
    }

    fn prepared_direct_grayscale_idwt_full_and_prepared_lens(
        plan: &super::PreparedDirectGrayscalePlan,
    ) -> Vec<(usize, usize)> {
        plan.steps
            .iter()
            .filter_map(|step| match step {
                PreparedDirectGrayscaleStep::Idwt(idwt) => Some((
                    idwt.step.rect.width() as usize * idwt.step.rect.height() as usize,
                    prepared_idwt_output_len(idwt),
                )),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn prepared_classic_direct_plan_groups_cleanup_subbands_before_idwt() {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode j2k gray8");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct grayscale plan");
        let classic_subband_steps = plan
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    step,
                    signinum_j2k_native::J2kDirectGrayscaleStep::ClassicSubBand(_)
                )
            })
            .count();
        assert!(
            classic_subband_steps > 1,
            "fixture must exercise multiple classic sub-band cleanup steps"
        );

        let prepared = prepare_direct_grayscale_plan(&plan).expect("prepared direct plan");
        assert_eq!(
            prepared.classic_groups.len(),
            1,
            "classic J2K direct decode should group adjacent sub-band cleanups before IDWT"
        );
        assert_eq!(
            prepared.classic_groups[0].members.len(),
            classic_subband_steps
        );
        assert!(matches!(
            prepared.steps[prepared.classic_groups[0].start_step],
            PreparedDirectGrayscaleStep::ClassicSubBand(_)
        ));
    }

    #[test]
    fn classic_plain_fast_path_accepts_style_zero_arithmetic_jobs() {
        let jobs = [J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: 1,
            segment_offset: 0,
            segment_count: 1,
            width: 64,
            height: 64,
            output_stride: 64,
            output_offset: 0,
            missing_msbs: 0,
            total_bitplanes: 8,
            roi_shift: 0,
            number_of_coding_passes: 1,
            sub_band_type: 0,
            style_flags: 0,
            strict: 1,
            dequantization_step: 1.0,
        }];
        let segments = [J2kClassicSegment {
            data_offset: 0,
            data_length: 1,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: 1,
        }];

        assert!(
            classic_batch_uses_plain_fast_path(&jobs, &segments),
            "style-0 arithmetic-only classic J2K jobs should use the fused plain cleanup/store kernel"
        );
    }

    #[test]
    fn classic_repeated_plain_fast_path_stays_off_for_wsi_batch_size() {
        let jobs = [J2kClassicCleanupBatchJob {
            coded_offset: 0,
            coded_len: 1,
            segment_offset: 0,
            segment_count: 1,
            width: 64,
            height: 64,
            output_stride: 64,
            output_offset: 0,
            missing_msbs: 0,
            total_bitplanes: 8,
            roi_shift: 0,
            number_of_coding_passes: 1,
            sub_band_type: 0,
            style_flags: 0,
            strict: 1,
            dequantization_step: 1.0,
        }];
        let segments = [J2kClassicSegment {
            data_offset: 0,
            data_length: 1,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: 1,
        }];

        assert!(
            !classic_repeated_uses_plain_fast_path(16, &jobs, &segments),
            "batch-16 WSI classic J2K should keep the device-state cleanup plus separate store path"
        );
    }

    #[test]
    fn repeated_gray_store_detects_contiguous_full_wsi_tiles() {
        let full_tile = J2kRepeatedGrayStoreParams {
            input_width: 1024,
            input_height: 1024,
            source_x: 0,
            source_y: 0,
            copy_width: 1024,
            copy_height: 1024,
            output_width: 1024,
            output_height: 1024,
            output_x: 0,
            output_y: 0,
            addend: 0.0,
            batch_count: 16,
            max_value: 255.0,
            u8_scale: 1.0,
            u16_scale: 257.0,
        };
        assert!(
            repeated_gray_store_is_contiguous_full_surface(full_tile),
            "full repeated grayscale WSI stores should use the contiguous store kernel"
        );

        let windowed = J2kRepeatedGrayStoreParams {
            source_x: 1,
            copy_width: 1023,
            ..full_tile
        };
        assert!(
            !repeated_gray_store_is_contiguous_full_surface(windowed),
            "ROI/windowed repeated grayscale stores must stay on the generic store kernel"
        );
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut code_block_decoder = MetalCodeBlockDecoder::default();
        let decoded = image
            .decode_components_with_ht_decoder(context, &mut code_block_decoder)
            .map_err(|error| Error::Decode(signinum_j2k::J2kError::Backend(error.to_string())))?;
        let stage = select_plane_stage(runtime, image, &decoded, &mut code_block_decoder)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_to_surface_with_device<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| decode_image_to_surface(image, context, fmt))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_region_to_surface<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut code_block_decoder = MetalCodeBlockDecoder::default();
        let decoded = image
            .decode_region_components_with_ht_decoder(
                context,
                (roi.x, roi.y, roi.w, roi.h),
                &mut code_block_decoder,
            )
            .map_err(|error| Error::Decode(signinum_j2k::J2kError::Backend(error.to_string())))?;
        let stage = select_plane_stage(runtime, image, &decoded, &mut code_block_decoder)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_image_region_to_surface_with_device<'a>(
    image: &NativeImage<'a>,
    context: &mut NativeDecoderContext<'a>,
    fmt: PixelFormat,
    roi: Rect,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_image_region_to_surface(image, context, fmt, roi)
    })
}

#[cfg(target_os = "macos")]
fn select_plane_stage(
    runtime: &MetalRuntime,
    image: &NativeImage<'_>,
    decoded: &NativeDecodedComponents<'_>,
    code_block_decoder: &mut MetalCodeBlockDecoder,
) -> Result<PlaneStage, Error> {
    if image.supports_direct_device_plane_reuse() {
        if matches!(decoded.color_space(), NativeColorSpace::RGB)
            && !decoded.has_alpha()
            && decoded.planes().len() == 3
        {
            if let Some(stage) = PlaneStage::from_captured_planes(
                decoded,
                code_block_decoder.mct.take_captured_planes(),
            ) {
                return Ok(stage);
            }
        }
        if matches!(decoded.color_space(), NativeColorSpace::Gray)
            && !decoded.has_alpha()
            && decoded.planes().len() == 1
        {
            if let Some(stage) = PlaneStage::from_captured_planes(
                decoded,
                code_block_decoder.store.take_captured_planes(),
            ) {
                return Ok(stage);
            }
        }
    }

    PlaneStage::from_planes(&runtime.device, decoded, None)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<Surface, Error> {
    let target_dims = (
        dims.0.div_ceil(scale.denominator()),
        dims.1.div_ceil(scale.denominator()),
    );
    let settings = NativeDecodeSettings {
        target_resolution: Some(target_dims),
        ..NativeDecodeSettings::default()
    };
    let image = NativeImage::new(bytes, &settings)
        .map_err(|error| Error::Decode(signinum_j2k::J2kError::Backend(error.to_string())))?;
    let mut context = NativeDecoderContext::default();
    decode_image_to_surface(&image, &mut context, fmt)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: signinum_core::Rect,
    scale: signinum_core::Downscale,
) -> Result<Surface, Error> {
    let target_dims = (
        dims.0.div_ceil(scale.denominator()),
        dims.1.div_ceil(scale.denominator()),
    );
    let settings = NativeDecodeSettings {
        target_resolution: Some(target_dims),
        ..NativeDecodeSettings::default()
    };
    let image = NativeImage::new(bytes, &settings)
        .map_err(|error| Error::Decode(signinum_j2k::J2kError::Backend(error.to_string())))?;
    let mut context = NativeDecoderContext::default();
    decode_image_region_to_surface(&image, &mut context, fmt, roi.scaled_covering(scale))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface_with_device(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_scaled_to_surface(bytes, dims, fmt, scale)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface_with_device(
    bytes: &[u8],
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: signinum_core::Rect,
    scale: signinum_core::Downscale,
    device: &Device,
) -> Result<Surface, Error> {
    with_runtime_for_device(device, |_| {
        decode_region_scaled_to_surface(bytes, dims, fmt, roi, scale)
    })
}
