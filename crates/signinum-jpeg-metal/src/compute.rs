// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::similar_names)]

#[cfg(all(target_os = "macos", test))]
use metal::foreign_types::ForeignType;
#[cfg(all(target_os = "macos", test))]
use signinum_metal_support::system_default_device;
#[cfg(all(target_os = "macos", test))]
use std::cell::Cell;
#[cfg(target_os = "macos")]
use std::{
    cell::RefCell,
    ffi::OsStr,
    mem::{size_of, size_of_val},
    sync::{Mutex, MutexGuard},
    time::{Duration, Instant},
};

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device,
    MTLPixelFormat, MTLResourceOptions, MTLSize,
};
use signinum_core::{BackendRequest, BufferError, PixelFormat, Rect};
use signinum_jpeg::{
    adapter::{
        JpegEntropyCheckpointV1, JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
        JpegHuffmanTable as PacketHuffmanTable,
    },
    ColorSpace as JpegColorSpace, ComponentRowWriter, Decoder as CpuDecoder,
};
#[cfg(target_os = "macos")]
use signinum_metal_support::{
    checked_command_queue, private_buffer, shared_buffer, shared_buffer_with_bytes,
    MetalPipelineLoader, MetalSupportError,
};

use crate::viewport::ViewportTile;
use crate::{batch, Error, Surface};

#[cfg(target_os = "macos")]
const SHADER_SOURCE: &str = include_str!("shaders.metal");

#[cfg(target_os = "macos")]
const MODE_GRAY: u32 = 0;
#[cfg(target_os = "macos")]
const MODE_YCBCR: u32 = 1;
#[cfg(target_os = "macos")]
const MODE_RGB: u32 = 2;

#[cfg(target_os = "macos")]
const OUT_GRAY: u32 = 0;
#[cfg(target_os = "macos")]
const OUT_RGB: u32 = 1;
#[cfg(target_os = "macos")]
const OUT_RGBA: u32 = 2;

#[cfg(target_os = "macos")]
pub(crate) const JPEG_BASELINE_ENCODE_FORMAT_GRAY8: u32 = 0;
#[cfg(target_os = "macos")]
pub(crate) const JPEG_BASELINE_ENCODE_FORMAT_RGB8: u32 = 1;
#[cfg(target_os = "macos")]
const JPEG_BASELINE_ENCODE_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const JPEG_BASELINE_ENCODE_STATUS_OVERFLOW: u32 = 1;
#[cfg(target_os = "macos")]
const JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN: u32 = 2;
#[cfg(target_os = "macos")]
const JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS: u32 = 3;

#[cfg(target_os = "macos")]
const FAST420_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const FAST420_STATUS_TRUNCATED: u32 = 1;
#[cfg(target_os = "macos")]
const FAST420_STATUS_HUFFMAN: u32 = 2;
#[cfg(target_os = "macos")]
const AUTO_METAL_MIN_BATCH_REQUESTS: usize = 8;
#[cfg(target_os = "macos")]
const AUTO_METAL_MIN_BATCH_EDGE: u32 = 512;
#[cfg(target_os = "macos")]
const REGION_SCALED_BATCH_CHUNK: usize = 8;
#[cfg(target_os = "macos")]
const FAST420_BATCH_TIMING_ENV: &str = "SIGNINUM_JPEG_METAL_FAST420_BATCH_TIMING";

#[cfg(all(target_os = "macos", test))]
std::thread_local! {
    static JPEG_PRIVATE_BUFFER_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static JPEG_SHARED_BUFFER_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_jpeg_private_buffer_allocations_for_test() {
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn reset_jpeg_shared_buffer_allocations_for_test() {
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(0));
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn jpeg_private_buffer_allocations_for_test() -> usize {
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(Cell::get)
}

#[cfg(all(target_os = "macos", test))]
pub(crate) fn jpeg_shared_buffer_allocations_for_test() -> usize {
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(Cell::get)
}

#[cfg(target_os = "macos")]
fn new_shared_buffer(device: &Device, bytes: usize) -> Buffer {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    shared_buffer(device, bytes)
}

#[cfg(target_os = "macos")]
fn new_shared_buffer_with_data(device: &Device, bytes: &[u8]) -> Buffer {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    shared_buffer_with_bytes(device, bytes)
}

#[cfg(target_os = "macos")]
fn new_private_buffer(device: &Device, bytes: usize) -> Buffer {
    #[cfg(test)]
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    private_buffer(device, bytes)
}

#[cfg(target_os = "macos")]
fn new_decode_plane_buffer(device: &Device, bytes: usize, returned_publicly: bool) -> Buffer {
    if returned_publicly {
        new_shared_buffer(device, bytes)
    } else {
        new_private_buffer(device, bytes)
    }
}

#[cfg(target_os = "macos")]
struct ReusablePrivateBuffer {
    key: &'static str,
    capacity: usize,
    buffer: Buffer,
}

#[cfg(target_os = "macos")]
struct ReusableSharedBuffer {
    key: &'static str,
    capacity: usize,
    buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct MetalBatchScratch {
    private_buffers: Vec<ReusablePrivateBuffer>,
    shared_buffers: Vec<ReusableSharedBuffer>,
}

#[cfg(target_os = "macos")]
impl MetalBatchScratch {
    fn private_buffer(&mut self, device: &Device, key: &'static str, bytes: usize) -> Buffer {
        let bytes = bytes.max(1);
        if let Some(entry) = self
            .private_buffers
            .iter()
            .find(|entry| entry.key == key && entry.capacity >= bytes)
        {
            return entry.buffer.clone();
        }

        let buffer = new_private_buffer(device, bytes);
        if let Some(entry) = self
            .private_buffers
            .iter_mut()
            .find(|entry| entry.key == key)
        {
            entry.capacity = bytes;
            entry.buffer = buffer.clone();
        } else {
            self.private_buffers.push(ReusablePrivateBuffer {
                key,
                capacity: bytes,
                buffer: buffer.clone(),
            });
        }
        buffer
    }

    fn shared_buffer_with_bytes(
        &mut self,
        device: &Device,
        key: &'static str,
        bytes: &[u8],
    ) -> Buffer {
        let capacity = bytes.len().max(1);
        let buffer = if let Some(entry) = self
            .shared_buffers
            .iter()
            .find(|entry| entry.key == key && entry.capacity >= capacity)
        {
            entry.buffer.clone()
        } else {
            let buffer = new_shared_buffer(device, capacity);
            if let Some(entry) = self
                .shared_buffers
                .iter_mut()
                .find(|entry| entry.key == key)
            {
                entry.capacity = capacity;
                entry.buffer = buffer.clone();
            } else {
                self.shared_buffers.push(ReusableSharedBuffer {
                    key,
                    capacity,
                    buffer: buffer.clone(),
                });
            }
            buffer
        };

        if !bytes.is_empty() {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.contents().cast::<u8>(),
                    bytes.len(),
                );
            }
        }
        buffer
    }

    fn shared_buffer_with_slice<T>(
        &mut self,
        device: &Device,
        key: &'static str,
        values: &[T],
    ) -> Buffer {
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let bytes = unsafe {
            core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values))
        };
        self.shared_buffer_with_bytes(device, key, bytes)
    }
}

#[cfg(target_os = "macos")]
struct FastRgbDecodeBuffer {
    buffer: Buffer,
    dimensions: (u32, u32),
    status_buffer: Buffer,
    command_buffer: CommandBuffer,
}

#[cfg(target_os = "macos")]
fn private_jpeg_tile_from_fast_rgb_buffer(
    decoded: FastRgbDecodeBuffer,
) -> crate::ResidentPrivateJpegTile {
    crate::ResidentPrivateJpegTile {
        buffer: decoded.buffer,
        byte_offset: 0,
        dimensions: decoded.dimensions,
        pixel_format: PixelFormat::Rgb8,
        pitch_bytes: decoded.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel(),
        status_buffer: decoded.status_buffer,
        command_buffer: decoded.command_buffer,
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegPackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    alpha: u32,
    mode: u32,
    out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegBaselineEncodeParams {
    pub(crate) input_offset_bytes: u32,
    pub(crate) input_width: u32,
    pub(crate) input_height: u32,
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) pitch_bytes: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) restart_interval_mcus: u32,
    pub(crate) format: u32,
    pub(crate) components: u32,
    pub(crate) max_h: u32,
    pub(crate) max_v: u32,
    pub(crate) h0: u32,
    pub(crate) v0: u32,
    pub(crate) h1: u32,
    pub(crate) v1: u32,
    pub(crate) h2: u32,
    pub(crate) v2: u32,
    pub(crate) entropy_offset_bytes: u32,
    pub(crate) entropy_capacity: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct JpegBaselineEncodeHuffmanTable {
    pub(crate) codes: [u16; 256],
    pub(crate) lens: [u8; 256],
}

#[cfg(target_os = "macos")]
impl Default for JpegBaselineEncodeHuffmanTable {
    fn default() -> Self {
        Self {
            codes: [0; 256],
            lens: [0; 256],
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct JpegBaselineEncodeStatus {
    code: u32,
    entropy_len: u32,
    detail: u32,
    reserved: u32,
}

#[cfg(target_os = "macos")]
pub(crate) struct JpegBaselineEntropyEncodeJob<'a> {
    pub(crate) input: &'a Buffer,
    pub(crate) input_offset: usize,
    pub(crate) params: JpegBaselineEncodeParams,
    pub(crate) q_luma: [u8; 64],
    pub(crate) q_chroma: [u8; 64],
    pub(crate) huff_dc_luma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_ac_luma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_dc_chroma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_ac_chroma: JpegBaselineEncodeHuffmanTable,
    pub(crate) entropy_capacity: usize,
}

#[cfg(target_os = "macos")]
pub(crate) struct JpegBaselineEntropyEncodeBatchJob<'a> {
    pub(crate) input: &'a Buffer,
    pub(crate) params: Vec<JpegBaselineEncodeParams>,
    pub(crate) q_luma: [u8; 64],
    pub(crate) q_chroma: [u8; 64],
    pub(crate) huff_dc_luma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_ac_luma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_dc_chroma: JpegBaselineEncodeHuffmanTable,
    pub(crate) huff_ac_chroma: JpegBaselineEncodeHuffmanTable,
    pub(crate) entropy_capacity: usize,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast420Params {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    restart_start_mcu: u32,
    entropy_len: u32,
    out_stride: u32,
    alpha: u32,
    out_format: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct JpegFast420ScaledParams {
    scaled_width: u32,
    scaled_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    restart_start_mcu: u32,
    entropy_len: u32,
    scale_shift: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast444Params {
    width: u32,
    height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    restart_start_mcu: u32,
    entropy_len: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct JpegFast444ScaledParams {
    scaled_width: u32,
    scaled_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    restart_interval_mcus: u32,
    restart_offset_count: u32,
    restart_start_mcu: u32,
    entropy_len: u32,
    scale_shift: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct JpegFast420WindowedPackParams {
    src_width: u32,
    src_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    src_x: u32,
    src_y: u32,
    width: u32,
    height: u32,
    out_stride: u32,
    alpha: u32,
    out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast420BatchParams {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    segment_count: u32,
    tile_count: u32,
    out_stride: u32,
    alpha: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct JpegFastRegionScaledBatchParams {
    scaled_width: u32,
    scaled_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    segment_count: u32,
    tile_count: u32,
    scale_shift: u32,
    origin_x: u32,
    origin_y: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast444TextureBatchParams {
    width: u32,
    height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    segment_count: u32,
    tile_index: u32,
    alpha: u32,
    mode: u32,
}

#[cfg(target_os = "macos")]
const FAST422_TEXTURE_BOUNDARY_META_WORDS: usize = 4;
#[cfg(target_os = "macos")]
const FAST422_TEXTURE_BOUNDARY_SAMPLE_BYTES: usize = 48;
#[cfg(target_os = "macos")]
const FAST420_TEXTURE_BOUNDARY_META_WORDS: usize = 4;
#[cfg(target_os = "macos")]
const FAST420_TEXTURE_BOUNDARY_SAMPLE_BYTES: usize = 64;
#[cfg(target_os = "macos")]
const FAST420_TEXTURE_VERTICAL_META_WORDS: usize = 4;
#[cfg(target_os = "macos")]
const FAST420_TEXTURE_VERTICAL_SAMPLE_BYTES: usize = 64;

/// Direct-to-texture decode params shared by the 4:2:0 and 4:2:2 texture
/// kernels (identical layout; both kernel families read the same fields).
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast420TextureBatchParams {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    segment_count: u32,
    tile_index: u32,
    alpha: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct JpegWindowedPackBatchParams {
    src_width: u32,
    src_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    src_x: u32,
    src_y: u32,
    width: u32,
    height: u32,
    tile_count: u32,
    out_stride: u32,
    alpha: u32,
    mode: u32,
    out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegWindowedTexturePackBatchParams {
    src_width: u32,
    src_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    src_x: u32,
    src_y: u32,
    width: u32,
    height: u32,
    tile_index: u32,
    alpha: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegTexturePackBatchParams {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    tile_index: u32,
    alpha: u32,
    mode: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegRgb8ToRgbaTextureParams {
    width: u32,
    height: u32,
    in_stride: u32,
    alpha: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct PreparedHuffmanHost {
    min_code: [i32; 17],
    max_code: [i32; 17],
    val_offset: [i32; 17],
    values: [u8; 256],
    fast_symbol: [u8; 512],
    fast_len: [u8; 512],
    values_len: u16,
    reserved: u16,
}

#[cfg(target_os = "macos")]
impl From<&PacketHuffmanTable> for PreparedHuffmanHost {
    fn from(value: &PacketHuffmanTable) -> Self {
        let mut min_code = [i32::MAX; 17];
        let mut max_code = [-1i32; 17];
        let mut val_offset = [0i32; 17];
        let mut values = [0u8; 256];
        let mut fast_symbol = [0u8; 512];
        let mut fast_len = [0u8; 512];
        let values_len = usize::from(value.values_len);
        values[..values_len].copy_from_slice(&value.values[..values_len]);

        let mut huffsize = [0u8; 256];
        let mut huffsize_len = 0usize;
        for (len_minus_1, &count) in value.bits.iter().enumerate() {
            let len = u8::try_from(len_minus_1 + 1).expect("JPEG Huffman code length fits in u8");
            for _ in 0..count {
                huffsize[huffsize_len] = len;
                huffsize_len += 1;
            }
        }

        let mut huffcode = [0u16; 256];
        let mut code = 0u32;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for (idx, &size) in huffsize[..huffsize_len].iter().enumerate() {
            while size != si {
                code <<= 1;
                si += 1;
            }
            huffcode[idx] = u16::try_from(code).expect("JPEG Huffman code fits in u16");
            code += 1;
        }

        let mut idx = 0usize;
        for (len_minus_1, &count) in value.bits.iter().enumerate() {
            let len = len_minus_1 + 1;
            let count = usize::from(count);
            if count == 0 {
                continue;
            }
            min_code[len] = i32::from(huffcode[idx]);
            max_code[len] = i32::from(huffcode[idx + count - 1]);
            val_offset[len] =
                i32::try_from(idx).expect("JPEG Huffman value index fits in i32") - min_code[len];
            idx += count;
        }

        for idx in 0..huffsize_len {
            let len = usize::from(huffsize[idx]);
            if len == 0 || len > 9 {
                continue;
            }
            let code = usize::from(huffcode[idx]);
            let prefix = code << (9 - len);
            let fill = 1usize << (9 - len);
            for suffix in 0..fill {
                fast_symbol[prefix | suffix] = values[idx];
                fast_len[prefix | suffix] = huffsize[idx];
            }
        }

        Self {
            min_code,
            max_code,
            val_offset,
            values,
            fast_symbol,
            fast_len,
            values_len: value.values_len,
            reserved: 0,
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct JpegDecodeStatus {
    code: u32,
    detail: u32,
    position: u32,
    reserved: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegEntropyCheckpointHost {
    mcu_index: u32,
    entropy_pos: u32,
    bit_acc: u64,
    bit_count: u32,
    y_prev_dc: i32,
    cb_prev_dc: i32,
    cr_prev_dc: i32,
    reserved: u32,
}

#[cfg(target_os = "macos")]
impl From<JpegEntropyCheckpointV1> for JpegEntropyCheckpointHost {
    fn from(value: JpegEntropyCheckpointV1) -> Self {
        Self {
            mcu_index: value.mcu_index,
            entropy_pos: value.entropy_pos,
            bit_acc: value.bit_acc,
            bit_count: value.bit_count,
            y_prev_dc: value.y_prev_dc,
            cb_prev_dc: value.cb_prev_dc,
            cr_prev_dc: value.cr_prev_dc,
            reserved: value.reserved,
        }
    }
}

#[cfg(target_os = "macos")]
thread_local! {
    static DEFAULT_METAL_SESSION: RefCell<Option<Result<crate::MetalBackendSession, MetalSupportError>>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
pub(crate) struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    pack_pipeline: ComputePipelineState,
    jpeg_baseline_encode_pipeline: ComputePipelineState,
    jpeg_baseline_encode_batch_pipeline: ComputePipelineState,
    pack_420_pipeline: ComputePipelineState,
    pack_420_rgb_pipeline: ComputePipelineState,
    pack_420_rgba_pipeline: ComputePipelineState,
    pack_420_rgb_batch_pipeline: ComputePipelineState,
    pack_420_rgba_texture_pipeline: ComputePipelineState,
    pack_420_windowed_rgb_batch_pipeline: ComputePipelineState,
    pack_420_windowed_rgba_texture_pipeline: ComputePipelineState,
    pack_422_rgb_pipeline: ComputePipelineState,
    pack_422_rgba_pipeline: ComputePipelineState,
    pack_422_rgb_batch_pipeline: ComputePipelineState,
    pack_422_rgba_texture_pipeline: ComputePipelineState,
    pack_422_windowed_rgb_batch_pipeline: ComputePipelineState,
    pack_422_windowed_rgba_texture_pipeline: ComputePipelineState,
    pack_444_rgb_batch_pipeline: ComputePipelineState,
    pack_444_rgba_texture_pipeline: ComputePipelineState,
    pack_422_windowed_pipeline: ComputePipelineState,
    pack_422_windowed_rgb_pipeline: ComputePipelineState,
    pack_422_windowed_rgba_pipeline: ComputePipelineState,
    pack_420_windowed_pipeline: ComputePipelineState,
    pack_420_windowed_rgb_pipeline: ComputePipelineState,
    pack_420_windowed_rgba_pipeline: ComputePipelineState,
    fast420_decode_pipeline: ComputePipelineState,
    fast420_batch_decode_pipeline: ComputePipelineState,
    #[cfg(test)]
    fast420_batch_coeffs_decode_pipeline: ComputePipelineState,
    #[cfg(test)]
    fast420_batch_idct_deposit_pipeline: ComputePipelineState,
    fast420_scaled_region_batch_decode_pipeline: ComputePipelineState,
    fast420_rgba_texture_batch_decode_pipeline: ComputePipelineState,
    fast420_rgba_texture_boundary_pipeline: ComputePipelineState,
    fast420_rgba_texture_vertical_boundary_pipeline: ComputePipelineState,
    fast420_rgba_texture_corner_pipeline: ComputePipelineState,
    fast422_decode_pipeline: ComputePipelineState,
    fast422_batch_decode_pipeline: ComputePipelineState,
    fast422_scaled_region_batch_decode_pipeline: ComputePipelineState,
    fast422_rgba_texture_batch_decode_pipeline: ComputePipelineState,
    fast422_rgba_texture_boundary_pipeline: ComputePipelineState,
    fast422_region_decode_pipeline: ComputePipelineState,
    fast422_scaled_decode_pipeline: ComputePipelineState,
    fast422_scaled_region_decode_pipeline: ComputePipelineState,
    fast420_region_decode_pipeline: ComputePipelineState,
    fast420_scaled_decode_pipeline: ComputePipelineState,
    fast420_scaled_region_decode_pipeline: ComputePipelineState,
    fast444_decode_pipeline: ComputePipelineState,
    fast444_region_decode_pipeline: ComputePipelineState,
    fast444_scaled_decode_pipeline: ComputePipelineState,
    fast444_scaled_region_decode_pipeline: ComputePipelineState,
    fast444_scaled_region_batch_decode_pipeline: ComputePipelineState,
    fast444_rgba_texture_batch_decode_pipeline: ComputePipelineState,
    rgb8_to_rgba_texture_pipeline: ComputePipelineState,
    batch_scratch: Mutex<MetalBatchScratch>,
    viewport_plane_cache: Mutex<Option<CachedViewportPlanes>>,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    #[cfg(test)]
    fn new() -> Result<Self, MetalSupportError> {
        let device = system_default_device()?;
        Self::new_with_device(device)
    }

    pub(crate) fn new_with_device(device: Device) -> Result<Self, MetalSupportError> {
        let loader = MetalPipelineLoader::new(&device, SHADER_SOURCE)?;
        let pipeline = |name: &str| loader.pipeline(name);
        let queue = checked_command_queue(&device)?;
        Ok(Self {
            device,
            queue,
            pack_pipeline: pipeline("jpeg_pack")?,
            jpeg_baseline_encode_pipeline: pipeline("jpeg_encode_baseline_entropy")?,
            jpeg_baseline_encode_batch_pipeline: pipeline("jpeg_encode_baseline_entropy_batch")?,
            pack_420_pipeline: pipeline("jpeg_pack_420")?,
            pack_420_rgb_pipeline: pipeline("jpeg_pack_420_rgb")?,
            pack_420_rgba_pipeline: pipeline("jpeg_pack_420_rgba")?,
            pack_420_rgb_batch_pipeline: pipeline("jpeg_pack_420_rgb_batch")?,
            pack_420_rgba_texture_pipeline: pipeline("jpeg_pack_420_rgba_texture")?,
            pack_420_windowed_rgb_batch_pipeline: pipeline("jpeg_pack_420_windowed_rgb_batch")?,
            pack_420_windowed_rgba_texture_pipeline: pipeline(
                "jpeg_pack_420_windowed_rgba_texture",
            )?,
            pack_422_rgb_pipeline: pipeline("jpeg_pack_422_rgb")?,
            pack_422_rgba_pipeline: pipeline("jpeg_pack_422_rgba")?,
            pack_422_rgb_batch_pipeline: pipeline("jpeg_pack_422_rgb_batch")?,
            pack_422_rgba_texture_pipeline: pipeline("jpeg_pack_422_rgba_texture")?,
            pack_422_windowed_rgb_batch_pipeline: pipeline("jpeg_pack_422_windowed_rgb_batch")?,
            pack_422_windowed_rgba_texture_pipeline: pipeline(
                "jpeg_pack_422_windowed_rgba_texture",
            )?,
            pack_444_rgb_batch_pipeline: pipeline("jpeg_pack_444_rgb_batch")?,
            pack_444_rgba_texture_pipeline: pipeline("jpeg_pack_444_rgba_texture")?,
            pack_422_windowed_pipeline: pipeline("jpeg_pack_422_windowed")?,
            pack_422_windowed_rgb_pipeline: pipeline("jpeg_pack_422_windowed_rgb")?,
            pack_422_windowed_rgba_pipeline: pipeline("jpeg_pack_422_windowed_rgba")?,
            pack_420_windowed_pipeline: pipeline("jpeg_pack_420_windowed")?,
            pack_420_windowed_rgb_pipeline: pipeline("jpeg_pack_420_windowed_rgb")?,
            pack_420_windowed_rgba_pipeline: pipeline("jpeg_pack_420_windowed_rgba")?,
            fast420_decode_pipeline: pipeline("jpeg_decode_fast420")?,
            fast420_batch_decode_pipeline: pipeline("jpeg_decode_fast420_batch")?,
            #[cfg(test)]
            fast420_batch_coeffs_decode_pipeline: pipeline("jpeg_decode_fast420_batch_coeffs")?,
            #[cfg(test)]
            fast420_batch_idct_deposit_pipeline: pipeline("jpeg_idct_deposit_fast420_batch")?,
            fast420_scaled_region_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast420_scaled_region_batch",
            )?,
            fast420_rgba_texture_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast420_rgba_texture_batch",
            )?,
            fast420_rgba_texture_boundary_pipeline: pipeline(
                "jpeg_resolve_fast420_rgba_texture_boundaries",
            )?,
            fast420_rgba_texture_vertical_boundary_pipeline: pipeline(
                "jpeg_resolve_fast420_rgba_texture_vertical_boundaries",
            )?,
            fast420_rgba_texture_corner_pipeline: pipeline(
                "jpeg_resolve_fast420_rgba_texture_corners",
            )?,
            fast422_decode_pipeline: pipeline("jpeg_decode_fast422")?,
            fast422_batch_decode_pipeline: pipeline("jpeg_decode_fast422_batch")?,
            fast422_scaled_region_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast422_scaled_region_batch",
            )?,
            fast422_rgba_texture_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast422_rgba_texture_batch",
            )?,
            fast422_rgba_texture_boundary_pipeline: pipeline(
                "jpeg_resolve_fast422_rgba_texture_boundaries",
            )?,
            fast422_region_decode_pipeline: pipeline("jpeg_decode_fast422_region")?,
            fast422_scaled_decode_pipeline: pipeline("jpeg_decode_fast422_scaled")?,
            fast422_scaled_region_decode_pipeline: pipeline("jpeg_decode_fast422_scaled_region")?,
            fast420_region_decode_pipeline: pipeline("jpeg_decode_fast420_region")?,
            fast420_scaled_decode_pipeline: pipeline("jpeg_decode_fast420_scaled")?,
            fast420_scaled_region_decode_pipeline: pipeline("jpeg_decode_fast420_scaled_region")?,
            fast444_decode_pipeline: pipeline("jpeg_decode_fast444")?,
            fast444_region_decode_pipeline: pipeline("jpeg_decode_fast444_region")?,
            fast444_scaled_decode_pipeline: pipeline("jpeg_decode_fast444_scaled")?,
            fast444_scaled_region_decode_pipeline: pipeline("jpeg_decode_fast444_scaled_region")?,
            fast444_scaled_region_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast444_scaled_region_batch",
            )?,
            fast444_rgba_texture_batch_decode_pipeline: pipeline(
                "jpeg_decode_fast444_rgba_texture_batch",
            )?,
            rgb8_to_rgba_texture_pipeline: pipeline("jpeg_copy_rgb8_to_rgba_texture")?,
            batch_scratch: Mutex::new(MetalBatchScratch::default()),
            viewport_plane_cache: Mutex::new(None),
        })
    }

    fn batch_scratch(&self) -> Result<MutexGuard<'_, MetalBatchScratch>, Error> {
        self.batch_scratch
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "JPEG Metal batch scratch",
            })
    }

    fn viewport_plane_cache(&self) -> Result<MutexGuard<'_, Option<CachedViewportPlanes>>, Error> {
        self.viewport_plane_cache
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "JPEG Metal viewport plane cache",
            })
    }

    #[cfg(test)]
    fn viewport_plane_cache_id_for_test(&self) -> Result<Option<usize>, Error> {
        Ok(self
            .viewport_plane_cache()?
            .as_ref()
            .map(|cached| cached.plane0.as_ptr() as usize))
    }
}

#[cfg(target_os = "macos")]
fn pack_420_pipeline_for_format(runtime: &MetalRuntime, fmt: PixelFormat) -> &ComputePipelineState {
    match fmt {
        PixelFormat::Rgb8 => &runtime.pack_420_rgb_pipeline,
        PixelFormat::Rgba8 => &runtime.pack_420_rgba_pipeline,
        _ => &runtime.pack_420_pipeline,
    }
}

#[cfg(target_os = "macos")]
fn pack_420_windowed_pipeline_for_format(
    runtime: &MetalRuntime,
    fmt: PixelFormat,
) -> &ComputePipelineState {
    match fmt {
        PixelFormat::Rgb8 => &runtime.pack_420_windowed_rgb_pipeline,
        PixelFormat::Rgba8 => &runtime.pack_420_windowed_rgba_pipeline,
        _ => &runtime.pack_420_windowed_pipeline,
    }
}

#[cfg(target_os = "macos")]
fn pack_422_pipeline_for_format(
    runtime: &MetalRuntime,
    fmt: PixelFormat,
) -> Option<&ComputePipelineState> {
    match fmt {
        PixelFormat::Rgb8 => Some(&runtime.pack_422_rgb_pipeline),
        PixelFormat::Rgba8 => Some(&runtime.pack_422_rgba_pipeline),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn pack_422_windowed_pipeline_for_format(
    runtime: &MetalRuntime,
    fmt: PixelFormat,
) -> &ComputePipelineState {
    match fmt {
        PixelFormat::Rgb8 => &runtime.pack_422_windowed_rgb_pipeline,
        PixelFormat::Rgba8 => &runtime.pack_422_windowed_rgba_pipeline,
        _ => &runtime.pack_422_windowed_pipeline,
    }
}

#[cfg(target_os = "macos")]
fn with_runtime<R>(f: impl FnOnce(&MetalRuntime) -> Result<R, Error>) -> Result<R, Error> {
    DEFAULT_METAL_SESSION.with(|session| {
        let mut session = session.borrow_mut();
        if session.is_none() {
            *session = Some(
                signinum_metal_support::system_default_device()
                    .map(crate::MetalBackendSession::new),
            );
        }
        let Some(session) = session.as_ref() else {
            return Err(Error::MetalRuntime {
                message: "JPEG Metal default session was not initialized".to_string(),
            });
        };
        match session {
            Ok(session) => with_runtime_for_session(session, f),
            Err(error) => Err(runtime_initialization_error(error)),
        }
    })
}

#[cfg(target_os = "macos")]
fn with_runtime_for_session<R>(
    session: &crate::MetalBackendSession,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = session
        .runtime
        .get_or_init(|| MetalRuntime::new_with_device(session.device.clone()));
    match runtime {
        Ok(runtime) => f(runtime),
        Err(error) => Err(runtime_initialization_error(error)),
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn runtime_initialization_error(error: &MetalSupportError) -> Error {
    if error.is_unavailable() {
        Error::MetalUnavailable
    } else {
        Error::MetalRuntime {
            message: error.to_string(),
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_jpeg_baseline_entropy_with_session(
    session: &crate::MetalBackendSession,
    job: &JpegBaselineEntropyEncodeJob<'_>,
) -> Result<Vec<u8>, Error> {
    with_runtime_for_session(session, |runtime| {
        let entropy_buffer = runtime.device.new_buffer(
            job.entropy_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status = JpegBaselineEncodeStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<JpegBaselineEncodeStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.jpeg_baseline_encode_pipeline);
        encoder.set_buffer(0, Some(job.input), job.input_offset as u64);
        encoder.set_buffer(1, Some(&entropy_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<JpegBaselineEncodeParams>() as u64,
            (&raw const job.params).cast(),
        );
        encoder.set_bytes(
            4,
            size_of_val(&job.q_luma) as u64,
            job.q_luma.as_ptr().cast(),
        );
        encoder.set_bytes(
            5,
            size_of_val(&job.q_chroma) as u64,
            job.q_chroma.as_ptr().cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_luma).cast(),
        );
        encoder.set_bytes(
            7,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_luma).cast(),
        );
        encoder.set_bytes(
            8,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_chroma).cast(),
        );
        encoder.set_bytes(
            9,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_chroma).cast(),
        );
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
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe { *(status_buffer.contents().cast::<JpegBaselineEncodeStatus>()) };
        if status.code != JPEG_BASELINE_ENCODE_STATUS_OK {
            return Err(jpeg_baseline_encode_status_error(status));
        }
        let entropy_len = usize::try_from(status.entropy_len).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal encode entropy length exceeds usize".to_string(),
        })?;
        if entropy_len > job.entropy_capacity {
            return Err(Error::MetalKernel {
                message: "JPEG Baseline Metal encode reported length exceeds output capacity"
                    .to_string(),
            });
        }
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let entropy = unsafe {
            core::slice::from_raw_parts(entropy_buffer.contents().cast::<u8>(), entropy_len)
        };
        Ok(entropy.to_vec())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_jpeg_baseline_entropy_batch_with_session(
    session: &crate::MetalBackendSession,
    job: &JpegBaselineEntropyEncodeBatchJob<'_>,
) -> Result<Vec<Vec<u8>>, Error> {
    if job.params.is_empty() {
        return Ok(Vec::new());
    }
    with_runtime_for_session(session, |runtime| {
        let entropy_buffer = runtime.device.new_buffer(
            job.entropy_capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let statuses = vec![JpegBaselineEncodeStatus::default(); job.params.len()];
        let status_buffer = runtime.device.new_buffer_with_data(
            statuses.as_ptr().cast(),
            size_of::<JpegBaselineEncodeStatus>() as u64 * statuses.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params_buffer = runtime.device.new_buffer_with_data(
            job.params.as_ptr().cast(),
            size_of::<JpegBaselineEncodeParams>() as u64 * job.params.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tile_count = u32::try_from(job.params.len()).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal batch tile count exceeds u32".to_string(),
        })?;

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.jpeg_baseline_encode_batch_pipeline);
        encoder.set_buffer(0, Some(job.input), 0);
        encoder.set_buffer(1, Some(&entropy_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_buffer(3, Some(&params_buffer), 0);
        encoder.set_bytes(
            4,
            size_of_val(&job.q_luma) as u64,
            job.q_luma.as_ptr().cast(),
        );
        encoder.set_bytes(
            5,
            size_of_val(&job.q_chroma) as u64,
            job.q_chroma.as_ptr().cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_luma).cast(),
        );
        encoder.set_bytes(
            7,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_luma).cast(),
        );
        encoder.set_bytes(
            8,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_chroma).cast(),
        );
        encoder.set_bytes(
            9,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_chroma).cast(),
        );
        encoder.set_bytes(10, size_of::<u32>() as u64, (&raw const tile_count).cast());
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(tile_count),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status_slice = unsafe {
            core::slice::from_raw_parts(
                status_buffer.contents().cast::<JpegBaselineEncodeStatus>(),
                job.params.len(),
            )
        };
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let entropy_bytes = unsafe {
            core::slice::from_raw_parts(
                entropy_buffer.contents().cast::<u8>(),
                job.entropy_capacity,
            )
        };
        let mut out = Vec::with_capacity(job.params.len());
        for (status, params) in status_slice.iter().copied().zip(job.params.iter()) {
            if status.code != JPEG_BASELINE_ENCODE_STATUS_OK {
                return Err(jpeg_baseline_encode_status_error(status));
            }
            let entropy_len =
                usize::try_from(status.entropy_len).map_err(|_| Error::MetalKernel {
                    message: "JPEG Baseline Metal encode entropy length exceeds usize".to_string(),
                })?;
            let offset =
                usize::try_from(params.entropy_offset_bytes).map_err(|_| Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy offset exceeds usize".to_string(),
                })?;
            let capacity =
                usize::try_from(params.entropy_capacity).map_err(|_| Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy capacity exceeds usize".to_string(),
                })?;
            if entropy_len > capacity {
                return Err(Error::MetalKernel {
                    message:
                        "JPEG Baseline Metal encode reported length exceeds tile output capacity"
                            .to_string(),
                });
            }
            let end = offset
                .checked_add(entropy_len)
                .ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy range overflow".to_string(),
                })?;
            if end > entropy_bytes.len() {
                return Err(Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy range exceeds buffer".to_string(),
                });
            }
            out.push(entropy_bytes[offset..end].to_vec());
        }
        Ok(out)
    })
}

#[cfg(target_os = "macos")]
fn jpeg_baseline_encode_status_error(status: JpegBaselineEncodeStatus) -> Error {
    let message = match status.code {
        JPEG_BASELINE_ENCODE_STATUS_OVERFLOW => {
            "JPEG Baseline Metal encode entropy output exceeded capacity".to_string()
        }
        JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN => format!(
            "JPEG Baseline Metal encode missing Huffman code for symbol {}",
            status.detail
        ),
        JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS => {
            "JPEG Baseline Metal encode received invalid kernel parameters".to_string()
        }
        other => format!("JPEG Baseline Metal encode failed with status {other}"),
    };
    Error::MetalKernel { message }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaneMode {
    Gray,
    YCbCr,
    Rgb,
}

#[cfg(target_os = "macos")]
struct PlaneStage {
    dims: (u32, u32),
    mode: PlaneMode,
    plane0: Buffer,
    plane1: Option<Buffer>,
    plane2: Option<Buffer>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
enum PlaneStageResidency {
    CpuStagedMetalUpload,
    MetalResidentDecode,
}

#[cfg(target_os = "macos")]
struct ViewportPlaneWriter<'a> {
    stage: &'a mut PlaneStage,
    dest: Rect,
}

#[cfg(target_os = "macos")]
struct CachedViewportPlanes {
    dims: (u32, u32),
    mode: PlaneMode,
    plane0: Buffer,
    plane1: Option<Buffer>,
    plane2: Option<Buffer>,
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    fn new(device: &Device, color_space: JpegColorSpace, dims: (u32, u32)) -> Result<Self, Error> {
        let len = dims.0 as usize * dims.1 as usize;
        let plane0 = device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
        let (mode, plane1, plane2) = match color_space {
            JpegColorSpace::Grayscale => (PlaneMode::Gray, None, None),
            JpegColorSpace::YCbCr => (
                PlaneMode::YCbCr,
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
            ),
            JpegColorSpace::Rgb => (
                PlaneMode::Rgb,
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
            ),
            JpegColorSpace::Cmyk | JpegColorSpace::Ycck => {
                return Err(Error::MetalKernel {
                    message: "Metal compute path does not support CMYK/YCCK JPEG output"
                        .to_string(),
                })
            }
        };

        Ok(Self {
            dims,
            mode,
            plane0,
            plane1,
            plane2,
        })
    }

    fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        self.finish_with_runtime_and_residency(
            runtime,
            fmt,
            PlaneStageResidency::CpuStagedMetalUpload,
        )
    }

    fn finish_resident_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        self.finish_with_runtime_and_residency(
            runtime,
            fmt,
            PlaneStageResidency::MetalResidentDecode,
        )
    }

    fn finish_with_runtime_and_residency(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
        residency: PlaneStageResidency,
    ) -> Result<Surface, Error> {
        match (self.mode, fmt) {
            (PlaneMode::Gray | PlaneMode::YCbCr, PixelFormat::Gray8) => Ok(
                surface_from_plane_buffer(self.plane0, self.dims, fmt, residency),
            ),
            (
                PlaneMode::Gray | PlaneMode::YCbCr | PlaneMode::Rgb,
                PixelFormat::Rgb8 | PixelFormat::Rgba8,
            )
            | (PlaneMode::Rgb, PixelFormat::Gray8) => {
                Ok(self.dispatch_with_runtime(runtime, fmt, residency))
            }
            _ => Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            }),
        }
    }

    fn dispatch_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
        residency: PlaneStageResidency,
    ) -> Surface {
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let out_buffer = runtime.device.new_buffer(
            (pitch_bytes * self.dims.1 as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: match fmt {
                PixelFormat::Gray8 => OUT_GRAY,
                PixelFormat::Rgb8 => OUT_RGB,
                PixelFormat::Rgba8 => OUT_RGBA,
                _ => unreachable!("validated by finish"),
            },
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref().map(std::convert::AsRef::as_ref),
                self.plane2.as_ref().map(std::convert::AsRef::as_ref),
            ],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        surface_from_plane_buffer(out_buffer, self.dims, fmt, residency)
    }

    fn finish_rgb8_into_output_with_runtime(
        self,
        runtime: &MetalRuntime,
        output: &crate::MetalBatchOutputBuffer,
    ) -> Result<Surface, Error> {
        let fmt = PixelFormat::Rgb8;
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let tile_len = pitch_bytes * self.dims.1 as usize;
        let out_buffer =
            batch_output_buffer_or_new(runtime, Some(output), self.dims, 1, pitch_bytes, tile_len)?;
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: OUT_RGB,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref().map(std::convert::AsRef::as_ref),
                self.plane2.as_ref().map(std::convert::AsRef::as_ref),
            ],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(Surface::from_metal_buffer_offset(
            out_buffer, self.dims, fmt, 0,
        ))
    }

    fn finish_rgba8_into_texture_output_with_runtime(
        self,
        runtime: &MetalRuntime,
        output: &crate::MetalBatchTextureOutput,
    ) -> Result<crate::MetalTextureTile, Error> {
        let rgb_pitch_bytes = self.dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let rgb_tile_len = rgb_pitch_bytes * self.dims.1 as usize;
        let rgba_tile_len =
            self.dims.0 as usize * self.dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, self.dims, 1, rgba_tile_len)?;
        let out_buffer = {
            let mut batch_scratch = runtime.batch_scratch()?;
            batch_scratch.private_buffer(
                &runtime.device,
                "viewport_sparse_rgba_texture_rgb8",
                rgb_tile_len,
            )
        };
        let texture = output.texture(0).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let pack_params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(rgb_pitch_bytes)
                .expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: OUT_RGB,
        };
        let texture_params = JpegRgb8ToRgbaTextureParams {
            width: self.dims.0,
            height: self.dims.1,
            in_stride: u32::try_from(rgb_pitch_bytes)
                .expect("JPEG Metal RGB texture input stride fits in u32"),
            alpha: u32::from(u8::MAX),
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            pack_encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref().map(std::convert::AsRef::as_ref),
                self.plane2.as_ref().map(std::convert::AsRef::as_ref),
            ],
            &out_buffer,
            &pack_params,
        );
        dispatch_2d_pipeline(pack_encoder, &runtime.pack_pipeline, self.dims);
        pack_encoder.end_encoding();

        let texture_encoder = command_buffer.new_compute_command_encoder();
        texture_encoder.set_compute_pipeline_state(&runtime.rgb8_to_rgba_texture_pipeline);
        texture_encoder.set_buffer(0, Some(&out_buffer), 0);
        texture_encoder.set_bytes(
            1,
            size_of::<JpegRgb8ToRgbaTextureParams>() as u64,
            (&raw const texture_params).cast(),
        );
        texture_encoder.set_texture(0, Some(texture));
        dispatch_2d_pipeline(
            texture_encoder,
            &runtime.rgb8_to_rgba_texture_pipeline,
            self.dims,
        );
        texture_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let texture = output.clone_texture(0).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        Ok(crate::MetalTextureTile::new(
            texture,
            self.dims,
            PixelFormat::Rgba8,
        ))
    }

    fn dispatch_private_rgb8_with_runtime(
        self,
        runtime: &MetalRuntime,
        status_buffer: Buffer,
    ) -> crate::ResidentPrivateJpegTile {
        let fmt = PixelFormat::Rgb8;
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let out_buffer = new_private_buffer(&runtime.device, pitch_bytes * self.dims.1 as usize);
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: OUT_RGB,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        bind_three_plane_pack::<JpegPackParams>(
            encoder,
            [
                Some(&self.plane0),
                self.plane1.as_ref().map(std::convert::AsRef::as_ref),
                self.plane2.as_ref().map(std::convert::AsRef::as_ref),
            ],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let command_buffer = command_buffer.to_owned();

        crate::ResidentPrivateJpegTile {
            buffer: out_buffer,
            byte_offset: 0,
            dimensions: self.dims,
            pixel_format: fmt,
            pitch_bytes,
            status_buffer,
            command_buffer,
        }
    }
}

#[cfg(target_os = "macos")]
fn surface_from_plane_buffer(
    buffer: Buffer,
    dims: (u32, u32),
    fmt: PixelFormat,
    residency: PlaneStageResidency,
) -> Surface {
    match residency {
        PlaneStageResidency::CpuStagedMetalUpload => {
            Surface::from_cpu_staged_metal_buffer(buffer, dims, fmt)
        }
        PlaneStageResidency::MetalResidentDecode => Surface::from_metal_buffer(buffer, dims, fmt),
    }
}

#[cfg(target_os = "macos")]
impl ComponentRowWriter for PlaneStage {
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), signinum_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, gray_row);
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), signinum_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, y_row);
        write_row_u8(
            self.plane1.as_ref().expect("Cb plane"),
            y,
            width,
            chroma_blue_row,
        );
        write_row_u8(
            self.plane2.as_ref().expect("Cr plane"),
            y,
            width,
            chroma_red_row,
        );
        Ok(())
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), signinum_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, r_row);
        write_row_u8(self.plane1.as_ref().expect("G plane"), y, width, g_row);
        write_row_u8(self.plane2.as_ref().expect("B plane"), y, width, b_row);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn write_row_u8(buffer: &Buffer, y: u32, width: usize, src: &[u8]) {
    let row_start = y as usize * width;
    let row_end = row_start + width;
    let len = width * (y as usize + 1);
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), len.max(row_end))
    };
    dst[row_start..row_end].copy_from_slice(&src[..width]);
}

#[cfg(target_os = "macos")]
fn write_row_u8_at(buffer: &Buffer, y: u32, x: u32, full_width: usize, src: &[u8]) {
    let row_start = y as usize * full_width + x as usize;
    let row_end = row_start + src.len();
    let len = full_width * (y as usize + 1);
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), len.max(row_end))
    };
    dst[row_start..row_end].copy_from_slice(src);
}

#[cfg(target_os = "macos")]
fn plane_mode_for_color_space(color_space: JpegColorSpace) -> Result<PlaneMode, Error> {
    match color_space {
        JpegColorSpace::Grayscale => Ok(PlaneMode::Gray),
        JpegColorSpace::YCbCr => Ok(PlaneMode::YCbCr),
        JpegColorSpace::Rgb => Ok(PlaneMode::Rgb),
        JpegColorSpace::Cmyk | JpegColorSpace::Ycck => Err(Error::MetalKernel {
            message: "Metal compute path does not support CMYK/YCCK JPEG output".to_string(),
        }),
    }
}

#[cfg(target_os = "macos")]
fn clear_buffer(buffer: &Buffer, len: usize) {
    fill_buffer(buffer, len, 0);
}

#[cfg(target_os = "macos")]
fn fill_buffer(buffer: &Buffer, len: usize, value: u8) {
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    unsafe {
        core::ptr::write_bytes(buffer.contents().cast::<u8>(), value, len);
    }
}

#[cfg(target_os = "macos")]
fn cached_plane_stage(
    runtime: &MetalRuntime,
    color_space: JpegColorSpace,
    dims: (u32, u32),
) -> Result<PlaneStage, Error> {
    let mode = plane_mode_for_color_space(color_space)?;
    let mut slot = runtime.viewport_plane_cache()?;
    let len = dims.0 as usize * dims.1 as usize;
    let refresh = slot
        .as_ref()
        .is_none_or(|cached| cached.dims != dims || cached.mode != mode);
    if refresh {
        let plane0 = runtime
            .device
            .new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
        let (plane1, plane2) = match mode {
            PlaneMode::Gray => (None, None),
            PlaneMode::YCbCr | PlaneMode::Rgb => (
                Some(
                    runtime
                        .device
                        .new_buffer(len as u64, MTLResourceOptions::StorageModeShared),
                ),
                Some(
                    runtime
                        .device
                        .new_buffer(len as u64, MTLResourceOptions::StorageModeShared),
                ),
            ),
        };
        *slot = Some(CachedViewportPlanes {
            dims,
            mode,
            plane0,
            plane1,
            plane2,
        });
    }

    let cached = slot.as_ref().expect("viewport plane cache");
    let stage = PlaneStage {
        dims,
        mode,
        plane0: cached.plane0.clone(),
        plane1: cached.plane1.clone(),
        plane2: cached.plane2.clone(),
    };
    drop(slot);

    clear_buffer(&stage.plane0, len);
    match stage.mode {
        PlaneMode::Gray => {}
        PlaneMode::YCbCr => {
            if let Some(plane1) = &stage.plane1 {
                fill_buffer(plane1, len, 128);
            }
            if let Some(plane2) = &stage.plane2 {
                fill_buffer(plane2, len, 128);
            }
        }
        PlaneMode::Rgb => {
            if let Some(plane1) = &stage.plane1 {
                clear_buffer(plane1, len);
            }
            if let Some(plane2) = &stage.plane2 {
                clear_buffer(plane2, len);
            }
        }
    }
    Ok(stage)
}

#[cfg(target_os = "macos")]
impl ComponentRowWriter for ViewportPlaneWriter<'_> {
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), signinum_jpeg::JpegError> {
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            self.stage.dims.0 as usize,
            gray_row,
        );
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), signinum_jpeg::JpegError> {
        let width = self.stage.dims.0 as usize;
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            width,
            y_row,
        );
        write_row_u8_at(
            self.stage.plane1.as_ref().expect("Cb plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            chroma_blue_row,
        );
        write_row_u8_at(
            self.stage.plane2.as_ref().expect("Cr plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            chroma_red_row,
        );
        Ok(())
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), signinum_jpeg::JpegError> {
        let width = self.stage.dims.0 as usize;
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            width,
            r_row,
        );
        write_row_u8_at(
            self.stage.plane1.as_ref().expect("G plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            g_row,
        );
        write_row_u8_at(
            self.stage.plane2.as_ref().expect("B plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            b_row,
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
/// Bind the shared fast-decode entropy kernel inputs at slots 0-16: entropy
/// bytes, the three component planes, the family params struct, the three
/// quantization tables, the per-component DC/AC Huffman table pairs, restart
/// offsets, decode status, and entropy checkpoints.
#[allow(clippy::too_many_arguments)]
fn bind_fast_decode_entropy_inputs<P>(
    encoder: &metal::ComputeCommandEncoderRef,
    entropy_buffer: &Buffer,
    planes: [&Buffer; 3],
    params: &P,
    quants: [&[u16; 64]; 3],
    dc_tables: &[PreparedHuffmanHost; 3],
    ac_tables: &[PreparedHuffmanHost; 3],
    restart_offsets_buffer: &Buffer,
    status_buffer: &Buffer,
    entropy_checkpoints_buffer: &Buffer,
) {
    encoder.set_buffer(0, Some(entropy_buffer), 0);
    encoder.set_buffer(1, Some(planes[0]), 0);
    encoder.set_buffer(2, Some(planes[1]), 0);
    encoder.set_buffer(3, Some(planes[2]), 0);
    encoder.set_bytes(4, size_of::<P>() as u64, (&raw const *params).cast());
    for (slot, quant) in (5u64..).zip(quants) {
        encoder.set_bytes(slot, size_of::<[u16; 64]>() as u64, quant.as_ptr().cast());
    }
    for (index, (dc, ac)) in dc_tables.iter().zip(ac_tables.iter()).enumerate() {
        let slot = 8 + 2 * index as u64;
        encoder.set_bytes(
            slot,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const *dc).cast(),
        );
        encoder.set_bytes(
            slot + 1,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const *ac).cast(),
        );
    }
    encoder.set_buffer(14, Some(restart_offsets_buffer), 0);
    encoder.set_buffer(15, Some(status_buffer), 0);
    encoder.set_buffer(16, Some(entropy_checkpoints_buffer), 0);
}

/// Bind the shared three-plane pack kernel layout at slots 0-4: the component
/// planes, the packed output buffer, and the pack params struct.
fn bind_three_plane_pack<P>(
    encoder: &metal::ComputeCommandEncoderRef,
    planes: [Option<&metal::BufferRef>; 3],
    out_buffer: &Buffer,
    params: &P,
) {
    encoder.set_buffer(0, planes[0], 0);
    encoder.set_buffer(1, planes[1], 0);
    encoder.set_buffer(2, planes[2], 0);
    encoder.set_buffer(3, Some(out_buffer), 0);
    encoder.set_bytes(4, size_of::<P>() as u64, (&raw const *params).cast());
}

fn dispatch_2d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32),
) {
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
fn dispatch_3d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32, u32),
) {
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: u64::from(dims.2),
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
fn packed_pair_extent(value: u32) -> u32 {
    value.div_ceil(2).max(1)
}

#[cfg(target_os = "macos")]
fn dispatch_1d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    threads: u32,
) {
    let threadgroup_width = choose_1d_threadgroup_width(
        pipeline.thread_execution_width(),
        pipeline.max_total_threads_per_threadgroup(),
        threads,
    );
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(threads.max(1)),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
fn choose_1d_threadgroup_width(simd_width: u64, max_threads: u64, threads: u32) -> u64 {
    let simd_width = simd_width.max(1);
    let max_threads = max_threads.max(simd_width);
    let requested = u64::from(threads.max(1));
    let rounded = requested.div_ceil(simd_width) * simd_width;
    rounded.clamp(simd_width, max_threads.min(256).max(simd_width))
}

#[cfg(target_os = "macos")]
fn pixel_format_to_out_format(fmt: PixelFormat) -> Option<u32> {
    match fmt {
        PixelFormat::Gray8 => Some(OUT_GRAY),
        PixelFormat::Rgb8 => Some(OUT_RGB),
        PixelFormat::Rgba8 => Some(OUT_RGBA),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn plane_mode_to_u32(mode: PlaneMode) -> u32 {
    match mode {
        PlaneMode::Gray => MODE_GRAY,
        PlaneMode::YCbCr => MODE_YCBCR,
        PlaneMode::Rgb => MODE_RGB,
    }
}

#[cfg(target_os = "macos")]
/// Field access over the sampling-family fast packets, which are
/// field-identical; the sampling factor is carried by the packet type.
trait FastPacketAccess {
    /// Family name used in diagnostics ("fast420" / "fast422" / "fast444").
    const FAMILY_NAME: &'static str;

    fn dimensions(&self) -> (u32, u32);
    fn mcus_per_row(&self) -> u32;
    fn mcu_rows(&self) -> u32;
    fn restart_interval_mcus(&self) -> u32;
    fn restart_offsets(&self) -> &[u32];
    fn entropy_checkpoints(&self) -> &[JpegEntropyCheckpointV1];
    fn entropy_bytes(&self) -> &[u8];
    fn y_quant(&self) -> &[u16; 64];
    fn cb_quant(&self) -> &[u16; 64];
    fn cr_quant(&self) -> &[u16; 64];
    fn y_dc_table(&self) -> &PacketHuffmanTable;
    fn y_ac_table(&self) -> &PacketHuffmanTable;
    fn cb_dc_table(&self) -> &PacketHuffmanTable;
    fn cb_ac_table(&self) -> &PacketHuffmanTable;
    fn cr_dc_table(&self) -> &PacketHuffmanTable;
    fn cr_ac_table(&self) -> &PacketHuffmanTable;
}

macro_rules! impl_fast_packet_access {
    ($packet:ty, $name:literal) => {
        impl FastPacketAccess for $packet {
            const FAMILY_NAME: &'static str = $name;

            fn dimensions(&self) -> (u32, u32) {
                self.dimensions
            }
            fn mcus_per_row(&self) -> u32 {
                self.mcus_per_row
            }
            fn mcu_rows(&self) -> u32 {
                self.mcu_rows
            }
            fn restart_interval_mcus(&self) -> u32 {
                self.restart_interval_mcus
            }
            fn restart_offsets(&self) -> &[u32] {
                &self.restart_offsets
            }
            fn entropy_checkpoints(&self) -> &[JpegEntropyCheckpointV1] {
                &self.entropy_checkpoints
            }
            fn entropy_bytes(&self) -> &[u8] {
                &self.entropy_bytes
            }
            fn y_quant(&self) -> &[u16; 64] {
                &self.y_quant
            }
            fn cb_quant(&self) -> &[u16; 64] {
                &self.cb_quant
            }
            fn cr_quant(&self) -> &[u16; 64] {
                &self.cr_quant
            }
            fn y_dc_table(&self) -> &PacketHuffmanTable {
                &self.y_dc_table
            }
            fn y_ac_table(&self) -> &PacketHuffmanTable {
                &self.y_ac_table
            }
            fn cb_dc_table(&self) -> &PacketHuffmanTable {
                &self.cb_dc_table
            }
            fn cb_ac_table(&self) -> &PacketHuffmanTable {
                &self.cb_ac_table
            }
            fn cr_dc_table(&self) -> &PacketHuffmanTable {
                &self.cr_dc_table
            }
            fn cr_ac_table(&self) -> &PacketHuffmanTable {
                &self.cr_ac_table
            }
        }
    };
}

impl_fast_packet_access!(JpegFast420PacketV1, "fast420");
impl_fast_packet_access!(JpegFast422PacketV1, "fast422");
impl_fast_packet_access!(JpegFast444PacketV1, "fast444");

/// Chroma geometry for the subsampled families that share the
/// `JpegFast420Params` kernel ABI (4:2:0 halves chroma rows, 4:2:2 keeps
/// them; both halve chroma columns).
trait FastSubsampledPacket: FastPacketAccess {
    /// Luma MCU width in pixels.
    const MCU_WIDTH: u32;
    /// Luma MCU height in pixels (4:2:0 MCUs are 16 rows, 4:2:2 are 8).
    const MCU_HEIGHT: u32;
    /// Whether the full-RGB batch path may group restart-interval packets.
    const FULL_RGB_BATCH_SUPPORTS_RESTART: bool;
    const ENTROPY_PAYLOAD_CTX: &'static str;
    const REGION_SCALED_BATCH_OUT_STRIDE_CTX: &'static str;
    const OUTPUT_STRIDE_CTX: &'static str;
    const REGION_OUTPUT_STRIDE_CTX: &'static str;
    const SCALED_ENTROPY_PAYLOAD_CTX: &'static str;
    /// Blocks per MCU when the full-RGB batch path needs block-count
    /// validation for the split coeff/IDCT debug mode (4:2:0 only).
    const FULL_RGB_BATCH_BLOCKS_PER_MCU: Option<usize>;

    fn chroma_height(height: u32) -> u32;
    /// Vertical dispatch extent for the full-frame pack kernels: 4:2:0 packs
    /// 2x2 pixel quads per thread, 4:2:2 packs 2x1 pairs (full-height rows).
    fn packed_height_extent(height: u32) -> u32;
}

impl FastSubsampledPacket for JpegFast420PacketV1 {
    const MCU_WIDTH: u32 = 16;
    const MCU_HEIGHT: u32 = 16;
    const FULL_RGB_BATCH_SUPPORTS_RESTART: bool = true;
    const ENTROPY_PAYLOAD_CTX: &'static str = "fast420 entropy payload";
    const REGION_SCALED_BATCH_OUT_STRIDE_CTX: &'static str =
        "fast420 region scaled batch output stride";
    const OUTPUT_STRIDE_CTX: &'static str = "fast420 output stride";
    const REGION_OUTPUT_STRIDE_CTX: &'static str = "fast420 region output stride";
    const SCALED_ENTROPY_PAYLOAD_CTX: &'static str = "fast420 scaled entropy payload";
    const FULL_RGB_BATCH_BLOCKS_PER_MCU: Option<usize> = Some(6);

    fn chroma_height(height: u32) -> u32 {
        height.div_ceil(2)
    }
    fn packed_height_extent(height: u32) -> u32 {
        height.div_ceil(2).max(1)
    }
}

impl FastSubsampledPacket for JpegFast422PacketV1 {
    const MCU_WIDTH: u32 = 16;
    const MCU_HEIGHT: u32 = 8;
    const FULL_RGB_BATCH_SUPPORTS_RESTART: bool = false;
    const ENTROPY_PAYLOAD_CTX: &'static str = "fast422 entropy payload";
    const REGION_SCALED_BATCH_OUT_STRIDE_CTX: &'static str =
        "fast422 region scaled batch output stride";
    const OUTPUT_STRIDE_CTX: &'static str = "fast422 output stride";
    const REGION_OUTPUT_STRIDE_CTX: &'static str = "fast422 region output stride";
    const SCALED_ENTROPY_PAYLOAD_CTX: &'static str = "fast422 scaled entropy payload";
    const FULL_RGB_BATCH_BLOCKS_PER_MCU: Option<usize> = None;

    fn chroma_height(height: u32) -> u32 {
        height
    }
    fn packed_height_extent(height: u32) -> u32 {
        height
    }
}

/// Scratch-pool cache keys for one batch driver's buffers; keys stay
/// per-family so pooled buffers are never shared across kernel families.
#[cfg(target_os = "macos")]
struct FastScratchKeys {
    y: &'static str,
    cb: &'static str,
    cr: &'static str,
    entropy: &'static str,
    entropy_offsets: &'static str,
    entropy_lens: &'static str,
    entropy_checkpoints: &'static str,
    status: &'static str,
}

/// Per-family vertical chroma-repair scratch layout for the direct-to-texture
/// full-frame path (4:2:0 only; 4:2:2 has no vertical MCU chroma boundary).
#[cfg(target_os = "macos")]
struct FastVerticalRepairSpec {
    meta_words: usize,
    sample_bytes: usize,
    meta_key: &'static str,
    samples_key: &'static str,
}

/// Metal-side hooks for the subsampled families: per-family pipelines,
/// scratch keys, and batched-packet extraction.
#[cfg(target_os = "macos")]
trait FastSubsampledMetal: FastSubsampledPacket {
    const REGION_SCALED_KEYS: FastScratchKeys;
    const REGION_SCALED_TEXTURE_KEYS: FastScratchKeys;
    /// Scratch keys for the full-frame RGB batch driver.
    const FULL_BATCH_KEYS: FastScratchKeys;
    /// Scratch keys for the full-frame RGBA texture batch driver.
    const TEXTURE_KEYS: FastScratchKeys;
    /// Profile tag for the full-RGB batch timing rows.
    const FULL_RGB_BATCH_TIMING_TAG: &'static str;
    /// Boundary chroma-repair record layout for the direct-to-texture path.
    const TEXTURE_BOUNDARY_META_WORDS: usize;
    const TEXTURE_BOUNDARY_SAMPLE_BYTES: usize;
    const TEXTURE_BOUNDARY_META_KEY: &'static str;
    const TEXTURE_BOUNDARY_SAMPLES_KEY: &'static str;
    const TEXTURE_VERTICAL_REPAIR: Option<FastVerticalRepairSpec>;

    fn from_batched<'a>(packet: &BatchedFastPacket<'a>) -> Option<&'a Self>;
    fn to_batched(&self) -> BatchedFastPacket<'_>;
    fn decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn scaled_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn scaled_region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    /// Full-frame pack pipeline for the requested output format, or `None`
    /// when the family has no kernel for that format (4:2:2 packs only RGB
    /// and RGBA; 4:2:0 falls back to its generic pack kernel).
    fn pack_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Option<&ComputePipelineState>;
    fn pack_windowed_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> &ComputePipelineState;
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn pack_windowed_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn full_rgb_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn pack_full_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn pack_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn rgba_texture_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    fn rgba_texture_boundary_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState;
    /// Whether the full-RGB batch driver should record per-stage timing rows.
    fn full_rgb_batch_timing_enabled() -> bool;
    /// Per-MCU dispatch width for the texture repair passes (4:2:0 only;
    /// 4:2:2 repairs per entropy segment instead).
    fn texture_mcu_dispatch_threads(total_mcus: usize) -> Result<Option<u32>, Error>;
    /// Repair record count for the direct-to-texture boundary scratch.
    fn texture_repair_record_count(
        tile_count: usize,
        total_mcus: usize,
        total_decode_threads: u32,
    ) -> Result<usize, Error>;
    /// Thread count for the horizontal boundary repair pass, or `None` when
    /// the pass is not needed for this batch shape.
    fn horizontal_repair_threads(
        first: &Self,
        segment_count_u32: u32,
        mcu_threads: Option<u32>,
    ) -> Option<u32>;
    /// Encode any family-specific repair passes beyond the shared horizontal
    /// pass (4:2:0 adds vertical and corner passes).
    fn encode_extra_texture_repair_passes(
        runtime: &MetalRuntime,
        ctx: &FastTextureRepairCtx<'_>,
    ) -> Result<(), Error>;
    /// Pipelines for the split coeff/IDCT debug decode mode, when supported.
    #[cfg(test)]
    fn split_coeff_idct_pipelines(
        runtime: &MetalRuntime,
    ) -> Option<(&ComputePipelineState, &ComputePipelineState)>;
    /// Scratch keys (coeff blocks, DC-only flags) for the split decode mode
    /// in the texture driver.
    #[cfg(test)]
    const SPLIT_TEXTURE_SCRATCH_KEYS: (&'static str, &'static str);
}

/// Shared context handed to the family-specific texture repair hooks.
#[cfg(target_os = "macos")]
struct FastTextureRepairCtx<'a> {
    command_buffer: &'a CommandBufferRef,
    output: &'a crate::MetalBatchTextureOutput,
    boundary_meta_buffer: &'a Buffer,
    vertical_buffers: Option<&'a (Buffer, Buffer)>,
    decode_params: JpegFast420TextureBatchParams,
    tile_count: usize,
    mcu_threads: Option<u32>,
    tile_index_ctx: &'a str,
}

#[cfg(target_os = "macos")]
impl FastSubsampledMetal for JpegFast420PacketV1 {
    const REGION_SCALED_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast420_region_scaled_y",
        cb: "fast420_region_scaled_cb",
        cr: "fast420_region_scaled_cr",
        entropy: "fast420_region_scaled_entropy",
        entropy_offsets: "fast420_region_scaled_entropy_offsets",
        entropy_lens: "fast420_region_scaled_entropy_lens",
        entropy_checkpoints: "fast420_region_scaled_entropy_checkpoints",
        status: "fast420_region_scaled_status",
    };
    const REGION_SCALED_TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast420_region_scaled_texture_y",
        cb: "fast420_region_scaled_texture_cb",
        cr: "fast420_region_scaled_texture_cr",
        entropy: "fast420_region_scaled_texture_entropy",
        entropy_offsets: "fast420_region_scaled_texture_entropy_offsets",
        entropy_lens: "fast420_region_scaled_texture_entropy_lens",
        entropy_checkpoints: "fast420_region_scaled_texture_entropy_checkpoints",
        status: "fast420_region_scaled_texture_status",
    };

    const FULL_BATCH_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast420_full_y",
        cb: "fast420_full_cb",
        cr: "fast420_full_cr",
        entropy: "fast420_full_entropy",
        entropy_offsets: "fast420_full_entropy_offsets",
        entropy_lens: "fast420_full_entropy_lens",
        entropy_checkpoints: "fast420_full_entropy_checkpoints",
        status: "fast420_full_status",
    };
    const TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast420_texture_y",
        cb: "fast420_texture_cb",
        cr: "fast420_texture_cr",
        entropy: "fast420_texture_entropy",
        entropy_offsets: "fast420_texture_entropy_offsets",
        entropy_lens: "fast420_texture_entropy_lens",
        entropy_checkpoints: "fast420_texture_entropy_checkpoints",
        status: "fast420_texture_status",
    };
    const FULL_RGB_BATCH_TIMING_TAG: &'static str = "metal_fast420_batch";
    const TEXTURE_BOUNDARY_META_WORDS: usize = FAST420_TEXTURE_BOUNDARY_META_WORDS;
    const TEXTURE_BOUNDARY_SAMPLE_BYTES: usize = FAST420_TEXTURE_BOUNDARY_SAMPLE_BYTES;
    const TEXTURE_BOUNDARY_META_KEY: &'static str = "fast420_texture_boundary_meta";
    const TEXTURE_BOUNDARY_SAMPLES_KEY: &'static str = "fast420_texture_boundary_samples";
    const TEXTURE_VERTICAL_REPAIR: Option<FastVerticalRepairSpec> = Some(FastVerticalRepairSpec {
        meta_words: FAST420_TEXTURE_VERTICAL_META_WORDS,
        sample_bytes: FAST420_TEXTURE_VERTICAL_SAMPLE_BYTES,
        meta_key: "fast420_texture_vertical_meta",
        samples_key: "fast420_texture_vertical_samples",
    });

    fn from_batched<'a>(packet: &BatchedFastPacket<'a>) -> Option<&'a Self> {
        match packet {
            BatchedFastPacket::Fast420(packet) => Some(packet),
            _ => None,
        }
    }
    fn to_batched(&self) -> BatchedFastPacket<'_> {
        BatchedFastPacket::Fast420(self)
    }
    fn decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_decode_pipeline
    }
    fn region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_region_decode_pipeline
    }
    fn scaled_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_scaled_decode_pipeline
    }
    fn scaled_region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_scaled_region_decode_pipeline
    }
    fn pack_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Option<&ComputePipelineState> {
        Some(pack_420_pipeline_for_format(runtime, fmt))
    }
    fn pack_windowed_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> &ComputePipelineState {
        pack_420_windowed_pipeline_for_format(runtime, fmt)
    }
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_scaled_region_batch_decode_pipeline
    }
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_420_windowed_rgb_batch_pipeline
    }
    fn pack_windowed_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_420_windowed_rgba_texture_pipeline
    }
    fn full_rgb_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_batch_decode_pipeline
    }
    fn pack_full_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_420_rgb_batch_pipeline
    }
    fn pack_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_420_rgba_texture_pipeline
    }
    fn rgba_texture_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_rgba_texture_batch_decode_pipeline
    }
    fn rgba_texture_boundary_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast420_rgba_texture_boundary_pipeline
    }
    fn full_rgb_batch_timing_enabled() -> bool {
        fast420_batch_timing_enabled()
    }
    fn texture_mcu_dispatch_threads(total_mcus: usize) -> Result<Option<u32>, Error> {
        checked_u32(total_mcus, "fast420 texture batch MCU count").map(Some)
    }
    fn texture_repair_record_count(
        tile_count: usize,
        total_mcus: usize,
        _total_decode_threads: u32,
    ) -> Result<usize, Error> {
        tile_count
            .checked_mul(total_mcus)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast420 texture repair record count overflowed".to_string(),
            })
    }
    fn horizontal_repair_threads(
        first: &Self,
        _segment_count_u32: u32,
        mcu_threads: Option<u32>,
    ) -> Option<u32> {
        mcu_threads.filter(|_| first.mcus_per_row > 1)
    }
    fn encode_extra_texture_repair_passes(
        runtime: &MetalRuntime,
        ctx: &FastTextureRepairCtx<'_>,
    ) -> Result<(), Error> {
        let (vertical_meta_buffer, vertical_samples_buffer) =
            ctx.vertical_buffers.ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast420 texture vertical repair scratch was missing"
                    .to_string(),
            })?;
        let Some(mcu_threads) = ctx.mcu_threads else {
            return Err(Error::MetalKernel {
                message: "JPEG Metal fast420 texture MCU dispatch width was missing".to_string(),
            });
        };
        if ctx.decode_params.mcu_rows > 1 {
            for index in 0..ctx.tile_count {
                let texture = ctx
                    .output
                    .texture(index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal batch texture output slot was missing".to_string(),
                    })?;
                let decode_params = JpegFast420TextureBatchParams {
                    tile_index: checked_u32(index, ctx.tile_index_ctx)?,
                    ..ctx.decode_params
                };
                let boundary_encoder = ctx.command_buffer.new_compute_command_encoder();
                boundary_encoder.set_compute_pipeline_state(
                    &runtime.fast420_rgba_texture_vertical_boundary_pipeline,
                );
                boundary_encoder.set_buffer(0, Some(vertical_meta_buffer), 0);
                boundary_encoder.set_buffer(1, Some(vertical_samples_buffer), 0);
                boundary_encoder.set_bytes(
                    2,
                    size_of::<JpegFast420TextureBatchParams>() as u64,
                    (&raw const decode_params).cast(),
                );
                boundary_encoder.set_texture(0, Some(texture));
                dispatch_1d_pipeline(
                    boundary_encoder,
                    &runtime.fast420_rgba_texture_vertical_boundary_pipeline,
                    mcu_threads,
                );
                boundary_encoder.end_encoding();
            }
        }
        if ctx.decode_params.mcus_per_row > 1 && ctx.decode_params.mcu_rows > 1 {
            for index in 0..ctx.tile_count {
                let texture = ctx
                    .output
                    .texture(index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal batch texture output slot was missing".to_string(),
                    })?;
                let decode_params = JpegFast420TextureBatchParams {
                    tile_index: checked_u32(index, ctx.tile_index_ctx)?,
                    ..ctx.decode_params
                };
                let corner_encoder = ctx.command_buffer.new_compute_command_encoder();
                corner_encoder
                    .set_compute_pipeline_state(&runtime.fast420_rgba_texture_corner_pipeline);
                corner_encoder.set_buffer(0, Some(ctx.boundary_meta_buffer), 0);
                corner_encoder.set_buffer(1, Some(vertical_meta_buffer), 0);
                corner_encoder.set_buffer(2, Some(vertical_samples_buffer), 0);
                corner_encoder.set_bytes(
                    3,
                    size_of::<JpegFast420TextureBatchParams>() as u64,
                    (&raw const decode_params).cast(),
                );
                corner_encoder.set_texture(0, Some(texture));
                dispatch_1d_pipeline(
                    corner_encoder,
                    &runtime.fast420_rgba_texture_corner_pipeline,
                    mcu_threads,
                );
                corner_encoder.end_encoding();
            }
        }
        Ok(())
    }
    #[cfg(test)]
    fn split_coeff_idct_pipelines(
        runtime: &MetalRuntime,
    ) -> Option<(&ComputePipelineState, &ComputePipelineState)> {
        Some((
            &runtime.fast420_batch_coeffs_decode_pipeline,
            &runtime.fast420_batch_idct_deposit_pipeline,
        ))
    }
    #[cfg(test)]
    const SPLIT_TEXTURE_SCRATCH_KEYS: (&'static str, &'static str) = (
        "fast420_texture_coeff_blocks",
        "fast420_texture_dc_only_flags",
    );
}

#[cfg(target_os = "macos")]
impl FastSubsampledMetal for JpegFast422PacketV1 {
    const REGION_SCALED_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast422_region_scaled_y",
        cb: "fast422_region_scaled_cb",
        cr: "fast422_region_scaled_cr",
        entropy: "fast422_region_scaled_entropy",
        entropy_offsets: "fast422_region_scaled_entropy_offsets",
        entropy_lens: "fast422_region_scaled_entropy_lens",
        entropy_checkpoints: "fast422_region_scaled_entropy_checkpoints",
        status: "fast422_region_scaled_status",
    };
    const REGION_SCALED_TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast422_region_scaled_texture_y",
        cb: "fast422_region_scaled_texture_cb",
        cr: "fast422_region_scaled_texture_cr",
        entropy: "fast422_region_scaled_texture_entropy",
        entropy_offsets: "fast422_region_scaled_texture_entropy_offsets",
        entropy_lens: "fast422_region_scaled_texture_entropy_lens",
        entropy_checkpoints: "fast422_region_scaled_texture_entropy_checkpoints",
        status: "fast422_region_scaled_texture_status",
    };

    const FULL_BATCH_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast422_full_y",
        cb: "fast422_full_cb",
        cr: "fast422_full_cr",
        entropy: "fast422_full_entropy",
        entropy_offsets: "fast422_full_entropy_offsets",
        entropy_lens: "fast422_full_entropy_lens",
        entropy_checkpoints: "fast422_full_entropy_checkpoints",
        status: "fast422_full_status",
    };
    const TEXTURE_KEYS: FastScratchKeys = FastScratchKeys {
        y: "fast422_texture_y",
        cb: "fast422_texture_cb",
        cr: "fast422_texture_cr",
        entropy: "fast422_texture_entropy",
        entropy_offsets: "fast422_texture_entropy_offsets",
        entropy_lens: "fast422_texture_entropy_lens",
        entropy_checkpoints: "fast422_texture_entropy_checkpoints",
        status: "fast422_texture_status",
    };
    const FULL_RGB_BATCH_TIMING_TAG: &'static str = "metal_fast422_batch";
    const TEXTURE_BOUNDARY_META_WORDS: usize = FAST422_TEXTURE_BOUNDARY_META_WORDS;
    const TEXTURE_BOUNDARY_SAMPLE_BYTES: usize = FAST422_TEXTURE_BOUNDARY_SAMPLE_BYTES;
    const TEXTURE_BOUNDARY_META_KEY: &'static str = "fast422_texture_boundary_meta";
    const TEXTURE_BOUNDARY_SAMPLES_KEY: &'static str = "fast422_texture_boundary_samples";
    const TEXTURE_VERTICAL_REPAIR: Option<FastVerticalRepairSpec> = None;

    fn from_batched<'a>(packet: &BatchedFastPacket<'a>) -> Option<&'a Self> {
        match packet {
            BatchedFastPacket::Fast422(packet) => Some(packet),
            _ => None,
        }
    }
    fn to_batched(&self) -> BatchedFastPacket<'_> {
        BatchedFastPacket::Fast422(self)
    }
    fn decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_decode_pipeline
    }
    fn region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_region_decode_pipeline
    }
    fn scaled_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_scaled_decode_pipeline
    }
    fn scaled_region_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_scaled_region_decode_pipeline
    }
    fn pack_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Option<&ComputePipelineState> {
        pack_422_pipeline_for_format(runtime, fmt)
    }
    fn pack_windowed_pipeline_for_format(
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> &ComputePipelineState {
        pack_422_windowed_pipeline_for_format(runtime, fmt)
    }
    fn scaled_region_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_scaled_region_batch_decode_pipeline
    }
    fn pack_windowed_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_422_windowed_rgb_batch_pipeline
    }
    fn pack_windowed_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_422_windowed_rgba_texture_pipeline
    }
    fn full_rgb_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_batch_decode_pipeline
    }
    fn pack_full_rgb_batch_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_422_rgb_batch_pipeline
    }
    fn pack_rgba_texture_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.pack_422_rgba_texture_pipeline
    }
    fn rgba_texture_batch_decode_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_rgba_texture_batch_decode_pipeline
    }
    fn rgba_texture_boundary_pipeline(runtime: &MetalRuntime) -> &ComputePipelineState {
        &runtime.fast422_rgba_texture_boundary_pipeline
    }
    fn full_rgb_batch_timing_enabled() -> bool {
        false
    }
    fn texture_mcu_dispatch_threads(_total_mcus: usize) -> Result<Option<u32>, Error> {
        Ok(None)
    }
    fn texture_repair_record_count(
        _tile_count: usize,
        _total_mcus: usize,
        total_decode_threads: u32,
    ) -> Result<usize, Error> {
        Ok(total_decode_threads as usize)
    }
    fn horizontal_repair_threads(
        _first: &Self,
        segment_count_u32: u32,
        _mcu_threads: Option<u32>,
    ) -> Option<u32> {
        (segment_count_u32 > 1).then_some(segment_count_u32)
    }
    fn encode_extra_texture_repair_passes(
        _runtime: &MetalRuntime,
        _ctx: &FastTextureRepairCtx<'_>,
    ) -> Result<(), Error> {
        Ok(())
    }
    #[cfg(test)]
    fn split_coeff_idct_pipelines(
        _runtime: &MetalRuntime,
    ) -> Option<(&ComputePipelineState, &ComputePipelineState)> {
        None
    }
    #[cfg(test)]
    const SPLIT_TEXTURE_SCRATCH_KEYS: (&'static str, &'static str) = (
        "fast422_texture_coeff_blocks",
        "fast422_texture_dc_only_flags",
    );
}

fn fast_subsampled_params<P: FastSubsampledPacket>(
    packet: &P,
    fmt: PixelFormat,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!(
            "unsupported JPEG Metal {} pixel format {fmt:?}",
            P::FAMILY_NAME
        ),
    })?;
    let out_stride = packet.dimensions().0 as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: packet.dimensions().0,
        height: packet.dimensions().1,
        chroma_width: packet.dimensions().0.div_ceil(2),
        chroma_height: P::chroma_height(packet.dimensions().1),
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        restart_interval_mcus: packet.restart_interval_mcus(),
        restart_offset_count: checked_entropy_segment_count(
            packet.restart_interval_mcus(),
            packet.restart_offsets().len(),
            packet.entropy_checkpoints().len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes().len(), P::ENTROPY_PAYLOAD_CTX)?,
        out_stride: checked_u32(out_stride, P::OUTPUT_STRIDE_CTX)?,
        alpha: u32::from(u8::MAX),
        out_format,
        origin_x: 0,
        origin_y: 0,
    })
}

fn fast_subsampled_region_params<P: FastSubsampledPacket>(
    packet: &P,
    fmt: PixelFormat,
    source_window: signinum_jpeg::Rect,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!(
            "unsupported JPEG Metal {} pixel format {fmt:?}",
            P::FAMILY_NAME
        ),
    })?;
    let out_stride = source_window.w as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: source_window.w,
        height: source_window.h,
        chroma_width: source_window.w.div_ceil(2),
        chroma_height: P::chroma_height(source_window.h),
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        restart_interval_mcus: packet.restart_interval_mcus(),
        restart_offset_count: checked_entropy_segment_count(
            packet.restart_interval_mcus(),
            packet.restart_offsets().len(),
            packet.entropy_checkpoints().len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes().len(), P::ENTROPY_PAYLOAD_CTX)?,
        out_stride: checked_u32(out_stride, P::REGION_OUTPUT_STRIDE_CTX)?,
        alpha: u32::from(u8::MAX),
        out_format,
        origin_x: source_window.x,
        origin_y: source_window.y,
    })
}

fn fast_subsampled_scaled_params<P: FastSubsampledPacket>(
    packet: &P,
    scale: signinum_core::Downscale,
) -> Option<JpegFast420ScaledParams> {
    let scale_shift = match scale {
        signinum_core::Downscale::Half => 1,
        signinum_core::Downscale::Quarter => 2,
        signinum_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    let scaled_width = packet.dimensions().0.div_ceil(denom);
    let scaled_height = packet.dimensions().1.div_ceil(denom);
    Some(JpegFast420ScaledParams {
        scaled_width,
        scaled_height,
        chroma_width: scaled_width.div_ceil(2),
        chroma_height: P::chroma_height(scaled_height),
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        restart_interval_mcus: packet.restart_interval_mcus(),
        restart_offset_count: optional_entropy_segment_count(
            packet.restart_interval_mcus(),
            packet.restart_offsets().len(),
            packet.entropy_checkpoints().len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes().len(), P::SCALED_ENTROPY_PAYLOAD_CTX)
            .ok()?,
        scale_shift,
        origin_x: 0,
        origin_y: 0,
    })
}

fn fast_subsampled_scaled_region_params<P: FastSubsampledPacket>(
    packet: &P,
    scale: signinum_core::Downscale,
    source_window: signinum_jpeg::Rect,
) -> Option<JpegFast420ScaledParams> {
    let full = fast_subsampled_scaled_params(packet, scale)?;
    Some(JpegFast420ScaledParams {
        scaled_width: source_window.w,
        scaled_height: source_window.h,
        chroma_width: source_window.w.div_ceil(2),
        chroma_height: P::chroma_height(source_window.h),
        origin_x: source_window.x,
        origin_y: source_window.y,
        ..full
    })
}

fn fast_subsampled_full_mcu_window<P: FastSubsampledPacket>(
    dims: (u32, u32),
    roi: signinum_jpeg::Rect,
) -> signinum_jpeg::Rect {
    let x0 = (roi.x / P::MCU_WIDTH) * P::MCU_WIDTH;
    let y0 = (roi.y / P::MCU_HEIGHT) * P::MCU_HEIGHT;
    let x1 = (roi.x + roi.w).div_ceil(P::MCU_WIDTH) * P::MCU_WIDTH;
    let y1 = (roi.y + roi.h).div_ceil(P::MCU_HEIGHT) * P::MCU_HEIGHT;
    signinum_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(dims.0).saturating_sub(x0),
        h: y1.min(dims.1).saturating_sub(y0),
    }
}

fn fast_subsampled_full_mcu_scaled_window<P: FastSubsampledPacket>(
    scaled_dims: (u32, u32),
    roi: signinum_jpeg::Rect,
    scale_shift: u32,
) -> signinum_jpeg::Rect {
    let mcu_width = P::MCU_WIDTH >> scale_shift;
    let mcu_height = P::MCU_HEIGHT >> scale_shift;
    let x0 = (roi.x / mcu_width) * mcu_width;
    let y0 = (roi.y / mcu_height) * mcu_height;
    let x1 = (roi.x + roi.w).div_ceil(mcu_width) * mcu_width;
    let y1 = (roi.y + roi.h).div_ceil(mcu_height) * mcu_height;
    signinum_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(scaled_dims.0).saturating_sub(x0),
        h: y1.min(scaled_dims.1).saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
fn fast444_params(packet: &JpegFast444PacketV1) -> Result<JpegFast444Params, Error> {
    Ok(JpegFast444Params {
        width: packet.dimensions.0,
        height: packet.dimensions.1,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: checked_entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes.len(), "fast444 entropy payload")?,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
fn fast444_region_params(
    packet: &JpegFast444PacketV1,
    roi: signinum_jpeg::Rect,
) -> Result<JpegFast444Params, Error> {
    Ok(JpegFast444Params {
        width: roi.w,
        height: roi.h,
        origin_x: roi.x,
        origin_y: roi.y,
        ..fast444_params(packet)?
    })
}

#[cfg(target_os = "macos")]
fn fast444_scaled_params(
    packet: &JpegFast444PacketV1,
    scale: signinum_core::Downscale,
) -> Option<JpegFast444ScaledParams> {
    let scale_shift = match scale {
        signinum_core::Downscale::Half => 1,
        signinum_core::Downscale::Quarter => 2,
        signinum_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    Some(JpegFast444ScaledParams {
        scaled_width: packet.dimensions.0.div_ceil(denom),
        scaled_height: packet.dimensions.1.div_ceil(denom),
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: optional_entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes.len(), "fast444 scaled entropy payload")
            .ok()?,
        scale_shift,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
fn fast444_scaled_region_params(
    packet: &JpegFast444PacketV1,
    scale: signinum_core::Downscale,
    roi: signinum_jpeg::Rect,
) -> Option<JpegFast444ScaledParams> {
    Some(JpegFast444ScaledParams {
        scaled_width: roi.w,
        scaled_height: roi.h,
        origin_x: roi.x,
        origin_y: roi.y,
        ..fast444_scaled_params(packet, scale)?
    })
}

#[cfg(target_os = "macos")]
fn fast_subsampled_windowed_pack_params_for_dims<P: FastSubsampledPacket>(
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
) -> Result<JpegFast420WindowedPackParams, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!(
            "unsupported JPEG Metal {} pixel format {fmt:?}",
            P::FAMILY_NAME
        ),
    })?;
    let out_stride = roi.w as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420WindowedPackParams {
        src_width: dims.0,
        src_height: dims.1,
        chroma_width: dims.0.div_ceil(2),
        chroma_height: P::chroma_height(dims.1),
        src_x: roi.x,
        src_y: roi.y,
        width: roi.w,
        height: roi.h,
        out_stride: checked_u32(
            out_stride,
            &format!("{} windowed output stride", P::FAMILY_NAME),
        )?,
        alpha: u32::from(u8::MAX),
        out_format,
    })
}

#[cfg(target_os = "macos")]
fn decode_error_from_cpu(
    decoder: &CpuDecoder<'_>,
    fmt: PixelFormat,
    status: JpegDecodeStatus,
) -> Error {
    if let Err(err) = decoder.decode(fmt) {
        Error::Decode(err)
    } else {
        let reason = match status.code {
            FAST420_STATUS_TRUNCATED => "truncated entropy stream",
            FAST420_STATUS_HUFFMAN => "invalid Huffman stream",
            _ => "unexpected Metal fast420 failure",
        };
        Error::MetalKernel {
            message: format!("{reason} at entropy byte {}", status.position),
        }
    }
}

#[cfg(target_os = "macos")]
fn restart_offsets_buffer(device: &Device, restart_offsets: &[u32]) -> Result<Buffer, Error> {
    if restart_offsets.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal restart offsets must contain at least one entry".to_string(),
        });
    }
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            restart_offsets.as_ptr().cast::<u8>(),
            size_of_val(restart_offsets),
        )
    };
    Ok(new_shared_buffer_with_data(device, bytes))
}

#[cfg(target_os = "macos")]
fn entropy_checkpoints_buffer(
    device: &Device,
    entropy_checkpoints: &[JpegEntropyCheckpointV1],
) -> Result<Buffer, Error> {
    if entropy_checkpoints.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal entropy checkpoints must contain at least one entry".to_string(),
        });
    }
    let checkpoints = entropy_checkpoint_hosts(entropy_checkpoints)?;
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            checkpoints.as_ptr().cast::<u8>(),
            size_of_val(checkpoints.as_slice()),
        )
    };
    Ok(new_shared_buffer_with_data(device, bytes))
}

#[cfg(target_os = "macos")]
fn entropy_checkpoint_hosts(
    entropy_checkpoints: &[JpegEntropyCheckpointV1],
) -> Result<Vec<JpegEntropyCheckpointHost>, Error> {
    if entropy_checkpoints.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal entropy checkpoints must contain at least one entry".to_string(),
        });
    }
    Ok(entropy_checkpoints
        .iter()
        .copied()
        .map(JpegEntropyCheckpointHost::from)
        .collect::<Vec<_>>())
}

#[cfg(target_os = "macos")]
fn entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> u32 {
    let len = if restart_interval_mcus == 0 {
        entropy_checkpoints_len
    } else {
        restart_offsets_len
    };
    u32::try_from(len)
        .expect("JPEG Metal entropy segment count fits in u32")
        .max(1)
}

#[cfg(target_os = "macos")]
fn optional_entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> Option<u32> {
    let len = if restart_interval_mcus == 0 {
        entropy_checkpoints_len
    } else {
        restart_offsets_len
    };
    u32::try_from(len).ok().map(|count| count.max(1))
}

#[cfg(target_os = "macos")]
fn checked_entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> Result<u32, Error> {
    optional_entropy_segment_count(
        restart_interval_mcus,
        restart_offsets_len,
        entropy_checkpoints_len,
    )
    .ok_or_else(|| Error::MetalKernel {
        message: "JPEG Metal entropy segment count does not fit in u32".to_string(),
    })
}

#[cfg(target_os = "macos")]
fn restart_work_for_mcu_range(
    restart_offsets: &[u32],
    restart_interval_mcus: u32,
    total_mcus: u32,
    first_mcu: u32,
    end_mcu: u32,
) -> (u32, &[u32]) {
    if restart_interval_mcus == 0 || restart_offsets.len() <= 1 {
        return (0, restart_offsets);
    }

    let first_mcu = first_mcu.min(total_mcus);
    let end_mcu = end_mcu.min(total_mcus).max(first_mcu + 1);
    let restart_offset_count =
        u32::try_from(restart_offsets.len()).expect("JPEG Metal restart offsets fit in u32");
    let first_segment = (first_mcu / restart_interval_mcus).min(restart_offset_count - 1);
    let end_segment = end_mcu
        .div_ceil(restart_interval_mcus)
        .min(restart_offset_count)
        .max(first_segment + 1);
    (
        first_segment * restart_interval_mcus,
        &restart_offsets[first_segment as usize..end_segment as usize],
    )
}

#[cfg(target_os = "macos")]
fn mcu_range_for_rect(
    rect: signinum_jpeg::Rect,
    mcus_per_row: u32,
    mcu_rows: u32,
    mcu_width: u32,
    mcu_height: u32,
) -> (u32, u32) {
    if rect.w == 0 || rect.h == 0 || mcus_per_row == 0 || mcu_rows == 0 {
        return (0, 0);
    }

    let max_col = mcus_per_row - 1;
    let max_row = mcu_rows - 1;
    let last_x = rect.x.saturating_add(rect.w).saturating_sub(1);
    let last_y = rect.y.saturating_add(rect.h).saturating_sub(1);
    let first_col = (rect.x / mcu_width).min(max_col);
    let last_col = (last_x / mcu_width).min(max_col);
    let first_row = (rect.y / mcu_height).min(max_row);
    let last_row = (last_y / mcu_height).min(max_row);
    let first_mcu = first_row * mcus_per_row + first_col;
    let end_mcu = last_row * mcus_per_row + last_col + 1;
    (first_mcu, end_mcu)
}

#[cfg(target_os = "macos")]
fn entropy_decode_thread_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> u32 {
    entropy_segment_count(
        restart_interval_mcus,
        restart_offsets_len,
        entropy_checkpoints_len,
    )
}

#[cfg(target_os = "macos")]
fn decode_status_buffer(device: &Device, count: u32) -> Buffer {
    let statuses = vec![JpegDecodeStatus::default(); count as usize];
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            statuses.as_ptr().cast::<u8>(),
            size_of_val(statuses.as_slice()),
        )
    };
    new_shared_buffer_with_data(device, bytes)
}

#[cfg(target_os = "macos")]
fn first_decode_error_status(buffer: &Buffer, count: u32) -> Option<JpegDecodeStatus> {
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let statuses = unsafe {
        core::slice::from_raw_parts(buffer.contents().cast::<JpegDecodeStatus>(), count as usize)
    };
    statuses
        .iter()
        .copied()
        .find(|status| status.code != FAST420_STATUS_OK)
}

#[cfg(target_os = "macos")]
enum BatchedFastPacket<'a> {
    Fast420(&'a JpegFast420PacketV1),
    Fast422(&'a JpegFast422PacketV1),
    Fast444(&'a JpegFast444PacketV1, PlaneMode),
}

#[cfg(target_os = "macos")]
struct BatchedDecodeItem {
    request_index: usize,
    surface: Surface,
    status_buffer: Buffer,
    decode_threads: u32,
    _decode_resources: Vec<Buffer>,
}

#[cfg(target_os = "macos")]
#[derive(Default)]
struct BatchDeviceBufferCache {
    packet_buffers: Vec<SharedPacketDeviceBuffers>,
}

#[cfg(target_os = "macos")]
struct SharedPacketDeviceBuffers {
    entropy_ptr: usize,
    entropy_len: usize,
    checkpoints_ptr: usize,
    checkpoints_len: usize,
    entropy_buffer: Buffer,
    entropy_checkpoints_buffer: Buffer,
}

#[cfg(target_os = "macos")]
impl BatchDeviceBufferCache {
    fn packet_buffers(
        &mut self,
        runtime: &MetalRuntime,
        entropy_bytes: &[u8],
        entropy_checkpoints: &[JpegEntropyCheckpointV1],
    ) -> Result<(Buffer, Buffer), Error> {
        let entropy_ptr = entropy_bytes.as_ptr() as usize;
        let entropy_len = entropy_bytes.len();
        let checkpoints_ptr = entropy_checkpoints.as_ptr() as usize;
        let checkpoints_len = entropy_checkpoints.len();
        if let Some(entry) = self.packet_buffers.iter().find(|entry| {
            entry.entropy_ptr == entropy_ptr
                && entry.entropy_len == entropy_len
                && entry.checkpoints_ptr == checkpoints_ptr
                && entry.checkpoints_len == checkpoints_len
        }) {
            return Ok((
                entry.entropy_buffer.clone(),
                entry.entropy_checkpoints_buffer.clone(),
            ));
        }

        let entropy_buffer = runtime.device.new_buffer_with_data(
            entropy_bytes.as_ptr().cast(),
            entropy_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let entropy_checkpoints_buffer =
            entropy_checkpoints_buffer(&runtime.device, entropy_checkpoints)?;
        self.packet_buffers.push(SharedPacketDeviceBuffers {
            entropy_ptr,
            entropy_len,
            checkpoints_ptr,
            checkpoints_len,
            entropy_buffer: entropy_buffer.clone(),
            entropy_checkpoints_buffer: entropy_checkpoints_buffer.clone(),
        });
        Ok((entropy_buffer, entropy_checkpoints_buffer))
    }
}

#[cfg(target_os = "macos")]
fn request_allows_batched_packet(
    requests: &[batch::QueuedRequest],
    request: &batch::QueuedRequest,
    restart_interval_mcus: u32,
    dimensions: (u32, u32),
) -> bool {
    match request.backend {
        BackendRequest::Metal => true,
        BackendRequest::Auto => match request.op {
            batch::BatchOp::RegionScaled { .. } => false,
            _ => {
                requests.len() >= AUTO_METAL_MIN_BATCH_REQUESTS
                    && (restart_interval_mcus != 0
                        || auto_batch_work_is_large_enough(request, dimensions))
            }
        },
        BackendRequest::Cpu | BackendRequest::Cuda => false,
    }
}

#[cfg(target_os = "macos")]
fn auto_batch_work_is_large_enough(request: &batch::QueuedRequest, dimensions: (u32, u32)) -> bool {
    let dims = match request.op {
        batch::BatchOp::Full | batch::BatchOp::Scaled(_) => dimensions,
        batch::BatchOp::Region(roi) => (roi.w, roi.h),
        batch::BatchOp::RegionScaled { .. } => return false,
    };
    dims.0 >= AUTO_METAL_MIN_BATCH_EDGE && dims.1 >= AUTO_METAL_MIN_BATCH_EDGE
}

#[cfg(target_os = "macos")]
fn batched_fast_packets(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<BatchedFastPacket<'_>>>, Error> {
    if requests.is_empty() {
        return Ok(None);
    }

    let mut packets = Vec::with_capacity(requests.len());
    for request in requests {
        let batchable_op = match request.op {
            batch::BatchOp::Full
            | batch::BatchOp::Region(_)
            | batch::BatchOp::Scaled(
                signinum_core::Downscale::Half
                | signinum_core::Downscale::Quarter
                | signinum_core::Downscale::Eighth,
            )
            | batch::BatchOp::RegionScaled {
                scale:
                    signinum_core::Downscale::Half
                    | signinum_core::Downscale::Quarter
                    | signinum_core::Downscale::Eighth,
                ..
            } => true,
            batch::BatchOp::Scaled(_) | batch::BatchOp::RegionScaled { .. } => false,
        };
        if !batchable_op
            || !matches!(
                request.backend,
                BackendRequest::Auto | BackendRequest::Metal
            )
            || pixel_format_to_out_format(request.fmt).is_none()
        {
            return Ok(None);
        }

        if let Some(packet) = request.fast420_packet.as_deref() {
            if !request_allows_batched_packet(
                requests,
                request,
                packet.restart_interval_mcus,
                packet.dimensions,
            ) {
                return Ok(None);
            }
            packets.push(BatchedFastPacket::Fast420(packet));
            continue;
        }

        if let Some(packet) = request.fast422_packet.as_deref() {
            if !request_allows_batched_packet(
                requests,
                request,
                packet.restart_interval_mcus,
                packet.dimensions,
            ) {
                return Ok(None);
            }
            packets.push(BatchedFastPacket::Fast422(packet));
            continue;
        }

        if let Some(packet) = request.fast444_packet.as_deref() {
            if !request_allows_batched_packet(
                requests,
                request,
                packet.restart_interval_mcus,
                packet.dimensions,
            ) {
                return Ok(None);
            }
            let decoder = CpuDecoder::new(request.input.as_ref())?;
            packets.push(BatchedFastPacket::Fast444(
                packet,
                fast444_plane_mode(&decoder),
            ));
            continue;
        }

        return Ok(None);
    }

    Ok(Some(packets))
}

#[cfg(target_os = "macos")]
fn core_rect_to_jpeg(rect: Rect) -> signinum_jpeg::Rect {
    signinum_jpeg::Rect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_jpeg_pack_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane0: &Buffer,
    plane1: Option<&Buffer>,
    plane2: Option<&Buffer>,
    dims: (u32, u32),
    mode: PlaneMode,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    match (mode, fmt) {
        (PlaneMode::Gray | PlaneMode::YCbCr, PixelFormat::Gray8) => {
            return Ok(Surface::from_metal_buffer(plane0.clone(), dims, fmt));
        }
        (
            PlaneMode::Gray | PlaneMode::YCbCr | PlaneMode::Rgb,
            PixelFormat::Rgb8 | PixelFormat::Rgba8,
        )
        | (PlaneMode::Rgb, PixelFormat::Gray8) => {}
        _ => {
            return Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            });
        }
    }

    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let params = JpegPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
        alpha: u32::from(u8::MAX),
        mode: match mode {
            PlaneMode::Gray => MODE_GRAY,
            PlaneMode::YCbCr => MODE_YCBCR,
            PlaneMode::Rgb => MODE_RGB,
        },
        out_format: match fmt {
            PixelFormat::Gray8 => OUT_GRAY,
            PixelFormat::Rgb8 => OUT_RGB,
            PixelFormat::Rgba8 => OUT_RGBA,
            _ => unreachable!("validated by caller"),
        },
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
    encoder.set_buffer(0, Some(plane0), 0);
    encoder.set_buffer(1, plane1.map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(2, plane2.map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<JpegPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, dims);
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_region_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<BatchedDecodeItem, Error> {
    let roi = core_rect_to_jpeg(roi);
    let source_window = fast_subsampled_full_mcu_window::<P>(packet.dimensions(), roi);
    let mut params = fast_subsampled_region_params(packet, fmt, source_window)?;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row(),
        packet.mcu_rows(),
        P::MCU_WIDTH,
        P::MCU_HEIGHT,
    );
    let total_mcus = packet.mcus_per_row() * packet.mcu_rows();
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        packet.restart_offsets(),
        packet.restart_interval_mcus(),
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    )?;

    let local_roi = signinum_jpeg::Rect {
        x: roi.x - source_window.x,
        y: roi.y - source_window.y,
        w: roi.w,
        h: roi.h,
    };
    let pack_params = fast_subsampled_windowed_pack_params_for_dims::<P>(
        (source_window.w, source_window.h),
        fmt,
        local_roi,
    )?;
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len =
        source_window.w.div_ceil(2) as usize * P::chroma_height(source_window.h) as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, false);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes().as_ptr().cast(),
        packet.entropy_bytes().len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let decode_pipeline = P::region_decode_pipeline(runtime);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = P::pack_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    bind_three_plane_pack::<JpegFast420WindowedPackParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (roi.w, roi.h));
    pack_encoder.end_encoding();

    Ok(BatchedDecodeItem {
        request_index,
        surface: Surface::from_metal_buffer(out_buffer, (roi.w, roi.h), fmt),
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_scaled_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(params) = fast_subsampled_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal {} scale {scale:?}", P::FAMILY_NAME),
        });
    };

    let y_len = params.scaled_width as usize * params.scaled_height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, fmt == PixelFormat::Gray8);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        packet.restart_offsets().len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes().as_ptr().cast(),
        packet.entropy_bytes().len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, packet.restart_offsets())?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let decode_pipeline = P::scaled_decode_pipeline(runtime);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    let out_buffer = (fmt != PixelFormat::Gray8).then(|| {
        runtime.device.new_buffer(
            (params.scaled_width as usize * fmt.bytes_per_pixel() * params.scaled_height as usize)
                as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    if let Some(out_buffer) = out_buffer.as_ref() {
        let pack_params = JpegFast420Params {
            width: params.scaled_width,
            height: params.scaled_height,
            chroma_width: params.chroma_width,
            chroma_height: params.chroma_height,
            mcus_per_row: params.mcus_per_row,
            mcu_rows: params.mcu_rows,
            restart_interval_mcus: params.restart_interval_mcus,
            restart_offset_count: params.restart_offset_count,
            restart_start_mcu: params.restart_start_mcu,
            entropy_len: params.entropy_len,
            out_stride: checked_u32(
                params.scaled_width as usize * fmt.bytes_per_pixel(),
                "scaled output stride",
            )?,
            alpha: u32::from(u8::MAX),
            out_format: pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            })?,
            origin_x: 0,
            origin_y: 0,
        };
        let Some(pack_pipeline) = P::pack_pipeline_for_format(runtime, fmt) else {
            return Err(Error::MetalKernel {
                message: format!(
                    "unsupported JPEG Metal {} pixel format {fmt:?}",
                    P::FAMILY_NAME
                ),
            });
        };
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&cb_plane), 0);
        pack_encoder.set_buffer(2, Some(&cr_plane), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const pack_params).cast(),
        );
        dispatch_2d_pipeline(
            pack_encoder,
            pack_pipeline,
            (params.scaled_width, params.scaled_height),
        );
        pack_encoder.end_encoding();
    }

    let surface = match out_buffer {
        Some(out_buffer) => {
            Surface::from_metal_buffer(out_buffer, (params.scaled_width, params.scaled_height), fmt)
        }
        None => Surface::from_metal_buffer(
            y_plane.clone(),
            (params.scaled_width, params.scaled_height),
            fmt,
        ),
    };

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_fast_subsampled_scaled_region_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    device_buffer_cache: &mut BatchDeviceBufferCache,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
    roi: Rect,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(full_params) = fast_subsampled_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal {} scale {scale:?}", P::FAMILY_NAME),
        });
    };
    let scaled_roi = roi.scaled_covering(scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let source_window = fast_subsampled_full_mcu_scaled_window::<P>(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let Some(mut decode_params) =
        fast_subsampled_scaled_region_params(packet, scale, source_window)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "unsupported JPEG Metal {} scaled region {scale:?}",
                P::FAMILY_NAME
            ),
        });
    };
    let mcu_width = P::MCU_WIDTH >> decode_params.scale_shift;
    let mcu_height = P::MCU_HEIGHT >> decode_params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row(),
        packet.mcu_rows(),
        mcu_width,
        mcu_height,
    );
    let total_mcus = packet.mcus_per_row() * packet.mcu_rows();
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        packet.restart_offsets(),
        packet.restart_interval_mcus(),
        total_mcus,
        first_mcu,
        end_mcu,
    );
    decode_params.restart_start_mcu = restart_start_mcu;
    decode_params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    )?;
    let local_roi = signinum_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params = fast_subsampled_windowed_pack_params_for_dims::<P>(
        (source_window.w, source_window.h),
        fmt,
        local_roi,
    )?;
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len =
        source_window.w.div_ceil(2) as usize * P::chroma_height(source_window.h) as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, false);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let (entropy_buffer, entropy_checkpoints_buffer) = device_buffer_cache.packet_buffers(
        runtime,
        packet.entropy_bytes(),
        packet.entropy_checkpoints(),
    )?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let decode_pipeline = P::scaled_region_decode_pipeline(runtime);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &decode_params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * scaled_roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = P::pack_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    bind_three_plane_pack::<JpegFast420WindowedPackParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (scaled_roi.w, scaled_roi.h));
    pack_encoder.end_encoding();

    Ok(BatchedDecodeItem {
        request_index,
        surface: Surface::from_metal_buffer(out_buffer, (scaled_roi.w, scaled_roi.h), fmt),
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
) -> Result<BatchedDecodeItem, Error> {
    let params = fast_subsampled_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, fmt == PixelFormat::Gray8);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        packet.restart_offsets().len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes().as_ptr().cast(),
        packet.entropy_bytes().len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, packet.restart_offsets())?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let decode_pipeline = P::decode_pipeline(runtime);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    let surface = if fmt == PixelFormat::Gray8 {
        Surface::from_metal_buffer(y_plane.clone(), packet.dimensions(), fmt)
    } else {
        let Some(pack_pipeline) = P::pack_pipeline_for_format(runtime, fmt) else {
            return Err(Error::MetalKernel {
                message: format!(
                    "unsupported JPEG Metal {} pixel format {fmt:?}",
                    P::FAMILY_NAME
                ),
            });
        };
        let out_buffer = runtime.device.new_buffer(
            (params.out_stride as usize * params.height as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        bind_three_plane_pack::<JpegFast420Params>(
            pack_encoder,
            [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(pack_encoder, pack_pipeline, packet.dimensions());
        pack_encoder.end_encoding();
        Surface::from_metal_buffer(out_buffer, packet.dimensions(), fmt)
    };

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

/// Route one batch request to the family's encode item for its op.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_fast_subsampled_op_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    device_buffer_cache: &mut BatchDeviceBufferCache,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
    op: batch::BatchOp,
) -> Result<BatchedDecodeItem, Error> {
    match op {
        batch::BatchOp::Full => {
            encode_fast_subsampled_batch_item(runtime, command_buffer, request_index, packet, fmt)
        }
        batch::BatchOp::Region(roi) => encode_fast_subsampled_region_batch_item(
            runtime,
            command_buffer,
            request_index,
            packet,
            fmt,
            roi,
        ),
        batch::BatchOp::Scaled(scale) => encode_fast_subsampled_scaled_batch_item(
            runtime,
            command_buffer,
            request_index,
            packet,
            fmt,
            scale,
        ),
        batch::BatchOp::RegionScaled { roi, scale } => {
            encode_fast_subsampled_scaled_region_batch_item(
                runtime,
                command_buffer,
                device_buffer_cache,
                request_index,
                packet,
                fmt,
                roi,
                scale,
            )
        }
    }
}

#[cfg(target_os = "macos")]
fn encode_fast444_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<BatchedDecodeItem, Error> {
    let roi = core_rect_to_jpeg(roi);
    let mut params = fast444_region_params(packet, roi)?;
    let (first_mcu, end_mcu) = mcu_range_for_rect(roi, packet.mcus_per_row, packet.mcu_rows, 8, 8);
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    )?;

    let plane_len = params.width as usize * params.height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let cb_plane = new_private_buffer(&runtime.device, plane_len);
    let cr_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_region_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        (roi.w, roi.h),
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast444_scaled_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(params) = fast444_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast444 scale {scale:?}"),
        });
    };

    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let cb_plane = new_private_buffer(&runtime.device, plane_len);
    let cr_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        (params.scaled_width, params.scaled_height),
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_fast444_scaled_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    device_buffer_cache: &mut BatchDeviceBufferCache,
    request_index: usize,
    packet: &JpegFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
    roi: Rect,
    scale: signinum_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let scaled_roi = roi.scaled_covering(scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let Some(mut params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast444 scaled region {scale:?}"),
        });
    };
    let mcu_size = 8u32 >> params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        scaled_roi,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_size,
        mcu_size,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    )?;

    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let cb_plane = new_private_buffer(&runtime.device, plane_len);
    let cr_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let (entropy_buffer, entropy_checkpoints_buffer) = device_buffer_cache.packet_buffers(
        runtime,
        &packet.entropy_bytes,
        &packet.entropy_checkpoints,
    )?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_region_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        (scaled_roi.w, scaled_roi.h),
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast444_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
) -> Result<BatchedDecodeItem, Error> {
    let params = fast444_params(packet)?;
    let plane_len = params.width as usize * params.height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let cb_plane = new_private_buffer(&runtime.device, plane_len);
    let cr_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        packet.dimensions,
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

fn fast_subsampled_packets_share_full_rgb_batch_shape<P: FastSubsampledPacket>(
    first: &P,
    packet: &P,
    segment_count: usize,
) -> bool {
    (P::FULL_RGB_BATCH_SUPPORTS_RESTART || packet.restart_interval_mcus() == 0)
        && packet.dimensions() == first.dimensions()
        && packet.mcus_per_row() == first.mcus_per_row()
        && packet.mcu_rows() == first.mcu_rows()
        && packet.entropy_checkpoints().len() == segment_count
        && packet.y_quant() == first.y_quant()
        && packet.cb_quant() == first.cb_quant()
        && packet.cr_quant() == first.cr_quant()
        && packet.y_dc_table() == first.y_dc_table()
        && packet.y_ac_table() == first.y_ac_table()
        && packet.cb_dc_table() == first.cb_dc_table()
        && packet.cb_ac_table() == first.cb_ac_table()
        && packet.cr_dc_table() == first.cr_dc_table()
        && packet.cr_ac_table() == first.cr_ac_table()
}

#[cfg(target_os = "macos")]
fn fast_subsampled_full_rgb_batch_groups<P: FastSubsampledPacket>(
    packets: &[&P],
) -> Option<Vec<Vec<usize>>> {
    let mut groups = Vec::<Vec<usize>>::new();
    'packet: for (index, packet) in packets.iter().copied().enumerate() {
        if packet.entropy_bytes().is_empty() || packet.entropy_checkpoints().is_empty() {
            return None;
        }

        for group in &mut groups {
            let first = packets[group[0]];
            if fast_subsampled_packets_share_full_rgb_batch_shape(
                first,
                packet,
                first.entropy_checkpoints().len(),
            ) {
                group.push(index);
                continue 'packet;
            }
        }
        groups.push(vec![index]);
    }
    Some(groups)
}

#[cfg(target_os = "macos")]
fn checked_u32(value: usize, label: &str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::MetalKernel {
        message: format!("JPEG Metal {label} does not fit in u32"),
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastBatchDecodeMode {
    Fused,
    #[cfg(test)]
    SplitCoeffIdct,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default)]
struct FastBatchTiming {
    accepted: Duration,
    entropy_concat: Duration,
    buffer_alloc: Duration,
    encode_decode: Duration,
    wait_decode: Duration,
    encode_pack: Duration,
    wait_pack: Duration,
    total: Duration,
}

#[cfg(target_os = "macos")]
impl FastBatchTiming {
    fn micros(duration: Duration) -> u128 {
        duration.as_micros()
    }

    fn log(
        self,
        tag: &'static str,
        label: &str,
        tile_count: usize,
        dimensions: (u32, u32),
        segment_count: usize,
    ) {
        signinum_profile::emit_profile_row_now(
            "jpeg",
            "decode",
            tag,
            &[
                ("mode", label.to_string()),
                ("tiles", tile_count.to_string()),
                ("dimensions", format!("{}x{}", dimensions.0, dimensions.1)),
                ("segments", segment_count.to_string()),
                ("accepted_us", Self::micros(self.accepted).to_string()),
                (
                    "entropy_concat_us",
                    Self::micros(self.entropy_concat).to_string(),
                ),
                (
                    "buffer_alloc_us",
                    Self::micros(self.buffer_alloc).to_string(),
                ),
                (
                    "encode_decode_us",
                    Self::micros(self.encode_decode).to_string(),
                ),
                ("wait_decode_us", Self::micros(self.wait_decode).to_string()),
                ("encode_pack_us", Self::micros(self.encode_pack).to_string()),
                ("wait_pack_us", Self::micros(self.wait_pack).to_string()),
                ("total_us", Self::micros(self.total).to_string()),
            ],
        );
    }
}

#[cfg(target_os = "macos")]
fn fast_batch_decode_mode() -> FastBatchDecodeMode {
    FastBatchDecodeMode::Fused
}

#[cfg(target_os = "macos")]
struct BatchEntropyBuffers {
    payload: Buffer,
    offsets: Buffer,
    lens: Buffer,
    checkpoints: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct BatchEntropyBufferKeys {
    payload: &'static str,
    offsets: &'static str,
    lens: &'static str,
    checkpoints: &'static str,
}

#[cfg(target_os = "macos")]
fn batch_entropy_buffers<'a>(
    runtime: &MetalRuntime,
    scratch: &mut MetalBatchScratch,
    keys: BatchEntropyBufferKeys,
    entropy_bytes_iter: impl Iterator<Item = &'a [u8]> + Clone,
    entropy_checkpoints_iter: impl Iterator<Item = &'a [JpegEntropyCheckpointV1]> + Clone,
    tile_count: usize,
    segment_count: usize,
) -> Result<Option<BatchEntropyBuffers>, Error> {
    let total_entropy_len = entropy_bytes_iter
        .clone()
        .map(<[u8]>::len)
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal region scaled batch entropy length overflowed".to_string(),
        })?;
    if total_entropy_len == 0 {
        return Ok(None);
    }

    let mut entropy_bytes = Vec::with_capacity(total_entropy_len);
    let mut entropy_offsets = Vec::with_capacity(tile_count);
    let mut entropy_lens = Vec::with_capacity(tile_count);
    let mut entropy_checkpoints = Vec::with_capacity(tile_count * segment_count);
    for (bytes, checkpoints) in entropy_bytes_iter.zip(entropy_checkpoints_iter) {
        entropy_offsets.push(checked_u32(
            entropy_bytes.len(),
            "region scaled batch entropy offset",
        )?);
        entropy_lens.push(checked_u32(
            bytes.len(),
            "region scaled batch entropy length",
        )?);
        entropy_bytes.extend_from_slice(bytes);
        entropy_checkpoints.extend(checkpoints.iter().copied());
    }

    let checkpoints = entropy_checkpoint_hosts(&entropy_checkpoints)?;
    Ok(Some(BatchEntropyBuffers {
        payload: scratch.shared_buffer_with_bytes(&runtime.device, keys.payload, &entropy_bytes),
        offsets: scratch.shared_buffer_with_slice(&runtime.device, keys.offsets, &entropy_offsets),
        lens: scratch.shared_buffer_with_slice(&runtime.device, keys.lens, &entropy_lens),
        checkpoints: scratch.shared_buffer_with_slice(
            &runtime.device,
            keys.checkpoints,
            &checkpoints,
        ),
    }))
}

#[cfg(target_os = "macos")]
fn region_scaled_batch_error_results(
    requests: &[batch::QueuedRequest],
    status_buffer: &Buffer,
    total_decode_threads: u32,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(status) = first_decode_error_status(status_buffer, total_decode_threads) else {
        return Ok(None);
    };
    let mut results = Vec::with_capacity(requests.len());
    for request in requests {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn texture_batch_error_results(
    requests: &[batch::QueuedRequest],
    status_buffer: &Buffer,
    total_decode_threads: u32,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(status) = first_decode_error_status(status_buffer, total_decode_threads) else {
        return Ok(None);
    };
    let mut results = Vec::with_capacity(requests.len());
    for request in requests {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
struct RegionScaledBatchPlan {
    decode_params: JpegFastRegionScaledBatchParams,
    pack_params: JpegWindowedPackBatchParams,
    y_len: usize,
    chroma_len: usize,
    out_tile_len: usize,
    out_dims: (u32, u32),
}

#[cfg(target_os = "macos")]
fn windowed_texture_pack_params(plan: RegionScaledBatchPlan) -> JpegWindowedTexturePackBatchParams {
    JpegWindowedTexturePackBatchParams {
        src_width: plan.pack_params.src_width,
        src_height: plan.pack_params.src_height,
        chroma_width: plan.pack_params.chroma_width,
        chroma_height: plan.pack_params.chroma_height,
        src_x: plan.pack_params.src_x,
        src_y: plan.pack_params.src_y,
        width: plan.pack_params.width,
        height: plan.pack_params.height,
        tile_index: 0,
        alpha: u32::from(u8::MAX),
    }
}

#[cfg(target_os = "macos")]
fn fast_subsampled_region_scaled_batch_plan<P: FastSubsampledPacket>(
    packet: &P,
    roi: Rect,
    scale: signinum_core::Downscale,
    tile_count: u32,
    segment_count: u32,
) -> Option<RegionScaledBatchPlan> {
    let full_params = fast_subsampled_scaled_params(packet, scale)?;
    let scaled_roi = roi.scaled_covering(scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let source_window = fast_subsampled_full_mcu_scaled_window::<P>(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let decode_params = fast_subsampled_scaled_region_params(packet, scale, source_window)?;
    let local_roi = signinum_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params = fast_subsampled_windowed_pack_params_for_dims::<P>(
        (source_window.w, source_window.h),
        PixelFormat::Rgb8,
        local_roi,
    )
    .ok()?;
    let out_stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    Some(RegionScaledBatchPlan {
        decode_params: JpegFastRegionScaledBatchParams {
            scaled_width: decode_params.scaled_width,
            scaled_height: decode_params.scaled_height,
            chroma_width: decode_params.chroma_width,
            chroma_height: decode_params.chroma_height,
            mcus_per_row: decode_params.mcus_per_row,
            mcu_rows: decode_params.mcu_rows,
            segment_count,
            tile_count,
            scale_shift: decode_params.scale_shift,
            origin_x: decode_params.origin_x,
            origin_y: decode_params.origin_y,
        },
        pack_params: JpegWindowedPackBatchParams {
            src_width: pack_params.src_width,
            src_height: pack_params.src_height,
            chroma_width: pack_params.chroma_width,
            chroma_height: pack_params.chroma_height,
            src_x: pack_params.src_x,
            src_y: pack_params.src_y,
            width: pack_params.width,
            height: pack_params.height,
            tile_count,
            out_stride: checked_u32(out_stride, P::REGION_SCALED_BATCH_OUT_STRIDE_CTX).ok()?,
            alpha: u32::from(u8::MAX),
            mode: MODE_YCBCR,
            out_format: OUT_RGB,
        },
        y_len: source_window.w as usize * source_window.h as usize,
        chroma_len: source_window.w.div_ceil(2) as usize
            * P::chroma_height(source_window.h) as usize,
        out_tile_len: out_stride * scaled_roi.h as usize,
        out_dims: (scaled_roi.w, scaled_roi.h),
    })
}

#[cfg(target_os = "macos")]
struct FastRegionScaledGroup {
    indices: Vec<usize>,
    scale: signinum_core::Downscale,
    plan: RegionScaledBatchPlan,
}

#[cfg(target_os = "macos")]
fn fast_subsampled_region_scaled_batch_groups<P: FastSubsampledMetal>(
    requests: &[batch::QueuedRequest],
    packets: &[&P],
) -> Result<Option<Vec<Vec<usize>>>, Error> {
    let mut groups = Vec::<FastRegionScaledGroup>::new();
    'packet: for (index, (request, packet)) in
        requests.iter().zip(packets.iter().copied()).enumerate()
    {
        if packet.restart_interval_mcus() != 0
            || packet.entropy_bytes().is_empty()
            || packet.entropy_checkpoints().is_empty()
        {
            return Ok(None);
        }
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let segment_count = packet.entropy_checkpoints().len();
        let segment_count_u32 = checked_u32(
            segment_count,
            &format!(
                "{} region scaled texture batch segment count",
                P::FAMILY_NAME
            ),
        )?;
        let Some(plan) =
            fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
        else {
            return Ok(None);
        };

        for group in &mut groups {
            let first = packets[group.indices[0]];
            let first_segment_count = first.entropy_checkpoints().len();
            if scale == group.scale
                && plan == group.plan
                && fast_subsampled_packets_share_full_rgb_batch_shape(
                    first,
                    packet,
                    first_segment_count,
                )
            {
                group.indices.push(index);
                continue 'packet;
            }
        }
        groups.push(FastRegionScaledGroup {
            indices: vec![index],
            scale,
            plan,
        });
    }
    Ok(Some(
        groups.into_iter().map(|group| group.indices).collect(),
    ))
}

#[cfg(target_os = "macos")]
fn batch_output_buffer_or_new(
    runtime: &MetalRuntime,
    output: Option<&crate::MetalBatchOutputBuffer>,
    dimensions: (u32, u32),
    tile_count: usize,
    out_stride: usize,
    out_tile_len: usize,
) -> Result<Buffer, Error> {
    let Some(output) = output else {
        let byte_len = out_tile_len
            .checked_mul(tile_count)
            .ok_or(BufferError::SizeOverflow {
                what: "JPEG Metal batch output bytes",
            })?;
        let byte_len_u64 = u64::try_from(byte_len).map_err(|_| BufferError::SizeOverflow {
            what: "JPEG Metal batch output bytes",
        })?;
        return Ok(runtime
            .device
            .new_buffer(byte_len_u64, MTLResourceOptions::StorageModeShared));
    };

    if output.dimensions() != dimensions
        || output.pixel_format() != PixelFormat::Rgb8
        || output.pitch_bytes() != out_stride
        || output.tile_stride_bytes() < out_tile_len
    {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal batch output buffer shape does not match requested RGB8 tiles",
        });
    }
    if output.tile_capacity() < tile_count {
        return Err(BufferError::OutputTooSmall {
            required: output.tile_stride_bytes().checked_mul(tile_count).ok_or(
                BufferError::SizeOverflow {
                    what: "JPEG Metal batch output bytes",
                },
            )?,
            have: output.byte_len(),
        }
        .into());
    }

    Ok(output.clone_buffer())
}

#[cfg(target_os = "macos")]
type GroupedSurfaceResult = (usize, Result<Surface, Error>);

#[cfg(target_os = "macos")]
type GroupedTextureResult = (usize, Result<crate::MetalTextureTile, Error>);

#[cfg(target_os = "macos")]
fn copy_grouped_surfaces_to_output(
    runtime: &MetalRuntime,
    output: &crate::MetalBatchOutputBuffer,
    dimensions: (u32, u32),
    out_tile_len: usize,
    group_indices: &[usize],
    group_results: Vec<Result<Surface, Error>>,
) -> Result<Vec<GroupedSurfaceResult>, Error> {
    if group_results.len() != group_indices.len() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal grouped buffer result count mismatch".to_string(),
        });
    }

    let output_buffer = output.clone_buffer();
    let mut copies = Vec::<(Buffer, usize, usize)>::new();
    let mut mapped_results = Vec::with_capacity(group_indices.len());
    for (original_index, result) in group_indices.iter().copied().zip(group_results) {
        match result {
            Ok(surface) => {
                let (source, source_offset) =
                    surface.metal_buffer().ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal grouped buffer source was not Metal-backed"
                            .to_string(),
                    })?;
                let destination_offset = original_index
                    .checked_mul(output.tile_stride_bytes())
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal grouped buffer destination offset overflowed"
                            .to_string(),
                    })?;
                copies.push((source.clone(), source_offset, destination_offset));
                mapped_results.push((
                    original_index,
                    Ok(Surface::from_metal_buffer_offset(
                        output_buffer.clone(),
                        dimensions,
                        PixelFormat::Rgb8,
                        destination_offset,
                    )),
                ));
            }
            Err(error) => mapped_results.push((original_index, Err(error))),
        }
    }

    if !copies.is_empty() {
        let command_buffer = runtime.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (source, source_offset, destination_offset) in copies {
            blit.copy_from_buffer(
                &source,
                u64::try_from(source_offset).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal grouped buffer source offset exceeds u64".to_string(),
                })?,
                &output_buffer,
                u64::try_from(destination_offset).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal grouped buffer destination offset exceeds u64".to_string(),
                })?,
                u64::try_from(out_tile_len).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal grouped buffer copy size exceeds u64".to_string(),
                })?,
            );
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    Ok(mapped_results)
}

#[cfg(target_os = "macos")]
fn validate_rgba_texture_batch_output(
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    tile_count: usize,
    out_tile_len: usize,
) -> Result<(), Error> {
    if output.dimensions() != dimensions
        || output.pixel_format() != PixelFormat::Rgba8
        || output.metal_pixel_format() != MTLPixelFormat::RGBA8Unorm
    {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal batch texture output shape does not match requested RGBA8 tiles",
        });
    }
    if output.tile_capacity() < tile_count {
        return Err(BufferError::OutputTooSmall {
            required: out_tile_len
                .checked_mul(tile_count)
                .ok_or(BufferError::SizeOverflow {
                    what: "JPEG Metal batch texture output bytes",
                })?,
            have: out_tile_len.checked_mul(output.tile_capacity()).ok_or(
                BufferError::SizeOverflow {
                    what: "JPEG Metal batch texture output bytes",
                },
            )?,
        }
        .into());
    }

    for index in 0..tile_count {
        let Some(texture) = output.texture(index) else {
            return Err(Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            });
        };
        if texture.width() != u64::from(dimensions.0)
            || texture.height() != u64::from(dimensions.1)
            || texture.pixel_format() != MTLPixelFormat::RGBA8Unorm
        {
            return Err(Error::UnsupportedMetalRequest {
                reason:
                    "JPEG Metal batch texture output texture does not match requested RGBA8 tiles",
            });
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn texture_batch_success_results(
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    tile_count: usize,
) -> Result<Vec<Result<crate::MetalTextureTile, Error>>, Error> {
    let mut results = Vec::with_capacity(tile_count);
    for index in 0..tile_count {
        let texture = output
            .clone_texture(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
        results.push(Ok(crate::MetalTextureTile::new(
            texture,
            dimensions,
            PixelFormat::Rgba8,
        )));
    }
    Ok(results)
}

#[cfg(target_os = "macos")]
fn copy_rgb8_surfaces_to_rgba_textures(
    runtime: &MetalRuntime,
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    tile_count: usize,
    group_indices: &[usize],
    group_results: Vec<Result<Surface, Error>>,
) -> Result<Vec<GroupedTextureResult>, Error> {
    if group_results.len() != group_indices.len() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal grouped texture result count mismatch".to_string(),
        });
    }
    let out_tile_len = dimensions
        .0
        .checked_mul(dimensions.1)
        .and_then(|pixels| {
            pixels.checked_mul(u32::try_from(PixelFormat::Rgba8.bytes_per_pixel()).ok()?)
        })
        .ok_or(BufferError::SizeOverflow {
            what: "JPEG Metal batch texture output bytes",
        })? as usize;
    validate_rgba_texture_batch_output(output, dimensions, tile_count, out_tile_len)?;

    let in_stride = dimensions
        .0
        .checked_mul(
            u32::try_from(PixelFormat::Rgb8.bytes_per_pixel()).map_err(|_| {
                BufferError::SizeOverflow {
                    what: "JPEG Metal RGB texture copy input stride",
                }
            })?,
        )
        .ok_or(BufferError::SizeOverflow {
            what: "JPEG Metal RGB texture copy input stride",
        })?;
    let params = JpegRgb8ToRgbaTextureParams {
        width: dimensions.0,
        height: dimensions.1,
        in_stride,
        alpha: u32::from(u8::MAX),
    };
    let mut copies = Vec::<(usize, Buffer, usize)>::new();
    let mut mapped_results = Vec::with_capacity(group_indices.len());
    for (original_index, result) in group_indices.iter().copied().zip(group_results) {
        match result {
            Ok(surface) => {
                if surface.dimensions != dimensions || surface.fmt != PixelFormat::Rgb8 {
                    return Err(Error::MetalKernel {
                        message: "JPEG Metal texture copy source shape mismatch".to_string(),
                    });
                }
                let (source, source_offset) =
                    surface.metal_buffer().ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal texture copy source was not Metal-backed".to_string(),
                    })?;
                let texture =
                    output
                        .clone_texture(original_index)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "JPEG Metal batch texture output slot was missing".to_string(),
                        })?;
                copies.push((original_index, source.clone(), source_offset));
                mapped_results.push((
                    original_index,
                    Ok(crate::MetalTextureTile::new(
                        texture,
                        dimensions,
                        PixelFormat::Rgba8,
                    )),
                ));
            }
            Err(error) => mapped_results.push((original_index, Err(error))),
        }
    }

    if !copies.is_empty() {
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.rgb8_to_rgba_texture_pipeline);
        for (original_index, source, source_offset) in copies {
            let texture = output
                .texture(original_index)
                .ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Metal batch texture output slot was missing".to_string(),
                })?;
            encoder.set_buffer(
                0,
                Some(&source),
                u64::try_from(source_offset).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal texture copy source offset exceeds u64".to_string(),
                })?,
            );
            encoder.set_bytes(
                1,
                size_of::<JpegRgb8ToRgbaTextureParams>() as u64,
                (&raw const params).cast(),
            );
            encoder.set_texture(0, Some(texture));
            dispatch_2d_pipeline(encoder, &runtime.rgb8_to_rgba_texture_pipeline, dimensions);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    Ok(mapped_results)
}

#[cfg(target_os = "macos")]
fn dispatch_rgba_texture_pack(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    planes: (&Buffer, &Buffer, &Buffer),
    output: &crate::MetalBatchTextureOutput,
    params: JpegTexturePackBatchParams,
    tile_count: usize,
    dispatch_dims: (u32, u32),
) -> Result<(), Error> {
    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(pipeline);
    pack_encoder.set_buffer(0, Some(planes.0), 0);
    pack_encoder.set_buffer(1, Some(planes.1), 0);
    pack_encoder.set_buffer(2, Some(planes.2), 0);
    for index in 0..tile_count {
        let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let mut params = params;
        params.tile_index = checked_u32(index, "texture batch tile index")?;
        pack_encoder.set_texture(0, Some(texture));
        pack_encoder.set_bytes(
            3,
            size_of::<JpegTexturePackBatchParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, pipeline, dispatch_dims);
    }
    pack_encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_windowed_rgba_texture_pack(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    planes: (&Buffer, &Buffer, &Buffer),
    output: &crate::MetalBatchTextureOutput,
    params: JpegWindowedTexturePackBatchParams,
    tile_count: usize,
    dispatch_dims: (u32, u32),
) -> Result<(), Error> {
    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(pipeline);
    pack_encoder.set_buffer(0, Some(planes.0), 0);
    pack_encoder.set_buffer(1, Some(planes.1), 0);
    pack_encoder.set_buffer(2, Some(planes.2), 0);
    for index in 0..tile_count {
        let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let mut params = params;
        params.tile_index = checked_u32(index, "windowed texture batch tile index")?;
        pack_encoder.set_texture(0, Some(texture));
        pack_encoder.set_bytes(
            3,
            size_of::<JpegWindowedTexturePackBatchParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, pipeline, dispatch_dims);
    }
    pack_encoder.end_encoding();
    Ok(())
}

/// Encode the split coeff-decode + IDCT-deposit passes shared by the surfaces
/// and texture drivers' `SplitCoeffIdct` debug mode.
#[cfg(all(target_os = "macos", test))]
#[allow(clippy::too_many_arguments)]
fn encode_split_coeff_idct_passes(
    command_buffer: &CommandBufferRef,
    pipelines: (&ComputePipelineState, &ComputePipelineState),
    params: &JpegFast420BatchParams,
    quants: [&[u16; 64]; 3],
    dc_tables: &[PreparedHuffmanHost; 3],
    ac_tables: &[PreparedHuffmanHost; 3],
    entropy: (&Buffer, &Buffer, &Buffer, &Buffer),
    status_buffer: &Buffer,
    planes: [&Buffer; 3],
    scratch: (&Buffer, &Buffer),
    total_decode_threads: u32,
    idct_grid: (u32, u32, u32),
) {
    let (coeffs_pipeline, idct_pipeline) = pipelines;
    let (entropy_payload, entropy_offsets, entropy_lens, entropy_checkpoints) = entropy;
    let (coeff_blocks, dc_only_flags) = scratch;

    let coeff_encoder = command_buffer.new_compute_command_encoder();
    coeff_encoder.set_compute_pipeline_state(coeffs_pipeline);
    coeff_encoder.set_buffer(0, Some(entropy_payload), 0);
    coeff_encoder.set_buffer(1, Some(coeff_blocks), 0);
    coeff_encoder.set_buffer(2, Some(dc_only_flags), 0);
    coeff_encoder.set_bytes(
        4,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const *params).cast(),
    );
    coeff_encoder.set_bytes(5, size_of::<[u16; 64]>() as u64, quants[0].as_ptr().cast());
    coeff_encoder.set_bytes(6, size_of::<[u16; 64]>() as u64, quants[1].as_ptr().cast());
    coeff_encoder.set_bytes(7, size_of::<[u16; 64]>() as u64, quants[2].as_ptr().cast());
    coeff_encoder.set_bytes(
        8,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    coeff_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    coeff_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    coeff_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    coeff_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    coeff_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    coeff_encoder.set_buffer(14, Some(entropy_offsets), 0);
    coeff_encoder.set_buffer(15, Some(entropy_lens), 0);
    coeff_encoder.set_buffer(16, Some(status_buffer), 0);
    coeff_encoder.set_buffer(17, Some(entropy_checkpoints), 0);
    dispatch_1d_pipeline(coeff_encoder, coeffs_pipeline, total_decode_threads);
    coeff_encoder.end_encoding();

    let idct_encoder = command_buffer.new_compute_command_encoder();
    idct_encoder.set_compute_pipeline_state(idct_pipeline);
    idct_encoder.set_buffer(0, Some(coeff_blocks), 0);
    idct_encoder.set_buffer(1, Some(dc_only_flags), 0);
    idct_encoder.set_buffer(2, Some(planes[0]), 0);
    idct_encoder.set_buffer(3, Some(planes[1]), 0);
    idct_encoder.set_buffer(4, Some(planes[2]), 0);
    idct_encoder.set_bytes(
        5,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const *params).cast(),
    );
    dispatch_3d_pipeline(idct_encoder, idct_pipeline, idct_grid);
    idct_encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_full_rgb_batch_to_surfaces<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output::<P>(
        runtime,
        requests,
        packets,
        fast_batch_decode_mode(),
        None,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output::<P>(
        runtime,
        requests,
        packets,
        fast_batch_decode_mode(),
        Some(output),
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    decode_mode: FastBatchDecodeMode,
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let timing_enabled =
        decode_mode == FastBatchDecodeMode::Fused && P::full_rgb_batch_timing_enabled();
    let timing_total_start = timing_enabled.then(Instant::now);
    let mut timing = FastBatchTiming::default();

    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut family_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let Some(packet) = P::from_batched(packet) else {
            return Ok(None);
        };
        family_packets.push(packet);
    }

    let Some(first) = family_packets.first().copied() else {
        return Ok(None);
    };
    if (!P::FULL_RGB_BATCH_SUPPORTS_RESTART && first.restart_interval_mcus() != 0)
        || first.entropy_checkpoints().is_empty()
    {
        return Ok(None);
    }

    let Some(groups) = fast_subsampled_full_rgb_batch_groups(&family_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_full_rgb_batch_to_surfaces_with_output::<P>(
            runtime,
            requests,
            &family_packets,
            decode_mode,
            output,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints().len();
    if !family_packets.iter().all(|packet| {
        fast_subsampled_packets_share_full_rgb_batch_shape(first, packet, segment_count)
    }) {
        return Ok(None);
    }

    let tile_count = family_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch decode thread count overflowed".to_string(),
            })?,
        "batch decode thread count",
    )?;

    let width = first.dimensions().0;
    let height = first.dimensions().1;
    let chroma_width = width.div_ceil(2);
    let chroma_height = P::chroma_height(height);
    let y_len = width as usize * height as usize;
    let chroma_len = chroma_width as usize * chroma_height as usize;
    let out_stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    #[cfg_attr(not(test), allow(unused_variables))]
    let total_blocks = match P::FULL_RGB_BATCH_BLOCKS_PER_MCU {
        Some(blocks_per_mcu) => {
            let total_mcus = first.mcus_per_row() as usize * first.mcu_rows() as usize;
            let blocks_per_tile =
                total_mcus
                    .checked_mul(blocks_per_mcu)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} batch block count overflowed",
                            P::FAMILY_NAME
                        ),
                    })?;
            let total_blocks =
                blocks_per_tile
                    .checked_mul(tile_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} batch total block count overflowed",
                            P::FAMILY_NAME
                        ),
                    })?;
            let _total_blocks_u32 = checked_u32(
                total_blocks,
                &format!("{} batch block count", P::FAMILY_NAME),
            )?;
            Some(total_blocks)
        }
        None => None,
    };

    let params = JpegFast420BatchParams {
        width,
        height,
        chroma_width,
        chroma_height,
        mcus_per_row: first.mcus_per_row(),
        mcu_rows: first.mcu_rows(),
        segment_count: segment_count_u32,
        tile_count: tile_count_u32,
        out_stride: checked_u32(out_stride, "batch output stride")?,
        alpha: u32::from(u8::MAX),
    };
    if timing_enabled {
        timing.accepted = timing_total_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let timing_entropy_start = timing_enabled.then(Instant::now);
    let total_entropy_len = family_packets
        .iter()
        .map(|packet| packet.entropy_bytes().len())
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch entropy length overflowed".to_string(),
        })?;
    if total_entropy_len == 0 {
        return Ok(None);
    }

    let mut entropy_bytes = Vec::with_capacity(total_entropy_len);
    let mut entropy_offsets = Vec::with_capacity(tile_count);
    let mut entropy_lens = Vec::with_capacity(tile_count);
    let mut entropy_checkpoints = Vec::with_capacity(tile_count * segment_count);
    for packet in &family_packets {
        entropy_offsets.push(checked_u32(entropy_bytes.len(), "batch entropy offset")?);
        entropy_lens.push(checked_u32(
            packet.entropy_bytes().len(),
            "batch entropy length",
        )?);
        entropy_bytes.extend_from_slice(packet.entropy_bytes());
        entropy_checkpoints.extend(packet.entropy_checkpoints().iter().copied());
    }
    if timing_enabled {
        timing.entropy_concat = timing_entropy_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let timing_buffer_start = timing_enabled.then(Instant::now);
    let mut batch_scratch = runtime.batch_scratch()?;
    let y_plane =
        batch_scratch.private_buffer(&runtime.device, P::FULL_BATCH_KEYS.y, y_len * tile_count);
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::FULL_BATCH_KEYS.cb,
        chroma_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::FULL_BATCH_KEYS.cr,
        chroma_len * tile_count,
    );
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        first.dimensions(),
        tile_count,
        out_stride,
        out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let checkpoint_hosts = entropy_checkpoint_hosts(&entropy_checkpoints)?;
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::FULL_BATCH_KEYS.status,
        &statuses,
    );
    let entropy_buffer = batch_scratch.shared_buffer_with_bytes(
        &runtime.device,
        P::FULL_BATCH_KEYS.entropy,
        &entropy_bytes,
    );
    let entropy_offsets_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::FULL_BATCH_KEYS.entropy_offsets,
        &entropy_offsets,
    );
    let entropy_lens_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::FULL_BATCH_KEYS.entropy_lens,
        &entropy_lens,
    );
    let entropy_checkpoints_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::FULL_BATCH_KEYS.entropy_checkpoints,
        &checkpoint_hosts,
    );
    if timing_enabled {
        timing.buffer_alloc = timing_buffer_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let dc_tables = [
        PreparedHuffmanHost::from(first.y_dc_table()),
        PreparedHuffmanHost::from(first.cb_dc_table()),
        PreparedHuffmanHost::from(first.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(first.y_ac_table()),
        PreparedHuffmanHost::from(first.cb_ac_table()),
        PreparedHuffmanHost::from(first.cr_ac_table()),
    ];

    let mut command_buffer = runtime.queue.new_command_buffer();
    #[cfg(test)]
    let mut split_scratch: Option<(Buffer, Buffer)> = None;
    match decode_mode {
        FastBatchDecodeMode::Fused => {
            let timing_encode_start = timing_enabled.then(Instant::now);
            let decode_pipeline = P::full_rgb_batch_decode_pipeline(runtime);
            let decoder_encoder = command_buffer.new_compute_command_encoder();
            decoder_encoder.set_compute_pipeline_state(decode_pipeline);
            bind_fast_decode_entropy_inputs::<JpegFast420BatchParams>(
                decoder_encoder,
                &entropy_buffer,
                [&y_plane, &cb_plane, &cr_plane],
                &params,
                [first.y_quant(), first.cb_quant(), first.cr_quant()],
                &dc_tables,
                &ac_tables,
                &entropy_offsets_buffer,
                &entropy_lens_buffer,
                &status_buffer,
            );
            decoder_encoder.set_buffer(17, Some(&entropy_checkpoints_buffer), 0);
            dispatch_1d_pipeline(decoder_encoder, decode_pipeline, total_decode_threads);
            decoder_encoder.end_encoding();
            if timing_enabled {
                timing.encode_decode = timing_encode_start
                    .expect("timing start is set when timing is enabled")
                    .elapsed();
                command_buffer.commit();
                let timing_wait_start = Instant::now();
                command_buffer.wait_until_completed();
                timing.wait_decode = timing_wait_start.elapsed();
                command_buffer = runtime.queue.new_command_buffer();
            }
        }
        #[cfg(test)]
        FastBatchDecodeMode::SplitCoeffIdct => {
            let Some((split, total_blocks)) =
                P::split_coeff_idct_pipelines(runtime).zip(total_blocks)
            else {
                return Err(Error::MetalKernel {
                    message: format!(
                        "JPEG Metal {} batch split coeff/IDCT decode mode is unsupported",
                        P::FAMILY_NAME
                    ),
                });
            };
            let coeff_bytes = total_blocks
                .checked_mul(64)
                .and_then(|bytes| bytes.checked_mul(size_of::<i16>()))
                .ok_or_else(|| Error::MetalKernel {
                    message: format!(
                        "JPEG Metal {} batch coefficient scratch overflowed",
                        P::FAMILY_NAME
                    ),
                })?;
            let idct_component_depth =
                tile_count_u32
                    .checked_mul(6)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} batch IDCT dispatch overflowed",
                            P::FAMILY_NAME
                        ),
                    })?;
            let coeff_blocks = runtime
                .device
                .new_buffer(coeff_bytes as u64, MTLResourceOptions::StorageModePrivate);
            let dc_only_flags = runtime
                .device
                .new_buffer(total_blocks as u64, MTLResourceOptions::StorageModePrivate);

            encode_split_coeff_idct_passes(
                command_buffer,
                split,
                &params,
                [first.y_quant(), first.cb_quant(), first.cr_quant()],
                &dc_tables,
                &ac_tables,
                (
                    &entropy_buffer,
                    &entropy_offsets_buffer,
                    &entropy_lens_buffer,
                    &entropy_checkpoints_buffer,
                ),
                &status_buffer,
                [&y_plane, &cb_plane, &cr_plane],
                (&coeff_blocks, &dc_only_flags),
                total_decode_threads,
                (first.mcus_per_row(), first.mcu_rows(), idct_component_depth),
            );

            split_scratch = Some((coeff_blocks, dc_only_flags));
        }
    }

    let timing_pack_encode_start = timing_enabled.then(Instant::now);
    let pack_pipeline = P::pack_full_rgb_batch_pipeline(runtime);
    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    bind_three_plane_pack::<JpegFast420BatchParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        pack_pipeline,
        (
            packed_pair_extent(width),
            P::packed_height_extent(height),
            tile_count_u32,
        ),
    );
    pack_encoder.end_encoding();
    if timing_enabled {
        timing.encode_pack = timing_pack_encode_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    command_buffer.commit();
    if timing_enabled {
        let timing_wait_start = Instant::now();
        command_buffer.wait_until_completed();
        timing.wait_pack = timing_wait_start.elapsed();
        timing.total = timing_total_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
        timing.log(
            P::FULL_RGB_BATCH_TIMING_TAG,
            "fused-stages",
            tile_count,
            first.dimensions(),
            segment_count,
        );
    } else {
        command_buffer.wait_until_completed();
    }
    #[cfg(test)]
    drop(split_scratch);
    drop(batch_scratch);

    if let Some(status) = first_decode_error_status(&status_buffer, total_decode_threads) {
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            let decoder = CpuDecoder::new(request.input.as_ref())?;
            results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
        }
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first.dimensions(),
            PixelFormat::Rgb8,
            index * out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_full_rgb_batch_to_surfaces_with_output<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    decode_mode: FastBatchDecodeMode,
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for packet in family_packets {
            let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * packet.dimensions().1 as usize;
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                packet.dimensions(),
                requests.len(),
                out_stride,
                out_tile_len,
            )?;
        }
    }

    let mut merged_results: Vec<Option<Result<Surface, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| family_packets[index].to_batched())
            .collect::<Vec<_>>();

        let Some(group_results) =
            try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output::<P>(
                runtime,
                &group_requests,
                &group_packets,
                decode_mode,
                None,
            )?
        else {
            return Ok(None);
        };

        if let Some(output) = output {
            let Some(&first_group_index) = group_indices.first() else {
                continue;
            };
            let packet = family_packets[first_group_index];
            let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * packet.dimensions().1 as usize;
            for (original_index, result) in copy_grouped_surfaces_to_output(
                runtime,
                output,
                packet.dimensions(),
                out_tile_len,
                &group_indices,
                group_results,
            )? {
                merged_results[original_index] = Some(result);
            }
        } else {
            if group_results.len() != group_indices.len() {
                return Err(Error::MetalKernel {
                    message: format!(
                        "JPEG Metal grouped {} buffer result count mismatch",
                        P::FAMILY_NAME
                    ),
                });
            }
            for (original_index, result) in group_indices.into_iter().zip(group_results) {
                merged_results[original_index] = Some(result);
            }
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} buffer result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_full_rgba_batch_to_textures<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
    decode_mode: FastBatchDecodeMode,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut family_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let Some(packet) = P::from_batched(packet) else {
            return Ok(None);
        };
        family_packets.push(packet);
    }

    let Some(first) = family_packets.first().copied() else {
        return Ok(None);
    };
    if (!P::FULL_RGB_BATCH_SUPPORTS_RESTART && first.restart_interval_mcus() != 0)
        || first.entropy_checkpoints().is_empty()
    {
        return Ok(None);
    }

    let Some(groups) = fast_subsampled_full_rgb_batch_groups(&family_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures::<P>(
            runtime,
            requests,
            &family_packets,
            output,
            decode_mode,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
    let tile_count_u32 = checked_u32(
        tile_count,
        &format!("{} texture batch tile count", P::FAMILY_NAME),
    )?;
    let segment_count_u32 = checked_u32(
        segment_count,
        &format!("{} texture batch segment count", P::FAMILY_NAME),
    )?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} texture batch decode thread count overflowed",
                    P::FAMILY_NAME
                ),
            })?,
        &format!("{} texture batch decode thread count", P::FAMILY_NAME),
    )?;

    let width = first.dimensions().0;
    let height = first.dimensions().1;
    let chroma_width = width.div_ceil(2);
    let chroma_height = P::chroma_height(height);
    let y_len = width as usize * height as usize;
    let chroma_len = chroma_width as usize * chroma_height as usize;
    let out_stride = width as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    validate_rgba_texture_batch_output(output, first.dimensions(), tile_count, out_tile_len)?;

    let total_mcus = first.mcus_per_row() as usize * first.mcu_rows() as usize;
    let mcu_threads = P::texture_mcu_dispatch_threads(total_mcus)?;
    #[cfg(test)]
    let total_blocks = match P::FULL_RGB_BATCH_BLOCKS_PER_MCU {
        Some(blocks_per_mcu) => {
            let blocks_per_tile =
                total_mcus
                    .checked_mul(blocks_per_mcu)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} texture batch block count overflowed",
                            P::FAMILY_NAME
                        ),
                    })?;
            Some(
                blocks_per_tile
                    .checked_mul(tile_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} texture batch total block count overflowed",
                            P::FAMILY_NAME
                        ),
                    })?,
            )
        }
        None => None,
    };

    let params = JpegFast420BatchParams {
        width,
        height,
        chroma_width,
        chroma_height,
        mcus_per_row: first.mcus_per_row(),
        mcu_rows: first.mcu_rows(),
        segment_count: segment_count_u32,
        tile_count: tile_count_u32,
        out_stride: checked_u32(
            out_stride,
            &format!("{} texture batch output stride", P::FAMILY_NAME),
        )?,
        alpha: u32::from(u8::MAX),
    };

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: P::TEXTURE_KEYS.entropy,
            offsets: P::TEXTURE_KEYS.entropy_offsets,
            lens: P::TEXTURE_KEYS.entropy_lens,
            checkpoints: P::TEXTURE_KEYS.entropy_checkpoints,
        },
        family_packets.iter().map(|packet| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    // Chroma reconstruction needs neighboring samples at MCU boundaries (4:2:0
    // repairs both axes with per-MCU records, 4:2:2 repairs horizontal
    // boundaries per entropy segment). The fused path carries same-segment
    // boundaries in-thread and resolves cross-segment boundaries from compact
    // shared records before returning the caller-owned texture.
    if decode_mode == FastBatchDecodeMode::Fused {
        let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
        let status_buffer = batch_scratch.shared_buffer_with_slice(
            &runtime.device,
            P::TEXTURE_KEYS.status,
            &statuses,
        );
        let total_repair_records =
            P::texture_repair_record_count(tile_count, total_mcus, total_decode_threads)?;
        let boundary_meta = vec![0u32; total_repair_records * P::TEXTURE_BOUNDARY_META_WORDS];
        let boundary_samples = vec![0u8; total_repair_records * P::TEXTURE_BOUNDARY_SAMPLE_BYTES];
        let boundary_meta_buffer = batch_scratch.shared_buffer_with_slice(
            &runtime.device,
            P::TEXTURE_BOUNDARY_META_KEY,
            &boundary_meta,
        );
        let boundary_samples_buffer = batch_scratch.shared_buffer_with_bytes(
            &runtime.device,
            P::TEXTURE_BOUNDARY_SAMPLES_KEY,
            &boundary_samples,
        );
        let vertical_buffers = match &P::TEXTURE_VERTICAL_REPAIR {
            Some(spec) => {
                let vertical_meta = vec![0u32; total_repair_records * spec.meta_words];
                let vertical_samples = vec![0u8; total_repair_records * spec.sample_bytes];
                let vertical_meta_buffer = batch_scratch.shared_buffer_with_slice(
                    &runtime.device,
                    spec.meta_key,
                    &vertical_meta,
                );
                let vertical_samples_buffer = batch_scratch.shared_buffer_with_bytes(
                    &runtime.device,
                    spec.samples_key,
                    &vertical_samples,
                );
                Some((vertical_meta_buffer, vertical_samples_buffer))
            }
            None => None,
        };
        let dc_tables = [
            PreparedHuffmanHost::from(first.y_dc_table()),
            PreparedHuffmanHost::from(first.cb_dc_table()),
            PreparedHuffmanHost::from(first.cr_dc_table()),
        ];
        let ac_tables = [
            PreparedHuffmanHost::from(first.y_ac_table()),
            PreparedHuffmanHost::from(first.cb_ac_table()),
            PreparedHuffmanHost::from(first.cr_ac_table()),
        ];

        let tile_index_ctx = format!("{} texture batch tile index", P::FAMILY_NAME);
        let texture_decode_pipeline = P::rgba_texture_batch_decode_pipeline(runtime);
        let command_buffer = runtime.queue.new_command_buffer();
        for index in 0..tile_count {
            let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
            let decode_params = JpegFast420TextureBatchParams {
                width,
                height,
                chroma_width,
                chroma_height,
                mcus_per_row: first.mcus_per_row(),
                mcu_rows: first.mcu_rows(),
                segment_count: segment_count_u32,
                tile_index: checked_u32(index, &tile_index_ctx)?,
                alpha: u32::from(u8::MAX),
            };
            let decoder_encoder = command_buffer.new_compute_command_encoder();
            decoder_encoder.set_compute_pipeline_state(texture_decode_pipeline);
            decoder_encoder.set_buffer(0, Some(&entropy_buffers.payload), 0);
            decoder_encoder.set_bytes(
                4,
                size_of::<JpegFast420TextureBatchParams>() as u64,
                (&raw const decode_params).cast(),
            );
            decoder_encoder.set_bytes(
                5,
                size_of::<[u16; 64]>() as u64,
                first.y_quant().as_ptr().cast(),
            );
            decoder_encoder.set_bytes(
                6,
                size_of::<[u16; 64]>() as u64,
                first.cb_quant().as_ptr().cast(),
            );
            decoder_encoder.set_bytes(
                7,
                size_of::<[u16; 64]>() as u64,
                first.cr_quant().as_ptr().cast(),
            );
            decoder_encoder.set_bytes(
                8,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const dc_tables[0]).cast(),
            );
            decoder_encoder.set_bytes(
                9,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const ac_tables[0]).cast(),
            );
            decoder_encoder.set_bytes(
                10,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const dc_tables[1]).cast(),
            );
            decoder_encoder.set_bytes(
                11,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const ac_tables[1]).cast(),
            );
            decoder_encoder.set_bytes(
                12,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const dc_tables[2]).cast(),
            );
            decoder_encoder.set_bytes(
                13,
                size_of::<PreparedHuffmanHost>() as u64,
                (&raw const ac_tables[2]).cast(),
            );
            decoder_encoder.set_buffer(14, Some(&entropy_buffers.offsets), 0);
            decoder_encoder.set_buffer(15, Some(&entropy_buffers.lens), 0);
            decoder_encoder.set_buffer(16, Some(&status_buffer), 0);
            decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
            decoder_encoder.set_buffer(18, Some(&boundary_meta_buffer), 0);
            decoder_encoder.set_buffer(19, Some(&boundary_samples_buffer), 0);
            if let Some((vertical_meta_buffer, vertical_samples_buffer)) = &vertical_buffers {
                decoder_encoder.set_buffer(20, Some(vertical_meta_buffer), 0);
                decoder_encoder.set_buffer(21, Some(vertical_samples_buffer), 0);
            }
            decoder_encoder.set_texture(0, Some(texture));
            dispatch_1d_pipeline(decoder_encoder, texture_decode_pipeline, segment_count_u32);
            decoder_encoder.end_encoding();
        }
        if let Some(repair_threads) =
            P::horizontal_repair_threads(first, segment_count_u32, mcu_threads)
        {
            let boundary_pipeline = P::rgba_texture_boundary_pipeline(runtime);
            for index in 0..tile_count {
                let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Metal batch texture output slot was missing".to_string(),
                })?;
                let decode_params = JpegFast420TextureBatchParams {
                    width,
                    height,
                    chroma_width,
                    chroma_height,
                    mcus_per_row: first.mcus_per_row(),
                    mcu_rows: first.mcu_rows(),
                    segment_count: segment_count_u32,
                    tile_index: checked_u32(index, &tile_index_ctx)?,
                    alpha: u32::from(u8::MAX),
                };
                let boundary_encoder = command_buffer.new_compute_command_encoder();
                boundary_encoder.set_compute_pipeline_state(boundary_pipeline);
                boundary_encoder.set_buffer(0, Some(&boundary_meta_buffer), 0);
                boundary_encoder.set_buffer(1, Some(&boundary_samples_buffer), 0);
                boundary_encoder.set_bytes(
                    2,
                    size_of::<JpegFast420TextureBatchParams>() as u64,
                    (&raw const decode_params).cast(),
                );
                boundary_encoder.set_texture(0, Some(texture));
                dispatch_1d_pipeline(boundary_encoder, boundary_pipeline, repair_threads);
                boundary_encoder.end_encoding();
            }
        }
        P::encode_extra_texture_repair_passes(
            runtime,
            &FastTextureRepairCtx {
                command_buffer,
                output,
                boundary_meta_buffer: &boundary_meta_buffer,
                vertical_buffers: vertical_buffers.as_ref(),
                decode_params: JpegFast420TextureBatchParams {
                    width,
                    height,
                    chroma_width,
                    chroma_height,
                    mcus_per_row: first.mcus_per_row(),
                    mcu_rows: first.mcu_rows(),
                    segment_count: segment_count_u32,
                    tile_index: 0,
                    alpha: u32::from(u8::MAX),
                },
                tile_count,
                mcu_threads,
                tile_index_ctx: &tile_index_ctx,
            },
        )?;

        command_buffer.commit();
        command_buffer.wait_until_completed();
        drop(batch_scratch);

        if let Some(results) =
            texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
        {
            return Ok(Some(results));
        }

        return Ok(Some(texture_batch_success_results(
            output,
            first.dimensions(),
            requests.len(),
        )?));
    }

    let y_plane =
        batch_scratch.private_buffer(&runtime.device, P::TEXTURE_KEYS.y, y_len * tile_count);
    let cb_plane =
        batch_scratch.private_buffer(&runtime.device, P::TEXTURE_KEYS.cb, chroma_len * tile_count);
    let cr_plane =
        batch_scratch.private_buffer(&runtime.device, P::TEXTURE_KEYS.cr, chroma_len * tile_count);
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer =
        batch_scratch.shared_buffer_with_slice(&runtime.device, P::TEXTURE_KEYS.status, &statuses);
    let dc_tables = [
        PreparedHuffmanHost::from(first.y_dc_table()),
        PreparedHuffmanHost::from(first.cb_dc_table()),
        PreparedHuffmanHost::from(first.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(first.y_ac_table()),
        PreparedHuffmanHost::from(first.cb_ac_table()),
        PreparedHuffmanHost::from(first.cr_ac_table()),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    match decode_mode {
        FastBatchDecodeMode::Fused => {
            let decode_pipeline = P::full_rgb_batch_decode_pipeline(runtime);
            let decoder_encoder = command_buffer.new_compute_command_encoder();
            decoder_encoder.set_compute_pipeline_state(decode_pipeline);
            bind_fast_decode_entropy_inputs::<JpegFast420BatchParams>(
                decoder_encoder,
                &entropy_buffers.payload,
                [&y_plane, &cb_plane, &cr_plane],
                &params,
                [first.y_quant(), first.cb_quant(), first.cr_quant()],
                &dc_tables,
                &ac_tables,
                &entropy_buffers.offsets,
                &entropy_buffers.lens,
                &status_buffer,
            );
            decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
            dispatch_1d_pipeline(decoder_encoder, decode_pipeline, total_decode_threads);
            decoder_encoder.end_encoding();
        }
        #[cfg(test)]
        FastBatchDecodeMode::SplitCoeffIdct => {
            let Some((split, total_blocks)) =
                P::split_coeff_idct_pipelines(runtime).zip(total_blocks)
            else {
                return Err(Error::MetalKernel {
                    message: format!(
                        "JPEG Metal {} texture batch split coeff/IDCT decode mode is unsupported",
                        P::FAMILY_NAME
                    ),
                });
            };
            let coeff_bytes = total_blocks
                .checked_mul(64)
                .and_then(|bytes| bytes.checked_mul(size_of::<i16>()))
                .ok_or_else(|| Error::MetalKernel {
                    message: format!(
                        "JPEG Metal {} texture batch coefficient scratch overflowed",
                        P::FAMILY_NAME
                    ),
                })?;
            let idct_component_depth =
                tile_count_u32
                    .checked_mul(6)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} texture batch IDCT dispatch overflowed",
                            P::FAMILY_NAME
                        ),
                    })?;
            let coeff_blocks = batch_scratch.private_buffer(
                &runtime.device,
                P::SPLIT_TEXTURE_SCRATCH_KEYS.0,
                coeff_bytes,
            );
            let dc_only_flags = batch_scratch.private_buffer(
                &runtime.device,
                P::SPLIT_TEXTURE_SCRATCH_KEYS.1,
                total_blocks,
            );

            encode_split_coeff_idct_passes(
                command_buffer,
                split,
                &params,
                [first.y_quant(), first.cb_quant(), first.cr_quant()],
                &dc_tables,
                &ac_tables,
                (
                    &entropy_buffers.payload,
                    &entropy_buffers.offsets,
                    &entropy_buffers.lens,
                    &entropy_buffers.checkpoints,
                ),
                &status_buffer,
                [&y_plane, &cb_plane, &cr_plane],
                (&coeff_blocks, &dc_only_flags),
                total_decode_threads,
                (first.mcus_per_row(), first.mcu_rows(), idct_component_depth),
            );
        }
    }

    let pack_params = JpegTexturePackBatchParams {
        width,
        height,
        chroma_width,
        chroma_height,
        tile_index: 0,
        alpha: u32::from(u8::MAX),
        mode: MODE_YCBCR,
    };
    dispatch_rgba_texture_pack(
        command_buffer,
        P::pack_rgba_texture_pipeline(runtime),
        (&y_plane, &cb_plane, &cr_plane),
        output,
        pack_params,
        tile_count,
        (packed_pair_extent(width), P::packed_height_extent(height)),
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        first.dimensions(),
        requests.len(),
    )?))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: &crate::MetalBatchTextureOutput,
    decode_mode: FastBatchDecodeMode,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for packet in family_packets {
        let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let out_tile_len = out_stride * packet.dimensions().1 as usize;
        validate_rgba_texture_batch_output(
            output,
            packet.dimensions(),
            requests.len(),
            out_tile_len,
        )?;
    }

    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_output = output.clone_slots(&group_indices)?;
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| family_packets[index].to_batched())
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast_subsampled_full_rgba_batch_to_textures::<P>(
            runtime,
            &group_requests,
            &group_packets,
            &group_output,
            decode_mode,
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: format!(
                    "JPEG Metal grouped {} texture result count mismatch",
                    P::FAMILY_NAME
                ),
            });
        }
        for (original_index, result) in group_indices.into_iter().zip(group_results) {
            merged_results[original_index] = Some(result);
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} texture result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn fast420_batch_timing_enabled() -> bool {
    fast420_batch_timing_value_enabled(std::env::var_os(FAST420_BATCH_TIMING_ENV).as_deref())
}

#[cfg(target_os = "macos")]
fn fast420_batch_timing_value_enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| value == OsStr::new("1"))
}

#[cfg(target_os = "macos")]
fn fast444_packets_share_region_scaled_batch_shape(
    first: &JpegFast444PacketV1,
    packet: &JpegFast444PacketV1,
    segment_count: usize,
) -> bool {
    packet.restart_interval_mcus == 0
        && packet.dimensions == first.dimensions
        && packet.mcus_per_row == first.mcus_per_row
        && packet.mcu_rows == first.mcu_rows
        && packet.entropy_checkpoints.len() == segment_count
        && packet.y_quant == first.y_quant
        && packet.cb_quant == first.cb_quant
        && packet.cr_quant == first.cr_quant
        && packet.y_dc_table == first.y_dc_table
        && packet.y_ac_table == first.y_ac_table
        && packet.cb_dc_table == first.cb_dc_table
        && packet.cb_ac_table == first.cb_ac_table
        && packet.cr_dc_table == first.cr_dc_table
        && packet.cr_ac_table == first.cr_ac_table
}

#[cfg(target_os = "macos")]
fn fast444_full_rgb_batch_groups(
    packets: &[(&JpegFast444PacketV1, PlaneMode)],
) -> Option<Vec<Vec<usize>>> {
    let mut groups = Vec::<Vec<usize>>::new();
    'packet: for (index, (packet, mode)) in packets.iter().copied().enumerate() {
        if packet.restart_interval_mcus != 0
            || packet.entropy_bytes.is_empty()
            || packet.entropy_checkpoints.is_empty()
        {
            return None;
        }

        for group in &mut groups {
            let (first, first_mode) = packets[group[0]];
            if mode == first_mode
                && fast444_packets_share_region_scaled_batch_shape(
                    first,
                    packet,
                    first.entropy_checkpoints.len(),
                )
            {
                group.push(index);
                continue 'packet;
            }
        }
        groups.push(vec![index]);
    }
    Some(groups)
}

#[cfg(target_os = "macos")]
struct Fast444RegionScaledGroup {
    indices: Vec<usize>,
    mode: PlaneMode,
    scale: signinum_core::Downscale,
    params: JpegFast444ScaledParams,
}

#[cfg(target_os = "macos")]
fn fast444_region_scaled_batch_groups(
    requests: &[batch::QueuedRequest],
    packets: &[(&JpegFast444PacketV1, PlaneMode)],
) -> Option<Vec<Vec<usize>>> {
    let mut groups = Vec::<Fast444RegionScaledGroup>::new();
    'packet: for (index, (request, (packet, mode))) in
        requests.iter().zip(packets.iter().copied()).enumerate()
    {
        if packet.restart_interval_mcus != 0
            || packet.entropy_bytes.is_empty()
            || packet.entropy_checkpoints.is_empty()
        {
            return None;
        }
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return None;
        };
        let scaled = roi.scaled_covering(scale);
        let scaled_roi = signinum_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        let params = fast444_scaled_region_params(packet, scale, scaled_roi)?;

        for group in &mut groups {
            let (first, _) = packets[group.indices[0]];
            if mode == group.mode
                && scale == group.scale
                && params == group.params
                && fast444_packets_share_region_scaled_batch_shape(
                    first,
                    packet,
                    first.entropy_checkpoints.len(),
                )
            {
                group.indices.push(index);
                continue 'packet;
            }
        }
        groups.push(Fast444RegionScaledGroup {
            indices: vec![index],
            mode,
            scale,
            params,
        });
    }
    Some(groups.into_iter().map(|group| group.indices).collect())
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast444_full_rgb_batch_to_surfaces_with_output(runtime, requests, packets, None)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast444_full_rgb_batch_to_surfaces_with_output(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast444_full_rgb_batch_groups(&fast444_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast444_full_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &fast444_packets,
            output,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints.len();
    if !fast444_packets.iter().all(|(packet, mode)| {
        *mode == first_mode
            && fast444_packets_share_region_scaled_batch_shape(first, packet, segment_count)
    }) {
        return Ok(None);
    }

    let tile_count = fast444_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "fast444 batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "fast444 batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast444 batch decode thread count overflowed".to_string(),
            })?,
        "fast444 batch decode thread count",
    )?;

    let width = first.dimensions.0;
    let height = first.dimensions.1;
    let out_stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    let plane_len = width as usize * height as usize;
    let decode_params = JpegFastRegionScaledBatchParams {
        scaled_width: width,
        scaled_height: height,
        chroma_width: width,
        chroma_height: height,
        mcus_per_row: first.mcus_per_row,
        mcu_rows: first.mcu_rows,
        segment_count: segment_count_u32,
        tile_count: tile_count_u32,
        scale_shift: 0,
        origin_x: 0,
        origin_y: 0,
    };
    let pack_params = JpegWindowedPackBatchParams {
        src_width: width,
        src_height: height,
        chroma_width: width,
        chroma_height: height,
        src_x: 0,
        src_y: 0,
        width,
        height,
        tile_count: tile_count_u32,
        out_stride: checked_u32(out_stride, "fast444 batch output stride")?,
        alpha: u32::from(u8::MAX),
        mode: plane_mode_to_u32(first_mode),
        out_format: OUT_RGB,
    };

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: "fast444_full_entropy",
            offsets: "fast444_full_entropy_offsets",
            lens: "fast444_full_entropy_lens",
            checkpoints: "fast444_full_entropy_checkpoints",
        },
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane =
        batch_scratch.private_buffer(&runtime.device, "fast444_full_y", plane_len * tile_count);
    let cb_plane =
        batch_scratch.private_buffer(&runtime.device, "fast444_full_cb", plane_len * tile_count);
    let cr_plane =
        batch_scratch.private_buffer(&runtime.device, "fast444_full_cr", plane_len * tile_count);
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        first.dimensions,
        tile_count,
        out_stride,
        out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer =
        batch_scratch.shared_buffer_with_slice(&runtime.device, "fast444_full_status", &statuses);
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder
        .set_compute_pipeline_state(&runtime.fast444_scaled_region_batch_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &decode_params,
        [&first.y_quant, &first.cb_quant, &first.cr_quant],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_444_rgb_batch_pipeline);
    bind_three_plane_pack::<JpegWindowedPackBatchParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        &runtime.pack_444_rgb_batch_pipeline,
        (width, height, tile_count_u32),
    );
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(batch_scratch);

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first.dimensions,
            PixelFormat::Rgb8,
            index * out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast444_full_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for (packet, _) in fast444_packets {
            let out_stride = packet.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * packet.dimensions.1 as usize;
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                packet.dimensions,
                requests.len(),
                out_stride,
                out_tile_len,
            )?;
        }
    }

    let mut merged_results: Vec<Option<Result<Surface, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| {
                let (packet, mode) = fast444_packets[index];
                BatchedFastPacket::Fast444(packet, mode)
            })
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast444_full_rgb_batch_to_surfaces_with_output(
            runtime,
            &group_requests,
            &group_packets,
            None,
        )?
        else {
            return Ok(None);
        };

        if let Some(output) = output {
            let Some(&first_group_index) = group_indices.first() else {
                continue;
            };
            let (packet, _) = fast444_packets[first_group_index];
            let out_stride = packet.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * packet.dimensions.1 as usize;
            for (original_index, result) in copy_grouped_surfaces_to_output(
                runtime,
                output,
                packet.dimensions,
                out_tile_len,
                &group_indices,
                group_results,
            )? {
                merged_results[original_index] = Some(result);
            }
        } else {
            if group_results.len() != group_indices.len() {
                return Err(Error::MetalKernel {
                    message: "JPEG Metal grouped fast444 buffer result count mismatch".to_string(),
                });
            }
            for (original_index, result) in group_indices.into_iter().zip(group_results) {
                merged_results[original_index] = Some(result);
            }
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped fast444 buffer result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast444_full_rgb_batch_groups(&fast444_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast444_full_rgba_batch_to_textures(
            runtime,
            requests,
            &fast444_packets,
            output,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints.len();
    let tile_count = fast444_packets.len();
    let width = first.dimensions.0;
    let height = first.dimensions.1;
    let out_stride = width as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    validate_rgba_texture_batch_output(output, first.dimensions, tile_count, out_tile_len)?;

    let segment_count_u32 = checked_u32(segment_count, "fast444 batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast444 texture batch decode thread count overflowed"
                    .to_string(),
            })?,
        "fast444 texture batch decode thread count",
    )?;

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: "fast444_texture_entropy",
            offsets: "fast444_texture_entropy_offsets",
            lens: "fast444_texture_entropy_lens",
            checkpoints: "fast444_texture_entropy_checkpoints",
        },
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        "fast444_texture_status",
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    for index in 0..tile_count {
        let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let decode_params = JpegFast444TextureBatchParams {
            width,
            height,
            mcus_per_row: first.mcus_per_row,
            mcu_rows: first.mcu_rows,
            segment_count: segment_count_u32,
            tile_index: checked_u32(index, "fast444 texture batch tile index")?,
            alpha: u32::from(u8::MAX),
            mode: plane_mode_to_u32(first_mode),
        };
        let decoder_encoder = command_buffer.new_compute_command_encoder();
        decoder_encoder
            .set_compute_pipeline_state(&runtime.fast444_rgba_texture_batch_decode_pipeline);
        decoder_encoder.set_buffer(0, Some(&entropy_buffers.payload), 0);
        decoder_encoder.set_bytes(
            4,
            size_of::<JpegFast444TextureBatchParams>() as u64,
            (&raw const decode_params).cast(),
        );
        decoder_encoder.set_bytes(
            5,
            size_of::<[u16; 64]>() as u64,
            first.y_quant.as_ptr().cast(),
        );
        decoder_encoder.set_bytes(
            6,
            size_of::<[u16; 64]>() as u64,
            first.cb_quant.as_ptr().cast(),
        );
        decoder_encoder.set_bytes(
            7,
            size_of::<[u16; 64]>() as u64,
            first.cr_quant.as_ptr().cast(),
        );
        decoder_encoder.set_bytes(
            8,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const dc_tables[0]).cast(),
        );
        decoder_encoder.set_bytes(
            9,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const ac_tables[0]).cast(),
        );
        decoder_encoder.set_bytes(
            10,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const dc_tables[1]).cast(),
        );
        decoder_encoder.set_bytes(
            11,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const ac_tables[1]).cast(),
        );
        decoder_encoder.set_bytes(
            12,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const dc_tables[2]).cast(),
        );
        decoder_encoder.set_bytes(
            13,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const ac_tables[2]).cast(),
        );
        decoder_encoder.set_buffer(14, Some(&entropy_buffers.offsets), 0);
        decoder_encoder.set_buffer(15, Some(&entropy_buffers.lens), 0);
        decoder_encoder.set_buffer(16, Some(&status_buffer), 0);
        decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
        decoder_encoder.set_texture(0, Some(texture));
        dispatch_1d_pipeline(
            decoder_encoder,
            &runtime.fast444_rgba_texture_batch_decode_pipeline,
            segment_count_u32,
        );
        decoder_encoder.end_encoding();
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        first.dimensions,
        requests.len(),
    )?))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast444_full_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: &crate::MetalBatchTextureOutput,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for (packet, _) in fast444_packets {
        let out_stride = packet.dimensions.0 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let out_tile_len = out_stride * packet.dimensions.1 as usize;
        validate_rgba_texture_batch_output(
            output,
            packet.dimensions,
            requests.len(),
            out_tile_len,
        )?;
    }

    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_output = output.clone_slots(&group_indices)?;
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| {
                let (packet, mode) = fast444_packets[index];
                BatchedFastPacket::Fast444(packet, mode)
            })
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast444_full_rgba_batch_to_textures(
            runtime,
            &group_requests,
            &group_packets,
            &group_output,
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: "JPEG Metal grouped fast444 texture result count mismatch".to_string(),
            });
        }
        for (original_index, result) in group_indices.into_iter().zip(group_results) {
            merged_results[original_index] = Some(result);
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped fast444 texture result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime, requests, packets, None,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

#[cfg(target_os = "macos")]
fn fast444_region_scaled_rgb_output_shape(
    packet: &JpegFast444PacketV1,
    roi: Rect,
    scale: signinum_core::Downscale,
) -> Option<((u32, u32), usize, usize)> {
    let scaled = roi.scaled_covering(scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled.x,
        y: scaled.y,
        w: scaled.w,
        h: scaled.h,
    };
    let params = fast444_scaled_region_params(packet, scale, scaled_roi)?;
    let out_dims = (params.scaled_width, params.scaled_height);
    let out_stride = out_dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * out_dims.1 as usize;
    Some((out_dims, out_stride, out_tile_len))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_restart_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if !fast444_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus != 0)
    {
        return Ok(None);
    }
    if fast444_packets
        .iter()
        .any(|(packet, _)| packet.entropy_bytes.is_empty() || packet.entropy_checkpoints.is_empty())
    {
        return Ok(None);
    }

    let mut first_shape = None;
    if output.is_some() {
        for (request, (packet, _)) in requests.iter().zip(fast444_packets.iter().copied()) {
            let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
                return Ok(None);
            };
            let Some((out_dims, out_stride, out_tile_len)) =
                fast444_region_scaled_rgb_output_shape(packet, roi, scale)
            else {
                return Ok(None);
            };
            batch_output_buffer_or_new(
                runtime,
                output,
                out_dims,
                requests.len(),
                out_stride,
                out_tile_len,
            )?;
            first_shape.get_or_insert((out_dims, out_tile_len));
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        let batched_packet = BatchedFastPacket::Fast444(packet, mode);
        results.push(decode_region_scaled_packet_surface(
            runtime,
            &decoder,
            request,
            &batched_packet,
        ));
    }

    let Some(output) = output else {
        return Ok(Some(results));
    };
    let Some((out_dims, out_tile_len)) = first_shape else {
        return Ok(Some(results));
    };
    let group_indices = (0..requests.len()).collect::<Vec<_>>();
    let copied = copy_grouped_surfaces_to_output(
        runtime,
        output,
        out_dims,
        out_tile_len,
        &group_indices,
        results,
    )?;
    let mut merged_results: Vec<Option<Result<Surface, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for (index, result) in copied {
        merged_results[index] = Some(result);
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart fast444 region scaled buffer result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
        return Ok(None);
    };
    if fast444_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus != 0)
    {
        return try_decode_fast444_restart_region_scaled_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &fast444_packets,
            output,
        );
    }
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast444_region_scaled_batch_groups(requests, &fast444_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &fast444_packets,
            output,
            groups,
        );
    }

    let first_scaled = first_roi.scaled_covering(first_scale);
    let first_scaled_roi = signinum_jpeg::Rect {
        x: first_scaled.x,
        y: first_scaled.y,
        w: first_scaled.w,
        h: first_scaled.h,
    };
    let Some(first_decode_params) =
        fast444_scaled_region_params(first, first_scale, first_scaled_roi)
    else {
        return Ok(None);
    };

    let segment_count = first.entropy_checkpoints.len();
    let tile_count = fast444_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "region scaled batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal region scaled batch decode thread count overflowed"
                    .to_string(),
            })?,
        "region scaled batch decode thread count",
    )?;

    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || mode != first_mode
            || !fast444_packets_share_region_scaled_batch_shape(first, packet, segment_count)
        {
            return Ok(None);
        }
        let scaled = roi.scaled_covering(scale);
        let scaled_roi = signinum_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        if fast444_scaled_region_params(packet, scale, scaled_roi) != Some(first_decode_params) {
            return Ok(None);
        }
    }

    let out_stride =
        first_decode_params.scaled_width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * first_decode_params.scaled_height as usize;

    let plane_len =
        first_decode_params.scaled_width as usize * first_decode_params.scaled_height as usize;
    let decode_params = JpegFastRegionScaledBatchParams {
        scaled_width: first_decode_params.scaled_width,
        scaled_height: first_decode_params.scaled_height,
        chroma_width: first_decode_params.scaled_width,
        chroma_height: first_decode_params.scaled_height,
        mcus_per_row: first_decode_params.mcus_per_row,
        mcu_rows: first_decode_params.mcu_rows,
        segment_count: segment_count_u32,
        tile_count: tile_count_u32,
        scale_shift: first_decode_params.scale_shift,
        origin_x: first_decode_params.origin_x,
        origin_y: first_decode_params.origin_y,
    };
    let pack_params = JpegWindowedPackBatchParams {
        src_width: first_decode_params.scaled_width,
        src_height: first_decode_params.scaled_height,
        chroma_width: first_decode_params.scaled_width,
        chroma_height: first_decode_params.scaled_height,
        src_x: 0,
        src_y: 0,
        width: first_decode_params.scaled_width,
        height: first_decode_params.scaled_height,
        tile_count: tile_count_u32,
        out_stride: checked_u32(out_stride, "region scaled batch output stride")?,
        alpha: u32::from(u8::MAX),
        mode: plane_mode_to_u32(first_mode),
        out_format: OUT_RGB,
    };

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: "fast444_region_scaled_entropy",
            offsets: "fast444_region_scaled_entropy_offsets",
            lens: "fast444_region_scaled_entropy_lens",
            checkpoints: "fast444_region_scaled_entropy_checkpoints",
        },
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_y",
        plane_len * tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_cb",
        plane_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_cr",
        plane_len * tile_count,
    );
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        (
            first_decode_params.scaled_width,
            first_decode_params.scaled_height,
        ),
        tile_count,
        out_stride,
        out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        "fast444_region_scaled_status",
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder
        .set_compute_pipeline_state(&runtime.fast444_scaled_region_batch_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &decode_params,
        [&first.y_quant, &first.cb_quant, &first.cr_quant],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_444_rgb_batch_pipeline);
    bind_three_plane_pack::<JpegWindowedPackBatchParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        &runtime.pack_444_rgb_batch_pipeline,
        (
            first_decode_params.scaled_width,
            first_decode_params.scaled_height,
            tile_count_u32,
        ),
    );
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(batch_scratch);

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            (
                first_decode_params.scaled_width,
                first_decode_params.scaled_height,
            ),
            PixelFormat::Rgb8,
            index * out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for (request, (packet, _)) in requests.iter().zip(fast444_packets.iter().copied()) {
            let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
                return Ok(None);
            };
            let scaled = roi.scaled_covering(scale);
            let scaled_roi = signinum_jpeg::Rect {
                x: scaled.x,
                y: scaled.y,
                w: scaled.w,
                h: scaled.h,
            };
            let Some(params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
                return Ok(None);
            };
            let out_dims = (params.scaled_width, params.scaled_height);
            let out_stride = out_dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * out_dims.1 as usize;
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                out_dims,
                requests.len(),
                out_stride,
                out_tile_len,
            )?;
        }
    }

    let mut merged_results: Vec<Option<Result<Surface, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| {
                let (packet, mode) = fast444_packets[index];
                BatchedFastPacket::Fast444(packet, mode)
            })
            .collect::<Vec<_>>();

        let Some(group_results) =
            try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
                runtime,
                &group_requests,
                &group_packets,
                None,
            )?
        else {
            return Ok(None);
        };

        if let Some(output) = output {
            let Some(&first_group_index) = group_indices.first() else {
                continue;
            };
            let batch::BatchOp::RegionScaled { roi, scale } = requests[first_group_index].op else {
                return Ok(None);
            };
            let (packet, _) = fast444_packets[first_group_index];
            let scaled = roi.scaled_covering(scale);
            let scaled_roi = signinum_jpeg::Rect {
                x: scaled.x,
                y: scaled.y,
                w: scaled.w,
                h: scaled.h,
            };
            let Some(params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
                return Ok(None);
            };
            let out_dims = (params.scaled_width, params.scaled_height);
            let out_tile_len =
                out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            for (original_index, result) in copy_grouped_surfaces_to_output(
                runtime,
                output,
                out_dims,
                out_tile_len,
                &group_indices,
                group_results,
            )? {
                merged_results[original_index] = Some(result);
            }
        } else {
            if group_results.len() != group_indices.len() {
                return Err(Error::MetalKernel {
                    message:
                        "JPEG Metal grouped fast444 region scaled buffer result count mismatch"
                            .to_string(),
                });
            }
            for (original_index, result) in group_indices.into_iter().zip(group_results) {
                merged_results[original_index] = Some(result);
            }
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped fast444 region scaled buffer result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_restart_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if !fast444_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus != 0)
    {
        return Ok(None);
    }
    if fast444_packets
        .iter()
        .any(|(packet, _)| packet.entropy_bytes.is_empty() || packet.entropy_checkpoints.is_empty())
    {
        return Ok(None);
    }

    let mut first_shape = None;
    for (request, (packet, _)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let Some((out_dims, _, _)) = fast444_region_scaled_rgb_output_shape(packet, roi, scale)
        else {
            return Ok(None);
        };
        let out_tile_len =
            out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, out_dims, requests.len(), out_tile_len)?;
        first_shape.get_or_insert(out_dims);
    }

    let Some(out_dims) = first_shape else {
        return Ok(Some(Vec::new()));
    };
    let mut surfaces = Vec::with_capacity(requests.len());
    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        let batched_packet = BatchedFastPacket::Fast444(packet, mode);
        surfaces.push(decode_region_scaled_packet_surface(
            runtime,
            &decoder,
            request,
            &batched_packet,
        ));
    }

    let group_indices = (0..requests.len()).collect::<Vec<_>>();
    let copied = copy_rgb8_surfaces_to_rgba_textures(
        runtime,
        output,
        out_dims,
        requests.len(),
        &group_indices,
        surfaces,
    )?;
    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for (index, result) in copied {
        merged_results[index] = Some(result);
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart fast444 region scaled texture result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
        return Ok(None);
    };
    if fast444_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus != 0)
    {
        return try_decode_fast444_restart_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &fast444_packets,
            output,
        );
    }
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast444_region_scaled_batch_groups(requests, &fast444_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast444_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &fast444_packets,
            output,
            groups,
        );
    }

    let first_scaled = first_roi.scaled_covering(first_scale);
    let first_scaled_roi = signinum_jpeg::Rect {
        x: first_scaled.x,
        y: first_scaled.y,
        w: first_scaled.w,
        h: first_scaled.h,
    };
    let Some(first_decode_params) =
        fast444_scaled_region_params(first, first_scale, first_scaled_roi)
    else {
        return Ok(None);
    };

    let segment_count = first.entropy_checkpoints.len();
    let tile_count = fast444_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled texture batch tile count")?;
    let segment_count_u32 =
        checked_u32(segment_count, "region scaled texture batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal region scaled texture batch decode thread count overflowed"
                    .to_string(),
            })?,
        "region scaled texture batch decode thread count",
    )?;

    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || mode != first_mode
            || !fast444_packets_share_region_scaled_batch_shape(first, packet, segment_count)
        {
            return Ok(None);
        }
        let scaled = roi.scaled_covering(scale);
        let scaled_roi = signinum_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        if fast444_scaled_region_params(packet, scale, scaled_roi) != Some(first_decode_params) {
            return Ok(None);
        }
    }

    let out_dims = (
        first_decode_params.scaled_width,
        first_decode_params.scaled_height,
    );
    let out_tile_len =
        out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
    validate_rgba_texture_batch_output(output, out_dims, tile_count, out_tile_len)?;

    let plane_len =
        first_decode_params.scaled_width as usize * first_decode_params.scaled_height as usize;
    let decode_params = JpegFastRegionScaledBatchParams {
        scaled_width: first_decode_params.scaled_width,
        scaled_height: first_decode_params.scaled_height,
        chroma_width: first_decode_params.scaled_width,
        chroma_height: first_decode_params.scaled_height,
        mcus_per_row: first_decode_params.mcus_per_row,
        mcu_rows: first_decode_params.mcu_rows,
        segment_count: segment_count_u32,
        tile_count: tile_count_u32,
        scale_shift: first_decode_params.scale_shift,
        origin_x: first_decode_params.origin_x,
        origin_y: first_decode_params.origin_y,
    };

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: "fast444_region_scaled_texture_entropy",
            offsets: "fast444_region_scaled_texture_entropy_offsets",
            lens: "fast444_region_scaled_texture_entropy_lens",
            checkpoints: "fast444_region_scaled_texture_entropy_checkpoints",
        },
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_y",
        plane_len * tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_cb",
        plane_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_cr",
        plane_len * tile_count,
    );
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        "fast444_region_scaled_texture_status",
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder
        .set_compute_pipeline_state(&runtime.fast444_scaled_region_batch_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &decode_params,
        [&first.y_quant, &first.cb_quant, &first.cr_quant],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_params = JpegTexturePackBatchParams {
        width: out_dims.0,
        height: out_dims.1,
        chroma_width: out_dims.0,
        chroma_height: out_dims.1,
        tile_index: 0,
        alpha: u32::from(u8::MAX),
        mode: plane_mode_to_u32(first_mode),
    };
    dispatch_rgba_texture_pack(
        command_buffer,
        &runtime.pack_444_rgba_texture_pipeline,
        (&y_plane, &cb_plane, &cr_plane),
        output,
        pack_params,
        tile_count,
        out_dims,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        out_dims,
        requests.len(),
    )?))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast444_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: &crate::MetalBatchTextureOutput,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for (request, (packet, _)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let scaled = roi.scaled_covering(scale);
        let scaled_roi = signinum_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        let Some(params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
            return Ok(None);
        };
        let out_dims = (params.scaled_width, params.scaled_height);
        let out_tile_len =
            out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, out_dims, requests.len(), out_tile_len)?;
    }

    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_output = output.clone_slots(&group_indices)?;
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| {
                let (packet, mode) = fast444_packets[index];
                BatchedFastPacket::Fast444(packet, mode)
            })
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast444_region_scaled_rgba_batch_to_textures(
            runtime,
            &group_requests,
            &group_packets,
            &group_output,
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: "JPEG Metal grouped fast444 region scaled texture result count mismatch"
                    .to_string(),
            });
        }
        for (original_index, result) in group_indices.into_iter().zip(group_results) {
            merged_results[original_index] = Some(result);
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped fast444 region scaled texture result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime, requests, packets, None,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_restart_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if !family_packets
        .iter()
        .any(|packet| packet.restart_interval_mcus() != 0)
    {
        return Ok(None);
    }
    if family_packets
        .iter()
        .any(|packet| packet.entropy_bytes().is_empty() || packet.entropy_checkpoints().is_empty())
    {
        return Ok(None);
    }

    let mut first_plan = None;
    if output.is_some() {
        for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
            let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
                return Ok(None);
            };
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} restart region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) =
                fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
            else {
                return Ok(None);
            };
            batch_output_buffer_or_new(
                runtime,
                output,
                plan.out_dims,
                requests.len(),
                plan.pack_params.out_stride as usize,
                plan.out_tile_len,
            )?;
            first_plan.get_or_insert(plan);
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        let batched_packet = packet.to_batched();
        results.push(decode_region_scaled_packet_surface(
            runtime,
            &decoder,
            request,
            &batched_packet,
        ));
    }

    let Some(output) = output else {
        return Ok(Some(results));
    };
    let Some(plan) = first_plan else {
        return Ok(Some(results));
    };
    let group_indices = (0..requests.len()).collect::<Vec<_>>();
    let copied = copy_grouped_surfaces_to_output(
        runtime,
        output,
        plan.out_dims,
        plan.out_tile_len,
        &group_indices,
        results,
    )?;
    let mut merged_results: Vec<Option<Result<Surface, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for (index, result) in copied {
        merged_results[index] = Some(result);
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart {} region scaled buffer result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut family_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let Some(packet) = P::from_batched(packet) else {
            return Ok(None);
        };
        family_packets.push(packet);
    }

    let Some(first) = family_packets.first().copied() else {
        return Ok(None);
    };
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
        return Ok(None);
    };
    if family_packets
        .iter()
        .any(|packet| packet.restart_interval_mcus() != 0)
    {
        return try_decode_fast_subsampled_restart_region_scaled_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &family_packets,
            output,
        );
    }
    if first.restart_interval_mcus() != 0 || first.entropy_checkpoints().is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast_subsampled_region_scaled_batch_groups(requests, &family_packets)?
    else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &family_packets,
            output,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "region scaled batch segment count")?;
    let Some(first_plan) = fast_subsampled_region_scaled_batch_plan(
        first,
        first_roi,
        first_scale,
        tile_count_u32,
        segment_count_u32,
    ) else {
        return Ok(None);
    };

    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} region scaled batch decode thread count overflowed",
                    P::FAMILY_NAME
                ),
            })?,
        &format!("{} region scaled batch decode thread count", P::FAMILY_NAME),
    )?;

    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || !fast_subsampled_packets_share_full_rgb_batch_shape(first, packet, segment_count)
            || fast_subsampled_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                tile_count_u32,
                segment_count_u32,
            ) != Some(first_plan)
        {
            return Ok(None);
        }
    }

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: P::REGION_SCALED_KEYS.entropy,
            offsets: P::REGION_SCALED_KEYS.entropy_offsets,
            lens: P::REGION_SCALED_KEYS.entropy_lens,
            checkpoints: P::REGION_SCALED_KEYS.entropy_checkpoints,
        },
        family_packets.iter().map(|packet| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_KEYS.y,
        first_plan.y_len * tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_KEYS.cb,
        first_plan.chroma_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_KEYS.cr,
        first_plan.chroma_len * tile_count,
    );
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        first_plan.out_dims,
        tile_count,
        first_plan.pack_params.out_stride as usize,
        first_plan.out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::REGION_SCALED_KEYS.status,
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(first.y_dc_table()),
        PreparedHuffmanHost::from(first.cb_dc_table()),
        PreparedHuffmanHost::from(first.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(first.y_ac_table()),
        PreparedHuffmanHost::from(first.cb_ac_table()),
        PreparedHuffmanHost::from(first.cr_ac_table()),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(P::scaled_region_batch_decode_pipeline(runtime));
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &first_plan.decode_params,
        [first.y_quant(), first.cb_quant(), first.cr_quant()],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        P::scaled_region_batch_decode_pipeline(runtime),
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(P::pack_windowed_rgb_batch_pipeline(runtime));
    bind_three_plane_pack::<JpegWindowedPackBatchParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &first_plan.pack_params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        P::pack_windowed_rgb_batch_pipeline(runtime),
        (first_plan.out_dims.0, first_plan.out_dims.1, tile_count_u32),
    );
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(batch_scratch);

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first_plan.out_dims,
            PixelFormat::Rgb8,
            index * first_plan.out_tile_len,
        )));
    }
    Ok(Some(results))
}

fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast420PacketV1>(
        runtime, requests, packets, output,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
            let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
                return Ok(None);
            };
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} grouped region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) =
                fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
            else {
                return Ok(None);
            };
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                plan.out_dims,
                requests.len(),
                plan.pack_params.out_stride as usize,
                plan.out_tile_len,
            )?;
        }
    }

    let mut merged_results: Vec<Option<Result<Surface, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| family_packets[index].to_batched())
            .collect::<Vec<_>>();

        let Some(group_results) =
            try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<P>(
                runtime,
                &group_requests,
                &group_packets,
                None,
            )?
        else {
            return Ok(None);
        };

        if let Some(output) = output {
            let Some(&first_group_index) = group_indices.first() else {
                continue;
            };
            let batch::BatchOp::RegionScaled { roi, scale } = requests[first_group_index].op else {
                return Ok(None);
            };
            let packet = family_packets[first_group_index];
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} grouped region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) =
                fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
            else {
                return Ok(None);
            };
            for (original_index, result) in copy_grouped_surfaces_to_output(
                runtime,
                output,
                plan.out_dims,
                plan.out_tile_len,
                &group_indices,
                group_results,
            )? {
                merged_results[original_index] = Some(result);
            }
        } else {
            if group_results.len() != group_indices.len() {
                return Err(Error::MetalKernel {
                    message: format!(
                        "JPEG Metal grouped {} region scaled buffer result count mismatch",
                        P::FAMILY_NAME
                    ),
                });
            }
            for (original_index, result) in group_indices.into_iter().zip(group_results) {
                merged_results[original_index] = Some(result);
            }
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} region scaled buffer result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_restart_region_scaled_rgba_batch_to_textures<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if !family_packets
        .iter()
        .any(|packet| packet.restart_interval_mcus() != 0)
    {
        return Ok(None);
    }
    if family_packets
        .iter()
        .any(|packet| packet.entropy_bytes().is_empty() || packet.entropy_checkpoints().is_empty())
    {
        return Ok(None);
    }

    let mut first_plan = None;
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let segment_count_u32 = checked_u32(
            packet.entropy_checkpoints().len(),
            &format!(
                "{} restart region scaled texture segment count",
                P::FAMILY_NAME
            ),
        )?;
        let Some(plan) =
            fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
        else {
            return Ok(None);
        };
        let out_tile_len = plan.out_dims.0 as usize
            * plan.out_dims.1 as usize
            * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, plan.out_dims, requests.len(), out_tile_len)?;
        first_plan.get_or_insert(plan);
    }

    let Some(plan) = first_plan else {
        return Ok(Some(Vec::new()));
    };
    let mut surfaces = Vec::with_capacity(requests.len());
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        let batched_packet = packet.to_batched();
        surfaces.push(decode_region_scaled_packet_surface(
            runtime,
            &decoder,
            request,
            &batched_packet,
        ));
    }

    let group_indices = (0..requests.len()).collect::<Vec<_>>();
    let copied = copy_rgb8_surfaces_to_rgba_textures(
        runtime,
        output,
        plan.out_dims,
        requests.len(),
        &group_indices,
        surfaces,
    )?;
    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for (index, result) in copied {
        merged_results[index] = Some(result);
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart {} region scaled texture result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut family_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let Some(packet) = P::from_batched(packet) else {
            return Ok(None);
        };
        family_packets.push(packet);
    }

    let Some(first) = family_packets.first().copied() else {
        return Ok(None);
    };
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
        return Ok(None);
    };
    if family_packets
        .iter()
        .any(|packet| packet.restart_interval_mcus() != 0)
    {
        return try_decode_fast_subsampled_restart_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &family_packets,
            output,
        );
    }
    if first.restart_interval_mcus() != 0 || first.entropy_checkpoints().is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast_subsampled_region_scaled_batch_groups(requests, &family_packets)?
    else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &family_packets,
            output,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled texture batch tile count")?;
    let segment_count_u32 =
        checked_u32(segment_count, "region scaled texture batch segment count")?;
    let Some(first_plan) = fast_subsampled_region_scaled_batch_plan(
        first,
        first_roi,
        first_scale,
        tile_count_u32,
        segment_count_u32,
    ) else {
        return Ok(None);
    };

    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} region scaled texture decode thread count overflowed",
                    P::FAMILY_NAME
                ),
            })?,
        &format!(
            "{} region scaled texture decode thread count",
            P::FAMILY_NAME
        ),
    )?;

    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || !fast_subsampled_packets_share_full_rgb_batch_shape(first, packet, segment_count)
            || fast_subsampled_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                tile_count_u32,
                segment_count_u32,
            ) != Some(first_plan)
        {
            return Ok(None);
        }
    }

    let out_tile_len = first_plan.out_dims.0 as usize
        * first_plan.out_dims.1 as usize
        * PixelFormat::Rgba8.bytes_per_pixel();
    validate_rgba_texture_batch_output(output, first_plan.out_dims, tile_count, out_tile_len)?;

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: P::REGION_SCALED_TEXTURE_KEYS.entropy,
            offsets: P::REGION_SCALED_TEXTURE_KEYS.entropy_offsets,
            lens: P::REGION_SCALED_TEXTURE_KEYS.entropy_lens,
            checkpoints: P::REGION_SCALED_TEXTURE_KEYS.entropy_checkpoints,
        },
        family_packets.iter().map(|packet| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.y,
        first_plan.y_len * tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.cb,
        first_plan.chroma_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.cr,
        first_plan.chroma_len * tile_count,
    );
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.status,
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(first.y_dc_table()),
        PreparedHuffmanHost::from(first.cb_dc_table()),
        PreparedHuffmanHost::from(first.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(first.y_ac_table()),
        PreparedHuffmanHost::from(first.cb_ac_table()),
        PreparedHuffmanHost::from(first.cr_ac_table()),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(P::scaled_region_batch_decode_pipeline(runtime));
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &first_plan.decode_params,
        [first.y_quant(), first.cb_quant(), first.cr_quant()],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        P::scaled_region_batch_decode_pipeline(runtime),
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    dispatch_windowed_rgba_texture_pack(
        command_buffer,
        P::pack_windowed_rgba_texture_pipeline(runtime),
        (&y_plane, &cb_plane, &cr_plane),
        output,
        windowed_texture_pack_params(first_plan),
        tile_count,
        first_plan.out_dims,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        first_plan.out_dims,
        requests.len(),
    )?))
}

fn try_decode_fast420_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures::<JpegFast420PacketV1>(
        runtime, requests, packets, output,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_region_scaled_rgba_batch_to_textures<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: &crate::MetalBatchTextureOutput,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let segment_count_u32 = checked_u32(
            packet.entropy_checkpoints().len(),
            &format!(
                "{} grouped region scaled texture batch segment count",
                P::FAMILY_NAME
            ),
        )?;
        let Some(plan) =
            fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
        else {
            return Ok(None);
        };
        let out_tile_len = plan.out_dims.0 as usize
            * plan.out_dims.1 as usize
            * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, plan.out_dims, requests.len(), out_tile_len)?;
    }

    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_output = output.clone_slots(&group_indices)?;
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| family_packets[index].to_batched())
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures::<
            P,
        >(
            runtime, &group_requests, &group_packets, &group_output
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: format!(
                    "JPEG Metal grouped {} region scaled texture result count mismatch",
                    P::FAMILY_NAME
                ),
            });
        }
        for (original_index, result) in group_indices.into_iter().zip(group_results) {
            merged_results[original_index] = Some(result);
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} region scaled texture result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime, requests, packets, None,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast422PacketV1>(
        runtime, requests, packets, output,
    )
}

fn try_decode_fast422_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures::<JpegFast422PacketV1>(
        runtime, requests, packets, output,
    )
}

#[cfg(target_os = "macos")]
fn requests_share_one_input(requests: &[batch::QueuedRequest]) -> bool {
    let Some(first) = requests.first() else {
        return false;
    };
    requests.iter().all(|request| {
        request.input.as_ptr() == first.input.as_ptr() && request.input.len() == first.input.len()
    })
}

#[cfg(target_os = "macos")]
fn requests_share_one_region_scaled_work(requests: &[batch::QueuedRequest]) -> bool {
    let Some(first) = requests.first() else {
        return false;
    };
    requests_share_one_input(requests)
        && requests.iter().all(|request| {
            request.fmt == first.fmt && request.backend == first.backend && request.op == first.op
        })
}

#[cfg(target_os = "macos")]
fn decode_region_scaled_packet_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    request: &batch::QueuedRequest,
    packet: &BatchedFastPacket<'_>,
) -> Result<Surface, Error> {
    let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
        return Err(Error::MetalKernel {
            message: "JPEG Metal expected a region scaled batch request".to_string(),
        });
    };
    let scaled = roi.scaled_covering(scale);
    let scaled_roi = signinum_jpeg::Rect {
        x: scaled.x,
        y: scaled.y,
        w: scaled.w,
        h: scaled.h,
    };
    match packet {
        BatchedFastPacket::Fast420(packet) => try_decode_fast420_scaled_region_to_surface(
            runtime,
            decoder,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
        BatchedFastPacket::Fast422(packet) => try_decode_fast422_scaled_region_to_surface(
            runtime,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
        BatchedFastPacket::Fast444(packet, _) => try_decode_fast444_scaled_region_to_surface(
            runtime,
            decoder,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
    }
    .and_then(|surface| {
        surface.ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal repeated region scaled batch was not packet-decodable".to_string(),
        })
    })
}

#[cfg(target_os = "macos")]
fn try_decode_repeated_region_scaled_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.len() <= REGION_SCALED_BATCH_CHUNK
        || !requests_share_one_input(requests)
        || !requests
            .iter()
            .all(|request| matches!(request.op, batch::BatchOp::RegionScaled { .. }))
    {
        return Ok(None);
    }

    let decoder = CpuDecoder::new(requests[0].input.as_ref())?;
    if requests_share_one_region_scaled_work(requests) {
        let surface =
            decode_region_scaled_packet_surface(runtime, &decoder, &requests[0], &packets[0])?;
        return Ok(Some(
            (0..requests.len())
                .map(|_| Ok(surface.clone()))
                .collect::<Vec<_>>(),
        ));
    }

    let mut results = Vec::with_capacity(requests.len());
    for (request, packet) in requests.iter().zip(packets.iter()) {
        results.push(decode_region_scaled_packet_surface(
            runtime, &decoder, request, packet,
        ));
    }

    Ok(Some(results))
}

#[cfg(target_os = "macos")]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn decode_full_batch_to_surfaces(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime(|runtime| decode_full_batch_to_surfaces_with_runtime(runtime, requests, &packets))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_batch_to_surfaces_with_session(
    requests: &[batch::QueuedRequest],
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_full_batch_to_surfaces_with_runtime(runtime, requests, &packets)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_batch_to_surfaces_with_session_state(
    requests: &[batch::QueuedRequest],
    session: &mut crate::session::SessionState,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    let backend_session = session.backend_session()?;
    with_runtime_for_session(backend_session, |runtime| {
        decode_full_batch_to_surfaces_with_runtime(runtime, requests, &packets)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_rgb8_batch_into_output_with_session(
    requests: &[batch::QueuedRequest],
    output: &crate::MetalBatchOutputBuffer,
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_full_rgb8_batch_into_output_with_runtime(runtime, requests, &packets, output)
    })
}

#[cfg(target_os = "macos")]
fn decode_full_rgb8_batch_into_output_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(results) = try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output::<
        JpegFast420PacketV1,
    >(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output::<
        JpegFast422PacketV1,
    >(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast444_full_rgb_batch_to_surfaces_into_output(
        runtime, requests, packets, output,
    )? {
        return Ok(Some(results));
    }

    Ok(None)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_rgb8_batch_into_textures_with_session(
    requests: &[batch::QueuedRequest],
    output: &crate::MetalBatchTextureOutput,
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_full_rgb8_batch_into_textures_with_runtime(runtime, requests, &packets, output)
    })
}

#[cfg(target_os = "macos")]
fn decode_full_rgb8_batch_into_textures_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if let Some(results) = try_decode_fast_subsampled_full_rgba_batch_to_textures::<
        JpegFast420PacketV1,
    >(runtime, requests, packets, output, fast_batch_decode_mode())?
    {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast_subsampled_full_rgba_batch_to_textures::<
        JpegFast422PacketV1,
    >(runtime, requests, packets, output, fast_batch_decode_mode())?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast444_full_rgba_batch_to_textures(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }

    Ok(None)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_rgb8_batch_into_output_with_session(
    requests: &[batch::QueuedRequest],
    output: &crate::MetalBatchOutputBuffer,
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_region_scaled_rgb8_batch_into_output_with_runtime(
            runtime, requests, &packets, output,
        )
    })
}

#[cfg(target_os = "macos")]
fn decode_region_scaled_rgb8_batch_into_output_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(results) = try_decode_fast444_region_scaled_rgb_batch_to_surfaces_into_output(
        runtime, requests, packets, output,
    )? {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast420_region_scaled_rgb_batch_to_surfaces_into_output(
        runtime, requests, packets, output,
    )? {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast422_region_scaled_rgb_batch_to_surfaces_into_output(
        runtime, requests, packets, output,
    )? {
        return Ok(Some(results));
    }

    Ok(None)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_rgb8_batch_into_textures_with_session(
    requests: &[batch::QueuedRequest],
    output: &crate::MetalBatchTextureOutput,
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_region_scaled_rgb8_batch_into_textures_with_runtime(
            runtime, requests, &packets, output,
        )
    })
}

#[cfg(target_os = "macos")]
fn decode_region_scaled_rgb8_batch_into_textures_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if let Some(results) =
        try_decode_fast444_region_scaled_rgba_batch_to_textures(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast420_region_scaled_rgba_batch_to_textures(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast422_region_scaled_rgba_batch_to_textures(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }

    Ok(None)
}

#[cfg(target_os = "macos")]
fn decode_full_batch_to_surfaces_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(results) = try_decode_fast_subsampled_full_rgb_batch_to_surfaces::<
        JpegFast420PacketV1,
    >(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast_subsampled_full_rgb_batch_to_surfaces::<
        JpegFast422PacketV1,
    >(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast444_full_rgb_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_repeated_region_scaled_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast444_region_scaled_rgb_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast420_region_scaled_rgb_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast422_region_scaled_rgb_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    let has_region_scaled = requests
        .iter()
        .any(|request| matches!(request.op, batch::BatchOp::RegionScaled { .. }));
    let chunk_size = if has_region_scaled {
        REGION_SCALED_BATCH_CHUNK
    } else {
        requests.len().max(1)
    };
    for chunk_start in (0..requests.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(requests.len());
        let command_buffer = runtime.queue.new_command_buffer();
        let mut encoded = Vec::with_capacity(chunk_end - chunk_start);
        let mut device_buffer_cache = BatchDeviceBufferCache::default();
        for index in chunk_start..chunk_end {
            let request = &requests[index];
            let packet = &packets[index];
            let item = match packet {
                BatchedFastPacket::Fast420(packet) => encode_fast_subsampled_op_batch_item(
                    runtime,
                    command_buffer,
                    &mut device_buffer_cache,
                    index,
                    *packet,
                    request.fmt,
                    request.op,
                )?,
                BatchedFastPacket::Fast422(packet) => encode_fast_subsampled_op_batch_item(
                    runtime,
                    command_buffer,
                    &mut device_buffer_cache,
                    index,
                    *packet,
                    request.fmt,
                    request.op,
                )?,
                BatchedFastPacket::Fast444(packet, mode) => match request.op {
                    batch::BatchOp::Full => encode_fast444_batch_item(
                        runtime,
                        command_buffer,
                        index,
                        packet,
                        *mode,
                        request.fmt,
                    )?,
                    batch::BatchOp::Region(roi) => encode_fast444_region_batch_item(
                        runtime,
                        command_buffer,
                        index,
                        packet,
                        *mode,
                        request.fmt,
                        roi,
                    )?,
                    batch::BatchOp::Scaled(scale) => encode_fast444_scaled_batch_item(
                        runtime,
                        command_buffer,
                        index,
                        packet,
                        *mode,
                        request.fmt,
                        scale,
                    )?,
                    batch::BatchOp::RegionScaled { roi, scale } => {
                        encode_fast444_scaled_region_batch_item(
                            runtime,
                            command_buffer,
                            &mut device_buffer_cache,
                            index,
                            packet,
                            *mode,
                            request.fmt,
                            roi,
                            scale,
                        )?
                    }
                },
            };
            encoded.push(item);
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        for item in encoded {
            if let Some(status) =
                first_decode_error_status(&item.status_buffer, item.decode_threads)
            {
                let request = &requests[item.request_index];
                let decoder = CpuDecoder::new(request.input.as_ref())?;
                results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
            } else {
                results.push(Ok(item.surface));
            }
        }
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_to_surface(runtime, packet, fmt, fast422_status_error)
}

#[cfg(target_os = "macos")]
fn decode_fast422_to_rgb_buffer(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
    output_storage: MTLResourceOptions,
) -> Result<Option<FastRgbDecodeBuffer>, Error> {
    decode_fast_subsampled_to_rgb_buffer(runtime, packet, fmt, output_storage, fast422_status_error)
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_to_surface<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    map_status: impl Fn(JpegDecodeStatus) -> Error,
) -> Result<Option<Surface>, Error> {
    let Some(decoded) = decode_fast_subsampled_to_rgb_buffer(
        runtime,
        packet,
        fmt,
        MTLResourceOptions::StorageModeShared,
        map_status,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(Surface::from_metal_buffer(
        decoded.buffer,
        decoded.dimensions,
        fmt,
    )))
}

#[cfg(target_os = "macos")]
fn decode_fast_subsampled_to_rgb_buffer<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    output_storage: MTLResourceOptions,
    map_status: impl Fn(JpegDecodeStatus) -> Error,
) -> Result<Option<FastRgbDecodeBuffer>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_out_format) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let params = fast_subsampled_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, fmt == PixelFormat::Gray8);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        packet.restart_offsets().len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes().as_ptr().cast(),
        packet.entropy_bytes().len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, packet.restart_offsets())?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let out_buffer = (fmt != PixelFormat::Gray8).then(|| {
        runtime.device.new_buffer(
            (params.out_stride as usize * params.height as usize) as u64,
            output_storage,
        )
    });

    let decode_pipeline = P::decode_pipeline(runtime);
    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    if let Some(out_buffer) = out_buffer.as_ref() {
        let Some(pack_pipeline) = P::pack_pipeline_for_format(runtime, fmt) else {
            return Ok(None);
        };
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&cb_plane), 0);
        pack_encoder.set_buffer(2, Some(&cr_plane), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, pack_pipeline, packet.dimensions());
        pack_encoder.end_encoding();
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();
    let command_buffer = command_buffer.to_owned();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(map_status(status));
    }

    Ok(Some(FastRgbDecodeBuffer {
        buffer: out_buffer.unwrap_or(y_plane),
        dimensions: packet.dimensions(),
        status_buffer,
        command_buffer,
    }))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_region_to_surface<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
    map_status: impl Fn(JpegDecodeStatus) -> Error,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let command_buffer = runtime.queue.new_command_buffer();
    let item = encode_fast_subsampled_region_batch_item(
        runtime,
        command_buffer,
        0,
        packet,
        fmt,
        Rect {
            x: roi.x,
            y: roi.y,
            w: roi.w,
            h: roi.h,
        },
    )?;
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&item.status_buffer, item.decode_threads) {
        return Err(map_status(status));
    }

    Ok(Some(item.surface))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_scaled_to_surface<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
    map_status: impl Fn(JpegDecodeStatus) -> Error,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    if fast_subsampled_scaled_params(packet, scale).is_none() {
        return Ok(None);
    }

    let command_buffer = runtime.queue.new_command_buffer();
    let item =
        encode_fast_subsampled_scaled_batch_item(runtime, command_buffer, 0, packet, fmt, scale)?;
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&item.status_buffer, item.decode_threads) {
        return Err(map_status(status));
    }

    Ok(Some(item.surface))
}

#[cfg(target_os = "macos")]
fn fast422_status_error(status: JpegDecodeStatus) -> Error {
    Error::MetalKernel {
        message: format!(
            "unexpected Metal fast422 failure at entropy byte {}",
            status.position
        ),
    }
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_region_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_region_to_surface(runtime, packet, fmt, roi, fast422_status_error)
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_scaled_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_scaled_to_surface(runtime, packet, fmt, scale, fast422_status_error)
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_scaled_region_to_surface(
    runtime: &MetalRuntime,
    packet: Option<&JpegFast422PacketV1>,
    fmt: PixelFormat,
    scaled_roi: signinum_jpeg::Rect,
    scale: signinum_core::Downscale,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_scaled_region_to_surface(
        runtime,
        packet,
        fmt,
        scaled_roi,
        scale,
        fast422_status_error,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_scaled_region_to_surface<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    packet: Option<&P>,
    fmt: PixelFormat,
    scaled_roi: signinum_jpeg::Rect,
    scale: signinum_core::Downscale,
    map_status: impl Fn(JpegDecodeStatus) -> Error,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(full_params) = fast_subsampled_scaled_params(packet, scale) else {
        return Ok(None);
    };
    let source_window = fast_subsampled_full_mcu_scaled_window::<P>(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let Some(mut decode_params) =
        fast_subsampled_scaled_region_params(packet, scale, source_window)
    else {
        return Ok(None);
    };
    let mcu_width = P::MCU_WIDTH >> decode_params.scale_shift;
    let mcu_height = P::MCU_HEIGHT >> decode_params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row(),
        packet.mcu_rows(),
        mcu_width,
        mcu_height,
    );
    let total_mcus = packet.mcus_per_row() * packet.mcu_rows();
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        packet.restart_offsets(),
        packet.restart_interval_mcus(),
        total_mcus,
        first_mcu,
        end_mcu,
    );
    decode_params.restart_start_mcu = restart_start_mcu;
    decode_params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    )?;
    let local_roi = signinum_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params = fast_subsampled_windowed_pack_params_for_dims::<P>(
        (source_window.w, source_window.h),
        fmt,
        local_roi,
    )?;
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len =
        source_window.w.div_ceil(2) as usize * P::chroma_height(source_window.h) as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, false);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes().as_ptr().cast(),
        packet.entropy_bytes().len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * scaled_roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let decode_pipeline = P::scaled_region_decode_pipeline(runtime);
    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &decode_params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = P::pack_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    bind_three_plane_pack::<JpegFast420WindowedPackParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (scaled_roi.w, scaled_roi.h));
    pack_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(map_status(status));
    }

    Ok(Some(Surface::from_metal_buffer(
        out_buffer,
        (scaled_roi.w, scaled_roi.h),
        fmt,
    )))
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_to_surface(runtime, packet, fmt, |status| {
        decode_error_from_cpu(decoder, fmt, status)
    })
}

#[cfg(target_os = "macos")]
fn decode_fast420_to_rgb_buffer(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
    output_storage: MTLResourceOptions,
) -> Result<Option<FastRgbDecodeBuffer>, Error> {
    decode_fast_subsampled_to_rgb_buffer(runtime, packet, fmt, output_storage, |status| {
        decode_error_from_cpu(decoder, fmt, status)
    })
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_region_to_surface(runtime, packet, fmt, roi, |status| {
        decode_error_from_cpu(decoder, fmt, status)
    })
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_scaled_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_scaled_to_surface(runtime, packet, fmt, scale, |status| {
        decode_error_from_cpu(decoder, fmt, status)
    })
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_scaled_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast420PacketV1>,
    fmt: PixelFormat,
    scaled_roi: signinum_jpeg::Rect,
    scale: signinum_core::Downscale,
) -> Result<Option<Surface>, Error> {
    try_decode_fast_subsampled_scaled_region_to_surface(
        runtime,
        packet,
        fmt,
        scaled_roi,
        scale,
        |status| decode_error_from_cpu(decoder, fmt, status),
    )
}

#[cfg(target_os = "macos")]
fn fast444_plane_mode(decoder: &CpuDecoder<'_>) -> PlaneMode {
    match decoder.info().color_space {
        JpegColorSpace::Rgb => PlaneMode::Rgb,
        _ => PlaneMode::YCbCr,
    }
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let params = fast444_params(packet)?;
    let mode = fast444_plane_mode(decoder);
    let plane_len = params.width as usize * params.height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let chroma_blue_plane = new_private_buffer(&runtime.device, plane_len);
    let chroma_red_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &chroma_blue_plane, &chroma_red_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    PlaneStage {
        dims: packet.dimensions,
        mode,
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
    }
    .finish_resident_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_to_private_rgb8_tile(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
) -> Result<Option<crate::ResidentPrivateJpegTile>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };

    let params = fast444_params(packet)?;
    let mode = fast444_plane_mode(decoder);
    let plane_len = params.width as usize * params.height as usize;
    let y_plane = new_private_buffer(&runtime.device, plane_len);
    let chroma_blue_plane = new_private_buffer(&runtime.device, plane_len);
    let chroma_red_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &chroma_blue_plane, &chroma_red_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, PixelFormat::Rgb8, status));
    }

    Ok(Some(
        PlaneStage {
            dims: packet.dimensions,
            mode,
            plane0: y_plane,
            plane1: Some(chroma_blue_plane),
            plane2: Some(chroma_red_plane),
        }
        .dispatch_private_rgb8_with_runtime(runtime, status_buffer),
    ))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let mut params = fast444_region_params(packet, roi)?;
    let (first_mcu, end_mcu) = mcu_range_for_rect(roi, packet.mcus_per_row, packet.mcu_rows, 8, 8);
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    )?;
    let mode = fast444_plane_mode(decoder);
    let plane_len = params.width as usize * params.height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let chroma_blue_plane = new_private_buffer(&runtime.device, plane_len);
    let chroma_red_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_region_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &chroma_blue_plane, &chroma_red_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    PlaneStage {
        dims: (roi.w, roi.h),
        mode,
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
    }
    .finish_resident_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_scaled_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(params) = fast444_scaled_params(packet, scale) else {
        return Ok(None);
    };

    let mode = fast444_plane_mode(decoder);
    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let chroma_blue_plane = new_private_buffer(&runtime.device, plane_len);
    let chroma_red_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &chroma_blue_plane, &chroma_red_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    PlaneStage {
        dims: (params.scaled_width, params.scaled_height),
        mode,
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
    }
    .finish_resident_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_scaled_region_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegFast444PacketV1>,
    fmt: PixelFormat,
    scaled_roi: signinum_jpeg::Rect,
    scale: signinum_core::Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };
    let Some(mut params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
        return Ok(None);
    };
    let mcu_size = 8u32 >> params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        scaled_roi,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_size,
        mcu_size,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    )?;

    let mode = fast444_plane_mode(decoder);
    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let chroma_blue_plane = new_private_buffer(&runtime.device, plane_len);
    let chroma_red_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_region_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &chroma_blue_plane, &chroma_red_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(status) = first_decode_error_status(&status_buffer, decode_threads) {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    PlaneStage {
        dims: (scaled_roi.w, scaled_roi.h),
        mode,
        plane0: y_plane,
        plane1: Some(chroma_blue_plane),
        plane2: Some(chroma_red_plane),
    }
    .finish_resident_with_runtime(runtime, fmt)
    .map(Some)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        decode_to_surface_with_runtime(
            runtime,
            decoder,
            pool,
            fmt,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        )
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_to_surface_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
    session: &crate::MetalBackendSession,
) -> Result<Surface, Error> {
    with_runtime_for_session(session, |runtime| {
        decode_to_surface_with_runtime(
            runtime,
            decoder,
            pool,
            fmt,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        )
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_private_rgb8_tile_with_session(
    decoder: &CpuDecoder<'_>,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
    session: &crate::MetalBackendSession,
) -> Result<crate::ResidentPrivateJpegTile, Error> {
    with_runtime_for_session(session, |runtime| {
        if let Some(tile) =
            try_decode_fast444_to_private_rgb8_tile(runtime, decoder, fast444_packet)?
        {
            return Ok(tile);
        }
        if let Some(decoded) = decode_fast422_to_rgb_buffer(
            runtime,
            fast422_packet,
            PixelFormat::Rgb8,
            MTLResourceOptions::StorageModePrivate,
        )? {
            return Ok(private_jpeg_tile_from_fast_rgb_buffer(decoded));
        }
        if let Some(decoded) = decode_fast420_to_rgb_buffer(
            runtime,
            decoder,
            fast420_packet,
            PixelFormat::Rgb8,
            MTLResourceOptions::StorageModePrivate,
        )? {
            return Ok(private_jpeg_tile_from_fast_rgb_buffer(decoded));
        }
        Err(Error::UnsupportedMetalRequest {
            reason:
                "private JPEG Metal output supports only fast baseline 4:4:4, 4:2:2, or 4:2:0 RGB8 full-tile decode",
        })
    })
}

#[cfg(target_os = "macos")]
fn decode_to_surface_with_runtime(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
) -> Result<Surface, Error> {
    if let Some(surface) = try_decode_fast444_to_surface(runtime, decoder, fast444_packet, fmt)? {
        return Ok(surface);
    }
    if let Some(surface) = try_decode_fast422_to_surface(runtime, fast422_packet, fmt)? {
        return Ok(surface);
    }
    if let Some(surface) = try_decode_fast420_to_surface(runtime, decoder, fast420_packet, fmt)? {
        return Ok(surface);
    }
    let mut stage = PlaneStage::new(
        &runtime.device,
        decoder.info().color_space,
        decoder.info().dimensions,
    )?;
    decoder.decode_component_rows_with_scratch(pool, &mut stage)?;
    stage.finish_with_runtime(runtime, fmt)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) =
            try_decode_fast444_region_to_surface(runtime, decoder, fast444_packet, fmt, roi)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast422_region_to_surface(runtime, fast422_packet, fmt, roi)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast420_region_to_surface(runtime, decoder, fast420_packet, fmt, roi)?
        {
            return Ok(surface);
        }
        let dims = (roi.w, roi.h);
        let mut stage = cached_plane_stage(runtime, decoder.info().color_space, dims)?;
        decoder.decode_region_component_rows_with_scratch(
            pool,
            &mut stage,
            roi,
            signinum_core::Downscale::None,
        )?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    scale: signinum_core::Downscale,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) =
            try_decode_fast444_scaled_to_surface(runtime, decoder, fast444_packet, fmt, scale)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast422_scaled_to_surface(runtime, fast422_packet, fmt, scale)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast420_scaled_to_surface(runtime, decoder, fast420_packet, fmt, scale)?
        {
            return Ok(surface);
        }
        let full = decoder.info().dimensions;
        let roi = signinum_jpeg::Rect {
            x: 0,
            y: 0,
            w: full.0,
            h: full.1,
        };
        let scaled = (Rect {
            x: 0,
            y: 0,
            w: full.0,
            h: full.1,
        })
        .scaled_covering(scale);
        let mut stage =
            cached_plane_stage(runtime, decoder.info().color_space, (scaled.w, scaled.h))?;
        decoder.decode_region_component_rows_with_scratch(pool, &mut stage, roi, scale)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_region_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: signinum_jpeg::Rect,
    scale: signinum_core::Downscale,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let scaled_roi = (Rect {
            x: roi.x,
            y: roi.y,
            w: roi.w,
            h: roi.h,
        })
        .scaled_covering(scale);
        if let Some(surface) = try_decode_fast444_scaled_region_to_surface(
            runtime,
            decoder,
            fast444_packet,
            fmt,
            signinum_jpeg::Rect {
                x: scaled_roi.x,
                y: scaled_roi.y,
                w: scaled_roi.w,
                h: scaled_roi.h,
            },
            scale,
        )? {
            return Ok(surface);
        }
        if let Some(surface) = try_decode_fast422_scaled_region_to_surface(
            runtime,
            fast422_packet,
            fmt,
            signinum_jpeg::Rect {
                x: scaled_roi.x,
                y: scaled_roi.y,
                w: scaled_roi.w,
                h: scaled_roi.h,
            },
            scale,
        )? {
            return Ok(surface);
        }
        if let Some(surface) = try_decode_fast420_scaled_region_to_surface(
            runtime,
            decoder,
            fast420_packet,
            fmt,
            signinum_jpeg::Rect {
                x: scaled_roi.x,
                y: scaled_roi.y,
                w: scaled_roi.w,
                h: scaled_roi.h,
            },
            scale,
        )? {
            return Ok(surface);
        }
        let scaled = (Rect {
            x: roi.x,
            y: roi.y,
            w: roi.w,
            h: roi.h,
        })
        .scaled_covering(scale);
        let mut stage =
            cached_plane_stage(runtime, decoder.info().color_space, (scaled.w, scaled.h))?;
        decoder.decode_region_component_rows_with_scratch(pool, &mut stage, roi, scale)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn compose_rgb_viewport_from_regions(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    scale: signinum_core::Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut stage = cached_plane_stage(runtime, decoder.info().color_space, viewport_dims)?;
        for tile in tiles {
            let dims = tile.source_roi.scaled_covering(scale);
            if (dims.w, dims.h) != (tile.dest.w, tile.dest.h) {
                return Err(Error::MetalKernel {
                    message: format!(
                        "viewport tile dims {:?} do not match destination rect {:?}",
                        (dims.w, dims.h),
                        tile.dest
                    ),
                });
            }
            let mut writer = ViewportPlaneWriter {
                stage: &mut stage,
                dest: tile.dest,
            };
            decoder.decode_region_component_rows_with_scratch(
                pool,
                &mut writer,
                signinum_jpeg::Rect {
                    x: tile.source_roi.x,
                    y: tile.source_roi.y,
                    w: tile.source_roi.w,
                    h: tile.source_roi.h,
                },
                scale,
            )?;
        }
        stage.finish_with_runtime(runtime, PixelFormat::Rgb8)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn compose_rgb_viewport_from_regions_into_output_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    scale: signinum_core::Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
    output: &crate::MetalBatchOutputBuffer,
    session: &crate::MetalBackendSession,
) -> Result<Surface, Error> {
    with_runtime_for_session(session, |runtime| {
        let mut stage = cached_plane_stage(runtime, decoder.info().color_space, viewport_dims)?;
        for tile in tiles {
            let dims = tile.source_roi.scaled_covering(scale);
            if (dims.w, dims.h) != (tile.dest.w, tile.dest.h) {
                return Err(Error::MetalKernel {
                    message: format!(
                        "viewport tile dims {:?} do not match destination rect {:?}",
                        (dims.w, dims.h),
                        tile.dest
                    ),
                });
            }
            let mut writer = ViewportPlaneWriter {
                stage: &mut stage,
                dest: tile.dest,
            };
            decoder.decode_region_component_rows_with_scratch(
                pool,
                &mut writer,
                signinum_jpeg::Rect {
                    x: tile.source_roi.x,
                    y: tile.source_roi.y,
                    w: tile.source_roi.w,
                    h: tile.source_roi.h,
                },
                scale,
            )?;
        }
        stage.finish_rgb8_into_output_with_runtime(runtime, output)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn compose_rgb_viewport_from_regions_into_textures_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut signinum_jpeg::ScratchPool,
    scale: signinum_core::Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
    output: &crate::MetalBatchTextureOutput,
    session: &crate::MetalBackendSession,
) -> Result<crate::MetalTextureTile, Error> {
    with_runtime_for_session(session, |runtime| {
        let mut stage = cached_plane_stage(runtime, decoder.info().color_space, viewport_dims)?;
        for tile in tiles {
            let dims = tile.source_roi.scaled_covering(scale);
            if (dims.w, dims.h) != (tile.dest.w, tile.dest.h) {
                return Err(Error::MetalKernel {
                    message: format!(
                        "viewport tile dims {:?} do not match destination rect {:?}",
                        (dims.w, dims.h),
                        tile.dest
                    ),
                });
            }
            let mut writer = ViewportPlaneWriter {
                stage: &mut stage,
                dest: tile.dest,
            };
            decoder.decode_region_component_rows_with_scratch(
                pool,
                &mut writer,
                signinum_jpeg::Rect {
                    x: tile.source_roi.x,
                    y: tile.source_roi.y,
                    w: tile.source_roi.w,
                    h: tile.source_roi.h,
                },
                scale,
            )?;
        }
        stage.finish_rgba8_into_texture_output_with_runtime(runtime, output)
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::Storage;
    use std::sync::Arc;

    const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
    const BASELINE_420_RESTART: &[u8] =
        include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");
    const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");

    #[test]
    fn mcu_range_for_rect_covers_only_touched_rows_and_columns() {
        let roi = signinum_jpeg::Rect {
            x: 19,
            y: 35,
            w: 11,
            h: 34,
        };

        assert_eq!(mcu_range_for_rect(roi, 8, 6, 16, 16), (17, 34));
    }

    #[test]
    fn restart_work_for_mcu_range_slices_to_overlapping_restart_segments() {
        let restart_offsets = [0, 10, 20, 30, 40, 50];

        let (restart_start_mcu, offsets) =
            restart_work_for_mcu_range(&restart_offsets, 4, 24, 9, 15);

        assert_eq!(restart_start_mcu, 8);
        assert_eq!(offsets, &[20, 30]);
    }

    #[test]
    fn runtime_initialization_error_classifies_device_unavailable() {
        assert!(matches!(
            runtime_initialization_error(&MetalSupportError::MetalUnavailable),
            Error::MetalUnavailable
        ));
        assert!(matches!(
            runtime_initialization_error(&MetalSupportError::ShaderLibrary {
                message: "failed to compile Metal library".to_string()
            }),
            Error::MetalRuntime { .. }
        ));
    }

    #[test]
    fn fast420_params_rejects_output_stride_overflow_without_panic() {
        let packet = minimal_fast420_packet((u32::MAX, 1));

        let Err(err) = fast_subsampled_params(&packet, PixelFormat::Rgba8) else {
            panic!("expected stride overflow");
        };

        assert!(matches!(err, Error::MetalKernel { .. }));
        assert!(
            err.to_string().contains("fast420 output stride"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fast422_region_params_rejects_output_stride_overflow_without_panic() {
        let packet = minimal_fast422_packet((16, 8));
        let source_window = signinum_jpeg::Rect {
            x: 0,
            y: 0,
            w: u32::MAX,
            h: 1,
        };

        let Err(err) = fast_subsampled_region_params(&packet, PixelFormat::Rgba8, source_window)
        else {
            panic!("expected stride overflow");
        };

        assert!(matches!(err, Error::MetalKernel { .. }));
        assert!(
            err.to_string().contains("fast422 region output stride"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fast444_params_accepts_minimal_packet() {
        let packet = minimal_fast444_packet((8, 8));

        let params = fast444_params(&packet).expect("fast444 params");

        assert_eq!(params.width, 8);
        assert_eq!(params.height, 8);
        assert_eq!(params.restart_offset_count, 1);
        assert_eq!(params.entropy_len, 0);
    }

    #[test]
    fn viewport_plane_cache_is_runtime_local() {
        let runtime_a = MetalRuntime::new().expect("Metal runtime");
        let runtime_b =
            MetalRuntime::new_with_device(runtime_a.device.clone()).expect("Metal runtime");

        let stage_a = cached_plane_stage(&runtime_a, JpegColorSpace::YCbCr, (8, 8)).expect("stage");
        let stage_b = cached_plane_stage(&runtime_b, JpegColorSpace::YCbCr, (8, 8)).expect("stage");
        drop((stage_a, stage_b));

        assert_ne!(
            runtime_a
                .viewport_plane_cache_id_for_test()
                .expect("cache")
                .expect("cache entry"),
            runtime_b
                .viewport_plane_cache_id_for_test()
                .expect("cache")
                .expect("cache entry")
        );
    }

    #[test]
    fn shader_decode_block_clears_coefficients_with_vector_stores() {
        assert!(
            SHADER_SOURCE.contains("thread short4 *coeff_chunks"),
            "decode_block should clear coeffs with packed short4 stores"
        );
        assert!(
            SHADER_SOURCE.contains("coeff_chunks[i] = short4(0);"),
            "decode_block should zero each packed coefficient chunk"
        );
    }

    #[test]
    fn shader_source_keeps_entropy_fast_paths() {
        assert!(SHADER_SOURCE.contains("inline bool refill_four_bytes("));
        assert!(
            SHADER_SOURCE.contains("return refill_four_bytes(br, bytes, len) || refill_one_byte")
        );
        assert!(SHADER_SOURCE.contains("ensure_bits_padded(br, bytes, len, 9)"));
        assert!(SHADER_SOURCE.contains("table.fast_len[fast_index]"));
        assert!(SHADER_SOURCE.contains("inline bool decode_block_skip("));
        assert!(SHADER_SOURCE.contains("skip_receive_extend(br, bytes, len, ssss, status)"));
        assert!(SHADER_SOURCE.contains("inline bool configure_batch_entropy_thread("));
    }

    #[test]
    fn shader_kernels_use_incremental_mx_my() {
        assert!(
            SHADER_SOURCE.contains("inline void init_mcu_cursor("),
            "fast decode kernels should seed mx/my via init_mcu_cursor instead of dividing per MCU"
        );
        assert!(
            SHADER_SOURCE.contains("inline void advance_mcu_cursor("),
            "fast decode kernels should carry mx/my via advance_mcu_cursor instead of dividing per MCU"
        );
        assert!(
            !SHADER_SOURCE.contains("mcu_index / params.mcus_per_row"),
            "no fast kernel should still divide mcu_index by mcus_per_row inside the MCU loop"
        );
        assert!(
            !SHADER_SOURCE.contains("mcu_index % params.mcus_per_row"),
            "no fast kernel should still modulo mcu_index by mcus_per_row inside the MCU loop"
        );
    }

    #[test]
    fn fast420_batch_timing_env_requires_explicit_one() {
        assert!(fast420_batch_timing_value_enabled(Some(
            std::ffi::OsStr::new("1")
        )));
        assert!(!fast420_batch_timing_value_enabled(Some(
            std::ffi::OsStr::new("true")
        )));
        assert!(!fast420_batch_timing_value_enabled(None));
    }

    #[test]
    fn one_dimensional_dispatch_width_tracks_work_without_full_threadgroup_waste() {
        assert_eq!(choose_1d_threadgroup_width(32, 1024, 1), 32);
        assert_eq!(choose_1d_threadgroup_width(32, 1024, 33), 64);
        assert_eq!(choose_1d_threadgroup_width(32, 1024, 256), 256);
        assert_eq!(choose_1d_threadgroup_width(32, 1024, 257), 256);
    }

    #[test]
    fn auto_batched_packets_skip_distinct_region_scaled_requests() {
        let packet =
            signinum_jpeg::adapter::build_fast420_packet(BASELINE_420_RESTART).expect("packet");
        let roi = Rect {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
        };
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::<[u8]>::from(BASELINE_420_RESTART),
                PixelFormat::Rgb8,
                BackendRequest::Auto,
                batch::BatchOp::RegionScaled {
                    roi,
                    scale: signinum_core::Downscale::Quarter,
                },
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::<[u8]>::from(BASELINE_420_RESTART),
                PixelFormat::Rgb8,
                BackendRequest::Auto,
                batch::BatchOp::RegionScaled {
                    roi,
                    scale: signinum_core::Downscale::Quarter,
                },
                None,
                None,
                Some(packet),
            ),
        ];

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_none());
    }

    #[test]
    fn auto_batched_packets_keep_large_repeated_region_scaled_requests_off_metal() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet = signinum_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
        let roi = Rect {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
        };
        let requests = (0..=REGION_SCALED_BATCH_CHUNK)
            .map(|_| {
                batch::QueuedRequest::new(
                    Arc::clone(&input),
                    PixelFormat::Rgb8,
                    BackendRequest::Auto,
                    batch::BatchOp::RegionScaled {
                        roi,
                        scale: signinum_core::Downscale::Quarter,
                    },
                    None,
                    None,
                    Some(packet.clone()),
                )
            })
            .collect::<Vec<_>>();

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_none());
    }

    #[test]
    fn auto_batched_packets_require_wsi_batch_threshold() {
        let input = Arc::<[u8]>::from(BASELINE_420_RESTART);
        let packet =
            signinum_jpeg::adapter::build_fast420_packet(BASELINE_420_RESTART).expect("packet");
        let requests = (0..7)
            .map(|_| {
                batch::QueuedRequest::new(
                    Arc::clone(&input),
                    PixelFormat::Rgb8,
                    BackendRequest::Auto,
                    batch::BatchOp::Full,
                    None,
                    None,
                    Some(packet.clone()),
                )
            })
            .collect::<Vec<_>>();

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_none());
    }

    #[test]
    fn auto_batched_packets_accept_restart_wsi_batch_at_threshold() {
        let input = Arc::<[u8]>::from(BASELINE_420_RESTART);
        let packet =
            signinum_jpeg::adapter::build_fast420_packet(BASELINE_420_RESTART).expect("packet");
        let requests = (0..8)
            .map(|_| {
                batch::QueuedRequest::new(
                    Arc::clone(&input),
                    PixelFormat::Rgb8,
                    BackendRequest::Auto,
                    batch::BatchOp::Full,
                    None,
                    None,
                    Some(packet.clone()),
                )
            })
            .collect::<Vec<_>>();

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_some());
    }

    #[test]
    fn auto_batched_packets_accept_large_nonrestart_wsi_batch_at_threshold() {
        let input = Arc::<[u8]>::from(generated_rgb_jpeg(512));
        let fast444_packet = signinum_jpeg::adapter::build_fast444_packet(input.as_ref()).ok();
        let fast422_packet = signinum_jpeg::adapter::build_fast422_packet(input.as_ref()).ok();
        let fast420_packet = signinum_jpeg::adapter::build_fast420_packet(input.as_ref()).ok();
        assert!(
            fast444_packet.is_some() || fast422_packet.is_some() || fast420_packet.is_some(),
            "generated JPEG must be packet-decodable"
        );
        let requests = (0..8)
            .map(|_| {
                batch::QueuedRequest::new(
                    Arc::clone(&input),
                    PixelFormat::Rgb8,
                    BackendRequest::Auto,
                    batch::BatchOp::Full,
                    fast444_packet.clone(),
                    fast422_packet.clone(),
                    fast420_packet.clone(),
                )
            })
            .collect::<Vec<_>>();

        assert!(batched_fast_packets(&requests)
            .expect("packet lookup")
            .is_some());
    }

    fn generated_rgb_jpeg(dim: u16) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(dim as usize * dim as usize * 3);
        for y in 0..dim {
            for x in 0..dim {
                let xf = u32::from(x);
                let yf = u32::from(y);
                rgb.push(((xf * 13 + yf * 3) & 0xff) as u8);
                rgb.push(((xf * 5 + yf * 11 + (xf ^ yf)) & 0xff) as u8);
                rgb.push(((xf * 7 + yf * 17 + (xf.wrapping_mul(yf) >> 5)) & 0xff) as u8);
            }
        }

        let mut jpeg = Vec::new();
        let mut encoder = jpeg_encoder::Encoder::new(&mut jpeg, 90);
        encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::F_2_2);
        encoder
            .encode(&rgb, dim, dim, jpeg_encoder::ColorType::Rgb)
            .expect("encode generated JPEG");
        jpeg
    }

    #[test]
    fn fast420_packet_scaled_decode_matches_cpu_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
        for scale in [
            signinum_core::Downscale::Half,
            signinum_core::Downscale::Quarter,
        ] {
            let (expected, _) = decoder
                .decode_scaled(PixelFormat::Rgb8, scale)
                .expect("cpu scaled");

            let surface = with_runtime(|runtime| {
                let surface = try_decode_fast420_scaled_to_surface(
                    runtime,
                    &decoder,
                    Some(&packet),
                    PixelFormat::Rgb8,
                    scale,
                )?
                .expect("fast420 scaled surface");
                Ok::<_, Error>(surface)
            })
            .expect("metal scaled");

            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_packet_region_decode_matches_cpu_region_bytes() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 3,
            y: 2,
            w: 9,
            h: 10,
        };
        let (expected, _) = decoder
            .decode_region(PixelFormat::Rgb8, roi)
            .expect("cpu region");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast420_region_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                roi,
            )?
            .expect("fast420 region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region");

        assert_eq!(surface.dimensions, (roi.w, roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast420_region_batch_decode_matches_cpu_region_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet = signinum_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
        let roi = Rect {
            x: 4,
            y: 4,
            w: 8,
            h: 8,
        };
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let (expected, _) = decoder
            .decode_region(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
            )
            .expect("cpu region");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("region batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (roi.w, roi.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_full_batch_decode_uses_shared_surface_offsets() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet = signinum_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast420 full batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for (index, result) in results.iter().enumerate() {
            let surface = result.as_ref().expect("surface");
            assert_eq!(surface.dimensions, (16, 16));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
            let Storage::Metal { offset, .. } = &surface.storage else {
                panic!("expected Metal storage");
            };
            assert_eq!(*offset, index * expected.len());
        }
    }

    #[test]
    fn fast420_split_full_batch_decode_matches_cpu_bytes() {
        let jpeg = generated_rgb_jpeg(32);
        let input = Arc::<[u8]>::from(jpeg.into_boxed_slice());
        let packet = signinum_jpeg::adapter::build_fast420_packet(input.as_ref()).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet),
            ),
        ];
        let packets = batched_fast_packets(&requests)
            .expect("packet lookup")
            .expect("packets");
        let decoder = CpuDecoder::new(input.as_ref()).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = with_runtime(|runtime| {
            try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output::<
                JpegFast420PacketV1,
            >(
                runtime,
                &requests,
                &packets,
                FastBatchDecodeMode::SplitCoeffIdct,
                None,
            )
        })
        .expect("batch result")
        .expect("split fast420 full batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (32, 32));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_batch_clears_high_ac_before_dc_only_blocks() {
        let input = Arc::<[u8]>::from(fast420_high_ac_then_dc_only_jpeg(1));
        let packet = signinum_jpeg::adapter::build_fast420_packet(input.as_ref()).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(input.as_ref()).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast420 full batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (16, 16));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_batch_matches_cpu_for_high_ac_overflow_coefficients() {
        let input = Arc::<[u8]>::from(fast420_high_ac_then_dc_only_jpeg(u8::MAX));
        let packet = signinum_jpeg::adapter::build_fast420_packet(input.as_ref()).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(input.as_ref()).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast420 full batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (16, 16));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 3,
            y: 2,
            w: 9,
            h: 10,
        };
        let scale = signinum_core::Downscale::Quarter;
        let (expected, _) = decoder
            .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
            .expect("cpu region scaled");
        let scaled_roi = signinum_jpeg::Rect {
            x: roi.x / 4,
            y: roi.y / 4,
            w: (roi.x + roi.w).div_ceil(4) - (roi.x / 4),
            h: (roi.y + roi.h).div_ceil(4) - (roi.y / 4),
        };

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast420_scaled_region_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                scaled_roi,
                scale,
            )?
            .expect("fast420 scaled region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region scaled");

        assert_eq!(surface.dimensions, (scaled_roi.w, scaled_roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast420_region_scaled_batch_decode_matches_cpu_region_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet = signinum_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
        let roi = Rect {
            x: 3,
            y: 2,
            w: 9,
            h: 10,
        };
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let (expected, _) = decoder
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled");
        let scaled = roi.scaled_covering(scale);

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("region scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (scaled.w, scaled.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast420_scaled_batch_decode_matches_cpu_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_420);
        let packet = signinum_jpeg::adapter::build_fast420_packet(BASELINE_420).expect("packet");
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                None,
                None,
                Some(packet.clone()),
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                None,
                None,
                Some(packet),
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (4, 4));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast422_packet_full_decode_matches_cpu_bytes() {
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast422_to_surface(runtime, Some(&packet), PixelFormat::Rgb8)?
                .expect("fast422 surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal full decode");

        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast422_full_batch_decode_matches_cpu_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_422);
        let packet = signinum_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                Some(packet.clone()),
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                Some(packet),
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast422 batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for (index, result) in results.iter().enumerate() {
            let surface = result.as_ref().expect("surface");
            assert_eq!(surface.dimensions, (16, 8));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
            let Storage::Metal { offset, .. } = &surface.storage else {
                panic!("expected Metal storage");
            };
            assert_eq!(*offset, index * expected.len());
        }
    }

    #[test]
    fn fast422_packet_region_decode_matches_cpu_region_bytes() {
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 3,
            y: 1,
            w: 9,
            h: 5,
        };
        let (expected, _) = decoder
            .decode_region(PixelFormat::Rgb8, roi)
            .expect("cpu region");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast422_region_to_surface(
                runtime,
                Some(&packet),
                PixelFormat::Rgb8,
                roi,
            )?
            .expect("fast422 region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region");

        assert_eq!(surface.dimensions, (roi.w, roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast422_region_batch_decode_matches_cpu_region_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_422);
        let packet = signinum_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
        let roi = Rect {
            x: 3,
            y: 1,
            w: 9,
            h: 5,
        };
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                None,
                Some(packet.clone()),
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                None,
                Some(packet),
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let (expected, _) = decoder
            .decode_region(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
            )
            .expect("cpu region");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast422 region batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (roi.w, roi.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast422_packet_scaled_decode_matches_cpu_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
        for (scale, dims) in [
            (signinum_core::Downscale::Half, (8, 4)),
            (signinum_core::Downscale::Quarter, (4, 2)),
        ] {
            let (expected, _) = decoder
                .decode_scaled(PixelFormat::Rgb8, scale)
                .expect("cpu scaled");

            let surface = with_runtime(|runtime| {
                let surface = try_decode_fast422_scaled_to_surface(
                    runtime,
                    Some(&packet),
                    PixelFormat::Rgb8,
                    scale,
                )?
                .expect("fast422 scaled surface");
                Ok::<_, Error>(surface)
            })
            .expect("metal scaled");

            assert_eq!(surface.dimensions, dims);
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast422_scaled_batch_decode_matches_cpu_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_422);
        let packet = signinum_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                None,
                Some(packet.clone()),
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                None,
                Some(packet),
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast422 scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (4, 2));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast422_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 3,
            y: 1,
            w: 9,
            h: 5,
        };
        let scale = signinum_core::Downscale::Half;
        let (expected, _) = decoder
            .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
            .expect("cpu region scaled");
        let scaled_roi = signinum_jpeg::Rect {
            x: roi.x / 2,
            y: roi.y / 2,
            w: (roi.x + roi.w).div_ceil(2) - (roi.x / 2),
            h: (roi.y + roi.h).div_ceil(2) - (roi.y / 2),
        };

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast422_scaled_region_to_surface(
                runtime,
                Some(&packet),
                PixelFormat::Rgb8,
                scaled_roi,
                scale,
            )?
            .expect("fast422 scaled region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region scaled");

        assert_eq!(surface.dimensions, (scaled_roi.w, scaled_roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast422_region_scaled_batch_decode_matches_cpu_region_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_422);
        let packet = signinum_jpeg::adapter::build_fast422_packet(BASELINE_422).expect("packet");
        let roi = Rect {
            x: 3,
            y: 1,
            w: 9,
            h: 5,
        };
        let scale = signinum_core::Downscale::Half;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                None,
                Some(packet.clone()),
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                None,
                Some(packet),
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");
        let (expected, _) = decoder
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled");
        let scaled = roi.scaled_covering(scale);

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("fast422 region scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (scaled.w, scaled.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast444_packet_full_decode_matches_cpu_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
        let (expected, _) = decoder.decode(PixelFormat::Rgb8).expect("cpu full decode");

        let surface = with_runtime(|runtime| {
            let surface =
                try_decode_fast444_to_surface(runtime, &decoder, Some(&packet), PixelFormat::Rgb8)?
                    .expect("fast444 surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal full decode");

        assert_eq!(
            surface.residency(),
            crate::SurfaceResidency::MetalResidentDecode
        );
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast444_packet_scaled_decode_matches_cpu_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
        for scale in [
            signinum_core::Downscale::Half,
            signinum_core::Downscale::Quarter,
        ] {
            let (expected, _) = decoder
                .decode_scaled(PixelFormat::Rgb8, scale)
                .expect("cpu scaled");

            let surface = with_runtime(|runtime| {
                let surface = try_decode_fast444_scaled_to_surface(
                    runtime,
                    &decoder,
                    Some(&packet),
                    PixelFormat::Rgb8,
                    scale,
                )?
                .expect("fast444 scaled surface");
                Ok::<_, Error>(surface)
            })
            .expect("metal scaled");

            assert_eq!(
                surface.residency(),
                crate::SurfaceResidency::MetalResidentDecode
            );
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast444_packet_region_decode_matches_cpu_region_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let (expected, _) = decoder
            .decode_region(PixelFormat::Rgb8, roi)
            .expect("cpu region");

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast444_region_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                roi,
            )?
            .expect("fast444 region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region");

        assert_eq!(surface.dimensions, (roi.w, roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(
            surface.residency(),
            crate::SurfaceResidency::MetalResidentDecode
        );
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast444_region_batch_decode_matches_cpu_region_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_444);
        let packet = signinum_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
        let roi = Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                Some(packet.clone()),
                None,
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Region(roi),
                Some(packet),
                None,
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let (expected, _) = decoder
            .decode_region(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
            )
            .expect("cpu region");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("region batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (roi.w, roi.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(
                surface.residency(),
                crate::SurfaceResidency::MetalResidentDecode
            );
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast444_packet_region_scaled_decode_matches_cpu_region_scaled_bytes() {
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let packet = signinum_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
        let roi = signinum_jpeg::Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let scale = signinum_core::Downscale::Quarter;
        let (expected, _) = decoder
            .decode_region_scaled(PixelFormat::Rgb8, roi, scale)
            .expect("cpu region scaled");
        let scaled_roi = signinum_jpeg::Rect {
            x: roi.x / 4,
            y: roi.y / 4,
            w: (roi.x + roi.w).div_ceil(4) - (roi.x / 4),
            h: (roi.y + roi.h).div_ceil(4) - (roi.y / 4),
        };

        let surface = with_runtime(|runtime| {
            let surface = try_decode_fast444_scaled_region_to_surface(
                runtime,
                &decoder,
                Some(&packet),
                PixelFormat::Rgb8,
                scaled_roi,
                scale,
            )?
            .expect("fast444 scaled region surface");
            Ok::<_, Error>(surface)
        })
        .expect("metal region scaled");

        assert_eq!(surface.dimensions, (scaled_roi.w, scaled_roi.h));
        assert_eq!(surface.fmt, PixelFormat::Rgb8);
        assert_eq!(
            surface.residency(),
            crate::SurfaceResidency::MetalResidentDecode
        );
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[test]
    fn fast444_region_scaled_batch_decode_matches_cpu_region_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_444);
        let packet = signinum_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
        let roi = Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                Some(packet.clone()),
                None,
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::RegionScaled { roi, scale },
                Some(packet),
                None,
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let (expected, _) = decoder
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled");
        let scaled = roi.scaled_covering(scale);

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("region scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (scaled.w, scaled.h));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(
                surface.residency(),
                crate::SurfaceResidency::MetalResidentDecode
            );
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn fast444_scaled_batch_decode_matches_cpu_scaled_bytes() {
        let input = Arc::<[u8]>::from(BASELINE_444);
        let packet = signinum_jpeg::adapter::build_fast444_packet(BASELINE_444).expect("packet");
        let scale = signinum_core::Downscale::Quarter;
        let requests = vec![
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                Some(packet.clone()),
                None,
                None,
            ),
            batch::QueuedRequest::new(
                Arc::clone(&input),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Scaled(scale),
                Some(packet),
                None,
                None,
            ),
        ];
        let decoder = CpuDecoder::new(BASELINE_444).expect("decoder");
        let (expected, _) = decoder
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled");

        let results = decode_full_batch_to_surfaces(&requests)
            .expect("batch result")
            .expect("scaled batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.dimensions, (2, 2));
            assert_eq!(surface.fmt, PixelFormat::Rgb8);
            assert_eq!(
                surface.residency(),
                crate::SurfaceResidency::MetalResidentDecode
            );
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    fn fast420_high_ac_then_dc_only_jpeg(ac_quant: u8) -> Vec<u8> {
        assert!(ac_quant > 0, "JPEG quant entries must be nonzero");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0xff, 0xd8]);

        let mut quant = [1u8; 64];
        quant[63] = ac_quant;
        let mut dqt = Vec::with_capacity(65);
        dqt.push(0x00);
        dqt.extend_from_slice(&quant);
        append_jpeg_segment(&mut bytes, 0xdb, &dqt);

        append_jpeg_segment(
            &mut bytes,
            0xc0,
            &[
                8,
                0,
                16,
                0,
                16,
                3,
                1,
                (2 << 4) | 2,
                0,
                2,
                (1 << 4) | 1,
                0,
                3,
                (1 << 4) | 1,
                0,
            ],
        );

        let mut dc_bits = [0u8; 16];
        dc_bits[0] = 1;
        let mut dht_dc = Vec::with_capacity(18);
        dht_dc.push(0x00);
        dht_dc.extend_from_slice(&dc_bits);
        dht_dc.push(0x00);
        append_jpeg_segment(&mut bytes, 0xc4, &dht_dc);

        let mut ac_bits = [0u8; 16];
        ac_bits[1] = 3;
        let mut dht_ac = Vec::with_capacity(20);
        dht_ac.push(0x10);
        dht_ac.extend_from_slice(&ac_bits);
        dht_ac.extend_from_slice(&[0x00, 0xf0, 0xea]);
        append_jpeg_segment(&mut bytes, 0xc4, &dht_ac);

        append_jpeg_segment(&mut bytes, 0xda, &[3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]);

        bytes.extend_from_slice(&fast420_high_ac_entropy());
        bytes.extend_from_slice(&[0xff, 0xd9]);
        bytes
    }

    fn append_jpeg_segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        bytes.extend_from_slice(&[0xff, marker]);
        let len = u16::try_from(payload.len() + 2).expect("JPEG segment length fits in u16");
        bytes.extend_from_slice(&len.to_be_bytes());
        bytes.extend_from_slice(payload);
    }

    fn minimal_fast420_packet(dimensions: (u32, u32)) -> JpegFast420PacketV1 {
        let [y_dc_table, y_ac_table, cb_dc_table, cb_ac_table, cr_dc_table, cr_ac_table] =
            empty_packet_huffman_tables();
        JpegFast420PacketV1 {
            dimensions,
            mcus_per_row: 1,
            mcu_rows: 1,
            restart_interval_mcus: 0,
            restart_offsets: vec![0],
            entropy_checkpoints: vec![empty_entropy_checkpoint()],
            y_quant: [1; 64],
            cb_quant: [1; 64],
            cr_quant: [1; 64],
            y_dc_table,
            y_ac_table,
            cb_dc_table,
            cb_ac_table,
            cr_dc_table,
            cr_ac_table,
            entropy_bytes: Vec::new(),
        }
    }

    fn minimal_fast422_packet(dimensions: (u32, u32)) -> JpegFast422PacketV1 {
        let [y_dc_table, y_ac_table, cb_dc_table, cb_ac_table, cr_dc_table, cr_ac_table] =
            empty_packet_huffman_tables();
        JpegFast422PacketV1 {
            dimensions,
            mcus_per_row: 1,
            mcu_rows: 1,
            restart_interval_mcus: 0,
            restart_offsets: vec![0],
            entropy_checkpoints: vec![empty_entropy_checkpoint()],
            y_quant: [1; 64],
            cb_quant: [1; 64],
            cr_quant: [1; 64],
            y_dc_table,
            y_ac_table,
            cb_dc_table,
            cb_ac_table,
            cr_dc_table,
            cr_ac_table,
            entropy_bytes: Vec::new(),
        }
    }

    fn minimal_fast444_packet(dimensions: (u32, u32)) -> JpegFast444PacketV1 {
        let [y_dc_table, y_ac_table, cb_dc_table, cb_ac_table, cr_dc_table, cr_ac_table] =
            empty_packet_huffman_tables();
        JpegFast444PacketV1 {
            dimensions,
            mcus_per_row: 1,
            mcu_rows: 1,
            restart_interval_mcus: 0,
            restart_offsets: vec![0],
            entropy_checkpoints: vec![empty_entropy_checkpoint()],
            y_quant: [1; 64],
            cb_quant: [1; 64],
            cr_quant: [1; 64],
            y_dc_table,
            y_ac_table,
            cb_dc_table,
            cb_ac_table,
            cr_dc_table,
            cr_ac_table,
            entropy_bytes: Vec::new(),
        }
    }

    fn empty_packet_huffman_tables() -> [PacketHuffmanTable; 6] {
        std::array::from_fn(|_| PacketHuffmanTable {
            bits: [0; 16],
            values_len: 0,
            values: [0; 256],
        })
    }

    fn empty_entropy_checkpoint() -> JpegEntropyCheckpointV1 {
        JpegEntropyCheckpointV1 {
            mcu_index: 0,
            entropy_pos: 0,
            bit_acc: 0,
            bit_count: 0,
            y_prev_dc: 0,
            cb_prev_dc: 0,
            cr_prev_dc: 0,
            reserved: 0,
        }
    }

    fn fast420_high_ac_entropy() -> Vec<u8> {
        let mut writer = EntropyBitWriter::default();
        emit_high_ac_block(&mut writer);
        for _ in 0..5 {
            emit_dc_only_block(&mut writer);
        }
        writer.finish()
    }

    fn emit_high_ac_block(writer: &mut EntropyBitWriter) {
        writer.push_bits(0, 1);
        for _ in 0..3 {
            writer.push_bits(0b01, 2);
        }
        writer.push_bits(0b10, 2);
        writer.push_bits(0b11_1111_1111, 10);
    }

    fn emit_dc_only_block(writer: &mut EntropyBitWriter) {
        writer.push_bits(0, 1);
        writer.push_bits(0b00, 2);
    }

    #[derive(Default)]
    struct EntropyBitWriter {
        bytes: Vec<u8>,
        current: u8,
        bit_count: u8,
    }

    impl EntropyBitWriter {
        fn push_bits(&mut self, bits: u16, len: u8) {
            for shift in (0..len).rev() {
                let bit = u8::from(((bits >> shift) & 1) != 0);
                self.current = (self.current << 1) | bit;
                self.bit_count += 1;
                if self.bit_count == 8 {
                    self.push_current_byte();
                }
            }
        }

        fn finish(mut self) -> Vec<u8> {
            if self.bit_count != 0 {
                let pad_bits = 8 - self.bit_count;
                self.current = (self.current << pad_bits) | ((1u8 << pad_bits) - 1);
                self.push_current_byte();
            }
            self.bytes
        }

        fn push_current_byte(&mut self) {
            self.bytes.push(self.current);
            if self.current == 0xff {
                self.bytes.push(0x00);
            }
            self.current = 0;
            self.bit_count = 0;
        }
    }
}
