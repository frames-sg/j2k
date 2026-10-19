#![allow(
    clippy::cast_possible_truncation,
    reason = "lossless device math narrows bounded 16-bit samples and 32-bit stream words"
)]
#![allow(
    clippy::too_many_arguments,
    reason = "flat device kernels mirror the CUDA ABI buffers of each stage"
)]
#![allow(
    static_mut_refs,
    reason = "CUDA shared-memory statics are accessed through device-scoped references"
)]

//! Lossless (SOF3) JPEG encode kernels.
//!
//! The stages, buffers, and parameters follow
//! `j2k_jpeg::adapter::JpegLosslessGpuEncodeParams`, whose host emulation is
//! tested against CPU encode, and match the Metal kernels in
//! `shaders_lossless_encode.metal`. Prediction and difference categories
//! come from `j2k_codec_math::jpeg::lossless`, the CPU encoder's own code.

use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::{SharedArray, kernel, thread};
use cuda_host::cuda_module;
use j2k_codec_math::jpeg::lossless::{
    DIFFERENCE_SYMBOLS, additional_bits, difference_category, predict_at,
};

include!("../../../cuda_oxide_simt_prelude.rs");

/// Status words: first out-of-range sample index, its value, packed bytes.
const STATUS_BAD_INDEX: usize = 0;
const STATUS_BAD_VALUE: usize = 1;
const STATUS_PACKED_BYTES: usize = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct J2kJpegLosslessEncodeParams {
    width: u32,
    height: u32,
    components: u32,
    sample_bytes: u32,
    pitch_bytes: u32,
    input_offset_bytes: u32,
    predictor: u32,
    point_transform: u32,
    initial_predictor: u32,
    max_sample: u32,
    interval_rows: u32,
    chunk_pixels: u32,
    chunks_per_row: u32,
    intervals: u32,
    sample_count: u32,
    packed_words: u32,
}

fn sample(input: *const u8, p: &J2kJpegLosslessEncodeParams, row: u32, index: u32) -> u32 {
    let offset = (p.input_offset_bytes + row * p.pitch_bytes + index * p.sample_bytes) as usize;
    if p.sample_bytes == 2 {
        u32::from(u16::from_ne_bytes([
            simt_load(input, offset),
            simt_load(input, offset + 1),
        ]))
    } else {
        u32::from(simt_load(input, offset))
    }
}

/// The `[start, end)` token range of one chunk.
fn chunk_tokens(p: &J2kJpegLosslessEncodeParams, chunk: u32) -> (u32, u32) {
    let row = chunk / p.chunks_per_row;
    let x0 = (chunk - row * p.chunks_per_row) * p.chunk_pixels;
    let x1 = (x0 + p.chunk_pixels).min(p.width);
    let base = row * p.width * p.components;
    (base + x0 * p.components, base + x1 * p.components)
}

fn device_atomic<'a>(words: *mut u32, index: usize) -> &'a DeviceAtomicU32 {
    // SAFETY: callers pass a host-allocated device buffer and an index the
    // host sized it for; these words are accessed only atomically while the
    // kernel runs.
    unsafe { DeviceAtomicU32::from_ptr(simt_mut_ptr_at(words, index)) }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Stage 1, one thread per sample: token `category | bits << 16` and
    /// the category histogram; out-of-range samples record their least index.
    #[kernel]
    pub unsafe fn j2k_jpeg_lossless_encode_classify(
        input: *const u8,
        tokens: *mut u32,
        histogram: *mut u32,
        status: *mut u32,
        params: *const J2kJpegLosslessEncodeParams,
    ) {
        static mut LOCAL: SharedArray<u32, DIFFERENCE_SYMBOLS> = SharedArray::UNINIT;
        let p = simt_load(params, 0);
        let lane = thread::threadIdx_x() as usize;
        // SAFETY: the shared histogram belongs to this block and is accessed
        // only atomically.
        let local = unsafe { LOCAL.as_mut_ptr() };
        let local_bin = |bin: usize| {
            // SAFETY: as above; `bin` is a category below DIFFERENCE_SYMBOLS.
            unsafe { BlockAtomicU32::from_ptr(simt_mut_ptr_at(local, bin)) }
        };
        if lane < DIFFERENCE_SYMBOLS {
            local_bin(lane).store(0, AtomicOrdering::Relaxed);
        }
        thread::sync_threads();
        let gid = thread::index_1d().get();
        if gid < p.sample_count as usize {
            let gid = gid as u32;
            let row_samples = p.width * p.components;
            let row = gid / row_samples;
            let index = gid - row * row_samples;
            let x = index / p.components;
            let value = sample(input, &p, row, index);
            if value > p.max_sample {
                device_atomic(status, STATUS_BAD_INDEX).fetch_min(gid, AtomicOrdering::Relaxed);
            }
            let shift = p.point_transform;
            let at = |row: u32, index: u32| (sample(input, &p, row, index) >> shift) as u16;
            let ra = if x > 0 {
                at(row, index - p.components)
            } else {
                0
            };
            let rb = if row > 0 { at(row - 1, index) } else { 0 };
            let rc = if x > 0 && row > 0 {
                at(row - 1, index - p.components)
            } else {
                0
            };
            let prediction = predict_at(
                p.predictor as u8,
                p.initial_predictor as u16,
                x as usize,
                (row % p.interval_rows) as usize,
                [ra, rb, rc],
            );
            let (category, bits) =
                difference_category(((value >> shift) as u16).wrapping_sub(prediction));
            simt_store(
                tokens,
                gid as usize,
                u32::from(category) | (u32::from(bits) << 16),
            );
            local_bin(usize::from(category)).fetch_add(1, AtomicOrdering::Relaxed);
        }
        thread::sync_threads();
        if lane < DIFFERENCE_SYMBOLS {
            let count = local_bin(lane).load(AtomicOrdering::Relaxed);
            if count != 0 {
                device_atomic(histogram, lane).fetch_add(count, AtomicOrdering::Relaxed);
            }
        }
    }

    /// One thread: the value of the first out-of-range sample, for the error.
    #[kernel]
    pub unsafe fn j2k_jpeg_lossless_encode_read_sample(
        input: *const u8,
        status: *mut u32,
        params: *const J2kJpegLosslessEncodeParams,
    ) {
        let p = simt_load(params, 0);
        let index = simt_load(status.cast_const(), STATUS_BAD_INDEX);
        if thread::index_1d().get() != 0 || index >= p.sample_count {
            return;
        }
        let row_samples = p.width * p.components;
        let row = index / row_samples;
        simt_store(
            status,
            STATUS_BAD_VALUE,
            sample(input, &p, row, index - row * row_samples),
        );
    }

    /// Stage 3, one thread per chunk: bits its tokens take.
    #[kernel]
    pub unsafe fn j2k_jpeg_lossless_encode_chunk_bits(
        tokens: *const u32,
        chunk_bits: *mut u32,
        params: *const J2kJpegLosslessEncodeParams,
        table: *const u32,
    ) {
        let p = simt_load(params, 0);
        let gid = thread::index_1d().get();
        if gid >= (p.chunks_per_row * p.height) as usize {
            return;
        }
        let (start, end) = chunk_tokens(&p, gid as u32);
        let mut bits = 0;
        for i in start..end {
            let category = simt_load(tokens, i as usize) & 0xFF;
            bits += (simt_load(table, category as usize) & 0xFF)
                + u32::from(additional_bits(category as u8));
        }
        simt_store(chunk_bits, gid, bits);
    }

    /// Stage 4a, one thread per row: each chunk's bit offset within its row.
    #[kernel]
    pub unsafe fn j2k_jpeg_lossless_encode_row_offsets(
        chunk_bits: *const u32,
        chunk_offsets: *mut u32,
        row_bits: *mut u32,
        params: *const J2kJpegLosslessEncodeParams,
    ) {
        let p = simt_load(params, 0);
        let row = thread::index_1d().get();
        if row >= p.height as usize {
            return;
        }
        let mut running = 0;
        for k in 0..p.chunks_per_row as usize {
            let chunk = row * p.chunks_per_row as usize + k;
            simt_store(chunk_offsets, chunk, running);
            running += simt_load(chunk_bits, chunk);
        }
        simt_store(row_bits, row, running);
    }

    /// Stage 4b, one thread per restart interval: each row's bit offset
    /// within its interval.
    #[kernel]
    pub unsafe fn j2k_jpeg_lossless_encode_interval_offsets(
        row_bits: *const u32,
        row_offsets: *mut u32,
        interval_bits: *mut u32,
        params: *const J2kJpegLosslessEncodeParams,
    ) {
        let p = simt_load(params, 0);
        let interval = thread::index_1d().get();
        if interval >= p.intervals as usize {
            return;
        }
        let first = interval as u32 * p.interval_rows;
        let last = (first + p.interval_rows).min(p.height);
        let mut running = 0;
        for row in first..last {
            simt_store(row_offsets, row as usize, running);
            running += simt_load(row_bits, row as usize);
        }
        simt_store(interval_bits, interval, running);
    }

    /// Stage 4c, one thread: byte lengths and byte bases of the intervals.
    #[kernel]
    pub unsafe fn j2k_jpeg_lossless_encode_interval_bases(
        interval_bits: *const u32,
        interval_bases: *mut u32,
        interval_bytes: *mut u32,
        status: *mut u32,
        params: *const J2kJpegLosslessEncodeParams,
    ) {
        let p = simt_load(params, 0);
        if thread::index_1d().get() != 0 {
            return;
        }
        let mut running = 0;
        for interval in 0..p.intervals as usize {
            let bytes = simt_load(interval_bits, interval).div_ceil(8);
            simt_store(interval_bytes, interval, bytes);
            simt_store(interval_bases, interval, running);
            running += bytes;
        }
        simt_store(status, STATUS_PACKED_BYTES, running);
    }

    /// Zero the words the pack stage ORs into.
    #[kernel]
    pub unsafe fn j2k_jpeg_lossless_encode_clear(
        packed: *mut u32,
        status: *const u32,
        params: *const J2kJpegLosslessEncodeParams,
    ) {
        let p = simt_load(params, 0);
        let used_words = (simt_load(status, STATUS_PACKED_BYTES) / 4 + 1).min(p.packed_words);
        let word = thread::index_1d().get();
        if word < used_words as usize {
            simt_store(packed, word, 0);
        }
    }

    /// Stage 5, one thread per chunk: OR its codes into the packed stream,
    /// held as words whose bytes are in stream order, and pad the last byte
    /// of an interval with 1-bits.
    #[kernel]
    pub unsafe fn j2k_jpeg_lossless_encode_pack(
        tokens: *const u32,
        packed: *mut u32,
        chunk_offsets: *const u32,
        row_offsets: *const u32,
        interval_bases: *const u32,
        interval_bits: *const u32,
        params: *const J2kJpegLosslessEncodeParams,
        table: *const u32,
    ) {
        let p = simt_load(params, 0);
        let gid = thread::index_1d().get();
        if gid >= (p.chunks_per_row * p.height) as usize {
            return;
        }
        let chunk = gid as u32;
        let row = chunk / p.chunks_per_row;
        let interval = row / p.interval_rows;
        let base = simt_load(interval_bases, interval as usize);
        let start = base * 8 + simt_load(row_offsets, row as usize) + simt_load(chunk_offsets, gid);
        let mut word = start / 32;
        let mut acc = 0_u64;
        let mut acc_bits = start % 32;
        // A word is shared with the previous chunk only when it starts
        // mid-word; words wholly inside this chunk are stored outright.
        let mut shared_first = acc_bits != 0;
        let (first_token, end_token) = chunk_tokens(&p, chunk);
        for i in first_token..end_token {
            let token = simt_load(tokens, i as usize);
            let category = token & 0xFF;
            let entry = simt_load(table, category as usize);
            let extra = u32::from(additional_bits(category as u8));
            let len = (entry & 0xFF) + extra;
            let bits = u64::from(token >> 16) & ((1_u64 << extra) - 1);
            acc = (acc << len) | (u64::from(entry >> 8) << extra) | bits;
            acc_bits += len;
            while acc_bits >= 32 {
                acc_bits -= 32;
                let stream = ((acc >> acc_bits) as u32).swap_bytes();
                if shared_first {
                    device_atomic(packed, word as usize).fetch_or(stream, AtomicOrdering::Relaxed);
                    shared_first = false;
                } else {
                    device_atomic(packed, word as usize).store(stream, AtomicOrdering::Relaxed);
                }
                acc &= (1_u64 << acc_bits) - 1;
                word += 1;
            }
        }
        if acc_bits > 0 {
            let stream = ((acc << (32 - acc_bits)) as u32).swap_bytes();
            device_atomic(packed, word as usize).fetch_or(stream, AtomicOrdering::Relaxed);
        }
        let last_row = ((interval + 1) * p.interval_rows).min(p.height) - 1;
        if row == last_row && chunk - row * p.chunks_per_row == p.chunks_per_row - 1 {
            let end = base * 8 + simt_load(interval_bits, interval as usize);
            let pad = (8 - end % 8) % 8;
            if pad > 0 {
                let ones = ((1_u32 << pad) - 1) << (32 - end % 32 - pad);
                device_atomic(packed, (end / 32) as usize)
                    .fetch_or(ones.swap_bytes(), AtomicOrdering::Relaxed);
            }
        }
    }
}

fn main() {}
